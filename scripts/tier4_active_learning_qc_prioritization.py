#!/usr/bin/env python3
"""Rank ROI candidates for manual QC using weak AI and physics evidence.

This is an active-learning queue builder, not a labeler. It merges the existing
manual-QC workbook, precursor review manifest, multi-cohort weak future-drop
predictions, and transfer-ranked front/timing audits into a single table that
prioritizes which ROI videos/panels a human should inspect next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val, errors="coerce")


def norm01(series: pd.Series, *, inverse: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(0.0, index=series.index)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi == lo:
        out = pd.Series(0.5, index=series.index)
    else:
        out = (values - lo) / (hi - lo)
    out = out.clip(0.0, 1.0).fillna(0.0)
    if inverse:
        out = 1.0 - out
    return out


def nonempty_path(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.len().gt(0)


def merge_unique(left: pd.DataFrame, right: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if right.empty or "roi_id" not in right.columns:
        return left
    right = right.drop_duplicates("roi_id").copy()
    overlap = [c for c in right.columns if c in left.columns and c != "roi_id"]
    right = right.rename(columns={c: f"{c}_{suffix}" for c in overlap})
    return left.merge(right, on="roi_id", how="outer")


def first_present(df: pd.DataFrame, cols: Iterable[str], default: Any = np.nan) -> pd.Series:
    out = pd.Series(default, index=df.index)
    for col in cols:
        if col in df.columns:
            vals = df[col]
            out = out.where(out.notna() & out.astype(str).ne(""), vals)
    return out


def aggregate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    if oof.empty or not {"roi_id", "model", "oof_probability"}.issubset(oof.columns):
        return pd.DataFrame({"roi_id": []})
    rows = []
    for roi_id, grp in oof.groupby("roi_id", dropna=False):
        row: Dict[str, Any] = {"roi_id": roi_id}
        probs = pd.to_numeric(grp["oof_probability"], errors="coerce")
        row["model_probability_mean"] = float(probs.mean()) if probs.notna().any() else np.nan
        row["model_probability_max"] = float(probs.max()) if probs.notna().any() else np.nan
        for model, mgrp in grp.groupby("model", dropna=False):
            prob = pd.to_numeric(mgrp["oof_probability"], errors="coerce").mean()
            safe_model = str(model).replace(" ", "_")
            row[f"{safe_model}_oof_probability"] = float(prob) if np.isfinite(prob) else np.nan
        if "weak_future_drop_label" in grp.columns:
            labels = pd.to_numeric(grp["weak_future_drop_label"], errors="coerce").dropna()
            row["weak_future_drop_label"] = int(labels.iloc[0]) if len(labels) else np.nan
        if "video_cohort" in grp.columns:
            row["video_cohort_model"] = ";".join(sorted(set(grp["video_cohort"].dropna().astype(str))))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_timing(timing: pd.DataFrame) -> pd.DataFrame:
    if timing.empty or "roi_id" not in timing.columns:
        return pd.DataFrame({"roi_id": []})
    value_cols = [
        "near_transition_residual_fraction",
        "particle_to_nonparticle_mse_ratio_median",
        "residual_peak_particle_mse",
        "weighted_center_distance_to_transition_frac",
        "peak_distance_to_transition_frac",
        "near_minus_far_particle_mse_median",
        "mask_fallback_fraction_eval",
    ]
    pieces = []
    for method, grp in timing.groupby("method", dropna=False):
        cols = [c for c in value_cols if c in grp.columns]
        if not cols:
            continue
        part = grp.groupby("roi_id", as_index=False)[cols].median(numeric_only=True)
        part = part.rename(columns={c: f"{method}_{c}" for c in cols})
        pieces.append(part)
    if pieces:
        out = pieces[0]
        for part in pieces[1:]:
            out = out.merge(part, on="roi_id", how="outer")
    else:
        out = timing[["roi_id"]].drop_duplicates().copy()
    base_cols = [
        "cycleNo",
        "cohort_role",
        "event_reference_cycle",
        "transfer_rank",
        "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles",
        "any_abrupt_drop",
        "threshold_robust_phase_score",
        "threshold_robust_diffusion_score",
        "diffusion_proxy_median_um2_per_s",
        "radius2_slope_median_px2_per_s",
        "persistence_particle_mse_mean",
        "low_rank_dmd_particle_mse_mean",
        "transferred_masked_residual_signature",
    ]
    base = timing[["roi_id"] + [c for c in base_cols if c in timing.columns]].drop_duplicates("roi_id")
    return base.merge(out, on="roi_id", how="outer")


def add_reasons(row: pd.Series) -> str:
    reasons: List[str] = []
    if row.get("model_uncertainty_score", 0) >= 0.75:
        reasons.append("model_boundary_case")
    if row.get("future_risk_score", 0) >= 0.75:
        reasons.append("high_future_drop_probability")
    if row.get("front_guardrail_score", 0) >= 0.70:
        reasons.append("front_diffusion_guardrail_review")
    if row.get("timing_residual_score", 0) >= 0.70:
        reasons.append("transition_timing_residual_review")
    if row.get("control_balance_score", 0) >= 0.65:
        reasons.append("control_balance_review")
    if row.get("visual_asset_score", 0) > 0:
        reasons.append("visual_asset_available")
    if not reasons:
        reasons.append("general_pending_qc")
    return ";".join(reasons)


def add_tier(row: pd.Series) -> str:
    if row.get("manual_qc_status", "pending") not in {"", "pending", np.nan}:
        return "already_reviewed_or_deferred"
    if row.get("active_learning_qc_score", 0) >= 0.40 and row.get("visual_asset_score", 0) > 0:
        return "immediate_manual_qc"
    if row.get("model_uncertainty_score", 0) >= 0.80:
        return "model_boundary_case"
    if row.get("front_guardrail_score", 0) >= 0.75:
        return "front_diffusion_guardrail_review"
    if row.get("control_balance_score", 0) >= 0.75:
        return "control_balance_review"
    return "standard_manual_qc"


def write_readme(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Active-Learning QC Prioritization",
        "",
        "This audit ranks ROI candidates for manual review. It does not assign degradation labels or validate diffusion/front claims.",
        "",
        f"- Candidate rows: {summary['n_candidate_rows']}",
        f"- Immediate manual-QC rows: {summary['tier_counts'].get('immediate_manual_qc', 0)}",
        f"- Rows with visual assets: {summary['n_rows_with_visual_asset']}",
        f"- Guardrail: {summary['guardrail']}",
        "",
        "Primary outputs:",
        "",
        "- `active_learning_qc_priority_table.csv`",
        "- `active_learning_qc_cycle_summary.csv`",
        "- `active_learning_qc_reason_counts.csv`",
        "- `active_learning_qc_summary.json`",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/active_learning_qc_prioritization")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manual = read_csv(derived / "manual_qc_label_workbook" / "manual_qc_label_template.csv")
    precursor = read_csv(derived / "precursor_informed_roi_review" / "precursor_informed_roi_review_manifest.csv")
    oof = aggregate_oof(read_csv(derived / "multicohort_future_drop_model" / "multicohort_future_drop_oof_predictions.csv"))
    feature_table = read_csv(derived / "multicohort_future_drop_model" / "multicohort_future_drop_feature_table.csv")
    front_review = read_csv(derived / "transfer_ranked_front_physics_audit" / "transfer_ranked_front_physics_review_priority.csv")
    timing = aggregate_timing(read_csv(derived / "transfer_ranked_residual_transition_timing" / "transfer_ranked_residual_transition_timing_per_roi.csv"))
    transfer_table = read_csv(derived / "transfer_ranked_roi_reconstruction" / "transfer_ranked_roi_table.csv")

    frames = [df[["roi_id"]].drop_duplicates() for df in [manual, precursor, oof, feature_table, front_review, timing, transfer_table] if not df.empty and "roi_id" in df.columns]
    if not frames:
        raise SystemExit("No ROI candidate tables found.")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("roi_id")

    for src, suffix in [
        (manual, "manual"),
        (precursor, "precursor"),
        (oof, "model"),
        (feature_table, "feature"),
        (front_review, "front"),
        (timing, "timing"),
        (transfer_table, "transfer"),
    ]:
        df = merge_unique(df, src, suffix)

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["cycleNo"] = first_present(df, ["cycleNo", "cycleNo_manual", "cycleNo_precursor", "cycleNo_front", "cycleNo_timing", "cycleNo_feature", "cycleNo_transfer"])
    df["cohort_role"] = first_present(df, ["cohort_role", "cohort_role_manual", "cohort_role_precursor", "cohort_role_timing", "video_cohort", "video_cohort_feature", "video_cohort_model"])
    df["manual_qc_status"] = first_present(df, ["manual_qc_status", "manual_qc_status_manual"], "pending").fillna("pending")

    rf_prob = first_present(df, ["random_forest_oof_probability", "random_forest_oof_probability_model", "model_probability_max"])
    logistic_prob = first_present(df, ["logistic_l2_oof_probability", "logistic_l2_oof_probability_model"])
    model_prob = pd.to_numeric(rf_prob.fillna(logistic_prob), errors="coerce")
    df["model_future_drop_probability"] = model_prob
    df["model_uncertainty"] = (1.0 - (model_prob - 0.5).abs() * 2.0).clip(0.0, 1.0)

    df["manual_review_score_norm"] = norm01(first_present(df, ["combined_review_priority_score", "combined_review_priority_score_manual"]))
    df["precursor_review_score_norm"] = norm01(first_present(df, ["precursor_informed_review_score", "precursor_informed_review_score_precursor", "combined_review_priority_score_precursor"]))
    df["future_risk_score"] = model_prob.fillna(0.0).clip(0.0, 1.0)
    df["model_uncertainty_score"] = df["model_uncertainty"].fillna(0.0)
    df["front_guardrail_score"] = (
        0.35 * norm01(first_present(df, ["front_physics_review_score", "front_physics_review_score_front"]))
        + 0.25 * norm01(first_present(df, ["threshold_robust_phase_score", "threshold_robust_phase_score_front", "threshold_robust_phase_score_timing"]))
        + 0.20 * norm01(first_present(df, ["threshold_robust_diffusion_score", "threshold_robust_diffusion_score_timing"]))
        + 0.20 * norm01(first_present(df, ["diffusion_proxy_median_um2_per_s", "diffusion_proxy_median_um2_per_s_front", "diffusion_proxy_median_um2_per_s_timing"]).abs())
    ).clip(0.0, 1.0)
    df["timing_residual_score"] = (
        0.35 * norm01(first_present(df, ["persistence_particle_to_nonparticle_mse_ratio_median", "persistence_particle_to_nonparticle_mse_ratio_median_timing"]))
        + 0.25 * norm01(first_present(df, ["low_rank_dmd_near_transition_residual_fraction", "low_rank_dmd_near_transition_residual_fraction_timing"]))
        + 0.20 * norm01(first_present(df, ["low_rank_dmd_residual_peak_particle_mse", "low_rank_dmd_residual_peak_particle_mse_timing"]))
        + 0.20 * norm01(first_present(df, ["low_rank_dmd_weighted_center_distance_to_transition_frac", "low_rank_dmd_weighted_center_distance_to_transition_frac_timing"]), inverse=True)
    ).clip(0.0, 1.0)
    role = df["cohort_role"].fillna("").astype(str).str.lower()
    label = pd.to_numeric(first_present(df, ["weak_future_drop_label", "weak_future_drop_label_model", "future_any_drop_within_8cycles", "future_any_drop_within_8cycles_timing", "future_any_drop_within_8cycles_front"]), errors="coerce")
    df["control_balance_score"] = ((role.str.contains("control") | label.eq(0)).astype(float) * 0.75 + (role.str.contains("selected|event").astype(float) * 0.25)).clip(0.0, 1.0)

    visual_cols = [
        "primary_qc_png",
        "control_balanced_qc_png",
        "roi_preview_path",
        "front_crop_preview_path",
        "front_tracking_plot_path",
        "rollout_preview_path",
        "primary_qc_png_manual",
        "control_balanced_qc_png_manual",
        "roi_preview_path_manual",
        "roi_preview_path_transfer",
    ]
    has_visual = pd.Series(False, index=df.index)
    for col in visual_cols:
        if col in df.columns:
            has_visual = has_visual | nonempty_path(df[col])
    df["visual_asset_score"] = has_visual.astype(float)

    df["active_learning_qc_score"] = (
        0.18 * df["manual_review_score_norm"]
        + 0.14 * df["precursor_review_score_norm"]
        + 0.18 * df["future_risk_score"]
        + 0.16 * df["model_uncertainty_score"]
        + 0.16 * df["front_guardrail_score"]
        + 0.12 * df["timing_residual_score"]
        + 0.04 * df["control_balance_score"]
        + 0.02 * df["visual_asset_score"]
    ).clip(0.0, 1.0)

    df["review_reason_tags"] = df.apply(add_reasons, axis=1)
    df["recommended_qc_tier"] = df.apply(add_tier, axis=1)
    df["active_learning_rank"] = df["active_learning_qc_score"].rank(method="first", ascending=False).astype(int)
    df = df.sort_values(["active_learning_qc_score", "visual_asset_score"], ascending=[False, False]).reset_index(drop=True)
    df["active_learning_rank"] = np.arange(1, len(df) + 1)
    pending_visual = df["manual_qc_status"].fillna("pending").eq("pending") & df["visual_asset_score"].gt(0)
    top_visual = pending_visual & df["active_learning_rank"].le(20)
    df.loc[top_visual, "recommended_qc_tier"] = "immediate_manual_qc"

    front_cols = [
        "roi_id",
        "active_learning_rank",
        "recommended_qc_tier",
        "active_learning_qc_score",
        "review_reason_tags",
        "manual_qc_status",
        "cohort_role",
        "cycleNo",
        "model_future_drop_probability",
        "model_uncertainty_score",
        "future_risk_score",
        "front_guardrail_score",
        "timing_residual_score",
        "control_balance_score",
        "visual_asset_score",
        "weak_future_drop_label",
        "combined_review_priority_score",
        "precursor_informed_review_score",
        "primary_qc_png",
        "control_balanced_qc_png",
        "roi_preview_path",
        "rollout_preview_path",
    ]
    ordered = [c for c in front_cols if c in df.columns] + [c for c in df.columns if c not in front_cols]
    df[ordered].to_csv(out / "active_learning_qc_priority_table.csv", index=False)

    cycle_summary = (
        df.groupby("cycleNo", dropna=False)
        .agg(
            n_roi=("roi_id", "count"),
            max_active_learning_qc_score=("active_learning_qc_score", "max"),
            mean_active_learning_qc_score=("active_learning_qc_score", "mean"),
            n_immediate=("recommended_qc_tier", lambda s: int((s == "immediate_manual_qc").sum())),
            n_with_visual_asset=("visual_asset_score", "sum"),
        )
        .reset_index()
        .sort_values(["max_active_learning_qc_score", "n_roi"], ascending=[False, False])
    )
    cycle_summary.to_csv(out / "active_learning_qc_cycle_summary.csv", index=False)

    reasons = []
    for tags in df["review_reason_tags"].fillna(""):
        reasons.extend([tag for tag in str(tags).split(";") if tag])
    reason_counts = pd.Series(reasons).value_counts().rename_axis("reason_tag").reset_index(name="n_roi")
    reason_counts.to_csv(out / "active_learning_qc_reason_counts.csv", index=False)

    summary: Dict[str, Any] = {
        "derived_dir": str(derived),
        "n_candidate_rows": int(len(df)),
        "n_cycles": int(pd.to_numeric(df["cycleNo"], errors="coerce").nunique()),
        "n_rows_with_visual_asset": int(df["visual_asset_score"].sum()),
        "tier_counts": df["recommended_qc_tier"].value_counts().to_dict(),
        "reason_counts": reason_counts.to_dict("records"),
        "top_priority_rows": df[ordered].head(15).to_dict("records"),
        "top_cycles": cycle_summary.head(12).to_dict("records"),
        "score_components": {
            "manual_review_score_norm": 0.18,
            "precursor_review_score_norm": 0.14,
            "future_risk_score": 0.18,
            "model_uncertainty_score": 0.16,
            "front_guardrail_score": 0.16,
            "timing_residual_score": 0.12,
            "control_balance_score": 0.04,
            "visual_asset_score": 0.02,
        },
        "guardrail": "Active-learning QC ranks pending ROI review candidates from automatic and weak-label evidence only; manual labels, diffusion claims, and deployment decisions remain withheld until human QC.",
    }
    summary = clean_json(summary)
    (out / "active_learning_qc_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_readme(out / "README.md", summary)


if __name__ == "__main__":
    main()
