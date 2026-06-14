#!/usr/bin/env python3
"""Automatic QC triage surrogate for NMC ROI/front candidates.

This is not a manual-label generator.  It consolidates existing QC manifests,
front metrics, active-learning priority, and visual asset availability into a
reproducible triage table that separates likely interpretable candidates from
artifact-risk candidates while preserving manual_qc_status=pending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def text(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return df[col].fillna("").astype(str)


def rank01(series: pd.Series, high: bool = True) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() <= 1:
        return pd.Series(np.nan, index=series.index)
    ranks = vals.rank(pct=True, method="average")
    return ranks if high else 1.0 - ranks


def rank01_filled(series: pd.Series, high: bool = True, fill: float = 0.0) -> pd.Series:
    return rank01(series, high=high).fillna(fill)


def any_path_exists(row: pd.Series, cols: Iterable[str]) -> bool:
    for col in cols:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return True
    return False


def merge_if(base: pd.DataFrame, other: pd.DataFrame, cols: List[str], suffix: str = "") -> pd.DataFrame:
    if other.empty or "roi_id" not in other.columns:
        return base
    keep = ["roi_id"] + [c for c in cols if c in other.columns and c != "roi_id"]
    tmp = other[keep].drop_duplicates("roi_id", keep="first").copy()
    if suffix:
        tmp = tmp.rename(columns={c: f"{c}{suffix}" for c in tmp.columns if c != "roi_id"})
    return base.merge(tmp, on="roi_id", how="left")


def binary_test(df: pd.DataFrame, feature: str, target: str) -> Dict[str, Any]:
    if feature not in df.columns or target not in df.columns:
        return {}
    tmp = df[[feature, target]].copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
    tmp = tmp.dropna()
    if len(tmp) < 8 or tmp[target].nunique() < 2:
        return {}
    pos = tmp.loc[tmp[target] > 0, feature].to_numpy(float)
    neg = tmp.loc[tmp[target] <= 0, feature].to_numpy(float)
    if len(pos) < 2 or len(neg) < 2:
        return {}
    return {
        "feature": feature,
        "target": target,
        "n": int(len(tmp)),
        "n_positive": int(len(pos)),
        "median_positive": float(np.nanmedian(pos)),
        "median_negative": float(np.nanmedian(neg)),
        "median_positive_minus_negative": float(np.nanmedian(pos) - np.nanmedian(neg)),
        "mannwhitney_p": float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue),
    }


def correlation(df: pd.DataFrame, x: str, y: str) -> Dict[str, Any]:
    if x not in df.columns or y not in df.columns:
        return {}
    tmp = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(tmp) < 8 or tmp[x].nunique() < 3 or tmp[y].nunique() < 3:
        return {}
    res = spearmanr(tmp[x], tmp[y])
    return {"x": x, "y": y, "n": int(len(tmp)), "spearman_rho": float(res.statistic), "p_value": float(res.pvalue)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/automatic_qc_triage_surrogate")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manual = read_csv(derived / "manual_qc_label_workbook" / "manual_qc_label_template.csv")
    active = read_csv(derived / "active_learning_qc_prioritization" / "active_learning_qc_priority_table.csv")
    qc_sens = read_csv(derived / "control_balanced_front_qc_sensitivity" / "control_balanced_front_qc_sensitivity_joined.csv")
    tracking = read_csv(derived / "control_balanced_front_tracking_table" / "control_balanced_front_rois_for_tracking.csv")
    physics = read_csv(derived / "physics_consistency_claim_matrix" / "physics_consistency_claim_matrix.csv")
    if manual.empty or "roi_id" not in manual.columns:
        raise FileNotFoundError("Need manual_qc_label_workbook/manual_qc_label_template.csv")

    df = manual.copy()
    df = merge_if(
        df,
        active,
        [
            "active_learning_rank", "recommended_qc_tier", "active_learning_qc_score",
            "review_reason_tags", "model_future_drop_probability", "model_uncertainty_score",
            "visual_asset_score", "front_guardrail_score", "timing_residual_score",
            "weak_future_drop_label", "physics_consistency_score", "physics_consistency_tier",
            "claim_readiness", "transferred_masked_residual_signature", "future_any_drop_within_8cycles",
            "future_any_drop_within_16cycles", "any_abrupt_drop", "is_cross_modal_priority",
        ],
        suffix="_active",
    )
    df = merge_if(
        df,
        qc_sens,
        [
            "validation_score", "front_quality_score", "thresholds_finite", "phase_slope_positive_fraction",
            "threshold_robust_phase_score", "threshold_robust_diffusion_score", "q70_phase_ci_excludes_zero",
            "q70_phase_ci_positive", "original_qc_n_components", "original_qc_largest_component_fraction",
            "original_qc_edge_touch_fraction", "original_qc_fragmented_q70_mask", "original_qc_diffusion_ci_crosses_zero",
            "original_qc_active_control", "original_qc_no_auto_flags", "balanced_qc_n_components",
            "balanced_qc_largest_component_fraction", "balanced_qc_edge_touch_fraction", "balanced_qc_fragmented_q70_mask",
            "balanced_qc_diffusion_ci_crosses_zero", "balanced_qc_active_control", "balanced_qc_no_auto_flags",
            "original_qc_auto_review_flags", "balanced_qc_auto_review_flags",
        ],
        suffix="_sens",
    )
    df = merge_if(
        df,
        tracking,
        [
            "front_radius_slope_r2", "front_radius_monotonic_fraction", "front_radius2_slope_r2",
            "apparent_diffusion_proxy_ds_px2_per_frame", "object_match_distance_ds", "object_area_ds_px",
            "object_mean_abs_z", "selected_for_next_tracking", "validation_score", "validation_tier",
        ],
        suffix="_track",
    )
    df = merge_if(
        df,
        physics,
        [
            "physics_consistency_score", "physics_pillar_support_count", "physics_pillar_strong_support_count",
            "physics_pillar_contradiction_count", "front_direction_score", "optical_change_score",
            "rollout_residual_score", "kinetic_transition_score", "precursor_context_score", "mode_taxonomy_score",
            "claim_readiness", "physics_consistency_tier",
        ],
        suffix="_physics",
    )

    flags = (
        text(df, "auto_review_flags") + ";"
        + text(df, "auto_review_flags_active") + ";"
        + text(df, "original_qc_auto_review_flags_sens") + ";"
        + text(df, "balanced_qc_auto_review_flags_sens")
    ).str.lower()
    df["auto_flag_fragmented"] = flags.str.contains("fragmented", regex=False).astype(int)
    df["auto_flag_diffusion_ci"] = flags.str.contains("diffusion_ci", regex=False).astype(int)
    df["auto_flag_threshold_sign"] = flags.str.contains("threshold_sign", regex=False).astype(int)
    df["auto_flag_active_control"] = flags.str.contains("active_control", regex=False).astype(int)
    df["auto_flag_none"] = flags.str.contains("none", regex=False).astype(int)
    visual_cols = ["primary_qc_png", "control_balanced_qc_png", "roi_preview_path", "front_crop_preview_path", "front_tracking_plot_path", "rollout_preview_path"]
    df["has_visual_asset"] = df.apply(lambda row: any_path_exists(row, visual_cols), axis=1).astype(int)

    largest_component = pd.concat([
        num(df, "largest_component_fraction"),
        num(df, "balanced_qc_largest_component_fraction_sens"),
        num(df, "original_qc_largest_component_fraction_sens"),
    ], axis=1).max(axis=1, skipna=True)
    n_components = pd.concat([
        num(df, "n_components"),
        num(df, "balanced_qc_n_components_sens"),
        num(df, "original_qc_n_components_sens"),
    ], axis=1).min(axis=1, skipna=True)
    edge_touch = pd.concat([
        num(df, "edge_touch_fraction"),
        num(df, "balanced_qc_edge_touch_fraction_sens"),
        num(df, "original_qc_edge_touch_fraction_sens"),
    ], axis=1).max(axis=1, skipna=True)
    df["auto_qc_largest_component_fraction"] = largest_component
    df["auto_qc_n_components_min"] = n_components
    df["auto_qc_edge_touch_fraction"] = edge_touch

    df["surrogate_particle_identity_score"] = (
        0.30 * df["has_visual_asset"]
        + 0.25 * rank01_filled(num(df, "front_quality_score"))
        + 0.20 * rank01_filled(largest_component)
        + 0.15 * rank01_filled(n_components, high=False)
        + 0.10 * rank01_filled(edge_touch.fillna(1.0), high=False)
    )
    df["surrogate_front_mask_score"] = (
        0.25 * (1 - df["auto_flag_fragmented"])
        + 0.20 * (1 - df["auto_flag_threshold_sign"])
        + 0.20 * num(df, "q70_phase_ci_positive").fillna(num(df, "q70_phase_ci_positive_sens")).fillna(0)
        + 0.15 * num(df, "q70_phase_ci_excludes_zero_sens").fillna(0)
        + 0.10 * rank01_filled(num(df, "threshold_robust_phase_score").abs())
        + 0.10 * rank01_filled(num(df, "q70_fraction_delta").abs())
    )
    df["surrogate_diffusion_interpretability_score"] = (
        0.20 * (1 - df["auto_flag_diffusion_ci"])
        + 0.20 * (1 - num(df, "q70_radius_ci_crosses_zero").fillna(1).clip(0, 1))
        + 0.20 * rank01_filled(num(df, "front_radius2_slope_r2_track"))
        + 0.20 * rank01_filled(num(df, "front_radius_monotonic_fraction_track"))
        + 0.10 * rank01_filled(num(df, "threshold_robust_diffusion_score_sens").abs())
        + 0.10 * rank01_filled(num(df, "object_match_distance_ds_track"), high=False)
    )
    df["automatic_qc_surrogate_score"] = (
        0.40 * df["surrogate_particle_identity_score"]
        + 0.35 * df["surrogate_front_mask_score"]
        + 0.15 * df["surrogate_diffusion_interpretability_score"]
        + 0.10 * rank01_filled(num(df, "combined_review_priority_score"))
    )
    risk_components = pd.concat([
        df["auto_flag_fragmented"],
        df["auto_flag_diffusion_ci"],
        df["auto_flag_threshold_sign"],
        df["auto_flag_active_control"],
        (largest_component < 0.45).astype(int),
        (n_components > 70).astype(int),
        (edge_touch > 0.05).astype(int),
        (num(df, "front_radius2_slope_r2_track") < 0.10).astype(int),
    ], axis=1)
    df["automatic_artifact_risk_score"] = risk_components.mean(axis=1, skipna=True)

    df["auto_qc_tier"] = "standard_review"
    df.loc[(df["automatic_qc_surrogate_score"] >= 0.68) & (df["automatic_artifact_risk_score"] <= 0.25), "auto_qc_tier"] = "auto_surrogate_likely_interpretable"
    df.loc[df["automatic_artifact_risk_score"] >= 0.50, "auto_qc_tier"] = "auto_surrogate_artifact_risk"
    df.loc[(df["surrogate_diffusion_interpretability_score"] < 0.35) | df["auto_flag_diffusion_ci"].eq(1), "auto_diffusion_guardrail"] = 1
    df["auto_diffusion_guardrail"] = df["auto_diffusion_guardrail"].fillna(0).astype(int)
    df["manual_qc_preserved_status"] = text(df, "manual_qc_status").replace("", "pending")

    sort_cols = ["auto_qc_tier", "automatic_qc_surrogate_score", "combined_review_priority_score"]
    triage = df.sort_values(sort_cols, ascending=[True, False, False]).copy()
    preferred_cols = [
        "roi_id", "cohort_role", "cycleNo", "event_reference_cycle", "manual_qc_preserved_status",
        "auto_qc_tier", "automatic_qc_surrogate_score", "automatic_artifact_risk_score",
        "surrogate_particle_identity_score", "surrogate_front_mask_score", "surrogate_diffusion_interpretability_score",
        "auto_diffusion_guardrail", "has_visual_asset", "auto_flag_fragmented", "auto_flag_diffusion_ci",
        "auto_flag_threshold_sign", "auto_flag_active_control", "auto_qc_largest_component_fraction",
        "auto_qc_n_components_min", "auto_qc_edge_touch_fraction", "front_radius2_slope_r2_track",
        "front_radius_monotonic_fraction_track", "combined_review_priority_score", "review_priority_tier",
        "recommended_review_reason", "primary_qc_png", "control_balanced_qc_png", "roi_preview_path",
        "front_crop_preview_path", "front_tracking_plot_path", "rollout_preview_path",
    ]
    triage_out = triage[[c for c in preferred_cols if c in triage.columns]]
    triage_out.to_csv(out / "automatic_qc_triage_table.csv", index=False)

    tier_summary = (
        df.groupby("auto_qc_tier", dropna=False)
        .agg(
            n_candidates=("roi_id", "count"),
            event_fraction=("cohort_role", lambda s: float((s.astype(str) == "event").mean())),
            median_surrogate_score=("automatic_qc_surrogate_score", "median"),
            median_artifact_risk=("automatic_artifact_risk_score", "median"),
            diffusion_guardrail_rate=("auto_diffusion_guardrail", "mean"),
            visual_asset_rate=("has_visual_asset", "mean"),
        )
        .reset_index()
        .sort_values(["median_surrogate_score", "n_candidates"], ascending=[False, False])
    )
    tier_summary.to_csv(out / "automatic_qc_triage_tier_summary.csv", index=False)

    feature_cols = [
        "combined_review_priority_score", "front_quality_score", "rollout_mobility_difficulty_score",
        "threshold_robust_phase_score", "phase_slope_positive_fraction_protocol_residual",
        "diffusion_proxy_abs_median_um2_per_s", "physics_consistency_score_physics",
        "surrogate_particle_identity_score", "surrogate_front_mask_score",
        "surrogate_diffusion_interpretability_score", "automatic_artifact_risk_score",
    ]
    df["is_likely_interpretable"] = (df["auto_qc_tier"] == "auto_surrogate_likely_interpretable").astype(int)
    df["is_artifact_risk"] = (df["auto_qc_tier"] == "auto_surrogate_artifact_risk").astype(int)
    tests = []
    for feature in feature_cols:
        for target in ["is_likely_interpretable", "is_artifact_risk", "auto_diffusion_guardrail"]:
            row = binary_test(df, feature, target)
            if row:
                tests.append(row)
    tests_df = pd.DataFrame(tests).sort_values(["mannwhitney_p", "feature"]) if tests else pd.DataFrame()
    tests_df.to_csv(out / "automatic_qc_triage_feature_tests.csv", index=False)

    corr_rows = []
    for x in feature_cols:
        for y in ["automatic_qc_surrogate_score", "automatic_artifact_risk_score"]:
            if x != y:
                row = correlation(df, x, y)
                if row:
                    corr_rows.append(row)
    corr_df = pd.DataFrame(corr_rows).sort_values(["p_value", "x"]) if corr_rows else pd.DataFrame()
    corr_df.to_csv(out / "automatic_qc_triage_correlations.csv", index=False)

    top_review = triage_out.head(30)
    top_review.to_csv(out / "automatic_qc_triage_top_review_queue.csv", index=False)

    summary = {
        "n_candidates": int(len(df)),
        "manual_status_counts": df["manual_qc_preserved_status"].value_counts(dropna=False).to_dict(),
        "tier_summary": tier_summary.to_dict("records"),
        "diffusion_guardrail_count": int(df["auto_diffusion_guardrail"].sum()),
        "likely_interpretable_count": int((df["auto_qc_tier"] == "auto_surrogate_likely_interpretable").sum()),
        "artifact_risk_count": int((df["auto_qc_tier"] == "auto_surrogate_artifact_risk").sum()),
        "top_likely_interpretable": triage_out[triage_out["auto_qc_tier"].eq("auto_surrogate_likely_interpretable")].head(12).to_dict("records"),
        "top_artifact_risk": triage_out[triage_out["auto_qc_tier"].eq("auto_surrogate_artifact_risk")].head(12).to_dict("records"),
        "top_feature_tests": tests_df.head(20).to_dict("records") if not tests_df.empty else [],
        "top_correlations": corr_df.head(20).to_dict("records") if not corr_df.empty else [],
        "guardrail": "Automatic QC triage surrogate preserves manual_qc_status as pending. It ranks candidate review priority from existing automatic/visual QC diagnostics and must not be treated as manual particle identity, front-mask, or diffusion-interpretable labels.",
        "outputs": {
            "triage_table": str(out / "automatic_qc_triage_table.csv"),
            "tier_summary": str(out / "automatic_qc_triage_tier_summary.csv"),
            "feature_tests": str(out / "automatic_qc_triage_feature_tests.csv"),
            "correlations": str(out / "automatic_qc_triage_correlations.csv"),
            "top_review_queue": str(out / "automatic_qc_triage_top_review_queue.csv"),
        },
    }
    with (out / "automatic_qc_triage_surrogate_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    readme = [
        "# Automatic QC Triage Surrogate",
        "",
        "Consolidates existing automatic/visual QC diagnostics for manual-review prioritization.",
        "",
        f"- Candidates: {summary['n_candidates']}",
        f"- Likely interpretable by surrogate: {summary['likely_interpretable_count']}",
        f"- Artifact-risk by surrogate: {summary['artifact_risk_count']}",
        f"- Diffusion guardrail count: {summary['diffusion_guardrail_count']}",
        "",
        "Guardrail: no manual labels are assigned; all manual status values remain pending.",
    ]
    (out / "README.md").write_text("\n".join(readme).rstrip() + "\n")
    print(json.dumps(clean_json({
        "out_dir": str(out),
        "n_candidates": summary["n_candidates"],
        "tier_summary": summary["tier_summary"],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
