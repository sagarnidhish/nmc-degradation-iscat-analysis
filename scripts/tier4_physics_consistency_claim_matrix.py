#!/usr/bin/env python3
"""Build a conservative multimodal physics-consistency matrix for NMC ROI claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


PILLARS: Dict[str, List[tuple[str, float]]] = {
    "front_direction": [
        ("phase_slope_positive_fraction_protocol_residual", 1.0),
        ("threshold_robust_phase_score_protocol_residual", 1.0),
        ("phase_slope_median_per_s_protocol_residual", 1.0),
        ("q70_fraction_delta", 1.0),
        ("q70_positive_rate_fraction", 1.0),
    ],
    "optical_change": [
        ("cumulative_abs_norm_change_protocol_residual", 1.0),
        ("first_last_corr_protocol_residual", -1.0),
        ("high_fraction_delta_protocol_residual", 1.0),
        ("roi_norm_mean_delta_protocol_residual", 1.0),
        ("roi_norm_total_variation", 1.0),
    ],
    "rollout_residual": [
        ("rollout_mobility_difficulty_score", 1.0),
        ("latent_net_displacement_protocol_residual", 1.0),
        ("dmd_minus_persistence_mse_protocol_residual", 1.0),
        ("temporal_diff_energy_protocol_residual", 1.0),
    ],
    "kinetic_transition": [
        ("roi_norm_max_abs_rate_per_s", 1.0),
        ("q70_max_abs_rate_per_s", 1.0),
        ("q70_transformed_fraction_delta", 1.0),
        ("q70_logistic_k_per_s", 1.0),
        ("q70_logistic_r2", 1.0),
    ],
    "precursor_context": [
        ("precursor_informed_review_score", 1.0),
        ("score_component_precursor_context", 1.0),
        ("precursor_signed_severity_mean", 1.0),
        ("precursor_signed_severity_sum", 1.0),
    ],
    "echem_shape_context": [
        ("shape_V_q95", -1.0),
        ("neg_dq_abs_peak_frac", 1.0),
        ("all_dq_abs_entropy", 1.0),
        ("shape_charge_mAh_abs", 1.0),
        ("shape_dVdt_abs_p95", 1.0),
    ],
    "mode_taxonomy": [
        ("mode_review_priority", 1.0),
        ("is_event_enriched_mode", 1.0),
        ("physics_no_qc_rf_event_probability_protocol_residual", 1.0),
    ],
}


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


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def robust_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = float(np.nanmedian(x)) if x.notna().any() else np.nan
    mad = float(np.nanmedian(np.abs(x - med))) if x.notna().any() else np.nan
    if np.isfinite(mad) and mad > 1e-12:
        z = (x - med) / (1.4826 * mad)
    else:
        std = float(np.nanstd(x))
        z = (x - float(np.nanmean(x))) / std if np.isfinite(std) and std > 1e-12 else x * np.nan
    return z.replace([np.inf, -np.inf], np.nan).clip(-5, 5)


def signed_feature_score(df: pd.DataFrame, col: str, sign: float) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    if col in {"q70_logistic_k_per_s", "q70_max_abs_rate_per_s", "roi_norm_max_abs_rate_per_s"}:
        x = x.abs()
    if col == "is_event_enriched_mode":
        x = x.astype(float)
    return robust_z(x) * sign


def add_pillar_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for pillar, specs in PILLARS.items():
        score_cols = []
        used = []
        for col, sign in specs:
            if col not in out.columns:
                continue
            zcol = f"{pillar}__{col}__signed_z"
            out[zcol] = signed_feature_score(out, col, sign)
            score_cols.append(zcol)
            used.append(col)
        if score_cols:
            out[f"{pillar}_score"] = out[score_cols].mean(axis=1, skipna=True)
            out[f"{pillar}_n_features"] = out[score_cols].notna().sum(axis=1)
            out[f"{pillar}_features_used"] = ";".join(used)
        else:
            out[f"{pillar}_score"] = np.nan
            out[f"{pillar}_n_features"] = 0
            out[f"{pillar}_features_used"] = ""
    pillar_cols = [f"{p}_score" for p in PILLARS]
    out["physics_pillar_support_count"] = out[pillar_cols].ge(0.5).sum(axis=1)
    out["physics_pillar_strong_support_count"] = out[pillar_cols].ge(1.0).sum(axis=1)
    out["physics_pillar_contradiction_count"] = out[pillar_cols].le(-0.5).sum(axis=1)
    out["physics_pillar_score_mean"] = out[pillar_cols].mean(axis=1, skipna=True)
    out["physics_pillar_score_min"] = out[pillar_cols].min(axis=1, skipna=True)
    out["physics_consistency_score"] = (
        out["physics_pillar_score_mean"].fillna(0)
        + 0.25 * out["physics_pillar_support_count"]
        + 0.25 * out["physics_pillar_strong_support_count"]
        - 0.35 * out["physics_pillar_contradiction_count"]
    )
    return out


def classify_rows(df: pd.DataFrame, calibration: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    auto_flags = out.get("auto_review_flags", pd.Series("", index=out.index)).fillna("").astype(str)
    manual_status = out.get("manual_qc_status", pd.Series("pending", index=out.index)).fillna("pending").astype(str)
    manual_decision = out.get("manual_qc_decision", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    accepted = manual_decision.eq("accept")
    artifact_flag = auto_flags.str.contains("fragment|edge|component|flag", case=False, na=False)

    out["manual_qc_accepted_candidate"] = accepted
    out["automatic_artifact_caution"] = artifact_flag
    out["calibration_time_evidence"] = f"{calibration.get('h5_files_with_camera_timing', 0)}/{calibration.get('h5_files_scanned', 0)} h5 timing files"
    out["calibration_spatial_evidence"] = str(calibration.get("spatial_status", "unknown"))

    tiers = []
    readiness = []
    reasons = []
    for _, row in out.iterrows():
        support = int(row.get("physics_pillar_support_count", 0))
        strong = int(row.get("physics_pillar_strong_support_count", 0))
        contradiction = int(row.get("physics_pillar_contradiction_count", 0))
        score = float(row.get("physics_consistency_score", np.nan))
        front = float(row.get("front_direction_score", np.nan))
        kinetic = float(row.get("kinetic_transition_score", np.nan))
        rollout = float(row.get("rollout_residual_score", np.nan))
        precursor = float(row.get("precursor_context_score", np.nan))
        mode = float(row.get("mode_taxonomy_score", np.nan))
        flag = bool(row.get("automatic_artifact_caution", False))
        reason_bits = []
        if support >= 4:
            reason_bits.append("multi-pillar support")
        if strong >= 2:
            reason_bits.append("multiple strong pillars")
        if np.isfinite(front) and np.isfinite(kinetic) and front >= 0.5 and kinetic >= 0.5:
            reason_bits.append("front/kinetic agreement")
        if np.isfinite(rollout) and np.isfinite(mode) and rollout >= 0.5 and mode >= 0.5:
            reason_bits.append("rollout/mode agreement")
        if np.isfinite(precursor) and precursor >= 0.5:
            reason_bits.append("precursor context")
        if contradiction:
            reason_bits.append("discordant pillars")
        if flag:
            reason_bits.append("automatic artifact caution")

        if support >= 5 and score >= 1.5 and contradiction <= 1 and not flag:
            tier = "cross_modal_high_priority"
        elif support >= 4 and score >= 1.0 and contradiction <= 2:
            tier = "cross_modal_review_priority"
        elif front >= 0.5 and kinetic >= 0.5 and contradiction <= 2:
            tier = "front_kinetic_consistent"
        elif rollout >= 0.5 and mode >= 0.5 and contradiction <= 2:
            tier = "rollout_mode_consistent"
        elif contradiction >= 3:
            tier = "discordant_guardrail"
        else:
            tier = "routine_or_low_consistency"
        tiers.append(tier)

        if bool(row.get("manual_qc_accepted_candidate", False)):
            readiness.append("manual_qc_accepted_recompute_required")
        else:
            readiness.append("manual_qc_required_no_physics_claim")
        reasons.append(";".join(reason_bits) if reason_bits else "limited cross-modal agreement")

    out["physics_consistency_tier"] = tiers
    out["claim_readiness"] = readiness
    out["physics_consistency_reason"] = reasons
    out["recommended_claim_language"] = np.where(
        out["manual_qc_accepted_candidate"],
        "manual label present; recompute gated front/effect tables before claim",
        "review-prioritization only; do not claim calibrated diffusion or validated degradation mode",
    )
    return out


def mann_whitney_p(x: np.ndarray, y: np.ndarray) -> float | None:
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return None


def event_tests(df: pd.DataFrame, features: List[str], rng: np.random.Generator, n_perm: int) -> pd.DataFrame:
    rows = []
    role = df.get("cohort_role", pd.Series("", index=df.index)).astype(str)
    for feature in features:
        vals = pd.to_numeric(df.get(feature), errors="coerce")
        mask = vals.notna() & role.isin(["event", "control"])
        if mask.sum() < 6:
            continue
        x = vals[mask & role.eq("event")].to_numpy(float)
        y = vals[mask & role.eq("control")].to_numpy(float)
        if len(x) < 2 or len(y) < 2:
            continue
        obs = float(np.nanmedian(x) - np.nanmedian(y))
        pooled = vals[mask].to_numpy(float)
        labels = role[mask].to_numpy()
        null = []
        for _ in range(n_perm):
            shuf = rng.permutation(labels)
            null.append(float(np.nanmedian(pooled[shuf == "event"]) - np.nanmedian(pooled[shuf == "control"])))
        null_arr = np.asarray(null)
        p_emp = float((np.sum(np.abs(null_arr) >= abs(obs)) + 1) / (len(null_arr) + 1)) if len(null_arr) else None
        rows.append(
            {
                "feature": feature,
                "n_event": int(len(x)),
                "n_control": int(len(y)),
                "event_median": float(np.nanmedian(x)),
                "control_median": float(np.nanmedian(y)),
                "median_event_minus_control": obs,
                "mannwhitney_p": mann_whitney_p(x, y),
                "permutation_p": p_emp,
                "null_p95_abs": float(np.nanpercentile(np.abs(null_arr), 95)) if len(null_arr) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("permutation_p", na_position="last") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/physics_consistency_claim_matrix")
    parser.add_argument("--n-permutation", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = read_csv(derived / "within_cycle_echem_shape_audit" / "within_cycle_echem_roi_joined.csv")
    if base.empty:
        base = read_csv(derived / "roi_trace_fusion_audit" / "roi_trace_fusion_joined.csv")
    if base.empty:
        raise FileNotFoundError("No ROI multimodal table found.")

    precursor = read_csv(derived / "precursor_informed_roi_review" / "precursor_informed_roi_review_manifest.csv")
    manual = read_csv(derived / "manual_qc_label_workbook" / "manual_qc_label_template.csv")
    calibration_risk = read_json(derived / "calibration_claim_risk_register" / "calibration_claim_risk_summary.json")
    calibration = calibration_risk.get("calibration_evidence", {})

    df = base.copy()
    if "roi_id" not in df.columns:
        raise ValueError("Base ROI table lacks roi_id.")
    if not precursor.empty:
        precursor_cols = [
            "roi_id",
            "precursor_informed_review_score",
            "precursor_review_tier",
            "precursor_review_reason",
            "score_component_precursor_context",
            "precursor_signed_severity_mean",
            "precursor_signed_severity_sum",
            "precursor_min_p",
            "top_precursor_terms",
            "auto_review_flags",
        ]
        keep = [c for c in precursor_cols if c in precursor.columns]
        df = df.merge(precursor[keep].drop_duplicates("roi_id"), on="roi_id", how="left", suffixes=("", "_precursor"))
    if not manual.empty:
        manual_cols = [
            "roi_id",
            "review_sources",
            "review_priority_tier",
            "combined_review_priority_score",
            "recommended_review_reason",
            "manual_qc_status",
            "manual_qc_decision",
            "manual_particle_identity_ok",
            "manual_front_mask_ok",
            "manual_diffusion_interpretable",
            "primary_qc_png",
            "control_balanced_qc_png",
            "front_tracking_plot_path",
            "rollout_preview_path",
        ]
        keep = [c for c in manual_cols if c in manual.columns]
        df = df.merge(manual[keep].drop_duplicates("roi_id"), on="roi_id", how="left", suffixes=("", "_manual"))

    for col in ["manual_qc_status", "manual_qc_decision", "auto_review_flags"]:
        dup = f"{col}_manual" if f"{col}_manual" in df.columns else f"{col}_precursor"
        if col not in df.columns and dup in df.columns:
            df[col] = df[dup]
        elif col in df.columns and dup in df.columns:
            df[col] = df[col].fillna(df[dup])

    df = add_pillar_scores(df)
    df = classify_rows(df, calibration)
    df = df.sort_values(["physics_consistency_score", "physics_pillar_support_count"], ascending=False).reset_index(drop=True)
    df.insert(0, "physics_consistency_rank", np.arange(1, len(df) + 1))

    pillar_cols = [f"{p}_score" for p in PILLARS]
    feature_cols = [
        "physics_consistency_rank",
        "roi_id",
        "cohort_role",
        "cycleNo",
        "event_reference_cycle",
        "physics_consistency_tier",
        "claim_readiness",
        "physics_consistency_score",
        "physics_pillar_support_count",
        "physics_pillar_strong_support_count",
        "physics_pillar_contradiction_count",
        *pillar_cols,
        "mode_label",
        "manual_qc_status",
        "manual_qc_decision",
        "review_sources",
        "auto_review_flags",
        "precursor_review_tier",
        "physics_consistency_reason",
        "recommended_claim_language",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    matrix_path = out / "physics_consistency_claim_matrix.csv"
    df[feature_cols].to_csv(matrix_path, index=False)

    cycle = (
        df.groupby("cycleNo", dropna=False)
        .agg(
            n_roi=("roi_id", "count"),
            n_event=("cohort_role", lambda s: int((s == "event").sum())),
            median_consistency=("physics_consistency_score", "median"),
            max_consistency=("physics_consistency_score", "max"),
            median_support=("physics_pillar_support_count", "median"),
            top_tier=("physics_consistency_tier", lambda s: s.value_counts().index[0] if len(s) else ""),
        )
        .reset_index()
        .sort_values("max_consistency", ascending=False)
    )
    cycle_path = out / "physics_consistency_cycle_summary.csv"
    cycle.to_csv(cycle_path, index=False)

    tier_summary = (
        df.groupby(["physics_consistency_tier", "claim_readiness"], dropna=False)
        .agg(
            n_roi=("roi_id", "count"),
            n_event=("cohort_role", lambda s: int((s == "event").sum())),
            median_score=("physics_consistency_score", "median"),
            median_support=("physics_pillar_support_count", "median"),
        )
        .reset_index()
        .sort_values(["median_score", "n_roi"], ascending=False)
    )
    tier_path = out / "physics_consistency_tier_summary.csv"
    tier_summary.to_csv(tier_path, index=False)

    rng = np.random.default_rng(args.seed)
    test_features = ["physics_consistency_score", "physics_pillar_support_count", *pillar_cols]
    tests = event_tests(df, test_features, rng, args.n_permutation)
    tests_path = out / "physics_consistency_event_tests.csv"
    tests.to_csv(tests_path, index=False)

    top_rows = df[feature_cols].head(15).to_dict("records")
    summary = {
        "n_roi": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()) if "cycleNo" in df else 0,
        "tier_counts": clean_json(df["physics_consistency_tier"].value_counts().to_dict()),
        "claim_readiness_counts": clean_json(df["claim_readiness"].value_counts().to_dict()),
        "manual_qc_accepted": int(df["manual_qc_accepted_candidate"].sum()),
        "calibration_evidence": clean_json(calibration),
        "top_consistency_rows": clean_json(top_rows),
        "top_event_tests": clean_json(tests.head(12).to_dict("records")) if not tests.empty else [],
        "guardrail": "This matrix is a multimodal consistency and review-prioritization audit. It does not assign manual QC labels and does not validate calibrated diffusion or material degradation mechanisms.",
        "outputs": {
            "matrix": str(matrix_path),
            "cycle_summary": str(cycle_path),
            "tier_summary": str(tier_path),
            "event_tests": str(tests_path),
            "summary": str(out / "physics_consistency_claim_matrix_summary.json"),
        },
    }
    with (out / "physics_consistency_claim_matrix_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)

    lines = [
        "# Physics Consistency Claim Matrix",
        "",
        "Multimodal ROI-level consistency audit across front direction, optical change, rollout residuals, kinetics, precursor context, echem-shape context, and residual mode taxonomy.",
        "",
        f"- ROI rows scored: {summary['n_roi']}",
        f"- Cycles represented: {summary['n_cycles']}",
        f"- Tier counts: {summary['tier_counts']}",
        f"- Claim readiness counts: {summary['claim_readiness_counts']}",
        f"- Manual-QC accepted rows: {summary['manual_qc_accepted']}",
        f"- Calibration evidence: {calibration.get('h5_files_with_camera_timing', 0)}/{calibration.get('h5_files_scanned', 0)} HDF5 timing files; spatial status {calibration.get('spatial_status', 'unknown')}",
        "",
        "## Top Candidates",
        "",
    ]
    for row in top_rows[:8]:
        lines.append(
            f"- {row.get('roi_id')} ({row.get('cohort_role')}, cycle {row.get('cycleNo')}): "
            f"score {row.get('physics_consistency_score'):.3f}, support {row.get('physics_pillar_support_count')}, "
            f"tier {row.get('physics_consistency_tier')}; {row.get('physics_consistency_reason')}"
        )
    lines.extend(["", "## Interpretation", "", summary["guardrail"]])
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
