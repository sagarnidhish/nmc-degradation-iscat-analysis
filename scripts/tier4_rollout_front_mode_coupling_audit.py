#!/usr/bin/env python3
"""Couple particle-only rollout residuals to front, transport, and mode physics.

The history/fallback rollout ablation says whether particle-only prediction is
possible under robust masks. This audit asks what those residuals mean: do they
track front/phase/transport descriptors and unsupervised degradation modes after
source centering, or are they mostly acquisition/mask artifacts?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


TARGETS = {
    "future8": "future_any_drop_within_8cycles",
    "future16": "future_any_drop_within_16cycles",
    "near_pre": "near_pre_flag",
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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def source_residual(df: pd.DataFrame, col: str) -> pd.Series:
    x = numeric(df, col)
    return x - x.groupby(df["source_stem"].astype(str)).transform("mean")


def orient_auc(y: pd.Series, x: pd.Series) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
        return {}
    direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
    score = xx if direction == "higher_in_positive" else -xx
    try:
        _, p_mwu = mannwhitneyu(xx[yy == 1], xx[yy == 0], alternative="two-sided")
    except ValueError:
        p_mwu = np.nan
    rho = spearmanr(yy, score)
    return {
        "n": int(valid.sum()),
        "n_positive": int(yy.sum()),
        "direction": direction,
        "oriented_auc": float(roc_auc_score(yy, score)),
        "average_precision": float(average_precision_score(yy, score)),
        "median_positive_minus_negative": float(xx[yy == 1].median() - xx[yy == 0].median()),
        "mwu_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
        "spearman_rho": float(rho.statistic) if np.isfinite(rho.statistic) else np.nan,
        "spearman_p": float(rho.pvalue) if np.isfinite(rho.pvalue) else np.nan,
    }


def feature_tests(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target_name, target_col in TARGETS.items():
        if target_col not in df.columns:
            continue
        y = numeric(df, target_col).fillna(0).astype(int)
        for feature in features:
            for transform in ["raw", "source_residual"]:
                x = numeric(df, feature)
                if transform == "source_residual":
                    x = source_residual(df, feature)
                result = orient_auc(y, x)
                if result:
                    rows.append({"target": target_name, "feature": feature, "transform": transform, **result})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mwu_p", "oriented_auc", "average_precision"], ascending=[True, False, False])
    return out


def pair_correlations(df: pd.DataFrame, rollout_features: Iterable[str], physics_features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rx in rollout_features:
        for py in physics_features:
            for transform in ["raw", "source_residual"]:
                x = numeric(df, rx)
                y = numeric(df, py)
                if transform == "source_residual":
                    x = source_residual(df, rx)
                    y = source_residual(df, py)
                valid = x.notna() & y.notna()
                if valid.sum() < 12 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
                    continue
                rho = spearmanr(x[valid], y[valid])
                rows.append({
                    "rollout_feature": rx,
                    "physics_feature": py,
                    "transform": transform,
                    "n": int(valid.sum()),
                    "spearman_rho": float(rho.statistic) if np.isfinite(rho.statistic) else np.nan,
                    "spearman_p": float(rho.pvalue) if np.isfinite(rho.pvalue) else np.nan,
                    "abs_rho": abs(float(rho.statistic)) if np.isfinite(rho.statistic) else np.nan,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["spearman_p", "abs_rho"], ascending=[True, False])
    return out


def mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "mode" not in df.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for mode, sub in df.groupby("mode", dropna=False):
        rows.append({
            "mode": mode,
            "n_roi": int(len(sub)),
            "n_sources": int(sub["source_stem"].nunique()),
            "near_pre_fraction": float(numeric(sub, "near_pre_flag").mean()),
            "future8_fraction": float(numeric(sub, "future_any_drop_within_8cycles").mean()),
            "future16_fraction": float(numeric(sub, "future_any_drop_within_16cycles").mean()),
            "median_fallback": float(numeric(sub, "fallback_frame_fraction").median()),
            "median_one_step_hybrid_mse": float(numeric(sub, "one_step_hybrid_mse").median()),
            "median_latent_gain_history": float(numeric(sub, "latent_linear_gain_vs_persistence_history").median()),
            "median_transport_score": float(numeric(sub, "transport_mechanism_score").median()),
            "median_front_kinetic_score": float(numeric(sub, "front_kinetic_score").median()),
            "median_q65_phase_delta": float(numeric(sub, "q65_phase_fraction_delta").median()),
        })
    return pd.DataFrame(rows).sort_values(["near_pre_fraction", "future8_fraction"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/rollout_front_mode_coupling_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollout = pd.read_csv(derived / "history_fallback_masked_rollout_ablation" / "history_fallback_masked_rollout_ablation_per_roi.csv")
    dossier = pd.read_csv(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_dossier.csv")
    phase = pd.read_csv(derived / "source_balanced_pre_event_phase_kinetics_audit" / "source_balanced_pre_event_phase_kinetics_features.csv")
    modes = pd.read_csv(derived / "source_balanced_pre_event_physics_mode_taxonomy" / "source_balanced_pre_event_physics_mode_assignments.csv")

    keep_dossier = [
        "roi_id", "transport_review_rank", "transport_mechanism_score", "transport_source_residual_score",
        "front_kinetic_score", "observable_tail_score", "qc_review_score", "abs_radial_flow_mean",
        "abs_radial_flow_mean_source_residual", "particle_flow_mag_mean", "curl_mean",
        "front_kinetic_concordance_score", "strict_qc_priority_score", "front_evidence_score",
        "kinetic_evidence_score", "qc_evidence_score", "front_consensus_score",
        "front_radius2_slope_px2_per_norm_time", "has_visual_assets", "near_pre_flag", "visual_asset_flag",
    ]
    keep_phase = [
        "roi_id", "capacity_fraction_of_first", "coulombic_inefficiency_pct", "echem_regime_pc1",
        "echem_regime_pc2", "masked_minus_bg_delta", "masked_minus_bg_slope",
        "q55_phase_fraction_delta", "q55_phase_fraction_slope", "q55_logistic_r2",
        "q65_phase_fraction_delta", "q65_phase_fraction_slope", "q65_logistic_r2",
        "q75_phase_fraction_delta", "q75_phase_fraction_slope", "q75_logistic_r2",
        "front_radius_slope_px_per_norm_time_source_echem_context_residual",
        "front_radius2_slope_px2_per_norm_time_source_echem_context_residual",
        "phase_fraction_slope_per_norm_time_source_echem_context_residual",
    ]
    keep_modes = ["roi_id", "mode", "pre_event_clock_closer", "is_pre_event", "is_near_pre", "is_mid_pre", "is_far_pre", "is_post_event", "is_control"]

    merged = rollout.merge(dossier[[c for c in keep_dossier if c in dossier.columns]], on="roi_id", how="left")
    merged = merged.merge(phase[[c for c in keep_phase if c in phase.columns]], on="roi_id", how="left")
    merged = merged.merge(modes[[c for c in keep_modes if c in modes.columns]], on="roi_id", how="left")

    rollout_features = [
        "fallback_frame_fraction",
        "one_step_hybrid_mse",
        "one_step_hybrid_minus_adaptive_mse",
        "persistence_rollout_history_mse",
        "persistence_rollout_hybrid_mse",
        "latent_linear_gain_vs_persistence_history",
        "latent_linear_gain_vs_persistence_hybrid",
        "late_minus_early_one_step_hybrid_mse",
    ]
    physics_features = [
        "transport_mechanism_score",
        "transport_source_residual_score",
        "front_kinetic_score",
        "observable_tail_score",
        "qc_review_score",
        "abs_radial_flow_mean",
        "abs_radial_flow_mean_source_residual",
        "front_kinetic_concordance_score",
        "front_evidence_score",
        "kinetic_evidence_score",
        "masked_minus_bg_delta",
        "masked_minus_bg_slope",
        "q55_phase_fraction_delta",
        "q65_phase_fraction_delta",
        "q75_phase_fraction_delta",
        "q65_phase_fraction_slope",
        "front_radius2_slope_px2_per_norm_time",
        "front_radius2_slope_px2_per_norm_time_source_echem_context_residual",
    ]
    score_features = rollout_features + physics_features
    tests = feature_tests(merged, [c for c in score_features if c in merged.columns])
    correlations = pair_correlations(merged, [c for c in rollout_features if c in merged.columns], [c for c in physics_features if c in merged.columns])
    modes_out = mode_summary(merged)

    for col in ["one_step_hybrid_mse", "latent_linear_gain_vs_persistence_history", "transport_mechanism_score", "front_kinetic_score", "fallback_frame_fraction"]:
        if col in merged.columns:
            merged[f"{col}_source_residual"] = source_residual(merged, col)
    merged["rollout_front_review_score"] = (
        numeric(merged, "one_step_hybrid_mse_source_residual").rank(pct=True)
        + numeric(merged, "latent_linear_gain_vs_persistence_history_source_residual").rank(pct=True)
        + numeric(merged, "transport_mechanism_score_source_residual").rank(pct=True)
        + numeric(merged, "front_kinetic_score_source_residual").rank(pct=True)
        - numeric(merged, "fallback_frame_fraction_source_residual").abs().rank(pct=True) * 0.25
    )
    review_cols = [
        "rollout_front_review_score", "roi_id", "cycleNo", "source_stem", "event_relative_bin",
        "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "mode",
        "one_step_hybrid_mse", "latent_linear_gain_vs_persistence_history", "fallback_frame_fraction",
        "transport_mechanism_score", "front_kinetic_score", "q65_phase_fraction_delta",
        "has_visual_assets",
    ]
    review = merged.sort_values("rollout_front_review_score", ascending=False)[[c for c in review_cols if c in merged.columns]].head(40)

    paths = {
        "merged": out / "rollout_front_mode_coupling_merged.csv",
        "feature_tests": out / "rollout_front_mode_coupling_feature_tests.csv",
        "correlations": out / "rollout_front_mode_coupling_correlations.csv",
        "mode_summary": out / "rollout_front_mode_coupling_mode_summary.csv",
        "review_queue": out / "rollout_front_mode_coupling_review_queue.csv",
        "summary": out / "rollout_front_mode_coupling_summary.json",
    }
    merged.to_csv(paths["merged"], index=False)
    tests.to_csv(paths["feature_tests"], index=False)
    correlations.to_csv(paths["correlations"], index=False)
    modes_out.to_csv(paths["mode_summary"], index=False)
    review.to_csv(paths["review_queue"], index=False)

    top_source_resid_corr = correlations[correlations["transform"].eq("source_residual")].head(12).to_dict("records") if not correlations.empty else []
    top_raw_corr = correlations[correlations["transform"].eq("raw")].head(12).to_dict("records") if not correlations.empty else []
    top_tests = tests.head(16).to_dict("records") if not tests.empty else []
    summary = {
        "overall_status": "rollout_front_mode_coupling_audit_ready",
        "n_rows": int(len(merged)),
        "n_sources": int(merged["source_stem"].nunique()),
        "n_cycles": int(merged["cycleNo"].nunique()),
        "n_modes": int(merged["mode"].nunique()) if "mode" in merged else 0,
        "top_feature_tests": clean_json(top_tests),
        "top_source_residual_correlations": clean_json(top_source_resid_corr),
        "top_raw_correlations": clean_json(top_raw_corr),
        "mode_summary": clean_json(modes_out.to_dict("records")) if not modes_out.empty else [],
        "top_review_queue": clean_json(review.head(12).to_dict("records")),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": (
            "This audit couples automatic particle-only rollout residuals to automatic front/phase/transport/mode descriptors. "
            "It nominates physically interpretable review rows and source-robust associations, but does not validate manual phase boundaries or calibrated diffusion coefficients."
        ),
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    readme = [
        "# Rollout Front/Mode Coupling Audit",
        "",
        f"- Rows / sources / cycles / modes: {summary['n_rows']} / {summary['n_sources']} / {summary['n_cycles']} / {summary['n_modes']}",
        f"- Top feature test: {top_tests[0] if top_tests else 'NA'}",
        f"- Top source-residual correlation: {top_source_resid_corr[0] if top_source_resid_corr else 'NA'}",
        "",
        "Outputs:",
        *[f"- `{path.name}`" for path in paths.values()],
        "",
        "Guardrail:",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(readme).rstrip() + "\n")
    summary["outputs"]["readme"] = str(out / "README.md")
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
