#!/usr/bin/env python3
"""Exact optical-feature mechanism consistency audit.

The source-invariant exact-feature audit nominated low positive particle-vs-context
frame-to-frame change as a future16 descriptor. This script joins that descriptor
back to front, rollout, temporal, echem, and diffusion-consistency outputs to ask
whether it behaves like optical loss/contraction rather than a clean expanding
front or calibrated diffusion signal.
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
from sklearn.preprocessing import StandardScaler

TARGET = "future_any_drop_within_16cycles"
PRIMARY_FEATURES = [
    "particle_vs_context_mean_diff_positive_fraction",
    "particle_std_diff_positive_fraction",
    "particle_mean_last_minus_first",
    "particle_gradient_diff_q90",
]
MECHANISM_FEATURES = [
    "phase_slope_median_per_s",
    "phase_slope_positive_fraction",
    "radius2_slope_median_px2_per_s",
    "radius2_slope_positive_fraction",
    "diffusion_proxy_median_um2_per_s",
    "diffusion_proxy_abs_median_um2_per_s",
    "threshold_robust_phase_score",
    "threshold_robust_diffusion_score",
    "q70_radius2_slope_bootstrap_p50_px2_per_s",
    "positive_D_fraction",
    "negative_D_fraction",
    "median_apparent_D_um2_per_s",
    "median_abs_apparent_D_um2_per_s",
    "median_radius2_fit_r2",
    "threshold_sensitivity_iqr_over_median_abs",
    "diffusion_physics_gate_count",
    "physics_consistency_score",
    "transferred_masked_residual_signature",
    "low_rank_dmd_particle_mse_fraction_of_full_mean",
    "low_rank_dmd_particle_to_nonparticle_mse_ratio_mean",
    "persistence_particle_mse_fraction_of_full_mean",
    "persistence_particle_to_nonparticle_mse_ratio_mean",
    "roi_norm_mean_delta_last_minus_first",
    "object_mean_residual",
    "object_mean_abs_z",
    "stage_drift_xy_sampled",
    "mask_instability_score",
    "fallback_frame_fraction",
    "accepted_area_cv",
    "capacity_fraction_of_first",
    "state_step_norm",
    "axis_step",
    "frames_percentile",
    "cycle_index_rank",
]


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
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def suffix_except(df: pd.DataFrame, keep: Iterable[str], suffix: str) -> pd.DataFrame:
    keep_set = set(keep)
    rename = {c: f"{c}{suffix}" for c in df.columns if c not in keep_set}
    return df.rename(columns=rename)


def coalesce_columns(df: pd.DataFrame, base: str) -> pd.Series:
    cols = [c for c in [base, f"{base}_temporal", f"{base}_diffusion", f"{base}_echemfront"] if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    out = pd.to_numeric(df[cols[0]], errors="coerce")
    for col in cols[1:]:
        out = out.combine_first(pd.to_numeric(df[col], errors="coerce"))
    return out


def build_joined(derived: Path) -> pd.DataFrame:
    fusion = read_csv(derived / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_joined.csv")
    keys = ["roi_id", "cycleNo"]
    temporal = read_csv(derived / "temporal_directionality_physics_audit" / "temporal_directionality_joined.csv")
    diffusion = read_csv(derived / "diffusion_physics_consistency_audit" / "diffusion_physics_consistency_roi_scores.csv")
    efront = read_csv(derived / "echem_conditioned_roi_rollout_front_audit" / "echem_conditioned_roi_rollout_front_joined.csv")

    base_cols = list(dict.fromkeys(keys + ["embedding_row_id", "source_stem", "cohort_role", "selection_subrole", TARGET, "future_any_drop_within_8cycles", "any_abrupt_drop"] + PRIMARY_FEATURES + [c for c in fusion.columns if c.startswith("particle_norm_")]))
    out = fusion[[c for c in base_cols if c in fusion.columns]].copy()

    temporal_cols = [c for c in keys + ["source_stem", "cohort_role", "selection_subrole"] + MECHANISM_FEATURES if c in temporal.columns]
    out = out.merge(suffix_except(temporal[temporal_cols].drop_duplicates(keys), keys, "_temporal"), on=keys, how="left")
    diffusion_cols = [c for c in keys + ["source_stem", "cohort_role"] + MECHANISM_FEATURES if c in diffusion.columns]
    out = out.merge(suffix_except(diffusion[diffusion_cols].drop_duplicates(keys), keys, "_diffusion"), on=keys, how="left")
    efront_cols = [c for c in keys + ["source_stem", "cohort_role", "selection_subrole"] + MECHANISM_FEATURES if c in efront.columns]
    out = out.merge(suffix_except(efront[efront_cols].drop_duplicates(keys), keys, "_echemfront"), on=keys, how="left")

    for col in ["source_stem", "cohort_role", "selection_subrole"] + MECHANISM_FEATURES:
        if col not in out.columns:
            out[col] = coalesce_columns(out, col)
    return out.loc[:, ~out.columns.duplicated()].copy()


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() < 3 or vals.std(skipna=True) == 0:
        return pd.Series(0.0, index=series.index)
    return (vals - vals.mean()) / vals.std()


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["exact_optical_loss_score"] = (
        -zscore(out["particle_vs_context_mean_diff_positive_fraction"])
        -zscore(out["particle_std_diff_positive_fraction"])
        +zscore(out["particle_mean_last_minus_first"])
        +zscore(out["particle_gradient_diff_q90"])
    ) / 4.0
    out["exact_low_context_change_score"] = -zscore(out["particle_vs_context_mean_diff_positive_fraction"])
    out["front_contraction_score"] = (
        -zscore(out["radius2_slope_median_px2_per_s"])
        -zscore(out["radius2_slope_positive_fraction"])
        -zscore(out["positive_D_fraction"])
        +zscore(out["negative_D_fraction"])
    ) / 4.0
    out["diffusion_guardrail_score"] = (
        -zscore(out["median_radius2_fit_r2"])
        +zscore(out["threshold_sensitivity_iqr_over_median_abs"])
        -zscore(out["diffusion_physics_gate_count"])
    ) / 3.0
    out["rollout_residual_score"] = (
        zscore(out["low_rank_dmd_particle_mse_fraction_of_full_mean"])
        +zscore(out["low_rank_dmd_particle_to_nonparticle_mse_ratio_mean"])
        +zscore(out["transferred_masked_residual_signature"])
    ) / 3.0
    return out


def source_eta2(series: pd.Series, sources: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.notna() & sources.notna()
    vals = vals[valid]
    src = sources[valid]
    if vals.nunique() < 2 or src.nunique() < 2:
        return float("nan")
    overall = vals.mean()
    total = float(((vals - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in vals.groupby(src):
        between += len(sub) * float((sub.mean() - overall) ** 2)
    return between / total


def residualize_by_group(series: pd.Series, groups: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    global_mean = vals.mean()
    centered = vals.copy()
    for group, idx in groups.groupby(groups).groups.items():
        centered.loc[idx] = vals.loc[idx] - vals.loc[idx].mean() + global_mean
    return centered


def corr_rows(df: pd.DataFrame, anchors: List[str], features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for anchor in anchors:
        ax = pd.to_numeric(df[anchor], errors="coerce")
        for feat in features:
            if feat not in df.columns or feat == anchor:
                continue
            yy = pd.to_numeric(df[feat], errors="coerce")
            valid = ax.notna() & yy.notna()
            if valid.sum() < 12 or ax[valid].nunique() < 2 or yy[valid].nunique() < 2:
                continue
            rho, p = spearmanr(ax[valid], yy[valid])
            axr = residualize_by_group(ax, df["source_stem"])
            yyr = residualize_by_group(yy, df["source_stem"])
            vr = axr.notna() & yyr.notna()
            rrho, rp = spearmanr(axr[vr], yyr[vr]) if vr.sum() >= 12 and axr[vr].nunique() > 1 and yyr[vr].nunique() > 1 else (np.nan, np.nan)
            rows.append({
                "anchor_feature": anchor,
                "mechanism_feature": feat,
                "n": int(valid.sum()),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(p) if np.isfinite(p) else np.nan,
                "source_residual_spearman_rho": float(rrho) if np.isfinite(rrho) else np.nan,
                "source_residual_spearman_p": float(rp) if np.isfinite(rp) else np.nan,
                "mechanism_source_eta2": source_eta2(yy, df["source_stem"]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["anchor_feature", "spearman_p", "source_residual_spearman_p"])


def target_metric_rows(df: pd.DataFrame, score_cols: List[str], target: str) -> pd.DataFrame:
    y = pd.to_numeric(df[target], errors="coerce")
    rows = []
    for score in score_cols:
        x = pd.to_numeric(df[score], errors="coerce")
        valid = y.isin([0, 1]) & x.notna()
        if valid.sum() < 12 or y[valid].nunique() < 2:
            continue
        yy = y[valid].astype(int)
        xx = x[valid]
        auc = roc_auc_score(yy, xx)
        if auc < 0.5:
            oriented = -xx
            direction = "lower_in_positive"
            auc = 1.0 - auc
        else:
            oriented = xx
            direction = "higher_in_positive"
        pos = xx[yy == 1]
        neg = xx[yy == 0]
        try:
            mwp = mannwhitneyu(pos, neg, alternative="two-sided").pvalue
        except ValueError:
            mwp = np.nan
        rows.append({
            "score": score,
            "target": target,
            "n": int(valid.sum()),
            "n_positive": int(yy.sum()),
            "orientation": direction,
            "oriented_auc": float(auc),
            "average_precision": float(average_precision_score(yy, oriented)),
            "median_positive": float(pos.median()),
            "median_negative": float(neg.median()),
            "median_positive_minus_negative": float(pos.median() - neg.median()),
            "mannwhitney_p": float(mwp) if np.isfinite(mwp) else np.nan,
            "source_eta2": source_eta2(x, df["source_stem"]),
        })
    return pd.DataFrame(rows).sort_values(["target", "oriented_auc"], ascending=[True, False])


def stratum_tests(df: pd.DataFrame, score: str, features: List[str]) -> pd.DataFrame:
    x = pd.to_numeric(df[score], errors="coerce")
    hi_thr = x.quantile(0.75)
    lo_thr = x.quantile(0.25)
    high = x >= hi_thr
    low = x <= lo_thr
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        vals = pd.to_numeric(df[feat], errors="coerce")
        valid_hi = high & vals.notna()
        valid_lo = low & vals.notna()
        if valid_hi.sum() < 4 or valid_lo.sum() < 4:
            continue
        try:
            p = mannwhitneyu(vals[valid_hi], vals[valid_lo], alternative="two-sided").pvalue
        except ValueError:
            p = np.nan
        rows.append({
            "score": score,
            "feature": feat,
            "high_threshold": float(hi_thr),
            "low_threshold": float(lo_thr),
            "n_high": int(valid_hi.sum()),
            "n_low": int(valid_lo.sum()),
            "median_high_score": float(vals[valid_hi].median()),
            "median_low_score": float(vals[valid_lo].median()),
            "median_high_minus_low": float(vals[valid_hi].median() - vals[valid_lo].median()),
            "mannwhitney_p": float(p) if np.isfinite(p) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mannwhitney_p") if rows else pd.DataFrame()


def source_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, sub in df.groupby("source_stem"):
        rows.append({
            "source_stem": src,
            "n_rows": int(len(sub)),
            "n_cycles": int(sub["cycleNo"].nunique()),
            "future16_rate": float(pd.to_numeric(sub[TARGET], errors="coerce").mean()),
            "mean_exact_optical_loss_score": float(pd.to_numeric(sub["exact_optical_loss_score"], errors="coerce").mean()),
            "mean_front_contraction_score": float(pd.to_numeric(sub["front_contraction_score"], errors="coerce").mean()),
            "mean_diffusion_guardrail_score": float(pd.to_numeric(sub["diffusion_guardrail_score"], errors="coerce").mean()),
            "mean_rollout_residual_score": float(pd.to_numeric(sub["rollout_residual_score"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_exact_optical_loss_score", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/exact_feature_mechanism_consistency_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = add_scores(build_joined(derived))
    score_cols = [
        "exact_low_context_change_score",
        "exact_optical_loss_score",
        "front_contraction_score",
        "diffusion_guardrail_score",
        "rollout_residual_score",
    ]
    mechanism_cols = [c for c in list(dict.fromkeys(MECHANISM_FEATURES + score_cols + PRIMARY_FEATURES)) if c in joined.columns]
    anchors = ["exact_low_context_change_score", "exact_optical_loss_score", "particle_vs_context_mean_diff_positive_fraction"]

    correlations = corr_rows(joined, anchors, mechanism_cols)
    target_metrics = target_metric_rows(joined, score_cols + PRIMARY_FEATURES, TARGET)
    strata = stratum_tests(joined, "exact_optical_loss_score", mechanism_cols)
    sources = source_table(joined)

    paths = {
        "joined": out / "exact_feature_mechanism_joined.csv",
        "correlations": out / "exact_feature_mechanism_correlations.csv",
        "target_metrics": out / "exact_feature_mechanism_target_metrics.csv",
        "strata": out / "exact_feature_mechanism_stratum_tests.csv",
        "source_summary": out / "exact_feature_mechanism_source_summary.csv",
        "summary": out / "exact_feature_mechanism_summary.json",
    }
    joined.to_csv(paths["joined"], index=False)
    correlations.to_csv(paths["correlations"], index=False)
    target_metrics.to_csv(paths["target_metrics"], index=False)
    strata.to_csv(paths["strata"], index=False)
    sources.to_csv(paths["source_summary"], index=False)

    contraction_corr = correlations[
        correlations["mechanism_feature"].isin(["front_contraction_score", "radius2_slope_median_px2_per_s", "positive_D_fraction", "diffusion_guardrail_score", "rollout_residual_score"])
    ] if not correlations.empty else pd.DataFrame()
    summary = clean_json({
        "n_rows": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "n_sources": int(joined["source_stem"].nunique()),
        "target": TARGET,
        "primary_features": PRIMARY_FEATURES,
        "score_definitions": {
            "exact_low_context_change_score": "negative z-score of particle_vs_context_mean_diff_positive_fraction",
            "exact_optical_loss_score": "average of low particle-vs-context positive-change, low particle-std positive-change, higher particle mean last-first, and higher gradient diff q90 z-scores",
            "front_contraction_score": "higher values indicate more negative radius2/front-D signs and lower positive-D fraction",
            "diffusion_guardrail_score": "higher values indicate lower fit/gate quality and higher threshold sensitivity",
            "rollout_residual_score": "higher values indicate larger particle-masked rollout residual burden",
        },
        "top_target_metrics": target_metrics.head(20).to_dict("records"),
        "top_mechanism_correlations": correlations.head(40).to_dict("records") if not correlations.empty else [],
        "contraction_related_correlations": contraction_corr.sort_values("spearman_p").head(20).to_dict("records") if not contraction_corr.empty else [],
        "top_stratum_tests": strata.head(30).to_dict("records") if not strata.empty else [],
        "source_summary": sources.to_dict("records"),
        "guardrail": "Mechanism consistency joins automatic particle descriptors to automatic front, rollout, echem, and diffusion-proxy outputs. It tests whether the exact feature behaves like optical loss/contraction, but all front/diffusion quantities remain apparent proxies pending manual QC and calibration.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Exact Feature Mechanism Consistency Audit\n\n"
        "Joins source-invariant exact optical descriptors to front/rollout/diffusion/echem derived tables.\n\n"
        f"- Rows: {summary['n_rows']}\n- Cycles: {summary['n_cycles']}\n- Sources: {summary['n_sources']}\n"
        f"- Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
