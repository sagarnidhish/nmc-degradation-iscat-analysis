#!/usr/bin/env python3
"""Cycle-level early-warning audit from masked particle-rollout residuals.

The masked rollout audit showed that prediction errors inside the accepted
particle support are more informative than full-crop errors. This script asks
whether those particle-local residuals line up with, or precede, cycle-level
abrupt-drop labels from the larger particle-trace table.

The cohort is small and ROI-cycle selected, so outputs are guardrail statistics
and ranked hypotheses, not a deployable early-warning detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


TARGETS = [
    "any_abrupt_drop",
    "synchronized_drop_2plus",
    "future_any_drop_within_4cycles",
    "future_any_drop_within_8cycles",
    "future_any_drop_within_16cycles",
    "future_sync2_drop_within_4cycles",
    "future_sync2_drop_within_8cycles",
    "future_sync2_drop_within_16cycles",
]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
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


def finite_float(value: Any, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_mwu(pos: Iterable[float], neg: Iterable[float]) -> Dict[str, float]:
    pp = pd.to_numeric(pd.Series(pos), errors="coerce").dropna()
    nn = pd.to_numeric(pd.Series(neg), errors="coerce").dropna()
    if len(pp) < 2 or len(nn) < 2:
        return {
            "n_positive": int(len(pp)),
            "n_negative": int(len(nn)),
            "median_positive": np.nan,
            "median_negative": np.nan,
            "median_positive_minus_negative": np.nan,
            "mannwhitney_p": np.nan,
        }
    try:
        _, p = mannwhitneyu(pp, nn, alternative="two-sided")
    except Exception:
        p = np.nan
    return {
        "n_positive": int(len(pp)),
        "n_negative": int(len(nn)),
        "median_positive": float(pp.median()),
        "median_negative": float(nn.median()),
        "median_positive_minus_negative": float(pp.median() - nn.median()),
        "mannwhitney_p": float(p) if np.isfinite(p) else np.nan,
    }


def permutation_p_binary(df: pd.DataFrame, feature: str, target: str, observed: float, n_perm: int, seed: int) -> float:
    if not np.isfinite(observed):
        return np.nan
    rng = np.random.default_rng(seed)
    x = pd.to_numeric(df[feature], errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    ok = x.notna() & y.notna()
    x = x[ok].to_numpy(dtype=float)
    y = y[ok].to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or len(x) < 5:
        return np.nan
    stats = []
    for _ in range(n_perm):
        yp = y.copy()
        rng.shuffle(yp)
        pos = x[yp == 1]
        neg = x[yp == 0]
        if len(pos) < 2 or len(neg) < 2:
            continue
        stats.append(float(np.nanmedian(pos) - np.nanmedian(neg)))
    arr = np.asarray(stats, dtype=float)
    if arr.size == 0:
        return np.nan
    return float((np.sum(np.abs(arr) >= abs(observed)) + 1) / (arr.size + 1))


def permutation_p_spearman(df: pd.DataFrame, x_col: str, y_col: str, observed: float, n_perm: int, seed: int) -> float:
    if not np.isfinite(observed):
        return np.nan
    rng = np.random.default_rng(seed)
    tmp = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 5 or tmp[x_col].nunique() < 2 or tmp[y_col].nunique() < 2:
        return np.nan
    x = tmp[x_col].to_numpy(dtype=float)
    y = tmp[y_col].to_numpy(dtype=float)
    stats = []
    for _ in range(n_perm):
        yp = y.copy()
        rng.shuffle(yp)
        rho, _ = spearmanr(x, yp)
        if np.isfinite(rho):
            stats.append(float(rho))
    arr = np.asarray(stats, dtype=float)
    if arr.size == 0:
        return np.nan
    return float((np.sum(np.abs(arr) >= abs(observed)) + 1) / (arr.size + 1))


def collapse_masked_rollout(per_roi: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, grp in per_roi.groupby("cycleNo", dropna=False):
        base: Dict[str, Any] = {
            "cycleNo": float(cycle),
            "n_roi": int(grp["roi_id"].nunique()),
            "event_roi_fraction": float((grp.drop_duplicates("roi_id")["cohort_role"] == "event").mean()),
            "mean_validation_score": float(pd.to_numeric(grp["validation_score_first"], errors="coerce").mean()),
        }
        for method, mgrp in grp.groupby("method"):
            prefix = str(method)
            for col in [
                "particle_mse_mean",
                "particle_mae_mean",
                "nonparticle_mse_mean",
                "particle_to_nonparticle_mse_ratio_mean",
                "particle_mse_fraction_of_full_mean",
                "mask_fallback_used_mean",
                "accepted_area_fraction_median",
            ]:
                vals = pd.to_numeric(mgrp[col], errors="coerce")
                base[f"{prefix}_{col}_median"] = float(vals.median()) if vals.notna().any() else np.nan
                base[f"{prefix}_{col}_max"] = float(vals.max()) if vals.notna().any() else np.nan
        rgrp = ratios[ratios["cycleNo"] == cycle]
        for method, mgrp in rgrp.groupby("method"):
            vals = pd.to_numeric(mgrp["particle_mse_ratio_vs_persistence"], errors="coerce")
            base[f"{method}_particle_mse_ratio_vs_persistence_median"] = float(vals.median()) if vals.notna().any() else np.nan
            base[f"{method}_particle_mse_ratio_vs_persistence_max"] = float(vals.max()) if vals.notna().any() else np.nan
        rows.append(base)
    return pd.DataFrame(rows).sort_values("cycleNo")


def add_neighbor_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.sort_values("cycleNo").copy()
    for col in feature_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_delta_prev_observed_roi_cycle"] = vals.diff()
        out[f"{col}_rolling2_mean"] = vals.rolling(2, min_periods=1).mean()
    return out


def feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"cycleNo", *TARGETS}
    cols = []
    for col in df.columns:
        if col in excluded or col.startswith("cycles_to_next"):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 6 and vals.nunique(dropna=True) > 2:
            cols.append(col)
    return cols


def target_tests(df: pd.DataFrame, features: List[str], n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    for target in [t for t in TARGETS if t in df.columns]:
        y = pd.to_numeric(df[target], errors="coerce")
        if y.dropna().nunique() < 2:
            continue
        for feat in features:
            x = pd.to_numeric(df[feat], errors="coerce")
            out = safe_mwu(x[y == 1], x[y == 0])
            out.update({
                "target": target,
                "feature": feat,
                "positive_rate": float(y.mean()),
                "permutation_p_abs_median_diff": permutation_p_binary(df, feat, target, out["median_positive_minus_negative"], n_perm, seed + len(rows)),
            })
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["permutation_p_abs_median_diff", "mannwhitney_p"], na_position="last") if rows else pd.DataFrame()


def correlation_tests(df: pd.DataFrame, features: List[str], n_perm: int, seed: int) -> pd.DataFrame:
    context_cols = [c for c in [
        "cycle_state_pc2",
        "degradation_state_axis",
        "mean_abs_delta_prev",
        "particle_norm_range",
        "capacity_mAh",
        "coulombic_efficiency_pct",
        "frames_percentile",
        "cycles_to_next_drop_within_8",
        "cycles_to_next_drop_within_16",
    ] if c in df.columns]
    rows = []
    for feat in features:
        for ctx in context_cols:
            tmp = df[[feat, ctx]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(tmp) < 6 or tmp[feat].nunique() < 2 or tmp[ctx].nunique() < 2:
                continue
            rho, p = spearmanr(tmp[feat], tmp[ctx])
            rows.append({
                "feature": feat,
                "context": ctx,
                "n": int(len(tmp)),
                "rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "permutation_p_abs_rho": permutation_p_spearman(df, feat, ctx, float(rho), n_perm, seed + len(rows)) if np.isfinite(rho) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["permutation_p_abs_rho", "p_value"], na_position="last") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-per-roi", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_roi_rollout_audit/masked_roi_rollout_per_roi_metrics.csv")
    parser.add_argument("--masked-ratios", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_roi_rollout_audit/masked_roi_rollout_method_ratios.csv")
    parser.add_argument("--trace-cycles", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_trace_physics_audit/particle_trace_cycle_features.csv")
    parser.add_argument("--cycle-state", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_state_space_transition_audit/cycle_state_space_table.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_rollout_cycle_warning")
    parser.add_argument("--n-permutation", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_roi = pd.read_csv(args.masked_per_roi)
    ratios = pd.read_csv(args.masked_ratios)
    trace = pd.read_csv(args.trace_cycles)
    cycle = collapse_masked_rollout(per_roi, ratios)
    trace["cycleNo"] = pd.to_numeric(trace["cycleNo"], errors="coerce")
    joined = cycle.merge(trace, how="left", on="cycleNo", suffixes=("", "_trace"))
    if Path(args.cycle_state).exists():
        state = pd.read_csv(args.cycle_state)
        state_cols = [c for c in ["cycleNo", "cycle_state_pc1", "cycle_state_pc2", "degradation_state_axis", "cycle_state_cluster"] if c in state.columns]
        joined = joined.merge(state[state_cols].drop_duplicates("cycleNo"), how="left", on="cycleNo", suffixes=("", "_state"))

    base_features = feature_columns(joined)
    rollout_feature_subset = [c for c in base_features if any(key in c for key in ["particle_mse", "nonparticle_mse", "particle_to_nonparticle", "mask_fallback", "accepted_area", "validation_score", "event_roi_fraction"])]
    joined = add_neighbor_features(joined, rollout_feature_subset)
    features = feature_columns(joined)
    # Keep focus on masked-rollout features and their observed-ROI-cycle deltas.
    features = [c for c in features if c in rollout_feature_subset or c.endswith("_delta_prev_observed_roi_cycle") or c.endswith("_rolling2_mean")]

    tests = target_tests(joined, features, args.n_permutation, args.seed)
    corrs = correlation_tests(joined, features, args.n_permutation, args.seed)
    ranked = joined.copy()
    score_parts = []
    for col in [
        "low_rank_dmd_particle_mse_mean_median",
        "low_rank_dmd_particle_mse_ratio_vs_persistence_median",
        "persistence_particle_mse_fraction_of_full_mean_median",
        "low_rank_dmd_particle_to_nonparticle_mse_ratio_mean_median",
    ]:
        if col in ranked.columns:
            score_parts.append(pd.to_numeric(ranked[col], errors="coerce").rank(pct=True))
    ranked["masked_rollout_cycle_warning_score"] = np.sum(score_parts, axis=0) if score_parts else np.nan
    ranked = ranked.sort_values("masked_rollout_cycle_warning_score", ascending=False, na_position="last")

    joined.to_csv(out / "masked_rollout_cycle_warning_joined.csv", index=False)
    tests.to_csv(out / "masked_rollout_cycle_warning_target_tests.csv", index=False)
    corrs.to_csv(out / "masked_rollout_cycle_warning_correlations.csv", index=False)
    ranked.to_csv(out / "masked_rollout_cycle_warning_ranked_cycles.csv", index=False)

    summary = {
        "n_roi_cycles": int(len(joined)),
        "n_rollout_features_tested": int(len(features)),
        "n_permutation": int(args.n_permutation),
        "cycle_min": finite_float(joined["cycleNo"].min()),
        "cycle_max": finite_float(joined["cycleNo"].max()),
        "target_positive_counts": {t: int(pd.to_numeric(joined[t], errors="coerce").fillna(0).sum()) for t in TARGETS if t in joined.columns},
        "top_target_tests": clean_json(tests.head(18).to_dict("records")) if not tests.empty else [],
        "top_correlations": clean_json(corrs.head(18).to_dict("records")) if not corrs.empty else [],
        "top_warning_cycles": clean_json(ranked.head(12).to_dict("records")),
        "guardrail": "Cycle-level audit from selected masked ROI rollout residuals; small selected cycle set, no deployable warning model.",
    }
    with (out / "masked_rollout_cycle_warning_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2)

    with (out / "README.md").open("w") as f:
        f.write("# Masked Rollout Cycle Warning Audit\n\n")
        f.write("This folder collapses masked particle-rollout residuals to observed ROI cycles and joins larger particle-trace future-drop labels.\n\n")
        f.write(f"- ROI cycles: {summary['n_roi_cycles']}\n")
        f.write(f"- Rollout-derived features tested: {summary['n_rollout_features_tested']}\n")
        f.write(f"- Permutations per test: {summary['n_permutation']}\n\n")
        f.write("Outputs:\n\n")
        f.write("- `masked_rollout_cycle_warning_joined.csv`: cycle-level masked-rollout features plus trace targets.\n")
        f.write("- `masked_rollout_cycle_warning_target_tests.csv`: binary target tests with permutation p-values.\n")
        f.write("- `masked_rollout_cycle_warning_correlations.csv`: correlations with trace/state context.\n")
        f.write("- `masked_rollout_cycle_warning_ranked_cycles.csv`: cycles ranked by masked rollout warning score.\n")
        f.write("- `masked_rollout_cycle_warning_summary.json`: compact summary.\n")

    print(json.dumps(clean_json({
        "n_roi_cycles": summary["n_roi_cycles"],
        "target_positive_counts": summary["target_positive_counts"],
        "top_target_tests": summary["top_target_tests"][:5],
    }), indent=2))


if __name__ == "__main__":
    main()
