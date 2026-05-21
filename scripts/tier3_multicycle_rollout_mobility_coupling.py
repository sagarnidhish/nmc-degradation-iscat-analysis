#!/usr/bin/env python3
"""Couple multi-cycle ROI rollout errors to mobility/degradation descriptors."""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def wide_rollout(per_roi: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["roi_id", "cycleNo", "validation_score"]
    metrics = ["mse", "mae", "psnr", "ssim"]
    rows: List[Dict[str, object]] = []
    for (roi_id, cycle), grp in per_roi.groupby(["roi_id", "cycleNo"], dropna=False):
        row: Dict[str, object] = {"roi_id": roi_id, "cycleNo": cycle}
        if "validation_score" in grp.columns:
            row["validation_score"] = finite_float(grp["validation_score"].iloc[0])
        for _, g in grp.iterrows():
            method = str(g["method"])
            for metric in metrics:
                row[f"{method}_{metric}"] = finite_float(g.get(metric))
        if "persistence_mse" in row and "velocity_mse" in row:
            row["velocity_mse_ratio_vs_persistence"] = row["velocity_mse"] / row["persistence_mse"] if row["persistence_mse"] > 0 else np.nan
        if "persistence_mse" in row and "low_rank_dmd_mse" in row:
            row["dmd_mse_ratio_vs_persistence"] = row["low_rank_dmd_mse"] / row["persistence_mse"] if row["persistence_mse"] > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def feature_tests(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    event = df[df["is_event_roi"] == 1]
    control = df[df["is_event_roi"] == 0]
    for feat in features:
        a = pd.to_numeric(event[feat], errors="coerce").dropna()
        b = pd.to_numeric(control[feat], errors="coerce").dropna()
        if len(a) and len(b):
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        rows.append({
            "feature": feat,
            "event_mean": float(a.mean()) if len(a) else np.nan,
            "control_mean": float(b.mean()) if len(b) else np.nan,
            "event_minus_control": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
            "n_event": int(len(a)),
            "n_control": int(len(b)),
            "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


def spearman_table(df: pd.DataFrame, x_cols: List[str], y_cols: List[str]) -> pd.DataFrame:
    rows = []
    for x in x_cols:
        for y in y_cols:
            pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 4:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(pair[x], pair[y])
            rows.append({
                "x_feature": x,
                "y_rollout_metric": y,
                "n": int(len(pair)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False, na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobility-descriptors", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_mobility/multi_cycle_roi_mobility_descriptors.csv")
    parser.add_argument("--rollout-per-roi", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_rollout_baselines/roi_rollout_per_roi_metrics.csv")
    parser.add_argument("--latent-summary", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_rollout_baselines/roi_latent_dynamics_summary.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_rollout_mobility_coupling")
    args = parser.parse_args()

    mobility = pd.read_csv(args.mobility_descriptors)
    rollout = wide_rollout(pd.read_csv(args.rollout_per_roi))
    merged = mobility.merge(rollout, how="left", on=["roi_id", "cycleNo"], suffixes=("", "_rollout"))
    if os.path.exists(args.latent_summary):
        latent = pd.read_csv(args.latent_summary)
        latent_cols = [c for c in latent.columns if c not in {"validation_score"}]
        latent = latent[latent_cols].drop_duplicates(["roi_id", "cycleNo"])
        merged = merged.merge(latent, how="left", on=["roi_id", "cycleNo"], suffixes=("", "_latent"))

    rollout_features = [
        "persistence_mse",
        "persistence_mae",
        "persistence_ssim",
        "velocity_mse_ratio_vs_persistence",
        "dmd_mse_ratio_vs_persistence",
        "latent_path_length",
        "latent_net_displacement",
    ]
    rollout_features = [c for c in rollout_features if c in merged.columns]
    mobility_features = [
        "roi_mean_delta",
        "high_fraction_slope_per_s",
        "mid_fraction_slope_per_s",
        "high_radius2_slope_px2_per_s",
        "interface_density_delta",
        "centroid_path_px",
        "first_last_corr",
        "cumulative_abs_first_last",
        "temporal_diff_energy",
        "evidence_score",
        "mean_drop_frac",
    ]
    mobility_features = [c for c in mobility_features if c in merged.columns]

    tests = feature_tests(merged, rollout_features)
    corr = spearman_table(merged, mobility_features, rollout_features)
    cycle = merged.groupby(["cohort_role", "event_reference_cycle", "cycleNo"], dropna=False).agg({
        "roi_id": "count",
        "persistence_mse": "mean",
        "persistence_ssim": "mean",
        "velocity_mse_ratio_vs_persistence": "mean",
        "dmd_mse_ratio_vs_persistence": "mean",
        "high_fraction_slope_per_s": "mean",
        "cumulative_abs_first_last": "mean",
        "first_last_corr": "mean",
        "latent_path_length": "mean",
        "latent_net_displacement": "mean",
    }).reset_index().rename(columns={"roi_id": "n_roi"})

    ranked = merged.copy()
    score_parts = []
    for col, sign in [
        ("persistence_mse", 1),
        ("dmd_mse_ratio_vs_persistence", 1),
        ("cumulative_abs_first_last", 1),
        ("first_last_corr", -1),
        ("high_fraction_slope_per_s", 1),
    ]:
        if col in ranked.columns:
            score_parts.append(sign * ranked[col].rank(pct=True))
    ranked["rollout_mobility_difficulty_score"] = np.sum(score_parts, axis=0) if score_parts else np.nan
    ranked = ranked.sort_values("rollout_mobility_difficulty_score", ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    merged_path = os.path.join(args.out_dir, "multi_cycle_rollout_mobility_descriptors.csv")
    tests_path = os.path.join(args.out_dir, "multi_cycle_rollout_event_control_tests.csv")
    corr_path = os.path.join(args.out_dir, "multi_cycle_rollout_mobility_correlations.csv")
    cycle_path = os.path.join(args.out_dir, "multi_cycle_rollout_mobility_cycle_summary.csv")
    ranked_path = os.path.join(args.out_dir, "multi_cycle_rollout_mobility_ranked.csv")
    merged.to_csv(merged_path, index=False)
    tests.to_csv(tests_path, index=False)
    corr.to_csv(corr_path, index=False)
    cycle.to_csv(cycle_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    summary = {
        "n_rows": int(len(merged)),
        "n_event_roi": int((merged["is_event_roi"] == 1).sum()),
        "n_control_roi": int((merged["is_event_roi"] == 0).sum()),
        "top_event_control_rollout_tests": tests.head(10).to_dict(orient="records"),
        "top_mobility_rollout_correlations": corr.head(12).to_dict(orient="records"),
        "cycle_summary": cycle.to_dict(orient="records"),
        "top_ranked_rollout_mobility_rois": ranked.head(12)[[
            c for c in [
                "roi_id",
                "cohort_role",
                "cycleNo",
                "event_reference_cycle",
                "rollout_mobility_difficulty_score",
                "persistence_mse",
                "persistence_ssim",
                "dmd_mse_ratio_vs_persistence",
                "high_fraction_slope_per_s",
                "cumulative_abs_first_last",
                "first_last_corr",
                "latent_path_length",
            ]
            if c in ranked.columns
        ]].to_dict(orient="records"),
        "guardrail": "Rollout baselines are ROI-only and persistence-dominated; use difficulty correlations as physics-facing hypotheses, not causal proof.",
        "outputs": {
            "merged": merged_path,
            "event_control_tests": tests_path,
            "correlations": corr_path,
            "cycle_summary": cycle_path,
            "ranked": ranked_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "multi_cycle_rollout_mobility_coupling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Multi-Cycle Rollout Mobility Coupling\n\n")
        f.write("Joins ROI-only rollout baselines with timing-aware mobility and degradation descriptors.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
