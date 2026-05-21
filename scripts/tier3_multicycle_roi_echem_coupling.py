#!/usr/bin/env python3
"""Couple expanded NMC ROI physics descriptors to echem/protocol context.

This joins ROI-level optical/front/rollout descriptors to per-cycle
electrochemical summaries and protocol/frame-count context. The outputs are
correlation and stratification tables, not causal claims.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


ROI_FEATURES = [
    "roi_norm_mean_delta",
    "roi_norm_mean_slope",
    "high_fraction_delta",
    "high_fraction_slope",
    "low_fraction_delta",
    "high_radius2_delta_px2",
    "high_radius2_slope_px2_per_frame",
    "centroid_path_px",
    "first_last_corr",
    "cumulative_abs_norm_change",
    "temporal_diff_energy",
    "latent_path_length",
    "latent_mean_step",
    "latent_net_displacement",
    "persistence_mse",
    "velocity_mse",
    "low_rank_dmd_mse",
    "dmd_minus_persistence_mse",
]


ECHEM_FEATURES = [
    "n_frames",
    "n_frames_percentile",
    "echem_points",
    "duration_s",
    "V_min",
    "V_max",
    "V_range",
    "V_mean",
    "I_mean_mA",
    "I_abs_mean_mA",
    "I_pos_fraction",
    "I_neg_fraction",
    "V_mean_delta",
    "I_mean_delta",
    "V_range_delta",
    "cycles_from_block_start",
    "cycles_to_block_end",
    "block_fraction_elapsed",
]


def safe_spearman(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    mask = xx.notna() & yy.notna()
    if mask.sum() < 5 or xx[mask].nunique() < 2 or yy[mask].nunique() < 2:
        return {"rho": np.nan, "p_value": np.nan, "n": int(mask.sum())}
    rho, p_value = spearmanr(xx[mask], yy[mask])
    return {"rho": float(rho), "p_value": float(p_value), "n": int(mask.sum())}


def feature_tests(df: pd.DataFrame, features: List[str], group_col: str) -> pd.DataFrame:
    rows = []
    for feat in features:
        event = pd.to_numeric(df[df[group_col] == "event"][feat], errors="coerce").dropna()
        control = pd.to_numeric(df[df[group_col] == "control"][feat], errors="coerce").dropna()
        if len(event) and len(control):
            stat, p_value = mannwhitneyu(event, control, alternative="two-sided")
        else:
            stat, p_value = np.nan, np.nan
        rows.append({
            "feature": feat,
            "event_mean": float(event.mean()) if len(event) else np.nan,
            "control_mean": float(control.mean()) if len(control) else np.nan,
            "event_minus_control": float(event.mean() - control.mean()) if len(event) and len(control) else np.nan,
            "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
            "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("p_value")


def load_rollout_features(rollout_dir: str) -> pd.DataFrame:
    latent = pd.read_csv(os.path.join(rollout_dir, "roi_latent_dynamics_summary.csv"))
    roll = pd.read_csv(os.path.join(rollout_dir, "roi_rollout_per_roi_metrics.csv"))
    pivot = roll.pivot_table(index="roi_id", columns="method", values="mse", aggfunc="first").reset_index()
    rename = {c: f"{c}_mse" for c in pivot.columns if c != "roi_id"}
    pivot = pivot.rename(columns=rename)
    if {"low_rank_dmd_mse", "persistence_mse"}.issubset(pivot.columns):
        pivot["dmd_minus_persistence_mse"] = pivot["low_rank_dmd_mse"] - pivot["persistence_mse"]
    return latent.merge(pivot, on="roi_id", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_echem_coupling")
    args = parser.parse_args()

    roi_path = os.path.join(args.derived_dir, "multi_cycle_roi_analysis", "multi_cycle_roi_descriptors.csv")
    rollout_dir = os.path.join(args.derived_dir, "multi_cycle_roi_rollout_baselines")
    context_path = os.path.join(args.derived_dir, "event_protocol_context", "event_protocol_context_table.csv")
    local_path = os.path.join(args.derived_dir, "protocol_local_window_scan", "local_echem_window_features.csv")
    pred_path = os.path.join(args.derived_dir, "multi_cycle_roi_event_predictor", "leave_event_cycle_out_predictions.csv")

    roi = pd.read_csv(roi_path)
    rollout = load_rollout_features(rollout_dir)
    context = pd.read_csv(context_path)
    local = pd.read_csv(local_path) if os.path.exists(local_path) else pd.DataFrame()
    pred = pd.read_csv(pred_path) if os.path.exists(pred_path) else pd.DataFrame()

    for df in [roi, context, local]:
        if "cycleNo" in df.columns:
            df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    joined = roi.merge(rollout, on=["roi_id", "cycleNo", "validation_score"], how="left")
    keep_context = ["cycleNo"] + [c for c in ECHEM_FEATURES + ["block_mode", "protocol_block_segment", "near_block_boundary_10cycles"] if c in context.columns]
    joined = joined.merge(context[keep_context].drop_duplicates("cycleNo"), on="cycleNo", how="left")
    if not pred.empty:
        pred_keep = pred[pred["feature_set"] == "physics_no_selection_qc"].copy()
        pred_keep = pred_keep[pred_keep["model"] == "random_forest"]
        pred_keep = pred_keep[["roi_id", "event_probability", "predicted_event_roi"]].rename(
            columns={"event_probability": "physics_no_qc_rf_event_probability"}
        )
        joined = joined.merge(pred_keep, on="roi_id", how="left")

    roi_features = [f for f in ROI_FEATURES + ["physics_no_qc_rf_event_probability"] if f in joined.columns]
    echem_features = [f for f in ECHEM_FEATURES if f in joined.columns]
    corr_rows = []
    for roi_feat in roi_features:
        for echem_feat in echem_features:
            out = safe_spearman(joined[echem_feat], joined[roi_feat])
            out.update({"roi_feature": roi_feat, "echem_feature": echem_feat})
            corr_rows.append(out)
    corr = pd.DataFrame(corr_rows).sort_values("p_value")

    # Within-reference centering separates event-control ROI differences from
    # broader event-cycle/protocol context.
    centered = joined.copy()
    for feat in roi_features:
        centered[f"{feat}_centered_by_reference"] = (
            pd.to_numeric(centered[feat], errors="coerce")
            - centered.groupby("event_reference_cycle")[feat].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
        )
    centered_features = [f"{f}_centered_by_reference" for f in roi_features]
    centered_tests = feature_tests(centered, centered_features, "cohort_role")

    group = joined.groupby(["event_reference_cycle", "cohort_role"], dropna=False).agg({
        "roi_id": "count",
        "roi_norm_mean_delta": "mean",
        "high_fraction_slope": "mean",
        "cumulative_abs_norm_change": "mean",
        "first_last_corr": "mean",
        "latent_net_displacement": "mean",
        "physics_no_qc_rf_event_probability": "mean",
        "n_frames_percentile": "mean",
        "V_mean": "mean",
        "I_mean_mA": "mean",
        "I_abs_mean_mA": "mean",
        "duration_s": "mean",
    }).reset_index().rename(columns={"roi_id": "n_roi"})

    block_group = joined.groupby(["block_mode", "cohort_role"], dropna=False).agg({
        "roi_id": "count",
        "roi_norm_mean_delta": "mean",
        "high_fraction_slope": "mean",
        "cumulative_abs_norm_change": "mean",
        "first_last_corr": "mean",
        "latent_net_displacement": "mean",
    }).reset_index().rename(columns={"roi_id": "n_roi"})

    event_ref = (
        joined.groupby("event_reference_cycle", dropna=False)
        .agg({
            "n_event_particles": "first",
            "mean_drop_frac": "first",
            "evidence_score": "first",
            "n_frames_percentile": "mean",
            "V_mean": "mean",
            "I_abs_mean_mA": "mean",
            "roi_norm_mean_delta": "mean",
            "high_fraction_slope": "mean",
            "cumulative_abs_norm_change": "mean",
            "latent_net_displacement": "mean",
        })
        .reset_index()
    )
    event_ref_corr_rows = []
    for roi_feat in ["roi_norm_mean_delta", "high_fraction_slope", "cumulative_abs_norm_change", "latent_net_displacement"]:
        for meta_feat in ["n_event_particles", "mean_drop_frac", "evidence_score", "n_frames_percentile", "V_mean", "I_abs_mean_mA"]:
            out = safe_spearman(event_ref[meta_feat], event_ref[roi_feat])
            out.update({"roi_feature": roi_feat, "event_meta_feature": meta_feat})
            event_ref_corr_rows.append(out)
    event_ref_corr = pd.DataFrame(event_ref_corr_rows).sort_values("p_value")

    os.makedirs(args.out_dir, exist_ok=True)
    joined_path = os.path.join(args.out_dir, "multi_cycle_roi_echem_joined.csv")
    corr_path = os.path.join(args.out_dir, "roi_echem_spearman_correlations.csv")
    centered_path = os.path.join(args.out_dir, "within_reference_event_control_tests.csv")
    group_path = os.path.join(args.out_dir, "roi_echem_group_summary.csv")
    block_path = os.path.join(args.out_dir, "roi_protocol_block_summary.csv")
    event_ref_path = os.path.join(args.out_dir, "event_reference_roi_echem_summary.csv")
    event_ref_corr_path = os.path.join(args.out_dir, "event_reference_roi_echem_correlations.csv")
    joined.to_csv(joined_path, index=False)
    corr.to_csv(corr_path, index=False)
    centered_tests.to_csv(centered_path, index=False)
    group.to_csv(group_path, index=False)
    block_group.to_csv(block_path, index=False)
    event_ref.to_csv(event_ref_path, index=False)
    event_ref_corr.to_csv(event_ref_corr_path, index=False)

    summary = {
        "roi_source": roi_path,
        "context_source": context_path,
        "local_window_source": local_path if os.path.exists(local_path) else "",
        "n_joined_roi": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "n_event_reference_cycles": int(joined["event_reference_cycle"].nunique()),
        "top_roi_echem_correlations": corr.head(15).to_dict(orient="records"),
        "top_within_reference_event_control_tests": centered_tests.head(12).to_dict(orient="records"),
        "group_summary": group.to_dict(orient="records"),
        "event_reference_summary": event_ref.to_dict(orient="records"),
        "guardrail": (
            "Correlations join automatic ROI candidates to cycle-level echem/protocol summaries. "
            "They are hypothesis screens and are not causal evidence or calibrated transport constants."
        ),
        "outputs": {
            "joined": joined_path,
            "roi_echem_correlations": corr_path,
            "within_reference_tests": centered_path,
            "group_summary": group_path,
            "block_summary": block_path,
            "event_reference_summary": event_ref_path,
            "event_reference_correlations": event_ref_corr_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "multi_cycle_roi_echem_coupling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Multi-Cycle ROI Echem Coupling\n\n")
        f.write("Joins expanded ROI physics/rollout descriptors to cycle-level echem and protocol context.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
