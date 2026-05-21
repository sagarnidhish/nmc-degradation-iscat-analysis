#!/usr/bin/env python3
"""Analyze the expanded multi-cycle NMC ROI cohort.

This reads the cohort table and exported crop tensors, computes compact
physics-facing descriptors, and tests event ROIs against nearby controls across
synchronized and single-particle degradation candidate cycles.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


PIXEL_SIZE_UM = 0.096


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def slope(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return {"slope": np.nan, "r2": np.nan}
    coef = np.polyfit(x[mask], y[mask], 1)
    pred = coef[0] * x[mask] + coef[1]
    ss_res = float(np.sum((y[mask] - pred) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return {"slope": float(coef[0]), "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan}


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(av) & np.isfinite(bv)
    if mask.sum() < 3 or np.nanstd(av[mask]) == 0 or np.nanstd(bv[mask]) == 0:
        return np.nan
    return float(np.corrcoef(av[mask], bv[mask])[0, 1])


def central_mask(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= 0.48 * min(height, width)


def descriptors(npz_path: str) -> Dict[str, float]:
    with np.load(npz_path) as data:
        frames = np.asarray(data["frames_norm"], dtype=np.float32)
        roi_mean = np.asarray(data["roi_norm_mean"], dtype=float)
    t, h, w = frames.shape
    mask = central_mask(h, w)
    early = frames[: max(4, t // 8)][:, mask]
    high_thr = float(np.nanpercentile(early, 70))
    low_thr = float(np.nanpercentile(early, 30))
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    area = float(mask.sum())
    high_frac = []
    low_frac = []
    high_r2 = []
    centroid_x = []
    centroid_y = []
    for frame in frames:
        valid = mask & np.isfinite(frame)
        high = valid & (frame >= high_thr)
        low = valid & (frame <= low_thr)
        high_frac.append(float(high.sum() / area))
        low_frac.append(float(low.sum() / area))
        high_r2.append(float(np.mean(rr2[high])) if high.any() else np.nan)
        if high.any():
            centroid_x.append(float(np.mean(xx[high]) - cx))
            centroid_y.append(float(np.mean(yy[high]) - cy))
        else:
            centroid_x.append(np.nan)
            centroid_y.append(np.nan)
    high_frac = np.asarray(high_frac)
    low_frac = np.asarray(low_frac)
    high_r2 = np.asarray(high_r2)
    centroid_x = np.asarray(centroid_x)
    centroid_y = np.asarray(centroid_y)
    roi_fit = slope(roi_mean)
    high_fit = slope(high_frac)
    low_fit = slope(low_frac)
    r2_fit = slope(high_r2)
    centroid_step = np.sqrt(np.diff(centroid_x) ** 2 + np.diff(centroid_y) ** 2)
    diff = np.diff(frames, axis=0)
    return {
        "roi_norm_mean_delta": float(roi_mean[-1] - roi_mean[0]),
        "roi_norm_mean_slope": roi_fit["slope"],
        "roi_norm_mean_slope_r2": roi_fit["r2"],
        "high_fraction_delta": float(high_frac[-1] - high_frac[0]),
        "high_fraction_slope": high_fit["slope"],
        "low_fraction_delta": float(low_frac[-1] - low_frac[0]),
        "low_fraction_slope": low_fit["slope"],
        "high_radius2_delta_px2": float(high_r2[-1] - high_r2[0]),
        "high_radius2_slope_px2_per_frame": r2_fit["slope"],
        "high_radius2_slope_um2_per_frame": r2_fit["slope"] * PIXEL_SIZE_UM * PIXEL_SIZE_UM,
        "centroid_path_px": float(np.nansum(centroid_step)),
        "first_last_corr": safe_corr(frames[0][mask], frames[-1][mask]),
        "cumulative_abs_norm_change": float(np.nanmean(np.abs(frames[-1][mask] - frames[0][mask]))),
        "temporal_diff_energy": float(np.nanmean(diff ** 2)),
    }


def feature_tests(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    event = df[df["cohort_role"] == "event"]
    control = df[df["cohort_role"] == "control"]
    for feat in features:
        a = pd.to_numeric(event[feat], errors="coerce").dropna()
        b = pd.to_numeric(control[feat], errors="coerce").dropna()
        if len(a) and len(b):
            stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
        else:
            stat, p_value = np.nan, np.nan
        rows.append({
            "feature": feat,
            "event_mean": float(a.mean()) if len(a) else np.nan,
            "control_mean": float(b.mean()) if len(b) else np.nan,
            "event_minus_control": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
            "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
            "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("p_value")


def correlations(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    event = df[df["cohort_role"] == "event"].copy()
    for feat in features:
        x = pd.to_numeric(event["evidence_score"], errors="coerce")
        y = pd.to_numeric(event[feat], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() >= 4:
            rho, p_value = spearmanr(x[mask], y[mask])
        else:
            rho, p_value = np.nan, np.nan
        rows.append({"feature": feat, "spearman_rho_vs_evidence_score": rho, "p_value": p_value, "n": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("p_value")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-table", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_cohort/multi_cycle_roi_table.csv")
    parser.add_argument("--manifest", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_sequences/selected_roi_sequence_manifest.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_analysis")
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort_table)
    manifest = pd.read_csv(args.manifest)
    rows = []
    for _, row in manifest.iterrows():
        desc = descriptors(row["npz_path"])
        out = row.to_dict()
        out.update(desc)
        rows.append(out)
    df = pd.DataFrame(rows)
    merge_cols = [
        c for c in [
            "cycleNo", "front_candidate_rank", "object_candidate_rank",
            "cohort_role", "event_reference_cycle", "event_particles",
            "n_event_particles", "mean_drop_frac", "global_frame_percentile",
            "evidence_score", "degradation_mode_hypothesis", "front_quality_score",
            "apparent_diffusion_proxy_ds_px2_per_frame",
        ] if c in cohort.columns
    ]
    df = df.merge(
        cohort[merge_cols].drop_duplicates(["cycleNo", "front_candidate_rank", "object_candidate_rank"]),
        how="left",
        on=["cycleNo", "front_candidate_rank", "object_candidate_rank"],
    )
    df["is_synchronized_reference"] = (pd.to_numeric(df["n_event_particles"], errors="coerce") >= 3).astype(int)
    features = [
        "roi_norm_mean_delta",
        "roi_norm_mean_slope",
        "high_fraction_delta",
        "high_fraction_slope",
        "low_fraction_delta",
        "low_fraction_slope",
        "high_radius2_delta_px2",
        "high_radius2_slope_px2_per_frame",
        "centroid_path_px",
        "first_last_corr",
        "cumulative_abs_norm_change",
        "temporal_diff_energy",
    ]
    tests = feature_tests(df, features)
    corr = correlations(df, features)
    group = df.groupby(["event_reference_cycle", "cohort_role"], dropna=False).agg({
        "roi_id": "count",
        "roi_norm_mean_delta": "mean",
        "high_fraction_slope": "mean",
        "high_radius2_slope_px2_per_frame": "mean",
        "first_last_corr": "mean",
        "cumulative_abs_norm_change": "mean",
        "temporal_diff_energy": "mean",
        "evidence_score": "mean",
    }).reset_index().rename(columns={"roi_id": "n_roi"})
    sync_group = df.groupby(["is_synchronized_reference", "cohort_role"], dropna=False).agg({
        "roi_id": "count",
        "roi_norm_mean_delta": "mean",
        "high_fraction_slope": "mean",
        "first_last_corr": "mean",
        "cumulative_abs_norm_change": "mean",
        "temporal_diff_energy": "mean",
    }).reset_index().rename(columns={"roi_id": "n_roi"})
    ranked = df.copy()
    ranked["multicycle_physics_score"] = (
        ranked["cumulative_abs_norm_change"].rank(pct=True)
        + ranked["temporal_diff_energy"].rank(pct=True)
        - ranked["first_last_corr"].rank(pct=True)
        + ranked["high_fraction_slope"].rank(pct=True)
    )
    ranked = ranked.sort_values("multicycle_physics_score", ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    desc_path = os.path.join(args.out_dir, "multi_cycle_roi_descriptors.csv")
    tests_path = os.path.join(args.out_dir, "multi_cycle_event_control_feature_tests.csv")
    corr_path = os.path.join(args.out_dir, "multi_cycle_event_evidence_correlations.csv")
    group_path = os.path.join(args.out_dir, "multi_cycle_group_summary.csv")
    sync_path = os.path.join(args.out_dir, "multi_cycle_sync_vs_single_summary.csv")
    ranked_path = os.path.join(args.out_dir, "multi_cycle_roi_ranked.csv")
    df.to_csv(desc_path, index=False)
    tests.to_csv(tests_path, index=False)
    corr.to_csv(corr_path, index=False)
    group.to_csv(group_path, index=False)
    sync_group.to_csv(sync_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    summary = {
        "cohort_table": args.cohort_table,
        "manifest": args.manifest,
        "n_roi": int(len(df)),
        "n_event_roi": int((df["cohort_role"] == "event").sum()),
        "n_control_roi": int((df["cohort_role"] == "control").sum()),
        "event_reference_cycles": sorted(float(x) for x in df["event_reference_cycle"].dropna().unique()),
        "top_feature_tests": tests.head(10).to_dict(orient="records"),
        "group_summary": group.to_dict(orient="records"),
        "sync_vs_single_summary": sync_group.to_dict(orient="records"),
        "top_ranked_rois": ranked.head(12)[[
            "roi_id", "cohort_role", "cycleNo", "event_reference_cycle",
            "multicycle_physics_score", "roi_norm_mean_delta",
            "high_fraction_slope", "first_last_corr",
            "cumulative_abs_norm_change", "temporal_diff_energy",
            "evidence_score", "degradation_mode_hypothesis",
        ]].to_dict(orient="records"),
        "guardrail": "Expanded cohort descriptors are automatic ROI-crop measurements. They rank physics candidates but do not replace manual particle/front validation.",
        "outputs": {
            "descriptors": desc_path,
            "feature_tests": tests_path,
            "evidence_correlations": corr_path,
            "group_summary": group_path,
            "sync_vs_single_summary": sync_path,
            "ranked": ranked_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "multi_cycle_roi_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Multi-Cycle ROI Analysis\n\n")
        f.write("Descriptor and event/control tests for synchronized and single-particle NMC degradation candidate cycles.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
