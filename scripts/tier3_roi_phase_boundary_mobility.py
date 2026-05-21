#!/usr/bin/env python3
"""Estimate phase-boundary mobility proxies from event and control ROI tensors.

The ROI tensors are already particle-region crops. This script uses only those
fixed crops and derives threshold-front summaries that are deliberately
calibration-light: phase-like bright/dark fractions, apparent radius-squared
slopes, interface length density, and centroid motion. These are not calibrated
diffusion coefficients, but they give a common physics-facing descriptor table
for event ROIs and matched non-event controls.
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def slope(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
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
    if mask.sum() < 3:
        return np.nan
    if np.nanstd(av[mask]) == 0 or np.nanstd(bv[mask]) == 0:
        return np.nan
    return float(np.corrcoef(av[mask], bv[mask])[0, 1])


def central_mask(height: int, width: int, radius_frac: float = 0.48) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= radius_frac * min(height, width)


def mask_perimeter(mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    eroded = ndi.binary_erosion(mask)
    edge = mask & ~eroded
    return float(edge.sum())


def trace_features(frames: np.ndarray) -> Dict[str, object]:
    frames = np.asarray(frames, dtype=np.float32)
    t, h, w = frames.shape
    roi_mask = central_mask(h, w)
    early = frames[: max(4, t // 8)][:, roi_mask]
    low_thr = float(np.nanpercentile(early, 30))
    high_thr = float(np.nanpercentile(early, 70))
    mid_thr = 0.5 * (low_thr + high_thr)

    yy, xx = np.mgrid[:h, :w]
    cy0 = (h - 1) / 2.0
    cx0 = (w - 1) / 2.0
    rr2 = (yy - cy0) ** 2 + (xx - cx0) ** 2
    roi_area = float(roi_mask.sum())

    high_frac = []
    low_frac = []
    mid_frac = []
    high_radius2 = []
    low_radius2 = []
    interface_density = []
    centroid_x = []
    centroid_y = []
    for frame in frames:
        valid = roi_mask & np.isfinite(frame)
        high = valid & (frame >= high_thr)
        low = valid & (frame <= low_thr)
        mid = valid & (frame >= mid_thr)
        high_frac.append(float(high.sum() / roi_area))
        low_frac.append(float(low.sum() / roi_area))
        mid_frac.append(float(mid.sum() / roi_area))
        high_radius2.append(float(np.mean(rr2[high])) if high.any() else np.nan)
        low_radius2.append(float(np.mean(rr2[low])) if low.any() else np.nan)
        interface_density.append(mask_perimeter(mid) / roi_area)
        if mid.any():
            centroid_x.append(float(np.mean(xx[mid]) - cx0))
            centroid_y.append(float(np.mean(yy[mid]) - cy0))
        else:
            centroid_x.append(np.nan)
            centroid_y.append(np.nan)

    high_frac = np.asarray(high_frac)
    low_frac = np.asarray(low_frac)
    mid_frac = np.asarray(mid_frac)
    high_radius2 = np.asarray(high_radius2)
    low_radius2 = np.asarray(low_radius2)
    interface_density = np.asarray(interface_density)
    centroid_x = np.asarray(centroid_x)
    centroid_y = np.asarray(centroid_y)
    centroid_step = np.sqrt(np.diff(centroid_x) ** 2 + np.diff(centroid_y) ** 2)
    diff = np.diff(frames, axis=0)

    high_fit = slope(high_frac)
    low_fit = slope(low_frac)
    mid_fit = slope(mid_frac)
    high_r2_fit = slope(high_radius2)
    low_r2_fit = slope(low_radius2)
    iface_fit = slope(interface_density)
    return {
        "low_threshold": low_thr,
        "high_threshold": high_thr,
        "mid_threshold": mid_thr,
        "high_fraction_first": float(high_frac[0]),
        "high_fraction_last": float(high_frac[-1]),
        "high_fraction_delta": float(high_frac[-1] - high_frac[0]),
        "high_fraction_slope_per_frame": high_fit["slope"],
        "high_fraction_slope_r2": high_fit["r2"],
        "low_fraction_delta": float(low_frac[-1] - low_frac[0]),
        "low_fraction_slope_per_frame": low_fit["slope"],
        "mid_fraction_delta": float(mid_frac[-1] - mid_frac[0]),
        "mid_fraction_slope_per_frame": mid_fit["slope"],
        "high_radius2_delta_px2": float(high_radius2[-1] - high_radius2[0]),
        "high_radius2_slope_px2_per_frame": high_r2_fit["slope"],
        "high_radius2_slope_r2": high_r2_fit["r2"],
        "low_radius2_delta_px2": float(low_radius2[-1] - low_radius2[0]),
        "low_radius2_slope_px2_per_frame": low_r2_fit["slope"],
        "interface_density_mean": float(np.nanmean(interface_density)),
        "interface_density_delta": float(interface_density[-1] - interface_density[0]),
        "interface_density_slope_per_frame": iface_fit["slope"],
        "centroid_path_px": float(np.nansum(centroid_step)),
        "centroid_net_px": float(np.sqrt((centroid_x[-1] - centroid_x[0]) ** 2 + (centroid_y[-1] - centroid_y[0]) ** 2)),
        "temporal_diff_energy": float(np.nanmean(diff ** 2)),
        "first_last_corr": safe_corr(frames[0][roi_mask], frames[-1][roi_mask]),
        "cumulative_abs_first_last": float(np.nanmean(np.abs(frames[-1][roi_mask] - frames[0][roi_mask]))),
    }


def load_control_map(control_table_path: str) -> Dict[str, float]:
    if not control_table_path or not os.path.exists(control_table_path):
        return {}
    table = pd.read_csv(control_table_path)
    if "control_for_event_cycle" not in table.columns:
        return {}
    table = table.copy()
    table["roi_id"] = table.apply(
        lambda r: f"cycle{int(float(r['cycleNo']))}_rank{int(r['front_candidate_rank'])}_obj{int(r['object_candidate_rank'])}",
        axis=1,
    )
    return dict(zip(table["roi_id"], table["control_for_event_cycle"]))



def timing_seconds(root: str, source_stem: str, frame_indices: np.ndarray) -> Dict[str, float]:
    if not root:
        return {"timing_elapsed_s": np.nan, "seconds_per_sample": np.nan}
    h5_path = os.path.join(root, "NMC_degradation_3_160623_Halfthedata", f"{source_stem}.hdf5")
    if not os.path.exists(h5_path):
        return {"timing_elapsed_s": np.nan, "seconds_per_sample": np.nan}
    try:
        with h5py.File(h5_path, "r") as f:
            if "camera_timing" not in f:
                return {"timing_elapsed_s": np.nan, "seconds_per_sample": np.nan}
            timing = np.asarray(f["camera_timing"])
    except Exception:
        return {"timing_elapsed_s": np.nan, "seconds_per_sample": np.nan}
    if timing.ndim == 1:
        trace = timing
    else:
        trace = timing.reshape(-1, timing.shape[-1])[0]
    frame_indices = np.asarray(frame_indices, dtype=int)
    frame_indices = frame_indices[(frame_indices >= 0) & (frame_indices < len(trace))]
    if len(frame_indices) < 2:
        return {"timing_elapsed_s": np.nan, "seconds_per_sample": np.nan}
    t = trace[frame_indices].astype(float)
    elapsed_raw = float(np.nanmax(t) - np.nanmin(t))
    # HDF5 camera_timing is nanosecond-like in the NMC movies.
    elapsed_s = elapsed_raw / 1e9 if elapsed_raw > 1e6 else elapsed_raw
    return {
        "timing_elapsed_s": elapsed_s,
        "seconds_per_sample": elapsed_s / float(len(frame_indices) - 1) if elapsed_s > 0 else np.nan,
    }


def descriptors_for_manifest(manifest_path: str, roi_class: str, control_map: Dict[str, float], root: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows: List[Dict[str, object]] = []
    for _, row in manifest.iterrows():
        with np.load(row["npz_path"]) as data:
            features = trace_features(data["frames_norm"])
            frame_indices = data["frame_indices"] if "frame_indices" in data else np.array([], dtype=int)
        timing = timing_seconds(root, str(row.get("source_stem", "")), frame_indices)
        seconds_per_sample = timing["seconds_per_sample"]
        if np.isfinite(seconds_per_sample) and seconds_per_sample > 0:
            for col in [
                "high_fraction_slope_per_frame",
                "low_fraction_slope_per_frame",
                "mid_fraction_slope_per_frame",
                "high_radius2_slope_px2_per_frame",
                "low_radius2_slope_px2_per_frame",
                "interface_density_slope_per_frame",
            ]:
                features[col.replace("_per_frame", "_per_s")] = features[col] / seconds_per_sample
        event_cycle = float(row["cycleNo"]) if roi_class == "event" else finite_float(control_map.get(row["roi_id"]), np.nan)
        out = {
            "roi_id": row["roi_id"],
            "roi_class": roi_class,
            "is_event_roi": int(roi_class == "event"),
            "cycleNo": float(row["cycleNo"]),
            "event_cycle": event_cycle,
            "source_stem": row.get("source_stem", ""),
            "stage_drift_xy_sampled": finite_float(row.get("stage_drift_xy_sampled")),
            "roi_norm_mean_delta_last_minus_first": finite_float(row.get("roi_norm_mean_delta_last_minus_first")),
            **timing,
        }
        out.update(features)
        rows.append(out)
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
                stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                stat, p_value = np.nan, np.nan
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


def save_plot(df: pd.DataFrame, tests: pd.DataFrame, out_png: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label, grp in df.groupby("roi_class"):
        axes[0].scatter(grp["high_fraction_slope_per_frame"], grp["high_radius2_slope_px2_per_frame"], label=label, s=45)
    axes[0].axhline(0, color="0.7", lw=1)
    axes[0].axvline(0, color="0.7", lw=1)
    axes[0].set_xlabel("high-fraction slope")
    axes[0].set_ylabel("high radius^2 slope")
    axes[0].legend(fontsize=8)
    axes[0].set_title("phase-proxy expansion/contraction", fontsize=9)
    for label, grp in df.groupby("roi_class"):
        axes[1].scatter(grp["interface_density_mean"], grp["centroid_path_px"], label=label, s=45)
    axes[1].set_xlabel("mean interface density")
    axes[1].set_ylabel("centroid path px")
    axes[1].set_title("boundary complexity/motion", fontsize=9)
    top = tests.head(7).copy()
    axes[2].barh(top["feature"], top["event_minus_control"])
    axes[2].axvline(0, color="0.7", lw=1)
    axes[2].set_title("event-control feature shift", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-manifest", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/selected_roi_sequences/selected_roi_sequence_manifest.csv")
    parser.add_argument("--control-manifest", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_roi_sequences_expanded/selected_roi_sequence_manifest.csv")
    parser.add_argument("--control-table", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_roi_selection_expanded/selected_control_rois.csv")
    parser.add_argument("--root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_phase_boundary_mobility")
    args = parser.parse_args()

    control_map = load_control_map(args.control_table)
    event_df = descriptors_for_manifest(args.event_manifest, "event", control_map, args.root)
    control_df = descriptors_for_manifest(args.control_manifest, "control", control_map, args.root)
    df = pd.concat([event_df, control_df], ignore_index=True)
    features = [
        "high_fraction_delta",
        "high_fraction_slope_per_frame",
        "high_fraction_slope_per_s",
        "low_fraction_delta",
        "low_fraction_slope_per_frame",
        "low_fraction_slope_per_s",
        "mid_fraction_delta",
        "mid_fraction_slope_per_frame",
        "mid_fraction_slope_per_s",
        "high_radius2_delta_px2",
        "high_radius2_slope_px2_per_frame",
        "high_radius2_slope_px2_per_s",
        "low_radius2_delta_px2",
        "low_radius2_slope_px2_per_frame",
        "low_radius2_slope_px2_per_s",
        "interface_density_mean",
        "interface_density_delta",
        "interface_density_slope_per_frame",
        "interface_density_slope_per_s",
        "centroid_path_px",
        "centroid_net_px",
        "temporal_diff_energy",
        "first_last_corr",
        "cumulative_abs_first_last",
    ]
    tests = feature_tests(df, features)
    group = df.groupby(["roi_class", "event_cycle"], dropna=False).agg({
        "roi_id": "count",
        "high_fraction_slope_per_frame": "mean",
        "high_fraction_slope_per_s": "mean",
        "high_radius2_slope_px2_per_frame": "mean",
        "high_radius2_slope_px2_per_s": "mean",
        "interface_density_mean": "mean",
        "centroid_path_px": "mean",
        "first_last_corr": "mean",
        "cumulative_abs_first_last": "mean",
    }).reset_index().rename(columns={"roi_id": "n_roi"})
    ranked = df.copy()
    ranked["mobility_score"] = (
        ranked["cumulative_abs_first_last"].rank(pct=True)
        + ranked["centroid_path_px"].rank(pct=True)
        + ranked["interface_density_mean"].rank(pct=True)
        - ranked["first_last_corr"].rank(pct=True)
    )
    ranked = ranked.sort_values("mobility_score", ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    desc_path = os.path.join(args.out_dir, "roi_phase_boundary_mobility_descriptors.csv")
    tests_path = os.path.join(args.out_dir, "roi_phase_boundary_mobility_feature_tests.csv")
    group_path = os.path.join(args.out_dir, "roi_phase_boundary_mobility_group_summary.csv")
    ranked_path = os.path.join(args.out_dir, "roi_phase_boundary_mobility_ranked.csv")
    plot_path = os.path.join(args.out_dir, "roi_phase_boundary_mobility.png")
    df.to_csv(desc_path, index=False)
    tests.to_csv(tests_path, index=False)
    group.to_csv(group_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    save_plot(df, tests, plot_path)
    summary = {
        "event_manifest": args.event_manifest,
        "control_manifest": args.control_manifest,
        "control_table": args.control_table,
        "root": args.root,
        "n_event_roi": int((df["is_event_roi"] == 1).sum()),
        "n_control_roi": int((df["is_event_roi"] == 0).sum()),
        "event_cycles": sorted(float(x) for x in event_df["event_cycle"].dropna().unique()),
        "control_cycles": sorted(float(x) for x in control_df["cycleNo"].dropna().unique()),
        "top_feature_tests": tests.head(10).to_dict(orient="records"),
        "group_summary": group.to_dict(orient="records"),
        "top_mobility_rois": ranked.head(10)[[
            "roi_id",
            "roi_class",
            "cycleNo",
            "event_cycle",
            "mobility_score",
            "high_fraction_slope_per_frame",
            "high_fraction_slope_per_s",
            "high_radius2_slope_px2_per_frame",
            "high_radius2_slope_px2_per_s",
            "interface_density_mean",
            "centroid_path_px",
            "first_last_corr",
            "cumulative_abs_first_last",
        ]].to_dict(orient="records"),
        "guardrail": "Threshold-front features are apparent phase-boundary mobility proxies from fixed particle ROI crops; they are not calibrated diffusion coefficients.",
        "outputs": {
            "descriptors": desc_path,
            "feature_tests": tests_path,
            "group_summary": group_path,
            "ranked": ranked_path,
            "plot": plot_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "roi_phase_boundary_mobility_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# ROI Phase-Boundary Mobility Proxies\n\n")
        f.write("Threshold-front descriptors for selected event ROIs and expanded matched controls.\n")
        f.write("These features rank apparent phase-boundary mobility but are not calibrated diffusion coefficients.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
