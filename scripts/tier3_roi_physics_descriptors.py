#!/usr/bin/env python3
"""Extract physics-facing descriptors from selected NMC particle ROI sequences.

The selected ROI tensors are particle-region-only crops. This script turns them
into compact descriptors that can support degradation-mode hypotheses:

* ROI intensity trends and residual motion energy
* high/low optical-state fraction trends
* radial first/second moments of the optical state
* apparent front-radius and radius-squared slopes
* small-sample clustering of ROI dynamics modes

All front and transport quantities are image-coordinate proxies. They are useful
for ranking and hypothesis generation, not calibrated diffusion coefficients.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def fit_line(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
    xx = x[good].astype(float)
    yy = y[good].astype(float)
    slope, intercept, r, _, _ = stats.linregress(xx, yy)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r * r)}


def radial_grid(shape: Tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(shape)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def frame_descriptors(frames: np.ndarray) -> pd.DataFrame:
    radius = radial_grid(frames.shape[1:])
    first = frames[0]
    base_med = float(np.nanmedian(first))
    base_std = float(np.nanstd(first))
    high_thr = base_med + 0.50 * base_std
    low_thr = base_med - 0.50 * base_std
    rows = []
    for i, frame in enumerate(frames):
        high = frame >= high_thr
        low = frame <= low_thr
        weights = np.clip(frame - low_thr, 0, None)
        wsum = float(np.nansum(weights))
        if wsum > 0:
            radial_mean = float(np.nansum(weights * radius) / wsum)
            radial_second = float(np.nansum(weights * radius * radius) / wsum)
        else:
            radial_mean = np.nan
            radial_second = np.nan
        rows.append({
            "sample_idx": i,
            "roi_mean": float(np.nanmean(frame)),
            "roi_std": float(np.nanstd(frame)),
            "roi_p10": float(np.nanpercentile(frame, 10)),
            "roi_p90": float(np.nanpercentile(frame, 90)),
            "roi_contrast_p90_p10": float(np.nanpercentile(frame, 90) - np.nanpercentile(frame, 10)),
            "high_fraction": float(np.nanmean(high)),
            "low_fraction": float(np.nanmean(low)),
            "radial_mean_weighted": radial_mean,
            "radial_second_weighted": radial_second,
            "front_radius_high_mean": float(np.nanmean(radius[high])) if np.any(high) else np.nan,
            "front_radius_high_p90": float(np.nanpercentile(radius[high], 90)) if np.any(high) else np.nan,
            "front_radius_low_mean": float(np.nanmean(radius[low])) if np.any(low) else np.nan,
        })
    return pd.DataFrame(rows)


def sequence_summary(row: pd.Series, frames: np.ndarray, frame_indices: np.ndarray, out_trace: str) -> Dict[str, object]:
    desc = frame_descriptors(frames)
    desc.insert(0, "frame_index", frame_indices.astype(int))
    desc.insert(0, "roi_id", row["roi_id"])
    desc.to_csv(out_trace, index=False)
    x = frame_indices.astype(float) - float(frame_indices[0])
    mean_fit = fit_line(x, desc["roi_mean"].to_numpy(dtype=float))
    high_fit = fit_line(x, desc["high_fraction"].to_numpy(dtype=float))
    low_fit = fit_line(x, desc["low_fraction"].to_numpy(dtype=float))
    rad_fit = fit_line(x, desc["front_radius_high_mean"].to_numpy(dtype=float))
    rad2_fit = fit_line(x, desc["front_radius_high_mean"].to_numpy(dtype=float) ** 2)
    weighted_rad_fit = fit_line(x, desc["radial_mean_weighted"].to_numpy(dtype=float))
    diff_energy = float(np.nanmean(np.diff(frames, axis=0) ** 2))
    cumulative_abs_change = float(np.nanmean(np.abs(frames[-1] - frames[0])))
    front = desc["front_radius_high_mean"].to_numpy(dtype=float)
    front_steps = np.diff(front[np.isfinite(front)])
    monotonic_pos = float(np.mean(front_steps >= 0)) if front_steps.size else np.nan
    return {
        "roi_id": row["roi_id"],
        "cycleNo": float(row["cycleNo"]),
        "source_stem": row["source_stem"],
        "validation_score": float(row["validation_score"]),
        "object_x_full_approx": float(row["object_x_full_approx"]),
        "object_y_full_approx": float(row["object_y_full_approx"]),
        "n_frames": int(frames.shape[0]),
        "first_frame_index": int(frame_indices[0]),
        "last_frame_index": int(frame_indices[-1]),
        "roi_mean_first": float(desc["roi_mean"].iloc[0]),
        "roi_mean_last": float(desc["roi_mean"].iloc[-1]),
        "roi_mean_delta": float(desc["roi_mean"].iloc[-1] - desc["roi_mean"].iloc[0]),
        "roi_mean_slope_per_frame": mean_fit["slope"],
        "roi_mean_slope_r2": mean_fit["r2"],
        "high_fraction_first": float(desc["high_fraction"].iloc[0]),
        "high_fraction_last": float(desc["high_fraction"].iloc[-1]),
        "high_fraction_delta": float(desc["high_fraction"].iloc[-1] - desc["high_fraction"].iloc[0]),
        "high_fraction_slope_per_frame": high_fit["slope"],
        "high_fraction_slope_r2": high_fit["r2"],
        "low_fraction_delta": float(desc["low_fraction"].iloc[-1] - desc["low_fraction"].iloc[0]),
        "low_fraction_slope_per_frame": low_fit["slope"],
        "front_radius_high_delta": float(desc["front_radius_high_mean"].iloc[-1] - desc["front_radius_high_mean"].iloc[0]),
        "front_radius_high_slope_px_per_frame": rad_fit["slope"],
        "front_radius_high_slope_r2": rad_fit["r2"],
        "front_radius_high_monotonic_pos_fraction": monotonic_pos,
        "apparent_front_radius2_slope_px2_per_frame": rad2_fit["slope"],
        "apparent_diffusion_proxy_px2_per_frame": float(rad2_fit["slope"] / 4.0) if np.isfinite(rad2_fit["slope"]) else np.nan,
        "front_radius2_slope_r2": rad2_fit["r2"],
        "weighted_radial_mean_delta": float(desc["radial_mean_weighted"].iloc[-1] - desc["radial_mean_weighted"].iloc[0]),
        "weighted_radial_mean_slope_px_per_frame": weighted_rad_fit["slope"],
        "temporal_diff_energy": diff_energy,
        "cumulative_abs_change": cumulative_abs_change,
        "trace_csv": out_trace,
    }


def cluster_modes(summary_df: pd.DataFrame, max_k: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    features = [
        "roi_mean_slope_per_frame",
        "high_fraction_slope_per_frame",
        "low_fraction_slope_per_frame",
        "front_radius_high_slope_px_per_frame",
        "apparent_diffusion_proxy_px2_per_frame",
        "weighted_radial_mean_slope_px_per_frame",
        "temporal_diff_energy",
        "cumulative_abs_change",
    ]
    x = summary_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    xz = StandardScaler().fit_transform(x)
    best = {"k": 1, "silhouette": np.nan, "labels": np.zeros(len(summary_df), dtype=int)}
    for k in range(2, min(max_k, len(summary_df) - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=17, n_init=30).fit_predict(xz)
        try:
            sil = float(silhouette_score(xz, labels))
        except Exception:
            sil = np.nan
        if best["k"] == 1 or (np.isfinite(sil) and sil > best["silhouette"]):
            best = {"k": k, "silhouette": sil, "labels": labels}
    out = summary_df.copy()
    out["mode_cluster"] = best["labels"]
    mode_names = {}
    for cluster, grp in out.groupby("mode_cluster"):
        mean_slope = float(grp["roi_mean_slope_per_frame"].mean())
        front_slope = float(grp["front_radius_high_slope_px_per_frame"].mean())
        diff_proxy = float(grp["apparent_diffusion_proxy_px2_per_frame"].mean())
        if mean_slope < 0 and front_slope < 0:
            name = "darkening_contracting_front"
        elif mean_slope < 0:
            name = "darkening_or_optical_loss"
        elif front_slope > 0:
            name = "brightening_expanding_front"
        else:
            name = "near_static_or_mixed"
        mode_names[int(cluster)] = name
    out["mode_name"] = out["mode_cluster"].map(mode_names)
    meta = {
        "feature_columns": features,
        "selected_k": int(best["k"]),
        "silhouette": best["silhouette"],
        "mode_names": mode_names,
    }
    return out, meta


def save_plots(summary_df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    paths = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sc = axes[0].scatter(
        summary_df["front_radius_high_slope_px_per_frame"],
        summary_df["roi_mean_slope_per_frame"],
        c=summary_df["cycleNo"],
        s=70,
        cmap="viridis",
    )
    for _, row in summary_df.iterrows():
        axes[0].text(row["front_radius_high_slope_px_per_frame"], row["roi_mean_slope_per_frame"], str(int(row["cycleNo"])), fontsize=7)
    axes[0].axhline(0, color="k", linewidth=0.6)
    axes[0].axvline(0, color="k", linewidth=0.6)
    axes[0].set_xlabel("front radius slope px/frame")
    axes[0].set_ylabel("ROI mean slope / frame")
    axes[0].set_title("ROI optical/front trends")
    fig.colorbar(sc, ax=axes[0], label="cycle")
    for cluster, grp in summary_df.groupby("mode_cluster"):
        axes[1].scatter(grp["temporal_diff_energy"], grp["cumulative_abs_change"], s=70, label=f"{cluster}: {grp['mode_name'].iloc[0]}")
    axes[1].set_xlabel("temporal diff energy")
    axes[1].set_ylabel("cumulative abs change")
    axes[1].set_title("ROI dynamics modes")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "roi_physics_descriptor_modes.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["mode_plot"] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/selected_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/roi_physics_descriptors")
    parser.add_argument("--max-k", type=int, default=4)
    args = parser.parse_args()

    manifest_path = os.path.join(args.roi_dir, "selected_roi_sequence_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    os.makedirs(args.out_dir, exist_ok=True)
    trace_dir = os.path.join(args.out_dir, "frame_descriptor_traces")
    os.makedirs(trace_dir, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for _, row in manifest.iterrows():
        data = np.load(row["npz_path"])
        frames = data["frames_norm"].astype(np.float32)
        frame_indices = data["frame_indices"].astype(int)
        trace_path = os.path.join(trace_dir, f"{row['roi_id']}_frame_descriptors.csv")
        rows.append(sequence_summary(row, frames, frame_indices, trace_path))
        print(f"processed {row['roi_id']}")
    summary_df = pd.DataFrame(rows)
    clustered, cluster_meta = cluster_modes(summary_df, args.max_k)
    plot_paths = save_plots(clustered, args.out_dir)

    roi_path = os.path.join(args.out_dir, "roi_physics_descriptors.csv")
    cycle_path = os.path.join(args.out_dir, "roi_physics_cycle_summary.csv")
    clustered.to_csv(roi_path, index=False)
    cycle = clustered.groupby("cycleNo", as_index=False).agg({
        "roi_id": "count",
        "roi_mean_slope_per_frame": "mean",
        "high_fraction_slope_per_frame": "mean",
        "front_radius_high_slope_px_per_frame": "mean",
        "apparent_diffusion_proxy_px2_per_frame": "mean",
        "temporal_diff_energy": "mean",
        "cumulative_abs_change": "mean",
        "weighted_radial_mean_slope_px_per_frame": "mean",
    }).rename(columns={"roi_id": "n_roi"})
    cycle.to_csv(cycle_path, index=False)

    mode_counts = clustered.groupby(["mode_cluster", "mode_name", "cycleNo"]).size().reset_index(name="n_roi")
    mode_path = os.path.join(args.out_dir, "roi_degradation_mode_counts.csv")
    mode_counts.to_csv(mode_path, index=False)
    summary = {
        "roi_dir": args.roi_dir,
        "n_roi": int(len(clustered)),
        "cycle_summary": cycle.to_dict(orient="records"),
        "cluster_meta": cluster_meta,
        "mode_counts": mode_counts.to_dict(orient="records"),
        "guardrail": "Front/radius and diffusion values are normalized-image pixel/frame proxies, not calibrated micron-scale diffusion coefficients.",
        "plots": plot_paths,
    }
    summary_path = os.path.join(args.out_dir, "roi_physics_descriptor_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# ROI Physics Descriptors\n\n")
        f.write("Physics-facing descriptors extracted from selected particle-region NMC ROI sequences.\n\n")
        f.write("Outputs include ROI-level optical trends, front-radius proxies, apparent radius-squared slope proxies, and small-sample degradation-mode clusters.\n\n")
        f.write("All transport quantities are image-coordinate proxies and require spatial/time calibration plus manual ROI validation before physical interpretation.\n")
    for path in [roi_path, cycle_path, mode_path, summary_path, plot_paths["mode_plot"]]:
        print(f"Saved: {path}")
    print(json.dumps(summary, indent=2, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
