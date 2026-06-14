#!/usr/bin/env python3
"""High-resolution front tracking for selected NMC front ROI candidates.

This is the follow-up to validated front ROI selection. It re-reads full HDF5
movie crops around the selected front/object candidates, tracks signed optical
change fronts over the sampled event segment, and fits radius^2 versus elapsed
seconds. The fitted slopes are apparent pixel-scale transport proxies only; they
are not micron-scale diffusion coefficients without spatial calibration and
manual ROI QC.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def finite_float(x, default=np.nan) -> float:
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def crop_bounds(cx: float, cy: float, crop_size: int, width: int, height: int) -> Tuple[int, int, int, int]:
    half = crop_size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def robust_mad(x: np.ndarray) -> float:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return 1.4826 * mad if np.isfinite(mad) else np.nan


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(quantile, cdf, v))


def linear_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n_fit": int(mask.sum())}
    xx = x[mask].astype(float)
    yy = y[mask].astype(float)
    slope, intercept = np.polyfit(xx, yy, 1)
    pred = slope * xx + intercept
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "n_fit": int(mask.sum())}


def load_trace_indices(trace_csv: str) -> np.ndarray:
    trace = pd.read_csv(trace_csv)
    if "frame_index" not in trace:
        raise ValueError(f"trace_csv lacks frame_index: {trace_csv}")
    return pd.to_numeric(trace["frame_index"], errors="coerce").dropna().astype(int).to_numpy()


def read_crop_stack(root: str, row: pd.Series, crop_size: int) -> Dict[str, object]:
    source_stem = str(row["source_stem"])
    h5_path = os.path.join(root, "NMC_degradation_3_160623_Halfthedata", f"{source_stem}.hdf5")
    frame_indices = load_trace_indices(str(row["front_trace_csv"]))
    downsample = int(finite_float(row.get("downsample", 4), 4))
    cx = finite_float(row.get("object_x_ds"), finite_float(row.get("front_centroid_x_ds"))) * downsample
    cy = finite_float(row.get("object_y_ds"), finite_float(row.get("front_centroid_y_ds"))) * downsample
    with h5py.File(h5_path, "r") as f:
        movie = f["movie"]
        height, width = int(movie.shape[1]), int(movie.shape[2])
        x0, y0, x1, y1 = crop_bounds(cx, cy, crop_size, width, height)
        valid_idx = frame_indices[(frame_indices >= 0) & (frame_indices < movie.shape[0])]
        frames = np.stack([np.asarray(movie[int(idx), y0:y1, x0:x1], dtype=np.float32) for idx in valid_idx], axis=0)
        avg = np.asarray(f["average_intensity"][0, valid_idx], dtype=np.float32) if "average_intensity" in f else np.full(len(valid_idx), np.nan, dtype=np.float32)
        timing = np.asarray(f["camera_timing"], dtype=np.float64) if "camera_timing" in f else np.empty((0, 0))
        if timing.ndim == 2 and timing.shape[0] > 1 and timing.shape[1] > int(np.max(valid_idx)):
            seconds = timing[1, valid_idx].astype(float)
            seconds = seconds - float(seconds[0])
        else:
            seconds = (valid_idx - int(valid_idx[0])).astype(float)
        stage = np.asarray(f["stage_position"][:, valid_idx], dtype=np.float32) if "stage_position" in f else np.full((3, len(valid_idx)), np.nan, dtype=np.float32)
    return {
        "frames": frames,
        "frame_indices": valid_idx,
        "seconds": seconds,
        "average_intensity": avg,
        "stage_position": stage,
        "crop_bounds": (x0, y0, x1, y1),
        "center_crop_xy": (cx - x0, cy - y0),
        "center_full_xy": (cx, cy),
        "h5_path": h5_path,
    }


def track_front(stack: Dict[str, object], row: pd.Series, baseline_frames: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    frames = np.asarray(stack["frames"], dtype=np.float32)
    avg = np.asarray(stack["average_intensity"], dtype=np.float32)
    if np.isfinite(avg).all() and np.nanmedian(avg) > 0:
        frames = frames / avg[:, None, None] * float(np.nanmedian(avg))
    n_base = max(2, min(baseline_frames, max(2, frames.shape[0] // 6)))
    baseline = np.nanmedian(frames[:n_base], axis=0)
    last_delta = np.nanmedian(frames[-n_base:], axis=0) - baseline
    mean_delta = float(np.nanmean(last_delta))
    direction = 1.0 if mean_delta >= 0 else -1.0
    signed_last = direction * last_delta
    noise = robust_mad(direction * (frames[:n_base] - baseline))
    final_positive = signed_last[np.isfinite(signed_last)]
    q75 = float(np.nanpercentile(final_positive, 75)) if final_positive.size else np.nan
    threshold = max(3.0 * noise if np.isfinite(noise) else -np.inf, q75 if np.isfinite(q75) else -np.inf, 1e-6)
    yy, xx = np.indices(frames.shape[1:])
    cx, cy = stack["center_crop_xy"]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rows: List[Dict[str, object]] = []
    for i, frame in enumerate(frames):
        signed = direction * (frame - baseline)
        positive = np.clip(signed - threshold, 0, None)
        mask = positive > 0
        area = int(np.count_nonzero(mask))
        weights = positive[mask]
        radii = rr[mask]
        if area > 0 and np.sum(weights) > 0:
            weighted_radius = float(np.sum(radii * weights) / np.sum(weights))
            radius_p90 = weighted_quantile(radii, weights, 0.90)
            radius2_weighted = float(np.sum((radii ** 2) * weights) / np.sum(weights))
        else:
            weighted_radius = np.nan
            radius_p90 = np.nan
            radius2_weighted = np.nan
        rows.append({
            "roi_id": f"cycle{int(row['cycleNo'])}_front{int(row['front_candidate_rank'])}_obj{int(row['object_candidate_rank'])}",
            "cycleNo": float(row["cycleNo"]),
            "source_stem": row["source_stem"],
            "front_candidate_rank": int(row["front_candidate_rank"]),
            "object_candidate_rank": int(row["object_candidate_rank"]),
            "sample_index": int(i),
            "frame_index": int(stack["frame_indices"][i]),
            "elapsed_s": float(stack["seconds"][i]),
            "roi_mean_corrected": float(np.nanmean(frame)),
            "signed_delta_mean": float(np.nanmean(signed)),
            "signed_delta_p90": float(np.nanpercentile(signed, 90)),
            "active_threshold": float(threshold),
            "active_area_full_px": area,
            "active_fraction": float(area / signed.size),
            "front_weighted_radius_full_px": weighted_radius,
            "front_radius_p90_full_px": radius_p90,
            "front_radius2_weighted_full_px2": radius2_weighted,
            "validation_score": finite_float(row.get("validation_score")),
        })
    trace = pd.DataFrame(rows)
    fit_area = linear_fit(trace["elapsed_s"].to_numpy(), (trace["active_area_full_px"].to_numpy(dtype=float) / math.pi))
    fit_radius = linear_fit(trace["elapsed_s"].to_numpy(), trace["front_radius2_weighted_full_px2"].to_numpy(dtype=float))
    fit_p90 = linear_fit(trace["elapsed_s"].to_numpy(), trace["front_radius_p90_full_px"].to_numpy(dtype=float) ** 2)
    stage = np.asarray(stack["stage_position"], dtype=float)
    stage_drift = float(np.sqrt((np.nanmax(stage[0]) - np.nanmin(stage[0])) ** 2 + (np.nanmax(stage[1]) - np.nanmin(stage[1])) ** 2)) if np.isfinite(stage).any() else np.nan
    meta = {
        "direction": float(direction),
        "mean_last_delta": mean_delta,
        "baseline_frames": int(n_base),
        "threshold": float(threshold),
        "noise_mad_sigma": float(noise) if np.isfinite(noise) else np.nan,
        "radius2_slope_full_px2_per_s": fit_radius["slope"],
        "radius2_slope_r2": fit_radius["r2"],
        "radius2_slope_n_fit": fit_radius["n_fit"],
        "apparent_diffusion_proxy_full_px2_per_s": fit_radius["slope"] / 4.0 if np.isfinite(fit_radius["slope"]) else np.nan,
        "area_equiv_radius2_slope_full_px2_per_s": fit_area["slope"],
        "area_equiv_radius2_slope_r2": fit_area["r2"],
        "p90_radius2_slope_full_px2_per_s": fit_p90["slope"],
        "p90_radius2_slope_r2": fit_p90["r2"],
        "stage_drift_xy": stage_drift,
    }
    return trace, meta


def save_front_plot(trace: pd.DataFrame, out_png: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes[0, 0].plot(trace["elapsed_s"], trace["roi_mean_corrected"], marker="o", ms=3)
    axes[0, 0].set_title("corrected ROI mean", fontsize=9)
    axes[0, 1].plot(trace["elapsed_s"], trace["active_fraction"], marker="o", ms=3)
    axes[0, 1].set_title("active signed-change fraction", fontsize=9)
    axes[1, 0].plot(trace["elapsed_s"], trace["front_radius2_weighted_full_px2"], marker="o", ms=3)
    axes[1, 0].set_title("weighted radius^2", fontsize=9)
    axes[1, 1].plot(trace["elapsed_s"], trace["front_radius_p90_full_px"] ** 2, marker="o", ms=3)
    axes[1, 1].set_title("p90 radius^2", fontsize=9)
    for ax in axes.ravel():
        ax.set_xlabel("elapsed seconds")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_crop_preview(stack: Dict[str, object], out_png: str, title: str) -> None:
    frames = np.asarray(stack["frames"], dtype=float)
    idxs = [0, len(frames) // 2, len(frames) - 1]
    baseline = np.nanmedian(frames[:max(2, min(5, len(frames)))], axis=0)
    diff = frames[idxs[-1]] - baseline
    lo, hi = np.nanpercentile(frames, [1, 99])
    vmax = max(1.0, float(np.nanpercentile(np.abs(diff), 99)))
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, idx, label in zip(axes[:3], idxs, ["first", "middle", "last"]):
        ax.imshow(frames[idx], cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    axes[3].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[3].set_title("last - baseline", fontsize=9)
    axes[3].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--roi-table", default="/scratch/<account>/<username>/Alek_Jiho/derived/validated_front_rois/selected_front_rois_for_tracking.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/selected_front_roi_tracking")
    parser.add_argument("--crop-size-full", type=int, default=192)
    parser.add_argument("--baseline-frames", type=int, default=5)
    args = parser.parse_args()

    rois = pd.read_csv(args.roi_table)
    if "selected_for_next_tracking" in rois:
        rois = rois[rois["selected_for_next_tracking"].astype(bool)].copy()
    os.makedirs(args.out_dir, exist_ok=True)
    trace_dir = os.path.join(args.out_dir, "roi_traces")
    plot_dir = os.path.join(args.out_dir, "plots")
    preview_dir = os.path.join(args.out_dir, "crop_previews")
    for path in [trace_dir, plot_dir, preview_dir]:
        os.makedirs(path, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    for _, row in rois.sort_values(["cycleNo", "validation_score"], ascending=[True, False]).iterrows():
        roi_id = f"cycle{int(row['cycleNo'])}_front{int(row['front_candidate_rank'])}_obj{int(row['object_candidate_rank'])}"
        stack = read_crop_stack(args.root, row, args.crop_size_full)
        trace, meta = track_front(stack, row, args.baseline_frames)
        trace_path = os.path.join(trace_dir, f"{roi_id}_front_tracking.csv")
        trace.to_csv(trace_path, index=False)
        plot_path = os.path.join(plot_dir, f"{roi_id}_front_tracking.png")
        preview_path = os.path.join(preview_dir, f"{roi_id}_crop_preview.png")
        save_front_plot(trace, plot_path, roi_id)
        save_crop_preview(stack, preview_path, roi_id)
        x0, y0, x1, y1 = stack["crop_bounds"]
        cx, cy = stack["center_full_xy"]
        out = {
            "roi_id": roi_id,
            "cycleNo": float(row["cycleNo"]),
            "source_stem": row["source_stem"],
            "front_candidate_rank": int(row["front_candidate_rank"]),
            "object_candidate_rank": int(row["object_candidate_rank"]),
            "validation_score": finite_float(row.get("validation_score")),
            "validation_tier": row.get("validation_tier", ""),
            "crop_x0": int(x0),
            "crop_y0": int(y0),
            "crop_x1": int(x1),
            "crop_y1": int(y1),
            "center_x_full_px": float(cx),
            "center_y_full_px": float(cy),
            "n_frames": int(len(trace)),
            "elapsed_s": float(trace["elapsed_s"].max()),
            "roi_mean_delta_corrected": float(trace["roi_mean_corrected"].iloc[-1] - trace["roi_mean_corrected"].iloc[0]),
            "active_fraction_first": float(trace["active_fraction"].iloc[0]),
            "active_fraction_last": float(trace["active_fraction"].iloc[-1]),
            "trace_csv": trace_path,
            "plot_png": plot_path,
            "crop_preview_png": preview_path,
        }
        out.update(meta)
        summary_rows.append(out)
        print(f"tracked {roi_id}: slope={out['radius2_slope_full_px2_per_s']:.6g} r2={out['radius2_slope_r2']:.3g}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "selected_front_roi_tracking_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    cycle_summary = summary_df.groupby("cycleNo").agg({
        "roi_id": "count",
        "radius2_slope_full_px2_per_s": "mean",
        "radius2_slope_r2": "mean",
        "apparent_diffusion_proxy_full_px2_per_s": "mean",
        "roi_mean_delta_corrected": "mean",
        "active_fraction_last": "mean",
        "stage_drift_xy": "mean",
    }).reset_index().rename(columns={"roi_id": "n_tracked_rois"})
    cycle_path = os.path.join(args.out_dir, "selected_front_roi_cycle_summary.csv")
    cycle_summary.to_csv(cycle_path, index=False)
    result = {
        "roi_table": args.roi_table,
        "n_tracked_rois": int(len(summary_df)),
        "crop_size_full": int(args.crop_size_full),
        "cycle_summary": cycle_summary.to_dict(orient="records"),
        "guardrail": "Radius-squared slopes are apparent full-pixel/second front-motion proxies from automatic ROIs, not calibrated diffusion coefficients without spatial calibration and manual QC.",
    }
    json_path = os.path.join(args.out_dir, "selected_front_roi_tracking_summary.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Selected Front ROI Tracking\n\n")
        f.write("High-resolution crop tracking for the validated cycle-86/116 front ROI candidates.\n\n")
        f.write("The radius-squared fits are apparent full-pixel/second transport proxies, not calibrated diffusion coefficients without spatial calibration and manual QC.\n")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cycle_path}")
    print(f"Saved: {json_path}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
