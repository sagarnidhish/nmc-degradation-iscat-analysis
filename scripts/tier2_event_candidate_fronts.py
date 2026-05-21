#!/usr/bin/env python3
"""Candidate ROI and apparent front-motion extraction around NMC event cycles.

This is a bounded full-HDF5 analysis for the Alek/Jiho NMC degradation data. It
uses event-frame proxy QC rows to locate source HDF5 movie segments, detects
candidate particle-like connected components from temporal mean/std images, and
extracts optical state/front proxies for each candidate.

Important: outputs are not calibrated diffusion coefficients. Front-radius and
D_app values are downsampled-pixel/frame proxies intended to rank candidates for
manual validation and future exact object-detector ROI recovery.
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
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation


def robust_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(arr, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def sample_indices(start: int, end: int, n_samples: int) -> np.ndarray:
    n = min(n_samples, max(1, end - start))
    return np.unique(np.linspace(start, end - 1, n, dtype=int))


def read_downsampled(movie, indices: np.ndarray, downsample: int) -> np.ndarray:
    frames = []
    for idx in indices:
        frames.append(movie[int(idx), ::downsample, ::downsample].astype(np.float32))
    return np.stack(frames, axis=0)


def border_values(img: np.ndarray, width: int = 8) -> np.ndarray:
    if min(img.shape) <= 2 * width:
        return img.ravel()
    return np.concatenate([img[:width, :].ravel(), img[-width:, :].ravel(), img[:, :width].ravel(), img[:, -width:].ravel()])


def detect_candidates(frames: np.ndarray, max_candidates: int, min_area: int, max_area_frac: float) -> Tuple[List[Dict[str, object]], np.ndarray]:
    mean_img = np.nanmean(frames, axis=0)
    std_img = np.nanstd(frames, axis=0)
    score = robust_norm(mean_img - np.nanmedian(border_values(mean_img))) + robust_norm(std_img)
    try:
        thr = filters.threshold_otsu(score[np.isfinite(score)])
    except Exception:
        thr = float(np.nanpercentile(score, 88))
    mask = score >= max(thr, float(np.nanpercentile(score, 82)))
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area)
    mask = morphology.binary_closing(mask, morphology.disk(2))
    labels = measure.label(mask)
    props = measure.regionprops(labels, intensity_image=score)
    img_area = float(mask.size)
    candidates = []
    for prop in props:
        area = int(prop.area)
        if area < min_area or area > max_area_frac * img_area:
            continue
        rr, cc = prop.coords[:, 0], prop.coords[:, 1]
        candidate_score = float(np.nanmean(score[rr, cc]) * math.sqrt(area))
        candidates.append({
            "label": int(prop.label),
            "area_ds_px": area,
            "centroid_y_ds": float(prop.centroid[0]),
            "centroid_x_ds": float(prop.centroid[1]),
            "bbox_min_row": int(prop.bbox[0]),
            "bbox_min_col": int(prop.bbox[1]),
            "bbox_max_row": int(prop.bbox[2]),
            "bbox_max_col": int(prop.bbox[3]),
            "eccentricity": float(prop.eccentricity),
            "solidity": float(prop.solidity),
            "candidate_score": candidate_score,
        })
    candidates = sorted(candidates, key=lambda r: r["candidate_score"], reverse=True)[:max_candidates]
    return candidates, labels


def fit_line(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
    xx = x[good].astype(float)
    yy = y[good].astype(float)
    A = np.vstack([xx, np.ones_like(xx)]).T
    slope, intercept = np.linalg.lstsq(A, yy, rcond=None)[0]
    pred = slope * xx + intercept
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def candidate_time_metrics(frames: np.ndarray, mask: np.ndarray, frame_indices: np.ndarray, downsample: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return {}, pd.DataFrame()
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    radius = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    first_vals = frames[0][mask]
    base_med = float(np.nanmedian(first_vals))
    base_std = float(np.nanstd(first_vals))
    high_thr = base_med + 0.50 * base_std
    rows = []
    for t, fidx in enumerate(frame_indices):
        vals = frames[t][mask].astype(float)
        high = vals >= high_thr
        if np.any(high):
            front_radius = float(np.nanmean(radius[high]))
            front_p90 = float(np.nanpercentile(radius[high], 90))
        else:
            front_radius = np.nan
            front_p90 = np.nan
        rows.append({
            "sample_idx": int(t),
            "frame_index": int(fidx),
            "relative_frame": int(fidx - frame_indices[0]),
            "roi_mean": float(np.nanmean(vals)),
            "roi_p10": float(np.nanpercentile(vals, 10)),
            "roi_p90": float(np.nanpercentile(vals, 90)),
            "high_fraction": float(np.mean(high)),
            "front_radius_ds_px": front_radius,
            "front_radius_p90_ds_px": front_p90,
        })
    trace = pd.DataFrame(rows)
    x = trace["relative_frame"].to_numpy(dtype=float)
    mean_fit = fit_line(x, trace["roi_mean"].to_numpy(dtype=float))
    high_fit = fit_line(x, trace["high_fraction"].to_numpy(dtype=float))
    rad = trace["front_radius_ds_px"].to_numpy(dtype=float)
    rad_fit = fit_line(x, rad)
    rad2_fit = fit_line(x, rad ** 2)
    monotonic_steps = np.diff(rad[np.isfinite(rad)])
    monotonic_fraction = float(np.mean(monotonic_steps >= 0)) if monotonic_steps.size else np.nan
    metrics = {
        "downsample": int(downsample),
        "high_threshold_from_first_frame": high_thr,
        "roi_mean_first": float(trace["roi_mean"].iloc[0]),
        "roi_mean_last": float(trace["roi_mean"].iloc[-1]),
        "roi_mean_delta_last_minus_first": float(trace["roi_mean"].iloc[-1] - trace["roi_mean"].iloc[0]),
        "roi_mean_slope_per_frame": mean_fit["slope"],
        "roi_mean_slope_r2": mean_fit["r2"],
        "high_fraction_first": float(trace["high_fraction"].iloc[0]),
        "high_fraction_last": float(trace["high_fraction"].iloc[-1]),
        "high_fraction_slope_per_frame": high_fit["slope"],
        "front_radius_slope_ds_px_per_frame": rad_fit["slope"],
        "front_radius_slope_r2": rad_fit["r2"],
        "front_radius_monotonic_fraction": monotonic_fraction,
        "apparent_diffusion_proxy_ds_px2_per_frame": float(rad2_fit["slope"] / 4.0) if np.isfinite(rad2_fit["slope"]) else np.nan,
        "front_radius2_slope_r2": rad2_fit["r2"],
    }
    return metrics, trace


def save_candidate_preview(frames: np.ndarray, labels: np.ndarray, candidates: List[Dict[str, object]], path: str, title: str) -> None:
    mean_img = np.nanmean(frames, axis=0)
    std_img = np.nanstd(frames, axis=0)
    outline = np.zeros_like(mean_img, dtype=bool)
    for cand in candidates:
        outline |= segmentation.find_boundaries(labels == int(cand["label"]), mode="outer")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, img, label in zip(axes, [frames[0], frames[-1], mean_img, std_img], ["first", "last", "mean", "temporal std"]):
        ax.imshow(img, cmap="gray")
        ax.contour(outline, colors="r", linewidths=0.6)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--proxy-qc", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_frame_proxy_qc/event_frame_proxy_qc.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_candidate_fronts")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--samples-per-segment", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=60)
    parser.add_argument("--max-area-frac", type=float, default=0.08)
    parser.add_argument("--event-only", action="store_true")
    args = parser.parse_args()

    proxy = pd.read_csv(args.proxy_qc)
    if args.event_only:
        proxy = proxy[pd.to_numeric(proxy["is_event_cycle"], errors="coerce").fillna(0).astype(bool)].copy()
    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "candidate_previews")
    trace_dir = os.path.join(args.out_dir, "candidate_traces")
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)

    candidate_rows = []
    trace_paths = []
    for _, row in proxy.sort_values(["source_stem", "cycleNo"]).iterrows():
        h5_path = os.path.join(args.root, "NMC_degradation_3_160623_Halfthedata", f"{row['source_stem']}.hdf5")
        if not os.path.exists(h5_path):
            print(f"warning: missing {h5_path}")
            continue
        start = int(row["segment_start_frame"])
        end = int(row["segment_end_frame"])
        frame_indices = sample_indices(start, end, args.samples_per_segment)
        with h5py.File(h5_path, "r") as f:
            frames = read_downsampled(f["movie"], frame_indices, args.downsample)
        candidates, labels = detect_candidates(frames, args.max_candidates, args.min_area, args.max_area_frac)
        preview_path = os.path.join(preview_dir, f"cycle_{int(row['cycleNo'])}_{row['source_stem']}_candidates.png")
        save_candidate_preview(frames, labels, candidates, preview_path, f"cycle {row['cycleNo']} {row['source_stem']}")
        for rank, cand in enumerate(candidates, 1):
            mask = labels == int(cand["label"])
            metrics, trace = candidate_time_metrics(frames, mask, frame_indices, args.downsample)
            trace_name = f"cycle_{int(row['cycleNo'])}_{row['source_stem']}_cand{rank}.csv"
            trace_path = os.path.join(trace_dir, trace_name)
            trace.insert(0, "candidate_rank", rank)
            trace.insert(0, "cycleNo", float(row["cycleNo"]))
            trace.to_csv(trace_path, index=False)
            trace_paths.append(trace_path)
            out = {
                "cycleNo": float(row["cycleNo"]),
                "source_stem": row["source_stem"],
                "is_event_cycle": bool(row["is_event_cycle"]),
                "particle_event_count": int(row.get("particle_event_count", 0)),
                "candidate_rank": rank,
                "preview_png": preview_path,
                "trace_csv": trace_path,
            }
            out.update(cand)
            out.update(metrics)
            candidate_rows.append(out)
        print(f"processed cycle={row['cycleNo']} candidates={len(candidates)}")

    cdf = pd.DataFrame(candidate_rows)
    if not cdf.empty:
        # Rank candidates useful for physical follow-up: coherent front movement plus event relevance.
        cdf["front_quality_score"] = (
            cdf["front_radius_slope_r2"].fillna(0).clip(lower=0)
            + cdf["front_radius2_slope_r2"].fillna(0).clip(lower=0)
            + cdf["front_radius_monotonic_fraction"].fillna(0)
            + 0.5 * cdf["is_event_cycle"].astype(float)
        )
        cdf = cdf.sort_values(["front_quality_score", "candidate_score"], ascending=[False, False]).reset_index(drop=True)
    cdf.to_csv(os.path.join(args.out_dir, "candidate_front_metrics.csv"), index=False)

    summary = {
        "proxy_qc": args.proxy_qc,
        "n_segment_rows": int(len(proxy)),
        "n_candidate_rows": int(len(cdf)),
        "downsample": args.downsample,
        "samples_per_segment": args.samples_per_segment,
        "top_candidates": cdf.head(12).to_dict(orient="records") if not cdf.empty else [],
        "guardrail": "apparent_diffusion_proxy_ds_px2_per_frame is in downsampled-pixel^2/frame units and is not a calibrated diffusion coefficient.",
    }
    with open(os.path.join(args.out_dir, "event_candidate_fronts_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Event Candidate Fronts\n\n")
        f.write("Candidate particle-region and apparent front-motion proxy extraction around NMC optical event cycles and neighbors.\n\n")
        f.write("Outputs are ranking/QC proxies, not calibrated diffusion coefficients.\n")
    print(f"[done] wrote {len(cdf)} candidate rows to {args.out_dir}")


if __name__ == "__main__":
    main()
