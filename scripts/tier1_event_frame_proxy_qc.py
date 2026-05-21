#!/usr/bin/env python3
"""Sample full NMC HDF5 frames around optical event cycles for proxy ROI QC.

The chopped cycle HDF5 files referenced in exampleParticles.csv are not present
on Isambard, but the full session HDF5 files are. This script uses the source
stem and local cycle index from the address field to sample a bounded number of
frames from the corresponding full movie segment. It builds a fixed proxy
particle-region mask per sampled segment from temporal variability and brightness
relative to the border background, then compares ROI/background intensity and
motion/QC metrics.

This is intentionally a proxy QC, not final segmentation. It avoids loading full
movies and does not mutate raw files.
"""

import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def resolve_existing_path(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def read_cycle_frames_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "cycleNo" in df.columns:
        return df
    raw = pd.read_csv(path, encoding="utf-8-sig", header=None)
    if raw.shape[1] < 2:
        raise ValueError("cycleFrames.csv must have at least two columns")
    raw = raw.iloc[:, :2].copy()
    raw.columns = ["cycleNo", "n_frames"]
    return raw


def parse_addr(addr: str) -> Dict[str, object]:
    s = str(addr)
    m = re.search(r"Halfthedata\\([^\\]+)chopped\\[^\\]+_cycle(\d+)\.hdf5", s)
    if not m:
        return {"source_stem": "", "local_cycle_index": np.nan}
    return {"source_stem": m.group(1), "local_cycle_index": int(m.group(2))}


def infer_n_segments(source_stem: str, source_rows: pd.DataFrame) -> int:
    m = re.search(r"_x(\d+)(?:_|$)", str(source_stem))
    if m:
        return max(1, int(m.group(1)))
    vals = pd.to_numeric(source_rows["local_cycle_index"], errors="coerce").dropna()
    if vals.empty:
        return 1
    return int(vals.max()) + 1


def segment_bounds(n_frames: int, local_idx: int, n_segments: int) -> Tuple[int, int]:
    start = int(math.floor(n_frames * local_idx / n_segments))
    end = int(math.floor(n_frames * (local_idx + 1) / n_segments))
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames))
    return start, end


def sample_indices(start: int, end: int, n_samples: int) -> np.ndarray:
    if end <= start:
        return np.array([start], dtype=int)
    n = min(n_samples, max(1, end - start))
    idx = np.linspace(start, end - 1, n, dtype=int)
    return np.unique(idx)


def read_downsampled_movie(movie, indices: np.ndarray, downsample: int) -> np.ndarray:
    frames = []
    for idx in indices:
        arr = movie[int(idx), ::downsample, ::downsample].astype(np.float32)
        frames.append(arr)
    return np.stack(frames, axis=0)


def robust_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def border_values(img: np.ndarray, width: int = 8) -> np.ndarray:
    if min(img.shape) <= 2 * width:
        return img.ravel()
    return np.concatenate([img[:width, :].ravel(), img[-width:, :].ravel(), img[:, :width].ravel(), img[:, -width:].ravel()])


def build_proxy_roi(frames: np.ndarray, min_fraction: float = 0.002, max_fraction: float = 0.65) -> Tuple[np.ndarray, Dict[str, float]]:
    mean_img = np.nanmean(frames, axis=0)
    std_img = np.nanstd(frames, axis=0)
    med_img = np.nanmedian(frames, axis=0)
    border_mean = border_values(mean_img)
    border_std = border_values(std_img)
    bri_thr = float(np.nanmedian(border_mean) + 3.0 * np.nanstd(border_mean))
    dyn_thr = float(np.nanmedian(border_std) + 3.0 * np.nanstd(border_std))
    mask = (mean_img > bri_thr) | (std_img > dyn_thr)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=2)
    labels, nlab = ndi.label(mask)
    if nlab > 0:
        sizes = ndi.sum(mask, labels, index=np.arange(1, nlab + 1))
        largest = int(np.argmax(sizes) + 1)
        mask = labels == largest
        mask = ndi.binary_fill_holes(mask)
    frac = float(mask.mean())
    fallback = False
    if frac < min_fraction or frac > max_fraction:
        score = robust_norm(mean_img) + robust_norm(std_img)
        q = max(0.0, min(100.0, 100.0 * (1.0 - 0.12)))
        mask = score >= np.nanpercentile(score, q)
        mask = ndi.binary_closing(mask, iterations=2)
        labels, nlab = ndi.label(mask)
        if nlab > 0:
            sizes = ndi.sum(mask, labels, index=np.arange(1, nlab + 1))
            mask = labels == int(np.argmax(sizes) + 1)
            mask = ndi.binary_fill_holes(mask)
        fallback = True
    meta = {
        "roi_fraction": float(mask.mean()),
        "brightness_threshold": bri_thr,
        "dynamic_threshold": dyn_thr,
        "fallback_mask": bool(fallback),
    }
    return mask.astype(bool), meta


def summarize_frames(frames: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    bg = ~mask
    if mask.sum() == 0:
        return {}
    roi_vals = frames[:, mask]
    bg_vals = frames[:, bg] if bg.sum() else frames.reshape(frames.shape[0], -1)
    roi_mean_t = np.nanmean(roi_vals, axis=1)
    bg_mean_t = np.nanmean(bg_vals, axis=1)
    contrast_t = roi_mean_t - bg_mean_t
    return {
        "roi_mean": float(np.nanmean(roi_mean_t)),
        "roi_std_time": float(np.nanstd(roi_mean_t)),
        "background_mean": float(np.nanmean(bg_mean_t)),
        "background_std_time": float(np.nanstd(bg_mean_t)),
        "roi_background_contrast": float(np.nanmean(contrast_t)),
        "roi_cv_time": float(np.nanstd(roi_mean_t) / np.nanmean(roi_mean_t)) if np.nanmean(roi_mean_t) else np.nan,
        "contrast_cv_time": float(np.nanstd(contrast_t) / abs(np.nanmean(contrast_t))) if np.nanmean(contrast_t) else np.nan,
        "frame0_roi_mean": float(roi_mean_t[0]),
        "frame_last_roi_mean": float(roi_mean_t[-1]),
        "roi_mean_delta_last_minus_first": float(roi_mean_t[-1] - roi_mean_t[0]),
    }


def save_preview(frames: np.ndarray, mask: np.ndarray, out_png: str, title: str) -> None:
    mean_img = np.nanmean(frames, axis=0)
    std_img = np.nanstd(frames, axis=0)
    first = frames[0]
    last = frames[-1]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    imgs = [first, last, mean_img, std_img, mask.astype(float)]
    titles = ["first", "last", "mean", "temporal std", "proxy ROI"]
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--event-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_targets")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_frame_proxy_qc")
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--samples-per-segment", type=int, default=12)
    args = parser.parse_args()

    particles_path = resolve_existing_path([
        args.particles_csv,
        os.path.join(args.root, "exampleParticles.csv"),
        os.path.join(args.root, "alek_jiho_nmc_deg/echemDF_full/exampleParticles.csv"),
    ])
    frames_path = resolve_existing_path([
        args.cycle_frames_csv,
        os.path.join(args.root, "cycleFrames.csv"),
        os.path.join(args.root, "alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv"),
    ])
    events_path = os.path.join(args.event_dir, "particle_abrupt_events.csv")
    if not particles_path or not frames_path or not os.path.exists(events_path):
        raise SystemExit("Missing required particles/cycle/event files")

    particles = pd.read_csv(particles_path, encoding="utf-8-sig")
    frames_meta = read_cycle_frames_csv(frames_path)
    events = pd.read_csv(events_path)
    particles["cycleNo"] = pd.to_numeric(particles["cycleNo"], errors="coerce")
    frames_meta["cycleNo"] = pd.to_numeric(frames_meta["cycleNo"], errors="coerce")
    frames_meta["n_frames_table"] = pd.to_numeric(frames_meta["n_frames"], errors="coerce")
    parsed = particles["addrs"].map(parse_addr).apply(pd.Series)
    particles = pd.concat([particles, parsed], axis=1).merge(frames_meta[["cycleNo", "n_frames_table"]], on="cycleNo", how="left")
    events["cycleNo"] = pd.to_numeric(events["cycleNo"], errors="coerce")

    event_cycles = sorted(events["cycleNo"].dropna().astype(float).unique())
    # Include immediate represented neighbors from the same source where available.
    target_rows = []
    for cyc in event_cycles:
        row = particles[particles["cycleNo"] == cyc]
        if row.empty:
            continue
        stem = row.iloc[0]["source_stem"]
        local_idx = row.iloc[0]["local_cycle_index"]
        same = particles[particles["source_stem"] == stem].copy()
        same["local_dist"] = (pd.to_numeric(same["local_cycle_index"], errors="coerce") - float(local_idx)).abs()
        target_rows.append(same.sort_values(["local_dist", "cycleNo"]).head(3))
    targets = pd.concat(target_rows, ignore_index=True).drop_duplicates(subset=["cycleNo", "source_stem", "local_cycle_index"]) if target_rows else pd.DataFrame()

    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)

    rows = []
    for stem, source_rows in targets.groupby("source_stem", sort=True):
        if not stem:
            continue
        h5_path = os.path.join(args.root, "NMC_degradation_3_160623_Halfthedata", f"{stem}.hdf5")
        if not os.path.exists(h5_path):
            print(f"warning: missing {h5_path}")
            continue
        n_segments = infer_n_segments(stem, particles[particles["source_stem"] == stem])
        with h5py.File(h5_path, "r") as f:
            movie = f["movie"]
            total_frames = int(movie.shape[0])
            stage = np.asarray(f["stage_position"][:, :], dtype=float) if "stage_position" in f else None
            avg = np.asarray(f["average_intensity"][0, :], dtype=float) if "average_intensity" in f else None
            for _, row in source_rows.sort_values("local_cycle_index").iterrows():
                local_idx = int(row["local_cycle_index"])
                start, end = segment_bounds(total_frames, local_idx, n_segments)
                idx = sample_indices(start, end, args.samples_per_segment)
                sampled = read_downsampled_movie(movie, idx, args.downsample)
                mask, mask_meta = build_proxy_roi(sampled)
                metrics = summarize_frames(sampled, mask)
                stage_drift_xy = np.nan
                stage_drift_z = np.nan
                if stage is not None and stage.shape[1] > max(idx):
                    st = stage[:, idx]
                    stage_drift_xy = float(np.sqrt((np.nanmax(st[0])-np.nanmin(st[0]))**2 + (np.nanmax(st[1])-np.nanmin(st[1]))**2))
                    stage_drift_z = float(np.nanmax(st[2]) - np.nanmin(st[2]))
                avg_mean = float(np.nanmean(avg[idx])) if avg is not None and len(avg) > max(idx) else np.nan
                avg_delta = float(avg[idx[-1]] - avg[idx[0]]) if avg is not None and len(avg) > max(idx) else np.nan
                cycle = float(row["cycleNo"])
                is_event = cycle in event_cycles
                particle_event_count = int((events["cycleNo"] == cycle).sum())
                preview_path = os.path.join(preview_dir, f"cycle_{int(cycle)}_{stem}_local{local_idx}.png")
                save_preview(sampled, mask, preview_path, f"cycle {cycle:g} | {stem} local {local_idx} | event={is_event}")
                out = {
                    "cycleNo": cycle,
                    "source_stem": stem,
                    "local_cycle_index": local_idx,
                    "is_event_cycle": bool(is_event),
                    "particle_event_count": particle_event_count,
                    "n_segments_inferred": n_segments,
                    "h5_total_frames": total_frames,
                    "segment_start_frame": int(start),
                    "segment_end_frame": int(end),
                    "n_sampled_frames": int(len(idx)),
                    "sampled_first_frame": int(idx[0]),
                    "sampled_last_frame": int(idx[-1]),
                    "cycle_table_n_frames": float(row.get("n_frames_table", np.nan)),
                    "stage_drift_xy_sampled": stage_drift_xy,
                    "stage_drift_z_sampled": stage_drift_z,
                    "average_intensity_sample_mean": avg_mean,
                    "average_intensity_sample_delta": avg_delta,
                    "preview_png": preview_path,
                }
                out.update(mask_meta)
                out.update(metrics)
                rows.append(out)

    result = pd.DataFrame(rows).sort_values(["source_stem", "local_cycle_index", "cycleNo"]) if rows else pd.DataFrame()
    csv_path = os.path.join(args.out_dir, "event_frame_proxy_qc.csv")
    result.to_csv(csv_path, index=False)

    event_df = result[result["is_event_cycle"] == True]
    non_event_df = result[result["is_event_cycle"] == False]
    summary = {
        "particles_csv": particles_path,
        "cycle_frames_csv": frames_path,
        "event_dir": args.event_dir,
        "downsample": args.downsample,
        "samples_per_segment": args.samples_per_segment,
        "n_rows": int(len(result)),
        "n_event_rows": int(len(event_df)),
        "n_neighbor_rows": int(len(non_event_df)),
        "event_cycles_sampled": sorted(event_df["cycleNo"].astype(float).tolist()) if not event_df.empty else [],
        "mean_roi_fraction_event": float(event_df["roi_fraction"].mean()) if not event_df.empty else np.nan,
        "mean_roi_fraction_neighbors": float(non_event_df["roi_fraction"].mean()) if not non_event_df.empty else np.nan,
        "mean_stage_drift_xy_event": float(event_df["stage_drift_xy_sampled"].mean()) if not event_df.empty else np.nan,
        "mean_stage_drift_xy_neighbors": float(non_event_df["stage_drift_xy_sampled"].mean()) if not non_event_df.empty else np.nan,
        "mean_roi_cv_event": float(event_df["roi_cv_time"].mean()) if not event_df.empty else np.nan,
        "mean_roi_cv_neighbors": float(non_event_df["roi_cv_time"].mean()) if not non_event_df.empty else np.nan,
        "rows": result.to_dict(orient="records") if not result.empty else [],
    }
    summary_path = os.path.join(args.out_dir, "event_frame_proxy_qc_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    readme_path = os.path.join(args.out_dir, "README.md")
    lines = [
        "# NMC Event Frame Proxy QC",
        "",
        "Samples full HDF5 movies around event-cycle segments and builds a fixed proxy ROI mask from temporal variability and brightness. This is a bounded visual/QC check because chopped cycle HDF5 files are unavailable on Isambard.",
        "",
        "## Summary",
        "",
        f"- Rows sampled: {summary['n_rows']}",
        f"- Event rows: {summary['n_event_rows']}",
        f"- Neighbor rows: {summary['n_neighbor_rows']}",
        f"- Mean event ROI fraction: {summary['mean_roi_fraction_event']:.4f}",
        f"- Mean neighbor ROI fraction: {summary['mean_roi_fraction_neighbors']:.4f}",
        f"- Mean event sampled XY stage drift: {summary['mean_stage_drift_xy_event']:.4g}",
        f"- Mean neighbor sampled XY stage drift: {summary['mean_stage_drift_xy_neighbors']:.4g}",
        "",
        "## Sampled Segments",
        "",
        "| cycleNo | event | source_stem | local_idx | proxy_roi_fraction | roi_cv_time | stage_drift_xy | preview |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in result.iterrows():
        rel_preview = os.path.relpath(row["preview_png"], args.out_dir)
        lines.append("| {cycleNo:.1f} | {event} | `{stem}` | {local} | {roi:.4f} | {cv:.4g} | {drift:.4g} | `{preview}` |".format(
            cycleNo=float(row["cycleNo"]),
            event="yes" if bool(row["is_event_cycle"]) else "no",
            stem=row["source_stem"],
            local=int(row["local_cycle_index"]),
            roi=float(row["roi_fraction"]),
            cv=float(row["roi_cv_time"]),
            drift=float(row["stage_drift_xy_sampled"]),
            preview=rel_preview,
        ))
    lines.extend([
        "",
        "## Interpretation Guardrail",
        "",
        "The ROI is a proxy mask at downsampled full-frame resolution. It is useful for checking gross frame quality, drift, and whether a stable bright/dynamic particle-like region exists, but it is not the original object detector output and should not be used as final particle segmentation.",
        "",
    ])
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))

    for p in [csv_path, summary_path, readme_path]:
        print(f"Saved: {p}")
    print(json.dumps(summary, indent=2, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
