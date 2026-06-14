#!/usr/bin/env python3
"""Reconstruct approximate NMC object candidates from sampled full HDF5 movies.

The original object-detector outputs referenced by exampleParticles.csv are not
available on Isambard, so this script rebuilds a bounded, auditable set of
candidate particle-like regions from the full session movies. It samples a small
number of frames per event/neighbor segment, detects robust bright/dark
components on a downsampled mean image, and matches candidates between adjacent
segments from the same source movie.

These candidates are not the legacy particleInfo.csv coordinates. They are a
practical validation layer for checking whether event segments contain stable
object-like regions that can be inspected and compared with neighboring cycles.
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
from matplotlib.patches import Rectangle
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
    return int(vals.max()) + 1 if not vals.empty else 1


def segment_bounds(n_frames: int, local_idx: int, n_segments: int) -> Tuple[int, int]:
    start = int(math.floor(n_frames * local_idx / n_segments))
    end = int(math.floor(n_frames * (local_idx + 1) / n_segments))
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames))
    return start, end


def sample_indices(start: int, end: int, n_samples: int) -> np.ndarray:
    n = min(n_samples, max(1, end - start))
    return np.unique(np.linspace(start, end - 1, n, dtype=int))


def read_downsampled_movie(movie, indices: np.ndarray, downsample: int) -> np.ndarray:
    return np.stack([movie[int(idx), ::downsample, ::downsample].astype(np.float32) for idx in indices], axis=0)


def robust_scale(arr: np.ndarray) -> Tuple[float, float]:
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(arr))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return med, sigma


def detect_candidates(
    frames: np.ndarray,
    downsample: int,
    z_threshold: float,
    min_area: int,
    max_area_fraction: float,
    max_candidates: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mean_img = np.nanmean(frames, axis=0)
    std_img = np.nanstd(frames, axis=0)
    smooth = ndi.gaussian_filter(mean_img, sigma=8.0)
    residual = mean_img - smooth
    med, sigma = robust_scale(residual)
    z = (residual - med) / sigma
    score = np.maximum(np.abs(z), np.nan_to_num(std_img / (np.nanmedian(std_img) + 1e-6), nan=0.0))

    mask = (np.abs(z) >= z_threshold) | (score >= z_threshold + 0.5)
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=1)

    labels, nlab = ndi.label(mask)
    objects = ndi.find_objects(labels)
    max_area = max(min_area, int(mask.size * max_area_fraction))
    rows = []
    for lab, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        yy, xx = np.where(labels[slc] == lab)
        if yy.size == 0:
            continue
        y0 = int(slc[0].start)
        x0 = int(slc[1].start)
        ys = yy + y0
        xs = xx + x0
        area = int(ys.size)
        if area < min_area or area > max_area:
            continue
        vals = residual[ys, xs]
        zvals = z[ys, xs]
        std_vals = std_img[ys, xs]
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        contrast = float(np.nanmean(vals))
        abs_z_mean = float(np.nanmean(np.abs(zvals)))
        rank_score = float(area * abs(contrast) * max(abs_z_mean, 1.0))
        rows.append({
            "candidate_id": lab,
            "x_ds": cx,
            "y_ds": cy,
            "x_full_approx": cx * downsample,
            "y_full_approx": cy * downsample,
            "area_ds_px": area,
            "area_full_approx_px": area * downsample * downsample,
            "bbox_x0_ds": int(xs.min()),
            "bbox_y0_ds": int(ys.min()),
            "bbox_x1_ds": int(xs.max()) + 1,
            "bbox_y1_ds": int(ys.max()) + 1,
            "mean_residual": contrast,
            "mean_abs_z": abs_z_mean,
            "mean_temporal_std": float(np.nanmean(std_vals)),
            "polarity": "bright" if contrast >= 0 else "dark",
            "rank_score": rank_score,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("rank_score", ascending=False).head(max_candidates).reset_index(drop=True)
        out["candidate_rank"] = np.arange(1, len(out) + 1)
    return out, mean_img, z


def save_overlay(mean_img: np.ndarray, z_img: np.ndarray, candidates: pd.DataFrame, out_png: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(mean_img, cmap="gray")
    axes[0].set_title("sampled mean", fontsize=9)
    vmax = float(np.nanpercentile(np.abs(z_img), 99))
    vmax = max(vmax, 3.0)
    axes[1].imshow(z_img, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("background-subtracted z", fontsize=9)
    for ax in axes:
        ax.axis("off")
        for _, row in candidates.head(20).iterrows():
            x0 = float(row["bbox_x0_ds"])
            y0 = float(row["bbox_y0_ds"])
            w = float(row["bbox_x1_ds"] - row["bbox_x0_ds"])
            h = float(row["bbox_y1_ds"] - row["bbox_y0_ds"])
            color = "lime" if row["polarity"] == "bright" else "cyan"
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=0.8))
            ax.text(float(row["x_ds"]), float(row["y_ds"]), str(int(row["candidate_rank"])), color=color, fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_targets(particles: pd.DataFrame, events: pd.DataFrame, neighbor_count: int) -> pd.DataFrame:
    event_cycles = sorted(events["cycleNo"].dropna().astype(float).unique())
    chunks = []
    for cyc in event_cycles:
        event_rows = particles[particles["cycleNo"] == cyc]
        if event_rows.empty:
            continue
        stem = event_rows.iloc[0]["source_stem"]
        local_idx = float(event_rows.iloc[0]["local_cycle_index"])
        same = particles[particles["source_stem"] == stem].copy()
        same["local_dist"] = (pd.to_numeric(same["local_cycle_index"], errors="coerce") - local_idx).abs()
        chunks.append(same.sort_values(["local_dist", "cycleNo"]).head(1 + neighbor_count))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["cycleNo", "source_stem", "local_cycle_index"])


def match_adjacent(candidates: pd.DataFrame, max_distance_ds: float) -> pd.DataFrame:
    rows = []
    if candidates.empty:
        return pd.DataFrame()
    for stem, src in candidates.groupby("source_stem"):
        segment_keys = src[["cycleNo", "local_cycle_index", "is_event_cycle"]].drop_duplicates().sort_values("local_cycle_index")
        for _, left in segment_keys.iterrows():
            local_idx = int(left["local_cycle_index"])
            right_rows = segment_keys[segment_keys["local_cycle_index"] == local_idx + 1]
            if right_rows.empty:
                continue
            right = right_rows.iloc[0]
            a = src[(src["cycleNo"] == left["cycleNo"]) & (src["local_cycle_index"] == local_idx)]
            b = src[(src["cycleNo"] == right["cycleNo"]) & (src["local_cycle_index"] == int(right["local_cycle_index"]))]
            for _, ca in a.iterrows():
                dx = b["x_ds"].to_numpy(dtype=float) - float(ca["x_ds"])
                dy = b["y_ds"].to_numpy(dtype=float) - float(ca["y_ds"])
                dist = np.sqrt(dx * dx + dy * dy)
                if dist.size == 0 or float(np.nanmin(dist)) > max_distance_ds:
                    continue
                cb = b.iloc[int(np.nanargmin(dist))]
                rows.append({
                    "source_stem": stem,
                    "cycle_a": float(left["cycleNo"]),
                    "cycle_b": float(right["cycleNo"]),
                    "local_a": local_idx,
                    "local_b": int(right["local_cycle_index"]),
                    "event_a": bool(left["is_event_cycle"]),
                    "event_b": bool(right["is_event_cycle"]),
                    "candidate_rank_a": int(ca["candidate_rank"]),
                    "candidate_rank_b": int(cb["candidate_rank"]),
                    "distance_ds": float(np.nanmin(dist)),
                    "distance_full_approx_px": float(np.nanmin(dist) * ca["downsample"]),
                    "mean_residual_a": float(ca["mean_residual"]),
                    "mean_residual_b": float(cb["mean_residual"]),
                    "residual_delta_b_minus_a": float(cb["mean_residual"] - ca["mean_residual"]),
                    "area_ds_a": int(ca["area_ds_px"]),
                    "area_ds_b": int(cb["area_ds_px"]),
                    "area_delta_b_minus_a": int(cb["area_ds_px"] - ca["area_ds_px"]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--event-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/particle_event_targets")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_object_candidate_reconstruction")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--samples-per-segment", type=int, default=10)
    parser.add_argument("--neighbor-count", type=int, default=3)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--max-area-fraction", type=float, default=0.015)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--max-match-distance-ds", type=float, default=18.0)
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
        raise SystemExit("Missing particles, cycle-frames, or event target file")

    particles = pd.read_csv(particles_path, encoding="utf-8-sig")
    frames_meta = read_cycle_frames_csv(frames_path)
    events = pd.read_csv(events_path)
    particles["cycleNo"] = pd.to_numeric(particles["cycleNo"], errors="coerce")
    frames_meta["cycleNo"] = pd.to_numeric(frames_meta["cycleNo"], errors="coerce")
    frames_meta["n_frames_table"] = pd.to_numeric(frames_meta["n_frames"], errors="coerce")
    parsed = particles["addrs"].map(parse_addr).apply(pd.Series)
    particles = pd.concat([particles, parsed], axis=1).merge(frames_meta[["cycleNo", "n_frames_table"]], on="cycleNo", how="left")
    events["cycleNo"] = pd.to_numeric(events["cycleNo"], errors="coerce")

    targets = build_targets(particles, events, args.neighbor_count)
    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "overlays")
    os.makedirs(preview_dir, exist_ok=True)

    all_candidates = []
    segment_rows = []
    event_cycles = set(events["cycleNo"].dropna().astype(float).tolist())
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
            for _, row in source_rows.sort_values("local_cycle_index").iterrows():
                cycle = float(row["cycleNo"])
                local_idx = int(row["local_cycle_index"])
                start, end = segment_bounds(total_frames, local_idx, n_segments)
                idx = sample_indices(start, end, args.samples_per_segment)
                sampled = read_downsampled_movie(movie, idx, args.downsample)
                cand, mean_img, z_img = detect_candidates(
                    sampled,
                    args.downsample,
                    args.z_threshold,
                    args.min_area,
                    args.max_area_fraction,
                    args.max_candidates,
                )
                is_event = cycle in event_cycles
                overlay_path = os.path.join(preview_dir, f"cycle_{int(cycle)}_{stem}_local{local_idx}.png")
                save_overlay(mean_img, z_img, cand, overlay_path, f"cycle {cycle:g} | {stem} local {local_idx} | event={is_event}")
                segment_meta = {
                    "cycleNo": cycle,
                    "source_stem": stem,
                    "local_cycle_index": local_idx,
                    "is_event_cycle": bool(is_event),
                    "particle_event_count": int((events["cycleNo"] == cycle).sum()),
                    "n_segments_inferred": int(n_segments),
                    "h5_total_frames": total_frames,
                    "segment_start_frame": int(start),
                    "segment_end_frame": int(end),
                    "n_sampled_frames": int(len(idx)),
                    "sampled_first_frame": int(idx[0]),
                    "sampled_last_frame": int(idx[-1]),
                    "cycle_table_n_frames": float(row.get("n_frames_table", np.nan)),
                    "downsample": int(args.downsample),
                    "n_candidates": int(len(cand)),
                    "overlay_png": overlay_path,
                }
                segment_rows.append(segment_meta)
                if not cand.empty:
                    for key, val in segment_meta.items():
                        cand[key] = val
                    all_candidates.append(cand)

    candidates = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    segments = pd.DataFrame(segment_rows).sort_values(["source_stem", "local_cycle_index", "cycleNo"]) if segment_rows else pd.DataFrame()
    matches = match_adjacent(candidates, args.max_match_distance_ds)

    candidate_path = os.path.join(args.out_dir, "reconstructed_object_candidates.csv")
    segment_path = os.path.join(args.out_dir, "reconstructed_object_segments.csv")
    match_path = os.path.join(args.out_dir, "adjacent_candidate_matches.csv")
    candidates.to_csv(candidate_path, index=False)
    segments.to_csv(segment_path, index=False)
    matches.to_csv(match_path, index=False)

    event_candidates = candidates[candidates["is_event_cycle"] == True] if not candidates.empty else pd.DataFrame()
    neighbor_candidates = candidates[candidates["is_event_cycle"] == False] if not candidates.empty else pd.DataFrame()
    summary = {
        "particles_csv": particles_path,
        "cycle_frames_csv": frames_path,
        "event_dir": args.event_dir,
        "downsample": args.downsample,
        "samples_per_segment": args.samples_per_segment,
        "z_threshold": args.z_threshold,
        "n_segments_sampled": int(len(segments)),
        "n_candidates": int(len(candidates)),
        "n_event_candidates": int(len(event_candidates)),
        "n_neighbor_candidates": int(len(neighbor_candidates)),
        "n_adjacent_matches": int(len(matches)),
        "event_cycles_sampled": sorted(segments.loc[segments["is_event_cycle"] == True, "cycleNo"].astype(float).tolist()) if not segments.empty else [],
        "mean_candidates_per_event_segment": float(segments.loc[segments["is_event_cycle"] == True, "n_candidates"].mean()) if not segments.empty else np.nan,
        "mean_candidates_per_neighbor_segment": float(segments.loc[segments["is_event_cycle"] == False, "n_candidates"].mean()) if not segments.empty else np.nan,
        "guardrail": "Reconstructed candidates are approximate connected components from sampled downsampled full-frame movies, not legacy object-detector outputs.",
    }
    summary_path = os.path.join(args.out_dir, "reconstructed_object_candidate_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    readme_path = os.path.join(args.out_dir, "README.md")
    lines = [
        "# NMC Event Object Candidate Reconstruction",
        "",
        "Bounded reconstruction of candidate particle-like regions from sampled full HDF5 movie segments. This fills the current validation gap caused by missing chopped cycle HDF5 files and missing legacy object-detector outputs.",
        "",
        "## Summary",
        "",
        f"- Segments sampled: {summary['n_segments_sampled']}",
        f"- Candidate components: {summary['n_candidates']}",
        f"- Event candidate components: {summary['n_event_candidates']}",
        f"- Neighbor candidate components: {summary['n_neighbor_candidates']}",
        f"- Adjacent-segment candidate matches: {summary['n_adjacent_matches']}",
        f"- Mean candidates per event segment: {summary['mean_candidates_per_event_segment']:.3g}",
        f"- Mean candidates per neighbor segment: {summary['mean_candidates_per_neighbor_segment']:.3g}",
        "",
        "## Outputs",
        "",
        "- `reconstructed_object_segments.csv`: sampled segment metadata and overlay paths.",
        "- `reconstructed_object_candidates.csv`: ranked component candidates with approximate full-frame centroids.",
        "- `adjacent_candidate_matches.csv`: nearest candidate matches from local segment `i` to `i+1` within each source movie.",
        "- `overlays/`: visual previews with top component boxes on mean and background-subtracted images.",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
        "",
    ]
    if not segments.empty:
        lines.extend([
            "## Sampled Segments",
            "",
            "| cycleNo | event | source_stem | local_idx | candidates | overlay |",
            "|---:|---|---|---:|---:|---|",
        ])
        for _, row in segments.iterrows():
            lines.append("| {cycleNo:.1f} | {event} | `{stem}` | {local} | {n} | `{overlay}` |".format(
                cycleNo=float(row["cycleNo"]),
                event="yes" if bool(row["is_event_cycle"]) else "no",
                stem=row["source_stem"],
                local=int(row["local_cycle_index"]),
                n=int(row["n_candidates"]),
                overlay=os.path.relpath(row["overlay_png"], args.out_dir),
            ))
        lines.append("")
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))

    for path in [candidate_path, segment_path, match_path, summary_path, readme_path]:
        print(f"Saved: {path}")
    print(json.dumps(summary, indent=2, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
