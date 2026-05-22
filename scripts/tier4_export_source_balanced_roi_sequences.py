#!/usr/bin/env python3
"""Export particle-region ROI sequences for source-balanced expansion candidates.

The source-balanced manifest expands beyond the earlier event/control and
transfer-ranked cohorts, but it only stores reconstructed particle coordinates.
This exporter turns those automatic coordinates into fixed padded crop tensors
from the full HDF5 movies. The fixed crop approximates the particle region from
prior reconstruction history and keeps drift-correction blur or small motion
inside the model input region.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")[:64]


def sample_indices(start: int, end: int, n_samples: int) -> np.ndarray:
    n = min(n_samples, max(1, end - start))
    return np.unique(np.linspace(start, end - 1, n, dtype=int))


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


def resize_crop(crop: np.ndarray, output_size: int) -> np.ndarray:
    if crop.shape == (output_size, output_size):
        return crop.astype(np.float32)
    return ndi.zoom(crop.astype(np.float32), (output_size / crop.shape[0], output_size / crop.shape[1]), order=1)


def robust_normalize(frames: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(frames, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(frames, dtype=np.float32)
    return np.clip((frames - lo) / (hi - lo), 0, 1).astype(np.float32)


def save_preview(frames: np.ndarray, out_png: Path, title: str) -> None:
    norm = robust_normalize(frames)
    idx = [0, len(norm) // 2, len(norm) - 1]
    imgs = [norm[i] for i in idx]
    diff = imgs[-1] - imgs[0]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, img, label in zip(axes[:3], imgs, ["first", "middle", "last"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    vmax = max(0.05, float(np.nanpercentile(np.abs(diff), 99)))
    axes[3].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[3].set_title("last - first", fontsize=9)
    axes[3].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def export_one(root: Path, row: pd.Series, out_dir: Path, crop_size_full: int, output_size: int, samples_per_roi: int, save_previews: bool) -> Dict[str, object]:
    source = str(row["source_stem"])
    h5_path = root / "NMC_degradation_3_160623_Halfthedata" / f"{source}.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    start = int(row["segment_start_frame"])
    end = int(row["segment_end_frame"])
    cx = finite_float(row["object_x_full_approx"])
    cy = finite_float(row["object_y_full_approx"])
    cycle = int(round(finite_float(row["cycleNo"])))
    rank = int(round(finite_float(row.get("expansion_cycle_rank", 0), 0)))
    obj = int(round(finite_float(row.get("object_candidate_rank", 0), 0)))
    roi_id = f"source_balanced_cycle{cycle}_rank{rank}_obj{obj}_{safe_stem(source)}"
    frame_indices = sample_indices(start, end, samples_per_roi)

    with h5py.File(h5_path, "r") as f:
        movie = f["movie"]
        height, width = int(movie.shape[1]), int(movie.shape[2])
        x0, y0, x1, y1 = crop_bounds(cx, cy, crop_size_full, width, height)
        crops = [resize_crop(movie[int(idx), y0:y1, x0:x1], output_size) for idx in frame_indices]
        frames = np.stack(crops, axis=0).astype(np.float32)
        avg_intensity = np.asarray(f["average_intensity"][0, frame_indices], dtype=float) if "average_intensity" in f else np.full(len(frame_indices), np.nan)
        stage = np.asarray(f["stage_position"][:, frame_indices], dtype=float) if "stage_position" in f else np.full((3, len(frame_indices)), np.nan)

    norm = robust_normalize(frames)
    roi_mean = np.nanmean(frames, axis=(1, 2))
    roi_norm_mean = np.nanmean(norm, axis=(1, 2))
    npz_path = out_dir / f"{roi_id}.npz"
    np.savez_compressed(
        npz_path,
        frames=frames.astype(np.float32),
        frames_norm=norm.astype(np.float32),
        frame_indices=frame_indices.astype(np.int64),
        roi_mean=roi_mean.astype(np.float32),
        roi_norm_mean=roi_norm_mean.astype(np.float32),
        average_intensity=avg_intensity.astype(np.float32),
        stage_position=stage.astype(np.float32),
    )
    preview_path = out_dir / "previews" / f"{roi_id}.png"
    if save_previews:
        save_preview(frames, preview_path, f"{roi_id} | {source}")
    drift = float(np.sqrt((np.nanmax(stage[0]) - np.nanmin(stage[0])) ** 2 + (np.nanmax(stage[1]) - np.nanmin(stage[1])) ** 2)) if np.isfinite(stage).any() else np.nan
    out = {
        "roi_id": roi_id,
        "cycleNo": float(row["cycleNo"]),
        "source_stem": source,
        "local_cycle_index": int(row.get("local_cycle_index", -1)),
        "expansion_cycle_rank": rank,
        "object_candidate_rank": obj,
        "selection_reason": row.get("selection_reason", ""),
        "event_relative_bin": row.get("event_relative_bin", ""),
        "cohort_role": row.get("cohort_role", ""),
        "cycles_to_next_event": finite_float(row.get("cycles_to_next_event")),
        "cycles_since_prev_event": finite_float(row.get("cycles_since_prev_event")),
        "future_event_within_32cycles": finite_float(row.get("future_event_within_32cycles")),
        "past_event_within_16cycles": finite_float(row.get("past_event_within_16cycles")),
        "already_in_existing_video_cohort": bool(row.get("already_in_existing_video_cohort", False)),
        "validation_score": finite_float(row.get("validation_score")),
        "validation_label": row.get("validation_label", ""),
        "object_x_full_approx": cx,
        "object_y_full_approx": cy,
        "object_area_ds_px": finite_float(row.get("object_area_ds_px")),
        "object_mean_residual": finite_float(row.get("object_mean_residual")),
        "object_mean_abs_z": finite_float(row.get("object_mean_abs_z")),
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "crop_size_full": int(crop_size_full),
        "output_size": int(output_size),
        "n_frames": int(len(frame_indices)),
        "first_frame_index": int(frame_indices[0]),
        "last_frame_index": int(frame_indices[-1]),
        "roi_mean_first": float(roi_mean[0]),
        "roi_mean_last": float(roi_mean[-1]),
        "roi_mean_delta_last_minus_first": float(roi_mean[-1] - roi_mean[0]),
        "roi_norm_mean_first": float(roi_norm_mean[0]),
        "roi_norm_mean_last": float(roi_norm_mean[-1]),
        "roi_norm_mean_delta_last_minus_first": float(roi_norm_mean[-1] - roi_norm_mean[0]),
        "stage_drift_xy_sampled": drift,
        "future_any_drop_within_8cycles": finite_float(row.get("future_any_drop_within_8cycles")),
        "future_any_drop_within_16cycles": finite_float(row.get("future_any_drop_within_16cycles")),
        "any_abrupt_drop": finite_float(row.get("any_abrupt_drop")),
        "npz_path": str(npz_path),
        "preview_png": str(preview_path) if save_previews else "",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--roi-table", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_roi_expansion_manifest/source_balanced_roi_table.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_roi_sequences")
    parser.add_argument("--crop-size-full", type=int, default=192)
    parser.add_argument("--output-size", type=int, default=96)
    parser.add_argument("--samples-per-roi", type=int, default=96)
    parser.add_argument("--max-rois", type=int, default=0)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    rois = pd.read_csv(args.roi_table)
    rois = rois.sort_values(["source_stem", "cycleNo", "object_candidate_rank"]).reset_index(drop=True)
    if args.max_rois and args.max_rois > 0:
        rois = rois.head(args.max_rois).copy()

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for _, row in rois.iterrows():
        try:
            rec = export_one(root, row, out_dir, args.crop_size_full, args.output_size, args.samples_per_roi, not args.no_previews)
            rows.append(rec)
            print(f"exported {rec['roi_id']} frames={rec['n_frames']}")
        except Exception as exc:
            failures.append({"cycleNo": row.get("cycleNo"), "source_stem": row.get("source_stem"), "object_candidate_rank": row.get("object_candidate_rank"), "error": str(exc)})
            print(f"failed cycle={row.get('cycleNo')} source={row.get('source_stem')}: {exc}")

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "selected_roi_sequence_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    failures_df = pd.DataFrame(failures)
    failures_path = out_dir / "source_balanced_sequence_failures.csv"
    failures_df.to_csv(failures_path, index=False)
    cycle_summary = {}
    if not manifest.empty:
        for cyc, grp in manifest.groupby("cycleNo"):
            cycle_summary[str(float(cyc))] = {
                "n_roi_sequences": int(len(grp)),
                "mean_roi_delta": float(grp["roi_mean_delta_last_minus_first"].mean()),
                "mean_norm_roi_delta": float(grp["roi_norm_mean_delta_last_minus_first"].mean()),
                "mean_stage_drift_xy_sampled": float(grp["stage_drift_xy_sampled"].mean()),
                "future8_positive": int(pd.to_numeric(grp["future_any_drop_within_8cycles"], errors="coerce").fillna(0).max() > 0),
                "future16_positive": int(pd.to_numeric(grp["future_any_drop_within_16cycles"], errors="coerce").fillna(0).max() > 0),
            }
    summary = {
        "roi_table": args.roi_table,
        "n_input_roi_rows": int(len(rois)),
        "n_roi_sequences": int(len(manifest)),
        "n_failed": int(len(failures)),
        "n_cycles": int(manifest["cycleNo"].nunique()) if not manifest.empty else 0,
        "n_sources": int(manifest["source_stem"].nunique()) if not manifest.empty else 0,
        "crop_size_full": int(args.crop_size_full),
        "output_size": int(args.output_size),
        "samples_per_roi": int(args.samples_per_roi),
        "future8_positive_sequences": int(pd.to_numeric(manifest.get("future_any_drop_within_8cycles", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not manifest.empty else 0,
        "future16_positive_sequences": int(pd.to_numeric(manifest.get("future_any_drop_within_16cycles", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not manifest.empty else 0,
        "cycle_summary": cycle_summary,
        "outputs": {
            "manifest": str(manifest_path),
            "failures": str(failures_path),
            "summary": str(out_dir / "selected_roi_sequence_summary.json"),
        },
        "guardrail": "Source-balanced sequences are fixed padded particle-region crops around automatic reconstructed candidates. They broaden source coverage for model/physics tests, but are not manual particle annotations or validated front labels.",
    }
    (out_dir / "selected_roi_sequence_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "README.md").write_text(
        "# Source-Balanced ROI Sequences\n\n"
        "Particle-region-only crop tensors for source-balanced expansion candidates. Use `frames_norm` from each NPZ as model inputs for next-frame and rollout experiments.\n\n"
        "Crops are fixed and padded around automatically reconstructed particle coordinates to tolerate drift-correction blur and small coordinate shifts.\n\n"
        f"- ROI sequences: {summary['n_roi_sequences']}\n"
        f"- Cycles: {summary['n_cycles']}\n"
        f"- Sources: {summary['n_sources']}\n"
        f"- Failures: {summary['n_failed']}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
