#!/usr/bin/env python3
"""Export particle-region-only NMC sequences for selected event ROIs.

The selected ROI table gives approximate full-frame particle coordinates. This
script turns those coordinates into fixed-size crop tensors from the full HDF5
movies. Crops are centered on the selected object and padded beyond the detected
box so drift-corrected blur or small coordinate shifts stay inside the model
input region.
"""

import argparse
import json
import os
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
    x0 = max(0, x0)
    y0 = max(0, y0)
    return x0, y0, x1, y1


def resize_crop(crop: np.ndarray, output_size: int) -> np.ndarray:
    if crop.shape == (output_size, output_size):
        return crop.astype(np.float32)
    zoom_y = output_size / float(crop.shape[0])
    zoom_x = output_size / float(crop.shape[1])
    return ndi.zoom(crop.astype(np.float32), (zoom_y, zoom_x), order=1)


def robust_normalize(frames: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(frames, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(frames, dtype=np.float32)
    return np.clip((frames - lo) / (hi - lo), 0, 1).astype(np.float32)


def save_preview(frames: np.ndarray, out_png: str, title: str) -> None:
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
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def export_one_roi(
    root: str,
    row: pd.Series,
    out_dir: str,
    crop_size_full: int,
    output_size: int,
    samples_per_roi: int,
) -> Dict[str, object]:
    source_stem = str(row["source_stem"])
    h5_path = os.path.join(root, "NMC_degradation_3_160623_Halfthedata", f"{source_stem}.hdf5")
    start = int(row["segment_start_frame"])
    end = int(row["segment_end_frame"])
    cx = finite_float(row["object_x_full_approx"])
    cy = finite_float(row["object_y_full_approx"])
    roi_id = f"cycle{int(row['cycleNo'])}_rank{int(row['front_candidate_rank'])}_obj{int(row['object_candidate_rank'])}"
    frame_indices = sample_indices(start, end, samples_per_roi)

    with h5py.File(h5_path, "r") as f:
        movie = f["movie"]
        height, width = int(movie.shape[1]), int(movie.shape[2])
        x0, y0, x1, y1 = crop_bounds(cx, cy, crop_size_full, width, height)
        crops = []
        for frame_idx in frame_indices:
            crop = movie[int(frame_idx), y0:y1, x0:x1]
            crops.append(resize_crop(crop, output_size))
        frames = np.stack(crops, axis=0).astype(np.float32)
        avg_intensity = np.asarray(f["average_intensity"][0, frame_indices], dtype=float) if "average_intensity" in f else np.full(len(frame_indices), np.nan)
        stage = np.asarray(f["stage_position"][:, frame_indices], dtype=float) if "stage_position" in f else np.full((3, len(frame_indices)), np.nan)

    norm = robust_normalize(frames)
    roi_mean = np.nanmean(frames, axis=(1, 2))
    roi_norm_mean = np.nanmean(norm, axis=(1, 2))
    npz_path = os.path.join(out_dir, f"{roi_id}.npz")
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
    preview_path = os.path.join(out_dir, "previews", f"{roi_id}.png")
    save_preview(frames, preview_path, f"{roi_id} | {source_stem}")
    return {
        "roi_id": roi_id,
        "cycleNo": float(row["cycleNo"]),
        "source_stem": source_stem,
        "local_cycle_index": int(row["local_cycle_index"]),
        "front_candidate_rank": int(row["front_candidate_rank"]),
        "object_candidate_rank": int(row["object_candidate_rank"]),
        "validation_score": finite_float(row["validation_score"]),
        "validation_label": row.get("validation_label", ""),
        "object_x_full_approx": cx,
        "object_y_full_approx": cy,
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
        "stage_drift_xy_sampled": float(np.sqrt((np.nanmax(stage[0]) - np.nanmin(stage[0])) ** 2 + (np.nanmax(stage[1]) - np.nanmin(stage[1])) ** 2)) if np.isfinite(stage).any() else np.nan,
        "cohort_role": row.get("cohort_role", ""),
        "selection_subrole": row.get("selection_subrole", ""),
        "event_reference_cycle": finite_float(row.get("event_reference_cycle", np.nan)),
        "future_any_drop_within_8cycles": finite_float(row.get("future_any_drop_within_8cycles", np.nan)),
        "future_any_drop_within_16cycles": finite_float(row.get("future_any_drop_within_16cycles", np.nan)),
        "any_abrupt_drop": finite_float(row.get("any_abrupt_drop", np.nan)),
        "transferred_masked_residual_signature": finite_float(row.get("transferred_masked_residual_signature", np.nan)),
        "npz_path": npz_path,
        "preview_png": preview_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--roi-table", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_roi_validation/selected_event_rois.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/selected_roi_sequences")
    parser.add_argument("--crop-size-full", type=int, default=192)
    parser.add_argument("--output-size", type=int, default=96)
    parser.add_argument("--samples-per-roi", type=int, default=96)
    parser.add_argument("--only-selected-label", action="store_true")
    args = parser.parse_args()

    rois = pd.read_csv(args.roi_table)
    if args.only_selected_label and "validation_label" in rois.columns:
        rois = rois[rois["validation_label"] == "selected_roi_candidate"].copy()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "previews"), exist_ok=True)

    rows: List[Dict[str, object]] = []
    for _, row in rois.sort_values(["cycleNo", "validation_score"], ascending=[True, False]).iterrows():
        rows.append(export_one_roi(args.root, row, args.out_dir, args.crop_size_full, args.output_size, args.samples_per_roi))
        print(f"exported {rows[-1]['roi_id']} frames={rows[-1]['n_frames']}")

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "selected_roi_sequence_manifest.csv")
    summary_df.to_csv(summary_path, index=False)
    cycle_summary = {}
    for cyc, grp in summary_df.groupby("cycleNo"):
        cycle_summary[str(float(cyc))] = {
            "n_roi_sequences": int(len(grp)),
            "mean_roi_delta": float(grp["roi_mean_delta_last_minus_first"].mean()),
            "mean_norm_roi_delta": float(grp["roi_norm_mean_delta_last_minus_first"].mean()),
            "mean_stage_drift_xy_sampled": float(grp["stage_drift_xy_sampled"].mean()),
        }
    summary = {
        "roi_table": args.roi_table,
        "n_roi_sequences": int(len(summary_df)),
        "crop_size_full": args.crop_size_full,
        "output_size": args.output_size,
        "samples_per_roi": args.samples_per_roi,
        "cycle_summary": cycle_summary,
        "guardrail": "Sequences are fixed padded crops around automatically selected ROIs; they are particle-region model inputs, not manual annotations.",
    }
    summary_json = os.path.join(args.out_dir, "selected_roi_sequence_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Selected ROI Sequences\n\n")
        f.write("Particle-region-only crop tensors for synchronized NMC event cycles. Use `frames_norm` from each NPZ as model inputs for next-frame or rollout experiments.\n\n")
        f.write("Crops are fixed and padded around selected object coordinates to tolerate drift-correction blur and small coordinate shifts.\n")
    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_json}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
