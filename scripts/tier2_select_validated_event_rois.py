#!/usr/bin/env python3
"""Select auditable NMC event ROI candidates by fusing reconstruction and front proxies.

This script is a validation bridge between two bounded analyses:

1. high-recall reconstructed object candidates from sampled full HDF5 segments
2. candidate front/intensity proxies for particle-like ROIs

It focuses on the synchronized event cycles by default (86 and 116), links each
front candidate to the nearest reconstructed object candidate, tests whether a
nearby object candidate is found in the next sampled segment from the same
source movie, and writes a compact selected-ROI table plus focused overlays.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def distance(a_x: float, a_y: float, b_x: np.ndarray, b_y: np.ndarray) -> np.ndarray:
    return np.sqrt((b_x.astype(float) - a_x) ** 2 + (b_y.astype(float) - a_y) ** 2)


def nearest_row(df: pd.DataFrame, x: float, y: float) -> Tuple[Optional[pd.Series], float]:
    if df.empty:
        return None, np.nan
    d = distance(x, y, df["x_ds"].to_numpy(dtype=float), df["y_ds"].to_numpy(dtype=float))
    idx = int(np.nanargmin(d))
    return df.iloc[idx], float(d[idx])


def sample_indices(start: int, end: int, n_samples: int) -> np.ndarray:
    n = min(n_samples, max(1, end - start))
    return np.unique(np.linspace(start, end - 1, n, dtype=int))


def read_mean_image(root: str, source_stem: str, start: int, end: int, downsample: int, n_samples: int) -> np.ndarray:
    h5_path = os.path.join(root, "NMC_degradation_3_160623_Halfthedata", f"{source_stem}.hdf5")
    idx = sample_indices(start, end, n_samples)
    frames = []
    with h5py.File(h5_path, "r") as f:
        movie = f["movie"]
        for frame_idx in idx:
            frames.append(movie[int(frame_idx), ::downsample, ::downsample].astype(np.float32))
    return np.nanmean(np.stack(frames, axis=0), axis=0)


def normalize_series(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").astype(float)
    lo = vals.quantile(0.05)
    hi = vals.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(vals)), index=s.index)
    return ((vals - lo) / (hi - lo)).clip(0, 1).fillna(0)


def build_roi_candidates(
    front: pd.DataFrame,
    recon: pd.DataFrame,
    target_cycles: List[float],
    max_front_to_object_distance: float,
    max_neighbor_distance: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    front = front.copy()
    front["cycleNo"] = pd.to_numeric(front["cycleNo"], errors="coerce")
    recon = recon.copy()
    recon["cycleNo"] = pd.to_numeric(recon["cycleNo"], errors="coerce")
    for cyc in target_cycles:
        fcyc = front[np.isclose(front["cycleNo"], cyc)].copy()
        if fcyc.empty:
            continue
        for _, frow in fcyc.iterrows():
            stem = str(frow["source_stem"])
            same_recon = recon[(recon["source_stem"] == stem) & (np.isclose(recon["cycleNo"], cyc))]
            obj, obj_dist = nearest_row(
                same_recon,
                finite_float(frow["centroid_x_ds"]),
                finite_float(frow["centroid_y_ds"]),
            )
            if obj is None or not np.isfinite(obj_dist):
                continue
            local_idx = int(obj["local_cycle_index"])
            later_segments = (
                recon[(recon["source_stem"] == stem) & (pd.to_numeric(recon["local_cycle_index"], errors="coerce") > local_idx)]
                .sort_values(["local_cycle_index", "candidate_rank"])
            )
            neighbor = pd.DataFrame()
            next_local = np.nan
            if not later_segments.empty:
                next_local = int(later_segments["local_cycle_index"].min())
                neighbor = later_segments[later_segments["local_cycle_index"] == next_local]
            nobj, n_dist = nearest_row(neighbor, finite_float(obj["x_ds"]), finite_float(obj["y_ds"]))
            area_ratio = np.nan
            residual_delta = np.nan
            neighbor_cycle = np.nan
            neighbor_rank = np.nan
            if nobj is not None and np.isfinite(n_dist):
                area_a = finite_float(obj["area_ds_px"])
                area_b = finite_float(nobj["area_ds_px"])
                area_ratio = area_b / area_a if area_a > 0 else np.nan
                residual_delta = finite_float(nobj["mean_residual"]) - finite_float(obj["mean_residual"])
                neighbor_cycle = finite_float(nobj["cycleNo"])
                neighbor_rank = int(nobj["candidate_rank"])
            bbox_min_dist = min(
                finite_float(obj["bbox_x0_ds"]),
                finite_float(obj["bbox_y0_ds"]),
                480.0 - finite_float(obj["bbox_x1_ds"]),
                300.0 - finite_float(obj["bbox_y1_ds"]),
            )
            rows.append({
                "cycleNo": float(cyc),
                "source_stem": stem,
                "local_cycle_index": local_idx,
                "front_candidate_rank": int(frow["candidate_rank"]),
                "front_quality_score": finite_float(frow.get("front_quality_score")),
                "front_candidate_score": finite_float(frow.get("candidate_score")),
                "front_radius_slope_r2": finite_float(frow.get("front_radius_slope_r2")),
                "front_radius_monotonic_fraction": finite_float(frow.get("front_radius_monotonic_fraction")),
                "apparent_diffusion_proxy_ds_px2_per_frame": finite_float(frow.get("apparent_diffusion_proxy_ds_px2_per_frame")),
                "roi_mean_delta_last_minus_first": finite_float(frow.get("roi_mean_delta_last_minus_first")),
                "front_centroid_x_ds": finite_float(frow["centroid_x_ds"]),
                "front_centroid_y_ds": finite_float(frow["centroid_y_ds"]),
                "object_candidate_rank": int(obj["candidate_rank"]),
                "object_candidate_id": int(obj["candidate_id"]),
                "object_x_ds": finite_float(obj["x_ds"]),
                "object_y_ds": finite_float(obj["y_ds"]),
                "object_x_full_approx": finite_float(obj["x_full_approx"]),
                "object_y_full_approx": finite_float(obj["y_full_approx"]),
                "object_area_ds_px": finite_float(obj["area_ds_px"]),
                "object_mean_residual": finite_float(obj["mean_residual"]),
                "object_mean_abs_z": finite_float(obj["mean_abs_z"]),
                "front_to_object_distance_ds": obj_dist,
                "neighbor_cycle": neighbor_cycle,
                "neighbor_local_cycle_index": next_local,
                "neighbor_candidate_rank": neighbor_rank,
                "neighbor_distance_ds": n_dist,
                "neighbor_distance_full_approx_px": n_dist * finite_float(obj.get("downsample", 4.0), 4.0) if np.isfinite(n_dist) else np.nan,
                "neighbor_area_ratio": area_ratio,
                "neighbor_residual_delta": residual_delta,
                "bbox_x0_ds": int(obj["bbox_x0_ds"]),
                "bbox_y0_ds": int(obj["bbox_y0_ds"]),
                "bbox_x1_ds": int(obj["bbox_x1_ds"]),
                "bbox_y1_ds": int(obj["bbox_y1_ds"]),
                "edge_margin_ds": bbox_min_dist,
                "object_overlay_png": obj.get("overlay_png", ""),
                "front_preview_png": frow.get("preview_png", ""),
                "front_trace_csv": frow.get("trace_csv", ""),
                "passes_front_object_distance": bool(obj_dist <= max_front_to_object_distance),
                "passes_neighbor_distance": bool(np.isfinite(n_dist) and n_dist <= max_neighbor_distance),
                "passes_area_ratio": bool(np.isfinite(area_ratio) and 0.45 <= area_ratio <= 2.25),
                "passes_edge_margin": bool(np.isfinite(bbox_min_dist) and bbox_min_dist >= 4.0),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["front_quality_norm"] = normalize_series(out["front_quality_score"])
    out["object_z_norm"] = normalize_series(out["object_mean_abs_z"])
    out["distance_score"] = (1.0 - out["front_to_object_distance_ds"].clip(0, max_front_to_object_distance) / max_front_to_object_distance).fillna(0)
    out["neighbor_score"] = (1.0 - out["neighbor_distance_ds"].clip(0, max_neighbor_distance) / max_neighbor_distance).fillna(0)
    out["validation_score"] = (
        2.0 * out["front_quality_norm"]
        + 1.0 * out["object_z_norm"]
        + 1.0 * out["distance_score"]
        + 1.0 * out["neighbor_score"]
        + 0.75 * out["passes_area_ratio"].astype(float)
        + 0.50 * out["passes_edge_margin"].astype(float)
    )
    out["validation_label"] = np.where(
        out["passes_front_object_distance"] & out["passes_neighbor_distance"] & out["passes_area_ratio"] & (out["validation_score"] >= 3.0),
        "selected_roi_candidate",
        "lower_priority_or_needs_manual_review",
    )
    return out.sort_values(["cycleNo", "validation_score"], ascending=[True, False]).reset_index(drop=True)


def save_selected_overlay(root: str, rows: pd.DataFrame, out_png: str, downsample: int, n_samples: int) -> None:
    if rows.empty:
        return
    first = rows.iloc[0]
    mean_img = read_mean_image(
        root,
        str(first["source_stem"]),
        int(first["segment_start_frame"]),
        int(first["segment_end_frame"]),
        downsample,
        n_samples,
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.imshow(mean_img, cmap="gray")
    for idx, (_, row) in enumerate(rows.head(8).iterrows(), start=1):
        color = "lime" if row["validation_label"] == "selected_roi_candidate" else "yellow"
        x0 = float(row["bbox_x0_ds"])
        y0 = float(row["bbox_y0_ds"])
        w = float(row["bbox_x1_ds"] - row["bbox_x0_ds"])
        h = float(row["bbox_y1_ds"] - row["bbox_y0_ds"])
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=1.2))
        ax.text(x0, max(0, y0 - 2), f"{idx}", color=color, fontsize=8)
    ax.set_title(f"cycle {first['cycleNo']:.0f} selected ROI candidates", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_roi_validation")
    parser.add_argument("--target-cycles", default="86,116")
    parser.add_argument("--top-per-cycle", type=int, default=6)
    parser.add_argument("--max-front-to-object-distance", type=float, default=12.0)
    parser.add_argument("--max-neighbor-distance", type=float, default=18.0)
    parser.add_argument("--overlay-samples", type=int, default=12)
    args = parser.parse_args()

    front_path = os.path.join(args.derived_dir, "event_candidate_fronts", "candidate_front_metrics.csv")
    recon_path = os.path.join(args.derived_dir, "event_object_candidate_reconstruction", "reconstructed_object_candidates.csv")
    if not os.path.exists(front_path) or not os.path.exists(recon_path):
        raise SystemExit("Missing candidate front or reconstructed object table")
    front = pd.read_csv(front_path)
    recon = pd.read_csv(recon_path)
    target_cycles = [float(x) for x in args.target_cycles.split(",") if x.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    overlay_dir = os.path.join(args.out_dir, "selected_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    all_rows = build_roi_candidates(
        front,
        recon,
        target_cycles,
        args.max_front_to_object_distance,
        args.max_neighbor_distance,
    )
    if all_rows.empty:
        raise SystemExit("No ROI candidates could be linked")

    # Add segment bounds needed for focused overlays.
    segment_cols = [
        "source_stem",
        "cycleNo",
        "local_cycle_index",
        "segment_start_frame",
        "segment_end_frame",
        "downsample",
    ]
    seg = recon[segment_cols].drop_duplicates()
    seg["cycleNo"] = pd.to_numeric(seg["cycleNo"], errors="coerce")
    all_rows = all_rows.merge(seg, on=["source_stem", "cycleNo", "local_cycle_index"], how="left")
    selected = (
        all_rows.sort_values(["cycleNo", "validation_score"], ascending=[True, False])
        .groupby("cycleNo", as_index=False, group_keys=False)
        .head(args.top_per_cycle)
        .reset_index(drop=True)
    )
    all_path = os.path.join(args.out_dir, "event_roi_validation_candidates.csv")
    selected_path = os.path.join(args.out_dir, "selected_event_rois.csv")
    all_rows.to_csv(all_path, index=False)
    selected.to_csv(selected_path, index=False)

    overlay_paths = []
    for cyc, rows in selected.groupby("cycleNo"):
        out_png = os.path.join(overlay_dir, f"cycle_{int(cyc)}_selected_rois.png")
        ds = int(rows["downsample"].dropna().iloc[0]) if rows["downsample"].notna().any() else 4
        save_selected_overlay(args.root, rows, out_png, ds, args.overlay_samples)
        overlay_paths.append(out_png)

    cycle_summary = {}
    for cyc, rows in selected.groupby("cycleNo"):
        cycle_summary[str(float(cyc))] = {
            "n_selected_rows": int(len(rows)),
            "n_selected_roi_candidates": int((rows["validation_label"] == "selected_roi_candidate").sum()),
            "top_validation_score": float(rows["validation_score"].max()),
            "top_object_full_xy": [
                float(rows.iloc[0]["object_x_full_approx"]),
                float(rows.iloc[0]["object_y_full_approx"]),
            ],
            "top_front_quality_score": float(rows.iloc[0]["front_quality_score"]),
            "top_neighbor_distance_ds": float(rows.iloc[0]["neighbor_distance_ds"]),
        }
    summary = {
        "front_path": front_path,
        "recon_path": recon_path,
        "target_cycles": target_cycles,
        "n_linked_candidates": int(len(all_rows)),
        "n_selected_rows": int(len(selected)),
        "selection_rule": "top validation_score per target cycle after linking front candidates to reconstructed objects and nearest next sampled segment",
        "cycle_summary": cycle_summary,
        "guardrail": "Selected ROIs are automatically validated candidate regions, not manually confirmed legacy detector outputs.",
        "overlay_paths": overlay_paths,
    }
    summary_path = os.path.join(args.out_dir, "event_roi_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    readme_path = os.path.join(args.out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Event ROI Validation\n\n")
        f.write("Fuses candidate front metrics with reconstructed object candidates to select auditable particle-region candidates for synchronized NMC event cycles.\n\n")
        f.write("## Outputs\n\n")
        f.write("- `event_roi_validation_candidates.csv`: all linked front/object candidates for target cycles.\n")
        f.write("- `selected_event_rois.csv`: top scored ROI candidates per cycle.\n")
        f.write("- `selected_overlays/`: focused overlays showing selected candidate boxes.\n\n")
        f.write("## Guardrail\n\n")
        f.write(summary["guardrail"] + "\n")

    for path in [all_path, selected_path, summary_path, readme_path]:
        print(f"Saved: {path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
