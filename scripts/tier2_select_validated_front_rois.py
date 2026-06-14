#!/usr/bin/env python3
"""Select validated candidate ROIs for NMC event front tracking.

This combines two independent candidate sources:
1. `event_candidate_fronts`: front/high-fraction metrics from connected regions.
2. `event_object_candidate_reconstruction`: dense object-like candidates from
   background-subtracted full-frame segments.

The output is an algorithmic validation/ranking table for cycles 86 and 116. It
is still a QC selection layer, not a final manual annotation. Apparent transport
metrics are reported in downsampled pixels and approximate full pixels per second
where camera timing is available; no micron-scale diffusion coefficient is claimed.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def nearest_object(front_row: pd.Series, objects: pd.DataFrame) -> Dict[str, object]:
    subset = objects[(objects["cycleNo"] == front_row["cycleNo"]) & (objects["source_stem"] == front_row["source_stem"])]
    if subset.empty:
        return {}
    dx = pd.to_numeric(subset["x_ds"], errors="coerce") - finite_float(front_row["centroid_x_ds"])
    dy = pd.to_numeric(subset["y_ds"], errors="coerce") - finite_float(front_row["centroid_y_ds"])
    dist = np.sqrt(dx.to_numpy(dtype=float) ** 2 + dy.to_numpy(dtype=float) ** 2)
    if not np.isfinite(dist).any():
        return {}
    pos = int(np.nanargmin(dist))
    obj = subset.iloc[pos].to_dict()
    obj["object_match_distance_ds"] = float(dist[pos])
    return obj


def nearest_neighbor_object(front_row: pd.Series, objects: pd.DataFrame, min_local_gap: int = 1) -> Dict[str, object]:
    same = objects[(objects["source_stem"] == front_row["source_stem"]) & (objects["cycleNo"] != front_row["cycleNo"])].copy()
    if same.empty:
        return {}
    same["local_gap"] = (pd.to_numeric(same["local_cycle_index"], errors="coerce") - finite_float(front_row.get("local_cycle_index", np.nan))).abs()
    same = same[same["local_gap"] >= min_local_gap]
    if same.empty:
        return {}
    # Prefer nearest represented segment, then nearest centroid.
    min_gap = same["local_gap"].min()
    same = same[same["local_gap"] == min_gap].copy()
    dx = pd.to_numeric(same["x_ds"], errors="coerce") - finite_float(front_row["centroid_x_ds"])
    dy = pd.to_numeric(same["y_ds"], errors="coerce") - finite_float(front_row["centroid_y_ds"])
    dist = np.sqrt(dx.to_numpy(dtype=float) ** 2 + dy.to_numpy(dtype=float) ** 2)
    if not np.isfinite(dist).any():
        return {}
    pos = int(np.nanargmin(dist))
    obj = same.iloc[pos].to_dict()
    obj["neighbor_distance_ds"] = float(dist[pos])
    return obj


def time_span_for_trace(root: str, trace_csv: str, source_stem: str) -> Dict[str, float]:
    try:
        trace = pd.read_csv(trace_csv)
    except Exception:
        return {}
    if trace.empty or "frame_index" not in trace.columns:
        return {}
    idx = pd.to_numeric(trace["frame_index"], errors="coerce").dropna().astype(int).to_numpy()
    if idx.size < 2:
        return {}
    h5_path = os.path.join(root, "NMC_degradation_3_160623_Halfthedata", f"{source_stem}.hdf5")
    if not os.path.exists(h5_path):
        return {}
    with h5py.File(h5_path, "r") as f:
        if "camera_timing" not in f:
            return {}
        timing = np.asarray(f["camera_timing"])
    # Row 1 is relative seconds in the inspected files.
    if timing.ndim != 2 or timing.shape[0] < 2:
        return {}
    idx = idx[(idx >= 0) & (idx < timing.shape[1])]
    if idx.size < 2:
        return {}
    secs = timing[1, idx].astype(float)
    elapsed = float(np.nanmax(secs) - np.nanmin(secs))
    frame_elapsed = float(np.nanmax(idx) - np.nanmin(idx))
    sec_per_frame = elapsed / frame_elapsed if frame_elapsed > 0 and np.isfinite(elapsed) else np.nan
    return {"elapsed_s": elapsed, "frame_elapsed": frame_elapsed, "sec_per_frame": sec_per_frame}


def validation_score(row: Dict[str, object]) -> float:
    score = 0.0
    score += 1.2 * finite_float(row.get("front_quality_score"), 0.0)
    score += 0.6 * min(finite_float(row.get("object_mean_abs_z"), 0.0) / 20.0, 2.0)
    score += 0.5 * min(finite_float(row.get("object_area_ds_px"), 0.0) / 120.0, 1.5)
    obj_rank = finite_float(row.get("object_candidate_rank"), np.nan)
    if np.isfinite(obj_rank):
        score += max(0.0, 1.0 - obj_rank / 80.0)
    dist = finite_float(row.get("object_match_distance_ds"), np.nan)
    if np.isfinite(dist):
        score += max(0.0, 1.0 - dist / 8.0)
    ndist = finite_float(row.get("neighbor_distance_ds"), np.nan)
    if np.isfinite(ndist):
        score += max(0.0, 1.0 - ndist / 12.0)
    # Penalize weak front fits and very small/negative radius2 trends only mildly:
    r2 = finite_float(row.get("front_radius2_slope_r2"), np.nan)
    if np.isfinite(r2):
        score += 0.6 * max(0.0, min(r2, 1.0))
    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--fronts", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_candidate_fronts/candidate_front_metrics.csv")
    parser.add_argument("--objects", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_object_candidate_reconstruction/reconstructed_object_candidates.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/validated_front_rois")
    parser.add_argument("--cycles", type=float, nargs="+", default=[86.0, 116.0])
    parser.add_argument("--top-per-cycle", type=int, default=5)
    args = parser.parse_args()

    fronts = pd.read_csv(args.fronts)
    objects = pd.read_csv(args.objects)
    for df in [fronts, objects]:
        df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    selected_rows: List[Dict[str, object]] = []
    for cyc in args.cycles:
        cyc_fronts = fronts[(fronts["cycleNo"] == cyc) & (fronts["is_event_cycle"].astype(bool))].copy()
        if cyc_fronts.empty:
            continue
        # Add local cycle index from object table by source/cycle.
        local_map = objects[objects["cycleNo"] == cyc].drop_duplicates(["cycleNo", "source_stem"])
        for _, frow in cyc_fronts.iterrows():
            fdict = frow.to_dict()
            match_meta = local_map[local_map["source_stem"] == frow["source_stem"]]
            if not match_meta.empty:
                fdict["local_cycle_index"] = int(match_meta.iloc[0]["local_cycle_index"])
            obj = nearest_object(pd.Series(fdict), objects)
            neigh = nearest_neighbor_object(pd.Series(fdict), objects)
            time_meta = time_span_for_trace(args.root, str(frow.get("trace_csv", "")), str(frow["source_stem"]))
            out = {
                "cycleNo": float(cyc),
                "source_stem": frow["source_stem"],
                "front_candidate_rank": int(frow["candidate_rank"]),
                "front_centroid_x_ds": finite_float(frow["centroid_x_ds"]),
                "front_centroid_y_ds": finite_float(frow["centroid_y_ds"]),
                "front_quality_score": finite_float(frow["front_quality_score"]),
                "front_radius_slope_ds_px_per_frame": finite_float(frow["front_radius_slope_ds_px_per_frame"]),
                "front_radius_slope_r2": finite_float(frow["front_radius_slope_r2"]),
                "front_radius_monotonic_fraction": finite_float(frow["front_radius_monotonic_fraction"]),
                "apparent_diffusion_proxy_ds_px2_per_frame": finite_float(frow["apparent_diffusion_proxy_ds_px2_per_frame"]),
                "front_radius2_slope_r2": finite_float(frow["front_radius2_slope_r2"]),
                "roi_mean_delta_last_minus_first": finite_float(frow["roi_mean_delta_last_minus_first"]),
                "high_fraction_first": finite_float(frow["high_fraction_first"]),
                "high_fraction_last": finite_float(frow["high_fraction_last"]),
                "front_preview_png": frow.get("preview_png", ""),
                "front_trace_csv": frow.get("trace_csv", ""),
                "downsample": int(frow.get("downsample", 4)),
            }
            if obj:
                out.update({
                    "object_candidate_rank": int(obj.get("candidate_rank", -1)),
                    "object_candidate_id": int(obj.get("candidate_id", -1)),
                    "object_x_ds": finite_float(obj.get("x_ds")),
                    "object_y_ds": finite_float(obj.get("y_ds")),
                    "object_area_ds_px": finite_float(obj.get("area_ds_px")),
                    "object_mean_residual": finite_float(obj.get("mean_residual")),
                    "object_mean_abs_z": finite_float(obj.get("mean_abs_z")),
                    "object_match_distance_ds": finite_float(obj.get("object_match_distance_ds")),
                    "object_overlay_png": obj.get("overlay_png", ""),
                })
            if neigh:
                out.update({
                    "neighbor_cycleNo": finite_float(neigh.get("cycleNo")),
                    "neighbor_local_gap": finite_float(neigh.get("local_gap")),
                    "neighbor_candidate_rank": int(neigh.get("candidate_rank", -1)),
                    "neighbor_distance_ds": finite_float(neigh.get("neighbor_distance_ds")),
                    "neighbor_mean_residual": finite_float(neigh.get("mean_residual")),
                    "neighbor_area_ds_px": finite_float(neigh.get("area_ds_px")),
                    "neighbor_residual_delta": finite_float(neigh.get("mean_residual")) - finite_float(obj.get("mean_residual")) if obj else np.nan,
                })
            out.update(time_meta)
            down = int(out.get("downsample", 4))
            out["apparent_diffusion_proxy_full_px2_per_frame"] = out["apparent_diffusion_proxy_ds_px2_per_frame"] * down * down
            spf = finite_float(out.get("sec_per_frame"), np.nan)
            out["apparent_diffusion_proxy_full_px2_per_s"] = out["apparent_diffusion_proxy_full_px2_per_frame"] / spf if np.isfinite(spf) and spf > 0 else np.nan
            out["validation_score"] = validation_score(out)
            if out.get("object_match_distance_ds", 999) <= 3 and out["front_quality_score"] >= 1.0:
                out["validation_tier"] = "candidate_roi_supported"
            elif out["front_quality_score"] >= 1.0:
                out["validation_tier"] = "front_only_candidate"
            else:
                out["validation_tier"] = "weak_candidate"
            selected_rows.append(out)

    out_df = pd.DataFrame(selected_rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["cycleNo", "validation_score"], ascending=[True, False])
        out_df["selected_for_next_tracking"] = False
        kept = []
        for cyc, g in out_df.groupby("cycleNo", sort=True):
            keep_idx = g.head(args.top_per_cycle).index.tolist()
            kept.extend(keep_idx)
        out_df.loc[kept, "selected_for_next_tracking"] = True
    os.makedirs(args.out_dir, exist_ok=True)
    all_path = os.path.join(args.out_dir, "validated_front_roi_candidates.csv")
    sel_path = os.path.join(args.out_dir, "selected_front_rois_for_tracking.csv")
    out_df.to_csv(all_path, index=False)
    out_df[out_df["selected_for_next_tracking"]].to_csv(sel_path, index=False)
    summary = {
        "fronts": args.fronts,
        "objects": args.objects,
        "cycles": args.cycles,
        "n_candidates_scored": int(len(out_df)),
        "n_selected_for_next_tracking": int(out_df["selected_for_next_tracking"].sum()) if not out_df.empty else 0,
        "top_selected": out_df[out_df["selected_for_next_tracking"]].head(20).to_dict(orient="records") if not out_df.empty else [],
        "guardrail": "Selection is algorithmic. Apparent full-pixel transport proxies are not micron-scale diffusion coefficients without spatial calibration.",
    }
    with open(os.path.join(args.out_dir, "validated_front_rois_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Validated Front ROI Candidates\n\n")
        f.write("Algorithmic selection of candidate ROIs for cycles 86 and 116 by combining front metrics, reconstructed object candidates, nearest-neighbor support, and camera timing.\n\n")
        f.write("Use `selected_front_rois_for_tracking.csv` as the next input for high-resolution/manual-QC tracking.\n")
    print(f"[done] scored {len(out_df)} candidates; selected {summary['n_selected_for_next_tracking']} -> {args.out_dir}")


if __name__ == "__main__":
    main()
