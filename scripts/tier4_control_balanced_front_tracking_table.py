#!/usr/bin/env python3
"""Build a tracker-ready table for control-balanced front QC candidates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_roi_id(roi_id: str) -> tuple[int, int, int]:
    parts = str(roi_id).split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected roi_id: {roi_id}")
    cycle = int(parts[0].replace("cycle", ""))
    front_rank = int(parts[1].replace("rank", ""))
    object_rank = int(parts[2].replace("obj", ""))
    return cycle, front_rank, object_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/scratch/<account>/<username>/Alek_Jiho/derived/control_balanced_front_qc_package/control_balanced_front_qc_manifest.csv")
    parser.add_argument("--front-candidates", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_candidate_fronts/candidate_front_metrics.csv")
    parser.add_argument("--object-candidates", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_object_candidate_reconstruction/reconstructed_object_candidates.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/control_balanced_front_tracking_table")
    parser.add_argument("--max-rois", type=int, default=40)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    fronts = pd.read_csv(args.front_candidates)
    objects = pd.read_csv(args.object_candidates)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    decoded = manifest["roi_id"].map(split_roi_id)
    manifest = manifest.copy()
    manifest["cycleNo_int"] = [x[0] for x in decoded]
    manifest["front_candidate_rank"] = [x[1] for x in decoded]
    manifest["object_candidate_rank"] = [x[2] for x in decoded]

    front_keep = fronts.rename(columns={"candidate_rank": "front_candidate_rank"}).copy()
    front_cols = [
        "cycleNo", "source_stem", "front_candidate_rank", "centroid_x_ds", "centroid_y_ds",
        "front_quality_score", "front_radius_slope_ds_px_per_frame", "front_radius_slope_r2",
        "front_radius_monotonic_fraction", "apparent_diffusion_proxy_ds_px2_per_frame",
        "front_radius2_slope_r2", "roi_mean_delta_last_minus_first", "high_fraction_first",
        "high_fraction_last", "preview_png", "trace_csv", "downsample",
    ]
    front_keep = front_keep[[c for c in front_cols if c in front_keep.columns]].copy()
    front_keep = front_keep.rename(columns={
        "centroid_x_ds": "front_centroid_x_ds",
        "centroid_y_ds": "front_centroid_y_ds",
        "preview_png": "front_preview_png",
        "trace_csv": "front_trace_csv",
    })
    front_keep["cycleNo_int"] = pd.to_numeric(front_keep["cycleNo"], errors="coerce").astype("Int64")

    object_keep = objects.rename(columns={"candidate_rank": "object_candidate_rank"}).copy()
    object_cols = [
        "cycleNo", "source_stem", "object_candidate_rank", "candidate_id", "x_ds", "y_ds",
        "area_ds_px", "mean_residual", "mean_abs_z", "overlay_png",
    ]
    object_keep = object_keep[[c for c in object_cols if c in object_keep.columns]].copy()
    object_keep = object_keep.rename(columns={
        "candidate_id": "object_candidate_id",
        "x_ds": "object_x_ds",
        "y_ds": "object_y_ds",
        "area_ds_px": "object_area_ds_px",
        "mean_residual": "object_mean_residual",
        "mean_abs_z": "object_mean_abs_z",
        "overlay_png": "object_overlay_png",
    })
    object_keep["cycleNo_int"] = pd.to_numeric(object_keep["cycleNo"], errors="coerce").astype("Int64")

    joined = manifest.merge(
        front_keep.drop(columns=["cycleNo"], errors="ignore"),
        on=["cycleNo_int", "source_stem", "front_candidate_rank"] if "source_stem" in manifest.columns else ["cycleNo_int", "front_candidate_rank"],
        how="left",
    )
    joined = joined.merge(
        object_keep.drop(columns=["cycleNo"], errors="ignore"),
        on=["cycleNo_int", "source_stem", "object_candidate_rank"] if "source_stem" in joined.columns else ["cycleNo_int", "object_candidate_rank"],
        how="left",
    )
    joined["cycleNo"] = joined["cycleNo_int"].astype(float)
    if "downsample" not in joined:
        joined["downsample"] = 4
    joined["object_match_distance_ds"] = np.sqrt(
        (pd.to_numeric(joined["object_x_ds"], errors="coerce") - pd.to_numeric(joined["front_centroid_x_ds"], errors="coerce")) ** 2
        + (pd.to_numeric(joined["object_y_ds"], errors="coerce") - pd.to_numeric(joined["front_centroid_y_ds"], errors="coerce")) ** 2
    )
    joined["validation_score"] = pd.to_numeric(joined.get("control_balance_priority_score", np.nan), errors="coerce").fillna(
        pd.to_numeric(joined.get("front_quality_score", np.nan), errors="coerce")
    )
    joined["validation_tier"] = joined.get("selection_group", "")
    joined["selected_for_next_tracking"] = True

    missing = joined[joined[["front_trace_csv", "object_x_ds", "object_y_ds", "source_stem"]].isna().any(axis=1)].copy()
    tracker = joined.drop(missing.index).copy()
    tracker = tracker.sort_values(
        ["cohort_role", "event_reference_cycle", "validation_score"],
        ascending=[True, True, False],
        na_position="last",
    ).head(args.max_rois)

    tracker_cols = [
        "cycleNo", "source_stem", "front_candidate_rank", "front_centroid_x_ds", "front_centroid_y_ds",
        "front_quality_score", "front_radius_slope_ds_px_per_frame", "front_radius_slope_r2",
        "front_radius_monotonic_fraction", "apparent_diffusion_proxy_ds_px2_per_frame",
        "front_radius2_slope_r2", "roi_mean_delta_last_minus_first", "high_fraction_first",
        "high_fraction_last", "front_preview_png", "front_trace_csv", "downsample",
        "object_candidate_rank", "object_candidate_id", "object_x_ds", "object_y_ds",
        "object_area_ds_px", "object_mean_residual", "object_mean_abs_z",
        "object_match_distance_ds", "object_overlay_png", "validation_score", "validation_tier",
        "selected_for_next_tracking", "roi_id", "cohort_role", "event_reference_cycle",
        "manual_qc_status", "auto_review_flags",
    ]
    tracker_path = out / "control_balanced_front_rois_for_tracking.csv"
    missing_path = out / "control_balanced_front_tracking_missing.csv"
    summary_path = out / "control_balanced_front_tracking_table_summary.json"
    tracker[[c for c in tracker_cols if c in tracker.columns]].to_csv(tracker_path, index=False)
    missing.to_csv(missing_path, index=False)

    summary = {
        "n_manifest_rows": int(len(manifest)),
        "n_tracker_rows": int(len(tracker)),
        "n_missing_rows": int(len(missing)),
        "tracker_role_counts": tracker.get("cohort_role", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "missing_roi_ids": missing.get("roi_id", pd.Series(dtype=object)).head(20).tolist(),
        "outputs": {
            "tracker_table": str(tracker_path),
            "missing_rows": str(missing_path),
        },
        "guardrail": "This table only joins automatic front/object candidates into the high-resolution tracker schema; it does not create manual QC labels.",
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    with (out / "README.md").open("w") as f:
        f.write("# Control-Balanced Front Tracking Table\n\n")
        f.write("Tracker-ready ROI table generated by joining the control-balanced QC manifest to front and object candidate coordinates.\n\n")
        f.write(f"- Manifest rows: {summary['n_manifest_rows']}\n")
        f.write(f"- Tracker rows: {summary['n_tracker_rows']}\n")
        f.write(f"- Missing rows: {summary['n_missing_rows']}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
