#!/usr/bin/env python3
"""Select matched non-event control ROI candidates for NMC particle modeling.

The event ROI table is useful but too small and cycle-specific for robust model
training. This script selects control particle-like ROIs from reconstructed
non-event segments in the same source movies as the synchronized event cycles.
It writes a table compatible with `tier2_export_selected_roi_sequences.py`.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def select_controls(
    recon: pd.DataFrame,
    event_rois: pd.DataFrame,
    controls_per_event_cycle: int,
    max_controls_per_control_cycle: int = 0,
) -> pd.DataFrame:
    recon = recon.copy()
    recon["cycleNo"] = pd.to_numeric(recon["cycleNo"], errors="coerce")
    recon["candidate_rank"] = pd.to_numeric(recon["candidate_rank"], errors="coerce")
    event_rois = event_rois.copy()
    event_rois["cycleNo"] = pd.to_numeric(event_rois["cycleNo"], errors="coerce")

    rows: List[Dict[str, object]] = []
    for event_cycle, ev in event_rois.groupby("cycleNo"):
        source = str(ev["source_stem"].iloc[0])
        event_local = int(ev["local_cycle_index"].iloc[0])
        controls = recon[
            (recon["source_stem"] == source)
            & (recon["is_event_cycle"].astype(str).str.lower().isin(["false", "0"]))
        ].copy()
        if controls.empty:
            continue
        controls["local_distance_from_event"] = (
            pd.to_numeric(controls["local_cycle_index"], errors="coerce") - event_local
        ).abs()
        # Favor nearby non-event segments, then strong/reliable reconstructed candidates.
        controls = controls.sort_values(
            ["local_distance_from_event", "cycleNo", "candidate_rank"],
            ascending=[True, True, True],
        )
        if max_controls_per_control_cycle > 0:
            selected = (
                controls.groupby("cycleNo", group_keys=False)
                .head(max_controls_per_control_cycle)
                .head(controls_per_event_cycle)
            )
        else:
            selected = controls.head(controls_per_event_cycle)
        for i, row in selected.iterrows():
            out = {
                "cycleNo": float(row["cycleNo"]),
                "source_stem": row["source_stem"],
                "local_cycle_index": int(row["local_cycle_index"]),
                "front_candidate_rank": int(row["candidate_rank"]),
                "object_candidate_rank": int(row["candidate_rank"]),
                "validation_score": float(1.0 / max(1.0, finite_float(row["candidate_rank"], 1.0))),
                "validation_label": "matched_control_roi_candidate",
                "object_x_full_approx": finite_float(row["x_full_approx"]),
                "object_y_full_approx": finite_float(row["y_full_approx"]),
                "object_x_ds": finite_float(row["x_ds"]),
                "object_y_ds": finite_float(row["y_ds"]),
                "object_area_ds_px": finite_float(row["area_ds_px"]),
                "object_mean_residual": finite_float(row["mean_residual"]),
                "object_mean_abs_z": finite_float(row["mean_abs_z"]),
                "segment_start_frame": int(row["segment_start_frame"]),
                "segment_end_frame": int(row["segment_end_frame"]),
                "downsample": int(row.get("downsample", 4)),
                "control_for_event_cycle": float(event_cycle),
                "local_distance_from_event": finite_float(row["local_distance_from_event"]),
                "source_reconstructed_overlay_png": row.get("overlay_png", ""),
            }
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--event-roi-table", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_roi_validation/selected_event_rois.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_roi_selection")
    parser.add_argument("--controls-per-event-cycle", type=int, default=8)
    parser.add_argument(
        "--max-controls-per-control-cycle",
        type=int,
        default=0,
        help="If positive, cap selections from any single non-event cycle before filling the event-cycle quota.",
    )
    args = parser.parse_args()

    recon_path = os.path.join(args.derived_dir, "event_object_candidate_reconstruction", "reconstructed_object_candidates.csv")
    if not os.path.exists(recon_path):
        raise SystemExit(f"missing reconstructed candidate table: {recon_path}")
    recon = pd.read_csv(recon_path)
    event_rois = pd.read_csv(args.event_roi_table)
    controls = select_controls(
        recon,
        event_rois,
        args.controls_per_event_cycle,
        args.max_controls_per_control_cycle,
    )
    if controls.empty:
        raise SystemExit("no controls selected")

    os.makedirs(args.out_dir, exist_ok=True)
    table_path = os.path.join(args.out_dir, "selected_control_rois.csv")
    controls.to_csv(table_path, index=False)
    cycle_counts = controls.groupby(["control_for_event_cycle", "cycleNo"]).size().reset_index(name="n_roi")
    cycle_counts_path = os.path.join(args.out_dir, "control_roi_cycle_counts.csv")
    cycle_counts.to_csv(cycle_counts_path, index=False)
    summary = {
        "recon_path": recon_path,
        "event_roi_table": args.event_roi_table,
        "n_control_rois": int(len(controls)),
        "controls_per_event_cycle": int(args.controls_per_event_cycle),
        "max_controls_per_control_cycle": int(args.max_controls_per_control_cycle),
        "control_cycle_counts": cycle_counts.to_dict(orient="records"),
        "guardrail": "Control ROIs are automatic reconstructed candidates from nearby non-event segments, not manual annotations.",
    }
    summary_path = os.path.join(args.out_dir, "control_roi_selection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Control ROI Selection\n\n")
        f.write("Matched non-event reconstructed ROI candidates selected from source movies adjacent to synchronized event cycles.\n\n")
        f.write("Use `selected_control_rois.csv` with `tier2_export_selected_roi_sequences.py` to export particle-region control tensors.\n")
    for path in [table_path, cycle_counts_path, summary_path]:
        print(f"Saved: {path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
