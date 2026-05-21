#!/usr/bin/env python3
"""Build a multi-cycle NMC ROI cohort from event and neighbor candidates.

This expands beyond the synchronized cycles by selecting candidate ROIs from
single-particle event cycles and nearby non-event controls. The output table is
compatible with `tier2_export_selected_roi_sequences.py`.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DEFAULT_CYCLES = [60.0, 86.0, 116.0, 156.0]


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def nearest_object(front_row: pd.Series, objects: pd.DataFrame) -> Dict[str, object]:
    subset = objects[
        (objects["cycleNo"] == front_row["cycleNo"])
        & (objects["source_stem"] == front_row["source_stem"])
    ].copy()
    if subset.empty:
        return {}
    dx = pd.to_numeric(subset["x_ds"], errors="coerce") - finite_float(front_row["centroid_x_ds"])
    dy = pd.to_numeric(subset["y_ds"], errors="coerce") - finite_float(front_row["centroid_y_ds"])
    dist = np.sqrt(dx.to_numpy(dtype=float) ** 2 + dy.to_numpy(dtype=float) ** 2)
    if not np.isfinite(dist).any():
        return {}
    pos = int(np.nanargmin(dist))
    out = subset.iloc[pos].to_dict()
    out["front_to_object_distance_ds"] = float(dist[pos])
    return out


def candidate_score(front_row: pd.Series, obj: Dict[str, object]) -> float:
    score = 0.0
    score += 1.4 * finite_float(front_row.get("front_quality_score"), 0.0)
    score += 0.8 * min(finite_float(obj.get("mean_abs_z"), 0.0) / 25.0, 2.0)
    score += 0.4 * min(finite_float(obj.get("area_ds_px"), 0.0) / 150.0, 1.5)
    dist = finite_float(obj.get("front_to_object_distance_ds"), np.nan)
    if np.isfinite(dist):
        score += max(0.0, 1.0 - dist / 10.0)
    front_rank = finite_float(front_row.get("candidate_rank"), np.nan)
    if np.isfinite(front_rank):
        score += max(0.0, 1.0 - front_rank / 12.0)
    obj_rank = finite_float(obj.get("candidate_rank"), np.nan)
    if np.isfinite(obj_rank):
        score += max(0.0, 1.0 - obj_rank / 120.0)
    return float(score)


def event_cycle_metadata(evidence: pd.DataFrame) -> Dict[float, Dict[str, object]]:
    out: Dict[float, Dict[str, object]] = {}
    if evidence.empty or "cycleNo" not in evidence.columns:
        return out
    for _, row in evidence.iterrows():
        cyc = finite_float(row.get("cycleNo"))
        if np.isfinite(cyc):
            out[float(cyc)] = row.to_dict()
    return out


def choose_control_cycles(
    fronts: pd.DataFrame,
    event_cycle: float,
    source_stem: str,
    local_cycle_index: float,
    max_control_cycles: int,
) -> List[float]:
    controls = fronts[
        (fronts["source_stem"] == source_stem)
        & (~fronts["is_event_cycle"].astype(bool))
    ].copy()
    if controls.empty:
        return []
    controls["local_distance"] = (
        pd.to_numeric(controls["local_cycle_index"], errors="coerce") - local_cycle_index
    ).abs()
    controls = controls[controls["cycleNo"] != event_cycle]
    ranked = (
        controls.sort_values(["local_distance", "cycleNo"])
        .drop_duplicates("cycleNo")
        .head(max_control_cycles)
    )
    return [float(x) for x in ranked["cycleNo"].tolist()]


def build_rows_for_cycle(
    fronts: pd.DataFrame,
    objects: pd.DataFrame,
    cycle: float,
    role: str,
    event_reference_cycle: float,
    top_n: int,
    evidence_meta: Dict[str, object],
) -> List[Dict[str, object]]:
    cyc_fronts = fronts[fronts["cycleNo"] == cycle].copy()
    if cyc_fronts.empty:
        return []
    rows: List[Dict[str, object]] = []
    for _, frow in cyc_fronts.iterrows():
        obj = nearest_object(frow, objects)
        if not obj:
            continue
        score = candidate_score(frow, obj)
        label = f"{role}_multicycle_roi_candidate"
        out = {
            "cycleNo": float(cycle),
            "source_stem": frow["source_stem"],
            "local_cycle_index": int(finite_float(frow.get("local_cycle_index"), finite_float(obj.get("local_cycle_index"), -1))),
            "front_candidate_rank": int(finite_float(frow.get("candidate_rank"), -1)),
            "object_candidate_rank": int(finite_float(obj.get("candidate_rank"), -1)),
            "validation_score": score,
            "validation_label": label,
            "cohort_role": role,
            "event_reference_cycle": float(event_reference_cycle),
            "object_x_full_approx": finite_float(obj.get("x_full_approx")),
            "object_y_full_approx": finite_float(obj.get("y_full_approx")),
            "object_x_ds": finite_float(obj.get("x_ds")),
            "object_y_ds": finite_float(obj.get("y_ds")),
            "object_area_ds_px": finite_float(obj.get("area_ds_px")),
            "object_mean_residual": finite_float(obj.get("mean_residual")),
            "object_mean_abs_z": finite_float(obj.get("mean_abs_z")),
            "front_to_object_distance_ds": finite_float(obj.get("front_to_object_distance_ds")),
            "front_quality_score": finite_float(frow.get("front_quality_score")),
            "front_radius_slope_r2": finite_float(frow.get("front_radius_slope_r2")),
            "front_radius_monotonic_fraction": finite_float(frow.get("front_radius_monotonic_fraction")),
            "apparent_diffusion_proxy_ds_px2_per_frame": finite_float(frow.get("apparent_diffusion_proxy_ds_px2_per_frame")),
            "roi_mean_delta_last_minus_first": finite_float(frow.get("roi_mean_delta_last_minus_first")),
            "high_fraction_first": finite_float(frow.get("high_fraction_first")),
            "high_fraction_last": finite_float(frow.get("high_fraction_last")),
            "segment_start_frame": int(finite_float(obj.get("segment_start_frame"), -1)),
            "segment_end_frame": int(finite_float(obj.get("segment_end_frame"), -1)),
            "downsample": int(finite_float(obj.get("downsample"), finite_float(frow.get("downsample"), 4))),
            "source_reconstructed_overlay_png": obj.get("overlay_png", ""),
            "front_preview_png": frow.get("preview_png", ""),
            "front_trace_csv": frow.get("trace_csv", ""),
            "event_particles": evidence_meta.get("particles", ""),
            "n_event_particles": finite_float(evidence_meta.get("n_event_particles"), 0.0),
            "mean_drop_frac": finite_float(evidence_meta.get("mean_drop_frac")),
            "global_frame_percentile": finite_float(evidence_meta.get("global_frame_percentile")),
            "evidence_score": finite_float(evidence_meta.get("evidence_score")),
            "degradation_mode_hypothesis": evidence_meta.get("degradation_mode_hypothesis", ""),
        }
        rows.append(out)
    rows = sorted(rows, key=lambda r: r["validation_score"], reverse=True)
    return rows[:top_n]


def parse_cycles(values: Iterable[str]) -> List[float]:
    if not values:
        return DEFAULT_CYCLES
    return [float(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_cohort")
    parser.add_argument("--event-cycles", nargs="*", default=[str(x) for x in DEFAULT_CYCLES])
    parser.add_argument("--top-events-per-cycle", type=int, default=6)
    parser.add_argument("--top-controls-per-cycle", type=int, default=4)
    parser.add_argument("--max-control-cycles", type=int, default=2)
    args = parser.parse_args()

    fronts_path = os.path.join(args.derived_dir, "event_candidate_fronts", "candidate_front_metrics.csv")
    objects_path = os.path.join(args.derived_dir, "event_object_candidate_reconstruction", "reconstructed_object_candidates.csv")
    evidence_path = os.path.join(args.derived_dir, "integrated_event_evidence", "integrated_event_evidence.csv")
    fronts = pd.read_csv(fronts_path)
    objects = pd.read_csv(objects_path)
    evidence = pd.read_csv(evidence_path) if os.path.exists(evidence_path) else pd.DataFrame()
    for df in [fronts, objects]:
        df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
        if "local_cycle_index" in df.columns:
            df["local_cycle_index"] = pd.to_numeric(df["local_cycle_index"], errors="coerce")
    if "local_cycle_index" not in fronts.columns:
        local_meta = objects[["cycleNo", "source_stem", "local_cycle_index"]].drop_duplicates(["cycleNo", "source_stem"])
        fronts = fronts.merge(local_meta, how="left", on=["cycleNo", "source_stem"])
    meta_by_cycle = event_cycle_metadata(evidence)
    event_cycles = parse_cycles(args.event_cycles)

    rows: List[Dict[str, object]] = []
    selected_control_cycles: Dict[str, List[float]] = {}
    for event_cycle in event_cycles:
        meta = meta_by_cycle.get(float(event_cycle), {})
        event_rows = build_rows_for_cycle(
            fronts,
            objects,
            event_cycle,
            "event",
            event_cycle,
            args.top_events_per_cycle,
            meta,
        )
        rows.extend(event_rows)
        if not event_rows:
            continue
        source = str(event_rows[0]["source_stem"])
        event_local = finite_float(event_rows[0]["local_cycle_index"])
        controls = choose_control_cycles(fronts, event_cycle, source, event_local, args.max_control_cycles)
        selected_control_cycles[str(float(event_cycle))] = controls
        for control_cycle in controls:
            rows.extend(
                build_rows_for_cycle(
                    fronts,
                    objects,
                    control_cycle,
                    "control",
                    event_cycle,
                    args.top_controls_per_cycle,
                    meta,
                )
            )

    cohort = pd.DataFrame(rows)
    if cohort.empty:
        raise SystemExit("no cohort rows selected")
    cohort = cohort.sort_values(
        ["event_reference_cycle", "cohort_role", "cycleNo", "validation_score"],
        ascending=[True, False, True, False],
    )

    os.makedirs(args.out_dir, exist_ok=True)
    table_path = os.path.join(args.out_dir, "multi_cycle_roi_table.csv")
    cohort.to_csv(table_path, index=False)
    counts = (
        cohort.groupby(["event_reference_cycle", "cohort_role", "cycleNo"])
        .size()
        .reset_index(name="n_roi")
    )
    counts_path = os.path.join(args.out_dir, "multi_cycle_roi_counts.csv")
    counts.to_csv(counts_path, index=False)
    summary = {
        "fronts_path": fronts_path,
        "objects_path": objects_path,
        "evidence_path": evidence_path,
        "event_cycles": event_cycles,
        "selected_control_cycles": selected_control_cycles,
        "n_rows": int(len(cohort)),
        "n_event_rows": int((cohort["cohort_role"] == "event").sum()),
        "n_control_rows": int((cohort["cohort_role"] == "control").sum()),
        "counts": counts.to_dict(orient="records"),
        "top_rows": cohort.head(20).to_dict(orient="records"),
        "guardrail": "ROIs are automatically selected candidate crops for modeling/QC; they are not manual particle annotations or validated diffusion measurements.",
        "outputs": {
            "multi_cycle_roi_table": table_path,
            "multi_cycle_roi_counts": counts_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "multi_cycle_roi_cohort_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Multi-Cycle ROI Cohort\n\n")
        f.write("Automatic event/control ROI cohort for synchronized and single-particle NMC degradation candidate cycles.\n\n")
        f.write("Use `multi_cycle_roi_table.csv` with `tier2_export_selected_roi_sequences.py` to export crop tensors.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
