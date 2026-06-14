#!/usr/bin/env python3
"""Reconstruct ROI candidates for cycles ranked by transferred residual warning.

The state-transfer audit ranks cycles that deserve direct video inspection, but
some high-ranked cycles do not yet have particle-region crop tensors. This
script samples the corresponding HDF5 cycle segments, rebuilds approximate
particle-like connected components, and writes a ROI table compatible with the
existing particle-region sequence exporter.

The outputs are automatic candidate crops for follow-up video modeling/QC, not
manual annotations.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd

from tier1_reconstruct_event_object_candidates import (
    detect_candidates,
    infer_n_segments,
    parse_addr,
    sample_indices,
    segment_bounds,
    read_downsampled_movie,
    save_overlay,
)


def clean_value(x):
    if isinstance(x, dict):
        return {str(k): clean_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_value(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    return x


def read_cycle_frames_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "cycleNo" in df.columns:
        return df
    raw = pd.read_csv(path, encoding="utf-8-sig", header=None)
    raw = raw.iloc[:, :2].copy()
    raw.columns = ["cycleNo", "n_frames"]
    return raw


def resolve_existing_path(candidates: List[str]) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def candidate_validation_score(cand: pd.Series, transfer_score: float, transfer_rank: int) -> float:
    rank_score = float(cand.get("rank_score", 0.0))
    abs_z = float(cand.get("mean_abs_z", 0.0))
    area = float(cand.get("area_ds_px", 0.0))
    transfer = float(transfer_score) if np.isfinite(transfer_score) else 0.0
    return float(np.log1p(max(rank_score, 0.0)) + 0.30 * abs_z + 0.02 * area + 0.15 * transfer - 0.02 * transfer_rank)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--ranked-cycles", default="/scratch/<account>/<username>/Alek_Jiho/derived/masked_residual_state_transfer_warning/masked_residual_state_transfer_ranked_cycles.csv")
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/transfer_ranked_roi_reconstruction")
    parser.add_argument("--top-cycles", type=int, default=12)
    parser.add_argument("--top-candidates-per-cycle", type=int, default=4)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--samples-per-segment", type=int, default=16)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--max-area-fraction", type=float, default=0.015)
    parser.add_argument("--max-candidates", type=int, default=80)
    args = parser.parse_args()

    out = Path(args.out_dir)
    overlay_dir = out / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

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
    if not particles_path:
        raise SystemExit("Missing exampleParticles.csv")

    ranked = pd.read_csv(args.ranked_cycles)
    ranked["cycleNo"] = pd.to_numeric(ranked["cycleNo"], errors="coerce")
    ranked = ranked.dropna(subset=["cycleNo"]).sort_values("transferred_masked_residual_signature", ascending=False).head(args.top_cycles).copy()
    ranked["transfer_rank"] = np.arange(1, len(ranked) + 1)
    parsed = ranked["addrs"].map(parse_addr).apply(pd.Series)
    ranked = pd.concat([ranked, parsed], axis=1)

    particles = pd.read_csv(particles_path, encoding="utf-8-sig")
    particles["cycleNo"] = pd.to_numeric(particles["cycleNo"], errors="coerce")
    particles = pd.concat([particles, particles["addrs"].map(parse_addr).apply(pd.Series)], axis=1)
    if frames_path:
        frames_meta = read_cycle_frames_csv(frames_path)
        frames_meta["cycleNo"] = pd.to_numeric(frames_meta["cycleNo"], errors="coerce")
        frames_meta["n_frames_table"] = pd.to_numeric(frames_meta["n_frames"], errors="coerce")
        ranked = ranked.merge(frames_meta[["cycleNo", "n_frames_table"]], how="left", on="cycleNo")
    else:
        ranked["n_frames_table"] = np.nan

    segment_rows: List[Dict[str, object]] = []
    candidate_tables: List[pd.DataFrame] = []
    roi_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []

    for _, cyc in ranked.iterrows():
        cycle = float(cyc["cycleNo"])
        stem = str(cyc.get("source_stem", ""))
        local_idx = int(cyc.get("local_cycle_index")) if np.isfinite(float(cyc.get("local_cycle_index", np.nan))) else -1
        h5_path = os.path.join(args.root, "NMC_degradation_3_160623_Halfthedata", f"{stem}.hdf5")
        if not stem or local_idx < 0 or not os.path.exists(h5_path):
            missing_rows.append({"cycleNo": cycle, "source_stem": stem, "local_cycle_index": local_idx, "reason": "missing_h5_or_parse"})
            continue
        n_segments = infer_n_segments(stem, particles[particles["source_stem"] == stem])
        with h5py.File(h5_path, "r") as f:
            movie = f["movie"]
            total_frames = int(movie.shape[0])
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
        overlay_path = overlay_dir / f"cycle_{int(cycle)}_{stem}_local{local_idx}.png"
        save_overlay(mean_img, z_img, cand, str(overlay_path), f"transfer rank {int(cyc['transfer_rank'])} | cycle {cycle:g} | {stem} local {local_idx}")
        transfer_score = float(cyc.get("transferred_masked_residual_signature", np.nan))
        future8 = int(cyc.get("future_any_drop_within_8cycles", 0)) if np.isfinite(float(cyc.get("future_any_drop_within_8cycles", 0))) else 0
        future16 = int(cyc.get("future_any_drop_within_16cycles", 0)) if np.isfinite(float(cyc.get("future_any_drop_within_16cycles", 0))) else 0
        segment_meta = {
            "cycleNo": cycle,
            "transfer_rank": int(cyc["transfer_rank"]),
            "transferred_masked_residual_signature": transfer_score,
            "future_any_drop_within_8cycles": future8,
            "future_any_drop_within_16cycles": future16,
            "any_abrupt_drop": int(cyc.get("any_abrupt_drop", 0)) if np.isfinite(float(cyc.get("any_abrupt_drop", 0))) else 0,
            "source_stem": stem,
            "local_cycle_index": local_idx,
            "n_segments_inferred": int(n_segments),
            "h5_total_frames": int(total_frames),
            "segment_start_frame": int(start),
            "segment_end_frame": int(end),
            "sampled_first_frame": int(idx[0]),
            "sampled_last_frame": int(idx[-1]),
            "n_sampled_frames": int(len(idx)),
            "cycle_table_n_frames": float(cyc.get("n_frames_table", np.nan)),
            "downsample": int(args.downsample),
            "n_candidates": int(len(cand)),
            "overlay_png": str(overlay_path),
        }
        segment_rows.append(segment_meta)
        if cand.empty:
            continue
        cand = cand.copy()
        for key, val in segment_meta.items():
            cand[key] = val
        candidate_tables.append(cand)
        for _, row in cand.head(args.top_candidates_per_cycle).iterrows():
            roi_rows.append({
                "cycleNo": cycle,
                "source_stem": stem,
                "local_cycle_index": local_idx,
                "front_candidate_rank": int(cyc["transfer_rank"]),
                "object_candidate_rank": int(row["candidate_rank"]),
                "validation_score": candidate_validation_score(row, transfer_score, int(cyc["transfer_rank"])),
                "validation_label": "transfer_warning_roi_candidate",
                "cohort_role": "transfer_warning",
                "event_reference_cycle": cycle,
                "object_x_full_approx": float(row["x_full_approx"]),
                "object_y_full_approx": float(row["y_full_approx"]),
                "object_x_ds": float(row["x_ds"]),
                "object_y_ds": float(row["y_ds"]),
                "object_area_ds_px": int(row["area_ds_px"]),
                "object_mean_residual": float(row["mean_residual"]),
                "object_mean_abs_z": float(row["mean_abs_z"]),
                "transfer_rank": int(cyc["transfer_rank"]),
                "transferred_masked_residual_signature": transfer_score,
                "future_any_drop_within_8cycles": future8,
                "future_any_drop_within_16cycles": future16,
                "any_abrupt_drop": int(segment_meta["any_abrupt_drop"]),
                "segment_start_frame": int(start),
                "segment_end_frame": int(end),
                "downsample": int(args.downsample),
                "source_reconstructed_overlay_png": str(overlay_path),
            })

    segments = pd.DataFrame(segment_rows).sort_values("transfer_rank") if segment_rows else pd.DataFrame()
    candidates = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame()
    roi_table = pd.DataFrame(roi_rows).sort_values(["transfer_rank", "validation_score"], ascending=[True, False]) if roi_rows else pd.DataFrame()
    missing = pd.DataFrame(missing_rows)

    segments.to_csv(out / "transfer_ranked_reconstructed_segments.csv", index=False)
    candidates.to_csv(out / "transfer_ranked_reconstructed_candidates.csv", index=False)
    roi_table.to_csv(out / "transfer_ranked_roi_table.csv", index=False)
    missing.to_csv(out / "transfer_ranked_missing_cycles.csv", index=False)

    summary = {
        "ranked_cycles": args.ranked_cycles,
        "n_ranked_cycles_requested": int(args.top_cycles),
        "n_cycles_sampled": int(len(segments)),
        "n_missing_cycles": int(len(missing)),
        "n_reconstructed_candidates": int(len(candidates)),
        "n_roi_rows": int(len(roi_table)),
        "top_candidates_per_cycle": int(args.top_candidates_per_cycle),
        "downsample": int(args.downsample),
        "samples_per_segment": int(args.samples_per_segment),
        "sampled_cycles": segments[["cycleNo", "transfer_rank", "transferred_masked_residual_signature", "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "n_candidates"]].to_dict("records") if not segments.empty else [],
        "top_roi_rows": roi_table.head(16).to_dict("records") if not roi_table.empty else [],
        "guardrail": "Automatic transfer-ranked ROI candidates reconstructed from sampled HDF5 cycle segments; compatible with particle-region sequence export, but not manual annotations or validated fronts.",
    }
    with (out / "transfer_ranked_roi_reconstruction_summary.json").open("w") as f:
        json.dump(clean_value(summary), f, indent=2)
    with (out / "README.md").open("w") as f:
        f.write("# Transfer-Ranked ROI Reconstruction\n\n")
        f.write("Reconstructs candidate particle-region ROIs for cycles ranked by transferred masked-residual warning score.\n\n")
        f.write(f"- Cycles sampled: {summary['n_cycles_sampled']}\n")
        f.write(f"- Candidate components: {summary['n_reconstructed_candidates']}\n")
        f.write(f"- ROI table rows: {summary['n_roi_rows']}\n\n")
        f.write(summary["guardrail"] + "\n")
    print(json.dumps(clean_value(summary), indent=2)[:8000])


if __name__ == "__main__":
    main()
