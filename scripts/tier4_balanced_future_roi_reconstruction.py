#!/usr/bin/env python3
"""Build a class-balanced direct-video ROI cohort for future-drop audits.

The transfer-ranked cohort was useful but biased toward warning-ranked cycles.
This script samples direct HDF5 cycle segments from both weak future8-positive
and future8-negative cycles, reconstructs automatic particle-like components,
and writes a ROI table compatible with the existing particle-region sequence
exporter.

The output is a balanced weak-label candidate cohort for video/physics audits,
not manual particle annotations or deployable degradation labels.
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



def select_balanced_cycles(ranked: pd.DataFrame, n_positive: int, n_negative: int) -> pd.DataFrame:
    df = ranked.copy()
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    df["future_any_drop_within_8cycles"] = pd.to_numeric(df.get("future_any_drop_within_8cycles"), errors="coerce")
    df["any_abrupt_drop"] = pd.to_numeric(df.get("any_abrupt_drop"), errors="coerce").fillna(0)
    df["transferred_masked_residual_signature"] = pd.to_numeric(df.get("transferred_masked_residual_signature"), errors="coerce")
    df = df.dropna(subset=["cycleNo", "future_any_drop_within_8cycles", "transferred_masked_residual_signature"])

    pos = df[df["future_any_drop_within_8cycles"] == 1].sort_values("transferred_masked_residual_signature", ascending=False).head(n_positive).copy()
    neg_pool = df[df["future_any_drop_within_8cycles"] == 0].copy()
    clean_neg = neg_pool[neg_pool["any_abrupt_drop"] == 0].copy()
    hard_n = min(len(clean_neg), max(1, n_negative // 2))
    easy_n = max(0, n_negative - hard_n)
    hard = clean_neg.sort_values("transferred_masked_residual_signature", ascending=False).head(hard_n)
    remaining = clean_neg.drop(index=hard.index, errors="ignore")
    easy = remaining.sort_values("transferred_masked_residual_signature", ascending=True).head(easy_n)
    neg = pd.concat([hard, easy], ignore_index=False)
    if len(neg) < n_negative:
        extra = neg_pool.drop(index=neg.index, errors="ignore").sort_values("transferred_masked_residual_signature", ascending=False).head(n_negative - len(neg))
        neg = pd.concat([neg, extra], ignore_index=False)

    pos["selection_role"] = "future8_positive"
    pos["selection_subrole"] = "high_warning_positive"
    neg = neg.head(n_negative).copy()
    neg["selection_role"] = "future8_negative"
    median_neg_score = neg["transferred_masked_residual_signature"].median() if len(neg) else np.nan
    neg["selection_subrole"] = np.where(
        neg["transferred_masked_residual_signature"] >= median_neg_score,
        "hard_negative_high_warning",
        "low_risk_negative",
    )
    out = pd.concat([pos, neg], ignore_index=True)
    out = out.sort_values(["selection_role", "transferred_masked_residual_signature"], ascending=[True, False]).reset_index(drop=True)
    out["balanced_cycle_rank"] = np.arange(1, len(out) + 1)
    return out


def candidate_validation_score(cand: pd.Series, warning_score: float, target_rank: int, is_positive: int) -> float:
    rank_score = float(cand.get("rank_score", 0.0))
    abs_z = float(cand.get("mean_abs_z", 0.0))
    area = float(cand.get("area_ds_px", 0.0))
    warning = float(warning_score) if np.isfinite(warning_score) else 0.0
    return float(np.log1p(max(rank_score, 0.0)) + 0.30 * abs_z + 0.02 * area + 0.08 * np.log1p(max(warning, 0.0)) + 0.05 * is_positive - 0.005 * target_rank)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--ranked-cycles", default="/scratch/<account>/<username>/Alek_Jiho/derived/masked_residual_state_transfer_warning/masked_residual_state_transfer_ranked_cycles.csv")
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/balanced_future_roi_reconstruction")
    parser.add_argument("--positive-cycles", type=int, default=12)
    parser.add_argument("--negative-cycles", type=int, default=12)
    parser.add_argument("--top-candidates-per-cycle", type=int, default=3)
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

    ranked_raw = pd.read_csv(args.ranked_cycles)
    selected = select_balanced_cycles(ranked_raw, args.positive_cycles, args.negative_cycles)
    parsed = selected["addrs"].map(parse_addr).apply(pd.Series)
    selected = pd.concat([selected, parsed], axis=1)

    particles = pd.read_csv(particles_path, encoding="utf-8-sig")
    particles["cycleNo"] = pd.to_numeric(particles["cycleNo"], errors="coerce")
    particles = pd.concat([particles, particles["addrs"].map(parse_addr).apply(pd.Series)], axis=1)
    if frames_path:
        frames_meta = read_cycle_frames_csv(frames_path)
        frames_meta["cycleNo"] = pd.to_numeric(frames_meta["cycleNo"], errors="coerce")
        frames_meta["n_frames_table"] = pd.to_numeric(frames_meta["n_frames"], errors="coerce")
        selected = selected.merge(frames_meta[["cycleNo", "n_frames_table"]], how="left", on="cycleNo")
    else:
        selected["n_frames_table"] = np.nan

    segment_rows: List[Dict[str, object]] = []
    candidate_tables: List[pd.DataFrame] = []
    roi_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []

    for _, cyc in selected.iterrows():
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
        cand, mean_img, z_img = detect_candidates(sampled, args.downsample, args.z_threshold, args.min_area, args.max_area_fraction, args.max_candidates)
        target_rank = int(cyc["balanced_cycle_rank"])
        overlay_path = overlay_dir / f"cycle_{int(cycle)}_rank{target_rank}_{stem}_local{local_idx}.png"
        save_overlay(mean_img, z_img, cand, str(overlay_path), f"balanced future {cyc['selection_role']} rank {target_rank} | cycle {cycle:g}")
        warning_score = float(cyc.get("transferred_masked_residual_signature", np.nan))
        future8 = int(cyc.get("future_any_drop_within_8cycles", 0)) if np.isfinite(float(cyc.get("future_any_drop_within_8cycles", 0))) else 0
        future16 = int(cyc.get("future_any_drop_within_16cycles", 0)) if np.isfinite(float(cyc.get("future_any_drop_within_16cycles", 0))) else 0
        abrupt = int(cyc.get("any_abrupt_drop", 0)) if np.isfinite(float(cyc.get("any_abrupt_drop", 0))) else 0
        segment_meta = {
            "cycleNo": cycle,
            "balanced_cycle_rank": target_rank,
            "selection_role": cyc.get("selection_role", ""),
            "selection_subrole": cyc.get("selection_subrole", ""),
            "transferred_masked_residual_signature": warning_score,
            "future_any_drop_within_8cycles": future8,
            "future_any_drop_within_16cycles": future16,
            "any_abrupt_drop": abrupt,
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
                "front_candidate_rank": target_rank,
                "object_candidate_rank": int(row["candidate_rank"]),
                "validation_score": candidate_validation_score(row, warning_score, target_rank, future8),
                "validation_label": "balanced_future_roi_candidate",
                "cohort_role": str(cyc.get("selection_role", "")),
                "selection_subrole": str(cyc.get("selection_subrole", "")),
                "event_reference_cycle": cycle,
                "object_x_full_approx": float(row["x_full_approx"]),
                "object_y_full_approx": float(row["y_full_approx"]),
                "object_x_ds": float(row["x_ds"]),
                "object_y_ds": float(row["y_ds"]),
                "object_area_ds_px": int(row["area_ds_px"]),
                "object_mean_residual": float(row["mean_residual"]),
                "object_mean_abs_z": float(row["mean_abs_z"]),
                "balanced_cycle_rank": target_rank,
                "transferred_masked_residual_signature": warning_score,
                "future_any_drop_within_8cycles": future8,
                "future_any_drop_within_16cycles": future16,
                "any_abrupt_drop": abrupt,
                "segment_start_frame": int(start),
                "segment_end_frame": int(end),
                "downsample": int(args.downsample),
                "source_reconstructed_overlay_png": str(overlay_path),
            })

    segments = pd.DataFrame(segment_rows).sort_values("balanced_cycle_rank") if segment_rows else pd.DataFrame()
    candidates = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame()
    roi_table = pd.DataFrame(roi_rows).sort_values(["balanced_cycle_rank", "validation_score"], ascending=[True, False]) if roi_rows else pd.DataFrame()
    missing = pd.DataFrame(missing_rows)

    segments.to_csv(out / "balanced_future_reconstructed_segments.csv", index=False)
    candidates.to_csv(out / "balanced_future_reconstructed_candidates.csv", index=False)
    roi_table.to_csv(out / "balanced_future_roi_table.csv", index=False)
    missing.to_csv(out / "balanced_future_missing_cycles.csv", index=False)

    label_counts = roi_table.groupby(["cohort_role", "future_any_drop_within_8cycles"], dropna=False).size().reset_index(name="n").to_dict("records") if not roi_table.empty else []
    cycle_counts = segments.groupby(["selection_role", "future_any_drop_within_8cycles"], dropna=False).size().reset_index(name="n_cycles").to_dict("records") if not segments.empty else []
    summary = {
        "ranked_cycles": args.ranked_cycles,
        "n_positive_cycles_requested": int(args.positive_cycles),
        "n_negative_cycles_requested": int(args.negative_cycles),
        "n_cycles_sampled": int(len(segments)),
        "n_missing_cycles": int(len(missing)),
        "n_reconstructed_candidates": int(len(candidates)),
        "n_roi_rows": int(len(roi_table)),
        "top_candidates_per_cycle": int(args.top_candidates_per_cycle),
        "downsample": int(args.downsample),
        "samples_per_segment": int(args.samples_per_segment),
        "cycle_label_counts": clean_value(cycle_counts),
        "roi_label_counts": clean_value(label_counts),
        "sampled_cycles": clean_value(segments[["cycleNo", "balanced_cycle_rank", "selection_role", "selection_subrole", "transferred_masked_residual_signature", "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "any_abrupt_drop", "n_candidates"]].to_dict("records")) if not segments.empty else [],
        "top_roi_rows": clean_value(roi_table.head(16).to_dict("records")) if not roi_table.empty else [],
        "guardrail": "Automatic class-balanced future-drop ROI candidates reconstructed from sampled HDF5 cycle segments. Labels are weak cycle-level future8 labels projected to particle-region candidates; outputs are not manual annotations, degradation labels, or validated fronts.",
    }
    with (out / "balanced_future_roi_reconstruction_summary.json").open("w") as f:
        json.dump(clean_value(summary), f, indent=2)
    with (out / "README.md").open("w") as f:
        f.write("# Balanced Future-Drop ROI Reconstruction\n\n")
        f.write("Reconstructs automatic particle-region ROI candidates from future8-positive and future8-negative cycles for direct video physics/model audits.\n\n")
        f.write(f"- Cycles sampled: {summary['n_cycles_sampled']}\n")
        f.write(f"- Candidate components: {summary['n_reconstructed_candidates']}\n")
        f.write(f"- ROI table rows: {summary['n_roi_rows']}\n")
        f.write(f"- ROI label counts: {summary['roi_label_counts']}\n\n")
        f.write(summary["guardrail"] + "\n")
    print(json.dumps(clean_value(summary), indent=2)[:8000])


if __name__ == "__main__":
    main()
