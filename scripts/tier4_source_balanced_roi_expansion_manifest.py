#!/usr/bin/env python3
"""Build a source-balanced ROI expansion manifest from full NMC HDF5 cycles.

Most downstream ROI video/physics audits use selected event/control,
transfer-ranked, or balanced-future crops. This script attacks that cohort
selection bottleneck by selecting additional cycle segments across source
movies and weak future-label strata, then reconstructing automatic
particle-like candidates from sampled HDF5 frames.

The output is an expansion manifest for follow-up sequence export and manual
QC, not a manual particle annotation or validated degradation label.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import h5py
import numpy as np
import pandas as pd

from tier1_reconstruct_event_object_candidates import (
    detect_candidates,
    infer_n_segments,
    parse_addr,
    read_downsampled_movie,
    sample_indices,
    save_overlay,
    segment_bounds,
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


def resolve_existing_path(candidates: Sequence[str]) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def read_cycle_frames_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "cycleNo" in df.columns:
        return df
    raw = pd.read_csv(path, encoding="utf-8-sig", header=None)
    raw = raw.iloc[:, :2].copy()
    raw.columns = ["cycleNo", "n_frames"]
    return raw


def existing_cycle_keys(paths: Iterable[str]) -> Set[Tuple[int, str]]:
    keys: Set[Tuple[int, str]] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "cycleNo" not in df.columns:
            continue
        if "source_stem" not in df.columns and "addrs" in df.columns:
            parsed = df["addrs"].map(parse_addr).apply(pd.Series)
            df = pd.concat([df, parsed], axis=1)
        for _, row in df.iterrows():
            cyc = pd.to_numeric(row.get("cycleNo"), errors="coerce")
            stem = str(row.get("source_stem", ""))
            if np.isfinite(cyc) and stem:
                keys.add((int(cyc), stem))
    return keys


def numeric01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1)


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = float(x.mean()) if x.notna().any() else 0.0
    sd = float(x.std(ddof=0)) if x.notna().any() else 0.0
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((x.fillna(mu) - mu) / sd).clip(-5, 5)


def prepare_ranked_cycles(path: str, existing_keys: Set[Tuple[int, str]]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    parsed = df["addrs"].map(parse_addr).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    df = df.dropna(subset=["cycleNo"])
    df["cycleNo_int"] = df["cycleNo"].round().astype(int)
    df["future8"] = numeric01(df.get("future_any_drop_within_8cycles", 0))
    df["future16"] = numeric01(df.get("future_any_drop_within_16cycles", 0))
    df["same_cycle_drop"] = numeric01(df.get("any_abrupt_drop", 0))
    df["warning_z"] = zscore(df.get("transferred_masked_residual_signature", 0))
    df["pc2_z"] = zscore(df.get("cycle_state_pc2", 0))
    df["frame_low_z"] = -zscore(df.get("n_frames", 0))
    df["selection_priority"] = (
        1.00 * df["future16"]
        + 0.65 * df["future8"]
        + 0.35 * df["same_cycle_drop"]
        + 0.20 * df["warning_z"].clip(lower=0)
        + 0.15 * df["pc2_z"].abs()
        + 0.10 * df["frame_low_z"].clip(lower=0)
    )
    df["already_in_existing_video_cohort"] = [
        (int(c), str(s)) in existing_keys for c, s in zip(df["cycleNo_int"], df["source_stem"])
    ]
    return df


def pick_rows(pool: pd.DataFrame, n: int, selected_idx: Set[int], prefer_new: bool = True) -> List[int]:
    if n <= 0 or pool.empty:
        return []
    work = pool[~pool.index.isin(selected_idx)].copy()
    if work.empty:
        return []
    work["new_bonus"] = (~work["already_in_existing_video_cohort"]).astype(float) if prefer_new else 0.0
    work["pick_score"] = work["selection_priority"] + 0.45 * work["new_bonus"]
    work = work.sort_values(["pick_score", "cycleNo"], ascending=[False, True])
    return list(work.head(n).index)


def source_balanced_selection(df: pd.DataFrame, per_source: int, max_cycles: int) -> pd.DataFrame:
    selected: List[int] = []
    reasons: Dict[int, str] = {}
    selected_set: Set[int] = set()

    for source, group in df.groupby("source_stem", sort=True):
        group = group.copy()
        target = max(1, min(per_source, len(group)))
        quotas = [
            ("future16_positive", group[group["future16"] == 1], 1),
            ("future16_negative", group[group["future16"] == 0], 1),
            ("future8_positive", group[(group["future8"] == 1) & (group["future16"] == 0)], 1),
            ("source_representative", group, max(0, target - 3)),
        ]
        for reason, pool, n in quotas:
            picks = pick_rows(pool, n, selected_set)
            for idx in picks:
                selected.append(idx)
                selected_set.add(idx)
                reasons[idx] = reason
        if sum(1 for idx in selected if df.loc[idx, "source_stem"] == source) < target:
            missing = target - sum(1 for idx in selected if df.loc[idx, "source_stem"] == source)
            picks = pick_rows(group, missing, selected_set)
            for idx in picks:
                selected.append(idx)
                selected_set.add(idx)
                reasons[idx] = "source_fill"

    out = df.loc[selected].copy()
    out["selection_reason"] = [reasons.get(idx, "source_fill") for idx in out.index]
    if len(out) > max_cycles:
        out = out.sort_values(
            ["already_in_existing_video_cohort", "selection_priority", "cycleNo"],
            ascending=[True, False, True],
        ).head(max_cycles)
    out = out.sort_values(["source_stem", "cycleNo"]).reset_index(drop=True)
    out["expansion_cycle_rank"] = np.arange(1, len(out) + 1)
    return out


def candidate_validation_score(cand: pd.Series, cyc: pd.Series) -> float:
    rank_score = float(cand.get("rank_score", 0.0))
    abs_z = float(cand.get("mean_abs_z", 0.0))
    area = float(cand.get("area_ds_px", 0.0))
    priority = float(cyc.get("selection_priority", 0.0))
    new_bonus = 0.35 if not bool(cyc.get("already_in_existing_video_cohort", False)) else 0.0
    return float(np.log1p(max(rank_score, 0.0)) + 0.25 * abs_z + 0.015 * area + 0.20 * priority + new_bonus)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--ranked-cycles", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_residual_state_transfer_warning/masked_residual_state_transfer_full_cycles.csv")
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_roi_expansion_manifest")
    parser.add_argument("--per-source", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=48)
    parser.add_argument("--top-candidates-per-cycle", type=int, default=2)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--samples-per-segment", type=int, default=12)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--max-area-fraction", type=float, default=0.015)
    parser.add_argument("--max-candidates", type=int, default=60)
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

    existing_keys = existing_cycle_keys([
        os.path.join(args.root, "derived/multi_cycle_roi_sequences/selected_roi_sequence_manifest.csv"),
        os.path.join(args.root, "derived/transfer_ranked_roi_sequences/selected_roi_sequence_manifest.csv"),
        os.path.join(args.root, "derived/balanced_future_roi_sequences/selected_roi_sequence_manifest.csv"),
        os.path.join(args.root, "derived/control_roi_sequences/selected_roi_sequence_manifest.csv"),
    ])
    ranked = prepare_ranked_cycles(args.ranked_cycles, existing_keys)
    selected = source_balanced_selection(ranked, args.per_source, args.max_cycles)

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

    selected.to_csv(out / "source_balanced_cycle_expansion_plan.csv", index=False)

    segment_rows: List[Dict[str, object]] = []
    candidate_tables: List[pd.DataFrame] = []
    roi_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []

    for _, cyc in selected.iterrows():
        cycle = int(cyc["cycleNo_int"])
        stem = str(cyc.get("source_stem", ""))
        local = cyc.get("local_cycle_index", np.nan)
        local_idx = int(local) if np.isfinite(float(local)) else -1
        h5_path = os.path.join(args.root, "NMC_degradation_3_160623_Halfthedata", f"{stem}.hdf5")
        if not stem or local_idx < 0 or not os.path.exists(h5_path):
            missing_rows.append({"cycleNo": cycle, "source_stem": stem, "local_cycle_index": local_idx, "reason": "missing_h5_or_parse"})
            continue
        n_segments = infer_n_segments(stem, particles[particles["source_stem"] == stem])
        try:
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
        except Exception as exc:
            missing_rows.append({"cycleNo": cycle, "source_stem": stem, "local_cycle_index": local_idx, "reason": f"read_or_detect_failed:{exc}"})
            continue

        overlay_path = overlay_dir / f"cycle_{cycle}_rank{int(cyc['expansion_cycle_rank'])}_{stem}_local{local_idx}.png"
        save_overlay(mean_img, z_img, cand, str(overlay_path), f"source-balanced expansion | cycle {cycle} | {stem} local {local_idx}")
        segment_meta = {
            "cycleNo": cycle,
            "expansion_cycle_rank": int(cyc["expansion_cycle_rank"]),
            "selection_reason": str(cyc.get("selection_reason", "")),
            "selection_priority": float(cyc.get("selection_priority", np.nan)),
            "already_in_existing_video_cohort": bool(cyc.get("already_in_existing_video_cohort", False)),
            "future_any_drop_within_8cycles": int(cyc.get("future8", 0)),
            "future_any_drop_within_16cycles": int(cyc.get("future16", 0)),
            "any_abrupt_drop": int(cyc.get("same_cycle_drop", 0)),
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
                "expansion_cycle_rank": int(cyc["expansion_cycle_rank"]),
                "object_candidate_rank": int(row["candidate_rank"]),
                "validation_score": candidate_validation_score(row, cyc),
                "validation_label": "source_balanced_expansion_candidate",
                "cohort_role": "source_balanced_expansion",
                "selection_reason": str(cyc.get("selection_reason", "")),
                "already_in_existing_video_cohort": bool(cyc.get("already_in_existing_video_cohort", False)),
                "future_any_drop_within_8cycles": int(cyc.get("future8", 0)),
                "future_any_drop_within_16cycles": int(cyc.get("future16", 0)),
                "any_abrupt_drop": int(cyc.get("same_cycle_drop", 0)),
                "object_x_full_approx": float(row["x_full_approx"]),
                "object_y_full_approx": float(row["y_full_approx"]),
                "object_x_ds": float(row["x_ds"]),
                "object_y_ds": float(row["y_ds"]),
                "object_area_ds_px": int(row["area_ds_px"]),
                "object_mean_residual": float(row["mean_residual"]),
                "object_mean_abs_z": float(row["mean_abs_z"]),
                "segment_start_frame": int(start),
                "segment_end_frame": int(end),
                "downsample": int(args.downsample),
                "source_reconstructed_overlay_png": str(overlay_path),
            })

    segments = pd.DataFrame(segment_rows).sort_values(["source_stem", "cycleNo"]) if segment_rows else pd.DataFrame()
    candidates = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame()
    roi_table = pd.DataFrame(roi_rows).sort_values(["source_stem", "cycleNo", "validation_score"], ascending=[True, True, False]) if roi_rows else pd.DataFrame()
    missing = pd.DataFrame(missing_rows)

    segments.to_csv(out / "source_balanced_reconstructed_segments.csv", index=False)
    candidates.to_csv(out / "source_balanced_reconstructed_candidates.csv", index=False)
    roi_table.to_csv(out / "source_balanced_roi_table.csv", index=False)
    missing.to_csv(out / "source_balanced_missing_cycles.csv", index=False)

    source_coverage = selected.groupby("source_stem", dropna=False).agg(
        selected_cycles=("cycleNo_int", "count"),
        new_cycles=("already_in_existing_video_cohort", lambda s: int((~s.astype(bool)).sum())),
        future8_positive=("future8", "sum"),
        future16_positive=("future16", "sum"),
        same_cycle_drop=("same_cycle_drop", "sum"),
        mean_selection_priority=("selection_priority", "mean"),
    ).reset_index()
    if not segments.empty:
        seg_cov = segments.groupby("source_stem").agg(
            sampled_cycles=("cycleNo", "count"),
            total_candidates=("n_candidates", "sum"),
        ).reset_index()
        source_coverage = source_coverage.merge(seg_cov, how="left", on="source_stem")
    source_coverage.to_csv(out / "source_balanced_source_coverage.csv", index=False)

    summary = {
        "ranked_cycles": args.ranked_cycles,
        "n_ranked_cycles": int(len(ranked)),
        "n_existing_cycle_source_keys": int(len(existing_keys)),
        "n_selected_cycles": int(len(selected)),
        "n_new_selected_cycles": int((~selected["already_in_existing_video_cohort"].astype(bool)).sum()),
        "n_sources_selected": int(selected["source_stem"].nunique()),
        "n_sampled_cycles": int(len(segments)),
        "n_missing_cycles": int(len(missing)),
        "n_reconstructed_candidates": int(len(candidates)),
        "n_roi_rows": int(len(roi_table)),
        "per_source": int(args.per_source),
        "max_cycles": int(args.max_cycles),
        "top_candidates_per_cycle": int(args.top_candidates_per_cycle),
        "source_coverage": clean_value(source_coverage.to_dict("records")),
        "selection_reason_counts": clean_value(selected["selection_reason"].value_counts().to_dict()),
        "selected_label_counts": {
            "future8_positive": int(selected["future8"].sum()),
            "future16_positive": int(selected["future16"].sum()),
            "same_cycle_drop": int(selected["same_cycle_drop"].sum()),
        },
        "top_roi_rows": clean_value(roi_table.head(20).to_dict("records")) if not roi_table.empty else [],
        "outputs": {
            "cycle_plan": str(out / "source_balanced_cycle_expansion_plan.csv"),
            "segments": str(out / "source_balanced_reconstructed_segments.csv"),
            "candidates": str(out / "source_balanced_reconstructed_candidates.csv"),
            "roi_table": str(out / "source_balanced_roi_table.csv"),
            "source_coverage": str(out / "source_balanced_source_coverage.csv"),
            "missing": str(out / "source_balanced_missing_cycles.csv"),
            "summary": str(out / "source_balanced_roi_expansion_summary.json"),
        },
        "guardrail": "Source-balanced expansion candidates are automatic proposals from sampled HDF5 frames. They reduce source/cycle selection bias for follow-up ROI export and manual QC, but do not validate particle identity, fronts, diffusion, or degradation mechanisms.",
    }
    (out / "source_balanced_roi_expansion_summary.json").write_text(json.dumps(clean_value(summary), indent=2, sort_keys=True))
    lines = [
        "# Source-Balanced ROI Expansion Manifest",
        "",
        f"- Ranked cycles: {summary['n_ranked_cycles']}",
        f"- Existing cycle/source keys: {summary['n_existing_cycle_source_keys']}",
        f"- Selected cycles: {summary['n_selected_cycles']} ({summary['n_new_selected_cycles']} new vs existing video cohorts)",
        f"- Sources selected: {summary['n_sources_selected']}",
        f"- Sampled cycles: {summary['n_sampled_cycles']}",
        f"- Reconstructed candidates: {summary['n_reconstructed_candidates']}",
        f"- ROI rows: {summary['n_roi_rows']}",
        "",
        "## Label Counts",
        "",
        json.dumps(summary["selected_label_counts"], indent=2, sort_keys=True),
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
