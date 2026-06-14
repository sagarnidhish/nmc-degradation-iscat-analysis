#!/usr/bin/env python3
"""Build a source-balanced pre-event ROI sampling manifest.

The future-specific residual audit showed that the current source-balanced
warning readout is heavily entangled with event proximity. This script creates
a broader, explicitly event-relative sampling cohort: near-pre-event,
far-pre-event, post-event washout, and no-near-event controls. It reconstructs
automatic particle-like ROI candidates from sampled HDF5 segments so the next
step can export particle-only videos and test whether pre-event dynamics remain
after source and past-event controls.

The output is a sampling/QC manifest, not a validated particle or mechanism
annotation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

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


EVENT_WINDOWS = {
    "near_pre_event_1_8": (1.0, 8.0),
    "mid_pre_event_9_16": (9.0, 16.0),
    "far_pre_event_17_32": (17.0, 32.0),
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def resolve_existing_path(candidates: Sequence[str]) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = float(x.mean()) if x.notna().any() else 0.0
    sd = float(x.std(ddof=0)) if x.notna().any() else 0.0
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((x.fillna(mu) - mu) / sd).clip(-5, 5)


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
            cycle = pd.to_numeric(row.get("cycleNo"), errors="coerce")
            source = str(row.get("source_stem", ""))
            if np.isfinite(cycle) and source:
                keys.add((int(round(float(cycle))), source))
    return keys


def load_event_cycles(path: Path) -> np.ndarray:
    events = pd.read_csv(path)
    return np.sort(pd.to_numeric(events["cycleNo"], errors="coerce").dropna().unique().astype(float))


def append_event_context(df: pd.DataFrame, event_cycles: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    cycles = numeric(out, "cycleNo").to_numpy(dtype=float)
    rows: List[Dict[str, Any]] = []
    for cycle in cycles:
        deltas = event_cycles - cycle
        future = deltas[deltas > 0]
        past = -deltas[deltas < 0]
        rows.append({
            "cycles_to_next_event": float(future.min()) if len(future) else np.nan,
            "cycles_since_prev_event": float(past.min()) if len(past) else np.nan,
            "nearest_event_abs_delta": float(np.min(np.abs(deltas))) if len(deltas) else np.nan,
            "current_any_event": int(np.any(deltas == 0)),
        })
    ctx = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, ctx], axis=1)
    out["event_relative_bin"] = "no_near_event_control"
    current = numeric(out, "current_any_event").eq(1)
    post = numeric(out, "cycles_since_prev_event").between(1, 16, inclusive="both")
    out.loc[current, "event_relative_bin"] = "current_event"
    out.loc[post & ~current, "event_relative_bin"] = "post_event_1_16"
    for label, (lo, hi) in EVENT_WINDOWS.items():
        mask = numeric(out, "cycles_to_next_event").between(lo, hi, inclusive="both") & ~current & ~post
        out.loc[mask, "event_relative_bin"] = label
    out["future_event_within_8cycles"] = numeric(out, "cycles_to_next_event").between(1, 8, inclusive="both").astype(int)
    out["future_event_within_16cycles"] = numeric(out, "cycles_to_next_event").between(1, 16, inclusive="both").astype(int)
    out["future_event_within_32cycles"] = numeric(out, "cycles_to_next_event").between(1, 32, inclusive="both").astype(int)
    out["past_event_within_16cycles"] = numeric(out, "cycles_since_prev_event").between(1, 16, inclusive="both").astype(int)
    return out


def prepare_cycles(path: Path, event_cycles: np.ndarray, existing_keys: Set[Tuple[int, str]]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cycleNo"] = numeric(df, "cycleNo")
    parsed = df["addrs"].map(parse_addr).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    df = df.dropna(subset=["cycleNo", "source_stem"]).copy()
    df["cycleNo_int"] = df["cycleNo"].round().astype(int)
    df = append_event_context(df, event_cycles)
    df["already_in_existing_video_cohort"] = [
        (int(c), str(s)) in existing_keys for c, s in zip(df["cycleNo_int"], df["source_stem"])
    ]
    df["masked_residual_z"] = zscore(df.get("transferred_masked_residual_signature", 0))
    df["cycle_state_abs_pc2_z"] = zscore(df.get("cycle_state_pc2", 0)).abs()
    df["low_frame_z"] = -zscore(df.get("n_frames", 0))
    pre_strength = (
        1.0 * numeric(df, "future_event_within_16cycles")
        + 0.45 * numeric(df, "future_event_within_32cycles")
        - 0.45 * numeric(df, "past_event_within_16cycles")
    )
    df["sampling_priority"] = (
        pre_strength
        + 0.25 * df["masked_residual_z"].clip(lower=0)
        + 0.15 * df["cycle_state_abs_pc2_z"]
        + 0.10 * df["low_frame_z"].clip(lower=0)
        + 0.40 * (~df["already_in_existing_video_cohort"]).astype(float)
    )
    return df


def pick_from_pool(pool: pd.DataFrame, n: int, selected: Set[int]) -> List[int]:
    if n <= 0 or pool.empty:
        return []
    work = pool[~pool.index.isin(selected)].copy()
    if work.empty:
        return []
    work = work.sort_values(["sampling_priority", "cycleNo"], ascending=[False, True])
    return list(work.head(n).index)


def source_balanced_pre_event_selection(df: pd.DataFrame, per_source: int, max_cycles: int) -> pd.DataFrame:
    selected: List[int] = []
    selected_set: Set[int] = set()
    reasons: Dict[int, str] = {}
    quotas = [
        ("near_pre_event_1_8", 1),
        ("mid_pre_event_9_16", 1),
        ("far_pre_event_17_32", 1),
        ("no_near_event_control", 1),
        ("post_event_1_16", 1),
    ]
    for source, group in df.groupby("source_stem", sort=True):
        group = group[group["event_relative_bin"].ne("current_event")].copy()
        if group.empty:
            continue
        target = max(1, min(per_source, len(group)))
        for bin_name, n in quotas:
            if sum(1 for idx in selected if df.loc[idx, "source_stem"] == source) >= target:
                break
            picks = pick_from_pool(group[group["event_relative_bin"].eq(bin_name)], n, selected_set)
            for idx in picks:
                selected.append(idx)
                selected_set.add(idx)
                reasons[idx] = bin_name
        current = sum(1 for idx in selected if df.loc[idx, "source_stem"] == source)
        if current < target:
            for idx in pick_from_pool(group, target - current, selected_set):
                selected.append(idx)
                selected_set.add(idx)
                reasons[idx] = "source_fill"

    out = df.loc[selected].copy()
    out["selection_reason"] = [reasons.get(idx, "source_fill") for idx in out.index]
    if len(out) > max_cycles:
        out = out.sort_values(
            ["already_in_existing_video_cohort", "event_relative_bin", "sampling_priority", "cycleNo"],
            ascending=[True, True, False, True],
        ).head(max_cycles)
    out = out.sort_values(["source_stem", "cycleNo"]).reset_index(drop=True)
    out["pre_event_sampling_rank"] = np.arange(1, len(out) + 1)
    return out


def candidate_validation_score(cand: pd.Series, cyc: pd.Series) -> float:
    rank_score = float(cand.get("rank_score", 0.0))
    abs_z = float(cand.get("mean_abs_z", 0.0))
    area = float(cand.get("area_ds_px", 0.0))
    priority = float(cyc.get("sampling_priority", 0.0))
    pre_bonus = 0.30 if str(cyc.get("event_relative_bin", "")).endswith("pre_event_1_8") else 0.0
    new_bonus = 0.25 if not bool(cyc.get("already_in_existing_video_cohort", False)) else 0.0
    return float(np.log1p(max(rank_score, 0.0)) + 0.25 * abs_z + 0.012 * area + 0.20 * priority + pre_bonus + new_bonus)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--ranked-cycles", default="/scratch/<account>/<username>/Alek_Jiho/derived/masked_residual_state_transfer_warning/masked_residual_state_transfer_full_cycles.csv")
    parser.add_argument("--event-cycles", default="/scratch/<account>/<username>/Alek_Jiho/derived/particle_event_targets/particle_abrupt_events.csv")
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_sampling_manifest")
    parser.add_argument("--per-source", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=64)
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
    if not particles_path:
        raise SystemExit("Missing exampleParticles.csv")
    particles = pd.read_csv(particles_path, encoding="utf-8-sig")
    particles["cycleNo"] = numeric(particles, "cycleNo")
    particles = pd.concat([particles, particles["addrs"].map(parse_addr).apply(pd.Series)], axis=1)

    existing_keys = existing_cycle_keys([
        os.path.join(args.root, "derived/multi_cycle_roi_sequences/selected_roi_sequence_manifest.csv"),
        os.path.join(args.root, "derived/transfer_ranked_roi_sequences/selected_roi_sequence_manifest.csv"),
        os.path.join(args.root, "derived/balanced_future_roi_sequences/selected_roi_sequence_manifest.csv"),
        os.path.join(args.root, "derived/control_roi_sequences/selected_roi_sequence_manifest.csv"),
        os.path.join(args.root, "derived/source_balanced_roi_sequences/selected_roi_sequence_manifest.csv"),
    ])
    event_cycles = load_event_cycles(Path(args.event_cycles))
    ranked = prepare_cycles(Path(args.ranked_cycles), event_cycles, existing_keys)
    selected = source_balanced_pre_event_selection(ranked, args.per_source, args.max_cycles)
    selected.to_csv(out / "source_balanced_pre_event_cycle_plan.csv", index=False)

    segment_rows: List[Dict[str, Any]] = []
    candidate_tables: List[pd.DataFrame] = []
    roi_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for _, cyc in selected.iterrows():
        cycle = int(cyc["cycleNo_int"])
        stem = str(cyc["source_stem"])
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

        overlay_path = overlay_dir / f"cycle_{cycle}_pre{int(cyc['pre_event_sampling_rank'])}_{stem}_local{local_idx}.png"
        save_overlay(mean_img, z_img, cand, str(overlay_path), f"pre-event sampling | cycle {cycle} | {stem} local {local_idx}")
        segment_meta = {
            "cycleNo": cycle,
            "pre_event_sampling_rank": int(cyc["pre_event_sampling_rank"]),
            "event_relative_bin": str(cyc.get("event_relative_bin", "")),
            "selection_reason": str(cyc.get("selection_reason", "")),
            "sampling_priority": float(cyc.get("sampling_priority", np.nan)),
            "cycles_to_next_event": float(cyc.get("cycles_to_next_event", np.nan)),
            "cycles_since_prev_event": float(cyc.get("cycles_since_prev_event", np.nan)),
            "already_in_existing_video_cohort": bool(cyc.get("already_in_existing_video_cohort", False)),
            "future_any_drop_within_8cycles": int(cyc.get("future_event_within_8cycles", 0)),
            "future_any_drop_within_16cycles": int(cyc.get("future_event_within_16cycles", 0)),
            "future_event_within_32cycles": int(cyc.get("future_event_within_32cycles", 0)),
            "past_event_within_16cycles": int(cyc.get("past_event_within_16cycles", 0)),
            "any_abrupt_drop": int(cyc.get("current_any_event", 0)),
            "source_stem": stem,
            "local_cycle_index": local_idx,
            "n_segments_inferred": int(n_segments),
            "h5_total_frames": int(total_frames),
            "segment_start_frame": int(start),
            "segment_end_frame": int(end),
            "sampled_first_frame": int(idx[0]),
            "sampled_last_frame": int(idx[-1]),
            "n_sampled_frames": int(len(idx)),
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
                "expansion_cycle_rank": int(cyc["pre_event_sampling_rank"]),
                "pre_event_sampling_rank": int(cyc["pre_event_sampling_rank"]),
                "object_candidate_rank": int(row["candidate_rank"]),
                "validation_score": candidate_validation_score(row, cyc),
                "validation_label": "source_balanced_pre_event_candidate",
                "cohort_role": "source_balanced_pre_event_sampling",
                "event_relative_bin": str(cyc.get("event_relative_bin", "")),
                "selection_reason": str(cyc.get("selection_reason", "")),
                "cycles_to_next_event": float(cyc.get("cycles_to_next_event", np.nan)),
                "cycles_since_prev_event": float(cyc.get("cycles_since_prev_event", np.nan)),
                "already_in_existing_video_cohort": bool(cyc.get("already_in_existing_video_cohort", False)),
                "future_any_drop_within_8cycles": int(cyc.get("future_event_within_8cycles", 0)),
                "future_any_drop_within_16cycles": int(cyc.get("future_event_within_16cycles", 0)),
                "future_event_within_32cycles": int(cyc.get("future_event_within_32cycles", 0)),
                "past_event_within_16cycles": int(cyc.get("past_event_within_16cycles", 0)),
                "any_abrupt_drop": int(cyc.get("current_any_event", 0)),
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

    segments.to_csv(out / "source_balanced_pre_event_segments.csv", index=False)
    candidates.to_csv(out / "source_balanced_pre_event_candidates.csv", index=False)
    roi_table.to_csv(out / "source_balanced_pre_event_roi_table.csv", index=False)
    missing.to_csv(out / "source_balanced_pre_event_missing_cycles.csv", index=False)

    source_coverage = selected.groupby("source_stem", dropna=False).agg(
        selected_cycles=("cycleNo_int", "count"),
        new_cycles=("already_in_existing_video_cohort", lambda s: int((~s.astype(bool)).sum())),
        near_pre=("event_relative_bin", lambda s: int(s.eq("near_pre_event_1_8").sum())),
        mid_pre=("event_relative_bin", lambda s: int(s.eq("mid_pre_event_9_16").sum())),
        far_pre=("event_relative_bin", lambda s: int(s.eq("far_pre_event_17_32").sum())),
        post_event=("event_relative_bin", lambda s: int(s.eq("post_event_1_16").sum())),
        controls=("event_relative_bin", lambda s: int(s.eq("no_near_event_control").sum())),
        mean_sampling_priority=("sampling_priority", "mean"),
    ).reset_index()
    if not segments.empty:
        seg_cov = segments.groupby("source_stem").agg(sampled_cycles=("cycleNo", "count"), total_candidates=("n_candidates", "sum")).reset_index()
        source_coverage = source_coverage.merge(seg_cov, how="left", on="source_stem")
    source_coverage.to_csv(out / "source_balanced_pre_event_source_coverage.csv", index=False)

    bin_counts = selected["event_relative_bin"].value_counts().to_dict()
    roi_bin_counts = roi_table["event_relative_bin"].value_counts().to_dict() if not roi_table.empty else {}
    summary = {
        "ranked_cycles": args.ranked_cycles,
        "event_cycles": clean_json(event_cycles.tolist()),
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
        "cycle_bin_counts": clean_json(bin_counts),
        "roi_bin_counts": clean_json(roi_bin_counts),
        "source_coverage": clean_json(source_coverage.to_dict("records")),
        "top_roi_rows": clean_json(roi_table.head(20).to_dict("records")) if not roi_table.empty else [],
        "outputs": {
            "cycle_plan": str(out / "source_balanced_pre_event_cycle_plan.csv"),
            "segments": str(out / "source_balanced_pre_event_segments.csv"),
            "candidates": str(out / "source_balanced_pre_event_candidates.csv"),
            "roi_table": str(out / "source_balanced_pre_event_roi_table.csv"),
            "source_coverage": str(out / "source_balanced_pre_event_source_coverage.csv"),
            "missing": str(out / "source_balanced_pre_event_missing_cycles.csv"),
            "summary": str(out / "source_balanced_pre_event_sampling_summary.json"),
        },
        "guardrail": "Pre-event sampling candidates are automatic particle-like proposals from sampled HDF5 frames. They broaden event-relative/source-balanced coverage for follow-up ROI video export and future-specific modeling, but do not validate particle identity, phase fronts, diffusion, or causal degradation mechanisms.",
    }
    (out / "source_balanced_pre_event_sampling_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    lines = [
        "# Source-Balanced Pre-Event Sampling Manifest",
        "",
        f"- Ranked cycles: {summary['n_ranked_cycles']}",
        f"- Event cycles: {summary['event_cycles']}",
        f"- Selected cycles: {summary['n_selected_cycles']} ({summary['n_new_selected_cycles']} new vs existing video cohorts)",
        f"- Sources selected: {summary['n_sources_selected']}",
        f"- Sampled cycles: {summary['n_sampled_cycles']}",
        f"- Reconstructed candidates: {summary['n_reconstructed_candidates']}",
        f"- ROI rows: {summary['n_roi_rows']}",
        "",
        "## Cycle Bins",
        "",
        json.dumps(summary["cycle_bin_counts"], indent=2, sort_keys=True),
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
