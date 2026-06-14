#!/usr/bin/env python3
"""All-cycle coverage atlas for the Alek/Jiho NMC degradation dataset.

This audit maps the complete cycle-level echem/optical ledger against all ROI/video
cohorts generated so far. It is a planning and coverage tool: it does not extract
new ROIs or make physical claims.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_RE = re.compile(r"NMC_degradation_3_160623_Halfthedata[\\/]+([^\\/]+?)chopped[\\/]+")
CYCLE_RE = re.compile(r"_cycle(\d+)")


ROI_COHORTS = {
    "selected_roi_sequences": "selected_roi_sequence_manifest.csv",
    "multi_cycle_roi_sequences": "selected_roi_sequence_manifest.csv",
    "balanced_future_roi_sequences": "selected_roi_sequence_manifest.csv",
    "control_roi_sequences": "selected_roi_sequence_manifest.csv",
    "control_roi_sequences_expanded": "selected_roi_sequence_manifest.csv",
    "source_balanced_roi_sequences": "selected_roi_sequence_manifest.csv",
    "source_balanced_pre_event_roi_sequences": "selected_roi_sequence_manifest.csv",
    "transfer_ranked_roi_sequences": "selected_roi_sequence_manifest.csv",
    "source_balanced_roi_expansion_manifest": "source_balanced_roi_table.csv",
    "source_balanced_pre_event_sampling_manifest": "source_balanced_pre_event_roi_table.csv",
    "balanced_future_roi_reconstruction": "balanced_future_roi_table.csv",
    "transfer_ranked_roi_reconstruction": "transfer_ranked_roi_table.csv",
    "multi_cycle_roi_cohort": "multi_cycle_roi_table.csv",
    "roi_front_qc_package": "roi_front_qc_manifest.csv",
    "control_balanced_front_qc_package": "control_balanced_front_qc_manifest.csv",
    "qc_review_packet": "qc_review_manifest.csv",
    "precursor_informed_roi_review": "precursor_informed_roi_review_manifest.csv",
}

CYCLE_OUTPUTS = {
    "event_echem_coupling": "event_echem_cycle_table.csv",
    "echem_optical_regime_atlas": "echem_optical_regime_cycle_table.csv",
    "cross_modal_degradation_consensus": "cross_modal_consensus_cycle_table.csv",
    "cycle_state_space_transition_audit": "cycle_state_space_table.csv",
    "cycle_hazard_warning_audit": "cycle_hazard_warning_predictions.csv",
    "masked_rollout_cycle_warning": "masked_rollout_cycle_warning_ranked_cycles.csv",
    "masked_residual_state_transfer_warning": "masked_residual_state_transfer_ranked_cycles.csv",
}


def scalar(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if math.isnan(float(v)):
            return None
        return float(v)
    if pd.isna(v):
        return None
    return v


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def parse_source(addr: Any) -> str | None:
    if not isinstance(addr, str):
        return None
    m = SOURCE_RE.search(addr)
    if m:
        return m.group(1)
    text = addr.replace("\\", "/")
    base = Path(text).stem
    if "chopped" in base:
        return base.split("chopped", 1)[0]
    m2 = re.search(r"([^/]+)_cycle\d+", base)
    return m2.group(1) if m2 else None


def parse_local_cycle(addr: Any) -> float | None:
    if not isinstance(addr, str):
        return None
    m = CYCLE_RE.search(addr)
    return float(m.group(1)) if m else None


def normalize_cycle_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "cycleNo" not in out.columns:
        for col in out.columns:
            if col.lower() in {"cycle", "cycleno", "cycleno_int"}:
                out = out.rename(columns={col: "cycleNo"})
                break
    if "cycleNo" in out.columns:
        out["cycleNo"] = pd.to_numeric(out["cycleNo"], errors="coerce")
    return out


def source_key_from_df(df: pd.DataFrame) -> pd.Series:
    if "source_stem" in df.columns:
        return df["source_stem"].astype("string")
    if "addrs" in df.columns:
        return df["addrs"].map(parse_source).astype("string")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")


def count_roi_coverage(derived: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    cohort_summary: list[dict[str, Any]] = []
    for cohort, rel in ROI_COHORTS.items():
        path = derived / cohort / rel
        df = normalize_cycle_col(read_csv(path))
        if df.empty or "cycleNo" not in df.columns:
            cohort_summary.append({"cohort": cohort, "path": str(path), "status": "missing_or_no_cycle", "n_rows": 0, "n_cycles": 0, "n_sources": 0})
            continue
        df["source_stem_norm"] = source_key_from_df(df)
        df = df.dropna(subset=["cycleNo"])
        grouped = df.groupby(["cycleNo", "source_stem_norm"], dropna=False).size().reset_index(name="n_roi")
        grouped["cohort"] = cohort
        rows.extend(grouped.to_dict("records"))
        cohort_summary.append({
            "cohort": cohort,
            "path": str(path),
            "status": "ok",
            "n_rows": int(len(df)),
            "n_cycles": int(df["cycleNo"].nunique()),
            "n_sources": int(df["source_stem_norm"].nunique(dropna=True)),
        })
    long = pd.DataFrame(rows)
    if long.empty:
        wide = pd.DataFrame(columns=["cycleNo", "source_stem"])
    else:
        wide = long.pivot_table(index=["cycleNo", "source_stem_norm"], columns="cohort", values="n_roi", aggfunc="sum", fill_value=0)
        wide.columns = [f"n_roi_{c}" for c in wide.columns]
        wide = wide.reset_index().rename(columns={"source_stem_norm": "source_stem"})
    return wide, pd.DataFrame(cohort_summary)


def add_cycle_output_flags(cycles: pd.DataFrame, derived: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = cycles.copy()
    rows: list[dict[str, Any]] = []
    for name, rel in CYCLE_OUTPUTS.items():
        path = derived / name / rel
        df = normalize_cycle_col(read_csv(path))
        if df.empty or "cycleNo" not in df.columns:
            rows.append({"output": name, "path": str(path), "status": "missing_or_no_cycle", "n_rows": 0, "n_cycles": 0})
            out[f"has_{name}"] = False
            continue
        present = set(pd.to_numeric(df["cycleNo"], errors="coerce").dropna().astype(float))
        out[f"has_{name}"] = out["cycleNo"].astype(float).isin(present)
        rows.append({"output": name, "path": str(path), "status": "ok", "n_rows": int(len(df)), "n_cycles": int(len(present))})
    return out, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--repo-dir", default="/scratch/<account>/<username>/Alek_Jiho/alek_jiho_nmc_deg")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/all_cycle_dataset_coverage_atlas")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    repo = Path(args.repo_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cycle_table_path = derived / "echem_optical_regime_atlas" / "echem_optical_regime_cycle_table.csv"
    cycles = normalize_cycle_col(read_csv(cycle_table_path))
    if cycles.empty:
        raise FileNotFoundError(cycle_table_path)
    keep = [
        "cycleNo", "addrs", "n_frames", "any_abrupt_drop", "drop_count", "synchronized_drop_2plus",
        "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "cycles_to_next_drop_within_16",
        "capacity_mAh", "capacity_fraction_of_first", "coulombic_efficiency_pct", "V_min", "V_max",
        "shape_I_abs_mean_mA", "cycle_state_pc1", "cycle_state_cluster", "cross_modal_consensus_score",
        "n_modal_votes", "consensus_class", "hazard_probability_mean", "echem_outlier_score",
    ]
    cycles = cycles[[c for c in keep if c in cycles.columns]].copy()
    cycles["source_stem"] = cycles.get("addrs", pd.Series([None] * len(cycles))).map(parse_source)
    cycles["local_cycle_index"] = cycles.get("addrs", pd.Series([None] * len(cycles))).map(parse_local_cycle)
    cycles["cycleNo"] = pd.to_numeric(cycles["cycleNo"], errors="coerce")
    cycles = cycles.dropna(subset=["cycleNo"]).sort_values("cycleNo").reset_index(drop=True)

    echem = normalize_cycle_col(read_csv(derived / "echem_per_cycle.csv"))
    if not echem.empty:
        echem_cols = [c for c in ["cycleNo", "charge_capacity_mAh", "discharge_capacity_mAh", "n_points"] if c in echem.columns]
        cycles = cycles.merge(echem[echem_cols], on="cycleNo", how="left", suffixes=("", "_echem_per_cycle"))

    roi_wide, roi_summary = count_roi_coverage(derived)
    if not roi_wide.empty:
        cycles = cycles.merge(roi_wide, on=["cycleNo", "source_stem"], how="left")
    roi_cols = [c for c in cycles.columns if c.startswith("n_roi_")]
    for c in roi_cols:
        cycles[c] = cycles[c].fillna(0).astype(int)
    cycles["n_roi_total_across_tracked_cohorts"] = cycles[roi_cols].sum(axis=1) if roi_cols else 0
    cycles["has_any_roi_video_sequence"] = cycles["n_roi_total_across_tracked_cohorts"] > 0
    primary_cols = [c for c in roi_cols if any(k in c for k in ["source_balanced_roi_sequences", "source_balanced_pre_event_roi_sequences", "multi_cycle_roi_sequences", "balanced_future_roi_sequences", "selected_roi_sequences"])]
    cycles["has_primary_roi_sequence"] = cycles[primary_cols].sum(axis=1) > 0 if primary_cols else False

    cycles, cycle_output_summary = add_cycle_output_flags(cycles, derived)

    h5 = read_csv(derived / "h5_inventory.csv")
    if not h5.empty:
        h5["source_stem"] = h5["filename"].map(lambda x: Path(str(x)).stem)
        h5["is_degradation3_cycle_source"] = h5["filename"].astype(str).str.contains("NMC_degradation_3_160623")
        source_cycle_counts = cycles.groupby("source_stem", dropna=False).agg(
            n_cycle_rows=("cycleNo", "nunique"),
            min_global_cycle=("cycleNo", "min"),
            max_global_cycle=("cycleNo", "max"),
            n_roi_cycles=("has_any_roi_video_sequence", "sum"),
            n_primary_roi_cycles=("has_primary_roi_sequence", "sum"),
            n_event_cycles=("any_abrupt_drop", "sum"),
            n_future16_positive=("future_any_drop_within_16cycles", "sum"),
        ).reset_index()
        h5_enriched = h5.merge(source_cycle_counts, on="source_stem", how="left")
    else:
        h5_enriched = pd.DataFrame()

    source_summary = cycles.groupby("source_stem", dropna=False).agg(
        n_cycle_rows=("cycleNo", "nunique"),
        min_cycle=("cycleNo", "min"),
        max_cycle=("cycleNo", "max"),
        min_local_cycle=("local_cycle_index", "min"),
        max_local_cycle=("local_cycle_index", "max"),
        n_any_drop_cycles=("any_abrupt_drop", "sum"),
        n_future8_positive=("future_any_drop_within_8cycles", "sum"),
        n_future16_positive=("future_any_drop_within_16cycles", "sum"),
        n_cycles_with_any_roi=("has_any_roi_video_sequence", "sum"),
        n_cycles_with_primary_roi=("has_primary_roi_sequence", "sum"),
        total_roi_rows=("n_roi_total_across_tracked_cohorts", "sum"),
        median_capacity_fraction=("capacity_fraction_of_first", "median"),
        max_consensus_score=("cross_modal_consensus_score", "max"),
        max_hazard_probability=("hazard_probability_mean", "max"),
    ).reset_index()
    source_summary["roi_cycle_coverage_fraction"] = source_summary["n_cycles_with_any_roi"] / source_summary["n_cycle_rows"].replace(0, np.nan)
    source_summary["primary_roi_cycle_coverage_fraction"] = source_summary["n_cycles_with_primary_roi"] / source_summary["n_cycle_rows"].replace(0, np.nan)

    candidate = cycles.copy()
    for col in ["future_any_drop_within_16cycles", "future_any_drop_within_8cycles", "any_abrupt_drop"]:
        if col in candidate.columns:
            candidate[col] = pd.to_numeric(candidate[col], errors="coerce").fillna(0)
    candidate["coverage_gap_priority"] = 0.0
    candidate.loc[~candidate["has_any_roi_video_sequence"], "coverage_gap_priority"] += 2.0
    candidate.loc[~candidate["has_primary_roi_sequence"], "coverage_gap_priority"] += 1.0
    for col, weight in [("future_any_drop_within_16cycles", 1.5), ("future_any_drop_within_8cycles", 2.0), ("any_abrupt_drop", 1.0)]:
        if col in candidate.columns:
            candidate["coverage_gap_priority"] += weight * candidate[col].astype(float)
    for col, weight in [("cross_modal_consensus_score", 1.0), ("hazard_probability_mean", 0.75), ("echem_outlier_score", 0.5)]:
        if col in candidate.columns:
            vals = pd.to_numeric(candidate[col], errors="coerce")
            if vals.notna().any() and vals.max() != vals.min():
                norm = (vals - vals.min()) / (vals.max() - vals.min())
                candidate["coverage_gap_priority"] += weight * norm.fillna(0)
    gap_cols = [
        "cycleNo", "source_stem", "local_cycle_index", "coverage_gap_priority", "has_any_roi_video_sequence", "has_primary_roi_sequence",
        "any_abrupt_drop", "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "cycles_to_next_drop_within_16",
        "n_roi_total_across_tracked_cohorts", "capacity_fraction_of_first", "cross_modal_consensus_score", "hazard_probability_mean",
        "echem_outlier_score", "consensus_class",
    ]
    gap_queue = candidate[[c for c in gap_cols if c in candidate.columns]].sort_values(
        ["coverage_gap_priority", "cycleNo"], ascending=[False, True]
    )

    cycles.to_csv(out / "all_cycle_coverage_table.csv", index=False)
    source_summary.to_csv(out / "all_cycle_source_coverage_summary.csv", index=False)
    roi_summary.to_csv(out / "all_cycle_roi_cohort_coverage_summary.csv", index=False)
    cycle_output_summary.to_csv(out / "all_cycle_output_coverage_summary.csv", index=False)
    gap_queue.to_csv(out / "all_cycle_coverage_gap_priority.csv", index=False)
    if not h5_enriched.empty:
        h5_enriched.to_csv(out / "all_cycle_h5_inventory_enriched.csv", index=False)

    n_cycles = int(len(cycles))
    n_sources = int(cycles["source_stem"].nunique(dropna=True))
    n_roi_cycles = int(cycles["has_any_roi_video_sequence"].sum())
    n_primary_cycles = int(cycles["has_primary_roi_sequence"].sum())
    n_gap_future16 = int(((~cycles["has_any_roi_video_sequence"]) & (pd.to_numeric(cycles.get("future_any_drop_within_16cycles", 0), errors="coerce").fillna(0) > 0)).sum())
    top_gap = gap_queue.head(20).to_dict("records")
    summary = {
        "overall_status": "all_cycle_coverage_atlas_ready",
        "n_cycle_rows": n_cycles,
        "n_sources": n_sources,
        "n_h5_inventory_rows": int(len(h5_enriched)) if not h5_enriched.empty else 0,
        "n_roi_cohorts_checked": len(ROI_COHORTS),
        "n_cycle_outputs_checked": len(CYCLE_OUTPUTS),
        "n_cycles_with_any_roi_video_sequence": n_roi_cycles,
        "n_cycles_with_primary_roi_sequence": n_primary_cycles,
        "any_roi_cycle_coverage_fraction": n_roi_cycles / n_cycles if n_cycles else None,
        "primary_roi_cycle_coverage_fraction": n_primary_cycles / n_cycles if n_cycles else None,
        "n_future16_positive_cycles_without_any_roi_sequence": n_gap_future16,
        "top_source_gaps": source_summary.sort_values(["primary_roi_cycle_coverage_fraction", "n_future16_positive", "n_cycle_rows"], ascending=[True, False, False]).head(12).to_dict("records"),
        "top_coverage_gap_cycles": top_gap,
        "roi_cohort_summary": roi_summary.to_dict("records"),
        "cycle_output_summary": cycle_output_summary.to_dict("records"),
        "outputs": {
            "cycle_table": str(out / "all_cycle_coverage_table.csv"),
            "source_summary": str(out / "all_cycle_source_coverage_summary.csv"),
            "roi_cohort_summary": str(out / "all_cycle_roi_cohort_coverage_summary.csv"),
            "cycle_output_summary": str(out / "all_cycle_output_coverage_summary.csv"),
            "gap_priority": str(out / "all_cycle_coverage_gap_priority.csv"),
            "h5_inventory_enriched": str(out / "all_cycle_h5_inventory_enriched.csv"),
            "summary": str(out / "all_cycle_dataset_coverage_summary.json"),
        },
        "guardrail": "This atlas audits dataset and ROI-analysis coverage only. It does not extract new ROIs, validate particle identity, train deployment models, or make calibrated diffusion/phase-boundary claims.",
    }
    with (out / "all_cycle_dataset_coverage_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)

    readme = [
        "# All-Cycle Dataset Coverage Atlas",
        "",
        "Maps the full cycle-level echem/optical ledger onto the ROI/video cohorts generated so far.",
        "",
        f"- Cycle rows: {n_cycles}",
        f"- Sources: {n_sources}",
        f"- Cycles with any tracked ROI/video sequence: {n_roi_cycles} ({summary['any_roi_cycle_coverage_fraction']:.3f})",
        f"- Cycles with primary ROI sequence coverage: {n_primary_cycles} ({summary['primary_roi_cycle_coverage_fraction']:.3f})",
        f"- Future16-positive cycles without any tracked ROI sequence: {n_gap_future16}",
        "",
        "Top gap-priority cycles are listed in `all_cycle_coverage_gap_priority.csv`.",
        "",
        f"Guardrail: {summary['guardrail']}",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")


if __name__ == "__main__":
    main()
