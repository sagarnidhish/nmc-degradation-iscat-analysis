#!/usr/bin/env python3
"""Integrate NMC event evidence into a degradation-mode ranking table.

This combines the outputs of event target, synchrony, echem coupling, protocol
context, recovery QC, and full-frame proxy QC analyses. The goal is to produce a
single auditable table that says which optical events look like credible
persistent degradation signals, which look more likely to need artifact/QC review,
and what the next physical interpretation step should be.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def read_csv_if_exists(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def classify_row(row: pd.Series) -> str:
    n_particles = safe_float(row.get("n_event_particles", row.get("n_particles_event", 0)), 0)
    sustained = safe_float(row.get("n_sustained_next_cycle", 0), 0)
    drop = safe_float(row.get("mean_drop_frac", np.nan), np.nan)
    frame_pct = safe_float(row.get("global_frame_percentile", row.get("n_frames_percentile", np.nan)), np.nan)
    roi_fallback = bool(row.get("fallback_mask", False)) if not pd.isna(row.get("fallback_mask", False)) else False
    if n_particles >= 2 and sustained >= n_particles and np.isfinite(drop) and drop >= 0.10:
        if np.isfinite(frame_pct) and frame_pct <= 0.10:
            return "synchronized_persistent_drop_low_frame_count"
        return "synchronized_persistent_drop"
    if n_particles >= 1 and sustained >= n_particles and np.isfinite(drop) and drop >= 0.10:
        if roi_fallback:
            return "single_persistent_drop_needs_roi_qc"
        return "single_persistent_drop"
    return "weak_or_unresolved_event"


def score_row(row: pd.Series) -> float:
    n_particles = safe_float(row.get("n_event_particles", row.get("n_particles_event", 0)), 0)
    drop = safe_float(row.get("mean_drop_frac", np.nan), 0)
    sustained = safe_float(row.get("n_sustained_next_cycle", 0), 0)
    frame_pct = safe_float(row.get("global_frame_percentile", row.get("n_frames_percentile", np.nan)), np.nan)
    stage_drift = safe_float(row.get("stage_drift_xy_sampled", np.nan), np.nan)
    roi_cv = safe_float(row.get("roi_cv_time", np.nan), np.nan)
    score = 0.0
    score += 2.0 * min(n_particles, 4.0)
    score += 8.0 * max(drop, 0.0)
    score += 1.5 * min(sustained, n_particles if n_particles else sustained)
    if np.isfinite(frame_pct):
        score += 1.0 * (1.0 - frame_pct)
    if np.isfinite(stage_drift):
        score -= 0.5 * min(stage_drift, 1.0)
    if np.isfinite(roi_cv):
        score -= 2.0 * min(roi_cv, 0.2)
    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/integrated_event_evidence")
    args = parser.parse_args()

    event_targets = read_csv_if_exists(os.path.join(args.derived_dir, "particle_event_targets", "particle_abrupt_events.csv"))
    synchrony = read_csv_if_exists(os.path.join(args.derived_dir, "event_synchrony", "event_synchrony_by_cycle.csv"))
    echem = read_csv_if_exists(os.path.join(args.derived_dir, "event_echem_coupling", "event_echem_cycle_table.csv"))
    recovery = read_csv_if_exists(os.path.join(args.derived_dir, "event_recovery_qc", "event_cycle_recovery_summary.csv"))
    frame_qc = read_csv_if_exists(os.path.join(args.derived_dir, "event_frame_proxy_qc", "event_frame_proxy_qc.csv"))

    if recovery.empty:
        raise SystemExit("event_cycle_recovery_summary.csv not found; run tier1_event_recovery_qc.py first")

    for df in [event_targets, synchrony, echem, recovery, frame_qc]:
        if not df.empty and "cycleNo" in df.columns:
            df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")

    base = recovery.copy()
    if not synchrony.empty:
        base = base.merge(synchrony[[c for c in synchrony.columns if c in ["cycleNo", "n_particles_event", "n_frames_z", "n_frames_percentile"]]], on="cycleNo", how="left", suffixes=("", "_sync"))
    if not echem.empty:
        ecols = [c for c in ["cycleNo", "any_event", "synchronized_event", "V_mean", "I_mean_mA", "I_abs_mean_mA", "V_range", "block_mode", "I_pos_fraction", "I_neg_fraction"] if c in echem.columns]
        base = base.merge(echem[ecols], on="cycleNo", how="left", suffixes=("", "_echem"))
    if not frame_qc.empty:
        fcols = [c for c in ["cycleNo", "roi_fraction", "fallback_mask", "roi_cv_time", "roi_background_contrast", "stage_drift_xy_sampled", "stage_drift_z_sampled", "average_intensity_sample_delta", "preview_png"] if c in frame_qc.columns]
        base = base.merge(frame_qc[fcols], on="cycleNo", how="left", suffixes=("", "_frame"))

    base["degradation_mode_hypothesis"] = base.apply(classify_row, axis=1)
    base["evidence_score"] = base.apply(score_row, axis=1)
    base["next_action"] = base["degradation_mode_hypothesis"].map({
        "synchronized_persistent_drop_low_frame_count": "manual ROI preview validation; compare with exact object detector coordinates; test if low frame count is protocol or acquisition artifact",
        "synchronized_persistent_drop": "manual ROI preview validation; model as candidate coordinated degradation onset",
        "single_persistent_drop_needs_roi_qc": "prioritize particle-level crop validation before mechanistic interpretation",
        "single_persistent_drop": "track as possible local particle failure mode",
        "weak_or_unresolved_event": "keep as low-priority candidate until more evidence is available",
    }).fillna("manual review")

    ordered = base.sort_values(["evidence_score", "cycleNo"], ascending=[False, True]).reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    table_path = os.path.join(args.out_dir, "integrated_event_evidence.csv")
    ordered.to_csv(table_path, index=False)

    mode_counts = ordered["degradation_mode_hypothesis"].value_counts().to_dict()
    summary: Dict[str, object] = {
        "derived_dir": args.derived_dir,
        "n_event_cycles": int(len(ordered)),
        "mode_counts": {str(k): int(v) for k, v in mode_counts.items()},
        "top_events": ordered.head(10).to_dict(orient="records"),
        "artifact_guardrails": [
            "Chopped cycle HDF5 files referenced by exampleParticles.csv are not present on Isambard.",
            "Full-frame proxy masks are not final particle/object masks; fallback thresholding was used in sampled segments.",
            "Frame-count associations are hypothesis-generating and may reflect protocol/acquisition coupling.",
            "No diffusion coefficient should be interpreted from these NMC event tables without spatial calibration and validated particle-boundary tracking.",
        ],
    }
    summary_path = os.path.join(args.out_dir, "integrated_event_evidence_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    readme_path = os.path.join(args.out_dir, "README.md")
    lines = [
        "# Integrated NMC Event Evidence",
        "",
        "Combines synchrony, electrochemistry, protocol context, persistence/recovery QC, and sampled full-frame proxy QC into one degradation-mode evidence table.",
        "",
        "## Mode Counts",
        "",
        "| Mode | Count |",
        "|---|---:|",
    ]
    for mode, count in mode_counts.items():
        lines.append(f"| `{mode}` | {int(count)} |")
    lines.extend([
        "",
        "## Ranked Event Cycles",
        "",
        "| rank | cycleNo | mode | score | particles | mean_drop_frac | sustained | frame_pct | V_mean | I_mean_mA | next_action |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for i, row in ordered.iterrows():
        lines.append("| {rank} | {cycle:.1f} | `{mode}` | {score:.3f} | `{particles}` | {drop:.3f} | {sust}/{n} | {fp} | {v} | {cur} | {action} |".format(
            rank=i + 1,
            cycle=float(row["cycleNo"]),
            mode=row["degradation_mode_hypothesis"],
            score=float(row["evidence_score"]),
            particles=row.get("particles", ""),
            drop=float(row["mean_drop_frac"]) if pd.notna(row.get("mean_drop_frac", np.nan)) else float("nan"),
            sust=int(row.get("n_sustained_next_cycle", 0)),
            n=int(row.get("n_event_particles", 0)),
            fp="" if pd.isna(row.get("global_frame_percentile", np.nan)) else f"{float(row['global_frame_percentile']):.3f}",
            v="" if pd.isna(row.get("V_mean", np.nan)) else f"{float(row['V_mean']):.4g}",
            cur="" if pd.isna(row.get("I_mean_mA", np.nan)) else f"{float(row['I_mean_mA']):.4g}",
            action=row.get("next_action", ""),
        ))
    lines.extend([
        "",
        "## Guardrails",
        "",
    ])
    for guard in summary["artifact_guardrails"]:
        lines.append(f"- {guard}")
    lines.append("")
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))

    for p in [table_path, summary_path, readme_path]:
        print(f"Saved: {p}")
    print(json.dumps(summary, indent=2, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
