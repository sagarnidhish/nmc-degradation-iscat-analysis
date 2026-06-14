#!/usr/bin/env python3
"""Audit raw source/event lattice coverage for pre-event control design.

Several source-balanced pre-event tests found useful near-pre particle-region
signals, but the same-source ladder audit showed zero near+far same-source
ladders. This script checks the raw cycle/source index rather than the sampled
ROI table, so the limitation is explicit: do same-source far-pre controls exist
in the underlying Alek/Jiho HDF5/source coverage, and what comparisons remain
feasible without fabricating unavailable controls?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from tier4_source_balanced_pre_event_sampling_manifest import (
    append_event_context,
    clean_json,
    load_event_cycles,
    numeric,
    parse_addr,
    resolve_existing_path,
)


ORDERED_BINS = [
    "near_pre_event_1_8",
    "mid_pre_event_9_16",
    "far_pre_event_17_32",
    "post_event_1_16",
    "no_near_event_control",
    "current_event",
]


def h5_segment_exists(root: Path, source: str) -> bool:
    return (root / "NMC_degradation_3_160623_Halfthedata" / f"{source}.hdf5").exists()


def source_class(source: str) -> str:
    if "HighHighCOV" in source:
        return "HighHighCOV"
    if "HighCOV" in source:
        return "HighCOV"
    return "nominal"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--event-cycles", default="/scratch/<account>/<username>/Alek_Jiho/derived/particle_event_targets/particle_abrupt_events.csv")
    parser.add_argument("--sampled-pre-event-summary", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_same_source_ladder_audit/source_balanced_pre_event_same_source_ladder_summary.json")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/pre_event_source_lattice_coverage_audit")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    particles_path = resolve_existing_path([
        args.particles_csv,
        str(root / "exampleParticles.csv"),
        str(root / "alek_jiho_nmc_deg" / "echemDF_full" / "exampleParticles.csv"),
    ])
    if not particles_path:
        raise SystemExit("Missing exampleParticles.csv")

    particles = pd.read_csv(particles_path, encoding="utf-8-sig")
    particles["cycleNo"] = numeric(particles, "cycleNo")
    particles = pd.concat([particles, particles["addrs"].map(parse_addr).apply(pd.Series)], axis=1)
    particles = particles.dropna(subset=["cycleNo", "source_stem", "local_cycle_index"]).copy()
    particles["cycleNo_int"] = particles["cycleNo"].round().astype(int)
    particles["local_cycle_index_int"] = numeric(particles, "local_cycle_index").round().astype(int)

    event_cycles = load_event_cycles(Path(args.event_cycles))
    cycles = particles.drop_duplicates(["cycleNo_int", "source_stem", "local_cycle_index_int"]).copy()
    cycles = append_event_context(cycles, event_cycles)
    cycles["source_class"] = cycles["source_stem"].astype(str).map(source_class)
    cycles["h5_available"] = cycles["source_stem"].astype(str).map(lambda s: h5_segment_exists(root, s))

    source_rows: List[Dict[str, Any]] = []
    for source, sub in cycles.groupby("source_stem", sort=True):
        counts = sub["event_relative_bin"].value_counts().to_dict()
        near = int(counts.get("near_pre_event_1_8", 0))
        mid = int(counts.get("mid_pre_event_9_16", 0))
        far = int(counts.get("far_pre_event_17_32", 0))
        post = int(counts.get("post_event_1_16", 0))
        control = int(counts.get("no_near_event_control", 0))
        current = int(counts.get("current_event", 0))
        source_rows.append({
            "source_stem": source,
            "source_class": source_class(str(source)),
            "h5_available": bool(sub["h5_available"].any()),
            "n_cycles": int(len(sub)),
            "cycle_min": int(sub["cycleNo_int"].min()),
            "cycle_max": int(sub["cycleNo_int"].max()),
            "near_pre_rows": near,
            "mid_pre_rows": mid,
            "far_pre_rows": far,
            "post_event_rows": post,
            "no_near_event_control_rows": control,
            "current_event_rows": current,
            "has_near": bool(near > 0),
            "has_near_mid": bool(near > 0 and mid > 0),
            "has_near_far": bool(near > 0 and far > 0),
            "has_near_post_or_control": bool(near > 0 and (post + control) > 0),
            "recommended_within_source_comparison": (
                "near_vs_mid_pre" if near > 0 and mid > 0
                else "near_vs_post_control" if near > 0 and (post + control) > 0
                else "near_only_no_within_source_control" if near > 0
                else "control_only"
            ),
        })
    source_summary = pd.DataFrame(source_rows).sort_values(["has_near", "source_stem"], ascending=[False, True])

    candidate_control_rows = cycles[cycles["event_relative_bin"].isin(["far_pre_event_17_32", "no_near_event_control"])].copy()
    candidate_control_rows = candidate_control_rows.sort_values(["event_relative_bin", "source_class", "source_stem", "cycleNo_int"])
    near_source_rows = source_summary[source_summary["has_near"]].copy()

    sampled_ladder = {}
    if os.path.exists(args.sampled_pre_event_summary):
        sampled_ladder = json.loads(Path(args.sampled_pre_event_summary).read_text())

    source_path = out / "pre_event_source_lattice_source_summary.csv"
    cycle_path = out / "pre_event_source_lattice_cycle_table.csv"
    controls_path = out / "pre_event_source_lattice_candidate_controls.csv"
    summary_path = out / "pre_event_source_lattice_coverage_summary.json"
    source_summary.to_csv(source_path, index=False)
    cycles.to_csv(cycle_path, index=False)
    candidate_control_rows.to_csv(controls_path, index=False)

    near_sources = source_summary[source_summary["has_near"]]
    summary = {
        "n_cycle_rows": int(len(cycles)),
        "n_sources": int(cycles["source_stem"].nunique()),
        "n_sources_with_h5": int(source_summary["h5_available"].sum()),
        "event_cycles": clean_json(event_cycles.tolist()),
        "event_relative_bin_counts": clean_json(cycles["event_relative_bin"].value_counts().to_dict()),
        "source_class_counts": clean_json(source_summary["source_class"].value_counts().to_dict()),
        "near_source_counts": {
            "sources_with_near": int(near_sources["has_near"].sum()),
            "sources_with_near_mid": int(near_sources["has_near_mid"].sum()),
            "sources_with_near_far": int(near_sources["has_near_far"].sum()),
            "sources_with_near_post_or_control": int(near_sources["has_near_post_or_control"].sum()),
            "near_sources_no_within_source_control": int(
                ((near_sources["has_near_mid"] == False) & (near_sources["has_near_post_or_control"] == False)).sum()
            ),
        },
        "near_source_rows": clean_json(near_source_rows.to_dict("records")),
        "candidate_far_control_sources": clean_json(
            source_summary[source_summary["far_pre_rows"] > 0][
                ["source_stem", "source_class", "far_pre_rows", "h5_available", "cycle_min", "cycle_max"]
            ].to_dict("records")
        ),
        "sampled_same_source_ladder_counts": clean_json(sampled_ladder.get("ladder_source_counts", {})),
        "recommended_design": [
            "Do not claim same-source near-vs-far evidence from current raw coverage: no source with near-pre rows also has far-pre rows.",
            "Use same-source near-vs-mid and near-vs-post/control as local opportunistic checks where available.",
            "Use cross-source far-pre controls only with explicit source/acquisition-class matching or source-residualization.",
            "Acquire or export additional far-pre cycles only if new raw source/movie coverage exists; it is not latent in exampleParticles for current near-pre sources.",
        ],
        "outputs": {
            "source_summary": str(source_path),
            "cycle_table": str(cycle_path),
            "candidate_controls": str(controls_path),
            "summary": str(summary_path),
        },
        "guardrail": "This is a coverage/design audit over the raw particle cycle/source index. It identifies feasible controls and missing same-source ladders; it does not add manual labels, validate particles/fronts, calibrate diffusion, or establish precursor causality.",
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")

    lines = [
        "# Pre-Event Source Lattice Coverage Audit",
        "",
        f"- Cycle rows/sources/HDF5-backed sources: {summary['n_cycle_rows']} / {summary['n_sources']} / {summary['n_sources_with_h5']}",
        f"- Event-relative bins: {summary['event_relative_bin_counts']}",
        f"- Near-source counts: {summary['near_source_counts']}",
        "",
        "## Recommended Design",
        "",
    ]
    lines.extend([f"- {row}" for row in summary["recommended_design"]])
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
