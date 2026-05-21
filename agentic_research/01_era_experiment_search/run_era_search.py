#!/usr/bin/env python3
"""ERA-style metric-driven experiment search for Alek/Jiho NMC degradation.

This script does not call an external LLM. It encodes the current scientific
objective as a quantitative, auditable ranking over computational experiments.
Each candidate gets scored by current evidence, expected physics yield, control
coverage, cost, and readiness of required inputs.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
from agentic_utils import markdown_table, output_root, read_csv_if_exists, read_json, resolve_root, summarize_available_artifacts, write_json


def candidate_experiments() -> List[Dict[str, Any]]:
    return [
        {
            "id": "E1_frame_count_matched_event_artifact_control",
            "question": "Are synchronized events still enriched after matching cycles by frame count and protocol block?",
            "script": "new: tier1_frame_count_matched_event_controls.py",
            "inputs": ["event_cycle_table", "event_training"],
            "physics_yield": 8,
            "control_strength": 10,
            "compute_cost": 1,
            "priority_reason": "Separates physical degradation from short-cycle imaging/protocol artifact.",
        },
        {
            "id": "E2_protocol_local_echem_window_scan",
            "question": "Do cycles 86/116 show local dV/dt, dQ/dV, current, or capacity anomalies compared with neighboring cycles?",
            "script": "new: tier1_protocol_local_window_scan.py",
            "inputs": ["event_cycle_table", "echem_cycle_summary"],
            "physics_yield": 9,
            "control_strength": 8,
            "compute_cost": 2,
            "priority_reason": "Directly tests electrochemical context around synchronized optical events.",
        },
        {
            "id": "E3_particle_event_hazard_calibration",
            "question": "Can grouped hazard models forecast abrupt particle events under leave-one-particle and blocked-cycle validation?",
            "script": "extend: tier2_crack_hazard.py",
            "inputs": ["event_training", "event_cycle_table"],
            "physics_yield": 7,
            "control_strength": 8,
            "compute_cost": 2,
            "priority_reason": "Turns weak precursor signal into a calibrated degradation-risk model.",
        },
        {
            "id": "E4_degradation_mode_clustering",
            "question": "Do particles/cycles cluster into gradual drift, abrupt drop, recovery, and stable modes?",
            "script": "new: tier2_degradation_mode_clustering.py",
            "inputs": ["event_training", "event_cycle_table"],
            "physics_yield": 8,
            "control_strength": 6,
            "compute_cost": 2,
            "priority_reason": "Extracts interpretable degradation modes rather than binary event labels only.",
        },
        {
            "id": "E5_roi_event_visual_qc_manifest",
            "question": "Do raw ROI frames around event cycles show real particle changes rather than threshold/drift artifacts?",
            "script": "new: tier1_event_roi_qc_manifest.py",
            "inputs": ["particle_events", "event_cycle_table"],
            "physics_yield": 9,
            "control_strength": 9,
            "compute_cost": 4,
            "priority_reason": "Required before claiming physical abrupt degradation from optical traces.",
        },
        {
            "id": "E6_observable_sequence_forecast",
            "question": "Can past particle optical observables plus future protocol/echem context predict future degradation observables?",
            "script": "extend: tier3_multimodal_transformer.py",
            "inputs": ["event_training", "event_cycle_table", "echem_cycle_summary"],
            "physics_yield": 7,
            "control_strength": 7,
            "compute_cost": 5,
            "priority_reason": "Bridges the NMC degradation data and the SP observable-forecasting strategy.",
        },
        {
            "id": "E7_front_proxy_readiness_scan",
            "question": "Which particles/cycles are suitable for phase-front/proxy tracking and apparent transport fits?",
            "script": "new: tier2_front_proxy_readiness_scan.py",
            "inputs": ["event_training", "recovery_qc"],
            "physics_yield": 6,
            "control_strength": 7,
            "compute_cost": 3,
            "priority_reason": "Prevents over-interpreting diffusion coefficients before front-quality validation.",
        },
    ]


def readiness_score(candidate: Dict[str, Any], artifacts: Dict[str, Any]) -> float:
    available = artifacts["available"]
    inputs = candidate["inputs"]
    if not inputs:
        return 0.5
    return sum(1 for name in inputs if available.get(name, False)) / len(inputs)


def evidence_bonus(root: Path, candidate: Dict[str, Any]) -> float:
    derived = root / "derived"
    bonus = 0.0
    sync = read_json(derived / "event_synchrony" / "event_synchrony_summary.json")
    if candidate["id"].startswith("E1") and sync:
        bonus += 1.0
    protocol = read_json(derived / "event_protocol_context" / "event_protocol_context_summary.json")
    if candidate["id"].startswith("E2") and protocol:
        bonus += 1.0
    baselines = read_csv_if_exists(derived / "particle_event_targets" / "particle_event_feature_baselines.csv")
    if candidate["id"].startswith("E3") and not baselines.empty:
        best_f1 = baselines.get("f1")
        if best_f1 is not None and best_f1.notna().any():
            bonus += min(1.5, float(best_f1.max()) * 3.0)
    return bonus


def rank_experiments(root: Path) -> List[Dict[str, Any]]:
    artifacts = summarize_available_artifacts(root)
    rows = []
    for cand in candidate_experiments():
        ready = readiness_score(cand, artifacts)
        score = (
            2.2 * cand["physics_yield"]
            + 2.0 * cand["control_strength"]
            + 6.0 * ready
            + evidence_bonus(root, cand)
            - 1.1 * cand["compute_cost"]
        )
        row = dict(cand)
        row["readiness"] = round(ready, 3)
        row["era_score"] = round(score, 3)
        row["status"] = "ready" if ready >= 0.99 else "needs_inputs"
        rows.append(row)
    return sorted(rows, key=lambda r: r["era_score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    root = resolve_root(args.root)
    out = output_root(root, args.out_dir) / "01_era_experiment_search"
    out.mkdir(parents=True, exist_ok=True)

    artifacts = summarize_available_artifacts(root)
    ranked = rank_experiments(root)
    write_json(out / "artifact_inventory.json", artifacts)
    write_json(out / "era_ranked_experiments.json", {"root": str(root), "ranked_experiments": ranked})

    pd_rows = ranked
    import pandas as pd

    pd.DataFrame(pd_rows).to_csv(out / "era_ranked_experiments.csv", index=False)
    md = [
        "# ERA-Style Experiment Search",
        "",
        f"Root: `{root}`",
        "",
        "## Top Ranked Experiments",
        "",
        markdown_table(ranked, ["id", "status", "readiness", "era_score", "question", "priority_reason"]),
        "",
        "## Interpretation",
        "",
        "The highest-ranked ready experiments should be executed first because they test artifact controls and protocol-local physics before adding heavier sequence models.",
        "",
    ]
    (out / "era_experiment_search_report.md").write_text("\n".join(md))
    print(f"[done] wrote ERA experiment search outputs to {out}")


if __name__ == "__main__":
    main()
