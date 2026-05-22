#!/usr/bin/env python3
"""Update diffusion-readiness gates after the cycle-78 remeasurement packet.

The cycle78 remeasurement clears the target's automatic q70 positive-CI gate,
but that does not make calibrated diffusion publication-ready. This audit
propagates the new target-specific evidence through the existing diffusion
readiness ledgers and records which blockers remain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


TARGET = "cycle78_rank22_obj2"
Q70_BLOCKER = "q70 positive CI"
Q70_GLOBAL_BLOCKER = "q70 positive confidence interval"


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


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def split_blockers(text: Any) -> List[str]:
    if text is None or (isinstance(text, float) and not np.isfinite(text)):
        return []
    return [x.strip() for x in str(text).split(";") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/post_remeasurement_diffusion_gate_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    claim_summary = read_json(derived / "diffusion_claim_readiness_audit" / "diffusion_claim_readiness_summary.json")
    unblock_summary = read_json(derived / "diffusion_unblock_sensitivity_audit" / "diffusion_unblock_sensitivity_summary.json")
    current_claim = read_json(derived / "current_claim_readiness_matrix" / "current_claim_readiness_summary.json")
    cycle78 = read_json(derived / "cycle78_diffusion_remeasurement_audit" / "cycle78_diffusion_remeasurement_summary.json")
    targeted = read_json(derived / "targeted_diffusion_blocker_diagnostic" / "targeted_diffusion_blocker_diagnostic_summary.json")

    candidates_path = derived / "diffusion_claim_readiness_audit" / "diffusion_claim_readiness_candidates.csv"
    criteria_path = derived / "diffusion_claim_readiness_audit" / "diffusion_claim_readiness_criteria.csv"
    candidates = pd.read_csv(candidates_path)
    criteria = pd.read_csv(criteria_path)

    cand = candidates.loc[candidates["roi_id"].astype(str).eq(TARGET)]
    if cand.empty:
        raise ValueError(f"{TARGET} missing from diffusion readiness candidates")
    cand_row = cand.iloc[0].to_dict()
    target = cycle78.get("target_summary", {}) or {}
    q70_pass = bool(target.get("default_q70_positive_ci", False))

    pre_blockers = split_blockers(cand_row.get("blockers"))
    post_candidate_blockers = [b for b in pre_blockers if b != Q70_BLOCKER]
    if not q70_pass and Q70_BLOCKER not in post_candidate_blockers:
        post_candidate_blockers.append(Q70_BLOCKER)
    if "publication-ready gate" not in post_candidate_blockers:
        post_candidate_blockers.append("publication-ready gate")

    global_hard_blockers = list(claim_summary.get("hard_blockers", []))
    remaining_global = []
    for blocker in global_hard_blockers:
        if blocker == Q70_GLOBAL_BLOCKER:
            remaining_global.append("q70 positive confidence interval (globally still fail; target-specific packet now passes)")
        else:
            remaining_global.append(blocker)

    gate_rows = []
    for _, row in criteria.iterrows():
        gate = str(row["gate"])
        post_status = row["status"]
        post_detail = row.get("blocking_detail", "")
        if gate == Q70_GLOBAL_BLOCKER:
            post_status = "target_specific_pass_global_not_rerun"
            post_detail = (
                f"{TARGET} default q70 CI is positive in the remeasurement packet, "
                "but the all-ROI diffusion-readiness gate has not been globally rerun."
            )
        if gate == "Automatic/publication diffusion candidates":
            post_status = "partial_target_specific_update"
            post_detail = (
                f"{TARGET} now clears q70 positive CI automatically, but publication-ready remains false "
                "because manual QC, spatial calibration, and independent sanity gates remain unresolved."
            )
        gate_rows.append({
            "gate": gate,
            "pre_status": row["status"],
            "post_status": post_status,
            "severity": row.get("severity"),
            "pre_evidence": row.get("evidence"),
            "post_detail": post_detail,
            "next_action": row.get("next_action"),
        })
    gate_table = pd.DataFrame(gate_rows)

    candidate_update = pd.DataFrame([{
        "roi_id": TARGET,
        "cycleNo": cand_row.get("cycleNo"),
        "source_stem": cand_row.get("source_stem"),
        "pre_blockers": "; ".join(pre_blockers),
        "post_candidate_blockers": "; ".join(post_candidate_blockers),
        "q70_candidate_blocker_removed": bool(q70_pass and Q70_BLOCKER in pre_blockers),
        "target_status": cycle78.get("target_status"),
        "default_q70_D_um2_per_s": target.get("default_q70_D_um2_per_s"),
        "default_q70_D_p05_um2_per_s": target.get("default_q70_D_p05_um2_per_s"),
        "default_q70_D_p95_um2_per_s": target.get("default_q70_D_p95_um2_per_s"),
        "default_q70_radius2_r2": target.get("default_q70_radius2_r2"),
        "max_radius2_r2": target.get("max_radius2_r2"),
        "median_D_um2_per_s": target.get("median_D_um2_per_s"),
        "positive_D_fraction": target.get("positive_D_fraction"),
        "positive_ci_fraction": target.get("positive_ci_fraction"),
        "same_source_default_q70_D_percentile": target.get("default_q70_D_um2_per_s_context_percentile"),
        "same_source_positive_ci_fraction_percentile": target.get("positive_ci_fraction_context_percentile"),
        "publication_ready_after_remeasurement": False,
    }])

    remaining_publication_blockers = [
        "manual front/QC labels not accepted",
        "raw spatial calibration metadata still not located in HDF5/microscope metadata",
        "control-balanced diffusion sanity remains negative",
        "event/control diffusion separability remains weak",
        "publication-ready diffusion gate has not been rerun with accepted labels",
    ]
    if "Per-ROI HDF5 timing stability" in global_hard_blockers:
        remaining_publication_blockers.append("global per-ROI timing-stability gate remains failed")

    scenario_rows = [
        {
            "scenario": "pre_remeasurement_readiness",
            "candidate_q70_status": "blocked",
            "publication_ready": False,
            "remaining_candidate_blockers": "; ".join(pre_blockers),
            "remaining_global_blockers": "; ".join(global_hard_blockers),
        },
        {
            "scenario": "post_cycle78_q70_remeasurement",
            "candidate_q70_status": "passed_for_target_packet" if q70_pass else "still_blocked",
            "publication_ready": False,
            "remaining_candidate_blockers": "; ".join(post_candidate_blockers),
            "remaining_global_blockers": "; ".join(remaining_global),
        },
        {
            "scenario": "next_manual_qc_and_metadata_handoff",
            "candidate_q70_status": "passed_for_target_packet" if q70_pass else "still_blocked",
            "publication_ready": False,
            "remaining_candidate_blockers": "publication-ready gate",
            "remaining_global_blockers": "; ".join(remaining_publication_blockers),
        },
    ]
    scenario_table = pd.DataFrame(scenario_rows)

    paths = {
        "candidate_update": out / "post_remeasurement_diffusion_candidate_update.csv",
        "gate_table": out / "post_remeasurement_diffusion_gate_table.csv",
        "scenario_table": out / "post_remeasurement_diffusion_scenario_table.csv",
        "summary": out / "post_remeasurement_diffusion_gate_summary.json",
    }
    candidate_update.to_csv(paths["candidate_update"], index=False)
    gate_table.to_csv(paths["gate_table"], index=False)
    scenario_table.to_csv(paths["scenario_table"], index=False)

    summary = {
        "overall_status": "diffusion_claim_still_blocked_after_target_q70_update",
        "target_roi_id": TARGET,
        "target_q70_status": "passed_for_target_packet" if q70_pass else "still_blocked",
        "candidate_q70_blocker_removed": bool(q70_pass and Q70_BLOCKER in pre_blockers),
        "publication_ready_after_remeasurement": False,
        "pre_candidate_blockers": pre_blockers,
        "post_candidate_blockers": post_candidate_blockers,
        "remaining_publication_blockers": remaining_publication_blockers,
        "target_remeasurement_evidence": clean_json(target),
        "claim_readiness_overall_status": claim_summary.get("overall_status"),
        "claim_readiness_publication_ready_candidates": claim_summary.get("n_publication_ready_candidates"),
        "current_claim_diffusion_status": (current_claim.get("top_negative_evidence", {}) or {}).get("diffusion_status"),
        "unblock_overall_status": unblock_summary.get("overall_status"),
        "targeted_diffusion_nearest_candidate": targeted.get("nearest_diffusion_candidate", {}),
        "scenario_table": clean_json(scenario_rows),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": (
            "This audit propagates a target-specific automatic q70 CI update through existing readiness gates. "
            "It does not rerun global diffusion readiness, accept manual labels, verify raw calibration metadata, or create publication-ready diffusion coefficients."
        ),
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    readme = [
        "# Post-Remeasurement Diffusion Gate Audit",
        "",
        f"- Target: `{TARGET}`",
        f"- Target q70 status: {summary['target_q70_status']}",
        f"- Candidate q70 blocker removed: {summary['candidate_q70_blocker_removed']}",
        f"- Publication-ready after remeasurement: {summary['publication_ready_after_remeasurement']}",
        "",
        "Outputs:",
        *[f"- `{path.name}`" for path in paths.values()],
        "",
        "Guardrail:",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(readme).rstrip() + "\n")
    summary["outputs"]["readme"] = str(out / "README.md")
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
