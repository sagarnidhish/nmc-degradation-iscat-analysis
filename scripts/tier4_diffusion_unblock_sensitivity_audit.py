#!/usr/bin/env python3
"""Quantify what would unblock calibrated diffusion claims.

The readiness audit already says "not ready"; this script asks a more
operational question: which blocker classes dominate the current candidate
ledger, and how many candidate rows would become eligible under explicit
what-if scenarios such as manual-QC acceptance or relaxed internal optical-front
gates. It does not relax the actual guardrail or create a diffusion claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import pandas as pd


CANONICAL_BLOCKERS = [
    "HDF5 spatial calibration metadata present",
    "Per-ROI HDF5 timing stability",
    "Radius2 linear-fit quality",
    "q70 positive confidence interval",
    "Automatic/publication diffusion candidates",
    "Manual QC accepted labels",
    "Control-balanced diffusion sanity candidates",
    "Event/control diffusion separability",
]

BLOCKER_ALIASES = {
    "publication-ready gate": "Automatic/publication diffusion candidates",
    "manual QC accepted": "Manual QC accepted labels",
    "selected fit quality": "Radius2 linear-fit quality",
    "radius2 fit quality": "Radius2 linear-fit quality",
    "q70 positive CI": "q70 positive confidence interval",
    "q70 bootstrap positive": "q70 positive confidence interval",
    "HDF5 timing stability": "Per-ROI HDF5 timing stability",
    "selected nonnegative": "Positive front expansion",
    "positive expansion": "Positive front expansion",
}

GLOBAL_BLOCKERS = {
    "HDF5 spatial calibration metadata present",
    "Control-balanced diffusion sanity candidates",
    "Event/control diffusion separability",
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


def split_blockers(text: Any) -> List[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    out: List[str] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        out.append(BLOCKER_ALIASES.get(item, item))
    return out


def canonical_set(row: pd.Series, global_blockers: Set[str]) -> Set[str]:
    blockers = set(split_blockers(row.get("blockers")))
    blockers |= global_blockers
    if not bool(row.get("publication_ready", False)):
        blockers.add("Automatic/publication diffusion candidates")
    return blockers


def scenario_rows(df: pd.DataFrame, removed: Iterable[str], name: str) -> Dict[str, Any]:
    removed_set = set(removed)
    remaining = df["all_blockers"].map(lambda x: sorted(set(x) - removed_set))
    eligible = remaining.map(len).eq(0)
    near = remaining.map(lambda x: len(x) <= 1)
    top = df.assign(remaining_blockers=remaining.map("; ".join), scenario_eligible=eligible).sort_values(
        ["scenario_eligible", "review_priority"], ascending=[False, False]
    )
    return {
        "scenario": name,
        "removed_blockers": sorted(removed_set),
        "n_eligible": int(eligible.sum()),
        "n_one_blocker_remaining": int(near.sum() - eligible.sum()),
        "top_eligible_or_near": top.head(12).to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/diffusion_unblock_sensitivity_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(derived / "diffusion_claim_readiness_audit" / "diffusion_claim_readiness_candidates.csv")
    criteria = pd.read_csv(derived / "diffusion_claim_readiness_audit" / "diffusion_claim_readiness_criteria.csv")

    hard = set(criteria.loc[criteria["severity"].eq("hard_blocker"), "gate"].astype(str))
    global_blockers = hard & GLOBAL_BLOCKERS
    candidates = candidates.copy()
    candidates["all_blockers"] = candidates.apply(lambda r: sorted(canonical_set(r, global_blockers)), axis=1)
    candidates["n_all_blockers"] = candidates["all_blockers"].map(len)
    candidates["blocker_summary"] = candidates["all_blockers"].map(lambda x: "; ".join(x))

    counts = Counter()
    for blockers in candidates["all_blockers"]:
        counts.update(blockers)
    blocker_rows = []
    for blocker, n in counts.most_common():
        crit = criteria.loc[criteria["gate"].eq(blocker)]
        blocker_rows.append({
            "blocker": blocker,
            "n_candidate_rows": int(n),
            "criterion_status": crit["status"].iloc[0] if not crit.empty else "",
            "criterion_severity": crit["severity"].iloc[0] if not crit.empty else "candidate_specific",
            "evidence": crit["evidence"].iloc[0] if not crit.empty else "",
            "next_action": crit["next_action"].iloc[0] if not crit.empty else "",
        })
    blocker_table = pd.DataFrame(blocker_rows)

    removed_spatial_manual = {
        "HDF5 spatial calibration metadata present",
        "Manual QC accepted labels",
    }
    removed_metadata_and_publication_gate = {
        "HDF5 spatial calibration metadata present",
        "Manual QC accepted labels",
        "Automatic/publication diffusion candidates",
    }
    removed_all_external = {
        "HDF5 spatial calibration metadata present",
        "Manual QC accepted labels",
        "Control-balanced diffusion sanity candidates",
        "Event/control diffusion separability",
        "Automatic/publication diffusion candidates",
    }
    removed_internal_liberal = removed_all_external | {
        "Per-ROI HDF5 timing stability",
        "q70 positive confidence interval",
    }
    scenarios = [
        scenario_rows(candidates, [], "current_all_guardrails"),
        scenario_rows(candidates, removed_spatial_manual, "confirm_spatial_calibration_plus_accept_manual_qc_only"),
        scenario_rows(candidates, removed_metadata_and_publication_gate, "metadata_manual_qc_and_publication_gate_rechecked"),
        scenario_rows(candidates, removed_all_external, "external_blockers_cleared_internal_gates_unchanged"),
        scenario_rows(candidates, removed_internal_liberal, "external_blockers_cleared_plus_timing_and_q70_relaxed"),
        scenario_rows(candidates, set(CANONICAL_BLOCKERS) | {"Positive front expansion"}, "all_current_blockers_removed_sanity_upper_bound"),
    ]
    scenario_table = pd.DataFrame([
        {
            "scenario": s["scenario"],
            "removed_blockers": "; ".join(s["removed_blockers"]),
            "n_eligible": s["n_eligible"],
            "n_one_blocker_remaining": s["n_one_blocker_remaining"],
        }
        for s in scenarios
    ])

    review_cols = [
        "candidate_source", "roi_id", "selected_roi_id", "cycleNo", "source_stem", "cohort_role",
        "automatic_physics_consistent", "publication_ready", "gate_count", "median_apparent_D_um2_per_s",
        "selected_diffusion_um2_per_s", "median_radius2_fit_r2", "selected_r2",
        "h5_dt_max_to_median_ratio", "manual_qc_status", "review_priority",
        "n_all_blockers", "blocker_summary",
    ]
    review_queue = candidates[[c for c in review_cols if c in candidates.columns]].sort_values(
        ["n_all_blockers", "review_priority"], ascending=[True, False]
    )

    paths = {
        "blocker_table": out / "diffusion_unblock_blocker_table.csv",
        "scenario_table": out / "diffusion_unblock_scenario_table.csv",
        "review_queue": out / "diffusion_unblock_review_queue.csv",
        "summary": out / "diffusion_unblock_sensitivity_summary.json",
    }
    blocker_table.to_csv(paths["blocker_table"], index=False)
    scenario_table.to_csv(paths["scenario_table"], index=False)
    review_queue.to_csv(paths["review_queue"], index=False)

    top_near = review_queue.head(12).to_dict("records")
    summary = clean_json({
        "overall_status": "diffusion_claim_still_blocked",
        "n_candidate_rows": int(len(candidates)),
        "n_criteria": int(len(criteria)),
        "n_global_hard_blockers_applied": int(len(global_blockers)),
        "global_hard_blockers_applied": sorted(global_blockers),
        "top_blockers": blocker_table.head(12).to_dict("records"),
        "scenario_summary": scenario_table.to_dict("records"),
        "top_nearest_unblock_candidates": top_near,
        "guardrail": "This sensitivity audit removes blockers only in explicit what-if scenarios. It does not change the diffusion readiness status, accept manual labels, relax gates in production, or create calibrated diffusion coefficients.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Diffusion Unblock Sensitivity Audit",
        "",
        "What-if accounting for calibrated diffusion blockers over the current candidate ledger.",
        "",
        f"- Candidate rows: {summary['n_candidate_rows']}",
        f"- Global hard blockers applied to every candidate: {summary['global_hard_blockers_applied']}",
        "",
        "## Blockers",
        "",
    ]
    for row in summary["top_blockers"][:10]:
        lines.append(f"- {row['blocker']}: {row['n_candidate_rows']} candidate rows; status {row.get('criterion_status', '')}")
    lines += ["", "## Scenarios", ""]
    for row in summary["scenario_summary"]:
        lines.append(f"- {row['scenario']}: eligible {row['n_eligible']}; one blocker remaining {row['n_one_blocker_remaining']}")
    lines += ["", "## Nearest Candidates", ""]
    for row in top_near[:8]:
        lines.append(
            f"- {row.get('roi_id')}: blockers {row.get('n_all_blockers')}, priority {row.get('review_priority')}, "
            f"remaining {row.get('blocker_summary')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
