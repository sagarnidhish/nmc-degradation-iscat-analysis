#!/usr/bin/env python3
"""Closed-loop hypothesis ledger for NMC agentic analyses."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
from agentic_utils import markdown_table, output_root, read_json, resolve_root, write_json


def get_nested(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_ledger(root: Path, out_base: Path) -> List[Dict[str, Any]]:
    derived = root / "derived"
    metric = read_json(out_base / "05_agentic_metric_search" / "agentic_metric_search_summary.json")
    tournament = read_json(derived / "agentic_current_hypothesis_tournament" / "agentic_current_hypothesis_tournament_summary.json")
    qc = read_json(derived / "qc_decision_evidence_ledger" / "qc_decision_evidence_ledger_summary.json")
    calibration = read_json(derived / "calibration_claim_risk_register" / "calibration_claim_risk_register_summary.json")

    n_promote = int(metric.get("n_promote_for_followup", 0) or 0)
    n_qc = int(qc.get("n_candidates", 0) or 0)
    possible_accept = int(get_nested(qc, "decision_action_counts", "review_for_possible_accept_first", default=0) or 0)
    top_hyp = tournament.get("top_hypothesis", {})

    rows = [
        {
            "claim_id": "C1_future16_echem_conditioned_video_residuals",
            "claim": "Echem-conditioned optical/video residuals are the strongest longer-horizon weak-label signal.",
            "status": "under_test" if n_promote else "proposed",
            "strongest_support": f"{n_promote} metric-search variants promoted for follow-up; top tournament hypothesis: {top_hyp.get('claim', top_hyp.get('statement', 'unavailable'))}",
            "strongest_counterevidence": "Future16 labels remain weak and source/acquisition imbalance is still present.",
            "required_falsification_test": "Require source-balanced or leave-source performance above matched null after acquisition/context residualization.",
            "next_script": "tier4_agentic_metric_search follow-up candidate from 05_agentic_metric_search",
            "do_not_claim": "Do not claim deployable degradation prediction or mechanism.",
        },
        {
            "claim_id": "C2_future8_context_dominated",
            "claim": "Short-horizon future8 labels are acquisition/context dominated and should not drive mechanisms alone.",
            "status": "supported_guardrail",
            "strongest_support": "Prior context audits and metric penalties flag acquisition/design-context variants as leakage guarded.",
            "strongest_counterevidence": "Some physics-only future8 signals remain above chance in selected audits.",
            "required_falsification_test": "Show future8 video/physics signal survives source-balanced acquisition residualization and shift/null controls.",
            "next_script": "tier4_residualized_future8_video_physics_benchmark.py",
            "do_not_claim": "Do not use future8 AUC as a physical degradation detector without residualized controls.",
        },
        {
            "claim_id": "C3_manual_qc_is_decisive",
            "claim": "Manual ROI/front QC is the decisive feedback signal before accepting front or diffusion-like claims.",
            "status": "blocked_by_qc" if n_qc else "proposed",
            "strongest_support": f"QC decision ledger has {n_qc} candidates, including {possible_accept} possible accept-first rows.",
            "strongest_counterevidence": "No manually accepted front labels are available in the current derived summaries.",
            "required_falsification_test": "Populate manual labels and rerun manual-QC-gated front/echem effects.",
            "next_script": "tier4_manual_qc_gated_front_effects.py",
            "do_not_claim": "Do not report calibrated front/diffusion mechanisms from automatic masks alone.",
        },
        {
            "claim_id": "C4_phase_front_direction_not_calibrated_diffusion",
            "claim": "The defensible physics claim is optical phase/front movement, not calibrated lithium diffusion.",
            "status": "supported_guardrail" if calibration else "proposed",
            "strongest_support": "Calibration risk register and front audits preserve apparent-proxy wording.",
            "strongest_counterevidence": "Some automatic front proxies show coherent directionality and kinetics.",
            "required_falsification_test": "Require pixel-size provenance, timing provenance, stable accepted masks, and manual front labels.",
            "next_script": "tier4_manual_qc_gated_diffusion_bounds.py",
            "do_not_claim": "Do not call apparent front slopes diffusion coefficients without calibration/QC evidence.",
        },
        {
            "claim_id": "C5_agentic_metric_search_as_planner",
            "claim": "Agentic AI is useful here as a benchmark-driven planner and critic over real derived evidence.",
            "status": "implemented",
            "strongest_support": f"Metric search aggregated {metric.get('n_metric_rows', 0)} rows from {metric.get('n_source_tables_present', 0)} metric tables.",
            "strongest_counterevidence": "The workflow is deterministic and does not yet use external literature retrieval or online model-generated code repair.",
            "required_falsification_test": "Show top-ranked agentic follow-up candidates reproduce or improve guarded held-out evidence.",
            "next_script": "agentic_research/05_agentic_metric_search/run_agentic_metric_search.py",
            "do_not_claim": "Do not frame the agentic layer as autonomous discovery; it is an auditable prioritizer.",
        },
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    root = resolve_root(args.root)
    out_base = output_root(root, args.out_dir)
    out = out_base / "06_closed_loop_hypothesis_ledger"
    out.mkdir(parents=True, exist_ok=True)
    rows = build_ledger(root, out_base)
    pd.DataFrame(rows).to_csv(out / "closed_loop_hypothesis_ledger.csv", index=False)
    write_json(out / "closed_loop_hypothesis_ledger_summary.json", {
        "root": str(root),
        "n_claims": len(rows),
        "status_counts": pd.Series([r["status"] for r in rows]).value_counts().to_dict(),
        "claims": rows,
        "guardrail": "The ledger tracks claim status and falsification requirements. It does not convert weak labels or automatic masks into validated mechanisms.",
    })
    md = [
        "# Closed-Loop Hypothesis Ledger",
        "",
        markdown_table(rows, ["claim_id", "status", "claim", "strongest_support", "strongest_counterevidence", "required_falsification_test", "next_script"]),
        "",
        "## Guardrail",
        "",
        "Claims marked supported are supported as computational guardrails or follow-up priorities, not as final battery mechanisms.",
        "",
    ]
    (out / "closed_loop_hypothesis_ledger_report.md").write_text("\n".join(md))
    print(f"[done] wrote hypothesis ledger outputs to {out}")


if __name__ == "__main__":
    main()
