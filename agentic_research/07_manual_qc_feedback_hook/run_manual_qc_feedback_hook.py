#!/usr/bin/env python3
"""Manual-QC feedback hook for lab-in-the-loop NMC photometry analysis."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
from agentic_utils import markdown_table, output_root, read_csv_if_exists, read_json, resolve_root, write_json


def choose_first_existing(paths: List[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()
    root = resolve_root(args.root)
    out = output_root(root, args.out_dir) / "07_manual_qc_feedback_hook"
    out.mkdir(parents=True, exist_ok=True)
    derived = root / "derived"
    repo_local = root / "alek_jiho_nmc_deg" / "derived_local"

    qc_path = choose_first_existing([
        derived / "qc_decision_evidence_ledger" / "qc_decision_evidence_ledger.csv",
        repo_local / "qc_decision_evidence_ledger" / "qc_decision_evidence_ledger.csv",
        root / "derived_local" / "qc_decision_evidence_ledger" / "qc_decision_evidence_ledger.csv",
    ])
    label_path = choose_first_existing([
        derived / "manual_qc_label_workbook" / "manual_qc_label_template.csv",
        repo_local / "manual_qc_label_workbook" / "manual_qc_label_template.csv",
        root / "derived_local" / "manual_qc_label_workbook" / "manual_qc_label_template.csv",
    ])
    gated_summary = read_json(derived / "manual_qc_gated_front_effects" / "manual_qc_gated_front_effects_summary.json")
    qc_df = read_csv_if_exists(qc_path)
    labels = read_csv_if_exists(label_path)

    if not qc_df.empty:
        rank_cols = [c for c in ["decision_priority", "priority_score", "qc_priority_score", "evidence_score"] if c in qc_df.columns]
        if rank_cols:
            qc_df = qc_df.sort_values(rank_cols, ascending=[True] + [False] * (len(rank_cols) - 1))
        review_cols = [c for c in [
            "roi_id", "cycleNo", "decision_action", "manual_status", "visual_asset_path",
            "claim_guardrail", "decision_reason", "review_priority", "priority_score"
        ] if c in qc_df.columns]
        queue = qc_df[review_cols].head(args.top_n).copy()
    else:
        queue = pd.DataFrame()

    label_status_counts: Dict[str, int] = {}
    accepted_labels = 0
    if not labels.empty:
        status_col = next((c for c in ["manual_status", "qc_label", "label", "decision"] if c in labels.columns), "")
        if status_col:
            clean = labels[status_col].fillna("").astype(str).str.strip()
            label_status_counts = clean.value_counts().to_dict()
            accepted_labels = int(clean.str.contains("accept", case=False, regex=False).sum())

    feedback_rules = [
        {
            "gate": "front_or_diffusion_claim",
            "required_feedback": "manual accept label for ROI/front mask plus timing and spatial calibration evidence",
            "current_status": "open" if accepted_labels else "waiting_for_labels",
            "downstream_script": "tier4_manual_qc_gated_front_effects.py",
        },
        {
            "gate": "degradation_mode_training_label",
            "required_feedback": "manual reject/accept/artifact label joined to automatic ROI evidence ledger",
            "current_status": "open" if accepted_labels else "waiting_for_labels",
            "downstream_script": "tier4_qc_decision_evidence_ledger.py",
        },
        {
            "gate": "agentic_hypothesis_status_update",
            "required_feedback": "ledger status update from proposed/blocked_by_qc to supported/weakened after manual labels",
            "current_status": "ready_for_update",
            "downstream_script": "agentic_research/06_closed_loop_hypothesis_ledger/run_hypothesis_ledger.py",
        },
    ]

    queue.to_csv(out / "manual_qc_feedback_queue.csv", index=False)
    pd.DataFrame(feedback_rules).to_csv(out / "manual_qc_feedback_rules.csv", index=False)
    summary = {
        "root": str(root),
        "qc_ledger_path": str(qc_path),
        "manual_label_template_path": str(label_path),
        "n_qc_rows": int(len(qc_df)),
        "n_queue_rows": int(len(queue)),
        "label_status_counts": label_status_counts,
        "accepted_label_count": accepted_labels,
        "manual_qc_gated_summary_present": bool(gated_summary),
        "feedback_state": "labels_available" if accepted_labels else "manual_labels_needed",
        "guardrail": "This hook defines how manual QC should update downstream AI/physics claims. It does not fabricate labels.",
    }
    write_json(out / "manual_qc_feedback_hook_summary.json", summary)
    md = [
        "# Manual-QC Feedback Hook",
        "",
        f"Feedback state: `{summary['feedback_state']}`",
        "",
        "## Feedback Rules",
        "",
        markdown_table(feedback_rules, ["gate", "required_feedback", "current_status", "downstream_script"]),
        "",
        "## Review Queue",
        "",
        markdown_table(queue.to_dict("records"), list(queue.columns)[:8]) if not queue.empty else "_No QC queue rows found._",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
        "",
    ]
    (out / "manual_qc_feedback_hook_report.md").write_text("\n".join(md))
    print(f"[done] wrote manual-QC feedback hook outputs to {out}")


if __name__ == "__main__":
    main()
