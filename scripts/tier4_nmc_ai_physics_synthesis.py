#!/usr/bin/env python3
"""Synthesize NMC photometry AI/physics experiment outputs.

This script turns the distributed derived outputs into a compact, auditable
project-level report. It does not rerun the expensive image/video analyses.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"_missing": str(path)}
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (abs(value) >= 1000 or (0 < abs(value) < 0.001)):
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}"
    return str(value)


def evidence_row(requirement: str, status: str, evidence: str, limitation: str, next_step: str) -> Dict[str, str]:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "limitation": limitation,
        "next_step": next_step,
    }


def first_summary(summary: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = summary.get(key, default)
    return value


def top_items(items: Iterable[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    return list(items)[:n]


def write_markdown(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/nmc_ai_physics_synthesis")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollout = read_json(derived / "multi_cycle_roi_rollout_baselines" / "roi_rollout_baseline_summary.json")
    mobility = read_json(derived / "multi_cycle_rollout_mobility_coupling" / "multi_cycle_rollout_mobility_coupling_summary.json")
    echem = read_json(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_coupling_summary.json")
    predictor = read_json(derived / "multi_cycle_roi_event_predictor" / "multi_cycle_roi_event_predictor_summary.json")
    calibration = read_json(derived / "front_roi_calibration_qc" / "front_roi_calibration_qc_summary.json")
    multicycle = read_json(derived / "multi_cycle_roi_analysis" / "multi_cycle_roi_analysis_summary.json")
    event_sync = read_json(derived / "event_synchrony" / "event_synchrony_summary.json")
    front_mobility = read_json(derived / "multi_cycle_roi_mobility" / "multi_cycle_roi_mobility_summary.json")
    modes = read_json(derived / "roi_joint_physics_degradation_modes" / "roi_joint_physics_degradation_modes_summary.json")

    rollout_cycle = read_csv(derived / "multi_cycle_roi_rollout_baselines" / "roi_rollout_cycle_method_summary.csv")
    echem_corr = read_csv(derived / "multi_cycle_roi_echem_coupling" / "roi_echem_spearman_correlations.csv")
    within_ref = read_csv(derived / "multi_cycle_roi_echem_coupling" / "within_reference_event_control_tests.csv")
    predictor_metrics = read_csv(derived / "multi_cycle_roi_event_predictor" / "leave_event_cycle_out_metrics.csv")
    calibration_table = read_csv(derived / "front_roi_calibration_qc" / "front_roi_qc_calibration.csv")
    ranked_mobility = read_csv(derived / "multi_cycle_rollout_mobility_coupling" / "multi_cycle_rollout_mobility_ranked.csv")

    n_roi = int(first_summary(echem, "n_joined_roi", first_summary(predictor, "n_roi", 0)) or 0)
    n_cycles = int(first_summary(echem, "n_cycles", 0) or 0)
    n_event_refs = int(first_summary(echem, "n_event_reference_cycles", 0) or 0)

    persistence_best = False
    if not rollout_cycle.empty and {"cycleNo", "method", "mse_mean"}.issubset(rollout_cycle.columns):
        winners = (
            rollout_cycle.sort_values("mse_mean")
            .groupby("cycleNo", as_index=False)
            .first()
        )
        persistence_best = bool((winners["method"] == "persistence").all())

    strict_rf = first_summary(predictor, "model_summary", {}).get("physics_no_selection_qc:random_forest", {})
    strict_logistic = first_summary(predictor, "model_summary", {}).get("physics_no_selection_qc:logistic_l2", {})
    all_rf = first_summary(predictor, "model_summary", {}).get("all_physics_plus_qc:random_forest", {})

    best_within = within_ref.head(8).to_dict("records") if not within_ref.empty else []
    best_echem_corr = echem_corr.head(8).to_dict("records") if not echem_corr.empty else []
    top_roi_cols = [
        "roi_id",
        "cohort_role",
        "cycleNo",
        "event_reference_cycle",
        "rollout_mobility_difficulty_score",
        "latent_path_length",
        "latent_net_displacement",
        "first_last_corr",
        "cumulative_abs_first_last",
        "dmd_mse_ratio_vs_persistence",
        "persistence_mse",
        "high_fraction_slope_per_s",
        "degradation_mode_hypothesis",
    ]
    if not ranked_mobility.empty:
        top_rois = ranked_mobility[[c for c in top_roi_cols if c in ranked_mobility.columns]].head(10).to_dict("records")
    else:
        top_rois = []

    qc_pending = 0
    if not calibration_table.empty and "manual_qc_status" in calibration_table.columns:
        qc_pending = int((calibration_table["manual_qc_status"] == "pending").sum())

    audit = [
        evidence_row(
            "Implement paper-inspired agentic workflows in separate Isambard folders",
            "implemented",
            "agentic_research outputs plus derived tier1/tier2/tier3 experiment folders were created on Isambard and compact outputs synced locally.",
            "The synthesis script summarizes the outputs but does not rerun the original literature analysis.",
            "Keep the agentic folders as provenance and use this synthesis as the project-level index.",
        ),
        evidence_row(
            "Focus on Alek_Jiho NMC degradation dataset on Isambard",
            "implemented",
            f"Synthesis reads Isambard derived directory with {n_roi} ROI rows, {n_cycles} cycles, and {n_event_refs} event-reference cycles.",
            "The current multi-cycle ROI cohort is selected around event/reference cycles, not every raw video in the full dataset.",
            "Expand ROI extraction to all cycles after manual QC stabilizes particle identity selection.",
        ),
        evidence_row(
            "Next-frame prediction and rollout",
            "implemented_with_guardrail",
            f"Persistence, velocity, low-rank DMD, PCA latent trajectories, PCA-ridge, and residual-CNN guardrails were run. Persistence is best across cycles: {persistence_best}.",
            "Learned/full rollout models do not yet beat persistence robustly; use residuals and latent paths as descriptors rather than claiming superior prediction.",
            "Train cycle-conditioned probabilistic video models only after growing the ROI set and validating particle masks.",
        ),
        evidence_row(
            "Track phase-boundary movement",
            "implemented_as_proxy",
            f"Front/phase mobility descriptors and selected-front tracking exist; calibration table has {len(calibration_table)} ROI rows.",
            "Front masks are automatic; all selected calibrated front ROIs remain manual-QC pending.",
            "Manually review selected crop previews and lock accepted/rejected masks before publication-scale interpretation.",
        ),
        evidence_row(
            "Extract diffusion coefficients",
            "partial_proxy_only",
            f"Provisional 0.096 um/px calibration and apparent diffusion proxies were computed for {len(calibration_table)} front ROIs.",
            "The values are apparent optical-front contraction/expansion proxies, not validated diffusion coefficients.",
            "Confirm microscope calibration/timebase and validate front geometry before converting to mechanistic diffusion coefficients.",
        ),
        evidence_row(
            "Identify degradation modes",
            "implemented_as_hypothesis_ranking",
            "Joint physics/rollout/echem degradation mode tables and multi-cycle ROI mobility rankings exist.",
            "Modes are unsupervised/automatic and tied to the selected ROI cohort.",
            "Use the top-ranked ROI list for manual labeling, then refit supervised or semi-supervised degradation-mode models.",
        ),
        evidence_row(
            "Correlate degradation with cycles, particle regions, and echem/protocol context",
            "implemented_with_guardrail",
            "Multi-cycle ROI echem coupling found strong frame-count/protocol correlations and within-reference event/control optical shifts.",
            "Protocol/frame-count confounding is strong; correlations are not causal physics.",
            "Use echem/protocol covariates in downstream models and avoid raw detector claims without conditioning.",
        ),
        evidence_row(
            "Keep objectives and observations updated",
            "implemented",
            "OBJECTIVES_OBSERVATIONS.md contains chronological experiment summaries and guardrails.",
            "It is long and narrative; the new synthesis is the compact index.",
            "Append this tier4 synthesis result to the observations file after each major rerun.",
        ),
        evidence_row(
            "Keep GitHub updated",
            "local_commits_ready_remote_unverified",
            "Local git commits exist through the latest echem coupling and this synthesis can be committed.",
            "Remote push must be verified separately because prior pushes were blocked by approval/network state.",
            "Attempt push after committing this synthesis and record the actual result.",
        ),
    ]

    audit_df = pd.DataFrame(audit)
    audit_path = out / "completion_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    report_lines = [
        "# NMC AI Physics Synthesis",
        "",
        "## Scope",
        "",
        "This report consolidates the Alek_Jiho NMC charge/discharge photometry analyses into one auditable view. It is generated from derived outputs on Isambard and should be treated as a synthesis of computational evidence, not as a manual curation substitute.",
        "",
        "## Current Evidence Base",
        "",
        f"- Multi-cycle ROI/echem rows: {n_roi}",
        f"- Distinct cycles in coupled ROI/echem table: {n_cycles}",
        f"- Event-reference cycles: {n_event_refs}",
        f"- Calibrated front-QC ROI rows: {len(calibration_table)}",
        f"- Manual-QC pending front ROIs: {qc_pending}",
        "",
        "## Main Findings",
        "",
        "- Persistence is the strongest raw next-frame baseline; DMD/velocity/learned residual experiments are most useful as residual and latent descriptors.",
        "- ROI event/control optical differences survive event-reference-cycle centering, especially cumulative normalized change, first-last decorrelation, latent net displacement, high-fraction growth, and ROI mean trend.",
        "- Frame count and protocol-block position strongly couple to ROI dynamics, so echem/protocol context must be a model covariate and a guardrail.",
        "- Cycles 86 and 116 remain the strongest synchronized event-timing regimes; cycles 60 and 156 provide stronger single-particle morphology/latent-movement examples.",
        "- Apparent front tracking currently indicates optical-front contraction/loss more than clean expanding diffusion fronts.",
        "",
        "## Model Readout",
        "",
        f"- Strict no-selection-QC random forest: ROC-AUC {fmt(strict_rf.get('mean_roc_auc'))}, balanced accuracy {fmt(strict_rf.get('mean_balanced_accuracy'))}.",
        f"- Strict no-selection-QC logistic: ROC-AUC {fmt(strict_logistic.get('mean_roc_auc'))}, balanced accuracy {fmt(strict_logistic.get('mean_balanced_accuracy'))}.",
        f"- All physics plus QC random forest: ROC-AUC {fmt(all_rf.get('mean_roc_auc'))}, balanced accuracy {fmt(all_rf.get('mean_balanced_accuracy'))}.",
        "",
        "Interpretation: the stricter model is above random but not deployable. QC/acquisition features improve apparent performance and should be treated as leakage-sensitive guardrails.",
        "",
        "## Top Within-Reference Optical Separations",
        "",
    ]
    for row in best_within:
        report_lines.append(
            f"- {row.get('feature')}: event-control {fmt(row.get('event_minus_control'))}, p={fmt(row.get('p_value'))}"
        )

    report_lines += [
        "",
        "## Top ROI/Echem Or Protocol Couplings",
        "",
    ]
    for row in best_echem_corr:
        report_lines.append(
            f"- {row.get('echem_feature')} vs {row.get('roi_feature')}: rho={fmt(row.get('rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )

    report_lines += [
        "",
        "## Highest-Priority ROI Candidates",
        "",
    ]
    for row in top_rois:
        report_lines.append(
            f"- {row.get('roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): rollout/mobility score {fmt(row.get('rollout_mobility_difficulty_score'))}, latent path {fmt(row.get('latent_path_length'))}, first-last corr {fmt(row.get('first_last_corr'))}"
        )

    report_lines += [
        "",
        "## Completion Audit",
        "",
    ]
    for row in audit:
        report_lines.append(f"- {row['requirement']}: {row['status']}. Evidence: {row['evidence']} Limitation: {row['limitation']}")

    report_lines += [
        "",
        "## Recommended Next Experiments",
        "",
        "1. Manual QC the selected front/particle ROI previews and update the manifest with accepted/rejected labels.",
        "2. Expand the ROI cohort across more cycles after QC to reduce event-reference and protocol confounding.",
        "3. Fit echem/protocol-conditioned rollout and event-ranking models, reporting persistence-normalized residuals and uncertainty coverage.",
        "4. Recompute apparent diffusion/front-motion proxies only on QC-accepted fronts with confirmed spatial/time calibration.",
        "5. Convert the top ROI candidates into a labeled degradation-mode benchmark for future self-supervised video models.",
        "",
        "## Guardrail",
        "",
        "The current outputs support a physics-extraction workflow and ranked hypotheses. They do not yet support a claim of calibrated diffusion coefficients or a deployable automated degradation detector.",
    ]

    report_path = out / "NMC_AI_PHYSICS_SYNTHESIS.md"
    write_markdown(report_path, report_lines)

    summary = {
        "derived_dir": str(derived),
        "n_joined_roi": n_roi,
        "n_cycles": n_cycles,
        "n_event_reference_cycles": n_event_refs,
        "n_calibration_rows": int(len(calibration_table)),
        "n_manual_qc_pending": qc_pending,
        "persistence_best_all_cycles": persistence_best,
        "strict_rf_mean_roc_auc": strict_rf.get("mean_roc_auc"),
        "strict_rf_mean_balanced_accuracy": strict_rf.get("mean_balanced_accuracy"),
        "strict_logistic_mean_roc_auc": strict_logistic.get("mean_roc_auc"),
        "all_qc_rf_mean_roc_auc": all_rf.get("mean_roc_auc"),
        "top_within_reference_tests": best_within,
        "top_roi_echem_correlations": best_echem_corr,
        "top_ranked_roi_candidates": top_rois,
        "requirement_audit": audit,
        "outputs": {
            "report": str(report_path),
            "audit": str(audit_path),
            "summary": str(out / "nmc_ai_physics_synthesis_summary.json"),
        },
    }
    with (out / "nmc_ai_physics_synthesis_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    readme = [
        "# NMC AI Physics Synthesis",
        "",
        "Generated project-level synthesis of the Alek_Jiho NMC photometry AI/physics analyses.",
        "",
        "Files:",
        "- `NMC_AI_PHYSICS_SYNTHESIS.md`: compact narrative report.",
        "- `completion_audit.csv`: requirement-by-requirement status table.",
        "- `nmc_ai_physics_synthesis_summary.json`: machine-readable summary.",
    ]
    write_markdown(out / "README.md", readme)


if __name__ == "__main__":
    main()
