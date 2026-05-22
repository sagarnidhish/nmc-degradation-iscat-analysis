#!/usr/bin/env python3
"""Build the current NMC physics claim-readiness matrix.

This matrix consolidates the latest source-balanced transport/fusion/falsification,
expanded-cohort generalization, rollout, manual-QC, residualized warning, and
diffusion-readiness audits into claim-level wording decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def fmt(value: Any, digits: int = 3) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def row(
    claim_id: str,
    claim: str,
    readiness: str,
    evidence_strength: str,
    evidence_summary: str,
    blockers: str,
    next_actions: str,
    allowed_wording: str,
    forbidden_wording: str,
    primary_outputs: str,
) -> Dict[str, str]:
    return {
        "claim_id": claim_id,
        "claim": claim,
        "readiness": readiness,
        "evidence_strength": evidence_strength,
        "evidence_summary": evidence_summary,
        "blockers_or_limitations": blockers,
        "next_actions": next_actions,
        "allowed_wording": allowed_wording,
        "forbidden_wording": forbidden_wording,
        "primary_outputs": primary_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/current_claim_readiness_matrix")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fusion = read_json(derived / "source_balanced_pre_event_transport_kinetic_fusion_audit" / "source_balanced_pre_event_transport_kinetic_fusion_summary.json")
    dossier = read_json(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_summary.json")
    falsification = read_json(derived / "source_balanced_transport_mechanism_falsification_audit" / "source_balanced_transport_mechanism_falsification_summary.json")
    expansion = read_json(derived / "source_balanced_expansion_transport_front_audit" / "source_balanced_expansion_transport_front_summary.json")
    diffusion = read_json(derived / "diffusion_claim_readiness_audit" / "diffusion_claim_readiness_summary.json")
    rollout = read_json(derived / "source_balanced_pre_event_masked_rollout_benchmark" / "source_balanced_pre_event_masked_rollout_summary.json")
    residualized = read_json(derived / "residualized_future8_video_physics_benchmark" / "residualized_future8_video_physics_summary.json")
    blind = read_json(derived / "source_balanced_pre_event_manual_qc_blind_workbook" / "source_balanced_pre_event_manual_qc_blind_summary.json")

    fusion_near_any = next((r for r in fusion.get("best_event_tests_by_target", []) if r.get("target") == "near_vs_any_non_near"), {})
    fusion_near_post = next((r for r in fusion.get("best_event_tests_by_target", []) if r.get("target") == "near_vs_post_control"), {})
    fals_event = falsification.get("lead_event_tests_for_transport_mechanism_score", [{}])[0]
    fals_pair = falsification.get("lead_pair_tests_for_transport_mechanism_score", [{}])[0]
    fals_source = falsification.get("lead_source_test_for_transport_mechanism_score", {})
    expansion_future8 = next((r for r in expansion.get("best_feature_tests_by_target", []) if r.get("target") == "future_any_drop_within_8cycles"), {})
    expansion_future16 = next((r for r in expansion.get("best_feature_tests_by_target", []) if r.get("target") == "future_any_drop_within_16cycles"), {})
    expansion_residual = next((r for r in expansion.get("top_feature_tests", []) if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature") == "divergence_q90" and r.get("transform") == "source_residual"), {})
    rollout_best = rollout.get("best_method_by_median_particle_gain", {})
    rollout_event = rollout.get("top_event_tests", [{}])[0]
    residual_decision = residualized.get("decision", {})

    rows: List[Dict[str, str]] = []
    rows.append(row(
        "C01_event_local_particle_readiness_ranker",
        "Automatic particle-region descriptors can rank near-pre-event ROIs for manual review.",
        "supported_for_review_ranking",
        "strong_internal_source_aware",
        (
            f"Fusion near-vs-any uses {fusion_near_any.get('score')} with AUC {fmt(fusion_near_any.get('oriented_auc'))}, "
            f"AP {fmt(fusion_near_any.get('average_precision'))}, source-stratified p {fmt(fusion_near_any.get('source_stratified_permutation_p'))}; "
            f"mechanism falsification near-vs-any AUC {fmt(fals_event.get('auc'))}, AP {fmt(fals_event.get('average_precision'))}, p {fmt(fals_event.get('source_stratified_permutation_p'))}; "
            f"same-source nearest controls median delta {fmt(fals_pair.get('median_delta_near_minus_control'))}, positive fraction {fmt(fals_pair.get('positive_delta_fraction'))}, sign-flip p {fmt(fals_pair.get('sign_flip_p'))}."
        ),
        f"Manual labels are pending; source-median test has only {fmt(fals_source.get('n_sources'), 0)} eligible sources and p {fmt(fals_source.get('sign_flip_p'))}; top-k still has source concentration.",
        "Use the immediate-review dossier and blinded workbook to collect manual particle/front labels; rerun gates after labels are frozen.",
        "Near-pre-event particle crops show coupled optical-transport, rollout-difficulty, front/kinetic evidence useful for source-aware manual-review ranking.",
        "Do not call this a causal degradation mechanism, a deployable detector, a calibrated velocity, or a diffusion coefficient.",
        "source_balanced_pre_event_transport_kinetic_fusion_audit; source_balanced_transport_mechanism_falsification_audit; source_balanced_transport_mechanism_dossier",
    ))
    rows.append(row(
        "C02_transport_mechanism_candidate_dossier",
        "The project has a concrete prioritized review dossier for apparent transport/front/kinetic candidates.",
        "ready_for_manual_qc_handoff",
        "operationally_strong",
        (
            f"Dossier has {dossier.get('n_rows')} rows, {dossier.get('n_sources')} sources, {dossier.get('n_immediate_review')} immediate-review rows, "
            f"{dossier.get('n_diffusion_claim_candidates')} diffusion-claim candidates; top candidate {dossier.get('top_candidate', {}).get('roi_id')} score {fmt(dossier.get('top_candidate', {}).get('transport_mechanism_score'))}. "
            f"Blinded workbook has {blind.get('n_blinded_rows')} rows across {blind.get('n_sources_hidden_key')} hidden-key sources."
        ),
        "No manual decisions have been imported yet; hidden key must remain unused until review is frozen.",
        "Have reviewers complete blinded workbook fields, then join labels back through the hidden key and recompute strict gates.",
        "A blinded, source-balanced manual-QC handoff exists for top transport/front/kinetic candidates.",
        "Do not treat the dossier scores as manual labels or accepted fronts.",
        "source_balanced_transport_mechanism_dossier; source_balanced_pre_event_manual_qc_blind_workbook",
    ))
    rows.append(row(
        "C03_broad_future_drop_generalization",
        "Apparent transport/front descriptors generalize as a broad future-drop predictor across expanded ROI crops.",
        "not_supported_as_broad_predictor",
        "mixed_weak_generalization",
        (
            f"Expanded cohort processed {expansion.get('n_ok')}/{expansion.get('n_input_rows')} rows across {expansion.get('n_sources')} sources. "
            f"Best future8 feature {expansion_future8.get('feature')} AUC {fmt(expansion_future8.get('oriented_auc'))}, AP {fmt(expansion_future8.get('average_precision'))}, source-stratified p {fmt(expansion_future8.get('source_stratified_permutation_p'))}. "
            f"Best future16 feature {expansion_future16.get('feature')} AUC {fmt(expansion_future16.get('oriented_auc'))}, p {fmt(expansion_future16.get('source_stratified_permutation_p'))}; source-residual divergence_q90 AUC {fmt(expansion_residual.get('oriented_auc'))}, p {fmt(expansion_residual.get('source_stratified_permutation_p'))}."
        ),
        "Future8 raw radial-flow rows are suggestive but not strict source-invariant; composite expansion score is weak and can rank high-scoring future-negative controls.",
        "Treat expansion rows as candidate generation and negative controls; build event-window-specific models rather than one broad predictor.",
        "Expanded-cohort transport/front features show modest, source-sensitive future-label associations and useful controls.",
        "Do not claim a general source-robust degradation-warning model from transport/front descriptors alone.",
        "source_balanced_expansion_transport_front_audit",
    ))
    rows.append(row(
        "C04_next_frame_rollout_physics_descriptor",
        "Next-frame prediction and rollout residuals are useful particle-region physics descriptors.",
        "supported_as_descriptor_not_superior_predictor",
        "moderate_descriptor_signal",
        (
            f"Best median particle baseline is {rollout_best.get('method')} with median MSE {fmt(rollout_best.get('particle_mse_median'))}; no non-persistence method wins by median. "
            f"Top event-relative rollout signal {rollout_event.get('feature')} has AUC {fmt(rollout_event.get('oriented_auc'))}, AP {fmt(rollout_event.get('average_precision'))}, MW p {fmt(rollout_event.get('mwu_p'))}."
        ),
        "Learned/trend baselines do not beat persistence robustly; rollout residuals indicate instability/difficulty rather than accurate long-horizon prediction.",
        "Use persistence-normalized particle residuals as interpretable features in source-aware review models; avoid claiming predictive superiority.",
        "Particle-region rollout difficulty is a useful instability descriptor near events.",
        "Do not claim the current rollout models outperform persistence or forecast full video evolution reliably.",
        "source_balanced_pre_event_masked_rollout_benchmark",
    ))
    rows.append(row(
        "C05_calibrated_diffusion_coefficients",
        "The current pipeline can report calibrated material diffusion coefficients.",
        "blocked",
        "strong_negative_gate_evidence",
        (
            f"Diffusion readiness status is {diffusion.get('overall_status')}; publication-ready candidates {diffusion.get('n_publication_ready_candidates')}; "
            f"hard blockers {diffusion.get('n_hard_blockers')}: {', '.join(diffusion.get('hard_blockers', []))}."
        ),
        "Missing/insufficient spatial calibration metadata, timing stability, front fit quality, positive CI, control-balanced sanity, event/control separability, and accepted manual QC.",
        "After manual QC and calibration provenance, recompute front slopes and diffusion gates only on accepted fronts.",
        "Only apparent optical-front diffusion proxies are available, and calibrated diffusion is not ready.",
        "Do not report Li diffusion coefficients, calibrated material diffusivity, or publication-ready diffusion candidates.",
        "diffusion_claim_readiness_audit; diffusion_physics_consistency_audit; control_balanced_diffusion_proxy_sanity_audit",
    ))
    rows.append(row(
        "C06_phase_boundary_tracking",
        "The videos support automatic phase-boundary/front tracking as a physical proxy.",
        "partial_proxy_only",
        "moderate_but_qc_limited",
        (
            f"Front/kinetic fusion near-vs-post/control AUC {fmt(fusion_near_post.get('oriented_auc'))}, AP {fmt(fusion_near_post.get('average_precision'))}, source-stratified p {fmt(fusion_near_post.get('source_stratified_permutation_p'))}; "
            f"mechanism top candidate has front-direction gate {dossier.get('top_candidate', {}).get('front_direction_agreement_gate')} and diffusion blocked {dossier.get('top_candidate', {}).get('diffusion_claim_blocked')}."
        ),
        "Automatic front/radius proxies are sensitive to masks, thresholds, drift, and missing manual accept labels.",
        "Use visual packet and blinded manual labels to decide which fronts are interpretable; then rerun strict front gates.",
        "Automatic front/kinetic proxies can prioritize candidate phase-boundary-like motion for review.",
        "Do not describe these as validated physical phase-boundary velocities before manual QC and calibration.",
        "source_balanced_pre_event_transport_kinetic_fusion_audit; source_balanced_transport_mechanism_dossier; source_balanced_pre_event_manual_qc_visual_packet",
    ))
    rows.append(row(
        "C07_deployable_future_warning_detector",
        "The current automatic video/echem features form a deployable future degradation warning detector.",
        "blocked_for_deployment",
        "negative_after_context_controls",
        (
            f"Residualized future8 decision is video={residual_decision.get('future8_video_physics_status')} and fused={residual_decision.get('future8_fused_video_echem_incremental_status')}; "
            f"acquisition context AUC {fmt(residual_decision.get('acquisition_source_cohort_metric', {}).get('oriented_auc'))}; fused minus echem residualized source-cohort delta AUC {fmt(residual_decision.get('fused_minus_echem_residualized_source_cohort_delta_auc'))}."
        ),
        "Acquisition/source context dominates weak future labels; optical/echem fusion is not incremental over echem context under strict source-cohort controls.",
        "Keep models as review-prioritization aids; collect manual labels and design prospective/source-heldout validation before any deployment claim.",
        "Current warning models expose context and review-prioritization signals.",
        "Do not claim deployable warning, prospective prediction, or source-transferable degradation detection.",
        "residualized_future8_video_physics_benchmark; source_balanced_video_echem_transfer_audit; source_invariant_video_echem_transfer_audit",
    ))

    matrix = pd.DataFrame(rows)
    status_counts = matrix["readiness"].value_counts().rename_axis("readiness").reset_index(name="n")
    by_strength = matrix["evidence_strength"].value_counts().rename_axis("evidence_strength").reset_index(name="n")

    paths = {
        "matrix_csv": out / "current_claim_readiness_matrix.csv",
        "status_counts_csv": out / "current_claim_readiness_status_counts.csv",
        "summary_json": out / "current_claim_readiness_summary.json",
        "readme": out / "README.md",
    }
    matrix.to_csv(paths["matrix_csv"], index=False)
    status_counts.to_csv(paths["status_counts_csv"], index=False)

    supported = matrix[matrix["readiness"].isin(["supported_for_review_ranking", "ready_for_manual_qc_handoff", "supported_as_descriptor_not_superior_predictor"])]
    blocked = matrix[matrix["readiness"].str.contains("blocked|not_supported", regex=True)]
    summary = {
        "n_claims": int(len(matrix)),
        "status_counts": clean_json(status_counts.to_dict("records")),
        "evidence_strength_counts": clean_json(by_strength.to_dict("records")),
        "supported_or_operational_claim_ids": supported["claim_id"].tolist(),
        "blocked_or_not_supported_claim_ids": blocked["claim_id"].tolist(),
        "overall_position": "The project supports source-aware particle-region review ranking and optical-proxy hypothesis generation, while calibrated diffusion, causal mechanism, and deployable warning claims remain blocked.",
        "top_positive_evidence": clean_json({
            "fusion_near_any_auc": fusion_near_any.get("oriented_auc"),
            "fusion_near_any_source_stratified_p": fusion_near_any.get("source_stratified_permutation_p"),
            "falsification_near_any_auc": fals_event.get("auc"),
            "falsification_source_stratified_p": fals_event.get("source_stratified_permutation_p"),
            "same_source_pair_median_delta": fals_pair.get("median_delta_near_minus_control"),
            "same_source_pair_sign_flip_p": fals_pair.get("sign_flip_p"),
        }),
        "top_negative_evidence": clean_json({
            "diffusion_status": diffusion.get("overall_status"),
            "diffusion_publication_ready_candidates": diffusion.get("n_publication_ready_candidates"),
            "diffusion_hard_blockers": diffusion.get("hard_blockers"),
            "future8_video_physics_status": residual_decision.get("future8_video_physics_status"),
            "future8_fused_video_echem_incremental_status": residual_decision.get("future8_fused_video_echem_incremental_status"),
            "expansion_future8_source_stratified_p": expansion_future8.get("source_stratified_permutation_p"),
        }),
        "guardrail": "Claim readiness is wording guidance over existing automatic audits. It does not add manual labels, calibrated spatial metadata, causal tests, or deployable validation.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))

    lines = [
        "# Current Claim Readiness Matrix",
        "",
        summary["overall_position"],
        "",
        f"- Claims audited: {summary['n_claims']}",
        f"- Supported/operational claim IDs: {', '.join(summary['supported_or_operational_claim_ids'])}",
        f"- Blocked/not-supported claim IDs: {', '.join(summary['blocked_or_not_supported_claim_ids'])}",
        "",
        "## Status Counts",
        "",
    ]
    for item in summary["status_counts"]:
        lines.append(f"- {item['readiness']}: {item['n']}")
    lines += ["", "## Claim Rows", ""]
    for item in rows:
        lines.append(f"- {item['claim_id']} {item['readiness']}: {item['claim']}")
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    paths["readme"].write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
