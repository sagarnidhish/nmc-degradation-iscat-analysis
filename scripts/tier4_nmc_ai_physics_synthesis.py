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
    conditioned = read_json(derived / "protocol_conditioned_roi_effects" / "protocol_conditioned_roi_effects_summary.json")
    robust_fronts = read_json(derived / "multi_cycle_threshold_robust_fronts" / "threshold_robust_front_summary.json")
    conditioned_fronts = read_json(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_effects_summary.json")
    qc_packet = read_json(derived / "qc_review_packet" / "qc_review_packet_summary.json")
    qc_package = read_json(derived / "roi_front_qc_package" / "roi_front_qc_package_summary.json")
    control_balanced_qc = read_json(derived / "control_balanced_front_qc_package" / "control_balanced_front_qc_summary.json")
    front_qc_sensitivity = read_json(derived / "front_qc_sensitivity" / "front_qc_sensitivity_summary.json")
    control_balanced_qc_sensitivity = read_json(derived / "control_balanced_front_qc_sensitivity" / "control_balanced_front_qc_sensitivity_summary.json")
    residual_modes = read_json(derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_taxonomy_summary.json")
    cycle_region_modes = read_json(derived / "cycle_region_mode_context" / "cycle_region_mode_context_summary.json")
    prefix_forecast = read_json(derived / "prefix_roi_forecast" / "prefix_roi_forecast_summary.json")
    prefix_importance = read_json(derived / "prefix_roi_feature_importance" / "prefix_feature_importance_summary.json")
    manual_qc_workbook = read_json(derived / "manual_qc_label_workbook" / "manual_qc_label_workbook_summary.json")
    manual_qc_gated = read_json(derived / "manual_qc_gated_front_effects" / "manual_qc_gated_front_effects_summary.json")
    spatiotemporal_graph = read_json(derived / "spatiotemporal_degradation_graph" / "spatiotemporal_degradation_graph_summary.json")
    phase_kinetics = read_json(derived / "phase_kinetics_avrami" / "phase_kinetics_avrami_summary.json")
    calibration_metadata = read_json(derived / "calibration_metadata_audit" / "calibration_metadata_audit_summary.json")
    calibration_provenance = read_json(derived / "calibration_provenance_evidence_audit" / "calibration_provenance_summary.json")
    calibration_claim_risk = read_json(derived / "calibration_claim_risk_register" / "calibration_claim_risk_summary.json")
    apparent_diffusion_calibration = read_json(derived / "apparent_diffusion_calibration_bounds" / "apparent_diffusion_calibration_bounds_summary.json")
    hdf5_timebase = read_json(derived / "hdf5_timebase_provenance_audit" / "hdf5_timebase_provenance_summary.json")
    diffusion_physics_consistency = read_json(derived / "diffusion_physics_consistency_audit" / "diffusion_physics_consistency_summary.json")
    diffusion_claim_readiness = read_json(derived / "diffusion_claim_readiness_audit" / "diffusion_claim_readiness_summary.json")
    all_cycle_coverage = read_json(derived / "all_cycle_dataset_coverage_atlas" / "all_cycle_dataset_coverage_summary.json")
    current_claim_readiness = read_json(derived / "current_claim_readiness_matrix" / "current_claim_readiness_summary.json")
    diffusion_unblock_sensitivity = read_json(derived / "diffusion_unblock_sensitivity_audit" / "diffusion_unblock_sensitivity_summary.json")
    targeted_diffusion_blocker = read_json(derived / "targeted_diffusion_blocker_diagnostic" / "targeted_diffusion_blocker_diagnostic_summary.json")
    cycle78_diffusion_remeasurement = read_json(derived / "cycle78_diffusion_remeasurement_audit" / "cycle78_diffusion_remeasurement_summary.json")
    post_remeasurement_diffusion_gate = read_json(derived / "post_remeasurement_diffusion_gate_audit" / "post_remeasurement_diffusion_gate_summary.json")
    cycle78_front_identity_review = read_json(derived / "cycle78_front_identity_review_packet" / "cycle78_front_identity_review_summary.json")
    cycle78_component_retracking = read_json(derived / "cycle78_component_front_retracking_audit" / "cycle78_component_front_retracking_summary.json")
    cross_modal_consensus = read_json(derived / "cross_modal_degradation_consensus" / "cross_modal_degradation_consensus_summary.json")
    particle_trace = read_json(derived / "particle_trace_physics_audit" / "particle_trace_physics_audit_summary.json")
    particle_precursor = read_json(derived / "particle_event_precursor_atlas" / "particle_event_precursor_atlas_summary.json")
    roi_trace_fusion = read_json(derived / "roi_trace_fusion_audit" / "roi_trace_fusion_audit_summary.json")
    roi_trace_cycle_null = read_json(derived / "roi_trace_fusion_cycle_null" / "roi_trace_fusion_cycle_null_summary.json")
    precursor_review = read_json(derived / "precursor_informed_roi_review" / "precursor_informed_roi_review_summary.json")
    precursor_visual_bundle = read_json(derived / "precursor_review_visual_bundle" / "precursor_review_visual_bundle_summary.json")
    within_cycle_echem = read_json(derived / "within_cycle_echem_shape_audit" / "within_cycle_echem_shape_audit_summary.json")
    echem_shape_conditioned = read_json(derived / "echem_shape_conditioned_roi_front_effects" / "echem_shape_conditioned_roi_front_effects_summary.json")
    echem_optical_breakpoint = read_json(derived / "echem_optical_breakpoint_audit" / "echem_optical_breakpoint_audit_summary.json")
    echem_optical_regime = read_json(derived / "echem_optical_regime_atlas" / "echem_optical_regime_atlas_summary.json")
    echem_conditioned_predictor = read_json(derived / "echem_conditioned_optical_predictor" / "echem_conditioned_optical_predictor_summary.json")
    echem_roi_rollout_front = read_json(derived / "echem_conditioned_roi_rollout_front_audit" / "echem_conditioned_roi_rollout_front_summary.json")
    echem_video_fusion = read_json(derived / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_summary.json")
    physics_consistency = read_json(derived / "physics_consistency_claim_matrix" / "physics_consistency_claim_matrix_summary.json")
    rollout_calibration = read_json(derived / "probabilistic_rollout_calibration" / "probabilistic_rollout_calibration_summary.json")
    cycle_state_space = read_json(derived / "cycle_state_space_transition_audit" / "cycle_state_space_transition_audit_summary.json")
    cycle_hazard_warning = read_json(derived / "cycle_hazard_warning_audit" / "cycle_hazard_warning_audit_summary.json")
    cycle_state_roi_bridge = read_json(derived / "cycle_state_roi_bridge" / "cycle_state_roi_bridge_summary.json")
    cycle_state_mode_frequency = read_json(derived / "cycle_state_mode_frequency_bridge" / "cycle_state_mode_frequency_bridge_summary.json")
    particle_mask = read_json(derived / "particle_mask_stability_audit" / "particle_mask_stability_audit_summary.json")
    particle_mask_history_fallback = read_json(derived / "particle_mask_history_fallback_audit" / "particle_mask_history_fallback_summary.json")
    history_fallback_rollout_ablation = read_json(derived / "history_fallback_masked_rollout_ablation" / "history_fallback_masked_rollout_ablation_summary.json")
    rollout_front_mode_coupling = read_json(derived / "rollout_front_mode_coupling_audit" / "rollout_front_mode_coupling_summary.json")
    masked_rollout = read_json(derived / "masked_roi_rollout_audit" / "masked_roi_rollout_audit_summary.json")
    masked_cycle_warning = read_json(derived / "masked_rollout_cycle_warning" / "masked_rollout_cycle_warning_summary.json")
    masked_residual_timing = read_json(derived / "masked_residual_transition_timing" / "masked_residual_transition_timing_summary.json")
    masked_residual_transfer = read_json(derived / "masked_residual_state_transfer_warning" / "masked_residual_state_transfer_warning_summary.json")
    transfer_ranked_reconstruction = read_json(derived / "transfer_ranked_roi_reconstruction" / "transfer_ranked_roi_reconstruction_summary.json")
    transfer_ranked_sequences = read_json(derived / "transfer_ranked_roi_sequences" / "selected_roi_sequence_summary.json")
    transfer_ranked_rollout = read_json(derived / "transfer_ranked_masked_roi_rollout_audit" / "masked_roi_rollout_audit_summary.json")
    transfer_ranked_fronts = read_json(derived / "transfer_ranked_threshold_robust_fronts" / "threshold_robust_front_summary.json")
    transfer_ranked_front_physics = read_json(derived / "transfer_ranked_front_physics_audit" / "transfer_ranked_front_physics_audit_summary.json")
    transfer_ranked_residual_timing = read_json(derived / "transfer_ranked_residual_transition_timing" / "transfer_ranked_residual_transition_timing_summary.json")
    multicohort_future_drop = read_json(derived / "multicohort_future_drop_model" / "multicohort_future_drop_model_summary.json")
    active_learning_qc = read_json(derived / "active_learning_qc_prioritization" / "active_learning_qc_summary.json")
    automatic_qc_triage = read_json(derived / "automatic_qc_triage_surrogate" / "automatic_qc_triage_surrogate_summary.json")
    qc_decision_ledger = read_json(derived / "qc_decision_evidence_ledger" / "qc_decision_evidence_ledger_summary.json")
    balanced_future_reconstruction = read_json(derived / "balanced_future_roi_reconstruction" / "balanced_future_roi_reconstruction_summary.json")
    source_balanced_expansion = read_json(derived / "source_balanced_roi_expansion_manifest" / "source_balanced_roi_expansion_summary.json")
    source_balanced_pre_event_sampling = read_json(derived / "source_balanced_pre_event_sampling_manifest" / "source_balanced_pre_event_sampling_summary.json")
    source_balanced_pre_event_sequences = read_json(derived / "source_balanced_pre_event_roi_sequences" / "selected_roi_sequence_summary.json")
    source_balanced_pre_event_sequence_audit = read_json(derived / "source_balanced_pre_event_sequence_audit" / "source_balanced_pre_event_sequence_audit_summary.json")
    source_balanced_pre_event_rollout = read_json(derived / "source_balanced_pre_event_sequence_rollout_audit" / "source_balanced_sequence_rollout_summary.json")
    source_balanced_pre_event_masked_rollout = read_json(derived / "source_balanced_pre_event_masked_rollout_benchmark" / "source_balanced_pre_event_masked_rollout_summary.json")
    source_balanced_pre_event_optical_flow = read_json(derived / "source_balanced_pre_event_optical_flow_transport_audit" / "source_balanced_pre_event_optical_flow_transport_summary.json")
    source_balanced_pre_event_mask_front = read_json(derived / "source_balanced_pre_event_mask_front_audit" / "source_balanced_mask_front_summary.json")
    source_balanced_pre_event_readout = read_json(derived / "source_balanced_pre_event_readout_audit" / "source_balanced_pre_event_readout_summary.json")
    source_balanced_pre_event_trajectory = read_json(derived / "source_balanced_pre_event_trajectory_audit" / "source_balanced_pre_event_trajectory_summary.json")
    source_balanced_pre_event_directionality = read_json(derived / "source_balanced_pre_event_directionality_audit" / "source_balanced_pre_event_directionality_summary.json")
    source_balanced_pre_event_source_invariant = read_json(derived / "source_balanced_pre_event_source_invariant_audit" / "source_balanced_pre_event_source_invariant_summary.json")
    source_balanced_pre_event_review_packet = read_json(derived / "source_balanced_pre_event_review_packet" / "source_balanced_pre_event_review_packet_summary.json")
    source_balanced_pre_event_matched_counterfactual = read_json(derived / "source_balanced_pre_event_matched_counterfactual_audit" / "source_balanced_pre_event_matched_counterfactual_summary.json")
    source_balanced_pre_event_same_source_ladder = read_json(derived / "source_balanced_pre_event_same_source_ladder_audit" / "source_balanced_pre_event_same_source_ladder_summary.json")
    pre_event_source_lattice = read_json(derived / "pre_event_source_lattice_coverage_audit" / "pre_event_source_lattice_coverage_summary.json")
    source_balanced_pre_event_radial_kymograph = read_json(derived / "source_balanced_pre_event_radial_kymograph_audit" / "source_balanced_pre_event_radial_kymograph_summary.json")
    source_balanced_pre_event_echem_front = read_json(derived / "source_balanced_pre_event_echem_front_coupling_audit" / "source_balanced_pre_event_echem_front_summary.json")
    source_balanced_pre_event_echem_matched_residual = read_json(derived / "source_balanced_pre_event_echem_matched_residual_audit" / "source_balanced_pre_event_echem_matched_summary.json")
    source_balanced_pre_event_front_consensus = read_json(derived / "source_balanced_pre_event_front_consensus_audit" / "source_balanced_pre_event_front_consensus_summary.json")
    source_balanced_pre_event_echem_matched_far = read_json(derived / "source_balanced_pre_event_echem_matched_far_control_audit" / "source_balanced_pre_event_echem_matched_far_summary.json")
    source_balanced_pre_event_consensus_review = read_json(derived / "source_balanced_pre_event_consensus_review_queue" / "source_balanced_pre_event_consensus_review_summary.json")
    source_balanced_pre_event_consensus_visual = read_json(derived / "source_balanced_pre_event_consensus_visual_packet" / "source_balanced_pre_event_consensus_visual_summary.json")
    source_balanced_pre_event_visual_sanity = read_json(derived / "source_balanced_pre_event_visual_sanity_audit" / "source_balanced_pre_event_visual_sanity_summary.json")
    source_balanced_pre_event_visual_qc_modes = read_json(derived / "source_balanced_pre_event_visual_qc_modes" / "source_balanced_pre_event_visual_qc_modes_summary.json")
    source_balanced_pre_event_phase_kinetics = read_json(derived / "source_balanced_pre_event_phase_kinetics_audit" / "source_balanced_pre_event_phase_kinetics_summary.json")
    source_balanced_pre_event_front_kinetic_concordance = read_json(derived / "source_balanced_pre_event_front_kinetic_concordance_audit" / "source_balanced_pre_event_front_kinetic_concordance_summary.json")
    source_balanced_pre_event_front_kinetic_null = read_json(derived / "source_balanced_pre_event_front_kinetic_null_audit" / "source_balanced_pre_event_front_kinetic_null_summary.json")
    source_balanced_pre_event_manual_qc_decision = read_json(derived / "source_balanced_pre_event_manual_qc_decision_packet" / "source_balanced_pre_event_manual_qc_decision_summary.json")
    source_balanced_pre_event_manual_qc_visual = read_json(derived / "source_balanced_pre_event_manual_qc_visual_packet" / "source_balanced_pre_event_manual_qc_visual_summary.json")
    source_balanced_pre_event_manual_qc_blind = read_json(derived / "source_balanced_pre_event_manual_qc_blind_workbook" / "source_balanced_pre_event_manual_qc_blind_summary.json")
    source_balanced_pre_event_multimodal = read_json(derived / "source_balanced_pre_event_multimodal_predictor" / "source_balanced_pre_event_multimodal_summary.json")
    source_balanced_pre_event_strict_qc_gated_front = read_json(derived / "source_balanced_pre_event_strict_qc_gated_front_audit" / "source_balanced_pre_event_strict_qc_gated_front_summary.json")
    source_balanced_pre_event_physics_modes = read_json(derived / "source_balanced_pre_event_physics_mode_taxonomy" / "source_balanced_pre_event_physics_mode_summary.json")
    source_balanced_sequences = read_json(derived / "source_balanced_roi_sequences" / "selected_roi_sequence_summary.json")
    source_balanced_sequence_rollout = read_json(derived / "source_balanced_sequence_rollout_audit" / "source_balanced_sequence_rollout_summary.json")
    source_balanced_sequence_source_control = read_json(derived / "source_balanced_sequence_source_control_audit" / "source_balanced_sequence_source_control_summary.json")
    source_balanced_expansion_transport_front = read_json(derived / "source_balanced_expansion_transport_front_audit" / "source_balanced_expansion_transport_front_summary.json")
    source_balanced_mask_front = read_json(derived / "source_balanced_mask_front_sanity_audit" / "source_balanced_mask_front_summary.json")
    source_balanced_mask_front_source_residual = read_json(derived / "source_balanced_mask_front_source_residual_audit" / "source_balanced_mask_front_source_residual_summary.json")
    source_balanced_residual_dictionary = read_json(derived / "source_balanced_residual_dictionary_audit" / "source_balanced_residual_dictionary_summary.json")
    source_balanced_resdict_source_residual = read_json(derived / "source_balanced_residual_dictionary_source_residual_audit" / "source_balanced_residual_dictionary_source_residual_summary.json")
    source_balanced_resdict_normalized_readout = read_json(derived / "source_balanced_residual_dictionary_normalized_readout" / "source_balanced_residual_dictionary_normalized_readout_summary.json")
    source_balanced_residual_temporal_specificity = read_json(derived / "source_balanced_residual_temporal_specificity_audit" / "source_balanced_residual_temporal_specificity_summary.json")
    source_balanced_future_specific_residual = read_json(derived / "source_balanced_future_specific_residual_audit" / "source_balanced_future_specific_residual_summary.json")
    source_balanced_degradation_modes = read_json(derived / "source_balanced_degradation_mode_audit" / "source_balanced_degradation_mode_summary.json")
    source_balanced_residual_physics_coupling = read_json(derived / "source_balanced_residual_physics_coupling_audit" / "source_balanced_residual_physics_coupling_summary.json")
    source_balanced_residual_candidate_review = read_json(derived / "source_balanced_residual_candidate_review_packet" / "source_balanced_residual_candidate_review_summary.json")
    balanced_future_sequences = read_json(derived / "balanced_future_roi_sequences" / "selected_roi_sequence_summary.json")
    balanced_future_fronts = read_json(derived / "balanced_future_threshold_robust_fronts" / "threshold_robust_front_summary.json")
    balanced_future_rollout = read_json(derived / "balanced_future_masked_roi_rollout_audit" / "masked_roi_rollout_audit_summary.json")
    balanced_future_mask = read_json(derived / "balanced_future_particle_mask_stability" / "particle_mask_stability_audit_summary.json")
    balanced_future_context = read_json(derived / "balanced_future_context_region_audit" / "balanced_future_context_region_summary.json")
    temporal_directionality = read_json(derived / "temporal_directionality_physics_audit" / "temporal_directionality_physics_audit_summary.json")
    balanced_spatial_propagation = read_json(derived / "balanced_spatial_front_propagation_audit" / "balanced_spatial_front_propagation_summary.json")
    masked_video_embedding = read_json(derived / "masked_video_embedding_audit" / "masked_video_embedding_audit_summary.json")
    learned_video_residual_embedding = read_json(derived / "learned_video_residual_embedding_audit" / "learned_video_residual_embedding_summary.json")
    residual_dictionary_embedding = read_json(derived / "residual_dictionary_embedding_audit" / "residual_dictionary_embedding_summary.json")
    echem_residual_dictionary_fusion = read_json(derived / "echem_residual_dictionary_fusion_audit" / "echem_residual_dictionary_summary.json")
    echem_conditioned_residual_dictionary = read_json(derived / "echem_conditioned_residual_dictionary" / "echem_conditioned_residual_dictionary_summary.json")
    conditioned_residual_physics_atlas = read_json(derived / "conditioned_residual_physics_atlas" / "conditioned_residual_physics_atlas_summary.json")
    acquisition_residualized_video = read_json(derived / "acquisition_residualized_video_physics_benchmark" / "acquisition_residualized_summary.json")
    acquisition_residualized_video_echem = read_json(derived / "acquisition_residualized_video_echem_warning" / "acquisition_residualized_video_echem_summary.json")
    residualized_future8_video_physics = read_json(derived / "residualized_future8_video_physics_benchmark" / "residualized_future8_video_physics_summary.json")
    source_balanced_pre_event_observable_forecast = read_json(derived / "source_balanced_pre_event_observable_forecast" / "source_balanced_pre_event_observable_forecast_summary.json")
    source_balanced_pre_event_optical_flow_transport = read_json(derived / "source_balanced_pre_event_optical_flow_transport_audit" / "source_balanced_pre_event_optical_flow_transport_summary.json")
    source_balanced_pre_event_transport_kinetic_fusion = read_json(derived / "source_balanced_pre_event_transport_kinetic_fusion_audit" / "source_balanced_pre_event_transport_kinetic_fusion_summary.json")
    source_balanced_transport_mechanism = read_json(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_summary.json")
    source_balanced_transport_mechanism_falsification = read_json(derived / "source_balanced_transport_mechanism_falsification_audit" / "source_balanced_transport_mechanism_falsification_summary.json")
    source_heldout_event_rank_transfer = read_json(derived / "source_heldout_event_rank_transfer_audit" / "source_heldout_event_rank_transfer_summary.json")
    pre_event_temporal_dose_response = read_json(derived / "pre_event_temporal_dose_response_audit" / "pre_event_temporal_dose_response_summary.json")
    targeted_densification_qc = read_json(derived / "targeted_densification_qc_plan" / "targeted_densification_qc_summary.json")
    source_domain_video_echem = read_json(derived / "source_domain_video_echem_adaptation_audit" / "source_domain_video_echem_summary.json")
    source_balanced_video_echem = read_json(derived / "source_balanced_video_echem_transfer_audit" / "source_balanced_video_echem_summary.json")
    source_invariant_video_echem = read_json(derived / "source_invariant_video_echem_transfer_audit" / "source_invariant_video_echem_summary.json")
    source_invariant_family = read_json(derived / "source_invariant_physical_family_audit" / "source_invariant_family_summary.json")
    source_invariant_interpretable = read_json(derived / "source_invariant_interpretable_feature_audit" / "source_invariant_interpretable_summary.json")
    exact_feature_mechanism = read_json(derived / "exact_feature_mechanism_consistency_audit" / "exact_feature_mechanism_summary.json")
    signed_optical_loss = read_json(derived / "signed_optical_loss_mechanism_audit" / "signed_optical_loss_mechanism_summary.json")
    signed_loss_source_robustness = read_json(derived / "signed_loss_source_robustness_audit" / "signed_loss_source_robustness_summary.json")
    echem_optical_source_residual = read_json(derived / "echem_optical_source_residual_audit" / "echem_optical_source_residual_summary.json")
    invariant_physics_rules = read_json(derived / "invariant_physics_rule_discovery" / "invariant_physics_rule_summary.json")
    agentic_current = read_json(derived / "agentic_current_hypothesis_tournament" / "agentic_current_hypothesis_tournament_summary.json")
    balanced_future_physics = read_json(derived / "balanced_future_roi_physics_audit" / "balanced_future_roi_physics_audit_summary.json")
    cross_cohort_rollout = read_json(derived / "cross_cohort_rollout_transfer_audit" / "cross_cohort_rollout_transfer_summary.json")
    diffusion_sanity = read_json(derived / "diffusion_proxy_sanity_audit" / "diffusion_proxy_sanity_audit_summary.json")
    control_balanced_front_tracking = read_json(derived / "control_balanced_front_tracking" / "selected_front_roi_tracking_summary.json")
    control_balanced_diffusion_sanity = read_json(derived / "control_balanced_diffusion_proxy_sanity_audit" / "diffusion_proxy_sanity_audit_summary.json")
    weak_label_benchmark = read_json(derived / "weak_label_degradation_benchmark" / "weak_label_degradation_benchmark_summary.json")

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

    conditioned_model = first_summary(conditioned, "model_summary", {})
    conditioned_tests = top_items(first_summary(conditioned, "top_protocol_conditioned_event_control_tests", []), 8)
    robust_front_tests = top_items(first_summary(robust_fronts, "top_overall_feature_tests", []), 8)
    robust_front_by_event = top_items(first_summary(robust_fronts, "top_by_event_feature_tests", []), 8)
    conditioned_front_model = first_summary(conditioned_fronts, "model_summary", {})
    conditioned_front_tests = top_items(first_summary(conditioned_fronts, "top_protocol_conditioned_front_tests", []), 8)
    residual_mode_enrichment = top_items(first_summary(residual_modes, "mode_enrichment", []), 8)
    residual_mode_top_rois = top_items(first_summary(residual_modes, "top_review_rois", []), 10)
    cycle_region_context = top_items(first_summary(cycle_region_modes, "top_cycle_summaries", []), 8)
    spatial_region_context = top_items(first_summary(cycle_region_modes, "top_spatial_region_summaries", []), 8)
    mode_context_correlations = top_items(first_summary(cycle_region_modes, "top_context_correlations", []), 8)
    prefix_top_classification = top_items(first_summary(prefix_forecast, "top_classification", []), 10)
    prefix_top_regression = top_items(first_summary(prefix_forecast, "top_regression", []), 8)
    prefix_null = top_items(first_summary(prefix_forecast, "permutation_null", []), 5)
    prefix_importance_model = first_summary(prefix_importance, "model_summary", {}) or {}
    prefix_importance_null = first_summary(prefix_importance, "permutation_null", {}) or {}
    prefix_importance_groups = top_items(first_summary(prefix_importance, "top_group_ablation", []), 6)
    prefix_importance_features = top_items(first_summary(prefix_importance, "top_permutation_importance", []), 8)
    top_prefix_classifier = prefix_top_classification[0] if prefix_top_classification else {}
    top_residual_mode = residual_mode_enrichment[0] if residual_mode_enrichment else {}
    control_balanced_nonfragment = first_summary(control_balanced_qc, "nonfragmented_by_role", {}) or {}
    front_qc_focus = top_items(first_summary(front_qc_sensitivity, "focus_tests", []), 25)
    front_qc_strata = top_items(first_summary(front_qc_sensitivity, "strata", []), 12)
    robust_phase_strata = first_summary(front_qc_sensitivity, "robust_positive_phase_residual_strata", [])
    balanced_qc_focus = first_summary(control_balanced_qc_sensitivity, "focus_tests", []) or []
    balanced_robust_phase_strata = first_summary(control_balanced_qc_sensitivity, "robust_positive_phase_residual_strata", [])
    graph_homophily = top_items(first_summary(spatiotemporal_graph, "top_homophily_tests", []), 10)
    graph_continuous = top_items(first_summary(spatiotemporal_graph, "top_continuous_neighbor_tests", []), 10)
    graph_lag = top_items(first_summary(spatiotemporal_graph, "temporal_lag_tests", []), 10)
    graph_distance = top_items(first_summary(spatiotemporal_graph, "distance_gradient_tests", []), 8)
    phase_kinetics_tests = top_items(first_summary(phase_kinetics, "top_group_tests", []), 12)
    phase_kinetics_corr = top_items(first_summary(phase_kinetics, "top_correlations", []), 12)
    particle_trace_event_tests = top_items(first_summary(particle_trace, "top_event_feature_tests", []), 12)
    particle_trace_echem_corr = top_items(first_summary(particle_trace, "top_echem_correlations", []), 12)
    particle_trace_classifiers = top_items(first_summary(particle_trace, "future_drop_classifier", []), 6)
    particle_trace_nulls = top_items(first_summary(particle_trace, "future_drop_classifier_null", []), 6)
    particle_precursor_tests = top_items(first_summary(particle_precursor, "top_precursor_window_tests", []), 12)
    particle_precursor_all_tests = top_items(first_summary(particle_precursor, "top_window_tests", []), 12)
    roi_trace_focus = top_items(first_summary(roi_trace_fusion, "top_precursor_context_residual_spearman", []), 12)
    roi_trace_mode_tests = top_items(first_summary(roi_trace_fusion, "top_precursor_event_enriched_mode_tests", []), 12)
    roi_trace_cycle_tests = top_items(first_summary(roi_trace_cycle_null, "top_cycle_collapsed_tests", []), 12)
    roi_trace_centered_tests = top_items(first_summary(roi_trace_cycle_null, "top_reference_centered_tests", []), 12)
    precursor_review_top = top_items(first_summary(precursor_review, "top_precursor_informed_candidates", []), 12)
    precursor_visual_top = top_items(first_summary(precursor_visual_bundle, "top_candidates", []), 12)
    within_cycle_roi_corr = top_items(first_summary(within_cycle_echem, "top_roi_shape_correlations", []), 12)
    within_cycle_cycle_corr = top_items(first_summary(within_cycle_echem, "top_cycle_shape_correlations", []), 12)
    within_cycle_event_tests = top_items(first_summary(within_cycle_echem, "top_event_shape_tests", []), 8)
    within_cycle_roi_binary = top_items(first_summary(within_cycle_echem, "top_roi_binary_shape_tests", []), 8)
    echem_shape_conditioned_tests = top_items(first_summary(echem_shape_conditioned, "top_shape_conditioned_event_control_tests", []), 12)
    echem_shape_retention = top_items(first_summary(echem_shape_conditioned, "top_effect_retention", []), 12)
    echem_shape_context = top_items(first_summary(echem_shape_conditioned, "top_shape_context_fits", []), 12)
    echem_shape_pc_corr = top_items(first_summary(echem_shape_conditioned, "top_shape_pc_target_correlations", []), 12)
    echem_shape_model = first_summary(echem_shape_conditioned, "model_summary", {}) or {}
    echem_breakpoint_event = top_items(first_summary(echem_optical_breakpoint, "top_event_centered_tests", []), 12)
    echem_breakpoint_future = top_items(first_summary(echem_optical_breakpoint, "top_future_label_tests", []), 12)
    echem_breakpoint_global = top_items(first_summary(echem_optical_breakpoint, "top_global_breakpoints", []), 8)
    echem_breakpoint_event_ranks = top_items(first_summary(echem_optical_breakpoint, "event_cycle_breakpoint_ranks", []), 12)
    echem_breakpoint_top = echem_breakpoint_event[0] if echem_breakpoint_event else {}
    echem_regime_summary = top_items(first_summary(echem_optical_regime, "regime_summary", []), 8)
    echem_regime_binary = top_items(first_summary(echem_optical_regime, "top_binary_tests", []), 12)
    echem_regime_corr = top_items(first_summary(echem_optical_regime, "top_correlations", []), 12)
    echem_regime_top_cycles = top_items(first_summary(echem_optical_regime, "top_cycles", []), 12)
    echem_regime_top_binary = echem_regime_binary[0] if echem_regime_binary else {}
    echem_regime_top_corr = echem_regime_corr[0] if echem_regime_corr else {}
    echem_predictor_metrics = top_items(first_summary(echem_conditioned_predictor, "top_metrics", []), 16)
    echem_predictor_deltas = top_items(first_summary(echem_conditioned_predictor, "top_feature_set_deltas", []), 16)
    echem_predictor_top_delta = echem_predictor_deltas[0] if echem_predictor_deltas else {}
    echem_roi_metrics = top_items(first_summary(echem_roi_rollout_front, "top_model_metrics", []), 16)
    echem_roi_deltas = top_items(first_summary(echem_roi_rollout_front, "top_feature_set_deltas", []), 16)
    echem_roi_residual = top_items(first_summary(echem_roi_rollout_front, "top_residual_correlations", []), 12)
    echem_roi_top_delta = echem_roi_deltas[0] if echem_roi_deltas else {}
    echem_video_class = top_items(first_summary(echem_video_fusion, "top_classification_metrics", []), 16)
    echem_video_reg = top_items(first_summary(echem_video_fusion, "top_regression_metrics", []), 16)
    echem_video_deltas = top_items(first_summary(echem_video_fusion, "top_feature_set_deltas", []), 16)
    echem_video_top_delta = echem_video_deltas[0] if echem_video_deltas else {}
    physics_consistency_top = top_items(first_summary(physics_consistency, "top_consistency_rows", []), 12)
    physics_consistency_tests = top_items(first_summary(physics_consistency, "top_event_tests", []), 12)
    rollout_calibration_coverage = top_items(first_summary(rollout_calibration, "coverage_summary", []), 30)
    rollout_calibration_tests = top_items(first_summary(rollout_calibration, "top_transition_error_tests", []), 12)
    rollout_calibration_corr = top_items(first_summary(rollout_calibration, "top_calibration_physics_correlations", []), 12)
    rollout_calibration_top = top_items(first_summary(rollout_calibration, "top_undercovered_roi_method_rows", []), 12)
    cycle_state_tests = top_items(first_summary(cycle_state_space, "top_future_drop_tests", []), 12)
    cycle_state_corr = top_items(first_summary(cycle_state_space, "top_cycle_state_correlations", []), 12)
    cycle_state_clusters = top_items(first_summary(cycle_state_space, "top_state_clusters", []), 8)
    cycle_state_transitions = top_items(first_summary(cycle_state_space, "top_transitions", []), 8)
    cycle_state_loadings = top_items(first_summary(cycle_state_space, "top_pc_loadings", []), 12)
    cycle_hazard_feature_sets = top_items(first_summary(cycle_hazard_warning, "feature_set_summary", []), 8)
    cycle_hazard_lead = top_items(first_summary(cycle_hazard_warning, "lead_time_summary", []), 8)
    cycle_hazard_ablation = top_items(first_summary(cycle_hazard_warning, "top_group_ablation", []), 8)
    cycle_hazard_corr = top_items(first_summary(cycle_hazard_warning, "top_probability_correlations", []), 8)
    cycle_hazard_null = first_summary(cycle_hazard_warning, "permutation_null", {}) or {}
    cycle_state_classifier = first_summary(cycle_state_space, "future_drop_classifier", {}) or {}
    cycle_state_temporal = cycle_state_classifier.get("temporal_holdout", {}) if isinstance(cycle_state_classifier, dict) else {}
    cycle_state_roi_row_tests = top_items(first_summary(cycle_state_roi_bridge, "top_row_tests", []), 12)
    cycle_state_roi_centered_tests = top_items(first_summary(cycle_state_roi_bridge, "top_reference_centered_tests", []), 12)
    cycle_state_roi_collapsed_tests = top_items(first_summary(cycle_state_roi_bridge, "top_cycle_collapsed_tests", []), 12)
    cycle_state_roi_clusters = top_items(first_summary(cycle_state_roi_bridge, "cluster_summary", []), 8)
    cycle_state_mode_metrics = top_items(first_summary(cycle_state_mode_frequency, "top_metrics", []), 30)
    cycle_state_mode_nulls = top_items(first_summary(cycle_state_mode_frequency, "permutation_null", []), 12)
    cycle_state_mode_clusters = top_items(first_summary(cycle_state_mode_frequency, "cluster_summary", []), 8)
    cycle_state_mode_best = first_summary(cycle_state_mode_frequency, "best_macro_model", {}) or {}
    cycle_state_mode_context = first_summary(cycle_state_mode_frequency, "context_macro_model", {}) or {}
    particle_mask_role_summary = top_items(first_summary(particle_mask, "role_summary", []), 8)
    particle_mask_tests = top_items(first_summary(particle_mask, "top_event_control_tests", []), 12)
    particle_mask_corr = top_items(first_summary(particle_mask, "top_correlations", []), 12)
    particle_mask_top = top_items(first_summary(particle_mask, "highest_instability_rois", []), 12)
    particle_mask_overall = first_summary(particle_mask, "overall", {}) or {}
    particle_mask_history_tests = top_items(first_summary(particle_mask_history_fallback, "top_event_tests", []), 16)
    particle_mask_history_sources = top_items(first_summary(particle_mask_history_fallback, "source_summary_top", []), 12)
    particle_mask_history_high = top_items(first_summary(particle_mask_history_fallback, "high_fallback_rois", []), 12)
    history_fallback_rollout_tests = top_items(first_summary(history_fallback_rollout_ablation, "top_event_tests", []), 16)
    history_fallback_rollout_methods = top_items(first_summary(history_fallback_rollout_ablation, "method_summary", []), 8)
    history_fallback_rollout_sources = top_items(first_summary(history_fallback_rollout_ablation, "source_summary_top", []), 12)
    rollout_front_mode_tests = top_items(first_summary(rollout_front_mode_coupling, "top_feature_tests", []), 16)
    rollout_front_mode_source_corr = top_items(first_summary(rollout_front_mode_coupling, "top_source_residual_correlations", []), 16)
    rollout_front_mode_raw_corr = top_items(first_summary(rollout_front_mode_coupling, "top_raw_correlations", []), 12)
    rollout_front_mode_modes = top_items(first_summary(rollout_front_mode_coupling, "mode_summary", []), 8)
    rollout_front_mode_queue = top_items(first_summary(rollout_front_mode_coupling, "top_review_queue", []), 12)
    masked_rollout_methods = top_items(first_summary(masked_rollout, "method_summary", []), 8)
    masked_rollout_tests = top_items(first_summary(masked_rollout, "top_event_control_tests", []), 12)
    masked_rollout_corr = top_items(first_summary(masked_rollout, "top_correlations", []), 12)
    masked_rollout_difficult = top_items(first_summary(masked_rollout, "top_particle_difficulty_rois", []), 12)
    masked_rollout_best_counts = first_summary(masked_rollout, "best_method_counts_inside_particle", {}) or {}
    masked_cycle_warning_tests = top_items(first_summary(masked_cycle_warning, "top_target_tests", []), 12)
    masked_cycle_warning_corr = top_items(first_summary(masked_cycle_warning, "top_correlations", []), 12)
    masked_cycle_warning_top = top_items(first_summary(masked_cycle_warning, "top_warning_cycles", []), 12)
    masked_cycle_warning_targets = first_summary(masked_cycle_warning, "target_positive_counts", {}) or {}
    masked_residual_timing_align = top_items(first_summary(masked_residual_timing, "top_alignment_tests", []), 10)
    masked_residual_timing_tests = top_items(first_summary(masked_residual_timing, "top_event_control_tests", []), 12)
    masked_residual_timing_corr = top_items(first_summary(masked_residual_timing, "top_correlations", []), 12)
    masked_residual_timing_top = top_items(first_summary(masked_residual_timing, "top_near_transition_residual_rois", []), 12)
    masked_residual_transfer_tests = top_items(first_summary(masked_residual_transfer, "top_target_tests", []), 15)
    masked_residual_transfer_features = top_items(first_summary(masked_residual_transfer, "top_signature_features", []), 8)
    masked_residual_transfer_coeff = top_items(first_summary(masked_residual_transfer, "top_transfer_coefficients", []), 10)
    masked_residual_transfer_corr = top_items(first_summary(masked_residual_transfer, "top_correlations", []), 12)
    masked_residual_transfer_ranked = top_items(first_summary(masked_residual_transfer, "top_ranked_cycles", []), 12)
    masked_residual_transfer_temporal = top_items(first_summary(masked_residual_transfer, "temporal_block_auc", []), 30)
    masked_residual_transfer_anchor = first_summary(masked_residual_transfer, "anchor_loo_summary", {}) or {}
    masked_residual_transfer_future8 = next((r for r in masked_residual_transfer_tests if r.get("target") == "future_any_drop_within_8cycles" and r.get("score") == "transferred_masked_residual_signature"), {})
    masked_residual_transfer_pc2_future8 = next((r for r in masked_residual_transfer_tests if r.get("target") == "future_any_drop_within_8cycles" and r.get("score") == "cycle_state_pc2"), {})
    transfer_ranked_sampled = top_items(first_summary(transfer_ranked_reconstruction, "sampled_cycles", []), 12)
    transfer_ranked_top_rois = top_items(first_summary(transfer_ranked_reconstruction, "top_roi_rows", []), 12)
    transfer_ranked_sequence_cycles = first_summary(transfer_ranked_sequences, "cycle_summary", {}) or {}
    transfer_ranked_rollout_methods = top_items(first_summary(transfer_ranked_rollout, "method_summary", []), 8)
    transfer_ranked_rollout_difficult = top_items(first_summary(transfer_ranked_rollout, "top_particle_difficulty_rois", []), 12)
    transfer_ranked_rollout_best_counts = first_summary(transfer_ranked_rollout, "best_method_counts_inside_particle", {}) or {}
    transfer_ranked_front_group = top_items(first_summary(transfer_ranked_fronts, "group_summary", []), 12)
    transfer_ranked_front_top = top_items(first_summary(transfer_ranked_fronts, "top_threshold_robust_phase_rois", []), 12)
    transfer_ranked_front_target_tests = top_items(first_summary(transfer_ranked_front_physics, "top_target_tests", []), 12)
    transfer_ranked_front_corr = top_items(first_summary(transfer_ranked_front_physics, "top_correlations", []), 12)
    transfer_ranked_front_cycle_tests = top_items(first_summary(transfer_ranked_front_physics, "top_cycle_target_tests", []), 12)
    transfer_ranked_front_cycle_corr = top_items(first_summary(transfer_ranked_front_physics, "top_cycle_correlations", []), 12)
    transfer_ranked_front_review = top_items(first_summary(transfer_ranked_front_physics, "top_review_rois", []), 12)
    transfer_ranked_timing_align = top_items(first_summary(transfer_ranked_residual_timing, "top_alignment_tests", []), 12)
    transfer_ranked_timing_target = top_items(first_summary(transfer_ranked_residual_timing, "top_timing_target_tests", []), 18)
    transfer_ranked_timing_target_corr = top_items(first_summary(transfer_ranked_residual_timing, "top_timing_target_correlations", []), 12)
    transfer_ranked_timing_transition_corr = top_items(first_summary(transfer_ranked_residual_timing, "top_transition_correlations", []), 12)
    transfer_ranked_timing_top_rois = top_items(first_summary(transfer_ranked_residual_timing, "top_near_transition_residual_rois", []), 12)
    multicohort_oof = top_items(first_summary(multicohort_future_drop, "cycle_group_oof_metrics", []), 6)
    multicohort_null = first_summary(multicohort_future_drop, "permutation_null", {}) or {}
    multicohort_features = top_items(first_summary(multicohort_future_drop, "top_feature_tests", []), 12)
    multicohort_importance = top_items(first_summary(multicohort_future_drop, "top_feature_importance", []), 12)
    multicohort_leave = top_items(first_summary(multicohort_future_drop, "leave_cohort_eval", []), 8)
    active_qc_top = top_items(first_summary(active_learning_qc, "top_priority_rows", []), 12)
    active_qc_cycles = top_items(first_summary(active_learning_qc, "top_cycles", []), 12)
    active_qc_reasons = top_items(first_summary(active_learning_qc, "reason_counts", []), 12)
    active_qc_tiers = first_summary(active_learning_qc, "tier_counts", {}) or {}
    auto_qc_triage_tiers = top_items(first_summary(automatic_qc_triage, "tier_summary", []), 8)
    auto_qc_likely = top_items(first_summary(automatic_qc_triage, "top_likely_interpretable", []), 12)
    auto_qc_artifact = top_items(first_summary(automatic_qc_triage, "top_artifact_risk", []), 12)
    auto_qc_tests = top_items(first_summary(automatic_qc_triage, "top_feature_tests", []), 12)
    auto_qc_corr = top_items(first_summary(automatic_qc_triage, "top_correlations", []), 12)
    qc_decision_top = top_items(first_summary(qc_decision_ledger, "top_review_queue", []), 12)
    qc_decision_accept = top_items(first_summary(qc_decision_ledger, "top_possible_accept_queue", []), 8)
    qc_decision_artifact = top_items(first_summary(qc_decision_ledger, "top_artifact_queue", []), 8)
    qc_decision_cycles = top_items(first_summary(qc_decision_ledger, "cycle_summary", []), 8)
    qc_decision_actions = first_summary(qc_decision_ledger, "decision_action_counts", {}) or {}
    source_balanced_expansion_sources = top_items(first_summary(source_balanced_expansion, "source_coverage", []), 20)
    source_balanced_expansion_top = top_items(first_summary(source_balanced_expansion, "top_roi_rows", []), 12)
    source_balanced_rollout_roi_tests = top_items(first_summary(source_balanced_sequence_rollout, "top_roi_feature_tests", []), 30)
    source_balanced_rollout_cycle_tests = top_items(first_summary(source_balanced_sequence_rollout, "top_cycle_feature_tests", []), 20)
    source_balanced_rollout_sources = top_items(first_summary(source_balanced_sequence_rollout, "source_summary", []), 20)
    source_balanced_source_control_scalars = top_items(first_summary(source_balanced_sequence_source_control, "top_source_stratified_scalars", []), 12)
    source_balanced_source_control_models = top_items(first_summary(source_balanced_sequence_source_control, "top_source_heldout_models", []), 12)
    source_balanced_source_control_deltas = top_items(first_summary(source_balanced_sequence_source_control, "top_model_deltas_vs_context", []), 12)
    source_balanced_source_control_best_scalar = source_balanced_source_control_scalars[0] if source_balanced_source_control_scalars else {}
    source_balanced_source_control_best_model = source_balanced_source_control_models[0] if source_balanced_source_control_models else {}
    source_balanced_rollout_top16 = next((r for r in source_balanced_rollout_roi_tests if r.get("target") == "future_any_drop_within_16cycles"), {})
    source_balanced_rollout_top8 = next((r for r in source_balanced_rollout_roi_tests if r.get("target") == "future_any_drop_within_8cycles"), {})
    source_balanced_expansion_transport_tests = top_items(first_summary(source_balanced_expansion_transport_front, "top_feature_tests", []), 120)
    source_balanced_expansion_transport_sources = top_items(first_summary(source_balanced_expansion_transport_front, "source_summary_top", []), 14)
    source_balanced_expansion_transport_candidates = top_items(first_summary(source_balanced_expansion_transport_front, "top_candidates", []), 12)
    source_balanced_expansion_transport_best_future8 = next((r for r in source_balanced_expansion_transport_tests if r.get("target") == "future_any_drop_within_8cycles"), {})
    source_balanced_expansion_transport_best_future16 = next((r for r in source_balanced_expansion_transport_tests if r.get("target") == "future_any_drop_within_16cycles"), {})
    source_balanced_expansion_transport_top_candidate = source_balanced_expansion_transport_candidates[0] if source_balanced_expansion_transport_candidates else {}
    source_balanced_mask_front_roi_tests = top_items(first_summary(source_balanced_mask_front, "top_roi_feature_tests", []), 30)
    source_balanced_mask_front_cycle_tests = top_items(first_summary(source_balanced_mask_front, "top_cycle_feature_tests", []), 20)
    source_balanced_mask_front_sources = top_items(first_summary(source_balanced_mask_front, "source_summary", []), 20)
    source_balanced_mask_front_top16 = next((r for r in source_balanced_mask_front_roi_tests if r.get("target") == "future_any_drop_within_16cycles"), {})
    source_balanced_mask_front_resid_best = first_summary(source_balanced_mask_front_source_residual, "future16_source_residual_best", {}) or {}
    source_balanced_mask_front_rank_best = first_summary(source_balanced_mask_front_source_residual, "future16_within_source_rank_best", {}) or {}
    source_balanced_resdict_metrics = top_items(first_summary(source_balanced_residual_dictionary, "top_metrics", []), 24)
    source_balanced_resdict_roi_tests = top_items(first_summary(source_balanced_residual_dictionary, "top_roi_feature_tests", []), 30)
    source_balanced_resdict_cycle_tests = top_items(first_summary(source_balanced_residual_dictionary, "top_cycle_feature_tests", []), 20)
    source_balanced_resdict_cycle16 = next((r for r in source_balanced_resdict_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "cycleNo" and r.get("feature_set") == "residual_dictionary"), {})
    source_balanced_resdict_source16 = next((r for r in source_balanced_resdict_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "source_stem" and r.get("feature_set") == "residual_dictionary"), {})
    source_balanced_resdict_top_roi16 = next((r for r in source_balanced_resdict_roi_tests if r.get("target") == "future_any_drop_within_16cycles" and str(r.get("feature", "")).startswith("resdict_")), {})
    source_balanced_resdict_top_cycle16 = next((r for r in source_balanced_resdict_cycle_tests if r.get("target") == "future_any_drop_within_16cycles" and str(r.get("feature", "")).startswith("resdict_")), {})
    source_balanced_resdict_sr_best = first_summary(source_balanced_resdict_source_residual, "future16_source_residual_residual_dictionary_best", {}) or {}
    source_balanced_resdict_rank_best = first_summary(source_balanced_resdict_source_residual, "future16_within_source_rank_residual_dictionary_best", {}) or {}
    source_balanced_resdict_sr_transform_best = first_summary(source_balanced_resdict_source_residual, "future16_source_residual_best", {}) or {}
    source_balanced_resdict_norm_metrics = top_items(first_summary(source_balanced_resdict_normalized_readout, "top_metrics", []), 32)
    source_balanced_resdict_norm_best = first_summary(source_balanced_resdict_normalized_readout, "future16_leave_source_best", {}) or {}
    source_balanced_resdict_norm_raw_source16 = first_summary(source_balanced_resdict_normalized_readout, "future16_leave_source_raw_residual_dictionary", {}) or {}
    source_balanced_resdict_norm_sr_source16 = first_summary(source_balanced_resdict_normalized_readout, "future16_leave_source_source_residual_residual_dictionary", {}) or {}
    source_balanced_resdict_norm_rank_source16 = first_summary(source_balanced_resdict_normalized_readout, "future16_leave_source_within_source_rank_residual_dictionary", {}) or {}
    source_balanced_resdict_norm_cycle_single16 = next((r for r in source_balanced_resdict_norm_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "cycleNo" and r.get("feature_set") == "dictionary_recon_error_last_minus_first_source_residual"), {})
    source_balanced_resdict_norm_perm = first_summary(source_balanced_resdict_normalized_readout, "permutation_summary", {}) or {}
    source_balanced_temporal_best = top_items(first_summary(source_balanced_residual_temporal_specificity, "best_future_specific", []), 16)
    source_balanced_temporal_primary = first_summary(source_balanced_residual_temporal_specificity, "primary_source_residual_rows", []) or []
    source_balanced_temporal_primary8 = next((r for r in source_balanced_temporal_primary if r.get("window_cycles") == 8), {})
    source_balanced_temporal_primary16 = next((r for r in source_balanced_temporal_primary if r.get("window_cycles") == 16), {})
    source_balanced_temporal_shift16 = first_summary(source_balanced_residual_temporal_specificity, "primary_future16_shift_null", {}) or {}
    source_balanced_temporal_top = source_balanced_temporal_best[0] if source_balanced_temporal_best else {}
    source_balanced_future_specific_primary = first_summary(source_balanced_future_specific_residual, "primary_source_residual_subset_rows", []) or []
    source_balanced_future_specific_primary16_all = next((r for r in source_balanced_future_specific_primary if r.get("target_window") == 16 and r.get("subset") == "all_rows"), {})
    source_balanced_future_specific_primary16_clean = next((r for r in source_balanced_future_specific_primary if r.get("target_window") == 16 and r.get("subset") == "exclude_past16"), {})
    source_balanced_future_specific_primary16_pre = next((r for r in source_balanced_future_specific_primary if r.get("target_window") == 16 and r.get("subset") == "pre_first_event"), {})
    source_balanced_future_specific_best_clean = top_items(first_summary(source_balanced_future_specific_residual, "best_clean_future_subset_rows", []), 16)
    source_balanced_future_specific_model_deltas = top_items(first_summary(source_balanced_future_specific_residual, "top_model_deltas_vs_past_context", []), 16)
    source_balanced_future_specific_top_delta = source_balanced_future_specific_model_deltas[0] if source_balanced_future_specific_model_deltas else {}
    source_balanced_future_specific_top_clean = source_balanced_future_specific_best_clean[0] if source_balanced_future_specific_best_clean else {}
    source_balanced_pre_event_bins = first_summary(source_balanced_pre_event_sampling, "cycle_bin_counts", {}) or {}
    source_balanced_pre_event_roi_bins = first_summary(source_balanced_pre_event_sampling, "roi_bin_counts", {}) or {}
    source_balanced_pre_event_sources = top_items(first_summary(source_balanced_pre_event_sampling, "source_coverage", []), 16)
    source_balanced_pre_event_audit_bins = first_summary(source_balanced_pre_event_sequence_audit, "bin_counts", {}) or {}
    source_balanced_pre_event_audit_targets = first_summary(source_balanced_pre_event_sequence_audit, "target_positive_counts", {}) or {}
    source_balanced_pre_event_audit_models = top_items(first_summary(source_balanced_pre_event_sequence_audit, "top_model_metrics", []), 24)
    source_balanced_pre_event_audit_scalars = top_items(first_summary(source_balanced_pre_event_sequence_audit, "top_scalar_tests", []), 24)
    source_balanced_pre_event_near_source = next((r for r in source_balanced_pre_event_audit_models if r.get("target") == "target_near_pre_vs_rest" and r.get("group_col") == "source_stem" and r.get("feature_set") == "spatial"), {})
    source_balanced_pre_event_near_cycle = next((r for r in source_balanced_pre_event_audit_models if r.get("target") == "target_near_pre_vs_rest" and r.get("group_col") == "cycleNo" and r.get("feature_set") == "spatial"), {})
    source_balanced_pre_event_clean_cycle = next((r for r in source_balanced_pre_event_audit_models if r.get("target") == "target_pre16_clean_vs_post_control" and r.get("group_col") == "cycleNo" and r.get("feature_set") == "all_video"), {})
    source_balanced_pre_event_any_source = next((r for r in source_balanced_pre_event_audit_models if r.get("target") == "target_any_pre_vs_post_control" and r.get("group_col") == "source_stem" and r.get("feature_set") == "all_video"), {})
    source_balanced_pre_event_scalar_best = source_balanced_pre_event_audit_scalars[0] if source_balanced_pre_event_audit_scalars else {}
    source_balanced_pre_event_rollout_top = top_items(first_summary(source_balanced_pre_event_rollout, "top_roi_feature_tests", []), 12)
    source_balanced_pre_event_masked_rollout_methods = top_items(first_summary(source_balanced_pre_event_masked_rollout, "method_summary", []), 8)
    source_balanced_pre_event_masked_rollout_tests = top_items(first_summary(source_balanced_pre_event_masked_rollout, "top_event_tests", []), 12)
    source_balanced_pre_event_flow_tests = top_items(first_summary(source_balanced_pre_event_optical_flow, "top_event_tests", []), 12)
    source_balanced_pre_event_flow_method = (first_summary(source_balanced_pre_event_optical_flow, "method_summary", []) or [{}])[0]
    source_balanced_pre_event_mask_top = top_items(first_summary(source_balanced_pre_event_mask_front, "top_roi_feature_tests", []), 12)
    source_balanced_pre_event_readout_best = top_items(first_summary(source_balanced_pre_event_readout, "best_by_target_transform", []), 24)
    source_balanced_pre_event_readout_clean = top_items(first_summary(source_balanced_pre_event_readout, "best_source_residual_clean_pre_readouts", []), 8)
    source_balanced_pre_event_rollout_best = source_balanced_pre_event_rollout_top[0] if source_balanced_pre_event_rollout_top else {}
    source_balanced_pre_event_masked_rollout_best_method = first_summary(source_balanced_pre_event_masked_rollout, "best_method_by_median_particle_gain", {}) or {}
    source_balanced_pre_event_masked_rollout_best_test = source_balanced_pre_event_masked_rollout_tests[0] if source_balanced_pre_event_masked_rollout_tests else {}
    source_balanced_pre_event_flow_best_test = source_balanced_pre_event_flow_tests[0] if source_balanced_pre_event_flow_tests else {}
    source_balanced_pre_event_flow_sr_best = first_summary(source_balanced_pre_event_optical_flow, "best_source_residual_test", {}) or next((r for r in source_balanced_pre_event_flow_tests if r.get("transform") == "source_residual"), {})
    source_balanced_pre_event_mask_physics = [
        row for row in source_balanced_pre_event_mask_top
        if any(key in str(row.get("feature", "")) for key in ("masked_", "front_", "mask_", "apparent_diffusion"))
    ]
    source_balanced_pre_event_mask_best = source_balanced_pre_event_mask_physics[0] if source_balanced_pre_event_mask_physics else (source_balanced_pre_event_mask_top[0] if source_balanced_pre_event_mask_top else {})
    source_balanced_pre_event_clean_best = source_balanced_pre_event_readout_clean[0] if source_balanced_pre_event_readout_clean else {}
    source_balanced_pre_event_traj_physics = top_items(first_summary(source_balanced_pre_event_trajectory, "top_physics_toward_event_tests", []), 12)
    source_balanced_pre_event_traj_source = top_items(first_summary(source_balanced_pre_event_trajectory, "top_source_residual_event_distance_tests", []), 12)
    source_balanced_pre_event_traj_best = source_balanced_pre_event_traj_physics[0] if source_balanced_pre_event_traj_physics else {}
    source_balanced_pre_event_dir_best = top_items(first_summary(source_balanced_pre_event_directionality, "best_by_target_transform", []), 18)
    source_balanced_pre_event_dir_clock = top_items(first_summary(source_balanced_pre_event_directionality, "best_pre_event_clock_features", []), 12)
    source_balanced_pre_event_dir_asym = top_items(first_summary(source_balanced_pre_event_directionality, "best_pre_vs_post_clock_asymmetry", []), 12)
    source_balanced_pre_event_dir_clean_sr = next((r for r in source_balanced_pre_event_dir_best if r.get("target") == "clean_pre8_vs_post_control" and r.get("transform") == "source_residual"), {})
    source_balanced_pre_event_dir_near_raw = next((r for r in source_balanced_pre_event_dir_best if r.get("target") == "near_pre_vs_far_pre" and r.get("transform") == "raw"), {})
    source_balanced_pre_event_dir_clock_physics = [
        row for row in source_balanced_pre_event_dir_clock
        if row.get("transform") in {"source_residual", "within_source_rank"}
        and any(key in str(row.get("feature", "")) for key in ("apparent_diffusion", "front_", "masked_minus_background", "mask_"))
        and (row.get("pre_clock_rho") or 0) > 0
    ]
    source_balanced_pre_event_dir_clock_top = source_balanced_pre_event_dir_clock_physics[0] if source_balanced_pre_event_dir_clock_physics else (source_balanced_pre_event_dir_clock[0] if source_balanced_pre_event_dir_clock else {})
    source_balanced_pre_event_dir_asym_top = source_balanced_pre_event_dir_asym[0] if source_balanced_pre_event_dir_asym else {}
    source_balanced_pre_event_si_best = top_items(first_summary(source_balanced_pre_event_source_invariant, "best_by_target", []), 16)
    source_balanced_pre_event_si_low_source = top_items(first_summary(source_balanced_pre_event_source_invariant, "best_low_source_eta2_models", []), 12)
    source_balanced_pre_event_si_uni = top_items(first_summary(source_balanced_pre_event_source_invariant, "top_univariate_rows", []), 16)
    source_balanced_pre_event_si_clean = next((r for r in source_balanced_pre_event_si_best if r.get("target") == "clean_pre8_vs_post_control"), {})
    source_balanced_pre_event_si_near_far = next((r for r in source_balanced_pre_event_si_best if r.get("target") == "near_pre_vs_far_pre"), {})
    source_balanced_pre_event_si_low_clean = next((r for r in source_balanced_pre_event_si_low_source if r.get("target") == "clean_pre8_vs_post_control"), {})
    source_balanced_pre_event_review_top = top_items(first_summary(source_balanced_pre_event_review_packet, "top_candidates", []), 12)
    source_balanced_pre_event_review_top1 = source_balanced_pre_event_review_top[0] if source_balanced_pre_event_review_top else {}
    source_balanced_pre_event_review_reasons = first_summary(source_balanced_pre_event_review_packet, "review_reason_counts", {}) or {}
    source_balanced_pre_event_matched_top = top_items(first_summary(source_balanced_pre_event_matched_counterfactual, "top_physics_matched_tests", []), 16)
    source_balanced_pre_event_matched_pair_counts = first_summary(source_balanced_pre_event_matched_counterfactual, "pair_counts", {}) or {}
    source_balanced_pre_event_matched_same_source = first_summary(source_balanced_pre_event_matched_counterfactual, "same_source_pair_fraction", {}) or {}
    source_balanced_pre_event_matched_best = source_balanced_pre_event_matched_top[0] if source_balanced_pre_event_matched_top else {}
    source_balanced_pre_event_matched_front = next((r for r in source_balanced_pre_event_matched_top if r.get("feature") == "front_radius_q60_slope_px_per_norm_time"), {})
    source_balanced_pre_event_matched_diffusion = next((r for r in source_balanced_pre_event_matched_top if r.get("feature") == "apparent_diffusion_q70_um2_per_norm_time"), {})
    source_balanced_pre_event_ladder_top = top_items(first_summary(source_balanced_pre_event_same_source_ladder, "top_same_source_paired_tests", []), 20)
    source_balanced_pre_event_ladder_clock = top_items(first_summary(source_balanced_pre_event_same_source_ladder, "top_within_source_clock_tests", []), 12)
    source_balanced_pre_event_ladder_counts = first_summary(source_balanced_pre_event_same_source_ladder, "ladder_source_counts", {}) or {}
    source_balanced_pre_event_ladder_pair_counts = first_summary(source_balanced_pre_event_same_source_ladder, "pair_counts", {}) or {}
    source_balanced_pre_event_ladder_best = source_balanced_pre_event_ladder_top[0] if source_balanced_pre_event_ladder_top else {}
    source_balanced_pre_event_ladder_front = next((r for r in source_balanced_pre_event_ladder_top if r.get("feature") == "front_radius_q60_slope_px_per_norm_time"), {})
    source_balanced_pre_event_ladder_diffusion = next((r for r in source_balanced_pre_event_ladder_top if r.get("feature") == "apparent_diffusion_q70_um2_per_norm_time"), {})
    source_balanced_pre_event_ladder_clock_top = source_balanced_pre_event_ladder_clock[0] if source_balanced_pre_event_ladder_clock else {}
    pre_event_lattice_near_counts = first_summary(pre_event_source_lattice, "near_source_counts", {}) or {}
    pre_event_lattice_near_sources = top_items(first_summary(pre_event_source_lattice, "near_source_rows", []), 8)
    pre_event_lattice_far_controls = top_items(first_summary(pre_event_source_lattice, "candidate_far_control_sources", []), 8)
    pre_event_lattice_design = first_summary(pre_event_source_lattice, "recommended_design", []) or []
    source_balanced_pre_event_kymo_near_far = top_items(first_summary(source_balanced_pre_event_radial_kymograph, "top_near_vs_far_tests", []), 8)
    source_balanced_pre_event_kymo_clean = top_items(first_summary(source_balanced_pre_event_radial_kymograph, "top_clean_pre_vs_post_control_tests", []), 8)
    source_balanced_pre_event_kymo_review = top_items(first_summary(source_balanced_pre_event_radial_kymograph, "top_review_candidate_kymograph_features", []), 12)
    source_balanced_pre_event_kymo_near_far_top = source_balanced_pre_event_kymo_near_far[0] if source_balanced_pre_event_kymo_near_far else {}
    source_balanced_pre_event_kymo_clean_top = source_balanced_pre_event_kymo_clean[0] if source_balanced_pre_event_kymo_clean else {}
    source_balanced_pre_event_kymo_review_top = source_balanced_pre_event_kymo_review[0] if source_balanced_pre_event_kymo_review else {}
    source_balanced_pre_event_echem_corr = top_items(first_summary(source_balanced_pre_event_echem_front, "top_echem_correlations", []), 12)
    source_balanced_pre_event_echem_raw = top_items(first_summary(source_balanced_pre_event_echem_front, "top_raw_effects", []), 12)
    source_balanced_pre_event_echem_resid = top_items(first_summary(source_balanced_pre_event_echem_front, "top_echem_source_residual_effects", []), 12)
    source_balanced_pre_event_echem_fit = top_items(first_summary(source_balanced_pre_event_echem_front, "top_residual_fits", []), 12)
    source_balanced_pre_event_echem_corr_top = source_balanced_pre_event_echem_corr[0] if source_balanced_pre_event_echem_corr else {}
    source_balanced_pre_event_echem_raw_top = source_balanced_pre_event_echem_raw[0] if source_balanced_pre_event_echem_raw else {}
    source_balanced_pre_event_echem_resid_top = source_balanced_pre_event_echem_resid[0] if source_balanced_pre_event_echem_resid else {}
    source_balanced_pre_event_echem_fit_top = source_balanced_pre_event_echem_fit[0] if source_balanced_pre_event_echem_fit else {}
    source_balanced_pre_event_echem_matched_resid_top = top_items(first_summary(source_balanced_pre_event_echem_matched_residual, "top_source_echem_residual_matched_tests", []), 16)
    source_balanced_pre_event_echem_matched_resid_pair_counts = first_summary(source_balanced_pre_event_echem_matched_residual, "pair_counts", {}) or {}
    source_balanced_pre_event_echem_matched_resid_same_source = first_summary(source_balanced_pre_event_echem_matched_residual, "same_source_pair_fraction", {}) or {}
    source_balanced_pre_event_echem_matched_resid_best = source_balanced_pre_event_echem_matched_resid_top[0] if source_balanced_pre_event_echem_matched_resid_top else {}
    source_balanced_pre_event_echem_matched_resid_far = next((r for r in source_balanced_pre_event_echem_matched_resid_top if r.get("comparison") == "near_vs_far_pre"), {})
    source_balanced_pre_event_front_consensus_event = top_items(first_summary(source_balanced_pre_event_front_consensus, "top_event_tests", []), 16)
    source_balanced_pre_event_front_consensus_matched = top_items(first_summary(source_balanced_pre_event_front_consensus, "top_matched_tests", []), 16)
    source_balanced_pre_event_front_consensus_clock = top_items(first_summary(source_balanced_pre_event_front_consensus, "top_clock_tests", []), 8)
    source_balanced_pre_event_front_consensus_ranked = top_items(first_summary(source_balanced_pre_event_front_consensus, "top_ranked_candidates", []), 8)
    source_balanced_pre_event_front_consensus_event_best = source_balanced_pre_event_front_consensus_event[0] if source_balanced_pre_event_front_consensus_event else {}
    source_balanced_pre_event_front_consensus_matched_best = source_balanced_pre_event_front_consensus_matched[0] if source_balanced_pre_event_front_consensus_matched else {}
    source_balanced_pre_event_front_consensus_ranked_best = source_balanced_pre_event_front_consensus_ranked[0] if source_balanced_pre_event_front_consensus_ranked else {}
    source_balanced_pre_event_echem_matched_far_top = top_items(first_summary(source_balanced_pre_event_echem_matched_far, "top_paired_tests", []), 16)
    source_balanced_pre_event_echem_matched_far_pair_counts = first_summary(source_balanced_pre_event_echem_matched_far, "pair_counts", {}) or {}
    source_balanced_pre_event_echem_matched_far_control_sources = first_summary(source_balanced_pre_event_echem_matched_far, "control_source_counts", {}) or {}
    source_balanced_pre_event_echem_matched_far_best = source_balanced_pre_event_echem_matched_far_top[0] if source_balanced_pre_event_echem_matched_far_top else {}
    source_balanced_pre_event_echem_matched_far_front = next((r for r in source_balanced_pre_event_echem_matched_far_top if r.get("match_scheme") == "same_source_class_echem_context" and r.get("feature") == "front_radius_slope_px_per_norm_time"), {})
    source_balanced_pre_event_consensus_top = top_items(first_summary(source_balanced_pre_event_consensus_review, "top_candidates", []), 20)
    source_balanced_pre_event_consensus_tiers = first_summary(source_balanced_pre_event_consensus_review, "priority_tier_counts", {}) or {}
    source_balanced_pre_event_consensus_best = source_balanced_pre_event_consensus_top[0] if source_balanced_pre_event_consensus_top else {}
    source_balanced_pre_event_visual_top = top_items(first_summary(source_balanced_pre_event_consensus_visual, "rendered_candidates", []), 12)
    source_balanced_pre_event_visual_best = source_balanced_pre_event_visual_top[0] if source_balanced_pre_event_visual_top else {}
    source_balanced_pre_event_visual_outputs = first_summary(source_balanced_pre_event_consensus_visual, "outputs", {}) or {}
    source_balanced_pre_event_visual_sanity_reviewable = top_items(first_summary(source_balanced_pre_event_visual_sanity, "top_reviewable_candidates", []), 12)
    source_balanced_pre_event_visual_sanity_artifact = top_items(first_summary(source_balanced_pre_event_visual_sanity, "top_artifact_risk_candidates", []), 8)
    source_balanced_pre_event_visual_sanity_sources = top_items(first_summary(source_balanced_pre_event_visual_sanity, "source_summary", []), 8)
    source_balanced_pre_event_visual_sanity_flags = first_summary(source_balanced_pre_event_visual_sanity, "visual_sanity_flag_counts", {}) or {}
    source_balanced_pre_event_visual_sanity_best = source_balanced_pre_event_visual_sanity_reviewable[0] if source_balanced_pre_event_visual_sanity_reviewable else {}
    source_balanced_pre_event_visual_qc_top = top_items(first_summary(source_balanced_pre_event_visual_qc_modes, "top_candidates", []), 12)
    source_balanced_pre_event_visual_qc_modes_summary = top_items(first_summary(source_balanced_pre_event_visual_qc_modes, "mode_summary", []), 8)
    source_balanced_pre_event_visual_qc_tiers = first_summary(source_balanced_pre_event_visual_qc_modes, "visual_qc_tier_counts", {}) or {}
    source_balanced_pre_event_visual_qc_mode_counts = first_summary(source_balanced_pre_event_visual_qc_modes, "visual_mode_counts", {}) or {}
    source_balanced_pre_event_visual_qc_best = source_balanced_pre_event_visual_qc_top[0] if source_balanced_pre_event_visual_qc_top else {}
    source_balanced_pre_event_phase_kinetics_event = top_items(first_summary(source_balanced_pre_event_phase_kinetics, "top_event_tests", []), 16)
    source_balanced_pre_event_phase_kinetics_matched = top_items(first_summary(source_balanced_pre_event_phase_kinetics, "top_matched_tests", []), 16)
    source_balanced_pre_event_phase_kinetics_corr = top_items(first_summary(source_balanced_pre_event_phase_kinetics, "top_correlations", []), 12)
    source_balanced_pre_event_phase_kinetics_sources = top_items(first_summary(source_balanced_pre_event_phase_kinetics, "source_summary", []), 8)
    source_balanced_pre_event_phase_kinetics_event_best = source_balanced_pre_event_phase_kinetics_event[0] if source_balanced_pre_event_phase_kinetics_event else {}
    source_balanced_pre_event_phase_kinetics_matched_best = source_balanced_pre_event_phase_kinetics_matched[0] if source_balanced_pre_event_phase_kinetics_matched else {}
    source_balanced_pre_event_fk_concordance_top = top_items(first_summary(source_balanced_pre_event_front_kinetic_concordance, "top_candidates", []), 16)
    source_balanced_pre_event_fk_concordance_tests = top_items(first_summary(source_balanced_pre_event_front_kinetic_concordance, "top_event_tests", []), 12)
    source_balanced_pre_event_fk_concordance_corr = top_items(first_summary(source_balanced_pre_event_front_kinetic_concordance, "top_correlations", []), 12)
    source_balanced_pre_event_fk_concordance_tiers = first_summary(source_balanced_pre_event_front_kinetic_concordance, "tier_counts", {}) or {}
    source_balanced_pre_event_fk_concordance_best = source_balanced_pre_event_fk_concordance_top[0] if source_balanced_pre_event_fk_concordance_top else {}
    source_balanced_pre_event_fk_concordance_test_best = source_balanced_pre_event_fk_concordance_tests[0] if source_balanced_pre_event_fk_concordance_tests else {}
    source_balanced_pre_event_fk_null_tests = top_items(first_summary(source_balanced_pre_event_front_kinetic_null, "top_null_tests", []), 16)
    source_balanced_pre_event_fk_null_proximity = top_items(first_summary(source_balanced_pre_event_front_kinetic_null, "top_proximity_tests", []), 12)
    source_balanced_pre_event_fk_null_best = source_balanced_pre_event_fk_null_tests[0] if source_balanced_pre_event_fk_null_tests else {}
    source_balanced_pre_event_fk_null_prox_best = source_balanced_pre_event_fk_null_proximity[0] if source_balanced_pre_event_fk_null_proximity else {}
    source_balanced_pre_event_manual_qc_top = top_items(first_summary(source_balanced_pre_event_manual_qc_decision, "top_candidates", []), 20)
    source_balanced_pre_event_manual_qc_actions = first_summary(source_balanced_pre_event_manual_qc_decision, "action_counts", {}) or {}
    source_balanced_pre_event_manual_qc_top40_actions = first_summary(source_balanced_pre_event_manual_qc_decision, "top40_action_counts", {}) or {}
    source_balanced_pre_event_manual_qc_best = source_balanced_pre_event_manual_qc_top[0] if source_balanced_pre_event_manual_qc_top else {}
    source_balanced_pre_event_manual_qc_visual_top = top_items(first_summary(source_balanced_pre_event_manual_qc_visual, "rendered_candidates", []), 12)
    source_balanced_pre_event_manual_qc_visual_actions = first_summary(source_balanced_pre_event_manual_qc_visual, "action_tier_counts_rendered", {}) or {}
    source_balanced_pre_event_manual_qc_visual_bins = first_summary(source_balanced_pre_event_manual_qc_visual, "event_relative_bin_counts_rendered", {}) or {}
    source_balanced_pre_event_manual_qc_visual_best = source_balanced_pre_event_manual_qc_visual_top[0] if source_balanced_pre_event_manual_qc_visual_top else {}
    source_balanced_pre_event_manual_qc_blind_actions = first_summary(source_balanced_pre_event_manual_qc_blind, "action_tier_counts_hidden_key", {}) or {}
    source_balanced_pre_event_manual_qc_blind_bins = first_summary(source_balanced_pre_event_manual_qc_blind, "event_relative_bin_counts_hidden_key", {}) or {}
    source_balanced_pre_event_multimodal_best = top_items(first_summary(source_balanced_pre_event_multimodal, "best_by_target", []), 24)
    source_balanced_pre_event_multimodal_deltas = top_items(first_summary(source_balanced_pre_event_multimodal, "best_family_deltas", []), 16)
    source_balanced_pre_event_multimodal_best_row = source_balanced_pre_event_multimodal_best[0] if source_balanced_pre_event_multimodal_best else {}
    source_balanced_pre_event_multimodal_delta_best = source_balanced_pre_event_multimodal_deltas[0] if source_balanced_pre_event_multimodal_deltas else {}
    source_balanced_pre_event_strict_qc_top = top_items(first_summary(source_balanced_pre_event_strict_qc_gated_front, "top_manual_front_review_candidates", []), 12)
    source_balanced_pre_event_strict_qc_ranked = top_items(first_summary(source_balanced_pre_event_strict_qc_gated_front, "top_strict_qc_ranked_candidates", []), 12)
    source_balanced_pre_event_strict_qc_gate_pass = first_summary(source_balanced_pre_event_strict_qc_gated_front, "gate_pass_counts", {}) or {}
    source_balanced_pre_event_strict_qc_best = source_balanced_pre_event_strict_qc_top[0] if source_balanced_pre_event_strict_qc_top else (source_balanced_pre_event_strict_qc_ranked[0] if source_balanced_pre_event_strict_qc_ranked else {})
    source_balanced_pre_event_mode_summary = top_items(first_summary(source_balanced_pre_event_physics_modes, "mode_summary", []), 12)
    source_balanced_pre_event_mode_enrich = top_items(first_summary(source_balanced_pre_event_physics_modes, "top_enrichment", []), 16)
    source_balanced_pre_event_mode_clocks = top_items(first_summary(source_balanced_pre_event_physics_modes, "clock_tests", []), 8)
    source_balanced_pre_event_mode_k = top_items(first_summary(source_balanced_pre_event_physics_modes, "k_scores", []), 6)
    source_balanced_pre_event_mode_best = source_balanced_pre_event_mode_summary[0] if source_balanced_pre_event_mode_summary else {}
    source_balanced_pre_event_mode_enrich_best = source_balanced_pre_event_mode_enrich[0] if source_balanced_pre_event_mode_enrich else {}
    source_balanced_degmode_clusters = top_items(first_summary(source_balanced_degradation_modes, "cluster_summary", []), 12)
    source_balanced_degmode_enrichment = top_items(first_summary(source_balanced_degradation_modes, "top_enrichment", []), 24)
    source_balanced_degmode_representatives = top_items(first_summary(source_balanced_degradation_modes, "representatives", []), 12)
    source_balanced_degmode_top = source_balanced_degmode_enrichment[0] if source_balanced_degmode_enrichment else {}
    source_balanced_resphys_by_transform = top_items(first_summary(source_balanced_residual_physics_coupling, "best_by_transform", []), 8)
    source_balanced_resphys_primary = top_items(first_summary(source_balanced_residual_physics_coupling, "best_source_residual_primary_candidate_correlations", []), 12)
    source_balanced_resphys_aligned = top_items(first_summary(source_balanced_residual_physics_coupling, "best_source_residual_target_aligned_pairs", []), 12)
    source_balanced_resphys_dict_recon = top_items(first_summary(source_balanced_residual_physics_coupling, "dictionary_recon_error_last_minus_first_source_residual_top_correlations", []), 12)
    source_balanced_resphys_top_aligned = source_balanced_resphys_aligned[0] if source_balanced_resphys_aligned else {}
    source_balanced_resphys_top_primary = source_balanced_resphys_primary[0] if source_balanced_resphys_primary else {}
    source_balanced_review_top = top_items(first_summary(source_balanced_residual_candidate_review, "top_review_candidates", []), 16)
    source_balanced_review_immediate = top_items(first_summary(source_balanced_residual_candidate_review, "immediate_manual_qc_candidates", []), 16)
    source_balanced_review_sources = top_items(first_summary(source_balanced_residual_candidate_review, "source_summary", []), 16)
    source_balanced_review_tiers = first_summary(source_balanced_residual_candidate_review, "review_tier_counts", {}) or {}
    source_balanced_review_top1 = source_balanced_review_top[0] if source_balanced_review_top else {}
    balanced_future_oof = top_items(first_summary(balanced_future_physics, "cycle_group_oof_metrics", []), 6)
    balanced_future_null = first_summary(balanced_future_physics, "permutation_null", {}) or {}
    balanced_future_roi_tests = top_items(first_summary(balanced_future_physics, "top_roi_feature_tests", []), 12)
    balanced_future_cycle_tests = top_items(first_summary(balanced_future_physics, "top_cycle_feature_tests", []), 12)
    balanced_future_corr = top_items(first_summary(balanced_future_physics, "top_correlations", []), 12)
    balanced_future_best_oof = max(balanced_future_oof, key=lambda r: (float(r.get("pooled_oof_roc_auc") or float("nan")) if r.get("pooled_oof_roc_auc") is not None else float("nan"))) if balanced_future_oof else {}
    balanced_future_rollout_best_counts = first_summary(balanced_future_rollout, "best_method_counts_inside_particle", {}) or {}
    balanced_future_mask_overall = first_summary(balanced_future_mask, "overall", {}) or {}
    balanced_future_mask_roles = top_items(first_summary(balanced_future_mask, "role_summary", []), 6)
    balanced_future_mask_tests = top_items(first_summary(balanced_future_mask, "top_future_label_tests", []), 8)
    balanced_future_mask_top_test = balanced_future_mask_tests[0] if balanced_future_mask_tests else {}
    balanced_future_context_oof = top_items(first_summary(balanced_future_context, "cycle_group_oof_metrics", []), 12)
    balanced_future_best_acq_context = first_summary(balanced_future_context, "best_acquisition_context_only", {}) or {}
    balanced_future_best_design_context = first_summary(balanced_future_context, "best_design_context_only", {}) or {}
    balanced_future_best_context_physics = first_summary(balanced_future_context, "best_physics_plus_acquisition_context", {}) or {}
    balanced_future_acq_resid = top_items(first_summary(balanced_future_context, "top_acquisition_context_residual_feature_tests", []), 10)
    balanced_future_spatial_tests = top_items(first_summary(balanced_future_context, "spatial_region_tests", []), 8)
    masked_video_metrics = top_items(first_summary(masked_video_embedding, "target_metrics", []), 8)
    masked_video_future_metric = next((r for r in masked_video_metrics if r.get("target") == "future_any_drop_within_8cycles"), {})
    masked_video_event_metric = next((r for r in masked_video_metrics if r.get("target") == "event_vs_control"), {})
    masked_video_null = first_summary(masked_video_embedding, "balanced_future_label_permutation_null", {}) or {}
    masked_video_feature_tests = top_items(first_summary(masked_video_embedding, "top_feature_tests", []), 12)
    masked_video_clusters = top_items(first_summary(masked_video_embedding, "cluster_summary", []), 8)
    learned_residual_class = top_items(first_summary(learned_video_residual_embedding, "top_classification_metrics", []), 12)
    learned_residual_reg = top_items(first_summary(learned_video_residual_embedding, "top_regression_metrics", []), 8)
    learned_residual_deltas = top_items(first_summary(learned_video_residual_embedding, "top_feature_set_deltas", []), 12)
    learned_residual_future8 = next((r for r in learned_residual_class if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "learned_all"), {})
    learned_residual_hand_future8 = next((r for r in learned_residual_class if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "handcrafted_scalar"), {})
    learned_residual_pca_future8 = next((r for r in learned_residual_class if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "pca_video"), {})
    learned_residual_future16 = next((r for r in learned_residual_class if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "learned_all"), {})
    learned_residual_hand_future16 = next((r for r in learned_residual_class if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "handcrafted_scalar"), {})
    residual_dict_class = top_items(first_summary(residual_dictionary_embedding, "top_classification_metrics", []), 8)
    residual_dict_reg = top_items(first_summary(residual_dictionary_embedding, "top_regression_metrics", []), 8)
    residual_dict_deltas = top_items(first_summary(residual_dictionary_embedding, "top_feature_set_deltas", []), 8)
    agentic_current_top = top_items(first_summary(agentic_current, "top_three", []), 3)
    agentic_current_specs = top_items(first_summary(agentic_current, "experiment_specs", []), 5)
    residual_dict_future8 = next((r for r in residual_dict_class if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "residual_dictionary"), {})
    residual_dict_plus_future8 = next((r for r in residual_dict_class if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "residual_dictionary_plus_handcrafted"), {})
    echem_resdict_class = top_items(first_summary(echem_residual_dictionary_fusion, "top_classification_metrics", []), 10)
    echem_resdict_reg = top_items(first_summary(echem_residual_dictionary_fusion, "top_regression_metrics", []), 8)
    echem_resdict_deltas = top_items(first_summary(echem_residual_dictionary_fusion, "top_feature_set_deltas", []), 10)
    echem_resdict_acq_future8 = next((r for r in echem_resdict_class if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "acquisition_context"), {})
    echem_resdict_res_future8 = next((r for r in echem_resdict_class if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "residual_dictionary_plus_echem"), {})
    echem_cond_resdict_metrics = top_items(first_summary(echem_conditioned_residual_dictionary, "top_metrics", []), 40)
    echem_cond_resdict_deltas = top_items(first_summary(echem_conditioned_residual_dictionary, "top_deltas", []), 30)
    echem_cond_resdict_context = top_items(first_summary(echem_conditioned_residual_dictionary, "top_context_fit_metrics", []), 20)
    echem_cond_resdict_lc16 = next((r for r in echem_cond_resdict_metrics if r.get("split") == "leave_cycle" and r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "conditioned_residual_dictionary"), {})
    echem_cond_resdict_ls16 = next((r for r in echem_cond_resdict_metrics if r.get("split") == "leave_source" and r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "conditioned_residual_dictionary"), {})
    echem_cond_resdict_lc16_plus = next((r for r in echem_cond_resdict_metrics if r.get("split") == "leave_cycle" and r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "conditioned_residual_plus_echem_context"), {})
    echem_cond_resdict_ls16_plus = next((r for r in echem_cond_resdict_metrics if r.get("split") == "leave_source" and r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "conditioned_residual_plus_echem_context"), {})
    echem_cond_resdict_raw_ls16 = next((r for r in echem_cond_resdict_metrics if r.get("split") == "leave_source" and r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "residual_dictionary_raw"), {})
    echem_cond_resdict_delta_ls16 = next((r for r in echem_cond_resdict_deltas if r.get("split") == "leave_source" and r.get("target") == "future_any_drop_within_16cycles" and r.get("comparison") == "conditioned_residual_dictionary_minus_residual_dictionary_raw"), {})
    cond_atlas_align = top_items(first_summary(conditioned_residual_physics_atlas, "top_interpretable_source_centered_alignments", []), 30)
    cond_atlas_targets = top_items(first_summary(conditioned_residual_physics_atlas, "top_target_tests", []), 30)
    cond_atlas_modes = top_items(first_summary(conditioned_residual_physics_atlas, "top_modes", []), 30)
    cond_atlas_categories = top_items(first_summary(conditioned_residual_physics_atlas, "category_summary", []), 20)
    cond_atlas_top_align = cond_atlas_align[0] if cond_atlas_align else {}
    cond_atlas_top_target = cond_atlas_targets[0] if cond_atlas_targets else {}
    acq_resid_metrics = top_items(first_summary(acquisition_residualized_video, "top_metrics", []), 40)
    acq_resid_deltas = top_items(first_summary(acquisition_residualized_video, "top_feature_set_deltas", []), 14)
    acq_resid_tests = top_items(first_summary(acquisition_residualized_video, "top_context_residual_feature_tests", []), 12)
    acq_resid_context8 = next((r for r in acq_resid_metrics if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "acquisition_context"), {})
    acq_resid_raw_video8 = next((r for r in acq_resid_metrics if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "all_video_raw"), {})
    acq_resid_video_only8 = next((r for r in acq_resid_metrics if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "residualized_all_video"), {})
    acq_resid_video_ctx8 = next((r for r in acq_resid_metrics if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "residualized_all_video_plus_context_logit"), {})
    acq_resid_context16 = next((r for r in acq_resid_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "acquisition_context"), {})
    acq_resid_raw_hand16 = next((r for r in acq_resid_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "handcrafted_particle_raw"), {})
    acq_resid_video_only16 = next((r for r in acq_resid_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "residualized_all_video"), {})
    acq_echem_metrics = top_items(first_summary(acquisition_residualized_video_echem, "top_metrics", []), 48)
    acq_echem_deltas = top_items(first_summary(acquisition_residualized_video_echem, "top_deltas", []), 40)
    future8_decision = first_summary(residualized_future8_video_physics, "decision", {}) or {}
    future8_top_metrics = top_items(first_summary(residualized_future8_video_physics, "top_metrics", []), 24)
    future8_key_deltas = top_items(first_summary(residualized_future8_video_physics, "key_deltas", []), 24)
    future8_strict_optical_source = future8_decision.get("strict_optical_source_stem_metric", {}) or {}
    future8_strict_optical_cohort = future8_decision.get("strict_optical_source_cohort_metric", {}) or {}
    future8_fused_cohort = future8_decision.get("fused_source_cohort_metric", {}) or {}
    future8_echem_cohort = future8_decision.get("echem_source_cohort_metric", {}) or {}
    future8_acq_cohort = future8_decision.get("acquisition_source_cohort_metric", {}) or {}
    observable_forecast_decision = first_summary(source_balanced_pre_event_observable_forecast, "decision", {}) or {}
    observable_forecast_best = observable_forecast_decision.get("best_leave_source_metric", {}) or {}
    observable_forecast_incremental_best = observable_forecast_decision.get("best_incremental_over_echem", {}) or {}
    observable_forecast_source_metrics = top_items(first_summary(source_balanced_pre_event_observable_forecast, "source_heldout_top_metrics", []), 12)
    observable_forecast_event_diag = top_items(first_summary(source_balanced_pre_event_observable_forecast, "event_relative_diagnostics", []), 12)
    observable_forecast_incremental = top_items(first_summary(source_balanced_pre_event_observable_forecast, "incremental_over_echem", []), 12)
    optical_flow_top_tests = top_items(first_summary(source_balanced_pre_event_optical_flow_transport, "top_event_tests", []), 16)
    optical_flow_source_resid_best = first_summary(source_balanced_pre_event_optical_flow_transport, "best_source_residual_test", {}) or next((r for r in optical_flow_top_tests if r.get("transform") == "source_residual"), {})
    optical_flow_source_resid_tests = [optical_flow_source_resid_best] if optical_flow_source_resid_best else []
    optical_flow_method_summary = (first_summary(source_balanced_pre_event_optical_flow_transport, "method_summary", []) or [{}])[0]
    optical_flow_best = optical_flow_top_tests[0] if optical_flow_top_tests else {}
    transport_fusion_tests = top_items(first_summary(source_balanced_pre_event_transport_kinetic_fusion, "top_event_tests", []), 16)
    transport_fusion_best_by_target = top_items(first_summary(source_balanced_pre_event_transport_kinetic_fusion, "best_event_tests_by_target", []), 8)
    transport_fusion_models = top_items(first_summary(source_balanced_pre_event_transport_kinetic_fusion, "top_leave_source_models", []), 12)
    transport_fusion_candidates = top_items(first_summary(source_balanced_pre_event_transport_kinetic_fusion, "top_ranked_candidates", []), 12)
    transport_fusion_best = transport_fusion_tests[0] if transport_fusion_tests else {}
    transport_fusion_near_post = next((r for r in transport_fusion_best_by_target if r.get("target") == "near_vs_post_control"), {})
    transport_fusion_top_model = transport_fusion_models[0] if transport_fusion_models else {}
    transport_fusion_top_candidate = transport_fusion_candidates[0] if transport_fusion_candidates else {}
    transport_mechanism_top = first_summary(source_balanced_transport_mechanism, "top_candidate", {}) or {}
    transport_mechanism_tiers = top_items(first_summary(source_balanced_transport_mechanism, "tier_counts", []), 12)
    transport_mechanism_sources = top_items(first_summary(source_balanced_transport_mechanism, "source_summary_top", []), 12)
    transport_mechanism_falsification_event = top_items(first_summary(source_balanced_transport_mechanism_falsification, "lead_event_tests_for_transport_mechanism_score", []), 3)
    transport_mechanism_falsification_pair = top_items(first_summary(source_balanced_transport_mechanism_falsification, "lead_pair_tests_for_transport_mechanism_score", []), 3)
    transport_mechanism_falsification_source = first_summary(source_balanced_transport_mechanism_falsification, "lead_source_test_for_transport_mechanism_score", {}) or {}
    transport_mechanism_falsification_topk = top_items(first_summary(source_balanced_transport_mechanism_falsification, "topk_enrichment", []), 4)
    heldout_rank_transfer_tests = top_items(first_summary(source_heldout_event_rank_transfer, "best_score_tests_by_target", []), 8)
    heldout_rank_transfer_transfer_tests = top_items(first_summary(source_heldout_event_rank_transfer, "transfer_score_tests", []), 4)
    heldout_rank_transfer_topk = top_items(first_summary(source_heldout_event_rank_transfer, "topk_summary", []), 12)
    temporal_dose_key_tests = top_items(first_summary(pre_event_temporal_dose_response, "key_feature_tests", []), 8)
    temporal_dose_centered = top_items(first_summary(pre_event_temporal_dose_response, "top_source_centered_tests", []), 8)
    temporal_dose_slopes = top_items(first_summary(pre_event_temporal_dose_response, "top_source_slope_tests", []), 8)
    targeted_densification_cycle_actions = top_items(first_summary(targeted_densification_qc, "top_cycle_actions", []), 12)
    targeted_densification_roi_actions = top_items(first_summary(targeted_densification_qc, "top_roi_actions", []), 12)
    targeted_densification_source_plan = top_items(first_summary(targeted_densification_qc, "source_plan_top", []), 10)
    targeted_densification_action_counts = top_items(first_summary(targeted_densification_qc, "action_counts", []), 10)
    targeted_densification_origin_counts = top_items(first_summary(targeted_densification_qc, "roi_origin_counts", []), 8)
    acq_echem_cycle_future16 = next((r for r in acq_echem_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "cycleNo" and r.get("feature_set") == "video_plus_echem" and r.get("mode") == "acquisition_residualized"), {})
    acq_echem_cycle_bal_future16 = next((r for r in acq_echem_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "cycleNo" and r.get("feature_set") == "video_plus_echem" and r.get("mode") == "acquisition_residualized_cycle_balanced"), {})
    acq_echem_cycle_acq_future16 = next((r for r in acq_echem_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "cycleNo" and r.get("feature_set") == "acquisition_context" and r.get("mode") == "raw"), {})
    acq_echem_source_future16 = next((r for r in acq_echem_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "source_stem" and r.get("feature_set") == "video_plus_echem" and r.get("mode") == "acquisition_residualized"), {})
    acq_echem_source_acq_future16 = next((r for r in acq_echem_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "source_stem" and r.get("feature_set") == "acquisition_context" and r.get("mode") == "raw"), {})
    acq_echem_sourcecohort_future16 = next((r for r in acq_echem_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "source_cohort_key" and r.get("feature_set") == "video_plus_echem" and r.get("mode") == "acquisition_residualized_cycle_balanced"), {})
    acq_echem_sourcecohort_acq_future16 = next((r for r in acq_echem_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("group_col") == "source_cohort_key" and r.get("feature_set") == "acquisition_context" and r.get("mode") == "raw"), {})
    source_domain_metrics = top_items(first_summary(source_domain_video_echem, "top_metrics", []), 24)
    source_domain_deltas = top_items(first_summary(source_domain_video_echem, "top_deltas", []), 24)
    source_domain_sources = top_items(first_summary(source_domain_video_echem, "source_summary", []), 12)
    source_domain_acq = next((r for r in source_domain_metrics if r.get("feature_set") == "acquisition_context" and r.get("method") == "raw"), {})
    source_domain_best = source_domain_metrics[0] if source_domain_metrics else {}
    source_domain_centered = next((r for r in source_domain_metrics if r.get("feature_set") == "video_plus_echem" and r.get("method") == "source_centered"), {})
    source_domain_coral = next((r for r in source_domain_metrics if r.get("feature_set") == "video_plus_echem" and r.get("method") == "coral"), {})
    source_balanced_metrics = top_items(first_summary(source_balanced_video_echem, "top_metrics", []), 48)
    source_balanced_deltas = top_items(first_summary(source_balanced_video_echem, "top_deltas", []), 24)
    source_balanced_sources = top_items(first_summary(source_balanced_video_echem, "source_summary", []), 12)
    source_balanced_acq16 = next((r for r in source_balanced_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "acquisition_context" and r.get("mode") == "raw_unweighted"), {})
    source_balanced_raw16 = next((r for r in source_balanced_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "video_plus_echem" and r.get("mode") == "raw_unweighted"), {})
    source_balanced_vpe16 = next((r for r in source_balanced_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "video_plus_echem" and r.get("mode") == "source_rank_weighted"), {})
    source_balanced_echem16 = next((r for r in source_balanced_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "echem_regime" and r.get("mode") == "source_rank_weighted"), {})
    source_balanced_rank8 = next((r for r in source_balanced_metrics if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "video_plus_echem" and r.get("mode") == "source_rank_weighted"), {})
    source_invariant_metrics = top_items(first_summary(source_invariant_video_echem, "top_metrics", []), 64)
    source_invariant_deltas = top_items(first_summary(source_invariant_video_echem, "top_deltas", []), 32)
    source_invariant_sources = top_items(first_summary(source_invariant_video_echem, "source_summary", []), 12)
    source_invariant_acq16 = next((r for r in source_invariant_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "acquisition_context" and r.get("method") == "raw"), {})
    source_invariant_raw16 = next((r for r in source_invariant_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "video_plus_echem" and r.get("method") == "raw"), {})
    source_invariant_best_vpe16 = max([r for r in source_invariant_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "video_plus_echem"], key=lambda r: (float(r.get("roc_auc") or float("nan")) if r.get("roc_auc") is not None else float("nan")), default={})
    source_invariant_best_video16 = max([r for r in source_invariant_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_set") == "video_all"], key=lambda r: (float(r.get("roc_auc") or float("nan")) if r.get("roc_auc") is not None else float("nan")), default={})
    source_invariant_best_vpe8 = max([r for r in source_invariant_metrics if r.get("target") == "future_any_drop_within_8cycles" and r.get("feature_set") == "video_plus_echem"], key=lambda r: (float(r.get("roc_auc") or float("nan")) if r.get("roc_auc") is not None else float("nan")), default={})
    source_family_metrics = top_items(first_summary(source_invariant_family, "top_metrics", []), 64)
    source_family_deltas = top_items(first_summary(source_invariant_family, "top_deltas", []), 32)
    source_family_confounds = top_items(first_summary(source_invariant_family, "top_source_confounded_features", []), 16)
    source_family_best16 = max([r for r in source_family_metrics if r.get("target") == "future_any_drop_within_16cycles"], key=lambda r: (float(r.get("roc_auc") or float("nan")) if r.get("roc_auc") is not None else float("nan")), default={})
    source_family_norm16 = next((r for r in source_family_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_family") == "norm_heterogeneity" and r.get("method") == "source_mean_resid_4"), {})
    source_family_contrast16 = next((r for r in source_family_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_family") == "particle_vs_context" and r.get("method") == "source_mean_resid_4"), {})
    source_family_embed16 = next((r for r in source_family_metrics if r.get("target") == "future_any_drop_within_16cycles" and r.get("feature_family") == "video_embedding" and r.get("method") == "raw"), {})
    source_family_best8 = max([r for r in source_family_metrics if r.get("target") == "future_any_drop_within_8cycles"], key=lambda r: (float(r.get("roc_auc") or float("nan")) if r.get("roc_auc") is not None else float("nan")), default={})
    source_interpretable_univariate = top_items(first_summary(source_invariant_interpretable, "top_univariate_features", []), 30)
    source_interpretable_single = top_items(first_summary(source_invariant_interpretable, "top_single_feature_transfer", []), 20)
    source_interpretable_combo = top_items(first_summary(source_invariant_interpretable, "top_combo_transfer", []), 20)
    source_interpretable_sets = top_items(first_summary(source_invariant_interpretable, "top_set_metrics", []), 40)
    source_interpretable_top_uni = source_interpretable_univariate[0] if source_interpretable_univariate else {}
    source_interpretable_top_single = source_interpretable_single[0] if source_interpretable_single else {}
    source_interpretable_top_combo = source_interpretable_combo[0] if source_interpretable_combo else {}
    exact_mech_target_metrics = top_items(first_summary(exact_feature_mechanism, "top_target_metrics", []), 20)
    exact_mech_correlations = top_items(first_summary(exact_feature_mechanism, "top_mechanism_correlations", []), 30)
    exact_mech_contraction = top_items(first_summary(exact_feature_mechanism, "contraction_related_correlations", []), 20)
    exact_mech_strata = top_items(first_summary(exact_feature_mechanism, "top_stratum_tests", []), 20)
    exact_mech_loss_metric = next((r for r in exact_mech_target_metrics if r.get("score") == "exact_optical_loss_score"), {})
    exact_mech_context_metric = next((r for r in exact_mech_target_metrics if r.get("score") == "exact_low_context_change_score"), {})
    exact_mech_contraction_metric = next((r for r in exact_mech_target_metrics if r.get("score") == "front_contraction_score"), {})
    exact_mech_radius_corr = next((r for r in exact_mech_contraction if r.get("anchor_feature") == "exact_optical_loss_score" and r.get("mechanism_feature") == "radius2_slope_median_px2_per_s"), {})
    exact_mech_primary_radius_corr = next((r for r in exact_mech_contraction if r.get("anchor_feature") == "particle_vs_context_mean_diff_positive_fraction" and r.get("mechanism_feature") == "radius2_slope_median_px2_per_s"), {})
    invariant_rule_best = first_summary(invariant_physics_rules, "best_rule", {}) or {}
    invariant_rule_top = top_items(first_summary(invariant_physics_rules, "top_rules", []), 16)
    invariant_rule_features = top_items(first_summary(invariant_physics_rules, "top_oriented_features", []), 16)
    signed_loss_tests = top_items(first_summary(signed_optical_loss, "top_future16_axis_tests", []), 12)
    signed_loss_models = top_items(first_summary(signed_optical_loss, "top_axis_model_metrics", []), 20)
    signed_loss_modes = top_items(first_summary(signed_optical_loss, "mechanism_mode_summary", []), 8)
    signed_loss_sources = top_items(first_summary(signed_optical_loss, "source_summary", []), 12)
    signed_loss_candidates = top_items(first_summary(signed_optical_loss, "top_candidates", []), 12)
    signed_source_key_metrics = top_items(first_summary(signed_loss_source_robustness, "key_future16_metrics", []), 30)
    signed_source_top_metrics = top_items(first_summary(signed_loss_source_robustness, "top_future16_metrics", []), 30)
    signed_source_influence = top_items(first_summary(signed_loss_source_robustness, "largest_negative_source_influence", []), 12)
    signed_source_summary = top_items(first_summary(signed_loss_source_robustness, "source_summary", []), 12)
    signed_source_raw_optical = next((r for r in signed_source_key_metrics if r.get("axis") == "signed_optical_loss_axis" and r.get("transform") == "raw"), {})
    signed_source_resid_optical = next((r for r in signed_source_key_metrics if r.get("axis") == "signed_optical_loss_axis" and r.get("transform") == "source_residual"), {})
    signed_source_rank_optical = next((r for r in signed_source_key_metrics if r.get("axis") == "signed_optical_loss_axis" and r.get("transform") == "within_source_rank"), {})
    signed_source_raw_combined = next((r for r in signed_source_key_metrics if r.get("axis") == "combined_loss_mechanism_axis" and r.get("transform") == "raw"), {})
    signed_source_resid_combined = next((r for r in signed_source_key_metrics if r.get("axis") == "combined_loss_mechanism_axis" and r.get("transform") == "source_residual"), {})
    signed_source_raw_echem = next((r for r in signed_source_key_metrics if r.get("axis") == "echem_degraded_state_axis" and r.get("transform") == "raw"), {})
    signed_source_resid_echem = next((r for r in signed_source_key_metrics if r.get("axis") == "echem_degraded_state_axis" and r.get("transform") == "source_residual"), {})
    signed_loss_combined16 = next((r for r in signed_loss_tests if r.get("target") == "future_any_drop_within_16cycles" and r.get("axis") == "combined_loss_mechanism_axis"), {})
    signed_loss_optical16 = next((r for r in signed_loss_tests if r.get("target") == "future_any_drop_within_16cycles" and r.get("axis") == "signed_optical_loss_axis"), {})
    signed_loss_echem16 = next((r for r in signed_loss_tests if r.get("target") == "future_any_drop_within_16cycles" and r.get("axis") == "echem_degraded_state_axis"), {})
    signed_loss_best_model16 = max([r for r in signed_loss_models if r.get("target") == "future_any_drop_within_16cycles"], key=lambda r: (float(r.get("roc_auc") or float("nan")) if r.get("roc_auc") is not None else float("nan")), default={})
    signed_loss_best_model8 = max([r for r in signed_loss_models if r.get("target") == "future_any_drop_within_8cycles"], key=lambda r: (float(r.get("roc_auc") or float("nan")) if r.get("roc_auc") is not None else float("nan")), default={})
    signed_loss_top_axis = signed_loss_combined16 or (signed_loss_tests[0] if signed_loss_tests else {})
    signed_loss_optical_axis = signed_loss_optical16
    signed_robust_key = top_items(first_summary(signed_loss_source_robustness, "key_future16_metrics", []), 40)
    signed_robust_influence = top_items(first_summary(signed_loss_source_robustness, "largest_negative_source_influence", []), 20)
    def signed_robust(axis: str, transform: str) -> Dict[str, Any]:
        return next((r for r in signed_robust_key if r.get("axis") == axis and r.get("transform") == transform), {})
    signed_robust_combined_raw = signed_robust("combined_loss_mechanism_axis", "raw")
    signed_robust_combined_source_mean = signed_robust("combined_loss_mechanism_axis", "source_mean_only")
    signed_robust_combined_rank = signed_robust("combined_loss_mechanism_axis", "within_source_rank")
    signed_robust_optical_raw = signed_robust("signed_optical_loss_axis", "raw")
    signed_robust_optical_source_mean = signed_robust("signed_optical_loss_axis", "source_mean_only")
    signed_robust_optical_rank = signed_robust("signed_optical_loss_axis", "within_source_rank")
    signed_robust_echem_resid = signed_robust("echem_degraded_state_axis", "source_residual")
    echem_optical_direct = top_items(first_summary(echem_optical_source_residual, "top_future16_direct_metrics", []), 20)
    echem_optical_models = top_items(first_summary(echem_optical_source_residual, "top_future16_model_metrics", []), 20)
    echem_optical_rules = top_items(first_summary(echem_optical_source_residual, "top_rules", []), 20)
    echem_optical_candidates = top_items(first_summary(echem_optical_source_residual, "top_candidates", []), 12)
    echem_optical_resid_direct = next((r for r in echem_optical_direct if r.get("feature_set") == "echem_plus_optical_source_residual"), {})
    echem_optical_front_direct = next((r for r in echem_optical_direct if r.get("feature_set") == "echem_optical_front_residual"), {})
    echem_optical_resid_model = next((r for r in echem_optical_models if r.get("feature_set") == "echem_plus_optical_source_residual"), {})
    echem_resid_direct = next((r for r in echem_optical_direct if r.get("feature_set") == "echem_source_residual"), {})
    optical_resid_direct = next((r for r in echem_optical_direct if r.get("feature_set") == "optical_source_residual"), {})
    balanced_future_top_acq_resid = balanced_future_acq_resid[0] if balanced_future_acq_resid else {}
    temporal_future8 = (first_summary(temporal_directionality, "best_future8_model", []) or [{}])[0]
    temporal_past8 = (first_summary(temporal_directionality, "best_past8_model", []) or [{}])[0]
    temporal_reversed8 = (first_summary(temporal_directionality, "best_reversed_future8_model", []) or [{}])[0]
    temporal_shift_null = first_summary(temporal_directionality, "shift_null_summary", {}) or {}
    temporal_counts = first_summary(temporal_directionality, "target_label_counts", {}) or {}
    temporal_past8_counts = temporal_counts.get("past_any_drop_within_8cycles", {}) if isinstance(temporal_counts, dict) else {}
    temporal_future_tests = top_items(first_summary(temporal_directionality, "top_future8_feature_tests", []), 10)
    temporal_past_tests = top_items(first_summary(temporal_directionality, "top_past8_feature_tests", []), 6)
    temporal_timing_corr = top_items(first_summary(temporal_directionality, "top_timing_correlations", []), 10)
    spatial_prop_homophily = top_items(first_summary(balanced_spatial_propagation, "top_homophily_tests", []), 8)
    spatial_prop_autocorr = top_items(first_summary(balanced_spatial_propagation, "top_feature_autocorrelation_tests", []), 12)
    spatial_prop_lag = top_items(first_summary(balanced_spatial_propagation, "top_lag_feature_label_tests", []), 10)
    spatial_prop_distance = top_items(first_summary(balanced_spatial_propagation, "distance_gradient_tests", []), 8)
    spatial_prop_top_homophily = spatial_prop_homophily[0] if spatial_prop_homophily else {}
    spatial_prop_top_autocorr = spatial_prop_autocorr[0] if spatial_prop_autocorr else {}
    spatial_prop_top_lag = spatial_prop_lag[0] if spatial_prop_lag else {}
    multicohort_best_oof = max(multicohort_oof, key=lambda r: (float(r.get("pooled_oof_roc_auc") or float("nan")) if r.get("pooled_oof_roc_auc") is not None else float("nan"))) if multicohort_oof else {}
    cross_cohort_shift = top_items(first_summary(cross_cohort_rollout, "domain_shift", []), 12)
    cross_cohort_corr = top_items(first_summary(cross_cohort_rollout, "top_correlations", []), 12)
    cross_cohort_difficult = top_items(first_summary(cross_cohort_rollout, "top_transfer_ranked_difficult_rois", []), 12)
    selected_to_transfer_shift = next((r for r in cross_cohort_shift if r.get("eval_cohort") == "transfer_ranked" and r.get("model_name") == "selected_internal"), {})
    pooled_to_transfer_shift = next((r for r in cross_cohort_shift if r.get("eval_cohort") == "transfer_ranked" and r.get("model_name") == "pooled"), {})
    apparent_diffusion_sources = top_items(first_summary(apparent_diffusion_calibration, "source_timing_summary", []), 12)
    apparent_diffusion_thresholds = top_items(first_summary(apparent_diffusion_calibration, "threshold_summary", []), 10)
    apparent_diffusion_q70_tests = top_items(first_summary(apparent_diffusion_calibration, "future8_feature_tests_q70", []), 10)
    apparent_diffusion_corr = top_items(first_summary(apparent_diffusion_calibration, "calibration_correlations", []), 8)
    apparent_diffusion_q70_test = apparent_diffusion_q70_tests[0] if apparent_diffusion_q70_tests else {}
    hdf5_timebase_scenarios = top_items(first_summary(hdf5_timebase, "scenario_summary", []), 8)
    hdf5_timebase_correlations = top_items(first_summary(hdf5_timebase, "top_timebase_correlations", []), 8)
    hdf5_timebase_pause_sources = top_items(first_summary(hdf5_timebase, "top_pause_heavy_sources", []), 8)
    hdf5_timebase_strict_scenario = next((r for r in hdf5_timebase_scenarios if r.get("scenario") == "strict_source_and_roi_aligned"), {})
    diffusion_physics_gates = top_items(first_summary(diffusion_physics_consistency, "gate_counts", []), 12)
    diffusion_physics_tests = top_items(first_summary(diffusion_physics_consistency, "top_feature_tests", []), 12)
    diffusion_physics_corr = top_items(first_summary(diffusion_physics_consistency, "top_correlations", []), 10)
    diffusion_physics_candidates = top_items(first_summary(diffusion_physics_consistency, "top_consistent_candidates", []), 8)
    diffusion_physics_top_scores = top_items(first_summary(diffusion_physics_consistency, "top_physics_scores", []), 8)
    diffusion_physics_source_summary = top_items(first_summary(diffusion_physics_consistency, "source_summary", []), 12)
    diffusion_claim_top_candidates = top_items(first_summary(diffusion_claim_readiness, "top_candidates", []), 8)
    diffusion_claim_hard_blockers = top_items(first_summary(diffusion_claim_readiness, "hard_blockers", []), 12)
    all_cycle_source_gaps = top_items(first_summary(all_cycle_coverage, "top_source_gaps", []), 12)
    all_cycle_gap_cycles = top_items(first_summary(all_cycle_coverage, "top_coverage_gap_cycles", []), 12)
    all_cycle_roi_cohorts = top_items(first_summary(all_cycle_coverage, "roi_cohort_summary", []), 18)
    all_cycle_outputs = top_items(first_summary(all_cycle_coverage, "cycle_output_summary", []), 10)
    current_claim_status_counts = top_items(first_summary(current_claim_readiness, "status_counts", []), 12)
    current_claim_positive = first_summary(current_claim_readiness, "top_positive_evidence", {}) or {}
    current_claim_negative = first_summary(current_claim_readiness, "top_negative_evidence", {}) or {}
    diffusion_unblock_blockers = top_items(first_summary(diffusion_unblock_sensitivity, "top_blockers", []), 12)
    diffusion_unblock_scenarios = top_items(first_summary(diffusion_unblock_sensitivity, "scenario_summary", []), 8)
    diffusion_unblock_candidates = top_items(first_summary(diffusion_unblock_sensitivity, "top_nearest_unblock_candidates", []), 8)
    targeted_diffusion_actions = top_items(first_summary(targeted_diffusion_blocker, "action_counts", []), 8)
    targeted_diffusion_candidates = top_items(first_summary(targeted_diffusion_blocker, "top_remeasurement_candidates", []), 10)
    targeted_diffusion_nearest = first_summary(targeted_diffusion_blocker, "nearest_diffusion_candidate", {}) or {}
    cycle78_diffusion_target = first_summary(cycle78_diffusion_remeasurement, "target_summary", {}) or {}
    cycle78_diffusion_context = top_items(first_summary(cycle78_diffusion_remeasurement, "context_summary", []), 8)
    post_remeasurement_scenarios = top_items(first_summary(post_remeasurement_diffusion_gate, "scenario_table", []), 8)
    post_remeasurement_target = first_summary(post_remeasurement_diffusion_gate, "target_remeasurement_evidence", {}) or {}
    post_remeasurement_publication_blockers = first_summary(post_remeasurement_diffusion_gate, "remaining_publication_blockers", []) or []
    cycle78_front_identity_target = first_summary(cycle78_front_identity_review, "target_summary", {}) or {}
    cycle78_component_raw = first_summary(cycle78_component_retracking, "target_raw", {}) or {}
    cycle78_component_largest = first_summary(cycle78_component_retracking, "target_largest_component", {}) or {}
    cycle78_component_central = first_summary(cycle78_component_retracking, "target_central_component", {}) or {}
    cycle78_component_top3 = first_summary(cycle78_component_retracking, "target_top3_components", {}) or {}
    cross_modal_top = top_items(first_summary(cross_modal_consensus, "top_cycles", []), 12)
    cross_modal_classes = top_items(first_summary(cross_modal_consensus, "class_summary", []), 8)
    cross_modal_contrasts = top_items(first_summary(cross_modal_consensus, "target_contrasts", []), 12)
    cross_modal_top_cycle = first_summary(cross_modal_consensus, "top_cycle", {}) or {}
    cross_modal_sync_cycles = [
        row for row in cross_modal_top
        if row.get("consensus_class") == "synchronized_multimodal_degradation_candidate"
    ]
    cross_modal_sync_cycle_labels = ", ".join(fmt(row.get("cycleNo"), 0) for row in cross_modal_sync_cycles[:3]) or "NA"
    diffusion_sanity_gate = top_items(first_summary(diffusion_sanity, "gate_counts", []), 12)
    diffusion_sanity_candidates = top_items(first_summary(diffusion_sanity, "top_automatic_candidates", []), 12)
    diffusion_sanity_corr = top_items(first_summary(diffusion_sanity, "top_correlations", []), 8)
    control_balanced_diffusion_gate = top_items(first_summary(control_balanced_diffusion_sanity, "gate_counts", []), 12)
    control_balanced_diffusion_tests = top_items(first_summary(control_balanced_diffusion_sanity, "top_event_control_tests", []), 8)
    control_balanced_diffusion_candidates = top_items(first_summary(control_balanced_diffusion_sanity, "top_automatic_candidates", []), 10)
    weak_label_top_pos = top_items(first_summary(weak_label_benchmark, "top_positive_training_rows", []), 8)
    weak_label_top_neg = top_items(first_summary(weak_label_benchmark, "top_negative_training_rows", []), 8)
    weak_label_leakage = first_summary(weak_label_benchmark, "leakage_audit", {}) or {}

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
            "implemented_with_guardrail",
            f"Synthesis reads Isambard derived directory with {n_roi} core ROI rows, while the all-cycle coverage atlas maps {first_summary(all_cycle_coverage, 'n_cycle_rows', 0)} cycle rows, {first_summary(all_cycle_coverage, 'n_sources', 0)} sources, {first_summary(all_cycle_coverage, 'n_h5_inventory_rows', 0)} HDF5 inventory rows, and {first_summary(all_cycle_coverage, 'n_cycles_with_any_roi_video_sequence', 0)} cycles with tracked ROI/video-sequence coverage.",
            "The atlas proves broad source/cycle coverage across accumulated cohorts, but it is still a manifest-level coverage audit rather than exhaustive per-particle segmentation of every raw movie.",
            "Use the all-cycle coverage gap queue to prioritize any remaining source/cycle extraction and manual-QC expansion.",
        ),
        evidence_row(
            "Next-frame prediction and rollout",
            "implemented_with_guardrail",
            f"Persistence, velocity, low-rank DMD, PCA latent trajectories, PCA-ridge, residual-CNN guardrails, and prefix-only ROI forecasts were run. Persistence is best across raw pixel rollouts: {persistence_best}; best prefix classifier target is {top_prefix_classifier.get('target', 'NA')} with AUC {fmt(top_prefix_classifier.get('mean_roc_auc'))}.",
            "Learned/full rollout models do not yet beat persistence robustly; use residuals, latent paths, and prefix forecasts as physics descriptors rather than claiming superior pixel prediction.",
            "Train cycle-conditioned probabilistic video models only after growing the ROI set and validating particle masks.",
        ),
        evidence_row(
            "Select and guard particle-region-only ROIs",
            "implemented_with_guardrail",
            f"Model inputs use cropped ROI tensors and the particle-mask stability audit covers {first_summary(particle_mask, 'n_roi', 0)} ROI rows / {first_summary(particle_mask, 'n_frames_total', 0)} frames; median fallback fraction is {fmt(particle_mask_overall.get('median_fallback_frame_fraction'))}.",
            "The audit uses automatic contrast/history masks and is not a manual segmentation of each particle boundary.",
            "Use the highest-instability ROI list during manual QC and avoid calibrated front/diffusion claims for unstable masks.",
        ),
        evidence_row(
            "Track phase-boundary movement",
            "implemented_as_proxy",
            f"Front/phase mobility descriptors, selected-front tracking, and threshold-robust sweeps exist; threshold sweep covers {first_summary(robust_fronts, 'n_roi', 0)} ROI rows.",
            f"Front masks are automatic; after protocol/echem conditioning, front-direction sign consistency survives more strongly than front-magnitude metrics and is robust in {len(robust_phase_strata)} automatic QC strata.",
            f"Use the primary and control-balanced QC packages to record accept/reject decisions, including {first_summary(control_balanced_qc, 'n_control_roi', 0)} control candidates.",
        ),
        evidence_row(
            "Extract diffusion coefficients",
            "partial_proxy_only",
            f"Provisional 0.096 um/px apparent diffusion proxies were computed and stress-tested across {len(first_summary(robust_fronts, 'threshold_quantiles', []))} thresholds with bootstrap slopes; the stricter diffusion sanity audit finds {first_summary(diffusion_sanity, 'n_automatic_positive_diffusion_proxy_candidates', 0)} automatic positive candidates and {first_summary(diffusion_sanity, 'n_publication_diffusion_candidates', 0)} publication candidates; the control-balanced high-resolution rerun tracks {first_summary(control_balanced_front_tracking, 'n_tracked_rois', 0)} ROIs and still finds {first_summary(control_balanced_diffusion_sanity, 'n_publication_diffusion_candidates', 0)} publication candidates.",
            "Global threshold-robust phase slopes separate event/control ROIs, but QC-stratified diffusion proxies are inconsistent, conditioned diffusion-proxy residuals remain non-significant, and selected-front radius-squared slopes fail sign/fit/manual-QC gates.",
            "Treat diffusion numbers as apparent optical-front proxies until microscope calibration, timebase, front masks, estimator agreement, and manual QC are jointly validated.",
        ),
        evidence_row(
            "Identify degradation modes",
            "implemented_as_hypothesis_ranking",
            f"Joint physics/rollout/echem mode tables exist, residual taxonomy found {first_summary(residual_modes, 'chosen_k', 0)} protocol-adjusted modes, and cycle/region context maps them across {first_summary(cycle_region_modes, 'n_cycles', 0)} cycles and {first_summary(cycle_region_modes, 'n_xy_regions', 0)} coarse regions.",
            "Modes are unsupervised/automatic and tied to the selected ROI cohort; residual taxonomy silhouette is modest and cycle/region context is descriptive.",
            "Use the residual-mode and cycle/region review lists for manual labeling, then refit supervised or semi-supervised degradation-mode models.",
        ),
        evidence_row(
            "Correlate degradation with cycles, particle regions, and echem/protocol context",
            "implemented_with_guardrail",
            "Multi-cycle ROI echem coupling found strong frame-count/protocol correlations; protocol-conditioned residual tests still show event/control optical shifts and phase-front sign consistency.",
            "Residualization reduces but does not eliminate confounding, and front residual classifiers are not deployable with 52 automatic ROIs.",
            "Use protocol-conditioned residuals as the default event-effect readout and expand after manual QC.",
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
            "implemented_with_verification",
            "Scoped analysis scripts, compact derived outputs, and observations are committed and pushed after each completed increment.",
            "The synthesis report records workflow state at generation time; final push status should still be checked with git status and git log.",
            "Continue committing and pushing each new analysis increment with compact local artifacts only.",
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
        f"- ROI/front QC package candidates: {first_summary(qc_package, 'n_selected_roi', 0)}",
        f"- Control-balanced front QC candidates: {first_summary(control_balanced_qc, 'n_selected_roi', 0)}",
        f"- Residual physics mode clusters: {first_summary(residual_modes, 'chosen_k', 0)}",
        f"- Manual-QC label workbook candidates: {first_summary(manual_qc_workbook, 'n_unique_roi', 0)}",
        f"- Manual-QC gated accepted fronts: {first_summary(manual_qc_gated, 'n_manual_front_effect_accepted', 0)}",
        f"- Prefix feature-importance audit features: {first_summary(prefix_importance, 'n_features', 0)}",
        f"- Spatiotemporal degradation graph nodes/edges: {first_summary(spatiotemporal_graph, 'n_nodes', 0)} / {first_summary(spatiotemporal_graph, 'n_edges', 0)}",
        f"- Phase-kinetics ROI rows/features: {first_summary(phase_kinetics, 'n_roi', 0)} / {first_summary(phase_kinetics, 'n_kinetic_features', 0)}",
        f"- Calibration metadata HDF5/camera-timing files: {first_summary(calibration_metadata, 'n_h5_files', 0)} / {first_summary(calibration_metadata, 'n_h5_with_camera_timing', 0)}",
        f"- Calibration provenance status: {first_summary(calibration_provenance, 'provenance_status', 'missing')} with {first_summary(calibration_provenance, 'n_near_96nm_px_statements', 0)} near-96 nm/px statements",
        f"- Calibration claim-risk families/source tables: {first_summary(calibration_claim_risk, 'n_claim_families', 0)} / {first_summary(calibration_claim_risk, 'n_source_tables_present', 0)}",
        f"- HDF5 timebase q70 rows strict/pause-heavy sources: {first_summary(hdf5_timebase, 'n_q70_roi_rows', 0)} rows, {first_summary(hdf5_timebase, 'n_strict_timebase_sources', 0)} strict sources, {first_summary(hdf5_timebase, 'n_pause_heavy_sources', 0)} pause-heavy sources",
        f"- Particle trace cycle rows/drop cycles: {first_summary(particle_trace, 'n_cycle_rows', 0)} / {first_summary(particle_trace, 'n_any_drop_cycles', 0)}",
        f"- Particle precursor event/control anchors: {first_summary(particle_precursor, 'n_event_anchors', 0)} / {first_summary(particle_precursor, 'n_matched_control_anchors', 0)}",
        f"- ROI trace-fusion rows/predictors: {first_summary(roi_trace_fusion, 'n_roi_rows', 0)} / {first_summary(roi_trace_fusion, 'n_predictors', 0)}",
        f"- ROI trace-fusion cycle-null points: {first_summary(roi_trace_cycle_null, 'n_cycle_points', 0)}",
        f"- Precursor-informed review candidates: {first_summary(precursor_review, 'n_review_candidates', 0)}",
        f"- Precursor visual-bundle candidates/assets: {first_summary(precursor_visual_bundle, 'n_ranked_candidates', 0)} / {first_summary(precursor_visual_bundle, 'n_candidates_with_visual_asset', 0)}",
        f"- Within-cycle echem shape cycles/features: {first_summary(within_cycle_echem, 'n_echem_shape_cycles', 0)} / {first_summary(within_cycle_echem, 'n_shape_features', 0)}",
        f"- Echem-shape-conditioned ROI/front rows/shape PCs: {first_summary(echem_shape_conditioned, 'n_rows', 0)} / {first_summary(echem_shape_conditioned, 'shape_pca', {}).get('n_components', 0)}",
        f"- Physics-consistency matrix ROI/cycles: {first_summary(physics_consistency, 'n_roi', 0)} / {first_summary(physics_consistency, 'n_cycles', 0)}",
        f"- Cycle state-space rows/clusters: {first_summary(cycle_state_space, 'n_cycles', 0)} / {first_summary(cycle_state_space, 'chosen_k', 0)}",
        f"- Cycle hazard warning evaluated cycles/events: {((cycle_hazard_feature_sets[0] if cycle_hazard_feature_sets else {}).get('n_evaluated_cycles', 0))} / {first_summary(cycle_hazard_warning, 'n_event_cycles', 0)}",
        f"- Cycle-state ROI bridge rows/cycles: {first_summary(cycle_state_roi_bridge, 'n_roi_rows', 0)} / {first_summary(cycle_state_roi_bridge, 'n_cycles', 0)}",
        f"- Cycle-state mode-frequency bridge cycles/modes: {first_summary(cycle_state_mode_frequency, 'n_cycles', 0)} / {first_summary(cycle_state_mode_frequency, 'n_mode_targets', 0)}",
        f"- Particle-mask stability ROI/frame rows: {first_summary(particle_mask, 'n_roi', 0)} / {first_summary(particle_mask, 'n_frames_total', 0)}",
        f"- Masked ROI rollout frame rows: {first_summary(masked_rollout, 'n_frame_metric_rows', 0)}",
        f"- Masked rollout cycle-warning ROI cycles/features: {first_summary(masked_cycle_warning, 'n_roi_cycles', 0)} / {first_summary(masked_cycle_warning, 'n_rollout_features_tested', 0)}",
        f"- Masked residual transition ROI/method rows: {first_summary(masked_residual_timing, 'n_roi_method_rows', 0)}",
        f"- Masked residual state-transfer anchor/full cycles: {first_summary(masked_residual_transfer, 'n_anchor_cycles', 0)} / {first_summary(masked_residual_transfer, 'n_full_cycles', 0)}",
        f"- Transfer-ranked reconstructed cycles/ROI rows: {first_summary(transfer_ranked_reconstruction, 'n_cycles_sampled', 0)} / {first_summary(transfer_ranked_reconstruction, 'n_roi_rows', 0)}",
        f"- Transfer-ranked masked rollout ROI/frame rows: {first_summary(transfer_ranked_rollout, 'n_roi', 0)} / {first_summary(transfer_ranked_rollout, 'n_frame_metric_rows', 0)}",
        f"- Transfer-ranked front physics ROI/cycles: {first_summary(transfer_ranked_front_physics, 'n_roi', 0)} / {first_summary(transfer_ranked_front_physics, 'n_cycles', 0)}",
        f"- Transfer-ranked residual transition timing ROI/method rows: {first_summary(transfer_ranked_residual_timing, 'n_roi', 0)} / {first_summary(transfer_ranked_residual_timing, 'n_roi_method_rows', 0)}",
        f"- Multi-cohort future-drop model rows/features: {first_summary(multicohort_future_drop, 'n_roi_rows', 0)} / {first_summary(multicohort_future_drop, 'n_features', 0)}",
        f"- Active-learning QC candidates/visual/immediate: {first_summary(active_learning_qc, 'n_candidate_rows', 0)} / {first_summary(active_learning_qc, 'n_rows_with_visual_asset', 0)} / {active_qc_tiers.get('immediate_manual_qc', 0)}",
        f"- Automatic QC triage surrogate candidates/likely/artifact/diffusion-guardrail: {first_summary(automatic_qc_triage, 'n_candidates', 0)} / {first_summary(automatic_qc_triage, 'likely_interpretable_count', 0)} / {first_summary(automatic_qc_triage, 'artifact_risk_count', 0)} / {first_summary(automatic_qc_triage, 'diffusion_guardrail_count', 0)}",
        f"- QC decision evidence ledger candidates/action tiers: {first_summary(qc_decision_ledger, 'n_candidates', 0)} / {qc_decision_actions}",
        f"- Balanced future-drop ROI rows/cycles/features: {first_summary(balanced_future_physics, 'n_roi', 0)} / {first_summary(balanced_future_physics, 'n_cycles', 0)} / {first_summary(balanced_future_physics, 'n_features', 0)}",
        f"- Cross-cohort rollout transfer selected/transfer ROIs: {first_summary(cross_cohort_rollout, 'n_selected_roi', 0)} / {first_summary(cross_cohort_rollout, 'n_transfer_ranked_roi', 0)}",
        f"- Diffusion sanity selected-front/publication candidates: {first_summary(diffusion_sanity, 'n_selected_front_rois', 0)} / {first_summary(diffusion_sanity, 'n_publication_diffusion_candidates', 0)}",
        f"- Control-balanced high-res front tracking/sanity candidates: {first_summary(control_balanced_front_tracking, 'n_tracked_rois', 0)} / {first_summary(control_balanced_diffusion_sanity, 'n_publication_diffusion_candidates', 0)}",
        f"- Weak-label benchmark trainable positives/negatives: {first_summary(weak_label_benchmark, 'n_positive_weak_labels', 0)} / {first_summary(weak_label_benchmark, 'n_negative_weak_labels', 0)}",
        f"- Control-balanced QC sensitivity robust strata: {len(first_summary(control_balanced_qc_sensitivity, 'robust_positive_phase_residual_strata', []))}",
        "",
        "## Main Findings",
        "",
        "- Persistence is the strongest raw next-frame baseline; DMD/velocity/learned residual experiments are most useful as residual and latent descriptors.",
        f"- Prefix-only cropped ROI forecasts still rank the front-direction residual class highest: {top_prefix_classifier.get('model', 'NA')} at prefix {fmt(top_prefix_classifier.get('prefix_fraction'))} gives AUC {fmt(top_prefix_classifier.get('mean_roc_auc'))}, but after excluding raw frame-index features the audited permutation null is not significant.",
        f"- The 75%-prefix feature-importance audit is descriptive but not independently significant: pooled OOF AUC {fmt(prefix_importance_model.get('pooled_oof_roc_auc'))}, null empirical p={fmt(prefix_importance_null.get('empirical_p_ge_observed'))}; strongest ablation groups are {', '.join(str(r.get('removed_group')) for r in prefix_importance_groups[:2]) or 'NA'}.",
        "- ROI event/control optical differences survive event-reference-cycle centering, especially cumulative normalized change, first-last decorrelation, latent net displacement, high-fraction growth, and ROI mean trend.",
        "- Frame count and protocol-block position strongly couple to ROI dynamics, so echem/protocol context must be a model covariate and a guardrail.",
        "- After residualizing available protocol/echem covariates and event-reference fixed effects, event/control separation remains in ROI mean delta, high-fraction delta, first-last correlation, cumulative change, DMD residual, and latent displacement.",
        "- Cycles 86 and 116 remain the strongest synchronized event-timing regimes; cycles 60 and 156 provide stronger single-particle morphology/latent-movement examples.",
        "- Apparent front tracking currently indicates optical-front contraction/loss more than clean expanding diffusion fronts.",
        "- Threshold sweeps show robust event/control differences in phase-fraction slope, but radius-derived diffusion proxies remain weaker and threshold-sensitive.",
        f"- Diffusion proxy sanity audit rejects calibrated-diffusion promotion for the selected high-resolution front set: {first_summary(diffusion_sanity, 'n_automatic_positive_diffusion_proxy_candidates', 0)} automatic positive candidates and {first_summary(diffusion_sanity, 'n_publication_diffusion_candidates', 0)} publication candidates; median selected-front apparent D is {fmt(first_summary(diffusion_sanity, 'median_selected_diffusion_um2_per_s'))} um2/s and only {fmt(first_summary(diffusion_sanity, 'selected_positive_fraction'))} of selected fronts are nonnegative.",
        f"- Control-balanced high-resolution front tracking expands this check to {first_summary(control_balanced_front_tracking, 'n_tracked_rois', 0)} ROIs ({first_summary(control_balanced_diffusion_sanity, 'selected_front_cohort_counts', {})}); it still yields {first_summary(control_balanced_diffusion_sanity, 'n_automatic_positive_diffusion_proxy_candidates', 0)} automatic positive diffusion candidates and event/control selected-D separation remains non-significant (top p={fmt((control_balanced_diffusion_tests[0] if control_balanced_diffusion_tests else {}).get('mannwhitney_p'))}).",
        f"- Calibration metadata audit finds camera-timing datasets in {first_summary(calibration_metadata, 'n_h5_with_camera_timing', 0)} of {first_summary(calibration_metadata, 'n_h5_files', 0)} scanned HDF5 files and no HDF5 pixel-size attributes; sampled timing rows can be sparse segment/cycle timing, while the 96 nm/px scale remains slide-derived pending raw microscope metadata confirmation.",
        f"- Calibration provenance evidence audit scans {first_summary(calibration_provenance, 'n_h5_files_scanned', 0)} raw HDF5 files and {first_summary(calibration_provenance, 'n_files_inventoried', 0)} total provenance files; status is `{first_summary(calibration_provenance, 'provenance_status', 'missing')}` with highest near-scale evidence `{first_summary(calibration_provenance, 'highest_scale_evidence_strength', 'none')}`, so 96 nm/px remains supported by slides/project text rather than primary raw microscope metadata.",
        f"- Calibration claim-risk register audits {first_summary(calibration_claim_risk, 'n_claim_families', 0)} front/kinetic/diffusion claim families; it classifies diffusion-like values as apparent proxies and keeps manual-QC-gated diffusion/front claims pending.",
        f"- Apparent diffusion calibration-bounds audit maps all {first_summary(apparent_diffusion_calibration, 'n_roi_with_h5_timing', 0)} balanced ROIs to HDF5 timing; ROI elapsed/HDF5 elapsed median ratio is {fmt(first_summary(apparent_diffusion_calibration, 'median_roi_elapsed_to_h5_median_ratio'))}, q70 median apparent D at 96 nm/px is {fmt(first_summary(apparent_diffusion_calibration, 'median_q70_apparent_D_h5median_um2_per_s'))} um2/s, and q70 future8 separation is non-significant (top p={fmt(apparent_diffusion_q70_test.get('mannwhitney_p'))}).",
        f"- HDF5 timebase provenance audit shows all q70 ROI spans align to median HDF5 timing ({first_summary(hdf5_timebase, 'n_roi_elapsed_h5_aligned', 0)}/{first_summary(hdf5_timebase, 'n_q70_roi_rows', 0)}), but {first_summary(hdf5_timebase, 'n_pause_heavy_sources', 0)} of {first_summary(hdf5_timebase, 'n_sources', 0)} sources have pause-heavy camera timing; in the strict source+ROI subset, q70 future8 apparent-D contrast has p={fmt(hdf5_timebase_strict_scenario.get('future8_mannwhitney_p'))} and median positive-negative {fmt(hdf5_timebase_strict_scenario.get('future8_median_positive_minus_negative'))}.",
        f"- Diffusion physics-consistency audit collapses {first_summary(diffusion_physics_consistency, 'n_threshold_rows', 0)} threshold rows to {first_summary(diffusion_physics_consistency, 'n_roi', 0)} ROI gates: {first_summary(diffusion_physics_consistency, 'n_automatic_diffusion_physics_consistent', 0)} automatic ROI passes the internal physics gate and {first_summary(diffusion_physics_consistency, 'n_publication_ready_diffusion_candidates', 0)} pass publication-ready diffusion gates; median radius2 fit R2 is only {fmt(first_summary(diffusion_physics_consistency, 'median_radius2_fit_r2'))}.",
        f"- Cross-modal consensus ranks cycles {cross_modal_sync_cycle_labels} as synchronized multimodal degradation candidates; the top cycle has {fmt(cross_modal_top_cycle.get('n_modal_votes'), 0)} modal votes and consensus score {fmt(cross_modal_top_cycle.get('cross_modal_consensus_score'))}, while the score remains an audit statistic rather than a calibrated probability.",
        f"- Echem/optical breakpoint audit tests {first_summary(echem_optical_breakpoint, 'n_features_tested', 0)} cycle-level echem/trace features around synchronized cycles {first_summary(echem_optical_breakpoint, 'event_cycles', [])}; strongest event-centered shift is {echem_breakpoint_top.get('feature', 'NA')} over +/-{fmt(echem_breakpoint_top.get('window_cycles'), 0)} cycles (scaled shift {fmt(echem_breakpoint_top.get('event_median_scaled_shift'))}, bootstrap p={fmt(echem_breakpoint_top.get('bootstrap_p_abs_vs_control_centers'))}).",
        f"- Echem-optical regime atlas organizes {first_summary(echem_optical_regime, 'n_cycles', 0)} cycles by charge/discharge asymmetry and dQ/dV-proxy shape; top binary contrast is {echem_regime_top_binary.get('feature', 'NA')} vs {echem_regime_top_binary.get('target', 'NA')} (median shift {fmt(echem_regime_top_binary.get('median_positive_minus_negative'))}, p={fmt(echem_regime_top_binary.get('mannwhitney_p'))}), and top continuous link is {echem_regime_top_corr.get('feature', 'NA')} vs {echem_regime_top_corr.get('target', 'NA')} (rho={fmt(echem_regime_top_corr.get('spearman_rho'))}).",
        f"- Echem-conditioned optical predictor shows the clearest echem gain for {echem_predictor_top_delta.get('target', 'NA')} under {echem_predictor_top_delta.get('split', 'NA')}: {echem_predictor_top_delta.get('comparison', 'NA')} changes AUC by {fmt(echem_predictor_top_delta.get('delta_roc_auc'))}; same-cycle synchronized candidates remain acquisition/context dominated and underpowered.",
        f"- Echem-conditioned ROI rollout/front audit joins {first_summary(echem_roi_rollout_front, 'n_roi_rows', 0)} ROI rows across {first_summary(echem_roi_rollout_front, 'n_cycles', 0)} cycles; strongest leave-cycle echem gain is {echem_roi_top_delta.get('target', 'NA')} {echem_roi_top_delta.get('comparison', 'NA')} with delta Spearman {fmt(echem_roi_top_delta.get('delta_spearman_rho'))} and delta R2 {fmt(echem_roi_top_delta.get('delta_r2'))}.",
        f"- Echem-video embedding fusion tests {first_summary(echem_video_fusion, 'n_embedding_rows', 0)} masked-video rows across {first_summary(echem_video_fusion, 'n_cycles', 0)} cycles; top fusion delta is {echem_video_top_delta.get('target', 'NA')} {echem_video_top_delta.get('comparison', 'NA')} with delta AUC {fmt(echem_video_top_delta.get('delta_roc_auc'))} and delta Spearman {fmt(echem_video_top_delta.get('delta_spearman_rho'))}.",
        "- Protocol-conditioned front residuals preserve phase-slope sign consistency, but not front-magnitude or diffusion-proxy separability.",
        f"- Automatic front-QC sensitivity keeps the positive phase-front residual in {len(robust_phase_strata)} strata: {', '.join(robust_phase_strata) if robust_phase_strata else 'none'}; review-panel diffusion proxy differences are selection-sensitive and not calibrated transport.",
        f"- Protocol-adjusted residual mode taxonomy chooses k={first_summary(residual_modes, 'chosen_k', 0)}; its most event-enriched mode is {top_residual_mode.get('mode_label', 'NA')} with event fraction {fmt(top_residual_mode.get('event_fraction'))} and Fisher p={fmt(top_residual_mode.get('fisher_p_value'))}.",
        f"- A QC review packet prioritizes {first_summary(qc_packet, 'n_candidates', 0)} ROI/front candidates, a control-balanced front package adds {first_summary(control_balanced_qc, 'n_control_roi', 0)} control candidates, and the manual-QC label workbook deduplicates these into {first_summary(manual_qc_workbook, 'n_unique_roi', 0)} pending ROI labels.",
        f"- Control-balanced QC sensitivity keeps positive phase-front residuals robust in {len(first_summary(control_balanced_qc_sensitivity, 'robust_positive_phase_residual_strata', []))} automatic strata, including the balanced selected panel; diffusion-proxy residuals remain non-significant in that balanced panel.",
        f"- Manual-QC gated front-effect tests are status `{first_summary(manual_qc_gated, 'status', 'missing')}` with {first_summary(manual_qc_gated, 'n_manual_front_effect_accepted', 0)} accepted fronts, so no manual-QC-filtered diffusion/front claim is emitted yet.",
        f"- Spatiotemporal graph tests show strong same-cycle spatial homophily in front-positive residuals and event-enriched residual modes, but cross-cycle nearest-neighbor front/event labels do not show simple propagation and remain cohort-design sensitive.",
        "- Optical phase-kinetics fits add transition-sharpness and Avrami-style descriptors: event-enriched residual modes have larger q70/q80 transformed-fraction deltas and faster q60/q70 logistic rates, while kinetic fit quality/rates remain strongly coupled to frame count.",
        f"- The larger four-particle cycle table shows leakage-conscious early-warning signal for future abrupt drops: any-drop within 8 cycles has mean AUC {fmt((particle_trace_classifiers[0] if particle_trace_classifiers else {}).get('mean_roc_auc'))} with empirical null p={fmt((particle_trace_nulls[0] if particle_trace_nulls else {}).get('empirical_p_ge_observed'))}; synchronized 2+ drops are also detectable but with only two positive cycles.",
        f"- Event-aligned precursor windows show lower pre-event capacity/CE and higher cross-particle delta dispersion versus matched non-event anchors; the strongest precursor window test is {((particle_precursor_tests[0] if particle_precursor_tests else {}).get('window', 'NA'))} {((particle_precursor_tests[0] if particle_precursor_tests else {}).get('feature', 'NA'))} with p={fmt((particle_precursor_tests[0] if particle_precursor_tests else {}).get('mannwhitney_p'))}.",
        f"- ROI trace-fusion links lagged global particle-trace state to localized front behavior at the ROI-row level: strongest focused context-residual association is {((roi_trace_focus[0] if roi_trace_focus else {}).get('predictor', 'NA'))} vs {((roi_trace_focus[0] if roi_trace_focus else {}).get('target', 'NA'))}, rho={fmt((roi_trace_focus[0] if roi_trace_focus else {}).get('rho'))}, p={fmt((roi_trace_focus[0] if roi_trace_focus else {}).get('p_value'))}.",
        f"- Cycle-collapsed ROI trace-fusion null audit reduces 52 ROI rows to {first_summary(roi_trace_cycle_null, 'n_cycle_points', 0)} cycle points; top surviving collapsed association is {((roi_trace_cycle_tests[0] if roi_trace_cycle_tests else {}).get('predictor', 'NA'))} vs {((roi_trace_cycle_tests[0] if roi_trace_cycle_tests else {}).get('target', 'NA'))}, rho={fmt((roi_trace_cycle_tests[0] if roi_trace_cycle_tests else {}).get('rho'))}, empirical p={fmt((roi_trace_cycle_tests[0] if roi_trace_cycle_tests else {}).get('empirical_p_abs_ge_observed'))}.",
        f"- Precursor-informed ROI review ranks {first_summary(precursor_review, 'n_review_candidates', 0)} pending manual-QC candidates; the top candidate is {(precursor_review_top[0] if precursor_review_top else {}).get('roi_id', 'NA')} with score {fmt((precursor_review_top[0] if precursor_review_top else {}).get('precursor_informed_review_score'))}.",
        f"- A visual review bundle now packages {first_summary(precursor_visual_bundle, 'n_ranked_candidates', 0)} top precursor-informed ROI candidates; {first_summary(precursor_visual_bundle, 'n_candidates_with_visual_asset', 0)} have at least one copied QC/preview asset and a contact sheet for manual inspection.",
        f"- Within-cycle echem shape descriptors add raw voltage/current trajectory and dQ/dV-proxy context for {first_summary(within_cycle_echem, 'n_echem_shape_cycles', 0)} observed cycles; strongest ROI association is {((within_cycle_roi_corr[0] if within_cycle_roi_corr else {}).get('feature', 'NA'))} vs {((within_cycle_roi_corr[0] if within_cycle_roi_corr else {}).get('target', 'NA'))}, rho={fmt((within_cycle_roi_corr[0] if within_cycle_roi_corr else {}).get('rho'))}, but direct event-cycle shape tests are weak and shape terms remain protocol/capacity guardrails.",
        f"- Echem-shape-conditioned residual audit uses {first_summary(echem_shape_conditioned, 'shape_pca', {}).get('n_shape_features_used', 0)} shape features compressed to {first_summary(echem_shape_conditioned, 'shape_pca', {}).get('n_components', 0)} PCs; phase-slope positive-fraction residual remains the strongest event/control readout after shape conditioning (p={fmt((echem_shape_conditioned_tests[0] if echem_shape_conditioned_tests else {}).get('p_value'))}), while diffusion residuals remain non-significant and the shape-residual classifier is poor.",
        f"- Physics-consistency claim matrix scores {first_summary(physics_consistency, 'n_roi', 0)} ROI rows across front, optical-change, rollout, kinetics, precursor, echem-shape, and mode-taxonomy pillars; {first_summary(physics_consistency, 'tier_counts', {}).get('cross_modal_high_priority', 0)} rows are cross-modal high priority, but all {first_summary(physics_consistency, 'n_roi', 0)} remain `manual_qc_required_no_physics_claim`.",
        f"- Cycle state-space transition audit builds a {first_summary(cycle_state_space, 'chosen_k', 0)}-state cycle manifold from trace plus echem-shape features; PC2 is the strongest future 8-cycle abrupt-drop separator (permutation p={fmt((cycle_state_tests[0] if cycle_state_tests else {}).get('permutation_p'))}), the shuffled-fold classifier reaches mean AUC {fmt(cycle_state_classifier.get('mean_roc_auc'))}, and stricter temporal holdout reaches AUC {fmt(cycle_state_temporal.get('mean_roc_auc'))} across {fmt(cycle_state_temporal.get('n_evaluated_blocks'), 0)} usable blocks.",
        f"- Rolling-origin cycle hazard warning audit evaluates {((cycle_hazard_feature_sets[0] if cycle_hazard_feature_sets else {}).get('n_evaluated_cycles', 0))} cycles for future 8-cycle abrupt drops; best AUC is {fmt((cycle_hazard_feature_sets[0] if cycle_hazard_feature_sets else {}).get('roc_auc'))} with permutation p={fmt(cycle_hazard_null.get('empirical_p_ge_observed'))}, and 8-cycle pre-event warnings hit {fmt((cycle_hazard_lead[1] if len(cycle_hazard_lead) > 1 else {}).get('hit_rate'))} of event cycles.",
        f"- Cycle-state to ROI/front bridge links state PC2 to ROI physics-consistency after collapsing repeated ROI rows to {first_summary(cycle_state_roi_bridge, 'n_cycles', 0)} cycles: top collapsed test {((cycle_state_roi_collapsed_tests[0] if cycle_state_roi_collapsed_tests else {}).get('predictor', 'NA'))} vs {((cycle_state_roi_collapsed_tests[0] if cycle_state_roi_collapsed_tests else {}).get('target', 'NA'))}, rho={fmt((cycle_state_roi_collapsed_tests[0] if cycle_state_roi_collapsed_tests else {}).get('rho'))}, permutation p={fmt((cycle_state_roi_collapsed_tests[0] if cycle_state_roi_collapsed_tests else {}).get('permutation_p'))}.",
        f"- Cycle-state mode-frequency bridge tests whether cycle/echem state organizes ROI degradation modes across {first_summary(cycle_state_mode_frequency, 'n_cycles', 0)} cycles: best macro model {cycle_state_mode_best.get('feature_set', 'NA')} has MAE {fmt(cycle_state_mode_best.get('mae'))} versus context-only MAE {fmt(cycle_state_mode_context.get('mae'))}; compact permutation p={fmt((cycle_state_mode_nulls[0] if cycle_state_mode_nulls else {}).get('empirical_p_le_observed_mae'))} keeps this as a guarded organization signal.",
        f"- Particle-mask stability audit confirms ROI-only crops can be processed with a history-aware particle support guardrail: median fallback fraction {fmt(particle_mask_overall.get('median_fallback_frame_fraction'))}, accepted-area CV {fmt(particle_mask_overall.get('median_accepted_area_cv'))}, centroid path {fmt(particle_mask_overall.get('median_centroid_path_px'))} px; event/control mask instability is not significantly different in the current cohort.",
        f"- Masked ROI rollout audit scores held-out predictions only inside accepted particle masks; persistence remains best for {masked_rollout_best_counts.get('persistence', 0)} of {first_summary(masked_rollout, 'n_roi', 0)} ROIs, while low-rank DMD particle MSE tracks cumulative optical change (top rho={fmt((masked_rollout_corr[0] if masked_rollout_corr else {}).get('spearman_rho'))}, p={fmt((masked_rollout_corr[0] if masked_rollout_corr else {}).get('p_value'))}).",
        f"- Cycle-collapsed masked-rollout warning audit covers {first_summary(masked_cycle_warning, 'n_roi_cycles', 0)} observed ROI cycles; strongest tests align residual jumps with same-cycle abrupt drops (top permutation p={fmt((masked_cycle_warning_tests[0] if masked_cycle_warning_tests else {}).get('permutation_p_abs_median_diff'))}), while future-drop evaluation is underpowered with only {masked_cycle_warning_targets.get('future_any_drop_within_8cycles', 0)} positive 8-cycle warning case.",
        f"- Masked residual state-transfer warning expands the masked-residual signature from {first_summary(masked_residual_transfer, 'n_anchor_cycles', 0)} video-backed cycles to {first_summary(masked_residual_transfer, 'n_full_cycles', 0)} cycle-state rows; the transferred score separates future 8-cycle drops (AUC {fmt(masked_residual_transfer_future8.get('abs_oriented_auc'))}, permutation p={fmt(masked_residual_transfer_future8.get('permutation_p_abs_median_diff'))}), but anchor leave-one-cycle transfer is weak (rho={fmt(masked_residual_transfer_anchor.get('rho'))}, p={fmt(masked_residual_transfer_anchor.get('p_value'))}) and cycle-state PC2 remains the stronger direct future8 baseline (AUC {fmt(masked_residual_transfer_pc2_future8.get('abs_oriented_auc'))}).",
        f"- Transfer-ranked ROI reconstruction converts that state-transfer hypothesis list back into direct video crops: {first_summary(transfer_ranked_reconstruction, 'n_cycles_sampled', 0)} cycles yielded {first_summary(transfer_ranked_reconstruction, 'n_reconstructed_candidates', 0)} reconstructed components and {first_summary(transfer_ranked_reconstruction, 'n_roi_rows', 0)} ROI rows; masked rollout on the exported crops again picks persistence as best for {transfer_ranked_rollout_best_counts.get('persistence', 0)} of {first_summary(transfer_ranked_rollout, 'n_roi', 0)} ROIs, while low-rank DMD particle residuals remain much larger than nonparticle context.",
        f"- Transfer-ranked front physics audit links those crops to phase/front proxies across {first_summary(transfer_ranked_front_physics, 'n_roi', 0)} ROIs; strongest ROI-level future8 association is {((transfer_ranked_front_target_tests[0] if transfer_ranked_front_target_tests else {}).get('feature', 'NA'))} with median positive-negative {fmt((transfer_ranked_front_target_tests[0] if transfer_ranked_front_target_tests else {}).get('median_positive_minus_negative'))}, AUC {fmt((transfer_ranked_front_target_tests[0] if transfer_ranked_front_target_tests else {}).get('abs_oriented_auc'))}, and permutation p={fmt((transfer_ranked_front_target_tests[0] if transfer_ranked_front_target_tests else {}).get('permutation_p_abs_median_diff'))}; radius/diffusion-like values remain apparent optical-front proxies only.",
        f"- Transfer-ranked residual transition timing gives a stronger temporal residual/phase link than the broader event-control cohort: {((transfer_ranked_timing_align[0] if transfer_ranked_timing_align else {}).get('method', 'NA'))} {((transfer_ranked_timing_align[0] if transfer_ranked_timing_align else {}).get('distance_feature', 'NA'))} median distance {fmt((transfer_ranked_timing_align[0] if transfer_ranked_timing_align else {}).get('median_distance_to_transition'))} versus null mean {fmt((transfer_ranked_timing_align[0] if transfer_ranked_timing_align else {}).get('null_median_distance_mean'))}, p={fmt((transfer_ranked_timing_align[0] if transfer_ranked_timing_align else {}).get('empirical_p_distance_le_observed'))}; top future8 timing target is {((transfer_ranked_timing_target[0] if transfer_ranked_timing_target else {}).get('method', 'NA'))} {((transfer_ranked_timing_target[0] if transfer_ranked_timing_target else {}).get('feature', 'NA'))}, AUC {fmt((transfer_ranked_timing_target[0] if transfer_ranked_timing_target else {}).get('abs_oriented_auc'))}.",
        f"- Cross-cohort rollout transfer audit shows the late transfer-ranked crops are a distinct video-dynamics domain: selected-cohort DMD evaluated on transfer-ranked ROIs has median particle MSE {fmt(selected_to_transfer_shift.get('median_particle_mse'))}, {fmt(selected_to_transfer_shift.get('median_particle_mse_ratio_vs_internal'))}x the transfer-internal DMD baseline (p={fmt(selected_to_transfer_shift.get('mwu_p_vs_internal'))}), while pooled training is close to transfer-internal ({fmt(pooled_to_transfer_shift.get('median_particle_mse_ratio_vs_internal'))}x).",
        f"- Multi-cohort future-drop model combines selected and transfer-ranked ROI physics features across {first_summary(multicohort_future_drop, 'n_roi_rows', 0)} rows / {first_summary(multicohort_future_drop, 'n_features', 0)} features; leave-cycle random forest reaches AUC {fmt(multicohort_best_oof.get('pooled_oof_roc_auc'))}/AP {fmt(multicohort_best_oof.get('pooled_oof_average_precision'))} with {fmt(multicohort_null.get('n_permutation'), 0)}-permutation p={fmt(multicohort_null.get('empirical_p_ge_observed'))}, but leave-cohort transfer is not evaluable because the selected cohort has no positive future8 labels.",
        f"- Active-learning QC prioritization merges manual-QC, precursor, weak-model, front, and timing evidence into {first_summary(active_learning_qc, 'n_candidate_rows', 0)} review candidates; {first_summary(active_learning_qc, 'n_rows_with_visual_asset', 0)} have visual assets and {active_qc_tiers.get('immediate_manual_qc', 0)} are immediate manual-QC picks, led by {(active_qc_top[0] if active_qc_top else {}).get('roi_id', 'NA')}. No manual labels are assigned.",
        f"- Automatic QC triage surrogate ranks the same pending-review bottleneck without assigning labels: {first_summary(automatic_qc_triage, 'likely_interpretable_count', 0)} likely interpretable candidates, {first_summary(automatic_qc_triage, 'artifact_risk_count', 0)} artifact-risk candidates, and {first_summary(automatic_qc_triage, 'diffusion_guardrail_count', 0)} diffusion-guardrail rows; top likely ROI is {(auto_qc_likely[0] if auto_qc_likely else {}).get('roi_id', 'NA')} and top artifact-risk ROI is {(auto_qc_artifact[0] if auto_qc_artifact else {}).get('roi_id', 'NA')}.",
        f"- QC decision evidence ledger converts the 47 pending labels into explicit reviewer actions without assigning labels: {qc_decision_actions}; top possible-accept ROI is {(qc_decision_accept[0] if qc_decision_accept else {}).get('roi_id', 'NA')}, while top artifact/reject-first ROI is {(qc_decision_artifact[0] if qc_decision_artifact else {}).get('roi_id', 'NA')}.",
        f"- Balanced future-drop direct-video audit removes the transfer-ranked class imbalance by sampling {first_summary(balanced_future_reconstruction, 'n_cycles_sampled', 0)} cycles and {first_summary(balanced_future_physics, 'n_roi', 0)} ROI rows with equal weak future8 positives/negatives; leave-cycle {balanced_future_best_oof.get('model', 'NA')} reaches AUC {fmt(balanced_future_best_oof.get('pooled_oof_roc_auc'))}/AP {fmt(balanced_future_best_oof.get('pooled_oof_average_precision'))}, permutation p={fmt(balanced_future_null.get('empirical_p_ge_observed'))}. Top positive-associated features are radius2/front-motion proxies and particle-mask rollout residual fractions, still under optical-proxy/manual-QC guardrails.",
        f"- Source-balanced ROI expansion attacks the remaining cohort-breadth bottleneck: it samples {first_summary(source_balanced_expansion, 'n_sampled_cycles', 0)} cycles across {first_summary(source_balanced_expansion, 'n_sources_selected', 0)} source movies, including {first_summary(source_balanced_expansion, 'n_new_selected_cycles', 0)} cycle/source pairs not already in video cohorts, and proposes {first_summary(source_balanced_expansion, 'n_roi_rows', 0)} automatic ROI rows for follow-up sequence export/QC.",
        f"- Source-balanced pre-event sampling addresses the future-specificity gap directly: it reconstructs {first_summary(source_balanced_pre_event_sampling, 'n_roi_rows', 0)} automatic ROI proposals from {first_summary(source_balanced_pre_event_sampling, 'n_sampled_cycles', 0)} event-relative cycles across {first_summary(source_balanced_pre_event_sampling, 'n_sources_selected', 0)} sources, with cycle bins {source_balanced_pre_event_bins} and {first_summary(source_balanced_pre_event_sampling, 'n_new_selected_cycles', 0)} new cycle/source pairs.",
        f"- Source-balanced pre-event sequence audit exports {first_summary(source_balanced_pre_event_sequences, 'n_roi_sequences', 0)} event-relative particle crops and finds near-pre-event spatial video structure under leave-source AUC {fmt(source_balanced_pre_event_near_source.get('roc_auc'))}/AP {fmt(source_balanced_pre_event_near_source.get('average_precision'))}, while broader any-pre transfer remains weak at AUC {fmt(source_balanced_pre_event_any_source.get('roc_auc'))}.",
        f"- Pre-event rollout/mask audits on those crops find top future16 ROI signals in {source_balanced_pre_event_rollout_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_rollout_best.get('oriented_auc'))} and {source_balanced_pre_event_mask_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_mask_best.get('oriented_auc'))}; event-relative clean-pre source-residual readout peaks at {source_balanced_pre_event_clean_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_clean_best.get('oriented_auc'))}.",
        f"- Pre-event event-distance trajectory audit collapses duplicate ROI proposals to {first_summary(source_balanced_pre_event_trajectory, 'n_cycle_rows', 0)} cycle rows and tests monotonic approach-to-event physics proxies; the leading source-residual physics trend is {source_balanced_pre_event_traj_best.get('feature', 'NA')} with rho {fmt(source_balanced_pre_event_traj_best.get('spearman_rho_vs_event_proximity'))} and source-stratified permutation p={fmt(source_balanced_pre_event_traj_best.get('within_source_permutation_abs_rho_p'))}.",
        f"- Pre-event directionality audit keeps ROI-level rows and compares pre-event versus post-event clocks with {fmt(first_summary(source_balanced_pre_event_directionality, 'n_permutations'), 0)} source-stratified clock permutations: physics-facing pre-clock feature {source_balanced_pre_event_dir_clock_top.get('feature', 'NA')} has rho {fmt(source_balanced_pre_event_dir_clock_top.get('pre_clock_rho'))}, permutation p={fmt(source_balanced_pre_event_dir_clock_top.get('pre_clock_perm_p_abs'))}, while the best source-residual clean-pre readout is {source_balanced_pre_event_dir_clean_sr.get('feature', 'NA')} at AUC {fmt(source_balanced_pre_event_dir_clean_sr.get('oriented_auc'))}.",
        f"- Pre-event source-invariant audit tests interpretable feature families under leave-source splits; clean-pre is led by {source_balanced_pre_event_si_clean.get('feature_family', 'NA')} / {source_balanced_pre_event_si_clean.get('method', 'NA')} at AUC {fmt(source_balanced_pre_event_si_clean.get('roc_auc'))}, while near-vs-far is led by {source_balanced_pre_event_si_near_far.get('feature_family', 'NA')} / {source_balanced_pre_event_si_near_far.get('method', 'NA')} at AUC {fmt(source_balanced_pre_event_si_near_far.get('roc_auc'))}.",
        f"- Pre-event review packet ranks {first_summary(source_balanced_pre_event_review_packet, 'n_candidates', 0)} automatic ROI crops and renders {first_summary(source_balanced_pre_event_review_packet, 'n_rendered_frame_strips', 0)} frame strips plus a contact sheet; top candidate is {source_balanced_pre_event_review_top1.get('roi_id', 'NA')} from {source_balanced_pre_event_review_top1.get('event_relative_bin', 'NA')} with review reason {source_balanced_pre_event_review_top1.get('review_reason', 'NA')}.",
        f"- Pre-event matched-counterfactual audit pairs near-pre ROIs against baseline/context-nearest controls; the strongest matched physics row is {source_balanced_pre_event_matched_best.get('comparison', 'NA')} {source_balanced_pre_event_matched_best.get('match_scheme', 'NA')} {source_balanced_pre_event_matched_best.get('feature', 'NA')} with median near-minus-control {fmt(source_balanced_pre_event_matched_best.get('median_treated_minus_control'))} and sign-flip p={fmt(source_balanced_pre_event_matched_best.get('signflip_mean_abs_p'))}, while q70 apparent diffusion remains weak at p={fmt(source_balanced_pre_event_matched_diffusion.get('signflip_mean_abs_p'))}.",
        f"- Same-source pre-event ladder audit shows the sampling gap explicitly: {source_balanced_pre_event_ladder_counts.get('sources_with_near_rows', 0)} sources have near-pre rows, {source_balanced_pre_event_ladder_counts.get('sources_with_near_mid_ladder', 0)} have near+mid ladders, and {source_balanced_pre_event_ladder_counts.get('sources_with_near_far_ladder', 0)} have near+far ladders; the strongest same-source paired row is {source_balanced_pre_event_ladder_best.get('comparison', 'NA')} {source_balanced_pre_event_ladder_best.get('feature', 'NA')} with p={fmt(source_balanced_pre_event_ladder_best.get('signflip_mean_abs_p'))}, while the global within-source clock remains weak (top rho {fmt(source_balanced_pre_event_ladder_clock_top.get('within_source_residual_spearman_rho_vs_event_proximity'))}).",
        f"- Raw pre-event source-lattice coverage confirms this is a real data-design gap, not a sampler artifact: {pre_event_lattice_near_counts.get('sources_with_near', 0)} raw sources have near-pre rows, {pre_event_lattice_near_counts.get('sources_with_near_far', 0)} have near+far rows, and {len(pre_event_lattice_far_controls)} separate sources can serve only as cross-source far-pre controls under source/acquisition matching.",
        f"- Pre-event radial-kymograph audit renders explicit front tracks for {first_summary(source_balanced_pre_event_radial_kymograph, 'n_roi', 0)} ROI crops; near-vs-far is led by {source_balanced_pre_event_kymo_near_far_top.get('feature', 'NA')} at AUC {fmt(source_balanced_pre_event_kymo_near_far_top.get('oriented_auc'))}, median diff {fmt(source_balanced_pre_event_kymo_near_far_top.get('median_positive_minus_negative'))}, p={fmt(source_balanced_pre_event_kymo_near_far_top.get('mwu_p'))}.",
        f"- Pre-event echem/front coupling audit joins {first_summary(source_balanced_pre_event_echem_front, 'n_rows', 0)} ROI rows to {fmt(first_summary(source_balanced_pre_event_echem_front, 'n_echem_features'), 0)} cycle-level echem descriptors; strongest echem link is {source_balanced_pre_event_echem_corr_top.get('echem_feature', 'NA')} vs {source_balanced_pre_event_echem_corr_top.get('target', 'NA')} rho {fmt(source_balanced_pre_event_echem_corr_top.get('spearman_rho'))}, while the best source+echem residual event-bin row is {source_balanced_pre_event_echem_resid_top.get('comparison', 'NA')} {source_balanced_pre_event_echem_resid_top.get('feature', 'NA')} at AUC {fmt(source_balanced_pre_event_echem_resid_top.get('roc_auc_treated_high'))}, p={fmt(source_balanced_pre_event_echem_resid_top.get('mannwhitney_p'))}.",
        f"- Pre-event echem-matched residual audit pairs near-pre rows on source/acquisition/echem context and tests source+echem residual outcomes; top row is {source_balanced_pre_event_echem_matched_resid_best.get('comparison', 'NA')} {source_balanced_pre_event_echem_matched_resid_best.get('match_scheme', 'NA')} {source_balanced_pre_event_echem_matched_resid_best.get('base_feature', 'NA')} with median near-control diff {fmt(source_balanced_pre_event_echem_matched_resid_best.get('median_treated_minus_control'))}, p={fmt(source_balanced_pre_event_echem_matched_resid_best.get('signflip_mean_abs_p'))}; best near-vs-far residual row is {source_balanced_pre_event_echem_matched_resid_far.get('base_feature', 'NA')} p={fmt(source_balanced_pre_event_echem_matched_resid_far.get('signflip_mean_abs_p'))}.",
        f"- Pre-event front-consensus audit combines residual outward-front, raw outward-front, quantile sign, and front-quality evidence across {first_summary(source_balanced_pre_event_front_consensus, 'n_rows', 0)} ROI rows; unpaired event-bin separation remains weak (best {source_balanced_pre_event_front_consensus_event_best.get('comparison', 'NA')} {source_balanced_pre_event_front_consensus_event_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_front_consensus_event_best.get('roc_auc_treated_high'))}, p={fmt(source_balanced_pre_event_front_consensus_event_best.get('mannwhitney_p'))}), but matched consensus is strong ({source_balanced_pre_event_front_consensus_matched_best.get('comparison', 'NA')} {source_balanced_pre_event_front_consensus_matched_best.get('match_scheme', 'NA')} {source_balanced_pre_event_front_consensus_matched_best.get('feature', 'NA')} median diff {fmt(source_balanced_pre_event_front_consensus_matched_best.get('median_treated_minus_control'))}, p={fmt(source_balanced_pre_event_front_consensus_matched_best.get('signflip_mean_abs_p'))}). Top consensus-ranked ROI is {source_balanced_pre_event_front_consensus_ranked_best.get('roi_id', 'NA')} from {source_balanced_pre_event_front_consensus_ranked_best.get('event_relative_bin', 'NA')}.",
        f"- Pre-event echem-matched far-control audit explicitly stress-tests the missing same-source near/far gap: source-class+echem/context matching forms {source_balanced_pre_event_echem_matched_far_pair_counts.get('same_source_class_echem_context', 0)} near-vs-far pairs from {source_balanced_pre_event_echem_matched_far_control_sources.get('same_source_class_echem_context', 0)} far-control sources; best row is {source_balanced_pre_event_echem_matched_far_best.get('match_scheme', 'NA')} {source_balanced_pre_event_echem_matched_far_best.get('feature', 'NA')} median diff {fmt(source_balanced_pre_event_echem_matched_far_best.get('median_treated_minus_control'))}, sign-flip p={fmt(source_balanced_pre_event_echem_matched_far_best.get('signflip_mean_abs_p'))}.",
        f"- Pre-event consensus review queue combines source-invariant review scores, radial fronts, source/echem residuals, and matched-control support into {first_summary(source_balanced_pre_event_consensus_review, 'n_candidates', 0)} ranked manual-QC candidates; {source_balanced_pre_event_consensus_tiers.get('matched_support_front_qc', 0)} are matched-support front-QC candidates, led by {source_balanced_pre_event_consensus_best.get('roi_id', 'NA')} at score {fmt(source_balanced_pre_event_consensus_best.get('consensus_review_score'))}.",
        f"- Pre-event consensus visual packet renders {first_summary(source_balanced_pre_event_consensus_visual, 'n_rendered', 0)} frame strips and {first_summary(source_balanced_pre_event_consensus_visual, 'n_rendered', 0)} radial kymographs for the consensus top candidates, with contact sheets for manual QC; top visual ROI is {source_balanced_pre_event_visual_best.get('roi_id', 'NA')}.",
        f"- Pre-event visual QC modes score the {first_summary(source_balanced_pre_event_visual_qc_modes, 'n_scored', 0)} rendered candidates with front-trace plausibility and artifact-risk features; none reach accepted-front status, {source_balanced_pre_event_visual_qc_tiers.get('front_plausible_followup', 0)} are follow-up candidates, and the top automatic QC candidate is {source_balanced_pre_event_visual_qc_best.get('roi_id', 'NA')} at score {fmt(source_balanced_pre_event_visual_qc_best.get('visual_review_score'))}.",
        f"- Pre-event phase-kinetics audit loads {first_summary(source_balanced_pre_event_phase_kinetics, 'n_ok', 0)} particle-region ROI tensors and tests {first_summary(source_balanced_pre_event_phase_kinetics, 'n_kinetic_features', 0)} masked optical kinetic features; best event-bin row is {source_balanced_pre_event_phase_kinetics_event_best.get('target', 'NA')} {source_balanced_pre_event_phase_kinetics_event_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_phase_kinetics_event_best.get('oriented_auc'))}, p={fmt(source_balanced_pre_event_phase_kinetics_event_best.get('mwu_p'))}.",
        f"- Pre-event front/kinetic concordance audit ranks {first_summary(source_balanced_pre_event_front_kinetic_concordance, 'n_rows', 0)} ROI rows by agreement between masked kinetics, front proxies, visual QC, and strict gates; {source_balanced_pre_event_fk_concordance_tiers.get('near_pre_front_kinetic_review', 0)} are near-pre front/kinetic review candidates, led by {source_balanced_pre_event_fk_concordance_best.get('roi_id', 'NA')} at score {fmt(source_balanced_pre_event_fk_concordance_best.get('front_kinetic_concordance_score'))}.",
        f"- Pre-event front/kinetic source-null audit stress-tests {first_summary(source_balanced_pre_event_front_kinetic_null, 'n_features_tested', 0)} concordance/kinetic/front features with {first_summary(source_balanced_pre_event_front_kinetic_null, 'n_permutations', 0)} source-stratified permutations; best row is {source_balanced_pre_event_fk_null_best.get('target', 'NA')} {source_balanced_pre_event_fk_null_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_fk_null_best.get('oriented_auc'))}, coarse perm p={fmt(source_balanced_pre_event_fk_null_best.get('source_stratified_perm_p_auc_ge_observed'))}.",
        f"- Pre-event manual-QC decision packet consolidates {first_summary(source_balanced_pre_event_manual_qc_decision, 'n_rows', 0)} ROI rows, {first_summary(source_balanced_pre_event_manual_qc_decision, 'n_visual_asset_rows', 0)} rendered visual-asset rows, and source-null evidence into action tiers; top action is {source_balanced_pre_event_manual_qc_best.get('manual_qc_action_tier', 'NA')} for {source_balanced_pre_event_manual_qc_best.get('roi_id', 'NA')} at decision score {fmt(source_balanced_pre_event_manual_qc_best.get('manual_qc_decision_score'))}.",
        f"- Manual-QC visual packet renders {first_summary(source_balanced_pre_event_manual_qc_visual, 'n_rendered', 0)} queue rows across {first_summary(source_balanced_pre_event_manual_qc_visual, 'n_sources_rendered', 0)} sources and action tiers {source_balanced_pre_event_manual_qc_visual_actions}; contact sheet {first_summary(source_balanced_pre_event_manual_qc_visual, 'contact_sheet', 'NA')}.",
        f"- Blinded manual-QC workbook randomizes {first_summary(source_balanced_pre_event_manual_qc_blind, 'n_blinded_rows', 0)} rendered candidates with seed {first_summary(source_balanced_pre_event_manual_qc_blind, 'seed', 'NA')}; reviewer-facing rows hide source/cycle/event/action metadata while the hidden key preserves action tiers {source_balanced_pre_event_manual_qc_blind_actions}.",
        f"- Pre-event multimodal leave-source predictor compares echem/front/QC/phase-kinetic families; best row is {source_balanced_pre_event_multimodal_best_row.get('target', 'NA')} {source_balanced_pre_event_multimodal_best_row.get('feature_family', 'NA')} {source_balanced_pre_event_multimodal_best_row.get('method', 'NA')} AUC {fmt(source_balanced_pre_event_multimodal_best_row.get('roc_auc'))}/AP {fmt(source_balanced_pre_event_multimodal_best_row.get('average_precision'))}, while the best kinetics-vs-front delta is dAUC {fmt(source_balanced_pre_event_multimodal_delta_best.get('roc_auc_delta'))}.",
        f"- Pre-event strict QC-gated front audit reduces {first_summary(source_balanced_pre_event_strict_qc_gated_front, 'n_candidates', 0)} rendered candidates to {first_summary(source_balanced_pre_event_strict_qc_gated_front, 'n_manual_front_review_candidates', 0)} automatic manual-front-review candidate and {first_summary(source_balanced_pre_event_strict_qc_gated_front, 'n_automatic_diffusion_claim_candidates', 0)} automatic diffusion-claim candidates; the surviving review candidate is {source_balanced_pre_event_strict_qc_best.get('roi_id', 'NA')} with strict QC score {fmt(source_balanced_pre_event_strict_qc_best.get('strict_qc_priority_score'))}.",
        f"- Pre-event physics-mode taxonomy clusters source-residual front/diffusion/heterogeneity features into k={fmt(first_summary(source_balanced_pre_event_physics_modes, 'chosen_k'), 0)} broad states but finds no strong near-pre enrichment (best Fisher p={fmt(source_balanced_pre_event_mode_enrich_best.get('fisher_p'))}), so continuous front/diffusion clocks remain more informative than coarse modes for this cohort.",
        f"- Source-balanced ROI sequence export converts that manifest into {first_summary(source_balanced_sequences, 'n_roi_sequences', 0)} particle-region crop tensors across {first_summary(source_balanced_sequences, 'n_cycles', 0)} cycles and {first_summary(source_balanced_sequences, 'n_sources', 0)} sources with {first_summary(source_balanced_sequences, 'n_failed', 0)} export failures; the fast rollout audit finds strongest future16 ROI signal in {source_balanced_rollout_top16.get('feature', 'NA')} at AUC {fmt(source_balanced_rollout_top16.get('oriented_auc'))}, while prediction-error features are highly source-structured.",
        f"- Source-balanced sequence source-control audit stress-tests those rollout features across {first_summary(source_balanced_sequence_source_control, 'n_rows', 0)} rows and {first_summary(source_balanced_sequence_source_control, 'n_sources', 0)} sources; verdict `{first_summary(source_balanced_sequence_source_control, 'verdict', 'missing')}` with {first_summary(source_balanced_sequence_source_control, 'n_strict_scalar_rows', 0)} strict scalar rows and {first_summary(source_balanced_sequence_source_control, 'n_source_model_auc_ge_065', 0)} source-heldout model rows above AUC 0.65.",
        f"- A source-balanced mask/front sanity audit adds crop-local particle masks, centroid stability, radial front proxies, and apparent q70 radius-squared slopes across {first_summary(source_balanced_mask_front, 'n_roi_sequences', 0)} ROI tensors; top future16 mask/front proxy is {source_balanced_mask_front_top16.get('feature', 'NA')} at AUC {fmt(source_balanced_mask_front_top16.get('oriented_auc'))}/AP {fmt(source_balanced_mask_front_top16.get('average_precision'))}, but source eta2 is {fmt(source_balanced_mask_front_top16.get('source_eta2'))}.",
        f"- A source-residual mask/front audit tests whether those crop-local descriptors survive source structure: best source-residual future16 proxy is {source_balanced_mask_front_resid_best.get('feature', 'NA')} at AUC {fmt(source_balanced_mask_front_resid_best.get('oriented_auc'))}/AP {fmt(source_balanced_mask_front_resid_best.get('average_precision'))}, and best within-source-rank proxy is {source_balanced_mask_front_rank_best.get('feature', 'NA')} at AUC {fmt(source_balanced_mask_front_rank_best.get('oriented_auc'))}/AP {fmt(source_balanced_mask_front_rank_best.get('average_precision'))}.",
        f"- A source-balanced residual dictionary learns label-free next-frame residual bases on the same 96 crop tensors; residual_dictionary leave-cycle future16 reaches AUC {fmt(source_balanced_resdict_cycle16.get('roc_auc'))}/AP {fmt(source_balanced_resdict_cycle16.get('average_precision'))}, but leave-source future16 drops to AUC {fmt(source_balanced_resdict_source16.get('roc_auc'))}, marking source transfer as the main failure mode.",
        f"- Source-normalizing the source-balanced residual dictionary leaves a source-residual future16 residual-dynamics candidate, {source_balanced_resdict_sr_best.get('feature', 'NA')}, at AUC {fmt(source_balanced_resdict_sr_best.get('oriented_auc'))}/AP {fmt(source_balanced_resdict_sr_best.get('average_precision'))} with source eta2 {fmt(source_balanced_resdict_sr_best.get('source_eta2_after_transform'))}; within-source-rank residual PCs are weaker at AUC {fmt(source_balanced_resdict_rank_best.get('oriented_auc'))}.",
        f"- The grouped normalized residual-dictionary readout partially rescues held-out-source future16 transfer: raw residual dictionary AUC {fmt(source_balanced_resdict_norm_raw_source16.get('roc_auc'))} improves to {fmt(source_balanced_resdict_norm_sr_source16.get('roc_auc'))} after source residualization, while the single {source_balanced_resdict_norm_best.get('feature_set', 'NA')} readout reaches AUC {fmt(source_balanced_resdict_norm_best.get('roc_auc'))}/AP {fmt(source_balanced_resdict_norm_best.get('average_precision'))}; permutation p={fmt(source_balanced_resdict_norm_perm.get('empirical_p_roc_auc'))} keeps it provisional.",
        f"- Temporal-specificity controls show the same source-residual reconstruction-error drift is temporally ordered but not cleanly precursor-specific: future16 AUC {fmt(source_balanced_temporal_primary16.get('future_auc'))} beats a within-source shift null (p={fmt(source_balanced_temporal_shift16.get('empirical_p_roc_auc'))}) but barely exceeds past16 AUC {fmt(source_balanced_temporal_primary16.get('past_auc_fixed_direction'))}; raw masked-minus-background slope is more future8-specific but source structured.",
        f"- Future-specific residual controls sharpen that guardrail: after excluding past16 rows, the primary source-residual reconstruction-error feature drops to AUC {fmt(source_balanced_future_specific_primary16_clean.get('oriented_auc'))}, while grouped future16 prediction gains only modestly over past-event context (best delta AUC {fmt(source_balanced_future_specific_top_delta.get('delta_roc_auc'))} from {source_balanced_future_specific_top_delta.get('feature_set', 'NA')}).",
        f"- Source-balanced degradation-mode audit clusters source-residual residual/front/contrast features into k={fmt(first_summary(source_balanced_degradation_modes, 'chosen_k'), 0)} modes; the strongest enrichment is mode {source_balanced_degmode_top.get('mode', 'NA')} for {source_balanced_degmode_top.get('label', 'NA')} (fraction {fmt(source_balanced_degmode_top.get('mode_fraction'))}, p={fmt(source_balanced_degmode_top.get('fisher_p'))}), but tiny outlier modes keep this as review triage rather than a stable taxonomy.",
        f"- Source-balanced residual-physics coupling links the best source-residual dictionary candidate to crop-local physics proxies: top target-aligned pair is {source_balanced_resphys_top_aligned.get('residual_feature', 'NA')} vs {source_balanced_resphys_top_aligned.get('physics_feature', 'NA')} with rho {fmt(source_balanced_resphys_top_aligned.get('spearman_rho'))}, residual AUC {fmt(source_balanced_resphys_top_aligned.get('residual_future16_auc'))}, and physics AUC {fmt(source_balanced_resphys_top_aligned.get('physics_future16_auc'))}; apparent diffusion coupling remains weak.",
        f"- Source-balanced residual candidate review packet converts the residual/readout/coupling evidence into {first_summary(source_balanced_residual_candidate_review, 'n_candidates', 0)} pending manual-QC candidates; {source_balanced_review_tiers.get('immediate_manual_qc', 0)} are immediate-review, led by {source_balanced_review_top1.get('roi_id', 'NA')} with score {fmt(source_balanced_review_top1.get('review_priority_score'))}.",
        f"- Balanced future particle-mask stability audit covers {first_summary(balanced_future_mask, 'n_roi', 0)} ROIs / {first_summary(balanced_future_mask, 'n_frames_total', 0)} frames; median fallback fraction is {fmt(balanced_future_mask_overall.get('median_fallback_frame_fraction'))}, and the strongest future8 mask-stability contrast is {balanced_future_mask_top_test.get('feature', 'NA')} with p={fmt(balanced_future_mask_top_test.get('p_value'))}, so the balanced future signal is not explained by a simple mask-instability split.",
        f"- Masked video embedding audit extracts particle-prior self-supervised descriptors across {first_summary(masked_video_embedding, 'n_embedding_rows', 0)} ROI tensors; balanced future leave-cycle AUC/AP is {fmt(masked_video_future_metric.get('pooled_oof_roc_auc'))}/{fmt(masked_video_future_metric.get('pooled_oof_average_precision'))} with label-permutation p={fmt(masked_video_null.get('empirical_p_ge_observed'))}, while selected event/control readout is weaker at AUC {fmt(masked_video_event_metric.get('pooled_oof_roc_auc'))}.",
        f"- Learned residual-CNN embeddings trained label-free for next-frame residual prediction reach future8 leave-cycle AUC {fmt(learned_residual_future8.get('roc_auc'))} versus PCA-video {fmt(learned_residual_pca_future8.get('roc_auc'))} and handcrafted scalar {fmt(learned_residual_hand_future8.get('roc_auc'))}; future16 learned_all remains weak at AUC {fmt(learned_residual_future16.get('roc_auc'))} versus handcrafted {fmt(learned_residual_hand_future16.get('roc_auc'))}.",
        f"- Residual dictionary embedding learns label-free next-frame residual bases over {first_summary(residual_dictionary_embedding, 'n_embedding_rows', 0)} ROI videos; residual-dictionary future8 AUC is {fmt(residual_dict_future8.get('roc_auc'))} with p={fmt(residual_dict_future8.get('empirical_p_ge_observed'))}, and residual_dictionary_plus_handcrafted reaches AUC {fmt(residual_dict_plus_future8.get('roc_auc'))}.",
        f"- Echem residual-dictionary fusion shows conditioning boosts residual-dictionary future8 AUC to {fmt(echem_resdict_res_future8.get('roc_auc'))}, while acquisition/context alone reaches {fmt(echem_resdict_acq_future8.get('roc_auc'))}; treat this as context-sensitive representation evidence rather than deployable warning.",
        f"- Echem-conditioned residual-dictionary audit converts post-hoc fusion into a split-specific residual objective: conditioned residual dictionary future16 reaches leave-source AUC {fmt(echem_cond_resdict_ls16.get('roc_auc'))} versus raw residual dictionary {fmt(echem_cond_resdict_raw_ls16.get('roc_auc'))} (delta {fmt(echem_cond_resdict_delta_ls16.get('delta_roc_auc'))}), while leave-cycle conditioned residual+echem reaches AUC {fmt(echem_cond_resdict_lc16_plus.get('roc_auc'))}; future8 remains context dominated.",
        f"- Conditioned residual physics atlas makes that objective interpretable: top source-centered physics alignment is {cond_atlas_top_align.get('split', 'NA')} {cond_atlas_top_align.get('residual_base', 'NA')} to {cond_atlas_top_align.get('physics_feature', 'NA')} ({cond_atlas_top_align.get('physics_category', 'NA')}) with rho {fmt(cond_atlas_top_align.get('source_centered_rho'))}; top single residual future16 mode is {cond_atlas_top_target.get('residual_base', 'NA')} at AUC {fmt(cond_atlas_top_target.get('oriented_auc'))}/AP {fmt(cond_atlas_top_target.get('average_precision'))}.",
        f"- Acquisition-residualized video benchmark confirms the context guardrail: future8 acquisition context reaches AUC {fmt(acq_resid_context8.get('roc_auc'))}, raw all-video reaches {fmt(acq_resid_raw_video8.get('roc_auc'))}, and context-residualized all-video alone reaches {fmt(acq_resid_video_only8.get('roc_auc'))}; future16 raw handcrafted reaches AUC {fmt(acq_resid_raw_hand16.get('roc_auc'))} but residualized all-video alone is {fmt(acq_resid_video_only16.get('roc_auc'))}.",
        f"- Acquisition-residualized video/echem warning audit executes the top tournament experiment: leave-cycle future16 residualized video_plus_echem reaches AUC {fmt(acq_echem_cycle_future16.get('roc_auc'))} versus acquisition-only {fmt(acq_echem_cycle_acq_future16.get('roc_auc'))}, but leave-source residualized AUC falls to {fmt(acq_echem_source_future16.get('roc_auc'))} versus acquisition-only {fmt(acq_echem_source_acq_future16.get('roc_auc'))}.",
        f"- Source-domain video/echem adaptation partially rescues leave-source future16 transfer: source-centered video_plus_echem reaches AUC {fmt(source_domain_centered.get('roc_auc'))} versus acquisition-only {fmt(source_domain_acq.get('roc_auc'))}, while CORAL reaches only {fmt(source_domain_coral.get('roc_auc'))}.",
        f"- Source-balanced transfer audit shows source-rank/weighting only modestly lifts video+echem future16 AUC to {fmt(source_balanced_vpe16.get('roc_auc'))} versus raw video+echem {fmt(source_balanced_raw16.get('roc_auc'))}, below acquisition context {fmt(source_balanced_acq16.get('roc_auc'))} and echem source-rank {fmt(source_balanced_echem16.get('roc_auc'))}; source label composition remains the dominant guardrail.",
        f"- Source-invariant projection is more promising but still guarded: best video_plus_echem future16 is {source_invariant_best_vpe16.get('method', 'NA')} at AUC {fmt(source_invariant_best_vpe16.get('roc_auc'))} versus raw {fmt(source_invariant_raw16.get('roc_auc'))} and acquisition context {fmt(source_invariant_acq16.get('roc_auc'))}; video-only source-confound filtering reaches AUC {fmt(source_invariant_best_video16.get('roc_auc'))}.",
        f"- Source-invariant physical-family audit localizes the future16 rescue to normalized heterogeneity and particle-vs-context contrast: norm-heterogeneity source_mean_resid_4 reaches AUC {fmt(source_family_norm16.get('roc_auc'))}, contrast source_mean_resid_4 reaches {fmt(source_family_contrast16.get('roc_auc'))}, while raw embedding alone is {fmt(source_family_embed16.get('roc_auc'))}.",
        f"- Exact-feature source-invariant audit nominates {source_interpretable_top_uni.get('feature', 'NA')} as the strongest univariate future16 descriptor (oriented AUC {fmt(source_interpretable_top_uni.get('roc_auc_oriented'))}, source eta2 {fmt(source_interpretable_top_uni.get('source_eta2'))}); best small transfer set {source_interpretable_top_combo.get('feature_set', 'NA')} reaches leave-source AUC {fmt(source_interpretable_top_combo.get('roc_auc'))}.",
        f"- Exact-feature mechanism consistency audit is a useful falsification check: exact_optical_loss_score predicts future16 with AUC {fmt(exact_mech_loss_metric.get('oriented_auc'))} but has source eta2 {fmt(exact_mech_loss_metric.get('source_eta2'))}; the primary particle-vs-context descriptor has weak radius-slope linkage after source residualization (rho {fmt(exact_mech_primary_radius_corr.get('source_residual_spearman_rho'))}).",
        f"- Signed optical-loss mechanism audit converts the pattern into interpretable axes: combined_loss_mechanism_axis future16 AUC is {fmt(signed_loss_top_axis.get('oriented_auc'))} but source eta2 is {fmt(signed_loss_top_axis.get('source_eta2'))}; leave-source all-axis AUC is {fmt(signed_loss_best_model16.get('roc_auc'))}, while optical-only AUC is {fmt((next((r for r in signed_loss_models if r.get('target') == 'future_any_drop_within_16cycles' and r.get('feature_set') == 'optical_loss_only'), {}) or {}).get('roc_auc'))}.",
        f"- Signed-loss source robustness audit shows why this remains guarded: combined-axis raw/source-mean/within-source-rank AUCs are {fmt(signed_robust_combined_raw.get('oriented_auc'))}/{fmt(signed_robust_combined_source_mean.get('oriented_auc'))}/{fmt(signed_robust_combined_rank.get('oriented_auc'))}; optical-axis raw/source-mean/within-source-rank AUCs are {fmt(signed_robust_optical_raw.get('oriented_auc'))}/{fmt(signed_robust_optical_source_mean.get('oriented_auc'))}/{fmt(signed_robust_optical_rank.get('oriented_auc'))}.",
        f"- Source-residual echem/optical audit finds low-source-eta residual evidence: echem+optical+front direct AUC {fmt(echem_optical_front_direct.get('oriented_auc'))}, echem+optical direct AUC {fmt(echem_optical_resid_direct.get('oriented_auc'))}, and leave-source echem+optical residual AUC {fmt(echem_optical_resid_model.get('roc_auc'))}.",
        f"- Invariant sparse rule discovery finds review-prioritization rules rather than a standalone predictor: best leave-source rule {invariant_rule_best.get('terms', 'NA')} covers {fmt(invariant_rule_best.get('n_covered'), 0)}/{fmt(invariant_rule_best.get('n_eval'), 0)} rows with precision {fmt(invariant_rule_best.get('precision'))}, lift {fmt(invariant_rule_best.get('lift'))}, and source-positive hits in {fmt(invariant_rule_best.get('n_sources_with_positive_hits'), 0)} sources.",
        f"- Current-evidence agentic hypothesis tournament ranks the next paper-inspired experiment as {first_summary(agentic_current, 'top_hypothesis', {}).get('title', 'NA')} with score {fmt(first_summary(agentic_current, 'top_hypothesis', {}).get('tournament_score'))}.",
        f"- Balanced future context/region guardrail shows acquisition/spatial context alone predicts weak future8 labels strongly (best AUC {fmt(balanced_future_best_acq_context.get('pooled_oof_roc_auc'))}), while selection-design context is perfect by construction (AUC {fmt(balanced_future_best_design_context.get('pooled_oof_roc_auc'))}); after acquisition-context residualization, the top physics residual is {balanced_future_top_acq_resid.get('feature', 'NA')} with p={fmt(balanced_future_top_acq_resid.get('mannwhitney_p'))}. Treat balanced physics features as review hypotheses, not context-independent degradation detectors.",
        f"- Temporal directionality audit supports a precursor interpretation but not a causal claim: balanced ROI physics predicts future8 with {temporal_future8.get('model', 'NA')} AUC {fmt(temporal_future8.get('pooled_oof_roc_auc'))}/AP {fmt(temporal_future8.get('pooled_oof_average_precision'))}, beating circular time-shift labels at empirical p={fmt(temporal_shift_null.get('empirical_p_ge_observed'))}; reversed labels remain nontrivial (best AUC {fmt(temporal_reversed8.get('pooled_oof_roc_auc'))}) and past8 is underpowered with {temporal_past8_counts.get('1', 0)} positives.",
        f"- Balanced spatial front-propagation audit builds {first_summary(balanced_spatial_propagation, 'n_edges', 0)} spatial kNN edges over {first_summary(balanced_spatial_propagation, 'n_nodes', 0)} balanced ROI nodes; nearest next-cycle front descriptors autocorrelate strongly ({spatial_prop_top_autocorr.get('feature', 'NA')} rho={fmt(spatial_prop_top_autocorr.get('spearman_src_dst'))}, p_perm={fmt(spatial_prop_top_autocorr.get('empirical_p_abs_ge_observed'))}) and same future8-label homophily is high ({fmt(spatial_prop_top_homophily.get('observed_same_future8_fraction'))}), but automatic ROI identity and cycle-level labels keep this as spatial hypothesis ranking.",
        f"- Masked residual transition timing finds low-rank DMD residual weighted centers are closer to automatic phase-transition centers than random at borderline strength (empirical p={fmt((masked_residual_timing_align[0] if masked_residual_timing_align else {}).get('empirical_p_distance_le_observed'))}), but peak-frame timing is not aligned and persistence particle/nonparticle ratios track kinetic rates.",
        f"- Weak-label degradation benchmark converts consensus physics/mode/mask evidence into a guarded manifest: {first_summary(weak_label_benchmark, 'n_trainable_weak_label_rows', 0)} trainable weak rows ({first_summary(weak_label_benchmark, 'n_positive_weak_labels', 0)} positive / {first_summary(weak_label_benchmark, 'n_negative_weak_labels', 0)} negative), and only {weak_label_leakage.get('n_usable_binary_folds', 0)} leave-reference fold is class-balanced enough for binary evaluation.",
        "",
        "## Model Readout",
        "",
        f"- Strict no-selection-QC random forest: ROC-AUC {fmt(strict_rf.get('mean_roc_auc'))}, balanced accuracy {fmt(strict_rf.get('mean_balanced_accuracy'))}.",
        f"- Strict no-selection-QC logistic: ROC-AUC {fmt(strict_logistic.get('mean_roc_auc'))}, balanced accuracy {fmt(strict_logistic.get('mean_balanced_accuracy'))}.",
        f"- All physics plus QC random forest: ROC-AUC {fmt(all_rf.get('mean_roc_auc'))}, balanced accuracy {fmt(all_rf.get('mean_balanced_accuracy'))}.",
        f"- Protocol-conditioned residual logistic: ROC-AUC {fmt(conditioned_model.get('mean_roc_auc'))}, balanced accuracy {fmt(conditioned_model.get('mean_balanced_accuracy'))}.",
        f"- Protocol-conditioned front-residual logistic: ROC-AUC {fmt(conditioned_front_model.get('mean_roc_auc'))}, balanced accuracy {fmt(conditioned_front_model.get('mean_balanced_accuracy'))}.",
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
        "## Prefix-Only ROI Forecasts",
        "",
    ]
    for row in prefix_top_classification:
        report_lines.append(
            f"- {row.get('target')} {row.get('feature_set')} {row.get('model')} f={fmt(row.get('prefix_fraction'))}: AUC {fmt(row.get('mean_roc_auc'))}, balanced accuracy {fmt(row.get('mean_balanced_accuracy'))}"
        )
    for row in prefix_null:
        report_lines.append(
            f"- Null check {row.get('target')} f={fmt(row.get('prefix_fraction'))}: observed AUC {fmt(row.get('observed_mean_auc'))}, null p95 {fmt(row.get('null_p95_auc'))}, empirical p={fmt(row.get('empirical_p_ge_observed'))}"
        )
    for row in prefix_top_regression[:4]:
        report_lines.append(
            f"- Regression {row.get('target')} {row.get('feature_set')} {row.get('model')} f={fmt(row.get('prefix_fraction'))}: MAE ratio vs median baseline {fmt(row.get('mean_mae_ratio_vs_baseline'))}, rho {fmt(row.get('mean_spearman_rho'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(prefix_forecast, 'guardrail', 'Small selected cohort; use as triage only.')}")

    report_lines += [
        "",
        "## Prefix ROI Feature Importance",
        "",
        f"- Rows/features: {first_summary(prefix_importance, 'n_rows', 0)} / {first_summary(prefix_importance, 'n_features', 0)}",
        f"- Model/target: {first_summary(prefix_importance, 'model', 'NA')} / {first_summary(prefix_importance, 'target', 'NA')}",
        f"- Pooled OOF AUC: {fmt(prefix_importance_model.get('pooled_oof_roc_auc'))}; balanced accuracy {fmt(prefix_importance_model.get('pooled_oof_balanced_accuracy'))}",
        f"- Null check: p95 AUC {fmt(prefix_importance_null.get('null_p95_auc'))}, empirical p={fmt(prefix_importance_null.get('empirical_p_ge_observed'))}",
    ]
    for row in prefix_importance_groups[:5]:
        report_lines.append(
            f"- Group ablation remove {row.get('removed_group')}: AUC drop {fmt(row.get('pooled_oof_auc_drop'))}, n_features={fmt(row.get('n_removed_features'), 0)}"
        )
    for row in prefix_importance_features[:6]:
        report_lines.append(
            f"- Feature permutation {row.get('feature')} ({row.get('feature_group')}): AUC drop {fmt(row.get('mean_auc_drop'))}, positive-fold fraction {fmt(row.get('positive_drop_fraction'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(prefix_importance, 'guardrail', 'Feature rankings are descriptive only.')}")

    report_lines += [
        "",
        "## Top Protocol-Conditioned Event Effects",
        "",
    ]
    for row in conditioned_tests:
        report_lines.append(
            f"- {row.get('feature')}: event-control {fmt(row.get('event_minus_control'))}, p={fmt(row.get('p_value'))}"
        )

    report_lines += [
        "",
        "## Threshold-Robust Front Readout",
        "",
    ]
    for row in robust_front_tests:
        report_lines.append(
            f"- {row.get('feature')}: event-control {fmt(row.get('event_minus_control'))}, p={fmt(row.get('p_value'))}"
        )
    report_lines.append("- By-event strongest cases include cycle 156 phase-slope separation and cycle 86 radius/diffusion-proxy differences; all remain apparent optical proxies.")


    report_lines += [
        "",
        "## Protocol-Conditioned Front Effects",
        "",
    ]
    for row in conditioned_front_tests:
        report_lines.append(
            f"- {row.get('feature')}: event-control {fmt(row.get('event_minus_control'))}, p={fmt(row.get('p_value'))}"
        )
    report_lines.append("- Conditioning preserves phase-slope sign consistency but weakens magnitude and diffusion-proxy effects; front residuals are not a standalone detector.")

    report_lines += [
        "",
        "## Front QC Sensitivity",
        "",
    ]
    for row in front_qc_strata:
        report_lines.append(
            f"- {row.get('stratum')}: n={fmt(row.get('n_roi'), 0)}, event/control {fmt(row.get('n_event'), 0)}/{fmt(row.get('n_control'), 0)}, phase-CI-positive fraction {fmt(row.get('phase_ci_positive_fraction'))}"
        )
    for row in front_qc_focus:
        if row.get("feature") == "phase_slope_positive_fraction_protocol_residual":
            report_lines.append(
                f"- {row.get('stratum')} phase-sign residual: median event-control {fmt(row.get('median_event_minus_control'))}, bootstrap p05 {fmt(row.get('bootstrap_p05'))}, MW p={fmt(row.get('mannwhitney_p'))}, permutation p={fmt(row.get('permutation_median_p'))}"
            )
    report_lines.append("- Diffusion-proxy separations do not remain globally robust; review-panel diffusion differences are selection-biased and still require manual QC.")
    report_lines.append(
        f"- Control-balanced QC augmentation selects {first_summary(control_balanced_qc, 'n_selected_roi', 0)} ROIs ({first_summary(control_balanced_qc, 'n_event_roi', 0)} event / {first_summary(control_balanced_qc, 'n_control_roi', 0)} control), with non-fragmented counts {control_balanced_nonfragment}."
    )
    for row in balanced_qc_focus:
        if row.get("feature") == "phase_slope_positive_fraction_protocol_residual" and row.get("stratum") in {"balanced_qc_selected", "balanced_qc_not_fragmented"}:
            report_lines.append(
                f"- Control-balanced sensitivity {row.get('stratum')} phase-sign residual: event/control {fmt(row.get('n_event'), 0)}/{fmt(row.get('n_control'), 0)}, median event-control {fmt(row.get('median_event_minus_control'))}, bootstrap p05 {fmt(row.get('bootstrap_p05'))}, MW p={fmt(row.get('mannwhitney_p'))}, permutation p={fmt(row.get('permutation_median_p'))}."
            )
    if balanced_robust_phase_strata:
        report_lines.append(f"- Control-balanced robust positive phase-residual strata: {', '.join(balanced_robust_phase_strata)}.")

    report_lines += [
        "",
        "## Residual Physics Mode Taxonomy",
        "",
        f"- Selected k={first_summary(residual_modes, 'chosen_k', 'NA')} with silhouette={fmt(first_summary(residual_modes, 'cluster_selection', {}).get('best', {}).get('silhouette'))} and mean seed-stability ARI={fmt(first_summary(residual_modes, 'cluster_stability', {}).get('mean_adjusted_rand_index'))}.",
    ]
    for row in residual_mode_enrichment:
        report_lines.append(
            f"- {row.get('mode_label')}: n={fmt(row.get('n_roi'), 0)}, event fraction {fmt(row.get('event_fraction'))}, p={fmt(row.get('fisher_p_value'))}, cycles {row.get('top_cycles')}"
        )
    report_lines.append(
        f"- Guardrail: {first_summary(residual_modes, 'guardrail', 'Treat these as computational hypotheses pending manual QC.')}"
    )

    report_lines += [
        "",
        "## Cycle/Region Mode Context",
        "",
    ]
    for row in cycle_region_context[:6]:
        report_lines.append(
            f"- cycle {fmt(row.get('cycleNo'), 0)}: n={fmt(row.get('n_roi'), 0)}, event-enriched mode fraction={fmt(row.get('event_enriched_mode_fraction'))}, top modes={row.get('top_modes')}"
        )
    for row in spatial_region_context[:4]:
        report_lines.append(
            f"- region {row.get('xy_region')}: n={fmt(row.get('n_roi'), 0)}, event-enriched mode fraction={fmt(row.get('event_enriched_mode_fraction'))}, event fraction={fmt(row.get('event_fraction'))}"
        )
    clf = first_summary(cycle_region_modes, 'context_only_classifier', {})
    report_lines.append(
        f"- Context-only leave-cycle-out classifier: pooled ROC-AUC {fmt(clf.get('pooled_roc_auc'))}, pooled balanced accuracy {fmt(clf.get('pooled_balanced_accuracy'))}; descriptive context is not a standalone detector."
    )

    report_lines += [
        "",
        "## Spatiotemporal Degradation Graph",
        "",
        f"- Graph size: {first_summary(spatiotemporal_graph, 'n_nodes', 0)} ROI nodes and {first_summary(spatiotemporal_graph, 'n_edges', 0)} directed nearest-neighbor edges.",
    ]
    for row in graph_homophily[:6]:
        report_lines.append(
            f"- Homophily {row.get('edge_type')} {row.get('target')}: same fraction {fmt(row.get('observed_same_fraction'))}, null mean {fmt(row.get('null_same_mean'))}, empirical p={fmt(row.get('empirical_p_same_ge_observed'))}"
        )
    for row in graph_continuous[:6]:
        report_lines.append(
            f"- Continuous neighbor {row.get('edge_type')} {row.get('target')}: rho {fmt(row.get('spearman_src_dst'))}, null p95 {fmt(row.get('null_p95'))}, empirical p={fmt(row.get('empirical_p_abs_ge_observed'))}"
        )
    for row in graph_lag[:4]:
        metric = row.get('roc_auc_previous_neighbor_predicts_current', row.get('spearman_previous_nearest_vs_current'))
        report_lines.append(
            f"- Temporal lag {row.get('target')}: n={fmt(row.get('n_lag_pairs'), 0)}, previous-neighbor metric {fmt(metric)}"
        )
    for row in graph_distance[:3]:
        report_lines.append(
            f"- Distance gradient {row.get('target')}: positive-positive median distance {fmt(row.get('median_distance_positive_positive'))} px vs other {fmt(row.get('median_distance_other'))} px, p={fmt(row.get('mannwhitney_p'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(spatiotemporal_graph, 'guardrail', 'Automatic graph audit only.')}")

    report_lines += [
        "",
        "## Phase Kinetics Avrami Audit",
        "",
        f"- ROI/features: {first_summary(phase_kinetics, 'n_roi', 0)} / {first_summary(phase_kinetics, 'n_kinetic_features', 0)}",
    ]
    for row in phase_kinetics_tests[:8]:
        report_lines.append(
            f"- {row.get('comparison')} {row.get('feature')}: median difference {fmt(row.get('median_positive_minus_negative'))}, p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in phase_kinetics_corr[:8]:
        report_lines.append(
            f"- Correlation {row.get('feature')} vs {row.get('target')}: rho {fmt(row.get('spearman_rho'))}, p={fmt(row.get('spearman_p'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(phase_kinetics, 'guardrail', 'Optical kinetic proxy only.')}")

    report_lines += [
        "",
        "## Calibration Metadata Audit",
        "",
        f"- HDF5 files/movie datasets/camera timing: {first_summary(calibration_metadata, 'n_h5_files', 0)} / {first_summary(calibration_metadata, 'n_h5_with_movie', 0)} / {first_summary(calibration_metadata, 'n_h5_with_camera_timing', 0)}",
        f"- HDF5 calibration-like attribute hits: {first_summary(calibration_metadata, 'n_h5_with_calibration_attr_hits', 0)}",
        f"- Median sampled HDF5 timing FPS proxy: {fmt(first_summary(calibration_metadata, 'fps_median_across_h5'))}",
        f"- PPTX files scanned/calibration hits: {first_summary(calibration_metadata, 'n_pptx_files_scanned', 0)} / {first_summary(calibration_metadata, 'n_pptx_calibration_hits', 0)}",
    ]
    for row in top_items(first_summary(calibration_metadata, 'top_pptx_hits', []), 3):
        report_lines.append(
            f"- PPTX hit {Path(str(row.get('pptx_path'))).name} slide {fmt(row.get('slide'), 0)}: {str(row.get('snippets', ''))[:220]}"
        )
    report_lines.append(f"- Guardrail: {first_summary(calibration_metadata, 'guardrail', 'Calibration metadata audit unavailable.')}")

    report_lines += [
        "",
        "## Calibration Provenance Evidence Audit",
        "",
        f"- Provenance status: {first_summary(calibration_provenance, 'provenance_status', 'missing')}",
        f"- Files inventoried / HDF5 scanned: {first_summary(calibration_provenance, 'n_files_inventoried', 0)} / {first_summary(calibration_provenance, 'n_h5_files_scanned', 0)}",
        f"- Raw HDF5 calibration-like / explicit spatial-scale statements: {first_summary(calibration_provenance, 'n_raw_h5_calibration_like_statements', 0)} / {first_summary(calibration_provenance, 'n_raw_h5_spatial_scale_statements', 0)}",
        f"- Near-96 nm/px / contradictory statements: {first_summary(calibration_provenance, 'n_near_96nm_px_statements', 0)} / {first_summary(calibration_provenance, 'n_contradictory_scale_statements', 0)}",
        f"- Highest near-scale evidence: {first_summary(calibration_provenance, 'highest_scale_evidence_strength', 'none')}",
        f"- Raw movie spatial shapes: {first_summary(calibration_provenance, 'unique_movie_spatial_shapes', [])}",
    ]
    for row in top_items(first_summary(calibration_provenance, 'near_96nm_px_examples', []), 3):
        report_lines.append(
            f"- Near-scale example {Path(str(row.get('relative_path'))).name} {row.get('statement_type')}: {fmt(row.get('inferred_nm_per_px'))} nm/px from {row.get('evidence_strength')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(calibration_provenance, 'interpretation', 'Calibration provenance evidence audit unavailable.')}")

    report_lines += [
        "",
        "## Calibration Claim Risk Register",
        "",
        f"- Claim families/source tables: {first_summary(calibration_claim_risk, 'n_claim_families', 0)} / {first_summary(calibration_claim_risk, 'n_source_tables_present', 0)}",
        f"- Calibration evidence: {first_summary(calibration_claim_risk, 'calibration_evidence', {}).get('h5_files_with_camera_timing', 0)} HDF5 timing files, {first_summary(calibration_claim_risk, 'calibration_evidence', {}).get('h5_files_with_spatial_calibration_attrs', 0)} HDF5 spatial-calibration attrs, {first_summary(calibration_claim_risk, 'calibration_evidence', {}).get('pptx_calibration_hits', 0)} PPTX hits",
    ]
    for row in top_items(first_summary(calibration_claim_risk, 'high_risk_claim_families', []), 8):
        report_lines.append(
            f"- {row.get('analysis')}: {row.get('claim_status')} - {row.get('recommended_wording')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(calibration_claim_risk, 'guardrail', 'Calibration claim risk register unavailable.')}")

    report_lines += [
        "",
        "## Apparent Diffusion Calibration Bounds",
        "",
        f"- Threshold rows/ROI/cycles: {first_summary(apparent_diffusion_calibration, 'n_threshold_rows', 0)} / {first_summary(apparent_diffusion_calibration, 'n_roi', 0)} / {first_summary(apparent_diffusion_calibration, 'n_cycles', 0)}",
        f"- ROI with HDF5 timing: {first_summary(apparent_diffusion_calibration, 'n_roi_with_h5_timing', 0)}",
        f"- Pixel-size assumptions: {first_summary(apparent_diffusion_calibration, 'pixel_size_um_assumptions', [])}; default {fmt(first_summary(apparent_diffusion_calibration, 'default_pixel_size_um'))} um/px",
        f"- Median ROI elapsed / HDF5 elapsed ratio: {fmt(first_summary(apparent_diffusion_calibration, 'median_roi_elapsed_to_h5_median_ratio'))}",
        f"- q70 median apparent D at 96 nm/px: {fmt(first_summary(apparent_diffusion_calibration, 'median_q70_apparent_D_h5median_um2_per_s'))} um2/s; median abs {fmt(first_summary(apparent_diffusion_calibration, 'median_q70_abs_apparent_D_h5median_um2_per_s'))} um2/s; positive fraction {fmt(first_summary(apparent_diffusion_calibration, 'q70_positive_D_fraction'))}",
    ]
    for row in apparent_diffusion_thresholds:
        report_lines.append(
            f"- Threshold {fmt(row.get('threshold_quantile'))}: median D {fmt(row.get('median_D_um2_per_s'))}, median abs D {fmt(row.get('median_abs_D_um2_per_s'))}, positive fraction {fmt(row.get('positive_D_fraction'))}"
        )
    for row in apparent_diffusion_q70_tests[:6]:
        report_lines.append(
            f"- q70 calibration future8 test {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in apparent_diffusion_sources[:8]:
        report_lines.append(
            f"- Source timing {row.get('source_stem')}: dt median {fmt(row.get('h5_dt_median_s'))}s, max/median {fmt(row.get('h5_dt_max_to_median_ratio'))}, ROI/H5 elapsed {fmt(row.get('roi_elapsed_to_h5_median_ratio'))}, median abs D {fmt(row.get('median_abs_D_um2_per_s'))}"
        )
    for row in apparent_diffusion_corr[:4]:
        report_lines.append(
            f"- Calibration-bound link {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(apparent_diffusion_calibration, 'guardrail', 'Apparent diffusion calibration-bounds audit unavailable.')}")

    report_lines += [
        "",
        "## HDF5 Timebase Provenance Audit",
        "",
        f"- Timebase status: {first_summary(hdf5_timebase, 'timebase_status', 'missing')}",
        f"- q70 ROI rows / sources: {first_summary(hdf5_timebase, 'n_q70_roi_rows', 0)} / {first_summary(hdf5_timebase, 'n_sources', 0)}",
        f"- Strict / pause-heavy sources: {first_summary(hdf5_timebase, 'n_strict_timebase_sources', 0)} / {first_summary(hdf5_timebase, 'n_pause_heavy_sources', 0)}",
        f"- ROI-HDF5 elapsed-aligned rows: {first_summary(hdf5_timebase, 'n_roi_elapsed_h5_aligned', 0)} / {first_summary(hdf5_timebase, 'n_q70_roi_rows', 0)}",
        f"- Median dt / ROI-HDF5 elapsed ratio / max source dt max-median ratio: {fmt(first_summary(hdf5_timebase, 'median_dt_median_s'))}s / {fmt(first_summary(hdf5_timebase, 'median_roi_elapsed_to_h5_ratio'))} / {fmt(first_summary(hdf5_timebase, 'max_source_dt_max_to_median_ratio'))}",
        f"- Pause-heavy sources: {', '.join(first_summary(hdf5_timebase, 'pause_heavy_sources', []) or []) or 'none'}",
    ]
    for row in hdf5_timebase_scenarios:
        report_lines.append(
            f"- Timebase scenario {row.get('scenario')}: n={fmt(row.get('n_roi'), 0)}, sources={fmt(row.get('n_sources'), 0)}, median D {fmt(row.get('median_D_um2_per_s'))}, future8 median positive-negative {fmt(row.get('future8_median_positive_minus_negative'))}, p={fmt(row.get('future8_mannwhitney_p'))}"
        )
    for row in hdf5_timebase_correlations[:4]:
        report_lines.append(
            f"- Timebase correlation {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(hdf5_timebase, 'guardrail', 'HDF5 timebase provenance audit unavailable.')}")

    report_lines += [
        "",
        "## Diffusion Physics Consistency Audit",
        "",
        f"- ROI/threshold rows/sources: {first_summary(diffusion_physics_consistency, 'n_roi', 0)} / {first_summary(diffusion_physics_consistency, 'n_threshold_rows', 0)} / {first_summary(diffusion_physics_consistency, 'n_sources', 0)}",
        f"- Automatic physics-consistent / publication-ready diffusion candidates: {first_summary(diffusion_physics_consistency, 'n_automatic_diffusion_physics_consistent', 0)} / {first_summary(diffusion_physics_consistency, 'n_publication_ready_diffusion_candidates', 0)}",
        f"- Median abs apparent D / positive-D fraction / radius2 fit R2 / threshold sensitivity: {fmt(first_summary(diffusion_physics_consistency, 'median_abs_apparent_D_um2_per_s'))} / {fmt(first_summary(diffusion_physics_consistency, 'median_positive_D_fraction'))} / {fmt(first_summary(diffusion_physics_consistency, 'median_radius2_fit_r2'))} / {fmt(first_summary(diffusion_physics_consistency, 'median_threshold_sensitivity_iqr_over_median_abs'))}",
    ]
    for row in diffusion_physics_gates:
        report_lines.append(
            f"- Diffusion physics gate {row.get('criterion')}: {fmt(row.get('n_pass'), 0)}/{fmt(row.get('n_total'), 0)} pass ({fmt(row.get('fraction_pass'))})"
        )
    for row in diffusion_physics_candidates[:4]:
        report_lines.append(
            f"- Physics-consistent candidate {row.get('roi_id')}: cycle {fmt(row.get('cycleNo'), 0)}, D {fmt(row.get('median_apparent_D_um2_per_s'))}, fit R2 {fmt(row.get('median_radius2_fit_r2'))}, future8={fmt(row.get('future_any_drop_within_8cycles'), 0)}, future16={fmt(row.get('future_any_drop_within_16cycles'), 0)}"
        )
    for row in diffusion_physics_tests[:6]:
        report_lines.append(
            f"- Diffusion consistency target {row.get('target')} {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, AUC {fmt(row.get('abs_oriented_auc'))}, p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in diffusion_physics_corr[:4]:
        report_lines.append(
            f"- Diffusion consistency correlation {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(diffusion_physics_consistency, 'guardrail', 'Diffusion physics consistency audit unavailable.')}")

    report_lines += [
        "",
        "## Diffusion Claim Readiness Audit",
        "",
        f"- Overall status: {first_summary(diffusion_claim_readiness, 'overall_status', 'unavailable')}",
        f"- Criteria / hard blockers / publication-ready candidates: {first_summary(diffusion_claim_readiness, 'n_criteria', 0)} / {first_summary(diffusion_claim_readiness, 'n_hard_blockers', 0)} / {first_summary(diffusion_claim_readiness, 'n_publication_ready_candidates', 0)}",
        f"- Status counts: {first_summary(diffusion_claim_readiness, 'status_counts', {})}",
    ]
    for blocker in diffusion_claim_hard_blockers[:8]:
        report_lines.append(f"- Hard blocker: {blocker}")
    for row in diffusion_claim_top_candidates[:4]:
        report_lines.append(
            f"- Readiness candidate {row.get('roi_id')}: source {row.get('candidate_source')}, cycle {fmt(row.get('cycleNo'), 0)}, publication_ready={row.get('publication_ready')}, blockers={row.get('blockers')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(diffusion_claim_readiness, 'guardrail', 'Diffusion claim readiness audit unavailable.')}")

    report_lines += [
        "",
        "## All-Cycle Dataset Coverage Atlas",
        "",
        f"- Status/cycles/sources/HDF5 rows: {first_summary(all_cycle_coverage, 'overall_status', 'unavailable')} / {first_summary(all_cycle_coverage, 'n_cycle_rows', 0)} / {first_summary(all_cycle_coverage, 'n_sources', 0)} / {first_summary(all_cycle_coverage, 'n_h5_inventory_rows', 0)}",
        f"- ROI/video cycle coverage: any tracked sequence {first_summary(all_cycle_coverage, 'n_cycles_with_any_roi_video_sequence', 0)} cycles ({fmt(first_summary(all_cycle_coverage, 'any_roi_cycle_coverage_fraction'))}); primary sequence {first_summary(all_cycle_coverage, 'n_cycles_with_primary_roi_sequence', 0)} cycles ({fmt(first_summary(all_cycle_coverage, 'primary_roi_cycle_coverage_fraction'))}).",
        f"- Future16-positive cycles without any tracked ROI sequence: {first_summary(all_cycle_coverage, 'n_future16_positive_cycles_without_any_roi_sequence', 0)}",
    ]
    for row in all_cycle_source_gaps[:6]:
        report_lines.append(
            f"- Source coverage {row.get('source_stem')}: cycles {fmt(row.get('n_cycle_rows'), 0)}, ROI-covered {fmt(row.get('n_cycles_with_any_roi'), 0)}, primary-covered {fmt(row.get('n_cycles_with_primary_roi'), 0)}, future16+ {fmt(row.get('n_future16_positive'), 0)}, primary fraction {fmt(row.get('primary_roi_cycle_coverage_fraction'))}"
        )
    for row in all_cycle_gap_cycles[:6]:
        report_lines.append(
            f"- Coverage-priority cycle {fmt(row.get('cycleNo'), 0)} / {row.get('source_stem')}: priority {fmt(row.get('coverage_gap_priority'))}, ROI rows {fmt(row.get('n_roi_total_across_tracked_cohorts'), 0)}, future8 {fmt(row.get('future_any_drop_within_8cycles'), 0)}, future16 {fmt(row.get('future_any_drop_within_16cycles'), 0)}, consensus {fmt(row.get('cross_modal_consensus_score'))}"
        )
    for row in all_cycle_roi_cohorts[:8]:
        report_lines.append(
            f"- ROI cohort coverage {row.get('cohort')}: rows {fmt(row.get('n_rows'), 0)}, cycles {fmt(row.get('n_cycles'), 0)}, sources {fmt(row.get('n_sources'), 0)}, status {row.get('status', 'NA')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(all_cycle_coverage, 'guardrail', 'All-cycle coverage atlas unavailable.')}")

    report_lines += [
        "",
        "## Targeted Densification and Manual-QC Plan",
        "",
        f"- Status/cycles/sources/ROI action rows: {first_summary(targeted_densification_qc, 'overall_status', 'unavailable')} / {first_summary(targeted_densification_qc, 'n_cycle_rows', 0)} / {first_summary(targeted_densification_qc, 'n_sources', 0)} / {first_summary(targeted_densification_qc, 'n_roi_queue_rows', 0)}",
        f"- Uncovered cycles / low-ROI cycles (<10 rows): {first_summary(targeted_densification_qc, 'n_uncovered_cycles', 0)} / {first_summary(targeted_densification_qc, 'n_low_roi_cycles_lt10', 0)}",
    ]
    for row in targeted_densification_action_counts:
        report_lines.append(
            f"- Recommended cycle action {row.get('recommended_cycle_action')}: n={fmt(row.get('n_cycles'), 0)}"
        )
    for row in targeted_densification_source_plan[:6]:
        report_lines.append(
            f"- Densification source priority {row.get('source_stem')}: max priority {fmt(row.get('max_cycle_action_priority'))}, future16+ {fmt(row.get('n_future16_positive'), 0)}, low-ROI cycles {fmt(row.get('n_low_roi_cycles'), 0)}, uncovered {fmt(row.get('n_uncovered_cycles'), 0)}"
        )
    for row in targeted_densification_cycle_actions[:6]:
        report_lines.append(
            f"- Densification cycle priority {fmt(row.get('cycleNo'), 0)} / {row.get('source_stem')}: action {row.get('recommended_cycle_action')}, priority {fmt(row.get('cycle_action_priority'))}, ROI rows {fmt(row.get('n_roi_total_across_tracked_cohorts'), 0)}"
        )
    for row in targeted_densification_roi_actions[:6]:
        report_lines.append(
            f"- Manual-QC ROI priority {row.get('roi_id')}: origin {row.get('candidate_origin')}, action {row.get('candidate_action')}, priority {fmt(row.get('roi_action_priority'))}, cycle {fmt(row.get('cycleNo'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(targeted_densification_qc, 'guardrail', 'Targeted densification/QC plan unavailable.')}")

    report_lines += [
        "",
        "## Current Claim Readiness Matrix",
        "",
        f"- Claims audited: {fmt(first_summary(current_claim_readiness, 'n_claims'), 0)}",
        f"- Overall position: {first_summary(current_claim_readiness, 'overall_position', 'NA')}",
        f"- Supported/operational claim IDs: {first_summary(current_claim_readiness, 'supported_or_operational_claim_ids', [])}",
        f"- Blocked/not-supported claim IDs: {first_summary(current_claim_readiness, 'blocked_or_not_supported_claim_ids', [])}",
        f"- Positive evidence: fusion near-any AUC {fmt(current_claim_positive.get('fusion_near_any_auc'))}, fusion source-stratified p {fmt(current_claim_positive.get('fusion_near_any_source_stratified_p'))}, falsification AUC {fmt(current_claim_positive.get('falsification_near_any_auc'))}, same-source median delta {fmt(current_claim_positive.get('same_source_pair_median_delta'))}.",
        f"- Negative evidence: diffusion status {current_claim_negative.get('diffusion_status', 'NA')} with {fmt(current_claim_negative.get('diffusion_publication_ready_candidates'), 0)} publication-ready candidates; future8 video status {current_claim_negative.get('future8_video_physics_status', 'NA')}; expansion future8 source-stratified p {fmt(current_claim_negative.get('expansion_future8_source_stratified_p'))}.",
    ]
    for row in current_claim_status_counts:
        report_lines.append(f"- Claim readiness {row.get('readiness', 'NA')}: n={fmt(row.get('n'), 0)}")
    report_lines.append(f"- Guardrail: {first_summary(current_claim_readiness, 'guardrail', 'Current claim readiness matrix unavailable.')}")

    report_lines += [
        "",
        "## Diffusion Unblock Sensitivity Audit",
        "",
        f"- Status/candidates/global hard blockers applied: {first_summary(diffusion_unblock_sensitivity, 'overall_status', 'unavailable')} / {first_summary(diffusion_unblock_sensitivity, 'n_candidate_rows', 0)} / {first_summary(diffusion_unblock_sensitivity, 'n_global_hard_blockers_applied', 0)}",
        f"- Global hard blockers applied to every candidate: {first_summary(diffusion_unblock_sensitivity, 'global_hard_blockers_applied', [])}",
    ]
    for row in diffusion_unblock_scenarios[:6]:
        report_lines.append(
            f"- Diffusion unblock scenario {row.get('scenario')}: eligible {fmt(row.get('n_eligible'), 0)}, one-blocker remaining {fmt(row.get('n_one_blocker_remaining'), 0)}"
        )
    for row in diffusion_unblock_blockers[:8]:
        report_lines.append(
            f"- Diffusion blocker sensitivity {row.get('blocker')}: {fmt(row.get('n_candidate_rows'), 0)} candidate rows, criterion status {row.get('criterion_status', 'NA')}"
        )
    for row in diffusion_unblock_candidates[:4]:
        report_lines.append(
            f"- Nearest diffusion-unblock candidate {row.get('roi_id')}: blockers {fmt(row.get('n_all_blockers'), 0)}, priority {fmt(row.get('review_priority'))}, blocker summary {row.get('blocker_summary', 'NA')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(diffusion_unblock_sensitivity, 'guardrail', 'Diffusion unblock sensitivity audit unavailable.')}")

    report_lines += [
        "",
        "## Targeted Diffusion Blocker Diagnostic",
        "",
        f"- Status/target candidates/threshold rows: {first_summary(targeted_diffusion_blocker, 'overall_status', 'unavailable')} / {first_summary(targeted_diffusion_blocker, 'n_target_candidates_with_thresholds', 0)} / {first_summary(targeted_diffusion_blocker, 'n_threshold_variant_rows', 0)}",
        f"- Nearest candidate: {targeted_diffusion_nearest.get('roi_id', 'NA')} from {targeted_diffusion_nearest.get('source_stem', 'NA')} cycle {fmt(targeted_diffusion_nearest.get('cycleNo'), 0)}; action {targeted_diffusion_nearest.get('diagnostic_action', 'NA')}; blockers {targeted_diffusion_nearest.get('blockers', 'NA')}",
        f"- Nearest candidate threshold evidence: positive-D fraction {fmt(targeted_diffusion_nearest.get('positive_D_fraction'))}, q70 D {fmt(targeted_diffusion_nearest.get('q70_D_um2_per_s'))} um2/s, q70 fit R2 {fmt(targeted_diffusion_nearest.get('q70_radius2_fit_r2'))}, same-source D percentile {fmt(targeted_diffusion_nearest.get('median_D_um2_per_s_same_source_percentile'))}",
    ]
    for row in targeted_diffusion_actions:
        report_lines.append(
            f"- Diffusion diagnostic action {row.get('diagnostic_action')}: n={fmt(row.get('n'), 0)}"
        )
    for row in targeted_diffusion_candidates[:5]:
        report_lines.append(
            f"- Targeted diffusion candidate {row.get('roi_id')}: action {row.get('diagnostic_action')}, score {fmt(row.get('targeted_remeasurement_score'))}, max positive-fit R2 {fmt(row.get('max_positive_fit_r2'))}, source percentile {fmt(row.get('median_D_um2_per_s_same_source_percentile'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(targeted_diffusion_blocker, 'guardrail', 'Targeted diffusion blocker diagnostic unavailable.')}")

    report_lines += [
        "",
        "## Cycle 78 Diffusion Remeasurement Audit",
        "",
        f"- Status/target/context ROIs/variant rows/bootstrap: {first_summary(cycle78_diffusion_remeasurement, 'overall_status', 'unavailable')} / {first_summary(cycle78_diffusion_remeasurement, 'target_status', 'NA')} / {first_summary(cycle78_diffusion_remeasurement, 'n_context_rois', 0)} / {first_summary(cycle78_diffusion_remeasurement, 'n_variant_rows', 0)} / {first_summary(cycle78_diffusion_remeasurement, 'n_bootstrap', 0)}",
        f"- Target {first_summary(cycle78_diffusion_remeasurement, 'target_roi_id', 'NA')}: default q70 D {fmt(cycle78_diffusion_target.get('default_q70_D_um2_per_s'))} um2/s, bootstrap p05/p95 {fmt(cycle78_diffusion_target.get('default_q70_D_p05_um2_per_s'))}/{fmt(cycle78_diffusion_target.get('default_q70_D_p95_um2_per_s'))}, positive CI {cycle78_diffusion_target.get('default_q70_positive_ci', 'NA')}",
        f"- Target robustness: median D {fmt(cycle78_diffusion_target.get('median_D_um2_per_s'))} um2/s, positive-D fraction {fmt(cycle78_diffusion_target.get('positive_D_fraction'))}, positive-CI fraction {fmt(cycle78_diffusion_target.get('positive_ci_fraction'))}, max radius2 R2 {fmt(cycle78_diffusion_target.get('max_radius2_r2'))}",
        f"- Same-source context percentiles: default q70 D {fmt(cycle78_diffusion_target.get('default_q70_D_um2_per_s_context_percentile'))}, q70 R2 {fmt(cycle78_diffusion_target.get('default_q70_radius2_r2_context_percentile'))}, positive-CI fraction {fmt(cycle78_diffusion_target.get('positive_ci_fraction_context_percentile'))}",
    ]
    for row in cycle78_diffusion_context:
        report_lines.append(
            f"- Cycle78 context {row.get('roi_id', 'NA')} ({row.get('context_role', 'NA')}): q70 D {fmt(row.get('default_q70_D_um2_per_s'))}, q70 positive CI {row.get('default_q70_positive_ci', 'NA')}, pos-CI fraction {fmt(row.get('positive_ci_fraction'))}, max R2 {fmt(row.get('max_radius2_r2'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(cycle78_diffusion_remeasurement, 'guardrail', 'Cycle 78 diffusion remeasurement audit unavailable.')}")

    report_lines += [
        "",
        "## Post-Remeasurement Diffusion Gate Audit",
        "",
        f"- Status/target/q70/publication-ready: {first_summary(post_remeasurement_diffusion_gate, 'overall_status', 'unavailable')} / {first_summary(post_remeasurement_diffusion_gate, 'target_roi_id', 'NA')} / {first_summary(post_remeasurement_diffusion_gate, 'target_q70_status', 'NA')} / {first_summary(post_remeasurement_diffusion_gate, 'publication_ready_after_remeasurement', 'NA')}",
        f"- Candidate q70 blocker removed: {first_summary(post_remeasurement_diffusion_gate, 'candidate_q70_blocker_removed', 'NA')}; claim readiness status remains {first_summary(post_remeasurement_diffusion_gate, 'claim_readiness_overall_status', 'NA')}; publication-ready candidates {fmt(first_summary(post_remeasurement_diffusion_gate, 'claim_readiness_publication_ready_candidates'), 0)}",
        f"- Remeasured target evidence: q70 D {fmt(post_remeasurement_target.get('default_q70_D_um2_per_s'))} um2/s, p05/p95 {fmt(post_remeasurement_target.get('default_q70_D_p05_um2_per_s'))}/{fmt(post_remeasurement_target.get('default_q70_D_p95_um2_per_s'))}, positive-CI fraction {fmt(post_remeasurement_target.get('positive_ci_fraction'))}",
    ]
    for row in post_remeasurement_scenarios:
        report_lines.append(
            f"- Diffusion post-remeasurement scenario {row.get('scenario', 'NA')}: q70 {row.get('candidate_q70_status', 'NA')}, publication_ready={row.get('publication_ready', 'NA')}, candidate blockers {row.get('remaining_candidate_blockers', 'NA')}"
        )
    for blocker in post_remeasurement_publication_blockers[:8]:
        report_lines.append(f"- Remaining publication blocker: {blocker}")
    report_lines.append(f"- Guardrail: {first_summary(post_remeasurement_diffusion_gate, 'guardrail', 'Post-remeasurement diffusion gate audit unavailable.')}")

    report_lines += [
        "",
        "## Cycle 78 Front Identity Review Packet",
        "",
        f"- Status/target/review rows/default-q70-CI rows/no-flag rows: {first_summary(cycle78_front_identity_review, 'overall_status', 'unavailable')} / {first_summary(cycle78_front_identity_review, 'target_roi_id', 'NA')} / {first_summary(cycle78_front_identity_review, 'n_review_rows', 0)} / {first_summary(cycle78_front_identity_review, 'n_rows_default_q70_positive_ci', 0)} / {first_summary(cycle78_front_identity_review, 'n_rows_no_automatic_flags', 0)}",
        f"- Target front-identity flags: {cycle78_front_identity_target.get('automatic_front_identity_flags', 'NA')}; score {fmt(cycle78_front_identity_target.get('automatic_front_identity_score'))}; q70 components median {fmt(cycle78_front_identity_target.get('q70_components_median'))}; largest-component fraction {fmt(cycle78_front_identity_target.get('q70_largest_component_fraction_median'))}",
        f"- Target mask stability: q70 area median/CV {fmt(cycle78_front_identity_target.get('q70_area_fraction_median'))}/{fmt(cycle78_front_identity_target.get('q70_area_fraction_cv'))}, centroid path/net {fmt(cycle78_front_identity_target.get('q70_centroid_path_px'))}/{fmt(cycle78_front_identity_target.get('q70_centroid_net_px'))} px, edge contact {fmt(cycle78_front_identity_target.get('q70_edge_contact_fraction_median'))}",
        f"- Target q70 diffusion context remains positive but manual fields are pending: D {fmt(cycle78_front_identity_target.get('default_q70_D_um2_per_s'))} um2/s, positive CI {cycle78_front_identity_target.get('default_q70_positive_ci', 'NA')}, manual particle/front/diffusion decisions are blank.",
        f"- Outputs: manifest {first_summary(cycle78_front_identity_review, 'outputs', {}).get('manifest', 'NA')}; HTML {first_summary(cycle78_front_identity_review, 'outputs', {}).get('html', 'NA')}",
    ]
    report_lines.append(f"- Guardrail: {first_summary(cycle78_front_identity_review, 'guardrail', 'Cycle 78 front identity review packet unavailable.')}")

    report_lines += [
        "",
        "## Cycle 78 Component Front Retracking Audit",
        "",
        f"- Status/target/context ROIs/strategy rows: {first_summary(cycle78_component_retracking, 'overall_status', 'unavailable')} / {first_summary(cycle78_component_retracking, 'target_roi_id', 'NA')} / {first_summary(cycle78_component_retracking, 'n_context_rois', 0)} / {first_summary(cycle78_component_retracking, 'n_strategy_rows', 0)}",
        f"- Largest component preserves positive slope/improves R2: {first_summary(cycle78_component_retracking, 'largest_component_preserves_positive_slope', 'NA')} / {first_summary(cycle78_component_retracking, 'largest_component_improves_r2', 'NA')}",
        f"- Target raw q70: slope {fmt(cycle78_component_raw.get('radius2_slope_px2_per_sample'))} px2/sample, R2 {fmt(cycle78_component_raw.get('radius2_slope_r2'))}, centroid path {fmt(cycle78_component_raw.get('centroid_path_px'))} px, area median {fmt(cycle78_component_raw.get('area_px_median'))}",
        f"- Target largest-component q70: slope {fmt(cycle78_component_largest.get('radius2_slope_px2_per_sample'))} px2/sample, R2 {fmt(cycle78_component_largest.get('radius2_slope_r2'))}, centroid path {fmt(cycle78_component_largest.get('centroid_path_px'))} px, area median {fmt(cycle78_component_largest.get('area_px_median'))}",
        f"- Target central/top3 q70 R2: {fmt(cycle78_component_central.get('radius2_slope_r2'))} / {fmt(cycle78_component_top3.get('radius2_slope_r2'))}; front flags remain {cycle78_component_raw.get('front_identity_flags', 'NA')}",
    ]
    report_lines.append(f"- Guardrail: {first_summary(cycle78_component_retracking, 'guardrail', 'Cycle 78 component retracking audit unavailable.')}")

    report_lines += [
        "",
        "## Cross-Modal Degradation Consensus",
        "",
        f"- Cycles scored/with votes: {first_summary(cross_modal_consensus, 'n_cycles', 0)} / {first_summary(cross_modal_consensus, 'n_cycles_with_any_modal_vote', 0)}",
        f"- Median consensus score: {fmt(first_summary(cross_modal_consensus, 'median_consensus_score'))}; modal vote threshold {fmt(first_summary(cross_modal_consensus, 'modal_vote_threshold'))}",
    ]
    for row in cross_modal_classes:
        report_lines.append(
            f"- Consensus class {row.get('consensus_class')}: n={fmt(row.get('n_cycles'), 0)}, median score {fmt(row.get('median_consensus_score'))}, modal votes {fmt(row.get('mean_modal_votes'))}, event rate {fmt(row.get('event_rate'))}, future8 rate {fmt(row.get('future8_rate'))}, median frame percentile {fmt(row.get('median_frames_percentile'))}"
        )
    for row in cross_modal_top[:8]:
        report_lines.append(
            f"- Consensus cycle {fmt(row.get('cycleNo'), 0)}: {row.get('consensus_class')}, score {fmt(row.get('cross_modal_consensus_score'))}, votes {fmt(row.get('n_modal_votes'), 0)}, event={fmt(row.get('any_abrupt_drop'), 0)}, future8={fmt(row.get('future_any_drop_within_8cycles'), 0)}, frame percentile {fmt(row.get('frames_percentile'))}"
        )
    for row in cross_modal_contrasts[:6]:
        report_lines.append(
            f"- Consensus contrast {row.get('feature')} vs {row.get('target')}: median positive-negative {fmt(row.get('median_difference'))}, rho={fmt(row.get('spearman_rho'))}, n={fmt(row.get('n'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(cross_modal_consensus, 'score_guardrail', 'Consensus audit unavailable.')}")

    report_lines += [
        "",
        "## Echem Optical Breakpoint Audit",
        "",
        f"- Cycles/features/permutations: {first_summary(echem_optical_breakpoint, 'n_cycles', 0)} / {first_summary(echem_optical_breakpoint, 'n_features_tested', 0)} / {first_summary(echem_optical_breakpoint, 'n_permutation', 0)}",
        f"- Event cycles tested: {first_summary(echem_optical_breakpoint, 'event_cycles', [])}",
    ]
    for row in echem_breakpoint_event[:8]:
        report_lines.append(
            f"- Event-centered breakpoint {row.get('feature')} +/-{fmt(row.get('window_cycles'), 0)} cycles: scaled shift {fmt(row.get('event_median_scaled_shift'))}, control p95 abs {fmt(row.get('control_p95_abs_scaled_shift'))}, empirical p={fmt(row.get('empirical_p_abs_vs_control_centers'))}, bootstrap p={fmt(row.get('bootstrap_p_abs_vs_control_centers'))}"
        )
    for row in echem_breakpoint_future[:8]:
        report_lines.append(
            f"- Echem/trace label link {row.get('feature')} vs {row.get('target')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, MW p={fmt(row.get('mannwhitney_p'))}, rho={fmt(row.get('spearman_rho'))}"
        )
    for row in echem_breakpoint_event_ranks[:6]:
        report_lines.append(
            f"- Event-cycle breakpoint rank cycle {fmt(row.get('center_cycle'), 0)} {row.get('feature')} +/-{fmt(row.get('window_cycles'), 0)}: rank {fmt(row.get('rank_abs_shift_desc'), 0)}, percentile {fmt(row.get('percentile_abs_shift'))}, scaled shift {fmt(row.get('post_minus_pre_iqr_scaled'))}"
        )
    for row in echem_breakpoint_global[:5]:
        report_lines.append(
            f"- Global echem breakpoint candidate cycle {fmt(row.get('center_cycle'), 0)} {row.get('feature')}: scaled shift {fmt(row.get('post_minus_pre_iqr_scaled'))}, rank {fmt(row.get('rank_abs_shift_desc'), 0)}, pre/post n={fmt(row.get('n_pre'), 0)}/{fmt(row.get('n_post'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(echem_optical_breakpoint, 'guardrail', 'Echem optical breakpoint audit unavailable.')}")

    report_lines += [
        "",
        "## Echem Optical Regime Atlas",
        "",
        f"- Cycles/features: {first_summary(echem_optical_regime, 'n_cycles', 0)} / {first_summary(echem_optical_regime, 'n_echem_features', 0)}",
        f"- Missing echem-shape cycles: {first_summary(echem_optical_regime, 'n_cycles_missing_echem_shape', 0)}; extreme-or-missing CE cycles: {first_summary(echem_optical_regime, 'n_cycles_extreme_or_missing_ce', 0)}",
    ]
    for row in echem_regime_summary:
        report_lines.append(
            f"- Echem PC1 regime {row.get('echem_pc1_tertile')}: n={fmt(row.get('n_cycles'), 0)}, median cycle {fmt(row.get('median_cycle'))}, median consensus {fmt(row.get('median_cross_modal_score'))}, future8 rate {fmt(row.get('future8_rate'))}, extreme/missing CE rate {fmt(row.get('echem_ce_extreme_or_missing_rate'))}"
        )
    for row in echem_regime_binary[:8]:
        report_lines.append(
            f"- Echem binary contrast {row.get('feature')} vs {row.get('target')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, p={fmt(row.get('mannwhitney_p'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in echem_regime_corr[:8]:
        report_lines.append(
            f"- Echem optical link {row.get('feature')} vs {row.get('target')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in echem_regime_top_cycles[:6]:
        report_lines.append(
            f"- Echem-optical priority cycle {fmt(row.get('cycleNo'), 0)}: score {fmt(row.get('echem_optical_priority_score'))}, regime {row.get('echem_pc1_tertile')}, consensus {fmt(row.get('cross_modal_consensus_score'))}, class {row.get('consensus_class')}, CE-flag={fmt(row.get('echem_ce_extreme_or_missing'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(echem_optical_regime, 'guardrail', 'Echem optical regime atlas unavailable.')}")

    report_lines += [
        "",
        "## Echem-Conditioned Optical Predictor",
        "",
        f"- Cycles/targets: {first_summary(echem_conditioned_predictor, 'n_cycles', 0)} / {len(first_summary(echem_conditioned_predictor, 'targets', []))}",
        f"- Feature set sizes: {first_summary(echem_conditioned_predictor, 'feature_set_sizes', {})}",
    ]
    for row in echem_predictor_deltas[:10]:
        report_lines.append(
            f"- Echem feature-set delta {row.get('split')} {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, base {fmt(row.get('base_roc_auc'))}, comparison {fmt(row.get('comparison_roc_auc'))}"
        )
    for row in echem_predictor_metrics[:10]:
        report_lines.append(
            f"- Echem-conditioned metric {row.get('split')} {row.get('target')} {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, n={fmt(row.get('n_eval'), 0)}, positives={fmt(row.get('n_positive'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(echem_conditioned_predictor, 'guardrail', 'Echem-conditioned optical predictor unavailable.')}")

    report_lines += [
        "",
        "## Echem-Conditioned ROI Rollout/Front Audit",
        "",
        f"- ROI rows/cycles/targets: {first_summary(echem_roi_rollout_front, 'n_roi_rows', 0)} / {first_summary(echem_roi_rollout_front, 'n_cycles', 0)} / {len(first_summary(echem_roi_rollout_front, 'targets', []))}",
        f"- Feature set sizes: {first_summary(echem_roi_rollout_front, 'feature_set_sizes', {})}",
    ]
    for row in echem_roi_deltas[:10]:
        report_lines.append(
            f"- ROI echem feature-set delta {row.get('target')} {row.get('comparison')}: delta Spearman {fmt(row.get('delta_spearman_rho'))}, delta R2 {fmt(row.get('delta_r2'))}, base rho {fmt(row.get('base_spearman_rho'))}, comparison rho {fmt(row.get('comparison_spearman_rho'))}"
        )
    for row in echem_roi_metrics[:10]:
        report_lines.append(
            f"- ROI leave-cycle metric {row.get('target')} {row.get('feature_set')}: rho {fmt(row.get('spearman_rho'))}, R2 {fmt(row.get('r2'))}, MAE {fmt(row.get('mae'))}, n={fmt(row.get('n_eval'), 0)}, cycles={fmt(row.get('n_cycles'), 0)}"
        )
    for row in echem_roi_residual[:8]:
        report_lines.append(
            f"- ROI acquisition-residual echem link {row.get('echem_feature')} vs residual {row.get('target')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(echem_roi_rollout_front, 'guardrail', 'Echem-conditioned ROI rollout/front audit unavailable.')}")

    report_lines += [
        "",
        "## Echem Video Embedding Fusion Audit",
        "",
        f"- Embedding rows/cycles: {first_summary(echem_video_fusion, 'n_embedding_rows', 0)} / {first_summary(echem_video_fusion, 'n_cycles', 0)}",
        f"- Cohort counts: {first_summary(echem_video_fusion, 'embedding_cohort_counts', {})}",
        f"- Feature set sizes: {first_summary(echem_video_fusion, 'feature_set_sizes', {})}",
    ]
    for row in echem_video_deltas[:10]:
        report_lines.append(
            f"- Echem-video fusion delta {row.get('task')} {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta R2 {fmt(row.get('delta_r2'))}, delta rho {fmt(row.get('delta_spearman_rho'))}"
        )
    for row in echem_video_class[:8]:
        report_lines.append(
            f"- Echem-video classification {row.get('target')} {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, rho {fmt(row.get('spearman_rho'))}, null p={fmt(row.get('empirical_p_ge_observed'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    for row in echem_video_reg[:8]:
        report_lines.append(
            f"- Echem-video regression {row.get('target')} {row.get('feature_set')}: rho {fmt(row.get('spearman_rho'))}, R2 {fmt(row.get('r2'))}, MAE {fmt(row.get('mae'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(echem_video_fusion, 'guardrail', 'Echem video embedding fusion audit unavailable.')}")

    report_lines += [
        "",
        "## Diffusion Proxy Sanity Audit",
        "",
        f"- Selected high-resolution front ROIs: {first_summary(diffusion_sanity, 'n_selected_front_rois', 0)}",
        f"- Automatic positive diffusion-proxy candidates: {first_summary(diffusion_sanity, 'n_automatic_positive_diffusion_proxy_candidates', 0)}",
        f"- Publication diffusion candidates after manual-QC gate: {first_summary(diffusion_sanity, 'n_publication_diffusion_candidates', 0)}",
        f"- Median selected/threshold apparent D: {fmt(first_summary(diffusion_sanity, 'median_selected_diffusion_um2_per_s'))} / {fmt(first_summary(diffusion_sanity, 'median_threshold_diffusion_um2_per_s'))} um2/s",
        f"- Estimator consensus counts: {first_summary(diffusion_sanity, 'estimator_consensus_counts', {})}",
    ]
    for row in diffusion_sanity_gate:
        report_lines.append(
            f"- Gate {row.get('criterion')}: {fmt(row.get('n_pass'), 0)}/{fmt(row.get('n_total'), 0)} pass ({fmt(row.get('fraction_pass'))})"
        )
    for row in diffusion_sanity_candidates[:6]:
        report_lines.append(
            f"- Candidate check {row.get('selected_roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): selected D {fmt(row.get('selected_diffusion_um2_per_s'))}, threshold D {fmt(row.get('diffusion_proxy_median_um2_per_s'))}, selected R2 {fmt(row.get('selected_r2'))}, consensus {row.get('estimator_consensus_sign')}, manual {row.get('manual_qc_status')}"
        )
    for row in diffusion_sanity_corr[:4]:
        report_lines.append(
            f"- Diffusion-sanity link {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(diffusion_sanity, 'guardrail', 'Diffusion proxy sanity audit unavailable.')}")

    report_lines += [
        "",
        "## Control-Balanced Front Tracking And Diffusion Sanity",
        "",
        f"- High-resolution tracked ROIs: {first_summary(control_balanced_front_tracking, 'n_tracked_rois', 0)}",
        f"- Cohort counts: {first_summary(control_balanced_diffusion_sanity, 'selected_front_cohort_counts', {})}",
        f"- Automatic/publication diffusion candidates: {first_summary(control_balanced_diffusion_sanity, 'n_automatic_positive_diffusion_proxy_candidates', 0)} / {first_summary(control_balanced_diffusion_sanity, 'n_publication_diffusion_candidates', 0)}",
        f"- Median selected/threshold apparent D: {fmt(first_summary(control_balanced_diffusion_sanity, 'median_selected_diffusion_um2_per_s'))} / {fmt(first_summary(control_balanced_diffusion_sanity, 'median_threshold_diffusion_um2_per_s'))} um2/s",
        f"- Estimator consensus counts: {first_summary(control_balanced_diffusion_sanity, 'estimator_consensus_counts', {})}",
    ]
    for row in control_balanced_diffusion_gate:
        report_lines.append(
            f"- Balanced gate {row.get('criterion')}: {fmt(row.get('n_pass'), 0)}/{fmt(row.get('n_total'), 0)} pass ({fmt(row.get('fraction_pass'))})"
        )
    for row in control_balanced_diffusion_tests[:5]:
        report_lines.append(
            f"- Balanced event/control {row.get('feature')}: median diff {fmt(row.get('event_control_median_diff'))}, p={fmt(row.get('mannwhitney_p'))}, n={fmt(row.get('n_event'), 0)}/{fmt(row.get('n_control'), 0)}"
        )
    for row in control_balanced_diffusion_candidates[:5]:
        report_lines.append(
            f"- Balanced candidate check {row.get('selected_roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): selected D {fmt(row.get('selected_diffusion_um2_per_s'))}, selected R2 {fmt(row.get('selected_r2'))}, consensus {row.get('estimator_consensus_sign')}, manual {row.get('manual_qc_status')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(control_balanced_diffusion_sanity, 'guardrail', 'Control-balanced diffusion sanity audit unavailable.')}")

    report_lines += [
        "",
        "## Particle Trace Physics Audit",
        "",
        f"- Cycle rows/range: {first_summary(particle_trace, 'n_cycle_rows', 0)} rows, cycles {fmt(first_summary(particle_trace, 'cycle_min'))}-{fmt(first_summary(particle_trace, 'cycle_max'))}",
        f"- Drop cycles: any={first_summary(particle_trace, 'n_any_drop_cycles', 0)}, synchronized 2+={first_summary(particle_trace, 'n_sync2_drop_cycles', 0)}, synchronized 3+={first_summary(particle_trace, 'n_sync3_drop_cycles', 0)}",
        f"- Trace-state clustering: k={first_summary(particle_trace, 'chosen_trace_state_k', 'NA')}, silhouette={fmt(first_summary(particle_trace, 'trace_state_best_silhouette'))}",
    ]
    for row in particle_trace_classifiers:
        report_lines.append(
            f"- Future-drop classifier {row.get('target')}: folds={fmt(row.get('n_folds'), 0)}, AUC {fmt(row.get('mean_roc_auc'))}, balanced accuracy {fmt(row.get('mean_balanced_accuracy'))}"
        )
    for row in particle_trace_nulls:
        report_lines.append(
            f"- Future-drop null {row.get('target')}: observed AUC {fmt(row.get('observed_mean_auc'))}, null p95 {fmt(row.get('null_p95_auc'))}, empirical p={fmt(row.get('empirical_p_ge_observed'))}"
        )
    for row in particle_trace_event_tests[:6]:
        report_lines.append(
            f"- Event feature {row.get('target')} {row.get('feature')}: median pos-neg {fmt(row.get('median_pos_minus_neg'))}, p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in particle_trace_echem_corr[:6]:
        report_lines.append(
            f"- Trace/echem correlation {row.get('left_feature')} vs {row.get('right_feature')}: rho {fmt(row.get('rho'))}, p={fmt(row.get('p_value'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(particle_trace, 'guardrail', 'Cycle-level trace audit only.')}")

    report_lines += [
        "",
        "## Particle Event Precursor Atlas",
        "",
        f"- Anchors: {first_summary(particle_precursor, 'n_event_anchors', 0)} event anchors, {first_summary(particle_precursor, 'n_candidate_control_anchors', 0)} candidate controls, {first_summary(particle_precursor, 'n_matched_control_anchors', 0)} matched controls",
    ]
    for row in particle_precursor_tests[:10]:
        report_lines.append(
            f"- Precursor {row.get('window')} {row.get('feature')} {row.get('statistic')}: event-control {fmt(row.get('event_minus_control_median'))}, p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in particle_precursor_all_tests[:5]:
        report_lines.append(
            f"- All-window {row.get('window')} {row.get('feature')} {row.get('statistic')}: event-control {fmt(row.get('event_minus_control_median'))}, p={fmt(row.get('mannwhitney_p'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(particle_precursor, 'guardrail', 'Event-aligned trace precursor atlas only.')}")

    report_lines += [
        "",
        "## ROI Trace Fusion Audit",
        "",
        f"- Fusion rows/predictors: {first_summary(roi_trace_fusion, 'n_roi_rows', 0)} ROI rows, {first_summary(roi_trace_fusion, 'n_predictors', 0)} lagged trace/context predictors",
    ]
    for row in roi_trace_focus[:8]:
        report_lines.append(
            f"- Focused residual association {row.get('predictor')} vs {row.get('target')}: rho={fmt(row.get('rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in roi_trace_mode_tests[:5]:
        report_lines.append(
            f"- Event-enriched mode precursor test {row.get('predictor')}: median diff {fmt(row.get('positive_minus_negative_median'))}, p={fmt(row.get('mannwhitney_p'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(roi_trace_fusion, 'guardrail', 'Trace-fusion associations are cycle-level linkage evidence only.')}")

    report_lines += [
        "",
        "## ROI Trace Fusion Cycle Null",
        "",
        f"- Cycle-collapsed points: {first_summary(roi_trace_cycle_null, 'n_cycle_points', 0)} from {first_summary(roi_trace_cycle_null, 'n_roi_rows', 0)} ROI rows",
        f"- Event-reference cycles: {first_summary(roi_trace_cycle_null, 'n_event_reference_cycles', 0)}",
        f"- Predictors/permutations: {first_summary(roi_trace_cycle_null, 'n_predictors_tested', 0)} / {first_summary(roi_trace_cycle_null, 'n_permutation', 0)}",
    ]
    for row in roi_trace_cycle_tests[:6]:
        report_lines.append(
            f"- Cycle-collapsed {row.get('predictor')} vs {row.get('target')}: rho {fmt(row.get('rho'))}, empirical p={fmt(row.get('empirical_p_abs_ge_observed'))}, n={fmt(row.get('n_cycle_points'), 0)}"
        )
    for row in roi_trace_centered_tests[:4]:
        report_lines.append(
            f"- Reference-centered {row.get('predictor')} vs {row.get('target')}: rho {fmt(row.get('rho'))}, empirical p={fmt(row.get('empirical_p_abs_ge_observed'))}, n={fmt(row.get('n_cycle_points'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(roi_trace_cycle_null, 'guardrail', 'Cycle-collapsed null audit only.')}")

    report_lines += [
        "",
        "## Precursor-Informed ROI Review",
        "",
        f"- Review candidates: {first_summary(precursor_review, 'n_review_candidates', 0)}",
        f"- Event/control candidates: {first_summary(precursor_review, 'n_event_candidates', 0)} / {first_summary(precursor_review, 'n_control_candidates', 0)}",
        f"- Review tiers: {first_summary(precursor_review, 'precursor_review_tier_counts', {})}",
    ]
    for row in precursor_review_top[:8]:
        report_lines.append(
            f"- {row.get('roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): score {fmt(row.get('precursor_informed_review_score'))}, tier {row.get('precursor_review_tier')}, reason {row.get('precursor_review_reason')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(precursor_review, 'guardrail', 'Review-prioritization manifest only.')}")

    report_lines += [
        "",
        "## Precursor Visual Review Bundle",
        "",
        f"- Ranked candidates included: {first_summary(precursor_visual_bundle, 'n_ranked_candidates', 0)}",
        f"- Candidates with visual assets: {first_summary(precursor_visual_bundle, 'n_candidates_with_visual_asset', 0)}",
        f"- Contact sheet: {first_summary(precursor_visual_bundle, 'contact_sheet', 'NA')}",
    ]
    for row in precursor_visual_top[:8]:
        report_lines.append(
            f"- Visual rank {fmt(row.get('visual_rank'), 0)} {row.get('roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): score {fmt(row.get('precursor_informed_review_score'))}, tier {row.get('precursor_review_tier')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(precursor_visual_bundle, 'guardrail', 'Visual bundle only; no manual labels assigned.')}")

    report_lines += [
        "",
        "## Within-Cycle Echem Shape Audit",
        "",
        f"- Echem shape cycles/features: {first_summary(within_cycle_echem, 'n_echem_shape_cycles', 0)} / {first_summary(within_cycle_echem, 'n_shape_features', 0)}",
        f"- ROI rows joined: {first_summary(within_cycle_echem, 'n_roi_rows', 0)}",
    ]
    for row in within_cycle_roi_corr[:8]:
        report_lines.append(
            f"- ROI shape correlation {row.get('feature')} vs {row.get('target')}: rho={fmt(row.get('rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in within_cycle_cycle_corr[:6]:
        report_lines.append(
            f"- Cycle shape correlation {row.get('feature')} vs {row.get('target')}: rho={fmt(row.get('rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in within_cycle_event_tests[:4]:
        report_lines.append(
            f"- Event shape test {row.get('feature')}: median event-control {fmt(row.get('positive_minus_negative_median'))}, p={fmt(row.get('mannwhitney_p'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(within_cycle_echem, 'guardrail', 'Within-cycle echem shape descriptors are proxy context only.')}")

    report_lines += [
        "",
        "## Echem-Shape-Conditioned ROI/Front Effects",
        "",
        f"- Rows event/control: {first_summary(echem_shape_conditioned, 'n_event_roi', 0)} / {first_summary(echem_shape_conditioned, 'n_control_roi', 0)}",
        f"- Shape PCA: {first_summary(echem_shape_conditioned, 'shape_pca', {}).get('n_shape_features_used', 0)} features, {first_summary(echem_shape_conditioned, 'shape_pca', {}).get('n_components', 0)} PCs, total explained variance {fmt(first_summary(echem_shape_conditioned, 'shape_pca', {}).get('explained_variance_total'))}",
        f"- Shape-residual classifier: ROC-AUC {fmt(echem_shape_model.get('mean_roc_auc'))}, balanced accuracy {fmt(echem_shape_model.get('mean_balanced_accuracy'))}",
    ]
    for row in echem_shape_conditioned_tests[:8]:
        report_lines.append(
            f"- Shape-conditioned {row.get('target_base')}: event-control residual median {fmt(row.get('event_minus_control_median'))}, p={fmt(row.get('p_value'))}"
        )
    for row in echem_shape_context[:6]:
        report_lines.append(
            f"- Shape context fit {row.get('target')}: variance explained {fmt(row.get('variance_explained_by_shape_pcs_and_event_ref'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in echem_shape_pc_corr[:4]:
        report_lines.append(
            f"- Shape PC correlation {row.get('pc')} vs {row.get('target')}: rho={fmt(row.get('rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(echem_shape_conditioned, 'guardrail', 'Echem-shape-conditioned residual audit unavailable.')}")

    report_lines += [
        "",
        "## Probabilistic Rollout Calibration",
        "",
        f"- Frame rows / ROI-method rows: {first_summary(rollout_calibration, 'n_frame_rows', 0)} / {first_summary(rollout_calibration, 'n_roi_method_rows', 0)}",
        f"- ROI / event-reference cycles: {first_summary(rollout_calibration, 'n_roi', 0)} / {first_summary(rollout_calibration, 'n_event_reference_cycles', 0)}",
        f"- Near-transition frame fraction: {fmt(first_summary(rollout_calibration, 'near_transition_frame_fraction'))}",
    ]
    for row in rollout_calibration_coverage:
        if row.get("alpha") == 0.05 and row.get("group") in {"all", "event_roi", "near_transition"}:
            report_lines.append(
                f"- 95% empirical coverage {row.get('method')} {row.get('group')}: global {fmt(row.get('coverage_global_weighted'))}, local {fmt(row.get('coverage_local_weighted'))}, n={fmt(row.get('n'), 0)}"
            )
    for row in rollout_calibration_tests[:5]:
        report_lines.append(
            f"- Residual test {row.get('method')} {row.get('contrast')}: median diff {fmt(row.get('median_diff'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n_a'), 0)}/{fmt(row.get('n_b'), 0)}"
        )
    for row in rollout_calibration_corr[:5]:
        report_lines.append(
            f"- Calibration/physics link {row.get('method')} {row.get('x_calibration_feature')} vs {row.get('y_physics_feature')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in rollout_calibration_top[:6]:
        report_lines.append(
            f"- Undercoverage priority {row.get('roi_id')} {row.get('method')}: q90 undercoverage {fmt(row.get('q90_undercoverage_rate'))}, priority {fmt(row.get('calibration_review_priority'))}, role {row.get('cohort_role_first')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(rollout_calibration, 'guardrail', 'Probabilistic rollout calibration unavailable.')}")




    report_lines += [
        "",
        "## Masked Residual State Transfer Warning",
        "",
        f"- Anchor/full cycles/permutations: {first_summary(masked_residual_transfer, 'n_anchor_cycles', 0)} / {first_summary(masked_residual_transfer, 'n_full_cycles', 0)} / {first_summary(masked_residual_transfer, 'n_permutation', 0)}",
        f"- Signature features: {first_summary(masked_residual_transfer, 'n_signature_features', 0)}",
        f"- Anchor leave-one-cycle transfer: rho={fmt(masked_residual_transfer_anchor.get('rho'))}, p={fmt(masked_residual_transfer_anchor.get('p_value'))}, n={fmt(masked_residual_transfer_anchor.get('n_anchor'), 0)}",
    ]
    for row in masked_residual_transfer_features[:6]:
        report_lines.append(
            f"- Signature feature {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, permutation p={fmt(row.get('permutation_p_abs_median_diff'))}"
        )
    for row in masked_residual_transfer_tests[:8]:
        report_lines.append(
            f"- Transfer target {row.get('target')} {row.get('score')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, AUC {fmt(row.get('abs_oriented_auc'))}, permutation p={fmt(row.get('permutation_p_abs_median_diff'))}, n={fmt(row.get('n_positive'), 0)}/{fmt(row.get('n_negative'), 0)}"
        )
    for row in masked_residual_transfer_corr[:6]:
        report_lines.append(
            f"- Transfer/context link {row.get('context')}: rho={fmt(row.get('rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in masked_residual_transfer_ranked[:6]:
        report_lines.append(
            f"- Transfer-ranked cycle {fmt(row.get('cycleNo'), 0)}: score {fmt(row.get('transferred_masked_residual_signature'))}, future8={fmt(row.get('future_any_drop_within_8cycles'), 0)}, future16={fmt(row.get('future_any_drop_within_16cycles'), 0)}, abrupt={fmt(row.get('any_abrupt_drop'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(masked_residual_transfer, 'guardrail', 'Masked residual state-transfer audit unavailable.')}")

    report_lines += [
        "",
        "## Transfer-Ranked ROI Reconstruction And Masked Rollout",
        "",
        f"- Sampled cycles/reconstructed candidates/ROI rows: {first_summary(transfer_ranked_reconstruction, 'n_cycles_sampled', 0)} / {first_summary(transfer_ranked_reconstruction, 'n_reconstructed_candidates', 0)} / {first_summary(transfer_ranked_reconstruction, 'n_roi_rows', 0)}",
        f"- Exported ROI sequences: {first_summary(transfer_ranked_sequences, 'n_roi_sequences', 0)} at {first_summary(transfer_ranked_sequences, 'samples_per_roi', 0)} frames per ROI",
        f"- Masked rollout ROI/frame rows: {first_summary(transfer_ranked_rollout, 'n_roi', 0)} / {first_summary(transfer_ranked_rollout, 'n_frame_metric_rows', 0)}",
        f"- Best method counts inside particle masks: {transfer_ranked_rollout_best_counts}",
    ]
    for row in transfer_ranked_sampled[:8]:
        report_lines.append(
            f"- Transfer sampled cycle {fmt(row.get('cycleNo'), 0)} rank {fmt(row.get('transfer_rank'), 0)}: score {fmt(row.get('transferred_masked_residual_signature'))}, future8={fmt(row.get('future_any_drop_within_8cycles'), 0)}, future16={fmt(row.get('future_any_drop_within_16cycles'), 0)}, candidates={fmt(row.get('n_candidates'), 0)}"
        )
    for cycle_key, row in list(transfer_ranked_sequence_cycles.items())[:8]:
        report_lines.append(
            f"- Transfer sequence cycle {cycle_key}: mean ROI delta {fmt(row.get('mean_roi_delta'))}, norm delta {fmt(row.get('mean_norm_roi_delta'))}, n={fmt(row.get('n_roi_sequences'), 0)}"
        )
    for row in transfer_ranked_rollout_methods:
        report_lines.append(
            f"- Transfer-ranked {row.get('method')}: particle-MSE median {fmt(row.get('particle_mse_mean_median'))}, nonparticle-MSE median {fmt(row.get('nonparticle_mse_mean_median'))}, particle/nonparticle ratio median {fmt(row.get('particle_to_nonparticle_mse_ratio_mean_median'))}"
        )
    for row in transfer_ranked_rollout_difficult[:6]:
        report_lines.append(
            f"- Transfer-ranked difficult ROI {row.get('roi_id')} {row.get('method')} cycle {fmt(row.get('cycleNo'), 0)}: particle MSE {fmt(row.get('particle_mse_mean'))}, particle/nonparticle ratio {fmt(row.get('particle_to_nonparticle_mse_ratio_mean'))}"
        )
    for row in transfer_ranked_top_rois[:6]:
        report_lines.append(
            f"- Transfer-ranked ROI candidate cycle {fmt(row.get('cycleNo'), 0)} obj {fmt(row.get('object_candidate_rank'), 0)}: validation score {fmt(row.get('validation_score'))}, mean abs z {fmt(row.get('object_mean_abs_z'))}, future8={fmt(row.get('future_any_drop_within_8cycles'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(transfer_ranked_reconstruction, 'guardrail', 'Automatic transfer-ranked reconstruction unavailable.')} {first_summary(transfer_ranked_rollout, 'guardrail', 'Masked rollout unavailable.')}")

    report_lines += [
        "",
        "## Transfer-Ranked Front Physics Audit",
        "",
        f"- Threshold-front ROI/cycles/quantiles: {first_summary(transfer_ranked_fronts, 'n_roi', 0)} / {first_summary(transfer_ranked_front_physics, 'n_cycles', 0)} / {len(first_summary(transfer_ranked_fronts, 'threshold_quantiles', []))}",
        f"- Front-physics target positives: {first_summary(transfer_ranked_front_physics, 'target_positive_counts', {})}",
    ]
    for row in transfer_ranked_front_target_tests[:8]:
        report_lines.append(
            f"- Front target {row.get('target')} {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, AUC {fmt(row.get('abs_oriented_auc'))}, permutation p={fmt(row.get('permutation_p_abs_median_diff'))}, n={fmt(row.get('n_positive'), 0)}/{fmt(row.get('n_negative'), 0)}"
        )
    for row in transfer_ranked_front_corr[:6]:
        report_lines.append(
            f"- Front/residual link {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in transfer_ranked_front_cycle_tests[:6]:
        report_lines.append(
            f"- Cycle-collapsed front target {row.get('target')} {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, AUC {fmt(row.get('abs_oriented_auc'))}, permutation p={fmt(row.get('permutation_p_abs_median_diff'))}, n={fmt(row.get('n_positive'), 0)}/{fmt(row.get('n_negative'), 0)}"
        )
    for row in transfer_ranked_front_group[:8]:
        report_lines.append(
            f"- Transfer front cycle {fmt(row.get('cycleNo'), 0)}: phase slope {fmt(row.get('phase_slope_median_per_s'))}, apparent D median {fmt(row.get('diffusion_proxy_median_um2_per_s'))}, phase score {fmt(row.get('threshold_robust_phase_score'))}, n={fmt(row.get('n_roi'), 0)}"
        )
    for row in transfer_ranked_front_review[:8]:
        report_lines.append(
            f"- Front-physics review ROI {row.get('roi_id')}: score {fmt(row.get('front_physics_review_score'))}, future8={fmt(row.get('future_any_drop_within_8cycles'), 0)}, phase score {fmt(row.get('threshold_robust_phase_score'))}, apparent D {fmt(row.get('diffusion_proxy_median_um2_per_s'))}, DMD particle MSE {fmt(row.get('low_rank_dmd_particle_mse_mean'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(transfer_ranked_front_physics, 'guardrail', first_summary(transfer_ranked_fronts, 'diffusion_guardrail', 'Automatic front physics audit unavailable.'))}")

    report_lines += [
        "",
        "## Transfer-Ranked Residual Transition Timing",
        "",
        f"- Phase-kinetic ROI/timing rows/permutations: {first_summary(transfer_ranked_residual_timing, 'n_roi', 0)} / {first_summary(transfer_ranked_residual_timing, 'n_roi_method_rows', 0)} / {first_summary(transfer_ranked_residual_timing, 'n_permutation', 0)}",
        f"- Timing target positives: {first_summary(transfer_ranked_residual_timing, 'target_positive_counts', {})}",
    ]
    for row in transfer_ranked_timing_align[:6]:
        report_lines.append(
            f"- Transfer timing alignment {row.get('method')} {row.get('distance_feature')}: median distance {fmt(row.get('median_distance_to_transition'))}, null mean {fmt(row.get('null_median_distance_mean'))}, empirical p={fmt(row.get('empirical_p_distance_le_observed'))}, n={fmt(row.get('n_roi'), 0)}"
        )
    for row in transfer_ranked_timing_target[:8]:
        report_lines.append(
            f"- Transfer timing target {row.get('target')} {row.get('method')} {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, AUC {fmt(row.get('abs_oriented_auc'))}, permutation p={fmt(row.get('permutation_p_abs_median_diff'))}, n={fmt(row.get('n_positive'), 0)}/{fmt(row.get('n_negative'), 0)}"
        )
    for row in transfer_ranked_timing_target_corr[:6]:
        report_lines.append(
            f"- Transfer timing/target link {row.get('method')} {row.get('x')} vs {row.get('target')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in transfer_ranked_timing_top_rois[:6]:
        report_lines.append(
            f"- Near-transition transfer ROI {row.get('roi_id')} {row.get('method')}: near residual fraction {fmt(row.get('near_transition_residual_fraction'))}, peak MSE {fmt(row.get('residual_peak_particle_mse'))}, future8={fmt(row.get('future_any_drop_within_8cycles'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(transfer_ranked_residual_timing, 'guardrail', 'Transfer-ranked timing audit unavailable.')}")

    report_lines += [
        "",
        "## Multi-Cohort Future-Drop Weak Model",
        "",
        f"- ROI rows selected/transfer/features: {first_summary(multicohort_future_drop, 'n_selected_rows', 0)} / {first_summary(multicohort_future_drop, 'n_transfer_ranked_rows', 0)} / {first_summary(multicohort_future_drop, 'n_features', 0)}",
        f"- RF trees/permutation null: {first_summary(multicohort_future_drop, 'rf_trees', 0)} / {multicohort_null.get('n_permutation', 0)}",
    ]
    for row in multicohort_oof:
        report_lines.append(
            f"- Multi-cohort OOF {row.get('model')}: AUC {fmt(row.get('pooled_oof_roc_auc'))}, AP {fmt(row.get('pooled_oof_average_precision'))}, scored {fmt(row.get('n_scored'), 0)} rows over {fmt(row.get('n_scored_folds'), 0)}/{fmt(row.get('n_total_folds'), 0)} folds"
        )
    report_lines.append(
        f"- Multi-cohort permutation null {multicohort_null.get('model', 'NA')}: observed AUC {fmt(multicohort_null.get('observed_auc'))}, null mean {fmt(multicohort_null.get('null_auc_mean'))}, p95 {fmt(multicohort_null.get('null_auc_p95'))}, empirical p={fmt(multicohort_null.get('empirical_p_ge_observed'))}"
    )
    for row in multicohort_features[:8]:
        report_lines.append(
            f"- Multi-cohort feature {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, oriented AUC {fmt(row.get('oriented_auc'))}, p={fmt(row.get('mannwhitney_p'))}, n={fmt(row.get('n_positive'), 0)}/{fmt(row.get('n_negative'), 0)}"
        )
    for row in multicohort_importance[:8]:
        report_lines.append(
            f"- Multi-cohort importance {row.get('model')} {row.get('feature')}: {fmt(row.get('importance'))}"
        )
    for row in multicohort_leave:
        report_lines.append(
            f"- Leave-cohort {row.get('train_cohort')} -> {row.get('test_cohort')} {row.get('model')}: status {row.get('status')}, train positives/negatives {fmt(row.get('n_positive_train'), 0)}/{fmt(row.get('n_negative_train'), 0)}, test positives/negatives {fmt(row.get('n_positive_test'), 0)}/{fmt(row.get('n_negative_test'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(multicohort_future_drop, 'guardrail', 'Multi-cohort future-drop model unavailable.')}")

    report_lines += [
        "",
        "## Active-Learning QC Prioritization",
        "",
        f"- Candidate/visual/immediate rows: {first_summary(active_learning_qc, 'n_candidate_rows', 0)} / {first_summary(active_learning_qc, 'n_rows_with_visual_asset', 0)} / {active_qc_tiers.get('immediate_manual_qc', 0)}",
        f"- Tier counts: {active_qc_tiers}",
    ]
    for row in active_qc_top[:8]:
        report_lines.append(
            f"- Active-QC ROI {row.get('roi_id')}: rank {fmt(row.get('active_learning_rank'), 0)}, tier {row.get('recommended_qc_tier')}, score {fmt(row.get('active_learning_qc_score'))}, cycle {fmt(row.get('cycleNo'), 0)}, tags {row.get('review_reason_tags')}, model p={fmt(row.get('model_future_drop_probability'))}"
        )
    for row in active_qc_cycles[:6]:
        report_lines.append(
            f"- Active-QC cycle {fmt(row.get('cycleNo'), 0)}: max score {fmt(row.get('max_active_learning_qc_score'))}, mean score {fmt(row.get('mean_active_learning_qc_score'))}, ROI {fmt(row.get('n_roi'), 0)}, immediate {fmt(row.get('n_immediate'), 0)}"
        )
    for row in active_qc_reasons[:8]:
        report_lines.append(
            f"- Active-QC reason {row.get('reason_tag')}: n={fmt(row.get('n_roi'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(active_learning_qc, 'guardrail', 'Active-learning QC prioritization unavailable.')}")

    report_lines += [
        "",
        "## Automatic QC Triage Surrogate",
        "",
        f"- Candidate/manual-status rows: {first_summary(automatic_qc_triage, 'n_candidates', 0)} / {first_summary(automatic_qc_triage, 'manual_status_counts', {})}",
        f"- Likely/artifact/diffusion-guardrail counts: {first_summary(automatic_qc_triage, 'likely_interpretable_count', 0)} / {first_summary(automatic_qc_triage, 'artifact_risk_count', 0)} / {first_summary(automatic_qc_triage, 'diffusion_guardrail_count', 0)}",
    ]
    for row in auto_qc_triage_tiers:
        report_lines.append(
            f"- Auto-QC tier {row.get('auto_qc_tier')}: n={fmt(row.get('n_candidates'), 0)}, median score {fmt(row.get('median_surrogate_score'))}, median risk {fmt(row.get('median_artifact_risk'))}, diffusion guardrail rate {fmt(row.get('diffusion_guardrail_rate'))}"
        )
    for row in auto_qc_likely[:6]:
        report_lines.append(
            f"- Auto-QC likely ROI {row.get('roi_id')}: score {fmt(row.get('automatic_qc_surrogate_score'))}, risk {fmt(row.get('automatic_artifact_risk_score'))}, cycle {fmt(row.get('cycleNo'), 0)}, role {row.get('cohort_role')}, reason {row.get('recommended_review_reason')}"
        )
    for row in auto_qc_artifact[:6]:
        report_lines.append(
            f"- Auto-QC artifact-risk ROI {row.get('roi_id')}: score {fmt(row.get('automatic_qc_surrogate_score'))}, risk {fmt(row.get('automatic_artifact_risk_score'))}, diffusion guardrail {fmt(row.get('auto_diffusion_guardrail'), 0)}, reason {row.get('recommended_review_reason')}"
        )
    for row in auto_qc_tests[:6]:
        report_lines.append(
            f"- Auto-QC contrast {row.get('feature')} vs {row.get('target')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, p={fmt(row.get('mannwhitney_p'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in auto_qc_corr[:6]:
        report_lines.append(
            f"- Auto-QC correlation {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(automatic_qc_triage, 'guardrail', 'Automatic QC triage surrogate unavailable.')}")

    report_lines += [
        "",
        "## QC Decision Evidence Ledger",
        "",
        f"- Candidate/manual-status rows: {first_summary(qc_decision_ledger, 'n_candidates', 0)} / {first_summary(qc_decision_ledger, 'manual_status_counts', {})}",
        f"- Decision action counts: {qc_decision_actions}",
        f"- Visual-asset candidates: {first_summary(qc_decision_ledger, 'n_visual_asset_candidates', 0)}",
    ]
    for row in qc_decision_accept[:5]:
        report_lines.append(
            f"- QC possible-accept ROI {row.get('roi_id')}: rank {fmt(row.get('qc_decision_rank'), 0)}, score {fmt(row.get('qc_decision_priority_score'))}, risk {fmt(row.get('automatic_artifact_risk_score'))}, cycle {fmt(row.get('cycleNo'), 0)}, reasons {row.get('decision_reason_tags')}"
        )
    for row in qc_decision_artifact[:5]:
        report_lines.append(
            f"- QC artifact/reject-first ROI {row.get('roi_id')}: artifact score {fmt(row.get('artifact_review_priority_score'))}, risk {fmt(row.get('automatic_artifact_risk_score'))}, cycle {fmt(row.get('cycleNo'), 0)}, reasons {row.get('decision_reason_tags')}"
        )
    for row in qc_decision_cycles[:5]:
        report_lines.append(
            f"- QC cycle {fmt(row.get('cycleNo'), 0)}: n={fmt(row.get('n_candidates'), 0)}, possible accept={fmt(row.get('n_review_for_accept'), 0)}, artifact-first={fmt(row.get('n_artifact_first'), 0)}, max score {fmt(row.get('max_decision_score'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(qc_decision_ledger, 'guardrail', 'QC decision evidence ledger unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced ROI Expansion Manifest",
        "",
        f"- Ranked/selected/sampled cycles: {first_summary(source_balanced_expansion, 'n_ranked_cycles', 0)} / {first_summary(source_balanced_expansion, 'n_selected_cycles', 0)} / {first_summary(source_balanced_expansion, 'n_sampled_cycles', 0)}",
        f"- Sources selected: {first_summary(source_balanced_expansion, 'n_sources_selected', 0)}",
        f"- New cycle/source pairs versus existing video cohorts: {first_summary(source_balanced_expansion, 'n_new_selected_cycles', 0)}",
        f"- Reconstructed candidates/ROI rows/missing cycles: {first_summary(source_balanced_expansion, 'n_reconstructed_candidates', 0)} / {first_summary(source_balanced_expansion, 'n_roi_rows', 0)} / {first_summary(source_balanced_expansion, 'n_missing_cycles', 0)}",
        f"- Selected label counts: {first_summary(source_balanced_expansion, 'selected_label_counts', {})}",
        f"- Selection reason counts: {first_summary(source_balanced_expansion, 'selection_reason_counts', {})}",
    ]
    for row in source_balanced_expansion_sources[:10]:
        report_lines.append(
            f"- Expansion source {row.get('source_stem')}: selected {fmt(row.get('selected_cycles'), 0)}, new {fmt(row.get('new_cycles'), 0)}, future16+ {fmt(row.get('future16_positive'), 0)}, candidates {fmt(row.get('total_candidates'), 0)}"
        )
    for row in source_balanced_expansion_top[:6]:
        report_lines.append(
            f"- Expansion ROI candidate cycle {fmt(row.get('cycleNo'), 0)} {row.get('source_stem')} rank {fmt(row.get('object_candidate_rank'), 0)}: score {fmt(row.get('validation_score'))}, future16 {fmt(row.get('future_any_drop_within_16cycles'), 0)}, existing cohort {row.get('already_in_existing_video_cohort')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_expansion, 'guardrail', 'Source-balanced expansion manifest unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced ROI Sequence Export and Rollout Audit",
        "",
        f"- Exported ROI sequences/cycles/sources/failures: {first_summary(source_balanced_sequences, 'n_roi_sequences', 0)} / {first_summary(source_balanced_sequences, 'n_cycles', 0)} / {first_summary(source_balanced_sequences, 'n_sources', 0)} / {first_summary(source_balanced_sequences, 'n_failed', 0)}",
        f"- Crop/output size/samples per ROI: {first_summary(source_balanced_sequences, 'crop_size_full', 0)} / {first_summary(source_balanced_sequences, 'output_size', 0)} / {first_summary(source_balanced_sequences, 'samples_per_roi', 0)}",
        f"- Rollout audit ROI sequences/cycles/sources: {first_summary(source_balanced_sequence_rollout, 'n_roi_sequences', 0)} / {first_summary(source_balanced_sequence_rollout, 'n_cycles', 0)} / {first_summary(source_balanced_sequence_rollout, 'n_sources', 0)}",
        f"- Future8/future16 positive sequences: {first_summary(source_balanced_sequence_rollout, 'future8_positive_sequences', 0)} / {first_summary(source_balanced_sequence_rollout, 'future16_positive_sequences', 0)}",
    ]
    for row in source_balanced_rollout_roi_tests[:10]:
        report_lines.append(
            f"- Source-balanced ROI feature {row.get('target')} {row.get('feature')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, source eta2 {fmt(row.get('source_eta2'))}, median pos-neg {fmt(row.get('median_positive_minus_negative'))}"
        )
    for row in source_balanced_rollout_cycle_tests[:6]:
        report_lines.append(
            f"- Source-balanced cycle feature {row.get('target')} {row.get('feature')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, median pos-neg {fmt(row.get('median_positive_minus_negative'))}"
        )
    for row in source_balanced_rollout_sources[:6]:
        report_lines.append(
            f"- Source-balanced rollout source {row.get('source_stem')}: ROI {fmt(row.get('n_roi'), 0)}, cycles {fmt(row.get('n_cycles'), 0)}, persistence MSE {fmt(row.get('persistence_mse_mean'))}, future16 seq {fmt(row.get('future_any_drop_within_16cycles'), 0)}"
        )
    report_lines.append(f"- Sequence guardrail: {first_summary(source_balanced_sequences, 'guardrail', 'Source-balanced sequence export unavailable.')}")
    report_lines.append(f"- Rollout guardrail: {first_summary(source_balanced_sequence_rollout, 'guardrail', 'Source-balanced rollout audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Sequence Source-Control Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_balanced_sequence_source_control, 'n_rows', 0)} / {first_summary(source_balanced_sequence_source_control, 'n_cycles', 0)} / {first_summary(source_balanced_sequence_source_control, 'n_sources', 0)}",
        f"- Features/scalar rows/permutations: {len(first_summary(source_balanced_sequence_source_control, 'features_tested', []))} / {first_summary(source_balanced_sequence_source_control, 'n_scalar_rows', 0)} / {first_summary(source_balanced_sequence_source_control, 'n_permutation', 0)}",
        f"- Strict scalar rows / source-heldout model rows AUC>=0.65: {first_summary(source_balanced_sequence_source_control, 'n_strict_scalar_rows', 0)} / {first_summary(source_balanced_sequence_source_control, 'n_source_model_auc_ge_065', 0)}",
        f"- Verdict: {first_summary(source_balanced_sequence_source_control, 'verdict', 'missing')}",
        f"- Best source-stratified scalar: {source_balanced_source_control_best_scalar.get('target', 'NA')} {source_balanced_source_control_best_scalar.get('feature', 'NA')} {source_balanced_source_control_best_scalar.get('transform', 'NA')} AUC {fmt(source_balanced_source_control_best_scalar.get('oriented_auc'))}, source p {fmt(source_balanced_source_control_best_scalar.get('source_stratified_auc_p'))}, AP {fmt(source_balanced_source_control_best_scalar.get('average_precision'))}",
        f"- Best source-heldout model: {source_balanced_source_control_best_model.get('target', 'NA')} {source_balanced_source_control_best_model.get('feature_set', 'NA')} AUC {fmt(source_balanced_source_control_best_model.get('roc_auc'))}, AP {fmt(source_balanced_source_control_best_model.get('average_precision'))}",
    ]
    for row in source_balanced_source_control_scalars[:6]:
        report_lines.append(
            f"- Source-control scalar {row.get('target')} {row.get('feature')} {row.get('transform')}: AUC {fmt(row.get('oriented_auc'))}, source p {fmt(row.get('source_stratified_auc_p'))}, cycle p {fmt(row.get('cycle_stratified_auc_p'))}, median pos-neg {fmt(row.get('median_positive_minus_negative'))}"
        )
    for row in source_balanced_source_control_deltas[:4]:
        report_lines.append(
            f"- Source-control model delta {row.get('target')} {row.get('group_col')} {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, delta vs context {fmt(row.get('delta_roc_auc_vs_context'))}, AP {fmt(row.get('average_precision'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_sequence_source_control, 'guardrail', 'Source-balanced sequence source-control audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Expansion Transport/Front Audit",
        "",
        f"- Expanded transport/front rows OK/failed/cycles/sources: {first_summary(source_balanced_expansion_transport_front, 'n_ok', 0)} / {first_summary(source_balanced_expansion_transport_front, 'n_failed', 0)} / {first_summary(source_balanced_expansion_transport_front, 'n_cycles', 0)} / {first_summary(source_balanced_expansion_transport_front, 'n_sources', 0)}",
        f"- Future8/future16 positive rows: {first_summary(source_balanced_expansion_transport_front, 'future8_positive_rows', 0)} / {first_summary(source_balanced_expansion_transport_front, 'future16_positive_rows', 0)}; flow method {first_summary(source_balanced_expansion_transport_front, 'flow_method', 'NA')}",
        f"- Best future8 transport/front row: {source_balanced_expansion_transport_best_future8.get('feature', 'NA')} {source_balanced_expansion_transport_best_future8.get('transform', 'NA')} AUC {fmt(source_balanced_expansion_transport_best_future8.get('oriented_auc'))}, AP {fmt(source_balanced_expansion_transport_best_future8.get('average_precision'))}, source-stratified p={fmt(source_balanced_expansion_transport_best_future8.get('source_stratified_permutation_p'))}",
        f"- Best future16 transport/front row: {source_balanced_expansion_transport_best_future16.get('feature', 'NA')} {source_balanced_expansion_transport_best_future16.get('transform', 'NA')} AUC {fmt(source_balanced_expansion_transport_best_future16.get('oriented_auc'))}, AP {fmt(source_balanced_expansion_transport_best_future16.get('average_precision'))}, source-stratified p={fmt(source_balanced_expansion_transport_best_future16.get('source_stratified_permutation_p'))}",
        f"- Top expanded candidate: {source_balanced_expansion_transport_top_candidate.get('roi_id', 'NA')} score {fmt(source_balanced_expansion_transport_top_candidate.get('expansion_transport_front_score'))}, future8 {fmt(source_balanced_expansion_transport_top_candidate.get('future_any_drop_within_8cycles'), 0)}, future16 {fmt(source_balanced_expansion_transport_top_candidate.get('future_any_drop_within_16cycles'), 0)}",
    ]
    for row in source_balanced_expansion_transport_tests[:8]:
        report_lines.append(
            f"- Expansion transport/front test {row.get('target', 'NA')} {row.get('feature', 'NA')} {row.get('transform', 'NA')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, MW p={fmt(row.get('mwu_p'))}, source-stratified p={fmt(row.get('source_stratified_permutation_p'))}"
        )
    for row in source_balanced_expansion_transport_sources[:6]:
        report_lines.append(
            f"- Expansion transport/front source {row.get('source_stem', 'NA')}: ROI {fmt(row.get('n_roi'), 0)}, cycles {fmt(row.get('n_cycles'), 0)}, max score {fmt(row.get('max_expansion_transport_front_score'))}, future16 rows {fmt(row.get('future16_positive_rows'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_expansion_transport_front, 'guardrail', 'Source-balanced expansion transport/front audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Mask/Front Sanity Audit",
        "",
        f"- Mask/front ROI sequences/cycles/sources: {first_summary(source_balanced_mask_front, 'n_roi_sequences', 0)} / {first_summary(source_balanced_mask_front, 'n_cycles', 0)} / {first_summary(source_balanced_mask_front, 'n_sources', 0)}",
        f"- Future8/future16 positive sequences: {first_summary(source_balanced_mask_front, 'future8_positive_sequences', 0)} / {first_summary(source_balanced_mask_front, 'future16_positive_sequences', 0)}",
        f"- Assumed output-pixel scale: {fmt(first_summary(source_balanced_mask_front, 'pixel_size_um_assumed'))} um",
    ]
    for row in source_balanced_mask_front_roi_tests[:10]:
        report_lines.append(
            f"- Source-balanced mask/front ROI feature {row.get('target')} {row.get('feature')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, source eta2 {fmt(row.get('source_eta2'))}, median pos-neg {fmt(row.get('median_positive_minus_negative'))}"
        )
    for row in source_balanced_mask_front_cycle_tests[:6]:
        report_lines.append(
            f"- Source-balanced mask/front cycle feature {row.get('target')} {row.get('feature')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, median pos-neg {fmt(row.get('median_positive_minus_negative'))}"
        )
    for row in source_balanced_mask_front_sources[:6]:
        report_lines.append(
            f"- Source-balanced mask/front source {row.get('source_stem')}: ROI {fmt(row.get('n_roi'), 0)}, cycles {fmt(row.get('n_cycles'), 0)}, q70 radius slope {fmt(row.get('front_radius_q70_slope_px_per_norm_time'))}, future16 seq {fmt(row.get('future_any_drop_within_16cycles'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_mask_front, 'guardrail', 'Source-balanced mask/front sanity audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Mask/Front Source-Residual Audit",
        "",
        f"- Rows/features/sources: {first_summary(source_balanced_mask_front_source_residual, 'n_rows', 0)} / {first_summary(source_balanced_mask_front_source_residual, 'n_features_tested', 0)} / {first_summary(source_balanced_mask_front_source_residual, 'n_sources', 0)}",
        f"- Best raw future16 feature: {first_summary(source_balanced_mask_front_source_residual, 'future16_raw_best', {}).get('feature', 'NA')} AUC {fmt(first_summary(source_balanced_mask_front_source_residual, 'future16_raw_best', {}).get('oriented_auc'))}, source eta2 {fmt(first_summary(source_balanced_mask_front_source_residual, 'future16_raw_best', {}).get('source_eta2_after_transform'))}",
        f"- Best source-residual future16 feature: {source_balanced_mask_front_resid_best.get('feature', 'NA')} AUC {fmt(source_balanced_mask_front_resid_best.get('oriented_auc'))}, AP {fmt(source_balanced_mask_front_resid_best.get('average_precision'))}, p={fmt(source_balanced_mask_front_resid_best.get('mwu_p'))}",
        f"- Best within-source-rank future16 feature: {source_balanced_mask_front_rank_best.get('feature', 'NA')} AUC {fmt(source_balanced_mask_front_rank_best.get('oriented_auc'))}, AP {fmt(source_balanced_mask_front_rank_best.get('average_precision'))}, p={fmt(source_balanced_mask_front_rank_best.get('mwu_p'))}",
        f"- Guardrail: {first_summary(source_balanced_mask_front_source_residual, 'guardrail', 'Source-balanced mask/front source-residual audit unavailable.')}",
    ]

    report_lines += [
        "",
        "## Source-Balanced Residual Dictionary Audit",
        "",
        f"- ROI sequences/cycles/sources: {first_summary(source_balanced_residual_dictionary, 'n_roi_sequences', 0)} / {first_summary(source_balanced_residual_dictionary, 'n_cycles', 0)} / {first_summary(source_balanced_residual_dictionary, 'n_sources', 0)}",
        f"- PCA components/downsample/variance explained: {first_summary(source_balanced_residual_dictionary, 'n_components', 0)} / {first_summary(source_balanced_residual_dictionary, 'downsample', 0)} / {fmt(first_summary(source_balanced_residual_dictionary, 'pca_explained_variance_ratio_sum'))}",
        f"- Feature set sizes: {first_summary(source_balanced_residual_dictionary, 'feature_set_sizes', {})}",
        f"- Residual dictionary leave-cycle future16: AUC {fmt(source_balanced_resdict_cycle16.get('roc_auc'))}, AP {fmt(source_balanced_resdict_cycle16.get('average_precision'))}; leave-source future16: AUC {fmt(source_balanced_resdict_source16.get('roc_auc'))}, AP {fmt(source_balanced_resdict_source16.get('average_precision'))}",
        f"- Top residual ROI/cycle future16 scalar: {source_balanced_resdict_top_roi16.get('feature', 'NA')} AUC {fmt(source_balanced_resdict_top_roi16.get('oriented_auc'))}, eta2 {fmt(source_balanced_resdict_top_roi16.get('source_eta2'))} / {source_balanced_resdict_top_cycle16.get('feature', 'NA')} AUC {fmt(source_balanced_resdict_top_cycle16.get('oriented_auc'))}, eta2 {fmt(source_balanced_resdict_top_cycle16.get('source_eta2'))}",
    ]
    for row in source_balanced_resdict_metrics[:10]:
        report_lines.append(
            f"- Source-balanced residual dictionary metric {row.get('group_col')} {row.get('target')} {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    for row in source_balanced_resdict_roi_tests[:8]:
        report_lines.append(
            f"- Source-balanced residual dictionary ROI feature {row.get('target')} {row.get('feature')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, source eta2 {fmt(row.get('source_eta2'))}"
        )
    for row in source_balanced_resdict_cycle_tests[:6]:
        report_lines.append(
            f"- Source-balanced residual dictionary cycle feature {row.get('target')} {row.get('feature')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, source eta2 {fmt(row.get('source_eta2'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_residual_dictionary, 'guardrail', 'Source-balanced residual dictionary audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Residual Dictionary Source-Residual Audit",
        "",
        f"- Rows/features/sources: {first_summary(source_balanced_resdict_source_residual, 'n_rows', 0)} / {first_summary(source_balanced_resdict_source_residual, 'n_features_tested', 0)} / {first_summary(source_balanced_resdict_source_residual, 'n_sources', 0)}",
        f"- Feature family counts: {first_summary(source_balanced_resdict_source_residual, 'feature_family_counts', {})}",
        f"- Best future16 source-residual residual dictionary feature: {source_balanced_resdict_sr_best.get('feature', 'NA')} AUC {fmt(source_balanced_resdict_sr_best.get('oriented_auc'))}, AP {fmt(source_balanced_resdict_sr_best.get('average_precision'))}, eta2 {fmt(source_balanced_resdict_sr_best.get('source_eta2_after_transform'))}, median pos-neg {fmt(source_balanced_resdict_sr_best.get('median_positive_minus_negative'))}",
        f"- Best future16 within-source-rank residual dictionary feature: {source_balanced_resdict_rank_best.get('feature', 'NA')} AUC {fmt(source_balanced_resdict_rank_best.get('oriented_auc'))}, AP {fmt(source_balanced_resdict_rank_best.get('average_precision'))}, eta2 {fmt(source_balanced_resdict_rank_best.get('source_eta2_after_transform'))}",
        f"- Best future16 source-residual feature overall: {source_balanced_resdict_sr_transform_best.get('feature', 'NA')} ({source_balanced_resdict_sr_transform_best.get('feature_family', 'NA')}) AUC {fmt(source_balanced_resdict_sr_transform_best.get('oriented_auc'))}, AP {fmt(source_balanced_resdict_sr_transform_best.get('average_precision'))}",
        f"- Guardrail: {first_summary(source_balanced_resdict_source_residual, 'guardrail', 'Source-balanced residual dictionary source-residual audit unavailable.')}",
    ]

    report_lines += [
        "",
        "## Source-Balanced Residual Dictionary Normalized Readout",
        "",
        f"- Rows/cycles/sources: {first_summary(source_balanced_resdict_normalized_readout, 'n_rows', 0)} / {first_summary(source_balanced_resdict_normalized_readout, 'n_cycles', 0)} / {first_summary(source_balanced_resdict_normalized_readout, 'n_sources', 0)}",
        f"- Feature set sizes: {first_summary(source_balanced_resdict_normalized_readout, 'feature_set_sizes', {})}",
        f"- Future16 leave-source raw residual dictionary: AUC {fmt(source_balanced_resdict_norm_raw_source16.get('roc_auc'))}, AP {fmt(source_balanced_resdict_norm_raw_source16.get('average_precision'))}",
        f"- Future16 leave-source source-residual residual dictionary: AUC {fmt(source_balanced_resdict_norm_sr_source16.get('roc_auc'))}, AP {fmt(source_balanced_resdict_norm_sr_source16.get('average_precision'))}",
        f"- Future16 leave-source within-source-rank residual dictionary: AUC {fmt(source_balanced_resdict_norm_rank_source16.get('roc_auc'))}, AP {fmt(source_balanced_resdict_norm_rank_source16.get('average_precision'))}",
        f"- Best future16 leave-source readout: {source_balanced_resdict_norm_best.get('feature_set', 'NA')} AUC {fmt(source_balanced_resdict_norm_best.get('roc_auc'))}, AP {fmt(source_balanced_resdict_norm_best.get('average_precision'))}, n={fmt(source_balanced_resdict_norm_best.get('n_eval'), 0)}",
        f"- Same single-feature leave-cycle future16 readout: AUC {fmt(source_balanced_resdict_norm_cycle_single16.get('roc_auc'))}, AP {fmt(source_balanced_resdict_norm_cycle_single16.get('average_precision'))}",
        f"- Permutation null: n={fmt(source_balanced_resdict_norm_perm.get('n_permutations'), 0)}, AUC p95={fmt(source_balanced_resdict_norm_perm.get('null_roc_auc_p95'))}, empirical p(AUC)={fmt(source_balanced_resdict_norm_perm.get('empirical_p_roc_auc'))}, empirical p(AP)={fmt(source_balanced_resdict_norm_perm.get('empirical_p_average_precision'))}",
        f"- Guardrail: {first_summary(source_balanced_resdict_normalized_readout, 'guardrail', 'Source-balanced normalized residual readout unavailable.')}",
    ]

    report_lines += [
        "",
        "## Source-Balanced Residual Temporal Specificity Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_balanced_residual_temporal_specificity, 'n_rows', 0)} / {first_summary(source_balanced_residual_temporal_specificity, 'n_cycles', 0)} / {first_summary(source_balanced_residual_temporal_specificity, 'n_sources', 0)}",
        f"- Event cycles and label counts: {first_summary(source_balanced_residual_temporal_specificity, 'event_cycles', [])} / {first_summary(source_balanced_residual_temporal_specificity, 'label_counts', {})}",
        f"- Primary source-residual reconstruction-error future8: AUC {fmt(source_balanced_temporal_primary8.get('future_auc'))}, past-control AUC {fmt(source_balanced_temporal_primary8.get('past_auc_fixed_direction'))}, future-minus-control {fmt(source_balanced_temporal_primary8.get('future_minus_max_control_auc'))}",
        f"- Primary source-residual reconstruction-error future16: AUC {fmt(source_balanced_temporal_primary16.get('future_auc'))}, AP {fmt(source_balanced_temporal_primary16.get('future_ap'))}, past-control AUC {fmt(source_balanced_temporal_primary16.get('past_auc_fixed_direction'))}, future-minus-control {fmt(source_balanced_temporal_primary16.get('future_minus_max_control_auc'))}",
        f"- Primary future16 within-source shift null: n={fmt(source_balanced_temporal_shift16.get('n_permutations'), 0)}, null AUC p95={fmt(source_balanced_temporal_shift16.get('null_roc_auc_p95'))}, empirical p(AUC)={fmt(source_balanced_temporal_shift16.get('empirical_p_roc_auc'))}",
        f"- Most future-specific row overall: {source_balanced_temporal_top.get('feature', 'NA')} / {source_balanced_temporal_top.get('transform', 'NA')} / {fmt(source_balanced_temporal_top.get('window_cycles'), 0)} cycles, future AUC {fmt(source_balanced_temporal_top.get('future_auc'))}, past-control AUC {fmt(source_balanced_temporal_top.get('past_auc_fixed_direction'))}, delta {fmt(source_balanced_temporal_top.get('future_minus_max_control_auc'))}",
        f"- Guardrail: {first_summary(source_balanced_residual_temporal_specificity, 'guardrail', 'Source-balanced residual temporal specificity audit unavailable.')}",
    ]

    report_lines += [
        "",
        "## Source-Balanced Future-Specific Residual Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_balanced_future_specific_residual, 'n_rows', 0)} / {first_summary(source_balanced_future_specific_residual, 'n_cycles', 0)} / {first_summary(source_balanced_future_specific_residual, 'n_sources', 0)}",
        f"- Event cycles and label counts: {first_summary(source_balanced_future_specific_residual, 'event_cycles', [])} / {first_summary(source_balanced_future_specific_residual, 'label_counts', {})}",
        f"- Primary source-residual reconstruction-error future16 all rows: AUC {fmt(source_balanced_future_specific_primary16_all.get('oriented_auc'))}, AP {fmt(source_balanced_future_specific_primary16_all.get('average_precision'))}",
        f"- Primary source-residual reconstruction-error future16 excluding past16 rows: AUC {fmt(source_balanced_future_specific_primary16_clean.get('oriented_auc'))}, AP {fmt(source_balanced_future_specific_primary16_clean.get('average_precision'))}, n={fmt(source_balanced_future_specific_primary16_clean.get('n'), 0)}",
        f"- Primary source-residual reconstruction-error future16 pre-first-event only: AUC {fmt(source_balanced_future_specific_primary16_pre.get('oriented_auc'))}, AP {fmt(source_balanced_future_specific_primary16_pre.get('average_precision'))}, n={fmt(source_balanced_future_specific_primary16_pre.get('n'), 0)}",
        f"- Top clean scalar row: {source_balanced_future_specific_top_clean.get('feature', 'NA')} / {source_balanced_future_specific_top_clean.get('transform', 'NA')} / window {fmt(source_balanced_future_specific_top_clean.get('target_window'), 0)} / {source_balanced_future_specific_top_clean.get('subset', 'NA')} AUC {fmt(source_balanced_future_specific_top_clean.get('oriented_auc'))}, AP {fmt(source_balanced_future_specific_top_clean.get('average_precision'))}",
        f"- Best grouped residual delta over past-event context: {source_balanced_future_specific_top_delta.get('group_col', 'NA')} {source_balanced_future_specific_top_delta.get('target', 'NA')} {source_balanced_future_specific_top_delta.get('feature_set', 'NA')} delta AUC {fmt(source_balanced_future_specific_top_delta.get('delta_roc_auc'))}, model AUC {fmt(source_balanced_future_specific_top_delta.get('roc_auc'))}, base AUC {fmt(source_balanced_future_specific_top_delta.get('base_roc_auc'))}",
        f"- Guardrail: {first_summary(source_balanced_future_specific_residual, 'guardrail', 'Source-balanced future-specific residual audit unavailable.')}",
    ]

    report_lines += [
        "",
        "## Source-Balanced Pre-Event Sampling Manifest",
        "",
        f"- Selected/sample cycles/sources: {first_summary(source_balanced_pre_event_sampling, 'n_selected_cycles', 0)} / {first_summary(source_balanced_pre_event_sampling, 'n_sampled_cycles', 0)} / {first_summary(source_balanced_pre_event_sampling, 'n_sources_selected', 0)}",
        f"- New cycle/source pairs vs existing video cohorts: {first_summary(source_balanced_pre_event_sampling, 'n_new_selected_cycles', 0)} of {first_summary(source_balanced_pre_event_sampling, 'n_selected_cycles', 0)}",
        f"- Event cycles: {first_summary(source_balanced_pre_event_sampling, 'event_cycles', [])}",
        f"- Cycle bins: {source_balanced_pre_event_bins}",
        f"- ROI proposal bins: {source_balanced_pre_event_roi_bins}",
        f"- Reconstructed candidates / ROI rows / missing cycles: {first_summary(source_balanced_pre_event_sampling, 'n_reconstructed_candidates', 0)} / {first_summary(source_balanced_pre_event_sampling, 'n_roi_rows', 0)} / {first_summary(source_balanced_pre_event_sampling, 'n_missing_cycles', 0)}",
        f"- Guardrail: {first_summary(source_balanced_pre_event_sampling, 'guardrail', 'Source-balanced pre-event sampling manifest unavailable.')}",
    ]
    for row in source_balanced_pre_event_sources[:8]:
        report_lines.append(
            f"- Source coverage {row.get('source_stem', 'NA')}: selected={fmt(row.get('selected_cycles'), 0)}, near={fmt(row.get('near_pre'), 0)}, mid={fmt(row.get('mid_pre'), 0)}, far={fmt(row.get('far_pre'), 0)}, post={fmt(row.get('post_event'), 0)}, controls={fmt(row.get('controls'), 0)}, new={fmt(row.get('new_cycles'), 0)}"
        )

    report_lines += [
        "",
        "## Source-Balanced Pre-Event ROI Sequence Export",
        "",
        f"- ROI sequences/cycles/sources/failures: {first_summary(source_balanced_pre_event_sequences, 'n_roi_sequences', 0)} / {first_summary(source_balanced_pre_event_sequences, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_sequences, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_sequences, 'n_failed', 0)}",
        f"- Future8/future16 sequence positives from event proximity: {first_summary(source_balanced_pre_event_sequences, 'future8_positive_sequences', 0)} / {first_summary(source_balanced_pre_event_sequences, 'future16_positive_sequences', 0)}",
        f"- Guardrail: {first_summary(source_balanced_pre_event_sequences, 'guardrail', 'Source-balanced pre-event ROI sequence export unavailable.')}",
        "",
        "## Source-Balanced Pre-Event Sequence Audit",
        "",
        f"- Feature rows/cycles/sources/failures: {first_summary(source_balanced_pre_event_sequence_audit, 'n_feature_rows', 0)} / {first_summary(source_balanced_pre_event_sequence_audit, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_sequence_audit, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_sequence_audit, 'n_failures', 0)}",
        f"- Bin counts: {source_balanced_pre_event_audit_bins}",
        f"- Target positives: {source_balanced_pre_event_audit_targets}",
        f"- Near-pre-event spatial readout: leave-cycle AUC {fmt(source_balanced_pre_event_near_cycle.get('roc_auc'))}/AP {fmt(source_balanced_pre_event_near_cycle.get('average_precision'))}; leave-source AUC {fmt(source_balanced_pre_event_near_source.get('roc_auc'))}/AP {fmt(source_balanced_pre_event_near_source.get('average_precision'))}",
        f"- Clean pre16 vs post/control all-video leave-cycle readout: AUC {fmt(source_balanced_pre_event_clean_cycle.get('roc_auc'))}/AP {fmt(source_balanced_pre_event_clean_cycle.get('average_precision'))}",
        f"- Any-pre vs post/control all-video leave-source guardrail: AUC {fmt(source_balanced_pre_event_any_source.get('roc_auc'))}/AP {fmt(source_balanced_pre_event_any_source.get('average_precision'))}",
        f"- Top scalar test: {source_balanced_pre_event_scalar_best.get('target', 'NA')} {source_balanced_pre_event_scalar_best.get('feature', 'NA')} / {source_balanced_pre_event_scalar_best.get('transform', 'NA')} AUC {fmt(source_balanced_pre_event_scalar_best.get('oriented_auc'))}, AP {fmt(source_balanced_pre_event_scalar_best.get('average_precision'))}, p={fmt(source_balanced_pre_event_scalar_best.get('mannwhitney_p'))}",
        f"- Guardrail: {first_summary(source_balanced_pre_event_sequence_audit, 'guardrail', 'Source-balanced pre-event sequence audit unavailable.')}",
    ]
    for row in source_balanced_pre_event_audit_models[:8]:
        report_lines.append(
            f"- Event-relative model {row.get('target', 'NA')} {row.get('group_col', 'NA')} {row.get('feature_set', 'NA')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, n={fmt(row.get('n_eval'), 0)}"
        )

    report_lines += [
        "",
        "## Source-Balanced Pre-Event Rollout, Mask/Front, and Event-Relative Readout",
        "",
        f"- Rollout audit rows/cycles/sources: {first_summary(source_balanced_pre_event_rollout, 'n_roi_sequences', 0)} / {first_summary(source_balanced_pre_event_rollout, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_rollout, 'n_sources', 0)}",
        f"- Top pre-event rollout future-label row: {source_balanced_pre_event_rollout_best.get('target', 'NA')} {source_balanced_pre_event_rollout_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_rollout_best.get('oriented_auc'))}, AP {fmt(source_balanced_pre_event_rollout_best.get('average_precision'))}, source eta2 {fmt(source_balanced_pre_event_rollout_best.get('source_eta2'))}",
        f"- Masked held-out rollout benchmark rows/cycles/sources/failures: {first_summary(source_balanced_pre_event_masked_rollout, 'n_ok', 0)} / {first_summary(source_balanced_pre_event_masked_rollout, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_masked_rollout, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_masked_rollout, 'n_failed', 0)}; best median particle baseline {source_balanced_pre_event_masked_rollout_best_method.get('method', 'NA')} median MSE {fmt(source_balanced_pre_event_masked_rollout_best_method.get('particle_mse_median'))}",
        f"- Top masked held-out event test: {source_balanced_pre_event_masked_rollout_best_test.get('target', 'NA')} {source_balanced_pre_event_masked_rollout_best_test.get('feature', 'NA')} {source_balanced_pre_event_masked_rollout_best_test.get('transform', 'NA')} AUC {fmt(source_balanced_pre_event_masked_rollout_best_test.get('oriented_auc'))}, AP {fmt(source_balanced_pre_event_masked_rollout_best_test.get('average_precision'))}, median positive-negative {fmt(source_balanced_pre_event_masked_rollout_best_test.get('median_positive_minus_negative'))}, p={fmt(source_balanced_pre_event_masked_rollout_best_test.get('mwu_p'))}",
        f"- Optical-flow transport rows/cycles/sources/failures: {first_summary(source_balanced_pre_event_optical_flow, 'n_ok', 0)} / {first_summary(source_balanced_pre_event_optical_flow, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_optical_flow, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_optical_flow, 'n_failed', 0)}; method {first_summary(source_balanced_pre_event_optical_flow, 'flow_method', 'NA')}, median particle/context flow ratio {fmt(source_balanced_pre_event_flow_method.get('median_particle_context_flow_ratio'))}",
        f"- Top optical-flow event test: {source_balanced_pre_event_flow_best_test.get('target', 'NA')} {source_balanced_pre_event_flow_best_test.get('feature', 'NA')} {source_balanced_pre_event_flow_best_test.get('transform', 'NA')} AUC {fmt(source_balanced_pre_event_flow_best_test.get('oriented_auc'))}, AP {fmt(source_balanced_pre_event_flow_best_test.get('average_precision'))}, median positive-negative {fmt(source_balanced_pre_event_flow_best_test.get('median_positive_minus_negative'))}, p={fmt(source_balanced_pre_event_flow_best_test.get('mwu_p'))}",
        f"- Best source-residual optical-flow row: {source_balanced_pre_event_flow_sr_best.get('target', 'NA')} {source_balanced_pre_event_flow_sr_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_flow_sr_best.get('oriented_auc'))}, AP {fmt(source_balanced_pre_event_flow_sr_best.get('average_precision'))}, p={fmt(source_balanced_pre_event_flow_sr_best.get('mwu_p'))}",
        f"- Top pre-event mask/front future-label row: {source_balanced_pre_event_mask_best.get('target', 'NA')} {source_balanced_pre_event_mask_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_mask_best.get('oriented_auc'))}, AP {fmt(source_balanced_pre_event_mask_best.get('average_precision'))}, source eta2 {fmt(source_balanced_pre_event_mask_best.get('source_eta2'))}",
        f"- Event-relative bins: {first_summary(source_balanced_pre_event_readout, 'event_relative_bin_counts', {})}",
        f"- Best source-residual clean-pre readout: {source_balanced_pre_event_clean_best.get('target', 'NA')} {source_balanced_pre_event_clean_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_clean_best.get('oriented_auc'))}, AP {fmt(source_balanced_pre_event_clean_best.get('average_precision'))}, p={fmt(source_balanced_pre_event_clean_best.get('mwu_p'))}",
        f"- Event-distance trajectory rows/sources/events/features: {first_summary(source_balanced_pre_event_trajectory, 'n_cycle_rows', 0)} / {first_summary(source_balanced_pre_event_trajectory, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_trajectory, 'n_event_cycles', 0)} / {first_summary(source_balanced_pre_event_trajectory, 'n_features_tested', 0)}; complete near/far event cycles {first_summary(source_balanced_pre_event_trajectory, 'full_event_cycles_with_near_and_far', 0)}",
        f"- Top trajectory physics trend: {source_balanced_pre_event_traj_best.get('transform', 'NA')} {source_balanced_pre_event_traj_best.get('feature', 'NA')} rho {fmt(source_balanced_pre_event_traj_best.get('spearman_rho_vs_event_proximity'))}, p={fmt(source_balanced_pre_event_traj_best.get('spearman_p'))}, source-stratified permutation p={fmt(source_balanced_pre_event_traj_best.get('within_source_permutation_abs_rho_p'))}, near-far median {fmt(source_balanced_pre_event_traj_best.get('near_minus_far_median'))}",
        f"- Trajectory guardrail: {first_summary(source_balanced_pre_event_trajectory, 'guardrail', 'Source-balanced pre-event trajectory audit unavailable.')}",
        f"- Directionality rows/cycles/sources/features: {first_summary(source_balanced_pre_event_directionality, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_directionality, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_directionality, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_directionality, 'n_features_tested', 0)}",
        f"- Top physics-facing pre-event clock feature: {source_balanced_pre_event_dir_clock_top.get('transform', 'NA')} {source_balanced_pre_event_dir_clock_top.get('feature', 'NA')} pre rho {fmt(source_balanced_pre_event_dir_clock_top.get('pre_clock_rho'))}, p={fmt(source_balanced_pre_event_dir_clock_top.get('pre_clock_p'))}, permutation p={fmt(source_balanced_pre_event_dir_clock_top.get('pre_clock_perm_p_abs'))}, post rho {fmt(source_balanced_pre_event_dir_clock_top.get('post_clock_rho'))}",
        f"- Top pre/post clock asymmetry: {source_balanced_pre_event_dir_asym_top.get('transform', 'NA')} {source_balanced_pre_event_dir_asym_top.get('feature', 'NA')} |pre|-|post| rho delta {fmt(source_balanced_pre_event_dir_asym_top.get('abs_pre_minus_abs_post_rho'))}",
        f"- Directionality near-pre vs far-pre: {source_balanced_pre_event_dir_near_raw.get('transform', 'NA')} {source_balanced_pre_event_dir_near_raw.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_dir_near_raw.get('oriented_auc'))}, AP {fmt(source_balanced_pre_event_dir_near_raw.get('average_precision'))}, pre rho {fmt(source_balanced_pre_event_dir_near_raw.get('pre_clock_rho'))}",
        f"- Directionality clean-pre source-residual readout: {source_balanced_pre_event_dir_clean_sr.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_dir_clean_sr.get('oriented_auc'))}, AP {fmt(source_balanced_pre_event_dir_clean_sr.get('average_precision'))}, pre rho {fmt(source_balanced_pre_event_dir_clean_sr.get('pre_clock_rho'))}, post rho {fmt(source_balanced_pre_event_dir_clean_sr.get('post_clock_rho'))}",
        f"- Directionality guardrail: {first_summary(source_balanced_pre_event_directionality, 'guardrail', 'Source-balanced pre-event directionality audit unavailable.')}",
        f"- Source-invariant rows/cycles/sources/features: {first_summary(source_balanced_pre_event_source_invariant, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_source_invariant, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_source_invariant, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_source_invariant, 'n_features_total', 0)}",
        f"- Source-invariant clean-pre best: {source_balanced_pre_event_si_clean.get('feature_family', 'NA')} {source_balanced_pre_event_si_clean.get('method', 'NA')} leave-source AUC {fmt(source_balanced_pre_event_si_clean.get('roc_auc'))}, AP {fmt(source_balanced_pre_event_si_clean.get('average_precision'))}, rho {fmt(source_balanced_pre_event_si_clean.get('spearman_rho'))}",
        f"- Source-invariant near-vs-far best: {source_balanced_pre_event_si_near_far.get('feature_family', 'NA')} {source_balanced_pre_event_si_near_far.get('method', 'NA')} leave-source AUC {fmt(source_balanced_pre_event_si_near_far.get('roc_auc'))}, AP {fmt(source_balanced_pre_event_si_near_far.get('average_precision'))}, rho {fmt(source_balanced_pre_event_si_near_far.get('spearman_rho'))}",
        f"- Source-invariant low-source-eta clean guardrail: {source_balanced_pre_event_si_low_clean.get('feature_family', 'NA')} {source_balanced_pre_event_si_low_clean.get('method', 'NA')} AUC {fmt(source_balanced_pre_event_si_low_clean.get('roc_auc'))}, max eta2 {fmt(source_balanced_pre_event_si_low_clean.get('max_raw_source_eta2'))}",
        f"- Source-invariant guardrail: {first_summary(source_balanced_pre_event_source_invariant, 'guardrail', 'Source-balanced pre-event source-invariant audit unavailable.')}",
        f"- Review packet candidates/cycles/sources/rendered strips: {first_summary(source_balanced_pre_event_review_packet, 'n_candidates', 0)} / {first_summary(source_balanced_pre_event_review_packet, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_review_packet, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_review_packet, 'n_rendered_frame_strips', 0)}",
        f"- Review packet reasons: {source_balanced_pre_event_review_reasons}",
        f"- Review packet contact sheet: {first_summary(source_balanced_pre_event_review_packet, 'contact_sheet', 'unavailable')}",
        f"- Top review candidate: rank {fmt(source_balanced_pre_event_review_top1.get('pre_event_review_rank'), 0)} {source_balanced_pre_event_review_top1.get('roi_id', 'NA')} cycle {fmt(source_balanced_pre_event_review_top1.get('cycleNo'), 0)} score {fmt(source_balanced_pre_event_review_top1.get('pre_event_review_score'))}, reason {source_balanced_pre_event_review_top1.get('review_reason', 'NA')}, source-invariant clean probability {fmt(source_balanced_pre_event_review_top1.get('si_clean_physics_prob'))}",
        f"- Review packet guardrail: {first_summary(source_balanced_pre_event_review_packet, 'guardrail', 'Source-balanced pre-event review packet unavailable.')}",
        f"- Matched-counterfactual rows/cycles/sources/permutations: {first_summary(source_balanced_pre_event_matched_counterfactual, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_matched_counterfactual, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_matched_counterfactual, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_matched_counterfactual, 'n_permutations', 0)}",
        f"- Matched-counterfactual pair counts: {source_balanced_pre_event_matched_pair_counts}; same-source fractions: {source_balanced_pre_event_matched_same_source}",
        f"- Top matched physics row: {source_balanced_pre_event_matched_best.get('comparison', 'NA')} {source_balanced_pre_event_matched_best.get('match_scheme', 'NA')} {source_balanced_pre_event_matched_best.get('feature', 'NA')} n={fmt(source_balanced_pre_event_matched_best.get('n_pairs'), 0)}, median near-control diff {fmt(source_balanced_pre_event_matched_best.get('median_treated_minus_control'))}, sign-flip p={fmt(source_balanced_pre_event_matched_best.get('signflip_mean_abs_p'))}",
        f"- Matched front/diffusion guardrails: q60 front-slope median diff {fmt(source_balanced_pre_event_matched_front.get('median_treated_minus_control'))}, p={fmt(source_balanced_pre_event_matched_front.get('signflip_mean_abs_p'))}; q70 apparent diffusion median diff {fmt(source_balanced_pre_event_matched_diffusion.get('median_treated_minus_control'))}, p={fmt(source_balanced_pre_event_matched_diffusion.get('signflip_mean_abs_p'))}",
        f"- Matched-counterfactual guardrail: {first_summary(source_balanced_pre_event_matched_counterfactual, 'guardrail', 'Source-balanced pre-event matched counterfactual audit unavailable.')}",
        f"- Same-source ladder rows/cycles/sources/permutations: {first_summary(source_balanced_pre_event_same_source_ladder, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_same_source_ladder, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_same_source_ladder, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_same_source_ladder, 'n_permutations', 0)}",
        f"- Same-source ladder coverage: {source_balanced_pre_event_ladder_counts}; pair counts: {source_balanced_pre_event_ladder_pair_counts}",
        f"- Top same-source paired row: {source_balanced_pre_event_ladder_best.get('comparison', 'NA')} {source_balanced_pre_event_ladder_best.get('feature', 'NA')} n={fmt(source_balanced_pre_event_ladder_best.get('n_pairs'), 0)}, median near-control diff {fmt(source_balanced_pre_event_ladder_best.get('median_treated_minus_control'))}, sign-flip p={fmt(source_balanced_pre_event_ladder_best.get('signflip_mean_abs_p'))}",
        f"- Same-source front/diffusion guardrails: q60 front-slope median diff {fmt(source_balanced_pre_event_ladder_front.get('median_treated_minus_control'))}, p={fmt(source_balanced_pre_event_ladder_front.get('signflip_mean_abs_p'))}; q70 apparent diffusion median diff {fmt(source_balanced_pre_event_ladder_diffusion.get('median_treated_minus_control'))}, p={fmt(source_balanced_pre_event_ladder_diffusion.get('signflip_mean_abs_p'))}; top continuous clock {source_balanced_pre_event_ladder_clock_top.get('feature', 'NA')} rho {fmt(source_balanced_pre_event_ladder_clock_top.get('within_source_residual_spearman_rho_vs_event_proximity'))}, p={fmt(source_balanced_pre_event_ladder_clock_top.get('spearman_p'))}",
        f"- Same-source ladder guardrail: {first_summary(source_balanced_pre_event_same_source_ladder, 'guardrail', 'Source-balanced pre-event same-source ladder audit unavailable.')}",
        f"- Source-lattice raw cycles/sources/HDF5 sources: {first_summary(pre_event_source_lattice, 'n_cycle_rows', 0)} / {first_summary(pre_event_source_lattice, 'n_sources', 0)} / {first_summary(pre_event_source_lattice, 'n_sources_with_h5', 0)}",
        f"- Source-lattice near-source counts: {pre_event_lattice_near_counts}; candidate far-control sources: {pre_event_lattice_far_controls}",
        f"- Source-lattice design recommendation: {'; '.join(pre_event_lattice_design[:3]) if pre_event_lattice_design else 'unavailable'}",
        f"- Source-lattice guardrail: {first_summary(pre_event_source_lattice, 'guardrail', 'Pre-event source-lattice coverage audit unavailable.')}",
        f"- Radial-kymograph ROI/cycles/sources/rendered: {first_summary(source_balanced_pre_event_radial_kymograph, 'n_roi', 0)} / {first_summary(source_balanced_pre_event_radial_kymograph, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_radial_kymograph, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_radial_kymograph, 'n_rendered_kymographs', 0)}",
        f"- Top radial-kymograph near-vs-far row: {source_balanced_pre_event_kymo_near_far_top.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_kymo_near_far_top.get('oriented_auc'))}, median diff {fmt(source_balanced_pre_event_kymo_near_far_top.get('median_positive_minus_negative'))}, p={fmt(source_balanced_pre_event_kymo_near_far_top.get('mwu_p'))}",
        f"- Top radial-kymograph clean-pre row: {source_balanced_pre_event_kymo_clean_top.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_kymo_clean_top.get('oriented_auc'))}, median diff {fmt(source_balanced_pre_event_kymo_clean_top.get('median_positive_minus_negative'))}, p={fmt(source_balanced_pre_event_kymo_clean_top.get('mwu_p'))}",
        f"- Top review-candidate kymograph: {source_balanced_pre_event_kymo_review_top.get('roi_id', 'NA')} rank {fmt(source_balanced_pre_event_kymo_review_top.get('pre_event_review_rank'), 0)} front radius2 slope {fmt(source_balanced_pre_event_kymo_review_top.get('front_radius2_slope_px2_per_norm_time'))}, front slope R2 {fmt(source_balanced_pre_event_kymo_review_top.get('front_radius_slope_r2'))}",
        f"- Radial-kymograph guardrail: {first_summary(source_balanced_pre_event_radial_kymograph, 'guardrail', 'Source-balanced pre-event radial kymograph audit unavailable.')}",
        f"- Echem/front coupling rows/cycles/sources/echem-features/targets: {first_summary(source_balanced_pre_event_echem_front, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_echem_front, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_echem_front, 'n_sources', 0)} / {fmt(first_summary(source_balanced_pre_event_echem_front, 'n_echem_features'), 0)} / {fmt(first_summary(source_balanced_pre_event_echem_front, 'n_targets'), 0)}",
        f"- Top raw echem/front event-bin row: {source_balanced_pre_event_echem_raw_top.get('comparison', 'NA')} {source_balanced_pre_event_echem_raw_top.get('feature', 'NA')} n={fmt(source_balanced_pre_event_echem_raw_top.get('n_treated'), 0)} vs {fmt(source_balanced_pre_event_echem_raw_top.get('n_control'), 0)}, median diff {fmt(source_balanced_pre_event_echem_raw_top.get('treated_minus_control_median'))}, AUC {fmt(source_balanced_pre_event_echem_raw_top.get('roc_auc_treated_high'))}, p={fmt(source_balanced_pre_event_echem_raw_top.get('mannwhitney_p'))}",
        f"- Top source+echem residual event-bin row: {source_balanced_pre_event_echem_resid_top.get('comparison', 'NA')} {source_balanced_pre_event_echem_resid_top.get('feature', 'NA')} n={fmt(source_balanced_pre_event_echem_resid_top.get('n_treated'), 0)} vs {fmt(source_balanced_pre_event_echem_resid_top.get('n_control'), 0)}, residual median diff {fmt(source_balanced_pre_event_echem_resid_top.get('treated_minus_control_median'))}, AUC {fmt(source_balanced_pre_event_echem_resid_top.get('roc_auc_treated_high'))}, p={fmt(source_balanced_pre_event_echem_resid_top.get('mannwhitney_p'))}",
        f"- Top echem/optical correlation: {source_balanced_pre_event_echem_corr_top.get('echem_feature', 'NA')} vs {source_balanced_pre_event_echem_corr_top.get('target', 'NA')} n={fmt(source_balanced_pre_event_echem_corr_top.get('n'), 0)}, rho {fmt(source_balanced_pre_event_echem_corr_top.get('spearman_rho'))}, p={fmt(source_balanced_pre_event_echem_corr_top.get('p_value'))}; strongest residual fit is {source_balanced_pre_event_echem_fit_top.get('target', 'NA')} variance explained {fmt(source_balanced_pre_event_echem_fit_top.get('variance_explained_by_model'))}",
        f"- Echem/front coupling guardrail: {first_summary(source_balanced_pre_event_echem_front, 'guardrail', 'Source-balanced pre-event echem/front coupling audit unavailable.')}",
        f"- Echem-matched residual rows/cycles/sources/match-features/outcomes: {first_summary(source_balanced_pre_event_echem_matched_residual, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_echem_matched_residual, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_echem_matched_residual, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_echem_matched_residual, 'n_match_features', 0)} / {first_summary(source_balanced_pre_event_echem_matched_residual, 'n_outcome_features', 0)}",
        f"- Echem-matched residual pair counts: {source_balanced_pre_event_echem_matched_resid_pair_counts}; same-source fractions: {source_balanced_pre_event_echem_matched_resid_same_source}",
        f"- Top echem-matched residual row: {source_balanced_pre_event_echem_matched_resid_best.get('comparison', 'NA')} {source_balanced_pre_event_echem_matched_resid_best.get('match_scheme', 'NA')} {source_balanced_pre_event_echem_matched_resid_best.get('base_feature', 'NA')} n={fmt(source_balanced_pre_event_echem_matched_resid_best.get('n_pairs'), 0)}, median near-control diff {fmt(source_balanced_pre_event_echem_matched_resid_best.get('median_treated_minus_control'))}, sign-flip p={fmt(source_balanced_pre_event_echem_matched_resid_best.get('signflip_mean_abs_p'))}",
        f"- Echem-matched residual guardrail: {first_summary(source_balanced_pre_event_echem_matched_residual, 'guardrail', 'Source-balanced pre-event echem-matched residual audit unavailable.')}",
        f"- Front-consensus rows/cycles/sources/matched-pairs/features: {first_summary(source_balanced_pre_event_front_consensus, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_front_consensus, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_front_consensus, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_front_consensus, 'n_matched_pairs', 0)} / {first_summary(source_balanced_pre_event_front_consensus, 'n_consensus_features', 0)}",
        f"- Top front-consensus event row: {source_balanced_pre_event_front_consensus_event_best.get('comparison', 'NA')} {source_balanced_pre_event_front_consensus_event_best.get('feature', 'NA')} n={fmt(source_balanced_pre_event_front_consensus_event_best.get('n_treated'), 0)} vs {fmt(source_balanced_pre_event_front_consensus_event_best.get('n_control'), 0)}, median diff {fmt(source_balanced_pre_event_front_consensus_event_best.get('treated_minus_control_median'))}, AUC {fmt(source_balanced_pre_event_front_consensus_event_best.get('roc_auc_treated_high'))}, p={fmt(source_balanced_pre_event_front_consensus_event_best.get('mannwhitney_p'))}",
        f"- Top front-consensus matched row: {source_balanced_pre_event_front_consensus_matched_best.get('comparison', 'NA')} {source_balanced_pre_event_front_consensus_matched_best.get('match_scheme', 'NA')} {source_balanced_pre_event_front_consensus_matched_best.get('feature', 'NA')} n={fmt(source_balanced_pre_event_front_consensus_matched_best.get('n_pairs'), 0)}, median near-control diff {fmt(source_balanced_pre_event_front_consensus_matched_best.get('median_treated_minus_control'))}, sign-flip p={fmt(source_balanced_pre_event_front_consensus_matched_best.get('signflip_mean_abs_p'))}",
        f"- Front-consensus guardrail: {first_summary(source_balanced_pre_event_front_consensus, 'guardrail', 'Source-balanced pre-event front consensus audit unavailable.')}",
        f"- Echem-matched far-control rows/cycles/sources/permutations: {first_summary(source_balanced_pre_event_echem_matched_far, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_echem_matched_far, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_echem_matched_far, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_echem_matched_far, 'n_permutations', 0)}",
        f"- Echem-matched far-control pair counts: {source_balanced_pre_event_echem_matched_far_pair_counts}; control source counts: {source_balanced_pre_event_echem_matched_far_control_sources}",
        f"- Top echem-matched far-control row: {source_balanced_pre_event_echem_matched_far_best.get('match_scheme', 'NA')} {source_balanced_pre_event_echem_matched_far_best.get('feature', 'NA')} n={fmt(source_balanced_pre_event_echem_matched_far_best.get('n_pairs'), 0)}, median near-far diff {fmt(source_balanced_pre_event_echem_matched_far_best.get('median_treated_minus_control'))}, positive fraction {fmt(source_balanced_pre_event_echem_matched_far_best.get('positive_difference_fraction'))}, sign-flip p={fmt(source_balanced_pre_event_echem_matched_far_best.get('signflip_mean_abs_p'))}",
        f"- Source-class+echem front-slope row: {source_balanced_pre_event_echem_matched_far_front.get('feature', 'NA')} n={fmt(source_balanced_pre_event_echem_matched_far_front.get('n_pairs'), 0)}, median near-far diff {fmt(source_balanced_pre_event_echem_matched_far_front.get('median_treated_minus_control'))}, p={fmt(source_balanced_pre_event_echem_matched_far_front.get('signflip_mean_abs_p'))}",
        f"- Echem-matched far-control guardrail: {first_summary(source_balanced_pre_event_echem_matched_far, 'guardrail', 'Source-balanced pre-event echem-matched far-control audit unavailable.')}",
        f"- Consensus review queue candidates/cycles/sources: {first_summary(source_balanced_pre_event_consensus_review, 'n_candidates', 0)} / {first_summary(source_balanced_pre_event_consensus_review, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_consensus_review, 'n_sources', 0)}; priority tiers: {source_balanced_pre_event_consensus_tiers}",
        f"- Top consensus review candidate: rank {fmt(source_balanced_pre_event_consensus_best.get('consensus_review_rank'), 0)} {source_balanced_pre_event_consensus_best.get('roi_id', 'NA')} {source_balanced_pre_event_consensus_best.get('event_relative_bin', 'NA')} score {fmt(source_balanced_pre_event_consensus_best.get('consensus_review_score'))}, matched support {fmt(source_balanced_pre_event_consensus_best.get('matched_positive_support_count'), 0)}, tier {source_balanced_pre_event_consensus_best.get('review_priority_tier', 'NA')}",
        f"- Consensus review queue guardrail: {first_summary(source_balanced_pre_event_consensus_review, 'guardrail', 'Source-balanced pre-event consensus review queue unavailable.')}",
        f"- Consensus visual packet queue/rendered/sources: {first_summary(source_balanced_pre_event_consensus_visual, 'n_queue_rows', 0)} / {first_summary(source_balanced_pre_event_consensus_visual, 'n_rendered', 0)} / {first_summary(source_balanced_pre_event_consensus_visual, 'n_sources_rendered', 0)}; contact sheet: {first_summary(source_balanced_pre_event_consensus_visual, 'contact_sheet', 'NA')}",
        f"- Top consensus visual candidate: rank {fmt(source_balanced_pre_event_visual_best.get('consensus_review_rank'), 0)} {source_balanced_pre_event_visual_best.get('roi_id', 'NA')} {source_balanced_pre_event_visual_best.get('event_relative_bin', 'NA')} score {fmt(source_balanced_pre_event_visual_best.get('consensus_review_score'))}, matched support {fmt(source_balanced_pre_event_visual_best.get('matched_positive_support_count'), 0)}",
        f"- Consensus visual packet guardrail: {first_summary(source_balanced_pre_event_consensus_visual, 'guardrail', 'Source-balanced pre-event consensus visual packet unavailable.')}",
        f"- Visual sanity audit scored {first_summary(source_balanced_pre_event_visual_sanity, 'n_ok', 0)} rendered candidates: flags {source_balanced_pre_event_visual_sanity_flags}, median score {fmt(first_summary(source_balanced_pre_event_visual_sanity, 'median_visual_sanity_score'))}; top reviewable candidate is rank {fmt(source_balanced_pre_event_visual_sanity_best.get('consensus_review_rank'), 0)} {source_balanced_pre_event_visual_sanity_best.get('roi_id', 'NA')} sanity {fmt(source_balanced_pre_event_visual_sanity_best.get('visual_sanity_score'))}.",
        f"- Visual sanity guardrail: {first_summary(source_balanced_pre_event_visual_sanity, 'guardrail', 'Source-balanced pre-event visual sanity audit unavailable.')}",
        f"- Visual QC modes scored/rendered: {first_summary(source_balanced_pre_event_visual_qc_modes, 'n_scored', 0)} / {first_summary(source_balanced_pre_event_consensus_visual, 'n_rendered', 0)}; tiers {source_balanced_pre_event_visual_qc_tiers}; mode counts {source_balanced_pre_event_visual_qc_mode_counts}",
        f"- Top visual QC mode candidate: rank {fmt(source_balanced_pre_event_visual_qc_best.get('consensus_review_rank'), 0)} {source_balanced_pre_event_visual_qc_best.get('roi_id', 'NA')} tier {source_balanced_pre_event_visual_qc_best.get('visual_qc_tier', 'NA')} score {fmt(source_balanced_pre_event_visual_qc_best.get('visual_review_score'))}, front score {fmt(source_balanced_pre_event_visual_qc_best.get('visual_front_plausibility_score'))}, artifact risk {fmt(source_balanced_pre_event_visual_qc_best.get('visual_artifact_risk_score'))}",
        f"- Visual QC modes guardrail: {first_summary(source_balanced_pre_event_visual_qc_modes, 'guardrail', 'Source-balanced pre-event visual QC modes unavailable.')}",
        f"- Phase-kinetics rows/sources/features: {first_summary(source_balanced_pre_event_phase_kinetics, 'n_ok', 0)} / {first_summary(source_balanced_pre_event_phase_kinetics, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_phase_kinetics, 'n_kinetic_features', 0)}; event bins {first_summary(source_balanced_pre_event_phase_kinetics, 'event_relative_bin_counts', {})}",
        f"- Top phase-kinetics event row: {source_balanced_pre_event_phase_kinetics_event_best.get('target', 'NA')} {source_balanced_pre_event_phase_kinetics_event_best.get('transform', 'NA')} {source_balanced_pre_event_phase_kinetics_event_best.get('feature', 'NA')} n={fmt(source_balanced_pre_event_phase_kinetics_event_best.get('n'), 0)}, median diff {fmt(source_balanced_pre_event_phase_kinetics_event_best.get('median_positive_minus_negative'))}, AUC {fmt(source_balanced_pre_event_phase_kinetics_event_best.get('oriented_auc'))}, p={fmt(source_balanced_pre_event_phase_kinetics_event_best.get('mwu_p'))}, source eta2 {fmt(source_balanced_pre_event_phase_kinetics_event_best.get('source_eta2_after_transform'))}",
        f"- Top phase-kinetics matched row: {source_balanced_pre_event_phase_kinetics_matched_best.get('match_scheme', 'NA')} near-vs-{source_balanced_pre_event_phase_kinetics_matched_best.get('control_bin', 'NA')} {source_balanced_pre_event_phase_kinetics_matched_best.get('feature', 'NA')} n={fmt(source_balanced_pre_event_phase_kinetics_matched_best.get('n_pairs'), 0)}, median diff {fmt(source_balanced_pre_event_phase_kinetics_matched_best.get('median_near_minus_control'))}, positive fraction {fmt(source_balanced_pre_event_phase_kinetics_matched_best.get('positive_fraction'))}, p={fmt(source_balanced_pre_event_phase_kinetics_matched_best.get('signflip_mean_abs_p'))}",
        f"- Phase-kinetics guardrail: {first_summary(source_balanced_pre_event_phase_kinetics, 'guardrail', 'Source-balanced pre-event phase kinetics audit unavailable.')}",
        f"- Front/kinetic concordance rows/sources/tier counts: {first_summary(source_balanced_pre_event_front_kinetic_concordance, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_front_kinetic_concordance, 'n_sources', 0)} / {source_balanced_pre_event_fk_concordance_tiers}",
        f"- Top front/kinetic concordance candidate: {source_balanced_pre_event_fk_concordance_best.get('roi_id', 'NA')} {source_balanced_pre_event_fk_concordance_best.get('event_relative_bin', 'NA')} score {fmt(source_balanced_pre_event_fk_concordance_best.get('front_kinetic_concordance_score'))}, kinetic {fmt(source_balanced_pre_event_fk_concordance_best.get('kinetic_evidence_score'))}, front {fmt(source_balanced_pre_event_fk_concordance_best.get('front_evidence_score'))}, tier {source_balanced_pre_event_fk_concordance_best.get('front_kinetic_tier', 'NA')}",
        f"- Best front/kinetic concordance event row: {source_balanced_pre_event_fk_concordance_test_best.get('target', 'NA')} {source_balanced_pre_event_fk_concordance_test_best.get('transform', 'NA')} {source_balanced_pre_event_fk_concordance_test_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_fk_concordance_test_best.get('oriented_auc'))}, p={fmt(source_balanced_pre_event_fk_concordance_test_best.get('mwu_p'))}",
        f"- Front/kinetic concordance guardrail: {first_summary(source_balanced_pre_event_front_kinetic_concordance, 'guardrail', 'Source-balanced pre-event front/kinetic concordance audit unavailable.')}",
        f"- Front/kinetic null rows/features/permutations/bootstrap: {first_summary(source_balanced_pre_event_front_kinetic_null, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_front_kinetic_null, 'n_features_tested', 0)} / {first_summary(source_balanced_pre_event_front_kinetic_null, 'n_permutations', 0)} / {first_summary(source_balanced_pre_event_front_kinetic_null, 'n_bootstrap', 0)}",
        f"- Top front/kinetic null row: {source_balanced_pre_event_fk_null_best.get('target', 'NA')} {source_balanced_pre_event_fk_null_best.get('transform', 'NA')} {source_balanced_pre_event_fk_null_best.get('feature', 'NA')} AUC {fmt(source_balanced_pre_event_fk_null_best.get('oriented_auc'))}, null median/p95 {fmt(source_balanced_pre_event_fk_null_best.get('null_auc_median'))}/{fmt(source_balanced_pre_event_fk_null_best.get('null_auc_p95'))}, coarse perm p={fmt(source_balanced_pre_event_fk_null_best.get('source_stratified_perm_p_auc_ge_observed'))}, eligible sources {fmt(source_balanced_pre_event_fk_null_best.get('n_eligible_sources'), 0)}",
        f"- Top front/kinetic proximity row: {source_balanced_pre_event_fk_null_prox_best.get('transform', 'NA')} {source_balanced_pre_event_fk_null_prox_best.get('feature', 'NA')} rho {fmt(source_balanced_pre_event_fk_null_prox_best.get('spearman_rho_vs_event_proximity'))}, coarse perm p={fmt(source_balanced_pre_event_fk_null_prox_best.get('source_stratified_perm_p_abs_rho'))}",
        f"- Front/kinetic null guardrail: {first_summary(source_balanced_pre_event_front_kinetic_null, 'guardrail', 'Source-balanced pre-event front/kinetic null audit unavailable.')}",
        f"- Manual-QC decision packet rows/sources/visual-assets: {first_summary(source_balanced_pre_event_manual_qc_decision, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_manual_qc_decision, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_manual_qc_decision, 'n_visual_asset_rows', 0)}; action counts {source_balanced_pre_event_manual_qc_actions}; top40 {source_balanced_pre_event_manual_qc_top40_actions}",
        f"- Top manual-QC decision candidate: rank {fmt(source_balanced_pre_event_manual_qc_best.get('manual_qc_rank'), 0)} {source_balanced_pre_event_manual_qc_best.get('roi_id', 'NA')} {source_balanced_pre_event_manual_qc_best.get('event_relative_bin', 'NA')} action {source_balanced_pre_event_manual_qc_best.get('manual_qc_action_tier', 'NA')} score {fmt(source_balanced_pre_event_manual_qc_best.get('manual_qc_decision_score'))}; question: {source_balanced_pre_event_manual_qc_best.get('review_question', 'NA')}",
        f"- Manual-QC visual packet rendered/action-tiers/event-bins: {first_summary(source_balanced_pre_event_manual_qc_visual, 'n_rendered', 0)} / {source_balanced_pre_event_manual_qc_visual_actions} / {source_balanced_pre_event_manual_qc_visual_bins}; contact sheet: {first_summary(source_balanced_pre_event_manual_qc_visual, 'contact_sheet', 'NA')}",
        f"- Top manual-QC visual candidate: rank {fmt(source_balanced_pre_event_manual_qc_visual_best.get('manual_qc_rank'), 0)} {source_balanced_pre_event_manual_qc_visual_best.get('roi_id', 'NA')} action {source_balanced_pre_event_manual_qc_visual_best.get('manual_qc_action_tier', 'NA')} status {source_balanced_pre_event_manual_qc_visual_best.get('manual_visual_render_status', 'NA')}",
        f"- Manual-QC visual guardrail: {first_summary(source_balanced_pre_event_manual_qc_visual, 'guardrail', 'Source-balanced pre-event manual-QC visual packet unavailable.')}",
        f"- Blinded manual-QC workbook rows/sources/action tiers: {first_summary(source_balanced_pre_event_manual_qc_blind, 'n_blinded_rows', 0)} / {first_summary(source_balanced_pre_event_manual_qc_blind, 'n_sources_hidden_key', 0)} / {source_balanced_pre_event_manual_qc_blind_actions}; event bins {source_balanced_pre_event_manual_qc_blind_bins}",
        f"- Blinded manual-QC workbook guardrail: {first_summary(source_balanced_pre_event_manual_qc_blind, 'guardrail', 'Source-balanced pre-event blinded manual-QC workbook unavailable.')}",
        f"- Manual-QC decision guardrail: {first_summary(source_balanced_pre_event_manual_qc_decision, 'guardrail', 'Source-balanced pre-event manual-QC decision packet unavailable.')}",
        f"- Multimodal predictor rows/cycles/sources: {first_summary(source_balanced_pre_event_multimodal, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_multimodal, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_multimodal, 'n_sources', 0)}; feature family sizes {first_summary(source_balanced_pre_event_multimodal, 'feature_family_sizes', {})}",
        f"- Top multimodal leave-source row: {source_balanced_pre_event_multimodal_best_row.get('target', 'NA')} {source_balanced_pre_event_multimodal_best_row.get('feature_family', 'NA')} {source_balanced_pre_event_multimodal_best_row.get('method', 'NA')} n={fmt(source_balanced_pre_event_multimodal_best_row.get('n_eval'), 0)}, AUC {fmt(source_balanced_pre_event_multimodal_best_row.get('roc_auc'))}, AP {fmt(source_balanced_pre_event_multimodal_best_row.get('average_precision'))}, source eta2 mean/max {fmt(source_balanced_pre_event_multimodal_best_row.get('mean_raw_source_eta2'))}/{fmt(source_balanced_pre_event_multimodal_best_row.get('max_raw_source_eta2'))}",
        f"- Best kinetics-family delta: {source_balanced_pre_event_multimodal_delta_best.get('target', 'NA')} {source_balanced_pre_event_multimodal_delta_best.get('method', 'NA')} {source_balanced_pre_event_multimodal_delta_best.get('richer_family', 'NA')} vs {source_balanced_pre_event_multimodal_delta_best.get('baseline_family', 'NA')} dAUC {fmt(source_balanced_pre_event_multimodal_delta_best.get('roc_auc_delta'))}, dAP {fmt(source_balanced_pre_event_multimodal_delta_best.get('average_precision_delta'))}",
        f"- Multimodal predictor guardrail: {first_summary(source_balanced_pre_event_multimodal, 'guardrail', 'Source-balanced pre-event multimodal predictor unavailable.')}",
        f"- Strict QC-gated front audit candidates/manual-review/diffusion-claim: {first_summary(source_balanced_pre_event_strict_qc_gated_front, 'n_candidates', 0)} / {first_summary(source_balanced_pre_event_strict_qc_gated_front, 'n_manual_front_review_candidates', 0)} / {first_summary(source_balanced_pre_event_strict_qc_gated_front, 'n_automatic_diffusion_claim_candidates', 0)}; gate pass counts {source_balanced_pre_event_strict_qc_gate_pass}",
        f"- Top strict QC-gated candidate: rank {fmt(source_balanced_pre_event_strict_qc_best.get('consensus_review_rank'), 0)} {source_balanced_pre_event_strict_qc_best.get('roi_id', 'NA')} {source_balanced_pre_event_strict_qc_best.get('event_relative_bin', 'NA')} score {fmt(source_balanced_pre_event_strict_qc_best.get('strict_qc_priority_score'))}, sanity {fmt(source_balanced_pre_event_strict_qc_best.get('visual_sanity_score'))}, visual QC {fmt(source_balanced_pre_event_strict_qc_best.get('visual_review_score'))}, front r2 {fmt(source_balanced_pre_event_strict_qc_best.get('front_trace_r2_mode'))}, manual gate {source_balanced_pre_event_strict_qc_best.get('manual_front_review_gate', 'NA')}, diffusion gate {source_balanced_pre_event_strict_qc_best.get('automatic_diffusion_claim_gate', 'NA')}",
        f"- Strict QC-gated front guardrail: {first_summary(source_balanced_pre_event_strict_qc_gated_front, 'guardrail', 'Source-balanced pre-event strict QC-gated front audit unavailable.')}",
        f"- Physics-mode taxonomy rows/cycles/sources/features/k: {first_summary(source_balanced_pre_event_physics_modes, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_physics_modes, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_physics_modes, 'n_sources', 0)} / {first_summary(source_balanced_pre_event_physics_modes, 'n_features_used', 0)} / {fmt(first_summary(source_balanced_pre_event_physics_modes, 'chosen_k'), 0)}",
        f"- Top physics mode: mode {source_balanced_pre_event_mode_best.get('mode', 'NA')} {source_balanced_pre_event_mode_best.get('mode_label', 'NA')} n={fmt(source_balanced_pre_event_mode_best.get('n_roi'), 0)}, near-pre fraction {fmt(source_balanced_pre_event_mode_best.get('near_pre_fraction'))}, post fraction {fmt(source_balanced_pre_event_mode_best.get('post_event_fraction'))}, loadings {source_balanced_pre_event_mode_best.get('top_loading_features', 'NA')}",
        f"- Best mode enrichment row: mode {source_balanced_pre_event_mode_enrich_best.get('mode', 'NA')} {source_balanced_pre_event_mode_enrich_best.get('label', 'NA')} fraction {fmt(source_balanced_pre_event_mode_enrich_best.get('mode_fraction'))} vs outside {fmt(source_balanced_pre_event_mode_enrich_best.get('outside_fraction'))}, p={fmt(source_balanced_pre_event_mode_enrich_best.get('fisher_p'))}",
        f"- Physics-mode guardrail: {first_summary(source_balanced_pre_event_physics_modes, 'guardrail', 'Source-balanced pre-event physics-mode taxonomy unavailable.')}",
        f"- Guardrail: {first_summary(source_balanced_pre_event_readout, 'guardrail', 'Source-balanced pre-event readout unavailable.')}",
    ]
    for row in source_balanced_pre_event_readout_best[:10]:
        report_lines.append(
            f"- Event-relative readout {row.get('target', 'NA')} {row.get('transform', 'NA')} {row.get('feature', 'NA')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, p={fmt(row.get('mwu_p'))}, eta2 {fmt(row.get('source_eta2_after_transform'))}"
        )
    for row in source_balanced_pre_event_traj_physics[:6]:
        report_lines.append(
            f"- Event-distance trajectory {row.get('transform', 'NA')} {row.get('feature', 'NA')}: rho {fmt(row.get('spearman_rho_vs_event_proximity'))}, permutation p={fmt(row.get('within_source_permutation_abs_rho_p'))}, near-far median {fmt(row.get('near_minus_far_median'))}"
        )

    report_lines += [
        "",
        "## Source-Balanced Pre-Event Observable Forecast",
        "",
        f"- Rows/cycles/sources: {first_summary(source_balanced_pre_event_observable_forecast, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_observable_forecast, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_observable_forecast, 'n_sources', 0)}; prefix fraction {fmt(first_summary(source_balanced_pre_event_observable_forecast, 'prefix_fraction'))}",
        f"- Feature set sizes: {first_summary(source_balanced_pre_event_observable_forecast, 'feature_set_sizes', {})}",
        f"- Best leave-source observable forecast: {observable_forecast_best.get('model', 'NA')} predicts {observable_forecast_best.get('target', 'NA')} with rho {fmt(observable_forecast_best.get('spearman_rho'))}, R2 {fmt(observable_forecast_best.get('r2'))}, MAE {fmt(observable_forecast_best.get('mae'))}",
        f"- Best prefix+echem incremental row: {observable_forecast_incremental_best.get('target', 'NA')} {observable_forecast_incremental_best.get('group_col', 'NA')} delta rho {fmt(observable_forecast_incremental_best.get('delta_spearman_prefix_plus_echem_minus_echem'))}, delta MAE {fmt(observable_forecast_incremental_best.get('delta_mae_prefix_plus_echem_minus_echem'))}",
    ]
    for row in observable_forecast_source_metrics[:8]:
        report_lines.append(
            f"- Observable forecast leave-source {row.get('target', 'NA')} {row.get('model', 'NA')}: rho {fmt(row.get('spearman_rho'))}, R2 {fmt(row.get('r2'))}, MAE {fmt(row.get('mae'))}"
        )
    for row in observable_forecast_event_diag[:8]:
        report_lines.append(
            f"- Observable event-relative diagnostic {row.get('target', 'NA')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, source-centered near-minus-other {fmt(row.get('source_centered_median_near_minus_other'))}"
        )
    for row in observable_forecast_incremental[:6]:
        report_lines.append(
            f"- Observable prefix+echem minus echem {row.get('target', 'NA')} {row.get('group_col', 'NA')}: delta rho {fmt(row.get('delta_spearman_prefix_plus_echem_minus_echem'))}, delta MAE {fmt(row.get('delta_mae_prefix_plus_echem_minus_echem'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_pre_event_observable_forecast, 'guardrail', 'Source-balanced pre-event observable forecast unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Pre-Event Optical-Flow Transport Audit",
        "",
        f"- Rows OK/failed/cycles/sources: {first_summary(source_balanced_pre_event_optical_flow_transport, 'n_ok', 0)} / {first_summary(source_balanced_pre_event_optical_flow_transport, 'n_failed', 0)} / {first_summary(source_balanced_pre_event_optical_flow_transport, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_optical_flow_transport, 'n_sources', 0)}",
        f"- Flow method/event bins: {first_summary(source_balanced_pre_event_optical_flow_transport, 'flow_method', 'NA')} / {first_summary(source_balanced_pre_event_optical_flow_transport, 'event_relative_bin_counts', {})}",
        f"- Median particle/context flow magnitude and ratio: {fmt(optical_flow_method_summary.get('median_particle_flow_mag'))} / {fmt(optical_flow_method_summary.get('median_context_flow_mag'))} / {fmt(optical_flow_method_summary.get('median_particle_context_flow_ratio'))}",
        f"- Top optical-flow transport row: {optical_flow_best.get('target', 'NA')} {optical_flow_best.get('feature', 'NA')} {optical_flow_best.get('transform', 'NA')} AUC {fmt(optical_flow_best.get('oriented_auc'))}, AP {fmt(optical_flow_best.get('average_precision'))}, MW p={fmt(optical_flow_best.get('mwu_p'))}, median positive-negative {fmt(optical_flow_best.get('median_positive_minus_negative'))}",
        f"- Best source-residual optical-flow row in top set: {optical_flow_source_resid_best.get('target', 'NA')} {optical_flow_source_resid_best.get('feature', 'NA')} AUC {fmt(optical_flow_source_resid_best.get('oriented_auc'))}, AP {fmt(optical_flow_source_resid_best.get('average_precision'))}, MW p={fmt(optical_flow_source_resid_best.get('mwu_p'))}",
    ]
    for row in optical_flow_top_tests[:10]:
        report_lines.append(
            f"- Optical-flow event test {row.get('target', 'NA')} {row.get('feature', 'NA')} {row.get('transform', 'NA')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, p={fmt(row.get('mwu_p'))}, rho {fmt(row.get('spearman_rho'))}"
        )
    for row in optical_flow_source_resid_tests[:5]:
        report_lines.append(
            f"- Optical-flow source-residual test {row.get('target', 'NA')} {row.get('feature', 'NA')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, p={fmt(row.get('mwu_p'))}, rho {fmt(row.get('spearman_rho'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_pre_event_optical_flow_transport, 'guardrail', 'Source-balanced pre-event optical-flow transport audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Pre-Event Transport/Kinetic Fusion Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_balanced_pre_event_transport_kinetic_fusion, 'n_rows', 0)} / {first_summary(source_balanced_pre_event_transport_kinetic_fusion, 'n_cycles', 0)} / {first_summary(source_balanced_pre_event_transport_kinetic_fusion, 'n_sources', 0)}",
        f"- Feature set sizes: {first_summary(source_balanced_pre_event_transport_kinetic_fusion, 'feature_set_sizes', {})}",
        f"- Top fused event test: {transport_fusion_best.get('target', 'NA')} {transport_fusion_best.get('score', 'NA')} AUC {fmt(transport_fusion_best.get('oriented_auc'))}, AP {fmt(transport_fusion_best.get('average_precision'))}, MW p={fmt(transport_fusion_best.get('mwu_p'))}, source-stratified p={fmt(transport_fusion_best.get('source_stratified_permutation_p'))}",
        f"- Near-vs-post/control fused row: {transport_fusion_near_post.get('score', 'NA')} AUC {fmt(transport_fusion_near_post.get('oriented_auc'))}, AP {fmt(transport_fusion_near_post.get('average_precision'))}, source-stratified p={fmt(transport_fusion_near_post.get('source_stratified_permutation_p'))}",
        f"- Top leave-source model: {transport_fusion_top_model.get('target', 'NA')} {transport_fusion_top_model.get('feature_set', 'NA')} AUC {fmt(transport_fusion_top_model.get('roc_auc'))}, AP {fmt(transport_fusion_top_model.get('average_precision'))}, n={fmt(transport_fusion_top_model.get('n_eval'), 0)}, sources={fmt(transport_fusion_top_model.get('n_sources_eval'), 0)}",
        f"- Top fusion review candidate: {transport_fusion_top_candidate.get('roi_id', 'NA')} {transport_fusion_top_candidate.get('event_relative_bin', 'NA')} score {fmt(transport_fusion_top_candidate.get('fusion_review_priority_score'))}, action {transport_fusion_top_candidate.get('manual_qc_action_tier', 'NA')}",
    ]
    for row in transport_fusion_tests[:8]:
        report_lines.append(
            f"- Fusion event test {row.get('target', 'NA')} {row.get('score', 'NA')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, MW p={fmt(row.get('mwu_p'))}, source-stratified p={fmt(row.get('source_stratified_permutation_p'))}"
        )
    for row in transport_fusion_candidates[:6]:
        report_lines.append(
            f"- Fusion candidate {row.get('roi_id', 'NA')} {row.get('event_relative_bin', 'NA')}: priority {fmt(row.get('fusion_review_priority_score'))}, source-guarded {fmt(row.get('source_guarded_transport_kinetic_front_score'))}, action {row.get('manual_qc_action_tier', 'NA')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_pre_event_transport_kinetic_fusion, 'guardrail', 'Source-balanced pre-event transport/kinetic fusion audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Transport Mechanism Dossier",
        "",
        f"- Rows/cycles/sources: {first_summary(source_balanced_transport_mechanism, 'n_rows', 0)} / {first_summary(source_balanced_transport_mechanism, 'n_cycles', 0)} / {first_summary(source_balanced_transport_mechanism, 'n_sources', 0)}",
        f"- Immediate-review rows / automatic diffusion-claim candidates: {fmt(first_summary(source_balanced_transport_mechanism, 'n_immediate_review'), 0)} / {fmt(first_summary(source_balanced_transport_mechanism, 'n_diffusion_claim_candidates'), 0)}",
        f"- Top40 event-bin counts: {first_summary(source_balanced_transport_mechanism, 'top40_event_bin_counts', {})}",
        f"- Top mechanism candidate: {transport_mechanism_top.get('roi_id', 'NA')} {transport_mechanism_top.get('event_relative_bin', 'NA')} score {fmt(transport_mechanism_top.get('transport_mechanism_score'))}, tier {transport_mechanism_top.get('transport_review_tier', 'NA')}, source {transport_mechanism_top.get('source_stem', 'NA')}, cycle {fmt(transport_mechanism_top.get('cycleNo'), 0)}",
        f"- Top candidate future labels and guardrail: future8={fmt(transport_mechanism_top.get('future_any_drop_within_8cycles'), 0)}, future16={fmt(transport_mechanism_top.get('future_any_drop_within_16cycles'), 0)}, diffusion blocked={transport_mechanism_top.get('diffusion_claim_blocked', 'NA')}, visual assets={transport_mechanism_top.get('has_visual_assets', 'NA')}",
    ]
    for row in transport_mechanism_tiers:
        report_lines.append(
            f"- Mechanism review tier {row.get('transport_review_tier', 'NA')}: n={fmt(row.get('n'), 0)}"
        )
    for row in transport_mechanism_sources[:6]:
        report_lines.append(
            f"- Mechanism source {row.get('source_stem', 'NA')}: max score {fmt(row.get('max_transport_mechanism_score'))}, near-pre rows {fmt(row.get('n_near_pre'), 0)}, priority rows {fmt(row.get('n_priority'), 0)}, median source-residual transport {fmt(row.get('median_transport_source_residual_score'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_transport_mechanism, 'guardrail', 'Source-balanced transport mechanism dossier unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Transport Mechanism Falsification Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_balanced_transport_mechanism_falsification, 'n_rows', 0)} / {first_summary(source_balanced_transport_mechanism_falsification, 'n_cycles', 0)} / {first_summary(source_balanced_transport_mechanism_falsification, 'n_sources', 0)}",
        f"- Near-pre rows / matched same-source pairs: {fmt(first_summary(source_balanced_transport_mechanism_falsification, 'n_near_pre'), 0)} / {fmt(first_summary(source_balanced_transport_mechanism_falsification, 'n_matched_pairs'), 0)}",
    ]
    for row in transport_mechanism_falsification_event:
        report_lines.append(
            f"- Event-local mechanism score test {row.get('target', 'NA')}: AUC {fmt(row.get('auc'))}, AP {fmt(row.get('average_precision'))}, median near-control diff {fmt(row.get('median_diff_pos_minus_neg'))}, source-stratified p {fmt(row.get('source_stratified_permutation_p'))}, n pos/neg {fmt(row.get('n_pos'), 0)}/{fmt(row.get('n_neg'), 0)}"
        )
    for row in transport_mechanism_falsification_pair:
        report_lines.append(
            f"- Same-source matched mechanism score {row.get('pair_set', 'NA')}: median near-control delta {fmt(row.get('median_delta_near_minus_control'))}, positive-delta fraction {fmt(row.get('positive_delta_fraction'))}, sign-flip p {fmt(row.get('sign_flip_p'))}, pairs/sources {fmt(row.get('n_pairs'), 0)}/{fmt(row.get('n_sources'), 0)}"
        )
    report_lines.append(
        f"- Source-median mechanism contrast: median delta {fmt(transport_mechanism_falsification_source.get('median_source_delta_near_minus_non'))}, positive-source fraction {fmt(transport_mechanism_falsification_source.get('positive_source_fraction'))}, sign-flip p {fmt(transport_mechanism_falsification_source.get('sign_flip_p'))}, eligible sources {fmt(transport_mechanism_falsification_source.get('n_sources'), 0)}"
    )
    for row in transport_mechanism_falsification_topk:
        report_lines.append(
            f"- Top-{fmt(row.get('top_k'), 0)} enrichment: near-pre fraction {fmt(row.get('near_pre_fraction'))}, sources {fmt(row.get('n_sources'), 0)}, dominant source {row.get('dominant_source', 'NA')} fraction {fmt(row.get('dominant_source_fraction'))}, diffusion candidates {fmt(row.get('n_diffusion_claim_candidates'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_transport_mechanism_falsification, 'guardrail', 'Source-balanced transport mechanism falsification audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Heldout Event Rank Transfer Audit",
        "",
        f"- Input rows/sources/candidate features/folds: {first_summary(source_heldout_event_rank_transfer, 'n_input_rows', 0)} / {first_summary(source_heldout_event_rank_transfer, 'n_sources', 0)} / {first_summary(source_heldout_event_rank_transfer, 'n_candidate_features', 0)} / {first_summary(source_heldout_event_rank_transfer, 'n_folds', 0)}",
    ]
    for row in heldout_rank_transfer_transfer_tests:
        report_lines.append(
            f"- Transfer-learned heldout score {row.get('target', 'NA')}: AUC {fmt(row.get('auc'))}, AP {fmt(row.get('average_precision'))}, median near-control diff {fmt(row.get('median_diff_pos_minus_neg'))}, eligible sources {fmt(row.get('n_eligible_heldout_sources'), 0)}, source sign-flip p {fmt(row.get('source_auc_minus_half_sign_flip_p'))}"
        )
    for row in heldout_rank_transfer_tests:
        report_lines.append(
            f"- Heldout comparator {row.get('score', 'NA')} {row.get('target', 'NA')}: AUC {fmt(row.get('auc'))}, AP {fmt(row.get('average_precision'))}, source sign-flip p {fmt(row.get('source_auc_minus_half_sign_flip_p'))}"
        )
    for row in heldout_rank_transfer_topk[:8]:
        report_lines.append(
            f"- Heldout top-{fmt(row.get('k'), 0)} {row.get('score', 'NA')}: near-pre fraction {fmt(row.get('near_pre_fraction'))}, sources {fmt(row.get('n_sources'), 0)}, dominant source {row.get('dominant_source', 'NA')} fraction {fmt(row.get('max_source_fraction'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_heldout_event_rank_transfer, 'guardrail', 'Source-heldout event rank transfer audit unavailable.')}")

    report_lines += [
        "",
        "## Pre-Event Temporal Dose-Response Audit",
        "",
        f"- Status/input/pre-event rows/cycles/sources: {first_summary(pre_event_temporal_dose_response, 'overall_status', 'unavailable')} / {first_summary(pre_event_temporal_dose_response, 'n_input_rows', 0)} / {first_summary(pre_event_temporal_dose_response, 'n_pre_event_rows', 0)} / {first_summary(pre_event_temporal_dose_response, 'n_pre_event_cycles', 0)} / {first_summary(pre_event_temporal_dose_response, 'n_pre_event_sources', 0)}",
        f"- Distance bin counts: {first_summary(pre_event_temporal_dose_response, 'distance_bin_counts', {})}",
    ]
    for row in temporal_dose_key_tests:
        report_lines.append(
            f"- Temporal key {row.get('feature', 'NA')}: raw rho {fmt(row.get('raw_spearman_rho'))} (p {fmt(row.get('raw_permutation_p'))}), source-centered rho {fmt(row.get('source_centered_spearman_rho'))} (p {fmt(row.get('source_centered_permutation_p'))}), positive source-slope fraction {fmt(row.get('positive_source_slope_fraction'))}, sign-flip p {fmt(row.get('source_slope_sign_flip_p'))}"
        )
    for row in temporal_dose_slopes[:5]:
        report_lines.append(
            f"- Directional source-slope candidate {row.get('feature', 'NA')}: median slope {fmt(row.get('median_source_slope'))}, positive source-slope fraction {fmt(row.get('positive_source_slope_fraction'))}, source sign-flip p {fmt(row.get('source_slope_sign_flip_p'))}, source-centered rho {fmt(row.get('source_centered_spearman_rho'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(pre_event_temporal_dose_response, 'guardrail', 'Pre-event temporal dose-response audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Degradation Mode Audit",
        "",
        f"- Rows/cycles/sources/features: {first_summary(source_balanced_degradation_modes, 'n_rows', 0)} / {first_summary(source_balanced_degradation_modes, 'n_cycles', 0)} / {first_summary(source_balanced_degradation_modes, 'n_sources', 0)} / {first_summary(source_balanced_degradation_modes, 'n_features_used', 0)}",
        f"- Chosen k and source-mode transitions: k={fmt(first_summary(source_balanced_degradation_modes, 'chosen_k'), 0)}, transition count {fmt(first_summary(source_balanced_degradation_modes, 'source_mode_transition_count'), 0)}, change fraction {fmt(first_summary(source_balanced_degradation_modes, 'source_mode_change_fraction'))}",
        f"- Strongest event-neighborhood enrichment: mode {source_balanced_degmode_top.get('mode', 'NA')} {source_balanced_degmode_top.get('label', 'NA')} fraction {fmt(source_balanced_degmode_top.get('mode_fraction'))} vs outside {fmt(source_balanced_degmode_top.get('outside_fraction'))}, p={fmt(source_balanced_degmode_top.get('fisher_p'))}",
    ]
    for row in source_balanced_degmode_clusters[:6]:
        report_lines.append(
            f"- Degradation mode {row.get('mode', 'NA')} {row.get('mode_label', 'NA')}: n={fmt(row.get('n_roi'), 0)}, cycles={fmt(row.get('n_cycles'), 0)}, sources={fmt(row.get('n_sources'), 0)}, future16 fraction {fmt(row.get('future16_fraction'))}, past16 fraction {fmt(row.get('past16_fraction'))}, phases {row.get('phase_counts', {})}"
        )
    for row in source_balanced_degmode_enrichment[:8]:
        report_lines.append(
            f"- Mode enrichment {row.get('mode', 'NA')} {row.get('label', 'NA')} ({row.get('enrichment_type', 'NA')}): fraction {fmt(row.get('mode_fraction'))} vs outside {fmt(row.get('outside_fraction'))}, p={fmt(row.get('fisher_p'))}, n={fmt(row.get('mode_total'), 0)}"
        )
    for row in source_balanced_degmode_representatives[:6]:
        report_lines.append(
            f"- Mode representative {row.get('mode', 'NA')}: {row.get('roi_id', 'NA')} cycle {fmt(row.get('cycleNo'), 0)}, phase {row.get('event_neighborhood_phase', 'NA')}, distance {fmt(row.get('mode_distance'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_degradation_modes, 'guardrail', 'Source-balanced degradation mode audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Residual-Physics Coupling Audit",
        "",
        f"- Rows/residual features/physics proxies/sources: {first_summary(source_balanced_residual_physics_coupling, 'n_rows', 0)} / {first_summary(source_balanced_residual_physics_coupling, 'n_residual_features', 0)} / {first_summary(source_balanced_residual_physics_coupling, 'n_physics_features', 0)} / {first_summary(source_balanced_residual_physics_coupling, 'n_sources', 0)}",
        f"- Top source-residual primary-candidate coupling: {source_balanced_resphys_top_primary.get('residual_feature', 'NA')} vs {source_balanced_resphys_top_primary.get('physics_feature', 'NA')} rho {fmt(source_balanced_resphys_top_primary.get('spearman_rho'))}, residual AUC {fmt(source_balanced_resphys_top_primary.get('residual_future16_auc'))}, physics AUC {fmt(source_balanced_resphys_top_primary.get('physics_future16_auc'))}",
        f"- Top source-residual target-aligned coupling: {source_balanced_resphys_top_aligned.get('residual_feature', 'NA')} vs {source_balanced_resphys_top_aligned.get('physics_feature', 'NA')} rho {fmt(source_balanced_resphys_top_aligned.get('spearman_rho'))}, residual AUC {fmt(source_balanced_resphys_top_aligned.get('residual_future16_auc'))}, physics AUC {fmt(source_balanced_resphys_top_aligned.get('physics_future16_auc'))}",
    ]
    for row in source_balanced_resphys_by_transform[:6]:
        report_lines.append(
            f"- Residual-physics coupling {row.get('transform')}: {row.get('residual_feature', 'NA')} vs {row.get('physics_feature', 'NA')} rho {fmt(row.get('spearman_rho'))}, residual AUC {fmt(row.get('residual_future16_auc'))}, physics AUC {fmt(row.get('physics_future16_auc'))}"
        )
    for row in source_balanced_resphys_dict_recon[:6]:
        report_lines.append(
            f"- Dictionary recon source-residual coupling to {row.get('physics_feature', 'NA')}: rho {fmt(row.get('spearman_rho'))}, p={fmt(row.get('spearman_p'))}, physics AUC {fmt(row.get('physics_future16_auc'))}, target aligned={row.get('target_aligned')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_residual_physics_coupling, 'guardrail', 'Source-balanced residual-physics coupling audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Residual Candidate Review Packet",
        "",
        f"- Candidates/sources/cycles: {first_summary(source_balanced_residual_candidate_review, 'n_candidates', 0)} / {first_summary(source_balanced_residual_candidate_review, 'n_sources', 0)} / {first_summary(source_balanced_residual_candidate_review, 'n_cycles', 0)}",
        f"- Review tier counts: {source_balanced_review_tiers}",
        f"- Top candidate: {source_balanced_review_top1.get('roi_id', 'NA')} score {fmt(source_balanced_review_top1.get('review_priority_score'))}, source {source_balanced_review_top1.get('source_stem', 'NA')}, cycle {fmt(source_balanced_review_top1.get('cycleNo'), 0)}, status {source_balanced_review_top1.get('manual_qc_status', 'NA')}",
    ]
    for row in source_balanced_review_top[:8]:
        report_lines.append(
            f"- Residual candidate review rank {row.get('review_rank')}: {row.get('roi_id', 'NA')} score {fmt(row.get('review_priority_score'))}, prob {fmt(row.get('source_heldout_future16_probability'))}, tier {row.get('review_tier', 'NA')}, status {row.get('manual_qc_status', 'NA')}"
        )
    for row in source_balanced_review_sources[:6]:
        report_lines.append(
            f"- Residual candidate source {row.get('source_stem', 'NA')}: n={fmt(row.get('n_candidates'), 0)}, max score {fmt(row.get('max_review_priority_score'))}, future16 rate {fmt(row.get('future16_positive_rate'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_residual_candidate_review, 'guardrail', 'Source-balanced residual candidate review packet unavailable.')}")

    report_lines += [
        "",
        "## Balanced Future-Drop Direct-Video ROI Audit",
        "",
        f"- Reconstructed cycles/candidates/ROI rows: {first_summary(balanced_future_reconstruction, 'n_cycles_sampled', 0)} / {first_summary(balanced_future_reconstruction, 'n_reconstructed_candidates', 0)} / {first_summary(balanced_future_reconstruction, 'n_roi_rows', 0)}",
        f"- Exported ROI sequences: {first_summary(balanced_future_sequences, 'n_roi_sequences', 0)}",
        f"- Physics ROI/cycles/features: {first_summary(balanced_future_physics, 'n_roi', 0)} / {first_summary(balanced_future_physics, 'n_cycles', 0)} / {first_summary(balanced_future_physics, 'n_features', 0)}",
        f"- Label counts: {first_summary(balanced_future_physics, 'label_counts', [])}",
        f"- Masked rollout best-method counts: {balanced_future_rollout_best_counts}",
        f"- Mask stability ROI/frames: {first_summary(balanced_future_mask, 'n_roi', 0)} / {first_summary(balanced_future_mask, 'n_frames_total', 0)}; overall {balanced_future_mask_overall}",
        f"- Mask stability role summary: {balanced_future_mask_roles}",
        f"- Context-only / physics-only / physics+acquisition best AUC: {fmt(balanced_future_best_acq_context.get('pooled_oof_roc_auc'))} / {fmt(balanced_future_best_oof.get('pooled_oof_roc_auc'))} / {fmt(balanced_future_best_context_physics.get('pooled_oof_roc_auc'))}",
        f"- Design-context best AUC: {fmt(balanced_future_best_design_context.get('pooled_oof_roc_auc'))} (includes selection metadata such as balanced rank/subrole and warning score)",
    ]
    for row in balanced_future_oof:
        report_lines.append(
            f"- Balanced future OOF {row.get('model')}: AUC {fmt(row.get('pooled_oof_roc_auc'))}, AP {fmt(row.get('pooled_oof_average_precision'))}, scored {fmt(row.get('n_scored'), 0)} rows ({fmt(row.get('n_positive_scored'), 0)}/{fmt(row.get('n_negative_scored'), 0)} pos/neg)"
        )
    report_lines.append(
        f"- Balanced future permutation null {balanced_future_null.get('model', 'NA')}: observed AUC {fmt(balanced_future_null.get('observed_auc'))}, null mean {fmt(balanced_future_null.get('null_auc_mean'))}, p95 {fmt(balanced_future_null.get('null_auc_p95'))}, empirical p={fmt(balanced_future_null.get('empirical_p_ge_observed'))}"
    )
    for row in balanced_future_roi_tests[:8]:
        report_lines.append(
            f"- Balanced future ROI feature {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, oriented AUC {fmt(row.get('oriented_auc'))}, permutation p={fmt(row.get('permutation_p_abs_median_diff'))}, n={fmt(row.get('n_positive'), 0)}/{fmt(row.get('n_negative'), 0)}"
        )
    for row in balanced_future_cycle_tests[:6]:
        report_lines.append(
            f"- Balanced future cycle feature {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, oriented AUC {fmt(row.get('oriented_auc'))}, permutation p={fmt(row.get('permutation_p_abs_median_diff'))}, n={fmt(row.get('n_positive'), 0)}/{fmt(row.get('n_negative'), 0)}"
        )
    for row in balanced_future_mask_tests[:6]:
        report_lines.append(
            f"- Balanced future mask-stability {row.get('feature')}: future8 positive-negative {fmt(row.get('median_diff_event_minus_control'))}, MW p={fmt(row.get('p_value'))}, n={fmt(row.get('n_event'), 0)}/{fmt(row.get('n_control'), 0)}"
        )
    for row in balanced_future_context_oof:
        report_lines.append(
            f"- Balanced future context OOF {row.get('feature_set')} {row.get('model')}: AUC {fmt(row.get('pooled_oof_roc_auc'))}, AP {fmt(row.get('pooled_oof_average_precision'))}"
        )
    for row in masked_video_metrics:
        report_lines.append(
            f"- Masked video embedding OOF {row.get('target')}: AUC {fmt(row.get('pooled_oof_roc_auc'))}, AP {fmt(row.get('pooled_oof_average_precision'))}, scored {fmt(row.get('n_scored'), 0)} rows ({fmt(row.get('n_positive_scored'), 0)}/{fmt(row.get('n_negative_scored'), 0)} pos/neg)"
        )
    report_lines.append(
        f"- Masked video embedding future8 permutation null: observed AUC {fmt(masked_video_null.get('observed_auc'))}, null mean {fmt(masked_video_null.get('null_auc_mean'))}, p95 {fmt(masked_video_null.get('null_auc_p95'))}, empirical p={fmt(masked_video_null.get('empirical_p_ge_observed'))}"
    )
    for row in masked_video_feature_tests[:6]:
        report_lines.append(
            f"- Masked video embedding feature {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, oriented AUC {fmt(row.get('oriented_auc'))}, MW p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in masked_video_clusters[:5]:
        report_lines.append(
            f"- Masked video embedding cluster {fmt(row.get('video_embedding_cluster'), 0)}: n={fmt(row.get('n_roi'), 0)}, future8 positive fraction {fmt(row.get('future8_positive_fraction'))}, prototype {row.get('prototype_roi', 'NA')}"
        )
    report_lines += [
        "",
        "## Learned Video Residual Embedding Audit",
        "",
        f"- Rows/cycles: {first_summary(learned_video_residual_embedding, 'n_embedding_rows', 0)} / {first_summary(learned_video_residual_embedding, 'n_cycles', 0)}",
        f"- Cohorts: {first_summary(learned_video_residual_embedding, 'embedding_cohort_counts', {})}",
        f"- Training: {first_summary(learned_video_residual_embedding, 'training', {})}",
        f"- Feature set sizes: {first_summary(learned_video_residual_embedding, 'feature_set_sizes', {})}",
        f"- Future8 learned_all / PCA-video / handcrafted AUC: {fmt(learned_residual_future8.get('roc_auc'))} / {fmt(learned_residual_pca_future8.get('roc_auc'))} / {fmt(learned_residual_hand_future8.get('roc_auc'))}",
        f"- Future16 learned_all / handcrafted AUC: {fmt(learned_residual_future16.get('roc_auc'))} / {fmt(learned_residual_hand_future16.get('roc_auc'))}",
    ]
    for row in learned_residual_class[:8]:
        report_lines.append(
            f"- Learned-residual classification {row.get('target')} {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, p={fmt(row.get('empirical_p_ge_observed'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    for row in learned_residual_deltas[:8]:
        report_lines.append(
            f"- Learned-residual delta {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta rho {fmt(row.get('delta_spearman_rho'))}, delta R2 {fmt(row.get('delta_r2'))}"
        )
    for row in learned_residual_reg[:4]:
        report_lines.append(
            f"- Learned-residual regression {row.get('target')} {row.get('feature_set')}: R2 {fmt(row.get('r2'))}, rho {fmt(row.get('spearman_rho'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(learned_video_residual_embedding, 'guardrail', 'Learned video residual embedding unavailable.')}")

    report_lines += [
        "",
        "## Residual Dictionary Embedding Audit",
        "",
        f"- Rows/cycles: {first_summary(residual_dictionary_embedding, 'n_embedding_rows', 0)} / {first_summary(residual_dictionary_embedding, 'n_cycles', 0)}",
        f"- Cohorts: {first_summary(residual_dictionary_embedding, 'embedding_cohort_counts', {})}",
        f"- Dictionary: {first_summary(residual_dictionary_embedding, 'dictionary', {})}",
        f"- Feature set sizes: {first_summary(residual_dictionary_embedding, 'feature_set_sizes', {})}",
    ]
    for row in residual_dict_class[:6]:
        report_lines.append(
            f"- Residual-dictionary classification {row.get('target')} {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, p={fmt(row.get('empirical_p_ge_observed'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    for row in residual_dict_deltas[:6]:
        report_lines.append(
            f"- Residual-dictionary delta {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta rho {fmt(row.get('delta_spearman_rho'))}"
        )
    for row in residual_dict_reg[:4]:
        report_lines.append(
            f"- Residual-dictionary regression {row.get('target')} {row.get('feature_set')}: R2 {fmt(row.get('r2'))}, rho {fmt(row.get('spearman_rho'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(residual_dictionary_embedding, 'guardrail', 'Residual dictionary embedding unavailable.')}")

    report_lines += [
        "",
        "## Echem Residual-Dictionary Fusion Audit",
        "",
        f"- Rows/cycles: {first_summary(echem_residual_dictionary_fusion, 'n_rows', 0)} / {first_summary(echem_residual_dictionary_fusion, 'n_cycles', 0)}",
        f"- Feature set sizes: {first_summary(echem_residual_dictionary_fusion, 'feature_set_sizes', {})}",
    ]
    for row in echem_resdict_class[:8]:
        report_lines.append(
            f"- Echem-resdict classification {row.get('target')} {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, p={fmt(row.get('empirical_p_ge_observed'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    for row in echem_resdict_deltas[:8]:
        report_lines.append(
            f"- Echem-resdict delta {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta rho {fmt(row.get('delta_spearman_rho'))}, delta R2 {fmt(row.get('delta_r2'))}"
        )
    for row in echem_resdict_reg[:4]:
        report_lines.append(
            f"- Echem-resdict regression {row.get('target')} {row.get('feature_set')}: R2 {fmt(row.get('r2'))}, rho {fmt(row.get('spearman_rho'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(echem_residual_dictionary_fusion, 'guardrail', 'Echem residual dictionary fusion unavailable.')}")

    report_lines += [
        "",
        "## Echem-Conditioned Residual Dictionary",
        "",
        f"- Rows/cycles/sources: {first_summary(echem_conditioned_residual_dictionary, 'n_rows', 0)} / {first_summary(echem_conditioned_residual_dictionary, 'n_cycles', 0)} / {first_summary(echem_conditioned_residual_dictionary, 'n_sources', 0)}",
        f"- Residual/context features: {first_summary(echem_conditioned_residual_dictionary, 'n_residual_features', 0)} / {first_summary(echem_conditioned_residual_dictionary, 'n_conditioning_features', 0)}",
        f"- Feature set sizes: {first_summary(echem_conditioned_residual_dictionary, 'feature_set_sizes', {})}",
        f"- Future16 conditioned residual dictionary leave-cycle/leave-source AUC: {fmt(echem_cond_resdict_lc16.get('roc_auc'))} / {fmt(echem_cond_resdict_ls16.get('roc_auc'))}",
        f"- Future16 conditioned residual+echem leave-cycle/leave-source AUC: {fmt(echem_cond_resdict_lc16_plus.get('roc_auc'))} / {fmt(echem_cond_resdict_ls16_plus.get('roc_auc'))}",
    ]
    for row in echem_cond_resdict_metrics[:10]:
        report_lines.append(
            f"- Echem-conditioned resdict {row.get('split')} {row.get('target')} {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, p={fmt(row.get('empirical_p_ge_observed'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    for row in echem_cond_resdict_deltas[:8]:
        report_lines.append(
            f"- Echem-conditioned resdict delta {row.get('split')} {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta AP {fmt(row.get('delta_average_precision'))}, delta rho {fmt(row.get('delta_spearman_rho'))}"
        )
    for row in echem_cond_resdict_context[:6]:
        report_lines.append(
            f"- Context fit {row.get('split')} {row.get('feature')}: R2 {fmt(row.get('r2'))}, rho {fmt(row.get('spearman_rho'))}, residual std {fmt(row.get('residual_std'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(echem_conditioned_residual_dictionary, 'guardrail', 'Echem-conditioned residual dictionary unavailable.')}")

    report_lines += [
        "",
        "## Conditioned Residual Physics Atlas",
        "",
        f"- Rows/cycles/sources: {first_summary(conditioned_residual_physics_atlas, 'n_rows', 0)} / {first_summary(conditioned_residual_physics_atlas, 'n_cycles', 0)} / {first_summary(conditioned_residual_physics_atlas, 'n_sources', 0)}",
        f"- Physics descriptor columns screened: {first_summary(conditioned_residual_physics_atlas, 'n_physics_features', 0)}",
        f"- Conditioned residual modes per split: {first_summary(conditioned_residual_physics_atlas, 'n_conditioned_residual_modes_per_split', {})}",
    ]
    for row in cond_atlas_align[:10]:
        report_lines.append(
            f"- Source-centered atlas {row.get('split')} {row.get('residual_base')} vs {row.get('physics_feature')} ({row.get('physics_category')}): centered rho {fmt(row.get('source_centered_rho'))}, raw rho {fmt(row.get('spearman_rho'))}"
        )
    for row in cond_atlas_targets[:10]:
        report_lines.append(
            f"- Residual-mode target {row.get('split')} {row.get('target')} {row.get('residual_base')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, source eta2 {fmt(row.get('source_eta2'))}"
        )
    for row in cond_atlas_categories[:8]:
        report_lines.append(
            f"- Atlas category {row.get('split')} {row.get('physics_category')}: max centered |rho| {fmt(row.get('max_abs_source_centered_rho'))}, strong centered pairs {fmt(row.get('n_strong_centered'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(conditioned_residual_physics_atlas, 'guardrail', 'Conditioned residual physics atlas unavailable.')}")

    report_lines += [
        "",
        "## Acquisition-Residualized Video Physics Benchmark",
        "",
        f"- Rows/cycles: {first_summary(acquisition_residualized_video, 'n_rows', 0)} / {first_summary(acquisition_residualized_video, 'n_cycles', 0)}",
        f"- Feature group sizes: {first_summary(acquisition_residualized_video, 'feature_group_sizes', {})}",
        f"- Future8 context/raw all-video/residualized all-video AUC: {fmt(acq_resid_context8.get('roc_auc'))} / {fmt(acq_resid_raw_video8.get('roc_auc'))} / {fmt(acq_resid_video_only8.get('roc_auc'))}",
        f"- Future16 context/raw handcrafted/residualized all-video AUC: {fmt(acq_resid_context16.get('roc_auc'))} / {fmt(acq_resid_raw_hand16.get('roc_auc'))} / {fmt(acq_resid_video_only16.get('roc_auc'))}",
    ]
    for row in acq_resid_metrics[:8]:
        report_lines.append(
            f"- Acquisition-residualized metric {row.get('target')} {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, cycle-block p={fmt(row.get('empirical_p_ge_observed_cycle_block'))}, n={fmt(row.get('n_eval'), 0)}"
        )
    for row in acq_resid_deltas[:8]:
        report_lines.append(
            f"- Acquisition-residualized delta {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta AP {fmt(row.get('delta_average_precision'))}, delta rho {fmt(row.get('delta_spearman_rho'))}"
        )
    for row in acq_resid_tests[:6]:
        report_lines.append(
            f"- Context-residual feature {row.get('feature_group')} {row.get('feature')} vs {row.get('target')}: |rho|={fmt(row.get('spearman_abs_rho_vs_context_label_residual'))}, direction-free AUC {fmt(row.get('direction_free_auc_vs_label'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(acquisition_residualized_video, 'guardrail', 'Acquisition-residualized video benchmark unavailable.')}")

    report_lines += [
        "",
        "## Acquisition-Residualized Video/Echem Warning Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(acquisition_residualized_video_echem, 'n_rows', 0)} / {first_summary(acquisition_residualized_video_echem, 'n_cycles', 0)} / {first_summary(acquisition_residualized_video_echem, 'n_sources', 0)}",
        f"- Feature set sizes: {first_summary(acquisition_residualized_video_echem, 'feature_set_sizes', {})}",
        f"- Leave-cycle future16 residualized video+echem: AUC {fmt(acq_echem_cycle_future16.get('roc_auc'))}, AP {fmt(acq_echem_cycle_future16.get('average_precision'))}, p={fmt(acq_echem_cycle_future16.get('empirical_p_ge_observed'))}; acquisition-only AUC {fmt(acq_echem_cycle_acq_future16.get('roc_auc'))}",
        f"- Cycle-balanced leave-cycle future16 residualized video+echem: AUC {fmt(acq_echem_cycle_bal_future16.get('roc_auc'))}, AP {fmt(acq_echem_cycle_bal_future16.get('average_precision'))}, p={fmt(acq_echem_cycle_bal_future16.get('empirical_p_ge_observed'))}",
        f"- Leave-source future16 residualized video+echem: AUC {fmt(acq_echem_source_future16.get('roc_auc'))}, AP {fmt(acq_echem_source_future16.get('average_precision'))}, p={fmt(acq_echem_source_future16.get('empirical_p_ge_observed'))}; acquisition-only AUC {fmt(acq_echem_source_acq_future16.get('roc_auc'))}",
        f"- Source-cohort future16 cycle-balanced residualized video+echem: AUC {fmt(acq_echem_sourcecohort_future16.get('roc_auc'))}, AP {fmt(acq_echem_sourcecohort_future16.get('average_precision'))}, p={fmt(acq_echem_sourcecohort_future16.get('empirical_p_ge_observed'))}; acquisition-only AUC {fmt(acq_echem_sourcecohort_acq_future16.get('roc_auc'))}",
    ]
    for row in acq_echem_deltas[:8]:
        report_lines.append(
            f"- Acquisition-residualized video/echem delta {row.get('target')} {row.get('group_col')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta rho {fmt(row.get('delta_spearman_rho'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(acquisition_residualized_video_echem, 'guardrail', 'Acquisition-residualized video/echem warning unavailable.')}")

    report_lines += [
        "",
        "## Residualized Future8 Video-Physics Benchmark",
        "",
        f"- Rows/cycles/sources: {first_summary(residualized_future8_video_physics, 'n_rows', 0)} / {first_summary(residualized_future8_video_physics, 'n_cycles', 0)} / {first_summary(residualized_future8_video_physics, 'n_sources', 0)}",
        f"- Feature set sizes: {first_summary(residualized_future8_video_physics, 'feature_set_sizes', {})}",
        f"- Decision: optical physics {future8_decision.get('future8_video_physics_status', 'unavailable')}; fused video+echem incremental {future8_decision.get('future8_fused_video_echem_incremental_status', 'unavailable')}",
        f"- Strict source-cohort residualized optical physics: AUC {fmt(future8_strict_optical_cohort.get('roc_auc'))}, AP {fmt(future8_strict_optical_cohort.get('average_precision'))}, source-stratified p={fmt(future8_strict_optical_cohort.get('empirical_p_ge_observed_source_stratified'))}",
        f"- Strict leave-source residualized optical physics: AUC {fmt(future8_strict_optical_source.get('roc_auc'))}, AP {fmt(future8_strict_optical_source.get('average_precision'))}, source-stratified p={fmt(future8_strict_optical_source.get('empirical_p_ge_observed_source_stratified'))}",
        f"- Source-cohort residualized fused video+echem / echem-only / acquisition-only AUC: {fmt(future8_fused_cohort.get('roc_auc'))} / {fmt(future8_echem_cohort.get('roc_auc'))} / {fmt(future8_acq_cohort.get('roc_auc'))}",
        f"- Fused minus residualized echem source-cohort delta AUC: {fmt(future8_decision.get('fused_minus_echem_residualized_source_cohort_delta_auc'))}",
    ]
    for row in future8_key_deltas[:8]:
        report_lines.append(
            f"- Future8 delta {row.get('group_col')} {row.get('feature_set')} {row.get('mode')} vs {row.get('baseline')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta AP {fmt(row.get('delta_average_precision'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(residualized_future8_video_physics, 'guardrail', 'Residualized future8 video-physics benchmark unavailable.')}")

    report_lines += [
        "",
        "## Source-Domain Video/Echem Adaptation Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_domain_video_echem, 'n_rows', 0)} / {first_summary(source_domain_video_echem, 'n_cycles', 0)} / {first_summary(source_domain_video_echem, 'n_sources', 0)}",
        f"- Feature set sizes: {first_summary(source_domain_video_echem, 'feature_set_sizes', {})}",
        f"- Best leave-source method: {source_domain_best.get('feature_set', 'NA')} {source_domain_best.get('method', 'NA')} AUC {fmt(source_domain_best.get('roc_auc'))}, AP {fmt(source_domain_best.get('average_precision'))}, p={fmt(source_domain_best.get('empirical_p_ge_observed'))}",
        f"- Acquisition-only / source-centered video+echem / CORAL video+echem AUC: {fmt(source_domain_acq.get('roc_auc'))} / {fmt(source_domain_centered.get('roc_auc'))} / {fmt(source_domain_coral.get('roc_auc'))}",
    ]
    for row in source_domain_deltas[:8]:
        report_lines.append(
            f"- Source-domain delta {row.get('feature_set')} {row.get('method')}: delta AUC {fmt(row.get('delta_auc_vs_acquisition_raw'))}, delta AP {fmt(row.get('delta_ap_vs_acquisition_raw'))}, delta rho {fmt(row.get('delta_rho_vs_acquisition_raw'))}"
        )
    for row in source_domain_sources[:6]:
        report_lines.append(
            f"- Source {row.get('source_stem')}: rows/cycles {fmt(row.get('n_rows'), 0)}/{fmt(row.get('n_cycles'), 0)}, future16 fraction {fmt(row.get('future16_positive_fraction'))}, mean feature z-shift {fmt(row.get('mean_abs_feature_z_shift'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_domain_video_echem, 'guardrail', 'Source-domain video/echem adaptation unavailable.')}")

    report_lines += [
        "",
        "## Source-Balanced Video/Echem Transfer Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_balanced_video_echem, 'n_rows', 0)} / {first_summary(source_balanced_video_echem, 'n_cycles', 0)} / {first_summary(source_balanced_video_echem, 'n_sources', 0)}",
        f"- Feature set sizes: {first_summary(source_balanced_video_echem, 'feature_set_sizes', {})}",
        f"- Future16 source-rank weighted video+echem: AUC {fmt(source_balanced_vpe16.get('roc_auc'))}, AP {fmt(source_balanced_vpe16.get('average_precision'))}, p={fmt(source_balanced_vpe16.get('empirical_p_ge_observed'))}",
        f"- Future16 acquisition raw / echem source-rank / video+echem source-rank AUC: {fmt(source_balanced_acq16.get('roc_auc'))} / {fmt(source_balanced_echem16.get('roc_auc'))} / {fmt(source_balanced_vpe16.get('roc_auc'))}",
    ]
    for row in source_balanced_deltas[:8]:
        report_lines.append(
            f"- Source-balanced delta {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta rho {fmt(row.get('delta_spearman_rho'))}"
        )
    for row in source_balanced_sources[:6]:
        report_lines.append(
            f"- Source {row.get('source_stem')}: rows/cycles {fmt(row.get('n_rows'), 0)}/{fmt(row.get('n_cycles'), 0)}, future16 labeled/positive/negative {fmt(row.get('future_any_drop_within_16cycles_labeled'), 0)}/{fmt(row.get('future_any_drop_within_16cycles_positive'), 0)}/{fmt(row.get('future_any_drop_within_16cycles_negative'), 0)}, future16 fraction {fmt(row.get('future_any_drop_within_16cycles_positive_fraction'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_balanced_video_echem, 'guardrail', 'Source-balanced video/echem transfer audit unavailable.')}")


    report_lines += [
        "",
        "## Source-Invariant Video/Echem Transfer Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_invariant_video_echem, 'n_rows', 0)} / {first_summary(source_invariant_video_echem, 'n_cycles', 0)} / {first_summary(source_invariant_video_echem, 'n_sources', 0)}",
        f"- Methods: {first_summary(source_invariant_video_echem, 'methods', [])}",
        f"- Future16 acquisition raw / video+echem raw / best video+echem invariant AUC: {fmt(source_invariant_acq16.get('roc_auc'))} / {fmt(source_invariant_raw16.get('roc_auc'))} / {fmt(source_invariant_best_vpe16.get('roc_auc'))}",
        f"- Best video+echem method: {source_invariant_best_vpe16.get('method', 'NA')} AP {fmt(source_invariant_best_vpe16.get('average_precision'))}, p={fmt(source_invariant_best_vpe16.get('empirical_p_ge_observed'))}",
        f"- Best video-only future16 method: {source_invariant_best_video16.get('method', 'NA')} AUC {fmt(source_invariant_best_video16.get('roc_auc'))}, AP {fmt(source_invariant_best_video16.get('average_precision'))}, p={fmt(source_invariant_best_video16.get('empirical_p_ge_observed'))}",
        f"- Best video+echem future8 invariant method: {source_invariant_best_vpe8.get('method', 'NA')} AUC {fmt(source_invariant_best_vpe8.get('roc_auc'))}, AP {fmt(source_invariant_best_vpe8.get('average_precision'))}",
    ]
    for row in source_invariant_deltas[:8]:
        report_lines.append(
            f"- Source-invariant delta {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta AP {fmt(row.get('delta_average_precision'))}, delta rho {fmt(row.get('delta_spearman_rho'))}"
        )
    for row in source_invariant_sources[:6]:
        report_lines.append(
            f"- Source {row.get('source_stem')}: rows/cycles {fmt(row.get('n_rows'), 0)}/{fmt(row.get('n_cycles'), 0)}, future16 fraction {fmt(row.get('future16_positive_fraction'))}, mean video/echem z-shift {fmt(row.get('mean_abs_video_echem_z_shift'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_invariant_video_echem, 'guardrail', 'Source-invariant video/echem transfer audit unavailable.')}")


    report_lines += [
        "",
        "## Source-Invariant Physical Family Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_invariant_family, 'n_rows', 0)} / {first_summary(source_invariant_family, 'n_cycles', 0)} / {first_summary(source_invariant_family, 'n_sources', 0)}",
        f"- Feature family sizes: {first_summary(source_invariant_family, 'feature_family_sizes', {})}",
        f"- Best future16 family/method: {source_family_best16.get('feature_family', 'NA')} {source_family_best16.get('method', 'NA')} AUC {fmt(source_family_best16.get('roc_auc'))}, AP {fmt(source_family_best16.get('average_precision'))}, p={fmt(source_family_best16.get('empirical_p_ge_observed'))}",
        f"- Norm heterogeneity source_mean_resid_4: AUC {fmt(source_family_norm16.get('roc_auc'))}, AP {fmt(source_family_norm16.get('average_precision'))}, p={fmt(source_family_norm16.get('empirical_p_ge_observed'))}",
        f"- Particle-vs-context source_mean_resid_4: AUC {fmt(source_family_contrast16.get('roc_auc'))}, AP {fmt(source_family_contrast16.get('average_precision'))}, p={fmt(source_family_contrast16.get('empirical_p_ge_observed'))}",
        f"- Raw video embedding future16: AUC {fmt(source_family_embed16.get('roc_auc'))}, AP {fmt(source_family_embed16.get('average_precision'))}; best future8 family/method {source_family_best8.get('feature_family', 'NA')} {source_family_best8.get('method', 'NA')} AUC {fmt(source_family_best8.get('roc_auc'))}",
    ]
    for row in source_family_deltas[:8]:
        report_lines.append(
            f"- Family delta {row.get('target')} {row.get('comparison')}: delta AUC {fmt(row.get('delta_roc_auc'))}, delta AP {fmt(row.get('delta_average_precision'))}, delta rho {fmt(row.get('delta_spearman_rho'))}"
        )
    for row in source_family_confounds[:8]:
        report_lines.append(
            f"- Source-confounded feature {row.get('feature_family')} {row.get('feature')}: eta2 {fmt(row.get('source_eta2'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_invariant_family, 'guardrail', 'Source-invariant physical family audit unavailable.')}")

    report_lines += [
        "",
        "## Source-Invariant Interpretable Feature Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(source_invariant_interpretable, 'n_rows', 0)} / {first_summary(source_invariant_interpretable, 'n_cycles', 0)} / {first_summary(source_invariant_interpretable, 'n_sources', 0)}",
        f"- Feature family sizes: {first_summary(source_invariant_interpretable, 'feature_family_sizes', {})}",
        f"- Top univariate descriptor: {source_interpretable_top_uni.get('feature', 'NA')} ({source_interpretable_top_uni.get('feature_family', 'NA')}), orientation {source_interpretable_top_uni.get('orientation', 'NA')}, AUC {fmt(source_interpretable_top_uni.get('roc_auc_oriented'))}, AP {fmt(source_interpretable_top_uni.get('average_precision_oriented'))}, eta2 {fmt(source_interpretable_top_uni.get('source_eta2'))}",
        f"- Top single-feature transfer set: {source_interpretable_top_single.get('feature_set', 'NA')} {source_interpretable_top_single.get('method', 'NA')} AUC {fmt(source_interpretable_top_single.get('roc_auc'))}, AP {fmt(source_interpretable_top_single.get('average_precision'))}, p={fmt(source_interpretable_top_single.get('empirical_p_ge_observed'))}",
        f"- Top small-combo transfer set: {source_interpretable_top_combo.get('feature_set', 'NA')} {source_interpretable_top_combo.get('method', 'NA')} AUC {fmt(source_interpretable_top_combo.get('roc_auc'))}, AP {fmt(source_interpretable_top_combo.get('average_precision'))}, p={fmt(source_interpretable_top_combo.get('empirical_p_ge_observed'))}",
    ]
    for row in source_interpretable_univariate[:8]:
        report_lines.append(
            f"- Exact feature {row.get('feature_family')} {row.get('feature')}: oriented AUC {fmt(row.get('roc_auc_oriented'))}, direction {row.get('orientation')}, eta2 {fmt(row.get('source_eta2'))}, median pos-neg {fmt(row.get('median_positive_minus_negative'))}"
        )
    for row in source_interpretable_sets[:8]:
        report_lines.append(
            f"- Feature set {row.get('feature_set')} {row.get('method')}: n_features {fmt(row.get('n_features'), 0)}, AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, p={fmt(row.get('empirical_p_ge_observed'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(source_invariant_interpretable, 'guardrail', 'Source-invariant interpretable feature audit unavailable.')}")

    report_lines += [
        "",
        "## Exact Feature Mechanism Consistency Audit",
        "",
        f"- Rows/cycles/sources: {first_summary(exact_feature_mechanism, 'n_rows', 0)} / {first_summary(exact_feature_mechanism, 'n_cycles', 0)} / {first_summary(exact_feature_mechanism, 'n_sources', 0)}",
        f"- Exact optical-loss composite future16 metric: AUC {fmt(exact_mech_loss_metric.get('oriented_auc'))}, AP {fmt(exact_mech_loss_metric.get('average_precision'))}, median positive-negative {fmt(exact_mech_loss_metric.get('median_positive_minus_negative'))}, source eta2 {fmt(exact_mech_loss_metric.get('source_eta2'))}",
        f"- Primary low context-change metric: AUC {fmt(exact_mech_context_metric.get('oriented_auc'))}, AP {fmt(exact_mech_context_metric.get('average_precision'))}, source eta2 {fmt(exact_mech_context_metric.get('source_eta2'))}",
        f"- Front-contraction composite future16 metric: AUC {fmt(exact_mech_contraction_metric.get('oriented_auc'))}, AP {fmt(exact_mech_contraction_metric.get('average_precision'))}, p={fmt(exact_mech_contraction_metric.get('mannwhitney_p'))}",
        f"- Composite exact-loss vs radius2 slope: raw rho {fmt(exact_mech_radius_corr.get('spearman_rho'))}, source-residual rho {fmt(exact_mech_radius_corr.get('source_residual_spearman_rho'))}; primary descriptor vs radius2 source-residual rho {fmt(exact_mech_primary_radius_corr.get('source_residual_spearman_rho'))}",
    ]
    for row in exact_mech_contraction[:8]:
        report_lines.append(
            f"- Mechanism correlation {row.get('anchor_feature')} vs {row.get('mechanism_feature')}: rho {fmt(row.get('spearman_rho'))}, p={fmt(row.get('spearman_p'))}, source-resid rho {fmt(row.get('source_residual_spearman_rho'))}, source-resid p={fmt(row.get('source_residual_spearman_p'))}"
        )
    for row in exact_mech_strata[:8]:
        report_lines.append(
            f"- High exact-loss stratum shift {row.get('feature')}: median high-low {fmt(row.get('median_high_minus_low'))}, p={fmt(row.get('mannwhitney_p'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(exact_feature_mechanism, 'guardrail', 'Exact feature mechanism consistency audit unavailable.')}")

    report_lines += [
        "",
        "## Signed Optical-Loss Mechanism Audit",
        "",
        f"- Rows/eval rows/cycles/sources: {first_summary(signed_optical_loss, 'n_rows', 0)} / {first_summary(signed_optical_loss, 'n_eval_rows', 0)} / {first_summary(signed_optical_loss, 'n_cycles', 0)} / {first_summary(signed_optical_loss, 'n_sources', 0)}",
        f"- Axis inputs: {first_summary(signed_optical_loss, 'axis_input_features', {})}",
        f"- Future16 combined/optical/echem axis AUCs: {fmt(signed_loss_combined16.get('oriented_auc'))} / {fmt(signed_loss_optical16.get('oriented_auc'))} / {fmt(signed_loss_echem16.get('oriented_auc'))}; source eta2 combined/optical {fmt(signed_loss_combined16.get('source_eta2'))} / {fmt(signed_loss_optical16.get('source_eta2'))}",
        f"- Best future16 leave-source axis model: {signed_loss_best_model16.get('feature_set', 'NA')} AUC {fmt(signed_loss_best_model16.get('roc_auc'))}, AP {fmt(signed_loss_best_model16.get('average_precision'))}, rho {fmt(signed_loss_best_model16.get('spearman_rho'))}",
        f"- Best future8 leave-source axis model: {signed_loss_best_model8.get('feature_set', 'NA')} AUC {fmt(signed_loss_best_model8.get('roc_auc'))}, AP {fmt(signed_loss_best_model8.get('average_precision'))}, rho {fmt(signed_loss_best_model8.get('spearman_rho'))}",
    ]
    for row in signed_loss_tests[:8]:
        report_lines.append(
            f"- Signed-loss axis {row.get('target')} {row.get('axis')}: AUC {fmt(row.get('oriented_auc'))}, median positive-negative {fmt(row.get('median_positive_minus_negative'))}, p={fmt(row.get('mannwhitney_p'))}, eta2 {fmt(row.get('source_eta2'))}"
        )
    for row in signed_loss_modes[:6]:
        report_lines.append(
            f"- Mechanism mode {row.get('mechanism_label')}: n={fmt(row.get('n_rows'), 0)}, cycles={fmt(row.get('n_cycles'), 0)}, sources={fmt(row.get('n_sources'), 0)}, future8/future16 {fmt(row.get('future8_rate'))}/{fmt(row.get('future16_rate'))}, median residual {fmt(row.get('median_transferred_masked_residual_signature'))}"
        )
    for row in signed_loss_candidates[:6]:
        report_lines.append(
            f"- Top signed-loss candidate {row.get('roi_id')}: cycle {fmt(row.get('cycleNo'), 0)}, source {row.get('source_stem')}, combined axis {fmt(row.get('combined_loss_mechanism_axis'))}, future8/future16 {fmt(row.get('future_any_drop_within_8cycles'), 0)}/{fmt(row.get('future_any_drop_within_16cycles'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(signed_optical_loss, 'guardrail', 'Signed optical-loss mechanism audit unavailable.')}")

    report_lines += [
        "",
        "## Signed-Loss Source Robustness Audit",
        "",
        f"- Rows/future16 labeled/cycles/sources: {first_summary(signed_loss_source_robustness, 'n_rows', 0)} / {first_summary(signed_loss_source_robustness, 'n_labeled_future16', 0)} / {first_summary(signed_loss_source_robustness, 'n_cycles', 0)} / {first_summary(signed_loss_source_robustness, 'n_sources', 0)}",
        f"- Combined axis raw/source-mean/within-source-rank AUC: {fmt(signed_robust_combined_raw.get('oriented_auc'))} / {fmt(signed_robust_combined_source_mean.get('oriented_auc'))} / {fmt(signed_robust_combined_rank.get('oriented_auc'))}",
        f"- Optical axis raw/source-mean/within-source-rank AUC: {fmt(signed_robust_optical_raw.get('oriented_auc'))} / {fmt(signed_robust_optical_source_mean.get('oriented_auc'))} / {fmt(signed_robust_optical_rank.get('oriented_auc'))}",
        f"- Echem degraded source-residual AUC/AP: {fmt(signed_robust_echem_resid.get('oriented_auc'))} / {fmt(signed_robust_echem_resid.get('average_precision'))}",
        f"- Optical within-source rank global/within-source permutation p: {fmt(signed_robust_optical_rank.get('global_label_shuffle'))} / {fmt(signed_robust_optical_rank.get('within_source_label_shuffle'))}",
    ]
    for row in signed_robust_key[:12]:
        report_lines.append(
            f"- Robustness {row.get('axis')} {row.get('transform')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, source eta2 {fmt(row.get('source_eta2'))}, balanced AUC mean {fmt(row.get('balanced_auc_mean'))}, within-source p {fmt(row.get('within_source_label_shuffle'))}"
        )
    for row in signed_robust_influence[:6]:
        report_lines.append(
            f"- Source influence {row.get('axis')} drop {row.get('dropped_source')}: full AUC {fmt(row.get('full_oriented_auc'))}, drop-source AUC {fmt(row.get('drop_source_oriented_auc'))}, delta {fmt(row.get('delta_auc_minus_full'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(signed_loss_source_robustness, 'guardrail', 'Signed-loss source robustness audit unavailable.')}")

    report_lines += [
        "",
        "## Echem/Optical Source-Residual Audit",
        "",
        f"- Rows/future16 labeled/cycles/sources: {first_summary(echem_optical_source_residual, 'n_rows', 0)} / {first_summary(echem_optical_source_residual, 'n_labeled_future16', 0)} / {first_summary(echem_optical_source_residual, 'n_cycles', 0)} / {first_summary(echem_optical_source_residual, 'n_sources', 0)}",
        f"- Direct future16 echem / optical / echem+optical residual AUCs: {fmt(echem_resid_direct.get('oriented_auc'))} / {fmt(optical_resid_direct.get('oriented_auc'))} / {fmt(echem_optical_resid_direct.get('oriented_auc'))}",
        f"- Direct echem+optical+front residual AUC/AP/source eta2: {fmt(echem_optical_front_direct.get('oriented_auc'))} / {fmt(echem_optical_front_direct.get('average_precision'))} / {fmt(echem_optical_front_direct.get('source_eta2'))}",
        f"- Leave-source echem+optical residual model AUC/AP: {fmt(echem_optical_resid_model.get('roc_auc'))} / {fmt(echem_optical_resid_model.get('average_precision'))}",
    ]
    for row in echem_optical_direct[:8]:
        report_lines.append(
            f"- Direct source-residual set {row.get('feature_set')}: AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, source eta2 {fmt(row.get('source_eta2'))}, p={fmt(row.get('spearman_p'))}"
        )
    for row in echem_optical_models[:8]:
        report_lines.append(
            f"- Leave-source residual model {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, rho {fmt(row.get('spearman_rho'))}"
        )
    for row in echem_optical_rules[:6]:
        report_lines.append(
            f"- Residual rule {row.get('target')} {row.get('rule')}: precision {fmt(row.get('precision'))}, recall {fmt(row.get('recall'))}, lift {fmt(row.get('lift'))}, source-positive hits {fmt(row.get('n_sources_with_positive_hits'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(echem_optical_source_residual, 'guardrail', 'Echem/optical source-residual audit unavailable.')}")

    report_lines += [
        "",
        "## Invariant Physics Rule Discovery",
        "",
        f"- Rows/eval rows/cycles/sources: {first_summary(invariant_physics_rules, 'n_rows', 0)} / {first_summary(invariant_physics_rules, 'n_eval_rows', 0)} / {first_summary(invariant_physics_rules, 'n_cycles', 0)} / {first_summary(invariant_physics_rules, 'n_sources', 0)}",
        f"- Target/positive rate/candidate rules: {first_summary(invariant_physics_rules, 'target', 'NA')} / {fmt(first_summary(invariant_physics_rules, 'positive_rate'))} / {first_summary(invariant_physics_rules, 'n_candidate_rules', 0)}",
        f"- Best rule: {invariant_rule_best.get('terms', 'NA')} with precision {fmt(invariant_rule_best.get('precision'))}, recall {fmt(invariant_rule_best.get('recall'))}, lift {fmt(invariant_rule_best.get('lift'))}, binary AUC {fmt(invariant_rule_best.get('binary_auc'))}, Fisher p={fmt(invariant_rule_best.get('fisher_p_greater'))}",
        f"- Best-rule source support: hits in {fmt(invariant_rule_best.get('n_sources_with_hits'), 0)} sources, positive hits in {fmt(invariant_rule_best.get('n_sources_with_positive_hits'), 0)} sources, max feature source eta2 {fmt(invariant_rule_best.get('max_feature_source_eta2'))}",
    ]
    for row in invariant_rule_top[:8]:
        report_lines.append(
            f"- Rule {row.get('terms')}: covered {fmt(row.get('n_covered'), 0)}/{fmt(row.get('n_eval'), 0)}, precision {fmt(row.get('precision'))}, recall {fmt(row.get('recall'))}, lift {fmt(row.get('lift'))}, positive-source hits {fmt(row.get('n_sources_with_positive_hits'), 0)}"
        )
    for row in invariant_rule_features[:8]:
        report_lines.append(
            f"- Oriented feature {row.get('direction')}({row.get('feature')}): AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, p={fmt(row.get('mannwhitney_p'))}, source eta2 {fmt(row.get('source_eta2'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(invariant_physics_rules, 'guardrail', 'Invariant physics rule discovery unavailable.')}")

    report_lines += [
        "",
        "## Agentic Current Hypothesis Tournament",
        "",
        f"- Hypotheses ranked: {first_summary(agentic_current, 'n_hypotheses', 0)}",
        f"- Paper-inspired roles: {first_summary(agentic_current, 'paper_inspiration', {})}",
    ]
    for row in agentic_current_top:
        report_lines.append(
            f"- Rank {fmt(row.get('rank'), 0)} {row.get('title')}: score {fmt(row.get('tournament_score'))}; next experiment: {row.get('next_experiment')}"
        )
    for row in agentic_current_specs[:3]:
        report_lines.append(
            f"- Next spec {fmt(row.get('rank'), 0)}: script {row.get('suggested_script')} with success evidence: {row.get('minimum_success_evidence')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(agentic_current, 'guardrail', 'Agentic current hypothesis tournament unavailable.')}")
    for row in balanced_future_acq_resid[:6]:
        report_lines.append(
            f"- Acquisition-context residual feature {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, AUC {fmt(row.get('oriented_auc'))}, MW p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in balanced_future_spatial_tests:
        report_lines.append(
            f"- Spatial region test {row.get('region_col')}: chi2={fmt(row.get('chi2'))}, p={fmt(row.get('p_value'))}, regions={fmt(row.get('n_regions'), 0)}"
        )
    report_lines.append(
        f"- Temporal directionality future8 model: {temporal_future8.get('model', 'NA')} AUC {fmt(temporal_future8.get('pooled_oof_roc_auc'))}, AP {fmt(temporal_future8.get('pooled_oof_average_precision'))}; shift-null mean AUC {fmt(temporal_shift_null.get('shift_null_auc_mean'))}, p95 {fmt(temporal_shift_null.get('shift_null_auc_p95'))}, empirical p={fmt(temporal_shift_null.get('empirical_p_ge_observed'))}"
    )
    report_lines.append(
        f"- Temporal directionality reversed/past8 guardrails: reversed best AUC {fmt(temporal_reversed8.get('pooled_oof_roc_auc'))}; past8 positives {temporal_past8_counts.get('1', 0)} and evaluable AUC {fmt(temporal_past8.get('pooled_oof_roc_auc'))}"
    )
    for row in temporal_future_tests[:6]:
        report_lines.append(
            f"- Temporal future8 feature {row.get('feature')}: AUC {fmt(row.get('auc'))}, median positive-negative {fmt(row.get('median_positive_minus_negative'))}, MW p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in temporal_timing_corr[:5]:
        report_lines.append(
            f"- Temporal timing correlation {row.get('feature')} vs {row.get('timing_target')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}"
        )
    for row in apparent_diffusion_q70_tests[:5]:
        report_lines.append(
            f"- Apparent diffusion q70 {row.get('feature')}: median future8 positive-negative {fmt(row.get('median_positive_minus_negative'))}, positive fractions {fmt(row.get('positive_fraction_positive'))}/{fmt(row.get('positive_fraction_negative'))}, MW p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in apparent_diffusion_corr[:4]:
        report_lines.append(
            f"- Apparent diffusion calibration correlation {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    report_lines.append(
        f"- Apparent diffusion timing guardrail: ROI/HDF5 elapsed median ratio {fmt(first_summary(apparent_diffusion_calibration, 'median_roi_elapsed_to_h5_median_ratio'))}, max source dt max/median ratio {fmt(first_summary(apparent_diffusion_calibration, 'max_source_h5_dt_max_to_median_ratio'))}, q70 positive-D fraction {fmt(first_summary(apparent_diffusion_calibration, 'q70_positive_D_fraction'))}"
    )
    report_lines.append(f"- Apparent diffusion guardrail: {first_summary(apparent_diffusion_calibration, 'guardrail', 'Apparent diffusion calibration audit unavailable.')}")
    report_lines.append(
        f"- Balanced spatial propagation graph: nodes {first_summary(balanced_spatial_propagation, 'n_nodes', 0)}, edges {first_summary(balanced_spatial_propagation, 'n_edges', 0)}, edge counts {first_summary(balanced_spatial_propagation, 'edge_counts', {})}"
    )
    for row in spatial_prop_homophily[:4]:
        report_lines.append(
            f"- Spatial propagation homophily {row.get('edge_type')}: same future8 {fmt(row.get('observed_same_future8_fraction'))} vs null mean {fmt(row.get('null_same_mean'))}, p={fmt(row.get('empirical_p_same_ge_observed'))}"
        )
    for row in spatial_prop_autocorr[:6]:
        report_lines.append(
            f"- Spatial propagation autocorr {row.get('edge_type')} {row.get('feature')}: rho={fmt(row.get('spearman_src_dst'))}, p={fmt(row.get('empirical_p_abs_ge_observed'))}"
        )
    for row in spatial_prop_lag[:4]:
        report_lines.append(
            f"- Spatial lag feature-to-next-label {row.get('feature')}: AUC {fmt(row.get('auc_src_feature_predicts_dst_future8'))}, null p={fmt(row.get('empirical_p_ge_observed'))}"
        )
    for row in spatial_prop_distance[:3]:
        report_lines.append(
            f"- Spatial distance gradient {row.get('edge_type')}: both-positive minus other {fmt(row.get('median_distance_both_positive_minus_other_um'))} um, p={fmt(row.get('mannwhitney_p'))}"
        )
    report_lines.append(f"- Spatial propagation guardrail: {first_summary(balanced_spatial_propagation, 'guardrail', 'Balanced spatial front propagation audit unavailable.')}")
    report_lines.append(f"- Temporal guardrail: {first_summary(temporal_directionality, 'guardrail', 'Temporal directionality audit unavailable.')}")
    report_lines.append(f"- Context guardrail: {first_summary(balanced_future_context, 'guardrail', 'Balanced future context/region audit unavailable.')}")
    report_lines.append(f"- Front-script guardrail: {first_summary(balanced_future_fronts, 'diffusion_guardrail', 'Balanced future fronts unavailable.')}")
    report_lines.append(f"- Audit guardrail: {first_summary(balanced_future_physics, 'guardrail', 'Balanced future physics audit unavailable.')}")

    report_lines += [
        "",
        "## Cross-Cohort Rollout Transfer",
        "",
        f"- ROI cohorts selected/transfer-ranked: {first_summary(cross_cohort_rollout, 'n_selected_roi', 0)} / {first_summary(cross_cohort_rollout, 'n_transfer_ranked_roi', 0)}",
        f"- Low-rank rank/train fraction: {first_summary(cross_cohort_rollout, 'rank', 0)} / {fmt(first_summary(cross_cohort_rollout, 'train_fraction'))}",
    ]
    for row in cross_cohort_shift:
        report_lines.append(
            f"- Transfer model {row.get('model_name')} on {row.get('eval_cohort')}: median particle MSE {fmt(row.get('median_particle_mse'))}, DMD/persistence {fmt(row.get('median_dmd_ratio_vs_persistence'))}, particle/nonparticle ratio {fmt(row.get('median_particle_to_nonparticle_ratio'))}, internal ratio {fmt(row.get('median_particle_mse_ratio_vs_internal'))}, p={fmt(row.get('mwu_p_vs_internal'))}"
        )
    for row in cross_cohort_corr[:6]:
        report_lines.append(
            f"- Transfer/error link {row.get('model_name')} on {row.get('eval_cohort')} {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in cross_cohort_difficult[:6]:
        report_lines.append(
            f"- Transfer-ranked hard ROI {row.get('roi_id')} via {row.get('model_name')}: particle MSE {fmt(row.get('particle_mse_mean'))}, DMD/persistence {fmt(row.get('dmd_particle_mse_ratio_vs_persistence'))}, cycle {fmt(row.get('cycleNo'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(cross_cohort_rollout, 'guardrail', 'Cross-cohort rollout transfer unavailable.')}")

    report_lines += [
        "",
        "## Masked Residual Transition Timing",
        "",
        f"- ROI/method rows/permutations: {first_summary(masked_residual_timing, 'n_roi_method_rows', 0)} / {first_summary(masked_residual_timing, 'n_permutation', 0)}",
        f"- ROI count: {first_summary(masked_residual_timing, 'n_roi', 0)}",
    ]
    for row in masked_residual_timing_align[:6]:
        report_lines.append(
            f"- Alignment {row.get('method')} {row.get('distance_feature')}: median distance {fmt(row.get('median_distance_to_transition'))}, null mean {fmt(row.get('null_median_distance_mean'))}, empirical p={fmt(row.get('empirical_p_distance_le_observed'))}, n={fmt(row.get('n_roi'), 0)}"
        )
    for row in masked_residual_timing_tests[:6]:
        report_lines.append(
            f"- Event/control timing {row.get('method')} {row.get('feature')}: median event-control {fmt(row.get('median_diff_a_minus_b'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n_a'), 0)}/{fmt(row.get('n_b'), 0)}"
        )
    for row in masked_residual_timing_corr[:6]:
        report_lines.append(
            f"- Timing/kinetics link {row.get('method')} {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in masked_residual_timing_top[:6]:
        report_lines.append(
            f"- Near-transition residual ROI {row.get('roi_id')} {row.get('method')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): near fraction {fmt(row.get('near_transition_residual_fraction'))}, peak distance {fmt(row.get('peak_distance_to_transition_frac'))}, weighted distance {fmt(row.get('weighted_center_distance_to_transition_frac'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(masked_residual_timing, 'guardrail', 'Masked residual timing audit unavailable.')}")

    report_lines += [
        "",
        "## Masked Rollout Cycle Warning",
        "",
        f"- ROI cycles/features/permutations: {first_summary(masked_cycle_warning, 'n_roi_cycles', 0)} / {first_summary(masked_cycle_warning, 'n_rollout_features_tested', 0)} / {first_summary(masked_cycle_warning, 'n_permutation', 0)}",
        f"- Target positive counts: {masked_cycle_warning_targets}",
    ]
    for row in masked_cycle_warning_tests[:6]:
        report_lines.append(
            f"- Target test {row.get('target')} {row.get('feature')}: median positive-negative {fmt(row.get('median_positive_minus_negative'))}, MW p={fmt(row.get('mannwhitney_p'))}, permutation p={fmt(row.get('permutation_p_abs_median_diff'))}, n={fmt(row.get('n_positive'), 0)}/{fmt(row.get('n_negative'), 0)}"
        )
    for row in masked_cycle_warning_corr[:6]:
        report_lines.append(
            f"- Cycle context link {row.get('feature')} vs {row.get('context')}: rho={fmt(row.get('rho'))}, p={fmt(row.get('p_value'))}, permutation p={fmt(row.get('permutation_p_abs_rho'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in masked_cycle_warning_top[:6]:
        report_lines.append(
            f"- Warning-ranked cycle {fmt(row.get('cycleNo'), 0)}: score {fmt(row.get('masked_rollout_cycle_warning_score'))}, abrupt_drop={fmt(row.get('any_abrupt_drop'), 0)}, future8={fmt(row.get('future_any_drop_within_8cycles'), 0)}, n_roi={fmt(row.get('n_roi'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(masked_cycle_warning, 'guardrail', 'Cycle-level masked rollout warning audit unavailable.')}")


    report_lines += [
        "",
        "## Masked ROI Rollout Audit",
        "",
        f"- ROI/frame metric rows: {first_summary(masked_rollout, 'n_roi', 0)} / {first_summary(masked_rollout, 'n_frame_metric_rows', 0)}",
        f"- Best method counts inside particle masks: {masked_rollout_best_counts}",
        f"- DMD spectral radius: {fmt(first_summary(masked_rollout, 'dmd_spectral_radius'))}",
    ]
    for row in masked_rollout_methods:
        report_lines.append(
            f"- {row.get('method')}: particle-MSE median {fmt(row.get('particle_mse_mean_median'))}, nonparticle-MSE median {fmt(row.get('nonparticle_mse_mean_median'))}, particle/nonparticle ratio median {fmt(row.get('particle_to_nonparticle_mse_ratio_mean_median'))}"
        )
    for row in masked_rollout_tests[:6]:
        report_lines.append(
            f"- Masked rollout event/control {row.get('method')} {row.get('feature')}: median event-control {fmt(row.get('median_diff_event_minus_control'))}, p={fmt(row.get('p_value'))}"
        )
    for row in masked_rollout_corr[:6]:
        report_lines.append(
            f"- Masked rollout/physics link {row.get('method')} {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in masked_rollout_difficult[:5]:
        report_lines.append(
            f"- Particle-rollout difficult ROI {row.get('roi_id')} {row.get('method')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): particle MSE {fmt(row.get('particle_mse_mean'))}, particle/nonparticle ratio {fmt(row.get('particle_to_nonparticle_mse_ratio_mean'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(masked_rollout, 'guardrail', 'Masked rollout audit unavailable.')}")

    report_lines += [
        "",
        "## Particle Mask Stability Audit",
        "",
        f"- ROI/frame rows: {first_summary(particle_mask, 'n_roi', 0)} / {first_summary(particle_mask, 'n_frames_total', 0)}",
        f"- Overall median fallback fraction: {fmt(particle_mask_overall.get('median_fallback_frame_fraction'))}",
        f"- Overall median accepted-area CV: {fmt(particle_mask_overall.get('median_accepted_area_cv'))}",
        f"- Overall median centroid path: {fmt(particle_mask_overall.get('median_centroid_path_px'))} px",
        f"- Overall median instability score: {fmt(particle_mask_overall.get('median_mask_instability_score'))}",
    ]
    for row in particle_mask_role_summary:
        report_lines.append(
            f"- {row.get('cohort_role')} masks: n={fmt(row.get('n_roi'), 0)}, fallback median {fmt(row.get('fallback_frame_fraction'))}, area-CV median {fmt(row.get('accepted_area_cv'))}, centroid-path median {fmt(row.get('accepted_centroid_path_px'))} px"
        )
    for row in particle_mask_tests[:6]:
        report_lines.append(
            f"- Event/control mask test {row.get('feature')}: median event-control {fmt(row.get('median_diff_event_minus_control'))}, p={fmt(row.get('p_value'))}"
        )
    for row in particle_mask_corr[:5]:
        report_lines.append(
            f"- Mask/physics link {row.get('x')} vs {row.get('y')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in particle_mask_top[:5]:
        report_lines.append(
            f"- High-instability ROI {row.get('roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): score {fmt(row.get('mask_instability_score'))}, area-CV {fmt(row.get('accepted_area_cv'))}, centroid path {fmt(row.get('accepted_centroid_path_px'))} px"
        )
    report_lines.append(f"- Guardrail: {first_summary(particle_mask, 'method', {}).get('interpretation', 'Automatic particle-mask stability audit only.')}")

    report_lines += [
        "",
        "## Particle Mask History/Fallback Audit",
        "",
        f"- Status/input/processed/failures/sources: {first_summary(particle_mask_history_fallback, 'overall_status', 'unavailable')} / {first_summary(particle_mask_history_fallback, 'n_input_rows', 0)} / {first_summary(particle_mask_history_fallback, 'n_ok', 0)} / {first_summary(particle_mask_history_fallback, 'n_failures', 0)} / {first_summary(particle_mask_history_fallback, 'n_sources', 0)}",
        f"- Median fallback fraction / q90 fallback / median history IoU: {fmt(first_summary(particle_mask_history_fallback, 'median_fallback_frame_fraction'))} / {fmt(first_summary(particle_mask_history_fallback, 'q90_fallback_frame_fraction'))} / {fmt(first_summary(particle_mask_history_fallback, 'median_history_iou'))}",
        f"- Median centroid q90 jitter / blur q10 ratio / near-non fallback median diff: {fmt(first_summary(particle_mask_history_fallback, 'median_centroid_jitter_q90_px'))} px / {fmt(first_summary(particle_mask_history_fallback, 'median_blur_ratio_q10'))} / {fmt(first_summary(particle_mask_history_fallback, 'near_vs_non_median_fallback_diff'))}",
    ]
    for row in particle_mask_history_tests[:8]:
        report_lines.append(
            f"- Mask-history test {row.get('target', 'NA')} {row.get('feature', 'NA')}: AUC {fmt(row.get('auc'))}, AP {fmt(row.get('average_precision'))}, med pos/neg {fmt(row.get('median_pos'))}/{fmt(row.get('median_neg'))}, source-strat p {fmt(row.get('source_stratified_permutation_p'))}"
        )
    for row in particle_mask_history_sources[:6]:
        report_lines.append(
            f"- Mask-history source {row.get('source_stem', 'NA')}: n={fmt(row.get('n_rows'), 0)}, median fallback {fmt(row.get('median_fallback_fraction'))}, median IoU {fmt(row.get('median_iou'))}, near-pre fraction {fmt(row.get('near_pre_fraction'))}"
        )
    for row in particle_mask_history_high[:4]:
        report_lines.append(
            f"- High fallback ROI {row.get('roi_id', 'NA')}: fallback {fmt(row.get('fallback_frame_fraction'))}, IoU {fmt(row.get('median_history_iou'))}, blur q10 {fmt(row.get('blur_ratio_q10'))}, event bin {row.get('event_relative_bin', 'NA')}"
        )
    report_lines.append(f"- Guardrail: {first_summary(particle_mask_history_fallback, 'guardrail', 'Particle mask history/fallback audit unavailable.')}")

    report_lines += [
        "",
        "## History/Fallback Masked Rollout Ablation",
        "",
        f"- Status/input/processed/failures/sources/cycles: {first_summary(history_fallback_rollout_ablation, 'overall_status', 'unavailable')} / {first_summary(history_fallback_rollout_ablation, 'n_input_rows', 0)} / {first_summary(history_fallback_rollout_ablation, 'n_ok', 0)} / {first_summary(history_fallback_rollout_ablation, 'n_failures', 0)} / {first_summary(history_fallback_rollout_ablation, 'n_sources', 0)} / {first_summary(history_fallback_rollout_ablation, 'n_cycles', 0)}",
        f"- Median fallback fraction / adaptive one-step MSE / hybrid one-step MSE: {fmt(first_summary(history_fallback_rollout_ablation, 'median_fallback_frame_fraction'))} / {fmt(first_summary(history_fallback_rollout_ablation, 'median_one_step_adaptive_mse'))} / {fmt(first_summary(history_fallback_rollout_ablation, 'median_one_step_hybrid_mse'))}",
        f"- Median hybrid-adaptive one-step delta / latent-linear gain history / latent-linear gain hybrid: {fmt(first_summary(history_fallback_rollout_ablation, 'median_hybrid_minus_adaptive_one_step_mse'))} / {fmt(first_summary(history_fallback_rollout_ablation, 'median_latent_gain_history'))} / {fmt(first_summary(history_fallback_rollout_ablation, 'median_latent_gain_hybrid'))}",
    ]
    for row in history_fallback_rollout_methods[:4]:
        report_lines.append(
            f"- Ablation summary {row.get('metric', 'NA')}: median {fmt(row.get('median'))}, q10 {fmt(row.get('q10'))}, q90 {fmt(row.get('q90'))}"
        )
    for row in history_fallback_rollout_tests[:8]:
        report_lines.append(
            f"- Rollout ablation test {row.get('target', 'NA')} {row.get('feature', 'NA')} ({row.get('transform', 'NA')}): AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, med delta {fmt(row.get('median_positive_minus_negative'))}, p {fmt(row.get('mwu_p'))}"
        )
    for row in history_fallback_rollout_sources[:5]:
        report_lines.append(
            f"- Ablation source {row.get('source_stem', 'NA')}: n={fmt(row.get('n_roi'), 0)}, median fallback {fmt(row.get('median_fallback'))}, latent gain history/hybrid {fmt(row.get('latent_gain_history'))}/{fmt(row.get('latent_gain_hybrid'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(history_fallback_rollout_ablation, 'guardrail', 'History/fallback masked rollout ablation unavailable.')}")

    report_lines += [
        "",
        "## Rollout Front/Mode Coupling Audit",
        "",
        f"- Status/rows/sources/cycles/modes: {first_summary(rollout_front_mode_coupling, 'overall_status', 'unavailable')} / {first_summary(rollout_front_mode_coupling, 'n_rows', 0)} / {first_summary(rollout_front_mode_coupling, 'n_sources', 0)} / {first_summary(rollout_front_mode_coupling, 'n_cycles', 0)} / {first_summary(rollout_front_mode_coupling, 'n_modes', 0)}",
    ]
    for row in rollout_front_mode_source_corr[:8]:
        report_lines.append(
            f"- Source-residual rollout/physics link {row.get('rollout_feature', 'NA')} vs {row.get('physics_feature', 'NA')}: rho {fmt(row.get('spearman_rho'))}, p {fmt(row.get('spearman_p'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in rollout_front_mode_tests[:6]:
        report_lines.append(
            f"- Coupled feature test {row.get('target', 'NA')} {row.get('feature', 'NA')} ({row.get('transform', 'NA')}): AUC {fmt(row.get('oriented_auc'))}, AP {fmt(row.get('average_precision'))}, med delta {fmt(row.get('median_positive_minus_negative'))}, p {fmt(row.get('mwu_p'))}"
        )
    for row in rollout_front_mode_modes[:4]:
        report_lines.append(
            f"- Coupled mode {row.get('mode', 'NA')}: n={fmt(row.get('n_roi'), 0)}, near {fmt(row.get('near_pre_fraction'))}, future8 {fmt(row.get('future8_fraction'))}, median latent gain {fmt(row.get('median_latent_gain_history'))}, median transport/front {fmt(row.get('median_transport_score'))}/{fmt(row.get('median_front_kinetic_score'))}"
        )
    for row in rollout_front_mode_queue[:5]:
        report_lines.append(
            f"- Coupled review ROI {row.get('roi_id', 'NA')}: score {fmt(row.get('rollout_front_review_score'))}, bin {row.get('event_relative_bin', 'NA')}, one-step MSE {fmt(row.get('one_step_hybrid_mse'))}, latent gain {fmt(row.get('latent_linear_gain_vs_persistence_history'))}, transport/front {fmt(row.get('transport_mechanism_score'))}/{fmt(row.get('front_kinetic_score'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(rollout_front_mode_coupling, 'guardrail', 'Rollout front/mode coupling audit unavailable.')}")

    report_lines += [
        "",
        "## Physics Consistency Claim Matrix",
        "",
        f"- ROI/cycles: {first_summary(physics_consistency, 'n_roi', 0)} / {first_summary(physics_consistency, 'n_cycles', 0)}",
        f"- Tier counts: {first_summary(physics_consistency, 'tier_counts', {})}",
        f"- Claim readiness: {first_summary(physics_consistency, 'claim_readiness_counts', {})}",
        f"- Manual-QC accepted rows: {first_summary(physics_consistency, 'manual_qc_accepted', 0)}",
    ]
    for row in physics_consistency_top[:8]:
        report_lines.append(
            f"- Rank {fmt(row.get('physics_consistency_rank'), 0)} {row.get('roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): score {fmt(row.get('physics_consistency_score'))}, support {fmt(row.get('physics_pillar_support_count'), 0)}, tier {row.get('physics_consistency_tier')}"
        )
    for row in physics_consistency_tests[:6]:
        report_lines.append(
            f"- Event/control pillar test {row.get('feature')}: median event-control {fmt(row.get('median_event_minus_control'))}, MW p={fmt(row.get('mannwhitney_p'))}, permutation p={fmt(row.get('permutation_p'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(physics_consistency, 'guardrail', 'Physics consistency matrix unavailable.')}")

    report_lines += [
        "",
        "## Cycle State-Space Transition Audit",
        "",
        f"- Cycle rows/features: {first_summary(cycle_state_space, 'n_cycles', 0)} / {first_summary(cycle_state_space, 'n_features_used', 0)}",
        f"- Echem-shape cycles joined: {first_summary(cycle_state_space, 'n_echem_shape_cycles_joined', 0)}",
        f"- Chosen state clusters: {first_summary(cycle_state_space, 'chosen_k', 0)} with silhouette {fmt(first_summary(cycle_state_space, 'best_silhouette'))}",
        f"- Degradation axis oriented to {first_summary(cycle_state_space, 'degradation_axis_oriented_to', 'NA')} with rho {fmt(first_summary(cycle_state_space, 'degradation_axis_orientation_rho'))}",
        f"- Future-drop classifier: shuffled-fold AUC {fmt(cycle_state_classifier.get('mean_roc_auc'))}, balanced accuracy {fmt(cycle_state_classifier.get('mean_balanced_accuracy'))}",
        f"- Expanding temporal holdout: AUC {fmt(cycle_state_temporal.get('mean_roc_auc'))}, balanced accuracy {fmt(cycle_state_temporal.get('mean_balanced_accuracy'))}, evaluated blocks {fmt(cycle_state_temporal.get('n_evaluated_blocks'), 0)} / {fmt(cycle_state_temporal.get('n_blocks'), 0)}, purge {fmt(cycle_state_temporal.get('purge_horizon_cycles'), 0)} cycles",
    ]
    for row in cycle_state_tests[:6]:
        report_lines.append(
            f"- Future-drop test {row.get('feature')}: positive-negative median {fmt(row.get('median_positive_minus_negative'))}, permutation p={fmt(row.get('permutation_p'))}, MW p={fmt(row.get('mannwhitney_p'))}"
        )
    for row in cycle_state_corr[:5]:
        report_lines.append(
            f"- State correlation {row.get('feature')} vs {row.get('target')}: rho={fmt(row.get('rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in cycle_state_clusters[:4]:
        report_lines.append(
            f"- State {row.get('cycle_state_cluster')}: n={fmt(row.get('n_cycles'), 0)}, cycles {fmt(row.get('cycle_min'), 0)}-{fmt(row.get('cycle_max'), 0)}, future8 rate={fmt(row.get('future_any_drop_within_8cycles_rate'))}"
        )
    for row in cycle_state_transitions[:4]:
        report_lines.append(
            f"- Transition {row.get('transition')}: n={fmt(row.get('n_transitions'), 0)}, next future8 rate={fmt(row.get('next_future8_rate'))}, step norm={fmt(row.get('median_state_step_norm'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(cycle_state_space, 'guardrail', 'Cycle state-space audit unavailable.')}")


    report_lines += [
        "",
        "## Cycle Hazard Warning Audit",
        "",
        f"- Target/events: {first_summary(cycle_hazard_warning, 'target', 'NA')} / {first_summary(cycle_hazard_warning, 'n_event_cycles', 0)} event cycles {first_summary(cycle_hazard_warning, 'event_cycles', [])}",
        f"- Rolling-origin purge/min-train/permutation nulls: {first_summary(cycle_hazard_warning, 'purge_cycles', 0)} / {first_summary(cycle_hazard_warning, 'min_train', 0)} / {cycle_hazard_null.get('n_permutations', 0)}",
    ]
    for row in cycle_hazard_feature_sets[:5]:
        report_lines.append(
            f"- Feature set {row.get('feature_set')}: AUC {fmt(row.get('roc_auc'))}, AP {fmt(row.get('average_precision'))}, Brier {fmt(row.get('brier_score'))}, balanced accuracy {fmt(row.get('balanced_accuracy_top_rate'))}, n={fmt(row.get('n_evaluated_cycles'), 0)}, positives={fmt(row.get('n_positive'), 0)}"
        )
    if cycle_hazard_null:
        report_lines.append(
            f"- Null check: observed AUC {fmt(cycle_hazard_null.get('observed_auc'))}, null p95 {fmt(cycle_hazard_null.get('null_p95_auc'))}, empirical p={fmt(cycle_hazard_null.get('empirical_p_ge_observed'))}"
        )
    for row in cycle_hazard_lead:
        report_lines.append(
            f"- Lead-time {fmt(row.get('lookback_horizon_cycles'), 0)} cycles: hit rate {fmt(row.get('hit_rate'))}, median max probability {fmt(row.get('median_max_probability'))}, median lead {fmt(row.get('median_lead_cycles'))} cycles"
        )
    for row in cycle_hazard_ablation[:5]:
        report_lines.append(
            f"- Ablation remove {row.get('removed_group')}: AUC {fmt(row.get('roc_auc'))}, drop vs best {fmt(row.get('auc_drop_vs_best'))}"
        )
    for row in cycle_hazard_corr[:5]:
        report_lines.append(
            f"- Warning probability link {row.get('feature')}: rho={fmt(row.get('spearman_rho'))}, p={fmt(row.get('p_value'))}, n={fmt(row.get('n'), 0)}"
        )
    report_lines.append(f"- Guardrail: {first_summary(cycle_hazard_warning, 'guardrail', 'Cycle hazard warning audit unavailable.')}")

    report_lines += [
        "",
        "## Cycle State To ROI/Front Bridge",
        "",
        f"- ROI rows/cycles joined: {first_summary(cycle_state_roi_bridge, 'n_roi_rows', 0)} / {first_summary(cycle_state_roi_bridge, 'n_cycles', 0)}",
        f"- Predictors/targets: {first_summary(cycle_state_roi_bridge, 'n_predictors', 0)} / {first_summary(cycle_state_roi_bridge, 'n_targets', 0)}",
    ]
    for row in cycle_state_roi_row_tests[:5]:
        report_lines.append(
            f"- Row bridge {row.get('predictor')} vs {row.get('target')}: rho={fmt(row.get('rho'))}, permutation p={fmt(row.get('permutation_p'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in cycle_state_roi_collapsed_tests[:5]:
        report_lines.append(
            f"- Cycle-collapsed bridge {row.get('predictor')} vs {row.get('target')}: rho={fmt(row.get('rho'))}, permutation p={fmt(row.get('permutation_p'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in cycle_state_roi_centered_tests[:4]:
        report_lines.append(
            f"- Reference-centered bridge {row.get('predictor')} vs {row.get('target')}: rho={fmt(row.get('rho'))}, permutation p={fmt(row.get('permutation_p'))}, n={fmt(row.get('n'), 0)}"
        )
    for row in cycle_state_roi_clusters[:4]:
        report_lines.append(
            f"- Cycle-state cluster {row.get('cycle_state_cluster')}: ROI n={fmt(row.get('n_roi'), 0)}, cycles={fmt(row.get('n_cycles'), 0)}, cross-modal priority fraction={fmt(row.get('cross_modal_priority_fraction'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(cycle_state_roi_bridge, 'guardrail', 'Cycle-state ROI bridge unavailable.')}")

    report_lines += [
        "",
        "## Cycle-State Mode-Frequency Bridge",
        "",
        f"- Cycles/ROI rows/mode targets: {first_summary(cycle_state_mode_frequency, 'n_cycles', 0)} / {first_summary(cycle_state_mode_frequency, 'n_roi_rows', 0)} / {first_summary(cycle_state_mode_frequency, 'n_mode_targets', 0)}",
        f"- Best macro model: {cycle_state_mode_best.get('feature_set', 'NA')} MAE {fmt(cycle_state_mode_best.get('mae'))}; context-only MAE {fmt(cycle_state_mode_context.get('mae'))}; reduction {fmt(first_summary(cycle_state_mode_frequency, 'best_minus_context_macro_mae_reduction'))}",
    ]
    for row in cycle_state_mode_metrics[:10]:
        report_lines.append(
            f"- Mode-frequency model {row.get('feature_set')} -> {row.get('target')}: MAE {fmt(row.get('mae'))}, R2 {fmt(row.get('r2'))}, rho {fmt(row.get('spearman_rho'))}"
        )
    for row in cycle_state_mode_nulls[:6]:
        report_lines.append(
            f"- Mode-frequency null {row.get('feature_set')}: observed macro MAE {fmt(row.get('observed_macro_mae'))}, null mean {fmt(row.get('null_mae_mean'))}, p={fmt(row.get('empirical_p_le_observed_mae'))}"
        )
    for row in cycle_state_mode_clusters[:4]:
        report_lines.append(
            f"- Cycle-state cluster {row.get('cycle_state_cluster')}: cycles={fmt(row.get('n_cycles'), 0)}, ROI={fmt(row.get('total_roi'), 0)}, median cycle={fmt(row.get('median_cycle'))}"
        )
    report_lines.append(f"- Guardrail: {first_summary(cycle_state_mode_frequency, 'guardrail', 'Cycle-state mode-frequency bridge unavailable.')}")

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
        "## Residual-Mode Review Priorities",
        "",
    ]
    for row in residual_mode_top_rois:
        report_lines.append(
            f"- {row.get('roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): {row.get('mode_label')}, priority {fmt(row.get('mode_review_priority'))}"
        )

    report_lines += [
        "",
        "## Manual QC Label Workbook",
        "",
        f"- Deduplicated ROI candidates: {first_summary(manual_qc_workbook, 'n_unique_roi', 0)}",
        f"- Role counts: {first_summary(manual_qc_workbook, 'role_counts', {})}",
        f"- Priority tiers: {first_summary(manual_qc_workbook, 'review_priority_tier_counts', {})}",
        f"- Manual-QC status counts: {first_summary(manual_qc_workbook, 'manual_qc_status_counts', {})}",
        f"- Guardrail: {first_summary(manual_qc_workbook, 'guardrail', 'This is a label template, not completed manual QC.')}",
    ]

    report_lines += [
        "",
        "## Weak-Label Degradation Benchmark",
        "",
        f"- ROI rows: {first_summary(weak_label_benchmark, 'n_roi_rows', 0)}",
        f"- Trainable weak-label rows: {first_summary(weak_label_benchmark, 'n_trainable_weak_label_rows', 0)}",
        f"- Positive/negative weak labels: {first_summary(weak_label_benchmark, 'n_positive_weak_labels', 0)} / {first_summary(weak_label_benchmark, 'n_negative_weak_labels', 0)}",
        f"- Label counts: {first_summary(weak_label_benchmark, 'label_counts', {})}",
        f"- Leave-reference usable binary folds: {weak_label_leakage.get('n_usable_binary_folds', 0)} / {weak_label_leakage.get('n_folds', 0)}",
    ]
    for row in weak_label_top_pos[:5]:
        report_lines.append(
            f"- Weak positive {row.get('roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): physics score {fmt(row.get('physics_consistency_score'))}, mode {row.get('mode_label')}"
        )
    for row in weak_label_top_neg[:5]:
        report_lines.append(
            f"- Weak negative {row.get('roi_id')} ({row.get('cohort_role')}, cycle {fmt(row.get('cycleNo'), 0)}): physics score {fmt(row.get('physics_consistency_score'))}, mode {row.get('mode_label')}"
        )
    for row in weak_label_leakage.get('folds_with_missing_class', [])[:4]:
        report_lines.append(
            f"- Split guardrail fold {row.get('fold')} holdout {fmt(row.get('holdout_event_reference_cycle'), 0)}: {row.get('trainable_fold_status')} ({fmt(row.get('n_positive_test'), 0)} positive / {fmt(row.get('n_negative_test'), 0)} negative test)"
        )
    report_lines.append(f"- Guardrail: {first_summary(weak_label_benchmark, 'guardrail', 'Weak labels are not manual QC labels.')}")

    report_lines += [
        "",
        "## Manual-QC Gated Front Effects",
        "",
        f"- Status: {first_summary(manual_qc_gated, 'status', 'missing')}",
        f"- Joined ROI rows: {first_summary(manual_qc_gated, 'n_joined_roi', 0)}",
        f"- Accepted front-effect rows: {first_summary(manual_qc_gated, 'n_manual_front_effect_accepted', 0)}",
        f"- Accepted diffusion-interpretable rows: {first_summary(manual_qc_gated, 'n_manual_diffusion_accepted', 0)}",
        f"- Guardrail: {first_summary(manual_qc_gated, 'guardrail', 'Manual-QC gate unavailable.')}",
        "",
        "## Control-Balanced QC Sensitivity",
        "",
        f"- Robust positive phase-residual strata: {', '.join(first_summary(control_balanced_qc_sensitivity, 'robust_positive_phase_residual_strata', []))}",
        f"- Guardrail: {first_summary(control_balanced_qc_sensitivity, 'interpretation', 'Automatic sensitivity only; not manual QC.')}",
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
        "1. Manual QC the ROI/front package panels and update `manual_qc_status` with accepted/rejected labels.",
        "2. Expand the ROI cohort across more cycles after QC to reduce event-reference and protocol confounding.",
        "3. Recompute apparent diffusion/front-motion proxies only on QC-accepted fronts with confirmed spatial/time calibration.",
        "4. Convert the top ROI candidates into a labeled degradation-mode benchmark for future self-supervised video models.",
        "5. Grow protocol-conditioned residual models after adding more QC-accepted cycles and particle regions.",
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
        "n_threshold_robust_front_roi": first_summary(robust_fronts, "n_roi"),
        "n_qc_review_candidates": first_summary(qc_packet, "n_candidates"),
        "n_roi_front_qc_package_candidates": first_summary(qc_package, "n_selected_roi"),
        "roi_front_qc_flag_counts": first_summary(qc_package, "flag_counts", {}),
        "control_balanced_front_qc": {
            "n_selected_roi": first_summary(control_balanced_qc, "n_selected_roi"),
            "n_event_roi": first_summary(control_balanced_qc, "n_event_roi"),
            "n_control_roi": first_summary(control_balanced_qc, "n_control_roi"),
            "n_new_roi": first_summary(control_balanced_qc, "n_new_roi"),
            "nonfragmented_by_role": first_summary(control_balanced_qc, "nonfragmented_by_role", {}),
            "no_auto_flag_by_role": first_summary(control_balanced_qc, "no_auto_flag_by_role", {}),
        },
        "n_residual_physics_mode_roi": first_summary(residual_modes, "n_roi"),
        "front_qc_sensitivity_n_roi": first_summary(front_qc_sensitivity, "n_roi"),
        "front_qc_robust_positive_phase_residual_strata": robust_phase_strata,
        "front_qc_sensitivity_strata": front_qc_strata,
        "front_qc_sensitivity_focus_tests": front_qc_focus,
        "control_balanced_front_qc_sensitivity_n_roi": first_summary(control_balanced_qc_sensitivity, "n_roi"),
        "control_balanced_front_qc_robust_positive_phase_residual_strata": balanced_robust_phase_strata,
        "control_balanced_front_qc_sensitivity_focus_tests": balanced_qc_focus,
        "residual_physics_mode_chosen_k": first_summary(residual_modes, "chosen_k"),
        "residual_physics_mode_cluster_stability": first_summary(residual_modes, "cluster_stability", {}),
        "residual_physics_mode_best_silhouette": first_summary(residual_modes, "cluster_selection", {}).get("best", {}).get("silhouette"),
        "residual_physics_mode_enrichment": residual_mode_enrichment,
        "top_residual_mode_review_rois": residual_mode_top_rois,
        "cycle_region_mode_context": {
            "n_cycles": first_summary(cycle_region_modes, "n_cycles"),
            "n_xy_regions": first_summary(cycle_region_modes, "n_xy_regions"),
            "event_enriched_mode_fraction": first_summary(cycle_region_modes, "event_enriched_mode_fraction"),
            "context_only_classifier": first_summary(cycle_region_modes, "context_only_classifier", {}),
        },
        "spatiotemporal_degradation_graph": {
            "n_nodes": first_summary(spatiotemporal_graph, "n_nodes"),
            "n_edges": first_summary(spatiotemporal_graph, "n_edges"),
            "edge_counts": first_summary(spatiotemporal_graph, "edge_counts", {}),
            "top_homophily_tests": graph_homophily,
            "top_continuous_neighbor_tests": graph_continuous,
            "temporal_lag_tests": graph_lag,
            "distance_gradient_tests": graph_distance,
            "guardrail": first_summary(spatiotemporal_graph, "guardrail"),
        },
        "phase_kinetics_avrami": {
            "n_roi": first_summary(phase_kinetics, "n_roi"),
            "n_kinetic_features": first_summary(phase_kinetics, "n_kinetic_features"),
            "top_group_tests": phase_kinetics_tests,
            "top_correlations": phase_kinetics_corr,
            "guardrail": first_summary(phase_kinetics, "guardrail"),
        },
        "control_balanced_front_tracking": {
            "n_tracked_rois": first_summary(control_balanced_front_tracking, "n_tracked_rois"),
            "cycle_summary": top_items(first_summary(control_balanced_front_tracking, "cycle_summary", []), 20),
            "guardrail": first_summary(control_balanced_front_tracking, "guardrail"),
        },
        "control_balanced_diffusion_proxy_sanity_audit": {
            "n_selected_front_rois": first_summary(control_balanced_diffusion_sanity, "n_selected_front_rois"),
            "selected_front_cohort_counts": first_summary(control_balanced_diffusion_sanity, "selected_front_cohort_counts", {}),
            "n_automatic_positive_diffusion_proxy_candidates": first_summary(control_balanced_diffusion_sanity, "n_automatic_positive_diffusion_proxy_candidates"),
            "n_publication_diffusion_candidates": first_summary(control_balanced_diffusion_sanity, "n_publication_diffusion_candidates"),
            "median_selected_diffusion_um2_per_s": first_summary(control_balanced_diffusion_sanity, "median_selected_diffusion_um2_per_s"),
            "selected_positive_fraction": first_summary(control_balanced_diffusion_sanity, "selected_positive_fraction"),
            "gate_counts": control_balanced_diffusion_gate,
            "top_event_control_tests": control_balanced_diffusion_tests,
            "top_automatic_candidates": control_balanced_diffusion_candidates,
            "guardrail": first_summary(control_balanced_diffusion_sanity, "guardrail"),
        },
        "diffusion_proxy_sanity_audit": {
            "n_selected_front_rois": first_summary(diffusion_sanity, "n_selected_front_rois"),
            "selected_front_cohort_counts": first_summary(diffusion_sanity, "selected_front_cohort_counts", {}),
            "n_automatic_positive_diffusion_proxy_candidates": first_summary(diffusion_sanity, "n_automatic_positive_diffusion_proxy_candidates"),
            "n_publication_diffusion_candidates": first_summary(diffusion_sanity, "n_publication_diffusion_candidates"),
            "median_selected_diffusion_um2_per_s": first_summary(diffusion_sanity, "median_selected_diffusion_um2_per_s"),
            "median_threshold_diffusion_um2_per_s": first_summary(diffusion_sanity, "median_threshold_diffusion_um2_per_s"),
            "selected_positive_fraction": first_summary(diffusion_sanity, "selected_positive_fraction"),
            "threshold_positive_fraction": first_summary(diffusion_sanity, "threshold_positive_fraction"),
            "estimator_consensus_counts": first_summary(diffusion_sanity, "estimator_consensus_counts", {}),
            "gate_counts": diffusion_sanity_gate,
            "top_automatic_candidates": diffusion_sanity_candidates,
            "top_correlations": diffusion_sanity_corr,
            "guardrail": first_summary(diffusion_sanity, "guardrail"),
        },
        "transfer_ranked_front_physics_audit": {
            "n_roi": first_summary(transfer_ranked_front_physics, "n_roi"),
            "n_cycles": first_summary(transfer_ranked_front_physics, "n_cycles"),
            "target_positive_counts": first_summary(transfer_ranked_front_physics, "target_positive_counts", {}),
            "cycle_target_positive_counts": first_summary(transfer_ranked_front_physics, "cycle_target_positive_counts", {}),
            "top_target_tests": transfer_ranked_front_target_tests,
            "top_correlations": transfer_ranked_front_corr,
            "top_cycle_target_tests": transfer_ranked_front_cycle_tests,
            "top_cycle_correlations": transfer_ranked_front_cycle_corr,
            "top_review_rois": transfer_ranked_front_review,
            "threshold_front_group_summary": transfer_ranked_front_group,
            "top_threshold_robust_phase_rois": transfer_ranked_front_top,
            "guardrail": first_summary(transfer_ranked_front_physics, "guardrail"),
        },
        "transfer_ranked_roi_reconstruction": {
            "n_cycles_sampled": first_summary(transfer_ranked_reconstruction, "n_cycles_sampled"),
            "n_reconstructed_candidates": first_summary(transfer_ranked_reconstruction, "n_reconstructed_candidates"),
            "n_roi_rows": first_summary(transfer_ranked_reconstruction, "n_roi_rows"),
            "n_missing_cycles": first_summary(transfer_ranked_reconstruction, "n_missing_cycles"),
            "sampled_cycles": transfer_ranked_sampled,
            "top_roi_rows": transfer_ranked_top_rois,
            "sequence_summary": {
                "n_roi_sequences": first_summary(transfer_ranked_sequences, "n_roi_sequences"),
                "samples_per_roi": first_summary(transfer_ranked_sequences, "samples_per_roi"),
                "cycle_summary": transfer_ranked_sequence_cycles,
                "guardrail": first_summary(transfer_ranked_sequences, "guardrail"),
            },
            "masked_rollout": {
                "n_roi": first_summary(transfer_ranked_rollout, "n_roi"),
                "n_frame_metric_rows": first_summary(transfer_ranked_rollout, "n_frame_metric_rows"),
                "best_method_counts_inside_particle": transfer_ranked_rollout_best_counts,
                "method_summary": transfer_ranked_rollout_methods,
                "top_particle_difficulty_rois": transfer_ranked_rollout_difficult,
                "guardrail": first_summary(transfer_ranked_rollout, "guardrail"),
            },
            "guardrail": first_summary(transfer_ranked_reconstruction, "guardrail"),
        },
        "transfer_ranked_residual_transition_timing": {
            "n_roi": first_summary(transfer_ranked_residual_timing, "n_roi"),
            "n_roi_method_rows": first_summary(transfer_ranked_residual_timing, "n_roi_method_rows"),
            "n_permutation": first_summary(transfer_ranked_residual_timing, "n_permutation"),
            "target_positive_counts": first_summary(transfer_ranked_residual_timing, "target_positive_counts", {}),
            "top_alignment_tests": transfer_ranked_timing_align,
            "top_timing_target_tests": transfer_ranked_timing_target,
            "top_timing_target_correlations": transfer_ranked_timing_target_corr,
            "top_transition_correlations": transfer_ranked_timing_transition_corr,
            "top_near_transition_residual_rois": transfer_ranked_timing_top_rois,
            "guardrail": first_summary(transfer_ranked_residual_timing, "guardrail"),
        },
        "source_balanced_roi_expansion_manifest": {
            "n_ranked_cycles": first_summary(source_balanced_expansion, "n_ranked_cycles"),
            "n_selected_cycles": first_summary(source_balanced_expansion, "n_selected_cycles"),
            "n_sampled_cycles": first_summary(source_balanced_expansion, "n_sampled_cycles"),
            "n_new_selected_cycles": first_summary(source_balanced_expansion, "n_new_selected_cycles"),
            "n_sources_selected": first_summary(source_balanced_expansion, "n_sources_selected"),
            "n_reconstructed_candidates": first_summary(source_balanced_expansion, "n_reconstructed_candidates"),
            "n_roi_rows": first_summary(source_balanced_expansion, "n_roi_rows"),
            "selected_label_counts": first_summary(source_balanced_expansion, "selected_label_counts", {}),
            "selection_reason_counts": first_summary(source_balanced_expansion, "selection_reason_counts", {}),
            "source_coverage": source_balanced_expansion_sources,
            "top_roi_rows": source_balanced_expansion_top,
            "guardrail": first_summary(source_balanced_expansion, "guardrail"),
        },
        "source_balanced_roi_sequences": {
            "n_input_roi_rows": first_summary(source_balanced_sequences, "n_input_roi_rows"),
            "n_roi_sequences": first_summary(source_balanced_sequences, "n_roi_sequences"),
            "n_cycles": first_summary(source_balanced_sequences, "n_cycles"),
            "n_sources": first_summary(source_balanced_sequences, "n_sources"),
            "n_failed": first_summary(source_balanced_sequences, "n_failed"),
            "crop_size_full": first_summary(source_balanced_sequences, "crop_size_full"),
            "output_size": first_summary(source_balanced_sequences, "output_size"),
            "samples_per_roi": first_summary(source_balanced_sequences, "samples_per_roi"),
            "future8_positive_sequences": first_summary(source_balanced_sequences, "future8_positive_sequences"),
            "future16_positive_sequences": first_summary(source_balanced_sequences, "future16_positive_sequences"),
            "guardrail": first_summary(source_balanced_sequences, "guardrail"),
        },
        "source_balanced_sequence_rollout_audit": {
            "n_roi_sequences": first_summary(source_balanced_sequence_rollout, "n_roi_sequences"),
            "n_cycles": first_summary(source_balanced_sequence_rollout, "n_cycles"),
            "n_sources": first_summary(source_balanced_sequence_rollout, "n_sources"),
            "future8_positive_sequences": first_summary(source_balanced_sequence_rollout, "future8_positive_sequences"),
            "future16_positive_sequences": first_summary(source_balanced_sequence_rollout, "future16_positive_sequences"),
            "top_roi_feature_tests": source_balanced_rollout_roi_tests,
            "top_cycle_feature_tests": source_balanced_rollout_cycle_tests,
            "source_summary": source_balanced_rollout_sources,
            "guardrail": first_summary(source_balanced_sequence_rollout, "guardrail"),
        },
        "source_balanced_sequence_source_control_audit": {
            "n_rows": first_summary(source_balanced_sequence_source_control, "n_rows"),
            "n_cycles": first_summary(source_balanced_sequence_source_control, "n_cycles"),
            "n_sources": first_summary(source_balanced_sequence_source_control, "n_sources"),
            "n_permutation": first_summary(source_balanced_sequence_source_control, "n_permutation"),
            "n_scalar_rows": first_summary(source_balanced_sequence_source_control, "n_scalar_rows"),
            "n_strict_scalar_rows": first_summary(source_balanced_sequence_source_control, "n_strict_scalar_rows"),
            "n_source_model_auc_ge_065": first_summary(source_balanced_sequence_source_control, "n_source_model_auc_ge_065"),
            "verdict": first_summary(source_balanced_sequence_source_control, "verdict"),
            "top_source_stratified_scalars": source_balanced_source_control_scalars,
            "top_source_heldout_models": source_balanced_source_control_models,
            "top_model_deltas_vs_context": source_balanced_source_control_deltas,
            "guardrail": first_summary(source_balanced_sequence_source_control, "guardrail"),
        },
        "source_balanced_expansion_transport_front_audit": {
            "n_input_rows": first_summary(source_balanced_expansion_transport_front, "n_input_rows"),
            "n_ok": first_summary(source_balanced_expansion_transport_front, "n_ok"),
            "n_failed": first_summary(source_balanced_expansion_transport_front, "n_failed"),
            "n_cycles": first_summary(source_balanced_expansion_transport_front, "n_cycles"),
            "n_sources": first_summary(source_balanced_expansion_transport_front, "n_sources"),
            "future8_positive_rows": first_summary(source_balanced_expansion_transport_front, "future8_positive_rows"),
            "future16_positive_rows": first_summary(source_balanced_expansion_transport_front, "future16_positive_rows"),
            "flow_method": first_summary(source_balanced_expansion_transport_front, "flow_method"),
            "top_feature_tests": source_balanced_expansion_transport_tests,
            "source_summary_top": source_balanced_expansion_transport_sources,
            "top_candidates": source_balanced_expansion_transport_candidates,
            "outputs": first_summary(source_balanced_expansion_transport_front, "outputs", {}),
            "guardrail": first_summary(source_balanced_expansion_transport_front, "guardrail"),
        },
        "source_balanced_mask_front_sanity_audit": {
            "n_roi_sequences": first_summary(source_balanced_mask_front, "n_roi_sequences"),
            "n_cycles": first_summary(source_balanced_mask_front, "n_cycles"),
            "n_sources": first_summary(source_balanced_mask_front, "n_sources"),
            "future8_positive_sequences": first_summary(source_balanced_mask_front, "future8_positive_sequences"),
            "future16_positive_sequences": first_summary(source_balanced_mask_front, "future16_positive_sequences"),
            "pixel_size_um_assumed": first_summary(source_balanced_mask_front, "pixel_size_um_assumed"),
            "top_roi_feature_tests": source_balanced_mask_front_roi_tests,
            "top_cycle_feature_tests": source_balanced_mask_front_cycle_tests,
            "source_summary": source_balanced_mask_front_sources,
            "guardrail": first_summary(source_balanced_mask_front, "guardrail"),
        },
        "source_balanced_residual_dictionary_audit": {
            "n_roi_sequences": first_summary(source_balanced_residual_dictionary, "n_roi_sequences"),
            "n_cycles": first_summary(source_balanced_residual_dictionary, "n_cycles"),
            "n_sources": first_summary(source_balanced_residual_dictionary, "n_sources"),
            "future8_positive_sequences": first_summary(source_balanced_residual_dictionary, "future8_positive_sequences"),
            "future16_positive_sequences": first_summary(source_balanced_residual_dictionary, "future16_positive_sequences"),
            "n_components": first_summary(source_balanced_residual_dictionary, "n_components"),
            "pca_explained_variance_ratio_sum": first_summary(source_balanced_residual_dictionary, "pca_explained_variance_ratio_sum"),
            "feature_set_sizes": first_summary(source_balanced_residual_dictionary, "feature_set_sizes", {}),
            "top_metrics": source_balanced_resdict_metrics,
            "top_roi_feature_tests": source_balanced_resdict_roi_tests,
            "top_cycle_feature_tests": source_balanced_resdict_cycle_tests,
            "guardrail": first_summary(source_balanced_residual_dictionary, "guardrail"),
        },
        "source_balanced_residual_dictionary_source_residual_audit": {
            "n_rows": first_summary(source_balanced_resdict_source_residual, "n_rows"),
            "n_cycles": first_summary(source_balanced_resdict_source_residual, "n_cycles"),
            "n_sources": first_summary(source_balanced_resdict_source_residual, "n_sources"),
            "n_features_tested": first_summary(source_balanced_resdict_source_residual, "n_features_tested"),
            "feature_family_counts": first_summary(source_balanced_resdict_source_residual, "feature_family_counts", {}),
            "future16_source_residual_residual_dictionary_best": source_balanced_resdict_sr_best,
            "future16_within_source_rank_residual_dictionary_best": source_balanced_resdict_rank_best,
            "future16_source_residual_best": source_balanced_resdict_sr_transform_best,
            "guardrail": first_summary(source_balanced_resdict_source_residual, "guardrail"),
        },
        "source_balanced_residual_dictionary_normalized_readout": {
            "n_rows": first_summary(source_balanced_resdict_normalized_readout, "n_rows"),
            "n_cycles": first_summary(source_balanced_resdict_normalized_readout, "n_cycles"),
            "n_sources": first_summary(source_balanced_resdict_normalized_readout, "n_sources"),
            "feature_set_sizes": first_summary(source_balanced_resdict_normalized_readout, "feature_set_sizes", {}),
            "top_metrics": source_balanced_resdict_norm_metrics,
            "future16_leave_source_best": source_balanced_resdict_norm_best,
            "future16_leave_source_raw_residual_dictionary": source_balanced_resdict_norm_raw_source16,
            "future16_leave_source_source_residual_residual_dictionary": source_balanced_resdict_norm_sr_source16,
            "future16_leave_source_within_source_rank_residual_dictionary": source_balanced_resdict_norm_rank_source16,
            "future16_leave_cycle_dictionary_recon_error_last_minus_first_source_residual": source_balanced_resdict_norm_cycle_single16,
            "permutation_summary": source_balanced_resdict_norm_perm,
            "guardrail": first_summary(source_balanced_resdict_normalized_readout, "guardrail"),
        },
        "source_balanced_residual_temporal_specificity_audit": {
            "n_rows": first_summary(source_balanced_residual_temporal_specificity, "n_rows"),
            "n_cycles": first_summary(source_balanced_residual_temporal_specificity, "n_cycles"),
            "n_sources": first_summary(source_balanced_residual_temporal_specificity, "n_sources"),
            "event_cycles": first_summary(source_balanced_residual_temporal_specificity, "event_cycles", []),
            "label_counts": first_summary(source_balanced_residual_temporal_specificity, "label_counts", {}),
            "best_future_specific": source_balanced_temporal_best,
            "primary_source_residual_rows": source_balanced_temporal_primary,
            "primary_future16_shift_null": source_balanced_temporal_shift16,
            "guardrail": first_summary(source_balanced_residual_temporal_specificity, "guardrail"),
        },
        "source_balanced_pre_event_sampling_manifest": {
            "n_ranked_cycles": first_summary(source_balanced_pre_event_sampling, "n_ranked_cycles"),
            "n_selected_cycles": first_summary(source_balanced_pre_event_sampling, "n_selected_cycles"),
            "n_sampled_cycles": first_summary(source_balanced_pre_event_sampling, "n_sampled_cycles"),
            "n_sources_selected": first_summary(source_balanced_pre_event_sampling, "n_sources_selected"),
            "n_new_selected_cycles": first_summary(source_balanced_pre_event_sampling, "n_new_selected_cycles"),
            "n_reconstructed_candidates": first_summary(source_balanced_pre_event_sampling, "n_reconstructed_candidates"),
            "n_roi_rows": first_summary(source_balanced_pre_event_sampling, "n_roi_rows"),
            "n_missing_cycles": first_summary(source_balanced_pre_event_sampling, "n_missing_cycles"),
            "event_cycles": first_summary(source_balanced_pre_event_sampling, "event_cycles", []),
            "cycle_bin_counts": source_balanced_pre_event_bins,
            "roi_bin_counts": source_balanced_pre_event_roi_bins,
            "source_coverage": source_balanced_pre_event_sources,
            "guardrail": first_summary(source_balanced_pre_event_sampling, "guardrail"),
        },
        "source_balanced_pre_event_roi_sequences": {
            "n_roi_sequences": first_summary(source_balanced_pre_event_sequences, "n_roi_sequences"),
            "n_cycles": first_summary(source_balanced_pre_event_sequences, "n_cycles"),
            "n_sources": first_summary(source_balanced_pre_event_sequences, "n_sources"),
            "n_failed": first_summary(source_balanced_pre_event_sequences, "n_failed"),
            "future8_positive_sequences": first_summary(source_balanced_pre_event_sequences, "future8_positive_sequences"),
            "future16_positive_sequences": first_summary(source_balanced_pre_event_sequences, "future16_positive_sequences"),
            "guardrail": first_summary(source_balanced_pre_event_sequences, "guardrail"),
        },
        "source_balanced_pre_event_sequence_audit": {
            "n_feature_rows": first_summary(source_balanced_pre_event_sequence_audit, "n_feature_rows"),
            "n_cycles": first_summary(source_balanced_pre_event_sequence_audit, "n_cycles"),
            "n_sources": first_summary(source_balanced_pre_event_sequence_audit, "n_sources"),
            "n_failures": first_summary(source_balanced_pre_event_sequence_audit, "n_failures"),
            "bin_counts": source_balanced_pre_event_audit_bins,
            "target_positive_counts": source_balanced_pre_event_audit_targets,
            "top_model_metrics": source_balanced_pre_event_audit_models,
            "top_scalar_tests": source_balanced_pre_event_audit_scalars,
            "guardrail": first_summary(source_balanced_pre_event_sequence_audit, "guardrail"),
        },
        "source_balanced_pre_event_rollout_mask_readout": {
            "rollout_summary": {
                "n_roi_sequences": first_summary(source_balanced_pre_event_rollout, "n_roi_sequences"),
                "n_cycles": first_summary(source_balanced_pre_event_rollout, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_rollout, "n_sources"),
                "top_roi_feature_tests": source_balanced_pre_event_rollout_top,
                "guardrail": first_summary(source_balanced_pre_event_rollout, "guardrail"),
            },
            "masked_rollout_benchmark": {
                "n_input_rows": first_summary(source_balanced_pre_event_masked_rollout, "n_input_rows"),
                "n_ok": first_summary(source_balanced_pre_event_masked_rollout, "n_ok"),
                "n_failed": first_summary(source_balanced_pre_event_masked_rollout, "n_failed"),
                "n_cycles": first_summary(source_balanced_pre_event_masked_rollout, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_masked_rollout, "n_sources"),
                "train_fraction": first_summary(source_balanced_pre_event_masked_rollout, "train_fraction"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_masked_rollout, "event_relative_bin_counts", {}),
                "best_method_by_median_particle_gain": source_balanced_pre_event_masked_rollout_best_method,
                "method_summary": source_balanced_pre_event_masked_rollout_methods,
                "top_event_tests": source_balanced_pre_event_masked_rollout_tests,
                "outputs": first_summary(source_balanced_pre_event_masked_rollout, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_masked_rollout, "guardrail"),
            },
            "optical_flow_transport_audit": {
                "n_input_rows": first_summary(source_balanced_pre_event_optical_flow, "n_input_rows"),
                "n_ok": first_summary(source_balanced_pre_event_optical_flow, "n_ok"),
                "n_failed": first_summary(source_balanced_pre_event_optical_flow, "n_failed"),
                "n_cycles": first_summary(source_balanced_pre_event_optical_flow, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_optical_flow, "n_sources"),
                "flow_method": first_summary(source_balanced_pre_event_optical_flow, "flow_method"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_optical_flow, "event_relative_bin_counts", {}),
                "method_summary": first_summary(source_balanced_pre_event_optical_flow, "method_summary", []),
                "top_event_tests": source_balanced_pre_event_flow_tests,
                "best_source_residual_test": source_balanced_pre_event_flow_sr_best,
                "outputs": first_summary(source_balanced_pre_event_optical_flow, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_optical_flow, "guardrail"),
            },
            "mask_front_summary": {
                "n_roi_sequences": first_summary(source_balanced_pre_event_mask_front, "n_roi_sequences"),
                "n_cycles": first_summary(source_balanced_pre_event_mask_front, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_mask_front, "n_sources"),
                "top_roi_feature_tests": source_balanced_pre_event_mask_top,
                "guardrail": first_summary(source_balanced_pre_event_mask_front, "guardrail"),
            },
            "event_relative_readout": {
                "n_rows": first_summary(source_balanced_pre_event_readout, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_readout, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_readout, "n_sources"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_readout, "event_relative_bin_counts", {}),
                "best_by_target_transform": source_balanced_pre_event_readout_best,
                "best_source_residual_clean_pre_readouts": source_balanced_pre_event_readout_clean,
                "guardrail": first_summary(source_balanced_pre_event_readout, "guardrail"),
            },
            "event_distance_trajectory": {
                "n_cycle_rows": first_summary(source_balanced_pre_event_trajectory, "n_cycle_rows"),
                "n_sources": first_summary(source_balanced_pre_event_trajectory, "n_sources"),
                "n_event_cycles": first_summary(source_balanced_pre_event_trajectory, "n_event_cycles"),
                "n_features_tested": first_summary(source_balanced_pre_event_trajectory, "n_features_tested"),
                "n_permutations": first_summary(source_balanced_pre_event_trajectory, "n_permutations"),
                "full_event_cycles_with_near_and_far": first_summary(source_balanced_pre_event_trajectory, "full_event_cycles_with_near_and_far"),
                "top_physics_toward_event_tests": source_balanced_pre_event_traj_physics,
                "top_source_residual_event_distance_tests": source_balanced_pre_event_traj_source,
                "guardrail": first_summary(source_balanced_pre_event_trajectory, "guardrail"),
            },
            "directionality_audit": {
                "n_rows": first_summary(source_balanced_pre_event_directionality, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_directionality, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_directionality, "n_sources"),
                "n_features_tested": first_summary(source_balanced_pre_event_directionality, "n_features_tested"),
                "n_permutations": first_summary(source_balanced_pre_event_directionality, "n_permutations"),
                "best_by_target_transform": source_balanced_pre_event_dir_best,
                "best_pre_event_clock_features": source_balanced_pre_event_dir_clock,
                "best_pre_vs_post_clock_asymmetry": source_balanced_pre_event_dir_asym,
                "guardrail": first_summary(source_balanced_pre_event_directionality, "guardrail"),
            },
            "source_invariant_audit": {
                "n_rows": first_summary(source_balanced_pre_event_source_invariant, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_source_invariant, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_source_invariant, "n_sources"),
                "n_features_total": first_summary(source_balanced_pre_event_source_invariant, "n_features_total"),
                "targets": first_summary(source_balanced_pre_event_source_invariant, "targets", []),
                "methods": first_summary(source_balanced_pre_event_source_invariant, "methods", []),
                "best_by_target": source_balanced_pre_event_si_best,
                "best_low_source_eta2_models": source_balanced_pre_event_si_low_source,
                "top_univariate_rows": source_balanced_pre_event_si_uni,
                "guardrail": first_summary(source_balanced_pre_event_source_invariant, "guardrail"),
            },
            "review_packet": {
                "n_candidates": first_summary(source_balanced_pre_event_review_packet, "n_candidates"),
                "n_cycles": first_summary(source_balanced_pre_event_review_packet, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_review_packet, "n_sources"),
                "top_n_rendered": first_summary(source_balanced_pre_event_review_packet, "top_n_rendered"),
                "n_rendered_frame_strips": first_summary(source_balanced_pre_event_review_packet, "n_rendered_frame_strips"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_review_packet, "event_relative_bin_counts", {}),
                "review_reason_counts": source_balanced_pre_event_review_reasons,
                "top_candidates": source_balanced_pre_event_review_top,
                "contact_sheet": first_summary(source_balanced_pre_event_review_packet, "contact_sheet"),
                "guardrail": first_summary(source_balanced_pre_event_review_packet, "guardrail"),
            },
            "matched_counterfactual_audit": {
                "n_rows": first_summary(source_balanced_pre_event_matched_counterfactual, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_matched_counterfactual, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_matched_counterfactual, "n_sources"),
                "n_permutations": first_summary(source_balanced_pre_event_matched_counterfactual, "n_permutations"),
                "context_features": first_summary(source_balanced_pre_event_matched_counterfactual, "context_features", []),
                "outcome_features": first_summary(source_balanced_pre_event_matched_counterfactual, "outcome_features", []),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_matched_counterfactual, "event_relative_bin_counts", {}),
                "pair_counts": source_balanced_pre_event_matched_pair_counts,
                "same_source_pair_fraction": source_balanced_pre_event_matched_same_source,
                "top_physics_matched_tests": source_balanced_pre_event_matched_top,
                "guardrail": first_summary(source_balanced_pre_event_matched_counterfactual, "guardrail"),
            },
            "same_source_ladder_audit": {
                "n_rows": first_summary(source_balanced_pre_event_same_source_ladder, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_same_source_ladder, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_same_source_ladder, "n_sources"),
                "n_permutations": first_summary(source_balanced_pre_event_same_source_ladder, "n_permutations"),
                "ladder_source_counts": source_balanced_pre_event_ladder_counts,
                "pair_counts": source_balanced_pre_event_ladder_pair_counts,
                "comparison_source_counts": first_summary(source_balanced_pre_event_same_source_ladder, "comparison_source_counts", {}),
                "top_same_source_paired_tests": source_balanced_pre_event_ladder_top,
                "top_within_source_clock_tests": source_balanced_pre_event_ladder_clock,
                "guardrail": first_summary(source_balanced_pre_event_same_source_ladder, "guardrail"),
            },
            "source_lattice_coverage_audit": {
                "n_cycle_rows": first_summary(pre_event_source_lattice, "n_cycle_rows"),
                "n_sources": first_summary(pre_event_source_lattice, "n_sources"),
                "n_sources_with_h5": first_summary(pre_event_source_lattice, "n_sources_with_h5"),
                "event_relative_bin_counts": first_summary(pre_event_source_lattice, "event_relative_bin_counts", {}),
                "near_source_counts": pre_event_lattice_near_counts,
                "near_source_rows": pre_event_lattice_near_sources,
                "candidate_far_control_sources": pre_event_lattice_far_controls,
                "recommended_design": pre_event_lattice_design,
                "guardrail": first_summary(pre_event_source_lattice, "guardrail"),
            },
            "radial_kymograph_audit": {
                "n_roi": first_summary(source_balanced_pre_event_radial_kymograph, "n_roi"),
                "n_cycles": first_summary(source_balanced_pre_event_radial_kymograph, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_radial_kymograph, "n_sources"),
                "n_rendered_kymographs": first_summary(source_balanced_pre_event_radial_kymograph, "n_rendered_kymographs"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_radial_kymograph, "event_relative_bin_counts", {}),
                "top_near_vs_far_tests": source_balanced_pre_event_kymo_near_far,
                "top_clean_pre_vs_post_control_tests": source_balanced_pre_event_kymo_clean,
                "top_review_candidate_kymograph_features": source_balanced_pre_event_kymo_review,
                "guardrail": first_summary(source_balanced_pre_event_radial_kymograph, "guardrail"),
            },
            "echem_front_coupling_audit": {
                "n_rows": first_summary(source_balanced_pre_event_echem_front, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_echem_front, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_echem_front, "n_sources"),
                "n_echem_features": first_summary(source_balanced_pre_event_echem_front, "n_echem_features"),
                "n_context_features": first_summary(source_balanced_pre_event_echem_front, "n_context_features"),
                "n_targets": first_summary(source_balanced_pre_event_echem_front, "n_targets"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_echem_front, "event_relative_bin_counts", {}),
                "top_raw_effects": source_balanced_pre_event_echem_raw,
                "top_source_echem_residual_effects": source_balanced_pre_event_echem_resid,
                "top_echem_correlations": source_balanced_pre_event_echem_corr,
                "top_residual_fits": source_balanced_pre_event_echem_fit,
                "guardrail": first_summary(source_balanced_pre_event_echem_front, "guardrail"),
            },
            "echem_matched_residual_audit": {
                "n_rows": first_summary(source_balanced_pre_event_echem_matched_residual, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_echem_matched_residual, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_echem_matched_residual, "n_sources"),
                "n_context_features": first_summary(source_balanced_pre_event_echem_matched_residual, "n_context_features"),
                "n_echem_features": first_summary(source_balanced_pre_event_echem_matched_residual, "n_echem_features"),
                "n_match_features": first_summary(source_balanced_pre_event_echem_matched_residual, "n_match_features"),
                "n_outcome_features": first_summary(source_balanced_pre_event_echem_matched_residual, "n_outcome_features"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_echem_matched_residual, "event_relative_bin_counts", {}),
                "pair_counts": source_balanced_pre_event_echem_matched_resid_pair_counts,
                "same_source_pair_fraction": source_balanced_pre_event_echem_matched_resid_same_source,
                "top_source_echem_residual_matched_tests": source_balanced_pre_event_echem_matched_resid_top,
                "guardrail": first_summary(source_balanced_pre_event_echem_matched_residual, "guardrail"),
            },
            "front_consensus_audit": {
                "n_rows": first_summary(source_balanced_pre_event_front_consensus, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_front_consensus, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_front_consensus, "n_sources"),
                "n_matched_pairs": first_summary(source_balanced_pre_event_front_consensus, "n_matched_pairs"),
                "n_consensus_features": first_summary(source_balanced_pre_event_front_consensus, "n_consensus_features"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_front_consensus, "event_relative_bin_counts", {}),
                "consensus_features": first_summary(source_balanced_pre_event_front_consensus, "consensus_features", []),
                "top_event_tests": source_balanced_pre_event_front_consensus_event,
                "top_matched_tests": source_balanced_pre_event_front_consensus_matched,
                "top_clock_tests": source_balanced_pre_event_front_consensus_clock,
                "top_ranked_candidates": source_balanced_pre_event_front_consensus_ranked,
                "guardrail": first_summary(source_balanced_pre_event_front_consensus, "guardrail"),
            },
            "echem_matched_far_control_audit": {
                "n_rows": first_summary(source_balanced_pre_event_echem_matched_far, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_echem_matched_far, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_echem_matched_far, "n_sources"),
                "n_permutations": first_summary(source_balanced_pre_event_echem_matched_far, "n_permutations"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_echem_matched_far, "event_relative_bin_counts", {}),
                "source_class_counts": first_summary(source_balanced_pre_event_echem_matched_far, "source_class_counts", {}),
                "context_features": first_summary(source_balanced_pre_event_echem_matched_far, "context_features", []),
                "echem_features": first_summary(source_balanced_pre_event_echem_matched_far, "echem_features", []),
                "outcome_features": first_summary(source_balanced_pre_event_echem_matched_far, "outcome_features", []),
                "pair_counts": source_balanced_pre_event_echem_matched_far_pair_counts,
                "control_source_counts": source_balanced_pre_event_echem_matched_far_control_sources,
                "top_paired_tests": source_balanced_pre_event_echem_matched_far_top,
                "guardrail": first_summary(source_balanced_pre_event_echem_matched_far, "guardrail"),
            },
            "consensus_review_queue": {
                "n_candidates": first_summary(source_balanced_pre_event_consensus_review, "n_candidates"),
                "n_cycles": first_summary(source_balanced_pre_event_consensus_review, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_consensus_review, "n_sources"),
                "priority_tier_counts": source_balanced_pre_event_consensus_tiers,
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_consensus_review, "event_relative_bin_counts", {}),
                "top_candidates": source_balanced_pre_event_consensus_top,
                "guardrail": first_summary(source_balanced_pre_event_consensus_review, "guardrail"),
            },
            "consensus_visual_packet": {
                "n_queue_rows": first_summary(source_balanced_pre_event_consensus_visual, "n_queue_rows"),
                "top_n": first_summary(source_balanced_pre_event_consensus_visual, "top_n"),
                "n_rendered": first_summary(source_balanced_pre_event_consensus_visual, "n_rendered"),
                "n_sources_rendered": first_summary(source_balanced_pre_event_consensus_visual, "n_sources_rendered"),
                "event_relative_bin_counts_rendered": first_summary(source_balanced_pre_event_consensus_visual, "event_relative_bin_counts_rendered", {}),
                "rendered_candidates": source_balanced_pre_event_visual_top,
                "contact_sheet": first_summary(source_balanced_pre_event_consensus_visual, "contact_sheet"),
                "outputs": source_balanced_pre_event_visual_outputs,
                "guardrail": first_summary(source_balanced_pre_event_consensus_visual, "guardrail"),
            },
            "visual_sanity_audit": {
                "n_audited": first_summary(source_balanced_pre_event_visual_sanity, "n_audited"),
                "n_ok": first_summary(source_balanced_pre_event_visual_sanity, "n_ok"),
                "n_sources": first_summary(source_balanced_pre_event_visual_sanity, "n_sources"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_visual_sanity, "event_relative_bin_counts", {}),
                "visual_sanity_flag_counts": source_balanced_pre_event_visual_sanity_flags,
                "median_visual_sanity_score": first_summary(source_balanced_pre_event_visual_sanity, "median_visual_sanity_score"),
                "source_summary": source_balanced_pre_event_visual_sanity_sources,
                "top_reviewable_candidates": source_balanced_pre_event_visual_sanity_reviewable,
                "top_artifact_risk_candidates": source_balanced_pre_event_visual_sanity_artifact,
                "guardrail": first_summary(source_balanced_pre_event_visual_sanity, "guardrail"),
            },
            "visual_qc_modes": {
                "n_input_candidates": first_summary(source_balanced_pre_event_visual_qc_modes, "n_input_candidates"),
                "n_scored": first_summary(source_balanced_pre_event_visual_qc_modes, "n_scored"),
                "visual_qc_tier_counts": source_balanced_pre_event_visual_qc_tiers,
                "visual_mode_counts": source_balanced_pre_event_visual_qc_mode_counts,
                "mode_summary": source_balanced_pre_event_visual_qc_modes_summary,
                "top_candidates": source_balanced_pre_event_visual_qc_top,
                "outputs": first_summary(source_balanced_pre_event_visual_qc_modes, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_visual_qc_modes, "guardrail"),
            },
            "phase_kinetics_audit": {
                "n_input": first_summary(source_balanced_pre_event_phase_kinetics, "n_input"),
                "n_ok": first_summary(source_balanced_pre_event_phase_kinetics, "n_ok"),
                "n_sources": first_summary(source_balanced_pre_event_phase_kinetics, "n_sources"),
                "n_kinetic_features": first_summary(source_balanced_pre_event_phase_kinetics, "n_kinetic_features"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_phase_kinetics, "event_relative_bin_counts", {}),
                "top_event_tests": source_balanced_pre_event_phase_kinetics_event,
                "top_matched_tests": source_balanced_pre_event_phase_kinetics_matched,
                "top_correlations": source_balanced_pre_event_phase_kinetics_corr,
                "source_summary": source_balanced_pre_event_phase_kinetics_sources,
                "outputs": first_summary(source_balanced_pre_event_phase_kinetics, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_phase_kinetics, "guardrail"),
            },
            "front_kinetic_concordance_audit": {
                "n_rows": first_summary(source_balanced_pre_event_front_kinetic_concordance, "n_rows"),
                "n_ok": first_summary(source_balanced_pre_event_front_kinetic_concordance, "n_ok"),
                "n_sources": first_summary(source_balanced_pre_event_front_kinetic_concordance, "n_sources"),
                "tier_counts": source_balanced_pre_event_fk_concordance_tiers,
                "top_candidates": source_balanced_pre_event_fk_concordance_top,
                "top_event_tests": source_balanced_pre_event_fk_concordance_tests,
                "top_correlations": source_balanced_pre_event_fk_concordance_corr,
                "outputs": first_summary(source_balanced_pre_event_front_kinetic_concordance, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_front_kinetic_concordance, "guardrail"),
            },
            "front_kinetic_null_audit": {
                "n_rows": first_summary(source_balanced_pre_event_front_kinetic_null, "n_rows"),
                "n_sources": first_summary(source_balanced_pre_event_front_kinetic_null, "n_sources"),
                "n_features_tested": first_summary(source_balanced_pre_event_front_kinetic_null, "n_features_tested"),
                "n_permutations": first_summary(source_balanced_pre_event_front_kinetic_null, "n_permutations"),
                "n_bootstrap": first_summary(source_balanced_pre_event_front_kinetic_null, "n_bootstrap"),
                "top_null_tests": source_balanced_pre_event_fk_null_tests,
                "top_proximity_tests": source_balanced_pre_event_fk_null_proximity,
                "outputs": first_summary(source_balanced_pre_event_front_kinetic_null, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_front_kinetic_null, "guardrail"),
            },
            "manual_qc_decision_packet": {
                "n_rows": first_summary(source_balanced_pre_event_manual_qc_decision, "n_rows"),
                "n_sources": first_summary(source_balanced_pre_event_manual_qc_decision, "n_sources"),
                "n_visual_asset_rows": first_summary(source_balanced_pre_event_manual_qc_decision, "n_visual_asset_rows"),
                "action_counts": source_balanced_pre_event_manual_qc_actions,
                "top40_action_counts": source_balanced_pre_event_manual_qc_top40_actions,
                "top40_source_counts": first_summary(source_balanced_pre_event_manual_qc_decision, "top40_source_counts", {}),
                "n_manual_front_review_gate": first_summary(source_balanced_pre_event_manual_qc_decision, "n_manual_front_review_gate"),
                "n_automatic_diffusion_claim_gate": first_summary(source_balanced_pre_event_manual_qc_decision, "n_automatic_diffusion_claim_gate"),
                "top_candidates": source_balanced_pre_event_manual_qc_top,
                "outputs": first_summary(source_balanced_pre_event_manual_qc_decision, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_manual_qc_decision, "guardrail"),
            },
            "manual_qc_visual_packet": {
                "n_queue_rows": first_summary(source_balanced_pre_event_manual_qc_visual, "n_queue_rows"),
                "top_n": first_summary(source_balanced_pre_event_manual_qc_visual, "top_n"),
                "n_rendered": first_summary(source_balanced_pre_event_manual_qc_visual, "n_rendered"),
                "n_sources_rendered": first_summary(source_balanced_pre_event_manual_qc_visual, "n_sources_rendered"),
                "action_tier_counts_rendered": source_balanced_pre_event_manual_qc_visual_actions,
                "event_relative_bin_counts_rendered": source_balanced_pre_event_manual_qc_visual_bins,
                "rendered_candidates": source_balanced_pre_event_manual_qc_visual_top,
                "contact_sheet": first_summary(source_balanced_pre_event_manual_qc_visual, "contact_sheet"),
                "outputs": first_summary(source_balanced_pre_event_manual_qc_visual, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_manual_qc_visual, "guardrail"),
            },
            "manual_qc_blind_workbook": {
                "seed": first_summary(source_balanced_pre_event_manual_qc_blind, "seed"),
                "n_input_rows": first_summary(source_balanced_pre_event_manual_qc_blind, "n_input_rows"),
                "n_blinded_rows": first_summary(source_balanced_pre_event_manual_qc_blind, "n_blinded_rows"),
                "n_sources_hidden_key": first_summary(source_balanced_pre_event_manual_qc_blind, "n_sources_hidden_key"),
                "action_tier_counts_hidden_key": source_balanced_pre_event_manual_qc_blind_actions,
                "event_relative_bin_counts_hidden_key": source_balanced_pre_event_manual_qc_blind_bins,
                "review_fields": first_summary(source_balanced_pre_event_manual_qc_blind, "review_fields", []),
                "outputs": first_summary(source_balanced_pre_event_manual_qc_blind, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_manual_qc_blind, "guardrail"),
            },
            "multimodal_predictor": {
                "n_rows": first_summary(source_balanced_pre_event_multimodal, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_multimodal, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_multimodal, "n_sources"),
                "event_relative_bin_counts": first_summary(source_balanced_pre_event_multimodal, "event_relative_bin_counts", {}),
                "targets": first_summary(source_balanced_pre_event_multimodal, "targets", []),
                "methods": first_summary(source_balanced_pre_event_multimodal, "methods", []),
                "feature_family_sizes": first_summary(source_balanced_pre_event_multimodal, "feature_family_sizes", {}),
                "best_by_target": source_balanced_pre_event_multimodal_best,
                "best_family_deltas": source_balanced_pre_event_multimodal_deltas,
                "outputs": first_summary(source_balanced_pre_event_multimodal, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_multimodal, "guardrail"),
            },
            "strict_qc_gated_front_audit": {
                "n_candidates": first_summary(source_balanced_pre_event_strict_qc_gated_front, "n_candidates"),
                "n_manual_front_review_candidates": first_summary(source_balanced_pre_event_strict_qc_gated_front, "n_manual_front_review_candidates"),
                "n_automatic_diffusion_claim_candidates": first_summary(source_balanced_pre_event_strict_qc_gated_front, "n_automatic_diffusion_claim_candidates"),
                "gate_pass_counts": source_balanced_pre_event_strict_qc_gate_pass,
                "gate_fail_counts": first_summary(source_balanced_pre_event_strict_qc_gated_front, "gate_fail_counts", {}),
                "top_manual_front_review_candidates": source_balanced_pre_event_strict_qc_top,
                "top_strict_qc_ranked_candidates": source_balanced_pre_event_strict_qc_ranked,
                "outputs": first_summary(source_balanced_pre_event_strict_qc_gated_front, "outputs", {}),
                "guardrail": first_summary(source_balanced_pre_event_strict_qc_gated_front, "guardrail"),
            },
            "physics_mode_taxonomy": {
                "n_rows": first_summary(source_balanced_pre_event_physics_modes, "n_rows"),
                "n_cycles": first_summary(source_balanced_pre_event_physics_modes, "n_cycles"),
                "n_sources": first_summary(source_balanced_pre_event_physics_modes, "n_sources"),
                "n_features_used": first_summary(source_balanced_pre_event_physics_modes, "n_features_used"),
                "chosen_k": first_summary(source_balanced_pre_event_physics_modes, "chosen_k"),
                "k_scores": source_balanced_pre_event_mode_k,
                "mode_summary": source_balanced_pre_event_mode_summary,
                "top_enrichment": source_balanced_pre_event_mode_enrich,
                "clock_tests": source_balanced_pre_event_mode_clocks,
                "guardrail": first_summary(source_balanced_pre_event_physics_modes, "guardrail"),
            },
        },
        "source_balanced_future_specific_residual_audit": {
            "n_rows": first_summary(source_balanced_future_specific_residual, "n_rows"),
            "n_cycles": first_summary(source_balanced_future_specific_residual, "n_cycles"),
            "n_sources": first_summary(source_balanced_future_specific_residual, "n_sources"),
            "event_cycles": first_summary(source_balanced_future_specific_residual, "event_cycles", []),
            "label_counts": first_summary(source_balanced_future_specific_residual, "label_counts", {}),
            "primary_source_residual_subset_rows": source_balanced_future_specific_primary,
            "best_clean_future_subset_rows": source_balanced_future_specific_best_clean,
            "top_model_deltas_vs_past_context": source_balanced_future_specific_model_deltas,
            "guardrail": first_summary(source_balanced_future_specific_residual, "guardrail"),
        },
        "source_balanced_degradation_mode_audit": {
            "n_rows": first_summary(source_balanced_degradation_modes, "n_rows"),
            "n_cycles": first_summary(source_balanced_degradation_modes, "n_cycles"),
            "n_sources": first_summary(source_balanced_degradation_modes, "n_sources"),
            "n_features_used": first_summary(source_balanced_degradation_modes, "n_features_used"),
            "chosen_k": first_summary(source_balanced_degradation_modes, "chosen_k"),
            "k_scores": first_summary(source_balanced_degradation_modes, "k_scores", []),
            "cluster_summary": source_balanced_degmode_clusters,
            "top_enrichment": source_balanced_degmode_enrichment,
            "representatives": source_balanced_degmode_representatives,
            "source_mode_transition_count": first_summary(source_balanced_degradation_modes, "source_mode_transition_count"),
            "source_mode_change_fraction": first_summary(source_balanced_degradation_modes, "source_mode_change_fraction"),
            "guardrail": first_summary(source_balanced_degradation_modes, "guardrail"),
        },
        "source_balanced_residual_physics_coupling_audit": {
            "n_rows": first_summary(source_balanced_residual_physics_coupling, "n_rows"),
            "n_cycles": first_summary(source_balanced_residual_physics_coupling, "n_cycles"),
            "n_sources": first_summary(source_balanced_residual_physics_coupling, "n_sources"),
            "n_residual_features": first_summary(source_balanced_residual_physics_coupling, "n_residual_features"),
            "n_physics_features": first_summary(source_balanced_residual_physics_coupling, "n_physics_features"),
            "best_by_transform": source_balanced_resphys_by_transform,
            "best_source_residual_primary_candidate_correlations": source_balanced_resphys_primary,
            "best_source_residual_target_aligned_pairs": source_balanced_resphys_aligned,
            "dictionary_recon_error_last_minus_first_source_residual_top_correlations": source_balanced_resphys_dict_recon,
            "guardrail": first_summary(source_balanced_residual_physics_coupling, "guardrail"),
        },
        "source_balanced_residual_candidate_review_packet": {
            "n_candidates": first_summary(source_balanced_residual_candidate_review, "n_candidates"),
            "n_sources": first_summary(source_balanced_residual_candidate_review, "n_sources"),
            "n_cycles": first_summary(source_balanced_residual_candidate_review, "n_cycles"),
            "review_tier_counts": source_balanced_review_tiers,
            "top_review_candidates": source_balanced_review_top,
            "immediate_manual_qc_candidates": source_balanced_review_immediate,
            "source_summary": source_balanced_review_sources,
            "guardrail": first_summary(source_balanced_residual_candidate_review, "guardrail"),
        },
        "balanced_future_roi_physics_audit": {
            "n_cycles_sampled": first_summary(balanced_future_reconstruction, "n_cycles_sampled"),
            "n_reconstructed_candidates": first_summary(balanced_future_reconstruction, "n_reconstructed_candidates"),
            "n_roi_rows": first_summary(balanced_future_reconstruction, "n_roi_rows"),
            "n_roi_sequences": first_summary(balanced_future_sequences, "n_roi_sequences"),
            "n_roi": first_summary(balanced_future_physics, "n_roi"),
            "n_cycles": first_summary(balanced_future_physics, "n_cycles"),
            "n_features": first_summary(balanced_future_physics, "n_features"),
            "label_counts": first_summary(balanced_future_physics, "label_counts", []),
            "cycle_group_oof_metrics": balanced_future_oof,
            "permutation_null": balanced_future_null,
            "top_roi_feature_tests": balanced_future_roi_tests,
            "top_cycle_feature_tests": balanced_future_cycle_tests,
            "top_correlations": balanced_future_corr,
            "masked_rollout_best_method_counts_inside_particle": balanced_future_rollout_best_counts,
            "particle_mask_stability": {
                "n_roi": first_summary(balanced_future_mask, "n_roi"),
                "n_frames_total": first_summary(balanced_future_mask, "n_frames_total"),
                "overall": balanced_future_mask_overall,
                "role_summary": balanced_future_mask_roles,
                "top_future_label_tests": balanced_future_mask_tests,
            },
            "context_region_audit": {
                "n_roi": first_summary(balanced_future_context, "n_roi"),
                "n_cycles": first_summary(balanced_future_context, "n_cycles"),
                "best_acquisition_context_only": balanced_future_best_acq_context,
                "best_design_context_only": balanced_future_best_design_context,
                "best_physics_plus_acquisition_context": balanced_future_best_context_physics,
                "cycle_group_oof_metrics": balanced_future_context_oof,
                "top_acquisition_context_residual_feature_tests": balanced_future_acq_resid,
                "spatial_region_tests": balanced_future_spatial_tests,
                "guardrail": first_summary(balanced_future_context, "guardrail"),
            },
            "temporal_directionality_audit": {
                "n_roi": first_summary(temporal_directionality, "n_roi"),
                "n_cycles": first_summary(temporal_directionality, "n_cycles"),
                "n_physics_features": first_summary(temporal_directionality, "n_physics_features"),
                "target_label_counts": temporal_counts,
                "best_future8_model": temporal_future8,
                "best_past8_model": temporal_past8,
                "best_reversed_future8_model": temporal_reversed8,
                "shift_null_summary": temporal_shift_null,
                "top_future8_feature_tests": temporal_future_tests,
                "top_timing_correlations": temporal_timing_corr,
                "guardrail": first_summary(temporal_directionality, "guardrail"),
            },
            "balanced_spatial_front_propagation_audit": {
                "n_nodes": first_summary(balanced_spatial_propagation, "n_nodes"),
                "n_edges": first_summary(balanced_spatial_propagation, "n_edges"),
                "n_cycles": first_summary(balanced_spatial_propagation, "n_cycles"),
                "n_source_stems": first_summary(balanced_spatial_propagation, "n_source_stems"),
                "edge_counts": first_summary(balanced_spatial_propagation, "edge_counts", {}),
                "top_homophily_tests": spatial_prop_homophily,
                "top_feature_autocorrelation_tests": spatial_prop_autocorr,
                "top_lag_feature_label_tests": spatial_prop_lag,
                "distance_gradient_tests": spatial_prop_distance,
                "guardrail": first_summary(balanced_spatial_propagation, "guardrail"),
            },
            "front_diffusion_guardrail": first_summary(balanced_future_fronts, "diffusion_guardrail"),
            "guardrail": first_summary(balanced_future_physics, "guardrail"),
        },
        "masked_video_embedding_audit": {
            "n_manifest_rows": first_summary(masked_video_embedding, "n_manifest_rows"),
            "n_embedding_rows": first_summary(masked_video_embedding, "n_embedding_rows"),
            "cohort_counts": first_summary(masked_video_embedding, "cohort_counts", {}),
            "particle_region_method": first_summary(masked_video_embedding, "particle_region_method"),
            "pca_summary": first_summary(masked_video_embedding, "pca_summary", {}),
            "target_metrics": masked_video_metrics,
            "balanced_future_label_permutation_null": masked_video_null,
            "top_feature_tests": masked_video_feature_tests,
            "cluster_summary": masked_video_clusters,
            "guardrail": first_summary(masked_video_embedding, "guardrail"),
        },
        "learned_video_residual_embedding_audit": {
            "n_embedding_rows": first_summary(learned_video_residual_embedding, "n_embedding_rows"),
            "n_cycles": first_summary(learned_video_residual_embedding, "n_cycles"),
            "embedding_cohort_counts": first_summary(learned_video_residual_embedding, "embedding_cohort_counts", {}),
            "training": first_summary(learned_video_residual_embedding, "training", {}),
            "feature_set_sizes": first_summary(learned_video_residual_embedding, "feature_set_sizes", {}),
            "top_classification_metrics": learned_residual_class,
            "top_regression_metrics": learned_residual_reg,
            "top_feature_set_deltas": learned_residual_deltas,
            "guardrail": first_summary(learned_video_residual_embedding, "guardrail"),
        },
        "residual_dictionary_embedding_audit": {
            "n_embedding_rows": first_summary(residual_dictionary_embedding, "n_embedding_rows"),
            "n_cycles": first_summary(residual_dictionary_embedding, "n_cycles"),
            "embedding_cohort_counts": first_summary(residual_dictionary_embedding, "embedding_cohort_counts", {}),
            "dictionary": first_summary(residual_dictionary_embedding, "dictionary", {}),
            "feature_set_sizes": first_summary(residual_dictionary_embedding, "feature_set_sizes", {}),
            "top_classification_metrics": residual_dict_class,
            "top_regression_metrics": residual_dict_reg,
            "top_feature_set_deltas": residual_dict_deltas,
            "guardrail": first_summary(residual_dictionary_embedding, "guardrail"),
        },
        "echem_residual_dictionary_fusion_audit": {
            "n_rows": first_summary(echem_residual_dictionary_fusion, "n_rows"),
            "n_cycles": first_summary(echem_residual_dictionary_fusion, "n_cycles"),
            "embedding_cohort_counts": first_summary(echem_residual_dictionary_fusion, "embedding_cohort_counts", {}),
            "feature_set_sizes": first_summary(echem_residual_dictionary_fusion, "feature_set_sizes", {}),
            "top_classification_metrics": echem_resdict_class,
            "top_regression_metrics": echem_resdict_reg,
            "top_feature_set_deltas": echem_resdict_deltas,
            "guardrail": first_summary(echem_residual_dictionary_fusion, "guardrail"),
        },
        "echem_conditioned_residual_dictionary": {
            "n_rows": first_summary(echem_conditioned_residual_dictionary, "n_rows"),
            "n_cycles": first_summary(echem_conditioned_residual_dictionary, "n_cycles"),
            "n_sources": first_summary(echem_conditioned_residual_dictionary, "n_sources"),
            "n_residual_features": first_summary(echem_conditioned_residual_dictionary, "n_residual_features"),
            "n_conditioning_features": first_summary(echem_conditioned_residual_dictionary, "n_conditioning_features"),
            "feature_set_sizes": first_summary(echem_conditioned_residual_dictionary, "feature_set_sizes", {}),
            "top_metrics": echem_cond_resdict_metrics,
            "top_deltas": echem_cond_resdict_deltas,
            "top_context_fit_metrics": echem_cond_resdict_context,
            "guardrail": first_summary(echem_conditioned_residual_dictionary, "guardrail"),
        },
        "conditioned_residual_physics_atlas": {
            "n_rows": first_summary(conditioned_residual_physics_atlas, "n_rows"),
            "n_cycles": first_summary(conditioned_residual_physics_atlas, "n_cycles"),
            "n_sources": first_summary(conditioned_residual_physics_atlas, "n_sources"),
            "n_physics_features": first_summary(conditioned_residual_physics_atlas, "n_physics_features"),
            "physics_category_counts": first_summary(conditioned_residual_physics_atlas, "physics_category_counts", {}),
            "n_conditioned_residual_modes_per_split": first_summary(conditioned_residual_physics_atlas, "n_conditioned_residual_modes_per_split", {}),
            "top_interpretable_source_centered_alignments": cond_atlas_align,
            "top_target_tests": cond_atlas_targets,
            "top_modes": cond_atlas_modes,
            "category_summary": cond_atlas_categories,
            "guardrail": first_summary(conditioned_residual_physics_atlas, "guardrail"),
        },
        "acquisition_residualized_video_physics_benchmark": {
            "n_rows": first_summary(acquisition_residualized_video, "n_rows"),
            "n_cycles": first_summary(acquisition_residualized_video, "n_cycles"),
            "feature_group_sizes": first_summary(acquisition_residualized_video, "feature_group_sizes", {}),
            "residualized_video_group_sizes": first_summary(acquisition_residualized_video, "residualized_video_group_sizes", {}),
            "top_metrics": acq_resid_metrics,
            "top_feature_set_deltas": acq_resid_deltas,
            "top_context_residual_feature_tests": acq_resid_tests,
            "guardrail": first_summary(acquisition_residualized_video, "guardrail"),
        },
        "acquisition_residualized_video_echem_warning": {
            "n_rows": first_summary(acquisition_residualized_video_echem, "n_rows"),
            "n_cycles": first_summary(acquisition_residualized_video_echem, "n_cycles"),
            "n_sources": first_summary(acquisition_residualized_video_echem, "n_sources"),
            "feature_set_sizes": first_summary(acquisition_residualized_video_echem, "feature_set_sizes", {}),
            "top_metrics": acq_echem_metrics,
            "top_deltas": acq_echem_deltas,
            "strict_cycle_balanced_source_cohort_metrics": first_summary(acquisition_residualized_video_echem, "strict_cycle_balanced_source_cohort_metrics", []),
            "strict_cycle_balanced_source_cohort_deltas": first_summary(acquisition_residualized_video_echem, "strict_cycle_balanced_source_cohort_deltas", []),
            "guardrail": first_summary(acquisition_residualized_video_echem, "guardrail"),
        },
        "residualized_future8_video_physics_benchmark": {
            "n_rows": first_summary(residualized_future8_video_physics, "n_rows"),
            "n_cycles": first_summary(residualized_future8_video_physics, "n_cycles"),
            "n_sources": first_summary(residualized_future8_video_physics, "n_sources"),
            "target": first_summary(residualized_future8_video_physics, "target"),
            "feature_set_sizes": first_summary(residualized_future8_video_physics, "feature_set_sizes", {}),
            "decision": future8_decision,
            "top_metrics": future8_top_metrics,
            "key_deltas": future8_key_deltas,
            "guardrail": first_summary(residualized_future8_video_physics, "guardrail"),
        },
        "source_balanced_pre_event_observable_forecast": {
            "n_rows": first_summary(source_balanced_pre_event_observable_forecast, "n_rows"),
            "n_cycles": first_summary(source_balanced_pre_event_observable_forecast, "n_cycles"),
            "n_sources": first_summary(source_balanced_pre_event_observable_forecast, "n_sources"),
            "prefix_fraction": first_summary(source_balanced_pre_event_observable_forecast, "prefix_fraction"),
            "targets": first_summary(source_balanced_pre_event_observable_forecast, "targets", []),
            "feature_set_sizes": first_summary(source_balanced_pre_event_observable_forecast, "feature_set_sizes", {}),
            "decision": observable_forecast_decision,
            "source_heldout_top_metrics": observable_forecast_source_metrics,
            "event_relative_diagnostics": observable_forecast_event_diag,
            "incremental_over_echem": observable_forecast_incremental,
            "guardrail": first_summary(source_balanced_pre_event_observable_forecast, "guardrail"),
        },
        "source_balanced_pre_event_transport_kinetic_fusion_audit": {
            "n_rows": first_summary(source_balanced_pre_event_transport_kinetic_fusion, "n_rows"),
            "n_cycles": first_summary(source_balanced_pre_event_transport_kinetic_fusion, "n_cycles"),
            "n_sources": first_summary(source_balanced_pre_event_transport_kinetic_fusion, "n_sources"),
            "event_relative_bin_counts": first_summary(source_balanced_pre_event_transport_kinetic_fusion, "event_relative_bin_counts", {}),
            "feature_set_sizes": first_summary(source_balanced_pre_event_transport_kinetic_fusion, "feature_set_sizes", {}),
            "score_columns": first_summary(source_balanced_pre_event_transport_kinetic_fusion, "score_columns", []),
            "best_event_tests_by_target": transport_fusion_best_by_target,
            "top_event_tests": transport_fusion_tests,
            "top_leave_source_models": transport_fusion_models,
            "top_ranked_candidates": transport_fusion_candidates,
            "outputs": first_summary(source_balanced_pre_event_transport_kinetic_fusion, "outputs", {}),
            "guardrail": first_summary(source_balanced_pre_event_transport_kinetic_fusion, "guardrail"),
        },
        "source_balanced_pre_event_optical_flow_transport_audit": {
            "n_input_rows": first_summary(source_balanced_pre_event_optical_flow_transport, "n_input_rows"),
            "n_ok": first_summary(source_balanced_pre_event_optical_flow_transport, "n_ok"),
            "n_failed": first_summary(source_balanced_pre_event_optical_flow_transport, "n_failed"),
            "n_cycles": first_summary(source_balanced_pre_event_optical_flow_transport, "n_cycles"),
            "n_sources": first_summary(source_balanced_pre_event_optical_flow_transport, "n_sources"),
            "flow_method": first_summary(source_balanced_pre_event_optical_flow_transport, "flow_method"),
            "event_relative_bin_counts": first_summary(source_balanced_pre_event_optical_flow_transport, "event_relative_bin_counts", {}),
            "method_summary": first_summary(source_balanced_pre_event_optical_flow_transport, "method_summary", []),
            "top_event_tests": optical_flow_top_tests,
            "top_source_residual_event_tests": optical_flow_source_resid_tests,
            "best_source_residual_test": optical_flow_source_resid_best,
            "guardrail": first_summary(source_balanced_pre_event_optical_flow_transport, "guardrail"),
        },
        "source_balanced_transport_mechanism_dossier": {
            "n_rows": first_summary(source_balanced_transport_mechanism, "n_rows"),
            "n_cycles": first_summary(source_balanced_transport_mechanism, "n_cycles"),
            "n_sources": first_summary(source_balanced_transport_mechanism, "n_sources"),
            "n_immediate_review": first_summary(source_balanced_transport_mechanism, "n_immediate_review"),
            "n_diffusion_claim_candidates": first_summary(source_balanced_transport_mechanism, "n_diffusion_claim_candidates"),
            "tier_counts": transport_mechanism_tiers,
            "top40_event_bin_counts": first_summary(source_balanced_transport_mechanism, "top40_event_bin_counts", {}),
            "top_candidate": transport_mechanism_top,
            "source_summary_top": transport_mechanism_sources,
            "outputs": first_summary(source_balanced_transport_mechanism, "outputs", {}),
            "guardrail": first_summary(source_balanced_transport_mechanism, "guardrail"),
        },
        "source_balanced_transport_mechanism_falsification_audit": {
            "n_rows": first_summary(source_balanced_transport_mechanism_falsification, "n_rows"),
            "n_cycles": first_summary(source_balanced_transport_mechanism_falsification, "n_cycles"),
            "n_sources": first_summary(source_balanced_transport_mechanism_falsification, "n_sources"),
            "n_near_pre": first_summary(source_balanced_transport_mechanism_falsification, "n_near_pre"),
            "n_matched_pairs": first_summary(source_balanced_transport_mechanism_falsification, "n_matched_pairs"),
            "lead_event_tests_for_transport_mechanism_score": transport_mechanism_falsification_event,
            "lead_pair_tests_for_transport_mechanism_score": transport_mechanism_falsification_pair,
            "lead_source_test_for_transport_mechanism_score": transport_mechanism_falsification_source,
            "topk_enrichment": transport_mechanism_falsification_topk,
            "outputs": first_summary(source_balanced_transport_mechanism_falsification, "outputs", {}),
            "guardrail": first_summary(source_balanced_transport_mechanism_falsification, "guardrail"),
        },
        "source_heldout_event_rank_transfer_audit": {
            "n_input_rows": first_summary(source_heldout_event_rank_transfer, "n_input_rows"),
            "n_sources": first_summary(source_heldout_event_rank_transfer, "n_sources"),
            "n_candidate_features": first_summary(source_heldout_event_rank_transfer, "n_candidate_features"),
            "n_folds": first_summary(source_heldout_event_rank_transfer, "n_folds"),
            "targets": first_summary(source_heldout_event_rank_transfer, "targets", []),
            "transfer_score_tests": heldout_rank_transfer_transfer_tests,
            "best_score_tests_by_target": heldout_rank_transfer_tests,
            "topk_summary": heldout_rank_transfer_topk,
            "outputs": first_summary(source_heldout_event_rank_transfer, "outputs", {}),
            "guardrail": first_summary(source_heldout_event_rank_transfer, "guardrail"),
        },
        "pre_event_temporal_dose_response_audit": {
            "overall_status": first_summary(pre_event_temporal_dose_response, "overall_status"),
            "n_input_rows": first_summary(pre_event_temporal_dose_response, "n_input_rows"),
            "n_pre_event_rows": first_summary(pre_event_temporal_dose_response, "n_pre_event_rows"),
            "n_pre_event_cycles": first_summary(pre_event_temporal_dose_response, "n_pre_event_cycles"),
            "n_pre_event_sources": first_summary(pre_event_temporal_dose_response, "n_pre_event_sources"),
            "distance_bin_counts": first_summary(pre_event_temporal_dose_response, "distance_bin_counts", {}),
            "key_feature_tests": temporal_dose_key_tests,
            "top_source_centered_tests": temporal_dose_centered,
            "top_source_slope_tests": temporal_dose_slopes,
            "outputs": first_summary(pre_event_temporal_dose_response, "outputs", {}),
            "guardrail": first_summary(pre_event_temporal_dose_response, "guardrail"),
        },
        "targeted_densification_qc_plan": {
            "overall_status": first_summary(targeted_densification_qc, "overall_status"),
            "n_cycle_rows": first_summary(targeted_densification_qc, "n_cycle_rows"),
            "n_sources": first_summary(targeted_densification_qc, "n_sources"),
            "n_uncovered_cycles": first_summary(targeted_densification_qc, "n_uncovered_cycles"),
            "n_low_roi_cycles_lt10": first_summary(targeted_densification_qc, "n_low_roi_cycles_lt10"),
            "n_roi_queue_rows": first_summary(targeted_densification_qc, "n_roi_queue_rows"),
            "action_counts": targeted_densification_action_counts,
            "roi_origin_counts": targeted_densification_origin_counts,
            "source_plan_top": targeted_densification_source_plan,
            "top_cycle_actions": targeted_densification_cycle_actions,
            "top_roi_actions": targeted_densification_roi_actions,
            "outputs": first_summary(targeted_densification_qc, "outputs", {}),
            "guardrail": first_summary(targeted_densification_qc, "guardrail"),
        },
        "source_domain_video_echem_adaptation_audit": {
            "n_rows": first_summary(source_domain_video_echem, "n_rows"),
            "n_cycles": first_summary(source_domain_video_echem, "n_cycles"),
            "n_sources": first_summary(source_domain_video_echem, "n_sources"),
            "feature_set_sizes": first_summary(source_domain_video_echem, "feature_set_sizes", {}),
            "top_metrics": source_domain_metrics,
            "top_deltas": source_domain_deltas,
            "source_summary": source_domain_sources,
            "guardrail": first_summary(source_domain_video_echem, "guardrail"),
        },
        "source_balanced_video_echem_transfer_audit": {
            "n_rows": first_summary(source_balanced_video_echem, "n_rows"),
            "n_cycles": first_summary(source_balanced_video_echem, "n_cycles"),
            "n_sources": first_summary(source_balanced_video_echem, "n_sources"),
            "modes": first_summary(source_balanced_video_echem, "modes", []),
            "feature_set_sizes": first_summary(source_balanced_video_echem, "feature_set_sizes", {}),
            "top_metrics": source_balanced_metrics,
            "top_deltas": source_balanced_deltas,
            "source_summary": source_balanced_sources,
            "guardrail": first_summary(source_balanced_video_echem, "guardrail"),
        },
        "source_invariant_video_echem_transfer_audit": {
            "n_rows": first_summary(source_invariant_video_echem, "n_rows"),
            "n_cycles": first_summary(source_invariant_video_echem, "n_cycles"),
            "n_sources": first_summary(source_invariant_video_echem, "n_sources"),
            "methods": first_summary(source_invariant_video_echem, "methods", []),
            "feature_set_sizes": first_summary(source_invariant_video_echem, "feature_set_sizes", {}),
            "top_metrics": source_invariant_metrics,
            "top_deltas": source_invariant_deltas,
            "source_summary": source_invariant_sources,
            "guardrail": first_summary(source_invariant_video_echem, "guardrail"),
        },
        "source_invariant_physical_family_audit": {
            "n_rows": first_summary(source_invariant_family, "n_rows"),
            "n_cycles": first_summary(source_invariant_family, "n_cycles"),
            "n_sources": first_summary(source_invariant_family, "n_sources"),
            "methods": first_summary(source_invariant_family, "methods", []),
            "feature_family_sizes": first_summary(source_invariant_family, "feature_family_sizes", {}),
            "top_metrics": source_family_metrics,
            "top_deltas": source_family_deltas,
            "top_source_confounded_features": source_family_confounds,
            "guardrail": first_summary(source_invariant_family, "guardrail"),
        },
        "source_invariant_interpretable_feature_audit": {
            "n_rows": first_summary(source_invariant_interpretable, "n_rows"),
            "n_cycles": first_summary(source_invariant_interpretable, "n_cycles"),
            "n_sources": first_summary(source_invariant_interpretable, "n_sources"),
            "target": first_summary(source_invariant_interpretable, "target"),
            "methods": first_summary(source_invariant_interpretable, "methods", []),
            "feature_family_sizes": first_summary(source_invariant_interpretable, "feature_family_sizes", {}),
            "top_univariate_features": source_interpretable_univariate,
            "top_single_feature_transfer": source_interpretable_single,
            "top_combo_transfer": source_interpretable_combo,
            "top_set_metrics": source_interpretable_sets,
            "guardrail": first_summary(source_invariant_interpretable, "guardrail"),
        },
        "exact_feature_mechanism_consistency_audit": {
            "n_rows": first_summary(exact_feature_mechanism, "n_rows"),
            "n_cycles": first_summary(exact_feature_mechanism, "n_cycles"),
            "n_sources": first_summary(exact_feature_mechanism, "n_sources"),
            "target": first_summary(exact_feature_mechanism, "target"),
            "top_target_metrics": exact_mech_target_metrics,
            "contraction_related_correlations": exact_mech_contraction,
            "top_stratum_tests": exact_mech_strata,
            "guardrail": first_summary(exact_feature_mechanism, "guardrail"),
        },
        "invariant_physics_rule_discovery": {
            "n_rows": first_summary(invariant_physics_rules, "n_rows"),
            "n_eval_rows": first_summary(invariant_physics_rules, "n_eval_rows"),
            "n_cycles": first_summary(invariant_physics_rules, "n_cycles"),
            "n_sources": first_summary(invariant_physics_rules, "n_sources"),
            "target": first_summary(invariant_physics_rules, "target"),
            "positive_rate": first_summary(invariant_physics_rules, "positive_rate"),
            "feature_family_sizes": first_summary(invariant_physics_rules, "feature_family_sizes", {}),
            "n_candidate_rules": first_summary(invariant_physics_rules, "n_candidate_rules"),
            "best_rule": invariant_rule_best,
            "top_rules": invariant_rule_top,
            "top_oriented_features": invariant_rule_features,
            "guardrail": first_summary(invariant_physics_rules, "guardrail"),
        },
        "signed_optical_loss_mechanism_audit": {
            "n_rows": first_summary(signed_optical_loss, "n_rows"),
            "n_eval_rows": first_summary(signed_optical_loss, "n_eval_rows"),
            "n_cycles": first_summary(signed_optical_loss, "n_cycles"),
            "n_sources": first_summary(signed_optical_loss, "n_sources"),
            "axes": first_summary(signed_optical_loss, "axes", []),
            "axis_input_features": first_summary(signed_optical_loss, "axis_input_features", {}),
            "top_future16_axis_tests": signed_loss_tests,
            "top_axis_model_metrics": signed_loss_models,
            "mechanism_mode_summary": signed_loss_modes,
            "source_summary": signed_loss_sources,
            "top_candidates": signed_loss_candidates,
            "guardrail": first_summary(signed_optical_loss, "guardrail"),
        },
        "signed_loss_source_robustness_audit": {
            "n_rows": first_summary(signed_loss_source_robustness, "n_rows"),
            "n_labeled_future16": first_summary(signed_loss_source_robustness, "n_labeled_future16"),
            "n_cycles": first_summary(signed_loss_source_robustness, "n_cycles"),
            "n_sources": first_summary(signed_loss_source_robustness, "n_sources"),
            "axes": first_summary(signed_loss_source_robustness, "axes", []),
            "transforms": first_summary(signed_loss_source_robustness, "transforms", []),
            "key_future16_metrics": signed_source_key_metrics,
            "top_future16_metrics": signed_source_top_metrics,
            "largest_negative_source_influence": signed_source_influence,
            "source_summary": signed_source_summary,
            "guardrail": first_summary(signed_loss_source_robustness, "guardrail"),
        },
        "echem_optical_source_residual_audit": {
            "n_rows": first_summary(echem_optical_source_residual, "n_rows"),
            "n_labeled_future16": first_summary(echem_optical_source_residual, "n_labeled_future16"),
            "n_cycles": first_summary(echem_optical_source_residual, "n_cycles"),
            "n_sources": first_summary(echem_optical_source_residual, "n_sources"),
            "top_future16_direct_metrics": echem_optical_direct,
            "top_future16_model_metrics": echem_optical_models,
            "top_rules": echem_optical_rules,
            "top_candidates": echem_optical_candidates,
            "guardrail": first_summary(echem_optical_source_residual, "guardrail"),
        },
        "agentic_current_hypothesis_tournament": {
            "n_hypotheses": first_summary(agentic_current, "n_hypotheses"),
            "top_hypothesis": first_summary(agentic_current, "top_hypothesis", {}),
            "top_three": agentic_current_top,
            "experiment_specs": agentic_current_specs,
            "paper_inspiration": first_summary(agentic_current, "paper_inspiration", {}),
            "guardrail": first_summary(agentic_current, "guardrail"),
        },
        "active_learning_qc_prioritization": {
            "n_candidate_rows": first_summary(active_learning_qc, "n_candidate_rows"),
            "n_cycles": first_summary(active_learning_qc, "n_cycles"),
            "n_rows_with_visual_asset": first_summary(active_learning_qc, "n_rows_with_visual_asset"),
            "tier_counts": active_qc_tiers,
            "reason_counts": active_qc_reasons,
            "top_priority_rows": active_qc_top,
            "top_cycles": active_qc_cycles,
            "guardrail": first_summary(active_learning_qc, "guardrail"),
        },
        "automatic_qc_triage_surrogate": {
            "n_candidates": first_summary(automatic_qc_triage, "n_candidates"),
            "manual_status_counts": first_summary(automatic_qc_triage, "manual_status_counts", {}),
            "likely_interpretable_count": first_summary(automatic_qc_triage, "likely_interpretable_count"),
            "artifact_risk_count": first_summary(automatic_qc_triage, "artifact_risk_count"),
            "diffusion_guardrail_count": first_summary(automatic_qc_triage, "diffusion_guardrail_count"),
            "tier_summary": auto_qc_triage_tiers,
            "top_likely_interpretable": auto_qc_likely,
            "top_artifact_risk": auto_qc_artifact,
            "top_feature_tests": auto_qc_tests,
            "top_correlations": auto_qc_corr,
            "guardrail": first_summary(automatic_qc_triage, "guardrail"),
        },
        "qc_decision_evidence_ledger": {
            "n_candidates": first_summary(qc_decision_ledger, "n_candidates"),
            "manual_status_counts": first_summary(qc_decision_ledger, "manual_status_counts", {}),
            "decision_action_counts": qc_decision_actions,
            "n_visual_asset_candidates": first_summary(qc_decision_ledger, "n_visual_asset_candidates"),
            "top_review_queue": qc_decision_top,
            "top_possible_accept_queue": qc_decision_accept,
            "top_artifact_queue": qc_decision_artifact,
            "cycle_summary": qc_decision_cycles,
            "guardrail": first_summary(qc_decision_ledger, "guardrail"),
        },
        "multicohort_future_drop_model": {
            "n_roi_rows": first_summary(multicohort_future_drop, "n_roi_rows"),
            "n_selected_rows": first_summary(multicohort_future_drop, "n_selected_rows"),
            "n_transfer_ranked_rows": first_summary(multicohort_future_drop, "n_transfer_ranked_rows"),
            "n_features": first_summary(multicohort_future_drop, "n_features"),
            "rf_trees": first_summary(multicohort_future_drop, "rf_trees"),
            "label_counts": first_summary(multicohort_future_drop, "label_counts", []),
            "cycle_group_oof_metrics": multicohort_oof,
            "permutation_null": multicohort_null,
            "top_feature_tests": multicohort_features,
            "top_feature_importance": multicohort_importance,
            "leave_cohort_eval": multicohort_leave,
            "guardrail": first_summary(multicohort_future_drop, "guardrail"),
        },
        "cross_cohort_rollout_transfer_audit": {
            "n_selected_roi": first_summary(cross_cohort_rollout, "n_selected_roi"),
            "n_transfer_ranked_roi": first_summary(cross_cohort_rollout, "n_transfer_ranked_roi"),
            "rank": first_summary(cross_cohort_rollout, "rank"),
            "train_fraction": first_summary(cross_cohort_rollout, "train_fraction"),
            "model_spectral_radius": first_summary(cross_cohort_rollout, "model_spectral_radius", {}),
            "domain_shift": cross_cohort_shift,
            "top_correlations": cross_cohort_corr,
            "top_transfer_ranked_difficult_rois": cross_cohort_difficult,
            "guardrail": first_summary(cross_cohort_rollout, "guardrail"),
        },
        "masked_residual_state_transfer_warning": {
            "n_anchor_cycles": first_summary(masked_residual_transfer, "n_anchor_cycles"),
            "n_full_cycles": first_summary(masked_residual_transfer, "n_full_cycles"),
            "n_signature_features": first_summary(masked_residual_transfer, "n_signature_features"),
            "n_permutation": first_summary(masked_residual_transfer, "n_permutation"),
            "anchor_loo_summary": masked_residual_transfer_anchor,
            "top_signature_features": masked_residual_transfer_features,
            "top_transfer_coefficients": masked_residual_transfer_coeff,
            "top_target_tests": masked_residual_transfer_tests,
            "temporal_block_auc": masked_residual_transfer_temporal,
            "top_correlations": masked_residual_transfer_corr,
            "top_ranked_cycles": masked_residual_transfer_ranked,
            "guardrail": first_summary(masked_residual_transfer, "guardrail"),
        },
        "masked_residual_transition_timing": {
            "n_roi": first_summary(masked_residual_timing, "n_roi"),
            "n_roi_method_rows": first_summary(masked_residual_timing, "n_roi_method_rows"),
            "n_permutation": first_summary(masked_residual_timing, "n_permutation"),
            "top_alignment_tests": masked_residual_timing_align,
            "top_event_control_tests": masked_residual_timing_tests,
            "top_correlations": masked_residual_timing_corr,
            "top_near_transition_residual_rois": masked_residual_timing_top,
            "guardrail": first_summary(masked_residual_timing, "guardrail"),
        },
        "masked_rollout_cycle_warning": {
            "n_roi_cycles": first_summary(masked_cycle_warning, "n_roi_cycles"),
            "n_rollout_features_tested": first_summary(masked_cycle_warning, "n_rollout_features_tested"),
            "n_permutation": first_summary(masked_cycle_warning, "n_permutation"),
            "target_positive_counts": masked_cycle_warning_targets,
            "top_target_tests": masked_cycle_warning_tests,
            "top_correlations": masked_cycle_warning_corr,
            "top_warning_cycles": masked_cycle_warning_top,
            "guardrail": first_summary(masked_cycle_warning, "guardrail"),
        },
        "masked_roi_rollout_audit": {
            "n_roi": first_summary(masked_rollout, "n_roi"),
            "n_frame_metric_rows": first_summary(masked_rollout, "n_frame_metric_rows"),
            "rank": first_summary(masked_rollout, "rank"),
            "train_fraction": first_summary(masked_rollout, "train_fraction"),
            "dmd_spectral_radius": first_summary(masked_rollout, "dmd_spectral_radius"),
            "best_method_counts_inside_particle": masked_rollout_best_counts,
            "method_summary": masked_rollout_methods,
            "top_event_control_tests": masked_rollout_tests,
            "top_correlations": masked_rollout_corr,
            "top_particle_difficulty_rois": masked_rollout_difficult,
            "guardrail": first_summary(masked_rollout, "guardrail"),
        },
        "probabilistic_rollout_calibration": {
            "n_frame_rows": first_summary(rollout_calibration, "n_frame_rows"),
            "n_roi_method_rows": first_summary(rollout_calibration, "n_roi_method_rows"),
            "n_roi": first_summary(rollout_calibration, "n_roi"),
            "n_event_reference_cycles": first_summary(rollout_calibration, "n_event_reference_cycles"),
            "near_transition_frame_fraction": first_summary(rollout_calibration, "near_transition_frame_fraction"),
            "coverage_summary": rollout_calibration_coverage,
            "top_transition_error_tests": rollout_calibration_tests,
            "top_calibration_physics_correlations": rollout_calibration_corr,
            "top_undercovered_roi_method_rows": rollout_calibration_top,
            "guardrail": first_summary(rollout_calibration, "guardrail"),
        },
        "physics_consistency_claim_matrix": {
            "n_roi": first_summary(physics_consistency, "n_roi"),
            "n_cycles": first_summary(physics_consistency, "n_cycles"),
            "tier_counts": first_summary(physics_consistency, "tier_counts", {}),
            "claim_readiness_counts": first_summary(physics_consistency, "claim_readiness_counts", {}),
            "manual_qc_accepted": first_summary(physics_consistency, "manual_qc_accepted"),
            "calibration_evidence": first_summary(physics_consistency, "calibration_evidence", {}),
            "top_consistency_rows": physics_consistency_top,
            "top_event_tests": physics_consistency_tests,
            "guardrail": first_summary(physics_consistency, "guardrail"),
        },
        "cycle_hazard_warning_audit": {
            "target": first_summary(cycle_hazard_warning, "target"),
            "n_cycles": first_summary(cycle_hazard_warning, "n_cycles"),
            "n_event_cycles": first_summary(cycle_hazard_warning, "n_event_cycles"),
            "event_cycles": first_summary(cycle_hazard_warning, "event_cycles", []),
            "purge_cycles": first_summary(cycle_hazard_warning, "purge_cycles"),
            "min_train": first_summary(cycle_hazard_warning, "min_train"),
            "feature_group_counts": first_summary(cycle_hazard_warning, "feature_group_counts", {}),
            "feature_set_summary": cycle_hazard_feature_sets,
            "best_feature_set": first_summary(cycle_hazard_warning, "best_feature_set"),
            "permutation_null": cycle_hazard_null,
            "lead_time_summary": cycle_hazard_lead,
            "top_group_ablation": cycle_hazard_ablation,
            "top_probability_correlations": cycle_hazard_corr,
            "guardrail": first_summary(cycle_hazard_warning, "guardrail"),
        },
        "cycle_state_space_transition_audit": {
            "n_cycles": first_summary(cycle_state_space, "n_cycles"),
            "n_echem_shape_cycles_joined": first_summary(cycle_state_space, "n_echem_shape_cycles_joined"),
            "n_features_used": first_summary(cycle_state_space, "n_features_used"),
            "feature_group_counts": first_summary(cycle_state_space, "feature_group_counts", {}),
            "chosen_k": first_summary(cycle_state_space, "chosen_k"),
            "best_silhouette": first_summary(cycle_state_space, "best_silhouette"),
            "pca_explained_variance": first_summary(cycle_state_space, "pca_explained_variance", []),
            "degradation_axis_oriented_to": first_summary(cycle_state_space, "degradation_axis_oriented_to"),
            "degradation_axis_orientation_rho": first_summary(cycle_state_space, "degradation_axis_orientation_rho"),
            "future_drop_classifier": cycle_state_classifier,
            "top_future_drop_tests": cycle_state_tests,
            "top_cycle_state_correlations": cycle_state_corr,
            "top_state_clusters": cycle_state_clusters,
            "top_transitions": cycle_state_transitions,
            "top_pc_loadings": cycle_state_loadings,
            "guardrail": first_summary(cycle_state_space, "guardrail"),
        },
        "weak_label_degradation_benchmark": {
            "n_roi_rows": first_summary(weak_label_benchmark, "n_roi_rows"),
            "n_trainable_weak_label_rows": first_summary(weak_label_benchmark, "n_trainable_weak_label_rows"),
            "n_positive_weak_labels": first_summary(weak_label_benchmark, "n_positive_weak_labels"),
            "n_negative_weak_labels": first_summary(weak_label_benchmark, "n_negative_weak_labels"),
            "label_counts": first_summary(weak_label_benchmark, "label_counts", {}),
            "confidence_counts": first_summary(weak_label_benchmark, "confidence_counts", {}),
            "leakage_audit": weak_label_leakage,
            "top_positive_training_rows": weak_label_top_pos,
            "top_negative_training_rows": weak_label_top_neg,
            "guardrail": first_summary(weak_label_benchmark, "guardrail"),
        },
        "particle_mask_stability_audit": {
            "n_roi": first_summary(particle_mask, "n_roi"),
            "n_frames_total": first_summary(particle_mask, "n_frames_total"),
            "overall": particle_mask_overall,
            "role_summary": particle_mask_role_summary,
            "top_event_control_tests": particle_mask_tests,
            "top_correlations": particle_mask_corr,
            "highest_instability_rois": particle_mask_top,
            "method": first_summary(particle_mask, "method", {}),
        },
        "particle_mask_history_fallback_audit": {
            "overall_status": first_summary(particle_mask_history_fallback, "overall_status"),
            "n_input_rows": first_summary(particle_mask_history_fallback, "n_input_rows"),
            "n_ok": first_summary(particle_mask_history_fallback, "n_ok"),
            "n_failures": first_summary(particle_mask_history_fallback, "n_failures"),
            "n_sources": first_summary(particle_mask_history_fallback, "n_sources"),
            "median_fallback_frame_fraction": first_summary(particle_mask_history_fallback, "median_fallback_frame_fraction"),
            "q90_fallback_frame_fraction": first_summary(particle_mask_history_fallback, "q90_fallback_frame_fraction"),
            "median_history_iou": first_summary(particle_mask_history_fallback, "median_history_iou"),
            "median_centroid_jitter_q90_px": first_summary(particle_mask_history_fallback, "median_centroid_jitter_q90_px"),
            "median_blur_ratio_q10": first_summary(particle_mask_history_fallback, "median_blur_ratio_q10"),
            "near_vs_non_median_fallback_diff": first_summary(particle_mask_history_fallback, "near_vs_non_median_fallback_diff"),
            "top_event_tests": particle_mask_history_tests,
            "source_summary_top": particle_mask_history_sources,
            "high_fallback_rois": particle_mask_history_high,
            "outputs": first_summary(particle_mask_history_fallback, "outputs", {}),
            "guardrail": first_summary(particle_mask_history_fallback, "guardrail"),
        },
        "history_fallback_masked_rollout_ablation": {
            "overall_status": first_summary(history_fallback_rollout_ablation, "overall_status"),
            "n_input_rows": first_summary(history_fallback_rollout_ablation, "n_input_rows"),
            "n_ok": first_summary(history_fallback_rollout_ablation, "n_ok"),
            "n_failures": first_summary(history_fallback_rollout_ablation, "n_failures"),
            "n_sources": first_summary(history_fallback_rollout_ablation, "n_sources"),
            "n_cycles": first_summary(history_fallback_rollout_ablation, "n_cycles"),
            "median_fallback_frame_fraction": first_summary(history_fallback_rollout_ablation, "median_fallback_frame_fraction"),
            "median_one_step_adaptive_mse": first_summary(history_fallback_rollout_ablation, "median_one_step_adaptive_mse"),
            "median_one_step_hybrid_mse": first_summary(history_fallback_rollout_ablation, "median_one_step_hybrid_mse"),
            "median_hybrid_minus_adaptive_one_step_mse": first_summary(history_fallback_rollout_ablation, "median_hybrid_minus_adaptive_one_step_mse"),
            "median_latent_gain_history": first_summary(history_fallback_rollout_ablation, "median_latent_gain_history"),
            "median_latent_gain_hybrid": first_summary(history_fallback_rollout_ablation, "median_latent_gain_hybrid"),
            "method_summary": history_fallback_rollout_methods,
            "top_event_tests": history_fallback_rollout_tests,
            "source_summary_top": history_fallback_rollout_sources,
            "outputs": first_summary(history_fallback_rollout_ablation, "outputs", {}),
            "guardrail": first_summary(history_fallback_rollout_ablation, "guardrail"),
        },
        "rollout_front_mode_coupling_audit": {
            "overall_status": first_summary(rollout_front_mode_coupling, "overall_status"),
            "n_rows": first_summary(rollout_front_mode_coupling, "n_rows"),
            "n_sources": first_summary(rollout_front_mode_coupling, "n_sources"),
            "n_cycles": first_summary(rollout_front_mode_coupling, "n_cycles"),
            "n_modes": first_summary(rollout_front_mode_coupling, "n_modes"),
            "top_feature_tests": rollout_front_mode_tests,
            "top_source_residual_correlations": rollout_front_mode_source_corr,
            "top_raw_correlations": rollout_front_mode_raw_corr,
            "mode_summary": rollout_front_mode_modes,
            "top_review_queue": rollout_front_mode_queue,
            "outputs": first_summary(rollout_front_mode_coupling, "outputs", {}),
            "guardrail": first_summary(rollout_front_mode_coupling, "guardrail"),
        },
        "cycle_state_roi_bridge": {
            "n_roi_rows": first_summary(cycle_state_roi_bridge, "n_roi_rows"),
            "n_cycles": first_summary(cycle_state_roi_bridge, "n_cycles"),
            "n_predictors": first_summary(cycle_state_roi_bridge, "n_predictors"),
            "n_targets": first_summary(cycle_state_roi_bridge, "n_targets"),
            "top_row_tests": cycle_state_roi_row_tests,
            "top_reference_centered_tests": cycle_state_roi_centered_tests,
            "top_cycle_collapsed_tests": cycle_state_roi_collapsed_tests,
            "cluster_summary": cycle_state_roi_clusters,
            "guardrail": first_summary(cycle_state_roi_bridge, "guardrail"),
        },
        "cycle_state_mode_frequency_bridge": {
            "n_cycles": first_summary(cycle_state_mode_frequency, "n_cycles"),
            "n_roi_rows": first_summary(cycle_state_mode_frequency, "n_roi_rows"),
            "n_mode_targets": first_summary(cycle_state_mode_frequency, "n_mode_targets"),
            "feature_set_sizes": first_summary(cycle_state_mode_frequency, "feature_set_sizes", {}),
            "best_macro_model": cycle_state_mode_best,
            "context_macro_model": cycle_state_mode_context,
            "best_minus_context_macro_mae_reduction": first_summary(cycle_state_mode_frequency, "best_minus_context_macro_mae_reduction"),
            "top_metrics": cycle_state_mode_metrics,
            "permutation_null": cycle_state_mode_nulls,
            "cluster_summary": cycle_state_mode_clusters,
            "guardrail": first_summary(cycle_state_mode_frequency, "guardrail"),
        },
        "echem_shape_conditioned_roi_front_effects": {
            "n_rows": first_summary(echem_shape_conditioned, "n_rows"),
            "n_event_roi": first_summary(echem_shape_conditioned, "n_event_roi"),
            "n_control_roi": first_summary(echem_shape_conditioned, "n_control_roi"),
            "shape_pca": first_summary(echem_shape_conditioned, "shape_pca", {}),
            "model_summary": echem_shape_model,
            "top_shape_conditioned_event_control_tests": echem_shape_conditioned_tests,
            "top_effect_retention": echem_shape_retention,
            "top_shape_context_fits": echem_shape_context,
            "top_shape_pc_target_correlations": echem_shape_pc_corr,
            "guardrail": first_summary(echem_shape_conditioned, "guardrail"),
        },
        "echem_optical_breakpoint_audit": {
            "n_cycles": first_summary(echem_optical_breakpoint, "n_cycles"),
            "n_features_tested": first_summary(echem_optical_breakpoint, "n_features_tested"),
            "event_cycles": first_summary(echem_optical_breakpoint, "event_cycles", []),
            "n_permutation": first_summary(echem_optical_breakpoint, "n_permutation"),
            "top_event_centered_tests": echem_breakpoint_event,
            "top_future_label_tests": echem_breakpoint_future,
            "top_global_breakpoints": echem_breakpoint_global,
            "event_cycle_breakpoint_ranks": echem_breakpoint_event_ranks,
            "guardrail": first_summary(echem_optical_breakpoint, "guardrail"),
        },
        "echem_optical_regime_atlas": {
            "n_cycles": first_summary(echem_optical_regime, "n_cycles"),
            "n_echem_features": first_summary(echem_optical_regime, "n_echem_features"),
            "n_cycles_missing_echem_shape": first_summary(echem_optical_regime, "n_cycles_missing_echem_shape"),
            "n_cycles_extreme_or_missing_ce": first_summary(echem_optical_regime, "n_cycles_extreme_or_missing_ce"),
            "regime_summary": echem_regime_summary,
            "top_binary_tests": echem_regime_binary,
            "top_correlations": echem_regime_corr,
            "top_cycles": echem_regime_top_cycles,
            "guardrail": first_summary(echem_optical_regime, "guardrail"),
        },
        "echem_conditioned_optical_predictor": {
            "n_cycles": first_summary(echem_conditioned_predictor, "n_cycles"),
            "targets": first_summary(echem_conditioned_predictor, "targets", []),
            "feature_set_sizes": first_summary(echem_conditioned_predictor, "feature_set_sizes", {}),
            "top_metrics": echem_predictor_metrics,
            "top_feature_set_deltas": echem_predictor_deltas,
            "guardrail": first_summary(echem_conditioned_predictor, "guardrail"),
        },
        "echem_conditioned_roi_rollout_front_audit": {
            "n_roi_rows": first_summary(echem_roi_rollout_front, "n_roi_rows"),
            "n_cycles": first_summary(echem_roi_rollout_front, "n_cycles"),
            "targets": first_summary(echem_roi_rollout_front, "targets", []),
            "feature_set_sizes": first_summary(echem_roi_rollout_front, "feature_set_sizes", {}),
            "top_model_metrics": echem_roi_metrics,
            "top_feature_set_deltas": echem_roi_deltas,
            "top_residual_correlations": echem_roi_residual,
            "guardrail": first_summary(echem_roi_rollout_front, "guardrail"),
        },
        "echem_video_embedding_fusion_audit": {
            "n_embedding_rows": first_summary(echem_video_fusion, "n_embedding_rows"),
            "n_cycles": first_summary(echem_video_fusion, "n_cycles"),
            "embedding_cohort_counts": first_summary(echem_video_fusion, "embedding_cohort_counts", {}),
            "classification_targets": first_summary(echem_video_fusion, "classification_targets", []),
            "regression_targets": first_summary(echem_video_fusion, "regression_targets", []),
            "feature_set_sizes": first_summary(echem_video_fusion, "feature_set_sizes", {}),
            "top_classification_metrics": echem_video_class,
            "top_regression_metrics": echem_video_reg,
            "top_feature_set_deltas": echem_video_deltas,
            "guardrail": first_summary(echem_video_fusion, "guardrail"),
        },
        "calibration_claim_risk_register": {
            "n_claim_families": first_summary(calibration_claim_risk, "n_claim_families"),
            "n_source_tables_present": first_summary(calibration_claim_risk, "n_source_tables_present"),
            "claim_status_counts": first_summary(calibration_claim_risk, "claim_status_counts", {}),
            "risk_level_counts": first_summary(calibration_claim_risk, "risk_level_counts", {}),
            "high_risk_claim_families": first_summary(calibration_claim_risk, "high_risk_claim_families", []),
            "calibration_evidence": first_summary(calibration_claim_risk, "calibration_evidence", {}),
            "guardrail": first_summary(calibration_claim_risk, "guardrail"),
        },
        "apparent_diffusion_calibration_bounds": {
            "n_threshold_rows": first_summary(apparent_diffusion_calibration, "n_threshold_rows"),
            "n_roi": first_summary(apparent_diffusion_calibration, "n_roi"),
            "n_cycles": first_summary(apparent_diffusion_calibration, "n_cycles"),
            "n_roi_with_h5_timing": first_summary(apparent_diffusion_calibration, "n_roi_with_h5_timing"),
            "pixel_size_um_assumptions": first_summary(apparent_diffusion_calibration, "pixel_size_um_assumptions", []),
            "median_roi_elapsed_to_h5_median_ratio": first_summary(apparent_diffusion_calibration, "median_roi_elapsed_to_h5_median_ratio"),
            "median_q70_apparent_D_h5median_um2_per_s": first_summary(apparent_diffusion_calibration, "median_q70_apparent_D_h5median_um2_per_s"),
            "median_q70_abs_apparent_D_h5median_um2_per_s": first_summary(apparent_diffusion_calibration, "median_q70_abs_apparent_D_h5median_um2_per_s"),
            "q70_positive_D_fraction": first_summary(apparent_diffusion_calibration, "q70_positive_D_fraction"),
            "threshold_summary": apparent_diffusion_thresholds,
            "source_timing_summary": apparent_diffusion_sources,
            "future8_feature_tests_q70": apparent_diffusion_q70_tests,
            "calibration_correlations": apparent_diffusion_corr,
            "guardrail": first_summary(apparent_diffusion_calibration, "guardrail"),
        },
        "hdf5_timebase_provenance_audit": {
            "timebase_status": first_summary(hdf5_timebase, "timebase_status"),
            "n_h5_files": first_summary(hdf5_timebase, "n_h5_files"),
            "n_q70_roi_rows": first_summary(hdf5_timebase, "n_q70_roi_rows"),
            "n_sources": first_summary(hdf5_timebase, "n_sources"),
            "n_strict_timebase_sources": first_summary(hdf5_timebase, "n_strict_timebase_sources"),
            "n_pause_heavy_sources": first_summary(hdf5_timebase, "n_pause_heavy_sources"),
            "n_roi_elapsed_h5_aligned": first_summary(hdf5_timebase, "n_roi_elapsed_h5_aligned"),
            "n_q70_rows_passing_full_timebase_gate": first_summary(hdf5_timebase, "n_q70_rows_passing_full_timebase_gate"),
            "median_dt_median_s": first_summary(hdf5_timebase, "median_dt_median_s"),
            "median_roi_elapsed_to_h5_ratio": first_summary(hdf5_timebase, "median_roi_elapsed_to_h5_ratio"),
            "max_source_dt_max_to_median_ratio": first_summary(hdf5_timebase, "max_source_dt_max_to_median_ratio"),
            "pause_heavy_sources": first_summary(hdf5_timebase, "pause_heavy_sources", []),
            "scenario_summary": hdf5_timebase_scenarios,
            "top_timebase_correlations": hdf5_timebase_correlations,
            "top_pause_heavy_sources": hdf5_timebase_pause_sources,
            "guardrail": first_summary(hdf5_timebase, "guardrail"),
        },
        "diffusion_physics_consistency_audit": {
            "n_roi": first_summary(diffusion_physics_consistency, "n_roi"),
            "n_cycles": first_summary(diffusion_physics_consistency, "n_cycles"),
            "n_sources": first_summary(diffusion_physics_consistency, "n_sources"),
            "n_threshold_rows": first_summary(diffusion_physics_consistency, "n_threshold_rows"),
            "n_automatic_diffusion_physics_consistent": first_summary(diffusion_physics_consistency, "n_automatic_diffusion_physics_consistent"),
            "n_publication_ready_diffusion_candidates": first_summary(diffusion_physics_consistency, "n_publication_ready_diffusion_candidates"),
            "median_abs_apparent_D_um2_per_s": first_summary(diffusion_physics_consistency, "median_abs_apparent_D_um2_per_s"),
            "median_positive_D_fraction": first_summary(diffusion_physics_consistency, "median_positive_D_fraction"),
            "median_radius2_fit_r2": first_summary(diffusion_physics_consistency, "median_radius2_fit_r2"),
            "median_threshold_sensitivity_iqr_over_median_abs": first_summary(diffusion_physics_consistency, "median_threshold_sensitivity_iqr_over_median_abs"),
            "gate_counts": diffusion_physics_gates,
            "top_consistent_candidates": diffusion_physics_candidates,
            "top_physics_scores": diffusion_physics_top_scores,
            "top_feature_tests": diffusion_physics_tests,
            "top_correlations": diffusion_physics_corr,
            "source_summary": diffusion_physics_source_summary,
            "guardrail": first_summary(diffusion_physics_consistency, "guardrail"),
        },
        "diffusion_claim_readiness_audit": {
            "overall_status": first_summary(diffusion_claim_readiness, "overall_status"),
            "n_criteria": first_summary(diffusion_claim_readiness, "n_criteria"),
            "n_hard_blockers": first_summary(diffusion_claim_readiness, "n_hard_blockers"),
            "status_counts": first_summary(diffusion_claim_readiness, "status_counts", {}),
            "hard_blockers": diffusion_claim_hard_blockers,
            "n_candidate_rows": first_summary(diffusion_claim_readiness, "n_candidate_rows"),
            "n_automatic_consistent_candidates": first_summary(diffusion_claim_readiness, "n_automatic_consistent_candidates"),
            "n_publication_ready_candidates": first_summary(diffusion_claim_readiness, "n_publication_ready_candidates"),
            "top_candidates": diffusion_claim_top_candidates,
            "guardrail": first_summary(diffusion_claim_readiness, "guardrail"),
        },
        "all_cycle_dataset_coverage_atlas": {
            "overall_status": first_summary(all_cycle_coverage, "overall_status"),
            "n_cycle_rows": first_summary(all_cycle_coverage, "n_cycle_rows"),
            "n_sources": first_summary(all_cycle_coverage, "n_sources"),
            "n_h5_inventory_rows": first_summary(all_cycle_coverage, "n_h5_inventory_rows"),
            "n_roi_cohorts_checked": first_summary(all_cycle_coverage, "n_roi_cohorts_checked"),
            "n_cycle_outputs_checked": first_summary(all_cycle_coverage, "n_cycle_outputs_checked"),
            "n_cycles_with_any_roi_video_sequence": first_summary(all_cycle_coverage, "n_cycles_with_any_roi_video_sequence"),
            "n_cycles_with_primary_roi_sequence": first_summary(all_cycle_coverage, "n_cycles_with_primary_roi_sequence"),
            "any_roi_cycle_coverage_fraction": first_summary(all_cycle_coverage, "any_roi_cycle_coverage_fraction"),
            "primary_roi_cycle_coverage_fraction": first_summary(all_cycle_coverage, "primary_roi_cycle_coverage_fraction"),
            "n_future16_positive_cycles_without_any_roi_sequence": first_summary(all_cycle_coverage, "n_future16_positive_cycles_without_any_roi_sequence"),
            "top_source_gaps": all_cycle_source_gaps,
            "top_coverage_gap_cycles": all_cycle_gap_cycles,
            "roi_cohort_summary": all_cycle_roi_cohorts,
            "cycle_output_summary": all_cycle_outputs,
            "outputs": first_summary(all_cycle_coverage, "outputs", {}),
            "guardrail": first_summary(all_cycle_coverage, "guardrail"),
        },
        "current_claim_readiness_matrix": {
            "n_claims": first_summary(current_claim_readiness, "n_claims"),
            "status_counts": current_claim_status_counts,
            "supported_or_operational_claim_ids": first_summary(current_claim_readiness, "supported_or_operational_claim_ids", []),
            "blocked_or_not_supported_claim_ids": first_summary(current_claim_readiness, "blocked_or_not_supported_claim_ids", []),
            "overall_position": first_summary(current_claim_readiness, "overall_position"),
            "top_positive_evidence": current_claim_positive,
            "top_negative_evidence": current_claim_negative,
            "guardrail": first_summary(current_claim_readiness, "guardrail"),
        },
        "diffusion_unblock_sensitivity_audit": {
            "overall_status": first_summary(diffusion_unblock_sensitivity, "overall_status"),
            "n_candidate_rows": first_summary(diffusion_unblock_sensitivity, "n_candidate_rows"),
            "n_criteria": first_summary(diffusion_unblock_sensitivity, "n_criteria"),
            "n_global_hard_blockers_applied": first_summary(diffusion_unblock_sensitivity, "n_global_hard_blockers_applied"),
            "global_hard_blockers_applied": first_summary(diffusion_unblock_sensitivity, "global_hard_blockers_applied", []),
            "top_blockers": diffusion_unblock_blockers,
            "scenario_summary": diffusion_unblock_scenarios,
            "top_nearest_unblock_candidates": diffusion_unblock_candidates,
            "outputs": first_summary(diffusion_unblock_sensitivity, "outputs", {}),
            "guardrail": first_summary(diffusion_unblock_sensitivity, "guardrail"),
        },
        "targeted_diffusion_blocker_diagnostic": {
            "overall_status": first_summary(targeted_diffusion_blocker, "overall_status"),
            "n_target_candidates_with_thresholds": first_summary(targeted_diffusion_blocker, "n_target_candidates_with_thresholds"),
            "n_threshold_variant_rows": first_summary(targeted_diffusion_blocker, "n_threshold_variant_rows"),
            "action_counts": targeted_diffusion_actions,
            "nearest_diffusion_candidate": targeted_diffusion_nearest,
            "top_remeasurement_candidates": targeted_diffusion_candidates,
            "outputs": first_summary(targeted_diffusion_blocker, "outputs", {}),
            "guardrail": first_summary(targeted_diffusion_blocker, "guardrail"),
        },
        "cycle78_diffusion_remeasurement_audit": {
            "overall_status": first_summary(cycle78_diffusion_remeasurement, "overall_status"),
            "target_roi_id": first_summary(cycle78_diffusion_remeasurement, "target_roi_id"),
            "target_status": first_summary(cycle78_diffusion_remeasurement, "target_status"),
            "n_context_rois": first_summary(cycle78_diffusion_remeasurement, "n_context_rois"),
            "n_variant_rows": first_summary(cycle78_diffusion_remeasurement, "n_variant_rows"),
            "n_bootstrap": first_summary(cycle78_diffusion_remeasurement, "n_bootstrap"),
            "block_len": first_summary(cycle78_diffusion_remeasurement, "block_len"),
            "pixel_size_um_assumed": first_summary(cycle78_diffusion_remeasurement, "pixel_size_um_assumed"),
            "target_summary": cycle78_diffusion_target,
            "context_summary": cycle78_diffusion_context,
            "outputs": first_summary(cycle78_diffusion_remeasurement, "outputs", {}),
            "guardrail": first_summary(cycle78_diffusion_remeasurement, "guardrail"),
        },
        "post_remeasurement_diffusion_gate_audit": {
            "overall_status": first_summary(post_remeasurement_diffusion_gate, "overall_status"),
            "target_roi_id": first_summary(post_remeasurement_diffusion_gate, "target_roi_id"),
            "target_q70_status": first_summary(post_remeasurement_diffusion_gate, "target_q70_status"),
            "candidate_q70_blocker_removed": first_summary(post_remeasurement_diffusion_gate, "candidate_q70_blocker_removed"),
            "publication_ready_after_remeasurement": first_summary(post_remeasurement_diffusion_gate, "publication_ready_after_remeasurement"),
            "claim_readiness_overall_status": first_summary(post_remeasurement_diffusion_gate, "claim_readiness_overall_status"),
            "claim_readiness_publication_ready_candidates": first_summary(post_remeasurement_diffusion_gate, "claim_readiness_publication_ready_candidates"),
            "pre_candidate_blockers": first_summary(post_remeasurement_diffusion_gate, "pre_candidate_blockers", []),
            "post_candidate_blockers": first_summary(post_remeasurement_diffusion_gate, "post_candidate_blockers", []),
            "remaining_publication_blockers": post_remeasurement_publication_blockers,
            "target_remeasurement_evidence": post_remeasurement_target,
            "scenario_table": post_remeasurement_scenarios,
            "outputs": first_summary(post_remeasurement_diffusion_gate, "outputs", {}),
            "guardrail": first_summary(post_remeasurement_diffusion_gate, "guardrail"),
        },
        "cycle78_front_identity_review_packet": {
            "overall_status": first_summary(cycle78_front_identity_review, "overall_status"),
            "target_roi_id": first_summary(cycle78_front_identity_review, "target_roi_id"),
            "n_review_rows": first_summary(cycle78_front_identity_review, "n_review_rows"),
            "n_rows_default_q70_positive_ci": first_summary(cycle78_front_identity_review, "n_rows_default_q70_positive_ci"),
            "n_rows_no_automatic_flags": first_summary(cycle78_front_identity_review, "n_rows_no_automatic_flags"),
            "target_summary": cycle78_front_identity_target,
            "outputs": first_summary(cycle78_front_identity_review, "outputs", {}),
            "guardrail": first_summary(cycle78_front_identity_review, "guardrail"),
        },
        "cycle78_component_front_retracking_audit": {
            "overall_status": first_summary(cycle78_component_retracking, "overall_status"),
            "target_roi_id": first_summary(cycle78_component_retracking, "target_roi_id"),
            "n_context_rois": first_summary(cycle78_component_retracking, "n_context_rois"),
            "n_strategy_rows": first_summary(cycle78_component_retracking, "n_strategy_rows"),
            "largest_component_preserves_positive_slope": first_summary(cycle78_component_retracking, "largest_component_preserves_positive_slope"),
            "largest_component_improves_r2": first_summary(cycle78_component_retracking, "largest_component_improves_r2"),
            "target_raw": cycle78_component_raw,
            "target_largest_component": cycle78_component_largest,
            "target_central_component": cycle78_component_central,
            "target_top3_components": cycle78_component_top3,
            "outputs": first_summary(cycle78_component_retracking, "outputs", {}),
            "guardrail": first_summary(cycle78_component_retracking, "guardrail"),
        },
        "cross_modal_degradation_consensus": {
            "n_cycles": first_summary(cross_modal_consensus, "n_cycles"),
            "n_cycles_with_any_modal_vote": first_summary(cross_modal_consensus, "n_cycles_with_any_modal_vote"),
            "median_consensus_score": first_summary(cross_modal_consensus, "median_consensus_score"),
            "modal_vote_threshold": first_summary(cross_modal_consensus, "modal_vote_threshold"),
            "top_cycle": cross_modal_top_cycle,
            "class_summary": cross_modal_classes,
            "target_contrasts": cross_modal_contrasts,
            "top_cycles": cross_modal_top,
            "guardrail": first_summary(cross_modal_consensus, "score_guardrail"),
        },
        "calibration_provenance_evidence_audit": {
            "provenance_status": first_summary(calibration_provenance, "provenance_status"),
            "n_files_inventoried": first_summary(calibration_provenance, "n_files_inventoried"),
            "n_h5_files_scanned": first_summary(calibration_provenance, "n_h5_files_scanned"),
            "n_raw_h5_calibration_like_statements": first_summary(calibration_provenance, "n_raw_h5_calibration_like_statements"),
            "n_raw_h5_spatial_scale_statements": first_summary(calibration_provenance, "n_raw_h5_spatial_scale_statements"),
            "n_near_96nm_px_statements": first_summary(calibration_provenance, "n_near_96nm_px_statements"),
            "n_contradictory_scale_statements": first_summary(calibration_provenance, "n_contradictory_scale_statements"),
            "highest_scale_evidence_strength": first_summary(calibration_provenance, "highest_scale_evidence_strength"),
            "unique_movie_spatial_shapes": first_summary(calibration_provenance, "unique_movie_spatial_shapes", []),
            "near_96nm_px_examples": top_items(first_summary(calibration_provenance, "near_96nm_px_examples", []), 5),
            "interpretation": first_summary(calibration_provenance, "interpretation"),
        },
        "calibration_metadata_audit": {
            "n_h5_discovered_before_cap": first_summary(calibration_metadata, "n_h5_discovered_before_cap"),
            "max_h5_files": first_summary(calibration_metadata, "max_h5_files"),
            "n_h5_files": first_summary(calibration_metadata, "n_h5_files"),
            "n_h5_with_movie": first_summary(calibration_metadata, "n_h5_with_movie"),
            "n_h5_with_camera_timing": first_summary(calibration_metadata, "n_h5_with_camera_timing"),
            "n_h5_with_calibration_attr_hits": first_summary(calibration_metadata, "n_h5_with_calibration_attr_hits"),
            "fps_median_across_h5": first_summary(calibration_metadata, "fps_median_across_h5"),
            "n_pptx_files_scanned": first_summary(calibration_metadata, "n_pptx_files_scanned"),
            "n_pptx_calibration_hits": first_summary(calibration_metadata, "n_pptx_calibration_hits"),
            "top_pptx_hits": first_summary(calibration_metadata, "top_pptx_hits", []),
            "guardrail": first_summary(calibration_metadata, "guardrail"),
        },
        "particle_trace_physics_audit": {
            "n_cycle_rows": first_summary(particle_trace, "n_cycle_rows"),
            "cycle_min": first_summary(particle_trace, "cycle_min"),
            "cycle_max": first_summary(particle_trace, "cycle_max"),
            "n_any_drop_cycles": first_summary(particle_trace, "n_any_drop_cycles"),
            "n_sync2_drop_cycles": first_summary(particle_trace, "n_sync2_drop_cycles"),
            "n_sync3_drop_cycles": first_summary(particle_trace, "n_sync3_drop_cycles"),
            "chosen_trace_state_k": first_summary(particle_trace, "chosen_trace_state_k"),
            "trace_state_best_silhouette": first_summary(particle_trace, "trace_state_best_silhouette"),
            "top_trace_state_clusters": first_summary(particle_trace, "top_trace_state_clusters", []),
            "top_event_feature_tests": particle_trace_event_tests,
            "top_echem_correlations": particle_trace_echem_corr,
            "future_drop_classifier": particle_trace_classifiers,
            "future_drop_classifier_null": particle_trace_nulls,
            "guardrail": first_summary(particle_trace, "guardrail"),
        },
        "particle_event_precursor_atlas": {
            "n_cycle_rows": first_summary(particle_precursor, "n_cycle_rows"),
            "n_event_anchors": first_summary(particle_precursor, "n_event_anchors"),
            "n_candidate_control_anchors": first_summary(particle_precursor, "n_candidate_control_anchors"),
            "n_matched_control_anchors": first_summary(particle_precursor, "n_matched_control_anchors"),
            "event_cycles": first_summary(particle_precursor, "event_cycles", []),
            "top_precursor_window_tests": particle_precursor_tests,
            "top_window_tests": particle_precursor_all_tests,
            "guardrail": first_summary(particle_precursor, "guardrail"),
        },
        "roi_trace_fusion_audit": {
            "n_roi_rows": first_summary(roi_trace_fusion, "n_roi_rows"),
            "n_event_roi": first_summary(roi_trace_fusion, "n_event_roi"),
            "n_event_enriched_mode": first_summary(roi_trace_fusion, "n_event_enriched_mode"),
            "n_predictors": first_summary(roi_trace_fusion, "n_predictors"),
            "context_residualized_against": first_summary(roi_trace_fusion, "context_residualized_against", []),
            "top_precursor_context_residual_spearman": roi_trace_focus,
            "top_precursor_event_enriched_mode_tests": roi_trace_mode_tests,
            "guardrail": first_summary(roi_trace_fusion, "guardrail"),
        },
        "roi_trace_fusion_cycle_null": {
            "n_roi_rows": first_summary(roi_trace_cycle_null, "n_roi_rows"),
            "n_cycle_points": first_summary(roi_trace_cycle_null, "n_cycle_points"),
            "n_event_reference_cycles": first_summary(roi_trace_cycle_null, "n_event_reference_cycles"),
            "n_predictors_tested": first_summary(roi_trace_cycle_null, "n_predictors_tested"),
            "n_permutation": first_summary(roi_trace_cycle_null, "n_permutation"),
            "top_cycle_collapsed_tests": roi_trace_cycle_tests,
            "top_reference_centered_tests": roi_trace_centered_tests,
            "guardrail": first_summary(roi_trace_cycle_null, "guardrail"),
        },
        "precursor_informed_roi_review": {
            "n_review_candidates": first_summary(precursor_review, "n_review_candidates"),
            "n_event_candidates": first_summary(precursor_review, "n_event_candidates"),
            "n_control_candidates": first_summary(precursor_review, "n_control_candidates"),
            "precursor_review_tier_counts": first_summary(precursor_review, "precursor_review_tier_counts", {}),
            "precursor_event_cycles_scored": first_summary(precursor_review, "precursor_event_cycles_scored", []),
            "top_precursor_informed_candidates": precursor_review_top,
            "score_weights": first_summary(precursor_review, "score_weights", {}),
            "guardrail": first_summary(precursor_review, "guardrail"),
        },
        "precursor_review_visual_bundle": {
            "n_ranked_candidates": first_summary(precursor_visual_bundle, "n_ranked_candidates"),
            "n_candidates_with_visual_asset": first_summary(precursor_visual_bundle, "n_candidates_with_visual_asset"),
            "contact_sheet": first_summary(precursor_visual_bundle, "contact_sheet"),
            "top_candidates": precursor_visual_top,
            "guardrail": first_summary(precursor_visual_bundle, "guardrail"),
        },
        "persistence_best_all_cycles": persistence_best,
        "prefix_forecast_n_roi": first_summary(prefix_forecast, "n_roi"),
        "prefix_forecast_n_prefix_features": first_summary(prefix_forecast, "n_prefix_features"),
        "top_prefix_roi_classification": prefix_top_classification,
        "top_prefix_roi_regression": prefix_top_regression,
        "prefix_roi_permutation_null": prefix_null,
        "prefix_roi_feature_importance": {
            "n_roi": first_summary(prefix_importance, "n_roi"),
            "n_rows": first_summary(prefix_importance, "n_rows"),
            "prefix_fraction": first_summary(prefix_importance, "prefix_fraction"),
            "target": first_summary(prefix_importance, "target"),
            "model": first_summary(prefix_importance, "model"),
            "n_features": first_summary(prefix_importance, "n_features"),
            "feature_group_counts": first_summary(prefix_importance, "feature_group_counts", {}),
            "model_summary": prefix_importance_model,
            "permutation_null": prefix_importance_null,
            "top_group_ablation": prefix_importance_groups,
            "top_permutation_importance": prefix_importance_features,
            "guardrail": first_summary(prefix_importance, "guardrail"),
        },
        "manual_qc_label_workbook": {
            "n_unique_roi": first_summary(manual_qc_workbook, "n_unique_roi"),
            "role_counts": first_summary(manual_qc_workbook, "role_counts", {}),
            "review_priority_tier_counts": first_summary(manual_qc_workbook, "review_priority_tier_counts", {}),
            "manual_qc_status_counts": first_summary(manual_qc_workbook, "manual_qc_status_counts", {}),
            "source_counts": first_summary(manual_qc_workbook, "source_counts", {}),
            "guardrail": first_summary(manual_qc_workbook, "guardrail"),
        },
        "manual_qc_gated_front_effects": {
            "status": first_summary(manual_qc_gated, "status"),
            "n_joined_roi": first_summary(manual_qc_gated, "n_joined_roi"),
            "n_manual_front_effect_accepted": first_summary(manual_qc_gated, "n_manual_front_effect_accepted"),
            "n_manual_diffusion_accepted": first_summary(manual_qc_gated, "n_manual_diffusion_accepted"),
            "manual_qc_status_counts": first_summary(manual_qc_gated, "manual_qc_status_counts", {}),
            "manual_qc_decision_counts": first_summary(manual_qc_gated, "manual_qc_decision_counts", {}),
            "guardrail": first_summary(manual_qc_gated, "guardrail"),
        },
        "control_balanced_front_qc_sensitivity": {
            "n_roi": first_summary(control_balanced_qc_sensitivity, "n_roi"),
            "n_event_roi": first_summary(control_balanced_qc_sensitivity, "n_event_roi"),
            "n_control_roi": first_summary(control_balanced_qc_sensitivity, "n_control_roi"),
            "robust_positive_phase_residual_strata": first_summary(control_balanced_qc_sensitivity, "robust_positive_phase_residual_strata", []),
            "interpretation": first_summary(control_balanced_qc_sensitivity, "interpretation"),
        },
        "strict_rf_mean_roc_auc": strict_rf.get("mean_roc_auc"),
        "strict_rf_mean_balanced_accuracy": strict_rf.get("mean_balanced_accuracy"),
        "strict_logistic_mean_roc_auc": strict_logistic.get("mean_roc_auc"),
        "all_qc_rf_mean_roc_auc": all_rf.get("mean_roc_auc"),
        "protocol_conditioned_residual_mean_roc_auc": conditioned_model.get("mean_roc_auc"),
        "protocol_conditioned_residual_mean_balanced_accuracy": conditioned_model.get("mean_balanced_accuracy"),
        "protocol_conditioned_front_mean_roc_auc": conditioned_front_model.get("mean_roc_auc"),
        "protocol_conditioned_front_mean_balanced_accuracy": conditioned_front_model.get("mean_balanced_accuracy"),
        "protocol_conditioned_front_residual_mean_roc_auc": conditioned_front_model.get("mean_roc_auc"),
        "protocol_conditioned_front_residual_mean_balanced_accuracy": conditioned_front_model.get("mean_balanced_accuracy"),
        "top_within_reference_tests": best_within,
        "top_roi_echem_correlations": best_echem_corr,
        "top_protocol_conditioned_event_control_tests": conditioned_tests,
        "top_threshold_robust_front_tests": robust_front_tests,
        "top_threshold_robust_front_by_event_tests": robust_front_by_event,
        "top_protocol_conditioned_front_tests": conditioned_front_tests,
        "top_qc_review_candidates": top_items(first_summary(qc_packet, "top_candidates", []), 8),
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
