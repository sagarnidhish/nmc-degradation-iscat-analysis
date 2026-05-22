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
    calibration_claim_risk = read_json(derived / "calibration_claim_risk_register" / "calibration_claim_risk_summary.json")
    apparent_diffusion_calibration = read_json(derived / "apparent_diffusion_calibration_bounds" / "apparent_diffusion_calibration_bounds_summary.json")
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
    particle_mask = read_json(derived / "particle_mask_stability_audit" / "particle_mask_stability_audit_summary.json")
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
    balanced_future_sequences = read_json(derived / "balanced_future_roi_sequences" / "selected_roi_sequence_summary.json")
    balanced_future_fronts = read_json(derived / "balanced_future_threshold_robust_fronts" / "threshold_robust_front_summary.json")
    balanced_future_rollout = read_json(derived / "balanced_future_masked_roi_rollout_audit" / "masked_roi_rollout_audit_summary.json")
    balanced_future_mask = read_json(derived / "balanced_future_particle_mask_stability" / "particle_mask_stability_audit_summary.json")
    balanced_future_context = read_json(derived / "balanced_future_context_region_audit" / "balanced_future_context_region_summary.json")
    temporal_directionality = read_json(derived / "temporal_directionality_physics_audit" / "temporal_directionality_physics_audit_summary.json")
    balanced_spatial_propagation = read_json(derived / "balanced_spatial_front_propagation_audit" / "balanced_spatial_front_propagation_summary.json")
    masked_video_embedding = read_json(derived / "masked_video_embedding_audit" / "masked_video_embedding_audit_summary.json")
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
    particle_mask_role_summary = top_items(first_summary(particle_mask, "role_summary", []), 8)
    particle_mask_tests = top_items(first_summary(particle_mask, "top_event_control_tests", []), 12)
    particle_mask_corr = top_items(first_summary(particle_mask, "top_correlations", []), 12)
    particle_mask_top = top_items(first_summary(particle_mask, "highest_instability_rois", []), 12)
    particle_mask_overall = first_summary(particle_mask, "overall", {}) or {}
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
            "implemented",
            f"Synthesis reads Isambard derived directory with {n_roi} ROI rows, {n_cycles} cycles, and {n_event_refs} event-reference cycles.",
            "The current multi-cycle ROI cohort is selected around event/reference cycles, not every raw video in the full dataset.",
            "Expand ROI extraction to all cycles after manual QC stabilizes particle identity selection.",
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
        f"- Calibration claim-risk families/source tables: {first_summary(calibration_claim_risk, 'n_claim_families', 0)} / {first_summary(calibration_claim_risk, 'n_source_tables_present', 0)}",
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
        f"- Calibration claim-risk register audits {first_summary(calibration_claim_risk, 'n_claim_families', 0)} front/kinetic/diffusion claim families; it classifies diffusion-like values as apparent proxies and keeps manual-QC-gated diffusion/front claims pending.",
        f"- Apparent diffusion calibration-bounds audit maps all {first_summary(apparent_diffusion_calibration, 'n_roi_with_h5_timing', 0)} balanced ROIs to HDF5 timing; ROI elapsed/HDF5 elapsed median ratio is {fmt(first_summary(apparent_diffusion_calibration, 'median_roi_elapsed_to_h5_median_ratio'))}, q70 median apparent D at 96 nm/px is {fmt(first_summary(apparent_diffusion_calibration, 'median_q70_apparent_D_h5median_um2_per_s'))} um2/s, and q70 future8 separation is non-significant (top p={fmt(apparent_diffusion_q70_test.get('mannwhitney_p'))}).",
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
        f"- Balanced future particle-mask stability audit covers {first_summary(balanced_future_mask, 'n_roi', 0)} ROIs / {first_summary(balanced_future_mask, 'n_frames_total', 0)} frames; median fallback fraction is {fmt(balanced_future_mask_overall.get('median_fallback_frame_fraction'))}, and the strongest future8 mask-stability contrast is {balanced_future_mask_top_test.get('feature', 'NA')} with p={fmt(balanced_future_mask_top_test.get('p_value'))}, so the balanced future signal is not explained by a simple mask-instability split.",
        f"- Masked video embedding audit extracts particle-prior self-supervised descriptors across {first_summary(masked_video_embedding, 'n_embedding_rows', 0)} ROI tensors; balanced future leave-cycle AUC/AP is {fmt(masked_video_future_metric.get('pooled_oof_roc_auc'))}/{fmt(masked_video_future_metric.get('pooled_oof_average_precision'))} with label-permutation p={fmt(masked_video_null.get('empirical_p_ge_observed'))}, while selected event/control readout is weaker at AUC {fmt(masked_video_event_metric.get('pooled_oof_roc_auc'))}.",
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
