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
    particle_trace = read_json(derived / "particle_trace_physics_audit" / "particle_trace_physics_audit_summary.json")
    particle_precursor = read_json(derived / "particle_event_precursor_atlas" / "particle_event_precursor_atlas_summary.json")
    roi_trace_fusion = read_json(derived / "roi_trace_fusion_audit" / "roi_trace_fusion_audit_summary.json")
    precursor_review = read_json(derived / "precursor_informed_roi_review" / "precursor_informed_roi_review_summary.json")

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
    precursor_review_top = top_items(first_summary(precursor_review, "top_precursor_informed_candidates", []), 12)

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
            "Track phase-boundary movement",
            "implemented_as_proxy",
            f"Front/phase mobility descriptors, selected-front tracking, and threshold-robust sweeps exist; threshold sweep covers {first_summary(robust_fronts, 'n_roi', 0)} ROI rows.",
            f"Front masks are automatic; after protocol/echem conditioning, front-direction sign consistency survives more strongly than front-magnitude metrics and is robust in {len(robust_phase_strata)} automatic QC strata.",
            f"Use the primary and control-balanced QC packages to record accept/reject decisions, including {first_summary(control_balanced_qc, 'n_control_roi', 0)} control candidates.",
        ),
        evidence_row(
            "Extract diffusion coefficients",
            "partial_proxy_only",
            f"Provisional 0.096 um/px apparent diffusion proxies were computed and stress-tested across {len(first_summary(robust_fronts, 'threshold_quantiles', []))} thresholds with bootstrap slopes.",
            "Global threshold-robust phase slopes separate event/control ROIs, but QC-stratified diffusion proxies are inconsistent and conditioned diffusion-proxy residuals remain non-significant.",
            "Treat diffusion numbers as apparent optical-front proxies until microscope calibration, timebase, and front masks are manually validated.",
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
        f"- Particle trace cycle rows/drop cycles: {first_summary(particle_trace, 'n_cycle_rows', 0)} / {first_summary(particle_trace, 'n_any_drop_cycles', 0)}",
        f"- Particle precursor event/control anchors: {first_summary(particle_precursor, 'n_event_anchors', 0)} / {first_summary(particle_precursor, 'n_matched_control_anchors', 0)}",
        f"- ROI trace-fusion rows/predictors: {first_summary(roi_trace_fusion, 'n_roi_rows', 0)} / {first_summary(roi_trace_fusion, 'n_predictors', 0)}",
        f"- Precursor-informed review candidates: {first_summary(precursor_review, 'n_review_candidates', 0)}",
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
        f"- ROI trace-fusion links lagged global particle-trace state to localized front behavior: strongest focused context-residual association is {((roi_trace_focus[0] if roi_trace_focus else {}).get('predictor', 'NA'))} vs {((roi_trace_focus[0] if roi_trace_focus else {}).get('target', 'NA'))}, rho={fmt((roi_trace_focus[0] if roi_trace_focus else {}).get('rho'))}, p={fmt((roi_trace_focus[0] if roi_trace_focus else {}).get('p_value'))}.",
        f"- Precursor-informed ROI review ranks {first_summary(precursor_review, 'n_review_candidates', 0)} pending manual-QC candidates; the top candidate is {(precursor_review_top[0] if precursor_review_top else {}).get('roi_id', 'NA')} with score {fmt((precursor_review_top[0] if precursor_review_top else {}).get('precursor_informed_review_score'))}.",
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
