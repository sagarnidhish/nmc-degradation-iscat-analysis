#!/usr/bin/env python3
"""Robin-style closed-loop analysis summary and next-action queue."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
from agentic_utils import finite_float, markdown_table, output_root, read_csv_if_exists, read_json, resolve_root, summarize_available_artifacts, write_json


def extract_observations(root: Path) -> List[Dict[str, Any]]:
    derived = root / "derived"
    observations: List[Dict[str, Any]] = []
    sync = read_json(derived / "event_synchrony" / "event_synchrony_summary.json")
    if sync:
        observations.append({
            "topic": "event_synchrony",
            "observation": "Synchronized optical drop events have already passed a permutation sanity check.",
            "evidence": str(derived / "event_synchrony" / "event_synchrony_summary.json"),
            "strength": "moderate",
        })
    echem = read_json(derived / "event_echem_coupling" / "event_echem_coupling_summary.json")
    if echem:
        observations.append({
            "topic": "event_echem",
            "observation": f"{echem.get('n_cycles_with_echem_match', 'unknown')} cycles matched to electrochemistry; coarse cycle summaries do not fully explain events.",
            "evidence": str(derived / "event_echem_coupling" / "event_echem_coupling_summary.json"),
            "strength": "moderate",
        })
    protocol = read_json(derived / "event_protocol_context" / "event_protocol_context_summary.json")
    if protocol:
        observations.append({
            "topic": "protocol_context",
            "observation": "Synchronized event cycles are linked to unusually short frame-count regimes.",
            "evidence": str(derived / "event_protocol_context" / "event_protocol_context_summary.json"),
            "strength": "moderate",
        })

    fronts = read_json(derived / "event_candidate_fronts" / "event_candidate_fronts_summary.json")
    if fronts:
        observations.append({
            "topic": "event_candidate_fronts",
            "observation": f"Detected {fronts.get('n_candidate_rows', 'unknown')} candidate particle/front regions; outputs are apparent front proxies, not calibrated diffusion coefficients.",
            "evidence": str(derived / "event_candidate_fronts" / "event_candidate_fronts_summary.json"),
            "strength": "exploratory",
        })
    validated = read_json(derived / "validated_front_rois" / "validated_front_rois_summary.json")
    if validated:
        observations.append({
            "topic": "validated_front_rois",
            "observation": f"Selected {validated.get('n_selected_for_next_tracking', 'unknown')} cycle-86/116 candidate ROIs for next tracking from {validated.get('n_candidates_scored', 'unknown')} scored candidates; selections are algorithmic and still need manual QC/spatial calibration.",
            "evidence": str(derived / "validated_front_rois" / "validated_front_rois_summary.json"),
            "strength": "exploratory_to_moderate",
        })
    tracking = read_json(derived / "selected_front_roi_tracking" / "selected_front_roi_tracking_summary.json")
    if tracking:
        cycle_bits = []
        for rec in tracking.get("cycle_summary", []):
            cycle_bits.append(f"cycle {int(float(rec.get('cycleNo', 0)))} mean radius2 slope {finite_float(rec.get('radius2_slope_full_px2_per_s')):.3g} px2/s, R2 {finite_float(rec.get('radius2_slope_r2')):.3g}")
        observations.append({
            "topic": "selected_front_roi_tracking",
            "observation": f"Tracked {tracking.get('n_tracked_rois', 'unknown')} selected front ROIs at full-pixel crop scale; {'; '.join(cycle_bits)}. These are apparent front-motion proxies, not calibrated diffusion coefficients.",
            "evidence": str(derived / "selected_front_roi_tracking" / "selected_front_roi_tracking_summary.json"),
            "strength": "exploratory_to_moderate",
        })
    rollout = read_json(derived / "roi_rollout_baselines" / "roi_rollout_baseline_summary.json")
    if rollout:
        observations.append({
            "topic": "roi_rollout_baselines",
            "observation": f"Evaluated {rollout.get('n_roi_sequences', 'unknown')} particle-ROI sequences with persistence, velocity, and low-rank DMD rollouts; DMD spectral radius {finite_float(rollout.get('dmd_spectral_radius')):.3f}.",
            "evidence": str(derived / "roi_rollout_baselines" / "roi_rollout_baseline_summary.json"),
            "strength": "moderate_baseline",
        })
    event_model = read_json(derived / "roi_event_conditioned_nextframe" / "roi_event_model_summary.json")
    if event_model:
        residual_bits = []
        for rec in event_model.get("residual_cycle_summary", []):
            residual_bits.append(f"cycle {int(float(rec.get('cycleNo', 0)))} residual mean {finite_float(rec.get('rollout_residual_energy_mean')):.3g}")
        observations.append({
            "topic": "roi_event_conditioned_nextframe",
            "observation": f"Trained a PCA-ridge event-conditioned next-frame model on {event_model.get('n_roi_sequences', 'unknown')} selected ROI sequences; {'; '.join(residual_bits)}. Persistence remains a strong recursive baseline, so residuals are the main degradation descriptor.",
            "evidence": str(derived / "roi_event_conditioned_nextframe" / "roi_event_model_summary.json"),
            "strength": "moderate_baseline",
        })
    residual_cnn = read_json(derived / "roi_residual_cnn_fast" / "roi_residual_cnn_summary.json")
    if residual_cnn:
        observations.append({
            "topic": "roi_residual_cnn_fast",
            "observation": f"A leave-one-cycle residual CNN on {residual_cnn.get('n_roi', 'unknown')} selected ROIs did not beat persistence; overall relative MSE improvement {finite_float(residual_cnn.get('overall_relative_mse_improvement')):.3g}.",
            "evidence": str(derived / "roi_residual_cnn_fast" / "roi_residual_cnn_summary.json"),
            "strength": "negative_baseline",
        })
    joint = read_json(derived / "roi_joint_physics_degradation_modes" / "roi_joint_physics_degradation_modes_summary.json")
    if joint:
        top = joint.get("top_ranked_rois", [])[:2]
        top_bits = [f"{r.get('roi_key')} score {finite_float(r.get('joint_degradation_score')):.3g}" for r in top]
        observations.append({
            "topic": "roi_joint_physics_degradation_modes",
            "observation": f"Combined rollout residual, front tracking, ROI physics, residual-CNN guardrails, and cycle evidence into {joint.get('n_roi', 'unknown')} ROI joint modes; selected k={joint.get('selected_k', 'unknown')} with silhouette {finite_float(joint.get('silhouette')):.3g}. Top ROIs: {'; '.join(top_bits)}.",
            "evidence": str(derived / "roi_joint_physics_degradation_modes" / "roi_joint_physics_degradation_modes_summary.json"),
            "strength": "moderate_synthesis",
        })
    evctrl = read_json(derived / "event_vs_control_roi_physics" / "event_vs_control_roi_physics_summary.json")
    if evctrl:
        top_tests = evctrl.get("top_feature_tests", [])[:2]
        test_bits = [f"{t.get('feature')} p={finite_float(t.get('p_value')):.3g}" for t in top_tests]
        observations.append({
            "topic": "event_vs_control_roi_physics",
            "observation": f"Compared {evctrl.get('n_event_roi', 'unknown')} event ROIs against {evctrl.get('n_control_roi', 'unknown')} matched control ROIs; strongest shifts: {'; '.join(test_bits)}. Leave-pair classifier did not generalize well, so controls are a guardrail.",
            "evidence": str(derived / "event_vs_control_roi_physics" / "event_vs_control_roi_physics_summary.json"),
            "strength": "moderate_control_check",
        })
    evctrl_exp = read_json(derived / "event_control_roi_comparison_expanded" / "event_control_roi_comparison_summary.json")
    clf_exp = read_json(derived / "event_control_roi_classifier_expanded" / "event_control_roi_classifier_summary.json")
    if evctrl_exp:
        top = evctrl_exp.get("top_metric_differences", [])[:2]
        bits = [f"{r.get('metric')} d={finite_float(r.get('cohens_d_event_vs_control')):.3g}, p={finite_float(r.get('mannwhitney_p')):.3g}" for r in top]
        clf_bit = f"; expanded classifier mean ROC-AUC {finite_float(clf_exp.get('mean_roc_auc')):.3g}" if clf_exp else ""
        observations.append({
            "topic": "event_control_roi_comparison_expanded",
            "observation": f"Expanded matched controls to {evctrl_exp.get('n_control_roi', 'unknown')} ROIs; strongest event-control shifts: {'; '.join(bits)}{clf_bit}.",
            "evidence": str(derived / "event_control_roi_comparison_expanded" / "event_control_roi_comparison_summary.json"),
            "strength": "moderate_control_check",
        })
    baselines = read_csv_if_exists(derived / "particle_event_targets" / "particle_event_feature_baselines.csv")
    if not baselines.empty and "f1" in baselines:
        observations.append({
            "topic": "event_forecasting",
            "observation": f"Best transparent leave-one-particle event-forecast F1 is {finite_float(baselines['f1'].max()):.3f}.",
            "evidence": str(derived / "particle_event_targets" / "particle_event_feature_baselines.csv"),
            "strength": "weak_to_moderate",
        })
    return observations


def next_actions(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    topics = {o["topic"] for o in observations}
    actions = []
    if "protocol_context" in topics:
        actions.append({
            "priority": 1,
            "action": "Run frame-count-matched shuffled controls for synchronized event cycles.",
            "expected_output": "matched_control_p_values.csv and artifact-risk summary",
        })
    if "event_echem" in topics:
        actions.append({
            "priority": 2,
            "action": "Run protocol-local echem window scan around cycles 86 and 116.",
            "expected_output": "local_echem_window_features.csv with neighbor-cycle controls",
        })
    actions.append({
        "priority": 3,
        "action": "Generate ROI/event visual QC manifest for raw frame inspection.",
        "expected_output": "event_roi_qc_manifest.csv and review checklist",
    })
    if "selected_front_roi_tracking" in topics:
        actions.append({
            "priority": 4,
            "action": "Add manual QC/spatial calibration metadata for selected front ROIs and convert pixel-scale slopes to calibrated units.",
            "expected_output": "front_roi_qc_calibration.csv and calibrated_front_tracking.csv",
        })
    elif "validated_front_rois" in topics:
        actions.append({
            "priority": 4,
            "action": "Run high-resolution selected-ROI front tracking with manual QC and spatial calibration.",
            "expected_output": "calibrated_front_tracking.csv with QC decisions and micron-scale transport proxies",
        })
    else:
        actions.append({
            "priority": 4,
            "action": "Review candidate front overlays for cycles 86/116 and select validated ROIs for calibrated front tracking.",
            "expected_output": "validated_front_candidates.csv and manual QC decisions",
        })
    if "event_control_roi_comparison_expanded" in topics:
        actions.append({
            "priority": 5,
            "action": "Add manual QC/spatial calibration for top joint-mode ROIs and expand event/control extraction to additional non-synchronized candidate cycles.",
            "expected_output": "calibrated top-mode ROI report and expanded multi-cycle ROI cohort",
        })
    elif "event_vs_control_roi_physics" in topics:
        actions.append({
            "priority": 5,
            "action": "Scale ROI/control sampling to additional cycles and add manual QC/spatial calibration for the top joint-mode ROIs.",
            "expected_output": "expanded_roi_control_manifest.csv and calibrated top-mode ROI report",
        })
    elif "roi_joint_physics_degradation_modes" in topics:
        actions.append({
            "priority": 5,
            "action": "Expand ROI/control sampling beyond synchronized events and validate whether joint modes generalize to non-event particle regions.",
            "expected_output": "control_roi_sequences and event-vs-control joint physics report",
        })
    elif "roi_event_conditioned_nextframe" in topics:
        actions.append({
            "priority": 5,
            "action": "Use ROI rollout residual and front-tracking features in a joint degradation-mode/hazard model across selected and future ROIs.",
            "expected_output": "roi_joint_physics_degradation_modes.csv and calibrated event-hazard report",
        })
    elif "roi_rollout_baselines" in topics:
        actions.append({
            "priority": 5,
            "action": "Train event-conditioned ROI next-frame models against persistence/DMD baselines and test rollout residuals as degradation features.",
            "expected_output": "roi_event_model_metrics.csv and rollout_residual_degradation_features.csv",
        })
    else:
        actions.append({
            "priority": 5,
            "action": "Fit grouped degradation-mode clustering and hazard calibration.",
            "expected_output": "degradation_modes.csv, hazard_calibration.csv",
        })
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    root = resolve_root(args.root)
    out = output_root(root, args.out_dir) / "03_closed_loop_analysis"
    out.mkdir(parents=True, exist_ok=True)
    artifacts = summarize_available_artifacts(root)
    observations = extract_observations(root)
    actions = next_actions(observations)
    write_json(out / "closed_loop_state.json", {"root": str(root), "artifacts": artifacts, "observations": observations, "next_actions": actions})
    pd.DataFrame(observations).to_csv(out / "closed_loop_observations.csv", index=False)
    pd.DataFrame(actions).to_csv(out / "next_action_queue.csv", index=False)
    md = [
        "# Closed-Loop Analysis Update",
        "",
        "## Current Observations",
        "",
        markdown_table(observations, ["topic", "strength", "observation", "evidence"]),
        "",
        "## Next Action Queue",
        "",
        markdown_table(actions, ["priority", "action", "expected_output"]),
        "",
    ]
    (out / "closed_loop_analysis_report.md").write_text("\n".join(md))
    print(f"[done] wrote closed-loop analysis outputs to {out}")


if __name__ == "__main__":
    main()
