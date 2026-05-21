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
    if "roi_event_conditioned_nextframe" in topics:
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
