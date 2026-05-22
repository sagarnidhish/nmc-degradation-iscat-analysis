#!/usr/bin/env python3
"""Current-evidence hypothesis tournament for NMC photometry analysis.

This is a deterministic, audit-friendly adaptation of the multi-agent
scientific-discovery pattern: evidence agents summarize current outputs,
generator agents propose hypotheses, skeptic agents attach guardrails, and a
tournament ranker turns them into executable next-analysis specs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
        return {}
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_row(rows: Iterable[Dict[str, Any]], **filters: Any) -> Dict[str, Any]:
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    return {}


def first_float(row: Dict[str, Any], keys: Iterable[str], default: float = np.nan) -> float:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return default


def clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def add_hypothesis(
    rows: List[Dict[str, Any]],
    *,
    hypothesis_id: str,
    title: str,
    generator_agent: str,
    claim: str,
    evidence: List[str],
    falsification: str,
    next_experiment: str,
    suggested_script: str,
    evidence_support: float,
    novelty: float,
    falsifiability: float,
    implementability: float,
    guardrail_penalty: float,
    cost_penalty: float = 0.15,
    human_review_need: float = 0.0,
) -> None:
    score = (
        0.30 * clamp01(evidence_support)
        + 0.18 * clamp01(novelty)
        + 0.18 * clamp01(falsifiability)
        + 0.18 * clamp01(implementability)
        - 0.12 * clamp01(guardrail_penalty)
        - 0.04 * clamp01(cost_penalty)
    )
    rows.append(
        {
            "hypothesis_id": hypothesis_id,
            "title": title,
            "generator_agent": generator_agent,
            "claim": claim,
            "evidence_bullets": evidence,
            "falsification_test": falsification,
            "next_experiment": next_experiment,
            "suggested_script": suggested_script,
            "evidence_support": clamp01(evidence_support),
            "novelty": clamp01(novelty),
            "falsifiability": clamp01(falsifiability),
            "implementability": clamp01(implementability),
            "guardrail_penalty": clamp01(guardrail_penalty),
            "cost_penalty": clamp01(cost_penalty),
            "human_review_need": clamp01(human_review_need),
            "tournament_score": score,
        }
    )


def write_readme(path: Path, ranked: pd.DataFrame, summary: Dict[str, Any]) -> None:
    lines = [
        "# Agentic Current Hypothesis Tournament",
        "",
        "Deterministic co-scientist-style tournament over the current NMC photometry evidence.",
        "It uses generator, skeptic, and ranker roles without making unverified physical claims.",
        "",
        "## Inputs",
        "",
    ]
    for name, info in summary["evidence_inventory"].items():
        lines.append(f"- `{name}`: {info}")
    lines.extend(["", "## Top Hypotheses", ""])
    for row in ranked.head(8).to_dict("records"):
        lines.extend(
            [
                f"### {row['rank']}. {row['title']}",
                "",
                f"- Score: {fmt(row['tournament_score'])}",
                f"- Agent: `{row['generator_agent']}`",
                f"- Claim: {row['claim']}",
                f"- Falsification: {row['falsification_test']}",
                f"- Next experiment: {row['next_experiment']}",
                f"- Suggested script: `{row['suggested_script']}`",
                "",
                "Evidence:",
            ]
        )
        for item in row["evidence_bullets"]:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(
        [
            "## Guardrail",
            "",
            summary["guardrail"],
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/agentic_current_hypothesis_tournament")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    echem_video = read_json(derived / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_summary.json")
    residual_dict = read_json(derived / "residual_dictionary_embedding_audit" / "residual_dictionary_embedding_summary.json")
    qc_ledger = read_json(derived / "qc_decision_evidence_ledger" / "qc_decision_evidence_ledger_summary.json")
    diffusion = read_json(derived / "apparent_diffusion_calibration_bounds" / "apparent_diffusion_calibration_bounds_summary.json")
    temporal = read_json(derived / "temporal_directionality_physics_audit" / "temporal_directionality_physics_audit_summary.json")
    spatial = read_json(derived / "balanced_spatial_front_propagation_audit" / "balanced_spatial_front_propagation_summary.json")
    cycle_state = read_json(derived / "cycle_state_space_transition_audit" / "cycle_state_space_transition_audit_summary.json")
    prefix = read_json(derived / "prefix_roi_forecast" / "prefix_roi_forecast_summary.json")
    physics = read_json(derived / "physics_consistency_claim_matrix" / "physics_consistency_claim_matrix_summary.json")
    synthesis = read_json(derived / "nmc_ai_physics_synthesis" / "nmc_ai_physics_synthesis_summary.json")

    echem_metrics = echem_video.get("top_classification_metrics", [])
    future16_video_echem = metric_row(echem_metrics, target="future_any_drop_within_16cycles", feature_set="video_plus_echem")
    future16_echem = metric_row(echem_metrics, target="future_any_drop_within_16cycles", feature_set="echem_regime")
    future8_acq = metric_row(echem_metrics, target="future_any_drop_within_8cycles", feature_set="acquisition_context")
    future8_video_all = metric_row(echem_metrics, target="future_any_drop_within_8cycles", feature_set="video_all")

    residual_metrics = residual_dict.get("top_classification_metrics", [])
    residual_future8 = metric_row(residual_metrics, target="future_any_drop_within_8cycles", feature_set="residual_dictionary")
    residual_plus_future8 = metric_row(residual_metrics, target="future_any_drop_within_8cycles", feature_set="residual_dictionary_plus_handcrafted")
    handcrafted_future8 = metric_row(residual_metrics, target="future_any_drop_within_8cycles", feature_set="handcrafted_scalar")
    pca_future8 = metric_row(residual_metrics, target="future_any_drop_within_8cycles", feature_set="pca_video")

    qc_actions = qc_ledger.get("decision_action_counts", {})
    n_qc = float(qc_ledger.get("n_candidates", 0) or 0)
    accept_first = float(qc_actions.get("review_for_possible_accept_first", 0) or 0)
    artifact_first = float(qc_actions.get("review_artifact_or_reject_first", 0) or 0)

    diffusion_guardrail = diffusion.get("guardrail") or diffusion.get("diffusion_guardrail") or "calibration guardrail unavailable"
    temporal_guardrail = temporal.get("guardrail", "temporal audit guardrail unavailable")
    spatial_guardrail = spatial.get("guardrail", "spatial propagation guardrail unavailable")
    cycle_state_guardrail = cycle_state.get("guardrail", "cycle-state guardrail unavailable")
    physics_guardrail = physics.get("guardrail", "physics consistency guardrail unavailable")

    rows: List[Dict[str, Any]] = []

    auc_future16_fused = first_float(future16_video_echem, ["roc_auc"])
    auc_future16_echem = first_float(future16_echem, ["roc_auc"])
    add_hypothesis(
        rows,
        hypothesis_id="H1",
        title="Echem-conditioned video residuals are the best longer-horizon weak-label signal",
        generator_agent="multimodal_evidence_agent",
        claim="Longer-horizon future-drop signal is strongest when optical video descriptors are conditioned by echem regime.",
        evidence=[
            f"future16 video+echem AUC {fmt(auc_future16_fused)} versus echem-only AUC {fmt(auc_future16_echem)}",
            f"future16 video+echem empirical p {fmt(future16_video_echem.get('empirical_p_ge_observed'))}",
            "Masked residual signatures also become interpretable when echem features are included in prior fusion audits.",
        ],
        falsification="Require the video+echem gain to persist under cycle-balanced acquisition residualization and source-cohort holdouts.",
        next_experiment="Run an acquisition-residualized video+echem future16 audit with leave-source and leave-cycle splits.",
        suggested_script="tier4_acquisition_residualized_video_echem_warning.py",
        evidence_support=0.55 + 0.8 * max(0.0, auc_future16_fused - auc_future16_echem),
        novelty=0.72,
        falsifiability=0.86,
        implementability=0.80,
        guardrail_penalty=0.38,
        cost_penalty=0.25,
    )

    auc_future8_acq = first_float(future8_acq, ["roc_auc"])
    auc_future8_video = first_float(future8_video_all, ["roc_auc"])
    add_hypothesis(
        rows,
        hypothesis_id="H2",
        title="Short-horizon future8 labels are acquisition/context dominated",
        generator_agent="skeptical_confound_agent",
        claim="The future8 weak label is partly a cohort-design or acquisition proxy, so it should not drive mechanistic claims alone.",
        evidence=[
            f"future8 acquisition/context AUC {fmt(auc_future8_acq)} versus video-all AUC {fmt(auc_future8_video)}",
            "Prior context audits repeatedly flag frame count and protocol position as central confounders.",
            "This explains why high future8 performance can coexist with weak physical interpretability.",
        ],
        falsification="A future8 model must remain above null after acquisition, frame-count, cohort-role, and source-movie residualization.",
        next_experiment="Build a residualized future8 benchmark that removes acquisition/context signal before testing video physics features.",
        suggested_script="tier4_residualized_future8_video_physics_benchmark.py",
        evidence_support=0.60 + 0.35 * clamp01(auc_future8_acq - auc_future8_video),
        novelty=0.55,
        falsifiability=0.92,
        implementability=0.88,
        guardrail_penalty=0.18,
        cost_penalty=0.15,
    )

    residual_auc = first_float(residual_future8, ["roc_auc"])
    residual_plus_auc = first_float(residual_plus_future8, ["roc_auc"])
    handcrafted_auc = first_float(handcrafted_future8, ["roc_auc"])
    pca_auc = first_float(pca_future8, ["roc_auc"])
    add_hypothesis(
        rows,
        hypothesis_id="H3",
        title="Generic video embeddings underperform physics-aware temporal descriptors",
        generator_agent="representation_skeptic_agent",
        claim="Current label-free residual/PCA embeddings help triage but do not replace hand-built particle and mask dynamics.",
        evidence=[
            f"residual-dictionary future8 AUC {fmt(residual_auc)} versus raw PCA AUC {fmt(pca_auc)}",
            f"residual+handcrafted AUC {fmt(residual_plus_auc)} versus handcrafted-only AUC {fmt(handcrafted_auc)}",
            "This points toward physics-aware or echem-conditioned objectives rather than larger generic embeddings.",
        ],
        falsification="Show that a learned objective beats handcrafted features under leave-cycle splits without acquisition leakage.",
        next_experiment="Train or approximate an echem/context-conditioned residual objective and compare it against handcrafted scalars.",
        suggested_script="tier4_echem_conditioned_residual_dictionary.py",
        evidence_support=0.68,
        novelty=0.70,
        falsifiability=0.82,
        implementability=0.70,
        guardrail_penalty=0.32,
        cost_penalty=0.45,
    )

    add_hypothesis(
        rows,
        hypothesis_id="H4",
        title="Manual QC on the top cycle-156 panel is the highest-yield lab-in-the-loop step",
        generator_agent="lab_in_loop_qc_agent",
        claim="The next human intervention should be a small targeted cycle-156 review because it maximizes cross-modal support and artifact contrast.",
        evidence=[
            f"QC ledger has {fmt(n_qc, 0)} pending candidates, {fmt(accept_first, 0)} possible-accept-first and {fmt(artifact_first, 0)} artifact/reject-first candidates.",
            "Cycle 156 contains all current possible-accept-first candidates plus a strong artifact-risk foil.",
            "This mirrors a lab-in-the-loop discovery loop: rank candidates, inspect, feed labels back into models.",
        ],
        falsification="If manual review rejects the possible-accept candidates or accepts the artifact foil, current automatic physics support is over-weighted.",
        next_experiment="Complete the cycle-156 manual QC mini-batch and rerun manual-QC-gated front/echem effects.",
        suggested_script="tier4_manual_qc_gated_front_effects.py",
        evidence_support=0.62 + 0.02 * min(10.0, accept_first + artifact_first),
        novelty=0.50,
        falsifiability=0.95,
        implementability=0.60,
        guardrail_penalty=0.10,
        cost_penalty=0.70,
        human_review_need=1.0,
    )

    add_hypothesis(
        rows,
        hypothesis_id="H5",
        title="Phase-front directionality is stronger than calibrated diffusion",
        generator_agent="physics_guardrail_agent",
        claim="The current defensible physics claim is optical phase/front movement, not calibrated lithium diffusion.",
        evidence=[
            f"Diffusion/calibration guardrail: {diffusion_guardrail}",
            f"Temporal directionality guardrail: {temporal_guardrail}",
            "Several audits find front-direction or phase-fraction signal while diffusion magnitudes remain calibration- and mask-sensitive.",
        ],
        falsification="Require stable front signs, pixel/time calibration, and accepted particle/front masks before reporting diffusion-like coefficients.",
        next_experiment="Run a manual-QC-gated apparent-diffusion bounds update only on accepted particle/front masks.",
        suggested_script="tier4_manual_qc_gated_diffusion_bounds.py",
        evidence_support=0.76,
        novelty=0.58,
        falsifiability=0.84,
        implementability=0.62,
        guardrail_penalty=0.52,
        cost_penalty=0.40,
        human_review_need=0.85,
    )

    add_hypothesis(
        rows,
        hypothesis_id="H6",
        title="Spatial propagation should be tested as graph lag, not just ROI-wise classification",
        generator_agent="spatial_graph_agent",
        claim="If phase fronts are physically coordinated, nearby ROI/front events should show lagged graph structure beyond cycle-level labels.",
        evidence=[
            f"Spatial propagation guardrail: {spatial_guardrail}",
            "The existing graph outputs provide nodes and edges, but final interpretation still hinges on lag robustness and source-cycle balance.",
            "This is closer to a physics mechanism than a high-AUC weak-label classifier.",
        ],
        falsification="Shuffle cycle labels and spatial edges; true propagation should weaken under both nulls while retaining local lag consistency.",
        next_experiment="Expand graph lag tests with edge-shuffled and cycle-shuffled nulls plus echem-conditioned edge weights.",
        suggested_script="tier4_echem_conditioned_spatial_lag_nulls.py",
        evidence_support=0.57,
        novelty=0.78,
        falsifiability=0.88,
        implementability=0.66,
        guardrail_penalty=0.34,
        cost_penalty=0.35,
    )

    add_hypothesis(
        rows,
        hypothesis_id="H7",
        title="Cycle-state transitions can organize ROI degradation modes",
        generator_agent="state_space_agent",
        claim="Cycle-level electrochemical state transitions may explain which ROI physics modes appear at each cycle.",
        evidence=[
            f"Cycle-state guardrail: {cycle_state_guardrail}",
            "Echem regime, future-drop, and ROI physics audits repeatedly point to cycle-conditioned behavior.",
            "This provides a compact bridge between cell-level echem state and ROI-level optical heterogeneity.",
        ],
        falsification="Cycle-state modes must predict held-out ROI mode frequencies better than cycle number and acquisition context alone.",
        next_experiment="Join cycle-state transition features to ROI mode counts and run leave-cycle mode-frequency prediction.",
        suggested_script="tier4_cycle_state_mode_frequency_bridge.py",
        evidence_support=0.59,
        novelty=0.74,
        falsifiability=0.83,
        implementability=0.78,
        guardrail_penalty=0.30,
        cost_penalty=0.20,
    )

    prefix_class = prefix.get("top_classification", [])
    prefix_best = prefix_class[0] if prefix_class else {}
    add_hypothesis(
        rows,
        hypothesis_id="H8",
        title="Prefix traces are useful for review prioritization before event completion",
        generator_agent="early_warning_agent",
        claim="Partial ROI traces may rank candidate events early enough for review, but need stricter leakage and cycle nulls.",
        evidence=[
            f"Best prefix classification target: {prefix_best.get('target', 'NA')} with ROC AUC {fmt(first_float(prefix_best, ['roc_auc', 'mean_roc_auc']))}",
            "Prefix models already exist, making this a low-cost route to an online warning-style analysis.",
            "The result should be framed as review prioritization until manual QC labels exist.",
        ],
        falsification="Prefix performance must survive reference-cycle centering and label-time shift nulls.",
        next_experiment="Rerun prefix forecasting with manual-QC queue labels and stronger shift-null controls.",
        suggested_script="tier4_prefix_qc_queue_forecast.py",
        evidence_support=0.54,
        novelty=0.62,
        falsifiability=0.86,
        implementability=0.82,
        guardrail_penalty=0.28,
        cost_penalty=0.18,
    )

    ranked = pd.DataFrame(rows).sort_values(
        ["tournament_score", "evidence_support", "falsifiability"], ascending=False
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    flat = ranked.copy()
    flat["evidence_bullets"] = flat["evidence_bullets"].apply(lambda xs: " | ".join(xs))
    flat.to_csv(out / "agentic_current_hypothesis_tournament.csv", index=False)

    experiment_specs = []
    for row in ranked.head(5).to_dict("records"):
        experiment_specs.append(
            {
                "rank": int(row["rank"]),
                "hypothesis_id": row["hypothesis_id"],
                "title": row["title"],
                "next_experiment": row["next_experiment"],
                "suggested_script": row["suggested_script"],
                "minimum_success_evidence": row["falsification_test"],
                "required_guardrails": [
                    "leave-cycle or leave-source split where applicable",
                    "explicit acquisition/context controls",
                    "manual-QC gate before calibrated diffusion or validated degradation-mode claims",
                ],
                "do_not_claim": [
                    "calibrated lithium diffusion without accepted calibration and front masks",
                    "deployable warning model from weak labels alone",
                    "causal degradation mechanism from acquisition-confounded future8 labels",
                ],
            }
        )

    summary = {
        "n_hypotheses": int(len(ranked)),
        "top_hypothesis": ranked.iloc[0].to_dict(),
        "top_three": ranked.head(3).to_dict("records"),
        "paper_inspiration": {
            "robin": "lab-in-the-loop agents that generate, analyze, and update hypotheses",
            "co_scientist": "asynchronous generation, critique, tournament-style ranking, and refinement",
            "empirical_software": "guarded code/execution workflow with explicit evaluation criteria",
        },
        "evidence_inventory": {
            "echem_video_embedding_fusion_audit": f"{len(echem_metrics)} classification rows; future16 video+echem AUC {fmt(auc_future16_fused)}",
            "residual_dictionary_embedding_audit": f"{len(residual_metrics)} classification rows; residual future8 AUC {fmt(residual_auc)}",
            "qc_decision_evidence_ledger": f"{fmt(n_qc, 0)} pending candidates; {fmt(accept_first, 0)} possible accept-first",
            "apparent_diffusion_calibration_bounds": "used as diffusion-claim guardrail",
            "temporal_directionality_physics_audit": "used as phase-front directionality guardrail",
            "balanced_spatial_front_propagation_audit": "used for spatial-lag hypothesis",
            "cycle_state_space_transition_audit": "used for cycle-state hypothesis",
            "prefix_roi_forecast": "used for early warning/review prioritization hypothesis",
            "nmc_ai_physics_synthesis": "available" if synthesis else "missing",
        },
        "experiment_specs": experiment_specs,
        "guardrail": (
            "This tournament ranks next analyses from existing automatic evidence. It does not create manual QC labels, "
            "does not validate a deployable degradation detector, and does not license calibrated diffusion claims."
        ),
        "outputs": {
            "csv": str(out / "agentic_current_hypothesis_tournament.csv"),
            "json": str(out / "agentic_current_hypothesis_tournament.json"),
            "experiment_specs": str(out / "agentic_next_experiment_specs.json"),
            "readme": str(out / "README.md"),
        },
        "physics_guardrail": physics_guardrail,
    }

    (out / "agentic_current_hypothesis_tournament.json").write_text(json.dumps(clean_json(ranked.to_dict("records")), indent=2))
    (out / "agentic_next_experiment_specs.json").write_text(json.dumps(clean_json(experiment_specs), indent=2))
    (out / "agentic_current_hypothesis_tournament_summary.json").write_text(json.dumps(clean_json(summary), indent=2))
    write_readme(out / "README.md", ranked, clean_json(summary))


if __name__ == "__main__":
    main()
