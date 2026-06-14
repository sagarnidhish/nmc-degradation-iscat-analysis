#!/usr/bin/env python3
"""Build a cycle-level cross-modal degradation consensus table.

The goal is not to create another black-box classifier.  This script asks
whether independent modalities point at the same cycles: particle traces,
frame-count/acquisition context, electrochemistry state, ROI front motion,
masked-rollout residuals, and integrated event evidence.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rank01(series: pd.Series, high_risk: bool = True) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() <= 1:
        return pd.Series(np.nan, index=series.index)
    ranks = vals.rank(pct=True, method="average")
    return ranks if high_risk else 1.0 - ranks


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    std = vals.std(skipna=True)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (vals - vals.mean(skipna=True)) / std


def coalesce(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def merge_cycle(base: pd.DataFrame, other: pd.DataFrame, cols: Iterable[str], prefix: str = "") -> pd.DataFrame:
    if other.empty or "cycleNo" not in other.columns:
        return base
    keep = ["cycleNo"] + [c for c in cols if c in other.columns]
    if len(keep) == 1:
        return base
    tmp = other[keep].copy()
    if prefix:
        rename = {c: f"{prefix}{c}" for c in keep if c != "cycleNo"}
        tmp = tmp.rename(columns=rename)
    tmp["cycleNo"] = pd.to_numeric(tmp["cycleNo"], errors="coerce")
    tmp = tmp.dropna(subset=["cycleNo"]).drop_duplicates("cycleNo", keep="first")
    return base.merge(tmp, on="cycleNo", how="left")


def spearman_like(x: pd.Series, y: pd.Series) -> Optional[float]:
    joined = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(joined) < 4 or joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return None
    return float(joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank()))


def binary_contrast(frame: pd.DataFrame, feature: str, target: str) -> Dict[str, object]:
    if feature not in frame.columns or target not in frame.columns:
        return {}
    tmp = frame[[feature, target]].copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty or tmp[target].nunique() < 2:
        return {}
    pos = tmp.loc[tmp[target] > 0, feature]
    neg = tmp.loc[tmp[target] <= 0, feature]
    if pos.empty or neg.empty:
        return {}
    return {
        "feature": feature,
        "target": target,
        "n": int(len(tmp)),
        "n_positive": int((tmp[target] > 0).sum()),
        "median_positive": float(pos.median()),
        "median_negative": float(neg.median()),
        "median_difference": float(pos.median() - neg.median()),
        "spearman_rho": spearman_like(tmp[feature], tmp[target]),
    }


def classify(row: pd.Series) -> str:
    event_now = bool(row.get("any_abrupt_drop", 0) > 0)
    sync = bool(row.get("synchronized_drop_2plus", 0) > 0)
    low_frame = bool(row.get("acquisition_vote", 0) > 0)
    future = bool(row.get("future_any_drop_within_8cycles", 0) > 0)
    n_votes = int(row.get("n_modal_votes", 0))
    score = row.get("cross_modal_consensus_score", np.nan)
    if sync and event_now and n_votes >= 4:
        return "synchronized_multimodal_degradation_candidate"
    if event_now and low_frame and n_votes >= 3:
        return "event_with_acquisition_confounder"
    if (not event_now) and future and n_votes >= 3 and pd.notna(score) and score >= 0.65:
        return "pre_event_multimodal_warning"
    if (not event_now) and n_votes >= 3:
        return "multimodal_outlier_without_trace_drop"
    if event_now:
        return "trace_event_low_modal_support"
    return "low_consensus"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/cross_modal_degradation_consensus")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trace = read_csv(derived / "particle_trace_physics_audit" / "particle_trace_cycle_features.csv")
    if trace.empty:
        trace = read_csv(derived / "particle_event_targets" / "particle_event_training_table.csv")
    if trace.empty or "cycleNo" not in trace.columns:
        raise FileNotFoundError("Need particle trace cycle table with cycleNo")

    base_cols = [
        "cycleNo",
        "n_frames",
        "frames_percentile",
        "drop_count",
        "any_abrupt_drop",
        "synchronized_drop_2plus",
        "synchronized_drop_3plus",
        "mean_abs_delta_prev",
        "max_abs_delta_prev",
        "delta_std_across_particles",
        "particle_norm_cv",
        "future_any_drop_within_8cycles",
        "future_sync2_drop_within_8cycles",
        "cycles_to_next_drop_within_8",
        "capacity_mAh",
        "coulombic_efficiency_pct",
        "V_min",
        "V_max",
        "n_points",
    ]
    cycles = trace[[c for c in base_cols if c in trace.columns]].copy()
    cycles["cycleNo"] = pd.to_numeric(cycles["cycleNo"], errors="coerce")
    cycles = cycles.dropna(subset=["cycleNo"]).drop_duplicates("cycleNo", keep="first").sort_values("cycleNo")

    event_evidence = read_csv(derived / "integrated_event_evidence" / "integrated_event_evidence.csv")
    cycles = merge_cycle(
        cycles,
        event_evidence,
        [
            "n_event_particles",
            "mean_drop_frac",
            "n_sustained_next_cycle",
            "global_frame_percentile",
            "synchronized_event",
            "evidence_score",
            "degradation_mode_hypothesis",
        ],
        prefix="integrated_",
    )

    hazard = read_csv(derived / "cycle_hazard_warning_audit" / "cycle_hazard_warning_predictions.csv")
    if not hazard.empty and {"cycleNo", "predicted_probability"}.issubset(hazard.columns):
        hazard = hazard.copy()
        hazard["predicted_probability"] = pd.to_numeric(hazard["predicted_probability"], errors="coerce")
        hazard_agg = hazard.groupby("cycleNo", as_index=False).agg(
            hazard_probability_mean=("predicted_probability", "mean"),
            hazard_probability_max=("predicted_probability", "max"),
            hazard_n_predictions=("predicted_probability", "count"),
        )
        cycles = merge_cycle(cycles, hazard_agg, hazard_agg.columns.drop("cycleNo"))

    roi_front = read_csv(derived / "balanced_future_roi_physics_audit" / "balanced_future_cycle_collapsed_features.csv")
    cycles = merge_cycle(
        cycles,
        roi_front,
        [
            "phase_slope_abs_median_per_s",
            "radius2_slope_median_px2_per_s",
            "diffusion_proxy_abs_median_um2_per_s",
            "threshold_robust_phase_score",
            "threshold_robust_diffusion_score",
            "roi_norm_mean_delta_last_minus_first",
            "stage_drift_xy_sampled",
            "object_mean_residual",
            "transferred_masked_residual_signature",
            "persistence_particle_mse_fraction_of_full_mean",
            "low_rank_dmd_particle_mse_fraction_of_full_mean",
            "velocity_particle_mse_fraction_of_full_mean",
            "selection_subrole",
        ],
        prefix="roi_",
    )

    temporal = read_csv(derived / "temporal_directionality_physics_audit" / "temporal_directionality_cycle_summary.csv")
    cycles = merge_cycle(
        cycles,
        temporal,
        [
            "cycles_to_next_drop",
            "cycles_since_previous_drop",
            "radius2_slope_median_px2_per_s",
            "diffusion_proxy_median_um2_per_s",
            "persistence_particle_mse_fraction_of_full_mean",
        ],
        prefix="temporal_",
    )

    state = read_csv(derived / "cycle_state_space_transition_audit" / "cycle_state_space_table.csv")
    cycles = merge_cycle(
        cycles,
        state,
        ["degradation_state_axis", "cycle_state_cluster", "state_step_norm", "axis_step", "shape_V_range", "shape_I_abs_mean_mA", "shape_dVdt_abs_p95"],
        prefix="state_",
    )

    joint = read_csv(derived / "roi_joint_physics_degradation_modes" / "joint_cycle_summary.csv")
    cycles = merge_cycle(
        cycles,
        joint,
        ["n_roi", "joint_degradation_score", "rollout_residual_energy_mean", "radius2_slope_full_px2_per_s", "active_fraction_last", "evidence_score"],
        prefix="joint_",
    )

    score_components: Dict[str, pd.Series] = {
        "trace_delta_rank": rank01(coalesce(cycles, "max_abs_delta_prev")),
        "trace_sync_rank": rank01(coalesce(cycles, "drop_count")),
        "particle_heterogeneity_rank": rank01(coalesce(cycles, "particle_norm_cv")),
        "low_frame_rank": rank01(coalesce(cycles, "frames_percentile"), high_risk=False),
        "integrated_evidence_rank": rank01(coalesce(cycles, "integrated_evidence_score")),
        "hazard_probability_rank": rank01(coalesce(cycles, "hazard_probability_max")),
        "front_motion_rank": rank01(coalesce(cycles, "roi_diffusion_proxy_abs_median_um2_per_s")),
        "front_threshold_rank": rank01(coalesce(cycles, "roi_threshold_robust_phase_score")),
        "masked_residual_rank": rank01(coalesce(cycles, "roi_transferred_masked_residual_signature")),
        "rollout_residual_rank": rank01(coalesce(cycles, "joint_rollout_residual_energy_mean")),
        "state_transition_rank": rank01(coalesce(cycles, "state_state_step_norm")),
        "echem_shape_rank": rank01(coalesce(cycles, "state_shape_dVdt_abs_p95")),
    }
    for name, values in score_components.items():
        cycles[name] = values

    modality_map = {
        "trace_vote": ["trace_delta_rank", "trace_sync_rank", "particle_heterogeneity_rank"],
        "acquisition_vote": ["low_frame_rank"],
        "integrated_event_vote": ["integrated_evidence_rank"],
        "ai_rollout_vote": ["hazard_probability_rank", "masked_residual_rank", "rollout_residual_rank"],
        "front_motion_vote": ["front_motion_rank", "front_threshold_rank"],
        "echem_state_vote": ["state_transition_rank", "echem_shape_rank"],
    }
    for vote, cols in modality_map.items():
        present = [c for c in cols if c in cycles.columns]
        cycles[f"{vote}_score"] = cycles[present].max(axis=1, skipna=True)
        cycles[vote] = (cycles[f"{vote}_score"] >= 0.85).astype(int)

    score_cols = list(score_components)
    cycles["cross_modal_consensus_score"] = cycles[score_cols].mean(axis=1, skipna=True)
    vote_cols = list(modality_map)
    cycles["n_modal_votes"] = cycles[vote_cols].sum(axis=1)
    cycles["consensus_class"] = cycles.apply(classify, axis=1)

    cycles["cross_modal_consensus_z"] = zscore(cycles["cross_modal_consensus_score"])
    cycles["trace_delta_z"] = zscore(coalesce(cycles, "max_abs_delta_prev"))
    cycles["low_frame_z"] = zscore(1.0 - pd.to_numeric(coalesce(cycles, "frames_percentile"), errors="coerce"))

    ordered = cycles.sort_values(
        ["n_modal_votes", "cross_modal_consensus_score", "integrated_evidence_score"],
        ascending=[False, False, False],
    )
    ordered.to_csv(out / "cross_modal_consensus_cycle_table.csv", index=False)

    class_summary = (
        ordered.groupby("consensus_class", dropna=False)
        .agg(
            n_cycles=("cycleNo", "count"),
            median_consensus_score=("cross_modal_consensus_score", "median"),
            mean_modal_votes=("n_modal_votes", "mean"),
            event_rate=("any_abrupt_drop", "mean"),
            future8_rate=("future_any_drop_within_8cycles", "mean"),
            median_frames_percentile=("frames_percentile", "median"),
        )
        .reset_index()
        .sort_values(["median_consensus_score", "n_cycles"], ascending=[False, False])
    )
    class_summary.to_csv(out / "cross_modal_consensus_class_summary.csv", index=False)

    top_cols = [
        "cycleNo",
        "consensus_class",
        "cross_modal_consensus_score",
        "n_modal_votes",
        "any_abrupt_drop",
        "synchronized_drop_2plus",
        "future_any_drop_within_8cycles",
        "frames_percentile",
        "drop_count",
        "max_abs_delta_prev",
        "integrated_evidence_score",
        "hazard_probability_max",
        "roi_diffusion_proxy_abs_median_um2_per_s",
        "roi_transferred_masked_residual_signature",
        "joint_rollout_residual_energy_mean",
        "state_state_step_norm",
        "trace_vote",
        "acquisition_vote",
        "integrated_event_vote",
        "ai_rollout_vote",
        "front_motion_vote",
        "echem_state_vote",
        "integrated_degradation_mode_hypothesis",
    ]
    ordered[[c for c in top_cols if c in ordered.columns]].head(25).to_csv(out / "cross_modal_consensus_top_cycles.csv", index=False)

    target_tests: List[Dict[str, object]] = []
    for target in ["any_abrupt_drop", "future_any_drop_within_8cycles", "synchronized_drop_2plus"]:
        for feature in ["cross_modal_consensus_score", "n_modal_votes", "low_frame_rank", "hazard_probability_max", "roi_transferred_masked_residual_signature"]:
            row = binary_contrast(ordered, feature, target)
            if row:
                target_tests.append(row)
    pd.DataFrame(target_tests).to_csv(out / "cross_modal_consensus_target_contrasts.csv", index=False)

    top_records = ordered[[c for c in top_cols if c in ordered.columns]].head(10).to_dict("records")
    summary = {
        "n_cycles": int(len(ordered)),
        "n_cycles_with_any_modal_vote": int((ordered["n_modal_votes"] > 0).sum()),
        "median_consensus_score": float(ordered["cross_modal_consensus_score"].median()),
        "top_cycle": top_records[0] if top_records else {},
        "class_summary": class_summary.to_dict("records"),
        "target_contrasts": target_tests,
        "top_cycles": top_records,
        "modal_vote_threshold": 0.85,
        "score_guardrail": (
            "Consensus score is an audit/ranking statistic across already-derived modalities; "
            "it is not a calibrated probability and keeps frame-count/acquisition as an explicit confounder."
        ),
    }
    with (out / "cross_modal_degradation_consensus_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Cross-Modal Degradation Consensus",
        "",
        "This audit ranks cycles by agreement across particle traces, acquisition context, integrated event evidence, ROI fronts, masked/rollout residuals, and electrochemical state features.",
        "",
        f"- Cycles scored: {summary['n_cycles']}",
        f"- Cycles with at least one modal vote: {summary['n_cycles_with_any_modal_vote']}",
        f"- Median consensus score: {summary['median_consensus_score']:.3f}",
        f"- Modal vote threshold: {summary['modal_vote_threshold']:.2f}",
        "",
        "Guardrail: the score is not a calibrated degradation probability. Frame-count/acquisition receives its own vote so cycles 86/116 can remain high-confidence optical events while still flagged as acquisition-confounded.",
        "",
        "Top cycles are in `cross_modal_consensus_top_cycles.csv`; full joined table is in `cross_modal_consensus_cycle_table.csv`.",
    ]
    (out / "README.md").write_text("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
