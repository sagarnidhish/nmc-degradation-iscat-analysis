#!/usr/bin/env python3
"""Build a source-balanced transport/phase-front mechanism dossier.

This joins the newer source-balanced pre-event evidence streams into one
candidate-level table for manual review:

- apparent optical-flow transport inside history-derived particle masks
- front/kinetic concordance and strict QC gates
- observable-forecast tail descriptors
- source/echem residual front proxies
- visual asset availability and manual-QC action tiers

The result is a review dossier, not a claim generator. It explicitly keeps
diffusion and phase-boundary claims blocked unless existing strict gates pass.
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def rank01(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0 or x.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    r = x.rank(pct=True)
    if not higher_is_better:
        r = 1.0 - r
    return r.fillna(0.5)


def source_residual(df: pd.DataFrame, col: str) -> pd.Series:
    x = numeric(df, col)
    if "source_stem" not in df.columns:
        return x * np.nan
    return x - x.groupby(df["source_stem"].astype(str)).transform("mean")


def add_prefixed(base: pd.DataFrame, other: pd.DataFrame, cols: Iterable[str], prefix: str) -> pd.DataFrame:
    keep = ["roi_id"] + [c for c in cols if c in other.columns]
    tmp = other[keep].copy()
    tmp = tmp.rename(columns={c: f"{prefix}{c}" for c in tmp.columns if c != "roi_id"})
    return base.merge(tmp, on="roi_id", how="left")


def tier_row(row: pd.Series) -> str:
    if bool(row.get("automatic_diffusion_claim_gate", False)):
        return "unexpected_diffusion_gate_pass_recheck"
    if bool(row.get("manual_front_review_gate", False)) and row.get("event_relative_bin") == "near_pre_event_1_8":
        return "priority_manual_transport_front_review"
    if row.get("transport_mechanism_score", 0) >= 0.80 and row.get("event_relative_bin") == "near_pre_event_1_8":
        return "priority_transport_mechanism_review"
    if row.get("transport_mechanism_score", 0) >= 0.65:
        return "guarded_transport_hypothesis_review"
    return "routine_or_context_control"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_transport_mechanism_dossier")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    flow = read_csv(derived / "source_balanced_pre_event_optical_flow_transport_audit" / "source_balanced_pre_event_optical_flow_transport_per_roi.csv")
    manual = read_csv(derived / "source_balanced_pre_event_manual_qc_decision_packet" / "source_balanced_pre_event_manual_qc_decision_queue.csv")
    obs = read_csv(derived / "source_balanced_pre_event_observable_forecast" / "source_balanced_pre_event_observable_forecast_features.csv")
    concord = read_csv(derived / "source_balanced_pre_event_front_kinetic_concordance_audit" / "source_balanced_pre_event_front_kinetic_concordance_ranked_candidates.csv")

    dossier = flow.copy()
    manual_cols = [
        "manual_qc_rank",
        "manual_qc_decision_score",
        "manual_qc_action_tier",
        "evidence_tags",
        "review_question",
        "manual_front_review_gate",
        "automatic_diffusion_claim_gate",
        "strict_qc_priority_score",
        "n_passed_component_gates",
        "front_kinetic_concordance_score",
        "front_kinetic_tier",
        "front_evidence_score",
        "kinetic_evidence_score",
        "qc_evidence_score",
        "visual_sanity_score",
        "visual_review_score",
        "visual_front_plausibility_score",
        "visual_artifact_risk_score",
        "front_consensus_score",
        "front_radius2_slope_px2_per_norm_time",
        "front_radius2_slope_px2_per_norm_time_source_echem_context_residual",
        "frame_strip_png",
        "mask_overlay_png",
        "kymograph_png",
        "has_visual_assets",
        "front_direction_agreement_gate",
        "stable_mask_gate",
        "centroid_stability_gate",
    ]
    obs_cols = [
        "tail_particle_mean_delta",
        "tail_particle_minus_background_delta",
        "tail_contrast_delta",
        "tail_front_radius_q70_delta",
        "tail_front_radius2_slope",
        "tail_frame_diff_energy",
        "prefix_particle_mean_slope",
        "prefix_particle_minus_background_slope",
        "prefix_front_radius_q70_slope",
        "prefix_front_radius2_q70_slope",
    ]
    concord_cols = [
        "consensus_review_rank",
        "consensus_review_score",
        "matched_positive_support_count",
        "front_kinetic_concordance_score",
        "front_kinetic_tier",
        "kinetic_evidence_score",
        "front_evidence_score",
        "qc_evidence_score",
    ]
    dossier = add_prefixed(dossier, manual, manual_cols, "")
    dossier = add_prefixed(dossier, obs, obs_cols, "observable_")
    dossier = add_prefixed(dossier, concord, concord_cols, "concordance_")

    # Source-residual transport terms: high raw flow alone is too source-structured.
    for col in [
        "abs_radial_flow_mean",
        "abs_radial_flow_q90",
        "particle_flow_mag_mean",
        "curl_mean",
        "curl_q90",
        "apparent_transport_instability_score",
    ]:
        if col in dossier.columns:
            dossier[f"{col}_source_residual"] = source_residual(dossier, col)

    dossier["near_pre_flag"] = dossier["event_relative_bin"].astype(str).eq("near_pre_event_1_8").astype(int)
    dossier["visual_asset_flag"] = dossier.get("has_visual_assets", pd.Series(False, index=dossier.index)).fillna(False).astype(bool).astype(int)
    dossier["diffusion_claim_blocked"] = ~dossier.get("automatic_diffusion_claim_gate", pd.Series(False, index=dossier.index)).fillna(False).astype(bool)

    transport_raw = (
        0.35 * rank01(dossier.get("abs_radial_flow_mean", pd.Series(index=dossier.index)))
        + 0.25 * rank01(dossier.get("abs_radial_flow_q90", pd.Series(index=dossier.index)))
        + 0.20 * rank01(dossier.get("particle_flow_mag_mean", pd.Series(index=dossier.index)))
        + 0.20 * rank01(dossier.get("apparent_transport_instability_score", pd.Series(index=dossier.index)))
    )
    transport_source = (
        0.45 * rank01(dossier.get("abs_radial_flow_mean_source_residual", pd.Series(index=dossier.index)))
        + 0.25 * rank01(dossier.get("abs_radial_flow_q90_source_residual", pd.Series(index=dossier.index)))
        + 0.15 * rank01(dossier.get("particle_flow_mag_mean_source_residual", pd.Series(index=dossier.index)))
        + 0.15 * rank01(dossier.get("curl_mean_source_residual", pd.Series(index=dossier.index)))
    )
    front_kinetic = (
        0.35 * rank01(dossier.get("front_kinetic_concordance_score", pd.Series(index=dossier.index)))
        + 0.20 * rank01(dossier.get("front_evidence_score", pd.Series(index=dossier.index)))
        + 0.20 * rank01(dossier.get("kinetic_evidence_score", pd.Series(index=dossier.index)))
        + 0.15 * rank01(dossier.get("strict_qc_priority_score", pd.Series(index=dossier.index)))
        + 0.10 * rank01(dossier.get("front_consensus_score", pd.Series(index=dossier.index)))
    )
    observable = (
        0.35 * rank01(dossier.get("observable_tail_contrast_delta", pd.Series(index=dossier.index)).abs())
        + 0.25 * rank01(dossier.get("observable_tail_particle_mean_delta", pd.Series(index=dossier.index)).abs())
        + 0.20 * rank01(dossier.get("observable_tail_front_radius_q70_delta", pd.Series(index=dossier.index)).abs())
        + 0.20 * rank01(dossier.get("observable_tail_front_radius2_slope", pd.Series(index=dossier.index)).abs())
    )
    qc = (
        0.35 * rank01(dossier.get("manual_qc_decision_score", pd.Series(index=dossier.index)))
        + 0.25 * rank01(dossier.get("visual_sanity_score", pd.Series(index=dossier.index)))
        + 0.15 * rank01(dossier.get("visual_review_score", pd.Series(index=dossier.index)))
        + 0.15 * dossier["visual_asset_flag"]
        + 0.10 * dossier.get("manual_front_review_gate", pd.Series(False, index=dossier.index)).fillna(False).astype(bool).astype(int)
    )

    dossier["transport_raw_score"] = transport_raw
    dossier["transport_source_residual_score"] = transport_source
    dossier["front_kinetic_score"] = front_kinetic
    dossier["observable_tail_score"] = observable
    dossier["qc_review_score"] = qc
    dossier["transport_mechanism_score"] = (
        0.25 * dossier["transport_raw_score"]
        + 0.25 * dossier["transport_source_residual_score"]
        + 0.25 * dossier["front_kinetic_score"]
        + 0.15 * dossier["observable_tail_score"]
        + 0.10 * dossier["qc_review_score"]
    )
    dossier["transport_review_tier"] = dossier.apply(tier_row, axis=1)
    dossier["transport_review_rank"] = dossier["transport_mechanism_score"].rank(ascending=False, method="first").astype(int)
    dossier = dossier.sort_values("transport_review_rank")

    top = dossier.head(40).copy()
    immediate = dossier[dossier["transport_review_tier"].isin(["priority_manual_transport_front_review", "priority_transport_mechanism_review"])].head(24).copy()
    source_summary = dossier.groupby("source_stem", dropna=False).agg(
        n_roi=("roi_id", "count"),
        n_near_pre=("near_pre_flag", "sum"),
        max_transport_mechanism_score=("transport_mechanism_score", "max"),
        median_transport_source_residual_score=("transport_source_residual_score", "median"),
        n_priority=("transport_review_tier", lambda s: int(s.astype(str).str.startswith("priority").sum())),
    ).reset_index().sort_values("max_transport_mechanism_score", ascending=False)
    tier_summary = dossier["transport_review_tier"].value_counts().rename_axis("transport_review_tier").reset_index(name="n")

    cols_front = [
        "transport_review_rank",
        "roi_id",
        "cycleNo",
        "source_stem",
        "event_relative_bin",
        "cycles_to_next_event",
        "transport_review_tier",
        "transport_mechanism_score",
        "transport_raw_score",
        "transport_source_residual_score",
        "front_kinetic_score",
        "observable_tail_score",
        "qc_review_score",
        "abs_radial_flow_mean",
        "abs_radial_flow_mean_source_residual",
        "particle_flow_mag_mean",
        "curl_mean",
        "front_kinetic_concordance_score",
        "front_kinetic_tier",
        "manual_qc_action_tier",
        "manual_front_review_gate",
        "automatic_diffusion_claim_gate",
        "diffusion_claim_blocked",
        "strict_qc_priority_score",
        "visual_sanity_score",
        "visual_review_score",
        "visual_front_plausibility_score",
        "observable_tail_contrast_delta",
        "observable_tail_front_radius_q70_delta",
        "observable_tail_front_radius2_slope",
        "frame_strip_png",
        "mask_overlay_png",
        "kymograph_png",
        "review_question",
        "evidence_tags",
    ]
    ordered_cols = [c for c in cols_front if c in dossier.columns] + [c for c in dossier.columns if c not in cols_front]
    dossier = dossier[ordered_cols]
    top = top[[c for c in ordered_cols if c in top.columns]]
    immediate = immediate[[c for c in ordered_cols if c in immediate.columns]]

    paths = {
        "dossier": out / "source_balanced_transport_mechanism_dossier.csv",
        "top40": out / "source_balanced_transport_mechanism_top40.csv",
        "immediate_review": out / "source_balanced_transport_mechanism_immediate_review.csv",
        "source_summary": out / "source_balanced_transport_mechanism_source_summary.csv",
        "tier_summary": out / "source_balanced_transport_mechanism_tier_summary.csv",
        "summary": out / "source_balanced_transport_mechanism_summary.json",
    }
    dossier.to_csv(paths["dossier"], index=False)
    top.to_csv(paths["top40"], index=False)
    immediate.to_csv(paths["immediate_review"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)
    tier_summary.to_csv(paths["tier_summary"], index=False)

    top_record = top.head(1).to_dict("records")[0] if not top.empty else {}
    summary = {
        "n_rows": int(len(dossier)),
        "n_sources": int(dossier["source_stem"].nunique()),
        "n_cycles": int(pd.to_numeric(dossier["cycleNo"], errors="coerce").nunique()),
        "tier_counts": tier_summary.to_dict("records"),
        "n_immediate_review": int(len(immediate)),
        "n_diffusion_claim_candidates": int((~dossier["diffusion_claim_blocked"]).sum()),
        "top_candidate": clean_json(top_record),
        "top40_event_bin_counts": top["event_relative_bin"].value_counts().to_dict() if not top.empty else {},
        "source_summary_top": clean_json(source_summary.head(12).to_dict("records")),
        "guardrail": "This dossier ranks apparent transport/front/kinetic candidates for manual review. Automatic optical flow and front proxies are not calibrated phase-boundary velocities, material fluxes, or diffusion coefficients; diffusion claims remain blocked unless strict gates and manual validation pass.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Source-Balanced Transport Mechanism Dossier\n\n"
        "Candidate-level joined dossier for apparent optical transport, front/kinetic concordance, observable-tail dynamics, and manual-QC review priority.\n\n"
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}\n"
        f"- Immediate-review rows: {summary['n_immediate_review']}\n"
        f"- Diffusion-claim candidates: {summary['n_diffusion_claim_candidates']}\n"
        f"- Top candidate: {top_record.get('roi_id', 'NA')}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True)[:12000])


if __name__ == "__main__":
    main()
