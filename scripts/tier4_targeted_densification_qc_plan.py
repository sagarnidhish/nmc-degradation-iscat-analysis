#!/usr/bin/env python3
"""Targeted densification and manual-QC plan for Alek/Jiho NMC analyses.

This audit turns the all-cycle coverage atlas and candidate-level AI/physics ledgers
into an action queue. It is deliberately operational: identify which cycles need
more ROI density and which existing ROI assets should be manually reviewed first.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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


def norm01(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    lo, hi = vals.min(), vals.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.0, index=series.index)
    return ((vals - lo) / (hi - lo)).fillna(0.0)


def cycle_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cycleNo" in out.columns:
        out["cycleNo"] = pd.to_numeric(out["cycleNo"], errors="coerce")
    if "source_stem" not in out.columns:
        out["source_stem"] = pd.NA
    return out


def best_by_cycle(df: pd.DataFrame, prefix: str, score_cols: list[str]) -> pd.DataFrame:
    if df.empty or "cycleNo" not in df.columns:
        return pd.DataFrame(columns=["cycleNo", "source_stem"])
    out = cycle_key(df)
    agg: dict[str, tuple[str, str]] = {}
    for col in score_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            agg[f"{prefix}_max_{col}"] = (col, "max")
            agg[f"{prefix}_mean_{col}"] = (col, "mean")
    if "roi_id" in out.columns:
        agg[f"{prefix}_n_roi_rows"] = ("roi_id", "nunique")
    else:
        agg[f"{prefix}_n_rows"] = ("cycleNo", "size")
    if not agg:
        return out.groupby(["cycleNo", "source_stem"], dropna=False).size().reset_index(name=f"{prefix}_n_rows")
    return out.groupby(["cycleNo", "source_stem"], dropna=False).agg(**agg).reset_index()


def merge_cycle(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra.empty:
        return base
    return base.merge(extra, on=["cycleNo", "source_stem"], how="left")


def first_existing(row: pd.Series, cols: list[str]) -> Any:
    for c in cols:
        if c in row.index and pd.notna(row[c]) and row[c] != "":
            return row[c]
    return None


def make_roi_queue(derived: Path, cycle_plan: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    transfer = read_csv(derived / "source_heldout_event_rank_transfer_audit" / "source_heldout_event_rank_transfer_predictions.csv")
    if not transfer.empty:
        t = cycle_key(transfer)
        t["candidate_origin"] = "source_heldout_transfer"
        t["candidate_score"] = pd.to_numeric(t.get("transfer_oriented_feature_score"), errors="coerce")
        t["candidate_action"] = "manual_qc_transfer_ranked_roi"
        frames.append(t)

    mechanism = read_csv(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_top40.csv")
    if not mechanism.empty:
        m = cycle_key(mechanism)
        m["candidate_origin"] = "transport_mechanism_dossier"
        m["candidate_score"] = pd.to_numeric(m.get("transport_mechanism_score"), errors="coerce")
        m["candidate_action"] = "manual_qc_transport_mechanism_roi"
        frames.append(m)

    manual = read_csv(derived / "source_balanced_pre_event_manual_qc_decision_packet" / "source_balanced_pre_event_manual_qc_decision_queue.csv")
    if not manual.empty:
        q = cycle_key(manual)
        q["candidate_origin"] = "manual_qc_decision_packet"
        score_col = "manual_qc_decision_score" if "manual_qc_decision_score" in q.columns else "review_priority"
        q["candidate_score"] = pd.to_numeric(q.get(score_col), errors="coerce")
        q["candidate_action"] = "manual_qc_existing_visual_assets"
        frames.append(q)

    diffusion = read_csv(derived / "diffusion_unblock_sensitivity_audit" / "diffusion_unblock_review_queue.csv")
    if not diffusion.empty:
        d = cycle_key(diffusion)
        d["candidate_origin"] = "diffusion_unblock_review_queue"
        d["candidate_score"] = pd.to_numeric(d.get("review_priority"), errors="coerce")
        d["candidate_action"] = "manual_qc_diffusion_blocker_followup"
        frames.append(d)

    if not frames:
        return pd.DataFrame()
    cols_keep = set()
    for f in frames:
        cols_keep.update(f.columns)
    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    all_rows = all_rows.dropna(subset=["cycleNo"])
    cycle_cols = [
        "cycleNo", "source_stem", "cycle_action_priority", "recommended_cycle_action", "coverage_gap_priority",
        "n_roi_total_across_tracked_cohorts", "future_any_drop_within_8cycles", "future_any_drop_within_16cycles",
        "any_abrupt_drop", "cross_modal_consensus_score", "hazard_probability_mean", "echem_outlier_score",
    ]
    all_rows = all_rows.merge(cycle_plan[[c for c in cycle_cols if c in cycle_plan.columns]], on=["cycleNo", "source_stem"], how="left")
    all_rows["candidate_score_norm"] = norm01(all_rows["candidate_score"])
    all_rows["cycle_action_priority"] = pd.to_numeric(all_rows.get("cycle_action_priority"), errors="coerce").fillna(0)
    all_rows["roi_action_priority"] = all_rows["cycle_action_priority"] + 4.0 * all_rows["candidate_score_norm"]
    all_rows["has_visual_asset"] = False
    for c in ["frame_strip_png", "mask_overlay_png", "kymograph_png", "visual_asset_path", "preview_png"]:
        if c in all_rows.columns:
            all_rows["has_visual_asset"] = all_rows["has_visual_asset"] | all_rows[c].notna()
    all_rows.loc[all_rows["has_visual_asset"], "roi_action_priority"] += 0.75
    all_rows["roi_id"] = all_rows.get("roi_id", pd.Series([pd.NA] * len(all_rows))).fillna(all_rows.get("selected_roi_id", pd.Series([pd.NA] * len(all_rows))))
    selected_cols = [
        "roi_action_priority", "candidate_origin", "candidate_action", "roi_id", "selected_roi_id", "cycleNo", "source_stem",
        "event_relative_bin", "target_label", "candidate_score", "candidate_score_norm", "cycle_action_priority",
        "recommended_cycle_action", "transport_mechanism_score", "transfer_oriented_feature_score", "fixed_transport_mechanism_score",
        "qc_review_score", "manual_qc_action_tier", "manual_front_review_gate", "visual_sanity_score", "review_question",
        "evidence_tags", "automatic_physics_consistent", "publication_ready", "n_all_blockers", "blocker_summary",
        "frame_strip_png", "mask_overlay_png", "kymograph_png", "npz_path", "has_visual_asset",
    ]
    selected_cols = [c for c in selected_cols if c in all_rows.columns]
    out = all_rows[selected_cols].sort_values("roi_action_priority", ascending=False)
    out = out.drop_duplicates(subset=[c for c in ["candidate_origin", "roi_id", "cycleNo", "source_stem"] if c in out.columns])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/targeted_densification_qc_plan")
    args = parser.parse_args()
    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cycle = cycle_key(read_csv(derived / "all_cycle_dataset_coverage_atlas" / "all_cycle_coverage_gap_priority.csv"))
    if cycle.empty:
        raise FileNotFoundError("all_cycle_coverage_gap_priority.csv")

    source_summary = read_csv(derived / "all_cycle_dataset_coverage_atlas" / "all_cycle_source_coverage_summary.csv")
    transfer = read_csv(derived / "source_heldout_event_rank_transfer_audit" / "source_heldout_event_rank_transfer_predictions.csv")
    mechanism = read_csv(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_top40.csv")
    manual = read_csv(derived / "source_balanced_pre_event_manual_qc_decision_packet" / "source_balanced_pre_event_manual_qc_decision_queue.csv")
    diffusion = read_csv(derived / "diffusion_unblock_sensitivity_audit" / "diffusion_unblock_review_queue.csv")

    plan = cycle.copy()
    plan = merge_cycle(plan, best_by_cycle(transfer, "transfer", ["transfer_oriented_feature_score", "fixed_transport_mechanism_score", "fixed_qc_review_score"]))
    plan = merge_cycle(plan, best_by_cycle(mechanism, "mechanism", ["transport_mechanism_score", "front_kinetic_score", "qc_review_score", "visual_sanity_score"]))
    plan = merge_cycle(plan, best_by_cycle(manual, "manual_qc", ["manual_qc_decision_score", "strict_qc_priority_score", "front_evidence_score", "kinetic_evidence_score"]))
    plan = merge_cycle(plan, best_by_cycle(diffusion, "diffusion", ["review_priority", "gate_count", "median_radius2_fit_r2"]))

    numeric_cols = [c for c in plan.columns if c.startswith(("transfer_max_", "mechanism_max_", "manual_qc_max_", "diffusion_max_"))]
    plan["cycle_action_priority"] = pd.to_numeric(plan.get("coverage_gap_priority"), errors="coerce").fillna(0)
    for c in numeric_cols:
        weight = 1.0
        if "transfer_oriented" in c or "transport_mechanism" in c or "manual_qc_decision" in c:
            weight = 1.5
        if "diffusion_max_review_priority" in c:
            weight = 0.08
        plan["cycle_action_priority"] += weight * norm01(plan[c])
    plan["roi_count_penalty_bonus"] = 0.0
    plan.loc[pd.to_numeric(plan.get("n_roi_total_across_tracked_cohorts"), errors="coerce").fillna(0) == 0, "roi_count_penalty_bonus"] = 3.0
    plan.loc[pd.to_numeric(plan.get("n_roi_total_across_tracked_cohorts"), errors="coerce").fillna(0).between(1, 8), "roi_count_penalty_bonus"] = 1.0
    plan["cycle_action_priority"] += plan["roi_count_penalty_bonus"]

    no_roi = ~plan.get("has_any_roi_video_sequence", pd.Series(False, index=plan.index)).astype(bool)
    few_roi = pd.to_numeric(plan.get("n_roi_total_across_tracked_cohorts"), errors="coerce").fillna(0) < 10
    high_manual = plan[[c for c in plan.columns if c.startswith("manual_qc_")]].notna().any(axis=1) if any(c.startswith("manual_qc_") for c in plan.columns) else pd.Series(False, index=plan.index)
    high_diff = plan[[c for c in plan.columns if c.startswith("diffusion_")]].notna().any(axis=1) if any(c.startswith("diffusion_") for c in plan.columns) else pd.Series(False, index=plan.index)
    plan["recommended_cycle_action"] = "manual_qc_existing_roi_assets"
    plan.loc[few_roi & ~no_roi, "recommended_cycle_action"] = "densify_existing_cycle_roi_candidates"
    plan.loc[no_roi, "recommended_cycle_action"] = "extract_roi_candidates_for_uncovered_cycle"
    plan.loc[high_diff & ~no_roi, "recommended_cycle_action"] = "manual_qc_diffusion_blocker_followup"
    plan.loc[high_manual & ~no_roi, "recommended_cycle_action"] = "manual_qc_existing_visual_assets"

    plan = plan.sort_values("cycle_action_priority", ascending=False)
    roi_queue = make_roi_queue(derived, plan)

    source_plan = plan.groupby("source_stem", dropna=False).agg(
        n_cycles=("cycleNo", "nunique"),
        max_cycle_action_priority=("cycle_action_priority", "max"),
        mean_cycle_action_priority=("cycle_action_priority", "mean"),
        n_uncovered_cycles=("has_any_roi_video_sequence", lambda s: int((~s.astype(bool)).sum())),
        n_low_roi_cycles=("n_roi_total_across_tracked_cohorts", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) < 10).sum())),
        n_future8_positive=("future_any_drop_within_8cycles", "sum"),
        n_future16_positive=("future_any_drop_within_16cycles", "sum"),
        total_roi_rows=("n_roi_total_across_tracked_cohorts", "sum"),
    ).reset_index().sort_values("max_cycle_action_priority", ascending=False)
    if not source_summary.empty and "source_stem" in source_summary.columns:
        source_plan = source_plan.merge(source_summary, on="source_stem", how="left", suffixes=("", "_coverage_atlas"))

    action_counts = plan["recommended_cycle_action"].value_counts(dropna=False).reset_index()
    action_counts.columns = ["recommended_cycle_action", "n_cycles"]
    roi_origin_counts = roi_queue["candidate_origin"].value_counts(dropna=False).reset_index() if not roi_queue.empty else pd.DataFrame(columns=["candidate_origin", "n_roi_rows"])
    if not roi_origin_counts.empty:
        roi_origin_counts.columns = ["candidate_origin", "n_roi_rows"]

    plan.to_csv(out / "targeted_densification_cycle_plan.csv", index=False)
    source_plan.to_csv(out / "targeted_densification_source_plan.csv", index=False)
    action_counts.to_csv(out / "targeted_densification_action_counts.csv", index=False)
    roi_origin_counts.to_csv(out / "targeted_densification_roi_origin_counts.csv", index=False)
    if not roi_queue.empty:
        roi_queue.to_csv(out / "targeted_manual_qc_roi_queue.csv", index=False)

    summary = {
        "overall_status": "targeted_densification_qc_plan_ready",
        "n_cycle_rows": int(len(plan)),
        "n_sources": int(plan["source_stem"].nunique(dropna=True)),
        "n_uncovered_cycles": int((~plan.get("has_any_roi_video_sequence", pd.Series(False, index=plan.index)).astype(bool)).sum()),
        "n_low_roi_cycles_lt10": int((pd.to_numeric(plan.get("n_roi_total_across_tracked_cohorts"), errors="coerce").fillna(0) < 10).sum()),
        "n_roi_queue_rows": int(len(roi_queue)),
        "top_cycle_actions": plan.head(20).to_dict("records"),
        "top_roi_actions": roi_queue.head(30).to_dict("records") if not roi_queue.empty else [],
        "source_plan_top": source_plan.head(12).to_dict("records"),
        "action_counts": action_counts.to_dict("records"),
        "roi_origin_counts": roi_origin_counts.to_dict("records"),
        "inputs": {
            "all_cycle_coverage": str(derived / "all_cycle_dataset_coverage_atlas" / "all_cycle_coverage_gap_priority.csv"),
            "heldout_transfer_predictions": str(derived / "source_heldout_event_rank_transfer_audit" / "source_heldout_event_rank_transfer_predictions.csv"),
            "mechanism_top40": str(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_top40.csv"),
            "manual_qc_decision_queue": str(derived / "source_balanced_pre_event_manual_qc_decision_packet" / "source_balanced_pre_event_manual_qc_decision_queue.csv"),
            "diffusion_unblock_queue": str(derived / "diffusion_unblock_sensitivity_audit" / "diffusion_unblock_review_queue.csv"),
        },
        "outputs": {
            "cycle_plan": str(out / "targeted_densification_cycle_plan.csv"),
            "source_plan": str(out / "targeted_densification_source_plan.csv"),
            "roi_queue": str(out / "targeted_manual_qc_roi_queue.csv"),
            "action_counts": str(out / "targeted_densification_action_counts.csv"),
            "roi_origin_counts": str(out / "targeted_densification_roi_origin_counts.csv"),
            "summary": str(out / "targeted_densification_qc_summary.json"),
        },
        "guardrail": "This plan prioritizes ROI densification and manual-QC review using existing automatic ledgers. It does not extract new ROIs, accept labels, relax gates, or create calibrated diffusion/phase-boundary claims.",
    }
    (out / "targeted_densification_qc_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    readme = [
        "# Targeted Densification and Manual-QC Plan",
        "",
        "Combines the all-cycle coverage atlas with transfer, mechanism, manual-QC, and diffusion-blocker ledgers.",
        "",
        f"- Cycle rows: {summary['n_cycle_rows']}",
        f"- Sources: {summary['n_sources']}",
        f"- Uncovered cycles: {summary['n_uncovered_cycles']}",
        f"- Low-ROI cycles (<10 rows across tracked cohorts): {summary['n_low_roi_cycles_lt10']}",
        f"- ROI action rows: {summary['n_roi_queue_rows']}",
        "",
        f"Guardrail: {summary['guardrail']}",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")


if __name__ == "__main__":
    main()
