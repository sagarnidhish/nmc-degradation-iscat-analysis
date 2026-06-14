#!/usr/bin/env python3
"""QC decision evidence ledger for NMC ROI/front candidates.

This script does not assign manual labels. It consolidates the manual-QC
workbook, automatic QC surrogate, physics-consistency matrix, active-learning
queue, and balanced future-drop ROI evidence into a reviewer-facing ledger that
prioritizes which particle-region candidates should be inspected first and why.
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
        return pd.DataFrame()
    return pd.read_csv(path)


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def text(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    return df[col].fillna("").astype(str)


def rank01(series: pd.Series, high: bool = True, fill: float = 0.0) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() <= 1:
        return pd.Series(fill, index=series.index, dtype=float)
    r = vals.rank(pct=True, method="average")
    if not high:
        r = 1.0 - r
    return r.fillna(fill)


def merge_if(base: pd.DataFrame, other: pd.DataFrame, cols: Iterable[str], suffix: str = "") -> pd.DataFrame:
    if other.empty or "roi_id" not in other.columns:
        return base
    keep = ["roi_id"] + [c for c in cols if c in other.columns and c != "roi_id"]
    if len(keep) <= 1:
        return base
    tmp = other[keep].drop_duplicates("roi_id", keep="first").copy()
    if suffix:
        tmp = tmp.rename(columns={c: f"{c}{suffix}" for c in tmp.columns if c != "roi_id"})
    return base.merge(tmp, on="roi_id", how="left")


def any_path(row: pd.Series, cols: Iterable[str]) -> bool:
    for col in cols:
        value = row.get(col, "")
        if isinstance(value, str) and value.strip():
            return True
    return False


def top_reason_tokens(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    tier = str(row.get("auto_qc_tier", ""))
    if tier == "auto_surrogate_likely_interpretable":
        reasons.append("auto_likely_interpretable")
    if tier == "auto_surrogate_artifact_risk":
        reasons.append("artifact_risk_review")
    if str(row.get("physics_consistency_tier", "")).startswith("cross_modal"):
        reasons.append(str(row.get("physics_consistency_tier")))
    if float(row.get("physics_pillar_strong_support_count", 0) or 0) >= 4:
        reasons.append("multi_pillar_support")
    if float(row.get("surrogate_front_mask_score", 0) or 0) >= 0.65:
        reasons.append("front_mask_candidate")
    if float(row.get("surrogate_diffusion_interpretability_score", 0) or 0) >= 0.55:
        reasons.append("diffusion_proxy_candidate")
    if float(row.get("automatic_artifact_risk_score", 0) or 0) >= 0.5:
        reasons.append("artifact_guardrail_needed")
    if float(row.get("auto_diffusion_guardrail", 0) or 0) >= 1:
        reasons.append("diffusion_claim_guardrail")
    if float(row.get("future_any_drop_within_8cycles", 0) or 0) >= 1:
        reasons.append("future8_positive_context")
    if any_path(row, ["primary_qc_png", "control_balanced_qc_png", "roi_preview_path", "rollout_preview_path"]):
        reasons.append("visual_assets_available")
    seen = set()
    out = []
    for reason in reasons:
        if reason and reason not in seen:
            out.append(reason)
            seen.add(reason)
    return out


def action_tier(row: pd.Series) -> str:
    risk = float(row.get("automatic_artifact_risk_score", 0) or 0)
    decision_score = float(row.get("qc_decision_priority_score", 0) or 0)
    tier = str(row.get("auto_qc_tier", ""))
    physics = str(row.get("physics_consistency_tier", ""))
    if risk >= 0.55:
        return "review_artifact_or_reject_first"
    if tier == "auto_surrogate_likely_interpretable" and physics.startswith("cross_modal"):
        return "review_for_possible_accept_first"
    if decision_score >= 0.70:
        return "high_priority_review"
    if float(row.get("auto_diffusion_guardrail", 0) or 0) >= 1:
        return "review_but_diffusion_guarded"
    return "routine_pending_review"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/qc_decision_evidence_ledger")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manual = read_csv(derived / "manual_qc_label_workbook" / "manual_qc_label_template.csv")
    auto = read_csv(derived / "automatic_qc_triage_surrogate" / "automatic_qc_triage_table.csv")
    physics = read_csv(derived / "physics_consistency_claim_matrix" / "physics_consistency_claim_matrix.csv")
    active = read_csv(derived / "active_learning_qc_prioritization" / "active_learning_qc_priority_table.csv")
    balanced = read_csv(derived / "balanced_future_roi_physics_audit" / "balanced_future_roi_physics_joined.csv")
    echem_roi = read_csv(derived / "echem_conditioned_roi_rollout_front_audit" / "echem_conditioned_roi_front_joined.csv")

    if manual.empty or "roi_id" not in manual.columns:
        raise FileNotFoundError("Need manual_qc_label_workbook/manual_qc_label_template.csv")

    table = manual.copy()
    table = merge_if(
        table,
        auto,
        [
            "auto_qc_tier",
            "automatic_qc_surrogate_score",
            "automatic_artifact_risk_score",
            "surrogate_particle_identity_score",
            "surrogate_front_mask_score",
            "surrogate_diffusion_interpretability_score",
            "auto_diffusion_guardrail",
            "has_visual_asset",
            "front_radius2_slope_r2_track",
            "front_radius_monotonic_fraction_track",
        ],
    )
    table = merge_if(
        table,
        physics,
        [
            "physics_consistency_rank",
            "physics_consistency_tier",
            "claim_readiness",
            "physics_consistency_score",
            "physics_pillar_support_count",
            "physics_pillar_strong_support_count",
            "physics_pillar_contradiction_count",
            "physics_consistency_reason",
            "recommended_claim_language",
        ],
    )
    table = merge_if(
        table,
        active,
        [
            "active_learning_rank",
            "active_learning_qc_score",
            "recommended_qc_tier",
            "review_reason_tags",
            "model_future_drop_probability",
            "uncertainty_score",
        ],
    )
    table = merge_if(
        table,
        balanced,
        [
            "future_any_drop_within_8cycles",
            "future_any_drop_within_16cycles",
            "any_abrupt_drop",
            "transferred_masked_residual_signature",
            "phase_slope_positive_fraction",
            "radius2_slope_median_px2_per_s",
            "persistence_particle_mse_fraction_of_full_mean",
            "low_rank_dmd_particle_mse_fraction_of_full_mean",
        ],
        suffix="_balanced_future",
    )
    table = merge_if(
        table,
        echem_roi,
        [
            "transferred_masked_residual_signature",
            "phase_slope_positive_fraction_protocol_residual",
            "radius2_slope_median_px2_per_s_protocol_residual",
            "echem_regime_transfer_score",
        ],
        suffix="_echem_roi",
    )

    table["manual_qc_status"] = text(table, "manual_qc_status").replace("", "pending")
    table["visual_asset_count"] = table.apply(
        lambda r: int(sum(1 for c in ["primary_qc_png", "control_balanced_qc_png", "roi_preview_path", "front_crop_preview_path", "front_tracking_plot_path", "rollout_preview_path"] if isinstance(r.get(c, ""), str) and r.get(c, "").strip())),
        axis=1,
    )

    interpretable = rank01(num(table, "automatic_qc_surrogate_score"), high=True)
    artifact_safe = rank01(num(table, "automatic_artifact_risk_score"), high=False)
    physics_rank = rank01(num(table, "physics_consistency_score"), high=True)
    active_rank = rank01(num(table, "active_learning_qc_score"), high=True)
    visual_rank = rank01(num(table, "visual_asset_count"), high=True)
    front_rank = rank01(num(table, "surrogate_front_mask_score"), high=True)
    diffusion_rank = rank01(num(table, "surrogate_diffusion_interpretability_score"), high=True)
    table["qc_decision_priority_score"] = (
        0.24 * interpretable
        + 0.20 * artifact_safe
        + 0.20 * physics_rank
        + 0.14 * active_rank
        + 0.10 * visual_rank
        + 0.07 * front_rank
        + 0.05 * diffusion_rank
    )
    table["artifact_review_priority_score"] = (
        0.45 * rank01(num(table, "automatic_artifact_risk_score"), high=True)
        + 0.20 * rank01(num(table, "physics_consistency_score"), high=True)
        + 0.15 * visual_rank
        + 0.10 * rank01(num(table, "auto_diffusion_guardrail"), high=True)
        + 0.10 * active_rank
    )
    table["decision_action_tier"] = table.apply(action_tier, axis=1)
    table["decision_reason_tags"] = table.apply(lambda r: ";".join(top_reason_tokens(r)), axis=1)
    table["manual_label_allowed_values"] = "accept_particle_front;accept_particle_only;reject_artifact;reject_not_particle;uncertain"
    table["physics_claim_guardrail"] = np.where(
        num(table, "auto_diffusion_guardrail", 0).fillna(0) > 0,
        "no_diffusion_claim_until_manual_front_qc_and_calibration",
        "review_priority_only_no_validated_mode_claim",
    )

    sort_cols = ["qc_decision_priority_score", "physics_consistency_score", "automatic_qc_surrogate_score"]
    table = table.sort_values(sort_cols, ascending=False, na_position="last").reset_index(drop=True)
    table.insert(0, "qc_decision_rank", np.arange(1, len(table) + 1))

    artifact_queue = table.sort_values("artifact_review_priority_score", ascending=False, na_position="last").head(12).copy()
    top_queue = table.head(20).copy()
    cycle_summary = (
        table.groupby("cycleNo", dropna=False)
        .agg(
            n_candidates=("roi_id", "count"),
            n_review_for_accept=("decision_action_tier", lambda s: int((s == "review_for_possible_accept_first").sum())),
            n_artifact_first=("decision_action_tier", lambda s: int((s == "review_artifact_or_reject_first").sum())),
            max_decision_score=("qc_decision_priority_score", "max"),
            median_decision_score=("qc_decision_priority_score", "median"),
            max_artifact_score=("artifact_review_priority_score", "max"),
        )
        .reset_index()
        .sort_values(["max_decision_score", "n_candidates"], ascending=[False, False])
    )

    paths = {
        "ledger": out / "qc_decision_evidence_ledger.csv",
        "top_queue": out / "qc_decision_top_review_queue.csv",
        "artifact_queue": out / "qc_decision_artifact_review_queue.csv",
        "cycle_summary": out / "qc_decision_cycle_summary.csv",
        "summary": out / "qc_decision_evidence_ledger_summary.json",
        "readme": out / "README.md",
    }
    table.to_csv(paths["ledger"], index=False)
    top_queue.to_csv(paths["top_queue"], index=False)
    artifact_queue.to_csv(paths["artifact_queue"], index=False)
    cycle_summary.to_csv(paths["cycle_summary"], index=False)

    action_counts = table["decision_action_tier"].value_counts(dropna=False).to_dict()
    status_counts = table["manual_qc_status"].value_counts(dropna=False).to_dict()
    top_accept = table[table["decision_action_tier"] == "review_for_possible_accept_first"].head(10)
    summary = {
        "n_candidates": int(len(table)),
        "manual_status_counts": clean_json(status_counts),
        "decision_action_counts": clean_json(action_counts),
        "n_visual_asset_candidates": int((table["visual_asset_count"] > 0).sum()),
        "top_review_queue": clean_json(top_queue.head(12).to_dict("records")),
        "top_possible_accept_queue": clean_json(top_accept.to_dict("records")),
        "top_artifact_queue": clean_json(artifact_queue.head(12).to_dict("records")),
        "cycle_summary": clean_json(cycle_summary.head(12).to_dict("records")),
        "guardrail": "This ledger does not assign manual QC labels. It prioritizes pending particle/front candidates for human review using existing automatic evidence and keeps all physics claims guarded until particle identity, front mask, and calibration checks are manually accepted.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))

    lines = [
        "# QC Decision Evidence Ledger",
        "",
        "Reviewer-facing consolidation of pending ROI/front candidates. No manual labels are assigned.",
        "",
        "## Summary",
        "",
        f"- Candidates: {len(table)}",
        f"- Manual status counts: {status_counts}",
        f"- Decision action counts: {action_counts}",
        f"- Visual-asset candidates: {int((table['visual_asset_count'] > 0).sum())}",
        "",
        "## Top Review Queue",
        "",
    ]
    for _, row in top_queue.head(12).iterrows():
        lines.append(
            f"- rank {int(row['qc_decision_rank'])}: {row['roi_id']} cycle {row.get('cycleNo')} action={row['decision_action_tier']} score={row['qc_decision_priority_score']:.3f} risk={float(row.get('automatic_artifact_risk_score') or 0):.3f} reasons={row['decision_reason_tags']}"
        )
    lines += ["", "## Artifact/Reject-First Queue", ""]
    for _, row in artifact_queue.head(8).iterrows():
        lines.append(
            f"- {row['roi_id']} cycle {row.get('cycleNo')} artifact_score={row['artifact_review_priority_score']:.3f} risk={float(row.get('automatic_artifact_risk_score') or 0):.3f} reasons={row['decision_reason_tags']}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    paths["readme"].write_text("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
