#!/usr/bin/env python3
"""Assemble a go/no-go ledger for calibrated diffusion claims.

This audit does not estimate new diffusion values. It joins the calibration,
apparent-diffusion, physics-consistency, control-balanced sanity, and manual-QC
outputs into one readiness table so the diffusion requirement is auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def gate(summary: dict[str, Any], criterion: str) -> dict[str, Any]:
    for row in summary.get("gate_counts", []) or []:
        if row.get("criterion") == criterion:
            return row
    return {}


def criterion(
    gate_name: str,
    status: str,
    evidence: str,
    blocker: str,
    next_action: str,
    severity: str = "hard_blocker",
) -> dict[str, Any]:
    return {
        "gate": gate_name,
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "blocking_detail": blocker,
        "next_action": next_action,
    }


def pass_count(row: dict[str, Any]) -> tuple[int, int]:
    return int(row.get("n_pass") or 0), int(row.get("n_total") or 0)


def bool_text(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    return "pass" if bool(value) else "fail"


def blockers_from_row(row: pd.Series, checks: list[tuple[str, str]]) -> str:
    blockers: list[str] = []
    for col, label in checks:
        if col not in row:
            continue
        value = row.get(col)
        if pd.isna(value) or not bool(value):
            blockers.append(label)
    return "; ".join(blockers) if blockers else "none"


def build_candidate_ledger(derived: Path, out_dir: Path) -> pd.DataFrame:
    physics = read_csv(derived / "diffusion_physics_consistency_audit" / "diffusion_physics_consistency_roi_scores.csv")
    sanity = read_csv(derived / "control_balanced_diffusion_proxy_sanity_audit" / "diffusion_proxy_sanity_joined.csv")
    rows: list[dict[str, Any]] = []

    if not physics.empty:
        ranked = physics.copy()
        ranked["review_sort"] = (
            pd.to_numeric(ranked.get("publication_ready_diffusion_candidate"), errors="coerce").fillna(0) * 100
            + pd.to_numeric(ranked.get("automatic_diffusion_physics_consistent"), errors="coerce").fillna(0) * 50
            + pd.to_numeric(ranked.get("diffusion_physics_gate_count"), errors="coerce").fillna(0)
            + pd.to_numeric(ranked.get("physics_consistency_score"), errors="coerce").fillna(0) / 10.0
        )
        ranked = ranked.sort_values("review_sort", ascending=False).head(30)
        checks = [
            ("gate_positive_expansion", "positive expansion"),
            ("gate_fit_quality", "radius2 fit quality"),
            ("gate_threshold_stability", "threshold stability"),
            ("gate_h5_timing_stable", "HDF5 timing stability"),
            ("gate_low_drift", "low drift"),
            ("gate_q70_positive_ci", "q70 positive CI"),
            ("publication_ready_diffusion_candidate", "publication-ready gate"),
        ]
        for _, row in ranked.iterrows():
            rows.append(
                {
                    "candidate_source": "diffusion_physics_consistency",
                    "roi_id": row.get("roi_id"),
                    "selected_roi_id": row.get("roi_id"),
                    "cycleNo": row.get("cycleNo"),
                    "source_stem": row.get("source_stem"),
                    "cohort_role": row.get("cohort_role"),
                    "is_event_roi": row.get("is_event_roi"),
                    "future_any_drop_within_8cycles": row.get("future_any_drop_within_8cycles"),
                    "future_any_drop_within_16cycles": row.get("future_any_drop_within_16cycles"),
                    "automatic_physics_consistent": row.get("automatic_diffusion_physics_consistent"),
                    "publication_ready": row.get("publication_ready_diffusion_candidate"),
                    "gate_count": row.get("diffusion_physics_gate_count"),
                    "median_apparent_D_um2_per_s": row.get("median_apparent_D_um2_per_s"),
                    "selected_diffusion_um2_per_s": row.get("selected_diffusion_um2_per_s"),
                    "median_radius2_fit_r2": row.get("median_radius2_fit_r2"),
                    "selected_r2": row.get("selected_r2"),
                    "h5_dt_max_to_median_ratio": row.get("h5_dt_max_to_median_ratio"),
                    "manual_qc_status": row.get("manual_qc_status", "missing"),
                    "blockers": blockers_from_row(row, checks),
                    "review_priority": row.get("review_sort"),
                }
            )

    if not sanity.empty:
        ranked = sanity.copy()
        ranked["review_sort"] = (
            pd.to_numeric(ranked.get("publication_diffusion_candidate"), errors="coerce").fillna(0) * 100
            + pd.to_numeric(ranked.get("automatic_positive_diffusion_proxy_candidate"), errors="coerce").fillna(0) * 50
            + pd.to_numeric(ranked.get("selected_nonnegative"), errors="coerce").fillna(0) * 5
            + pd.to_numeric(ranked.get("threshold_nonnegative"), errors="coerce").fillna(0) * 5
            + pd.to_numeric(ranked.get("q70_bootstrap_positive"), errors="coerce").fillna(0) * 5
            + pd.to_numeric(ranked.get("selected_r2"), errors="coerce").fillna(0)
        )
        ranked = ranked.sort_values("review_sort", ascending=False).head(30)
        checks = [
            ("selected_nonnegative", "selected nonnegative"),
            ("threshold_nonnegative", "threshold nonnegative"),
            ("selected_fit_good", "selected fit quality"),
            ("low_drift_relative_to_motion", "low drift"),
            ("q70_bootstrap_positive", "q70 bootstrap positive"),
            ("manual_qc_accepted", "manual QC accepted"),
            ("publication_diffusion_candidate", "publication-ready gate"),
        ]
        for _, row in ranked.iterrows():
            rows.append(
                {
                    "candidate_source": "control_balanced_selected_front",
                    "roi_id": row.get("roi_id"),
                    "selected_roi_id": row.get("selected_roi_id"),
                    "cycleNo": row.get("cycleNo"),
                    "source_stem": row.get("source_stem"),
                    "cohort_role": row.get("cohort_role"),
                    "is_event_roi": row.get("is_event_roi"),
                    "future_any_drop_within_8cycles": np.nan,
                    "future_any_drop_within_16cycles": np.nan,
                    "automatic_physics_consistent": row.get("automatic_positive_diffusion_proxy_candidate"),
                    "publication_ready": row.get("publication_diffusion_candidate"),
                    "gate_count": np.nan,
                    "median_apparent_D_um2_per_s": row.get("diffusion_proxy_median_um2_per_s"),
                    "selected_diffusion_um2_per_s": row.get("selected_diffusion_um2_per_s"),
                    "median_radius2_fit_r2": np.nan,
                    "selected_r2": row.get("selected_r2"),
                    "h5_dt_max_to_median_ratio": np.nan,
                    "manual_qc_status": row.get("manual_qc_status", "missing"),
                    "blockers": blockers_from_row(row, checks),
                    "review_priority": row.get("review_sort"),
                }
            )

    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger = ledger.sort_values(["publication_ready", "automatic_physics_consistent", "review_priority"], ascending=False)
    ledger.to_csv(out_dir / "diffusion_claim_readiness_candidates.csv", index=False)
    return ledger


def build_criteria(derived: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    calibration = read_json(derived / "calibration_metadata_audit" / "calibration_metadata_audit_summary.json")
    apparent = read_json(derived / "apparent_diffusion_calibration_bounds" / "apparent_diffusion_calibration_bounds_summary.json")
    physics = read_json(derived / "diffusion_physics_consistency_audit" / "diffusion_physics_consistency_summary.json")
    sanity = read_json(derived / "control_balanced_diffusion_proxy_sanity_audit" / "diffusion_proxy_sanity_audit_summary.json")
    manual = read_json(derived / "manual_qc_gated_front_effects" / "manual_qc_gated_front_effects_summary.json")
    visual = read_json(derived / "source_balanced_pre_event_manual_qc_visual_packet" / "source_balanced_pre_event_manual_qc_visual_summary.json")

    criteria: list[dict[str, Any]] = []
    n_h5 = int(calibration.get("n_h5_files") or 0)
    n_h5_timing = int(calibration.get("n_h5_with_camera_timing") or 0)
    n_h5_cal = int(calibration.get("n_h5_with_calibration_attr_hits") or 0)
    criteria.append(
        criterion(
            "HDF5 camera timing present",
            "partial" if n_h5_timing else "fail",
            f"{n_h5_timing}/{n_h5} scanned HDF5 files have camera timing datasets.",
            "Timing rows exist, but sampled rows may represent sparse segment/cycle timing rather than raw frame cadence.",
            "Confirm raw acquisition frame cadence semantics from microscope/HDF5 metadata.",
            "guardrail",
        )
    )
    criteria.append(
        criterion(
            "HDF5 spatial calibration metadata present",
            "fail" if n_h5_cal == 0 else "pass",
            f"{n_h5_cal}/{n_h5} scanned HDF5 files have spatial-calibration-like attribute hits; slide evidence gives 96 nm/px.",
            "Pixel size is slide-derived, not confirmed in raw HDF5/microscope metadata.",
            "Locate raw microscope calibration metadata or independently verify the 96 nm/px scale.",
        )
    )
    ratio = finite(apparent.get("median_roi_elapsed_to_h5_median_ratio"))
    criteria.append(
        criterion(
            "ROI/HDF5 elapsed-time consistency",
            "partial" if np.isfinite(ratio) else "fail",
            f"Median ROI elapsed / HDF5 elapsed ratio is {ratio:.4g} across {apparent.get('n_roi_with_h5_timing', 0)} ROI with HDF5 timing.",
            "Global elapsed-time ratio is good, but source-level timing instability remains.",
            "Audit frame-by-frame timestamps for the high max/median-ratio sources.",
            "guardrail",
        )
    )

    timing_pass, timing_total = pass_count(gate(physics, "gate_h5_timing_stable"))
    criteria.append(
        criterion(
            "Per-ROI HDF5 timing stability",
            "fail" if (not timing_total or timing_pass < timing_total) else "pass",
            f"{timing_pass}/{timing_total} ROI pass the HDF5 timing-stability gate; max source dt max/median ratio is {finite(apparent.get('max_source_h5_dt_max_to_median_ratio')):.4g}.",
            "Only a minority of ROI pass the stricter timing-stability gate.",
            "Prioritize source-level timestamp validation before physical units are used in claims.",
        )
    )

    for name, label in [
        ("gate_positive_expansion", "Positive radius2/front expansion"),
        ("gate_fit_quality", "Radius2 linear-fit quality"),
        ("gate_threshold_stability", "Threshold stability"),
        ("gate_low_drift", "Low drift"),
        ("gate_q70_positive_ci", "q70 positive confidence interval"),
    ]:
        n_pass, n_total = pass_count(gate(physics, name))
        status = "pass" if n_total and n_pass == n_total else ("partial" if n_pass else "fail")
        if name in {"gate_fit_quality", "gate_q70_positive_ci"} and n_pass < n_total:
            status = "fail"
        criteria.append(
            criterion(
                label,
                status,
                f"{n_pass}/{n_total} ROI pass.",
                "Internal optical-front evidence is not strong enough for a calibrated diffusion claim." if status == "fail" else ("This gate currently passes all audited ROI." if status == "pass" else "Some ROI remain excluded by this gate."),
                "Use this gate to restrict manual review and avoid reporting failed candidates as diffusion coefficients.",
                "hard_blocker" if status == "fail" else "guardrail",
            )
        )

    n_auto = int(physics.get("n_automatic_diffusion_physics_consistent") or 0)
    n_pub = int(physics.get("n_publication_ready_diffusion_candidates") or 0)
    criteria.append(
        criterion(
            "Automatic/publication diffusion candidates",
            "fail" if n_pub == 0 else "pass",
            f"{n_auto} automatic physics-consistent ROI and {n_pub} publication-ready diffusion candidates.",
            "No ROI passes the publication-ready diffusion gate.",
            "Do not state calibrated material diffusion coefficients until manual labels and gate pass criteria are satisfied.",
        )
    )

    n_manual_front = int(manual.get("n_manual_front_effect_accepted") or 0)
    n_manual_diff = int(manual.get("n_manual_diffusion_accepted") or 0)
    criteria.append(
        criterion(
            "Manual QC accepted labels",
            "blocked_pending_manual_qc" if n_manual_front == 0 or n_manual_diff == 0 else "pass",
            f"{n_manual_front} manual front-effect labels and {n_manual_diff} manual diffusion-interpretable labels accepted; {manual.get('n_pending_or_not_accepted', 0)} pending/not accepted.",
            "Manual labels have not yet validated particle identity, front masks, or diffusion interpretability.",
            "Review and label the manual-QC packet before treating front motion as physical diffusion.",
        )
    )

    sanity_pub = int(sanity.get("n_publication_diffusion_candidates") or 0)
    sanity_auto = int(sanity.get("n_automatic_positive_diffusion_proxy_candidates") or 0)
    criteria.append(
        criterion(
            "Control-balanced diffusion sanity candidates",
            "fail" if sanity_pub == 0 else "pass",
            f"{sanity_auto} automatic positive proxy candidates and {sanity_pub} publication diffusion candidates in the control-balanced sanity audit.",
            "Control-balanced selected-front checks produce no publication-ready diffusion candidates.",
            "Keep diffusion wording as optical-front proxy unless this independent sanity audit produces accepted candidates.",
        )
    )

    tests = sanity.get("top_event_control_tests", []) or sanity.get("event_control_tests", []) or []
    best_p = None
    if tests:
        best_p = min((finite(row.get("mannwhitney_p")) for row in tests), default=np.nan)
    criteria.append(
        criterion(
            "Event/control diffusion separability",
            "fail" if not np.isfinite(best_p) or best_p >= 0.05 else "partial",
            f"Best control-balanced diffusion-proxy event/control p-value is {best_p:.4g}." if np.isfinite(best_p) else "No valid event/control diffusion test found.",
            "Diffusion proxies do not show robust event/control separation under this sanity design.",
            "Use source/event-balanced tests for prioritization only, not physical diffusion inference.",
        )
    )

    rendered = int(visual.get("n_rendered") or 0)
    strict = int((visual.get("action_tier_counts_rendered") or {}).get("review_strict_front_gate", 0) or 0)
    criteria.append(
        criterion(
            "Manual visual review packet available",
            "partial" if rendered else "fail",
            f"{rendered} manual-review crops rendered across {visual.get('n_sources_rendered', 0)} sources; {strict} strict-front-gate candidate(s).",
            "Visual packet supports review but does not provide accepted labels.",
            "Use the packet to create manual labels, then rerun manual-QC-gated front and diffusion audits.",
            "guardrail",
        )
    )

    criteria_df = pd.DataFrame(criteria)
    status_counts = criteria_df["status"].value_counts().to_dict()
    hard_blockers = criteria_df[criteria_df["severity"].eq("hard_blocker") & ~criteria_df["status"].eq("pass")]["gate"].tolist()
    summary = {
        "overall_status": "not_ready_for_calibrated_diffusion_claim" if hard_blockers else "ready_for_limited_calibrated_diffusion_review",
        "n_criteria": int(len(criteria_df)),
        "status_counts": status_counts,
        "hard_blockers": hard_blockers,
        "n_hard_blockers": int(len(hard_blockers)),
        "guardrail": "Calibrated diffusion remains blocked unless raw spatial calibration, timestamp semantics, internal front-motion gates, control-balanced sanity checks, and manual QC labels all pass. Current outputs support optical-front proxy review only.",
    }
    return criteria_df, summary


def write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    blockers = summary.get("hard_blockers", [])
    lines = [
        "# Diffusion Claim Readiness Audit",
        "",
        "This folder assembles a go/no-go ledger for calibrated diffusion claims from existing calibration, apparent-diffusion, physics-consistency, control-balanced sanity, and manual-QC outputs.",
        "",
        f"- Overall status: `{summary.get('overall_status')}`",
        f"- Hard blockers: {summary.get('n_hard_blockers', 0)}",
        f"- Status counts: `{summary.get('status_counts', {})}`",
        "",
        "Hard blockers:",
    ]
    lines += [f"- {item}" for item in blockers] or ["- none"]
    lines += [
        "",
        "Outputs:",
        "- `diffusion_claim_readiness_criteria.csv`: gate-level pass/partial/fail ledger.",
        "- `diffusion_claim_readiness_candidates.csv`: ranked ROI/candidate ledger with blockers.",
        "- `diffusion_claim_readiness_summary.json`: compact summary for the synthesis report.",
        "",
        f"Guardrail: {summary.get('guardrail')}",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    criteria, summary = build_criteria(args.derived_dir)
    candidates = build_candidate_ledger(args.derived_dir, args.out_dir)
    criteria.to_csv(args.out_dir / "diffusion_claim_readiness_criteria.csv", index=False)

    top_candidates = []
    if not candidates.empty:
        for _, row in candidates.head(12).iterrows():
            top_candidates.append(clean_json(row.to_dict()))
    summary.update(
        {
            "n_candidate_rows": int(len(candidates)),
            "n_publication_ready_candidates": int(pd.to_numeric(candidates.get("publication_ready"), errors="coerce").fillna(0).sum()) if not candidates.empty else 0,
            "n_automatic_consistent_candidates": int(pd.to_numeric(candidates.get("automatic_physics_consistent"), errors="coerce").fillna(0).sum()) if not candidates.empty else 0,
            "top_candidates": top_candidates,
            "outputs": {
                "criteria": str(args.out_dir / "diffusion_claim_readiness_criteria.csv"),
                "candidates": str(args.out_dir / "diffusion_claim_readiness_candidates.csv"),
                "summary": str(args.out_dir / "diffusion_claim_readiness_summary.json"),
                "readme": str(args.out_dir / "README.md"),
            },
        }
    )
    (args.out_dir / "diffusion_claim_readiness_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    write_readme(args.out_dir, summary)


if __name__ == "__main__":
    main()
