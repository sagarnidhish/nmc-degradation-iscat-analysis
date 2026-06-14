#!/usr/bin/env python3
"""Build a calibration-risk register for NMC front/kinetic/diffusion claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def file_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"source_exists": False, "n_rows": 0, "n_columns": 0}
    try:
        df = pd.read_csv(path)
        return {"source_exists": True, "n_rows": int(len(df)), "n_columns": int(len(df.columns))}
    except Exception as exc:
        return {"source_exists": True, "n_rows": 0, "n_columns": 0, "read_error": str(exc)}


def add(rows: List[Dict[str, Any]], **kwargs: Any) -> None:
    rows.append(kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/calibration_claim_risk_register")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    calibration = read_json(derived / "calibration_metadata_audit" / "calibration_metadata_audit_summary.json")
    h5_spatial_attrs = int(calibration.get("n_h5_with_calibration_attr_hits") or 0)
    h5_timing = int(calibration.get("n_h5_with_camera_timing") or 0)
    h5_scanned = int(calibration.get("n_h5_files") or 0)
    pptx_hits = int(calibration.get("n_pptx_calibration_hits") or 0)
    timing_guardrail = calibration.get("guardrail", "Calibration metadata audit unavailable.")

    spatial_status = "raw_h5_confirmed" if h5_spatial_attrs else ("slide_derived_96nm_px" if pptx_hits else "unknown")
    time_status = "h5_camera_timing_present_but_sparse_proxy" if h5_timing else "unknown"

    rows: List[Dict[str, Any]] = []

    specs = [
        {
            "analysis": "event_candidate_fronts",
            "source": "event_candidate_fronts/candidate_front_metrics.csv",
            "claim_family": "early downsampled front candidates",
            "metric_examples": "apparent_diffusion_proxy_ds_px2_per_frame, front/radius descriptors",
            "spatial_evidence": "downsampled pixels only",
            "time_evidence": "frame index only",
            "claim_status": "proxy_only",
            "risk_level": "high",
            "recommended_wording": "Use as automatic front-candidate ranking only; do not interpret as calibrated diffusion.",
        },
        {
            "analysis": "roi_phase_boundary_mobility",
            "source": "roi_phase_boundary_mobility/roi_phase_boundary_mobility_descriptors.csv",
            "claim_family": "selected ROI phase-boundary mobility",
            "metric_examples": "phase fraction slopes, apparent diffusion proxy in px^2/frame",
            "spatial_evidence": "pixels; no raw HDF5 pixel-size attr",
            "time_evidence": "frame index/provisional timing",
            "claim_status": "optical_front_proxy",
            "risk_level": "high",
            "recommended_wording": "Describe as apparent optical boundary mobility in cropped particle ROIs.",
        },
        {
            "analysis": "multi_cycle_roi_mobility",
            "source": "multi_cycle_roi_mobility/multi_cycle_roi_mobility_descriptors.csv",
            "claim_family": "multi-cycle front/mobility descriptors",
            "metric_examples": "front displacement, phase fraction, apparent diffusion proxies",
            "spatial_evidence": "automatic ROI pixels; no raw HDF5 spatial calibration",
            "time_evidence": "cycle/ROI timing context, not direct true cadence proof",
            "claim_status": "optical_front_proxy",
            "risk_level": "high",
            "recommended_wording": "Use for event/control morphology comparisons, not transport constants.",
        },
        {
            "analysis": "multi_cycle_threshold_robust_fronts",
            "source": "multi_cycle_threshold_robust_fronts/threshold_sweep_per_roi.csv",
            "claim_family": "threshold-robust front and phase-slope readout",
            "metric_examples": "phase_slope_*_per_s, diffusion_proxy_*_um2_per_s",
            "spatial_evidence": spatial_status,
            "time_evidence": time_status,
            "claim_status": "phase_sign_usable_diffusion_proxy_only",
            "risk_level": "medium_high",
            "recommended_wording": "Phase-slope sign/fraction trends are usable optical proxies; diffusion numbers are apparent um^2/s proxies pending calibration provenance.",
        },
        {
            "analysis": "front_roi_calibration_qc",
            "source": "front_roi_calibration_qc/calibrated_front_tracking.csv",
            "claim_family": "calibrated-front QC table",
            "metric_examples": "diffusion_proxy_median_um2_per_s, radius2 slopes, CI flags",
            "spatial_evidence": spatial_status,
            "time_evidence": time_status,
            "claim_status": "apparent_um_scale_proxy",
            "risk_level": "medium_high",
            "recommended_wording": "Keep units as apparent um^2/s optical-front proxies and pair with QC flags/manual labels.",
        },
        {
            "analysis": "protocol_conditioned_front_effects",
            "source": "protocol_conditioned_front_effects/protocol_conditioned_front_residuals.csv",
            "claim_family": "protocol-conditioned front residuals",
            "metric_examples": "phase_slope_positive_fraction_protocol_residual, diffusion residuals",
            "spatial_evidence": spatial_status,
            "time_evidence": time_status,
            "claim_status": "conditioned_optical_proxy",
            "risk_level": "medium",
            "recommended_wording": "Emphasize robust phase-front direction residuals; do not upgrade diffusion residuals to physical transport.",
        },
        {
            "analysis": "front_qc_sensitivity",
            "source": "front_qc_sensitivity/front_qc_sensitivity_focus_tests.csv",
            "claim_family": "automatic QC sensitivity of front residuals",
            "metric_examples": "phase-sign residual strata, diffusion-proxy strata",
            "spatial_evidence": spatial_status,
            "time_evidence": time_status,
            "claim_status": "qc_guardrail",
            "risk_level": "medium",
            "recommended_wording": "Use as sensitivity support for phase-sign residuals and as negative evidence for diffusion robustness.",
        },
        {
            "analysis": "control_balanced_front_qc_sensitivity",
            "source": "control_balanced_front_qc_sensitivity/control_balanced_front_qc_sensitivity_focus_tests.csv",
            "claim_family": "balanced QC front sensitivity",
            "metric_examples": "balanced event/control phase-sign and diffusion residual tests",
            "spatial_evidence": spatial_status,
            "time_evidence": time_status,
            "claim_status": "qc_guardrail",
            "risk_level": "medium",
            "recommended_wording": "Use the balanced panel to support phase-sign robustness; keep diffusion as non-significant/proxy.",
        },
        {
            "analysis": "phase_kinetics_avrami",
            "source": "phase_kinetics_avrami/phase_kinetics_avrami_roi_table.csv",
            "claim_family": "optical phase kinetics / Avrami-style descriptors",
            "metric_examples": "logistic_k_per_s, transformed_fraction_delta, Avrami proxy terms",
            "spatial_evidence": "particle ROI intensity/phase fractions; spatial calibration mostly not required except front-linked covariates",
            "time_evidence": time_status,
            "claim_status": "descriptive_optical_kinetics_proxy",
            "risk_level": "medium_high",
            "recommended_wording": "Report as optical transition-sharpness/rate descriptors, not reaction constants.",
        },
        {
            "analysis": "manual_qc_gated_front_effects",
            "source": "manual_qc_gated_front_effects/manual_qc_gated_front_effect_tests.csv",
            "claim_family": "manual-QC-gated front/diffusion effects",
            "metric_examples": "only emitted after manual particle/front/diffusion labels accept rows",
            "spatial_evidence": spatial_status,
            "time_evidence": time_status,
            "claim_status": "publication_gate_pending",
            "risk_level": "low_when_populated_high_now",
            "recommended_wording": "No manual-QC accepted front/diffusion claim should be made until this table is populated.",
        },
        {
            "analysis": "roi_front_qc_package",
            "source": "roi_front_qc_package/roi_front_qc_manifest.csv",
            "claim_family": "manual visual review package",
            "metric_examples": "QC panels and pending manual labels",
            "spatial_evidence": "visual particle crop review, not measurement calibration",
            "time_evidence": "not a timebase measurement",
            "claim_status": "review_prioritization_only",
            "risk_level": "low_for_review_high_for_physics_claim",
            "recommended_wording": "Use to assign labels and inspect artifacts; it does not validate diffusion by itself.",
        },
    ]

    for spec in specs:
        stats = file_stats(derived / spec["source"])
        add(
            rows,
            **spec,
            **stats,
            h5_files_scanned=h5_scanned,
            h5_files_with_camera_timing=h5_timing,
            h5_files_with_spatial_calibration_attrs=h5_spatial_attrs,
            pptx_calibration_hits=pptx_hits,
            calibration_guardrail=timing_guardrail,
        )

    register = pd.DataFrame(rows)
    register_path = out / "calibration_claim_risk_register.csv"
    register.to_csv(register_path, index=False)

    by_status = register["claim_status"].value_counts().to_dict()
    by_risk = register["risk_level"].value_counts().to_dict()
    high_risk = register[register["risk_level"].astype(str).str.contains("high", case=False, na=False)]
    summary = {
        "n_claim_families": int(len(register)),
        "n_source_tables_present": int(register["source_exists"].sum()),
        "claim_status_counts": by_status,
        "risk_level_counts": by_risk,
        "high_risk_claim_families": high_risk[["analysis", "claim_family", "claim_status", "recommended_wording"]].to_dict("records"),
        "calibration_evidence": {
            "h5_files_scanned": h5_scanned,
            "h5_files_with_camera_timing": h5_timing,
            "h5_files_with_spatial_calibration_attrs": h5_spatial_attrs,
            "pptx_calibration_hits": pptx_hits,
            "spatial_status": spatial_status,
            "time_status": time_status,
        },
        "guardrail": "This register audits wording risk, not numerical correctness. Current front/kinetic outputs are strongest as optical particle-region proxies; diffusion-like values remain apparent proxies until spatial calibration, true frame cadence, masks, and manual QC are jointly validated.",
        "outputs": {
            "register": str(register_path),
            "summary": str(out / "calibration_claim_risk_summary.json"),
        },
    }
    with (out / "calibration_claim_risk_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    lines = [
        "# Calibration Claim Risk Register",
        "",
        "Claim-level calibration guardrail for front, kinetic, mobility, and diffusion-like outputs.",
        "",
        f"- Claim families audited: {summary['n_claim_families']}",
        f"- Source tables present: {summary['n_source_tables_present']}",
        f"- HDF5 timing evidence: {h5_timing}/{h5_scanned} scanned files have camera_timing",
        f"- HDF5 spatial calibration attrs: {h5_spatial_attrs}",
        f"- PPTX calibration hits: {pptx_hits}",
        "",
        "## Interpretation",
        "",
        summary["guardrail"],
        "",
        "## Highest-Risk Claim Families",
        "",
    ]
    for row in summary["high_risk_claim_families"]:
        lines.append(f"- {row['analysis']}: {row['recommended_wording']}")
    (out / "README.md").write_text("\n".join(lines).rstrip() + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
