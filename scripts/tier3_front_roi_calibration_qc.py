#!/usr/bin/env python3
"""Build a QC/calibration table for selected NMC front ROIs.

The upstream front-tracking scripts report apparent slopes in pixel units.
This step records the calibration provenance, converts slopes to micrometre
units when a pixel size is provided, and writes a manual-review manifest for
the highest-priority event ROIs.
"""

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd


DEFAULT_PIXEL_SIZE_UM = 0.096
DEFAULT_CALIBRATION_SOURCE = (
    "Battery_Degradation_Project/Degradation Paper Outline.pptx slide 3: "
    "improved microscope design offers 96 nm pixel size and 180x120 um FoV"
)


def read_csv_optional(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def finite_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def add_calibrated_columns(df: pd.DataFrame, pixel_size_um: float) -> pd.DataFrame:
    out = df.copy()
    px = finite_float(pixel_size_um)
    px2 = px * px if np.isfinite(px) and px > 0 else np.nan
    out["pixel_size_um"] = px
    out["pixel_area_um2"] = px2

    for col in [
        "radius2_slope_full_px2_per_s",
        "apparent_diffusion_proxy_full_px2_per_s",
        "area_equiv_radius2_slope_full_px2_per_s",
        "p90_radius2_slope_full_px2_per_s",
    ]:
        if col in out.columns:
            out[col.replace("px2", "um2")] = pd.to_numeric(out[col], errors="coerce") * px2

    for col in ["stage_drift_xy", "center_x_full_px", "center_y_full_px"]:
        if col in out.columns:
            out[col.replace("_px", "_um").replace("stage_drift_xy", "stage_drift_xy_um")] = (
                pd.to_numeric(out[col], errors="coerce") * px
            )
    return out


def build_qc_flags(row: pd.Series) -> Dict[str, object]:
    r2 = finite_float(row.get("radius2_slope_r2"))
    drift = finite_float(row.get("stage_drift_xy"))
    elapsed = finite_float(row.get("elapsed_s"))
    active0 = finite_float(row.get("active_fraction_first"))
    active1 = finite_float(row.get("active_fraction_last"))
    notes = []
    if np.isfinite(r2) and r2 < 0.15:
        notes.append("low radius2 fit R2")
    if np.isfinite(drift) and drift > 1.0:
        notes.append("stage drift exceeds 1 px")
    if np.isfinite(elapsed) and elapsed <= 0:
        notes.append("non-positive elapsed time")
    if np.isfinite(active0) and np.isfinite(active1) and max(active0, active1) < 0.03:
        notes.append("small active-front area")
    return {
        "manual_qc_status": "pending",
        "manual_qc_required": True,
        "qc_flag_count": len(notes),
        "auto_qc_notes": "; ".join(notes) if notes else "no automatic front-tracking warning",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-tracking-csv", default="derived_local/selected_front_roi_tracking/selected_front_roi_tracking_summary.csv")
    parser.add_argument("--joint-modes-csv", default="derived_local/roi_joint_physics_degradation_modes/roi_joint_physics_degradation_modes.csv")
    parser.add_argument("--out-dir", default="derived_local/front_roi_calibration_qc")
    parser.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM)
    parser.add_argument("--calibration-source", default=DEFAULT_CALIBRATION_SOURCE)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    front = read_csv_optional(args.front_tracking_csv)
    if front.empty:
        raise FileNotFoundError(args.front_tracking_csv)
    joint = read_csv_optional(args.joint_modes_csv)

    calibrated = add_calibrated_columns(front, args.pixel_size_um)
    if not joint.empty and "roi_id_front" in joint.columns:
        keep = [
            c for c in [
                "roi_id_front",
                "joint_degradation_score",
                "joint_mode_name",
                "roi_id",
                "rollout_residual_energy_mean",
                "mean_drop_frac",
                "evidence_score",
                "degradation_mode_hypothesis",
            ]
            if c in joint.columns
        ]
        calibrated = calibrated.merge(
            joint[keep].drop_duplicates("roi_id_front"),
            how="left",
            left_on="roi_id",
            right_on="roi_id_front",
        )

    score_col = "joint_degradation_score" if "joint_degradation_score" in calibrated.columns else "validation_score"
    calibrated = calibrated.sort_values(score_col, ascending=False, na_position="last")
    qc = calibrated.head(args.top_n).copy()
    flag_rows = [build_qc_flags(row) for _, row in qc.iterrows()]
    qc = pd.concat([qc.reset_index(drop=True), pd.DataFrame(flag_rows)], axis=1)
    qc["calibration_status"] = "provisional_from_project_slides"
    qc["calibration_source"] = args.calibration_source
    qc["front_tracking_interpretation"] = (
        "calibrated apparent front/radius slopes; not a physical diffusion coefficient "
        "until particle identity, front mask, and pixel calibration are manually confirmed"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    calibrated_path = os.path.join(args.out_dir, "calibrated_front_tracking.csv")
    qc_path = os.path.join(args.out_dir, "front_roi_qc_calibration.csv")
    summary_path = os.path.join(args.out_dir, "front_roi_calibration_qc_summary.json")
    readme_path = os.path.join(args.out_dir, "README.md")
    calibrated.to_csv(calibrated_path, index=False)
    qc.to_csv(qc_path, index=False)

    summary = {
        "n_front_rois": int(len(calibrated)),
        "n_qc_rois": int(len(qc)),
        "pixel_size_um": float(args.pixel_size_um),
        "pixel_area_um2": float(args.pixel_size_um * args.pixel_size_um),
        "calibration_source": args.calibration_source,
        "calibration_status": "provisional_from_project_slides",
        "top_qc_rois": qc[[
            c for c in [
                "roi_id",
                "cycleNo",
                "joint_degradation_score",
                "joint_mode_name",
                "radius2_slope_full_um2_per_s",
                "apparent_diffusion_proxy_full_um2_per_s",
                "radius2_slope_r2",
                "manual_qc_status",
                "auto_qc_notes",
            ]
            if c in qc.columns
        ]].to_dict(orient="records"),
        "guardrail": (
            "The 96 nm/px calibration comes from project-slide text and should be "
            "confirmed against raw microscope metadata before publication-scale claims."
        ),
        "outputs": {
            "calibrated_front_tracking": calibrated_path,
            "front_roi_qc_calibration": qc_path,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(readme_path, "w") as f:
        f.write("# Front ROI Calibration QC\n\n")
        f.write("Converts selected front-tracking slopes from pixel units to provisional micrometre units.\n")
        f.write("Calibration source: " + args.calibration_source + "\n\n")
        f.write("Manual QC remains required before interpreting these as mechanistic diffusion or mobility constants.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
