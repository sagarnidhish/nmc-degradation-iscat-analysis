#!/usr/bin/env python3
"""Build a control-balanced front-QC augmentation package for NMC ROI review.

The first ROI/front QC package intentionally prioritizes high-signal candidates,
but the downstream sensitivity analysis showed that strict no-fragment/no-flag
review strata contain event ROIs but no controls. This package keeps the visual
panel format and selects a balanced event/control set so manual front-mask review
can validate controls under comparable automatic criteria.
"""

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tier4_roi_front_qc_package import finite_float, render_roi_qc


FOCUS_FEATURES = [
    "phase_slope_positive_fraction_protocol_residual",
    "phase_slope_median_per_s",
    "threshold_robust_phase_score",
    "diffusion_proxy_abs_median_um2_per_s",
    "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    "rollout_mobility_difficulty_score",
]


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
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


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return (x - x.mean(skipna=True)) / sd


def prepare_table(derived: Path) -> pd.DataFrame:
    echem = pd.read_csv(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_joined.csv")
    fronts = pd.read_csv(derived / "multi_cycle_threshold_robust_fronts" / "threshold_robust_front_summary.csv")
    residuals = pd.read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")
    mobility = pd.read_csv(derived / "multi_cycle_rollout_mobility_coupling" / "multi_cycle_rollout_mobility_ranked.csv")
    modes_path = derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv"
    modes = pd.read_csv(modes_path) if modes_path.exists() else pd.DataFrame({"roi_id": []})
    original_manifest_path = derived / "roi_front_qc_package" / "roi_front_qc_manifest.csv"
    original_manifest = pd.read_csv(original_manifest_path) if original_manifest_path.exists() else pd.DataFrame({"roi_id": []})

    base_cols = [
        "roi_id",
        "cycleNo",
        "cohort_role",
        "event_reference_cycle",
        "npz_path",
        "preview_png",
        "object_x_full_approx",
        "object_y_full_approx",
        "n_frames_percentile",
        "V_mean",
        "I_mean_mA",
        "degradation_mode_hypothesis",
    ]
    df = echem[[c for c in base_cols if c in echem.columns]].drop_duplicates("roi_id")
    for src in [
        fronts.drop(columns=[c for c in ["cohort_role", "cycleNo", "event_reference_cycle"] if c in fronts.columns], errors="ignore").drop_duplicates("roi_id"),
        residuals.drop(columns=[c for c in ["cohort_role", "cycleNo", "event_reference_cycle"] if c in residuals.columns], errors="ignore").drop_duplicates("roi_id"),
        mobility[[c for c in ["roi_id", "rollout_mobility_difficulty_score", "latent_path_length", "first_last_corr"] if c in mobility.columns]].drop_duplicates("roi_id"),
        modes[[c for c in ["roi_id", "mode_label", "mode_review_priority"] if c in modes.columns]].drop_duplicates("roi_id"),
    ]:
        if "roi_id" in src.columns:
            df = df.merge(src, on="roi_id", how="left")

    if not original_manifest.empty:
        original = original_manifest[[c for c in ["roi_id", "auto_review_flags", "manual_qc_status"] if c in original_manifest.columns]].copy()
        original = original.rename(columns={"auto_review_flags": "original_auto_review_flags", "manual_qc_status": "original_manual_qc_status"})
        original["in_original_qc_package"] = True
        df = df.merge(original.drop_duplicates("roi_id"), on="roi_id", how="left")
    df["in_original_qc_package"] = df.get("in_original_qc_package", False).fillna(False).astype(bool)

    for col in FOCUS_FEATURES + ["front_quality_score", "q70_phase_slope_bootstrap_p05", "q70_phase_slope_bootstrap_p95", "q70_radius2_slope_bootstrap_p05_px2_per_s", "q70_radius2_slope_bootstrap_p95_px2_per_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["q70_phase_ci_positive"] = df.get("q70_phase_slope_bootstrap_p05", pd.Series(np.nan, index=df.index)).gt(0)
    df["q70_radius_ci_crosses_zero"] = (
        df.get("q70_radius2_slope_bootstrap_p05_px2_per_s", pd.Series(np.nan, index=df.index)).le(0)
        & df.get("q70_radius2_slope_bootstrap_p95_px2_per_s", pd.Series(np.nan, index=df.index)).ge(0)
    )
    df["front_quality_score"] = df.get("front_quality_score", pd.Series(np.nan, index=df.index))
    df["front_quality_rank_score"] = zscore(df["front_quality_score"]).fillna(0)
    df["phase_sign_rank_score"] = zscore(df.get("phase_slope_positive_fraction_protocol_residual", pd.Series(0, index=df.index))).fillna(0)
    df["phase_strength_rank_score"] = zscore(df.get("threshold_robust_phase_score", pd.Series(0, index=df.index))).fillna(0)
    df["rollout_rank_score"] = zscore(df.get("rollout_mobility_difficulty_score", pd.Series(0, index=df.index))).fillna(0)
    df["diffusion_abs_rank_score"] = zscore(df.get("diffusion_proxy_abs_median_um2_per_s", pd.Series(0, index=df.index)).abs()).fillna(0)
    df["control_balance_priority_score"] = (
        1.2 * df["front_quality_rank_score"]
        + 1.0 * df["phase_sign_rank_score"].abs()
        + 0.8 * df["phase_strength_rank_score"].abs()
        + 0.6 * df["rollout_rank_score"]
        + 0.4 * df["diffusion_abs_rank_score"]
        + 0.5 * df["q70_phase_ci_positive"].astype(float)
    )
    return df


def select_balanced(df: pd.DataFrame, n_per_role: int, n_extra_controls: int) -> pd.DataFrame:
    controls = df[df["cohort_role"] == "control"].copy()
    events = df[df["cohort_role"] == "event"].copy()

    selected_parts = []
    # Include controls that were already in the primary package, then add high-quality controls not yet selected.
    selected_parts.append(controls[controls["in_original_qc_package"]].sort_values("control_balance_priority_score", ascending=False).head(max(4, n_per_role // 2)))
    remaining_controls = controls[~controls["roi_id"].isin(pd.concat(selected_parts, ignore_index=True).get("roi_id", pd.Series(dtype=str)))]
    selected_parts.append(remaining_controls.sort_values("control_balance_priority_score", ascending=False).head(n_per_role + n_extra_controls))

    control_selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates("roi_id").head(n_per_role + n_extra_controls)
    event_selected = events.sort_values("control_balance_priority_score", ascending=False).head(n_per_role)
    selected = pd.concat([event_selected, control_selected], ignore_index=True).drop_duplicates("roi_id")
    selected["selection_group"] = np.where(selected["cohort_role"].eq("control"), "control_balancing", "event_comparator")
    return selected.sort_values(["cohort_role", "control_balance_priority_score"], ascending=[True, False])


def review_flags(row: pd.Series, stats: Dict[str, object]) -> List[str]:
    flags = []
    if finite_float(stats.get("edge_touch_fraction"), 0) > 0.03:
        flags.append("front_touches_crop_edge")
    if finite_float(stats.get("largest_component_fraction"), 0) < 0.45:
        flags.append("fragmented_q70_mask")
    if finite_float(row.get("phase_slope_positive_fraction"), 0) < 0.75 and finite_float(row.get("phase_slope_negative_fraction"), 0) < 0.75:
        flags.append("threshold_sign_unstable")
    if finite_float(row.get("q70_radius2_slope_bootstrap_p05_px2_per_s")) < 0 < finite_float(row.get("q70_radius2_slope_bootstrap_p95_px2_per_s")):
        flags.append("diffusion_ci_crosses_zero")
    if row.get("cohort_role") == "control" and finite_float(row.get("rollout_mobility_difficulty_score"), 0) > 2:
        flags.append("active_control")
    if row.get("in_original_qc_package"):
        flags.append("already_in_primary_qc")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_qc_package")
    parser.add_argument("--n-per-role", type=int, default=16)
    parser.add_argument("--n-extra-controls", type=int, default=8)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out_dir = Path(args.out_dir)
    png_dir = out_dir / "qc_panels"
    png_dir.mkdir(parents=True, exist_ok=True)

    table = prepare_table(derived)
    selected = select_balanced(table, args.n_per_role, args.n_extra_controls)

    rows: List[Dict[str, object]] = []
    for _, row in selected.iterrows():
        roi_id = str(row["roi_id"])
        out_png = png_dir / f"{roi_id}_qc.png"
        stats = render_roi_qc(roi_id, str(row["npz_path"]), row, out_png)
        flags = review_flags(row, stats)
        rows.append({
            "roi_id": roi_id,
            "cohort_role": row.get("cohort_role", ""),
            "selection_group": row.get("selection_group", ""),
            "cycleNo": finite_float(row.get("cycleNo")),
            "event_reference_cycle": finite_float(row.get("event_reference_cycle")),
            "mode_label": row.get("mode_label", ""),
            "mode_review_priority": finite_float(row.get("mode_review_priority")),
            "control_balance_priority_score": finite_float(row.get("control_balance_priority_score")),
            "front_quality_score": finite_float(row.get("front_quality_score")),
            "phase_slope_positive_fraction_protocol_residual": finite_float(row.get("phase_slope_positive_fraction_protocol_residual")),
            "threshold_robust_phase_score": finite_float(row.get("threshold_robust_phase_score")),
            "diffusion_proxy_abs_median_um2_per_s": finite_float(row.get("diffusion_proxy_abs_median_um2_per_s")),
            "diffusion_proxy_abs_median_um2_per_s_protocol_residual": finite_float(row.get("diffusion_proxy_abs_median_um2_per_s_protocol_residual")),
            "rollout_mobility_difficulty_score": finite_float(row.get("rollout_mobility_difficulty_score")),
            "q70_phase_ci_positive": bool(row.get("q70_phase_ci_positive", False)),
            "q70_radius_ci_crosses_zero": bool(row.get("q70_radius_ci_crosses_zero", False)),
            "in_original_qc_package": bool(row.get("in_original_qc_package", False)),
            **stats,
            "auto_review_flags": ";".join(flags) if flags else "none",
            "manual_qc_status": "pending",
            "manual_qc_decision": "",
            "manual_qc_notes": "",
        })

    manifest = pd.DataFrame(rows).sort_values(["cohort_role", "control_balance_priority_score"], ascending=[True, False])
    manifest_path = out_dir / "control_balanced_front_qc_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Control-Balanced NMC Front QC Package</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ccc;padding:4px;font-size:12px;} img{max-width:520px;} .flags{font-weight:bold;color:#7a2100;}</style>",
        "</head><body>",
        "<h1>Control-Balanced NMC Front QC Package</h1>",
        "<p>Augments the primary high-signal QC package with additional controls for manual event/control front-mask review. Green contours show q70 bright-front masks; yellow contours show the central ROI guard mask.</p>",
        "<table><tr><th>ROI</th><th>Role</th><th>Cycle</th><th>Priority</th><th>Flags</th><th>Panel</th></tr>",
    ]
    for _, row in manifest.iterrows():
        rel = Path(row["qc_png"]).relative_to(out_dir)
        html_lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['roi_id']))}</td>"
            f"<td>{html.escape(str(row['cohort_role']))}</td>"
            f"<td>{finite_float(row['cycleNo']):.0f}</td>"
            f"<td>{finite_float(row['control_balance_priority_score']):.3f}</td>"
            f"<td class='flags'>{html.escape(str(row['auto_review_flags']))}</td>"
            f"<td><img src='{html.escape(str(rel))}'></td>"
            "</tr>"
        )
    html_lines += ["</table>", "</body></html>"]
    html_path = out_dir / "control_balanced_front_qc_index.html"
    html_path.write_text("\n".join(html_lines) + "\n")

    role_counts = manifest["cohort_role"].value_counts().to_dict()
    flag_counts = manifest["auto_review_flags"].str.get_dummies(sep=";").sum().sort_values(ascending=False).to_dict()
    original_overlap = int(manifest["in_original_qc_package"].sum())
    nonfragment_by_role = (
        manifest[~manifest["auto_review_flags"].str.contains("fragmented_q70_mask", regex=False)]
        .groupby("cohort_role")
        .size()
        .to_dict()
    )
    noflag_by_role = manifest[manifest["auto_review_flags"].eq("none")].groupby("cohort_role").size().to_dict()
    summary = {
        "n_selected_roi": int(len(manifest)),
        "n_event_roi": int(role_counts.get("event", 0)),
        "n_control_roi": int(role_counts.get("control", 0)),
        "n_original_qc_overlap": original_overlap,
        "n_new_roi": int(len(manifest) - original_overlap),
        "flag_counts": flag_counts,
        "nonfragmented_by_role": nonfragment_by_role,
        "no_auto_flag_by_role": noflag_by_role,
        "top_controls": clean_json(manifest[manifest["cohort_role"] == "control"].head(10).to_dict("records")),
        "top_events": clean_json(manifest[manifest["cohort_role"] == "event"].head(10).to_dict("records")),
        "guardrail": "This package is a manual-review augmentation. It balances control/event visual candidates but does not assign accept/reject QC labels or validate calibrated diffusion.",
        "outputs": {
            "manifest": str(manifest_path),
            "html_index": str(html_path),
            "qc_panels": str(png_dir),
            "summary": str(out_dir / "control_balanced_front_qc_summary.json"),
        },
    }
    with (out_dir / "control_balanced_front_qc_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    readme = [
        "# Control-Balanced Front QC Package",
        "",
        "Manual-review augmentation for balanced event/control NMC front-mask QC.",
        "",
        f"- Selected ROI: {summary['n_selected_roi']}",
        f"- Event/control ROI: {summary['n_event_roi']} / {summary['n_control_roi']}",
        f"- New ROI beyond primary QC package: {summary['n_new_roi']}",
        f"- Non-fragmented by role: {summary['nonfragmented_by_role']}",
        f"- No-auto-flag by role: {summary['no_auto_flag_by_role']}",
        "",
        "Guardrail: this package enables manual review balance; it does not itself validate diffusion or final front masks.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n")
    print(json.dumps({k: summary[k] for k in ["n_selected_roi", "n_event_roi", "n_control_roi", "n_new_roi", "nonfragmented_by_role", "no_auto_flag_by_role"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
