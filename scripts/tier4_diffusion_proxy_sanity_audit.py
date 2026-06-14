#!/usr/bin/env python3
"""Sanity audit for diffusion-like optical-front proxies.

The project has several radius-squared slope outputs that are useful optical
front descriptors, but they should not be promoted to diffusion coefficients
unless they pass basic physical and measurement-consistency checks. This audit
joins the selected high-resolution front tracking table to the threshold-sweep
ROI table and asks whether any ROI has nonnegative, estimator-consistent,
low-drift, threshold-robust apparent transport.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def finite_float(x, default=np.nan) -> float:
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def sign_label(x: float, eps: float = 0.0) -> str:
    val = finite_float(x)
    if not np.isfinite(val) or abs(val) <= eps:
        return "zero_or_missing"
    return "positive" if val > 0 else "negative"


def selected_to_rank_roi_id(roi_id: str) -> str:
    parts = str(roi_id).split("_")
    if len(parts) != 3:
        return str(roi_id)
    return f"{parts[0]}_{parts[1].replace('front', 'rank')}_{parts[2]}"


def top_records(df: pd.DataFrame, sort_col: str, n: int = 10, ascending: bool = False) -> List[Dict[str, object]]:
    if df.empty or sort_col not in df.columns:
        return []
    sdf = df.sort_values(sort_col, ascending=ascending).head(n)
    out = []
    for _, row in sdf.iterrows():
        rec = {}
        for col, val in row.items():
            if isinstance(val, (np.integer,)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                rec[col] = float(val) if np.isfinite(val) else None
            else:
                rec[col] = val
        out.append(rec)
    return out


def compact_candidate_records(df: pd.DataFrame, n: int = 10) -> List[Dict[str, object]]:
    cols = [
        "selected_roi_id", "roi_id", "cycleNo", "cohort_role", "selected_diffusion_um2_per_s",
        "diffusion_proxy_median_um2_per_s", "selected_r2", "drift_to_motion_ratio",
        "positive_estimator_count", "negative_estimator_count", "estimator_consensus_sign",
        "selected_nonnegative", "threshold_nonnegative", "q70_bootstrap_positive",
        "automatic_positive_diffusion_proxy_candidate", "publication_diffusion_candidate",
        "manual_qc_status", "auto_qc_notes",
    ]
    keep = [c for c in cols if c in df.columns]
    if df.empty or not keep:
        return []
    sdf = df.sort_values(
        ["automatic_positive_diffusion_proxy_candidate", "positive_estimator_count", "selected_r2"],
        ascending=[False, False, False],
    ).head(n)
    return top_records(sdf[keep], keep[0], n=n, ascending=True)


def event_control_tests(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows = []
    if "cohort_role" not in df.columns:
        return pd.DataFrame(rows)
    for feature in features:
        if feature not in df.columns:
            continue
        event = pd.to_numeric(df.loc[df["cohort_role"] == "event", feature], errors="coerce").dropna()
        control = pd.to_numeric(df.loc[df["cohort_role"] == "control", feature], errors="coerce").dropna()
        if len(event) < 2 or len(control) < 2:
            p = np.nan
        else:
            p = float(mannwhitneyu(event, control, alternative="two-sided").pvalue)
        rows.append({
            "feature": feature,
            "n_event": int(len(event)),
            "n_control": int(len(control)),
            "event_median": float(np.nanmedian(event)) if len(event) else np.nan,
            "control_median": float(np.nanmedian(control)) if len(control) else np.nan,
            "event_control_median_diff": float(np.nanmedian(event) - np.nanmedian(control)) if len(event) and len(control) else np.nan,
            "mannwhitney_p": p,
        })
    return pd.DataFrame(rows).sort_values("mannwhitney_p", na_position="last")


def correlations(df: pd.DataFrame, x_cols: Iterable[str], y_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for x_col in x_cols:
        for y_col in y_cols:
            if x_col not in df.columns or y_col not in df.columns:
                continue
            x = pd.to_numeric(df[x_col], errors="coerce")
            y = pd.to_numeric(df[y_col], errors="coerce")
            mask = x.notna() & y.notna()
            if int(mask.sum()) < 5:
                continue
            rho, p = spearmanr(x[mask], y[mask])
            rows.append({
                "x": x_col,
                "y": y_col,
                "n": int(mask.sum()),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_rho"] = out["spearman_rho"].abs()
    return out.sort_values(["p_value", "abs_rho"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/diffusion_proxy_sanity_audit")
    parser.add_argument("--front-calibration-dir", default=None)
    parser.add_argument("--min-r2", type=float, default=0.50)
    parser.add_argument("--max-drift-ratio", type=float, default=0.25)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    front_calibration_dir = Path(args.front_calibration_dir) if args.front_calibration_dir else derived / "front_roi_calibration_qc"
    selected = read_csv(front_calibration_dir / "calibrated_front_tracking.csv")
    qc = read_csv(front_calibration_dir / "front_roi_qc_calibration.csv")
    threshold = read_csv(derived / "multi_cycle_threshold_robust_fronts" / "threshold_robust_front_summary.csv")
    masks = read_csv(derived / "particle_mask_stability_audit" / "particle_mask_stability_per_roi.csv")

    if selected.empty:
        raise FileNotFoundError("Missing calibrated_front_tracking.csv")
    selected = selected.copy()
    selected["selected_roi_id"] = selected["roi_id_x"] if "roi_id_x" in selected.columns else selected["roi_id"]
    selected["roi_id"] = selected["selected_roi_id"].map(selected_to_rank_roi_id)
    selected["selected_diffusion_um2_per_s"] = pd.to_numeric(selected["apparent_diffusion_proxy_full_um2_per_s"], errors="coerce")
    selected["selected_radius2_um2_per_s"] = pd.to_numeric(selected["radius2_slope_full_um2_per_s"], errors="coerce")
    selected["selected_r2"] = pd.to_numeric(selected["radius2_slope_r2"], errors="coerce")
    selected["area_radius2_um2_per_s"] = pd.to_numeric(selected.get("area_equiv_radius2_slope_full_um2_per_s", np.nan), errors="coerce")
    selected["p90_radius2_um2_per_s"] = pd.to_numeric(selected.get("p90_radius2_slope_full_um2_per_s", np.nan), errors="coerce")
    selected["apparent_motion_um"] = np.sqrt(np.abs(selected["selected_radius2_um2_per_s"]) * pd.to_numeric(selected["elapsed_s"], errors="coerce"))
    selected["stage_drift_xy_um"] = pd.to_numeric(selected.get("stage_drift_xy_um", np.nan), errors="coerce")
    selected["drift_to_motion_ratio"] = selected["stage_drift_xy_um"] / selected["apparent_motion_um"].replace(0, np.nan)

    keep_threshold = [
        "roi_id", "cohort_role", "is_event_roi", "event_reference_cycle", "front_quality_score",
        "diffusion_proxy_median_um2_per_s", "diffusion_proxy_abs_median_um2_per_s",
        "radius2_slope_positive_fraction", "radius2_slope_negative_fraction",
        "threshold_robust_diffusion_score", "q70_radius2_slope_bootstrap_p05_px2_per_s",
        "q70_radius2_slope_bootstrap_p95_px2_per_s", "phase_slope_positive_fraction",
        "threshold_robust_phase_score", "degradation_mode_hypothesis",
    ]
    joined = selected.merge(threshold[[c for c in keep_threshold if c in threshold.columns]], on="roi_id", how="left", suffixes=("", "_threshold"))

    if not qc.empty:
        qid = "roi_id_x" if "roi_id_x" in qc.columns else ("roi_id" if "roi_id" in qc.columns else None)
        if qid:
            q = qc.copy()
            q["selected_roi_id"] = q[qid]
            qc_keep = [c for c in ["selected_roi_id", "manual_qc_status", "auto_qc_notes", "front_tracking_interpretation"] if c in q.columns]
            joined = joined.merge(q[qc_keep].drop_duplicates("selected_roi_id"), on="selected_roi_id", how="left")

    if not masks.empty:
        mask_keep = [c for c in ["roi_id", "fallback_frame_fraction", "accepted_area_cv", "mask_instability_score", "accepted_centroid_path_px"] if c in masks.columns]
        joined = joined.merge(masks[mask_keep].drop_duplicates("roi_id"), on="roi_id", how="left")

    sign_cols = {
        "selected_radius2_sign": "selected_radius2_um2_per_s",
        "area_radius2_sign": "area_radius2_um2_per_s",
        "p90_radius2_sign": "p90_radius2_um2_per_s",
        "threshold_median_sign": "diffusion_proxy_median_um2_per_s",
    }
    for out_col, src_col in sign_cols.items():
        joined[out_col] = joined[src_col].map(sign_label)
    joined["positive_estimator_count"] = joined[list(sign_cols)].eq("positive").sum(axis=1)
    joined["negative_estimator_count"] = joined[list(sign_cols)].eq("negative").sum(axis=1)
    joined["estimator_consensus_sign"] = np.where(
        joined["positive_estimator_count"] >= 3,
        "positive",
        np.where(joined["negative_estimator_count"] >= 3, "negative", "mixed"),
    )
    joined["selected_nonnegative"] = joined["selected_radius2_um2_per_s"] > 0
    joined["threshold_nonnegative"] = pd.to_numeric(joined["diffusion_proxy_median_um2_per_s"], errors="coerce") > 0
    joined["selected_fit_good"] = joined["selected_r2"] >= args.min_r2
    joined["low_drift_relative_to_motion"] = joined["drift_to_motion_ratio"] <= args.max_drift_ratio
    joined["q70_bootstrap_positive"] = pd.to_numeric(joined["q70_radius2_slope_bootstrap_p05_px2_per_s"], errors="coerce") > 0
    joined["q70_bootstrap_negative"] = pd.to_numeric(joined["q70_radius2_slope_bootstrap_p95_px2_per_s"], errors="coerce") < 0
    joined["manual_qc_accepted"] = joined.get("manual_qc_status", "pending").astype(str).str.lower().eq("accepted")
    joined["automatic_positive_diffusion_proxy_candidate"] = (
        joined["selected_nonnegative"]
        & joined["threshold_nonnegative"]
        & joined["selected_fit_good"]
        & joined["low_drift_relative_to_motion"]
        & joined["q70_bootstrap_positive"]
        & (joined["positive_estimator_count"] >= 3)
    )
    joined["publication_diffusion_candidate"] = joined["automatic_positive_diffusion_proxy_candidate"] & joined["manual_qc_accepted"]
    joined["selected_diffusion_m2_per_s"] = joined["selected_diffusion_um2_per_s"] * 1e-12
    joined["threshold_diffusion_m2_per_s"] = pd.to_numeric(joined["diffusion_proxy_median_um2_per_s"], errors="coerce") * 1e-12

    criteria = [
        "selected_nonnegative", "threshold_nonnegative", "selected_fit_good",
        "low_drift_relative_to_motion", "q70_bootstrap_positive", "manual_qc_accepted",
    ]
    gate_rows = []
    for crit in criteria:
        gate_rows.append({
            "criterion": crit,
            "n_pass": int(joined[crit].fillna(False).sum()),
            "n_total": int(len(joined)),
            "fraction_pass": float(joined[crit].fillna(False).mean()) if len(joined) else np.nan,
        })
    gate = pd.DataFrame(gate_rows)

    tests = event_control_tests(joined, [
        "selected_diffusion_um2_per_s", "selected_radius2_um2_per_s", "selected_r2",
        "drift_to_motion_ratio", "positive_estimator_count", "negative_estimator_count",
        "diffusion_proxy_median_um2_per_s", "threshold_robust_diffusion_score",
    ])
    corr = correlations(
        joined,
        ["selected_diffusion_um2_per_s", "selected_r2", "positive_estimator_count", "drift_to_motion_ratio"],
        ["front_quality_score", "mask_instability_score", "fallback_frame_fraction", "threshold_robust_diffusion_score"],
    )

    joined_path = out / "diffusion_proxy_sanity_joined.csv"
    gate_path = out / "diffusion_proxy_sanity_gate_counts.csv"
    tests_path = out / "diffusion_proxy_sanity_event_control_tests.csv"
    corr_path = out / "diffusion_proxy_sanity_correlations.csv"
    joined.to_csv(joined_path, index=False)
    gate.to_csv(gate_path, index=False)
    tests.to_csv(tests_path, index=False)
    corr.to_csv(corr_path, index=False)

    summary = {
        "n_selected_front_rois": int(len(joined)),
        "selected_front_cohort_counts": joined.get("cohort_role", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "n_automatic_positive_diffusion_proxy_candidates": int(joined["automatic_positive_diffusion_proxy_candidate"].sum()),
        "n_publication_diffusion_candidates": int(joined["publication_diffusion_candidate"].sum()),
        "median_selected_diffusion_um2_per_s": float(np.nanmedian(joined["selected_diffusion_um2_per_s"])),
        "median_threshold_diffusion_um2_per_s": float(np.nanmedian(pd.to_numeric(joined["diffusion_proxy_median_um2_per_s"], errors="coerce"))),
        "selected_positive_fraction": float(joined["selected_nonnegative"].mean()),
        "threshold_positive_fraction": float(joined["threshold_nonnegative"].mean()),
        "estimator_consensus_counts": joined["estimator_consensus_sign"].value_counts(dropna=False).to_dict(),
        "gate_counts": top_records(gate, "n_pass", n=20, ascending=False),
        "top_automatic_candidates": compact_candidate_records(joined, n=10),
        "top_event_control_tests": top_records(tests, "mannwhitney_p", n=10, ascending=True),
        "top_correlations": top_records(corr, "p_value", n=10, ascending=True),
        "guardrail": (
            "This audit checks whether optical radius-squared front proxies behave like calibrated diffusion estimates. "
            "Publication diffusion claims require manual QC plus positive, estimator-consistent, low-drift, threshold-robust slopes; "
            "otherwise values remain apparent optical-front proxies."
        ),
        "outputs": {
            "front_calibration_dir": str(front_calibration_dir),
            "joined": str(joined_path),
            "gate_counts": str(gate_path),
            "event_control_tests": str(tests_path),
            "correlations": str(corr_path),
        },
    }
    with (out / "diffusion_proxy_sanity_audit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out / "README.md").open("w") as f:
        f.write("# Diffusion Proxy Sanity Audit\n\n")
        f.write("Checks whether selected optical-front radius-squared slopes satisfy basic diffusion-claim guardrails.\n\n")
        f.write(f"- Selected front ROIs: {summary['n_selected_front_rois']}\n")
        f.write(f"- Automatic positive proxy candidates: {summary['n_automatic_positive_diffusion_proxy_candidates']}\n")
        f.write(f"- Publication diffusion candidates after manual QC gate: {summary['n_publication_diffusion_candidates']}\n")
        f.write(f"- Estimator consensus counts: {summary['estimator_consensus_counts']}\n\n")
        f.write("Guardrail: values remain apparent optical-front proxies unless the full gate passes.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
