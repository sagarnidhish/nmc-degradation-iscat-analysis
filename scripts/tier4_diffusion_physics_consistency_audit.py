#!/usr/bin/env python3
"""Audit physical consistency of apparent diffusion/front-motion proxies.

This is a stricter follow-up to the apparent-diffusion calibration and
control-balanced diffusion sanity audits. It collapses the threshold sweep to
per-ROI consistency scores and asks whether any automatic ROI satisfies the
minimum conditions needed before interpreting radius^2 front slopes as a
material-like diffusion proxy:

- positive radius^2 expansion across most thresholds,
- adequate radius^2 linear-fit quality,
- low threshold sensitivity of the apparent D estimate,
- stable HDF5 timing metadata,
- low drift/context artifacts when available.

The output is a physics-claim gate, not a label generator. It intentionally
keeps calibrated diffusion claims blocked unless the automatic evidence is
internally consistent before manual QC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score


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


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def entropy_binary(frac: float) -> float:
    if not np.isfinite(frac) or frac <= 0 or frac >= 1:
        return 0.0
    return float(-(frac * np.log2(frac) + (1.0 - frac) * np.log2(1.0 - frac)))


def signed_auc(y: pd.Series, x: pd.Series) -> float:
    valid = y.isin([0, 1]) & x.notna()
    if valid.sum() < 8 or y[valid].nunique() != 2:
        return np.nan
    auc = roc_auc_score(y[valid].astype(int), x[valid].astype(float))
    return float(max(auc, 1.0 - auc))


def mannwhitney_test(df: pd.DataFrame, feature: str, target: str) -> dict[str, Any] | None:
    if feature not in df or target not in df:
        return None
    y = pd.to_numeric(df[target], errors="coerce")
    x = pd.to_numeric(df[feature], errors="coerce")
    valid = y.isin([0, 1]) & x.notna()
    if valid.sum() < 8 or y[valid].nunique() != 2:
        return None
    pos = x[valid & y.eq(1)]
    neg = x[valid & y.eq(0)]
    if len(pos) < 3 or len(neg) < 3:
        return None
    p = mannwhitneyu(pos, neg, alternative="two-sided").pvalue
    auc = roc_auc_score(y[valid].astype(int), x[valid].astype(float))
    return {
        "target": target,
        "feature": feature,
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "median_positive": float(pos.median()),
        "median_negative": float(neg.median()),
        "median_positive_minus_negative": float(pos.median() - neg.median()),
        "roc_auc": float(auc),
        "abs_oriented_auc": float(max(auc, 1.0 - auc)),
        "mannwhitney_p": float(p),
    }


def spearman_pair(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any] | None:
    if x_col not in df or y_col not in df:
        return None
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 10 or x[valid].nunique() < 3 or y[valid].nunique() < 3:
        return None
    rho, p = spearmanr(x[valid], y[valid])
    return {
        "x": x_col,
        "y": y_col,
        "n": int(valid.sum()),
        "spearman_rho": float(rho),
        "abs_rho": float(abs(rho)),
        "p_value": float(p),
    }


def threshold_slope(thresholds: pd.Series, values: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(thresholds, errors="coerce")
    y = pd.to_numeric(values, errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 4 or x[valid].nunique() < 3 or y[valid].nunique() < 3:
        return np.nan, np.nan
    rho, p = spearmanr(x[valid], y[valid])
    return float(rho), float(p)


def collapse_roi(calib: pd.DataFrame, sanity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for roi_id, sub in calib.groupby("roi_id", dropna=False):
        sub = sub.sort_values("threshold_quantile")
        d = pd.to_numeric(sub["apparent_D_h5median_px0p096_um2_per_s"], errors="coerce")
        abs_d = d.abs()
        r2 = pd.to_numeric(sub.get("radius2_slope_r2"), errors="coerce")
        phase_slope = pd.to_numeric(sub.get("phase_fraction_slope_per_s"), errors="coerce")
        radius_slope = pd.to_numeric(sub.get("radius2_slope_h5_median_px2_per_s"), errors="coerce")
        thresholds = pd.to_numeric(sub.get("threshold_quantile"), errors="coerce")
        pos_frac = float((d > 0).mean()) if d.notna().any() else np.nan
        neg_frac = float((d < 0).mean()) if d.notna().any() else np.nan
        robust_positive = float(((d > 0) & r2.ge(0.20)).mean()) if d.notna().any() else np.nan
        median_abs = float(abs_d.median()) if abs_d.notna().any() else np.nan
        iqr_abs = float(abs_d.quantile(0.75) - abs_d.quantile(0.25)) if abs_d.notna().sum() >= 4 else np.nan
        threshold_cv = float(iqr_abs / median_abs) if np.isfinite(median_abs) and median_abs > 0 else np.nan
        d_rho, d_p = threshold_slope(thresholds, d)
        abs_d_rho, abs_d_p = threshold_slope(thresholds, abs_d)
        first = sub.iloc[0]
        row = {
            "roi_id": roi_id,
            "cycleNo": finite_float(first.get("cycleNo")),
            "source_stem": first.get("source_stem", ""),
            "cohort_role": first.get("cohort_role", ""),
            "is_event_roi": finite_float(first.get("is_event_roi")),
            "event_reference_cycle": finite_float(first.get("event_reference_cycle")),
            "future_any_drop_within_8cycles": finite_float(first.get("future_any_drop_within_8cycles")),
            "future_any_drop_within_16cycles": finite_float(first.get("future_any_drop_within_16cycles")),
            "any_abrupt_drop": finite_float(first.get("any_abrupt_drop")),
            "n_thresholds": int(len(sub)),
            "median_apparent_D_um2_per_s": float(d.median()) if d.notna().any() else np.nan,
            "median_abs_apparent_D_um2_per_s": median_abs,
            "iqr_abs_apparent_D_um2_per_s": iqr_abs,
            "threshold_sensitivity_iqr_over_median_abs": threshold_cv,
            "positive_D_fraction": pos_frac,
            "negative_D_fraction": neg_frac,
            "sign_entropy_bits": entropy_binary(pos_frac),
            "robust_positive_fit_fraction": robust_positive,
            "median_radius2_fit_r2": float(r2.median()) if r2.notna().any() else np.nan,
            "min_radius2_fit_r2": float(r2.min()) if r2.notna().any() else np.nan,
            "median_phase_slope_per_s": float(phase_slope.median()) if phase_slope.notna().any() else np.nan,
            "median_radius2_slope_px2_per_s": float(radius_slope.median()) if radius_slope.notna().any() else np.nan,
            "D_vs_threshold_spearman_rho": d_rho,
            "D_vs_threshold_spearman_p": d_p,
            "abs_D_vs_threshold_spearman_rho": abs_d_rho,
            "abs_D_vs_threshold_spearman_p": abs_d_p,
            "roi_elapsed_to_h5_median_ratio": finite_float(first.get("roi_elapsed_to_h5_median_ratio")),
            "h5_dt_max_to_median_ratio": finite_float(first.get("h5_dt_max_to_median_ratio")),
            "transferred_masked_residual_signature": finite_float(first.get("transferred_masked_residual_signature")),
            "validation_score_recon": finite_float(first.get("validation_score_recon")),
        }
        rows.append(row)
    roi = pd.DataFrame(rows)

    if not sanity.empty and "roi_id" in sanity:
        keep = [
            "roi_id", "selected_diffusion_um2_per_s", "selected_r2", "drift_to_motion_ratio",
            "front_quality_score", "diffusion_proxy_median_um2_per_s", "threshold_robust_diffusion_score",
            "q70_radius2_slope_bootstrap_p05_px2_per_s", "q70_radius2_slope_bootstrap_p95_px2_per_s",
            "mask_instability_score", "fallback_frame_fraction", "accepted_area_cv", "manual_qc_status",
            "positive_estimator_count", "negative_estimator_count", "automatic_positive_diffusion_proxy_candidate",
            "publication_diffusion_candidate",
        ]
        available = [c for c in keep if c in sanity.columns]
        roi = roi.merge(sanity[available].drop_duplicates("roi_id"), on="roi_id", how="left")

    # Conservative automatic gates. These are intentionally difficult to pass.
    roi["gate_all_thresholds_present"] = roi["n_thresholds"].ge(7)
    roi["gate_positive_expansion"] = roi["positive_D_fraction"].ge(0.80) & roi["median_apparent_D_um2_per_s"].gt(0)
    roi["gate_fit_quality"] = roi["median_radius2_fit_r2"].ge(0.20)
    roi["gate_threshold_stability"] = roi["threshold_sensitivity_iqr_over_median_abs"].le(1.50)
    roi["gate_h5_timing_stable"] = roi["h5_dt_max_to_median_ratio"].le(2.0)
    roi["gate_low_drift"] = roi.get("drift_to_motion_ratio", pd.Series(np.nan, index=roi.index)).fillna(0.0).le(0.05)
    roi["gate_q70_positive_ci"] = pd.to_numeric(roi.get("q70_radius2_slope_bootstrap_p05_px2_per_s", np.nan), errors="coerce").gt(0)
    gate_cols = [c for c in roi.columns if c.startswith("gate_")]
    roi["diffusion_physics_gate_count"] = roi[gate_cols].sum(axis=1).astype(int)
    roi["automatic_diffusion_physics_consistent"] = (
        roi["gate_all_thresholds_present"]
        & roi["gate_positive_expansion"]
        & roi["gate_fit_quality"]
        & roi["gate_threshold_stability"]
        & roi["gate_h5_timing_stable"]
        & roi["gate_low_drift"]
    )
    roi["publication_ready_diffusion_candidate"] = roi["automatic_diffusion_physics_consistent"] & roi["gate_q70_positive_ci"] & roi.get("manual_qc_status", "pending").eq("accepted")
    roi["physics_consistency_score"] = (
        roi["positive_D_fraction"].fillna(0)
        + roi["robust_positive_fit_fraction"].fillna(0)
        + roi["median_radius2_fit_r2"].clip(lower=0, upper=1).fillna(0)
        + (1.0 - roi["threshold_sensitivity_iqr_over_median_abs"].clip(lower=0, upper=2).fillna(2) / 2.0)
        + roi["gate_h5_timing_stable"].astype(float)
        + roi["gate_low_drift"].astype(float)
    )
    return roi.sort_values(["automatic_diffusion_physics_consistent", "physics_consistency_score"], ascending=[False, False])


def gate_summary(roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in roi.columns if c.startswith("gate_")] + ["automatic_diffusion_physics_consistent", "publication_ready_diffusion_candidate"]:
        vals = roi[col].fillna(False).astype(bool)
        rows.append({"criterion": col, "n_pass": int(vals.sum()), "n_total": int(len(vals)), "fraction_pass": float(vals.mean()) if len(vals) else np.nan})
    return pd.DataFrame(rows)


def source_summary(roi: pd.DataFrame) -> pd.DataFrame:
    if roi.empty:
        return pd.DataFrame()
    return (
        roi.groupby("source_stem", dropna=False)
        .agg(
            n_roi=("roi_id", "count"),
            n_cycles=("cycleNo", "nunique"),
            median_abs_D=("median_abs_apparent_D_um2_per_s", "median"),
            median_positive_fraction=("positive_D_fraction", "median"),
            median_fit_r2=("median_radius2_fit_r2", "median"),
            median_h5_dt_max_to_median=("h5_dt_max_to_median_ratio", "median"),
            physics_consistent_count=("automatic_diffusion_physics_consistent", "sum"),
            future8_positive_fraction=("future_any_drop_within_8cycles", "mean"),
        )
        .reset_index()
        .sort_values(["physics_consistent_count", "n_roi"], ascending=[False, False])
    )


def tests(roi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "median_apparent_D_um2_per_s",
        "median_abs_apparent_D_um2_per_s",
        "positive_D_fraction",
        "robust_positive_fit_fraction",
        "median_radius2_fit_r2",
        "threshold_sensitivity_iqr_over_median_abs",
        "physics_consistency_score",
        "diffusion_physics_gate_count",
        "h5_dt_max_to_median_ratio",
        "transferred_masked_residual_signature",
    ]
    targets = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "any_abrupt_drop"]
    rows = []
    for target in targets:
        for feature in features:
            row = mannwhitney_test(roi, feature, target)
            if row is not None:
                rows.append(row)
    feature_tests = pd.DataFrame(rows)
    if not feature_tests.empty:
        feature_tests = feature_tests.sort_values(["target", "mannwhitney_p", "abs_oriented_auc"], ascending=[True, True, False])

    pairs = []
    for x in features:
        for y in ["transferred_masked_residual_signature", "validation_score_recon", "cycleNo"]:
            if x != y:
                row = spearman_pair(roi, x, y)
                if row is not None:
                    pairs.append(row)
    corr = pd.DataFrame(pairs)
    if not corr.empty:
        corr = corr.sort_values("abs_rho", ascending=False)
    return feature_tests, corr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/diffusion_physics_consistency_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    calib = read_csv(derived / "apparent_diffusion_calibration_bounds" / "apparent_diffusion_calibration_joined.csv")
    sanity_path = derived / "control_balanced_diffusion_proxy_sanity_audit" / "diffusion_proxy_sanity_joined.csv"
    sanity = read_csv(sanity_path) if sanity_path.exists() else pd.DataFrame()

    roi = collapse_roi(calib, sanity)
    gates = gate_summary(roi)
    sources = source_summary(roi)
    feature_tests, corr = tests(roi)

    paths = {
        "roi_scores": out / "diffusion_physics_consistency_roi_scores.csv",
        "gate_counts": out / "diffusion_physics_consistency_gate_counts.csv",
        "source_summary": out / "diffusion_physics_consistency_source_summary.csv",
        "feature_tests": out / "diffusion_physics_consistency_feature_tests.csv",
        "correlations": out / "diffusion_physics_consistency_correlations.csv",
        "summary": out / "diffusion_physics_consistency_summary.json",
    }
    roi.to_csv(paths["roi_scores"], index=False)
    gates.to_csv(paths["gate_counts"], index=False)
    sources.to_csv(paths["source_summary"], index=False)
    feature_tests.to_csv(paths["feature_tests"], index=False)
    corr.to_csv(paths["correlations"], index=False)

    consistent = roi[roi["automatic_diffusion_physics_consistent"]].copy()
    publication = roi[roi["publication_ready_diffusion_candidate"]].copy()
    summary = clean_json({
        "n_roi": int(len(roi)),
        "n_sources": int(roi["source_stem"].nunique()),
        "n_cycles": int(roi["cycleNo"].nunique()),
        "n_threshold_rows": int(len(calib)),
        "n_automatic_diffusion_physics_consistent": int(len(consistent)),
        "n_publication_ready_diffusion_candidates": int(len(publication)),
        "median_abs_apparent_D_um2_per_s": float(roi["median_abs_apparent_D_um2_per_s"].median()),
        "median_positive_D_fraction": float(roi["positive_D_fraction"].median()),
        "median_radius2_fit_r2": float(roi["median_radius2_fit_r2"].median()),
        "median_threshold_sensitivity_iqr_over_median_abs": float(roi["threshold_sensitivity_iqr_over_median_abs"].median()),
        "gate_counts": gates.to_dict("records"),
        "top_consistent_candidates": consistent.head(12).to_dict("records"),
        "top_physics_scores": roi.head(12).to_dict("records"),
        "top_feature_tests": feature_tests.head(16).to_dict("records") if not feature_tests.empty else [],
        "top_correlations": corr.head(16).to_dict("records") if not corr.empty else [],
        "source_summary": sources.head(12).to_dict("records") if not sources.empty else [],
        "guardrail": "Automatic apparent-D candidates must pass positive expansion, radius^2 fit, threshold-stability, timing, drift, q70 CI, and manual-QC gates before any calibrated diffusion claim. This audit is a physics-consistency filter over optical front proxies, not a material diffusion measurement.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "README.md").write_text(
        "# Diffusion Physics Consistency Audit\n\n"
        "Collapses calibrated threshold-front apparent-D rows to ROI-level consistency gates.\n\n"
        f"- ROI rows: {summary['n_roi']}\n"
        f"- Threshold rows: {summary['n_threshold_rows']}\n"
        f"- Automatic physics-consistent candidates: {summary['n_automatic_diffusion_physics_consistent']}\n"
        f"- Publication-ready diffusion candidates: {summary['n_publication_ready_diffusion_candidates']}\n\n"
        f"Guardrail: {summary['guardrail']}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
