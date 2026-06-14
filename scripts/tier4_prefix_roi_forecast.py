#!/usr/bin/env python3
"""Prefix-only ROI sequence forecasts for NMC degradation physics outcomes.

This experiment asks whether early particle-region video content contains
predictive signal for later degradation/front outcomes. It computes features
from only the first 25/50/75% of each cropped ROI sequence and evaluates
leave-event-reference-cycle-out classifiers/regressors against event labels,
residual mode labels, and protocol-conditioned front-direction residuals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EVENT_MODE = "optical_brightening_decorrelating_rollout_hard_front_positive"
PREFIX_FRACTIONS = (0.25, 0.50, 0.75)


def clean_json(value: Any) -> Any:
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


def slope_features(y: np.ndarray, prefix: str) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    t = np.linspace(0.0, 1.0, len(y)) if len(y) else np.array([])
    out: Dict[str, float] = {}
    if len(y) < 3 or not np.isfinite(y).any():
        for name in ["first", "last", "delta", "mean", "std", "slope", "absdiff_mean", "absdiff_sum"]:
            out[f"{prefix}_{name}"] = np.nan
        return out
    valid = np.isfinite(y)
    yy = y[valid]
    tt = t[valid]
    out[f"{prefix}_first"] = float(yy[0])
    out[f"{prefix}_last"] = float(yy[-1])
    out[f"{prefix}_delta"] = float(yy[-1] - yy[0])
    out[f"{prefix}_mean"] = float(np.nanmean(yy))
    out[f"{prefix}_std"] = float(np.nanstd(yy))
    out[f"{prefix}_slope"] = float(np.polyfit(tt, yy, 1)[0]) if len(yy) >= 3 else np.nan
    diffs = np.diff(yy)
    out[f"{prefix}_absdiff_mean"] = float(np.nanmean(np.abs(diffs))) if len(diffs) else np.nan
    out[f"{prefix}_absdiff_sum"] = float(np.nansum(np.abs(diffs))) if len(diffs) else np.nan
    return out


def frame_prefix_features(npz_path: str, frac: float) -> Dict[str, float]:
    z = np.load(npz_path)
    frames = np.asarray(z["frames_norm"], dtype=np.float32)
    roi_norm_mean = np.asarray(z["roi_norm_mean"], dtype=float)
    roi_mean = np.asarray(z["roi_mean"], dtype=float)
    avg = np.asarray(z["average_intensity"], dtype=float)
    stage = np.asarray(z["stage_position"], dtype=float) if "stage_position" in z.files else None
    n_total = int(frames.shape[0])
    n = max(6, int(np.ceil(n_total * frac)))
    n = min(n, n_total)
    prefix_frames = frames[:n]
    first = prefix_frames[0]
    last = prefix_frames[-1]
    diff = last - first
    thresh = float(np.quantile(first, 0.70))
    low_thresh = float(np.quantile(first, 0.30))
    high_frac = (prefix_frames >= thresh).mean(axis=(1, 2))
    low_frac = (prefix_frames <= low_thresh).mean(axis=(1, 2))
    temporal_abs = np.abs(np.diff(prefix_frames, axis=0)).mean(axis=(1, 2)) if n > 1 else np.array([np.nan])
    row: Dict[str, float] = {
        "prefix_fraction": float(frac),
        "prefix_n_frames": int(n),
        "sequence_n_frames": n_total,
        "first_frame_mean": float(np.mean(first)),
        "last_prefix_frame_mean": float(np.mean(last)),
        "prefix_frame_delta_mean": float(np.mean(diff)),
        "prefix_frame_delta_abs_mean": float(np.mean(np.abs(diff))),
        "prefix_frame_delta_p95_abs": float(np.quantile(np.abs(diff), 0.95)),
        "first_q70_threshold": thresh,
        "first_q30_threshold": low_thresh,
        "prefix_temporal_absdiff_mean": float(np.nanmean(temporal_abs)),
        "prefix_temporal_absdiff_p95": float(np.nanquantile(temporal_abs, 0.95)),
        "prefix_spatial_std_first": float(np.std(first)),
        "prefix_spatial_std_last": float(np.std(last)),
    }
    row.update(slope_features(roi_norm_mean[:n], "roi_norm_mean_prefix"))
    row.update(slope_features(roi_mean[:n], "roi_mean_prefix"))
    row.update(slope_features(avg[:n], "average_intensity_prefix"))
    row.update(slope_features(high_frac, "high_fraction_prefix"))
    row.update(slope_features(low_frac, "low_fraction_prefix"))
    if stage is not None and stage.ndim == 2 and stage.shape[1] >= n:
        xy = stage[:2, :n].T
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if n > 1 else np.array([np.nan])
        drift = np.linalg.norm(xy[-1] - xy[0]) if n > 1 else np.nan
        row["stage_prefix_net_drift"] = float(drift)
        row["stage_prefix_step_mean"] = float(np.nanmean(steps))
        row["stage_prefix_step_max"] = float(np.nanmax(steps))
    else:
        row["stage_prefix_net_drift"] = np.nan
        row["stage_prefix_step_mean"] = np.nan
        row["stage_prefix_step_max"] = np.nan
    return row


def build_prefix_table(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in assignments.iterrows():
        npz_path = row.get("npz_path")
        if not isinstance(npz_path, str) or not os.path.exists(npz_path):
            continue
        for frac in PREFIX_FRACTIONS:
            feats = frame_prefix_features(npz_path, frac)
            base = row.to_dict()
            base.update(feats)
            rows.append(base)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "roi_id",
        "cohort_role",
        "mode_label",
        "degradation_mode_hypothesis",
        "npz_path",
        "preview_png",
        "source_stem",
        "validation_label",
    }
    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("target_") or col.startswith("outcome_"):
            continue
        if col in {"is_event_roi", "is_event_enriched_mode", "front_positive_residual_binary"}:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 8 and vals.nunique(dropna=True) > 1:
            numeric_cols.append(col)
    return numeric_cols


def logo_classification(df: pd.DataFrame, target: str, features: List[str], model_name: str) -> pd.DataFrame:
    x = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce").astype(int)
    groups = df["event_reference_cycle"]
    if y.nunique() < 2:
        return pd.DataFrame()
    if model_name == "logistic_l2":
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        ])
    elif model_name == "random_forest":
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=17,
            )),
        ])
    else:
        raise ValueError(model_name)
    rows = []
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(x, y, groups):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
            continue
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(x.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "target": target,
            "model": model_name,
            "prefix_fraction": float(df["prefix_fraction"].iloc[0]),
            "heldout_event_reference_cycle": float(groups.iloc[test_idx].iloc[0]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "roc_auc": float(roc_auc_score(y.iloc[test_idx], prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y.iloc[test_idx], pred)),
        })
    return pd.DataFrame(rows)


def logo_regression(df: pd.DataFrame, target: str, features: List[str], model_name: str) -> pd.DataFrame:
    x = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    groups = df["event_reference_cycle"]
    valid = y.notna()
    x = x.loc[valid]
    y = y.loc[valid]
    groups = groups.loc[valid]
    if len(y) < 12 or y.nunique() < 4:
        return pd.DataFrame()
    if model_name == "ridge":
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("reg", RidgeCV(alphas=np.logspace(-4, 4, 25))),
        ])
    elif model_name == "random_forest":
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=3,
                random_state=23,
            )),
        ])
    else:
        raise ValueError(model_name)
    rows = []
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(x, y, groups):
        if len(test_idx) < 2:
            continue
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        baseline = np.repeat(float(np.nanmedian(y.iloc[train_idx])), len(test_idx))
        rho = spearmanr(y.iloc[test_idx], pred).statistic if y.iloc[test_idx].nunique() > 1 else np.nan
        rows.append({
            "target": target,
            "model": model_name,
            "prefix_fraction": float(df["prefix_fraction"].iloc[0]),
            "heldout_event_reference_cycle": float(groups.iloc[test_idx].iloc[0]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "mae": float(mean_absolute_error(y.iloc[test_idx], pred)),
            "baseline_median_mae": float(mean_absolute_error(y.iloc[test_idx], baseline)),
            "mae_ratio_vs_baseline": float(mean_absolute_error(y.iloc[test_idx], pred) / max(mean_absolute_error(y.iloc[test_idx], baseline), 1e-12)),
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_cols = ["target", "model", "prefix_fraction"]
    rows = []
    for keys, grp in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["folds"] = int(len(grp))
        for col in metric_cols:
            if col in grp.columns:
                row[f"mean_{col}"] = float(pd.to_numeric(grp[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values([c for c in ["target", "mean_roc_auc", "mean_mae_ratio_vs_baseline"] if c in pd.DataFrame(rows).columns])


def permutation_null_auc(df: pd.DataFrame, target: str, features: List[str], prefix_fraction: float, observed_auc: float, n_perm: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    base = df[df["prefix_fraction"] == prefix_fraction].copy()
    y = pd.to_numeric(base[target], errors="coerce").astype(int).to_numpy()
    if len(np.unique(y)) < 2 or not np.isfinite(observed_auc):
        return {"target": target, "prefix_fraction": prefix_fraction, "status": "skipped"}
    null_aucs = []
    for _ in range(n_perm):
        perm = y.copy()
        rng.shuffle(perm)
        base["_perm_target"] = perm
        fold = logo_classification(base, "_perm_target", features, "logistic_l2")
        if not fold.empty:
            null_aucs.append(float(fold["roc_auc"].mean()))
    null = np.asarray(null_aucs, dtype=float)
    p = float((np.sum(null >= observed_auc) + 1) / (len(null) + 1)) if len(null) else np.nan
    return {
        "target": target,
        "prefix_fraction": float(prefix_fraction),
        "observed_mean_auc": float(observed_auc),
        "n_permutations": int(len(null)),
        "null_mean_auc": float(np.nanmean(null)) if len(null) else None,
        "null_p95_auc": float(np.nanpercentile(null, 95)) if len(null) else None,
        "empirical_p_ge_observed": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/prefix_roi_forecast")
    parser.add_argument("--n-permutation", type=int, default=200)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    echem = pd.read_csv(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_joined.csv")
    modes = pd.read_csv(derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv")
    residuals = pd.read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")
    keep_mode = [
        "roi_id",
        "mode_label",
        "mode_review_priority",
        "mode_pc1",
        "mode_pc2",
    ]
    keep_resid = [
        "roi_id",
        "phase_slope_positive_fraction_protocol_residual",
        "phase_slope_median_per_s_protocol_residual",
        "threshold_robust_phase_score_protocol_residual",
        "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    ]
    df = echem.merge(modes[[c for c in keep_mode if c in modes.columns]], on="roi_id", how="inner")
    df = df.merge(residuals[[c for c in keep_resid if c in residuals.columns]], on="roi_id", how="left")
    df["is_event_roi"] = (df["cohort_role"] == "event").astype(int)
    df["is_event_enriched_mode"] = (df["mode_label"] == EVENT_MODE).astype(int)
    df["front_positive_residual_binary"] = (pd.to_numeric(df["phase_slope_positive_fraction_protocol_residual"], errors="coerce") > 0).astype(int)

    prefix_df = build_prefix_table(df)
    prefix_feature_cols = [c for c in feature_columns(prefix_df) if (
        "frame_index" not in c
        and (
            c.startswith("prefix_")
            or c.endswith("_prefix_first")
            or "_prefix_" in c
            or c.startswith("stage_prefix")
            or c.startswith("first_q")
            or c.startswith("first_frame_mean")
            or c.startswith("last_prefix")
        )
    )]
    candidate_context_cols = [
        "cycleNo",
        "event_reference_cycle",
        "n_frames_percentile",
        "V_mean",
        "I_mean_mA",
        "I_abs_mean_mA",
        "cycles_from_block_start",
        "front_candidate_rank",
        "object_candidate_rank",
    ]
    context_cols = [
        c for c in candidate_context_cols
        if c in prefix_df.columns and pd.to_numeric(prefix_df[c], errors="coerce").notna().sum() >= 8
    ]
    feature_sets = {
        "prefix_only": prefix_feature_cols,
        "prefix_plus_context": list(dict.fromkeys(prefix_feature_cols + context_cols)),
    }
    class_targets = ["is_event_roi", "is_event_enriched_mode", "front_positive_residual_binary"]
    reg_targets = [
        "phase_slope_positive_fraction_protocol_residual",
        "threshold_robust_phase_score_protocol_residual",
        "mode_review_priority",
    ]

    class_rows = []
    reg_rows = []
    for frac in PREFIX_FRACTIONS:
        sub = prefix_df[prefix_df["prefix_fraction"] == frac].copy()
        for feature_set, cols in feature_sets.items():
            cols = [c for c in cols if c in sub.columns]
            for target in class_targets:
                for model_name in ["logistic_l2", "random_forest"]:
                    m = logo_classification(sub, target, cols, model_name)
                    if not m.empty:
                        m["feature_set"] = feature_set
                        class_rows.append(m)
            for target in reg_targets:
                for model_name in ["ridge", "random_forest"]:
                    m = logo_regression(sub, target, cols, model_name)
                    if not m.empty:
                        m["feature_set"] = feature_set
                        reg_rows.append(m)

    class_metrics = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()
    reg_metrics = pd.concat(reg_rows, ignore_index=True) if reg_rows else pd.DataFrame()
    class_summary = (
        class_metrics.groupby(["target", "feature_set", "model", "prefix_fraction"], dropna=False)
        .agg(folds=("roc_auc", "size"), mean_roc_auc=("roc_auc", "mean"), mean_balanced_accuracy=("balanced_accuracy", "mean"))
        .reset_index()
        .sort_values(["target", "mean_roc_auc"], ascending=[True, False])
    ) if not class_metrics.empty else pd.DataFrame()
    reg_summary = (
        reg_metrics.groupby(["target", "feature_set", "model", "prefix_fraction"], dropna=False)
        .agg(folds=("mae", "size"), mean_mae=("mae", "mean"), mean_baseline_median_mae=("baseline_median_mae", "mean"), mean_mae_ratio_vs_baseline=("mae_ratio_vs_baseline", "mean"), mean_spearman_rho=("spearman_rho", "mean"))
        .reset_index()
        .sort_values(["target", "mean_mae_ratio_vs_baseline"], ascending=[True, True])
    ) if not reg_metrics.empty else pd.DataFrame()

    null_rows = []
    if not class_summary.empty:
        for target in class_targets:
            best = class_summary[(class_summary["target"] == target) & (class_summary["feature_set"] == "prefix_only") & (class_summary["model"] == "logistic_l2")].head(1)
            if not best.empty:
                frac = float(best["prefix_fraction"].iloc[0])
                auc = float(best["mean_roc_auc"].iloc[0])
                null_rows.append(permutation_null_auc(prefix_df, target, prefix_feature_cols, frac, auc, args.n_permutation, 1000 + len(null_rows)))
    null_summary = pd.DataFrame(null_rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix_path = out / "prefix_roi_feature_table.csv"
    class_metrics_path = out / "prefix_roi_classification_folds.csv"
    class_summary_path = out / "prefix_roi_classification_summary.csv"
    reg_metrics_path = out / "prefix_roi_regression_folds.csv"
    reg_summary_path = out / "prefix_roi_regression_summary.csv"
    null_path = out / "prefix_roi_permutation_null.csv"
    prefix_df.to_csv(prefix_path, index=False)
    class_metrics.to_csv(class_metrics_path, index=False)
    class_summary.to_csv(class_summary_path, index=False)
    reg_metrics.to_csv(reg_metrics_path, index=False)
    reg_summary.to_csv(reg_summary_path, index=False)
    null_summary.to_csv(null_path, index=False)

    summary = {
        "n_roi": int(df["roi_id"].nunique()),
        "n_prefix_rows": int(len(prefix_df)),
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "n_prefix_features": int(len(prefix_feature_cols)),
        "class_targets": class_targets,
        "regression_targets": reg_targets,
        "top_classification": class_summary.head(12).to_dict("records") if not class_summary.empty else [],
        "top_regression": reg_summary.head(12).to_dict("records") if not reg_summary.empty else [],
        "permutation_null": null_summary.to_dict("records") if not null_summary.empty else [],
        "guardrail": "Prefix-only ROI forecasts use cropped particle-region sequences, but the cohort is small and selected around event-reference cycles; treat results as physics-signal triage, not a deployable predictor.",
        "outputs": {
            "prefix_features": str(prefix_path),
            "classification_folds": str(class_metrics_path),
            "classification_summary": str(class_summary_path),
            "regression_folds": str(reg_metrics_path),
            "regression_summary": str(reg_summary_path),
            "permutation_null": str(null_path),
            "summary": str(out / "prefix_roi_forecast_summary.json"),
        },
    }
    with (out / "prefix_roi_forecast_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Prefix ROI Forecast",
        "",
        "Prefix-only forecasts from early cropped particle ROI frames.",
        "",
        f"- ROI rows: {summary['n_roi']}",
        f"- Prefix rows: {summary['n_prefix_rows']}",
        f"- Prefix features: {summary['n_prefix_features']}",
        "",
        "## Top Classification",
    ]
    for row in summary["top_classification"][:8]:
        lines.append(
            f"- {row.get('target')} {row.get('feature_set')} {row.get('model')} f={row.get('prefix_fraction')}: AUC={row.get('mean_roc_auc'):.3f}, bal_acc={row.get('mean_balanced_accuracy'):.3f}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out), "n_roi": summary["n_roi"], "top_classification": summary["top_classification"][:3]}, indent=2))


if __name__ == "__main__":
    main()
