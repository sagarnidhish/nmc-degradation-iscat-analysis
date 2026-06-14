#!/usr/bin/env python3
"""Timebase-aware audit for source-balanced ROI rollout features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
ROLLOUT_FEATURES = [
    "roi_norm_mean_delta_last_minus_first",
    "roi_norm_mean_late_minus_early",
    "roi_norm_mean_positive_step_fraction",
    "raw_roi_mean_delta_last_minus_first",
    "temporal_energy_mean",
    "temporal_energy_p95",
    "temporal_energy_late_minus_early",
    "persistence_mse_mean",
    "persistence_mse_p95",
    "persistence_mse_late_mean",
    "velocity_mse_mean",
    "velocity_mse_p95",
    "velocity_minus_persistence_mse",
]
ACQUISITION_FEATURES = [
    "stage_drift_xy_recomputed",
    "object_area_ds_px",
    "object_mean_residual",
    "object_mean_abs_z",
]
TIMEBASE_FEATURES = [
    "h5_dt_median_s",
    "h5_dt_min_s",
    "h5_dt_max_s",
    "h5_dt_max_to_median_ratio",
    "roi_elapsed_to_h5_median_ratio",
    "roi_elapsed_to_h5_max_abs_error",
    "strict_timebase_fraction",
    "roi_h5_elapsed_aligned_fraction",
]


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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def orient_metrics(y: pd.Series, x: pd.Series) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    out: Dict[str, Any] = {
        "n": int(valid.sum()),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "direction": "NA",
        "oriented_auc": np.nan,
        "average_precision": np.nan,
        "mwu_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "median_positive": np.nan,
        "median_negative": np.nan,
        "median_positive_minus_negative": np.nan,
    }
    if len(yy) >= 8 and yy.nunique() == 2 and xx.nunique() > 1:
        direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
        score = xx if direction == "higher_in_positive" else -xx
        out["direction"] = direction
        out["oriented_auc"] = float(roc_auc_score(yy, score))
        out["average_precision"] = float(average_precision_score(yy, score))
        pos = xx[yy == 1]
        neg = xx[yy == 0]
        try:
            _, p_mwu = mannwhitneyu(pos, neg, alternative="two-sided")
            out["mwu_p"] = float(p_mwu)
        except ValueError:
            pass
        rho, sp = spearmanr(yy, score)
        out["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
        out["spearman_p"] = float(sp) if np.isfinite(sp) else np.nan
        out["median_positive"] = float(pos.median())
        out["median_negative"] = float(neg.median())
        out["median_positive_minus_negative"] = out["median_positive"] - out["median_negative"]
    return out


def source_transform(values: pd.Series, sources: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    src = sources.astype(str)
    if mode == "raw":
        return x
    means = x.groupby(src).transform("mean")
    if mode == "source_residual":
        return x - means
    if mode == "within_source_rank":
        return x.groupby(src).rank(pct=True) - 0.5
    raise ValueError(mode)


def feature_target_tests(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario, mask in scenario_masks(df).items():
        sub = df[mask].copy()
        for target in [t for t in TARGETS if t in sub.columns]:
            y = numeric(sub, target)
            for feature in features:
                if feature not in sub.columns:
                    continue
                for transform in ["raw", "source_residual", "within_source_rank"]:
                    if transform != "raw" and sub["source_stem"].nunique() < 2:
                        continue
                    x = source_transform(numeric(sub, feature), sub["source_stem"], transform)
                    rows.append({
                        "scenario": scenario,
                        "target": target,
                        "feature": feature,
                        "transform": transform,
                        **orient_metrics(y, x),
                    })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["target", "scenario", "oriented_auc", "average_precision"],
            ascending=[True, True, False, False],
        )
    return out


def scenario_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    known = df["timebase_class"].isin(["strict", "pause_heavy"])
    return {
        "all_rows": pd.Series(True, index=df.index),
        "timebase_known": known,
        "strict_timebase_sources": df["timebase_class"].eq("strict"),
        "pause_heavy_sources": df["timebase_class"].eq("pause_heavy"),
        "timebase_unknown_sources": df["timebase_class"].eq("unknown"),
    }


def feature_timebase_correlations(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature in features:
        if feature not in df.columns:
            continue
        for tb in ["h5_dt_max_to_median_ratio", "roi_elapsed_to_h5_max_abs_error", "strict_timebase_fraction"]:
            if tb not in df.columns:
                continue
            valid = numeric(df, feature).notna() & numeric(df, tb).notna()
            if valid.sum() >= 8 and numeric(df.loc[valid], feature).nunique() > 1 and numeric(df.loc[valid], tb).nunique() > 1:
                rho, p = spearmanr(numeric(df.loc[valid], tb), numeric(df.loc[valid], feature))
                rows.append({
                    "feature": feature,
                    "timebase_feature": tb,
                    "n": int(valid.sum()),
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "abs_rho": float(abs(rho)) if np.isfinite(rho) else np.nan,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_rho", "p_value"], ascending=[False, True])
    return out


def timebase_class_tests(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    known = df[df["timebase_class"].isin(["strict", "pause_heavy"])].copy()
    if known.empty:
        return pd.DataFrame()
    y = known["timebase_class"].eq("pause_heavy").astype(int)
    for feature in features:
        if feature not in known.columns:
            continue
        rows.append({"feature": feature, **orient_metrics(y, numeric(known, feature))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["oriented_auc", "average_precision"], ascending=[False, False])
    return out


def make_model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, solver="liblinear", class_weight="balanced", C=0.25, random_state=seed),
    )


def grouped_predictions(df: pd.DataFrame, cols: List[str], target: str, group_col: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = numeric(df, target)
    valid = y.isin([0, 1]) & df[group_col].notna()
    for group in sorted(df.loc[valid, group_col].dropna().unique()):
        test = valid & df[group_col].eq(group)
        train = valid & ~test
        meta = df.loc[test, ["roi_id", "cycleNo", "source_stem", "timebase_class", target]].rename(columns={target: "observed"}).copy()
        meta[group_col] = group
        if train.sum() < 16 or y[train].nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
        else:
            clf = make_model(seed)
            clf.fit(df.loc[train, cols], y[train].astype(int))
            meta["predicted_probability"] = clf.predict_proba(df.loc[test, cols])[:, 1]
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    out = pd.DataFrame(rows)
    out["target"] = target
    out["group_col"] = group_col
    return out


def prediction_metrics(pred: pd.DataFrame, feature_set: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int) if len(tmp) else pd.Series(dtype=int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce") if len(tmp) else pd.Series(dtype=float)
    group_col = pred["group_col"].iloc[0] if len(pred) else ""
    out: Dict[str, Any] = {
        "feature_set": feature_set,
        "target": pred["target"].iloc[0] if len(pred) else "",
        "group_col": group_col,
        "n_eval": int(len(tmp)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "n_groups": int(tmp[group_col].nunique()) if len(tmp) and group_col in tmp else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2 and p.nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
        rho, sp = spearmanr(y, p)
        out["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
        out["spearman_p"] = float(sp) if np.isfinite(sp) else np.nan
    return out


def model_audit(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rollout_cols = [c for c in ROLLOUT_FEATURES if c in df.columns]
    acq_cols = [c for c in ACQUISITION_FEATURES if c in df.columns]
    tb_cols = [c for c in TIMEBASE_FEATURES if c in df.columns]
    feature_sets = {
        "acquisition_context": acq_cols,
        "timebase_context": tb_cols,
        "rollout_only": rollout_cols,
        "rollout_plus_acquisition": rollout_cols + acq_cols,
        "rollout_plus_timebase": rollout_cols + tb_cols,
        "rollout_plus_acquisition_timebase": rollout_cols + acq_cols + tb_cols,
    }
    preds: List[pd.DataFrame] = []
    metrics: List[Dict[str, Any]] = []
    for name, cols in feature_sets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        for target in [t for t in TARGETS if t in df.columns]:
            for group_col in ["cycleNo", "source_stem"]:
                pred = grouped_predictions(df, cols, target, group_col, seed)
                pred["feature_set"] = name
                preds.append(pred)
                metrics.append(prediction_metrics(pred, name))
    pred_df = pd.concat(preds, ignore_index=True, sort=False) if preds else pd.DataFrame()
    metric_df = pd.DataFrame(metrics)
    deltas: List[Dict[str, Any]] = []
    for _, row in metric_df.iterrows():
        base = metric_df[
            (metric_df["target"] == row["target"])
            & (metric_df["group_col"] == row["group_col"])
            & (metric_df["feature_set"] == "acquisition_context")
        ]
        if len(base) and row["feature_set"] != "acquisition_context":
            b = base.iloc[0]
            rec = row.to_dict()
            rec["base_roc_auc"] = b.get("roc_auc")
            rec["base_average_precision"] = b.get("average_precision")
            rec["delta_roc_auc_vs_acquisition"] = row.get("roc_auc") - b.get("roc_auc") if pd.notna(row.get("roc_auc")) and pd.notna(b.get("roc_auc")) else np.nan
            rec["delta_average_precision_vs_acquisition"] = row.get("average_precision") - b.get("average_precision") if pd.notna(row.get("average_precision")) and pd.notna(b.get("average_precision")) else np.nan
            deltas.append(rec)
    delta_df = pd.DataFrame(deltas)
    if not metric_df.empty:
        metric_df = metric_df.sort_values(["target", "group_col", "roc_auc"], ascending=[True, True, False])
    if not delta_df.empty:
        delta_df = delta_df.sort_values(["target", "group_col", "delta_roc_auc_vs_acquisition"], ascending=[True, True, False])
    return pred_df, metric_df, delta_df


def load_joined(rollout_dir: Path, timebase_dir: Path) -> pd.DataFrame:
    rollout = pd.read_csv(rollout_dir / "source_balanced_sequence_rollout_features.csv")
    source = pd.read_csv(timebase_dir / "hdf5_timebase_source_summary.csv")
    keep = ["source_stem"] + [c for c in TIMEBASE_FEATURES + ["source_strict_timebase"] if c in source.columns]
    joined = rollout.merge(source[keep], on="source_stem", how="left", suffixes=("", "_h5"))
    strict = joined.get("source_strict_timebase", pd.Series(False, index=joined.index)).astype("boolean")
    joined["timebase_class"] = np.where(
        joined["h5_dt_max_to_median_ratio"].isna(),
        "unknown",
        np.where(strict.fillna(False), "strict", "pause_heavy"),
    )
    return joined


def verdict(summary: Dict[str, Any]) -> str:
    strict_rows = summary.get("scenario_counts", {}).get("strict_timebase_sources", {}).get("n_rows", 0)
    pause_rows = summary.get("scenario_counts", {}).get("pause_heavy_sources", {}).get("n_rows", 0)
    best_strict = summary.get("best_strict_timebase_target_auc")
    best_pause = summary.get("best_pause_heavy_target_auc")
    source_delta = summary.get("best_source_heldout_timebase_delta_auc")
    if strict_rows >= 16 and pause_rows >= 16 and best_strict is not None and best_pause is not None:
        if best_pause - best_strict > 0.10:
            return "rollout_signal_timebase_sensitive_pause_heavy_enriched"
    if source_delta is not None and source_delta > 0.05:
        return "timebase_context_adds_source_heldout_signal"
    return "timebase_is_guardrail_not_validated_predictor"


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Timebase-Aware Source-Balanced Rollout Audit",
        "",
        "Joins HDF5 timebase quality onto source-balanced particle-region rollout features.",
        "",
        f"- ROI rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Timebase classes: {summary['timebase_class_counts']}",
        f"- Verdict: {summary['verdict']}",
        "",
        "## Key Results",
        "",
        f"- Best strict-timebase target AUC: {summary.get('best_strict_timebase_target_auc')}",
        f"- Best pause-heavy target AUC: {summary.get('best_pause_heavy_target_auc')}",
        f"- Best source-heldout timebase delta AUC: {summary.get('best_source_heldout_timebase_delta_auc')}",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_sequence_rollout_audit")
    parser.add_argument("--timebase-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/hdf5_timebase_provenance_audit")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/timebase_aware_source_balanced_rollout_audit")
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_joined(Path(args.rollout_dir), Path(args.timebase_dir))
    features = [c for c in ROLLOUT_FEATURES + ACQUISITION_FEATURES if c in df.columns]

    target_tests = feature_target_tests(df, features)
    timebase_corr = feature_timebase_correlations(df[df["timebase_class"].isin(["strict", "pause_heavy"])], features)
    class_tests = timebase_class_tests(df, features)
    predictions, model_metrics, model_deltas = model_audit(df, args.seed)

    scenario_counts = {}
    for name, mask in scenario_masks(df).items():
        sub = df[mask]
        scenario_counts[name] = {
            "n_rows": int(len(sub)),
            "n_cycles": int(pd.to_numeric(sub["cycleNo"], errors="coerce").nunique()),
            "n_sources": int(sub["source_stem"].nunique()),
            "future8_positive": int(numeric(sub, "future_any_drop_within_8cycles").sum()) if "future_any_drop_within_8cycles" in sub else 0,
            "future16_positive": int(numeric(sub, "future_any_drop_within_16cycles").sum()) if "future_any_drop_within_16cycles" in sub else 0,
        }

    strict_tests = target_tests[target_tests["scenario"].eq("strict_timebase_sources")]
    pause_tests = target_tests[target_tests["scenario"].eq("pause_heavy_sources")]
    source_deltas = model_deltas[model_deltas["group_col"].eq("source_stem")] if not model_deltas.empty else pd.DataFrame()
    best_source_delta = pd.to_numeric(source_deltas.get("delta_roc_auc_vs_acquisition", pd.Series(dtype=float)), errors="coerce").max() if not source_deltas.empty else np.nan

    paths = {
        "joined": out / "timebase_aware_rollout_joined.csv",
        "target_tests": out / "timebase_aware_rollout_target_tests.csv",
        "timebase_correlations": out / "timebase_aware_rollout_timebase_correlations.csv",
        "timebase_class_tests": out / "timebase_aware_rollout_class_tests.csv",
        "model_metrics": out / "timebase_aware_rollout_model_metrics.csv",
        "model_deltas": out / "timebase_aware_rollout_model_deltas.csv",
        "predictions": out / "timebase_aware_rollout_predictions.csv",
        "summary": out / "timebase_aware_rollout_summary.json",
        "readme": out / "README.md",
    }
    df.to_csv(paths["joined"], index=False)
    target_tests.to_csv(paths["target_tests"], index=False)
    timebase_corr.to_csv(paths["timebase_correlations"], index=False)
    class_tests.to_csv(paths["timebase_class_tests"], index=False)
    model_metrics.to_csv(paths["model_metrics"], index=False)
    model_deltas.to_csv(paths["model_deltas"], index=False)
    predictions.to_csv(paths["predictions"], index=False)

    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cycles": int(pd.to_numeric(df["cycleNo"], errors="coerce").nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "timebase_class_counts": {str(k): int(v) for k, v in df["timebase_class"].value_counts().to_dict().items()},
        "scenario_counts": scenario_counts,
        "best_strict_timebase_target_auc": float(pd.to_numeric(strict_tests["oriented_auc"], errors="coerce").max()) if len(strict_tests) else None,
        "best_pause_heavy_target_auc": float(pd.to_numeric(pause_tests["oriented_auc"], errors="coerce").max()) if len(pause_tests) else None,
        "best_source_heldout_timebase_delta_auc": float(best_source_delta) if pd.notna(best_source_delta) else None,
        "top_timebase_correlations": timebase_corr.head(20).to_dict("records") if not timebase_corr.empty else [],
        "top_timebase_class_features": class_tests.head(20).to_dict("records") if not class_tests.empty else [],
        "top_target_tests": target_tests.head(40).to_dict("records") if not target_tests.empty else [],
        "top_model_metrics": model_metrics.head(40).to_dict("records") if not model_metrics.empty else [],
        "top_model_deltas": model_deltas.head(40).to_dict("records") if not model_deltas.empty else [],
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "This audit tests whether source-balanced ROI rollout signals are stable under HDF5 timebase quality classes. It uses automatic ROI crops and source-level timing provenance, not manual particle labels, prospective validation, or calibrated diffusion.",
    }
    summary["verdict"] = verdict(summary)
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_readme(out, clean_json(summary))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
