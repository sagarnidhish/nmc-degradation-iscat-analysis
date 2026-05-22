#!/usr/bin/env python3
"""Grouped readouts on source-normalized residual dictionary features.

The residual dictionary scalar audit found source-residual reconstruction-error
drift as a source-robust weak-label candidate. This script asks whether those
source-normalized descriptors improve grouped predictive readouts under
leave-cycle and leave-source splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
TRANSFORMS = ["raw", "source_residual", "within_source_z", "within_source_rank"]


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


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 16) -> List[str]:
    keep = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = numeric(df, col)
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def residual_dictionary_columns(df: pd.DataFrame) -> List[str]:
    return available_numeric(
        df,
        [c for c in df.columns if c.startswith("resdict_") or c.startswith("dictionary_") or c.startswith("residual_energy_")],
    )


def mask_front_columns(df: pd.DataFrame) -> List[str]:
    wanted = [
        "mask_base_area_fraction", "mask_area_fraction_median", "mask_area_fraction_iqr", "mask_area_fraction_slope",
        "mask_centroid_path_px", "mask_centroid_max_step_px", "mask_centroid_drift_px",
        "masked_minus_background_mean_median", "masked_minus_background_mean_slope",
        "front_radius_q60_median_px", "front_radius_q60_delta_px", "front_radius_q60_slope_px_per_norm_time",
        "front_radius_q70_median_px", "front_radius_q70_delta_px", "front_radius_q70_slope_px_per_norm_time",
        "front_radius_q80_median_px", "front_radius_q80_delta_px", "front_radius_q80_slope_px_per_norm_time",
        "front_radius_q70_positive_step_fraction",
        "front_gradient_peak_radius_median_px", "front_gradient_peak_radius_slope_px_per_norm_time",
        "apparent_diffusion_q70_px2_per_norm_time", "apparent_diffusion_q70_um2_per_norm_time",
        "roi_norm_mean_delta_last_minus_first",
    ]
    return available_numeric(df, wanted)


def source_transform(values: pd.Series, sources: pd.Series, transform: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    src = sources.astype(str)
    means = x.groupby(src).transform("mean")
    if transform == "raw":
        return x
    if transform == "source_residual":
        return x - means
    if transform == "within_source_z":
        stds = x.groupby(src).transform("std").replace(0, np.nan)
        return (x - means) / stds
    if transform == "within_source_rank":
        return x.groupby(src).rank(pct=True) - 0.5
    raise ValueError(transform)


def append_transforms(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    sources = df["source_stem"].astype(str)
    transformed: Dict[str, pd.Series] = {}
    for col in cols:
        for transform in TRANSFORMS:
            new_col = col if transform == "raw" else f"{col}__{transform}"
            transformed[new_col] = source_transform(df[col], sources, transform)
    existing = df.drop(columns=[c for c in transformed if c in df.columns], errors="ignore")
    return pd.concat([existing, pd.DataFrame(transformed, index=df.index)], axis=1)


def class_model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.2, solver="liblinear", random_state=seed),
    )


def grouped_predictions(df: pd.DataFrame, features: List[str], target: str, group_col: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = numeric(df, target)
    valid = y.isin([0, 1]) & df[group_col].notna()
    for group in sorted(df.loc[valid, group_col].dropna().unique()):
        test = valid & (df[group_col] == group)
        train = valid & ~test
        meta_cols = ["roi_id", "cycleNo", "source_stem", target]
        meta = df.loc[test, meta_cols].rename(columns={target: "observed"}).copy()
        meta[group_col] = group
        if train.sum() < 16 or y[train].nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
        else:
            model = class_model(seed)
            model.fit(df.loc[train, features], y[train].astype(int))
            meta["predicted_probability"] = model.predict_proba(df.loc[test, features])[:, 1]
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    out = pd.DataFrame(rows)
    out["target"] = target
    out["group_col"] = group_col
    return out


def prediction_metrics(pred: pd.DataFrame, feature_set: str, transform: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    group_col = pred["group_col"].iloc[0] if len(pred) else ""
    row: Dict[str, Any] = {
        "feature_set": feature_set,
        "transform": transform,
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
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def build_feature_sets(residual_cols: List[str], mask_cols: List[str]) -> Dict[str, Dict[str, List[str] | str]]:
    def suffixed(cols: List[str], transform: str) -> List[str]:
        return cols if transform == "raw" else [f"{c}__{transform}" for c in cols]

    sets: Dict[str, Dict[str, List[str] | str]] = {}
    for transform in TRANSFORMS:
        sets[f"residual_dictionary_{transform}"] = {"transform": transform, "features": suffixed(residual_cols, transform)}
    for transform in ["source_residual", "within_source_rank"]:
        sets[f"mask_front_{transform}"] = {"transform": transform, "features": suffixed(mask_cols, transform)}
        sets[f"residual_dictionary_plus_mask_front_{transform}"] = {
            "transform": transform,
            "features": suffixed(residual_cols + mask_cols, transform),
        }
    if "dictionary_recon_error_last_minus_first" in residual_cols:
        sets["dictionary_recon_error_last_minus_first_source_residual"] = {
            "transform": "source_residual",
            "features": ["dictionary_recon_error_last_minus_first__source_residual"],
        }
    return sets


def permutation_null(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    group_col: str,
    feature_set: str,
    transform: str,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    if permutations <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    valid = numeric(df, target).isin([0, 1])
    y = numeric(df, target).copy()
    rows = []
    for i in range(permutations):
        work = df.copy()
        work.loc[valid, target] = rng.permutation(y[valid].astype(int).to_numpy())
        pred = grouped_predictions(work, features, target, group_col, seed + i + 1)
        metric = prediction_metrics(pred, feature_set, transform)
        metric["permutation"] = i
        rows.append(metric)
    return pd.DataFrame(rows)


def summarize_permutation(observed: Dict[str, Any], null_df: pd.DataFrame) -> Dict[str, Any]:
    if null_df.empty or observed.get("roc_auc") is None or pd.isna(observed.get("roc_auc")):
        return {}
    null_auc = pd.to_numeric(null_df["roc_auc"], errors="coerce").dropna()
    null_ap = pd.to_numeric(null_df["average_precision"], errors="coerce").dropna()
    obs_auc = float(observed["roc_auc"])
    obs_ap = float(observed["average_precision"])
    return {
        "feature_set": observed.get("feature_set"),
        "transform": observed.get("transform"),
        "target": observed.get("target"),
        "group_col": observed.get("group_col"),
        "observed_roc_auc": obs_auc,
        "observed_average_precision": obs_ap,
        "n_permutations": int(len(null_df)),
        "null_roc_auc_mean": float(null_auc.mean()) if len(null_auc) else np.nan,
        "null_roc_auc_p95": float(null_auc.quantile(0.95)) if len(null_auc) else np.nan,
        "null_average_precision_mean": float(null_ap.mean()) if len(null_ap) else np.nan,
        "null_average_precision_p95": float(null_ap.quantile(0.95)) if len(null_ap) else np.nan,
        "empirical_p_roc_auc": float((1 + (null_auc >= obs_auc).sum()) / (1 + len(null_auc))) if len(null_auc) else np.nan,
        "empirical_p_average_precision": float((1 + (null_ap >= obs_ap).sum()) / (1 + len(null_ap))) if len(null_ap) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_dictionary_audit")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_dictionary_normalized_readout")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_dir / "source_balanced_residual_dictionary_features.csv")
    residual_cols = residual_dictionary_columns(df)
    mask_cols = mask_front_columns(df)
    transformed = append_transforms(df, residual_cols + mask_cols)
    feature_sets = build_feature_sets(residual_cols, mask_cols)

    pred_rows: List[pd.DataFrame] = []
    metrics: List[Dict[str, Any]] = []
    for feature_set, spec in feature_sets.items():
        cols = [c for c in spec["features"] if c in transformed.columns]  # type: ignore[index]
        if not cols:
            continue
        transform = str(spec["transform"])
        for target in [t for t in TARGETS if t in transformed.columns]:
            for group_col in ["cycleNo", "source_stem"]:
                pred = grouped_predictions(transformed, cols, target, group_col, args.seed)
                pred["feature_set"] = feature_set
                pred["transform"] = transform
                pred_rows.append(pred)
                metrics.append(prediction_metrics(pred, feature_set, transform))

    predictions = pd.concat(pred_rows, ignore_index=True, sort=False) if pred_rows else pd.DataFrame()
    metric_df = pd.DataFrame(metrics)
    if not metric_df.empty:
        metric_df = metric_df.sort_values(
            ["target", "group_col", "roc_auc", "average_precision"],
            ascending=[True, True, False, False],
        )

    future16_source = metric_df[
        (metric_df["target"] == "future_any_drop_within_16cycles") &
        (metric_df["group_col"] == "source_stem")
    ].copy() if not metric_df.empty else pd.DataFrame()
    best_future16_source = future16_source.sort_values(["roc_auc", "average_precision"], ascending=False).iloc[0].to_dict() if not future16_source.empty else {}
    perm_df = pd.DataFrame()
    permutation_summary: Dict[str, Any] = {}
    if best_future16_source and args.permutations > 0:
        spec = feature_sets[str(best_future16_source["feature_set"])]
        cols = [c for c in spec["features"] if c in transformed.columns]  # type: ignore[index]
        perm_df = permutation_null(
            transformed,
            cols,
            str(best_future16_source["target"]),
            str(best_future16_source["group_col"]),
            str(best_future16_source["feature_set"]),
            str(best_future16_source["transform"]),
            args.permutations,
            args.seed,
        )
        permutation_summary = summarize_permutation(best_future16_source, perm_df)

    paths = {
        "metrics": out / "source_balanced_residual_dictionary_normalized_readout_metrics.csv",
        "predictions": out / "source_balanced_residual_dictionary_normalized_readout_predictions.csv",
        "permutation": out / "source_balanced_residual_dictionary_normalized_readout_permutation.csv",
        "summary": out / "source_balanced_residual_dictionary_normalized_readout_summary.json",
    }
    metric_df.to_csv(paths["metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    perm_df.to_csv(paths["permutation"], index=False)

    def pick(target: str, group_col: str, feature_set: str) -> Dict[str, Any]:
        sub = metric_df[
            (metric_df["target"] == target) &
            (metric_df["group_col"] == group_col) &
            (metric_df["feature_set"] == feature_set)
        ]
        return sub.iloc[0].to_dict() if len(sub) else {}

    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()) if "cycleNo" in df.columns else None,
        "n_sources": int(df["source_stem"].nunique()) if "source_stem" in df.columns else None,
        "future8_positive_rows": int(numeric(df, "future_any_drop_within_8cycles").sum()) if "future_any_drop_within_8cycles" in df.columns else None,
        "future16_positive_rows": int(numeric(df, "future_any_drop_within_16cycles").sum()) if "future_any_drop_within_16cycles" in df.columns else None,
        "residual_dictionary_feature_count": int(len(residual_cols)),
        "mask_front_feature_count": int(len(mask_cols)),
        "feature_set_sizes": {k: len(v["features"]) for k, v in feature_sets.items()},
        "top_metrics": metric_df.head(32).to_dict(orient="records") if not metric_df.empty else [],
        "future16_leave_source_best": best_future16_source,
        "future16_leave_source_raw_residual_dictionary": pick("future_any_drop_within_16cycles", "source_stem", "residual_dictionary_raw"),
        "future16_leave_source_source_residual_residual_dictionary": pick("future_any_drop_within_16cycles", "source_stem", "residual_dictionary_source_residual"),
        "future16_leave_source_within_source_rank_residual_dictionary": pick("future_any_drop_within_16cycles", "source_stem", "residual_dictionary_within_source_rank"),
        "future16_leave_cycle_source_residual_residual_dictionary": pick("future_any_drop_within_16cycles", "cycleNo", "residual_dictionary_source_residual"),
        "future16_leave_cycle_within_source_rank_residual_dictionary": pick("future_any_drop_within_16cycles", "cycleNo", "residual_dictionary_within_source_rank"),
        "permutation_summary": permutation_summary,
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Source transforms are unsupervised within-source normalizations computed from ROI feature distributions, including held-out source rows without labels. This tests source-normalized readout stability, not a deployable source-transfer warning model.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Source-Balanced Residual Dictionary Normalized Readout",
        "",
        f"Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"Residual dictionary features: {summary['residual_dictionary_feature_count']}; mask/front features: {summary['mask_front_feature_count']}",
        "",
        "## Future16 Leave-Source Readouts",
    ]
    for key in [
        "future16_leave_source_raw_residual_dictionary",
        "future16_leave_source_source_residual_residual_dictionary",
        "future16_leave_source_within_source_rank_residual_dictionary",
        "future16_leave_source_best",
    ]:
        row = summary.get(key, {}) or {}
        readme.append(
            f"- {key}: {row.get('feature_set', 'NA')} AUC={row.get('roc_auc', np.nan):.3f}, "
            f"AP={row.get('average_precision', np.nan):.3f}, n={row.get('n_eval', 0)}"
        )
    if permutation_summary:
        readme.extend([
            "",
            "## Permutation Null",
            f"- {permutation_summary.get('feature_set')} observed source-heldout future16 AUC={permutation_summary.get('observed_roc_auc'):.3f}; "
            f"null p95={permutation_summary.get('null_roc_auc_p95'):.3f}; empirical p={permutation_summary.get('empirical_p_roc_auc'):.3f}",
        ])
    readme.extend(["", "## Guardrail", summary["guardrail"], ""])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()
