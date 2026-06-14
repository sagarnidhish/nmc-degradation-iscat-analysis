#!/usr/bin/env python3
"""Source/cycle-control audit for source-balanced ROI rollout features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
PRIMARY_FEATURES = [
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
CONTEXT_FEATURES = [
    "stage_drift_xy_recomputed",
    "object_area_ds_px",
    "object_mean_residual",
    "object_mean_abs_z",
    "n_loaded_frames",
    "frame_height",
    "frame_width",
]
TRANSFORMS = ["raw", "source_residual", "within_source_rank", "within_source_z"]


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


def source_eta2(series: pd.Series, sources: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.notna() & sources.notna()
    vals = vals[valid]
    src = sources[valid].astype(str)
    if vals.nunique() < 2 or src.nunique() < 2:
        return np.nan
    total = float(((vals - vals.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in vals.groupby(src):
        between += len(sub) * float((sub.mean() - vals.mean()) ** 2)
    return between / total


def source_transform(values: pd.Series, sources: pd.Series, transform: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    src = sources.astype(str)
    means = x.groupby(src).transform("mean")
    if transform == "raw":
        return x
    if transform == "source_residual":
        return x - means
    if transform == "within_source_rank":
        return x.groupby(src).rank(pct=True) - 0.5
    if transform == "within_source_z":
        stds = x.groupby(src).transform("std").replace(0, np.nan)
        return (x - means) / stds
    raise ValueError(transform)


def oriented_scores(y: pd.Series, x: pd.Series, direction: str | None = None) -> Tuple[Dict[str, Any], pd.Series]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    out: Dict[str, Any] = {
        "n": int(valid.sum()),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "direction": direction or "NA",
        "oriented_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "median_positive": np.nan,
        "median_negative": np.nan,
        "median_positive_minus_negative": np.nan,
    }
    score = pd.Series(np.nan, index=x.index, dtype=float)
    if len(yy) >= 8 and yy.nunique() == 2 and xx.nunique() > 1:
        if direction is None:
            direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
        oriented = xx if direction == "higher_in_positive" else -xx
        score.loc[valid] = oriented
        out["direction"] = direction
        out["oriented_auc"] = float(roc_auc_score(yy, oriented))
        out["average_precision"] = float(average_precision_score(yy, oriented))
        rho, sp = spearmanr(yy, oriented)
        out["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
        out["spearman_p"] = float(sp) if np.isfinite(sp) else np.nan
        out["median_positive"] = float(xx[yy == 1].median())
        out["median_negative"] = float(xx[yy == 0].median())
        out["median_positive_minus_negative"] = out["median_positive"] - out["median_negative"]
    return out, score


def stratified_label_permutation_p(
    y: pd.Series,
    score: pd.Series,
    strata: pd.Series,
    observed_auc: float,
    seed: int,
    n_perm: int,
) -> Tuple[float, float, float]:
    valid = y.isin([0, 1]) & score.notna() & strata.notna()
    yy = y[valid].astype(int).to_numpy()
    ss = score[valid].to_numpy(dtype=float)
    groups = strata[valid].astype(str).to_numpy()
    if len(yy) < 8 or len(np.unique(yy)) < 2 or not np.isfinite(observed_auc):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = yy.copy()
        for group in np.unique(groups):
            idx = np.flatnonzero(groups == group)
            if len(idx) > 1:
                yp[idx] = rng.permutation(yp[idx])
        if len(np.unique(yp)) == 2:
            null.append(roc_auc_score(yp, ss))
    if not null:
        return np.nan, np.nan, np.nan
    arr = np.asarray(null, dtype=float)
    p_ge = (float(np.sum(arr >= observed_auc)) + 1.0) / (len(arr) + 1.0)
    return p_ge, float(np.mean(arr)), float(np.quantile(arr, 0.95))


def scalar_tests(df: pd.DataFrame, features: Iterable[str], seed: int, n_perm: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    cycles = numeric(df, "cycleNo").astype(str)
    for target in [t for t in TARGETS if t in df.columns]:
        y = numeric(df, target)
        for feature in features:
            if feature not in df.columns:
                continue
            for transform in TRANSFORMS:
                x = source_transform(numeric(df, feature), sources, transform)
                met, score = oriented_scores(y, x)
                src_p, src_mean, src_p95 = stratified_label_permutation_p(
                    y, score, sources, met["oriented_auc"], seed, n_perm
                )
                cyc_p, cyc_mean, cyc_p95 = stratified_label_permutation_p(
                    y, score, cycles, met["oriented_auc"], seed + 17, n_perm
                )
                rows.append({
                    "target": target,
                    "feature": feature,
                    "transform": transform,
                    **met,
                    "source_eta2_raw_feature": source_eta2(numeric(df, feature), sources),
                    "source_stratified_auc_p": src_p,
                    "source_stratified_null_auc_mean": src_mean,
                    "source_stratified_null_auc_p95": src_p95,
                    "cycle_stratified_auc_p": cyc_p,
                    "cycle_stratified_null_auc_mean": cyc_mean,
                    "cycle_stratified_null_auc_p95": cyc_p95,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["target", "transform", "oriented_auc", "average_precision"],
            ascending=[True, True, False, False],
        )
    return out


def add_transforms(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    sources = out["source_stem"].astype(str)
    for feature in features:
        if feature not in out.columns:
            continue
        for transform in ["source_residual", "within_source_rank", "within_source_z"]:
            out[f"{feature}__{transform}"] = source_transform(out[feature], sources, transform)
    return out


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    primary = [c for c in PRIMARY_FEATURES if c in df.columns]
    context = [c for c in CONTEXT_FEATURES if c in df.columns]
    residual = [f"{c}__source_residual" for c in primary if f"{c}__source_residual" in df.columns]
    rank = [f"{c}__within_source_rank" for c in primary if f"{c}__within_source_rank" in df.columns]
    zed = [f"{c}__within_source_z" for c in primary if f"{c}__within_source_z" in df.columns]
    return {
        "context_only": context,
        "rollout_raw": primary,
        "rollout_raw_plus_context": primary + context,
        "rollout_source_residual": residual,
        "rollout_within_source_rank": rank,
        "rollout_within_source_z": zed,
        "residual_plus_context": residual + context,
        "rank_plus_context": rank + context,
    }


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
        meta_cols = ["roi_id", "cycleNo", "source_stem", target]
        meta = df.loc[test, [c for c in meta_cols if c in df.columns]].rename(columns={target: "observed"}).copy()
        meta[group_col] = group
        if train.sum() < 16 or y[train].nunique() < 2 or y[test].nunique() < 1:
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


def metric_row(pred: pd.DataFrame, feature_set: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce")
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
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
        out["roc_auc"] = float(roc_auc_score(y.astype(int), p))
        out["average_precision"] = float(average_precision_score(y.astype(int), p))
        rho, sp = spearmanr(y, p)
        out["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
        out["spearman_p"] = float(sp) if np.isfinite(sp) else np.nan
    return out


def grouped_model_audit(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fsets = {k: v for k, v in feature_sets(df).items() if v}
    preds: List[pd.DataFrame] = []
    metrics: List[Dict[str, Any]] = []
    for name, cols in fsets.items():
        for target in [t for t in TARGETS if t in df.columns]:
            for group_col in ["cycleNo", "source_stem"]:
                pred = grouped_predictions(df, cols, target, group_col, seed)
                pred["feature_set"] = name
                preds.append(pred)
                metrics.append(metric_row(pred, name))
    pred_df = pd.concat(preds, ignore_index=True, sort=False) if preds else pd.DataFrame()
    metric_df = pd.DataFrame(metrics)
    if not metric_df.empty:
        metric_df = metric_df.sort_values(
            ["target", "group_col", "roc_auc", "average_precision"],
            ascending=[True, True, False, False],
        )
    deltas: List[Dict[str, Any]] = []
    for _, row in metric_df.iterrows():
        if row["feature_set"] == "context_only":
            continue
        base = metric_df[
            (metric_df["target"] == row["target"])
            & (metric_df["group_col"] == row["group_col"])
            & (metric_df["feature_set"] == "context_only")
        ]
        if len(base):
            b = base.iloc[0]
            rec = row.to_dict()
            rec["base_roc_auc"] = b.get("roc_auc")
            rec["base_average_precision"] = b.get("average_precision")
            rec["delta_roc_auc_vs_context"] = row.get("roc_auc") - b.get("roc_auc") if pd.notna(row.get("roc_auc")) and pd.notna(b.get("roc_auc")) else np.nan
            rec["delta_average_precision_vs_context"] = row.get("average_precision") - b.get("average_precision") if pd.notna(row.get("average_precision")) and pd.notna(b.get("average_precision")) else np.nan
            deltas.append(rec)
    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_df = delta_df.sort_values(
            ["target", "group_col", "delta_roc_auc_vs_context", "delta_average_precision_vs_context"],
            ascending=[True, True, False, False],
        )
    return pred_df, metric_df, delta_df


def cohort_verdict(scalar: pd.DataFrame, metrics: pd.DataFrame, deltas: pd.DataFrame) -> str:
    strict_scalar = scalar[
        (scalar["transform"].isin(["source_residual", "within_source_rank", "within_source_z"]))
        & (pd.to_numeric(scalar["source_stratified_auc_p"], errors="coerce") <= 0.05)
        & (pd.to_numeric(scalar["oriented_auc"], errors="coerce") >= 0.65)
    ]
    source_models = metrics[
        (metrics["group_col"] == "source_stem")
        & (pd.to_numeric(metrics["roc_auc"], errors="coerce") >= 0.65)
    ]
    positive_deltas = deltas[
        (deltas["group_col"] == "source_stem")
        & (pd.to_numeric(deltas["delta_roc_auc_vs_context"], errors="coerce") > 0.05)
    ]
    if len(strict_scalar) and len(source_models) and len(positive_deltas):
        return "source_controlled_rollout_signal_candidate"
    if len(strict_scalar):
        return "scalar_source_controlled_signal_only"
    return "not_source_controlled_predictive;use_for_review_negative_controls"


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Sequence Source-Control Audit",
        "",
        "Source/cycle-control stress test for source-balanced particle-region rollout features.",
        "",
        f"- ROI rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Source/cycle-stratified permutations per scalar test: {summary['n_permutation']}",
        f"- Verdict: {summary['verdict']}",
        f"- Strict scalar rows: {summary['n_strict_scalar_rows']}",
        f"- Source-heldout model rows with AUC >= 0.65: {summary['n_source_model_auc_ge_065']}",
        "",
        "## Top Source-Stratified Scalars",
    ]
    for row in summary["top_source_stratified_scalars"][:8]:
        lines.append(
            f"- {row['target']} {row['feature']} ({row['transform']}): "
            f"AUC {row['oriented_auc']:.3f}, source p {row['source_stratified_auc_p']:.3f}, AP {row['average_precision']:.3f}"
        )
    lines.extend(["", "## Top Source-Heldout Models"])
    for row in summary["top_source_heldout_models"][:8]:
        lines.append(
            f"- {row['target']} {row['feature_set']}: AUC {row['roc_auc']:.3f}, AP {row['average_precision']:.3f}, n {row['n_eval']}"
        )
    lines.extend(["", "## Guardrail", summary["guardrail"], ""])
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_sequence_rollout_audit")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_sequence_source_control_audit")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--n-permutation", type=int, default=1000)
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(rollout_dir / "source_balanced_sequence_rollout_features.csv")
    features = [c for c in PRIMARY_FEATURES + CONTEXT_FEATURES if c in df.columns]
    work = add_transforms(df, [c for c in PRIMARY_FEATURES if c in df.columns])

    scalar = scalar_tests(work, features, args.seed, args.n_permutation)
    predictions, model_metrics, model_deltas = grouped_model_audit(work, args.seed)

    strict_scalar = scalar[
        (scalar["transform"].isin(["source_residual", "within_source_rank", "within_source_z"]))
        & (pd.to_numeric(scalar["source_stratified_auc_p"], errors="coerce") <= 0.05)
        & (pd.to_numeric(scalar["oriented_auc"], errors="coerce") >= 0.65)
    ].copy()
    source_models = model_metrics[
        (model_metrics["group_col"] == "source_stem")
        & (pd.to_numeric(model_metrics["roc_auc"], errors="coerce") >= 0.65)
    ].copy()

    paths = {
        "scalar_tests": out / "source_balanced_sequence_source_control_scalar_tests.csv",
        "model_metrics": out / "source_balanced_sequence_source_control_model_metrics.csv",
        "model_deltas": out / "source_balanced_sequence_source_control_model_deltas.csv",
        "predictions": out / "source_balanced_sequence_source_control_predictions.csv",
        "summary": out / "source_balanced_sequence_source_control_summary.json",
        "readme": out / "README.md",
    }
    scalar.to_csv(paths["scalar_tests"], index=False)
    model_metrics.to_csv(paths["model_metrics"], index=False)
    model_deltas.to_csv(paths["model_deltas"], index=False)
    predictions.to_csv(paths["predictions"], index=False)

    top_source_scalars = scalar.sort_values(
        ["source_stratified_auc_p", "oriented_auc", "average_precision"],
        ascending=[True, False, False],
    ).head(20)
    top_source_models = model_metrics[model_metrics["group_col"] == "source_stem"].sort_values(
        ["roc_auc", "average_precision"],
        ascending=[False, False],
    ).head(20)
    top_deltas = model_deltas.sort_values(
        ["delta_roc_auc_vs_context", "delta_average_precision_vs_context"],
        ascending=[False, False],
    ).head(20) if not model_deltas.empty else pd.DataFrame()

    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(pd.to_numeric(df["cycleNo"], errors="coerce").nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "targets": [t for t in TARGETS if t in df.columns],
        "features_tested": features,
        "n_scalar_rows": int(len(scalar)),
        "n_permutation": int(args.n_permutation),
        "n_strict_scalar_rows": int(len(strict_scalar)),
        "n_source_model_auc_ge_065": int(len(source_models)),
        "verdict": cohort_verdict(scalar, model_metrics, model_deltas),
        "top_source_stratified_scalars": top_source_scalars.to_dict("records"),
        "strict_source_controlled_scalars": strict_scalar.head(20).to_dict("records"),
        "top_source_heldout_models": top_source_models.to_dict("records"),
        "top_model_deltas_vs_context": top_deltas.to_dict("records") if not top_deltas.empty else [],
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "This audit stress-tests source-balanced rollout descriptors under source/cycle controls and weak future-drop labels. Within-source transforms are useful for review and negative-control design, but are not prospective source-transfer models. Results do not assign manual QC labels, validate degradation mechanisms, or calibrate diffusion.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_readme(out, clean_json(summary))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
