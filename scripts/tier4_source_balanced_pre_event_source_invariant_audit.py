#!/usr/bin/env python3
"""Source-invariant pre-event mechanism audit.

This audit uses the source-balanced pre-event ROI readout table and asks which
interpretable feature families preserve event-relative signal under
leave-source evaluation after simple source-confound handling. It is a compact
mechanism screen: source-normalized pre-event front/contrast/heterogeneity
features are treated as hypotheses, not validated precursors.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

TARGETS = ["clean_pre8_vs_post_control", "near_pre_vs_far_pre"]
METHODS = ["raw_standard", "source_residual", "within_source_rank", "source_confound_filter_0.25", "source_mean_resid_2"]
PRE_BINS = {"near_pre_event_1_8", "mid_pre_event_9_16", "far_pre_event_17_32"}
CONTROL_BINS = {"post_event_1_16", "no_near_event_control"}


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


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 16) -> List[str]:
    keep: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    families = {
        "roi_intensity": [c for c in df.columns if c.startswith("roi_norm_mean") or c.startswith("raw_roi_mean") or c.startswith("roi_mean")],
        "video_dynamics": [c for c in df.columns if c.startswith("temporal_energy") or c.startswith("persistence") or c.startswith("velocity") or c.startswith("frame_diff")],
        "heterogeneity": [c for c in df.columns if c.startswith("spatial_std") or c.startswith("bright_fraction") or c.startswith("dark_fraction")],
        "mask_contrast": [c for c in df.columns if c.startswith("masked_minus_background") or c.startswith("mask_")],
        "front_geometry": [c for c in df.columns if c.startswith("front_radius") or c.startswith("front_gradient")],
        "apparent_diffusion": [c for c in df.columns if c.startswith("apparent_diffusion")],
        "object_context_guardrail": [c for c in df.columns if c.startswith("object_") or c in {"object_x_full_approx", "object_y_full_approx"}],
    }
    families = {k: available_numeric(df, v) for k, v in families.items()}
    families["physics_front_combo"] = sorted(set(families["mask_contrast"] + families["front_geometry"] + families["apparent_diffusion"]))
    families["all_interpretable"] = sorted(set(sum(families.values(), [])))
    return {k: v for k, v in families.items() if v}


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = out["event_relative_bin"].astype(str)
    out["clean_pre8_vs_post_control"] = np.where(bins.eq("near_pre_event_1_8"), 1, np.where(bins.isin(CONTROL_BINS), 0, np.nan))
    out["near_pre_vs_far_pre"] = np.where(bins.eq("near_pre_event_1_8"), 1, np.where(bins.eq("far_pre_event_17_32"), 0, np.nan))
    out["pre_event_clock_closer"] = np.where(bins.isin(PRE_BINS), -pd.to_numeric(out["cycles_to_next_event"], errors="coerce"), np.nan)
    return out


def source_eta2(x: pd.DataFrame, sources: pd.Series) -> pd.Series:
    vals = x.apply(pd.to_numeric, errors="coerce")
    overall = vals.mean(axis=0)
    total = ((vals - overall) ** 2).sum(axis=0)
    between = pd.Series(0.0, index=vals.columns)
    for _, sub in vals.groupby(sources.astype(str)):
        between += len(sub) * (sub.mean(axis=0) - overall) ** 2
    return (between / total.replace(0, np.nan)).fillna(0.0)


def source_mean_basis(x_scaled: np.ndarray, sources: pd.Series, max_k: int) -> np.ndarray:
    means = []
    src = sources.reset_index(drop=True).astype(str)
    for source in sorted(src.unique()):
        rows = np.asarray(src == source)
        if rows.sum():
            means.append(x_scaled[rows].mean(axis=0))
    if len(means) < 2:
        return np.zeros((x_scaled.shape[1], 0))
    mat = np.vstack(means)
    mat = mat - mat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(mat, full_matrices=False)
    k = min(max_k, vt.shape[0], x_scaled.shape[1])
    return vt[:k].T if k > 0 else np.zeros((x_scaled.shape[1], 0))


def transform_fold(train_x: pd.DataFrame, test_x: pd.DataFrame, train_sources: pd.Series, method: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"n_features_before": int(train_x.shape[1]), "n_features_after": int(train_x.shape[1]), "removed_source_mean_components": 0}
    if method == "source_residual":
        train_x = train_x - train_x.groupby(train_sources.astype(str)).transform("mean")
        global_means = train_x.mean(axis=0)
        test_x = test_x - global_means
        meta["test_source_residual_reference"] = "train_global_mean"
    elif method == "within_source_rank":
        train_x = train_x.groupby(train_sources.astype(str)).rank(pct=True) - 0.5
        global_medians = train_x.median(axis=0)
        test_x = test_x.rank(pct=True) - 0.5
        test_x = test_x.fillna(global_medians)
        meta["test_rank_reference"] = "heldout_source_rank"
    elif method.startswith("source_confound_filter_") and train_x.shape[1] > 1:
        frac = float(method.rsplit("_", 1)[-1])
        eta = source_eta2(train_x, train_sources)
        n_drop = min(int(np.floor(len(eta) * frac)), len(eta) - 1)
        keep = eta.sort_values(ascending=False).index[n_drop:].tolist() if n_drop > 0 else eta.index.tolist()
        train_x = train_x[keep]
        test_x = test_x[keep]
        meta.update({"filter_fraction": frac, "n_dropped_source_confounded_features": n_drop, "n_features_after": len(keep)})

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(imputer.fit_transform(train_x))
    xte = scaler.transform(imputer.transform(test_x))
    if method.startswith("source_mean_resid_"):
        k = int(method.rsplit("_", 1)[-1])
        basis = source_mean_basis(xtr, train_sources, k)
        if basis.shape[1] > 0:
            xtr = xtr - xtr @ basis @ basis.T
            xte = xte - xte @ basis @ basis.T
        meta["removed_source_mean_components"] = int(basis.shape[1])
    return xtr, xte, meta


def leave_source_predictions(df: pd.DataFrame, cols: List[str], target: str, family: str, method: str, seed: int) -> pd.DataFrame:
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    features = df[cols].apply(pd.to_numeric, errors="coerce")
    rows: List[Dict[str, Any]] = []
    for source in sorted(df.loc[valid, "source_stem"].astype(str).unique()):
        test = valid & df["source_stem"].astype(str).eq(source)
        train = valid & ~test
        meta_cols = [c for c in ["roi_id", "cycleNo", "source_stem", "event_relative_bin", target] if c in df.columns]
        meta = df.loc[test, meta_cols].rename(columns={target: "observed"}).copy()
        fold_meta: Dict[str, Any] = {}
        if train.sum() < 12 or y[train].nunique() < 2 or len(cols) == 0:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped"
        else:
            xtr, xte, fold_meta = transform_fold(features.loc[train], features.loc[test], df.loc[train, "source_stem"], method)
            model = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.25, solver="liblinear", random_state=seed)
            model.fit(xtr, y[train].astype(int))
            meta["predicted_probability"] = model.predict_proba(xte)[:, 1]
            meta["status"] = "ok"
        meta["heldout_source"] = source
        meta["feature_family"] = family
        meta["method"] = method
        meta["target"] = target
        meta["features"] = ";".join(cols)
        for key, val in fold_meta.items():
            meta[key] = val
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def metric_row(pred: pd.DataFrame, family: str, method: str, target: str, cols: List[str], source_eta: pd.Series) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    rho = pval = np.nan
    if len(tmp) >= 8 and y.nunique() == 2:
        rho, pval = spearmanr(y, p)
    return {
        "target": target,
        "feature_family": family,
        "method": method,
        "n_features": int(len(cols)),
        "n_eval": int(len(tmp)),
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_sources": int(tmp["heldout_source"].nunique()) if "heldout_source" in tmp else 0,
        "n_positive": int(y.sum()) if len(y) else 0,
        "roc_auc": float(roc_auc_score(y, p)) if len(tmp) >= 8 and y.nunique() == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p)) if len(tmp) >= 8 and y.nunique() == 2 else np.nan,
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "spearman_p": float(pval) if np.isfinite(pval) else np.nan,
        "mean_raw_source_eta2": float(source_eta.reindex(cols).mean()) if cols else np.nan,
        "max_raw_source_eta2": float(source_eta.reindex(cols).max()) if cols else np.nan,
    }


def top_univariate_rows(df: pd.DataFrame, features: List[str], target: str, source_eta: pd.Series, n: int = 24) -> List[Dict[str, Any]]:
    rows = []
    y = pd.to_numeric(df[target], errors="coerce")
    for feature in features:
        x = pd.to_numeric(df[feature], errors="coerce")
        valid = y.isin([0, 1]) & x.notna()
        yy = y[valid].astype(int)
        xx = x[valid]
        if valid.sum() < 8 or yy.nunique() < 2 or xx.nunique() < 2:
            continue
        direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
        score = xx if direction == "higher_in_positive" else -xx
        rows.append({
            "target": target,
            "feature": feature,
            "direction": direction,
            "n": int(valid.sum()),
            "n_positive": int(yy.sum()),
            "oriented_auc": float(roc_auc_score(yy, score)),
            "average_precision": float(average_precision_score(yy, score)),
            "median_positive_minus_negative": float(xx[yy == 1].median() - xx[yy == 0].median()),
            "raw_source_eta2": float(source_eta.get(feature, np.nan)),
        })
    return sorted(rows, key=lambda r: (r["oriented_auc"], r["average_precision"]), reverse=True)[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_readout_audit/source_balanced_pre_event_readout_features.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_source_invariant_audit")
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = add_targets(pd.read_csv(args.features).loc[:, lambda d: ~d.columns.duplicated()].copy())
    families = feature_families(df)
    all_features = sorted(set(sum(families.values(), [])))
    raw_eta = source_eta2(df[all_features], df["source_stem"]) if all_features else pd.Series(dtype=float)

    preds = []
    metrics = []
    for target in TARGETS:
        for family, cols in families.items():
            for method in METHODS:
                pred = leave_source_predictions(df, cols, target, family, method, args.seed)
                preds.append(pred)
                metrics.append(metric_row(pred, family, method, target, cols, raw_eta))

    metrics_df = pd.DataFrame(metrics).sort_values(["target", "roc_auc", "average_precision"], ascending=[True, False, False])
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    uni_rows = []
    for target in TARGETS:
        uni_rows.extend(top_univariate_rows(df, all_features, target, raw_eta, n=24))
    uni_df = pd.DataFrame(uni_rows)

    best_by_target = []
    for target, sub in metrics_df.groupby("target", dropna=False):
        best_by_target.extend(sub.head(8).to_dict("records"))
    best_low_source = (
        metrics_df[metrics_df["max_raw_source_eta2"].fillna(1.0) <= 0.5]
        .sort_values(["target", "roc_auc", "average_precision"], ascending=[True, False, False])
        .groupby("target", dropna=False)
        .head(8)
        .to_dict("records")
    )

    paths = {
        "metrics": out / "source_balanced_pre_event_source_invariant_metrics.csv",
        "predictions": out / "source_balanced_pre_event_source_invariant_predictions.csv",
        "univariate": out / "source_balanced_pre_event_source_invariant_univariate.csv",
        "summary": out / "source_balanced_pre_event_source_invariant_summary.json",
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    uni_df.to_csv(paths["univariate"], index=False)

    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "event_relative_bin_counts": clean_json(df["event_relative_bin"].value_counts().to_dict()),
        "targets": TARGETS,
        "methods": METHODS,
        "feature_families": {k: v for k, v in families.items()},
        "n_features_total": int(len(all_features)),
        "best_by_target": clean_json(best_by_target),
        "best_low_source_eta2_models": clean_json(best_low_source),
        "top_univariate_rows": clean_json(uni_rows[:32]),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Leave-source source-invariant pre-event models use automatic ROI features and weak event-relative labels. Source residual/rank transforms are analysis-time normalizations. These results nominate pre-event mechanism families for review; they are not causal precursors, deployable warnings, manual particle identities, calibrated phase boundaries, or diffusion coefficients.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text(
        "# Source-Balanced Pre-Event Source-Invariant Audit\n\n"
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}\n"
        f"- Feature families: {', '.join(families.keys())}\n"
        f"- Targets: {', '.join(TARGETS)}\n\n"
        "## Guardrail\n\n"
        f"{summary['guardrail']}\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
