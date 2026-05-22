#!/usr/bin/env python3
"""Source-invariant physical family audit for video/echem transfer.

The prior source-invariant audit showed that video-only features can recover
future16 leave-source signal after removing source-dominated directions. This
script decomposes that rescue into interpretable particle-region feature
families: intensity, particle-vs-context contrast, gradients, normalized
heterogeneity scalars, and self-supervised video embedding PCs.
"""

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
from sklearn.preprocessing import StandardScaler


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 12) -> List[str]:
    keep: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    intensity = available_numeric(df, [c for c in df.columns if c.startswith("particle_mean_") or c.startswith("particle_std_")])
    contrast = available_numeric(df, [c for c in df.columns if c.startswith("particle_vs_context_")])
    gradient = available_numeric(df, [c for c in df.columns if c.startswith("particle_gradient_")])
    heterogeneity = available_numeric(
        df,
        [
            "particle_norm_mean",
            "particle_norm_std",
            "particle_norm_min",
            "particle_norm_max",
            "particle_norm_range",
            "particle_norm_cv",
            "particle_heterogeneity_rank",
            "particle_prior_area_fraction",
        ],
    )
    embedding = available_numeric(df, [c for c in df.columns if c.startswith("video_embed_pc")])
    handcrafted = sorted(set(intensity + contrast + gradient + heterogeneity))
    all_video = sorted(set(handcrafted + embedding))
    return {
        "particle_intensity": intensity,
        "particle_vs_context": contrast,
        "particle_gradient": gradient,
        "norm_heterogeneity": heterogeneity,
        "video_embedding": embedding,
        "handcrafted_particle": handcrafted,
        "all_video": all_video,
    }


def source_eta2(x: pd.DataFrame, sources: pd.Series) -> pd.Series:
    vals = x.apply(pd.to_numeric, errors="coerce")
    overall = vals.mean(axis=0)
    total = ((vals - overall) ** 2).sum(axis=0)
    between = pd.Series(0.0, index=vals.columns)
    for _, sub in vals.groupby(sources):
        between += len(sub) * (sub.mean(axis=0) - overall) ** 2
    return (between / total.replace(0, np.nan)).fillna(0.0)


def source_mean_basis(x_scaled: np.ndarray, sources: pd.Series, max_k: int) -> np.ndarray:
    means = []
    for source in sorted(sources.dropna().unique()):
        rows = np.asarray(sources == source)
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
    if method.startswith("source_confound_filter_"):
        frac = float(method.rsplit("_", 1)[-1])
        eta = source_eta2(train_x, train_sources)
        n_drop = int(np.floor(len(eta) * frac))
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
        basis = source_mean_basis(xtr, train_sources.reset_index(drop=True), k)
        if basis.shape[1] > 0:
            xtr = xtr - xtr @ basis @ basis.T
            xte = xte - xte @ basis @ basis.T
        meta["removed_source_mean_components"] = int(basis.shape[1])
    return xtr, xte, meta


def fit_predict(xtr: np.ndarray, ytr: pd.Series, xte: np.ndarray, seed: int) -> np.ndarray:
    model = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed)
    model.fit(xtr, ytr.astype(int))
    return model.predict_proba(xte)[:, 1]


def leave_source_predictions(df: pd.DataFrame, cols: List[str], target: str, family: str, method: str, seed: int) -> pd.DataFrame:
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    features = df[cols].apply(pd.to_numeric, errors="coerce")
    rows = []
    for source in sorted(df.loc[valid, "source_stem"].dropna().unique()):
        test = valid & (df["source_stem"] == source)
        train = valid & ~test
        meta = df.loc[test, ["embedding_row_id", "roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
        fold_meta: Dict[str, Any] = {}
        if len(cols) == 0 or train.sum() < 12 or y[train].nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped"
        else:
            xtr, xte, fold_meta = transform_fold(features.loc[train], features.loc[test], df.loc[train, "source_stem"], method)
            meta["predicted_probability"] = fit_predict(xtr, y[train], xte, seed)
            meta["status"] = "ok"
        meta["heldout_source"] = source
        meta["feature_family"] = family
        meta["method"] = method
        meta["target"] = target
        for key, val in fold_meta.items():
            meta[key] = val
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def metric_row(pred: pd.DataFrame, family: str, method: str, target: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    out: Dict[str, Any] = {
        "feature_family": family,
        "method": method,
        "target": target,
        "n_eval": int(len(tmp)),
        "n_sources": int(tmp["heldout_source"].nunique()) if "heldout_source" in tmp else 0,
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_positive": int(y.sum()) if len(y) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "mean_features_after": float(tmp.get("n_features_after", pd.Series(dtype=float)).mean()) if "n_features_after" in tmp else np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
        rho, pval = spearmanr(y, p)
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(pval)
    return out


def permutation_null(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int) -> Dict[str, Any]:
    if not np.isfinite(observed_auc):
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce").to_numpy()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        yy = y.copy(); rng.shuffle(yy)
        if len(np.unique(yy)) == 2:
            vals.append(float(roc_auc_score(yy, p)))
    arr = np.asarray(vals)
    if len(arr) == 0:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    return {"n_permutation": int(len(arr)), "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)), "null_auc_mean": float(arr.mean()), "null_auc_p95": float(np.quantile(arr, 0.95))}


def feature_source_confounds(df: pd.DataFrame, families: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for family, cols in families.items():
        if not cols:
            continue
        eta = source_eta2(df[cols].apply(pd.to_numeric, errors="coerce"), df["source_stem"])
        for feature, val in eta.sort_values(ascending=False).items():
            rows.append({"feature_family": family, "feature": feature, "source_eta2": float(val)})
    return pd.DataFrame(rows)


def deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in sorted(metrics["target"].dropna().unique()):
        sub = metrics[metrics["target"] == target]
        all_raw = sub[(sub["feature_family"] == "all_video") & (sub["method"] == "raw")]
        for _, row in sub.iterrows():
            base = sub[(sub["feature_family"] == row["feature_family"]) & (sub["method"] == "raw")]
            if not base.empty and row["method"] != "raw":
                b = base.iloc[0]
                rows.append({"target": target, "feature_family": row["feature_family"], "method": row["method"], "comparison": f"{row['feature_family']}_{row['method']}_minus_same_raw", "delta_roc_auc": row["roc_auc"] - b["roc_auc"], "delta_average_precision": row["average_precision"] - b["average_precision"], "delta_spearman_rho": row["spearman_rho"] - b["spearman_rho"], "comparison_metric": row["roc_auc"], "base_metric": b["roc_auc"]})
            if not all_raw.empty and row["feature_family"] != "all_video":
                a = all_raw.iloc[0]
                rows.append({"target": target, "feature_family": row["feature_family"], "method": row["method"], "comparison": f"{row['feature_family']}_{row['method']}_minus_all_video_raw", "delta_roc_auc": row["roc_auc"] - a["roc_auc"], "delta_average_precision": row["average_precision"] - a["average_precision"], "delta_spearman_rho": row["spearman_rho"] - a["spearman_rho"], "comparison_metric": row["roc_auc"], "base_metric": a["roc_auc"]})
    return pd.DataFrame(rows).sort_values(["target", "delta_roc_auc"], ascending=[True, False]) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_invariant_physical_family_audit")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--n-permutation", type=int, default=500)
    args = parser.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = read_csv(Path(args.derived_dir) / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_joined.csv")
    families = feature_families(df)
    targets = [t for t in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"] if t in df.columns and pd.to_numeric(df[t], errors="coerce").isin([0, 1]).sum() >= 16]
    methods = ["raw", "source_mean_resid_2", "source_mean_resid_4", "source_confound_filter_0.50"]

    preds = []
    metric_rows = []
    null_rows = []
    for target in targets:
        for family, cols in families.items():
            for method in methods:
                pred = leave_source_predictions(df, cols, target, family, method, args.seed)
                preds.append(pred)
                met = metric_row(pred, family, method, target)
                metric_rows.append(met)
                if target == "future_any_drop_within_16cycles":
                    null = permutation_null(pred, met["roc_auc"], args.seed, args.n_permutation)
                    null.update({"target": target, "feature_family": family, "method": method, "observed_roc_auc": met["roc_auc"]})
                    null_rows.append(null)
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        metrics_df = metrics_df.merge(null_df[["target", "feature_family", "method", "empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]], on=["target", "feature_family", "method"], how="left")
    delta_df = deltas(metrics_df)
    confound_df = feature_source_confounds(df, families)
    family_df = pd.DataFrame([{"feature_family": k, "n_features": len(v), "features": ";".join(v)} for k, v in families.items()])

    paths = {
        "metrics": out / "source_invariant_family_metrics.csv",
        "predictions": out / "source_invariant_family_predictions.csv",
        "deltas": out / "source_invariant_family_deltas.csv",
        "permutation_null": out / "source_invariant_family_permutation_null.csv",
        "feature_source_confounds": out / "source_invariant_family_feature_source_confounds.csv",
        "family_features": out / "source_invariant_family_features.csv",
        "summary": out / "source_invariant_family_summary.json",
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)
    confound_df.to_csv(paths["feature_source_confounds"], index=False)
    family_df.to_csv(paths["family_features"], index=False)

    sorted_metrics = metrics_df.sort_values(["target", "roc_auc", "average_precision"], ascending=[True, False, False])
    summary = clean_json({
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "targets": targets,
        "methods": methods,
        "feature_family_sizes": {k: len(v) for k, v in families.items()},
        "top_metrics": sorted_metrics.to_dict("records"),
        "top_deltas": delta_df.to_dict("records") if not delta_df.empty else [],
        "top_source_confounded_features": confound_df.sort_values("source_eta2", ascending=False).head(30).to_dict("records"),
        "guardrail": "Physical family readouts use automatic particle-region descriptors and weak future labels under leave-source splits. They identify candidate physics families for review prioritization only; source/outcome imbalance, automatic masks, missing manual QC, and uncalibrated optical-front diffusion remain guardrails.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Source-Invariant Physical Family Audit\n\n"
        "Decomposes source-invariant leave-source video signal into particle-region feature families.\n\n"
        f"- Rows: {summary['n_rows']}\n- Cycles: {summary['n_cycles']}\n- Sources: {summary['n_sources']}\n"
        f"- Feature families: {summary['feature_family_sizes']}\n- Methods: {methods}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
