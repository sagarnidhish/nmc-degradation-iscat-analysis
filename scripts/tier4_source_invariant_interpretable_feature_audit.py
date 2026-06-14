#!/usr/bin/env python3
"""Interpretable source-invariant particle-feature audit.

The physical-family audit found that normalized heterogeneity and
particle-vs-context contrast preserve future16 signal after source-invariant
handling. This script drills down to exact automatic particle-region features
and small feature combinations under leave-source evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


TARGET = "future_any_drop_within_16cycles"


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


def feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "norm_heterogeneity": [
            "particle_norm_mean",
            "particle_norm_std",
            "particle_norm_min",
            "particle_norm_max",
            "particle_norm_range",
            "particle_norm_cv",
            "particle_heterogeneity_rank",
            "particle_prior_area_fraction",
        ],
        "particle_vs_context": [c for c in df.columns if c.startswith("particle_vs_context_")],
        "particle_gradient": [c for c in df.columns if c.startswith("particle_gradient_")],
        "particle_intensity": [c for c in df.columns if c.startswith("particle_mean_") or c.startswith("particle_std_")],
    }
    return {k: available_numeric(df, v) for k, v in groups.items()}


def feature_family(feature: str, groups: Dict[str, List[str]]) -> str:
    for family, cols in groups.items():
        if feature in cols:
            return family
    return "other"


def source_eta2(x: pd.DataFrame, sources: pd.Series) -> pd.Series:
    vals = x.apply(pd.to_numeric, errors="coerce")
    overall = vals.mean(axis=0)
    total = ((vals - overall) ** 2).sum(axis=0)
    between = pd.Series(0.0, index=vals.columns)
    for _, sub in vals.groupby(sources):
        between += len(sub) * (sub.mean(axis=0) - overall) ** 2
    return (between / total.replace(0, np.nan)).fillna(0.0)


def orient_univariate_metrics(df: pd.DataFrame, feature: str, target: str, sources: pd.Series) -> Dict[str, Any]:
    y = pd.to_numeric(df[target], errors="coerce")
    x = pd.to_numeric(df[feature], errors="coerce")
    valid = y.isin([0, 1]) & x.notna() & sources.notna()
    yy = y[valid].astype(int)
    xx = x[valid].astype(float)
    out: Dict[str, Any] = {
        "feature": feature,
        "n_eval": int(valid.sum()),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "n_sources": int(sources[valid].nunique()),
        "source_eta2": np.nan,
        "roc_auc_oriented": np.nan,
        "roc_auc_positive_direction": np.nan,
        "orientation": "NA",
        "average_precision_oriented": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "median_positive": np.nan,
        "median_negative": np.nan,
        "median_positive_minus_negative": np.nan,
        "mannwhitney_p": np.nan,
    }
    if valid.sum() >= 8:
        out["source_eta2"] = float(source_eta2(df.loc[valid, [feature]], sources[valid]).iloc[0])
    if len(yy) >= 8 and yy.nunique() == 2:
        auc = float(roc_auc_score(yy, xx))
        sign = 1.0 if auc >= 0.5 else -1.0
        oriented = xx * sign
        out["roc_auc_positive_direction"] = auc
        out["roc_auc_oriented"] = max(auc, 1.0 - auc)
        out["orientation"] = "higher_in_positive" if sign > 0 else "lower_in_positive"
        out["average_precision_oriented"] = float(average_precision_score(yy, oriented))
        rho, pval = spearmanr(yy, xx)
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(pval)
        pos = xx[yy == 1]
        neg = xx[yy == 0]
        out["median_positive"] = float(pos.median())
        out["median_negative"] = float(neg.median())
        out["median_positive_minus_negative"] = float(pos.median() - neg.median())
        try:
            out["mannwhitney_p"] = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
        except ValueError:
            pass
    return out


def transform_fold(train_x: pd.DataFrame, test_x: pd.DataFrame, train_sources: pd.Series, method: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"n_features_before": int(train_x.shape[1]), "n_features_after": int(train_x.shape[1])}
    if method.startswith("source_confound_filter_") and train_x.shape[1] > 1:
        frac = float(method.rsplit("_", 1)[-1])
        eta = source_eta2(train_x, train_sources)
        n_drop = min(int(np.floor(len(eta) * frac)), len(eta) - 1)
        keep = eta.sort_values(ascending=False).index[n_drop:].tolist() if n_drop > 0 else eta.index.tolist()
        train_x = train_x[keep]
        test_x = test_x[keep]
        meta.update({"filter_fraction": frac, "n_dropped_source_confounded_features": n_drop, "n_features_after": len(keep), "kept_features": ";".join(keep)})
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(imputer.fit_transform(train_x))
    xte = scaler.transform(imputer.transform(test_x))
    return xtr, xte, meta


def leave_source_predictions(df: pd.DataFrame, cols: List[str], target: str, feature_set: str, method: str, seed: int) -> pd.DataFrame:
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    features = df[cols].apply(pd.to_numeric, errors="coerce")
    rows = []
    for source in sorted(df.loc[valid, "source_stem"].dropna().unique()):
        test = valid & (df["source_stem"] == source)
        train = valid & ~test
        meta_cols = [c for c in ["embedding_row_id", "roi_id", "cycleNo", "source_stem", target] if c in df.columns]
        meta = df.loc[test, meta_cols].rename(columns={target: "observed"}).copy()
        fold_meta: Dict[str, Any] = {}
        if len(cols) == 0 or train.sum() < 12 or y[train].nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped"
        else:
            xtr, xte, fold_meta = transform_fold(features.loc[train], features.loc[test], df.loc[train, "source_stem"], method)
            model = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed)
            model.fit(xtr, y[train].astype(int))
            meta["predicted_probability"] = model.predict_proba(xte)[:, 1]
            meta["status"] = "ok"
        meta["heldout_source"] = source
        meta["feature_set"] = feature_set
        meta["method"] = method
        meta["features"] = ";".join(cols)
        for key, val in fold_meta.items():
            meta[key] = val
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def metric_row(pred: pd.DataFrame, feature_set: str, method: str, features: List[str], groups: Dict[str, List[str]]) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    families = sorted({feature_family(f, groups) for f in features})
    out: Dict[str, Any] = {
        "feature_set": feature_set,
        "method": method,
        "features": ";".join(features),
        "feature_families": ";".join(families),
        "n_features": len(features),
        "n_eval": int(len(tmp)),
        "n_sources": int(tmp["heldout_source"].nunique()) if "heldout_source" in tmp else 0,
        "n_positive": int(y.sum()) if len(y) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "empirical_p_ge_observed": np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
        rho, pval = spearmanr(y, p)
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(pval)
    return out


def source_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, method, source), sub in pred.groupby(["feature_set", "method", "heldout_source"], dropna=False):
        yy = pd.to_numeric(sub["observed"], errors="coerce")
        pp = pd.to_numeric(sub["predicted_probability"], errors="coerce")
        rows.append(
            {
                "feature_set": feature_set,
                "method": method,
                "heldout_source": source,
                "n_eval": int(pp.notna().sum()),
                "n_positive": int(yy.fillna(0).sum()),
                "observed_rate": float(yy.mean()) if yy.notna().any() else np.nan,
                "mean_predicted_probability": float(pp.mean()) if pp.notna().any() else np.nan,
                "predicted_minus_observed_rate": float(pp.mean() - yy.mean()) if pp.notna().any() and yy.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def permutation_p(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int) -> Dict[str, Any]:
    if not np.isfinite(observed_auc):
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan, "null_auc_mean": np.nan, "null_auc_p95": np.nan}
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce").to_numpy()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        yy = y.copy()
        rng.shuffle(yy)
        if len(np.unique(yy)) == 2:
            vals.append(float(roc_auc_score(yy, p)))
    arr = np.asarray(vals)
    if len(arr) == 0:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan, "null_auc_mean": np.nan, "null_auc_p95": np.nan}
    return {
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)),
        "null_auc_mean": float(arr.mean()),
        "null_auc_p95": float(np.quantile(arr, 0.95)),
    }


def candidate_sets(univariate: pd.DataFrame, groups: Dict[str, List[str]], max_pairs: int, max_triples: int) -> Dict[str, List[str]]:
    selected_features = (
        univariate.sort_values(["roc_auc_oriented", "average_precision_oriented"], ascending=[False, False])
        .head(16)["feature"]
        .tolist()
    )
    for family in ["norm_heterogeneity", "particle_vs_context"]:
        family_top = (
            univariate[univariate["feature_family"] == family]
            .sort_values(["roc_auc_oriented", "average_precision_oriented"], ascending=[False, False])
            .head(8)["feature"]
            .tolist()
        )
        selected_features = list(dict.fromkeys(selected_features + family_top))

    sets: Dict[str, List[str]] = {f"single::{f}": [f] for f in selected_features}
    ranked_pairs = []
    for a, b in itertools.combinations(selected_features, 2):
        families = {feature_family(a, groups), feature_family(b, groups)}
        bonus = 1 if {"norm_heterogeneity", "particle_vs_context"}.issubset(families) else 0
        score = float(
            univariate.set_index("feature").loc[[a, b], "roc_auc_oriented"].mean()
        ) + bonus
        ranked_pairs.append((score, a, b))
    for _, a, b in sorted(ranked_pairs, reverse=True)[:max_pairs]:
        sets[f"pair::{a}+{b}"] = [a, b]

    ranked_triples = []
    for combo in itertools.combinations(selected_features[:14], 3):
        families = {feature_family(f, groups) for f in combo}
        if len(families) < 2:
            continue
        score = float(univariate.set_index("feature").loc[list(combo), "roc_auc_oriented"].mean()) + 0.2 * len(families)
        ranked_triples.append((score, combo))
    for _, combo in sorted(ranked_triples, reverse=True)[:max_triples]:
        sets[f"trio::{'+'.join(combo)}"] = list(combo)

    for family, cols in groups.items():
        if cols:
            sets[f"family::{family}"] = cols
    sets["family::norm_heterogeneity_plus_particle_vs_context"] = sorted(
        set(groups.get("norm_heterogeneity", []) + groups.get("particle_vs_context", []))
    )
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_invariant_interpretable_feature_audit")
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--n-permutation", type=int, default=500)
    parser.add_argument("--max-pairs", type=int, default=36)
    parser.add_argument("--max-triples", type=int, default=24)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_csv(Path(args.derived_dir) / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_joined.csv")
    if TARGET not in df.columns:
        raise KeyError(TARGET)

    groups = feature_groups(df)
    all_features = sorted(set(itertools.chain.from_iterable(groups.values())))
    target_valid = pd.to_numeric(df[TARGET], errors="coerce").isin([0, 1])
    eval_df = df.loc[target_valid & df["source_stem"].notna()].copy()

    uni_rows = [orient_univariate_metrics(eval_df, f, TARGET, eval_df["source_stem"]) for f in all_features]
    univariate = pd.DataFrame(uni_rows)
    univariate["feature_family"] = univariate["feature"].map(lambda f: feature_family(f, groups))
    univariate = univariate.sort_values(["roc_auc_oriented", "source_eta2"], ascending=[False, True])

    sets = candidate_sets(univariate, groups, args.max_pairs, args.max_triples)
    methods = ["raw", "source_confound_filter_0.50"]
    pred_frames = []
    metric_rows = []
    pred_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}
    for feature_set, cols in sets.items():
        for method in methods:
            pred = leave_source_predictions(eval_df, cols, TARGET, feature_set, method, args.seed)
            pred_frames.append(pred)
            pred_by_key[(feature_set, method)] = pred
            metric_rows.append(metric_row(pred, feature_set, method, cols, groups))
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows).sort_values(["roc_auc", "average_precision"], ascending=[False, False])

    null_rows = []
    for _, row in metrics.head(30).iterrows():
        key = (row["feature_set"], row["method"])
        null = permutation_p(pred_by_key[key], row["roc_auc"], args.seed, args.n_permutation)
        null.update({"feature_set": row["feature_set"], "method": row["method"], "observed_roc_auc": row["roc_auc"]})
        null_rows.append(null)
    nulls = pd.DataFrame(null_rows)
    if not nulls.empty:
        metrics = metrics.merge(nulls.drop(columns=["observed_roc_auc"]), on=["feature_set", "method"], how="left", suffixes=("", "_perm"))
        for col in ["empirical_p_ge_observed", "n_permutation", "null_auc_mean", "null_auc_p95"]:
            perm_col = f"{col}_perm"
            if perm_col in metrics.columns:
                metrics[col] = metrics[perm_col].combine_first(metrics[col]) if col in metrics.columns else metrics[perm_col]
                metrics = metrics.drop(columns=[perm_col])

    source_stability = source_summary(predictions)
    feature_inventory = pd.DataFrame(
        [{"feature_family": family, "n_features": len(cols), "features": ";".join(cols)} for family, cols in groups.items()]
    )

    paths = {
        "univariate": out / "interpretable_feature_univariate_metrics.csv",
        "set_metrics": out / "interpretable_feature_set_metrics.csv",
        "predictions": out / "interpretable_feature_set_predictions.csv",
        "source_stability": out / "interpretable_feature_source_stability.csv",
        "permutation_null": out / "interpretable_feature_permutation_null.csv",
        "feature_inventory": out / "interpretable_feature_inventory.csv",
        "summary": out / "source_invariant_interpretable_summary.json",
    }
    univariate.to_csv(paths["univariate"], index=False)
    metrics.to_csv(paths["set_metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    source_stability.to_csv(paths["source_stability"], index=False)
    nulls.to_csv(paths["permutation_null"], index=False)
    feature_inventory.to_csv(paths["feature_inventory"], index=False)

    top_single_sets = metrics[metrics["n_features"] == 1].head(20)
    top_combo_sets = metrics[metrics["n_features"] > 1].head(20)
    summary = clean_json(
        {
            "n_rows": int(len(eval_df)),
            "n_cycles": int(eval_df["cycleNo"].nunique()),
            "n_sources": int(eval_df["source_stem"].nunique()),
            "target": TARGET,
            "methods": methods,
            "feature_family_sizes": {k: len(v) for k, v in groups.items()},
            "n_candidate_feature_sets": int(len(sets)),
            "top_univariate_features": univariate.head(30).to_dict("records"),
            "top_single_feature_transfer": top_single_sets.to_dict("records"),
            "top_combo_transfer": top_combo_sets.to_dict("records"),
            "top_set_metrics": metrics.head(40).to_dict("records"),
            "source_stability": source_stability.sort_values("predicted_minus_observed_rate", ascending=False).head(24).to_dict("records"),
            "guardrail": "Exact-feature readouts are automatic particle-region descriptors evaluated against weak future16 labels under leave-source splits. They are hypothesis-prioritization signals only; source imbalance, acquisition context, automatic masks, and absent manual QC still block mechanistic claims.",
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Source-Invariant Interpretable Feature Audit\n\n"
        "Ranks exact automatic particle-region descriptors and small descriptor combinations under leave-source future16 evaluation.\n\n"
        f"- Rows: {summary['n_rows']}\n- Cycles: {summary['n_cycles']}\n- Sources: {summary['n_sources']}\n"
        f"- Candidate feature sets: {summary['n_candidate_feature_sets']}\n- Methods: {methods}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
