#!/usr/bin/env python3
"""Interpret early ROI-prefix features for later NMC front-direction outcomes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "front_positive_residual_binary"
TARGET_RESIDUAL = "phase_slope_positive_fraction_protocol_residual"
PREFIX_FRACTION = 0.75
MODEL_NAME = "logistic_l2"


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


def is_prefix_feature(col: str) -> bool:
    if "frame_index" in col or col in {"first_frame", "last_frame"}:
        return False
    return (
        col.startswith("prefix_")
        or col.startswith("first_q")
        or col.startswith("first_frame_mean")
        or col.startswith("last_prefix")
        or col.startswith("stage_prefix")
        or "_prefix_" in col
        or col.endswith("_prefix_first")
    )


def feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "roi_id",
        "cohort_role",
        "mode_label",
        "degradation_mode_hypothesis",
        "npz_path",
        "preview_png",
        "source_stem",
        "validation_label",
        "is_event_roi",
        "is_event_enriched_mode",
        "front_positive_residual_binary",
    }
    out: List[str] = []
    for col in df.columns:
        if col in excluded or col.startswith("target_") or col.startswith("outcome_"):
            continue
        if not is_prefix_feature(col):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 8 and vals.nunique(dropna=True) > 1:
            out.append(col)
    return out


def feature_group(feature: str) -> str:
    if feature.startswith("stage_prefix"):
        return "stage_drift"
    if "temporal_absdiff" in feature or "absdiff" in feature or "delta_abs" in feature or "delta_p95" in feature:
        return "temporal_change_energy"
    if feature.startswith("high_fraction") or feature.startswith("low_fraction") or "q70" in feature or "q30" in feature:
        return "bright_dark_fraction"
    if feature.startswith("roi_norm_mean") or feature.startswith("roi_mean") or feature.startswith("average_intensity"):
        return "mean_intensity_trace"
    if "spatial_std" in feature or "frame_delta" in feature or feature.startswith("first_frame") or feature.startswith("last_prefix"):
        return "frame_texture_level"
    return "other_prefix"


def make_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "logistic_l2":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        ])
    if model_name == "random_forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=4,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=seed,
            )),
        ])
    raise ValueError(model_name)


def logo_predictions(df: pd.DataFrame, features: List[str], target: str, model_name: str, seed: int) -> Dict[str, Any]:
    x = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce").astype(int)
    groups = df["event_reference_cycle"]
    logo = LeaveOneGroupOut()
    rows = []
    coefs = []
    oof_prob = np.full(len(df), np.nan, dtype=float)
    oof_pred = np.full(len(df), np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(logo.split(x, y, groups)):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
            continue
        model = make_model(model_name, seed + fold)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(x.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_prob[test_idx] = prob
        oof_pred[test_idx] = pred
        rows.append({
            "heldout_event_reference_cycle": float(groups.iloc[test_idx].iloc[0]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "roc_auc": float(roc_auc_score(y.iloc[test_idx], prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y.iloc[test_idx], pred)),
        })
        if model_name == "logistic_l2":
            coef = model.named_steps["clf"].coef_[0]
            coefs.extend({
                "heldout_event_reference_cycle": float(groups.iloc[test_idx].iloc[0]),
                "feature": feat,
                "coefficient": float(val),
            } for feat, val in zip(features, coef))
    valid = np.isfinite(oof_prob)
    summary = {
        "folds": int(len(rows)),
        "mean_fold_roc_auc": float(pd.DataFrame(rows)["roc_auc"].mean()) if rows else np.nan,
        "mean_fold_balanced_accuracy": float(pd.DataFrame(rows)["balanced_accuracy"].mean()) if rows else np.nan,
        "pooled_oof_roc_auc": float(roc_auc_score(y[valid], oof_prob[valid])) if valid.sum() and y[valid].nunique() > 1 else np.nan,
        "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(y[valid], oof_pred[valid])) if valid.sum() and y[valid].nunique() > 1 else np.nan,
        "n_oof": int(valid.sum()),
    }
    return {
        "folds": pd.DataFrame(rows),
        "coef": pd.DataFrame(coefs),
        "oof": pd.DataFrame({
            "roi_id": df["roi_id"].to_numpy(),
            "cohort_role": df["cohort_role"].to_numpy(),
            "event_reference_cycle": groups.to_numpy(),
            "target": y.to_numpy(),
            "target_residual": pd.to_numeric(df.get(TARGET_RESIDUAL, pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(),
            "oof_probability": oof_prob,
            "oof_prediction": oof_pred,
        }),
        "summary": summary,
    }


def permutation_importance_oof(df: pd.DataFrame, features: List[str], target: str, model_name: str, n_repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce").astype(int)
    groups = df["event_reference_cycle"]
    logo = LeaveOneGroupOut()
    rows = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(x, y, groups)):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
            continue
        model = make_model(model_name, seed + fold)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        base_prob = model.predict_proba(x.iloc[test_idx])[:, 1]
        base_auc = float(roc_auc_score(y.iloc[test_idx], base_prob))
        base_bal = float(balanced_accuracy_score(y.iloc[test_idx], (base_prob >= 0.5).astype(int)))
        for feat in features:
            auc_drops = []
            bal_drops = []
            for _ in range(n_repeats):
                xp = x.iloc[test_idx].copy()
                vals = xp[feat].to_numpy(copy=True)
                rng.shuffle(vals)
                xp[feat] = vals
                prob = model.predict_proba(xp)[:, 1]
                auc_drops.append(base_auc - float(roc_auc_score(y.iloc[test_idx], prob)))
                bal_drops.append(base_bal - float(balanced_accuracy_score(y.iloc[test_idx], (prob >= 0.5).astype(int))))
            rows.append({
                "heldout_event_reference_cycle": float(groups.iloc[test_idx].iloc[0]),
                "feature": feat,
                "feature_group": feature_group(feat),
                "baseline_fold_auc": base_auc,
                "mean_auc_drop": float(np.nanmean(auc_drops)),
                "mean_balanced_accuracy_drop": float(np.nanmean(bal_drops)),
            })
    if not rows:
        return pd.DataFrame()
    fold = pd.DataFrame(rows)
    out_rows = []
    for (feat, grp), sub in fold.groupby(["feature", "feature_group"], dropna=False):
        out_rows.append({
            "feature": feat,
            "feature_group": grp,
            "folds": int(len(sub)),
            "mean_auc_drop": float(sub["mean_auc_drop"].mean()),
            "median_auc_drop": float(sub["mean_auc_drop"].median()),
            "positive_drop_fraction": float((sub["mean_auc_drop"] > 0).mean()),
            "mean_balanced_accuracy_drop": float(sub["mean_balanced_accuracy_drop"].mean()),
        })
    return pd.DataFrame(out_rows).sort_values(["mean_auc_drop", "positive_drop_fraction"], ascending=[False, False])


def group_ablation(df: pd.DataFrame, features: List[str], target: str, model_name: str, seed: int) -> pd.DataFrame:
    base = logo_predictions(df, features, target, model_name, seed)["summary"]
    base_auc = float(base["pooled_oof_roc_auc"])
    rows = []
    groups = sorted({feature_group(f) for f in features})
    for grp in groups:
        keep = [f for f in features if feature_group(f) != grp]
        if len(keep) < 2:
            continue
        result = logo_predictions(df, keep, target, model_name, seed + 100 + len(rows))["summary"]
        auc = float(result["pooled_oof_roc_auc"])
        rows.append({
            "removed_group": grp,
            "n_removed_features": int(len(features) - len(keep)),
            "remaining_features": int(len(keep)),
            "pooled_oof_auc_without_group": auc,
            "pooled_oof_auc_drop": float(base_auc - auc) if np.isfinite(base_auc) and np.isfinite(auc) else np.nan,
            "pooled_oof_balanced_accuracy_without_group": result["pooled_oof_balanced_accuracy"],
        })
    return pd.DataFrame(rows).sort_values("pooled_oof_auc_drop", ascending=False)


def coefficient_summary(coefs: pd.DataFrame) -> pd.DataFrame:
    if coefs.empty:
        return pd.DataFrame()
    rows = []
    for feat, sub in coefs.groupby("feature", dropna=False):
        vals = pd.to_numeric(sub["coefficient"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        rows.append({
            "feature": feat,
            "feature_group": feature_group(feat),
            "mean_coefficient": float(np.mean(vals)),
            "mean_abs_coefficient": float(np.mean(np.abs(vals))),
            "median_coefficient": float(np.median(vals)),
            "positive_fraction": float((vals > 0).mean()),
            "folds": int(len(vals)),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_coefficient", ascending=False)


def univariate_tests(df: pd.DataFrame, features: Iterable[str], target: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target], errors="coerce").astype(int)
    residual = pd.to_numeric(df.get(TARGET_RESIDUAL, pd.Series(np.nan, index=df.index)), errors="coerce")
    for feat in features:
        x = pd.to_numeric(df[feat], errors="coerce")
        valid = x.notna() & y.notna()
        event = x[valid & y.eq(1)].to_numpy(dtype=float)
        control = x[valid & y.eq(0)].to_numpy(dtype=float)
        if len(event) >= 2 and len(control) >= 2:
            _, p_value = mannwhitneyu(event, control, alternative="two-sided")
            med_diff = float(np.nanmedian(event) - np.nanmedian(control))
        else:
            p_value = np.nan
            med_diff = np.nan
        rv = valid & residual.notna()
        rho = spearmanr(x[rv], residual[rv]).statistic if int(rv.sum()) >= 6 else np.nan
        rho_p = spearmanr(x[rv], residual[rv]).pvalue if int(rv.sum()) >= 6 else np.nan
        rows.append({
            "feature": feat,
            "feature_group": feature_group(feat),
            "target_positive_median_minus_negative": med_diff,
            "mannwhitney_p": float(p_value) if np.isfinite(p_value) else np.nan,
            "spearman_with_front_residual": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_p": float(rho_p) if np.isfinite(rho_p) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["mannwhitney_p", "feature"], na_position="last")


def permutation_null(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    observed_auc: float,
    model_name: str,
    prefix_fraction: float,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = pd.to_numeric(df[target], errors="coerce").astype(int).to_numpy()
    aucs = []
    for i in range(n_perm):
        shuffled = df.copy()
        perm = y.copy()
        rng.shuffle(perm)
        shuffled["_perm_target"] = perm
        result = logo_predictions(shuffled, features, "_perm_target", model_name, seed + 5000 + i)["summary"]
        auc = result["pooled_oof_roc_auc"]
        if np.isfinite(auc):
            aucs.append(float(auc))
    arr = np.asarray(aucs, dtype=float)
    return {
        "target": target,
        "model": model_name,
        "prefix_fraction": prefix_fraction,
        "observed_pooled_oof_auc": float(observed_auc),
        "n_permutations": int(len(arr)),
        "null_mean_auc": float(np.mean(arr)) if len(arr) else None,
        "null_p95_auc": float(np.percentile(arr, 95)) if len(arr) else None,
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)) if len(arr) else None,
    }




def fmt_optional(value: Any, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{x:.{digits}f}" if np.isfinite(x) else "NA"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/prefix_roi_feature_importance")
    parser.add_argument("--prefix-fraction", type=float, default=PREFIX_FRACTION)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--model", default=MODEL_NAME, choices=["logistic_l2", "random_forest"])
    parser.add_argument("--n-permutation-repeats", type=int, default=50)
    parser.add_argument("--n-null", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    prefix_path = derived / "prefix_roi_forecast" / "prefix_roi_feature_table.csv"
    df = pd.read_csv(prefix_path)
    df = df[np.isclose(pd.to_numeric(df["prefix_fraction"], errors="coerce"), args.prefix_fraction)].copy()
    if df.empty:
        raise ValueError(f"No rows for prefix fraction {args.prefix_fraction}")
    features = feature_columns(df)
    if len(features) < 2:
        raise ValueError("Too few prefix features for importance analysis")

    model_result = logo_predictions(df, features, args.target, args.model, args.seed)
    importance = permutation_importance_oof(df, features, args.target, args.model, args.n_permutation_repeats, args.seed + 1000)
    ablation = group_ablation(df, features, args.target, args.model, args.seed + 2000)
    coefs = coefficient_summary(model_result["coef"])
    univariate = univariate_tests(df, features, args.target)
    null = permutation_null(
        df,
        features,
        args.target,
        model_result["summary"]["pooled_oof_roc_auc"],
        args.model,
        args.prefix_fraction,
        args.n_null,
        args.seed + 3000,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "fold_metrics": out / "prefix_feature_importance_folds.csv",
        "oof_predictions": out / "prefix_feature_importance_oof_predictions.csv",
        "permutation_importance": out / "prefix_feature_permutation_importance.csv",
        "group_ablation": out / "prefix_feature_group_ablation.csv",
        "coefficient_summary": out / "prefix_feature_coefficient_summary.csv",
        "univariate_tests": out / "prefix_feature_univariate_tests.csv",
        "summary": out / "prefix_feature_importance_summary.json",
    }
    model_result["folds"].to_csv(paths["fold_metrics"], index=False)
    model_result["oof"].to_csv(paths["oof_predictions"], index=False)
    importance.to_csv(paths["permutation_importance"], index=False)
    ablation.to_csv(paths["group_ablation"], index=False)
    coefs.to_csv(paths["coefficient_summary"], index=False)
    univariate.to_csv(paths["univariate_tests"], index=False)

    top_importance = importance.head(12).to_dict("records") if not importance.empty else []
    top_ablation = ablation.head(8).to_dict("records") if not ablation.empty else []
    top_coefficients = coefs.head(12).to_dict("records") if not coefs.empty else []
    top_univariate = univariate.head(12).to_dict("records") if not univariate.empty else []
    summary = {
        "source_prefix_features": str(prefix_path),
        "n_roi": int(df["roi_id"].nunique()),
        "n_rows": int(len(df)),
        "prefix_fraction": float(args.prefix_fraction),
        "target": args.target,
        "model": args.model,
        "n_features": int(len(features)),
        "feature_group_counts": pd.Series([feature_group(f) for f in features]).value_counts().to_dict(),
        "model_summary": model_result["summary"],
        "permutation_null": null,
        "top_permutation_importance": top_importance,
        "top_group_ablation": top_ablation,
        "top_coefficients": top_coefficients,
        "top_univariate": top_univariate,
        "guardrail": "Feature importance is computed on the small 52-ROI selected cohort with leave-event-reference-cycle-out folds. Treat it as mechanistic triage for early particle-region video signals, not causal proof or a deployable detector.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Prefix ROI Feature Importance",
        "",
        "Interpretability audit for early cropped particle-ROI features predicting later positive phase-front residual direction.",
        "",
        f"- ROI rows: {summary['n_roi']}",
        f"- Prefix fraction: {summary['prefix_fraction']}",
        f"- Target: {summary['target']}",
        f"- Pooled OOF AUC: {fmt_optional(summary['model_summary']['pooled_oof_roc_auc'])}",
        f"- Null p>=observed: {fmt_optional(summary['permutation_null']['empirical_p_ge_observed'], 4)}",
        "",
        "## Top Feature Groups",
    ]
    for row in top_ablation[:5]:
        lines.append(
            f"- remove {row.get('removed_group')}: AUC drop={row.get('pooled_oof_auc_drop'):.3f}, n_features={row.get('n_removed_features')}"
        )
    lines += ["", "## Top Permutation Features"]
    for row in top_importance[:8]:
        lines.append(
            f"- {row.get('feature')} ({row.get('feature_group')}): AUC drop={row.get('mean_auc_drop'):.3f}, positive-fold fraction={row.get('positive_drop_fraction'):.2f}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "out_dir": str(out),
        "n_roi": summary["n_roi"],
        "pooled_oof_auc": summary["model_summary"]["pooled_oof_roc_auc"],
        "top_groups": top_ablation[:3],
        "null": null,
    }, indent=2))


if __name__ == "__main__":
    main()
