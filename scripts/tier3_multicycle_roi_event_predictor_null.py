#!/usr/bin/env python3
"""Permutation null for the multi-cycle ROI event predictor.

This reuses the exported predictor feature table and the same leave-event-cycle
folds. Labels are permuted within each event-reference cycle so the null keeps
the small per-cycle class balance and fold geometry intact.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SELECTION_ARTIFACT_FEATURES = {
    "stage_drift_xy_sampled",
    "validation_score",
    "front_quality_score",
}


def finite_metric(func, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(func(y_true, y_score))
    except Exception:
        return np.nan


def make_models(random_state: int) -> Dict[str, object]:
    return {
        "logistic_l2": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }


def feature_sets(features: List[str]) -> Dict[str, List[str]]:
    return {
        "all_physics_plus_qc": features,
        "physics_no_selection_qc": [f for f in features if f not in SELECTION_ARTIFACT_FEATURES],
    }


def permute_labels_within_cycle(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    y_perm = df["target_event_roi"].astype(int).to_numpy().copy()
    for _, idx in df.groupby("event_reference_cycle").groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        y_perm[idx_arr] = rng.permutation(y_perm[idx_arr])
    return y_perm


def score_folds(df: pd.DataFrame, feature_map: Dict[str, List[str]], y: np.ndarray, random_state: int) -> pd.DataFrame:
    rows = []
    for feature_set_name, cols in feature_map.items():
        x_all = df[cols].apply(pd.to_numeric, errors="coerce")
        for holdout_ref in sorted(df["event_reference_cycle"].dropna().unique()):
            train_mask = (df["event_reference_cycle"] != holdout_ref).to_numpy()
            test_mask = (df["event_reference_cycle"] == holdout_ref).to_numpy()
            y_train = y[train_mask]
            y_test = y[test_mask]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            for model_name, model in make_models(random_state).items():
                model.fit(x_all.loc[train_mask], y_train)
                score = model.predict_proba(x_all.loc[test_mask])[:, 1]
                pred = (score >= 0.5).astype(int)
                rows.append({
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "holdout_event_reference_cycle": float(holdout_ref),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "roc_auc": finite_metric(roc_auc_score, y_test, score),
                    "average_precision": finite_metric(average_precision_score, y_test, score),
                })
    return pd.DataFrame(rows)


def summarize_metrics(metric_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metric_df.groupby(["feature_set", "model"], as_index=False)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            folds=("roc_auc", "count"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_event_predictor")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_event_predictor_null")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=29)
    args = parser.parse_args()

    feature_path = os.path.join(args.predictor_dir, "multi_cycle_roi_event_predictor_features.csv")
    observed_path = os.path.join(args.predictor_dir, "leave_event_cycle_out_metrics.csv")
    summary_path = os.path.join(args.predictor_dir, "multi_cycle_roi_event_predictor_summary.json")
    df = pd.read_csv(feature_path)
    observed_fold = pd.read_csv(observed_path)
    with open(summary_path) as f:
        predictor_summary = json.load(f)

    features = [f for f in predictor_summary["features"] if f in df.columns]
    feature_map = feature_sets(features)
    observed_summary = summarize_metrics(observed_fold)

    rng = np.random.default_rng(args.random_state)
    null_rows = []
    for permutation_index in range(args.n_permutations):
        y_perm = permute_labels_within_cycle(df, rng)
        perm_scores = score_folds(df, feature_map, y_perm, args.random_state + permutation_index + 1)
        perm_summary = summarize_metrics(perm_scores)
        perm_summary["permutation_index"] = permutation_index
        null_rows.append(perm_summary)
    null_summary = pd.concat(null_rows, ignore_index=True)

    comparison_rows = []
    for _, obs in observed_summary.iterrows():
        mask = (null_summary["feature_set"] == obs["feature_set"]) & (null_summary["model"] == obs["model"])
        null_group = null_summary.loc[mask]
        row = obs.to_dict()
        for metric in ["mean_balanced_accuracy", "mean_roc_auc", "mean_average_precision"]:
            null_vals = null_group[metric].dropna().to_numpy()
            obs_val = float(obs[metric])
            row[f"null_{metric}_mean"] = float(np.mean(null_vals))
            row[f"null_{metric}_std"] = float(np.std(null_vals, ddof=1))
            row[f"p_ge_observed_{metric}"] = float((np.sum(null_vals >= obs_val) + 1) / (len(null_vals) + 1))
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    null_path = os.path.join(args.out_dir, "permutation_null_model_summary.csv")
    comp_path = os.path.join(args.out_dir, "observed_vs_permutation_null.csv")
    null_summary.to_csv(null_path, index=False)
    comparison.to_csv(comp_path, index=False)

    out_summary = {
        "predictor_dir": args.predictor_dir,
        "n_roi": int(len(df)),
        "n_permutations": int(args.n_permutations),
        "label_permutation": "within event_reference_cycle",
        "observed_vs_null": comparison.to_dict(orient="records"),
        "guardrail": (
            "Permutation p-values test whether the small 52-ROI cohort predictor exceeds "
            "same-fold, same-cycle-class-balance shuffled labels. They are descriptive, not a final detector validation."
        ),
        "outputs": {
            "permutation_null_model_summary": null_path,
            "observed_vs_permutation_null": comp_path,
        },
    }
    out_json = os.path.join(args.out_dir, "multi_cycle_roi_event_predictor_null_summary.json")
    with open(out_json, "w") as f:
        json.dump(out_summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Multi-Cycle ROI Event Predictor Null\n\n")
        f.write("Within-cycle label-permutation null for leave-event-cycle-out ROI event/control prediction.\n")
    print(json.dumps(out_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
