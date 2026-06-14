#!/usr/bin/env python3
"""Leave-event-cycle-out event/control prediction for expanded NMC ROIs.

The purpose is not to deploy an event detector. It is a leakage-guarded test of
whether physically motivated ROI descriptors and rollout/latent features
generalize across candidate degradation episodes.
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
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "roi_norm_mean_delta",
    "roi_norm_mean_slope",
    "roi_norm_mean_slope_r2",
    "high_fraction_delta",
    "high_fraction_slope",
    "low_fraction_delta",
    "low_fraction_slope",
    "high_radius2_delta_px2",
    "high_radius2_slope_px2_per_frame",
    "high_radius2_slope_um2_per_frame",
    "centroid_path_px",
    "first_last_corr",
    "cumulative_abs_norm_change",
    "temporal_diff_energy",
    "stage_drift_xy_sampled",
    "validation_score",
    "front_quality_score",
    "apparent_diffusion_proxy_ds_px2_per_frame",
]


LATENT_FEATURES = [
    "latent_path_length",
    "latent_mean_step",
    "latent_net_displacement",
    "latent_component0_delta",
    "latent_component1_delta",
]


ROLLOUT_FEATURES = [
    "persistence_mse",
    "persistence_mae",
    "persistence_ssim",
    "velocity_mse",
    "velocity_mae",
    "velocity_ssim",
    "low_rank_dmd_mse",
    "low_rank_dmd_mae",
    "low_rank_dmd_ssim",
    "dmd_minus_persistence_mse",
    "velocity_minus_persistence_mse",
]


def finite_metric(func, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(func(y_true, y_score))
    except Exception:
        return np.nan


def load_feature_table(descriptors: str, latent: str, rollout: str) -> pd.DataFrame:
    df = pd.read_csv(descriptors)
    latent_df = pd.read_csv(latent)
    roll = pd.read_csv(rollout)
    roll_pivot = roll.pivot_table(
        index="roi_id",
        columns="method",
        values=["mse", "mae", "ssim"],
        aggfunc="first",
    )
    roll_pivot.columns = [f"{method}_{metric}" for metric, method in roll_pivot.columns.to_flat_index()]
    roll_pivot = roll_pivot.reset_index()
    if {"low_rank_dmd_mse", "persistence_mse"}.issubset(roll_pivot.columns):
        roll_pivot["dmd_minus_persistence_mse"] = roll_pivot["low_rank_dmd_mse"] - roll_pivot["persistence_mse"]
    if {"velocity_mse", "persistence_mse"}.issubset(roll_pivot.columns):
        roll_pivot["velocity_minus_persistence_mse"] = roll_pivot["velocity_mse"] - roll_pivot["persistence_mse"]
    df = df.merge(latent_df[["roi_id"] + [c for c in LATENT_FEATURES if c in latent_df.columns]], on="roi_id", how="left")
    df = df.merge(roll_pivot, on="roi_id", how="left")
    df["target_event_roi"] = (df["cohort_role"] == "event").astype(int)
    return df


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


def feature_sets(all_features: List[str]) -> Dict[str, List[str]]:
    selection_artifact_features = {
        "stage_drift_xy_sampled",
        "validation_score",
        "front_quality_score",
    }
    return {
        "all_physics_plus_qc": all_features,
        "physics_no_selection_qc": [f for f in all_features if f not in selection_artifact_features],
    }


def feature_importance(model_name: str, model: object, feature_names: List[str]) -> pd.DataFrame:
    fitted = model.named_steps["model"]
    if model_name == "logistic_l2":
        vals = fitted.coef_[0]
        return pd.DataFrame({"model": model_name, "feature": feature_names, "importance": vals, "abs_importance": np.abs(vals)})
    vals = fitted.feature_importances_
    return pd.DataFrame({"model": model_name, "feature": feature_names, "importance": vals, "abs_importance": np.abs(vals)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/multi_cycle_roi_analysis")
    parser.add_argument("--rollout-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/multi_cycle_roi_rollout_baselines")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/multi_cycle_roi_event_predictor")
    parser.add_argument("--random-state", type=int, default=29)
    args = parser.parse_args()

    desc_path = os.path.join(args.analysis_dir, "multi_cycle_roi_descriptors.csv")
    latent_path = os.path.join(args.rollout_dir, "roi_latent_dynamics_summary.csv")
    rollout_path = os.path.join(args.rollout_dir, "roi_rollout_per_roi_metrics.csv")
    df = load_feature_table(desc_path, latent_path, rollout_path)
    all_feature_cols = [c for c in BASE_FEATURES + LATENT_FEATURES + ROLLOUT_FEATURES if c in df.columns]
    y_all = df["target_event_roi"].astype(int).to_numpy()

    folds = []
    predictions = []
    importances = []
    feature_set_map = feature_sets(all_feature_cols)
    for feature_set_name, feature_cols in feature_set_map.items():
        x_all = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        for holdout_ref in sorted(df["event_reference_cycle"].dropna().unique()):
            train_mask = df["event_reference_cycle"] != holdout_ref
            test_mask = df["event_reference_cycle"] == holdout_ref
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            y_train = y_all[train_mask.to_numpy()]
            y_test = y_all[test_mask.to_numpy()]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            for model_name, model in make_models(args.random_state).items():
                model.fit(x_all.loc[train_mask], y_train)
                score = model.predict_proba(x_all.loc[test_mask])[:, 1]
                pred = (score >= 0.5).astype(int)
                folds.append({
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "holdout_event_reference_cycle": float(holdout_ref),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "n_event_test": int(y_test.sum()),
                    "n_control_test": int((1 - y_test).sum()),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                    "roc_auc": finite_metric(roc_auc_score, y_test, score),
                    "average_precision": finite_metric(average_precision_score, y_test, score),
                })
                pred_rows = df.loc[test_mask, [
                    "roi_id", "cycleNo", "event_reference_cycle", "cohort_role",
                    "degradation_mode_hypothesis", "is_synchronized_reference",
                ]].copy()
                pred_rows["feature_set"] = feature_set_name
                pred_rows["model"] = model_name
                pred_rows["target_event_roi"] = y_test
                pred_rows["event_probability"] = score
                pred_rows["predicted_event_roi"] = pred
                predictions.append(pred_rows)
                imp = feature_importance(model_name, model, feature_cols)
                imp["feature_set"] = feature_set_name
                imp["holdout_event_reference_cycle"] = float(holdout_ref)
                importances.append(imp)

    fold_df = pd.DataFrame(folds)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    if not imp_df.empty:
        imp_summary = (
            imp_df.groupby(["feature_set", "model", "feature"], as_index=False)["abs_importance"]
            .mean()
            .sort_values(["feature_set", "model", "abs_importance"], ascending=[True, True, False])
        )
    else:
        imp_summary = pd.DataFrame()

    os.makedirs(args.out_dir, exist_ok=True)
    feature_path = os.path.join(args.out_dir, "multi_cycle_roi_event_predictor_features.csv")
    folds_path = os.path.join(args.out_dir, "leave_event_cycle_out_metrics.csv")
    pred_path = os.path.join(args.out_dir, "leave_event_cycle_out_predictions.csv")
    imp_path = os.path.join(args.out_dir, "feature_importance_by_fold.csv")
    imp_summary_path = os.path.join(args.out_dir, "feature_importance_summary.csv")
    df.to_csv(feature_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    imp_df.to_csv(imp_path, index=False)
    imp_summary.to_csv(imp_summary_path, index=False)

    model_summary = {}
    if not fold_df.empty:
        for (feature_set_name, model_name), g in fold_df.groupby(["feature_set", "model"]):
            model_summary[f"{feature_set_name}:{model_name}"] = {
                "mean_balanced_accuracy": float(g["balanced_accuracy"].mean()),
                "mean_roc_auc": float(g["roc_auc"].mean()),
                "mean_average_precision": float(g["average_precision"].mean()),
                "folds": int(len(g)),
            }
    summary = {
        "descriptor_source": desc_path,
        "latent_source": latent_path,
        "rollout_source": rollout_path,
        "n_roi": int(len(df)),
        "n_event_roi": int(df["target_event_roi"].sum()),
        "n_control_roi": int((1 - df["target_event_roi"]).sum()),
        "feature_count": int(len(all_feature_cols)),
        "features": all_feature_cols,
        "feature_sets": feature_set_map,
        "model_summary": model_summary,
        "fold_metrics": fold_df.to_dict(orient="records"),
        "top_features": imp_summary.head(20).to_dict(orient="records") if not imp_summary.empty else [],
        "guardrail": (
            "Leave-event-cycle-out prediction tests whether ROI physics descriptors generalize. "
            "It is not a deployable detector and excludes direct evidence/cycle labels to reduce leakage."
        ),
        "outputs": {
            "features": feature_path,
            "fold_metrics": folds_path,
            "predictions": pred_path,
            "feature_importance_by_fold": imp_path,
            "feature_importance_summary": imp_summary_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "multi_cycle_roi_event_predictor_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Multi-Cycle ROI Event Predictor\n\n")
        f.write("Leakage-guarded leave-event-cycle-out event/control prediction from ROI physics and rollout descriptors.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
