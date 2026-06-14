#!/usr/bin/env python3
"""Multi-cohort weak future-drop model from ROI video physics features.

This audit combines the original selected ROI cohort and the later
transfer-ranked ROI cohort into one direct-video feature table. It evaluates
whether particle-mask rollout residuals plus phase/front optical proxies can
rank cycles/ROIs with future abrupt drops under leakage-aware grouped splits.

Labels are weak cycle-level future-drop labels projected onto ROI rows. They
are useful for model development and review prioritization, not manual ground
truth degradation annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FRONT_FEATURES = [
    "phase_slope_median_per_s",
    "phase_slope_abs_median_per_s",
    "phase_slope_positive_fraction",
    "radius2_slope_median_px2_per_s",
    "radius2_slope_positive_fraction",
    "diffusion_proxy_median_um2_per_s",
    "diffusion_proxy_abs_median_um2_per_s",
    "threshold_robust_phase_score",
    "threshold_robust_diffusion_score",
    "q70_phase_slope_bootstrap_p50",
    "q70_radius2_slope_bootstrap_p50_px2_per_s",
]

ROI_FEATURES = [
    "roi_mean_delta_last_minus_first",
    "roi_norm_mean_delta_last_minus_first",
    "stage_drift_xy_sampled",
    "validation_score",
    "front_quality_score",
    "object_mean_abs_z",
    "object_mean_residual",
    "transferred_masked_residual_signature",
    "transfer_rank",
]

ROLLOUT_FEATURES = [
    "persistence_particle_mse_mean",
    "persistence_particle_to_nonparticle_mse_ratio_mean",
    "persistence_particle_mse_fraction_of_full_mean",
    "low_rank_dmd_particle_mse_mean",
    "low_rank_dmd_particle_to_nonparticle_mse_ratio_mean",
    "low_rank_dmd_particle_mse_fraction_of_full_mean",
    "velocity_particle_mse_mean",
    "velocity_particle_to_nonparticle_mse_ratio_mean",
    "velocity_particle_mse_fraction_of_full_mean",
    "mask_fallback_used_mean",
    "accepted_area_fraction_median",
]

CYCLE_FEATURES = [
    "cycle_state_pc1",
    "cycle_state_pc2",
    "cycle_state_pc3",
    "cycle_state_pc4",
    "degradation_state_axis",
    "state_step_norm",
    "axis_step",
    "physics_consistency_score",
    "front_direction_score",
    "optical_change_score",
    "rollout_residual_score",
    "kinetic_transition_score",
    "mode_taxonomy_score",
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val, errors="coerce")


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def pivot_rollout(rollout: pd.DataFrame) -> pd.DataFrame:
    if rollout.empty or "method" not in rollout.columns:
        return pd.DataFrame({"roi_id": []})
    keep = [
        "particle_mse_mean",
        "particle_to_nonparticle_mse_ratio_mean",
        "particle_mse_fraction_of_full_mean",
        "mask_fallback_used_mean",
        "accepted_area_fraction_median",
    ]
    pieces = []
    for method, grp in rollout.groupby("method", dropna=False):
        cols = [c for c in keep if c in grp.columns]
        part = grp[["roi_id"] + cols].copy()
        rename = {c: f"{method}_{c}" for c in cols}
        if "mask_fallback_used_mean" in rename:
            rename["mask_fallback_used_mean"] = f"{method}_mask_fallback_used_mean"
        if "accepted_area_fraction_median" in rename:
            rename["accepted_area_fraction_median"] = f"{method}_accepted_area_fraction_median"
        pieces.append(part.rename(columns=rename))
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="roi_id", how="outer")
    fallback_cols = [c for c in out.columns if c.endswith("_mask_fallback_used_mean")]
    area_cols = [c for c in out.columns if c.endswith("_accepted_area_fraction_median")]
    if fallback_cols:
        out["mask_fallback_used_mean"] = out[fallback_cols].mean(axis=1)
    if area_cols:
        out["accepted_area_fraction_median"] = out[area_cols].mean(axis=1)
    return out


def build_selected(derived: Path) -> pd.DataFrame:
    manifest = read_csv(derived / "selected_roi_sequences" / "selected_roi_sequence_manifest.csv")
    if manifest.empty:
        manifest = read_csv(derived / "multi_cycle_roi_sequences" / "selected_roi_sequence_manifest.csv")
    front = read_csv(derived / "multi_cycle_threshold_robust_fronts" / "threshold_robust_front_summary.csv")
    bridge = read_csv(derived / "cycle_state_roi_bridge" / "cycle_state_roi_bridge_joined.csv")
    rollout = pivot_rollout(read_csv(derived / "masked_roi_rollout_audit" / "masked_roi_rollout_per_roi_metrics.csv"))

    if manifest.empty:
        return pd.DataFrame()
    df = manifest.copy()
    df["video_cohort"] = "selected"
    for src in [front, bridge, rollout]:
        if not src.empty and "roi_id" in src.columns:
            df = df.merge(src.drop_duplicates("roi_id"), on="roi_id", how="left", suffixes=("", "_src"))
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["weak_future_drop_label"] = numeric(df, "future_any_drop_within_8cycles")
    df["weak_future_drop_label_source"] = "cycle_state_roi_bridge_future8"
    return df


def build_transfer(derived: Path) -> pd.DataFrame:
    joined = read_csv(derived / "transfer_ranked_front_physics_audit" / "transfer_ranked_front_physics_joined.csv")
    if joined.empty:
        return pd.DataFrame()
    df = joined.copy()
    df["video_cohort"] = "transfer_ranked"
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["weak_future_drop_label"] = numeric(df, "future_any_drop_within_8cycles")
    df["weak_future_drop_label_source"] = "transfer_ranked_cycle_trace_future8"
    return df


def orient_auc(y: np.ndarray, score: np.ndarray) -> float:
    auc = roc_auc_score(y, score)
    return float(max(auc, 1.0 - auc))


def feature_tests(df: pd.DataFrame, features: List[str], label_col: str) -> pd.DataFrame:
    rows = []
    y = numeric(df, label_col)
    for feature in features:
        x = numeric(df, feature)
        mask = y.isin([0, 1]) & np.isfinite(x)
        pos = x[mask & y.eq(1)].to_numpy(dtype=float)
        neg = x[mask & y.eq(0)].to_numpy(dtype=float)
        if len(pos) and len(neg):
            try:
                _, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
            except Exception:
                p_value = np.nan
            auc = orient_auc(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
            diff = float(np.nanmedian(pos) - np.nanmedian(neg))
        else:
            p_value = np.nan
            auc = np.nan
            diff = np.nan
        rows.append({
            "feature": feature,
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "median_positive_minus_negative": diff,
            "oriented_auc": auc,
            "mannwhitney_p": float(p_value) if np.isfinite(p_value) else np.nan,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["mannwhitney_p", "oriented_auc"], ascending=[True, False], na_position="last")


def make_models(random_state: int, n_estimators: int) -> Dict[str, Pipeline]:
    return {
        "logistic_l2": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            )),
        ]),
    }


def grouped_oof(df: pd.DataFrame, features: List[str], group_col: str, random_state: int, n_estimators: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    data = df[df["weak_future_drop_label"].isin([0, 1])].copy()
    data = data[np.isfinite(numeric(data, group_col))]
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    x_all = data[features].apply(pd.to_numeric, errors="coerce")
    y_all = data["weak_future_drop_label"].astype(int).to_numpy()
    groups = data[group_col].to_numpy()

    for model_name in make_models(random_state, n_estimators):
        prob = np.full(len(data), np.nan, dtype=float)
        fold_status = []
        for group in sorted(pd.Series(groups).dropna().unique()):
            test = groups == group
            train = ~test
            y_train = y_all[train]
            y_test = y_all[test]
            status = {
                "model": model_name,
                "group_col": group_col,
                "holdout_group": float(group),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "n_positive_train": int(y_train.sum()),
                "n_negative_train": int(len(y_train) - y_train.sum()),
                "n_positive_test": int(y_test.sum()),
                "n_negative_test": int(len(y_test) - y_test.sum()),
            }
            if len(np.unique(y_train)) < 2:
                status["fold_status"] = "missing_train_class"
            else:
                model = make_models(random_state, n_estimators)[model_name]
                model.fit(x_all.iloc[train], y_train)
                prob[test] = model.predict_proba(x_all.iloc[test])[:, 1]
                status["fold_status"] = "scored"
            fold_status.append(status)
        scored = np.isfinite(prob)
        if scored.sum() and len(np.unique(y_all[scored])) == 2:
            auc = roc_auc_score(y_all[scored], prob[scored])
            ap = average_precision_score(y_all[scored], prob[scored])
        else:
            auc = np.nan
            ap = np.nan
        rows.append({
            "model": model_name,
            "group_col": group_col,
            "n_scored": int(scored.sum()),
            "n_positive_scored": int(y_all[scored].sum()) if scored.sum() else 0,
            "n_negative_scored": int(scored.sum() - y_all[scored].sum()) if scored.sum() else 0,
            "pooled_oof_roc_auc": float(auc) if np.isfinite(auc) else np.nan,
            "pooled_oof_average_precision": float(ap) if np.isfinite(ap) else np.nan,
            "n_scored_folds": int(sum(r["fold_status"] == "scored" for r in fold_status)),
            "n_total_folds": int(len(fold_status)),
        })
        for pos, (_, row) in enumerate(data.iterrows()):
            pred_rows.append({
                "roi_id": row["roi_id"],
                "video_cohort": row["video_cohort"],
                "cycleNo": row["cycleNo"],
                "model": model_name,
                "group_col": group_col,
                "weak_future_drop_label": int(row["weak_future_drop_label"]),
                "oof_probability": float(prob[pos]) if np.isfinite(prob[pos]) else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def leave_cohort_eval(df: pd.DataFrame, features: List[str], random_state: int, n_estimators: int) -> pd.DataFrame:
    rows = []
    data = df[df["weak_future_drop_label"].isin([0, 1])].copy()
    x_all = data[features].apply(pd.to_numeric, errors="coerce")
    y_all = data["weak_future_drop_label"].astype(int).to_numpy()
    cohorts = data["video_cohort"].astype(str).to_numpy()
    for train_cohort in sorted(data["video_cohort"].dropna().unique()):
        train = cohorts == train_cohort
        for test_cohort in sorted(data["video_cohort"].dropna().unique()):
            if test_cohort == train_cohort:
                continue
            test = cohorts == test_cohort
            y_train = y_all[train]
            y_test = y_all[test]
            for model_name in make_models(random_state, n_estimators):
                row = {
                    "model": model_name,
                    "train_cohort": train_cohort,
                    "test_cohort": test_cohort,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "n_positive_train": int(y_train.sum()),
                    "n_negative_train": int(len(y_train) - y_train.sum()),
                    "n_positive_test": int(y_test.sum()),
                    "n_negative_test": int(len(y_test) - y_test.sum()),
                }
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    row.update({"status": "missing_class", "roc_auc": np.nan, "average_precision": np.nan})
                else:
                    model = make_models(random_state, n_estimators)[model_name]
                    model.fit(x_all.iloc[train], y_train)
                    prob = model.predict_proba(x_all.iloc[test])[:, 1]
                    row.update({
                        "status": "scored",
                        "roc_auc": float(roc_auc_score(y_test, prob)),
                        "average_precision": float(average_precision_score(y_test, prob)),
                    })
                rows.append(row)
    return pd.DataFrame(rows)


def permutation_null(df: pd.DataFrame, features: List[str], observed_auc: float, group_col: str, model_name: str, n_perm: int, random_state: int, n_estimators: int) -> Dict[str, Any]:
    if not np.isfinite(observed_auc) or n_perm <= 0:
        return {"status": "not_run"}
    rng = np.random.default_rng(random_state)
    data = df[df["weak_future_drop_label"].isin([0, 1])].copy()
    labels = data["weak_future_drop_label"].astype(int).to_numpy()
    aucs = []
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        tmp = data.copy()
        tmp["weak_future_drop_label"] = shuffled
        metrics, _ = grouped_oof(tmp, features, group_col, random_state, n_estimators)
        vals = metrics[(metrics["model"] == model_name)]["pooled_oof_roc_auc"].dropna()
        if not vals.empty:
            aucs.append(float(vals.iloc[0]))
    arr = np.asarray(aucs, dtype=float)
    if arr.size == 0:
        return {"status": "no_valid_permutations"}
    return {
        "status": "ok",
        "model": model_name,
        "group_col": group_col,
        "n_permutation": int(arr.size),
        "observed_auc": float(observed_auc),
        "null_auc_mean": float(np.nanmean(arr)),
        "null_auc_p95": float(np.nanpercentile(arr, 95)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (arr.size + 1)),
    }


def fit_feature_importance(df: pd.DataFrame, features: List[str], random_state: int, n_estimators: int) -> pd.DataFrame:
    data = df[df["weak_future_drop_label"].isin([0, 1])].copy()
    x = data[features].apply(pd.to_numeric, errors="coerce")
    y = data["weak_future_drop_label"].astype(int).to_numpy()
    rows = []
    if len(np.unique(y)) < 2:
        return pd.DataFrame()
    for model_name, model in make_models(random_state, n_estimators).items():
        model.fit(x, y)
        if model_name == "logistic_l2":
            values = np.abs(model.named_steps["clf"].coef_[0])
        else:
            values = model.named_steps["clf"].feature_importances_
        for feature, value in zip(features, values):
            rows.append({"model": model_name, "feature": feature, "importance": float(value)})
    return pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/multicohort_future_drop_model")
    parser.add_argument("--n-permutation", type=int, default=200)
    parser.add_argument("--rf-trees", type=int, default=120)
    parser.add_argument("--random-state", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cohorts = [build_selected(derived), build_transfer(derived)]
    cohorts = [c.loc[:, ~c.columns.duplicated()].copy() for c in cohorts if not c.empty]
    df = pd.concat(cohorts, ignore_index=True, sort=False) if cohorts else pd.DataFrame()
    if df.empty:
        raise RuntimeError("No ROI rows available")
    for col in ["cycleNo", "weak_future_drop_label"]:
        df[col] = numeric(df, col)
    features = unique([c for c in FRONT_FEATURES + ROI_FEATURES + ROLLOUT_FEATURES + CYCLE_FEATURES if c in df.columns])
    for col in features:
        df[col] = numeric(df, col)
    df = df.drop_duplicates(["video_cohort", "roi_id"]).copy()

    label_counts = df.groupby("video_cohort")["weak_future_drop_label"].value_counts(dropna=False).rename("n").reset_index()
    feature_test_all = feature_tests(df, features, "weak_future_drop_label")
    feature_tests_by_cohort = []
    for cohort, sub in df.groupby("video_cohort", dropna=False):
        tmp = feature_tests(sub, features, "weak_future_drop_label")
        tmp["video_cohort"] = cohort
        feature_tests_by_cohort.append(tmp)
    feature_test_cohort = pd.concat(feature_tests_by_cohort, ignore_index=True) if feature_tests_by_cohort else pd.DataFrame()

    cycle_metrics, oof_pred = grouped_oof(df, features, "cycleNo", args.random_state, args.rf_trees)
    cohort_eval = leave_cohort_eval(df, features, args.random_state, args.rf_trees)
    best = cycle_metrics.sort_values("pooled_oof_roc_auc", ascending=False, na_position="last").head(1)
    if not best.empty:
        null = permutation_null(
            df,
            features,
            float(best.iloc[0]["pooled_oof_roc_auc"]),
            "cycleNo",
            str(best.iloc[0]["model"]),
            args.n_permutation,
            args.random_state,
            args.rf_trees,
        )
    else:
        null = {"status": "no_oof_model"}
    importance = fit_feature_importance(df, features, args.random_state, args.rf_trees)

    cycle_summary = df.groupby(["video_cohort", "cycleNo"], dropna=False).agg(
        n_roi=("roi_id", "count"),
        weak_future_drop_label=("weak_future_drop_label", "mean"),
        phase_slope_median_per_s=("phase_slope_median_per_s", "mean"),
        radius2_slope_median_px2_per_s=("radius2_slope_median_px2_per_s", "mean"),
        persistence_particle_mse_mean=("persistence_particle_mse_mean", "mean"),
        low_rank_dmd_particle_mse_mean=("low_rank_dmd_particle_mse_mean", "mean"),
        threshold_robust_phase_score=("threshold_robust_phase_score", "mean"),
    ).reset_index()

    paths = {
        "feature_table": out / "multicohort_future_drop_feature_table.csv",
        "label_counts": out / "multicohort_future_drop_label_counts.csv",
        "feature_tests": out / "multicohort_future_drop_feature_tests.csv",
        "feature_tests_by_cohort": out / "multicohort_future_drop_feature_tests_by_cohort.csv",
        "cycle_oof_metrics": out / "multicohort_future_drop_cycle_oof_metrics.csv",
        "oof_predictions": out / "multicohort_future_drop_oof_predictions.csv",
        "leave_cohort_eval": out / "multicohort_future_drop_leave_cohort_eval.csv",
        "feature_importance": out / "multicohort_future_drop_feature_importance.csv",
        "cycle_summary": out / "multicohort_future_drop_cycle_summary.csv",
    }
    df.to_csv(paths["feature_table"], index=False)
    label_counts.to_csv(paths["label_counts"], index=False)
    feature_test_all.to_csv(paths["feature_tests"], index=False)
    feature_test_cohort.to_csv(paths["feature_tests_by_cohort"], index=False)
    cycle_metrics.to_csv(paths["cycle_oof_metrics"], index=False)
    oof_pred.to_csv(paths["oof_predictions"], index=False)
    cohort_eval.to_csv(paths["leave_cohort_eval"], index=False)
    importance.to_csv(paths["feature_importance"], index=False)
    cycle_summary.to_csv(paths["cycle_summary"], index=False)

    summary = {
        "n_roi_rows": int(len(df)),
        "n_selected_rows": int((df["video_cohort"] == "selected").sum()),
        "n_transfer_ranked_rows": int((df["video_cohort"] == "transfer_ranked").sum()),
        "n_features": int(len(features)),
        "rf_trees": int(args.rf_trees),
        "features": features,
        "label_counts": clean_json(label_counts.to_dict("records")),
        "top_feature_tests": clean_json(feature_test_all.head(15).to_dict("records")),
        "top_feature_tests_by_cohort": clean_json(feature_test_cohort.sort_values(["mannwhitney_p", "oriented_auc"], ascending=[True, False], na_position="last").head(20).to_dict("records")) if not feature_test_cohort.empty else [],
        "cycle_group_oof_metrics": clean_json(cycle_metrics.to_dict("records")),
        "leave_cohort_eval": clean_json(cohort_eval.to_dict("records")),
        "permutation_null": clean_json(null),
        "top_feature_importance": clean_json(importance.groupby("model", group_keys=False).head(12).to_dict("records")) if not importance.empty else [],
        "top_cycle_summary": clean_json(cycle_summary.sort_values(["weak_future_drop_label", "radius2_slope_median_px2_per_s"], ascending=[False, False]).head(15).to_dict("records")),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": (
            "Future-drop labels are weak cycle-level labels projected onto ROI rows. "
            "Grouped splits reduce cycle leakage, but this is a review-prioritization model, "
            "not a deployment-ready degradation detector or manual-QC label set."
        ),
    }
    with (out / "multicohort_future_drop_model_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)
    with (out / "README.md").open("w") as f:
        f.write("# Multi-Cohort Future-Drop Model\n\n")
        f.write("Weak future-drop model table combining selected and transfer-ranked ROI video physics features.\n\n")
        f.write(f"- ROI rows: {summary['n_roi_rows']}\n")
        f.write(f"- Features: {summary['n_features']}\n")
        f.write(f"- Cycle-group OOF metrics: {summary['cycle_group_oof_metrics']}\n\n")
        f.write(f"Guardrail: {summary['guardrail']}\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
