#!/usr/bin/env python3
"""Audit a balanced direct-video future-drop ROI cohort.

This joins class-balanced ROI reconstruction rows, exported particle-region crop
manifests, threshold-robust front proxies, and masked rollout residuals. It
reports ROI-level tests, cycle-collapsed tests, grouped leave-cycle-out models,
and a small label-permutation null for the best grouped model.

All labels are weak cycle-level future-drop labels projected onto automatic ROI
candidates. This is a review-prioritization and physics-feature audit, not a
manual-QC label set or deployable detector.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clean_value(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_value(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    return x


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val, errors="coerce")


def add_roi_id(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "roi_id" not in out.columns:
        out["roi_id"] = out.apply(
            lambda r: f"cycle{int(float(r['cycleNo']))}_rank{int(float(r['front_candidate_rank']))}_obj{int(float(r['object_candidate_rank']))}",
            axis=1,
        )
    return out


def pivot_rollout(rollout: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "particle_mse_mean",
        "particle_to_nonparticle_mse_ratio_mean",
        "particle_mse_fraction_of_full_mean",
        "particle_mask_area_fraction_mean",
        "mask_fallback_used_mean",
        "accepted_area_fraction_median",
    ]
    pieces = []
    for method, grp in rollout.groupby("method", dropna=False):
        cols = [c for c in keep if c in grp.columns]
        if not cols:
            continue
        renamed = grp[["roi_id"] + cols].copy().rename(columns={c: f"{method}_{c}" for c in cols})
        pieces.append(renamed)
    if not pieces:
        return pd.DataFrame({"roi_id": []})
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="roi_id", how="outer")
    return out


def binary_tests(df: pd.DataFrame, features: List[str], target: str, n_perm: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    y = numeric(df, target)
    for feature in unique(features):
        x = numeric(df, feature)
        mask = y.isin([0, 1]) & np.isfinite(x)
        pos = x[mask & (y == 1)].to_numpy(float)
        neg = x[mask & (y == 0)].to_numpy(float)
        if len(pos) and len(neg):
            diff = float(np.nanmedian(pos) - np.nanmedian(neg))
            auc = float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
            try:
                _, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
            except Exception:
                p_value = np.nan
            pooled = np.concatenate([pos, neg])
            perm = []
            for _ in range(n_perm):
                shuffled = rng.permutation(pooled)
                perm.append(float(np.nanmedian(shuffled[: len(pos)]) - np.nanmedian(shuffled[len(pos) :])))
            perm = np.asarray(perm, dtype=float)
            perm_p = float((np.sum(np.abs(perm) >= abs(diff)) + 1) / (len(perm) + 1)) if len(perm) else np.nan
        else:
            diff = auc = p_value = perm_p = np.nan
        rows.append({
            "target": target,
            "feature": feature,
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "median_positive_minus_negative": diff,
            "oriented_auc": abs(auc - 0.5) + 0.5 if np.isfinite(auc) else np.nan,
            "mannwhitney_p": finite(p_value),
            "permutation_p_abs_median_diff": perm_p,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["permutation_p_abs_median_diff", "mannwhitney_p"], na_position="last")


def correlations(df: pd.DataFrame, xs: List[str], ys: List[str]) -> pd.DataFrame:
    rows = []
    for x_name in unique(xs):
        x = numeric(df, x_name)
        for y_name in unique(ys):
            if x_name == y_name:
                continue
            y = numeric(df, y_name)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) >= 4 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
                rho, p = spearmanr(x[mask], y[mask])
            else:
                rho, p = np.nan, np.nan
            rows.append({"x": x_name, "y": y_name, "n": int(mask.sum()), "spearman_rho": finite(rho), "p_value": finite(p)})
    out = pd.DataFrame(rows)
    out["abs_rho"] = pd.to_numeric(out["spearman_rho"], errors="coerce").abs()
    return out.sort_values(["p_value", "abs_rho"], ascending=[True, False], na_position="last")


def cycle_collapse(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    agg = {f: "median" for f in features if f in df.columns}
    keep_first = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "any_abrupt_drop", "selection_subrole", "transferred_masked_residual_signature"]
    for col in keep_first:
        if col in df.columns:
            agg[col] = "first"
    out = df.groupby("cycleNo", as_index=False).agg(agg)
    out[target] = numeric(out, target)
    return out


def make_model(name: str, seed: int):
    if name == "logistic_l2":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=2))
    raise ValueError(name)


def grouped_oof(df: pd.DataFrame, features: List[str], target: str, group_col: str, seed: int, y_override: np.ndarray | None = None) -> pd.DataFrame:
    y = numeric(df, target).to_numpy(float) if y_override is None else y_override.astype(float)
    groups = df[group_col].to_numpy()
    X = df[features].copy()
    rows = []
    for model_name in ["logistic_l2", "random_forest"]:
        pred = np.full(len(df), np.nan, dtype=float)
        for group in pd.unique(groups):
            test = groups == group
            train = ~test
            ok_train = train & np.isin(y, [0, 1])
            ok_test = test & np.isin(y, [0, 1])
            if len(np.unique(y[ok_train])) < 2 or not ok_test.any():
                continue
            model = make_model(model_name, seed)
            model.fit(X.loc[ok_train], y[ok_train].astype(int))
            pred[ok_test] = model.predict_proba(X.loc[ok_test])[:, 1]
        mask = np.isfinite(pred) & np.isin(y, [0, 1])
        y_eval = y[mask].astype(int)
        if len(np.unique(y_eval)) == 2:
            auc = float(roc_auc_score(y_eval, pred[mask]))
            ap = float(average_precision_score(y_eval, pred[mask]))
        else:
            auc = ap = np.nan
        rows.append({
            "model": model_name,
            "group_col": group_col,
            "n_scored": int(mask.sum()),
            "n_positive_scored": int((y_eval == 1).sum()) if len(y_eval) else 0,
            "n_negative_scored": int((y_eval == 0).sum()) if len(y_eval) else 0,
            "pooled_oof_roc_auc": auc,
            "pooled_oof_average_precision": ap,
        })
    return pd.DataFrame(rows)


def permutation_null(df: pd.DataFrame, features: List[str], target: str, group_col: str, observed_model: str, observed_auc: float, n_perm: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = numeric(df, target).to_numpy(float)
    groups = df[group_col].to_numpy()
    valid_groups = pd.Series(groups).drop_duplicates().to_numpy()
    group_labels = {g: y[groups == g][np.isin(y[groups == g], [0, 1])][0] for g in valid_groups if np.isin(y[groups == g], [0, 1]).any()}
    group_vals = np.array([group_labels[g] for g in group_labels], dtype=float)
    aucs = []
    for _ in range(n_perm):
        shuffled = rng.permutation(group_vals)
        mapping = dict(zip(group_labels.keys(), shuffled))
        y_perm = np.array([mapping.get(g, np.nan) for g in groups], dtype=float)
        met = grouped_oof(df, features, target, group_col, seed, y_override=y_perm)
        row = met[met["model"] == observed_model]
        if not row.empty and np.isfinite(row.iloc[0]["pooled_oof_roc_auc"]):
            aucs.append(float(row.iloc[0]["pooled_oof_roc_auc"]))
    arr = np.asarray(aucs, dtype=float)
    return {
        "model": observed_model,
        "group_col": group_col,
        "n_permutation": int(n_perm),
        "n_valid_permutation": int(len(arr)),
        "observed_auc": finite(observed_auc),
        "null_auc_mean": float(np.nanmean(arr)) if len(arr) else None,
        "null_auc_p95": float(np.nanpercentile(arr, 95)) if len(arr) else None,
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)) if len(arr) and np.isfinite(observed_auc) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_physics_audit")
    parser.add_argument("--n-permutation", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    roi = add_roi_id(read_csv(derived / "balanced_future_roi_reconstruction" / "balanced_future_roi_table.csv"))
    manifest = read_csv(derived / "balanced_future_roi_sequences" / "selected_roi_sequence_manifest.csv")
    front = read_csv(derived / "balanced_future_threshold_robust_fronts" / "threshold_robust_front_summary.csv")
    rollout = read_csv(derived / "balanced_future_masked_roi_rollout_audit" / "masked_roi_rollout_per_roi_metrics.csv")

    roi_keep = ["roi_id", "cycleNo", "source_stem", "local_cycle_index", "front_candidate_rank", "object_candidate_rank", "validation_score", "cohort_role", "selection_subrole", "balanced_cycle_rank", "transferred_masked_residual_signature", "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "any_abrupt_drop", "object_mean_abs_z", "object_mean_residual"]
    manifest_keep = ["roi_id", "roi_mean_delta_last_minus_first", "roi_norm_mean_delta_last_minus_first", "stage_drift_xy_sampled", "first_frame_index", "last_frame_index"]
    joined = front.merge(roi[[c for c in roi_keep if c in roi.columns]], on="roi_id", how="left", suffixes=("", "_recon"))
    joined = joined.merge(manifest[[c for c in manifest_keep if c in manifest.columns]], on="roi_id", how="left")
    joined = joined.merge(pivot_rollout(rollout), on="roi_id", how="left")
    joined = joined.loc[:, ~joined.columns.duplicated()].copy()

    feature_candidates = [
        "phase_slope_median_per_s", "phase_slope_abs_median_per_s", "phase_slope_positive_fraction", "radius2_slope_median_px2_per_s", "radius2_slope_positive_fraction", "diffusion_proxy_median_um2_per_s", "diffusion_proxy_abs_median_um2_per_s", "threshold_robust_phase_score", "threshold_robust_diffusion_score", "q70_phase_slope_bootstrap_p50", "q70_radius2_slope_bootstrap_p50_px2_per_s", "roi_mean_delta_last_minus_first", "roi_norm_mean_delta_last_minus_first", "stage_drift_xy_sampled", "validation_score", "object_mean_abs_z", "object_mean_residual", "transferred_masked_residual_signature", "persistence_particle_mse_mean", "persistence_particle_to_nonparticle_mse_ratio_mean", "persistence_particle_mse_fraction_of_full_mean", "low_rank_dmd_particle_mse_mean", "low_rank_dmd_particle_to_nonparticle_mse_ratio_mean", "low_rank_dmd_particle_mse_fraction_of_full_mean", "velocity_particle_mse_mean", "velocity_particle_to_nonparticle_mse_ratio_mean", "velocity_particle_mse_fraction_of_full_mean", "mask_fallback_used_mean",
    ]
    features = [c for c in unique(feature_candidates) if c in joined.columns and numeric(joined, c).notna().any()]
    target = "future_any_drop_within_8cycles"
    label_counts = joined.groupby(["cohort_role", target], dropna=False).size().reset_index(name="n")
    cycle_table = cycle_collapse(joined, features, target)

    roi_tests = binary_tests(joined, features, target, args.n_permutation, rng)
    cycle_tests = binary_tests(cycle_table, features, target, args.n_permutation, rng)
    corr = correlations(joined, ["transferred_masked_residual_signature", "stage_drift_xy_sampled", "validation_score"], features)
    oof = grouped_oof(joined.dropna(subset=["cycleNo"]), features, target, "cycleNo", args.random_state)
    best = oof.sort_values("pooled_oof_roc_auc", ascending=False, na_position="last").head(1)
    null = {}
    if not best.empty and np.isfinite(best.iloc[0]["pooled_oof_roc_auc"]):
        null = permutation_null(joined.dropna(subset=["cycleNo"]), features, target, "cycleNo", str(best.iloc[0]["model"]), float(best.iloc[0]["pooled_oof_roc_auc"]), args.n_permutation, args.random_state + 17)

    joined.to_csv(out / "balanced_future_roi_physics_joined.csv", index=False)
    cycle_table.to_csv(out / "balanced_future_cycle_collapsed_features.csv", index=False)
    roi_tests.to_csv(out / "balanced_future_roi_feature_tests.csv", index=False)
    cycle_tests.to_csv(out / "balanced_future_cycle_feature_tests.csv", index=False)
    corr.to_csv(out / "balanced_future_feature_correlations.csv", index=False)
    oof.to_csv(out / "balanced_future_leave_cycle_oof_metrics.csv", index=False)

    summary = {
        "n_roi": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "n_features": int(len(features)),
        "label_counts": clean_value(label_counts.to_dict("records")),
        "cycle_group_oof_metrics": clean_value(oof.to_dict("records")),
        "permutation_null": clean_value(null),
        "top_roi_feature_tests": clean_value(roi_tests.head(16).to_dict("records")),
        "top_cycle_feature_tests": clean_value(cycle_tests.head(16).to_dict("records")),
        "top_correlations": clean_value(corr.head(16).to_dict("records")),
        "guardrail": "Balanced direct-video ROI audit uses weak cycle-level future8 labels projected to automatic particle-region candidates. Cycle-group splits and cycle-collapsed tests reduce leakage, but this is still review-prioritization evidence, not manual QC or deployable detection.",
    }
    with (out / "balanced_future_roi_physics_audit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (out / "README.md").open("w") as f:
        f.write("# Balanced Future-Drop ROI Physics Audit\n\n")
        f.write("Joins balanced direct-video ROI candidates to front proxies and masked rollout residuals.\n\n")
        f.write(f"- ROI rows: {summary['n_roi']}\n")
        f.write(f"- Cycles: {summary['n_cycles']}\n")
        f.write(f"- Features: {summary['n_features']}\n")
        f.write(f"- Label counts: {summary['label_counts']}\n\n")
        f.write(summary["guardrail"] + "\n")
    print(json.dumps(summary, indent=2)[:8000])


if __name__ == "__main__":
    main()
