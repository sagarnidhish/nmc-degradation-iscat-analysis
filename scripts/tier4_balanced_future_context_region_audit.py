#!/usr/bin/env python3
"""Context and region guardrail for the balanced future-drop ROI audit.

The balanced future-drop audit found weak future8 signal in front-motion and
particle-mask rollout features. This follow-up asks whether that signal is
mostly explained by acquisition context: cycle number, source/local segment,
full-frame object position, stage drift, candidate score, or the transferred
cycle warning score.

Labels remain weak cycle-level future8 labels projected onto automatic ROI
candidates. This is a confound/region audit, not manual QC or deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def clean_value(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_value(v) for v in x]
    if isinstance(x, tuple):
        return [clean_value(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    try:
        if pd.isna(x):
            return None
    except TypeError:
        pass
    return x


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val, errors="coerce")


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def add_roi_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "roi_id" not in out.columns:
        out["roi_id"] = out.apply(
            lambda r: f"cycle{int(float(r['cycleNo']))}_rank{int(float(r['front_candidate_rank']))}_obj{int(float(r['object_candidate_rank']))}",
            axis=1,
        )
    return out


def onehot_context(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    pieces = []
    num = df[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=df.index)
    if not num.empty:
        pieces.append(num)
    cats = [c for c in categorical_cols if c in df.columns]
    if cats:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        arr = enc.fit_transform(df[cats].astype(str).fillna(""))
        pieces.append(pd.DataFrame(arr, index=df.index, columns=enc.get_feature_names_out(cats)))
    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces, axis=1)


def make_model(name: str, seed: int):
    if name == "logistic_l2":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=140, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=2))
    raise ValueError(name)


def grouped_oof(df: pd.DataFrame, features: List[str], target: str, group_col: str, feature_set: str, seed: int) -> pd.DataFrame:
    y = numeric(df, target).to_numpy(float)
    groups = df[group_col].to_numpy()
    x = df[features].copy()
    rows = []
    for model_name in ["logistic_l2", "random_forest"]:
        pred = np.full(len(df), np.nan, dtype=float)
        for group in pd.unique(groups):
            test = groups == group
            train = ~test
            train_ok_full = train & np.isin(y, [0, 1])
            test_ok_full = test & np.isin(y, [0, 1])
            y_train = y[train_ok_full]
            if len(np.unique(y_train)) < 2 or not test_ok_full.any():
                continue
            model = make_model(model_name, seed)
            model.fit(x.loc[train_ok_full], y_train.astype(int))
            pred[test_ok_full] = model.predict_proba(x.loc[test_ok_full])[:, 1]
        mask = np.isfinite(pred) & np.isin(y, [0, 1])
        y_eval = y[mask].astype(int)
        if len(np.unique(y_eval)) == 2:
            auc = float(roc_auc_score(y_eval, pred[mask]))
            ap = float(average_precision_score(y_eval, pred[mask]))
        else:
            auc = ap = np.nan
        rows.append({
            "feature_set": feature_set,
            "model": model_name,
            "group_col": group_col,
            "n_scored": int(mask.sum()),
            "n_positive_scored": int((y_eval == 1).sum()) if len(y_eval) else 0,
            "n_negative_scored": int((y_eval == 0).sum()) if len(y_eval) else 0,
            "pooled_oof_roc_auc": auc,
            "pooled_oof_average_precision": ap,
        })
    return pd.DataFrame(rows)


def residualize_features(df: pd.DataFrame, features: List[str], context: pd.DataFrame, target: str) -> pd.DataFrame:
    y_label = numeric(df, target)
    ctx = context.copy()
    out_rows = []
    residual_cols = []
    for feature in features:
        x = numeric(df, feature)
        mask = np.isfinite(x) & y_label.isin([0, 1])
        if int(mask.sum()) < 8:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        model.fit(ctx.loc[mask], x.loc[mask])
        pred = model.predict(ctx)
        resid = x - pred
        rcol = f"{feature}_context_residual"
        residual_cols.append(rcol)
        df[rcol] = resid
        pos = resid[y_label.eq(1) & np.isfinite(resid)]
        neg = resid[y_label.eq(0) & np.isfinite(resid)]
        if len(pos) and len(neg):
            try:
                _, p = mannwhitneyu(pos, neg, alternative="two-sided")
            except Exception:
                p = np.nan
            auc = float(np.mean(pos.to_numpy()[:, None] > neg.to_numpy()[None, :]) + 0.5 * np.mean(pos.to_numpy()[:, None] == neg.to_numpy()[None, :]))
            diff = float(np.nanmedian(pos) - np.nanmedian(neg))
        else:
            p = auc = diff = np.nan
        out_rows.append({
            "feature": feature,
            "residual_feature": rcol,
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "median_positive_minus_negative": diff,
            "oriented_auc": abs(auc - 0.5) + 0.5 if np.isfinite(auc) else np.nan,
            "mannwhitney_p": finite(p),
        })
    return pd.DataFrame(out_rows).sort_values(["mannwhitney_p", "oriented_auc"], ascending=[True, False], na_position="last")


def spatial_region_tests(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    y = numeric(df, target)
    for region_col in ["x_bin", "y_bin", "xy_region"]:
        if region_col not in df.columns:
            continue
        tab = pd.crosstab(df[region_col], y)
        if {0, 1}.issubset(set(tab.columns)) and tab.shape[0] >= 2:
            try:
                chi2, p, _, _ = chi2_contingency(tab[[0, 1]].to_numpy())
            except Exception:
                chi2, p = np.nan, np.nan
        else:
            chi2, p = np.nan, np.nan
        rows.append({"region_col": region_col, "n_regions": int(tab.shape[0]), "chi2": finite(chi2), "p_value": finite(p), "table": tab.to_dict()})
    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


def correlations(df: pd.DataFrame, xs: List[str], ys: List[str]) -> pd.DataFrame:
    rows = []
    for x_col in unique(xs):
        x = numeric(df, x_col)
        for y_col in unique(ys):
            y = numeric(df, y_col)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) >= 6 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
                rho, p = spearmanr(x[mask], y[mask])
            else:
                rho, p = np.nan, np.nan
            rows.append({"x": x_col, "y": y_col, "n": int(mask.sum()), "spearman_rho": finite(rho), "p_value": finite(p)})
    out = pd.DataFrame(rows)
    out["abs_rho"] = pd.to_numeric(out["spearman_rho"], errors="coerce").abs()
    return out.sort_values(["p_value", "abs_rho"], ascending=[True, False], na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_context_region_audit")
    parser.add_argument("--random-state", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = read_csv(derived / "balanced_future_roi_physics_audit" / "balanced_future_roi_physics_joined.csv")
    recon = add_roi_id(read_csv(derived / "balanced_future_roi_reconstruction" / "balanced_future_roi_table.csv"))
    mask = read_csv(derived / "balanced_future_particle_mask_stability" / "particle_mask_stability_per_roi.csv")

    spatial_cols = ["roi_id", "object_x_full_approx", "object_y_full_approx", "object_area_ds_px"]
    df = joined.merge(recon[[c for c in spatial_cols if c in recon.columns]], on="roi_id", how="left")
    mask_cols = ["roi_id", "fallback_frame_fraction", "accepted_area_cv", "accepted_centroid_path_px", "accepted_centroid_max_step_px", "mask_instability_score"]
    if not mask.empty:
        df = df.merge(mask[[c for c in mask_cols if c in mask.columns]], on="roi_id", how="left", suffixes=("", "_mask"))
    df = df.loc[:, ~df.columns.duplicated()].copy()

    target = "future_any_drop_within_8cycles"
    for col in ["object_x_full_approx", "object_y_full_approx"]:
        vals = numeric(df, col)
        if vals.notna().sum() >= 6:
            df[col.replace("object_", "").replace("_full_approx", "_bin")] = pd.qcut(vals.rank(method="first"), q=3, labels=["low", "mid", "high"])
    if {"x_bin", "y_bin"}.issubset(df.columns):
        df["xy_region"] = df["x_bin"].astype(str) + "_" + df["y_bin"].astype(str)

    physics_features = [c for c in [
        "radius2_slope_median_px2_per_s",
        "diffusion_proxy_median_um2_per_s",
        "q70_radius2_slope_bootstrap_p50_px2_per_s",
        "persistence_particle_mse_fraction_of_full_mean",
        "velocity_particle_mse_fraction_of_full_mean",
        "velocity_particle_to_nonparticle_mse_ratio_mean",
        "persistence_particle_to_nonparticle_mse_ratio_mean",
        "phase_slope_positive_fraction",
        "threshold_robust_phase_score",
        "threshold_robust_diffusion_score",
        "low_rank_dmd_particle_mse_fraction_of_full_mean",
        "accepted_area_cv",
        "mask_instability_score",
    ] if c in df.columns and numeric(df, c).notna().any()]
    acquisition_context_numeric = [c for c in [
        "local_cycle_index",
        "validation_score_recon",
        "stage_drift_xy_sampled",
        "object_x_full_approx",
        "object_y_full_approx",
        "object_area_ds_px",
        "first_frame_index",
        "last_frame_index",
    ] if c in df.columns and numeric(df, c).notna().any()]
    acquisition_context_categorical = [c for c in ["source_stem", "x_bin", "y_bin", "xy_region"] if c in df.columns]
    design_context_numeric = [c for c in [
        "cycleNo",
        "local_cycle_index",
        "balanced_cycle_rank",
        "transferred_masked_residual_signature",
        "validation_score_recon",
        "stage_drift_xy_sampled",
        "object_x_full_approx",
        "object_y_full_approx",
        "object_area_ds_px",
        "first_frame_index",
        "last_frame_index",
    ] if c in df.columns and numeric(df, c).notna().any()]
    design_context_categorical = [c for c in ["source_stem", "selection_subrole", "x_bin", "y_bin", "xy_region"] if c in df.columns]
    acquisition_context_matrix = onehot_context(df, acquisition_context_numeric, acquisition_context_categorical)
    design_context_matrix = onehot_context(df, design_context_numeric, design_context_categorical)
    acquisition_feature_cols = list(acquisition_context_matrix.columns)
    design_feature_cols = list(design_context_matrix.columns)
    model_df = pd.concat([
        df.reset_index(drop=True),
        acquisition_context_matrix.reset_index(drop=True).add_prefix("acqctx_"),
        design_context_matrix.reset_index(drop=True).add_prefix("designctx_"),
    ], axis=1)
    acquisition_model_cols = [f"acqctx_{c}" for c in acquisition_feature_cols]
    design_model_cols = [f"designctx_{c}" for c in design_feature_cols]

    model_rows = []
    for feature_set, cols in [
        ("acquisition_context_only", acquisition_model_cols),
        ("design_context_only", design_model_cols),
        ("physics_only", physics_features),
        ("physics_plus_acquisition_context", physics_features + acquisition_model_cols),
        ("physics_plus_design_context", physics_features + design_model_cols),
    ]:
        cols = [c for c in cols if c in model_df.columns and numeric(model_df, c).notna().any()]
        if cols:
            model_rows.append(grouped_oof(model_df.dropna(subset=["cycleNo"]), cols, target, "cycleNo", feature_set, args.random_state))
    oof = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()

    acquisition_residual_tests = residualize_features(model_df, physics_features, model_df[acquisition_model_cols], target) if acquisition_model_cols else pd.DataFrame()
    design_residual_tests = residualize_features(model_df, physics_features, model_df[design_model_cols], target) if design_model_cols else pd.DataFrame()
    region_tests = spatial_region_tests(df, target)
    context_corr = correlations(df, design_context_numeric, physics_features)

    cycle_summary = (
        df.groupby(["cycleNo", target], dropna=False)
        .agg(
            n_roi=("roi_id", "count"),
            x_median=("object_x_full_approx", "median"),
            y_median=("object_y_full_approx", "median"),
            warning_score=("transferred_masked_residual_signature", "first"),
            radius2_median=("radius2_slope_median_px2_per_s", "median"),
            persistence_fraction_median=("persistence_particle_mse_fraction_of_full_mean", "median"),
        )
        .reset_index()
        .sort_values(["cycleNo"])
    )

    df.to_csv(out / "balanced_future_context_region_joined.csv", index=False)
    oof.to_csv(out / "balanced_future_context_region_oof_metrics.csv", index=False)
    acquisition_residual_tests.to_csv(out / "balanced_future_acquisition_context_residual_feature_tests.csv", index=False)
    design_residual_tests.to_csv(out / "balanced_future_design_context_residual_feature_tests.csv", index=False)
    region_tests.to_csv(out / "balanced_future_spatial_region_tests.csv", index=False)
    context_corr.to_csv(out / "balanced_future_context_feature_correlations.csv", index=False)
    cycle_summary.to_csv(out / "balanced_future_context_cycle_summary.csv", index=False)

    best_acquisition_context = oof[oof["feature_set"] == "acquisition_context_only"].sort_values("pooled_oof_roc_auc", ascending=False, na_position="last").head(1).to_dict("records")
    best_design_context = oof[oof["feature_set"] == "design_context_only"].sort_values("pooled_oof_roc_auc", ascending=False, na_position="last").head(1).to_dict("records")
    best_physics = oof[oof["feature_set"] == "physics_only"].sort_values("pooled_oof_roc_auc", ascending=False, na_position="last").head(1).to_dict("records")
    best_acq_combined = oof[oof["feature_set"] == "physics_plus_acquisition_context"].sort_values("pooled_oof_roc_auc", ascending=False, na_position="last").head(1).to_dict("records")
    best_design_combined = oof[oof["feature_set"] == "physics_plus_design_context"].sort_values("pooled_oof_roc_auc", ascending=False, na_position="last").head(1).to_dict("records")
    summary: Dict[str, Any] = {
        "n_roi": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "label_counts": clean_value(df.groupby([target], dropna=False).size().reset_index(name="n").to_dict("records")),
        "n_physics_features": int(len(physics_features)),
        "n_acquisition_context_numeric_features": int(len(acquisition_context_numeric)),
        "n_acquisition_context_model_features": int(len(acquisition_model_cols)),
        "n_design_context_numeric_features": int(len(design_context_numeric)),
        "n_design_context_model_features": int(len(design_model_cols)),
        "physics_features": physics_features,
        "acquisition_context_numeric_features": acquisition_context_numeric,
        "acquisition_context_categorical_features": acquisition_context_categorical,
        "design_context_numeric_features": design_context_numeric,
        "design_context_categorical_features": design_context_categorical,
        "best_acquisition_context_only": clean_value(best_acquisition_context[0] if best_acquisition_context else {}),
        "best_design_context_only": clean_value(best_design_context[0] if best_design_context else {}),
        "best_physics_only": clean_value(best_physics[0] if best_physics else {}),
        "best_physics_plus_acquisition_context": clean_value(best_acq_combined[0] if best_acq_combined else {}),
        "best_physics_plus_design_context": clean_value(best_design_combined[0] if best_design_combined else {}),
        "cycle_group_oof_metrics": clean_value(oof.to_dict("records")),
        "top_acquisition_context_residual_feature_tests": clean_value(acquisition_residual_tests.head(12).to_dict("records")) if not acquisition_residual_tests.empty else [],
        "top_design_context_residual_feature_tests": clean_value(design_residual_tests.head(12).to_dict("records")) if not design_residual_tests.empty else [],
        "spatial_region_tests": clean_value(region_tests.to_dict("records")) if not region_tests.empty else [],
        "top_context_feature_correlations": clean_value(context_corr.head(12).to_dict("records")) if not context_corr.empty else [],
        "top_cycle_summary": clean_value(cycle_summary.head(24).to_dict("records")),
        "guardrail": "Context/region audit tests whether balanced future8 ROI signal is explainable by acquisition context or spatial position. It still uses weak cycle labels and automatic ROI candidates.",
    }
    with (out / "balanced_future_context_region_summary.json").open("w") as f:
        json.dump(clean_value(summary), f, indent=2, sort_keys=True)
    with (out / "README.md").open("w") as f:
        f.write("# Balanced Future Context/Region Audit\n\n")
        f.write("Checks whether balanced future8 ROI physics signal is explained by cycle/source/spatial/acquisition context.\n\n")
        f.write(f"- ROI rows: {summary['n_roi']}\n")
        f.write(f"- Cycles: {summary['n_cycles']}\n")
        f.write(f"- Physics features: {summary['n_physics_features']}\n")
        f.write(f"- Acquisition context model features: {summary['n_acquisition_context_model_features']}\n")
        f.write(f"- Design context model features: {summary['n_design_context_model_features']}\n\n")
        f.write(summary["guardrail"] + "\n")
    print(json.dumps(clean_value(summary), indent=2)[:8000])


if __name__ == "__main__":
    main()
