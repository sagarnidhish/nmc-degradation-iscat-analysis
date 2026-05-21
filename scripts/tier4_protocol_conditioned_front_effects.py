#!/usr/bin/env python3
"""Protocol/echem-conditioned tests for threshold-robust front metrics.

The threshold-front analysis shows event/control differences in bright-phase
front growth, but those raw metrics can still be confounded by protocol block,
frame count, voltage/current context, and event-reference cycle. This script
joins the threshold-robust front table to the ROI/echem table, residualizes
front metrics against those context covariates, and retests event/control
separation in residual space.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FRONT_FEATURES = [
    "phase_slope_median_per_s",
    "phase_slope_iqr_per_s",
    "phase_slope_positive_fraction",
    "phase_slope_negative_fraction",
    "phase_slope_abs_median_per_s",
    "radius2_slope_median_px2_per_s",
    "radius2_slope_iqr_px2_per_s",
    "radius2_slope_positive_fraction",
    "radius2_slope_negative_fraction",
    "diffusion_proxy_median_um2_per_s",
    "diffusion_proxy_iqr_um2_per_s",
    "default_q70_diffusion_proxy_um2_per_s",
    "diffusion_proxy_abs_median_um2_per_s",
    "q70_phase_slope_bootstrap_p50",
    "q70_radius2_slope_bootstrap_p50_px2_per_s",
    "threshold_robust_phase_score",
    "threshold_robust_diffusion_score",
]


BASE_COVARIATES = [
    "n_frames_percentile",
    "cycles_from_block_start",
    "cycles_to_block_end",
    "block_fraction_elapsed",
    "V_mean",
    "I_mean_mA",
    "I_abs_mean_mA",
    "duration_s",
    "timing_elapsed_s",
]


def clean_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def residualize(df: pd.DataFrame, features: List[str], covariates: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_base = clean_numeric(df, covariates)
    ref = pd.get_dummies(df["event_reference_cycle"].astype(str), prefix="event_ref", drop_first=True)
    x = pd.concat([x_base, ref], axis=1)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 25))),
    ])
    residuals = df[["roi_id", "cohort_role", "event_reference_cycle", "cycleNo"]].copy()
    model_rows = []
    for feat in features:
        y = pd.to_numeric(df[feat], errors="coerce")
        mask = y.notna()
        if mask.sum() < 8 or y[mask].nunique() < 3:
            continue
        model.fit(x.loc[mask], y.loc[mask])
        pred = model.predict(x.loc[mask])
        resid = pd.Series(np.nan, index=df.index, dtype=float)
        resid.loc[mask] = y.loc[mask] - pred
        residuals[f"{feat}_protocol_residual"] = resid
        y_var = float(np.var(y.loc[mask]))
        resid_var = float(np.var(resid.loc[mask]))
        model_rows.append({
            "feature": feat,
            "n": int(mask.sum()),
            "ridge_alpha": float(model.named_steps["ridge"].alpha_),
            "variance_explained_by_protocol_context": float(1.0 - resid_var / y_var) if y_var > 0 else np.nan,
        })
    return residuals, pd.DataFrame(model_rows)


def test_event_control(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        event = pd.to_numeric(df[df["cohort_role"] == "event"][feat], errors="coerce").dropna()
        control = pd.to_numeric(df[df["cohort_role"] == "control"][feat], errors="coerce").dropna()
        if len(event) and len(control):
            stat, p_value = mannwhitneyu(event, control, alternative="two-sided")
        else:
            stat, p_value = np.nan, np.nan
        rows.append({
            "feature": feat,
            "n_event": int(len(event)),
            "n_control": int(len(control)),
            "event_mean": float(event.mean()) if len(event) else np.nan,
            "control_mean": float(control.mean()) if len(control) else np.nan,
            "event_minus_control": float(event.mean() - control.mean()) if len(event) and len(control) else np.nan,
            "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
            "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("p_value")


def safe_spearman(df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, float]:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 5 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return {"rho": np.nan, "p_value": np.nan, "n": int(mask.sum())}
    rho, p_value = spearmanr(x[mask], y[mask])
    return {"rho": float(rho), "p_value": float(p_value), "n": int(mask.sum())}


def leave_reference_classifier(residuals: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    x = residuals[feature_cols].copy()
    y = (residuals["cohort_role"] == "event").astype(int)
    groups = residuals["event_reference_cycle"]
    rows = []
    logo = LeaveOneGroupOut()
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    ])
    for train_idx, test_idx in logo.split(x, y, groups):
        if y.iloc[test_idx].nunique() < 2 or y.iloc[train_idx].nunique() < 2:
            continue
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(x.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "holdout_event_reference_cycle": float(groups.iloc[test_idx].iloc[0]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "roc_auc": float(roc_auc_score(y.iloc[test_idx], prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y.iloc[test_idx], pred)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/protocol_conditioned_front_effects")
    args = parser.parse_args()

    echem_path = os.path.join(args.derived_dir, "multi_cycle_roi_echem_coupling", "multi_cycle_roi_echem_joined.csv")
    front_path = os.path.join(args.derived_dir, "multi_cycle_threshold_robust_fronts", "threshold_robust_front_summary.csv")
    echem = pd.read_csv(echem_path)
    front = pd.read_csv(front_path)
    keep_front = ["roi_id"] + [c for c in FRONT_FEATURES + ["timing_elapsed_s"] if c in front.columns]
    df = echem.merge(front[keep_front], on="roi_id", how="inner")
    features = [c for c in FRONT_FEATURES if c in df.columns]
    covariates = [c for c in BASE_COVARIATES if c in df.columns]

    residuals, context_fit = residualize(df, features, covariates)
    residual_cols = [c for c in residuals.columns if c.endswith("_protocol_residual")]
    raw_tests = test_event_control(df, features)
    residual_tests = test_event_control(residuals, residual_cols)

    corr_rows = []
    for feat in residual_cols:
        for covar in covariates:
            tmp = pd.concat([df[[covar]], residuals[[feat]]], axis=1)
            out = safe_spearman(tmp, covar, feat)
            out.update({"residual_feature": feat, "context_feature": covar})
            corr_rows.append(out)
    residual_context_corr = pd.DataFrame(corr_rows).sort_values("p_value")

    clf_metrics = leave_reference_classifier(residuals, residual_cols)
    model_summary = {}
    if not clf_metrics.empty:
        model_summary = {
            "folds": int(len(clf_metrics)),
            "mean_roc_auc": float(clf_metrics["roc_auc"].mean()),
            "mean_balanced_accuracy": float(clf_metrics["balanced_accuracy"].mean()),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    joined_path = os.path.join(args.out_dir, "protocol_conditioned_front_joined.csv")
    residual_path = os.path.join(args.out_dir, "protocol_conditioned_front_residuals.csv")
    context_path = os.path.join(args.out_dir, "front_protocol_context_fit_summary.csv")
    raw_tests_path = os.path.join(args.out_dir, "raw_front_event_control_tests.csv")
    residual_tests_path = os.path.join(args.out_dir, "protocol_conditioned_front_event_control_tests.csv")
    corr_path = os.path.join(args.out_dir, "front_residual_context_correlations.csv")
    clf_path = os.path.join(args.out_dir, "leave_reference_front_residual_classifier_metrics.csv")
    df.to_csv(joined_path, index=False)
    residuals.to_csv(residual_path, index=False)
    context_fit.to_csv(context_path, index=False)
    raw_tests.to_csv(raw_tests_path, index=False)
    residual_tests.to_csv(residual_tests_path, index=False)
    residual_context_corr.to_csv(corr_path, index=False)
    clf_metrics.to_csv(clf_path, index=False)

    summary = {
        "source_echem": echem_path,
        "source_fronts": front_path,
        "n_rows": int(len(df)),
        "n_event_roi": int((df["cohort_role"] == "event").sum()),
        "n_control_roi": int((df["cohort_role"] == "control").sum()),
        "features_residualized": features,
        "protocol_echem_covariates": covariates,
        "model_summary": model_summary,
        "top_raw_front_event_control_tests": raw_tests.head(12).to_dict("records"),
        "top_protocol_context_fits": context_fit.sort_values("variance_explained_by_protocol_context", ascending=False).head(12).to_dict("records"),
        "top_protocol_conditioned_front_tests": residual_tests.head(12).to_dict("records"),
        "top_residual_context_correlations": residual_context_corr.head(12).to_dict("records"),
        "guardrail": (
            "Residualized threshold-front effects test whether front metrics survive protocol/echem context. "
            "They remain automatic ROI/front proxies and are not calibrated diffusion coefficients."
        ),
        "outputs": {
            "joined": joined_path,
            "residuals": residual_path,
            "context_fit": context_path,
            "raw_tests": raw_tests_path,
            "residual_tests": residual_tests_path,
            "residual_context_correlations": corr_path,
            "leave_reference_classifier": clf_path,
        },
    }
    with open(os.path.join(args.out_dir, "protocol_conditioned_front_effects_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    lines = [
        "# Protocol-Conditioned Front Effects",
        "",
        "Residualized threshold-robust front metrics against protocol/echem context and event-reference-cycle fixed effects.",
        "",
        f"- Source rows: {len(df)}",
        f"- Event ROI: {summary['n_event_roi']}",
        f"- Control ROI: {summary['n_control_roi']}",
        f"- Residualized front features: {len(features)}",
        f"- Covariates: {', '.join(covariates)}",
        "",
        "## Top Adjusted Event-Control Tests",
        "",
    ]
    for row in summary["top_protocol_conditioned_front_tests"][:8]:
        lines.append(f"- {row['feature']}: event-control {row['event_minus_control']:.4g}, p={row['p_value']:.4g}")
    lines += [
        "",
        "## Residual Classifier",
        "",
        f"- Leave-event-reference-out logistic mean ROC-AUC: {model_summary.get('mean_roc_auc', np.nan):.3f}",
        f"- Leave-event-reference-out logistic mean balanced accuracy: {model_summary.get('mean_balanced_accuracy', np.nan):.3f}",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
