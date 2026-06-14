#!/usr/bin/env python3
"""Condition ROI/front optical effects on within-cycle echem shape descriptors.

The within-cycle echem shape audit found strong associations between voltage/current
trajectory descriptors and ROI/front/kinetic readouts. This script asks a narrower
question: after the existing protocol-conditioned optical/front residuals are further
regressed against a low-dimensional echem-shape representation, which event/control
readouts still remain?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = [
    "roi_norm_mean_delta_protocol_residual",
    "high_fraction_delta_protocol_residual",
    "low_fraction_delta_protocol_residual",
    "first_last_corr_protocol_residual",
    "cumulative_abs_norm_change_protocol_residual",
    "latent_net_displacement_protocol_residual",
    "dmd_minus_persistence_mse_protocol_residual",
    "phase_slope_median_per_s_protocol_residual",
    "phase_slope_positive_fraction_protocol_residual",
    "threshold_robust_phase_score_protocol_residual",
    "diffusion_proxy_median_um2_per_s_protocol_residual",
    "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    "q60_logistic_k_per_s",
    "q70_logistic_k_per_s",
    "q70_transformed_fraction_delta",
    "q80_transformed_fraction_delta",
    "mode_review_priority",
]

SHAPE_PREFIXES = (
    "echem_shape_duration_s",
    "shape_",
    "all_dq_",
    "all_dqdv_",
    "pos_dq_",
    "pos_dqdv_",
    "neg_dq_",
    "neg_dqdv_",
)


def finite_float(value: Any) -> Any:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


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
        return finite_float(value)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def numeric_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def shape_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col in {"shape_block_mode", "shape_block_nunique", "echem_shape_points"}:
            continue
        if col.startswith(SHAPE_PREFIXES):
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().sum() >= 12 and vals.nunique(dropna=True) >= 3:
                cols.append(col)
    return cols


def build_shape_pcs(df: pd.DataFrame, shape_cols: List[str], max_components: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    x = numeric_frame(df, shape_cols)
    valid_cols = [c for c in x.columns if x[c].notna().sum() >= 12 and x[c].nunique(dropna=True) >= 3]
    x = x[valid_cols]
    n_components = min(max_components, max(1, len(valid_cols)), max(1, len(df) // 6))
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=0)),
    ])
    scores = pipe.fit_transform(x)
    pc_cols = [f"echem_shape_pc{i+1}" for i in range(scores.shape[1])]
    score_df = pd.DataFrame(scores, columns=pc_cols, index=df.index)
    pca = pipe.named_steps["pca"]
    loading_rows = []
    for i, pc in enumerate(pc_cols):
        load = pd.Series(pca.components_[i], index=valid_cols)
        for feat, val in load.reindex(load.abs().sort_values(ascending=False).index).head(12).items():
            loading_rows.append({"pc": pc, "feature": feat, "loading": float(val), "abs_loading": float(abs(val))})
    meta = {
        "n_shape_features_used": int(len(valid_cols)),
        "n_components": int(len(pc_cols)),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "explained_variance_total": float(np.sum(pca.explained_variance_ratio_)),
    }
    return score_df, pd.DataFrame(loading_rows), meta


def residualize_targets(df: pd.DataFrame, targets: List[str], pc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = pd.get_dummies(df["event_reference_cycle"].astype(str), prefix="event_ref", drop_first=True)
    x = pd.concat([pc_df, ref], axis=1)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 25))),
    ])
    residuals = df[["roi_id", "cycleNo", "cohort_role", "event_reference_cycle"]].copy()
    fit_rows = []
    for target in targets:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce")
        mask = y.notna()
        if mask.sum() < 10 or y[mask].nunique() < 3:
            continue
        model.fit(x.loc[mask], y.loc[mask])
        pred = model.predict(x.loc[mask])
        resid = pd.Series(np.nan, index=df.index, dtype=float)
        resid.loc[mask] = y.loc[mask] - pred
        residuals[f"{target}_shape_residual"] = resid
        y_var = float(np.var(y.loc[mask]))
        resid_var = float(np.var(resid.loc[mask]))
        fit_rows.append({
            "target": target,
            "n": int(mask.sum()),
            "ridge_alpha": float(model.named_steps["ridge"].alpha_),
            "variance_explained_by_shape_pcs_and_event_ref": float(1.0 - resid_var / y_var) if y_var > 0 else np.nan,
        })
    return residuals, pd.DataFrame(fit_rows)


def event_control_tests(df: pd.DataFrame, cols: List[str], suffix: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        event = pd.to_numeric(df.loc[df["cohort_role"] == "event", col], errors="coerce").dropna()
        control = pd.to_numeric(df.loc[df["cohort_role"] == "control", col], errors="coerce").dropna()
        if len(event) and len(control):
            stat, p_value = mannwhitneyu(event, control, alternative="two-sided")
            event_minus_control = float(event.median() - control.median())
        else:
            stat, p_value, event_minus_control = np.nan, np.nan, np.nan
        rows.append({
            "feature": col,
            "target_base": col.removesuffix(suffix),
            "n_event": int(len(event)),
            "n_control": int(len(control)),
            "event_median": finite_float(event.median()) if len(event) else None,
            "control_median": finite_float(control.median()) if len(control) else None,
            "event_minus_control_median": finite_float(event_minus_control),
            "mannwhitney_u": finite_float(stat),
            "p_value": finite_float(p_value),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("p_value", na_position="last") if not out.empty else out


def compare_tests(raw_tests: pd.DataFrame, residual_tests: pd.DataFrame) -> pd.DataFrame:
    raw = raw_tests.rename(columns={
        "feature": "raw_feature",
        "p_value": "raw_p_value",
        "event_minus_control_median": "raw_event_minus_control_median",
    })[["target_base", "raw_feature", "raw_p_value", "raw_event_minus_control_median", "n_event", "n_control"]]
    res = residual_tests.rename(columns={
        "feature": "shape_residual_feature",
        "p_value": "shape_residual_p_value",
        "event_minus_control_median": "shape_residual_event_minus_control_median",
    })[["target_base", "shape_residual_feature", "shape_residual_p_value", "shape_residual_event_minus_control_median"]]
    merged = raw.merge(res, on="target_base", how="outer")
    merged["abs_effect_retention_ratio"] = (
        merged["shape_residual_event_minus_control_median"].abs() / merged["raw_event_minus_control_median"].abs().replace(0, np.nan)
    )
    merged["p_value_change_factor"] = merged["shape_residual_p_value"] / merged["raw_p_value"].replace(0, np.nan)
    return merged.sort_values("shape_residual_p_value", na_position="last")


def safe_spearman(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 8 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return {"rho": None, "p_value": None, "n": int(mask.sum())}
    rho, p_value = spearmanr(x[mask], y[mask])
    return {"rho": finite_float(rho), "p_value": finite_float(p_value), "n": int(mask.sum())}


def pc_target_correlations(df: pd.DataFrame, pc_df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    rows = []
    for pc in pc_df.columns:
        for target in targets:
            if target not in df.columns:
                continue
            out = safe_spearman(pc_df[pc], df[target])
            out.update({"pc": pc, "target": target})
            rows.append(out)
    table = pd.DataFrame(rows)
    return table.sort_values("p_value", na_position="last") if not table.empty else table


def leave_reference_classifier(residuals: pd.DataFrame, residual_cols: List[str]) -> pd.DataFrame:
    if len(residual_cols) == 0:
        return pd.DataFrame()
    x = residuals[residual_cols].copy()
    y = (residuals["cohort_role"] == "event").astype(int)
    groups = residuals["event_reference_cycle"]
    rows = []
    logo = LeaveOneGroupOut()
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3)),
    ])
    for train_idx, test_idx in logo.split(x, y, groups):
        if y.iloc[test_idx].nunique() < 2 or y.iloc[train_idx].nunique() < 2:
            continue
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(x.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "holdout_event_reference_cycle": finite_float(groups.iloc[test_idx].iloc[0]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "roc_auc": finite_float(roc_auc_score(y.iloc[test_idx], prob)),
            "balanced_accuracy": finite_float(balanced_accuracy_score(y.iloc[test_idx], pred)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/echem_shape_conditioned_roi_front_effects")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    source = derived / "within_cycle_echem_shape_audit" / "within_cycle_echem_roi_joined.csv"
    df = pd.read_csv(source)
    df = df[df["cohort_role"].isin(["event", "control"])].copy()
    shape_cols = shape_feature_columns(df)
    target_cols = [c for c in TARGETS if c in df.columns]
    pc_df, loadings, pc_meta = build_shape_pcs(df, shape_cols)
    residuals, fit_summary = residualize_targets(df, target_cols, pc_df)
    residual_cols = [c for c in residuals.columns if c.endswith("_shape_residual")]

    raw_tests = event_control_tests(df, target_cols, "")
    shape_tests = event_control_tests(residuals, residual_cols, "_shape_residual")
    comparison = compare_tests(raw_tests, shape_tests)
    pc_corr = pc_target_correlations(df, pc_df, target_cols)
    clf_metrics = leave_reference_classifier(residuals, residual_cols)
    model_summary = {}
    if not clf_metrics.empty:
        model_summary = {
            "folds": int(len(clf_metrics)),
            "mean_roc_auc": finite_float(clf_metrics["roc_auc"].mean()),
            "mean_balanced_accuracy": finite_float(clf_metrics["balanced_accuracy"].mean()),
        }

    joined = pd.concat([df.reset_index(drop=True), pc_df.reset_index(drop=True)], axis=1)
    joined_path = out / "echem_shape_conditioned_joined.csv"
    residuals_path = out / "echem_shape_conditioned_residuals.csv"
    fit_path = out / "echem_shape_context_fit_summary.csv"
    raw_tests_path = out / "raw_event_control_tests.csv"
    shape_tests_path = out / "shape_conditioned_event_control_tests.csv"
    comparison_path = out / "shape_conditioning_effect_retention.csv"
    pc_corr_path = out / "shape_pc_target_correlations.csv"
    loadings_path = out / "echem_shape_pc_loadings.csv"
    clf_path = out / "leave_reference_shape_residual_classifier_metrics.csv"

    joined.to_csv(joined_path, index=False)
    residuals.to_csv(residuals_path, index=False)
    fit_summary.to_csv(fit_path, index=False)
    raw_tests.to_csv(raw_tests_path, index=False)
    shape_tests.to_csv(shape_tests_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    pc_corr.to_csv(pc_corr_path, index=False)
    loadings.to_csv(loadings_path, index=False)
    clf_metrics.to_csv(clf_path, index=False)

    top_retained = comparison.head(12).to_dict("records")
    top_shape_fit = fit_summary.sort_values("variance_explained_by_shape_pcs_and_event_ref", ascending=False).head(12).to_dict("records") if not fit_summary.empty else []
    top_pc_corr = pc_corr.head(12).to_dict("records") if not pc_corr.empty else []
    summary = {
        "source": str(source),
        "n_rows": int(len(df)),
        "n_event_roi": int((df["cohort_role"] == "event").sum()),
        "n_control_roi": int((df["cohort_role"] == "control").sum()),
        "target_features": target_cols,
        "shape_features_used": shape_cols,
        "shape_pca": pc_meta,
        "model_summary": model_summary,
        "top_shape_context_fits": clean_json(top_shape_fit),
        "top_shape_conditioned_event_control_tests": clean_json(shape_tests.head(12).to_dict("records")),
        "top_effect_retention": clean_json(top_retained),
        "top_shape_pc_target_correlations": clean_json(top_pc_corr),
        "guardrail": "Echem-shape conditioning uses low-dimensional PCA/ridge covariates on a small automatically selected ROI cohort. Surviving residuals are evidence that optical/front signals are not fully explained by measured within-cycle echem shape, but they are not causal proof or calibrated transport constants.",
        "outputs": {
            "joined": str(joined_path),
            "residuals": str(residuals_path),
            "context_fit": str(fit_path),
            "raw_tests": str(raw_tests_path),
            "shape_tests": str(shape_tests_path),
            "effect_retention": str(comparison_path),
            "pc_correlations": str(pc_corr_path),
            "pc_loadings": str(loadings_path),
            "leave_reference_classifier": str(clf_path),
            "summary": str(out / "echem_shape_conditioned_roi_front_effects_summary.json"),
        },
    }
    with (out / "echem_shape_conditioned_roi_front_effects_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)

    lines = [
        "# Echem-Shape-Conditioned ROI/Front Effects",
        "",
        "Residualizes protocol-conditioned ROI/front optical readouts against low-dimensional within-cycle echem shape PCs and event-reference-cycle fixed effects.",
        "",
        f"- Source rows: {summary['n_rows']}",
        f"- Event/control ROI: {summary['n_event_roi']} / {summary['n_control_roi']}",
        f"- Shape features used: {pc_meta['n_shape_features_used']}",
        f"- Shape PCs: {pc_meta['n_components']} explaining {pc_meta['explained_variance_total']:.3f} variance",
        "",
        "## Top Shape-Conditioned Event-Control Tests",
        "",
    ]
    for row in summary["top_shape_conditioned_event_control_tests"][:8]:
        lines.append(f"- {row['target_base']}: residual median event-control {row['event_minus_control_median']:.4g}, p={row['p_value']:.4g}")
    lines += [
        "",
        "## Shape Context Fit",
        "",
    ]
    for row in top_shape_fit[:6]:
        lines.append(f"- {row['target']}: variance explained {row['variance_explained_by_shape_pcs_and_event_ref']:.3f}")
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
    (out / "README.md").write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
