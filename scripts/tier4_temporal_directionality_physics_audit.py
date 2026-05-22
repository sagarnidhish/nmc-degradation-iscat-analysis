#!/usr/bin/env python3
"""
Audit whether ROI physics descriptors behave more like degradation precursors
than generic cycle-age or aftermath markers.

The analysis joins balanced future-drop ROI physics descriptors with the
cycle-level particle trace table, computes past-drop windows at each ROI cycle,
and compares physics-only models/features against:
  - future abrupt-drop labels,
  - past abrupt-drop labels,
  - time-reversed future labels,
  - circular time-shifted future labels.

This is a guardrail, not a deployment model. It uses weak cycle labels and
automatic ROI/front descriptors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PHYSICS_FEATURE_HINTS = (
    "phase_slope",
    "radius2_slope",
    "diffusion_proxy",
    "threshold_robust",
    "q70_",
    "particle_mse",
    "particle_to_nonparticle",
    "particle_mse_fraction",
    "accepted_area_fraction",
    "mask_fallback",
    "roi_mean_delta",
    "roi_norm_mean_delta",
    "object_mean_abs_z",
    "object_mean_residual",
)

EXCLUDE_FEATURE_SUBSTRINGS = (
    "cycleNo",
    "rank",
    "future_",
    "past_",
    "label",
    "cohort",
    "source",
    "role",
    "reference",
    "validation",
    "path",
    "png",
    "stem",
    "subrole",
    "signature",
    "abrupt_drop",
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite_auc(y: pd.Series, score: pd.Series) -> float | None:
    valid = y.notna() & score.notna()
    yy = y[valid].astype(int)
    ss = score[valid].astype(float)
    if yy.nunique() != 2 or len(yy) < 6:
        return None
    return float(roc_auc_score(yy, ss))


def label_counts(df: pd.DataFrame, col: str) -> dict:
    if col not in df:
        return {}
    counts = df[col].dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}


def compute_past_future_timing(cycles: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    out = cycles[["cycleNo", "any_abrupt_drop"]].drop_duplicates("cycleNo").copy()
    out = out.sort_values("cycleNo").reset_index(drop=True)
    cycle_values = out["cycleNo"].to_numpy(dtype=float)
    drop_cycles = out.loc[out["any_abrupt_drop"].fillna(0).astype(int) == 1, "cycleNo"].to_numpy(dtype=float)

    for window in windows:
        past_labels = []
        future_labels = []
        for cycle in cycle_values:
            past = drop_cycles[(drop_cycles < cycle) & ((cycle - drop_cycles) <= window)]
            future = drop_cycles[(drop_cycles > cycle) & ((drop_cycles - cycle) <= window)]
            past_labels.append(int(len(past) > 0))
            future_labels.append(int(len(future) > 0))
        out[f"past_any_drop_within_{window}cycles"] = past_labels
        out[f"recomputed_future_any_drop_within_{window}cycles"] = future_labels

    time_since = []
    time_to = []
    for cycle in cycle_values:
        prior = drop_cycles[drop_cycles < cycle]
        future = drop_cycles[drop_cycles > cycle]
        time_since.append(float(cycle - prior.max()) if len(prior) else np.nan)
        time_to.append(float(future.min() - cycle) if len(future) else np.nan)
    out["cycles_since_previous_drop"] = time_since
    out["cycles_to_next_drop"] = time_to
    return out


def select_physics_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        lower = col.lower()
        if any(token in lower for token in EXCLUDE_FEATURE_SUBSTRINGS):
            continue
        if any(token in lower for token in PHYSICS_FEATURE_HINTS):
            if df[col].notna().sum() >= 12 and df[col].nunique(dropna=True) > 2:
                features.append(col)
    return sorted(set(features))


def feature_tests(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    rows = []
    if target not in df:
        return pd.DataFrame(rows)
    y = df[target]
    for feat in features:
        valid = y.notna() & df[feat].notna()
        yy = y[valid].astype(int)
        xx = df.loc[valid, feat].astype(float)
        if yy.nunique() != 2:
            continue
        pos = xx[yy == 1]
        neg = xx[yy == 0]
        if len(pos) < 3 or len(neg) < 3:
            continue
        auc = finite_auc(yy, xx)
        oriented = None if auc is None else float(max(auc, 1.0 - auc))
        try:
            mw_p = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
        except ValueError:
            mw_p = np.nan
        rows.append(
            {
                "target": target,
                "feature": feat,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                "median_positive": float(pos.median()),
                "median_negative": float(neg.median()),
                "median_positive_minus_negative": float(pos.median() - neg.median()),
                "auc": auc,
                "oriented_auc": oriented,
                "mannwhitney_p": mw_p,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "oriented_auc", "mannwhitney_p"], ascending=[True, False, True])
    return out


def timing_correlations(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for target in ["cycles_to_next_drop", "cycles_since_previous_drop"]:
        if target not in df:
            continue
        for feat in features:
            valid = df[target].notna() & df[feat].notna()
            if valid.sum() < 10 or df.loc[valid, target].nunique() < 3 or df.loc[valid, feat].nunique() < 3:
                continue
            rho, pval = spearmanr(df.loc[valid, feat], df.loc[valid, target])
            rows.append(
                {
                    "timing_target": target,
                    "feature": feat,
                    "n": int(valid.sum()),
                    "spearman_rho": float(rho),
                    "abs_rho": float(abs(rho)),
                    "p_value": float(pval),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["timing_target", "abs_rho"], ascending=[True, False])
    return out


def oof_model(df: pd.DataFrame, features: list[str], target: str, model_name: str) -> tuple[dict, pd.DataFrame]:
    valid_target = df[target].notna() if target in df else pd.Series(False, index=df.index)
    working = df.loc[valid_target, ["roi_id", "cycleNo", target] + features].copy()
    working[target] = working[target].astype(int)
    pred = np.full(len(working), np.nan)

    if model_name == "logistic_l2":
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=500, random_state=7),
        )
    elif model_name == "random_forest":
        estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=7,
                class_weight="balanced_subsample",
            ),
        )
    else:
        raise ValueError(model_name)

    cycles = sorted(working["cycleNo"].dropna().unique())
    for cycle in cycles:
        test = working["cycleNo"] == cycle
        train = ~test
        y_train = working.loc[train, target]
        y_test = working.loc[test, target]
        if y_train.nunique() != 2 or y_test.empty:
            continue
        estimator.fit(working.loc[train, features], y_train)
        pred[test.to_numpy()] = estimator.predict_proba(working.loc[test, features])[:, 1]

    scored = np.isfinite(pred)
    y_scored = working.loc[scored, target]
    if scored.sum() >= 6 and y_scored.nunique() == 2:
        auc = float(roc_auc_score(y_scored, pred[scored]))
        ap = float(average_precision_score(y_scored, pred[scored]))
    else:
        auc = np.nan
        ap = np.nan
    metrics = {
        "target": target,
        "model": model_name,
        "group_col": "cycleNo",
        "n_rows": int(len(working)),
        "n_scored": int(scored.sum()),
        "n_positive_scored": int(y_scored.sum()) if scored.sum() else 0,
        "n_negative_scored": int((1 - y_scored).sum()) if scored.sum() else 0,
        "pooled_oof_roc_auc": auc,
        "pooled_oof_average_precision": ap,
    }
    pred_df = working[["roi_id", "cycleNo", target]].copy()
    pred_df["model"] = model_name
    pred_df["predicted_probability"] = pred
    return metrics, pred_df


def circular_shift_null(df: pd.DataFrame, features: list[str], target: str = "future_any_drop_within_8cycles") -> pd.DataFrame:
    cycle_label = df[["cycleNo", target]].drop_duplicates("cycleNo").sort_values("cycleNo")
    labels = cycle_label[target].astype(int).to_numpy()
    cycles = cycle_label["cycleNo"].to_numpy()
    rows = []
    if len(labels) < 6:
        return pd.DataFrame(rows)
    for shift in range(1, len(labels)):
        shifted = np.roll(labels, shift)
        mapping = dict(zip(cycles, shifted))
        shifted_target = f"{target}_circular_shift_{shift}"
        shifted_df = df.copy()
        shifted_df[shifted_target] = shifted_df["cycleNo"].map(mapping)
        metrics, _ = oof_model(shifted_df, features, shifted_target, "logistic_l2")
        rows.append(
            {
                "shift": int(shift),
                "target": shifted_target,
                "pooled_oof_roc_auc": metrics["pooled_oof_roc_auc"],
                "pooled_oof_average_precision": metrics["pooled_oof_average_precision"],
                "n_scored": metrics["n_scored"],
            }
        )
    return pd.DataFrame(rows)



def fmt_num(value: object) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{val:.3f}" if np.isfinite(val) else "NA"

def top_records(df: pd.DataFrame, n: int, sort_col: str | None = None) -> list[dict]:
    if df.empty:
        return []
    out = df.copy()
    if sort_col and sort_col in out:
        out = out.sort_values(sort_col, ascending=False)
    return json.loads(out.head(n).to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/temporal_directionality_physics_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = ensure_dir(Path(args.out_dir))

    roi_path = derived / "balanced_future_roi_physics_audit" / "balanced_future_roi_physics_joined.csv"
    cycle_path = derived / "particle_trace_physics_audit" / "particle_trace_cycle_features.csv"
    roi = pd.read_csv(roi_path)
    cycles = pd.read_csv(cycle_path)

    timing = compute_past_future_timing(cycles, (4, 8, 16))
    df = roi.merge(timing, on="cycleNo", how="left", suffixes=("", "_cycle_timing"))
    if "future_any_drop_within_8cycles" not in df and "recomputed_future_any_drop_within_8cycles" in df:
        df["future_any_drop_within_8cycles"] = df["recomputed_future_any_drop_within_8cycles"]

    cycle_labels = df[["cycleNo", "future_any_drop_within_8cycles"]].drop_duplicates("cycleNo").sort_values("cycleNo")
    reversed_mapping = dict(zip(cycle_labels["cycleNo"], cycle_labels["future_any_drop_within_8cycles"].iloc[::-1].to_numpy()))
    df["reversed_future_any_drop_within_8cycles"] = df["cycleNo"].map(reversed_mapping).astype(int)

    features = select_physics_features(df)
    target_cols = [
        "future_any_drop_within_4cycles",
        "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles",
        "past_any_drop_within_4cycles",
        "past_any_drop_within_8cycles",
        "past_any_drop_within_16cycles",
        "reversed_future_any_drop_within_8cycles",
    ]
    target_cols = [col for col in target_cols if col in df and df[col].dropna().astype(int).nunique() == 2]

    all_tests = []
    for target in target_cols:
        all_tests.append(feature_tests(df, features, target))
    tests = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()

    corr = timing_correlations(df, features)

    model_rows = []
    pred_rows = []
    for target in target_cols:
        for model in ["logistic_l2", "random_forest"]:
            metrics, pred_df = oof_model(df, features, target, model)
            model_rows.append(metrics)
            pred_rows.append(pred_df)
    model_metrics = pd.DataFrame(model_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    shift_null = circular_shift_null(df, features)
    future_logit = model_metrics[(model_metrics["target"] == "future_any_drop_within_8cycles") & (model_metrics["model"] == "logistic_l2")]
    observed_auc = float(future_logit["pooled_oof_roc_auc"].iloc[0]) if not future_logit.empty else np.nan
    if not shift_null.empty and np.isfinite(observed_auc):
        null_auc = shift_null["pooled_oof_roc_auc"].dropna()
        null_summary = {
            "observed_future8_logistic_auc": observed_auc,
            "n_shift_nulls": int(len(null_auc)),
            "shift_null_auc_mean": float(null_auc.mean()) if len(null_auc) else np.nan,
            "shift_null_auc_p95": float(null_auc.quantile(0.95)) if len(null_auc) else np.nan,
            "empirical_p_ge_observed": float((1 + (null_auc >= observed_auc).sum()) / (len(null_auc) + 1)) if len(null_auc) else np.nan,
        }
    else:
        null_summary = {}

    cycle_summary = (
        df.groupby("cycleNo", as_index=False)
        .agg(
            n_roi=("roi_id", "count"),
            future_any_drop_within_8cycles=("future_any_drop_within_8cycles", "max"),
            past_any_drop_within_8cycles=("past_any_drop_within_8cycles", "max"),
            cycles_to_next_drop=("cycles_to_next_drop", "first"),
            cycles_since_previous_drop=("cycles_since_previous_drop", "first"),
            radius2_slope_median_px2_per_s=("radius2_slope_median_px2_per_s", "median"),
            diffusion_proxy_median_um2_per_s=("diffusion_proxy_median_um2_per_s", "median"),
            persistence_particle_mse_fraction_of_full_mean=("persistence_particle_mse_fraction_of_full_mean", "median"),
        )
        .sort_values("cycleNo")
    )

    df.to_csv(out / "temporal_directionality_joined.csv", index=False)
    tests.to_csv(out / "temporal_directionality_feature_tests.csv", index=False)
    corr.to_csv(out / "temporal_directionality_timing_correlations.csv", index=False)
    model_metrics.to_csv(out / "temporal_directionality_model_metrics.csv", index=False)
    predictions.to_csv(out / "temporal_directionality_oof_predictions.csv", index=False)
    shift_null.to_csv(out / "temporal_directionality_shift_null.csv", index=False)
    cycle_summary.to_csv(out / "temporal_directionality_cycle_summary.csv", index=False)

    best_future = model_metrics[model_metrics["target"].eq("future_any_drop_within_8cycles")].sort_values("pooled_oof_roc_auc", ascending=False).head(1)
    best_past = model_metrics[model_metrics["target"].eq("past_any_drop_within_8cycles")].sort_values("pooled_oof_roc_auc", ascending=False).head(1)
    best_reversed = model_metrics[model_metrics["target"].eq("reversed_future_any_drop_within_8cycles")].sort_values("pooled_oof_roc_auc", ascending=False).head(1)

    summary = {
        "n_roi": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_physics_features": int(len(features)),
        "physics_features": features,
        "target_label_counts": {target: label_counts(df, target) for target in target_cols},
        "best_future8_model": top_records(best_future, 1),
        "best_past8_model": top_records(best_past, 1),
        "best_reversed_future8_model": top_records(best_reversed, 1),
        "all_model_metrics": top_records(model_metrics.sort_values(["target", "pooled_oof_roc_auc"], ascending=[True, False]), 30),
        "top_future8_feature_tests": top_records(tests[tests["target"].eq("future_any_drop_within_8cycles")], 12),
        "top_past8_feature_tests": top_records(tests[tests["target"].eq("past_any_drop_within_8cycles")], 12),
        "top_timing_correlations": top_records(corr, 20, "abs_rho"),
        "shift_null_summary": null_summary,
        "guardrail": "Temporal directionality audit compares ROI physics against future, past, reversed, and circularly shifted weak degradation labels. It uses automatic ROI/front descriptors and weak cycle-level abrupt-drop labels.",
    }
    with (out / "temporal_directionality_physics_audit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with (out / "README.md").open("w") as f:
        f.write("# Temporal Directionality Physics Audit\n\n")
        f.write("Tests whether balanced ROI physics descriptors behave more like future degradation precursors than past-drop aftermath or time-shifted cycle context.\n\n")
        f.write(f"- ROI rows: {summary['n_roi']}\n")
        f.write(f"- Cycles: {summary['n_cycles']}\n")
        f.write(f"- Physics features: {summary['n_physics_features']}\n")
        if summary["best_future8_model"]:
            row = summary["best_future8_model"][0]
            f.write(f"- Best future8 model: {row['model']} AUC={fmt_num(row['pooled_oof_roc_auc'])}, AP={fmt_num(row['pooled_oof_average_precision'])}\n")
        if summary["best_past8_model"]:
            row = summary["best_past8_model"][0]
            f.write(f"- Best past8 model: {row['model']} AUC={fmt_num(row['pooled_oof_roc_auc'])}, AP={fmt_num(row['pooled_oof_average_precision'])}\n")
        f.write("\nGuardrail: automatic ROI/front descriptors and weak labels; use for hypothesis ranking, not a validated causal claim.\n")


if __name__ == "__main__":
    main()
