#!/usr/bin/env python3
"""Future-specific residual audit after controlling past-event proximity.

The temporal-specificity audit showed that the source-residual reconstruction
error drift is temporally ordered, but not cleanly future-specific. This audit
tests what remains when past-event windows are excluded or modeled explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WINDOWS = [8, 16]
TRANSFORMS = ["raw", "source_residual", "within_source_rank", "within_source_z"]
FEATURES = [
    "dictionary_recon_error_last_minus_first",
    "dictionary_recon_error_mse_slope",
    "resdict_pc01_mean",
    "resdict_pc02_slope",
    "resdict_pc09_slope",
    "masked_minus_background_mean_slope",
    "front_radius_q80_slope_px_per_norm_time",
    "roi_norm_mean_delta_last_minus_first",
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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def source_transform(values: pd.Series, sources: pd.Series, transform: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    src = sources.astype(str)
    means = x.groupby(src).transform("mean")
    if transform == "raw":
        return x
    if transform == "source_residual":
        return x - means
    if transform == "within_source_rank":
        return x.groupby(src).rank(pct=True) - 0.5
    if transform == "within_source_z":
        stds = x.groupby(src).transform("std").replace(0, np.nan)
        return (x - means) / stds
    raise ValueError(transform)


def load_event_cycles(path: Path) -> np.ndarray:
    events = pd.read_csv(path)
    return np.sort(pd.to_numeric(events["cycleNo"], errors="coerce").dropna().unique().astype(float))


def append_temporal_context(df: pd.DataFrame, event_cycles: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    cycles = numeric(out, "cycleNo").to_numpy(dtype=float)
    rows = []
    for cycle in cycles:
        deltas = event_cycles - cycle
        future_pos = deltas[deltas > 0]
        past_pos = -deltas[deltas < 0]
        rows.append({
            "current_any_event": int(np.any(deltas == 0)),
            "cycles_to_next_event": float(future_pos.min()) if len(future_pos) else np.nan,
            "cycles_since_prev_event": float(past_pos.min()) if len(past_pos) else np.nan,
            "nearest_event_abs_delta": float(np.min(np.abs(deltas))) if len(deltas) else np.nan,
        })
    ctx = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, ctx], axis=1)
    for window in WINDOWS:
        out[f"full_future_any_event_within_{window}cycles"] = (
            numeric(out, "cycles_to_next_event").gt(0) & numeric(out, "cycles_to_next_event").le(window)
        ).astype(int)
        out[f"full_past_any_event_within_{window}cycles"] = (
            numeric(out, "cycles_since_prev_event").gt(0) & numeric(out, "cycles_since_prev_event").le(window)
        ).astype(int)
    return out


def oriented_metrics(y: pd.Series, x: pd.Series) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    out: Dict[str, Any] = {
        "n": int(valid.sum()),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "direction": "NA",
        "oriented_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "median_positive_minus_negative": np.nan,
    }
    if len(yy) >= 8 and yy.nunique() == 2 and xx.nunique() > 1:
        direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
        score = xx if direction == "higher_in_positive" else -xx
        out["direction"] = direction
        out["oriented_auc"] = float(roc_auc_score(yy, score))
        out["average_precision"] = float(average_precision_score(yy, score))
        rho, sp = spearmanr(yy, score)
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(sp)
        out["median_positive_minus_negative"] = float(xx[yy == 1].median() - xx[yy == 0].median())
    return out


def subset_masks(df: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
    return {
        "all_rows": pd.Series(True, index=df.index),
        f"exclude_past{window}": numeric(df, f"full_past_any_event_within_{window}cycles").eq(0),
        "pre_first_event": numeric(df, "cycleNo").lt(float(numeric(df, "cycleNo")[numeric(df, "current_any_event").eq(1)].min()))
        if numeric(df, "current_any_event").sum() else numeric(df, "cycleNo").lt(60.0),
    }


def scalar_subset_tests(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    for feature in features:
        raw = numeric(df, feature)
        for transform in TRANSFORMS:
            x = source_transform(raw, sources, transform)
            for window in WINDOWS:
                y = numeric(df, f"full_future_any_event_within_{window}cycles")
                for subset_name, mask in subset_masks(df, window).items():
                    met = oriented_metrics(y[mask], x[mask])
                    rows.append({
                        "feature": feature,
                        "transform": transform,
                        "target_window": window,
                        "subset": subset_name,
                        **met,
                    })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target_window", "subset", "oriented_auc", "average_precision"], ascending=[True, True, False, False])
    return out


def model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.2, solver="liblinear", random_state=seed),
    )


def grouped_predictions(df: pd.DataFrame, cols: List[str], target: str, group_col: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = numeric(df, target)
    valid = y.isin([0, 1]) & df[group_col].notna()
    for group in sorted(df.loc[valid, group_col].dropna().unique()):
        test = valid & df[group_col].eq(group)
        train = valid & ~test
        meta = df.loc[test, ["roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
        meta[group_col] = group
        if train.sum() < 16 or y[train].nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
        else:
            clf = model(seed)
            clf.fit(df.loc[train, cols], y[train].astype(int))
            meta["predicted_probability"] = clf.predict_proba(df.loc[test, cols])[:, 1]
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    out = pd.DataFrame(rows)
    out["target"] = target
    out["group_col"] = group_col
    return out


def prediction_metrics(pred: pd.DataFrame, feature_set: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    group_col = pred["group_col"].iloc[0] if len(pred) else ""
    row: Dict[str, Any] = {
        "feature_set": feature_set,
        "target": pred["target"].iloc[0] if len(pred) else "",
        "group_col": group_col,
        "n_eval": int(len(tmp)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "n_groups": int(tmp[group_col].nunique()) if len(tmp) and group_col in tmp else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2 and p.nunique() > 1:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def build_model_table(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = df.copy()
    sources = out["source_stem"].astype(str)
    for feature in features:
        out[f"{feature}__source_residual"] = source_transform(out[feature], sources, "source_residual")
        out[f"{feature}__within_source_rank"] = source_transform(out[feature], sources, "within_source_rank")
    context = [
        "full_past_any_event_within_8cycles",
        "full_past_any_event_within_16cycles",
        "cycles_since_prev_event",
    ]
    feature_sets = {
        "past_event_context_only": context,
        "primary_source_residual_plus_context": context + ["dictionary_recon_error_last_minus_first__source_residual"],
        "primary_rank_plus_context": context + ["dictionary_recon_error_last_minus_first__within_source_rank"],
        "mask_contrast_source_residual_plus_context": context + ["masked_minus_background_mean_slope__source_residual"],
        "q80_front_source_residual_plus_context": context + ["front_radius_q80_slope_px_per_norm_time__source_residual"],
    }
    return out, {k: [c for c in v if c in out.columns] for k, v in feature_sets.items()}


def grouped_model_audit(df: pd.DataFrame, features: List[str], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work, feature_sets = build_model_table(df, features)
    preds = []
    metrics = []
    for feature_set, cols in feature_sets.items():
        if not cols:
            continue
        for window in WINDOWS:
            target = f"full_future_any_event_within_{window}cycles"
            for group_col in ["cycleNo", "source_stem"]:
                pred = grouped_predictions(work, cols, target, group_col, seed)
                pred["feature_set"] = feature_set
                preds.append(pred)
                metrics.append(prediction_metrics(pred, feature_set))
    pred_df = pd.concat(preds, ignore_index=True, sort=False) if preds else pd.DataFrame()
    metric_df = pd.DataFrame(metrics)
    if not metric_df.empty:
        metric_df = metric_df.sort_values(["target", "group_col", "roc_auc", "average_precision"], ascending=[True, True, False, False])
    return pred_df, metric_df


def deltas(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metric_df.empty:
        return pd.DataFrame()
    base_name = "past_event_context_only"
    for _, row in metric_df.iterrows():
        if row["feature_set"] == base_name:
            continue
        base = metric_df[
            (metric_df["target"] == row["target"])
            & (metric_df["group_col"] == row["group_col"])
            & (metric_df["feature_set"] == base_name)
        ]
        if len(base):
            b = base.iloc[0]
            out = row.to_dict()
            out["base_roc_auc"] = b.get("roc_auc")
            out["base_average_precision"] = b.get("average_precision")
            out["delta_roc_auc"] = row.get("roc_auc") - b.get("roc_auc") if pd.notna(row.get("roc_auc")) and pd.notna(b.get("roc_auc")) else np.nan
            out["delta_average_precision"] = row.get("average_precision") - b.get("average_precision") if pd.notna(row.get("average_precision")) and pd.notna(b.get("average_precision")) else np.nan
            rows.append(out)
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["target", "group_col", "delta_roc_auc", "delta_average_precision"], ascending=[True, True, False, False])
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_dictionary_audit")
    parser.add_argument("--event-cycles", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_targets/particle_abrupt_events.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_future_specific_residual_audit")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.feature_dir) / "source_balanced_residual_dictionary_features.csv")
    event_cycles = load_event_cycles(Path(args.event_cycles))
    df = append_temporal_context(df, event_cycles)
    features = [f for f in FEATURES if f in df.columns]

    scalar_tests = scalar_subset_tests(df, features)
    predictions, model_metrics = grouped_model_audit(df, features, args.seed)
    model_deltas = deltas(model_metrics)

    primary_rows = scalar_tests[
        (scalar_tests["feature"] == "dictionary_recon_error_last_minus_first")
        & (scalar_tests["transform"] == "source_residual")
    ].copy()
    best_clean_future = scalar_tests[scalar_tests["subset"].str.startswith("exclude_past")].sort_values(
        ["target_window", "oriented_auc", "average_precision"], ascending=[True, False, False]
    ).groupby("target_window", as_index=False).head(8)
    top_model_deltas = model_deltas.sort_values(["delta_roc_auc", "delta_average_precision"], ascending=False).head(16)

    paths = {
        "scalar_tests": out / "source_balanced_future_specific_scalar_tests.csv",
        "model_metrics": out / "source_balanced_future_specific_model_metrics.csv",
        "model_deltas": out / "source_balanced_future_specific_model_deltas.csv",
        "predictions": out / "source_balanced_future_specific_model_predictions.csv",
        "summary": out / "source_balanced_future_specific_residual_summary.json",
    }
    scalar_tests.to_csv(paths["scalar_tests"], index=False)
    model_metrics.to_csv(paths["model_metrics"], index=False)
    model_deltas.to_csv(paths["model_deltas"], index=False)
    predictions.to_csv(paths["predictions"], index=False)

    label_counts = {
        "current_any_event": int(numeric(df, "current_any_event").sum()),
        **{f"future{w}": int(numeric(df, f"full_future_any_event_within_{w}cycles").sum()) for w in WINDOWS},
        **{f"past{w}": int(numeric(df, f"full_past_any_event_within_{w}cycles").sum()) for w in WINDOWS},
        **{f"no_past{w}_rows": int(numeric(df, f"full_past_any_event_within_{w}cycles").eq(0).sum()) for w in WINDOWS},
    }
    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "event_cycles": [float(c) for c in event_cycles],
        "features_tested": features,
        "label_counts": label_counts,
        "primary_source_residual_subset_rows": primary_rows.to_dict(orient="records"),
        "best_clean_future_subset_rows": best_clean_future.to_dict(orient="records"),
        "top_model_metrics": model_metrics.head(32).to_dict(orient="records") if not model_metrics.empty else [],
        "top_model_deltas_vs_past_context": top_model_deltas.to_dict(orient="records") if not top_model_deltas.empty else [],
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Past-event context uses event-cycle labels derived from full particle abrupt-drop cycles. Excluding or modeling past windows tests future specificity on the source-balanced ROI table; it does not prove causality, source transfer, or calibrated phase/diffusion physics.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Source-Balanced Future-Specific Residual Audit",
        "",
        f"Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"Event cycles: {summary['event_cycles']}",
        f"Label counts: {summary['label_counts']}",
        "",
        "## Primary Source-Residual Candidate",
    ]
    for row in summary["primary_source_residual_subset_rows"]:
        if row.get("target_window") == 16:
            readme.append(
                f"- {row.get('subset')}: future16 AUC={row.get('oriented_auc'):.3f}, "
                f"AP={row.get('average_precision'):.3f}, n={row.get('n')}, positives={row.get('n_positive')}"
            )
    readme.extend(["", "## Top Model Deltas vs Past-Event Context"])
    for row in summary["top_model_deltas_vs_past_context"][:8]:
        readme.append(
            f"- {row.get('group_col')} {row.get('target')} {row.get('feature_set')}: "
            f"delta AUC={row.get('delta_roc_auc'):.3f}, AUC={row.get('roc_auc'):.3f}, base={row.get('base_roc_auc'):.3f}"
        )
    readme.extend(["", "## Guardrail", summary["guardrail"], ""])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()
