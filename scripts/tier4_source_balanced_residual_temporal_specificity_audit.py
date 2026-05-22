#!/usr/bin/env python3
"""Temporal specificity audit for source-balanced residual dynamics candidates.

The normalized residual readout found a provisional source-robust future16
signal in reconstruction-error drift. This audit asks whether that signal is
more future-specific than past/current event controls when labels are derived
from the full particle abrupt-drop cycle table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

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


def metric_with_fixed_score(y: pd.Series, score: pd.Series) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & score.notna()
    yy = y[valid].astype(int)
    ss = score[valid]
    out: Dict[str, Any] = {
        "n": int(valid.sum()),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "median_positive": np.nan,
        "median_negative": np.nan,
        "median_positive_minus_negative": np.nan,
    }
    if len(yy) >= 8 and yy.nunique() == 2 and ss.nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(yy, ss))
        out["average_precision"] = float(average_precision_score(yy, ss))
        rho, sp = spearmanr(yy, ss)
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(sp)
        out["median_positive"] = float(ss[yy == 1].median())
        out["median_negative"] = float(ss[yy == 0].median())
        out["median_positive_minus_negative"] = out["median_positive"] - out["median_negative"]
    return out


def future_oriented_score(y_future: pd.Series, x: pd.Series) -> Tuple[pd.Series, str]:
    valid = y_future.isin([0, 1]) & x.notna()
    yy = y_future[valid].astype(int)
    xx = x[valid]
    if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
        return x * np.nan, "NA"
    direction = "higher_in_future_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_future_positive"
    return (x if direction == "higher_in_future_positive" else -x), direction


def load_event_cycles(path: Path) -> np.ndarray:
    events = pd.read_csv(path)
    if "cycleNo" not in events.columns:
        raise ValueError(f"cycleNo missing from {path}")
    return np.sort(pd.to_numeric(events["cycleNo"], errors="coerce").dropna().unique().astype(float))


def append_temporal_labels(df: pd.DataFrame, event_cycles: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    cycles = numeric(out, "cycleNo").to_numpy(dtype=float)
    out["current_any_event"] = np.isin(cycles, event_cycles).astype(int)
    for window in WINDOWS:
        future = []
        past = []
        nearest_signed = []
        for cycle in cycles:
            deltas = event_cycles - cycle
            future.append(int(np.any((deltas > 0) & (deltas <= window))))
            past.append(int(np.any((deltas < 0) & (deltas >= -window))))
            nearest_signed.append(float(deltas[np.argmin(np.abs(deltas))]) if len(deltas) else np.nan)
        out[f"full_future_any_event_within_{window}cycles"] = future
        out[f"full_past_any_event_within_{window}cycles"] = past
        out[f"full_nearest_event_delta_cycles"] = nearest_signed
    return out


def temporal_tests(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    for feature in features:
        raw = numeric(df, feature)
        for transform in TRANSFORMS:
            x = source_transform(raw, sources, transform)
            for window in WINDOWS:
                future_col = f"full_future_any_event_within_{window}cycles"
                past_col = f"full_past_any_event_within_{window}cycles"
                future_score, direction = future_oriented_score(numeric(df, future_col), x)
                controls = {
                    future_col: numeric(df, future_col),
                    past_col: numeric(df, past_col),
                    "current_any_event": numeric(df, "current_any_event"),
                }
                metrics = {name: metric_with_fixed_score(y, future_score) for name, y in controls.items()}
                future_auc = metrics[future_col]["roc_auc"]
                control_auc = np.nanmax([
                    metrics[past_col]["roc_auc"],
                    metrics["current_any_event"]["roc_auc"],
                ])
                rows.append({
                    "feature": feature,
                    "transform": transform,
                    "window_cycles": window,
                    "future_label": future_col,
                    "future_direction": direction,
                    "future_auc": future_auc,
                    "future_ap": metrics[future_col]["average_precision"],
                    "future_spearman_rho": metrics[future_col]["spearman_rho"],
                    "future_spearman_p": metrics[future_col]["spearman_p"],
                    "future_n_positive": metrics[future_col]["n_positive"],
                    "past_auc_fixed_direction": metrics[past_col]["roc_auc"],
                    "past_ap_fixed_direction": metrics[past_col]["average_precision"],
                    "past_n_positive": metrics[past_col]["n_positive"],
                    "current_auc_fixed_direction": metrics["current_any_event"]["roc_auc"],
                    "current_ap_fixed_direction": metrics["current_any_event"]["average_precision"],
                    "current_n_positive": metrics["current_any_event"]["n_positive"],
                    "future_minus_max_control_auc": future_auc - control_auc if np.isfinite(future_auc) and np.isfinite(control_auc) else np.nan,
                    "future_minus_past_auc": future_auc - metrics[past_col]["roc_auc"] if np.isfinite(future_auc) and np.isfinite(metrics[past_col]["roc_auc"]) else np.nan,
                    "future_median_positive_minus_negative": metrics[future_col]["median_positive_minus_negative"],
                    "past_median_positive_minus_negative_fixed_direction": metrics[past_col]["median_positive_minus_negative"],
                    "current_median_positive_minus_negative_fixed_direction": metrics["current_any_event"]["median_positive_minus_negative"],
                    "n": metrics[future_col]["n"],
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["window_cycles", "future_minus_max_control_auc", "future_auc", "future_ap"], ascending=[True, False, False, False])
    return out


def shift_labels_within_source(df: pd.DataFrame, label_col: str, rng: np.random.Generator) -> pd.Series:
    shifted = pd.Series(index=df.index, dtype=float)
    cycle_table = df[["source_stem", "cycleNo", label_col]].drop_duplicates().copy()
    for source, sub in cycle_table.groupby("source_stem"):
        sub = sub.sort_values("cycleNo")
        labels = sub[label_col].to_numpy()
        if len(labels) <= 1:
            shift = 0
        else:
            shift = int(rng.integers(1, len(labels)))
        shifted_labels = np.roll(labels, shift)
        mapping = dict(zip(sub["cycleNo"].to_numpy(), shifted_labels))
        mask = df["source_stem"].eq(source)
        shifted.loc[mask] = df.loc[mask, "cycleNo"].map(mapping).astype(float)
    return shifted


def shift_null(df: pd.DataFrame, feature: str, transform: str, window: int, permutations: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    label_col = f"full_future_any_event_within_{window}cycles"
    x = source_transform(numeric(df, feature), df["source_stem"].astype(str), transform)
    score, direction = future_oriented_score(numeric(df, label_col), x)
    observed = metric_with_fixed_score(numeric(df, label_col), score)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(permutations):
        y_perm = shift_labels_within_source(df, label_col, rng)
        met = metric_with_fixed_score(y_perm, score)
        met.update({
            "permutation": i,
            "feature": feature,
            "transform": transform,
            "window_cycles": window,
            "label_col": label_col,
            "future_direction": direction,
        })
        rows.append(met)
    null = pd.DataFrame(rows)
    null_auc = pd.to_numeric(null["roc_auc"], errors="coerce").dropna()
    null_ap = pd.to_numeric(null["average_precision"], errors="coerce").dropna()
    summary = {
        "feature": feature,
        "transform": transform,
        "window_cycles": window,
        "label_col": label_col,
        "future_direction": direction,
        "observed_roc_auc": observed.get("roc_auc"),
        "observed_average_precision": observed.get("average_precision"),
        "observed_spearman_rho": observed.get("spearman_rho"),
        "n_permutations": int(len(null)),
        "null_roc_auc_mean": float(null_auc.mean()) if len(null_auc) else np.nan,
        "null_roc_auc_p95": float(null_auc.quantile(0.95)) if len(null_auc) else np.nan,
        "null_average_precision_mean": float(null_ap.mean()) if len(null_ap) else np.nan,
        "null_average_precision_p95": float(null_ap.quantile(0.95)) if len(null_ap) else np.nan,
        "empirical_p_roc_auc": float((1 + (null_auc >= observed.get("roc_auc", np.nan)).sum()) / (1 + len(null_auc))) if len(null_auc) else np.nan,
        "empirical_p_average_precision": float((1 + (null_ap >= observed.get("average_precision", np.nan)).sum()) / (1 + len(null_ap))) if len(null_ap) else np.nan,
    }
    return null, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_dictionary_audit")
    parser.add_argument("--event-cycles", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_targets/particle_abrupt_events.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_temporal_specificity_audit")
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(feature_dir / "source_balanced_residual_dictionary_features.csv")
    event_cycles = load_event_cycles(Path(args.event_cycles))
    df = append_temporal_labels(df, event_cycles)
    features = [f for f in FEATURES if f in df.columns]
    tests = temporal_tests(df, features)

    primary_feature = "dictionary_recon_error_last_minus_first"
    primary_transform = "source_residual"
    null, null_summary = shift_null(df, primary_feature, primary_transform, 16, args.permutations, args.seed)

    best_future_specific = tests.sort_values(
        ["future_minus_max_control_auc", "future_auc", "future_ap"], ascending=False
    ).head(16)
    primary_rows = tests[(tests["feature"] == primary_feature) & (tests["transform"] == primary_transform)].copy()

    paths = {
        "temporal_tests": out / "source_balanced_residual_temporal_specificity_tests.csv",
        "shift_null": out / "source_balanced_residual_temporal_specificity_shift_null.csv",
        "summary": out / "source_balanced_residual_temporal_specificity_summary.json",
    }
    tests.to_csv(paths["temporal_tests"], index=False)
    null.to_csv(paths["shift_null"], index=False)

    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "event_cycles": [float(c) for c in event_cycles],
        "features_tested": features,
        "transforms": TRANSFORMS,
        "windows": WINDOWS,
        "label_counts": {
            "current_any_event": int(numeric(df, "current_any_event").sum()),
            **{f"future{w}": int(numeric(df, f"full_future_any_event_within_{w}cycles").sum()) for w in WINDOWS},
            **{f"past{w}": int(numeric(df, f"full_past_any_event_within_{w}cycles").sum()) for w in WINDOWS},
        },
        "best_future_specific": best_future_specific.to_dict(orient="records"),
        "primary_source_residual_rows": primary_rows.to_dict(orient="records"),
        "primary_future16_shift_null": null_summary,
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Temporal labels come from full particle abrupt-drop cycles but are evaluated on the 96 source-balanced ROI rows. Future-oriented scores use the future-label sign and apply the same sign to past/current controls. This tests temporal specificity, not causality or deployable warning.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Source-Balanced Residual Temporal Specificity Audit",
        "",
        f"Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"Event cycles: {summary['event_cycles']}",
        f"Label counts: {summary['label_counts']}",
        "",
        "## Primary Candidate",
    ]
    for row in summary["primary_source_residual_rows"]:
        readme.append(
            f"- window {row.get('window_cycles')}: future AUC={row.get('future_auc'):.3f}, "
            f"past AUC={row.get('past_auc_fixed_direction'):.3f}, current AUC={row.get('current_auc_fixed_direction'):.3f}, "
            f"future-control delta={row.get('future_minus_max_control_auc'):.3f}"
        )
    readme.extend([
        "",
        "## Shift Null",
        f"- future16 source-residual {primary_feature}: observed AUC={null_summary.get('observed_roc_auc'):.3f}, "
        f"null p95={null_summary.get('null_roc_auc_p95'):.3f}, empirical p={null_summary.get('empirical_p_roc_auc'):.3f}",
        "",
        "## Guardrail",
        summary["guardrail"],
        "",
    ])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()
