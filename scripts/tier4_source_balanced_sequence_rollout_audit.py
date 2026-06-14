#!/usr/bin/env python3
"""Fast rollout/temporal-dynamics audit for source-balanced particle ROI crops.

This avoids the expensive global high-dimensional DMD fit used by the small
selected-ROI baseline and instead computes interpretable particle-region video
features from each exported source-balanced ROI sequence: one-step persistence
and velocity prediction error, temporal activity, intensity drift, and simple
label/source tests. The NPZ tensors stay on Isambard; outputs are compact CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]


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


def source_eta2(series: pd.Series, sources: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.notna() & sources.notna()
    vals = vals[valid]
    src = sources[valid]
    if vals.nunique() < 2 or src.nunique() < 2:
        return np.nan
    overall = vals.mean()
    total = float(((vals - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in vals.groupby(src):
        between += len(sub) * float((sub.mean() - overall) ** 2)
    return between / total


def orient_auc(y: pd.Series, x: pd.Series) -> Tuple[float, float, str]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
        return np.nan, np.nan, "NA"
    direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
    score = xx if direction == "higher_in_positive" else -xx
    return float(roc_auc_score(yy, score)), float(average_precision_score(yy, score)), direction


def feature_tests(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target in [t for t in TARGETS if t in df.columns]:
        y = numeric(df, target)
        for feat in features:
            x = numeric(df, feat)
            valid = y.isin([0, 1]) & x.notna()
            yy = y[valid].astype(int)
            xx = x[valid]
            auc, ap, direction = orient_auc(y, x)
            rho, sp = (np.nan, np.nan)
            p_mwu = np.nan
            med_pos = med_neg = np.nan
            if valid.sum() >= 8 and yy.nunique() == 2 and xx.nunique() > 1:
                rho, sp = spearmanr(yy, xx if direction == "higher_in_positive" else -xx)
                pos = xx[yy == 1]
                neg = xx[yy == 0]
                med_pos = float(pos.median())
                med_neg = float(neg.median())
                try:
                    _, p_mwu = mannwhitneyu(pos, neg, alternative="two-sided")
                except ValueError:
                    p_mwu = np.nan
            rows.append({
                "target": target,
                "feature": feat,
                "n": int(valid.sum()),
                "n_positive": int(yy.sum()) if len(yy) else 0,
                "direction": direction,
                "oriented_auc": auc,
                "average_precision": ap,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(sp) if np.isfinite(sp) else np.nan,
                "mwu_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
                "median_positive": med_pos,
                "median_negative": med_neg,
                "median_positive_minus_negative": med_pos - med_neg if np.isfinite(med_pos) and np.isfinite(med_neg) else np.nan,
                "source_eta2": source_eta2(x, df["source_stem"]),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "oriented_auc", "average_precision"], ascending=[True, False, False])
    return out


def compute_roi_features(row: pd.Series) -> Dict[str, Any]:
    data = np.load(row["npz_path"])
    frames = data["frames_norm"].astype(np.float32)
    raw = data["frames"].astype(np.float32)
    flat = frames.reshape(frames.shape[0], -1)
    diffs = np.diff(flat, axis=0)
    persistence_err = diffs ** 2
    velocity_pred = np.clip(flat[1:-1] + (flat[1:-1] - flat[:-2]), 0, 1)
    velocity_err = (velocity_pred - flat[2:]) ** 2 if len(flat) >= 3 else np.empty((0, flat.shape[1]))
    roi_mean = np.asarray(data.get("roi_norm_mean", frames.mean(axis=(1, 2))), dtype=float)
    raw_mean = np.asarray(data.get("roi_mean", raw.mean(axis=(1, 2))), dtype=float)
    stage = np.asarray(data.get("stage_position", np.full((3, len(frames)), np.nan)), dtype=float)
    half = len(roi_mean) // 2
    temporal_energy = np.sqrt(np.mean(diffs ** 2, axis=1)) if len(diffs) else np.array([np.nan])
    early_energy = float(np.nanmean(temporal_energy[:max(1, len(temporal_energy)//3)]))
    late_energy = float(np.nanmean(temporal_energy[-max(1, len(temporal_energy)//3):]))
    pos_frac = float(np.nanmean(np.diff(roi_mean) > 0)) if len(roi_mean) > 1 else np.nan
    drift = float(np.sqrt((np.nanmax(stage[0]) - np.nanmin(stage[0])) ** 2 + (np.nanmax(stage[1]) - np.nanmin(stage[1])) ** 2)) if np.isfinite(stage).any() else np.nan
    out = row.to_dict()
    out.update({
        "persistence_mse_mean": float(np.nanmean(persistence_err)),
        "persistence_mse_p95": float(np.nanpercentile(persistence_err, 95)),
        "persistence_mse_late_mean": float(np.nanmean(persistence_err[max(0, len(persistence_err)//2):])),
        "velocity_mse_mean": float(np.nanmean(velocity_err)) if velocity_err.size else np.nan,
        "velocity_mse_p95": float(np.nanpercentile(velocity_err, 95)) if velocity_err.size else np.nan,
        "velocity_minus_persistence_mse": float(np.nanmean(velocity_err) - np.nanmean(persistence_err[1:])) if velocity_err.size and len(persistence_err) > 1 else np.nan,
        "temporal_energy_mean": float(np.nanmean(temporal_energy)),
        "temporal_energy_p95": float(np.nanpercentile(temporal_energy, 95)),
        "temporal_energy_late_minus_early": late_energy - early_energy,
        "roi_norm_mean_first": float(roi_mean[0]),
        "roi_norm_mean_last": float(roi_mean[-1]),
        "roi_norm_mean_delta_last_minus_first": float(roi_mean[-1] - roi_mean[0]),
        "roi_norm_mean_late_minus_early": float(np.nanmean(roi_mean[half:]) - np.nanmean(roi_mean[:half])) if half > 0 else np.nan,
        "roi_norm_mean_positive_step_fraction": pos_frac,
        "raw_roi_mean_delta_last_minus_first": float(raw_mean[-1] - raw_mean[0]),
        "stage_drift_xy_recomputed": drift,
        "n_loaded_frames": int(len(frames)),
        "frame_height": int(frames.shape[1]),
        "frame_width": int(frames.shape[2]),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_sequence_rollout_audit")
    args = parser.parse_args()

    roi_dir = Path(args.roi_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(roi_dir / "selected_roi_sequence_manifest.csv")
    rows = [compute_roi_features(row) for _, row in manifest.iterrows()]
    features = pd.DataFrame(rows)

    rollout_features = [
        "persistence_mse_mean", "persistence_mse_p95", "persistence_mse_late_mean",
        "velocity_mse_mean", "velocity_mse_p95", "velocity_minus_persistence_mse",
        "temporal_energy_mean", "temporal_energy_p95", "temporal_energy_late_minus_early",
        "roi_norm_mean_delta_last_minus_first", "roi_norm_mean_late_minus_early", "roi_norm_mean_positive_step_fraction",
        "raw_roi_mean_delta_last_minus_first", "stage_drift_xy_recomputed", "object_area_ds_px", "object_mean_residual", "object_mean_abs_z",
    ]
    tests = feature_tests(features, rollout_features)
    cycle = features.groupby(["cycleNo", "source_stem"], as_index=False).agg({
        "roi_id": "count",
        "future_any_drop_within_8cycles": "max",
        "future_any_drop_within_16cycles": "max",
        "persistence_mse_mean": "mean",
        "velocity_mse_mean": "mean",
        "temporal_energy_mean": "mean",
        "temporal_energy_late_minus_early": "mean",
        "roi_norm_mean_delta_last_minus_first": "mean",
        "stage_drift_xy_recomputed": "mean",
        "object_mean_abs_z": "mean",
    }).rename(columns={"roi_id": "n_roi"})
    cycle_tests = feature_tests(cycle, [c for c in rollout_features if c in cycle.columns])
    source = features.groupby("source_stem", as_index=False).agg({
        "roi_id": "count",
        "cycleNo": "nunique",
        "future_any_drop_within_8cycles": "sum",
        "future_any_drop_within_16cycles": "sum",
        "persistence_mse_mean": "mean",
        "velocity_mse_mean": "mean",
        "temporal_energy_mean": "mean",
        "roi_norm_mean_delta_last_minus_first": "mean",
        "stage_drift_xy_recomputed": "mean",
    }).rename(columns={"roi_id": "n_roi", "cycleNo": "n_cycles"})

    paths = {
        "features": out / "source_balanced_sequence_rollout_features.csv",
        "feature_tests": out / "source_balanced_sequence_rollout_feature_tests.csv",
        "cycle_summary": out / "source_balanced_sequence_rollout_cycle_summary.csv",
        "cycle_tests": out / "source_balanced_sequence_rollout_cycle_tests.csv",
        "source_summary": out / "source_balanced_sequence_rollout_source_summary.csv",
        "summary": out / "source_balanced_sequence_rollout_summary.json",
    }
    features.to_csv(paths["features"], index=False)
    tests.to_csv(paths["feature_tests"], index=False)
    cycle.to_csv(paths["cycle_summary"], index=False)
    cycle_tests.to_csv(paths["cycle_tests"], index=False)
    source.to_csv(paths["source_summary"], index=False)
    summary = clean_json({
        "roi_dir": str(roi_dir),
        "n_roi_sequences": int(len(features)),
        "n_cycles": int(features["cycleNo"].nunique()),
        "n_sources": int(features["source_stem"].nunique()),
        "future8_positive_sequences": int(numeric(features, "future_any_drop_within_8cycles").fillna(0).sum()),
        "future16_positive_sequences": int(numeric(features, "future_any_drop_within_16cycles").fillna(0).sum()),
        "top_roi_feature_tests": tests.head(30).to_dict("records"),
        "top_cycle_feature_tests": cycle_tests.head(30).to_dict("records"),
        "source_summary": source.to_dict("records"),
        "guardrail": "Source-balanced rollout features are computed from automatic particle-centered crops and weak future labels. They quantify ROI-only temporal prediction difficulty and optical drift/intensity dynamics, not manual QC, causal degradation, or calibrated diffusion.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = [
        "# Source-Balanced Sequence Rollout Audit",
        "",
        "Fast ROI-only temporal prediction and dynamics features for the source-balanced particle crop cohort.",
        "",
        f"- ROI sequences: {summary['n_roi_sequences']}",
        f"- Cycles: {summary['n_cycles']}",
        f"- Sources: {summary['n_sources']}",
        f"- Future8 positive sequences: {summary['future8_positive_sequences']}",
        f"- Future16 positive sequences: {summary['future16_positive_sequences']}",
        "",
        "## Top ROI Feature Tests",
        "",
    ]
    for row in summary["top_roi_feature_tests"][:12]:
        lines.append(f"- {row.get('target')} {row.get('feature')}: AUC {row.get('oriented_auc')}, AP {row.get('average_precision')}, source eta2 {row.get('source_eta2')}")
    lines += ["", "## Top Cycle Feature Tests", ""]
    for row in summary["top_cycle_feature_tests"][:8]:
        lines.append(f"- {row.get('target')} {row.get('feature')}: AUC {row.get('oriented_auc')}, AP {row.get('average_precision')}")
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
