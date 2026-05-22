#!/usr/bin/env python3
"""Masked rollout benchmark for source-balanced pre-event particle crops.

This benchmark uses only the early frames of each exported particle-region ROI
sequence, rolls out the held-out tail with simple interpretable predictors, and
scores errors inside a history-derived particle mask. It is intentionally not a
large learned model: the point is to test whether physics-shaped baselines
(velocity, linear trend, particle-mask drift, radial-profile trend) beat
persistence in event-relative particle regions without using background pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


METHODS = ["persistence", "velocity", "pixel_linear", "particle_mean_drift", "radial_profile_trend"]
TARGETS = {
    "near_vs_any_non_near": ("near_pre_event_1_8", None),
    "near_vs_mid_pre": ("near_pre_event_1_8", "mid_pre_event_9_16"),
    "near_vs_far_pre": ("near_pre_event_1_8", "far_pre_event_17_32"),
    "near_vs_post_control": ("near_pre_event_1_8", {"post_event_1_16", "no_near_event_control"}),
}


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


def stable_mask_from_history(history: np.ndarray) -> np.ndarray:
    base = np.nanmedian(history, axis=0)
    high = base >= np.nanpercentile(base, 70)
    low = base <= np.nanpercentile(base, 30)
    mask = high if high.mean() <= low.mean() else low
    if mask.mean() < 0.03 or mask.mean() > 0.75:
        yy, xx = np.indices(base.shape)
        cy, cx = (np.array(base.shape) - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = rr <= 0.38 * min(base.shape)
    return mask.astype(bool)


def radial_bins(mask: np.ndarray, n_bins: int = 28) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices(mask.shape)
    if mask.any():
        cy = float(yy[mask].mean())
        cx = float(xx[mask].mean())
    else:
        cy, cx = (np.array(mask.shape) - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.linspace(0, float(rr.max()) + 1e-6, n_bins + 1)
    labels = np.digitize(rr, bins) - 1
    labels = np.clip(labels, 0, n_bins - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return labels, centers, rr


def radial_profile(frame: np.ndarray, labels: np.ndarray, n_bins: int) -> np.ndarray:
    prof = np.full(n_bins, np.nan, dtype=float)
    flat = frame.ravel()
    lab = labels.ravel()
    for b in range(n_bins):
        vals = flat[lab == b]
        if vals.size:
            prof[b] = np.nanmean(vals)
    return pd.Series(prof).interpolate(limit_direction="both").bfill().ffill().to_numpy()


def rollout_predictions(frames: np.ndarray, train_n: int, mask: np.ndarray) -> Dict[str, np.ndarray]:
    history = frames[:train_n]
    horizon = frames.shape[0] - train_n
    last = history[-1]
    prev = history[-2] if train_n >= 2 else history[-1]
    t = np.arange(train_n, dtype=float)
    t_center = t - t.mean()
    denom = float(np.sum(t_center ** 2)) or 1.0
    pix_slope = np.tensordot(t_center, history, axes=(0, 0)) / denom
    mean_series = history[:, mask].mean(axis=1) if mask.any() else history.mean(axis=(1, 2))
    mean_slope = float(np.dot(t_center, mean_series) / denom)

    n_bins = 28
    labels, _, _ = radial_bins(mask, n_bins=n_bins)
    profiles = np.vstack([radial_profile(f, labels, n_bins) for f in history])
    prof_slope = np.tensordot(t_center, profiles, axes=(0, 0)) / denom
    prof_last = profiles[-1]
    last_profile = radial_profile(last, labels, n_bins)
    profile_resid = last - last_profile[labels]

    preds: Dict[str, List[np.ndarray]] = {m: [] for m in METHODS}
    velocity = last - prev
    for step in range(1, horizon + 1):
        preds["persistence"].append(last.copy())
        preds["velocity"].append(last + step * velocity)
        preds["pixel_linear"].append(last + step * pix_slope)
        drift = np.zeros_like(last)
        drift[mask] = step * mean_slope
        preds["particle_mean_drift"].append(last + drift)
        prof_pred = prof_last + step * prof_slope
        preds["radial_profile_trend"].append(prof_pred[labels] + profile_resid)
    return {k: np.clip(np.stack(v), 0, 1) for k, v in preds.items()}


def mse_mae(pred: np.ndarray, truth: np.ndarray, region: np.ndarray) -> Tuple[float, float]:
    if not region.any():
        region = np.ones(truth.shape[1:], dtype=bool)
    diff = pred[:, region] - truth[:, region]
    return float(np.nanmean(diff ** 2)), float(np.nanmean(np.abs(diff)))


def target_series(df: pd.DataFrame, positive: Any, negative: Any) -> pd.Series:
    bins = df["event_relative_bin"].astype(str)
    pos = positive if isinstance(positive, set) else {positive}
    y = pd.Series(np.nan, index=df.index)
    y.loc[bins.isin(pos)] = 1
    if negative is None:
        y.loc[~bins.isin(pos)] = 0
    else:
        neg = negative if isinstance(negative, set) else {negative}
        y.loc[bins.isin(neg)] = 0
    return y


def source_residual(values: pd.Series, sources: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    return x - x.groupby(sources.astype(str)).transform("mean")


def event_tests(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target, (positive, negative) in TARGETS.items():
        y = target_series(df, positive, negative)
        for feature in features:
            for transform in ["raw", "source_residual"]:
                x = numeric(df, feature)
                if transform == "source_residual":
                    x = source_residual(x, df["source_stem"])
                valid = y.isin([0, 1]) & x.notna()
                yy = y[valid].astype(int)
                xx = x[valid]
                if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
                    continue
                direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
                score = xx if direction == "higher_in_positive" else -xx
                try:
                    _, p_mwu = mannwhitneyu(xx[yy == 1], xx[yy == 0], alternative="two-sided")
                except ValueError:
                    p_mwu = np.nan
                rho = spearmanr(yy, score)
                rows.append({
                    "target": target,
                    "feature": feature,
                    "transform": transform,
                    "n": int(valid.sum()),
                    "n_positive": int(yy.sum()),
                    "direction": direction,
                    "oriented_auc": float(roc_auc_score(yy, score)),
                    "average_precision": float(average_precision_score(yy, score)),
                    "median_positive_minus_negative": float(xx[yy == 1].median() - xx[yy == 0].median()),
                    "mwu_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
                    "spearman_rho": float(rho.statistic) if np.isfinite(rho.statistic) else np.nan,
                    "spearman_p": float(rho.pvalue) if np.isfinite(rho.pvalue) else np.nan,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mwu_p", "oriented_auc", "average_precision"], ascending=[True, False, False])
    return out


def analyze_row(row: pd.Series, train_fraction: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = np.load(row["npz_path"])
    frames = np.asarray(data["frames_norm"] if "frames_norm" in data else data["frames"], dtype=float)
    if frames.ndim != 3 or frames.shape[0] < 12:
        raise ValueError(f"bad frames for {row.get('roi_id')}")
    train_n = int(np.clip(round(frames.shape[0] * train_fraction), 6, frames.shape[0] - 4))
    truth = frames[train_n:]
    history = frames[:train_n]
    mask = stable_mask_from_history(history)
    context = ~mask
    preds = rollout_predictions(frames, train_n, mask)

    frame_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = row.to_dict()
    summary.update({
        "n_frames_loaded": int(frames.shape[0]),
        "train_n": int(train_n),
        "horizon_n": int(frames.shape[0] - train_n),
        "history_mask_fraction": float(mask.mean()),
    })
    for method, pred in preds.items():
        particle_mse, particle_mae = mse_mae(pred, truth, mask)
        context_mse, context_mae = mse_mae(pred, truth, context)
        full_mse, full_mae = mse_mae(pred, truth, np.ones(mask.shape, dtype=bool))
        summary[f"{method}_particle_mse"] = particle_mse
        summary[f"{method}_particle_mae"] = particle_mae
        summary[f"{method}_context_mse"] = context_mse
        summary[f"{method}_context_mae"] = context_mae
        summary[f"{method}_full_mse"] = full_mse
        summary[f"{method}_full_mae"] = full_mae
        for h in [1, max(1, len(truth) // 2), len(truth)]:
            p = pred[h - 1:h]
            t = truth[h - 1:h]
            pm, pa = mse_mae(p, t, mask)
            cm, ca = mse_mae(p, t, context)
            frame_rows.append({
                "roi_id": row["roi_id"],
                "method": method,
                "horizon_step": int(h),
                "particle_mse": pm,
                "particle_mae": pa,
                "context_mse": cm,
                "context_mae": ca,
            })
    base = summary["persistence_particle_mse"]
    for method in METHODS:
        val = summary[f"{method}_particle_mse"]
        summary[f"{method}_particle_mse_ratio_vs_persistence"] = val / base if base > 0 else np.nan
        summary[f"{method}_particle_mse_gain_vs_persistence"] = (base - val) / base if base > 0 else np.nan
    gains = {m: summary[f"{m}_particle_mse_gain_vs_persistence"] for m in METHODS if m != "persistence"}
    best_method = max(gains, key=lambda k: gains[k] if np.isfinite(gains[k]) else -np.inf)
    summary["best_nonpersistence_method"] = best_method
    summary["best_nonpersistence_particle_gain"] = float(gains[best_method])
    summary["radial_gain_minus_velocity_gain"] = summary["radial_profile_trend_particle_mse_gain_vs_persistence"] - summary["velocity_particle_mse_gain_vs_persistence"]
    summary["particle_vs_context_persistence_mse_ratio"] = summary["persistence_particle_mse"] / summary["persistence_context_mse"] if summary["persistence_context_mse"] > 0 else np.nan
    return frame_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_masked_rollout_benchmark")
    parser.add_argument("--train-fraction", type=float, default=0.60)
    args = parser.parse_args()

    roi_dir = Path(args.roi_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(roi_dir / "selected_roi_sequence_manifest.csv")

    frame_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        try:
            frows, summary = analyze_row(row, args.train_fraction)
            frame_rows.extend(frows)
            summaries.append(summary)
        except Exception as exc:
            failures.append({"roi_id": row.get("roi_id"), "npz_path": row.get("npz_path"), "error": str(exc)})

    per_roi = pd.DataFrame(summaries)
    frames = pd.DataFrame(frame_rows)
    failures_df = pd.DataFrame(failures)

    feature_cols = [
        "persistence_particle_mse",
        "velocity_particle_mse_gain_vs_persistence",
        "pixel_linear_particle_mse_gain_vs_persistence",
        "particle_mean_drift_particle_mse_gain_vs_persistence",
        "radial_profile_trend_particle_mse_gain_vs_persistence",
        "best_nonpersistence_particle_gain",
        "radial_gain_minus_velocity_gain",
        "particle_vs_context_persistence_mse_ratio",
        "history_mask_fraction",
    ]
    tests = event_tests(per_roi, feature_cols) if not per_roi.empty else pd.DataFrame()
    method_summary_rows = []
    for method in METHODS:
        if per_roi.empty:
            continue
        method_summary_rows.append({
            "method": method,
            "particle_mse_median": float(per_roi[f"{method}_particle_mse"].median()),
            "particle_mse_mean": float(per_roi[f"{method}_particle_mse"].mean()),
            "particle_mse_ratio_vs_persistence_median": float(per_roi[f"{method}_particle_mse_ratio_vs_persistence"].median()),
            "particle_mse_gain_vs_persistence_median": float(per_roi[f"{method}_particle_mse_gain_vs_persistence"].median()),
            "wins_vs_persistence": int((per_roi[f"{method}_particle_mse_gain_vs_persistence"] > 0).sum()),
        })
    method_summary = pd.DataFrame(method_summary_rows)
    source_summary = (
        per_roi.groupby("source_stem", as_index=False)
        .agg(
            n_roi=("roi_id", "count"),
            n_cycles=("cycleNo", "nunique"),
            future8=("future_any_drop_within_8cycles", "sum"),
            future16=("future_any_drop_within_16cycles", "sum"),
            persistence_particle_mse=("persistence_particle_mse", "mean"),
            best_nonpersistence_particle_gain=("best_nonpersistence_particle_gain", "mean"),
            radial_gain=("radial_profile_trend_particle_mse_gain_vs_persistence", "mean"),
            velocity_gain=("velocity_particle_mse_gain_vs_persistence", "mean"),
        )
        if not per_roi.empty
        else pd.DataFrame()
    )

    per_roi_path = out / "source_balanced_pre_event_masked_rollout_per_roi.csv"
    frame_path = out / "source_balanced_pre_event_masked_rollout_frame_samples.csv"
    method_path = out / "source_balanced_pre_event_masked_rollout_method_summary.csv"
    tests_path = out / "source_balanced_pre_event_masked_rollout_event_tests.csv"
    source_path = out / "source_balanced_pre_event_masked_rollout_source_summary.csv"
    failures_path = out / "source_balanced_pre_event_masked_rollout_failures.csv"
    summary_path = out / "source_balanced_pre_event_masked_rollout_summary.json"

    per_roi.to_csv(per_roi_path, index=False)
    frames.to_csv(frame_path, index=False)
    method_summary.to_csv(method_path, index=False)
    tests.to_csv(tests_path, index=False)
    source_summary.to_csv(source_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    top_tests = tests.head(16).to_dict("records") if not tests.empty else []
    best_method = (
        method_summary.sort_values("particle_mse_gain_vs_persistence_median", ascending=False).head(1).to_dict("records")[0]
        if not method_summary.empty
        else {}
    )
    summary = {
        "n_input_rows": int(len(manifest)),
        "n_ok": int(len(per_roi)),
        "n_failed": int(len(failures_df)),
        "n_sources": int(per_roi["source_stem"].nunique()) if not per_roi.empty else 0,
        "n_cycles": int(per_roi["cycleNo"].nunique()) if not per_roi.empty else 0,
        "train_fraction": float(args.train_fraction),
        "methods": METHODS,
        "best_method_by_median_particle_gain": clean_json(best_method),
        "method_summary": clean_json(method_summary.to_dict("records")),
        "top_event_tests": clean_json(top_tests),
        "event_relative_bin_counts": clean_json(per_roi["event_relative_bin"].value_counts().to_dict()) if not per_roi.empty else {},
        "outputs": {
            "per_roi_csv": str(per_roi_path),
            "frame_samples_csv": str(frame_path),
            "method_summary_csv": str(method_path),
            "event_tests_csv": str(tests_path),
            "source_summary_csv": str(source_path),
            "failures_csv": str(failures_path),
            "summary_json": str(summary_path),
            "readme": str(out / "README.md"),
        },
        "guardrail": (
            "Rollouts use history-derived automatic particle masks and held-out future frames within each ROI crop. "
            "They test interpretable particle-region prediction baselines, not manual segmentation, calibrated phase-boundary motion, or material diffusion."
        ),
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")

    readme = [
        "# Source-Balanced Pre-Event Masked Rollout Benchmark",
        "",
        f"- Input rows / OK / failed: {len(manifest)} / {len(per_roi)} / {len(failures_df)}",
        f"- Methods: {METHODS}",
        f"- Best method by median particle gain: {best_method}",
        f"- Top event test: {top_tests[0] if top_tests else 'NA'}",
        "",
        "Outputs:",
        f"- `{per_roi_path.name}`",
        f"- `{frame_path.name}`",
        f"- `{method_path.name}`",
        f"- `{tests_path.name}`",
        f"- `{source_path.name}`",
        f"- `{summary_path.name}`",
        "",
        "Guardrail:",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(readme).rstrip() + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
