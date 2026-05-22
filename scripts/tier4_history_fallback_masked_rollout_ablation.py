#!/usr/bin/env python3
"""Ablate history/fallback particle masks for next-frame and rollout residuals.

This audit directly tests the drift-correction blur concern in the predictive
setting. For each source-balanced pre-event particle crop it builds framewise
adaptive masks, an early-history particle prior, and a hybrid fallback mask that
uses the history prior on blurry/unstable frames. It then scores one-step
persistence residuals and held-out particle-only rollouts, including a compact
latent-linear model trained only on history-mask particle pixels.
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


TARGETS = {
    "near_vs_any_non_near": ("near_pre_event_1_8", None),
    "near_vs_post_control": ("near_pre_event_1_8", {"post_event_1_16", "no_near_event_control"}),
    "near_vs_mid_pre": ("near_pre_event_1_8", "mid_pre_event_9_16"),
    "near_vs_far_pre": ("near_pre_event_1_8", "far_pre_event_17_32"),
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


def topk_mask(frame: np.ndarray, area: int) -> np.ndarray:
    h, w = frame.shape
    yy, xx = np.mgrid[:h, :w]
    border = np.concatenate([frame[:4, :].ravel(), frame[-4:, :].ravel(), frame[:, :4].ravel(), frame[:, -4:].ravel()])
    baseline = np.nanmedian(border)
    contrast = np.abs(frame - baseline)
    radius = np.sqrt((yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2)
    central_weight = np.clip(1.0 - radius / (0.75 * min(h, w)), 0.25, 1.0)
    score = contrast * central_weight
    flat = score.ravel()
    area = int(np.clip(area, 16, h * w // 3))
    idx = np.argpartition(flat, -area)[-area:]
    mask = np.zeros(len(flat), dtype=bool)
    mask[idx] = True
    return mask.reshape(frame.shape)


def centroid(mask: np.ndarray) -> Tuple[float, float]:
    y, x = np.where(mask)
    if len(y) == 0:
        return float("nan"), float("nan")
    return float(y.mean()), float(x.mean())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def sharpness(frame: np.ndarray, mask: np.ndarray) -> float:
    gy, gx = np.gradient(frame.astype(float))
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.nanmean(mag[mask])) if mask.any() else float("nan")


def build_masks(frames: np.ndarray, object_area: float) -> Dict[str, Any]:
    n = len(frames)
    area = int(max(16, object_area if np.isfinite(object_area) else 64))
    adaptive = np.stack([topk_mask(frames[i], area) for i in range(n)])
    early_n = max(8, min(24, n // 4))
    vote = adaptive[:early_n].mean(axis=0)
    history = vote >= 0.5
    if history.sum() < 8:
        history = vote >= np.quantile(vote, 0.98)
    hist_cy, hist_cx = centroid(history)
    ious = np.array([iou(m, history) for m in adaptive], dtype=float)
    cents = np.array([centroid(m) for m in adaptive], dtype=float)
    dist = np.sqrt((cents[:, 0] - hist_cy) ** 2 + (cents[:, 1] - hist_cx) ** 2)
    sharp = np.array([sharpness(frames[i], adaptive[i]) for i in range(n)], dtype=float)
    early_sharp = np.nanmedian(sharp[:early_n])
    blur_ratio = sharp / early_sharp if np.isfinite(early_sharp) and early_sharp > 0 else np.full(n, np.nan)
    fallback = (ious < 0.20) | (dist > 8.0) | (blur_ratio < 0.65)
    hybrid = adaptive.copy()
    hybrid[fallback] = history
    return {
        "adaptive": adaptive,
        "history": history,
        "hybrid": hybrid,
        "fallback": fallback,
        "history_iou": ious,
        "centroid_dist": dist,
        "blur_ratio": blur_ratio,
    }


def masked_mse_by_time(diff: np.ndarray, masks: np.ndarray | np.ndarray) -> np.ndarray:
    vals: List[float] = []
    if masks.ndim == 2:
        for d in diff:
            vals.append(float(np.nanmean(d[masks]))) if masks.any() else vals.append(float(np.nanmean(d)))
    else:
        for d, m in zip(diff, masks):
            vals.append(float(np.nanmean(d[m]))) if m.any() else vals.append(float(np.nanmean(d)))
    return np.asarray(vals, dtype=float)


def fit_latent_linear_rollout(history_pixels: np.ndarray, horizon: int, rank: int = 8, ridge: float = 1e-3) -> np.ndarray:
    x = np.asarray(history_pixels, dtype=float)
    if x.shape[0] < 6 or x.shape[1] < 2 or horizon < 1:
        return np.repeat(x[-1:], horizon, axis=0)
    mu = x.mean(axis=0)
    xc = x - mu
    try:
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.repeat(x[-1:], horizon, axis=0)
    k = int(min(rank, vt.shape[0], x.shape[0] - 2, x.shape[1]))
    if k < 1:
        return np.repeat(x[-1:], horizon, axis=0)
    comp = vt[:k].T
    z = xc @ comp
    zx = z[:-1]
    zy = z[1:]
    lhs = zx.T @ zx + ridge * np.eye(k)
    rhs = zx.T @ zy
    try:
        a = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        a = np.linalg.pinv(lhs) @ rhs
    cur = z[-1]
    preds = []
    for _ in range(horizon):
        cur = cur @ a
        preds.append(mu + cur @ comp.T)
    return np.clip(np.vstack(preds), 0, 1)


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


def analyze_row(row: pd.Series, train_fraction: float, latent_rank: int) -> Dict[str, Any]:
    z = np.load(row["npz_path"])
    frames = np.asarray(z["frames_norm"] if "frames_norm" in z else z["frames"], dtype=float)
    if frames.ndim != 3 or len(frames) < 12:
        raise ValueError(f"bad frame tensor for {row.get('roi_id')}")
    masks = build_masks(frames, float(row.get("object_area_ds_px", np.nan)))
    history = masks["history"]
    train_n = int(np.clip(round(len(frames) * train_fraction), 6, len(frames) - 4))
    horizon = len(frames) - train_n

    one_step_err = (frames[1:] - frames[:-1]) ** 2
    adaptive_next = masks["adaptive"][1:]
    hybrid_next = masks["hybrid"][1:]
    history_one_step = masked_mse_by_time(one_step_err, history)
    adaptive_one_step = masked_mse_by_time(one_step_err, adaptive_next)
    hybrid_one_step = masked_mse_by_time(one_step_err, hybrid_next)

    truth_tail = frames[train_n:]
    persistence_tail = np.repeat(frames[train_n - 1:train_n], horizon, axis=0)
    mask_pixels = history.ravel()
    latent_pixels = fit_latent_linear_rollout(frames[:train_n].reshape(train_n, -1)[:, mask_pixels], horizon, rank=latent_rank)
    latent_tail = persistence_tail.copy().reshape(horizon, -1)
    latent_tail[:, mask_pixels] = latent_pixels
    latent_tail = latent_tail.reshape(truth_tail.shape)

    tail_err_persist = (persistence_tail - truth_tail) ** 2
    tail_err_latent = (latent_tail - truth_tail) ** 2
    tail_adaptive = masks["adaptive"][train_n:]
    tail_hybrid = masks["hybrid"][train_n:]
    persist_history = masked_mse_by_time(tail_err_persist, history)
    persist_hybrid = masked_mse_by_time(tail_err_persist, tail_hybrid)
    latent_history = masked_mse_by_time(tail_err_latent, history)
    latent_hybrid = masked_mse_by_time(tail_err_latent, tail_hybrid)

    one_step_delta = hybrid_one_step - adaptive_one_step
    latent_gain_history = (np.nanmean(persist_history) - np.nanmean(latent_history)) / np.nanmean(persist_history)
    latent_gain_hybrid = (np.nanmean(persist_hybrid) - np.nanmean(latent_hybrid)) / np.nanmean(persist_hybrid)
    out = row.to_dict()
    out.update({
        "n_frames_loaded": int(len(frames)),
        "train_n": int(train_n),
        "horizon_n": int(horizon),
        "history_mask_area_px": int(history.sum()),
        "history_mask_fraction": float(history.mean()),
        "fallback_frame_fraction": float(np.nanmean(masks["fallback"])),
        "fallback_frame_count": int(np.nansum(masks["fallback"])),
        "median_history_iou": float(np.nanmedian(masks["history_iou"])),
        "centroid_jitter_q90_px": float(np.nanquantile(masks["centroid_dist"], 0.90)),
        "blur_ratio_q10": float(np.nanquantile(masks["blur_ratio"], 0.10)),
        "one_step_history_mse": float(np.nanmean(history_one_step)),
        "one_step_adaptive_mse": float(np.nanmean(adaptive_one_step)),
        "one_step_hybrid_mse": float(np.nanmean(hybrid_one_step)),
        "one_step_hybrid_minus_adaptive_mse": float(np.nanmean(one_step_delta)),
        "one_step_hybrid_adaptive_abs_delta_median": float(np.nanmedian(np.abs(one_step_delta))),
        "one_step_hybrid_adaptive_rel_delta": float(np.nanmean(one_step_delta) / np.nanmean(adaptive_one_step)) if np.nanmean(adaptive_one_step) > 0 else np.nan,
        "persistence_rollout_history_mse": float(np.nanmean(persist_history)),
        "persistence_rollout_hybrid_mse": float(np.nanmean(persist_hybrid)),
        "latent_linear_rollout_history_mse": float(np.nanmean(latent_history)),
        "latent_linear_rollout_hybrid_mse": float(np.nanmean(latent_hybrid)),
        "latent_linear_gain_vs_persistence_history": float(latent_gain_history) if np.isfinite(latent_gain_history) else np.nan,
        "latent_linear_gain_vs_persistence_hybrid": float(latent_gain_hybrid) if np.isfinite(latent_gain_hybrid) else np.nan,
        "hybrid_vs_history_persistence_rollout_mse_delta": float(np.nanmean(persist_hybrid) - np.nanmean(persist_history)),
        "hybrid_vs_history_latent_rollout_mse_delta": float(np.nanmean(latent_hybrid) - np.nanmean(latent_history)),
        "late_one_step_hybrid_mse": float(np.nanmean(hybrid_one_step[len(hybrid_one_step) // 2:])),
        "early_one_step_hybrid_mse": float(np.nanmean(hybrid_one_step[:max(1, len(hybrid_one_step) // 2)])),
    })
    out["late_minus_early_one_step_hybrid_mse"] = out["late_one_step_hybrid_mse"] - out["early_one_step_hybrid_mse"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/history_fallback_masked_rollout_ablation")
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--latent-rank", type=int, default=8)
    args = parser.parse_args()

    roi_dir = Path(args.roi_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(roi_dir / "selected_roi_sequence_manifest.csv")

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        try:
            rows.append(analyze_row(row, args.train_fraction, args.latent_rank))
        except Exception as exc:
            failures.append({"roi_id": row.get("roi_id"), "npz_path": row.get("npz_path"), "error": repr(exc)})

    per_roi = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    features = [
        "fallback_frame_fraction",
        "median_history_iou",
        "centroid_jitter_q90_px",
        "blur_ratio_q10",
        "one_step_history_mse",
        "one_step_adaptive_mse",
        "one_step_hybrid_mse",
        "one_step_hybrid_minus_adaptive_mse",
        "one_step_hybrid_adaptive_abs_delta_median",
        "one_step_hybrid_adaptive_rel_delta",
        "persistence_rollout_history_mse",
        "persistence_rollout_hybrid_mse",
        "latent_linear_rollout_history_mse",
        "latent_linear_rollout_hybrid_mse",
        "latent_linear_gain_vs_persistence_history",
        "latent_linear_gain_vs_persistence_hybrid",
        "hybrid_vs_history_persistence_rollout_mse_delta",
        "hybrid_vs_history_latent_rollout_mse_delta",
        "late_minus_early_one_step_hybrid_mse",
    ]
    tests = event_tests(per_roi, features) if not per_roi.empty else pd.DataFrame()
    source_summary = (
        per_roi.groupby("source_stem", as_index=False)
        .agg(
            n_roi=("roi_id", "count"),
            n_cycles=("cycleNo", "nunique"),
            future8=("future_any_drop_within_8cycles", "sum"),
            future16=("future_any_drop_within_16cycles", "sum"),
            median_fallback=("fallback_frame_fraction", "median"),
            one_step_hybrid_mse=("one_step_hybrid_mse", "median"),
            latent_gain_history=("latent_linear_gain_vs_persistence_history", "median"),
            latent_gain_hybrid=("latent_linear_gain_vs_persistence_hybrid", "median"),
        )
        if not per_roi.empty
        else pd.DataFrame()
    )
    method_summary = pd.DataFrame([
        {
            "metric": "one_step_hybrid_minus_adaptive_mse",
            "median": float(per_roi["one_step_hybrid_minus_adaptive_mse"].median()) if not per_roi.empty else np.nan,
            "q10": float(per_roi["one_step_hybrid_minus_adaptive_mse"].quantile(0.10)) if not per_roi.empty else np.nan,
            "q90": float(per_roi["one_step_hybrid_minus_adaptive_mse"].quantile(0.90)) if not per_roi.empty else np.nan,
        },
        {
            "metric": "latent_linear_gain_vs_persistence_history",
            "median": float(per_roi["latent_linear_gain_vs_persistence_history"].median()) if not per_roi.empty else np.nan,
            "q10": float(per_roi["latent_linear_gain_vs_persistence_history"].quantile(0.10)) if not per_roi.empty else np.nan,
            "q90": float(per_roi["latent_linear_gain_vs_persistence_history"].quantile(0.90)) if not per_roi.empty else np.nan,
        },
        {
            "metric": "latent_linear_gain_vs_persistence_hybrid",
            "median": float(per_roi["latent_linear_gain_vs_persistence_hybrid"].median()) if not per_roi.empty else np.nan,
            "q10": float(per_roi["latent_linear_gain_vs_persistence_hybrid"].quantile(0.10)) if not per_roi.empty else np.nan,
            "q90": float(per_roi["latent_linear_gain_vs_persistence_hybrid"].quantile(0.90)) if not per_roi.empty else np.nan,
        },
    ])

    paths = {
        "per_roi": out / "history_fallback_masked_rollout_ablation_per_roi.csv",
        "event_tests": out / "history_fallback_masked_rollout_ablation_event_tests.csv",
        "source_summary": out / "history_fallback_masked_rollout_ablation_source_summary.csv",
        "method_summary": out / "history_fallback_masked_rollout_ablation_method_summary.csv",
        "failures": out / "history_fallback_masked_rollout_ablation_failures.csv",
        "summary": out / "history_fallback_masked_rollout_ablation_summary.json",
    }
    per_roi.to_csv(paths["per_roi"], index=False)
    tests.to_csv(paths["event_tests"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)
    method_summary.to_csv(paths["method_summary"], index=False)
    failures_df.to_csv(paths["failures"], index=False)

    top_tests = tests.head(16).to_dict("records") if not tests.empty else []
    summary = {
        "overall_status": "history_fallback_masked_rollout_ablation_ready",
        "n_input_rows": int(len(manifest)),
        "n_ok": int(len(per_roi)),
        "n_failures": int(len(failures_df)),
        "n_sources": int(per_roi["source_stem"].nunique()) if not per_roi.empty else 0,
        "n_cycles": int(per_roi["cycleNo"].nunique()) if not per_roi.empty else 0,
        "train_fraction": float(args.train_fraction),
        "latent_rank": int(args.latent_rank),
        "median_fallback_frame_fraction": float(per_roi["fallback_frame_fraction"].median()) if not per_roi.empty else np.nan,
        "median_one_step_hybrid_mse": float(per_roi["one_step_hybrid_mse"].median()) if not per_roi.empty else np.nan,
        "median_one_step_adaptive_mse": float(per_roi["one_step_adaptive_mse"].median()) if not per_roi.empty else np.nan,
        "median_hybrid_minus_adaptive_one_step_mse": float(per_roi["one_step_hybrid_minus_adaptive_mse"].median()) if not per_roi.empty else np.nan,
        "median_latent_gain_history": float(per_roi["latent_linear_gain_vs_persistence_history"].median()) if not per_roi.empty else np.nan,
        "median_latent_gain_hybrid": float(per_roi["latent_linear_gain_vs_persistence_hybrid"].median()) if not per_roi.empty else np.nan,
        "method_summary": clean_json(method_summary.to_dict("records")),
        "top_event_tests": clean_json(top_tests),
        "source_summary_top": clean_json(source_summary.sort_values("median_fallback", ascending=False).head(12).to_dict("records")) if not source_summary.empty else [],
        "event_relative_bin_counts": clean_json(per_roi["event_relative_bin"].value_counts().to_dict()) if not per_roi.empty else {},
        "outputs": {key: str(path) for key, path in paths.items()},
        "guardrail": (
            "This is an automatic particle-mask ablation and compact latent-linear rollout benchmark. "
            "It uses history/fallback particle support for robustness under drift-correction blur, but it does not provide manual segmentation, validated phase-boundary velocities, or calibrated diffusion coefficients."
        ),
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    readme = [
        "# History/Fallback Masked Rollout Ablation",
        "",
        f"- Input / OK / failures / sources: {len(manifest)} / {len(per_roi)} / {len(failures_df)} / {summary['n_sources']}",
        f"- Median fallback frame fraction: {summary['median_fallback_frame_fraction']}",
        f"- Median one-step adaptive/hybrid MSE: {summary['median_one_step_adaptive_mse']} / {summary['median_one_step_hybrid_mse']}",
        f"- Median latent-linear gain vs persistence, history/hybrid masks: {summary['median_latent_gain_history']} / {summary['median_latent_gain_hybrid']}",
        f"- Top event test: {top_tests[0] if top_tests else 'NA'}",
        "",
        "Outputs:",
        *[f"- `{path.name}`" for path in paths.values()],
        "",
        "Guardrail:",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(readme).rstrip() + "\n")
    summary["outputs"]["readme"] = str(out / "README.md")
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
