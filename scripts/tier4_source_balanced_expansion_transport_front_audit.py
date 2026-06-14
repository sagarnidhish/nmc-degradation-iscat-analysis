#!/usr/bin/env python3
"""Audit apparent transport/front proxies on the expanded source-balanced cohort.

The pre-event mechanism dossier is intentionally review-focused. This script
checks whether similar automatic transport/front descriptors carry signal in the
broader source-balanced expansion ROI sequences, which were sampled across more
cycles and sources. Outputs are hypothesis-ranking tables only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage, stats

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


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


def robust_scale(frames: np.ndarray) -> np.ndarray:
    arr = frames.astype(np.float32, copy=False)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(hi - lo) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def stable_mask_from_history(frames: np.ndarray, train_fraction: float) -> np.ndarray:
    n_train = max(8, min(frames.shape[0], int(round(frames.shape[0] * train_fraction))))
    hist = robust_scale(frames[:n_train])
    med = np.nanmedian(hist, axis=0)
    bg = ndimage.median_filter(med, size=21, mode="reflect")
    contrast = np.abs(med - bg)
    smooth = ndimage.gaussian_filter(contrast, sigma=1.2)
    thresh = np.nanpercentile(smooth, 72)
    mask = smooth >= thresh
    yy, xx = np.indices(mask.shape)
    cy = (mask.shape[0] - 1) / 2.0
    cx = (mask.shape[1] - 1) / 2.0
    center_prior = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (0.42 * min(mask.shape)) ** 2
    mask = mask | (center_prior & (smooth >= np.nanpercentile(smooth, 62)))
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labels, nlab = ndimage.label(mask)
    if nlab:
        center_label = labels[int(round(cy)), int(round(cx))]
        if center_label > 0:
            mask = labels == center_label
        else:
            sizes = ndimage.sum(mask, labels, np.arange(1, nlab + 1))
            mask = labels == (int(np.argmax(sizes)) + 1)
    if mask.mean() < 0.03 or mask.mean() > 0.90:
        radius = 0.30 * min(mask.shape)
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    return mask.astype(bool)


def flow_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = ndimage.gaussian_filter(a.astype(np.float32), sigma=0.8)
    b = ndimage.gaussian_filter(b.astype(np.float32), sigma=0.8)
    if cv2 is not None:
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)
    return np.zeros_like(a, dtype=np.float32), np.zeros_like(a, dtype=np.float32)


def slope(values: List[float]) -> float:
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(y[ok]) == 0:
        return np.nan
    x = np.linspace(0.0, 1.0, len(y))[ok]
    return float(np.polyfit(x, y[ok], 1)[0])


def front_radius_features(frames: np.ndarray, mask: np.ndarray, train_fraction: float) -> Dict[str, float]:
    n_train = max(8, min(frames.shape[0], int(round(frames.shape[0] * train_fraction))))
    hist = robust_scale(frames[:n_train])
    bg = np.nanmedian(hist, axis=0)
    yy, xx = np.indices(mask.shape)
    if mask.any():
        cy = float(yy[mask].mean())
        cx = float(xx[mask].mean())
    else:
        cy = (mask.shape[0] - 1) / 2.0
        cx = (mask.shape[1] - 1) / 2.0
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    radii: List[float] = []
    areas: List[float] = []
    for frame in robust_scale(frames):
        contrast = np.abs(frame - bg)
        vals = contrast[mask]
        if vals.size < 8:
            radii.append(np.nan)
            areas.append(np.nan)
            continue
        thr = np.nanpercentile(vals, 70)
        active = mask & (contrast >= thr)
        if active.sum() < 4:
            radii.append(np.nan)
            areas.append(float(active.mean()))
            continue
        radii.append(float(np.nanpercentile(rr2[active], 70)))
        areas.append(float(active.mean()))
    half = len(radii) // 2
    return {
        "front_radius2_q70_slope": slope(radii),
        "front_radius2_q70_late_minus_early": float(np.nanmedian(radii[half:]) - np.nanmedian(radii[:half])) if half else np.nan,
        "front_active_area_slope": slope(areas),
        "front_active_area_late_minus_early": float(np.nanmedian(areas[half:]) - np.nanmedian(areas[:half])) if half else np.nan,
    }


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
    between = sum(len(sub) * float((sub.mean() - overall) ** 2) for _, sub in vals.groupby(src))
    return float(between / total)


def oriented_auc_ap(y: np.ndarray, score: np.ndarray) -> Tuple[float, float, str, np.ndarray]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    if len(y) < 8 or len(np.unique(y)) < 2 or np.nanstd(score) == 0:
        return np.nan, np.nan, "NA", score
    ranks = stats.rankdata(score)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    direction = "higher_in_positive"
    oriented_score = score.copy()
    if auc < 0.5:
        auc = 1.0 - auc
        oriented_score = -score
        direction = "lower_in_positive"
    order = np.argsort(-oriented_score)
    yy = y[order]
    precision = np.cumsum(yy) / (np.arange(len(yy)) + 1)
    ap = float((precision * yy).sum() / max(1, n_pos))
    return float(auc), ap, direction, oriented_score


def stratified_permutation_p(y: np.ndarray, score: np.ndarray, source: np.ndarray, observed_auc: float, n_perm: int = 499) -> float:
    if not np.isfinite(observed_auc):
        return np.nan
    rng = np.random.default_rng(20260522)
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    source = np.asarray(source).astype(str)
    ok = np.isfinite(score)
    y, score, source = y[ok], score[ok], source[ok]
    if len(np.unique(y)) < 2:
        return np.nan
    hits = 0
    total = 0
    for _ in range(n_perm):
        yp = y.copy()
        for src in np.unique(source):
            idx = np.flatnonzero(source == src)
            if len(idx) > 1:
                yp[idx] = rng.permutation(yp[idx])
        if len(np.unique(yp)) < 2:
            continue
        auc, _, _, _ = oriented_auc_ap(yp, score)
        if np.isfinite(auc):
            total += 1
            hits += int(auc >= observed_auc)
    return float((hits + 1) / (total + 1)) if total else np.nan


def feature_tests(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        if target not in df.columns:
            continue
        y_all = numeric(df, target).fillna(0).clip(0, 1).astype(int)
        for feat in features:
            for transform in ["raw", "source_residual"]:
                x = numeric(df, feat)
                if transform == "source_residual":
                    x = x - x.groupby(df["source_stem"].astype(str)).transform("mean")
                valid = y_all.isin([0, 1]) & x.notna()
                y = y_all[valid].to_numpy(dtype=int)
                score = x[valid].to_numpy(dtype=float)
                if len(y) < 8 or len(np.unique(y)) < 2 or np.nanstd(score) == 0:
                    continue
                auc, ap, direction, oriented_score = oriented_auc_ap(y, score)
                pos = score[y == 1]
                neg = score[y == 0]
                try:
                    mwu_p = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
                except Exception:
                    mwu_p = np.nan
                try:
                    rho, sp = stats.spearmanr(y, oriented_score)
                    rho, sp = float(rho), float(sp)
                except Exception:
                    rho, sp = np.nan, np.nan
                rows.append({
                    "target": target,
                    "feature": feat,
                    "transform": transform,
                    "n": int(len(y)),
                    "n_positive": int(y.sum()),
                    "direction": direction,
                    "oriented_auc": auc,
                    "average_precision": ap,
                    "mwu_p": mwu_p,
                    "spearman_rho_oriented": rho,
                    "spearman_p": sp,
                    "median_positive": float(np.nanmedian(pos)),
                    "median_negative": float(np.nanmedian(neg)),
                    "median_positive_minus_negative": float(np.nanmedian(pos) - np.nanmedian(neg)),
                    "source_eta2": source_eta2(x[valid], df.loc[valid, "source_stem"]),
                    "source_stratified_permutation_p": stratified_permutation_p(
                        y, score, df.loc[valid, "source_stem"].to_numpy(), auc
                    ),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "oriented_auc", "average_precision"], ascending=[True, False, False])
    return out


def summarize_roi(row: pd.Series, train_fraction: float) -> Dict[str, Any]:
    z = np.load(row["npz_path"], allow_pickle=True)
    frames = z["frames_norm"] if "frames_norm" in z.files else robust_scale(z["frames"])
    frames = robust_scale(frames)
    n = frames.shape[0]
    split = max(8, min(n - 3, int(round(n * train_fraction))))
    mask = stable_mask_from_history(frames, train_fraction)
    context = ~ndimage.binary_dilation(mask, iterations=2)
    if context.mean() < 0.03:
        context = ~mask
    boundary = ndimage.binary_dilation(mask, iterations=2) ^ ndimage.binary_erosion(mask, iterations=2)
    yy, xx = np.indices(mask.shape)
    if mask.any():
        cy = float(yy[mask].mean())
        cx = float(xx[mask].mean())
    else:
        cy = (mask.shape[0] - 1) / 2.0
        cx = (mask.shape[1] - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) + 1e-6
    erx = (xx - cx) / rr
    ery = (yy - cy) / rr
    metrics: Dict[str, List[float]] = {k: [] for k in [
        "particle_flow_mag", "context_flow_mag", "boundary_flow_mag",
        "abs_radial_flow", "radial_flow", "tangential_flow_abs", "curl_abs",
        "divergence", "gradient_aligned_flow", "intensity_delta_abs",
    ]}
    for t in range(split, n - 1):
        a, b = frames[t], frames[t + 1]
        u, v = flow_pair(a, b)
        mag = np.sqrt(u * u + v * v)
        radial = u * erx + v * ery
        tang = -u * ery + v * erx
        curl = np.gradient(v, axis=1) - np.gradient(u, axis=0)
        div = np.gradient(u, axis=1) + np.gradient(v, axis=0)
        gy, gx = np.gradient(a)
        gnorm = np.sqrt(gx * gx + gy * gy) + 1e-6
        galign = (u * gx + v * gy) / gnorm
        delta_abs = np.abs(b - a)
        metrics["particle_flow_mag"].append(float(np.nanmean(mag[mask])))
        metrics["context_flow_mag"].append(float(np.nanmean(mag[context])))
        metrics["boundary_flow_mag"].append(float(np.nanmean(mag[boundary])))
        metrics["abs_radial_flow"].append(float(np.nanmean(np.abs(radial[mask]))))
        metrics["radial_flow"].append(float(np.nanmean(radial[mask])))
        metrics["tangential_flow_abs"].append(float(np.nanmean(np.abs(tang[mask]))))
        metrics["curl_abs"].append(float(np.nanmean(np.abs(curl[mask]))))
        metrics["divergence"].append(float(np.nanmean(div[mask])))
        metrics["gradient_aligned_flow"].append(float(np.nanmean(galign[mask])))
        metrics["intensity_delta_abs"].append(float(np.nanmean(delta_abs[mask])))
    out = row.to_dict()
    out.update({
        "flow_method": "farneback" if cv2 is not None else "none",
        "train_fraction": float(train_fraction),
        "mask_fraction": float(mask.mean()),
        "context_fraction": float(context.mean()),
        "boundary_fraction": float(boundary.mean()),
        "heldout_flow_pairs": int(max(0, n - split - 1)),
        "particle_context_flow_ratio": float(np.nanmean(metrics["particle_flow_mag"]) / (np.nanmean(metrics["context_flow_mag"]) + 1e-9)),
        "boundary_particle_flow_ratio": float(np.nanmean(metrics["boundary_flow_mag"]) / (np.nanmean(metrics["particle_flow_mag"]) + 1e-9)),
    })
    for name, vals in metrics.items():
        arr = np.asarray(vals, dtype=float)
        out[f"{name}_mean"] = float(np.nanmean(arr)) if arr.size else np.nan
        out[f"{name}_q90"] = float(np.nanpercentile(arr, 90)) if arr.size else np.nan
        out[f"{name}_slope"] = slope(vals)
    out.update(front_radius_features(frames, mask, train_fraction))
    out["apparent_transport_instability_score"] = float(
        np.nanmean([
            out.get("particle_flow_mag_q90", np.nan),
            out.get("abs_radial_flow_q90", np.nan),
            out.get("curl_abs_q90", np.nan),
            out.get("intensity_delta_abs_q90", np.nan),
        ])
    )
    return out


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score_cols = [
        "particle_flow_mag_mean", "abs_radial_flow_mean", "curl_abs_mean",
        "apparent_transport_instability_score", "front_radius2_q70_late_minus_early",
        "front_radius2_q70_slope", "intensity_delta_abs_mean",
    ]
    for col in score_cols:
        vals = numeric(out, col)
        resid = vals - vals.groupby(out["source_stem"].astype(str)).transform("mean")
        out[f"{col}_source_residual"] = resid
        if vals.notna().sum() > 1 and vals.nunique(dropna=True) > 1:
            out[f"{col}_rank01"] = vals.rank(pct=True).fillna(0.5)
        else:
            out[f"{col}_rank01"] = 0.5
        if resid.notna().sum() > 1 and resid.nunique(dropna=True) > 1:
            out[f"{col}_source_residual_rank01"] = resid.rank(pct=True).fillna(0.5)
        else:
            out[f"{col}_source_residual_rank01"] = 0.5
    out["expansion_transport_front_score"] = (
        0.20 * out["particle_flow_mag_mean_source_residual_rank01"]
        + 0.20 * out["abs_radial_flow_mean_source_residual_rank01"]
        + 0.15 * out["curl_abs_mean_source_residual_rank01"]
        + 0.15 * out["apparent_transport_instability_score_source_residual_rank01"]
        + 0.15 * out["front_radius2_q70_slope_source_residual_rank01"]
        + 0.15 * out["intensity_delta_abs_mean_source_residual_rank01"]
    )
    out = out.sort_values("expansion_transport_front_score", ascending=False).reset_index(drop=True)
    out["expansion_transport_front_rank"] = np.arange(1, len(out) + 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_expansion_transport_front_audit")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    args = parser.parse_args()

    roi_dir = Path(args.roi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(roi_dir / "selected_roi_sequence_manifest.csv")
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        try:
            rows.append(summarize_roi(row, args.train_fraction))
        except Exception as exc:
            failures.append({"roi_id": row.get("roi_id"), "npz_path": row.get("npz_path"), "error": str(exc)})
    features = add_scores(pd.DataFrame(rows)) if rows else pd.DataFrame()
    feature_cols = [
        c for c in features.columns
        if c.endswith("_mean") or c.endswith("_q90") or c.endswith("_slope")
        or c.endswith("_late_minus_early") or c in [
            "particle_context_flow_ratio", "boundary_particle_flow_ratio",
            "apparent_transport_instability_score", "expansion_transport_front_score",
        ]
    ]
    tests = feature_tests(features, feature_cols) if not features.empty else pd.DataFrame()
    source_summary = pd.DataFrame()
    if not features.empty:
        source_summary = features.groupby("source_stem", as_index=False).agg({
            "roi_id": "count",
            "cycleNo": "nunique",
            "future_any_drop_within_8cycles": "sum",
            "future_any_drop_within_16cycles": "sum",
            "expansion_transport_front_score": ["max", "median"],
            "apparent_transport_instability_score": "median",
            "front_radius2_q70_slope": "median",
        })
        source_summary.columns = [
            "source_stem", "n_roi", "n_cycles", "future8_positive_rows", "future16_positive_rows",
            "max_expansion_transport_front_score", "median_expansion_transport_front_score",
            "median_apparent_transport_instability_score", "median_front_radius2_q70_slope",
        ]
        source_summary = source_summary.sort_values("max_expansion_transport_front_score", ascending=False)
    top40 = features.head(40).copy() if not features.empty else pd.DataFrame()
    failures_df = pd.DataFrame(failures)

    paths = {
        "features": out_dir / "source_balanced_expansion_transport_front_features.csv",
        "feature_tests": out_dir / "source_balanced_expansion_transport_front_tests.csv",
        "source_summary": out_dir / "source_balanced_expansion_transport_front_source_summary.csv",
        "top40": out_dir / "source_balanced_expansion_transport_front_top40.csv",
        "failures": out_dir / "source_balanced_expansion_transport_front_failures.csv",
        "summary": out_dir / "source_balanced_expansion_transport_front_summary.json",
    }
    features.to_csv(paths["features"], index=False)
    tests.to_csv(paths["feature_tests"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)
    top40.to_csv(paths["top40"], index=False)
    failures_df.to_csv(paths["failures"], index=False)

    best_by_target = []
    if not tests.empty:
        best_by_target = tests.sort_values(["target", "oriented_auc", "average_precision"], ascending=[True, False, False]).groupby("target", as_index=False).head(1).to_dict("records")

    summary = clean_json({
        "roi_dir": str(roi_dir),
        "n_input_rows": int(len(manifest)),
        "n_ok": int(len(features)),
        "n_failed": int(len(failures)),
        "n_cycles": int(features["cycleNo"].nunique()) if not features.empty else 0,
        "n_sources": int(features["source_stem"].nunique()) if not features.empty else 0,
        "future8_positive_rows": int(numeric(features, "future_any_drop_within_8cycles").fillna(0).sum()) if not features.empty else 0,
        "future16_positive_rows": int(numeric(features, "future_any_drop_within_16cycles").fillna(0).sum()) if not features.empty else 0,
        "flow_method": "farneback" if cv2 is not None else "none",
        "best_feature_tests_by_target": best_by_target,
        "top_feature_tests": tests.head(90).to_dict("records") if not tests.empty else [],
        "source_summary_top": source_summary.head(14).to_dict("records") if not source_summary.empty else [],
        "top_candidates": top40.head(12).to_dict("records") if not top40.empty else [],
        "guardrail": "Expanded-cohort apparent transport/front descriptors are computed from automatic fixed ROI crops. They test generalization of image-motion and front-radius proxies across broader source-balanced cycles, but they are not manual particle labels, calibrated transport, diffusion coefficients, or degradation causality.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Source-Balanced Expansion Transport/Front Audit",
        "",
        "Apparent optical-flow transport and front-radius proxy audit on the expanded source-balanced ROI sequence cohort.",
        "",
        f"- Rows OK/failed: {summary['n_ok']} / {summary['n_failed']}",
        f"- Cycles/sources: {summary['n_cycles']} / {summary['n_sources']}",
        f"- Future8/future16 positive rows: {summary['future8_positive_rows']} / {summary['future16_positive_rows']}",
        f"- Flow method: {summary['flow_method']}",
        "",
        "## Top Feature Tests",
        "",
    ]
    for row in summary["top_feature_tests"][:10]:
        lines.append(
            f"- {row.get('target')} {row.get('feature')} {row.get('transform')}: "
            f"AUC {row.get('oriented_auc')}, AP {row.get('average_precision')}, "
            f"source-stratified p {row.get('source_stratified_permutation_p')}"
        )
    lines += ["", "## Top Candidates", ""]
    for row in summary["top_candidates"][:8]:
        lines.append(
            f"- {row.get('roi_id')}: score {row.get('expansion_transport_front_score')}, "
            f"future8={row.get('future_any_drop_within_8cycles')}, future16={row.get('future_any_drop_within_16cycles')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
