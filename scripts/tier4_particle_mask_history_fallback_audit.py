#!/usr/bin/env python3
"""Audit history-based particle masks under blur/drift-correction artifacts.

The project uses cropped particle-region sequences. This audit checks whether a
simple history prior would be needed when framewise contrast masks become blurry
or unstable, and whether those mask-instability metrics confound the
near-pre-event ranking scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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
    if area >= len(flat):
        return np.ones_like(frame, dtype=bool)
    idx = np.argpartition(flat, -area)[-area:]
    mask = np.zeros(len(flat), dtype=bool)
    mask[idx] = True
    return mask.reshape(frame.shape)


def centroid(mask: np.ndarray) -> tuple[float, float]:
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
    if mask.sum() == 0:
        return float("nan")
    return float(np.nanmean(mag[mask]))


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_score = score[order]
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    precision = np.cumsum(ys) / (np.arange(len(ys)) + 1.0)
    return float((precision * ys).sum() / ys.sum())


def source_residual(df: pd.DataFrame, feature: str) -> pd.Series:
    values = pd.to_numeric(df[feature], errors="coerce")
    return values - values.groupby(df["source_stem"]).transform("median")


def permutation_p_by_source(df: pd.DataFrame, label: str, score: str, n_perm: int = 2000) -> float:
    rng = np.random.default_rng(20260522)
    y = df[label].to_numpy(dtype=int)
    s = pd.to_numeric(df[score], errors="coerce").to_numpy(dtype=float)
    obs = abs(auc_score(y, s) - 0.5)
    count = 1
    sources = df["source_stem"].to_numpy()
    for _ in range(n_perm):
        yp = y.copy()
        for src in np.unique(sources):
            idx = np.where(sources == src)[0]
            yp[idx] = rng.permutation(yp[idx])
        if abs(auc_score(yp, s) - 0.5) >= obs - 1e-12:
            count += 1
    return float(count / (n_perm + 1))


def analyze_row(row: pd.Series) -> Dict[str, Any]:
    z = np.load(row["npz_path"])
    frames = z["frames_norm"].astype(np.float32)
    n, h, w = frames.shape
    # Manifest object area is measured at a downsampled scale close to the 96px output.
    area = int(max(16, row.get("object_area_ds_px", 64)))
    masks = np.stack([topk_mask(frames[i], area) for i in range(n)])
    early_n = max(8, min(24, n // 4))
    history_vote = masks[:early_n].mean(axis=0)
    history_mask = history_vote >= 0.5
    if history_mask.sum() < 8:
        history_mask = history_vote >= np.quantile(history_vote, 0.98)
    hist_cy, hist_cx = centroid(history_mask)
    ious = np.array([iou(m, history_mask) for m in masks], dtype=float)
    cents = np.array([centroid(m) for m in masks], dtype=float)
    dist = np.sqrt((cents[:, 0] - hist_cy) ** 2 + (cents[:, 1] - hist_cx) ** 2)
    sharp = np.array([sharpness(frames[i], masks[i]) for i in range(n)], dtype=float)
    early_sharp = np.nanmedian(sharp[:early_n])
    blur_ratio = sharp / early_sharp if np.isfinite(early_sharp) and early_sharp > 0 else np.full(n, np.nan)
    fallback = (ious < 0.20) | (dist > 8.0) | (blur_ratio < 0.65)
    return {
        "roi_id": row["roi_id"],
        "cycleNo": row.get("cycleNo"),
        "source_stem": row.get("source_stem"),
        "event_relative_bin": row.get("event_relative_bin"),
        "near_pre_flag": int(row.get("near_pre_flag", 0)),
        "future_any_drop_within_8cycles": row.get("future_any_drop_within_8cycles"),
        "future_any_drop_within_16cycles": row.get("future_any_drop_within_16cycles"),
        "n_frames": int(n),
        "history_mask_area_px": int(history_mask.sum()),
        "manifest_object_area_px": float(row.get("object_area_ds_px", np.nan)),
        "median_history_iou": float(np.nanmedian(ious)),
        "q10_history_iou": float(np.nanquantile(ious, 0.10)),
        "min_history_iou": float(np.nanmin(ious)),
        "centroid_jitter_median_px": float(np.nanmedian(dist)),
        "centroid_jitter_q90_px": float(np.nanquantile(dist, 0.90)),
        "blur_ratio_median": float(np.nanmedian(blur_ratio)),
        "blur_ratio_q10": float(np.nanquantile(blur_ratio, 0.10)),
        "fallback_frame_fraction": float(np.nanmean(fallback)),
        "fallback_frame_count": int(np.nansum(fallback)),
        "history_area_to_manifest_area": float(history_mask.sum() / max(float(row.get("object_area_ds_px", np.nan)), 1.0)),
        "transport_mechanism_score": row.get("transport_mechanism_score"),
        "qc_review_score": row.get("qc_review_score"),
        "front_kinetic_score": row.get("front_kinetic_score"),
    }


def event_tests(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    labels = {
        "near_vs_any_non_near": df["near_pre_flag"].astype(int),
        "future8": pd.to_numeric(df["future_any_drop_within_8cycles"], errors="coerce").fillna(0).astype(int),
        "future16": pd.to_numeric(df["future_any_drop_within_16cycles"], errors="coerce").fillna(0).astype(int),
    }
    for target, y in labels.items():
        for feature in features:
            if feature not in df:
                continue
            s = pd.to_numeric(df[feature], errors="coerce")
            if y.nunique() < 2:
                continue
            auc = auc_score(y.to_numpy(), s.to_numpy())
            rows.append({
                "target": target,
                "feature": feature,
                "n_rows": int(s.notna().sum()),
                "n_pos": int(y.sum()),
                "n_neg": int(len(y) - y.sum()),
                "n_sources": int(df["source_stem"].nunique()),
                "auc": auc,
                "average_precision": average_precision(y.to_numpy(), s.to_numpy()),
                "median_pos": float(s[y.eq(1)].median()),
                "median_neg": float(s[y.eq(0)].median()),
                "source_stratified_permutation_p": permutation_p_by_source(df.assign(_y=y), "_y", feature),
            })
    return pd.DataFrame(rows).sort_values(["target", "auc"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/particle_mask_history_fallback_audit")
    args = parser.parse_args()
    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dossier = pd.read_csv(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_dossier.csv")
    manifest_path = derived / "source_balanced_pre_event_roi_sequences" / "selected_roi_sequence_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        add_cols = [c for c in ["roi_id", "object_area_ds_px", "crop_size_full", "output_size"] if c in manifest.columns]
        if "object_area_ds_px" not in dossier.columns and "roi_id" in add_cols:
            dossier = dossier.merge(manifest[add_cols].drop_duplicates("roi_id"), on="roi_id", how="left")
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for _, row in dossier.iterrows():
        try:
            rows.append(analyze_row(row))
        except Exception as exc:
            failures.append({"roi_id": row.get("roi_id"), "npz_path": row.get("npz_path"), "error": repr(exc)})
    metrics = pd.DataFrame(rows)
    for feature in ["fallback_frame_fraction", "centroid_jitter_q90_px", "blur_ratio_q10", "median_history_iou"]:
        metrics[f"{feature}_source_residual"] = source_residual(metrics, feature)

    test_features = [
        "fallback_frame_fraction",
        "centroid_jitter_q90_px",
        "blur_ratio_q10",
        "median_history_iou",
        "fallback_frame_fraction_source_residual",
        "centroid_jitter_q90_px_source_residual",
        "blur_ratio_q10_source_residual",
        "median_history_iou_source_residual",
        "transport_mechanism_score",
        "qc_review_score",
    ]
    tests = event_tests(metrics, test_features)
    source_summary = metrics.groupby("source_stem").agg(
        n_rows=("roi_id", "count"),
        median_fallback_fraction=("fallback_frame_fraction", "median"),
        median_iou=("median_history_iou", "median"),
        median_centroid_jitter_q90_px=("centroid_jitter_q90_px", "median"),
        median_blur_ratio_q10=("blur_ratio_q10", "median"),
        near_pre_fraction=("near_pre_flag", "mean"),
    ).reset_index().sort_values("median_fallback_fraction", ascending=False)
    high_fallback = metrics.sort_values(["fallback_frame_fraction", "centroid_jitter_q90_px"], ascending=[False, False]).head(30)

    paths = {
        "metrics": out / "particle_mask_history_fallback_metrics.csv",
        "event_tests": out / "particle_mask_history_fallback_event_tests.csv",
        "source_summary": out / "particle_mask_history_fallback_source_summary.csv",
        "high_fallback": out / "particle_mask_history_fallback_high_risk_rois.csv",
        "failures": out / "particle_mask_history_fallback_failures.csv",
        "summary": out / "particle_mask_history_fallback_summary.json",
        "readme": out / "README.md",
    }
    metrics.to_csv(paths["metrics"], index=False)
    tests.to_csv(paths["event_tests"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)
    high_fallback.to_csv(paths["high_fallback"], index=False)
    pd.DataFrame(failures).to_csv(paths["failures"], index=False)

    fallback_near = metrics.loc[metrics["near_pre_flag"].eq(1), "fallback_frame_fraction"]
    fallback_non = metrics.loc[metrics["near_pre_flag"].eq(0), "fallback_frame_fraction"]
    summary = clean_json({
        "overall_status": "particle_mask_history_fallback_audit_ready",
        "n_input_rows": int(len(dossier)),
        "n_ok": int(len(metrics)),
        "n_failures": int(len(failures)),
        "n_sources": int(metrics["source_stem"].nunique()) if not metrics.empty else 0,
        "median_fallback_frame_fraction": float(metrics["fallback_frame_fraction"].median()) if not metrics.empty else None,
        "q90_fallback_frame_fraction": float(metrics["fallback_frame_fraction"].quantile(0.90)) if not metrics.empty else None,
        "median_history_iou": float(metrics["median_history_iou"].median()) if not metrics.empty else None,
        "median_centroid_jitter_q90_px": float(metrics["centroid_jitter_q90_px"].median()) if not metrics.empty else None,
        "median_blur_ratio_q10": float(metrics["blur_ratio_q10"].median()) if not metrics.empty else None,
        "near_vs_non_median_fallback_diff": float(fallback_near.median() - fallback_non.median()) if len(fallback_near) and len(fallback_non) else None,
        "top_event_tests": tests.head(20).to_dict("records"),
        "source_summary_top": source_summary.head(12).to_dict("records"),
        "high_fallback_rois": high_fallback.head(12).to_dict("records"),
        "guardrail": "History-based masks are an automatic robustness audit for cropped particle ROIs. Fallback flags indicate possible blur/drift mask instability; they are not manual segmentation labels or physical phase-boundary measurements.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = [
        "# Particle Mask History/Fallback Audit",
        "",
        "Audits whether framewise particle masks are stable or would need history-based fallback under blur/drift artifacts.",
        "",
        f"- Input rows: {summary['n_input_rows']}",
        f"- Processed rows: {summary['n_ok']}",
        f"- Median fallback frame fraction: {summary['median_fallback_frame_fraction']:.3f}",
        f"- Median history IoU: {summary['median_history_iou']:.3f}",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    paths["readme"].write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
