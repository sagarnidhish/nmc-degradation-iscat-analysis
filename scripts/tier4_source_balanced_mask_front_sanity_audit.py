#!/usr/bin/env python3
"""Mask/front sanity audit for source-balanced particle ROI crops.

This audit stays intentionally conservative: it estimates crop-local particle
masks and radial front proxies from the exported ROI tensors, then asks whether
those automatic summaries correlate with future degradation labels. The output
is a compact QC/physics proxy packet; it is not a calibrated diffusion or
manual particle-validation product.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress, mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
PIXEL_SIZE_UM = 0.192


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


def feature_tests(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
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
                signed_x = xx if direction == "higher_in_positive" else -xx
                rho, sp = spearmanr(yy, signed_x)
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


def robust_z(image: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(image))
    mad = float(np.nanmedian(np.abs(image - med)))
    scale = 1.4826 * mad if mad > 1e-8 else float(np.nanstd(image))
    if not np.isfinite(scale) or scale <= 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - med) / scale).astype(np.float32)


def slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.linspace(0.0, 1.0, len(y))
    valid = np.isfinite(y)
    if valid.sum() < 3 or np.nanstd(y[valid]) <= 1e-12:
        return np.nan
    return float(linregress(x[valid], y[valid]).slope)


def radial_profile(image: np.ndarray, radius: np.ndarray, max_r: int) -> np.ndarray:
    bins = np.arange(max_r + 1)
    labels = np.clip(radius.astype(int), 0, max_r)
    sums = np.bincount(labels.ravel(), weights=image.ravel(), minlength=max_r + 1)
    counts = np.bincount(labels.ravel(), minlength=max_r + 1)
    return sums / np.maximum(counts, 1)


def center_of_mass(mask: np.ndarray, weight: np.ndarray, cy: float, cx: float) -> Tuple[float, float]:
    values = np.where(mask, np.maximum(weight, 0), 0.0)
    total = float(values.sum())
    if total <= 1e-8:
        return cy, cx
    yy, xx = np.indices(mask.shape)
    return float((yy * values).sum() / total), float((xx * values).sum() / total)


def mask_for_frame(frame: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    z = robust_z(frame)
    candidate = (z >= 0.5) & base_mask
    frac = float(candidate.mean())
    if frac < 0.01 or frac > 0.65:
        q = float(np.nanquantile(frame[base_mask], 0.65)) if base_mask.any() else float(np.nanquantile(frame, 0.65))
        candidate = (frame >= q) & base_mask
    return candidate


def front_radius_from_profile(profile: np.ndarray, quantile: float) -> float:
    if len(profile) < 4 or not np.isfinite(profile).any():
        return np.nan
    lo, hi = np.nanpercentile(profile, [10, 90])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    threshold = lo + quantile * (hi - lo)
    above = np.where(profile >= threshold)[0]
    if len(above) == 0:
        return np.nan
    return float(above[-1])


def compute_roi_features(row: pd.Series) -> Dict[str, Any]:
    data = np.load(row["npz_path"])
    frames = data["frames_norm"].astype(np.float32)
    n, height, width = frames.shape
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = int(min(cy, cx))
    central_mask = radius <= max_r

    mean_img = np.nanmean(frames, axis=0)
    mean_z = robust_z(mean_img)
    base_mask = (mean_z >= 0.5) & central_mask
    base_frac = float(base_mask.mean())
    if base_frac < 0.01 or base_frac > 0.70:
        q = float(np.nanquantile(mean_img[central_mask], 0.65))
        base_mask = (mean_img >= q) & central_mask
        base_frac = float(base_mask.mean())

    areas: List[float] = []
    centroids_y: List[float] = []
    centroids_x: List[float] = []
    radii_q60: List[float] = []
    radii_q70: List[float] = []
    radii_q80: List[float] = []
    radial_grad_peak: List[float] = []
    masked_means: List[float] = []
    bg_means: List[float] = []

    for frame in frames:
        mask = mask_for_frame(frame, base_mask)
        areas.append(float(mask.mean()))
        my, mx = center_of_mass(mask, frame, cy, cx)
        centroids_y.append(my)
        centroids_x.append(mx)
        masked_means.append(float(np.nanmean(frame[mask])) if mask.any() else np.nan)
        bg = central_mask & ~mask
        bg_means.append(float(np.nanmean(frame[bg])) if bg.any() else np.nan)
        profile = radial_profile(frame, radius, max_r)
        radii_q60.append(front_radius_from_profile(profile, 0.60))
        radii_q70.append(front_radius_from_profile(profile, 0.70))
        radii_q80.append(front_radius_from_profile(profile, 0.80))
        grad = np.abs(np.gradient(profile))
        radial_grad_peak.append(float(np.nanargmax(grad[2:]) + 2) if len(grad) > 4 and np.isfinite(grad).any() else np.nan)

    areas_a = np.asarray(areas, dtype=float)
    cy_a = np.asarray(centroids_y, dtype=float)
    cx_a = np.asarray(centroids_x, dtype=float)
    r60 = np.asarray(radii_q60, dtype=float)
    r70 = np.asarray(radii_q70, dtype=float)
    r80 = np.asarray(radii_q80, dtype=float)
    rgp = np.asarray(radial_grad_peak, dtype=float)
    mm = np.asarray(masked_means, dtype=float)
    bm = np.asarray(bg_means, dtype=float)
    centroid_step = np.sqrt(np.diff(cy_a) ** 2 + np.diff(cx_a) ** 2) if len(cy_a) > 1 else np.array([np.nan])

    out = row.to_dict()
    for key in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "any_abrupt_drop"]:
        if key in out:
            out[key] = float(out[key])
    r70_slope = slope(r70)
    out.update({
        "mask_base_area_fraction": base_frac,
        "mask_area_fraction_median": float(np.nanmedian(areas_a)),
        "mask_area_fraction_iqr": float(np.nanpercentile(areas_a, 75) - np.nanpercentile(areas_a, 25)),
        "mask_area_fraction_slope": slope(areas_a),
        "mask_centroid_path_px": float(np.nansum(centroid_step)),
        "mask_centroid_max_step_px": float(np.nanmax(centroid_step)),
        "mask_centroid_drift_px": float(np.sqrt((cy_a[-1] - cy_a[0]) ** 2 + (cx_a[-1] - cx_a[0]) ** 2)) if len(cy_a) else np.nan,
        "masked_minus_background_mean_median": float(np.nanmedian(mm - bm)),
        "masked_minus_background_mean_slope": slope(mm - bm),
        "front_radius_q60_median_px": float(np.nanmedian(r60)),
        "front_radius_q60_delta_px": float(r60[-1] - r60[0]) if len(r60) else np.nan,
        "front_radius_q60_slope_px_per_norm_time": slope(r60),
        "front_radius_q70_median_px": float(np.nanmedian(r70)),
        "front_radius_q70_delta_px": float(r70[-1] - r70[0]) if len(r70) else np.nan,
        "front_radius_q70_slope_px_per_norm_time": r70_slope,
        "front_radius_q80_median_px": float(np.nanmedian(r80)),
        "front_radius_q80_delta_px": float(r80[-1] - r80[0]) if len(r80) else np.nan,
        "front_radius_q80_slope_px_per_norm_time": slope(r80),
        "front_radius_q70_positive_step_fraction": float(np.nanmean(np.diff(r70) > 0)) if len(r70) > 1 else np.nan,
        "front_gradient_peak_radius_median_px": float(np.nanmedian(rgp)),
        "front_gradient_peak_radius_slope_px_per_norm_time": slope(rgp),
        "apparent_diffusion_q70_px2_per_norm_time": slope(r70 ** 2),
        "apparent_diffusion_q70_um2_per_norm_time": slope((r70 * PIXEL_SIZE_UM) ** 2),
        "pixel_size_um_assumed": PIXEL_SIZE_UM,
        "n_loaded_frames": int(n),
        "frame_height": int(height),
        "frame_width": int(width),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_mask_front_sanity_audit")
    args = parser.parse_args()

    roi_dir = Path(args.roi_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(roi_dir / "selected_roi_sequence_manifest.csv")
    features = pd.DataFrame([compute_roi_features(row) for _, row in manifest.iterrows()])

    audit_features = [
        "mask_base_area_fraction", "mask_area_fraction_median", "mask_area_fraction_iqr", "mask_area_fraction_slope",
        "mask_centroid_path_px", "mask_centroid_max_step_px", "mask_centroid_drift_px",
        "masked_minus_background_mean_median", "masked_minus_background_mean_slope",
        "front_radius_q60_median_px", "front_radius_q60_delta_px", "front_radius_q60_slope_px_per_norm_time",
        "front_radius_q70_median_px", "front_radius_q70_delta_px", "front_radius_q70_slope_px_per_norm_time",
        "front_radius_q80_median_px", "front_radius_q80_delta_px", "front_radius_q80_slope_px_per_norm_time",
        "front_radius_q70_positive_step_fraction",
        "front_gradient_peak_radius_median_px", "front_gradient_peak_radius_slope_px_per_norm_time",
        "apparent_diffusion_q70_px2_per_norm_time", "apparent_diffusion_q70_um2_per_norm_time",
        "roi_norm_mean_delta_last_minus_first", "object_area_ds_px", "object_mean_residual", "object_mean_abs_z",
    ]
    tests = feature_tests(features, audit_features)

    cycle_aggs = {
        "roi_id": "count",
        "future_any_drop_within_8cycles": "max",
        "future_any_drop_within_16cycles": "max",
    }
    for feat in audit_features:
        if feat in features.columns:
            cycle_aggs[feat] = "mean"
    cycle = features.groupby(["cycleNo", "source_stem"], as_index=False).agg(cycle_aggs).rename(columns={"roi_id": "n_roi"})
    cycle_tests = feature_tests(cycle, [f for f in audit_features if f in cycle.columns])
    source = features.groupby("source_stem", as_index=False).agg({
        "roi_id": "count",
        "cycleNo": "nunique",
        "future_any_drop_within_8cycles": "sum",
        "future_any_drop_within_16cycles": "sum",
        "mask_area_fraction_median": "mean",
        "mask_centroid_path_px": "mean",
        "front_radius_q70_slope_px_per_norm_time": "mean",
        "apparent_diffusion_q70_um2_per_norm_time": "mean",
        "roi_norm_mean_delta_last_minus_first": "mean",
    }).rename(columns={"roi_id": "n_roi", "cycleNo": "n_cycles"})

    paths = {
        "features": out / "source_balanced_mask_front_features.csv",
        "feature_tests": out / "source_balanced_mask_front_feature_tests.csv",
        "cycle_summary": out / "source_balanced_mask_front_cycle_summary.csv",
        "cycle_tests": out / "source_balanced_mask_front_cycle_tests.csv",
        "source_summary": out / "source_balanced_mask_front_source_summary.csv",
        "summary": out / "source_balanced_mask_front_summary.json",
    }
    features.to_csv(paths["features"], index=False)
    tests.to_csv(paths["feature_tests"], index=False)
    cycle.to_csv(paths["cycle_summary"], index=False)
    cycle_tests.to_csv(paths["cycle_tests"], index=False)
    source.to_csv(paths["source_summary"], index=False)

    top_roi = tests.head(20).to_dict(orient="records") if not tests.empty else []
    top_cycle = cycle_tests.head(16).to_dict(orient="records") if not cycle_tests.empty else []
    summary = {
        "n_roi_sequences": int(len(features)),
        "n_cycles": int(features["cycleNo"].nunique()),
        "n_sources": int(features["source_stem"].nunique()),
        "future8_positive_sequences": int(numeric(features, "future_any_drop_within_8cycles").sum()),
        "future16_positive_sequences": int(numeric(features, "future_any_drop_within_16cycles").sum()),
        "pixel_size_um_assumed": PIXEL_SIZE_UM,
        "feature_columns": audit_features,
        "top_roi_feature_tests": top_roi,
        "top_cycle_feature_tests": top_cycle,
        "source_summary": source.to_dict(orient="records"),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Automatic crop-local masks/front radii are source-balanced sanity proxies from resized ROI tensors; they are not manual particle masks, not calibrated fronts, and apparent diffusion uses an approximate 0.192 um/output-pixel scale.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")
    readme = [
        "# Source-Balanced Mask/Front Sanity Audit",
        "",
        f"ROI sequences: {summary['n_roi_sequences']} across {summary['n_cycles']} cycles and {summary['n_sources']} sources.",
        f"Future8/future16 positive ROI sequences: {summary['future8_positive_sequences']} / {summary['future16_positive_sequences']}.",
        "",
        "This packet estimates automatic crop-local masks, centroid stability, radial front radii at q60/q70/q80, and an approximate q70 radius-squared diffusion proxy from source-balanced ROI tensors.",
        "",
        "## Top ROI Feature Tests",
    ]
    for row in top_roi[:10]:
        readme.append(
            f"- {row.get('target')} {row.get('feature')}: AUC={row.get('oriented_auc'):.3f}, "
            f"AP={row.get('average_precision'):.3f}, direction={row.get('direction')}, eta2={row.get('source_eta2'):.3f}"
        )
    readme.extend([
        "",
        "## Guardrail",
        summary["guardrail"],
        "",
    ])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()
