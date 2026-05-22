#!/usr/bin/env python3
"""Radial front-kymograph audit for source-balanced pre-event ROI crops.

This pass extracts a more explicit phase-front proxy from the source-balanced
pre-event ROI tensors: radial intensity kymographs, gradient-front tracks,
front-track fit quality, monotonicity, and radius-squared mobility proxies. It
is meant as a manual-QC and physics-hypothesis bridge after the matched
counterfactual audit narrowed robust effects to mask/contrast and weak front
motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

PRE_BINS = {"near_pre_event_1_8", "mid_pre_event_9_16", "far_pre_event_17_32"}
CONTROL_BINS = {"post_event_1_16", "no_near_event_control"}
TESTS = [
    ("near_pre_vs_far_pre", "near_pre_event_1_8", "far_pre_event_17_32"),
    ("near_pre_vs_post_control", "near_pre_event_1_8", CONTROL_BINS),
    ("clean_pre_vs_post_control", PRE_BINS, CONTROL_BINS),
]
SUMMARY_FEATURES = [
    "front_radius_slope_px_per_norm_time",
    "front_radius2_slope_px2_per_norm_time",
    "front_radius_slope_r2",
    "front_radius_monotonic_fraction",
    "front_gradient_strength_median",
    "front_gradient_coherence",
    "phase_fraction_slope_per_norm_time",
    "kymograph_temporal_energy",
    "radial_profile_last_minus_first_l1",
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or np.nanstd(x[valid]) <= 1e-12 or np.nanstd(y[valid]) <= 1e-12:
        return np.nan, np.nan, np.nan
    coef = np.polyfit(x[valid], y[valid], 1)
    pred = coef[0] * x[valid] + coef[1]
    ss_res = float(np.sum((y[valid] - pred) ** 2))
    ss_tot = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return float(coef[0]), float(coef[1]), float(r2)


def weighted_centroid(img: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(img, dtype=float)
    base = np.nanpercentile(arr, 35)
    w = np.clip(arr - base, 0, None)
    if not np.isfinite(w).all() or float(w.sum()) <= 1e-12:
        return (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0
    yy, xx = np.indices(arr.shape)
    return float((yy * w).sum() / w.sum()), float((xx * w).sum() / w.sum())


def radial_kymograph(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    early = np.nanmedian(frames[: max(3, min(10, frames.shape[0] // 5))], axis=0)
    cy, cx = weighted_centroid(early)
    yy, xx = np.indices(frames.shape[1:])
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rbin = np.floor(rr).astype(int)
    max_bin = int(min(np.nanmax(rbin), min(frames.shape[1:]) // 2))
    radii = np.arange(max_bin + 1, dtype=float)
    kymo = np.full((frames.shape[0], max_bin + 1), np.nan, dtype=float)
    for r in range(max_bin + 1):
        mask = rbin == r
        if mask.any():
            kymo[:, r] = np.nanmean(frames[:, mask], axis=1)
    col_med = np.nanmedian(kymo, axis=0)
    inds = np.where(np.isnan(kymo))
    if inds[0].size:
        kymo[inds] = np.take(col_med, inds[1])
    return kymo, radii, (cy, cx)


def extract_kymograph_features(frames: np.ndarray) -> Dict[str, Any]:
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3 or frames.shape[0] < 8:
        return {}
    lo, hi = np.nanpercentile(frames, [1, 99])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        frames = np.clip((frames - lo) / (hi - lo), 0, 1)
    kymo, radii, centroid = radial_kymograph(frames)
    if kymo.shape[1] < 4:
        return {}
    t = np.linspace(0.0, 1.0, kymo.shape[0])
    smooth = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3) / 3, mode="same"), 1, kymo)
    grad = np.gradient(smooth, axis=1)
    abs_grad = np.abs(grad)
    ignore = max(2, int(0.08 * len(radii)))
    usable = abs_grad[:, ignore:]
    front_idx_rel = np.nanargmax(usable, axis=1)
    front_idx = front_idx_rel + ignore
    front_radius = radii[front_idx]
    signed_grad = grad[np.arange(len(front_idx)), front_idx]
    strength = abs_grad[np.arange(len(front_idx)), front_idx]
    radius_slope, _, radius_r2 = linear_fit(t, front_radius)
    radius2_slope, _, radius2_r2 = linear_fit(t, front_radius ** 2)
    phase_thr = np.nanpercentile(kymo[: max(3, len(t) // 5)], 70)
    phase_fraction = np.nanmean(kymo >= phase_thr, axis=1)
    phase_slope, _, phase_r2 = linear_fit(t, phase_fraction)
    dr = np.diff(front_radius)
    mono_frac = float(np.mean(dr >= 0)) if dr.size else np.nan
    signed_coherence = float(abs(np.nanmean(np.sign(signed_grad)))) if len(signed_grad) else np.nan
    return {
        "centroid_y": centroid[0],
        "centroid_x": centroid[1],
        "n_kymograph_frames": int(kymo.shape[0]),
        "n_radial_bins": int(kymo.shape[1]),
        "front_radius_first_px": float(front_radius[0]),
        "front_radius_last_px": float(front_radius[-1]),
        "front_radius_delta_px": float(front_radius[-1] - front_radius[0]),
        "front_radius_slope_px_per_norm_time": radius_slope,
        "front_radius_slope_r2": radius_r2,
        "front_radius2_slope_px2_per_norm_time": radius2_slope,
        "front_radius2_slope_r2": radius2_r2,
        "front_radius_monotonic_fraction": mono_frac,
        "front_gradient_strength_median": float(np.nanmedian(strength)),
        "front_gradient_coherence": signed_coherence,
        "front_gradient_signed_median": float(np.nanmedian(signed_grad)),
        "phase_fraction_first": float(phase_fraction[0]),
        "phase_fraction_last": float(phase_fraction[-1]),
        "phase_fraction_delta": float(phase_fraction[-1] - phase_fraction[0]),
        "phase_fraction_slope_per_norm_time": phase_slope,
        "phase_fraction_slope_r2": phase_r2,
        "kymograph_temporal_energy": float(np.nanmean(np.diff(kymo, axis=0) ** 2)),
        "radial_profile_last_minus_first_l1": float(np.nanmean(np.abs(kymo[-1] - kymo[0]))),
        "front_radius_trace": ";".join(f"{v:.3f}" for v in front_radius),
        "phase_fraction_trace": ";".join(f"{v:.5f}" for v in phase_fraction),
    }


def render_kymograph(kymo: np.ndarray, out_path: Path, title: str) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    arr = np.asarray(kymo, dtype=float)
    lo, hi = np.nanpercentile(arr, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        img = np.zeros(arr.shape, dtype=np.uint8)
    else:
        img = (255 * np.clip((arr - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    pil = Image.fromarray(img).convert("RGB").resize((320, 160))
    canvas = Image.new("RGB", (320, 194), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), title[:76], fill=(0, 0, 0))
    canvas.paste(pil, (0, 34))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def load_frames(path: str) -> Optional[np.ndarray]:
    try:
        data = np.load(path)
        arr = data["frames_norm"] if "frames_norm" in data else data["frames"]
        return np.asarray(arr, dtype=float)
    except Exception:
        return None


def compare_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    bins = df["event_relative_bin"].astype(str)
    for test, pos_bins, neg_bins in TESTS:
        pos_set = {pos_bins} if isinstance(pos_bins, str) else set(pos_bins)
        neg_set = {neg_bins} if isinstance(neg_bins, str) else set(neg_bins)
        y = np.where(bins.isin(pos_set), 1, np.where(bins.isin(neg_set), 0, np.nan))
        yy = pd.Series(y, index=df.index)
        for feature in SUMMARY_FEATURES:
            x = numeric(df, feature)
            valid = yy.isin([0, 1]) & x.notna()
            if valid.sum() < 8 or yy[valid].nunique() < 2 or x[valid].nunique() < 2:
                continue
            pos = x[valid & yy.eq(1)]
            neg = x[valid & yy.eq(0)]
            direction = "higher_in_positive" if pos.median() >= neg.median() else "lower_in_positive"
            score = x[valid] if direction == "higher_in_positive" else -x[valid]
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score
                auc = float(roc_auc_score(yy[valid].astype(int), score))
                ap = float(average_precision_score(yy[valid].astype(int), score))
            except Exception:
                auc = ap = np.nan
            try:
                _, p_mwu = mannwhitneyu(pos, neg, alternative="two-sided")
            except ValueError:
                p_mwu = np.nan
            rho, sp = spearmanr(yy[valid].astype(float), score)
            rows.append({
                "test": test,
                "feature": feature,
                "n": int(valid.sum()),
                "n_positive": int(yy[valid].sum()),
                "direction": direction,
                "oriented_auc": auc,
                "average_precision": ap,
                "median_positive": float(pos.median()),
                "median_negative": float(neg.median()),
                "median_positive_minus_negative": float(pos.median() - neg.median()),
                "mwu_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(sp) if np.isfinite(sp) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["test", "oriented_auc", "average_precision"], ascending=[True, False, False])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_radial_kymograph_audit")
    parser.add_argument("--render-top-n", type=int, default=32)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset_dir = out / "kymograph_pngs"
    asset_dir.mkdir(exist_ok=True)

    ranked = read_csv(derived / "source_balanced_pre_event_review_packet" / "source_balanced_pre_event_review_ranked_candidates.csv")
    rows: List[Dict[str, Any]] = []
    assets: List[Dict[str, Any]] = []
    for _, row in ranked.iterrows():
        frames = load_frames(str(row.get("npz_path", "")))
        if frames is None:
            continue
        features = extract_kymograph_features(frames)
        if not features:
            continue
        out_row = row.to_dict()
        out_row.update(features)
        rows.append(out_row)
        rank = int(row.get("pre_event_review_rank", len(rows))) if pd.notna(row.get("pre_event_review_rank")) else len(rows)
        if rank <= args.render_top_n:
            kymo, _, _ = radial_kymograph(np.asarray(frames, dtype=float))
            png = render_kymograph(kymo, asset_dir / f"{rank:02d}_{row['roi_id']}_radial_kymograph.png", f"{rank}. {row['roi_id']} {row['event_relative_bin']}")
            assets.append({
                "pre_event_review_rank": rank,
                "roi_id": row["roi_id"],
                "event_relative_bin": row.get("event_relative_bin"),
                "radial_kymograph_png": png,
            })

    feature_df = pd.DataFrame(rows)
    test_df = compare_groups(feature_df) if not feature_df.empty else pd.DataFrame()
    asset_df = pd.DataFrame(assets)

    feature_path = out / "source_balanced_pre_event_radial_kymograph_features.csv"
    test_path = out / "source_balanced_pre_event_radial_kymograph_tests.csv"
    asset_path = out / "source_balanced_pre_event_radial_kymograph_assets.csv"
    summary_path = out / "source_balanced_pre_event_radial_kymograph_summary.json"
    feature_df.to_csv(feature_path, index=False)
    test_df.to_csv(test_path, index=False)
    asset_df.to_csv(asset_path, index=False)

    near_tests = test_df[test_df["test"].eq("near_pre_vs_far_pre")].head(8).to_dict("records") if not test_df.empty else []
    clean_tests = test_df[test_df["test"].eq("clean_pre_vs_post_control")].head(8).to_dict("records") if not test_df.empty else []
    top_review = feature_df.sort_values("pre_event_review_rank").head(12)[[
        "pre_event_review_rank", "roi_id", "cycleNo", "source_stem", "event_relative_bin",
        "front_radius_slope_px_per_norm_time", "front_radius2_slope_px2_per_norm_time",
        "front_radius_slope_r2", "front_radius_monotonic_fraction",
        "phase_fraction_slope_per_norm_time", "front_gradient_coherence",
    ]].to_dict("records") if not feature_df.empty else []
    summary = {
        "n_roi": int(len(feature_df)),
        "n_cycles": int(feature_df["cycleNo"].nunique()) if not feature_df.empty else 0,
        "n_sources": int(feature_df["source_stem"].nunique()) if not feature_df.empty else 0,
        "n_rendered_kymographs": int(asset_df["radial_kymograph_png"].notna().sum()) if not asset_df.empty and "radial_kymograph_png" in asset_df else 0,
        "event_relative_bin_counts": clean_json(feature_df["event_relative_bin"].value_counts().to_dict()) if not feature_df.empty else {},
        "top_near_vs_far_tests": clean_json(near_tests),
        "top_clean_pre_vs_post_control_tests": clean_json(clean_tests),
        "top_review_candidate_kymograph_features": clean_json(top_review),
        "outputs": {
            "features": str(feature_path),
            "tests": str(test_path),
            "assets": str(asset_path),
            "kymograph_pngs": str(asset_dir),
            "summary": str(summary_path),
        },
        "guardrail": "Radial kymographs are automatic fixed-crop optical front proxies from source-balanced ROI tensors. Centroids, thresholds, and gradient fronts are not manual segmentations. Radius-squared slopes are apparent mobility descriptors only, not calibrated diffusion coefficients or validated phase-boundary tracks.",
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")

    lines = [
        "# Source-Balanced Pre-Event Radial Kymograph Audit",
        "",
        f"- ROI/cycles/sources: {summary['n_roi']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Rendered kymographs: {summary['n_rendered_kymographs']}",
        "",
        "## Top Near-Vs-Far Tests",
    ]
    for row in near_tests[:6]:
        lines.append(
            f"- {row['feature']}: AUC={row['oriented_auc']:.3f}, median diff={row['median_positive_minus_negative']:.4g}, p={row['mwu_p']:.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
