#!/usr/bin/env python3
"""Threshold-robust phase-front and diffusion-proxy analysis for NMC ROI crops.

This script deliberately works on the particle-region ROI tensors, not full
frames. It sweeps bright-phase thresholds, estimates apparent front radius
motion, and bootstraps slope stability so that diffusion-like conclusions are
flagged as robust or threshold-sensitive.
"""

import argparse
import json
import os
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def linear_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or np.nanstd(x[mask]) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
    coef = np.polyfit(x[mask], y[mask], 1)
    pred = coef[0] * x[mask] + coef[1]
    ss_res = float(np.sum((y[mask] - pred) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return {
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def central_mask(height: int, width: int, radius_frac: float = 0.48) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= radius_frac * min(height, width)


def timing_seconds(root: str, source_stem: str, frame_indices: np.ndarray) -> Dict[str, object]:
    h5_path = os.path.join(root, "NMC_degradation_3_160623_Halfthedata", f"{source_stem}.hdf5")
    frame_indices = np.asarray(frame_indices, dtype=int)
    if not os.path.exists(h5_path) or frame_indices.size < 2:
        elapsed = float(frame_indices[-1] - frame_indices[0]) if frame_indices.size >= 2 else np.nan
        return {
            "time_s": np.arange(frame_indices.size, dtype=float),
            "timing_elapsed_s": elapsed,
            "seconds_per_sample": np.nan,
            "timing_source": "frame_index_fallback",
        }
    try:
        with h5py.File(h5_path, "r") as f:
            timing = np.asarray(f["camera_timing"])
    except Exception:
        elapsed = float(frame_indices[-1] - frame_indices[0])
        return {
            "time_s": np.arange(frame_indices.size, dtype=float),
            "timing_elapsed_s": elapsed,
            "seconds_per_sample": np.nan,
            "timing_source": "frame_index_fallback",
        }
    trace = timing if timing.ndim == 1 else timing.reshape(-1, timing.shape[-1])[0]
    valid = frame_indices[(frame_indices >= 0) & (frame_indices < len(trace))]
    if len(valid) != len(frame_indices) or len(valid) < 2:
        elapsed = float(frame_indices[-1] - frame_indices[0])
        return {
            "time_s": np.arange(frame_indices.size, dtype=float),
            "timing_elapsed_s": elapsed,
            "seconds_per_sample": np.nan,
            "timing_source": "frame_index_fallback",
        }
    raw = trace[valid].astype(float)
    time_s = raw - raw[0]
    if np.nanmax(time_s) > 1e6:
        time_s = time_s / 1e9
    return {
        "time_s": time_s,
        "timing_elapsed_s": float(time_s[-1] - time_s[0]),
        "seconds_per_sample": float(np.nanmedian(np.diff(time_s))) if len(time_s) > 2 else np.nan,
        "timing_source": "camera_timing",
    }


def add_roi_id(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["roi_id"] = out.apply(
        lambda r: f"cycle{int(float(r['cycleNo']))}_rank{int(r['front_candidate_rank'])}_obj{int(r['object_candidate_rank'])}",
        axis=1,
    )
    return out


def mask_perimeter(mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    return float((mask & ~ndi.binary_erosion(mask)).sum())


def threshold_trace(frames: np.ndarray, quantile: float) -> Dict[str, np.ndarray]:
    t, h, w = frames.shape
    roi_mask = central_mask(h, w)
    early = frames[: max(4, t // 8)][:, roi_mask]
    threshold = float(np.nanquantile(early, quantile))
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    roi_area = float(roi_mask.sum())

    fraction = []
    radius2 = []
    interface_density = []
    centroid_x = []
    centroid_y = []
    for frame in frames:
        phase = roi_mask & np.isfinite(frame) & (frame >= threshold)
        fraction.append(float(phase.sum() / roi_area))
        radius2.append(float(np.mean(rr2[phase])) if phase.any() else np.nan)
        interface_density.append(mask_perimeter(phase) / roi_area)
        if phase.any():
            centroid_x.append(float(np.mean(xx[phase]) - cx))
            centroid_y.append(float(np.mean(yy[phase]) - cy))
        else:
            centroid_x.append(np.nan)
            centroid_y.append(np.nan)
    return {
        "threshold": threshold,
        "fraction": np.asarray(fraction, dtype=float),
        "radius2_px2": np.asarray(radius2, dtype=float),
        "interface_density": np.asarray(interface_density, dtype=float),
        "centroid_x": np.asarray(centroid_x, dtype=float),
        "centroid_y": np.asarray(centroid_y, dtype=float),
    }


def bootstrap_slope_ci(time_s: np.ndarray, values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> Dict[str, float]:
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(time_s) & np.isfinite(values)
    x = time_s[mask]
    y = values[mask]
    if len(x) < 8:
        return {"slope_bootstrap_p05": np.nan, "slope_bootstrap_p50": np.nan, "slope_bootstrap_p95": np.nan}
    slopes = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), len(x))
        fit = linear_fit(x[idx], y[idx])
        if np.isfinite(fit["slope"]):
            slopes.append(fit["slope"])
    if not slopes:
        return {"slope_bootstrap_p05": np.nan, "slope_bootstrap_p50": np.nan, "slope_bootstrap_p95": np.nan}
    arr = np.asarray(slopes, dtype=float)
    return {
        "slope_bootstrap_p05": float(np.nanpercentile(arr, 5)),
        "slope_bootstrap_p50": float(np.nanpercentile(arr, 50)),
        "slope_bootstrap_p95": float(np.nanpercentile(arr, 95)),
    }


def per_threshold_rows(
    roi_row: pd.Series,
    frames: np.ndarray,
    time_s: np.ndarray,
    quantiles: List[float],
    pixel_size_um: float,
) -> pd.DataFrame:
    rows = []
    for q in quantiles:
        tr = threshold_trace(frames, q)
        frac_fit = linear_fit(time_s, tr["fraction"])
        radius_fit = linear_fit(time_s, tr["radius2_px2"])
        iface_fit = linear_fit(time_s, tr["interface_density"])
        centroid_step = np.sqrt(np.diff(tr["centroid_x"]) ** 2 + np.diff(tr["centroid_y"]) ** 2)
        rows.append({
            "roi_id": roi_row["roi_id"],
            "cycleNo": finite_float(roi_row.get("cycleNo")),
            "cohort_role": roi_row.get("cohort_role", ""),
            "is_event_roi": int(roi_row.get("cohort_role", "") == "event"),
            "event_reference_cycle": finite_float(roi_row.get("event_reference_cycle")),
            "degradation_mode_hypothesis": roi_row.get("degradation_mode_hypothesis", ""),
            "threshold_quantile": q,
            "threshold_value": tr["threshold"],
            "phase_fraction_first": float(tr["fraction"][0]),
            "phase_fraction_last": float(tr["fraction"][-1]),
            "phase_fraction_delta": float(tr["fraction"][-1] - tr["fraction"][0]),
            "phase_fraction_slope_per_s": frac_fit["slope"],
            "phase_fraction_slope_r2": frac_fit["r2"],
            "radius2_delta_px2": float(tr["radius2_px2"][-1] - tr["radius2_px2"][0]),
            "radius2_slope_px2_per_s": radius_fit["slope"],
            "radius2_slope_r2": radius_fit["r2"],
            "apparent_diffusion_proxy_um2_per_s": radius_fit["slope"] * (pixel_size_um ** 2) / 4.0 if np.isfinite(radius_fit["slope"]) else np.nan,
            "interface_density_delta": float(tr["interface_density"][-1] - tr["interface_density"][0]),
            "interface_density_slope_per_s": iface_fit["slope"],
            "interface_density_slope_r2": iface_fit["r2"],
            "centroid_path_px": float(np.nansum(centroid_step)),
            "centroid_net_px": float(np.sqrt((tr["centroid_x"][-1] - tr["centroid_x"][0]) ** 2 + (tr["centroid_y"][-1] - tr["centroid_y"][0]) ** 2)),
        })
    return pd.DataFrame(rows)


def robustness_row(
    roi_row: pd.Series,
    threshold_df: pd.DataFrame,
    default_trace: Dict[str, np.ndarray],
    time_s: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    pixel_size_um: float,
) -> Dict[str, object]:
    slope = pd.to_numeric(threshold_df["phase_fraction_slope_per_s"], errors="coerce").dropna().to_numpy()
    radius_slope = pd.to_numeric(threshold_df["radius2_slope_px2_per_s"], errors="coerce").dropna().to_numpy()
    d_proxy = pd.to_numeric(threshold_df["apparent_diffusion_proxy_um2_per_s"], errors="coerce").dropna().to_numpy()
    frac_ci = bootstrap_slope_ci(time_s, default_trace["fraction"], rng, n_bootstrap)
    radius_ci = bootstrap_slope_ci(time_s, default_trace["radius2_px2"], rng, n_bootstrap)
    default_radius_slope = linear_fit(time_s, default_trace["radius2_px2"])["slope"]
    return {
        "roi_id": roi_row["roi_id"],
        "cycleNo": finite_float(roi_row.get("cycleNo")),
        "cohort_role": roi_row.get("cohort_role", ""),
        "is_event_roi": int(roi_row.get("cohort_role", "") == "event"),
        "event_reference_cycle": finite_float(roi_row.get("event_reference_cycle")),
        "validation_score": finite_float(roi_row.get("validation_score")),
        "front_quality_score": finite_float(roi_row.get("front_quality_score")),
        "degradation_mode_hypothesis": roi_row.get("degradation_mode_hypothesis", ""),
        "timing_elapsed_s": finite_float(roi_row.get("timing_elapsed_s")),
        "thresholds_finite": int(len(slope)),
        "phase_slope_median_per_s": float(np.nanmedian(slope)) if len(slope) else np.nan,
        "phase_slope_iqr_per_s": float(np.nanpercentile(slope, 75) - np.nanpercentile(slope, 25)) if len(slope) else np.nan,
        "phase_slope_positive_fraction": float(np.mean(slope > 0)) if len(slope) else np.nan,
        "phase_slope_negative_fraction": float(np.mean(slope < 0)) if len(slope) else np.nan,
        "phase_slope_abs_median_per_s": float(np.nanmedian(np.abs(slope))) if len(slope) else np.nan,
        "radius2_slope_median_px2_per_s": float(np.nanmedian(radius_slope)) if len(radius_slope) else np.nan,
        "radius2_slope_iqr_px2_per_s": float(np.nanpercentile(radius_slope, 75) - np.nanpercentile(radius_slope, 25)) if len(radius_slope) else np.nan,
        "radius2_slope_positive_fraction": float(np.mean(radius_slope > 0)) if len(radius_slope) else np.nan,
        "radius2_slope_negative_fraction": float(np.mean(radius_slope < 0)) if len(radius_slope) else np.nan,
        "diffusion_proxy_median_um2_per_s": float(np.nanmedian(d_proxy)) if len(d_proxy) else np.nan,
        "diffusion_proxy_iqr_um2_per_s": float(np.nanpercentile(d_proxy, 75) - np.nanpercentile(d_proxy, 25)) if len(d_proxy) else np.nan,
        "default_q70_diffusion_proxy_um2_per_s": default_radius_slope * (pixel_size_um ** 2) / 4.0 if np.isfinite(default_radius_slope) else np.nan,
        "q70_phase_slope_bootstrap_p05": frac_ci["slope_bootstrap_p05"],
        "q70_phase_slope_bootstrap_p50": frac_ci["slope_bootstrap_p50"],
        "q70_phase_slope_bootstrap_p95": frac_ci["slope_bootstrap_p95"],
        "q70_radius2_slope_bootstrap_p05_px2_per_s": radius_ci["slope_bootstrap_p05"],
        "q70_radius2_slope_bootstrap_p50_px2_per_s": radius_ci["slope_bootstrap_p50"],
        "q70_radius2_slope_bootstrap_p95_px2_per_s": radius_ci["slope_bootstrap_p95"],
    }


def feature_tests(df: pd.DataFrame, features: List[str], group_col: str = "") -> pd.DataFrame:
    groups = [(np.nan, df)] if not group_col else list(df.groupby(group_col, dropna=False))
    rows = []
    for group_value, grp in groups:
        event = grp[grp["is_event_roi"] == 1]
        control = grp[grp["is_event_roi"] == 0]
        for feat in features:
            a = pd.to_numeric(event[feat], errors="coerce").dropna()
            b = pd.to_numeric(control[feat], errors="coerce").dropna()
            if len(a) and len(b):
                try:
                    stat, p = mannwhitneyu(a, b, alternative="two-sided")
                except Exception:
                    stat, p = np.nan, np.nan
            else:
                stat, p = np.nan, np.nan
            rows.append({
                "group": group_col or "all",
                "group_value": group_value,
                "feature": feat,
                "event_mean": float(a.mean()) if len(a) else np.nan,
                "control_mean": float(b.mean()) if len(b) else np.nan,
                "event_minus_control": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
                "n_event": int(len(a)),
                "n_control": int(len(b)),
                "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["group", "p_value"], na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_sequences/selected_roi_sequence_manifest.csv")
    parser.add_argument("--cohort-table", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_cohort/multi_cycle_roi_table.csv")
    parser.add_argument("--root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_threshold_robust_fronts")
    parser.add_argument("--threshold-quantiles", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--pixel-size-um", type=float, default=0.096)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=29)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    cohort = add_roi_id(pd.read_csv(args.cohort_table))
    meta_cols = [
        "roi_id",
        "cohort_role",
        "event_reference_cycle",
        "validation_score",
        "front_quality_score",
        "degradation_mode_hypothesis",
    ]
    table = manifest.merge(cohort[[c for c in meta_cols if c in cohort.columns]].drop_duplicates("roi_id"), on="roi_id", how="left")
    quantiles = [float(x) for x in args.threshold_quantiles.split(",") if x.strip()]
    rng = np.random.default_rng(args.random_state)

    threshold_tables = []
    robustness_rows = []
    for _, row in table.iterrows():
        with np.load(row["npz_path"]) as data:
            frames = np.asarray(data["frames_norm"], dtype=np.float32)
            frame_indices = data["frame_indices"] if "frame_indices" in data else np.arange(frames.shape[0])
        timing = timing_seconds(args.root, str(row.get("source_stem", "")), frame_indices)
        row = row.copy()
        row["timing_elapsed_s"] = timing["timing_elapsed_s"]
        row["timing_source"] = timing["timing_source"]
        th = per_threshold_rows(row, frames, timing["time_s"], quantiles, args.pixel_size_um)
        threshold_tables.append(th)
        default_trace = threshold_trace(frames, 0.70)
        robustness_rows.append(robustness_row(row, th, default_trace, timing["time_s"], rng, args.n_bootstrap, args.pixel_size_um))

    threshold_df = pd.concat(threshold_tables, ignore_index=True)
    robust_df = pd.DataFrame(robustness_rows)
    robust_df["threshold_robust_phase_score"] = (
        robust_df["phase_slope_abs_median_per_s"].rank(pct=True)
        - robust_df["phase_slope_iqr_per_s"].rank(pct=True)
        + np.maximum(robust_df["phase_slope_positive_fraction"], robust_df["phase_slope_negative_fraction"]).rank(pct=True)
    )
    robust_df["diffusion_proxy_abs_median_um2_per_s"] = robust_df["diffusion_proxy_median_um2_per_s"].abs()
    robust_df["threshold_robust_diffusion_score"] = (
        robust_df["diffusion_proxy_abs_median_um2_per_s"].rank(pct=True)
        - robust_df["diffusion_proxy_iqr_um2_per_s"].rank(pct=True)
        + np.maximum(robust_df["radius2_slope_positive_fraction"], robust_df["radius2_slope_negative_fraction"]).rank(pct=True)
    )
    robust_df = robust_df.sort_values(["threshold_robust_phase_score", "threshold_robust_diffusion_score"], ascending=False)

    test_features = [
        "phase_slope_median_per_s",
        "phase_slope_abs_median_per_s",
        "phase_slope_positive_fraction",
        "radius2_slope_median_px2_per_s",
        "diffusion_proxy_median_um2_per_s",
        "diffusion_proxy_abs_median_um2_per_s",
        "threshold_robust_phase_score",
        "threshold_robust_diffusion_score",
    ]
    tests_all = feature_tests(robust_df, test_features)
    tests_by_event = feature_tests(robust_df, test_features, "event_reference_cycle")
    group = robust_df.groupby(["cohort_role", "event_reference_cycle", "cycleNo"], dropna=False).agg({
        "roi_id": "count",
        "phase_slope_median_per_s": "mean",
        "phase_slope_abs_median_per_s": "mean",
        "diffusion_proxy_median_um2_per_s": "mean",
        "diffusion_proxy_abs_median_um2_per_s": "mean",
        "threshold_robust_phase_score": "mean",
        "threshold_robust_diffusion_score": "mean",
    }).reset_index().rename(columns={"roi_id": "n_roi"})

    os.makedirs(args.out_dir, exist_ok=True)
    threshold_path = os.path.join(args.out_dir, "threshold_sweep_per_roi.csv")
    robust_path = os.path.join(args.out_dir, "threshold_robust_front_summary.csv")
    tests_path = os.path.join(args.out_dir, "threshold_robust_front_feature_tests.csv")
    by_event_path = os.path.join(args.out_dir, "threshold_robust_front_by_event_tests.csv")
    group_path = os.path.join(args.out_dir, "threshold_robust_front_group_summary.csv")
    threshold_df.to_csv(threshold_path, index=False)
    robust_df.to_csv(robust_path, index=False)
    tests_all.to_csv(tests_path, index=False)
    tests_by_event.to_csv(by_event_path, index=False)
    group.to_csv(group_path, index=False)

    summary = {
        "n_roi": int(len(robust_df)),
        "n_event_roi": int((robust_df["is_event_roi"] == 1).sum()),
        "n_control_roi": int((robust_df["is_event_roi"] == 0).sum()),
        "threshold_quantiles": quantiles,
        "n_bootstrap": int(args.n_bootstrap),
        "pixel_size_um": float(args.pixel_size_um),
        "diffusion_guardrail": (
            "Diffusion values are apparent optical-front proxies computed as slope(radius^2) * pixel_size_um^2 / 4. "
            "They are not validated Li diffusion coefficients and require manual front QC."
        ),
        "top_overall_feature_tests": tests_all.head(12).to_dict(orient="records"),
        "top_by_event_feature_tests": tests_by_event.groupby("group_value", dropna=False).head(5).to_dict(orient="records"),
        "top_threshold_robust_phase_rois": robust_df.head(12).to_dict(orient="records"),
        "group_summary": group.to_dict(orient="records"),
        "outputs": {
            "threshold_sweep": threshold_path,
            "robust_summary": robust_path,
            "feature_tests": tests_path,
            "by_event_tests": by_event_path,
            "group_summary": group_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "threshold_robust_front_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Multi-Cycle Threshold-Robust Fronts\n\n")
        f.write("Threshold-sweep and bootstrap stability analysis for phase-front mobility and apparent diffusion proxies from particle ROI tensors.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
