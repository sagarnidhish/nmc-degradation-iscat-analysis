#!/usr/bin/env python3
"""
Quantify calibration bounds for apparent diffusion/front-motion proxies.

This audit maps each balanced future-drop ROI back to its source HDF5 camera
timing metadata, recomputes radius^2 slopes from ROI frame spans, and evaluates
how apparent diffusion-like values change under plausible pixel-size and
timebase assumptions.

The output is intentionally framed as an apparent optical-front calibration
guardrail. It does not promote these values to material diffusion coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


PIXEL_SIZES_UM = [0.08, 0.096, 0.12]
DEFAULT_PIXEL_SIZE_UM = 0.096


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_movie_frames(value: object) -> float:
    if not isinstance(value, str) or not value.startswith("("):
        return np.nan
    inner = value.strip().strip("()")
    first = inner.split(",", 1)[0].strip()
    try:
        return float(first)
    except ValueError:
        return np.nan


def match_h5_metadata(roi: pd.DataFrame, h5: pd.DataFrame) -> pd.DataFrame:
    h5 = h5.copy()
    h5["source_stem"] = h5["relative_path"].astype(str).str.rsplit("/", n=1).str[-1].str.replace(".hdf5", "", regex=False)
    keep_cols = [
        "source_stem",
        "relative_path",
        "sampled_timing_dt_median_s",
        "sampled_timing_dt_min_s",
        "sampled_timing_dt_max_s",
        "sampled_timing_fps_median",
        "sampled_timing_duration_s",
        "movie_shape",
        "camera_timing_shape",
        "camera_timing_unit_inference",
    ]
    h5 = h5[keep_cols].drop_duplicates("source_stem")
    merged = roi.merge(h5, on="source_stem", how="left")
    return merged


def front_feature_tests(df: pd.DataFrame, value_cols: list[str], target: str) -> pd.DataFrame:
    rows = []
    if target not in df:
        return pd.DataFrame(rows)
    y = to_float_series(df[target])
    for col in value_cols:
        if col not in df:
            continue
        x = to_float_series(df[col])
        valid = y.notna() & x.notna()
        yy = y[valid].astype(int)
        xx = x[valid].astype(float)
        if yy.nunique() != 2:
            continue
        pos = xx[yy == 1]
        neg = xx[yy == 0]
        if len(pos) < 3 or len(neg) < 3:
            continue
        try:
            p = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
        except ValueError:
            p = np.nan
        rows.append(
            {
                "target": target,
                "feature": col,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                "median_positive": float(pos.median()),
                "median_negative": float(neg.median()),
                "median_positive_minus_negative": float(pos.median() - neg.median()),
                "positive_fraction_positive": float((pos > 0).mean()),
                "positive_fraction_negative": float((neg > 0).mean()),
                "mannwhitney_p": p,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "mannwhitney_p", "feature"], ascending=[True, True, True])
    return out


def correlations(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for x_col, y_col in pairs:
        if x_col not in df or y_col not in df:
            continue
        x = to_float_series(df[x_col])
        y = to_float_series(df[y_col])
        valid = x.notna() & y.notna()
        if valid.sum() < 10 or x[valid].nunique() < 3 or y[valid].nunique() < 3:
            continue
        rho, p = spearmanr(x[valid], y[valid])
        rows.append(
            {
                "x": x_col,
                "y": y_col,
                "n": int(valid.sum()),
                "spearman_rho": float(rho),
                "abs_rho": float(abs(rho)),
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_rho", ascending=False)
    return out


def records(df: pd.DataFrame, n: int) -> list[dict]:
    if df.empty:
        return []
    return json.loads(df.head(n).to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/apparent_diffusion_calibration_bounds")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = ensure_dir(Path(args.out_dir))

    roi = pd.read_csv(derived / "balanced_future_roi_physics_audit" / "balanced_future_roi_physics_joined.csv")
    sweep = pd.read_csv(derived / "balanced_future_threshold_robust_fronts" / "threshold_sweep_per_roi.csv")
    h5 = pd.read_csv(derived / "calibration_metadata_audit" / "h5_calibration_metadata.csv")

    meta_cols = [
        "roi_id",
        "cycleNo",
        "source_stem",
        "cohort_role",
        "is_event_roi",
        "event_reference_cycle",
        "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles",
        "any_abrupt_drop",
        "first_frame_index",
        "last_frame_index",
        "timing_elapsed_s",
        "transferred_masked_residual_signature",
        "validation_score_recon",
    ]
    meta = roi[[c for c in meta_cols if c in roi.columns]].drop_duplicates("roi_id")
    joined = sweep.merge(meta, on=["roi_id", "cycleNo", "cohort_role", "is_event_roi", "event_reference_cycle"], how="left")
    joined = match_h5_metadata(joined, h5)

    joined["movie_n_frames"] = joined["movie_shape"].apply(parse_movie_frames)
    joined["frame_span"] = to_float_series(joined["last_frame_index"]) - to_float_series(joined["first_frame_index"])
    joined["frame_span_in_movie"] = joined["last_frame_index"].le(joined["movie_n_frames"] - 1)
    joined["h5_elapsed_median_s"] = joined["frame_span"] * to_float_series(joined["sampled_timing_dt_median_s"])
    joined["h5_elapsed_min_s"] = joined["frame_span"] * to_float_series(joined["sampled_timing_dt_min_s"])
    joined["h5_elapsed_max_s"] = joined["frame_span"] * to_float_series(joined["sampled_timing_dt_max_s"])
    joined["roi_elapsed_to_h5_median_ratio"] = to_float_series(joined["timing_elapsed_s"]) / joined["h5_elapsed_median_s"]
    joined["h5_dt_max_to_median_ratio"] = to_float_series(joined["sampled_timing_dt_max_s"]) / to_float_series(joined["sampled_timing_dt_median_s"])
    joined["h5_dt_min_to_median_ratio"] = to_float_series(joined["sampled_timing_dt_min_s"]) / to_float_series(joined["sampled_timing_dt_median_s"])

    joined["radius2_slope_h5_median_px2_per_s"] = to_float_series(joined["radius2_delta_px2"]) / joined["h5_elapsed_median_s"]
    joined["radius2_slope_h5_min_dt_px2_per_s"] = to_float_series(joined["radius2_delta_px2"]) / joined["h5_elapsed_min_s"]
    joined["radius2_slope_h5_max_dt_px2_per_s"] = to_float_series(joined["radius2_delta_px2"]) / joined["h5_elapsed_max_s"]
    for pixel_size in PIXEL_SIZES_UM:
        suffix = str(pixel_size).replace(".", "p")
        joined[f"apparent_D_h5median_px{suffix}_um2_per_s"] = joined["radius2_slope_h5_median_px2_per_s"] * (pixel_size**2) / 4.0
    joined["apparent_D_reported_to_h5median_ratio"] = (
        to_float_series(joined["apparent_diffusion_proxy_um2_per_s"]) / joined["apparent_D_h5median_px0p096_um2_per_s"]
    )
    joined["apparent_D_h5median_abs_um2_per_s"] = joined["apparent_D_h5median_px0p096_um2_per_s"].abs()

    # Conservative timing envelope using min/max sampled camera intervals and default 96 nm/px.
    joined["apparent_D_h5_fast_dt_px0p096_um2_per_s"] = joined["radius2_slope_h5_min_dt_px2_per_s"] * (DEFAULT_PIXEL_SIZE_UM**2) / 4.0
    joined["apparent_D_h5_slow_dt_px0p096_um2_per_s"] = joined["radius2_slope_h5_max_dt_px2_per_s"] * (DEFAULT_PIXEL_SIZE_UM**2) / 4.0
    joined["apparent_D_h5_timing_envelope_abs_min_um2_per_s"] = np.minimum(
        joined["apparent_D_h5_fast_dt_px0p096_um2_per_s"].abs(),
        joined["apparent_D_h5_slow_dt_px0p096_um2_per_s"].abs(),
    )
    joined["apparent_D_h5_timing_envelope_abs_max_um2_per_s"] = np.maximum(
        joined["apparent_D_h5_fast_dt_px0p096_um2_per_s"].abs(),
        joined["apparent_D_h5_slow_dt_px0p096_um2_per_s"].abs(),
    )

    source_summary = (
        joined.groupby("source_stem", dropna=False)
        .agg(
            n_rows=("roi_id", "count"),
            n_roi=("roi_id", "nunique"),
            n_cycles=("cycleNo", "nunique"),
            h5_dt_median_s=("sampled_timing_dt_median_s", "median"),
            h5_dt_min_s=("sampled_timing_dt_min_s", "median"),
            h5_dt_max_s=("sampled_timing_dt_max_s", "median"),
            h5_dt_max_to_median_ratio=("h5_dt_max_to_median_ratio", "median"),
            roi_elapsed_to_h5_median_ratio=("roi_elapsed_to_h5_median_ratio", "median"),
            frame_span_in_movie_fraction=("frame_span_in_movie", "mean"),
            positive_D_fraction=("apparent_D_h5median_px0p096_um2_per_s", lambda x: float((x > 0).mean())),
            median_abs_D_um2_per_s=("apparent_D_h5median_abs_um2_per_s", "median"),
        )
        .reset_index()
        .sort_values("source_stem")
    )

    threshold_summary = (
        joined.groupby("threshold_quantile", dropna=False)
        .agg(
            n_rows=("roi_id", "count"),
            n_roi=("roi_id", "nunique"),
            median_D_um2_per_s=("apparent_D_h5median_px0p096_um2_per_s", "median"),
            median_abs_D_um2_per_s=("apparent_D_h5median_abs_um2_per_s", "median"),
            positive_D_fraction=("apparent_D_h5median_px0p096_um2_per_s", lambda x: float((x > 0).mean())),
            median_timing_envelope_abs_min=("apparent_D_h5_timing_envelope_abs_min_um2_per_s", "median"),
            median_timing_envelope_abs_max=("apparent_D_h5_timing_envelope_abs_max_um2_per_s", "median"),
        )
        .reset_index()
        .sort_values("threshold_quantile")
    )

    feature_cols = [
        "apparent_D_h5median_px0p08_um2_per_s",
        "apparent_D_h5median_px0p096_um2_per_s",
        "apparent_D_h5median_px0p12_um2_per_s",
        "apparent_D_h5median_abs_um2_per_s",
        "apparent_D_h5_timing_envelope_abs_min_um2_per_s",
        "apparent_D_h5_timing_envelope_abs_max_um2_per_s",
        "roi_elapsed_to_h5_median_ratio",
        "h5_dt_max_to_median_ratio",
    ]
    tests = front_feature_tests(joined, feature_cols, "future_any_drop_within_8cycles")

    corr_pairs = [
        ("apparent_D_h5median_px0p096_um2_per_s", "transferred_masked_residual_signature"),
        ("apparent_D_h5median_abs_um2_per_s", "transferred_masked_residual_signature"),
        ("roi_elapsed_to_h5_median_ratio", "transferred_masked_residual_signature"),
        ("h5_dt_max_to_median_ratio", "transferred_masked_residual_signature"),
        ("apparent_D_h5median_px0p096_um2_per_s", "validation_score_recon"),
        ("roi_elapsed_to_h5_median_ratio", "validation_score_recon"),
    ]
    corr = correlations(joined, corr_pairs)

    top_threshold = threshold_summary.iloc[(threshold_summary["threshold_quantile"] - 0.70).abs().argsort()[:1]]
    q70_rows = joined[np.isclose(joined["threshold_quantile"].astype(float), 0.70)]
    q70_tests = front_feature_tests(q70_rows, feature_cols, "future_any_drop_within_8cycles")

    summary = {
        "n_threshold_rows": int(len(joined)),
        "n_roi": int(joined["roi_id"].nunique()),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "n_source_stems": int(joined["source_stem"].nunique()),
        "n_rows_with_h5_timing": int(joined["sampled_timing_dt_median_s"].notna().sum()),
        "n_roi_with_h5_timing": int(joined.loc[joined["sampled_timing_dt_median_s"].notna(), "roi_id"].nunique()),
        "pixel_size_um_assumptions": PIXEL_SIZES_UM,
        "default_pixel_size_um": DEFAULT_PIXEL_SIZE_UM,
        "median_roi_elapsed_to_h5_median_ratio": float(joined["roi_elapsed_to_h5_median_ratio"].median()),
        "max_source_h5_dt_max_to_median_ratio": float(source_summary["h5_dt_max_to_median_ratio"].max()),
        "median_q70_apparent_D_h5median_um2_per_s": float(q70_rows["apparent_D_h5median_px0p096_um2_per_s"].median()) if not q70_rows.empty else np.nan,
        "median_q70_abs_apparent_D_h5median_um2_per_s": float(q70_rows["apparent_D_h5median_abs_um2_per_s"].median()) if not q70_rows.empty else np.nan,
        "q70_positive_D_fraction": float((q70_rows["apparent_D_h5median_px0p096_um2_per_s"] > 0).mean()) if not q70_rows.empty else np.nan,
        "threshold_summary": records(threshold_summary, 20),
        "source_timing_summary": records(source_summary, 20),
        "future8_feature_tests_all_thresholds": records(tests, 20),
        "future8_feature_tests_q70": records(q70_tests, 20),
        "calibration_correlations": records(corr, 20),
        "guardrail": "Apparent diffusion values are recalibrated from HDF5 camera timing and slide-derived pixel-size assumptions. No HDF5 pixel-size attribute was found, and the values remain optical-front proxies, not validated material diffusion coefficients.",
    }

    joined.to_csv(out / "apparent_diffusion_calibration_joined.csv", index=False)
    source_summary.to_csv(out / "apparent_diffusion_source_timing_summary.csv", index=False)
    threshold_summary.to_csv(out / "apparent_diffusion_threshold_summary.csv", index=False)
    tests.to_csv(out / "apparent_diffusion_future8_tests.csv", index=False)
    q70_tests.to_csv(out / "apparent_diffusion_q70_future8_tests.csv", index=False)
    corr.to_csv(out / "apparent_diffusion_calibration_correlations.csv", index=False)
    with (out / "apparent_diffusion_calibration_bounds_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (out / "README.md").open("w") as f:
        f.write("# Apparent Diffusion Calibration Bounds\n\n")
        f.write("Maps balanced future-drop ROI front descriptors to source HDF5 camera timing and slide-derived pixel-size assumptions.\n\n")
        f.write(f"- Threshold rows: {summary['n_threshold_rows']}\n")
        f.write(f"- ROI with HDF5 timing: {summary['n_roi_with_h5_timing']} / {summary['n_roi']}\n")
        f.write(f"- Median ROI elapsed / HDF5 elapsed ratio: {summary['median_roi_elapsed_to_h5_median_ratio']:.3f}\n")
        f.write(f"- q70 median apparent D at 96 nm/px: {summary['median_q70_apparent_D_h5median_um2_per_s']:.3e} um2/s\n")
        f.write(f"- q70 positive-D fraction: {summary['q70_positive_D_fraction']:.3f}\n")
        f.write("\nGuardrail: apparent optical-front proxies only; no HDF5 pixel-size attribute confirms calibrated material diffusion.\n")


if __name__ == "__main__":
    main()
