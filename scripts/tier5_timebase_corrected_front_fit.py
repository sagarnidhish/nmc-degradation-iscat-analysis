#!/usr/bin/env python3
"""Timebase-corrected front-motion fits for ROI tensors.

Why this exists:
- The current "apparent diffusion" / front-motion pipeline is intentionally
  conservative and (correctly) blocks calibrated diffusion claims because:
  (1) source timebases can be irregular, and (2) radius^2-vs-time fits can be
  fragile under drift/blur and threshold sensitivity.
- Many existing summaries assume a near-uniform dt (or use coarse sampled dt).

This script re-fits simple front-motion proxies using *per-frame camera_timing*
for the exact ROI frame indices, without loading whole movies.

Scope / guardrail:
- Outputs are "apparent front kinematics" for experiment design and ranking.
- They are NOT a calibrated material diffusion coefficient.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import h5py
import numpy as np
import pandas as pd


DEFAULT_PIXEL_SIZE_UM = 0.096


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def central_mask(height: int, width: int, radius_frac: float = 0.48) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= radius_frac * min(height, width)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 3:
        return np.nan
    resid = y[mask] - yhat[mask]
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def infer_seconds(trace: np.ndarray) -> np.ndarray:
    """Convert raw camera_timing values to seconds with a conservative heuristic."""
    trace = np.asarray(trace, dtype=np.float64)
    trace = trace[np.isfinite(trace)]
    if trace.size == 0:
        return trace
    span = float(np.nanmax(trace) - np.nanmin(trace))
    # NMC HDF5 timing is often nanosecond-like; detect by magnitude.
    if span > 1e6:
        return trace / 1e9
    return trace


def read_camera_timing_seconds(h5_path: Path, frame_indices: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"h5_path": str(h5_path)}
    frame_indices = np.asarray(frame_indices, dtype=int)
    frame_indices = frame_indices[np.isfinite(frame_indices)]
    frame_indices = frame_indices[(frame_indices >= 0)]
    if frame_indices.size < 3:
        return np.array([], dtype=float), {**meta, "status": "insufficient_frame_indices"}

    with h5py.File(h5_path, "r") as f:
        if "camera_timing" not in f:
            return np.array([], dtype=float), {**meta, "status": "missing_camera_timing"}
        ds = f["camera_timing"]
        # camera_timing may be 1D (n_frames) or 2D (n_chunks, n_frames); handle both.
        if ds.ndim == 1:
            n = int(ds.shape[0])
            idx = frame_indices[frame_indices < n]
            raw = np.asarray(ds[idx], dtype=np.float64)
        else:
            flat = ds.reshape(-1)
            n = int(flat.shape[0])
            idx = frame_indices[frame_indices < n]
            raw = np.asarray(flat[idx], dtype=np.float64)

    t = infer_seconds(raw)
    if t.size < 3:
        return np.array([], dtype=float), {**meta, "status": "empty_timing"}

    # Normalize to start at 0 for numerical stability.
    t0 = float(np.nanmin(t))
    t = t - t0
    dt = np.diff(np.sort(t))
    meta.update(
        status="ok",
        n_timing=int(t.size),
        elapsed_s=float(np.nanmax(t) - np.nanmin(t)),
        dt_median_s=float(np.nanmedian(dt)) if dt.size else np.nan,
        dt_p10_s=float(np.nanpercentile(dt, 10)) if dt.size else np.nan,
        dt_p90_s=float(np.nanpercentile(dt, 90)) if dt.size else np.nan,
        dt_max_to_median_ratio=float(np.nanmax(dt) / np.nanmedian(dt)) if dt.size and np.nanmedian(dt) > 0 else np.nan,
    )
    return t, meta


@dataclass(frozen=True)
class FitResult:
    slope: float
    intercept: float
    r2: float
    n: int
    bootstrap_p05: float
    bootstrap_p50: float
    bootstrap_p95: float


def robust_linear_fit_with_bootstrap(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int = 400) -> FitResult:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 6:
        return FitResult(np.nan, np.nan, np.nan, int(x.size), np.nan, np.nan, np.nan)

    # Center x to reduce numerical issues.
    x0 = float(np.median(x))
    xc = x - x0

    # Base fit (least squares on centered x).
    coef = np.polyfit(xc, y, 1)
    slope = float(coef[0])
    intercept = float(coef[1] - slope * x0)
    yhat = slope * x + intercept
    r2 = r2_score(y, yhat)

    # Bootstrap slopes with subsampling to reduce leverage of timing outliers.
    slopes = []
    n = int(x.size)
    take = max(6, int(round(0.8 * n)))
    for _ in range(int(n_boot)):
        idx = rng.choice(n, size=take, replace=True)
        xb = x[idx]
        yb = y[idx]
        if np.nanstd(xb) == 0:
            continue
        coef_b = np.polyfit(xb - np.median(xb), yb, 1)
        slopes.append(float(coef_b[0]))
    if len(slopes) < 20:
        return FitResult(slope, intercept, r2, n, np.nan, np.nan, np.nan)
    slopes = np.asarray(slopes, dtype=float)
    return FitResult(
        slope=slope,
        intercept=intercept,
        r2=r2,
        n=n,
        bootstrap_p05=float(np.nanpercentile(slopes, 5)),
        bootstrap_p50=float(np.nanpercentile(slopes, 50)),
        bootstrap_p95=float(np.nanpercentile(slopes, 95)),
    )


def front_radius2_series(frames_norm: np.ndarray) -> Dict[str, np.ndarray]:
    frames = np.asarray(frames_norm, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(f"expected frames_norm (t,h,w), got shape {frames.shape}")
    t, h, w = frames.shape
    roi_mask = central_mask(h, w)
    yy, xx = np.mgrid[:h, :w]
    cy0 = (h - 1) / 2.0
    cx0 = (w - 1) / 2.0
    rr2 = (yy - cy0) ** 2 + (xx - cx0) ** 2

    early = frames[: max(6, t // 8)][:, roi_mask]
    low_thr = float(np.nanpercentile(early, 30))
    high_thr = float(np.nanpercentile(early, 70))
    mid_thr = 0.5 * (low_thr + high_thr)

    thr_values = {
        "thr30": low_thr,
        "thr50": mid_thr,
        "thr70": high_thr,
    }

    out: Dict[str, list[float]] = {k: [] for k in thr_values}
    out_frac: Dict[str, list[float]] = {f"{k}_frac": [] for k in thr_values}
    roi_area = float(roi_mask.sum())
    for frame in frames:
        valid = roi_mask & np.isfinite(frame)
        for name, thr in thr_values.items():
            mask = valid & (frame >= thr)
            out[name].append(float(np.mean(rr2[mask])) if mask.any() else np.nan)
            out_frac[f"{name}_frac"].append(float(mask.sum() / roi_area))

    series: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    series.update({k: np.asarray(v, dtype=float) for k, v in out_frac.items()})
    return series


def apparent_D_from_radius2_slope(slope_px2_per_s: float, pixel_size_um: float) -> float:
    # For diffusion-like scaling in 2D radial symmetry, <r^2> ~ 4 D t is common.
    return float(slope_px2_per_s) * (float(pixel_size_um) ** 2) / 4.0


def iter_roi_rows(manifest: pd.DataFrame) -> Iterable[dict[str, Any]]:
    for _, row in manifest.iterrows():
        yield {k: row.get(k) for k in manifest.columns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, help="ROI sequence manifest with npz_path and source_stem.")
    parser.add_argument(
        "--h5-root",
        default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/NMC_degradation_3_160623_Halfthedata",
        help="Folder containing per-cycle HDF5 files named {source_stem}.hdf5",
    )
    parser.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    manifest = pd.read_csv(args.manifest_csv)
    h5_root = Path(args.h5_root)
    rng = np.random.default_rng(int(args.seed))

    rows = []
    timing_rows = []
    for row in iter_roi_rows(manifest):
        roi_id = str(row.get("roi_id"))
        npz_path = Path(str(row.get("npz_path")))
        source_stem = str(row.get("source_stem"))
        h5_path = h5_root / f"{source_stem}.hdf5"

        if not npz_path.exists():
            rows.append({**row, "status": "missing_npz"})
            continue
        if not h5_path.exists():
            rows.append({**row, "status": "missing_h5"})
            continue

        try:
            with np.load(npz_path) as data:
                frames_norm = np.asarray(data["frames_norm"], dtype=np.float32)
                frame_indices = np.asarray(data["frame_indices"], dtype=int)
        except Exception as exc:
            rows.append({**row, "status": f"npz_read_error:{exc.__class__.__name__}"})
            continue

        t_s, timing_meta = read_camera_timing_seconds(h5_path, frame_indices)
        timing_rows.append({"roi_id": roi_id, "source_stem": source_stem, **timing_meta})
        if timing_meta.get("status") != "ok":
            rows.append({**row, "status": f"timing_{timing_meta.get('status')}"})
            continue

        try:
            series = front_radius2_series(frames_norm)
        except Exception as exc:
            rows.append({**row, "status": f"front_series_error:{exc.__class__.__name__}"})
            continue

        # Fit each threshold variant.
        roi_out: Dict[str, Any] = {**row, "status": "ok"}
        roi_out.update(
            {
                "timing_elapsed_s": timing_meta.get("elapsed_s"),
                "timing_dt_median_s": timing_meta.get("dt_median_s"),
                "timing_dt_max_to_median_ratio": timing_meta.get("dt_max_to_median_ratio"),
                "pixel_size_um": float(args.pixel_size_um),
            }
        )

        for key in ("thr30", "thr50", "thr70"):
            fit = robust_linear_fit_with_bootstrap(t_s, series[key], rng=rng)
            slope_col = f"{key}_radius2_slope_px2_per_s"
            roi_out[slope_col] = fit.slope
            roi_out[f"{key}_radius2_slope_r2"] = fit.r2
            roi_out[f"{key}_radius2_slope_n"] = fit.n
            roi_out[f"{key}_radius2_slope_boot_p05_px2_per_s"] = fit.bootstrap_p05
            roi_out[f"{key}_radius2_slope_boot_p50_px2_per_s"] = fit.bootstrap_p50
            roi_out[f"{key}_radius2_slope_boot_p95_px2_per_s"] = fit.bootstrap_p95
            roi_out[f"{key}_apparent_D_um2_per_s"] = apparent_D_from_radius2_slope(fit.slope, args.pixel_size_um)
            roi_out[f"{key}_apparent_D_boot_p05_um2_per_s"] = apparent_D_from_radius2_slope(fit.bootstrap_p05, args.pixel_size_um)
            roi_out[f"{key}_apparent_D_boot_p50_um2_per_s"] = apparent_D_from_radius2_slope(fit.bootstrap_p50, args.pixel_size_um)
            roi_out[f"{key}_apparent_D_boot_p95_um2_per_s"] = apparent_D_from_radius2_slope(fit.bootstrap_p95, args.pixel_size_um)
            roi_out[f"{key}_frac_mean"] = float(np.nanmean(series.get(f"{key}_frac", np.array([np.nan]))))
            roi_out[f"{key}_frac_delta"] = float(series.get(f"{key}_frac", np.array([np.nan]))[-1] - series.get(f"{key}_frac", np.array([np.nan]))[0])

        rows.append(roi_out)

    table = pd.DataFrame(rows)
    timing = pd.DataFrame(timing_rows)

    # A simple summary for quick scanning.
    ok = table[table["status"].eq("ok")].copy() if "status" in table else pd.DataFrame()
    summary = {
        "n_rows": int(len(table)),
        "n_ok": int(len(ok)),
        "pixel_size_um": float(args.pixel_size_um),
        "timing_dt_max_to_median_ratio_p90": float(np.nanpercentile(pd.to_numeric(ok.get("timing_dt_max_to_median_ratio"), errors="coerce"), 90)) if len(ok) else None,
        "guardrail": "Timebase-corrected front fits are apparent kinematics for ranking/diagnosis. They are not calibrated diffusion coefficients.",
        "outputs": {
            "roi_table": str(out_dir / "timebase_corrected_front_fit_rois.csv"),
            "timing_table": str(out_dir / "timebase_corrected_front_fit_timing.csv"),
            "summary_json": str(out_dir / "timebase_corrected_front_fit_summary.json"),
            "readme": str(out_dir / "README.md"),
        },
    }

    table.to_csv(out_dir / "timebase_corrected_front_fit_rois.csv", index=False)
    if not timing.empty:
        timing.to_csv(out_dir / "timebase_corrected_front_fit_timing.csv", index=False)
    (out_dir / "timebase_corrected_front_fit_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))

    lines = [
        "# Timebase-Corrected Front Fit",
        "",
        "Re-fit ROI front/radius^2 slopes using per-frame `camera_timing` values for the exact ROI frame indices.",
        "",
        f"- Rows in manifest: {summary['n_rows']}",
        f"- Rows with successful timebase-corrected fits: {summary['n_ok']}",
        f"- Pixel size assumption: {summary['pixel_size_um']} um/px",
        "",
        "## Outputs",
        "",
        f"- `timebase_corrected_front_fit_rois.csv`: per-ROI threshold front slopes (px^2/s) and apparent D (um^2/s) with bootstrap CIs.",
        f"- `timebase_corrected_front_fit_timing.csv`: per-ROI timing diagnostics (dt quantiles, dt max/median ratio).",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

