#!/usr/bin/env python3
"""Segmented timebase-corrected front-motion fits for ROI tensors.

This is a follow-up to `tier5_timebase_corrected_front_fit.py`.

Motivation:
- Many ROI sequences contain regime switches (phase boundary stalls, reversal,
  crack events, illumination drift) so a single global r^2-vs-time slope often
  has poor linear-fit quality.
- The diffusion readiness gates care about whether there exists a stable,
  positive-expansion interval with reasonable linearity.

Approach:
- Use per-frame `camera_timing` at the ROI's `frame_indices`.
- Compute simple threshold-based front radius^2 proxies (thr30/thr50/thr70).
- For each threshold, search over contiguous time windows and select the
  "best" linear segment under constraints:
  - minimum points per window
  - finite values
  - optional preference for positive slope

Guardrail:
- Output is an analysis aid for diagnosing which ROIs are timebase-stable and
  have any diffusion-like intervals. It is not a calibrated diffusion claim.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np


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
    return value


def infer_seconds(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    trace = trace[np.isfinite(trace)]
    if trace.size == 0:
        return trace
    span = float(np.nanmax(trace) - np.nanmin(trace))
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

    with h5py.File(str(h5_path), "r") as f:
        if "camera_timing" not in f:
            return np.array([], dtype=float), {**meta, "status": "missing_camera_timing"}
        ds = f["camera_timing"]
        if ds.ndim == 1:
            n = int(ds.shape[0])
            idx = frame_indices[frame_indices < n]
            raw = np.asarray(ds[idx], dtype=np.float64)
        else:
            n1, n2 = int(ds.shape[0]), int(ds.shape[1])
            n = n1 * n2
            idx = frame_indices[frame_indices < n]
            if idx.size == 0:
                return np.array([], dtype=float), {**meta, "status": "empty_timing"}
            rows = (idx // n2).astype(int)
            cols = (idx % n2).astype(int)
            raw = np.asarray([ds[int(r), int(c)] for r, c in zip(rows, cols)], dtype=np.float64)

    t = infer_seconds(raw)
    if t.size < 3:
        return np.array([], dtype=float), {**meta, "status": "empty_timing"}
    t0 = float(np.nanmin(t))
    t = t - t0
    dt = np.diff(np.sort(t))
    meta.update(
        status="ok",
        n_timing=int(t.size),
        elapsed_s=float(np.nanmax(t) - np.nanmin(t)),
        dt_median_s=float(np.nanmedian(dt)) if dt.size else float("nan"),
        dt_p10_s=float(np.nanpercentile(dt, 10)) if dt.size else float("nan"),
        dt_p90_s=float(np.nanpercentile(dt, 90)) if dt.size else float("nan"),
        dt_max_to_median_ratio=float(np.nanmax(dt) / np.nanmedian(dt)) if dt.size and np.nanmedian(dt) > 0 else float("nan"),
    )
    return t, meta


def central_mask(height: int, width: int, radius_frac: float = 0.48) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= radius_frac * min(height, width)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 3:
        return float("nan")
    resid = y[mask] - yhat[mask]
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def front_radius2_series(frames_norm: np.ndarray) -> Dict[str, np.ndarray]:
    frames = np.asarray(frames_norm, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("frames_norm must have shape (t,h,w)")
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

    thr_values = {"thr30": low_thr, "thr50": mid_thr, "thr70": high_thr}
    out: Dict[str, List[float]] = {k: [] for k in thr_values}
    for frame in frames:
        valid = roi_mask & np.isfinite(frame)
        for name, thr in thr_values.items():
            mask = valid & (frame >= thr)
            out[name].append(float(np.mean(rr2[mask])) if mask.any() else float("nan"))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def fit_segment(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    coef = np.polyfit(x - np.median(x), y, 1)
    slope = float(coef[0])
    intercept = float(np.median(y) - slope * np.median(x))
    yhat = slope * x + intercept
    return slope, intercept, r2_score(y, yhat)


def best_linear_window(x: np.ndarray, y: np.ndarray, min_pts: int, prefer_positive: bool) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < max(6, min_pts):
        return {"status": "too_few_points"}

    best = None
    # O(n^2) window search is fine at n~96.
    for i in range(0, n - min_pts + 1):
        for j in range(i + min_pts, n + 1):
            xx = x[i:j]
            yy = y[i:j]
            if np.nanstd(xx) == 0:
                continue
            slope, intercept, r2 = fit_segment(xx, yy)
            if not np.isfinite(r2):
                continue
            if prefer_positive and slope <= 0:
                continue
            # Score prefers high r2 and longer windows.
            score = float(r2) * math.log(float(len(xx)) + 1.0)
            cand = (score, r2, slope, intercept, i, j, len(xx))
            if best is None or cand > best:
                best = cand

    if best is None:
        return {"status": "no_valid_window"}
    score, r2, slope, intercept, i, j, nwin = best
    return {
        "status": "ok",
        "window_i": int(i),
        "window_j": int(j),
        "window_n": int(nwin),
        "window_t0_s": float(x[i]),
        "window_t1_s": float(x[j - 1]),
        "slope_px2_per_s": float(slope),
        "intercept_px2": float(intercept),
        "r2": float(r2),
        "score": float(score),
    }


def apparent_D_from_radius2_slope(slope_px2_per_s: float, pixel_size_um: float) -> float:
    return float(slope_px2_per_s) * (float(pixel_size_um) ** 2) / 4.0


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(str(path), newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--h5-root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/NMC_degradation_3_160623_Halfthedata")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM)
    ap.add_argument("--min-window-points", type=int, default=24)
    ap.add_argument("--prefer-positive", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    manifest = read_manifest(Path(args.manifest_csv))
    h5_root = Path(args.h5_root)

    rows: List[Dict[str, Any]] = []
    for row in manifest:
        roi_id = str(row.get("roi_id"))
        source_stem = str(row.get("source_stem"))
        npz_path = Path(str(row.get("npz_path")))
        h5_path = h5_root / (source_stem + ".hdf5")

        out: Dict[str, Any] = dict(row)
        out["status"] = "ok"
        if not npz_path.exists():
            out["status"] = "missing_npz"
            rows.append(out)
            continue
        if not h5_path.exists():
            out["status"] = "missing_h5"
            rows.append(out)
            continue

        with np.load(str(npz_path)) as data:
            frames_norm = np.asarray(data["frames_norm"], dtype=np.float32)
            frame_indices = np.asarray(data["frame_indices"], dtype=int)
        t_s, tmeta = read_camera_timing_seconds(h5_path, frame_indices)
        out.update(
            timing_status=tmeta.get("status"),
            timing_elapsed_s=tmeta.get("elapsed_s"),
            timing_dt_median_s=tmeta.get("dt_median_s"),
            timing_dt_max_to_median_ratio=tmeta.get("dt_max_to_median_ratio"),
            pixel_size_um=float(args.pixel_size_um),
        )
        if tmeta.get("status") != "ok":
            out["status"] = "timing_" + str(tmeta.get("status"))
            rows.append(out)
            continue

        series = front_radius2_series(frames_norm)
        for key in ("thr30", "thr50", "thr70"):
            res = best_linear_window(t_s, series[key], min_pts=int(args.min_window_points), prefer_positive=bool(args.prefer_positive))
            out[key + "_seg_status"] = res.get("status")
            if res.get("status") != "ok":
                continue
            out[key + "_seg_window_n"] = res.get("window_n")
            out[key + "_seg_window_t0_s"] = res.get("window_t0_s")
            out[key + "_seg_window_t1_s"] = res.get("window_t1_s")
            out[key + "_seg_slope_px2_per_s"] = res.get("slope_px2_per_s")
            out[key + "_seg_r2"] = res.get("r2")
            out[key + "_seg_apparent_D_um2_per_s"] = apparent_D_from_radius2_slope(res.get("slope_px2_per_s"), args.pixel_size_um)

        rows.append(out)

    table_path = out_dir / "segmented_timebase_front_fit_rois.csv"
    with open(str(table_path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "n_rows": int(len(rows)),
        "n_ok": int(sum(1 for r in rows if r.get("status") == "ok")),
        "min_window_points": int(args.min_window_points),
        "prefer_positive": bool(args.prefer_positive),
        "outputs": {"roi_table": str(table_path)},
        "guardrail": "Segmented timebase fits select the best linear window; this is for diagnosis/ranking and is not a calibrated diffusion claim.",
    }
    (out_dir / "segmented_timebase_front_fit_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

