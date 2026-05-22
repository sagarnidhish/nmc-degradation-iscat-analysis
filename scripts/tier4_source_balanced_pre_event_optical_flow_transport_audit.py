#!/usr/bin/env python3
"""Source-balanced pre-event optical-flow transport audit.

This audit estimates apparent particle-local motion from ROI videos using only a
history-derived particle mask. It is a guardrail experiment: the outputs are
optical transport proxies for ranking and hypothesis generation, not calibrated
material velocities or diffusion coefficients.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage, stats

try:
    from skimage.registration import optical_flow_tvl1
except Exception:  # pragma: no cover - remote dependency fallback
    optical_flow_tvl1 = None

try:
    import cv2
except Exception:  # pragma: no cover - remote dependency fallback
    cv2 = None


EVENT_ORDER = [
    "near_pre_event_1_8",
    "mid_pre_event_9_16",
    "far_pre_event_17_32",
    "post_event_1_16",
    "no_near_event_control",
]

METHOD = "farneback" if cv2 is not None else "tvl1"


def finite_float(x: Any) -> float | None:
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def robust_scale(frames: np.ndarray) -> np.ndarray:
    arr = frames.astype(np.float32, copy=False)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(hi - lo) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def stable_mask_from_history(frames: np.ndarray, train_fraction: float = 0.6) -> np.ndarray:
    n_train = max(8, int(round(frames.shape[0] * train_fraction)))
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
    if METHOD == "farneback" and cv2 is not None:
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)
    if optical_flow_tvl1 is not None:
        v, u = optical_flow_tvl1(a, b, attachment=15, tightness=0.3, num_warp=4, num_iter=40, tol=1e-4)
        return u.astype(np.float32), v.astype(np.float32)
    return np.zeros_like(a), np.zeros_like(a)


def auc_ap(y: np.ndarray, score: np.ndarray) -> Tuple[float, float, str]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, "NA"
    ranks = stats.rankdata(score)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    direction = "higher_in_positive"
    oriented = auc
    oriented_score = score.copy()
    if auc < 0.5:
        oriented = 1.0 - auc
        oriented_score = -score
        direction = "lower_in_positive"
    order = np.argsort(-oriented_score)
    yy = y[order]
    precision = np.cumsum(yy) / (np.arange(len(yy)) + 1)
    ap = float((precision * yy).sum() / max(1, n_pos))
    return float(oriented), ap, direction


def test_feature(df: pd.DataFrame, target: str, feature: str, transform: str = "raw") -> Dict[str, Any] | None:
    sub = df[["source_stem", "event_relative_bin", feature]].copy()
    if target == "near_vs_any_non_near":
        sub["y"] = (sub.event_relative_bin == "near_pre_event_1_8").astype(int)
    elif target == "near_vs_mid_pre":
        sub = sub[sub.event_relative_bin.isin(["near_pre_event_1_8", "mid_pre_event_9_16"])].copy()
        sub["y"] = (sub.event_relative_bin == "near_pre_event_1_8").astype(int)
    elif target == "near_vs_far_pre":
        sub = sub[sub.event_relative_bin.isin(["near_pre_event_1_8", "far_pre_event_17_32"])].copy()
        sub["y"] = (sub.event_relative_bin == "near_pre_event_1_8").astype(int)
    elif target == "near_vs_post_control":
        sub = sub[sub.event_relative_bin.isin(["near_pre_event_1_8", "post_event_1_16", "no_near_event_control"])].copy()
        sub["y"] = (sub.event_relative_bin == "near_pre_event_1_8").astype(int)
    else:
        return None
    sub = sub[np.isfinite(sub[feature])].copy()
    if sub.empty or sub.y.nunique() < 2 or len(sub) < 8:
        return None
    values = sub[feature].astype(float)
    if transform == "source_residual":
        values = values - sub.groupby("source_stem")[feature].transform("mean")
    y = sub.y.to_numpy()
    x = values.to_numpy(dtype=float)
    if np.nanstd(x) == 0:
        return None
    auc, ap, direction = auc_ap(y, x)
    pos = x[y == 1]
    neg = x[y == 0]
    try:
        mwu_p = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    except Exception:
        mwu_p = np.nan
    try:
        rho, sp = stats.spearmanr(y, x)
        rho, sp = float(rho), float(sp)
    except Exception:
        rho, sp = np.nan, np.nan
    return {
        "target": target,
        "feature": feature,
        "transform": transform,
        "n": int(len(sub)),
        "n_positive": int(y.sum()),
        "direction": direction,
        "oriented_auc": auc,
        "average_precision": ap,
        "median_positive_minus_negative": float(np.nanmedian(pos) - np.nanmedian(neg)),
        "mwu_p": mwu_p,
        "spearman_rho": rho,
        "spearman_p": sp,
    }


def summarize_roi(row: pd.Series, train_fraction: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    z = np.load(row["npz_path"], allow_pickle=True)
    frames = z["frames_norm"] if "frames_norm" in z.files else robust_scale(z["frames"])
    frames = robust_scale(frames)
    n = frames.shape[0]
    split = max(8, min(n - 3, int(round(n * train_fraction))))
    mask = stable_mask_from_history(frames, train_fraction=train_fraction)
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
    ry = yy - cy
    rx = xx - cx
    rr = np.sqrt(rx**2 + ry**2) + 1e-6
    erx = rx / rr
    ery = ry / rr

    metrics: Dict[str, List[float]] = {k: [] for k in [
        "particle_flow_mag", "context_flow_mag", "boundary_flow_mag", "radial_flow", "abs_radial_flow",
        "tangential_flow", "divergence", "curl", "gradient_aligned_flow", "intensity_delta_abs",
    ]}
    samples: List[Dict[str, Any]] = []
    sample_offsets = {split, (split + n - 2) // 2, n - 2}
    prev_mag = None
    accelerations: List[float] = []
    for t in range(split, n - 1):
        a, b = frames[t], frames[t + 1]
        u, v = flow_pair(a, b)
        mag = np.sqrt(u * u + v * v)
        radial = u * erx + v * ery
        tang = -u * ery + v * erx
        div = np.gradient(u, axis=1) + np.gradient(v, axis=0)
        curl = np.gradient(v, axis=1) - np.gradient(u, axis=0)
        gy, gx = np.gradient(a)
        gnorm = np.sqrt(gx * gx + gy * gy) + 1e-6
        galign = (u * gx + v * gy) / gnorm
        delta_abs = np.abs(b - a)
        metrics["particle_flow_mag"].append(float(np.nanmean(mag[mask])))
        metrics["context_flow_mag"].append(float(np.nanmean(mag[context])))
        metrics["boundary_flow_mag"].append(float(np.nanmean(mag[boundary])))
        metrics["radial_flow"].append(float(np.nanmean(radial[mask])))
        metrics["abs_radial_flow"].append(float(np.nanmean(np.abs(radial[mask]))))
        metrics["tangential_flow"].append(float(np.nanmean(np.abs(tang[mask]))))
        metrics["divergence"].append(float(np.nanmean(div[mask])))
        metrics["curl"].append(float(np.nanmean(np.abs(curl[mask]))))
        metrics["gradient_aligned_flow"].append(float(np.nanmean(galign[mask])))
        metrics["intensity_delta_abs"].append(float(np.nanmean(delta_abs[mask])))
        if prev_mag is not None:
            accelerations.append(float(abs(metrics["particle_flow_mag"][-1] - prev_mag)))
        prev_mag = metrics["particle_flow_mag"][-1]
        if t in sample_offsets:
            samples.append({
                "roi_id": row["roi_id"],
                "cycleNo": finite_float(row["cycleNo"]),
                "source_stem": row["source_stem"],
                "event_relative_bin": row["event_relative_bin"],
                "frame_index": int(t),
                "particle_flow_mag": metrics["particle_flow_mag"][-1],
                "context_flow_mag": metrics["context_flow_mag"][-1],
                "radial_flow": metrics["radial_flow"][-1],
                "divergence": metrics["divergence"][-1],
                "gradient_aligned_flow": metrics["gradient_aligned_flow"][-1],
            })

    out = {k: row.get(k) for k in [
        "roi_id", "cycleNo", "source_stem", "local_cycle_index", "object_candidate_rank", "event_relative_bin",
        "cycles_to_next_event", "cycles_since_prev_event", "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles", "any_abrupt_drop", "validation_score", "npz_path",
    ]}
    out.update({
        "n_frames": int(n),
        "flow_method": METHOD,
        "train_fraction": train_fraction,
        "mask_fraction": float(mask.mean()),
        "context_fraction": float(context.mean()),
        "heldout_flow_pairs": int(n - 1 - split),
    })
    for key, vals in metrics.items():
        arr = np.asarray(vals, dtype=float)
        out[f"{key}_mean"] = float(np.nanmean(arr))
        out[f"{key}_median"] = float(np.nanmedian(arr))
        out[f"{key}_q90"] = float(np.nanpercentile(arr, 90))
        out[f"{key}_slope"] = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 2 else np.nan
    out["particle_context_flow_ratio"] = out["particle_flow_mag_mean"] / (out["context_flow_mag_mean"] + 1e-9)
    out["boundary_particle_flow_ratio"] = out["boundary_flow_mag_mean"] / (out["particle_flow_mag_mean"] + 1e-9)
    out["radial_transport_bias"] = out["radial_flow_mean"] / (out["abs_radial_flow_mean"] + 1e-9)
    out["flow_acceleration_mean"] = float(np.nanmean(accelerations)) if accelerations else np.nan
    out["apparent_transport_instability_score"] = float(
        np.log1p(out["particle_flow_mag_q90"]) + np.log1p(abs(out["divergence_q90"])) + np.log1p(out["flow_acceleration_mean"] if np.isfinite(out["flow_acceleration_mean"]) else 0.0)
    )
    return out, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_roi_sequences/selected_roi_sequence_manifest.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_optical_flow_transport_audit")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        try:
            rec, sample_rows = summarize_roi(row, args.train_fraction)
            rows.append(rec)
            samples.extend(sample_rows)
        except Exception as exc:
            failures.append({"roi_id": row.get("roi_id"), "npz_path": row.get("npz_path"), "error": repr(exc)})

    per_roi = pd.DataFrame(rows)
    sample_df = pd.DataFrame(samples)
    fail_df = pd.DataFrame(failures)

    feature_cols = [
        c for c in per_roi.columns
        if c.endswith(("_mean", "_median", "_q90", "_slope", "_ratio", "_bias"))
        or c in {"mask_fraction", "flow_acceleration_mean", "apparent_transport_instability_score"}
    ]
    tests: List[Dict[str, Any]] = []
    for target in ["near_vs_any_non_near", "near_vs_mid_pre", "near_vs_far_pre", "near_vs_post_control"]:
        for feature in feature_cols:
            for transform in ["raw", "source_residual"]:
                res = test_feature(per_roi, target, feature, transform)
                if res is not None:
                    tests.append(res)
    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        tests_df = tests_df.sort_values(["mwu_p", "oriented_auc"], ascending=[True, False])

    method_summary = pd.DataFrame([
        {
            "flow_method": METHOD,
            "n_ok": int(len(per_roi)),
            "median_particle_flow_mag": float(per_roi["particle_flow_mag_mean"].median()) if not per_roi.empty else np.nan,
            "median_context_flow_mag": float(per_roi["context_flow_mag_mean"].median()) if not per_roi.empty else np.nan,
            "median_particle_context_flow_ratio": float(per_roi["particle_context_flow_ratio"].median()) if not per_roi.empty else np.nan,
            "median_mask_fraction": float(per_roi["mask_fraction"].median()) if not per_roi.empty else np.nan,
        }
    ])
    source_summary = per_roi.groupby("source_stem", dropna=False).agg(
        n_roi=("roi_id", "count"),
        n_cycles=("cycleNo", "nunique"),
        median_particle_flow_mag=("particle_flow_mag_mean", "median"),
        median_transport_instability=("apparent_transport_instability_score", "median"),
    ).reset_index() if not per_roi.empty else pd.DataFrame()

    paths = {
        "per_roi_csv": out_dir / "source_balanced_pre_event_optical_flow_transport_per_roi.csv",
        "frame_samples_csv": out_dir / "source_balanced_pre_event_optical_flow_transport_frame_samples.csv",
        "event_tests_csv": out_dir / "source_balanced_pre_event_optical_flow_transport_event_tests.csv",
        "method_summary_csv": out_dir / "source_balanced_pre_event_optical_flow_transport_method_summary.csv",
        "source_summary_csv": out_dir / "source_balanced_pre_event_optical_flow_transport_source_summary.csv",
        "failures_csv": out_dir / "source_balanced_pre_event_optical_flow_transport_failures.csv",
        "summary_json": out_dir / "source_balanced_pre_event_optical_flow_transport_summary.json",
        "readme": out_dir / "README.md",
    }
    per_roi.to_csv(paths["per_roi_csv"], index=False)
    sample_df.to_csv(paths["frame_samples_csv"], index=False)
    tests_df.to_csv(paths["event_tests_csv"], index=False)
    method_summary.to_csv(paths["method_summary_csv"], index=False)
    source_summary.to_csv(paths["source_summary_csv"], index=False)
    fail_df.to_csv(paths["failures_csv"], index=False)

    top_tests = tests_df.head(32).to_dict("records") if not tests_df.empty else []
    top_source_residual_tests = []
    best_source_residual = {}
    if not tests_df.empty:
        sr = tests_df[tests_df["transform"] == "source_residual"]
        if not sr.empty:
            top_source_residual_tests = sr.head(16).to_dict("records")
            best_source_residual = sr.iloc[0].to_dict()
    summary = {
        "n_input_rows": int(len(manifest)),
        "n_ok": int(len(per_roi)),
        "n_failed": int(len(failures)),
        "n_cycles": int(per_roi["cycleNo"].nunique()) if not per_roi.empty else 0,
        "n_sources": int(per_roi["source_stem"].nunique()) if not per_roi.empty else 0,
        "flow_method": METHOD,
        "train_fraction": args.train_fraction,
        "event_relative_bin_counts": per_roi["event_relative_bin"].value_counts().reindex(EVENT_ORDER).dropna().astype(int).to_dict() if not per_roi.empty else {},
        "method_summary": method_summary.to_dict("records"),
        "top_event_tests": top_tests,
        "top_source_residual_event_tests": top_source_residual_tests,
        "best_source_residual_test": best_source_residual,
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Apparent optical-flow transport is computed inside history-derived automatic particle masks on normalized ROI crops. It is an image-motion proxy for hypothesis ranking, not calibrated particle velocity, phase-boundary velocity, material flux, or diffusion.",
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    readme = [
        "# Source-Balanced Pre-Event Optical-Flow Transport Audit",
        "",
        f"- Input rows / OK / failed: {len(manifest)} / {len(per_roi)} / {len(failures)}",
        f"- Flow method: {METHOD}",
        f"- Event bins: {summary['event_relative_bin_counts']}",
        f"- Top event test: {top_tests[0] if top_tests else 'none'}",
        "",
        "Outputs:",
    ]
    readme += [f"- `{Path(v).name}`" for k, v in paths.items() if k not in {"summary_json", "readme"}]
    readme += ["", "Guardrail:", summary["guardrail"]]
    paths["readme"].write_text("\n".join(readme) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
