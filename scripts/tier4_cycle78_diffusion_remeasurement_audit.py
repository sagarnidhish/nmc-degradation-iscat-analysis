#!/usr/bin/env python3
"""Targeted cycle-78 apparent-diffusion remeasurement audit.

The broader diffusion-readiness ledgers identify ``cycle78_rank22_obj2`` as the
nearest candidate to a calibrated-diffusion follow-up, but the q70 positive-CI
gate remains blocked. This script remeasures that ROI in a local same-source
context using multiple threshold, central-mask, and time-window choices. It is a
diagnostic packet for manual/retracked front follow-up, not a diffusion claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


TARGET_ROI = "cycle78_rank22_obj2"
CONTEXT_ROIS = [
    "cycle76_rank8_obj2",
    "cycle78_rank22_obj1",
    "cycle78_rank22_obj2",
    "cycle78_rank22_obj3",
    "cycle84_rank23_obj2",
]
PIXEL_SIZE_UM = 0.096


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


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 5 or np.nanstd(x[mask]) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(mask.sum())}
    xx = x[mask]
    yy = y[mask]
    slope, intercept = np.polyfit(xx, yy, 1)
    pred = slope * xx + intercept
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "n": int(mask.sum()),
    }


def block_bootstrap_ci(
    time_s: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    block_len: int,
) -> dict[str, float]:
    mask = np.isfinite(time_s) & np.isfinite(values)
    x = np.asarray(time_s, dtype=float)[mask]
    y = np.asarray(values, dtype=float)[mask]
    n = len(x)
    if n < max(8, block_len * 2):
        return {"p05": np.nan, "p50": np.nan, "p95": np.nan, "positive_ci": False, "n_boot": 0}
    slopes: list[float] = []
    starts = np.arange(0, n - block_len + 1)
    for _ in range(n_bootstrap):
        chunks = []
        while sum(len(c) for c in chunks) < n:
            s = int(rng.choice(starts))
            chunks.append(np.arange(s, s + block_len))
        idx = np.concatenate(chunks)[:n]
        order = np.argsort(x[idx])
        fit = linear_fit(x[idx][order], y[idx][order])
        if np.isfinite(fit["slope"]):
            slopes.append(fit["slope"])
    if not slopes:
        return {"p05": np.nan, "p50": np.nan, "p95": np.nan, "positive_ci": False, "n_boot": 0}
    arr = np.asarray(slopes, dtype=float)
    p05, p50, p95 = np.nanpercentile(arr, [5, 50, 95])
    return {
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "positive_ci": bool(p05 > 0),
        "n_boot": int(len(arr)),
    }


def central_mask(shape: tuple[int, int], radius_frac: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= radius_frac * min(h, w)


def perimeter(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float((mask & ~ndi.binary_erosion(mask)).sum())


def timing_seconds(root: Path, source_stem: str, frame_indices: np.ndarray) -> tuple[np.ndarray, str, float]:
    h5_path = root / "NMC_degradation_3_160623_Halfthedata" / f"{source_stem}.hdf5"
    fallback = frame_indices.astype(float) - float(frame_indices[0])
    if not h5_path.exists():
        return fallback, "frame_index_fallback_missing_h5", np.nan
    try:
        with h5py.File(h5_path, "r") as f:
            timing = np.asarray(f["camera_timing"])
    except Exception:
        return fallback, "frame_index_fallback_timing_error", np.nan
    trace = timing if timing.ndim == 1 else timing.reshape(-1, timing.shape[-1])[0]
    valid = (frame_indices >= 0) & (frame_indices < len(trace))
    if not bool(valid.all()):
        return fallback, "frame_index_fallback_out_of_range", np.nan
    t = trace[frame_indices].astype(float)
    t = t - float(t[0])
    if np.nanmax(t) > 1e6:
        t = t / 1e9
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 2 else np.nan
    return t, "camera_timing", dt


def load_npz(row: pd.Series) -> dict[str, Any]:
    path = Path(str(row["npz_path"]))
    data = np.load(path)
    frames = np.asarray(data["frames_norm"] if "frames_norm" in data else data["frames"], dtype=float)
    frame_indices = np.asarray(data["frame_indices"], dtype=int)
    return {"frames": frames, "frame_indices": frame_indices, "npz_path": str(path)}


def front_trace(frames: np.ndarray, q: float, radius_frac: float, baseline_frac: float) -> dict[str, np.ndarray | float]:
    t, h, w = frames.shape
    mask0 = central_mask((h, w), radius_frac)
    n_base = int(np.clip(round(t * baseline_frac), 6, max(6, t // 3)))
    threshold = float(np.nanquantile(frames[:n_base, mask0], q))
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    area0 = float(mask0.sum())
    frac = []
    radius2 = []
    iface = []
    cx_trace = []
    cy_trace = []
    for frame in frames:
        phase = mask0 & np.isfinite(frame) & (frame >= threshold)
        frac.append(float(phase.sum() / area0))
        radius2.append(float(np.nanmean(rr2[phase])) if phase.any() else np.nan)
        iface.append(float(perimeter(phase) / area0))
        if phase.any():
            cx_trace.append(float(np.nanmean(xx[phase]) - cx))
            cy_trace.append(float(np.nanmean(yy[phase]) - cy))
        else:
            cx_trace.append(np.nan)
            cy_trace.append(np.nan)
    return {
        "threshold_value": threshold,
        "phase_fraction": np.asarray(frac, dtype=float),
        "radius2_px2": np.asarray(radius2, dtype=float),
        "interface_density": np.asarray(iface, dtype=float),
        "centroid_x": np.asarray(cx_trace, dtype=float),
        "centroid_y": np.asarray(cy_trace, dtype=float),
        "baseline_frames": int(n_base),
    }


def time_window_indices(n: int, window: str) -> np.ndarray:
    if window == "full":
        return np.arange(n)
    if window == "early70":
        return np.arange(max(8, int(round(n * 0.70))))
    if window == "late70":
        return np.arange(n - max(8, int(round(n * 0.70))), n)
    if window == "middle70":
        k = max(8, int(round(n * 0.70)))
        start = (n - k) // 2
        return np.arange(start, start + k)
    raise ValueError(window)


def context_role(roi_id: str) -> str:
    if roi_id == TARGET_ROI:
        return "target_nearest_diffusion_candidate"
    if roi_id.startswith("cycle78_"):
        return "same_cycle_neighbor_object"
    if roi_id.startswith("cycle76_"):
        return "same_source_prev_cycle_same_object_rank"
    if roi_id.startswith("cycle84_"):
        return "same_source_later_cycle_same_object_rank"
    return "context"


def analyze_roi(
    root: Path,
    row: pd.Series,
    quantiles: list[float],
    radius_fracs: list[float],
    baseline_fracs: list[float],
    windows: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
    block_len: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loaded = load_npz(row)
    frames = loaded["frames"]
    frame_indices = loaded["frame_indices"]
    time_s, timing_source, dt = timing_seconds(root, str(row["source_stem"]), frame_indices)
    rows: list[dict[str, Any]] = []
    default_preview = None
    for q in quantiles:
        for rf in radius_fracs:
            for bf in baseline_fracs:
                tr = front_trace(frames, q, rf, bf)
                if abs(q - 0.70) < 1e-9 and abs(rf - 0.48) < 1e-9 and abs(bf - 0.125) < 1e-9:
                    default_preview = tr
                for window in windows:
                    idx = time_window_indices(len(time_s), window)
                    radius = np.asarray(tr["radius2_px2"], dtype=float)[idx]
                    frac = np.asarray(tr["phase_fraction"], dtype=float)[idx]
                    t_win = time_s[idx]
                    rfit = linear_fit(t_win, radius)
                    ffit = linear_fit(t_win, frac)
                    ci = block_bootstrap_ci(t_win, radius, rng, n_bootstrap, block_len)
                    cdx = np.asarray(tr["centroid_x"], dtype=float)[idx]
                    cdy = np.asarray(tr["centroid_y"], dtype=float)[idx]
                    step = np.sqrt(np.diff(cdx) ** 2 + np.diff(cdy) ** 2)
                    slope = rfit["slope"]
                    rows.append(
                        {
                            "roi_id": row["roi_id"],
                            "context_role": context_role(str(row["roi_id"])),
                            "cycleNo": float(row["cycleNo"]),
                            "source_stem": row["source_stem"],
                            "cohort_role": row.get("cohort_role", ""),
                            "validation_score": float(row.get("validation_score", np.nan)),
                            "threshold_quantile": q,
                            "central_radius_fraction": rf,
                            "baseline_fraction": bf,
                            "baseline_frames": tr["baseline_frames"],
                            "time_window": window,
                            "n_fit": rfit["n"],
                            "timing_source": timing_source,
                            "median_dt_s": dt,
                            "threshold_value": tr["threshold_value"],
                            "phase_fraction_first": float(frac[0]) if len(frac) else np.nan,
                            "phase_fraction_last": float(frac[-1]) if len(frac) else np.nan,
                            "phase_fraction_delta": float(frac[-1] - frac[0]) if len(frac) else np.nan,
                            "phase_fraction_slope_per_s": ffit["slope"],
                            "phase_fraction_slope_r2": ffit["r2"],
                            "radius2_slope_px2_per_s": slope,
                            "radius2_slope_r2": rfit["r2"],
                            "apparent_D_um2_per_s": slope * (PIXEL_SIZE_UM**2) / 4.0 if np.isfinite(slope) else np.nan,
                            "radius2_slope_bootstrap_p05_px2_per_s": ci["p05"],
                            "radius2_slope_bootstrap_p50_px2_per_s": ci["p50"],
                            "radius2_slope_bootstrap_p95_px2_per_s": ci["p95"],
                            "apparent_D_bootstrap_p05_um2_per_s": ci["p05"] * (PIXEL_SIZE_UM**2) / 4.0 if np.isfinite(ci["p05"]) else np.nan,
                            "apparent_D_bootstrap_p50_um2_per_s": ci["p50"] * (PIXEL_SIZE_UM**2) / 4.0 if np.isfinite(ci["p50"]) else np.nan,
                            "apparent_D_bootstrap_p95_um2_per_s": ci["p95"] * (PIXEL_SIZE_UM**2) / 4.0 if np.isfinite(ci["p95"]) else np.nan,
                            "positive_ci": bool(ci["positive_ci"]),
                            "n_bootstrap_slopes": ci["n_boot"],
                            "centroid_path_px": float(np.nansum(step)),
                            "centroid_net_px": float(np.sqrt((cdx[-1] - cdx[0]) ** 2 + (cdy[-1] - cdy[0]) ** 2)) if len(cdx) else np.nan,
                            "npz_path": loaded["npz_path"],
                        }
                    )
    meta = {
        "roi_id": row["roi_id"],
        "frames": frames,
        "time_s": time_s,
        "default_trace": default_preview,
    }
    return rows, meta


def summarize(per_variant: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for roi_id, grp in per_variant.groupby("roi_id", sort=False):
        d = pd.to_numeric(grp["apparent_D_um2_per_s"], errors="coerce")
        r2 = pd.to_numeric(grp["radius2_slope_r2"], errors="coerce")
        full = grp.loc[grp["time_window"].eq("full")]
        q70 = grp.loc[
            grp["time_window"].eq("full")
            & np.isclose(grp["threshold_quantile"], 0.70)
            & np.isclose(grp["central_radius_fraction"], 0.48)
            & np.isclose(grp["baseline_fraction"], 0.125)
        ]
        q70_row = q70.iloc[0] if not q70.empty else pd.Series(dtype=object)
        mrow = manifest.loc[manifest["roi_id"].eq(roi_id)].iloc[0]
        rows.append(
            {
                "roi_id": roi_id,
                "context_role": context_role(str(roi_id)),
                "cycleNo": float(mrow["cycleNo"]),
                "source_stem": mrow["source_stem"],
                "cohort_role": mrow.get("cohort_role", ""),
                "validation_score": float(mrow.get("validation_score", np.nan)),
                "n_variants": int(len(grp)),
                "n_full_window_variants": int(len(full)),
                "median_D_um2_per_s": float(d.median()) if d.notna().any() else np.nan,
                "q10_D_um2_per_s": float(d.quantile(0.10)) if d.notna().any() else np.nan,
                "q90_D_um2_per_s": float(d.quantile(0.90)) if d.notna().any() else np.nan,
                "positive_D_fraction": float((d > 0).mean()) if d.notna().any() else np.nan,
                "positive_ci_fraction": float(pd.Series(grp["positive_ci"]).astype(bool).mean()) if len(grp) else np.nan,
                "full_window_positive_ci_fraction": float(pd.Series(full["positive_ci"]).astype(bool).mean()) if len(full) else np.nan,
                "median_radius2_r2": float(r2.median()) if r2.notna().any() else np.nan,
                "max_radius2_r2": float(r2.max()) if r2.notna().any() else np.nan,
                "default_q70_D_um2_per_s": q70_row.get("apparent_D_um2_per_s", np.nan),
                "default_q70_D_p05_um2_per_s": q70_row.get("apparent_D_bootstrap_p05_um2_per_s", np.nan),
                "default_q70_D_p95_um2_per_s": q70_row.get("apparent_D_bootstrap_p95_um2_per_s", np.nan),
                "default_q70_positive_ci": bool(q70_row.get("positive_ci", False)),
                "default_q70_radius2_r2": q70_row.get("radius2_slope_r2", np.nan),
                "roi_norm_mean_delta_last_minus_first": float(mrow.get("roi_norm_mean_delta_last_minus_first", np.nan)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        for metric in ["median_D_um2_per_s", "positive_ci_fraction", "default_q70_D_um2_per_s", "default_q70_radius2_r2"]:
            vals = pd.to_numeric(out[metric], errors="coerce")
            out[f"{metric}_context_percentile"] = [
                float((np.sum(vals < v) + 0.5 * np.sum(vals == v)) / vals.notna().sum()) if np.isfinite(v) and vals.notna().any() else np.nan
                for v in vals
            ]
    return out


def save_target_plot(meta: dict[str, Any], out_png: Path) -> None:
    frames = np.asarray(meta["frames"], dtype=float)
    time_s = np.asarray(meta["time_s"], dtype=float)
    tr = meta.get("default_trace")
    idxs = [0, len(frames) // 2, len(frames) - 1]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, idx, label in zip(axes[0], idxs, ["first", "middle", "last"]):
        ax.imshow(frames[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    if tr:
        axes[1, 0].plot(time_s, tr["phase_fraction"], lw=1.5)
        axes[1, 0].set_title("q70 phase fraction", fontsize=9)
        axes[1, 1].plot(time_s, tr["radius2_px2"], lw=1.5)
        axes[1, 1].set_title("q70 radius^2", fontsize=9)
        axes[1, 2].plot(time_s, tr["interface_density"], lw=1.5)
        axes[1, 2].set_title("q70 interface density", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("elapsed seconds")
    fig.suptitle(str(meta["roi_id"]), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--roi-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/balanced_future_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle78_diffusion_remeasurement_audit")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--block-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=78)
    args = parser.parse_args()

    root = Path(args.root)
    roi_dir = Path(args.roi_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    manifest = pd.read_csv(roi_dir / "selected_roi_sequence_manifest.csv")
    selected = manifest.loc[manifest["roi_id"].isin(CONTEXT_ROIS)].copy()
    missing = sorted(set(CONTEXT_ROIS) - set(selected["roi_id"].astype(str)))
    if missing:
        raise FileNotFoundError(f"Missing context ROIs from manifest: {missing}")
    selected["context_role"] = selected["roi_id"].map(context_role)
    selected = selected.sort_values(["cycleNo", "roi_id"]).reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, Any]] = []
    target_meta: dict[str, Any] | None = None
    for _, row in selected.iterrows():
        rows, meta = analyze_roi(
            root=root,
            row=row,
            quantiles=[0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
            radius_fracs=[0.42, 0.48, 0.55],
            baseline_fracs=[0.10, 0.125, 0.20],
            windows=["full", "early70", "middle70", "late70"],
            rng=rng,
            n_bootstrap=args.n_bootstrap,
            block_len=args.block_len,
        )
        all_rows.extend(rows)
        if row["roi_id"] == TARGET_ROI:
            target_meta = meta
    per_variant = pd.DataFrame(all_rows)
    roi_summary = summarize(per_variant, selected)

    per_variant_path = out_dir / "cycle78_diffusion_remeasurement_per_variant.csv"
    roi_summary_path = out_dir / "cycle78_diffusion_remeasurement_roi_summary.csv"
    context_path = out_dir / "cycle78_diffusion_remeasurement_context_manifest.csv"
    per_variant.to_csv(per_variant_path, index=False)
    roi_summary.to_csv(roi_summary_path, index=False)
    selected.to_csv(context_path, index=False)

    if target_meta is not None:
        save_target_plot(target_meta, out_dir / "plots" / f"{TARGET_ROI}_q70_remeasurement.png")

    target_row = roi_summary.loc[roi_summary["roi_id"].eq(TARGET_ROI)]
    target = target_row.iloc[0].to_dict() if not target_row.empty else {}
    positive_ci = int(bool(target.get("default_q70_positive_ci", False)))
    status = "q70_positive_ci_passed_automatically" if positive_ci else "q70_positive_ci_still_blocked"
    summary = {
        "overall_status": "cycle78_diffusion_remeasurement_ready",
        "target_roi_id": TARGET_ROI,
        "target_status": status,
        "n_context_rois": int(len(selected)),
        "n_variant_rows": int(len(per_variant)),
        "n_bootstrap": int(args.n_bootstrap),
        "block_len": int(args.block_len),
        "pixel_size_um_assumed": PIXEL_SIZE_UM,
        "target_summary": clean_json(target),
        "context_summary": clean_json(roi_summary.to_dict(orient="records")),
        "outputs": {
            "per_variant": str(per_variant_path),
            "roi_summary": str(roi_summary_path),
            "context_manifest": str(context_path),
            "target_plot": str(out_dir / "plots" / f"{TARGET_ROI}_q70_remeasurement.png"),
            "summary": str(out_dir / "cycle78_diffusion_remeasurement_summary.json"),
        },
        "guardrail": "Automatic threshold/mask/window remeasurement only; does not accept manual labels, validate front identity, or create calibrated diffusion coefficients.",
    }
    summary_path = out_dir / "cycle78_diffusion_remeasurement_summary.json"
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    readme = out_dir / "README.md"
    readme.write_text(
        "# Cycle 78 Diffusion Remeasurement Audit\n\n"
        "Targeted q70/threshold remeasurement packet for `cycle78_rank22_obj2`, the current nearest diffusion-follow-up candidate.\n\n"
        "The audit compares the target against same-source/same-object and same-cycle neighboring ROIs using multiple threshold quantiles, central support masks, baseline windows, and contiguous-block bootstrap CIs.\n\n"
        "Guardrail: this is an automatic diagnostic packet for manual/retracked front follow-up. It is not a calibrated diffusion claim.\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
