#!/usr/bin/env python3
"""Component-aware retracking for the cycle-78 front candidate.

The q70 front for ``cycle78_rank22_obj2`` has positive apparent expansion but a
fragmented threshold mask. This audit tests whether a simple connected-component
front repair can isolate a coherent particle-local component while preserving
the positive radius^2 trend. It remains an automatic diagnostic and does not
accept manual front labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    if int(mask.sum()) < 6 or np.nanstd(x[mask]) == 0:
        return {"slope": np.nan, "r2": np.nan, "n": int(mask.sum())}
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    pred = slope * x[mask] + intercept
    ss_res = float(np.sum((y[mask] - pred) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return {"slope": float(slope), "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan, "n": int(mask.sum())}


def central_mask(shape: tuple[int, int], radius_frac: float = 0.48) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= radius_frac * min(h, w)


def load_frames(row: pd.Series) -> np.ndarray:
    data = np.load(str(row["npz_path"]))
    return np.asarray(data["frames_norm"] if "frames_norm" in data else data["frames"], dtype=float)


def threshold_masks(frames: np.ndarray, quantile: float) -> tuple[np.ndarray, float, np.ndarray]:
    roi_mask = central_mask(frames.shape[1:])
    n_base = int(np.clip(round(len(frames) * 0.125), 6, max(6, len(frames) // 3)))
    threshold = float(np.nanquantile(frames[:n_base, roi_mask], quantile))
    masks = np.asarray([roi_mask & np.isfinite(frame) & (frame >= threshold) for frame in frames], dtype=bool)
    return masks, threshold, roi_mask


def component_repair(mask: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "raw":
        return mask
    labels, nlab = ndi.label(mask)
    if nlab == 0:
        return mask
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if strategy == "largest_component":
        keep = int(np.argmax(counts))
    elif strategy == "central_component":
        scores = np.full(nlab + 1, -np.inf, dtype=float)
        for lab in range(1, nlab + 1):
            pts = labels == lab
            if not pts.any():
                continue
            dist = np.sqrt((np.mean(yy[pts]) - cy) ** 2 + (np.mean(xx[pts]) - cx) ** 2)
            scores[lab] = counts[lab] / (1.0 + dist)
        keep = int(np.argmax(scores))
    elif strategy == "top3_components":
        labs = np.argsort(counts)[-3:]
        return np.isin(labels, labs[labs > 0])
    else:
        raise ValueError(strategy)
    return labels == keep


def trace_for_strategy(frames: np.ndarray, quantile: float, strategy: str) -> dict[str, Any]:
    masks, threshold, _ = threshold_masks(frames, quantile)
    repaired = np.asarray([component_repair(mask, strategy) for mask in masks], dtype=bool)
    h, w = frames.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    area = []
    radius2 = []
    largest_fraction = []
    component_count = []
    centroids = []
    for raw, mask in zip(masks, repaired):
        raw_labels, raw_n = ndi.label(raw)
        raw_area = float(raw.sum())
        component_count.append(int(raw_n))
        if raw_n and raw_area:
            counts = np.bincount(raw_labels.ravel())[1:]
            largest_fraction.append(float(counts.max() / raw_area))
        else:
            largest_fraction.append(np.nan)
        area.append(float(mask.sum()))
        radius2.append(float(np.nanmean(rr2[mask])) if mask.any() else np.nan)
        if mask.any():
            centroids.append((float(np.nanmean(yy[mask]) - cy), float(np.nanmean(xx[mask]) - cx)))
        else:
            centroids.append((np.nan, np.nan))
    area = np.asarray(area, dtype=float)
    radius2 = np.asarray(radius2, dtype=float)
    cents = np.asarray(centroids, dtype=float)
    t = np.arange(len(frames), dtype=float)
    fit = linear_fit(t, radius2)
    steps = np.sqrt(np.diff(cents[:, 0]) ** 2 + np.diff(cents[:, 1]) ** 2)
    return {
        "threshold_value": threshold,
        "area_px_median": float(np.nanmedian(area)),
        "area_px_cv": float(np.nanstd(area) / np.nanmean(area)) if np.nanmean(area) > 0 else np.nan,
        "raw_component_count_median": float(np.nanmedian(component_count)),
        "raw_largest_component_fraction_median": float(np.nanmedian(largest_fraction)),
        "radius2_slope_px2_per_sample": fit["slope"],
        "radius2_slope_r2": fit["r2"],
        "apparent_D_um2_per_sample": fit["slope"] * (PIXEL_SIZE_UM**2) / 4.0 if np.isfinite(fit["slope"]) else np.nan,
        "centroid_path_px": float(np.nansum(steps)),
        "centroid_net_px": float(np.sqrt((cents[-1, 0] - cents[0, 0]) ** 2 + (cents[-1, 1] - cents[0, 1]) ** 2)) if len(cents) else np.nan,
        "masks": repaired,
        "radius2": radius2,
    }


def save_plot(row: pd.Series, frames: np.ndarray, traces: dict[str, dict[str, Any]], out_png: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    idxs = [0, len(frames) // 2, len(frames) - 1]
    raw = traces["raw"]["masks"]
    repaired = traces["largest_component"]["masks"]
    for ax, idx, label in zip(axes[0, :3], idxs, ["first", "middle", "last"]):
        ax.imshow(frames[idx], cmap="gray", vmin=0, vmax=1)
        ax.contour(raw[idx].astype(float), levels=[0.5], colors="yellow", linewidths=0.5)
        ax.contour(repaired[idx].astype(float), levels=[0.5], colors="lime", linewidths=0.9)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    axes[0, 3].axis("off")
    axes[0, 3].text(0, 0.8, f"{row['roi_id']}\nyellow raw q70\nlime largest component", fontsize=9)
    t = np.arange(len(frames), dtype=float)
    for strategy, tr in traces.items():
        axes[1, 0].plot(t, tr["radius2"], label=strategy, lw=1)
        axes[1, 1].plot(t, tr["masks"].sum(axis=(1, 2)), label=strategy, lw=1)
    axes[1, 0].set_title("radius^2 trace", fontsize=8)
    axes[1, 1].set_title("area trace", fontsize=8)
    axes[1, 0].legend(fontsize=6)
    axes[1, 1].legend(fontsize=6)
    axes[1, 2].axis("off")
    axes[1, 3].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/balanced_future_roi_sequences")
    parser.add_argument("--review-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/cycle78_front_identity_review_packet")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/cycle78_component_front_retracking_audit")
    args = parser.parse_args()

    roi_dir = Path(args.roi_dir)
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    manifest = pd.read_csv(roi_dir / "selected_roi_sequence_manifest.csv")
    rows = manifest.loc[manifest["roi_id"].isin(CONTEXT_ROIS)].copy()
    if len(rows) != len(CONTEXT_ROIS):
        missing = sorted(set(CONTEXT_ROIS) - set(rows["roi_id"].astype(str)))
        raise FileNotFoundError(f"Missing ROI rows: {missing}")
    review = pd.read_csv(Path(args.review_dir) / "cycle78_front_identity_review_manifest.csv")
    rows = rows.merge(
        review[["roi_id", "context_role", "default_q70_D_um2_per_s", "default_q70_positive_ci", "automatic_front_identity_flags"]],
        on="roi_id",
        how="left",
    )

    out_rows = []
    for _, row in rows.sort_values(["cycleNo", "roi_id"]).iterrows():
        frames = load_frames(row)
        traces = {}
        for strategy in ["raw", "largest_component", "central_component", "top3_components"]:
            traces[strategy] = trace_for_strategy(frames, 0.70, strategy)
            tr = traces[strategy]
            out_rows.append({
                "roi_id": row["roi_id"],
                "context_role": row.get("context_role", ""),
                "cycleNo": float(row["cycleNo"]),
                "source_stem": row["source_stem"],
                "strategy": strategy,
                "default_q70_positive_ci": bool(row.get("default_q70_positive_ci", False)),
                "front_identity_flags": row.get("automatic_front_identity_flags", ""),
                "threshold_value": tr["threshold_value"],
                "area_px_median": tr["area_px_median"],
                "area_px_cv": tr["area_px_cv"],
                "raw_component_count_median": tr["raw_component_count_median"],
                "raw_largest_component_fraction_median": tr["raw_largest_component_fraction_median"],
                "radius2_slope_px2_per_sample": tr["radius2_slope_px2_per_sample"],
                "radius2_slope_r2": tr["radius2_slope_r2"],
                "apparent_D_um2_per_sample": tr["apparent_D_um2_per_sample"],
                "centroid_path_px": tr["centroid_path_px"],
                "centroid_net_px": tr["centroid_net_px"],
            })
        save_plot(row, frames, traces, plot_dir / f"{row['roi_id']}_component_retracking.png")

    per_strategy = pd.DataFrame(out_rows)
    target = per_strategy.loc[per_strategy["roi_id"].eq(TARGET_ROI)].copy()
    raw_target = target.loc[target["strategy"].eq("raw")].iloc[0].to_dict()
    largest_target = target.loc[target["strategy"].eq("largest_component")].iloc[0].to_dict()
    central_target = target.loc[target["strategy"].eq("central_component")].iloc[0].to_dict()
    top3_target = target.loc[target["strategy"].eq("top3_components")].iloc[0].to_dict()

    per_path = out_dir / "cycle78_component_front_retracking_per_strategy.csv"
    per_strategy.to_csv(per_path, index=False)
    summary = {
        "overall_status": "cycle78_component_front_retracking_ready",
        "target_roi_id": TARGET_ROI,
        "n_context_rois": int(rows["roi_id"].nunique()),
        "n_strategy_rows": int(len(per_strategy)),
        "target_raw": clean_json(raw_target),
        "target_largest_component": clean_json(largest_target),
        "target_central_component": clean_json(central_target),
        "target_top3_components": clean_json(top3_target),
        "largest_component_preserves_positive_slope": bool(largest_target.get("radius2_slope_px2_per_sample", np.nan) > 0),
        "largest_component_improves_r2": bool(largest_target.get("radius2_slope_r2", -np.inf) > raw_target.get("radius2_slope_r2", np.inf)),
        "outputs": {
            "per_strategy": str(per_path),
            "plots_dir": str(plot_dir),
            "summary": str(out_dir / "cycle78_component_front_retracking_summary.json"),
        },
        "guardrail": "Automatic connected-component front retracking only; does not validate the front or create calibrated diffusion coefficients.",
    }
    (out_dir / "cycle78_component_front_retracking_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    (out_dir / "README.md").write_text(
        "# Cycle 78 Component Front Retracking Audit\n\n"
        "Connected-component retracking diagnostic for the fragmented q70 front of `cycle78_rank22_obj2` and local context ROIs.\n\n"
        "This tests whether largest/central/top3 component choices preserve a positive radius-squared trend. It is automatic and does not replace manual front identity review.\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
