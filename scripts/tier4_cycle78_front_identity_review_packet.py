#!/usr/bin/env python3
"""Build a targeted cycle-78 front-identity review packet.

The cycle-78 remeasurement clears the narrow automatic q70 positive-CI blocker
for ``cycle78_rank22_obj2``. The next blocker is not another numeric sweep; it
is front identity: does the q70 thresholded structure look like a coherent
particle-local phase/front signal rather than a fragmented mask, crop artifact,
or drift/blur artifact? This packet renders the target and local context ROIs
with automatic QC metrics and an explicit pending manual-review rubric.
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
        return {"slope": np.nan, "r2": np.nan}
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    pred = slope * x[mask] + intercept
    ss_res = float(np.sum((y[mask] - pred) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return {"slope": float(slope), "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan}


def central_mask(shape: tuple[int, int], radius_frac: float = 0.48) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= radius_frac * min(h, w)


def load_frames(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(str(row["npz_path"]))
    frames = np.asarray(data["frames_norm"] if "frames_norm" in data else data["frames"], dtype=float)
    frame_indices = np.asarray(data["frame_indices"], dtype=int)
    return frames, frame_indices


def q70_masks(frames: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    roi_mask = central_mask(frames.shape[1:])
    n_base = int(np.clip(round(len(frames) * 0.125), 6, max(6, len(frames) // 3)))
    threshold = float(np.nanquantile(frames[:n_base, roi_mask], 0.70))
    masks = np.asarray([roi_mask & np.isfinite(frame) & (frame >= threshold) for frame in frames], dtype=bool)
    return masks, threshold, roi_mask, np.arange(len(frames), dtype=float)


def mask_metrics(frames: np.ndarray) -> dict[str, Any]:
    masks, threshold, roi_mask, t = q70_masks(frames)
    h, w = frames.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    roi_area = float(roi_mask.sum())
    areas = []
    n_components = []
    largest_component_fraction = []
    edge_contact = []
    radius2 = []
    centroids = []
    for mask in masks:
        area = float(mask.sum())
        areas.append(area / roi_area)
        labels, nlab = ndi.label(mask)
        n_components.append(int(nlab))
        if nlab > 0 and area > 0:
            counts = np.bincount(labels.ravel())[1:]
            largest_component_fraction.append(float(counts.max() / area))
        else:
            largest_component_fraction.append(np.nan)
        edge = np.zeros(mask.shape, dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        edge_contact.append(float((mask & edge).sum() / area) if area > 0 else np.nan)
        radius2.append(float(np.nanmean(rr2[mask])) if mask.any() else np.nan)
        if mask.any():
            centroids.append((float(np.nanmean(yy[mask]) - cy), float(np.nanmean(xx[mask]) - cx)))
        else:
            centroids.append((np.nan, np.nan))
    areas_arr = np.asarray(areas, dtype=float)
    radius2_arr = np.asarray(radius2, dtype=float)
    cents = np.asarray(centroids, dtype=float)
    steps = np.sqrt(np.diff(cents[:, 0]) ** 2 + np.diff(cents[:, 1]) ** 2)
    fit = linear_fit(t, radius2_arr)
    frag_fraction = float(np.mean(np.asarray(n_components) > 3))
    large_frac_med = float(np.nanmedian(largest_component_fraction))
    edge_med = float(np.nanmedian(edge_contact))
    area_cv = float(np.nanstd(areas_arr) / np.nanmean(areas_arr)) if np.nanmean(areas_arr) > 0 else np.nan
    flags: list[str] = []
    if frag_fraction > 0.30:
        flags.append("fragmented_q70_mask")
    if large_frac_med < 0.65:
        flags.append("no_dominant_component")
    if edge_med > 0.03:
        flags.append("edge_contact")
    if area_cv > 0.60:
        flags.append("area_unstable")
    if not np.isfinite(fit["r2"]) or fit["r2"] < 0.25:
        flags.append("weak_radius2_fit")
    return {
        "q70_threshold": threshold,
        "q70_area_fraction_median": float(np.nanmedian(areas_arr)),
        "q70_area_fraction_cv": area_cv,
        "q70_components_median": float(np.nanmedian(n_components)),
        "q70_fragmented_frame_fraction": frag_fraction,
        "q70_largest_component_fraction_median": large_frac_med,
        "q70_edge_contact_fraction_median": edge_med,
        "q70_centroid_path_px": float(np.nansum(steps)),
        "q70_centroid_net_px": float(np.sqrt((cents[-1, 0] - cents[0, 0]) ** 2 + (cents[-1, 1] - cents[0, 1]) ** 2)),
        "q70_radius2_slope_per_sample": fit["slope"],
        "q70_radius2_r2_per_sample": fit["r2"],
        "automatic_front_identity_flags": ";".join(flags) if flags else "none",
        "automatic_front_identity_score": float(
            (1.0 - min(frag_fraction, 1.0))
            + large_frac_med
            + (1.0 - min(edge_med * 10.0, 1.0))
            + min(max(fit["r2"], 0.0), 1.0)
        ),
    }


def save_panel(row: pd.Series, frames: np.ndarray, out_png: Path) -> None:
    masks, threshold, roi_mask, _ = q70_masks(frames)
    idxs = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
    fig, axes = plt.subplots(2, len(idxs), figsize=(12, 5))
    for j, idx in enumerate(idxs):
        ax = axes[0, j]
        ax.imshow(frames[idx], cmap="gray", vmin=0, vmax=1)
        ax.contour(masks[idx].astype(float), levels=[0.5], colors="lime", linewidths=0.8)
        ax.contour(roi_mask.astype(float), levels=[0.5], colors="yellow", linewidths=0.5)
        ax.set_title(f"f{idx}", fontsize=8)
        ax.axis("off")
        diff = frames[idx] - np.nanmedian(frames[: max(6, len(frames) // 8)], axis=0)
        vmax = max(0.01, float(np.nanpercentile(np.abs(diff), 99)))
        ax2 = axes[1, j]
        ax2.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax2.contour(masks[idx].astype(float), levels=[0.5], colors="black", linewidths=0.7)
        ax2.axis("off")
    fig.suptitle(f"{row['roi_id']} | q70={threshold:.4f} | manual QC pending", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_html(manifest: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for _, row in manifest.iterrows():
        png_rel = Path(row["panel_png"]).name
        rows.append(
            f"<section><h2>{row['review_id']}: {row['roi_id']}</h2>"
            f"<p>Role: {row['context_role']} | Flags: {row['automatic_front_identity_flags']} | "
            f"Score: {row['automatic_front_identity_score']:.3f}</p>"
            f"<img src='panels/{png_rel}' alt='{row['roi_id']} panel'>"
            "<p>Manual fields in CSV: particle identity, q70 front coherence, edge/drift artifact risk, diffusion interpretability, final decision.</p>"
            "</section>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Cycle 78 Front Identity Review</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;max-width:1200px}"
        "section{border-top:1px solid #ccc;padding:18px 0}img{max-width:100%;height:auto}"
        "h1,h2{font-weight:600}</style></head><body>"
        "<h1>Cycle 78 Front Identity Review Packet</h1>"
        "<p>Automatic panels for pending manual front identity review. No labels are assigned by this file.</p>"
        + "\n".join(rows)
        + "</body></html>"
    )
    (out_dir / "cycle78_front_identity_review.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/balanced_future_roi_sequences")
    parser.add_argument("--remeasure-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/cycle78_diffusion_remeasurement_audit")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/cycle78_front_identity_review_packet")
    args = parser.parse_args()

    roi_dir = Path(args.roi_dir)
    remeasure_dir = Path(args.remeasure_dir)
    out_dir = Path(args.out_dir)
    panel_dir = out_dir / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(exist_ok=True)

    manifest = pd.read_csv(roi_dir / "selected_roi_sequence_manifest.csv")
    context = manifest.loc[manifest["roi_id"].isin(CONTEXT_ROIS)].copy()
    if len(context) != len(CONTEXT_ROIS):
        missing = sorted(set(CONTEXT_ROIS) - set(context["roi_id"].astype(str)))
        raise FileNotFoundError(f"Missing ROI rows: {missing}")
    remeasure = pd.read_csv(remeasure_dir / "cycle78_diffusion_remeasurement_roi_summary.csv")
    context = context.merge(
        remeasure[[
            "roi_id",
            "context_role",
            "default_q70_D_um2_per_s",
            "default_q70_D_p05_um2_per_s",
            "default_q70_D_p95_um2_per_s",
            "default_q70_positive_ci",
            "default_q70_radius2_r2",
            "positive_ci_fraction",
        ]],
        on="roi_id",
        how="left",
    )
    rows = []
    for i, (_, row) in enumerate(context.sort_values(["cycleNo", "roi_id"]).iterrows(), start=1):
        frames, _ = load_frames(row)
        metrics = mask_metrics(frames)
        panel_png = panel_dir / f"{row['roi_id']}_front_identity_panel.png"
        save_panel(row, frames, panel_png)
        out = row.to_dict()
        out.update(metrics)
        out["review_id"] = f"C78-FRONT-{i:03d}"
        out["panel_png"] = str(panel_png)
        out["manual_particle_identity_ok"] = ""
        out["manual_q70_front_coherent"] = ""
        out["manual_artifact_or_drift_risk"] = ""
        out["manual_diffusion_interpretable"] = ""
        out["manual_final_decision"] = ""
        out["manual_reviewer"] = ""
        out["manual_review_date"] = ""
        rows.append(out)
    review = pd.DataFrame(rows)
    review = review.sort_values(["roi_id"]).reset_index(drop=True)
    review_path = out_dir / "cycle78_front_identity_review_manifest.csv"
    review.to_csv(review_path, index=False)
    write_html(review, out_dir)

    target = review.loc[review["roi_id"].eq(TARGET_ROI)].iloc[0].to_dict()
    summary = {
        "overall_status": "cycle78_front_identity_review_packet_ready",
        "target_roi_id": TARGET_ROI,
        "n_review_rows": int(len(review)),
        "n_target_rows": int(review["roi_id"].eq(TARGET_ROI).sum()),
        "n_rows_no_automatic_flags": int((review["automatic_front_identity_flags"] == "none").sum()),
        "n_rows_default_q70_positive_ci": int(review["default_q70_positive_ci"].astype(bool).sum()),
        "target_summary": clean_json(target),
        "outputs": {
            "manifest": str(review_path),
            "html": str(out_dir / "cycle78_front_identity_review.html"),
            "panels_dir": str(panel_dir),
            "summary": str(out_dir / "cycle78_front_identity_review_summary.json"),
        },
        "guardrail": "Review packet and automatic front-identity metrics only; manual fields are blank and no diffusion/front labels are accepted.",
    }
    (out_dir / "cycle78_front_identity_review_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    (out_dir / "README.md").write_text(
        "# Cycle 78 Front Identity Review Packet\n\n"
        "Rendered q70 front panels and a pending manual-review manifest for `cycle78_rank22_obj2` and same-source context ROIs.\n\n"
        "The packet is intended to resolve the front-identity/manual-QC blocker after the cycle-78 q70 positive-CI remeasurement. It does not assign labels or create calibrated diffusion coefficients.\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
