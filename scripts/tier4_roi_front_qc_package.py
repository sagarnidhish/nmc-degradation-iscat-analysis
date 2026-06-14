#!/usr/bin/env python3
"""Build a compact manual-QC package for high-priority NMC ROI/front candidates.

The package is intentionally visual and small: it selects the most informative
particle-region ROI tensors from rollout/mobility, threshold-front, and
protocol-conditioned front analyses, renders first/middle/last/delta panels
with q70 bright-front contours, and writes a triage table with automatic QC
flags. It does not make accept/reject decisions for publication.
"""

import argparse
import html
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def central_mask(height: int, width: int, radius_frac: float = 0.48) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= radius_frac * min(height, width)


def edge_touch_fraction(mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return np.nan
    edge = np.zeros_like(mask, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    return float((mask & edge).sum() / mask.sum())


def component_stats(mask: np.ndarray) -> Dict[str, float]:
    labels, n_comp = ndi.label(mask)
    if n_comp == 0:
        return {
            "n_components": 0,
            "largest_component_fraction": np.nan,
            "edge_touch_fraction": np.nan,
        }
    sizes = ndi.sum(mask, labels, index=np.arange(1, n_comp + 1))
    largest = float(np.max(sizes))
    return {
        "n_components": int(n_comp),
        "largest_component_fraction": float(largest / mask.sum()) if mask.sum() else np.nan,
        "edge_touch_fraction": edge_touch_fraction(mask),
    }


def q70_masks(frames: np.ndarray) -> Dict[str, object]:
    t, h, w = frames.shape
    roi_mask = central_mask(h, w)
    early = frames[: max(4, t // 8)][:, roi_mask]
    threshold = float(np.nanquantile(early, 0.70))
    masks = [(frame >= threshold) & roi_mask & np.isfinite(frame) for frame in frames]
    return {"threshold": threshold, "roi_mask": roi_mask, "masks": masks}


def normalize_for_display(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(frame, dtype=float)
    return np.clip((frame - lo) / (hi - lo), 0, 1)


def render_roi_qc(roi_id: str, npz_path: str, row: pd.Series, out_png: Path) -> Dict[str, object]:
    with np.load(npz_path) as data:
        frames = np.asarray(data["frames_norm"], dtype=np.float32)
    first = frames[0]
    mid = frames[len(frames) // 2]
    last = frames[-1]
    delta = last - first
    masks = q70_masks(frames)
    first_mask = masks["masks"][0]
    mid_mask = masks["masks"][len(frames) // 2]
    last_mask = masks["masks"][-1]
    stats = component_stats(last_mask)
    roi_mask = masks["roi_mask"]
    lo, hi = np.nanpercentile(frames[:, roi_mask], [1, 99])
    dlim = float(np.nanpercentile(np.abs(delta[roi_mask]), 99))
    dlim = dlim if dlim > 0 else 1.0

    fig, axes = plt.subplots(1, 4, figsize=(11, 3), constrained_layout=True)
    panels = [
        ("first", first, first_mask),
        ("mid", mid, mid_mask),
        ("last", last, last_mask),
        ("last-first", delta, last_mask),
    ]
    for ax, (title, image, mask) in zip(axes, panels):
        if title == "last-first":
            ax.imshow(image, cmap="coolwarm", vmin=-dlim, vmax=dlim)
        else:
            ax.imshow(normalize_for_display(image, lo, hi), cmap="gray", vmin=0, vmax=1)
        ax.contour(mask.astype(float), levels=[0.5], colors=["lime"], linewidths=0.7)
        ax.contour(roi_mask.astype(float), levels=[0.5], colors=["yellow"], linewidths=0.4, alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    fig.suptitle(
        f"{roi_id} | {row.get('cohort_role', '')} cycle {finite_float(row.get('cycleNo')):.0f} | "
        f"phase+frac {finite_float(row.get('phase_slope_positive_fraction')):.2f}",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    stats.update({
        "q70_threshold": masks["threshold"],
        "q70_first_fraction": float(first_mask.sum() / roi_mask.sum()),
        "q70_last_fraction": float(last_mask.sum() / roi_mask.sum()),
        "q70_fraction_delta": float(last_mask.sum() / roi_mask.sum() - first_mask.sum() / roi_mask.sum()),
        "qc_png": str(out_png),
    })
    return stats


def add_rank(source: pd.DataFrame, score_col: str, rank_name: str, ascending: bool = False) -> pd.DataFrame:
    if source.empty or score_col not in source.columns:
        return pd.DataFrame(columns=["roi_id", rank_name])
    out = source[["roi_id", score_col]].copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.sort_values(score_col, ascending=ascending).drop_duplicates("roi_id")
    out[rank_name] = np.arange(1, len(out) + 1)
    return out[["roi_id", rank_name, score_col]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/roi_front_qc_package")
    parser.add_argument("--top-n", type=int, default=24)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out_dir = Path(args.out_dir)
    png_dir = out_dir / "qc_panels"
    png_dir.mkdir(parents=True, exist_ok=True)

    echem = pd.read_csv(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_joined.csv")
    rollout = pd.read_csv(derived / "multi_cycle_rollout_mobility_coupling" / "multi_cycle_rollout_mobility_ranked.csv")
    fronts = pd.read_csv(derived / "multi_cycle_threshold_robust_fronts" / "threshold_robust_front_summary.csv")
    conditioned = pd.read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")

    candidate = echem.copy()
    for src in [
        rollout[["roi_id", "rollout_mobility_difficulty_score"]].drop_duplicates("roi_id"),
        fronts.drop(columns=[c for c in ["cohort_role", "cycleNo", "event_reference_cycle"] if c in fronts.columns], errors="ignore").drop_duplicates("roi_id"),
        conditioned.drop(columns=[c for c in ["cohort_role", "cycleNo", "event_reference_cycle"] if c in conditioned.columns], errors="ignore").drop_duplicates("roi_id"),
    ]:
        candidate = candidate.merge(src, on="roi_id", how="left")

    ranks = [
        add_rank(candidate, "rollout_mobility_difficulty_score", "rollout_rank"),
        add_rank(candidate, "threshold_robust_phase_score", "phase_rank"),
        add_rank(candidate, "diffusion_proxy_abs_median_um2_per_s", "diffusion_proxy_rank"),
        add_rank(candidate, "phase_slope_positive_fraction_protocol_residual", "conditioned_phase_sign_rank"),
    ]
    rank_table = candidate[["roi_id"]].drop_duplicates()
    for r in ranks:
        rank_table = rank_table.merge(r, on="roi_id", how="left")
    rank_cols = ["rollout_rank", "phase_rank", "diffusion_proxy_rank", "conditioned_phase_sign_rank"]
    for col in rank_cols:
        rank_table[col] = pd.to_numeric(rank_table[col], errors="coerce").fillna(999)
    rank_table["qc_priority_score"] = sum(1.0 / rank_table[col].clip(lower=1) for col in rank_cols)

    candidate = candidate.merge(rank_table[["roi_id", "qc_priority_score"] + rank_cols], on="roi_id", how="left")
    base = candidate.sort_values(["qc_priority_score", "rollout_mobility_difficulty_score"], ascending=[False, False]).drop_duplicates("roi_id")
    n_control_target = max(6, args.top_n // 4)
    n_event_target = args.top_n - n_control_target
    selected_events = base[base["cohort_role"] == "event"].head(n_event_target)
    selected_controls = base[base["cohort_role"] == "control"].head(n_control_target)
    active_controls = (
        candidate[candidate["cohort_role"] == "control"]
        .sort_values("rollout_mobility_difficulty_score", ascending=False)
        .drop_duplicates("roi_id")
        .head(max(4, args.top_n // 6))
    )
    selected = (
        pd.concat([selected_events, selected_controls, active_controls], ignore_index=True)
        .drop_duplicates("roi_id")
        .sort_values(["cohort_role", "qc_priority_score"], ascending=[False, False])
        .head(args.top_n)
    )

    qc_rows: List[Dict[str, object]] = []
    for _, row in selected.iterrows():
        roi_id = str(row["roi_id"])
        out_png = png_dir / f"{roi_id}_qc.png"
        stats = render_roi_qc(roi_id, str(row["npz_path"]), row, out_png)
        review_flags = []
        if finite_float(stats["edge_touch_fraction"], 0) > 0.03:
            review_flags.append("front_touches_crop_edge")
        if finite_float(stats["largest_component_fraction"], 0) < 0.45:
            review_flags.append("fragmented_q70_mask")
        if finite_float(row.get("phase_slope_positive_fraction"), 0) < 0.75 and finite_float(row.get("phase_slope_negative_fraction"), 0) < 0.75:
            review_flags.append("threshold_sign_unstable")
        if finite_float(row.get("q70_radius2_slope_bootstrap_p05_px2_per_s")) < 0 < finite_float(row.get("q70_radius2_slope_bootstrap_p95_px2_per_s")):
            review_flags.append("diffusion_ci_crosses_zero")
        if row.get("cohort_role") == "control" and finite_float(row.get("rollout_mobility_difficulty_score"), 0) > 2:
            review_flags.append("active_control")
        qc_rows.append({
            "roi_id": roi_id,
            "cohort_role": row.get("cohort_role", ""),
            "cycleNo": finite_float(row.get("cycleNo")),
            "event_reference_cycle": finite_float(row.get("event_reference_cycle")),
            "degradation_mode_hypothesis": row.get("degradation_mode_hypothesis", ""),
            "qc_priority_score": finite_float(row.get("qc_priority_score")),
            "rollout_rank": finite_float(row.get("rollout_rank")),
            "phase_rank": finite_float(row.get("phase_rank")),
            "diffusion_proxy_rank": finite_float(row.get("diffusion_proxy_rank")),
            "conditioned_phase_sign_rank": finite_float(row.get("conditioned_phase_sign_rank")),
            "rollout_mobility_difficulty_score": finite_float(row.get("rollout_mobility_difficulty_score")),
            "threshold_robust_phase_score": finite_float(row.get("threshold_robust_phase_score")),
            "diffusion_proxy_abs_median_um2_per_s": finite_float(row.get("diffusion_proxy_abs_median_um2_per_s")),
            "phase_slope_positive_fraction_protocol_residual": finite_float(row.get("phase_slope_positive_fraction_protocol_residual")),
            "front_quality_score": finite_float(row.get("front_quality_score")),
            "first_last_corr": finite_float(row.get("first_last_corr")),
            "latent_path_length": finite_float(row.get("latent_path_length")),
            **stats,
            "auto_review_flags": ";".join(review_flags) if review_flags else "none",
            "manual_qc_status": "pending",
        })

    qc = pd.DataFrame(qc_rows).sort_values("qc_priority_score", ascending=False)
    manifest_path = out_dir / "roi_front_qc_manifest.csv"
    qc.to_csv(manifest_path, index=False)

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>NMC ROI Front QC Package</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ccc;padding:4px;font-size:12px;} img{max-width:520px;} .flags{font-weight:bold;color:#7a2100;}</style>",
        "</head><body>",
        "<h1>NMC ROI Front QC Package</h1>",
        "<p>Manual-review package for high-priority particle ROI/front candidates. Green contours show q70 bright-front masks; yellow contours show the central ROI guard mask.</p>",
        "<table><tr><th>ROI</th><th>Role</th><th>Cycle</th><th>Priority</th><th>Flags</th><th>Panel</th></tr>",
    ]
    for _, row in qc.iterrows():
        rel = Path(row["qc_png"]).relative_to(out_dir)
        html_lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['roi_id']))}</td>"
            f"<td>{html.escape(str(row['cohort_role']))}</td>"
            f"<td>{finite_float(row['cycleNo']):.0f}</td>"
            f"<td>{finite_float(row['qc_priority_score']):.3f}</td>"
            f"<td class='flags'>{html.escape(str(row['auto_review_flags']))}</td>"
            f"<td><img src='{html.escape(str(rel))}'></td>"
            "</tr>"
        )
    html_lines += ["</table></body></html>"]
    html_path = out_dir / "roi_front_qc_index.html"
    html_path.write_text("\n".join(html_lines))

    summary = {
        "n_selected_roi": int(len(qc)),
        "n_event_roi": int((qc["cohort_role"] == "event").sum()),
        "n_control_roi": int((qc["cohort_role"] == "control").sum()),
        "top_n_requested": int(args.top_n),
        "flag_counts": qc["auto_review_flags"].str.get_dummies(sep=";").sum().to_dict(),
        "top_qc_rois": qc.head(12).to_dict("records"),
        "guardrail": "QC panels are review aids for automatic ROI/front candidates; manual_qc_status remains pending until human inspection.",
        "outputs": {
            "manifest": str(manifest_path),
            "html_index": str(html_path),
            "qc_panels": str(png_dir),
        },
    }
    with (out_dir / "roi_front_qc_package_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (out_dir / "README.md").open("w") as f:
        f.write("# ROI Front QC Package\n\n")
        f.write("Compact manual-review package for high-priority NMC particle ROI/front candidates.\n")
        f.write("Open `roi_front_qc_index.html` or inspect `roi_front_qc_manifest.csv`; all manual QC statuses are pending.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
