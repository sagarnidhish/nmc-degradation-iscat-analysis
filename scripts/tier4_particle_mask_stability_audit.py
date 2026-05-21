#!/usr/bin/env python3
"""Audit history-aware particle-mask stability on ROI-only NMC tensors.

The ROI tensors are already cropped around particle candidates, but drift
correction and soft boundaries can blur the apparent particle support. This
script builds a conservative per-frame candidate mask from temporal and local
contrast, then falls back to the previous accepted mask when the candidate
violates simple history constraints.

The output is a guardrail audit: it quantifies mask stability/fallback pressure
for downstream physics claims. It is not a replacement for manual particle
segmentation.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu, spearmanr


def finite_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_mwu(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    aa = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    bb = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if aa.empty or bb.empty:
        return {
            "n_event": int(len(aa)),
            "n_control": int(len(bb)),
            "median_event": np.nan,
            "median_control": np.nan,
            "median_diff_event_minus_control": np.nan,
            "p_value": np.nan,
        }
    try:
        _, p = mannwhitneyu(aa, bb, alternative="two-sided")
    except Exception:
        p = np.nan
    return {
        "n_event": int(len(aa)),
        "n_control": int(len(bb)),
        "median_event": float(aa.median()),
        "median_control": float(bb.median()),
        "median_diff_event_minus_control": float(aa.median() - bb.median()),
        "p_value": float(p) if np.isfinite(p) else np.nan,
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    scale = 1.4826 * mad if np.isfinite(mad) and mad > 1e-9 else np.nanstd(values)
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - med) / scale).astype(np.float32)


def central_ellipse(height: int, width: int, frac: float = 0.49) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    ry = max(frac * height, 1.0)
    rx = max(frac * width, 1.0)
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def component_features(mask: np.ndarray) -> Tuple[float, float, float, int, float]:
    if mask.sum() == 0:
        return 0.0, np.nan, np.nan, 0, np.nan
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return 0.0, np.nan, np.nan, 0, np.nan
    sizes = ndi.sum(mask, labels, index=np.arange(1, n_labels + 1))
    largest_idx = int(np.argmax(sizes) + 1)
    largest = labels == largest_idx
    yy, xx = np.nonzero(largest)
    area = float(largest.sum())
    frag = float(area / mask.sum()) if mask.sum() else np.nan
    return area, float(np.mean(xx)), float(np.mean(yy)), int(n_labels), frag


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask, labels, index=np.arange(1, n_labels + 1))
    largest_idx = int(np.argmax(sizes) + 1)
    return labels == largest_idx


def build_prior_mask(frames: np.ndarray) -> np.ndarray:
    t, h, w = frames.shape
    finite_frames = np.nan_to_num(frames, nan=float(np.nanmedian(frames)))
    median_img = np.nanmedian(finite_frames, axis=0)
    std_img = np.nanstd(finite_frames, axis=0)
    score = np.abs(robust_z(median_img)) + 0.75 * robust_z(std_img)
    score = ndi.gaussian_filter(score, sigma=1.0)
    center = central_ellipse(h, w, 0.49)
    score = np.where(center, score, -np.inf)
    finite_score = score[np.isfinite(score) & center]
    if finite_score.size == 0:
        return center
    q = float(np.nanquantile(finite_score, 0.45))
    prior = center & (score >= q)
    prior = ndi.binary_closing(prior, iterations=2)
    prior = ndi.binary_fill_holes(prior)
    prior = largest_component(prior)
    min_area = 0.12 * h * w
    if prior.sum() < min_area:
        prior = center
    return prior.astype(bool)


def candidate_mask(frame: np.ndarray, temporal_std: np.ndarray, prior: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    finite = np.isfinite(frame)
    fill = float(np.nanmedian(frame[finite])) if finite.any() else 0.0
    frame = np.where(finite, frame, fill)
    local = frame - ndi.gaussian_filter(frame, sigma=4.0)
    score = np.abs(robust_z(local)) + 0.35 * robust_z(temporal_std)
    score = ndi.gaussian_filter(score, sigma=0.8)
    vals = score[prior & np.isfinite(score)]
    if vals.size == 0:
        return np.zeros_like(prior, dtype=bool)
    thresh = float(np.nanquantile(vals, 0.48))
    mask = prior & (score >= thresh)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    return largest_component(mask)


def trace_masks(frames: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    t, h, w = frames.shape
    prior = build_prior_mask(frames)
    prior_area = float(prior.sum())
    temporal_std = np.nanstd(frames, axis=0)
    previous = prior.copy()
    accepted_areas: List[float] = []
    accepted_centroids: List[Tuple[float, float]] = []
    rows = []
    max_jump = 0.23 * min(h, w)

    for idx, frame in enumerate(frames):
        cand = candidate_mask(frame, temporal_std, prior)
        cand_area, cand_x, cand_y, cand_components, cand_largest_frac = component_features(cand)
        prev_area = float(previous.sum())
        hist_area = float(np.nanmedian(accepted_areas[-8:])) if accepted_areas else np.nan
        hist_x = float(np.nanmedian([p[0] for p in accepted_centroids[-8:]])) if accepted_centroids else (w - 1) / 2.0
        hist_y = float(np.nanmedian([p[1] for p in accepted_centroids[-8:]])) if accepted_centroids else (h - 1) / 2.0
        jump = float(np.hypot(cand_x - hist_x, cand_y - hist_y)) if np.isfinite(cand_x) and np.isfinite(hist_x) else np.nan

        low_floor = max(64.0, 0.08 * prior_area)
        high_ceiling = min(1.05 * prior_area, 0.85 * h * w)
        low_area = cand_area < low_floor
        high_area = cand_area > high_ceiling
        if accepted_areas and np.isfinite(hist_area):
            low_area = low_area or cand_area < 0.40 * hist_area
            high_area = high_area or cand_area > 2.25 * hist_area
        fragmented = (cand_components > 3) or (np.isfinite(cand_largest_frac) and cand_largest_frac < 0.72)
        jumpy = np.isfinite(jump) and jump > max_jump
        empty = cand_area <= 0
        fallback = bool(empty or low_area or high_area or fragmented or jumpy)
        reasons = []
        if empty:
            reasons.append("empty")
        if low_area:
            reasons.append("low_area")
        if high_area:
            reasons.append("high_area")
        if fragmented:
            reasons.append("fragmented")
        if jumpy:
            reasons.append("centroid_jump")

        accepted = previous.copy() if fallback else cand
        area, cx, cy, n_comp, largest_frac = component_features(accepted)
        previous = accepted
        accepted_areas.append(area)
        accepted_centroids.append((cx, cy))
        rows.append({
            "frame_local_index": idx,
            "candidate_area_px": cand_area,
            "accepted_area_px": area,
            "candidate_area_fraction_of_prior": cand_area / prior_area if prior_area else np.nan,
            "accepted_area_fraction_of_prior": area / prior_area if prior_area else np.nan,
            "candidate_components": cand_components,
            "candidate_largest_component_fraction": cand_largest_frac,
            "accepted_components": n_comp,
            "accepted_largest_component_fraction": largest_frac,
            "candidate_centroid_x": cand_x,
            "candidate_centroid_y": cand_y,
            "accepted_centroid_x": cx,
            "accepted_centroid_y": cy,
            "candidate_centroid_jump_from_history_px": jump,
            "fallback_used": int(fallback),
            "fallback_reason": ";".join(reasons),
            "low_area_flag": int(low_area),
            "high_area_flag": int(high_area),
            "fragmented_flag": int(fragmented),
            "centroid_jump_flag": int(jumpy),
        })

    frame_df = pd.DataFrame(rows)
    accepted_area = pd.to_numeric(frame_df["accepted_area_px"], errors="coerce")
    cand_area = pd.to_numeric(frame_df["candidate_area_px"], errors="coerce")
    cx = pd.to_numeric(frame_df["accepted_centroid_x"], errors="coerce").to_numpy()
    cy = pd.to_numeric(frame_df["accepted_centroid_y"], errors="coerce").to_numpy()
    centroid_step = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    accepted_cv = float(accepted_area.std() / accepted_area.mean()) if accepted_area.mean() else np.nan
    candidate_cv = float(cand_area.std() / cand_area.mean()) if cand_area.mean() else np.nan
    summary = {
        "prior_area_px": prior_area,
        "prior_area_fraction": prior_area / float(h * w),
        "fallback_frame_fraction": float(frame_df["fallback_used"].mean()),
        "low_area_fraction": float(frame_df["low_area_flag"].mean()),
        "high_area_fraction": float(frame_df["high_area_flag"].mean()),
        "fragmented_fraction": float(frame_df["fragmented_flag"].mean()),
        "centroid_jump_fraction": float(frame_df["centroid_jump_flag"].mean()),
        "candidate_area_cv": candidate_cv,
        "accepted_area_cv": accepted_cv,
        "accepted_area_fraction_median": float((accepted_area / prior_area).median()) if prior_area else np.nan,
        "accepted_area_fraction_iqr": float((accepted_area / prior_area).quantile(0.75) - (accepted_area / prior_area).quantile(0.25)) if prior_area else np.nan,
        "accepted_centroid_path_px": float(np.nansum(centroid_step)),
        "accepted_centroid_max_step_px": float(np.nanmax(centroid_step)) if centroid_step.size else np.nan,
        "candidate_centroid_max_jump_px": float(pd.to_numeric(frame_df["candidate_centroid_jump_from_history_px"], errors="coerce").max()),
    }
    summary["mask_instability_score"] = float(
        np.nansum([
            summary["fallback_frame_fraction"],
            summary["fragmented_fraction"],
            summary["low_area_fraction"],
            summary["high_area_fraction"],
            min(summary["accepted_area_cv"], 1.0) if np.isfinite(summary["accepted_area_cv"]) else np.nan,
            min(summary["accepted_centroid_path_px"] / max(min(h, w), 1), 2.0),
        ])
    )
    return frame_df, summary


def add_optional_tables(df: pd.DataFrame, mobility_path: Path, rollout_path: Path) -> pd.DataFrame:
    out = df.copy()
    if mobility_path.exists():
        mobility = pd.read_csv(mobility_path)
        keep = [c for c in [
            "roi_id",
            "cycleNo",
            "front_quality_score",
            "front_radius_slope_r2",
            "apparent_diffusion_proxy_ds_px2_per_frame",
            "high_fraction_slope_per_s",
            "first_last_corr",
            "cumulative_abs_first_last",
            "evidence_score",
            "mean_drop_frac",
        ] if c in mobility.columns]
        if {"roi_id", "cycleNo"}.issubset(keep):
            out = out.merge(mobility[keep].drop_duplicates(["roi_id", "cycleNo"]), how="left", on=["roi_id", "cycleNo"])
    if rollout_path.exists():
        rollout = pd.read_csv(rollout_path)
        if {"roi_id", "cycleNo", "method"}.issubset(rollout.columns):
            piv = rollout.pivot_table(
                index=["roi_id", "cycleNo"],
                columns="method",
                values=[c for c in ["q90_undercoverage_rate", "mae_mean", "mae_max"] if c in rollout.columns],
                aggfunc="first",
            )
            piv.columns = [f"{metric}_{method}" for metric, method in piv.columns.to_flat_index()]
            out = out.merge(piv.reset_index(), how="left", on=["roi_id", "cycleNo"])
    return out


def cohort_role_from_row(row: pd.Series) -> str:
    label = str(row.get("validation_label", "")).lower()
    if "event" in label:
        return "event"
    if "control" in label:
        return "control"
    role = str(row.get("cohort_role", "")).lower()
    if role in {"event", "control"}:
        return role
    return ""


def event_control_tests(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "fallback_frame_fraction",
        "low_area_fraction",
        "high_area_fraction",
        "fragmented_fraction",
        "centroid_jump_fraction",
        "candidate_area_cv",
        "accepted_area_cv",
        "accepted_area_fraction_iqr",
        "accepted_centroid_path_px",
        "accepted_centroid_max_step_px",
        "mask_instability_score",
    ]
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        event = df[df["cohort_role"] == "event"][feature]
        control = df[df["cohort_role"] == "control"][feature]
        row = safe_mwu(event, control)
        row.update({"feature": feature, "contrast": "event_minus_control"})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    x_cols = [
        "fallback_frame_fraction",
        "fragmented_fraction",
        "accepted_area_cv",
        "accepted_area_fraction_iqr",
        "accepted_centroid_path_px",
        "mask_instability_score",
    ]
    y_cols = [c for c in df.columns if c.startswith("q90_undercoverage_rate_") or c in [
        "front_quality_score",
        "front_radius_slope_r2",
        "high_fraction_slope_per_s",
        "first_last_corr",
        "cumulative_abs_first_last",
        "evidence_score",
        "mean_drop_frac",
    ]]
    rows = []
    for x in x_cols:
        for y in y_cols:
            tmp = df[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(tmp) < 6 or tmp[x].nunique() < 2 or tmp[y].nunique() < 2:
                continue
            rho, p = spearmanr(tmp[x], tmp[y])
            rows.append({
                "x": x,
                "y": y,
                "n": int(len(tmp)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(rows).sort_values("p_value", na_position="last") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_sequences/selected_roi_sequence_manifest.csv")
    parser.add_argument("--mobility", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_mobility/multi_cycle_roi_mobility_descriptors.csv")
    parser.add_argument("--rollout-calibration", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/probabilistic_rollout_calibration/probabilistic_rollout_roi_table.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_mask_stability_audit")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)

    roi_rows = []
    frame_rows = []
    for _, row in manifest.iterrows():
        roi_id = str(row["roi_id"])
        npz_path = Path(str(row["npz_path"]))
        if not npz_path.exists():
            continue
        with np.load(npz_path) as data:
            frames = np.asarray(data["frames_norm"], dtype=np.float32)
            frame_indices = np.asarray(data["frame_indices"], dtype=int) if "frame_indices" in data.files else np.arange(frames.shape[0])
        frame_df, summary = trace_masks(frames)
        frame_df.insert(0, "roi_id", roi_id)
        frame_df.insert(1, "cycleNo", finite_float(row.get("cycleNo")))
        if len(frame_indices) == len(frame_df):
            frame_df.insert(2, "frame_index", frame_indices)
        frame_rows.append(frame_df)
        roi_summary = {
            "roi_id": roi_id,
            "cycleNo": finite_float(row.get("cycleNo")),
            "cohort_role": cohort_role_from_row(row),
            "event_reference_cycle": finite_float(row.get("event_reference_cycle"), finite_float(row.get("cycleNo")) if cohort_role_from_row(row) == "event" else np.nan),
            "n_frames": int(frames.shape[0]),
            "height": int(frames.shape[1]),
            "width": int(frames.shape[2]),
            "stage_drift_xy_sampled": finite_float(row.get("stage_drift_xy_sampled")),
            "validation_score": finite_float(row.get("validation_score")),
            "validation_label": row.get("validation_label", ""),
            **summary,
        }
        roi_rows.append(roi_summary)

    per_roi = pd.DataFrame(roi_rows)
    if per_roi.empty:
        raise RuntimeError(f"No ROI NPZ tensors could be read from {manifest_path}")
    per_roi = add_optional_tables(per_roi, Path(args.mobility), Path(args.rollout_calibration))
    frame_summary = pd.concat(frame_rows, ignore_index=True) if frame_rows else pd.DataFrame()
    tests = event_control_tests(per_roi)
    corr = correlation_table(per_roi)

    per_roi.to_csv(out_dir / "particle_mask_stability_per_roi.csv", index=False)
    frame_summary.to_csv(out_dir / "particle_mask_stability_frame_summary.csv", index=False)
    tests.to_csv(out_dir / "particle_mask_stability_event_control_tests.csv", index=False)
    corr.to_csv(out_dir / "particle_mask_stability_correlations.csv", index=False)

    role_summary = (
        per_roi.groupby("cohort_role", dropna=False)
        .agg({
            "roi_id": "count",
            "fallback_frame_fraction": "median",
            "accepted_area_cv": "median",
            "accepted_centroid_path_px": "median",
            "mask_instability_score": "median",
        })
        .rename(columns={"roi_id": "n_roi"})
        .reset_index()
    )

    summary = {
        "n_roi": int(len(per_roi)),
        "n_frames_total": int(per_roi["n_frames"].sum()),
        "manifest": str(manifest_path),
        "method": {
            "candidate_mask": "per-frame local contrast plus temporal-std evidence inside a sequence-level central prior",
            "fallback": "previous accepted mask when area, fragmentation, or centroid jump violates rolling-history constraints",
            "interpretation": "guardrail audit of particle-region mask stability, not manual segmentation",
        },
        "overall": {
            "median_fallback_frame_fraction": float(per_roi["fallback_frame_fraction"].median()),
            "median_accepted_area_cv": float(per_roi["accepted_area_cv"].median()),
            "median_centroid_path_px": float(per_roi["accepted_centroid_path_px"].median()),
            "median_mask_instability_score": float(per_roi["mask_instability_score"].median()),
        },
        "role_summary": role_summary.to_dict("records"),
        "top_event_control_tests": tests.head(12).to_dict("records"),
        "top_correlations": corr.head(12).to_dict("records") if not corr.empty else [],
        "highest_instability_rois": per_roi.sort_values("mask_instability_score", ascending=False).head(12).to_dict("records"),
    }
    with (out_dir / "particle_mask_stability_audit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "README.md").open("w") as f:
        f.write("# Particle Mask Stability Audit\n\n")
        f.write("This folder audits a history-aware particle-mask guardrail on the existing ROI-only tensors.\n\n")
        f.write("The candidate mask is built from frame-local contrast plus temporal-standard-deviation evidence inside a sequence-level central prior. ")
        f.write("Frames with implausible area, fragmentation, or centroid jumps fall back to the previous accepted mask.\n\n")
        f.write("Outputs:\n\n")
        f.write("- `particle_mask_stability_per_roi.csv`: ROI-level fallback and stability metrics.\n")
        f.write("- `particle_mask_stability_frame_summary.csv`: per-frame candidate and accepted mask diagnostics.\n")
        f.write("- `particle_mask_stability_event_control_tests.csv`: event/control Mann-Whitney tests.\n")
        f.write("- `particle_mask_stability_correlations.csv`: Spearman correlations with rollout/front descriptors.\n")
        f.write("- `particle_mask_stability_audit_summary.json`: compact project summary.\n")

    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
