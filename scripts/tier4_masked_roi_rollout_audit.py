#!/usr/bin/env python3
"""Evaluate ROI rollout baselines inside history-aware particle masks.

Earlier rollout baselines used particle-centered crops, but every crop pixel
contributed equally to MSE/MAE. This audit keeps the same simple rollout
models, then scores held-out next-frame errors inside the accepted particle
support from the mask-stability guardrail and compares them with non-particle
crop pixels.

This is a particle-region scoring audit, not a new production video model.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.utils.extmath import randomized_svd

from tier4_particle_mask_stability_audit import (
    candidate_mask,
    central_ellipse,
    cohort_role_from_row,
    component_features,
    finite_float,
    build_prior_mask,
)


def safe_mwu(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    aa = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    bb = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if aa.empty or bb.empty:
        return {"n_event": int(len(aa)), "n_control": int(len(bb)), "median_event": np.nan, "median_control": np.nan, "median_diff_event_minus_control": np.nan, "p_value": np.nan}
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


def linear_fit(x_train: np.ndarray, y_train: np.ndarray, rank: int, ridge: float) -> Dict[str, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    x0 = x_train - mean
    k = int(min(rank, x0.shape[0] - 1, x0.shape[1]))
    _, s, vt = randomized_svd(x0, n_components=k, n_iter=4, random_state=29)
    basis = vt[:k].T.astype(np.float32)
    zx = (x_train - mean) @ basis
    zy = (y_train - mean) @ basis
    lhs = zx.T @ zx + ridge * np.eye(k, dtype=np.float32)
    rhs = zx.T @ zy
    a = np.linalg.solve(lhs, rhs).astype(np.float32)
    eig = np.linalg.eigvals(a)
    return {
        "mean": mean.astype(np.float32),
        "basis": basis,
        "a": a,
        "singular_values": s[:k].astype(np.float32),
        "spectral_radius": np.array([float(np.max(np.abs(eig)))], dtype=np.float32),
    }


def decode(z: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    return np.clip(model["mean"] + z @ model["basis"].T, 0, 1)


def accepted_mask_stack(frames: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    t, h, w = frames.shape
    prior = build_prior_mask(frames)
    temporal_std = np.nanstd(frames, axis=0)
    previous = prior.copy()
    accepted_areas: List[float] = []
    accepted_centroids: List[Tuple[float, float]] = []
    masks = []
    rows = []
    max_jump = 0.23 * min(h, w)
    prior_area = float(prior.sum())

    for idx, frame in enumerate(frames):
        cand = candidate_mask(frame, temporal_std, prior)
        cand_area, cand_x, cand_y, cand_components, cand_largest_frac = component_features(cand)
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
        fallback = bool(cand_area <= 0 or low_area or high_area or fragmented or jumpy)
        accepted = previous.copy() if fallback else cand
        area, cx, cy, n_comp, largest_frac = component_features(accepted)
        previous = accepted
        accepted_areas.append(area)
        accepted_centroids.append((cx, cy))
        masks.append(accepted.astype(bool))
        rows.append({
            "frame_local_index": idx,
            "accepted_area_px": area,
            "accepted_area_fraction": area / float(h * w),
            "fallback_used": int(fallback),
            "candidate_centroid_jump_from_history_px": jump,
            "accepted_centroid_x": cx,
            "accepted_centroid_y": cy,
        })
    return np.stack(masks, axis=0), pd.DataFrame(rows)


def masked_metrics(pred: np.ndarray, truth: np.ndarray, particle_mask: np.ndarray, context_mask: np.ndarray) -> Dict[str, float]:
    diff = pred - truth
    particle = particle_mask & np.isfinite(diff)
    context = context_mask & ~particle_mask & np.isfinite(diff)
    full = np.isfinite(diff)

    def mse(mask: np.ndarray) -> float:
        return float(np.mean(diff[mask] ** 2)) if mask.any() else np.nan

    def mae(mask: np.ndarray) -> float:
        return float(np.mean(np.abs(diff[mask]))) if mask.any() else np.nan

    particle_mse = mse(particle)
    nonparticle_mse = mse(context)
    full_mse = mse(full)
    return {
        "particle_mse": particle_mse,
        "particle_mae": mae(particle),
        "nonparticle_mse": nonparticle_mse,
        "nonparticle_mae": mae(context),
        "full_crop_mse": full_mse,
        "full_crop_mae": mae(full),
        "particle_to_nonparticle_mse_ratio": particle_mse / nonparticle_mse if np.isfinite(particle_mse) and np.isfinite(nonparticle_mse) and nonparticle_mse > 0 else np.nan,
        "particle_mse_fraction_of_full": particle_mse / full_mse if np.isfinite(particle_mse) and np.isfinite(full_mse) and full_mse > 0 else np.nan,
        "particle_mask_area_fraction": float(particle.mean()),
        "nonparticle_context_area_fraction": float(context.mean()),
    }


def load_sequences(manifest: pd.DataFrame) -> List[Dict[str, object]]:
    seqs = []
    for _, row in manifest.iterrows():
        npz_path = Path(str(row["npz_path"]))
        if not npz_path.exists():
            continue
        with np.load(npz_path) as data:
            frames = np.asarray(data["frames_norm"], dtype=np.float32)
            frame_indices = np.asarray(data["frame_indices"], dtype=int) if "frame_indices" in data.files else np.arange(frames.shape[0])
        masks, mask_rows = accepted_mask_stack(frames)
        context = central_ellipse(frames.shape[1], frames.shape[2], 0.49)
        seqs.append({
            "row": row,
            "frames": frames,
            "flat": frames.reshape(frames.shape[0], -1),
            "masks": masks,
            "mask_rows": mask_rows,
            "context_mask": context,
            "frame_indices": frame_indices,
        })
    return seqs


def split_pairs(seqs: List[Dict[str, object]], train_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for seq in seqs:
        frames = seq["flat"]
        split = max(3, min(frames.shape[0] - 2, int(frames.shape[0] * train_fraction)))
        xs.append(frames[:split - 1])
        ys.append(frames[1:split])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def evaluate_sequence(seq: Dict[str, object], model: Dict[str, np.ndarray], train_fraction: float) -> pd.DataFrame:
    row = seq["row"]
    frames = seq["frames"]
    flat = seq["flat"]
    masks = seq["masks"]
    context = seq["context_mask"]
    frame_indices = seq["frame_indices"]
    split = max(3, min(flat.shape[0] - 2, int(flat.shape[0] * train_fraction)))
    basis = model["basis"]
    mean = model["mean"]
    a = model["a"]
    z = (flat[split - 1:split] - mean) @ basis
    prev = flat[split - 2]
    cur = flat[split - 1]
    rows = []
    for t in range(split, flat.shape[0]):
        truth = frames[t]
        z = z @ a
        preds = {
            "persistence": cur.reshape(truth.shape),
            "velocity": np.clip(cur + (cur - prev), 0, 1).reshape(truth.shape),
            "low_rank_dmd": decode(z, model)[0].reshape(truth.shape),
        }
        for method, pred in preds.items():
            out = masked_metrics(pred, truth, masks[t], context)
            out.update({
                "roi_id": row["roi_id"],
                "cycleNo": finite_float(row.get("cycleNo")),
                "cohort_role": cohort_role_from_row(row),
                "event_reference_cycle": finite_float(row.get("event_reference_cycle"), finite_float(row.get("cycleNo")) if cohort_role_from_row(row) == "event" else np.nan),
                "method": method,
                "eval_step": int(t - split),
                "frame_index": int(frame_indices[t]) if t < len(frame_indices) else int(t),
                "validation_score": finite_float(row.get("validation_score")),
                "mask_fallback_used": int(seq["mask_rows"].iloc[t]["fallback_used"]),
                "accepted_area_fraction": finite_float(seq["mask_rows"].iloc[t]["accepted_area_fraction"]),
            })
            rows.append(out)
        prev, cur = cur, flat[t]
    return pd.DataFrame(rows)


def method_ratio_table(per_roi: pd.DataFrame) -> pd.DataFrame:
    tmp = per_roi.copy()
    tmp["event_reference_cycle_filled"] = pd.to_numeric(tmp["event_reference_cycle"], errors="coerce").fillna(pd.to_numeric(tmp["cycleNo"], errors="coerce"))
    piv = tmp.pivot_table(index=["roi_id", "cycleNo", "cohort_role", "event_reference_cycle_filled"], columns="method", values="particle_mse_mean", aggfunc="first")
    rows = []
    for idx, row in piv.iterrows():
        roi_id, cycle, role, ref = idx
        base = row.get("persistence", np.nan)
        for method in ["velocity", "low_rank_dmd"]:
            val = row.get(method, np.nan)
            rows.append({
                "roi_id": roi_id,
                "cycleNo": cycle,
                "cohort_role": role,
                "event_reference_cycle": ref,
                "method": method,
                "particle_mse_ratio_vs_persistence": val / base if np.isfinite(val) and np.isfinite(base) and base > 0 else np.nan,
                "particle_mse_delta_vs_persistence": val - base if np.isfinite(val) and np.isfinite(base) else np.nan,
            })
    return pd.DataFrame(rows)


def event_control_tests(per_roi: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = [
        "particle_mse_mean",
        "particle_mae_mean",
        "nonparticle_mse_mean",
        "particle_to_nonparticle_mse_ratio_mean",
        "particle_mse_fraction_of_full_mean",
        "mask_fallback_used_mean",
    ]
    for method, grp in per_roi.groupby("method"):
        for feature in features:
            if feature not in grp.columns:
                continue
            out = safe_mwu(grp[grp["cohort_role"] == "event"][feature], grp[grp["cohort_role"] == "control"][feature])
            out.update({"method": method, "feature": feature, "contrast": "event_minus_control"})
            rows.append(out)
    for method, grp in ratios.groupby("method"):
        out = safe_mwu(grp[grp["cohort_role"] == "event"]["particle_mse_ratio_vs_persistence"], grp[grp["cohort_role"] == "control"]["particle_mse_ratio_vs_persistence"])
        out.update({"method": method, "feature": "particle_mse_ratio_vs_persistence", "contrast": "event_minus_control"})
        rows.append(out)
    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


def correlation_table(per_roi: pd.DataFrame, ratios: pd.DataFrame, mask_stability_path: Path) -> pd.DataFrame:
    joined = per_roi.copy()
    if mask_stability_path.exists():
        mask_df = pd.read_csv(mask_stability_path)
        keep = [c for c in ["roi_id", "cycleNo", "fallback_frame_fraction", "accepted_area_cv", "accepted_centroid_path_px", "mask_instability_score", "front_quality_score", "high_fraction_slope_per_s", "first_last_corr", "cumulative_abs_first_last"] if c in mask_df.columns]
        joined = joined.merge(mask_df[keep].drop_duplicates(["roi_id", "cycleNo"]), how="left", on=["roi_id", "cycleNo"])
    ratio_wide = ratios.pivot_table(index=["roi_id", "cycleNo"], columns="method", values="particle_mse_ratio_vs_persistence", aggfunc="first")
    ratio_wide.columns = [f"particle_mse_ratio_vs_persistence_{c}" for c in ratio_wide.columns]
    joined = joined.merge(ratio_wide.reset_index(), how="left", on=["roi_id", "cycleNo"])

    x_cols = [c for c in [
        "particle_mse_mean",
        "particle_to_nonparticle_mse_ratio_mean",
        "particle_mse_fraction_of_full_mean",
        "particle_mse_ratio_vs_persistence_low_rank_dmd",
        "particle_mse_ratio_vs_persistence_velocity",
    ] if c in joined.columns]
    y_cols = [c for c in [
        "fallback_frame_fraction",
        "accepted_area_cv",
        "accepted_centroid_path_px",
        "mask_instability_score",
        "front_quality_score",
        "high_fraction_slope_per_s",
        "first_last_corr",
        "cumulative_abs_first_last",
    ] if c in joined.columns]
    rows = []
    for method, grp in joined.groupby("method"):
        for x in x_cols:
            for y in y_cols:
                tmp = grp[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(tmp) < 6 or tmp[x].nunique() < 2 or tmp[y].nunique() < 2:
                    continue
                rho, p = spearmanr(tmp[x], tmp[y])
                rows.append({"method": method, "x": x, "y": y, "n": int(len(tmp)), "spearman_rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("p_value", na_position="last") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/scratch/<account>/<username>/Alek_Jiho/derived/multi_cycle_roi_sequences/selected_roi_sequence_manifest.csv")
    parser.add_argument("--mask-stability", default="/scratch/<account>/<username>/Alek_Jiho/derived/particle_mask_stability_audit/particle_mask_stability_per_roi.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/masked_roi_rollout_audit")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    seqs = load_sequences(manifest)
    if not seqs:
        raise RuntimeError(f"No readable ROI tensors from {args.manifest}")
    x_train, y_train = split_pairs(seqs, args.train_fraction)
    model = linear_fit(x_train, y_train, args.rank, args.ridge)
    frame_tables = [evaluate_sequence(seq, model, args.train_fraction) for seq in seqs]
    frames = pd.concat(frame_tables, ignore_index=True)
    per_roi = frames.groupby(["roi_id", "cycleNo", "cohort_role", "event_reference_cycle", "method"], dropna=False).agg({
        "particle_mse": ["mean", "median", "max"],
        "particle_mae": ["mean", "median"],
        "nonparticle_mse": ["mean", "median"],
        "nonparticle_mae": ["mean", "median"],
        "full_crop_mse": ["mean", "median"],
        "particle_to_nonparticle_mse_ratio": ["mean", "median"],
        "particle_mse_fraction_of_full": ["mean", "median"],
        "particle_mask_area_fraction": "mean",
        "nonparticle_context_area_fraction": "mean",
        "mask_fallback_used": "mean",
        "accepted_area_fraction": "median",
        "validation_score": "first",
    }).reset_index()
    per_roi.columns = ["_".join(c).strip("_") for c in per_roi.columns.to_flat_index()]
    ratios = method_ratio_table(per_roi)
    tests = event_control_tests(per_roi, ratios)
    corr = correlation_table(per_roi, ratios, Path(args.mask_stability))
    cycle_method = per_roi.groupby(["cycleNo", "method"], as_index=False).agg({
        "particle_mse_mean": ["mean", "median"],
        "nonparticle_mse_mean": ["mean", "median"],
        "particle_to_nonparticle_mse_ratio_mean": ["mean", "median"],
        "mask_fallback_used_mean": "mean",
    })
    cycle_method.columns = ["_".join(c).strip("_") for c in cycle_method.columns.to_flat_index()]

    frames.to_csv(out / "masked_roi_rollout_frame_metrics.csv", index=False)
    per_roi.to_csv(out / "masked_roi_rollout_per_roi_metrics.csv", index=False)
    ratios.to_csv(out / "masked_roi_rollout_method_ratios.csv", index=False)
    tests.to_csv(out / "masked_roi_rollout_event_control_tests.csv", index=False)
    corr.to_csv(out / "masked_roi_rollout_correlations.csv", index=False)
    cycle_method.to_csv(out / "masked_roi_rollout_cycle_method_summary.csv", index=False)

    method_summary = per_roi.groupby("method").agg({
        "particle_mse_mean": ["median", "mean"],
        "nonparticle_mse_mean": ["median", "mean"],
        "particle_to_nonparticle_mse_ratio_mean": "median",
        "particle_mse_fraction_of_full_mean": "median",
        "mask_fallback_used_mean": "median",
    }).reset_index()
    method_summary.columns = ["_".join(c).strip("_") for c in method_summary.columns.to_flat_index()]
    method_summary = method_summary.sort_values("particle_mse_mean_median")

    best_by_roi = per_roi.sort_values("particle_mse_mean").groupby(["roi_id", "cycleNo"], as_index=False).first()
    summary = {
        "manifest": str(args.manifest),
        "n_roi": int(len(seqs)),
        "n_frame_metric_rows": int(len(frames)),
        "rank": int(args.rank),
        "train_fraction": float(args.train_fraction),
        "dmd_spectral_radius": float(model["spectral_radius"][0]),
        "method_summary": method_summary.to_dict("records"),
        "best_method_counts_inside_particle": best_by_roi["method"].value_counts().to_dict(),
        "top_event_control_tests": tests.head(18).to_dict("records"),
        "top_correlations": corr.head(18).to_dict("records") if not corr.empty else [],
        "top_particle_difficulty_rois": per_roi.sort_values("particle_mse_mean", ascending=False).head(12).to_dict("records"),
        "guardrail": "Held-out rollout errors are scored inside automatic history-aware particle masks; this is not manual segmentation or a new learned video model.",
    }
    with (out / "masked_roi_rollout_audit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out / "README.md").open("w") as f:
        f.write("# Masked ROI Rollout Audit\n\n")
        f.write("This audit reruns the existing simple rollout baseline scoring inside the accepted particle support from the history-aware mask guardrail.\n\n")
        f.write(f"- ROI tensors: {summary['n_roi']}\n")
        f.write(f"- Frame metric rows: {summary['n_frame_metric_rows']}\n")
        f.write(f"- Best-method counts inside particle masks: {summary['best_method_counts_inside_particle']}\n\n")
        f.write("Outputs:\n\n")
        f.write("- `masked_roi_rollout_frame_metrics.csv`: held-out frame metrics by ROI, method, and mask region.\n")
        f.write("- `masked_roi_rollout_per_roi_metrics.csv`: ROI-method aggregates.\n")
        f.write("- `masked_roi_rollout_method_ratios.csv`: particle-MSE ratios versus persistence.\n")
        f.write("- `masked_roi_rollout_event_control_tests.csv`: event/control tests on masked errors.\n")
        f.write("- `masked_roi_rollout_correlations.csv`: links between masked rollout errors and mask/front descriptors.\n")
        f.write("- `masked_roi_rollout_audit_summary.json`: compact summary.\n")

    print(json.dumps({
        "n_roi": summary["n_roi"],
        "best_method_counts_inside_particle": summary["best_method_counts_inside_particle"],
        "method_summary": summary["method_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
