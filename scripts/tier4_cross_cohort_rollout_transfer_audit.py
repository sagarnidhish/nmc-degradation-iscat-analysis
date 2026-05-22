#!/usr/bin/env python3
"""Cross-cohort masked rollout transfer audit for NMC ROI sequences.

The existing masked rollout audits fit and score a low-rank linear dynamics
model within one ROI cohort. This audit asks a stricter question: does a model
fit on one video-backed cohort transfer to another cohort, or are the late
transfer-ranked warning crops dynamically out of domain?

This is an interpretable generalization audit, not a production video model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from tier4_masked_roi_rollout_audit import (
    accepted_mask_stack,
    central_ellipse,
    finite_float,
    linear_fit,
    masked_metrics,
)


def safe_median(values: Iterable[float]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(series.median()) if not series.empty else np.nan


def safe_mwu(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    aa = pd.to_numeric(pd.Series(a), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    bb = pd.to_numeric(pd.Series(b), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "n_a": int(len(aa)),
        "n_b": int(len(bb)),
        "median_a": float(aa.median()) if not aa.empty else np.nan,
        "median_b": float(bb.median()) if not bb.empty else np.nan,
        "median_diff_a_minus_b": float(aa.median() - bb.median()) if not aa.empty and not bb.empty else np.nan,
        "p_value": np.nan,
    }
    if not aa.empty and not bb.empty:
        try:
            _, p = mannwhitneyu(aa, bb, alternative="two-sided")
            out["p_value"] = float(p)
        except Exception:
            pass
    return out


def read_manifest(path: Path, cohort_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sequence_cohort"] = cohort_name
    return df


def load_sequences(manifest: pd.DataFrame) -> List[Dict[str, object]]:
    seqs: List[Dict[str, object]] = []
    for _, row in manifest.iterrows():
        npz_path = Path(str(row["npz_path"]))
        if not npz_path.exists():
            continue
        with np.load(npz_path) as data:
            frames = np.asarray(data["frames_norm"], dtype=np.float32)
            frame_indices = np.asarray(data["frame_indices"], dtype=int) if "frame_indices" in data.files else np.arange(frames.shape[0])
        masks, mask_rows = accepted_mask_stack(frames)
        seqs.append({
            "row": row,
            "frames": frames,
            "flat": frames.reshape(frames.shape[0], -1),
            "masks": masks,
            "mask_rows": mask_rows,
            "context_mask": central_ellipse(frames.shape[1], frames.shape[2], 0.49),
            "frame_indices": frame_indices,
        })
    return seqs


def split_pairs(seqs: List[Dict[str, object]], train_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for seq in seqs:
        flat = seq["flat"]
        split = max(3, min(flat.shape[0] - 2, int(flat.shape[0] * train_fraction)))
        xs.append(flat[: split - 1])
        ys.append(flat[1:split])
    if not xs:
        raise RuntimeError("No readable training sequences")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def evaluate_model(
    seqs: List[Dict[str, object]],
    model: Dict[str, np.ndarray],
    model_name: str,
    train_fraction: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    basis = model["basis"]
    mean = model["mean"]
    a = model["a"]
    for seq in seqs:
        row = seq["row"]
        frames = seq["frames"]
        flat = seq["flat"]
        split = max(3, min(flat.shape[0] - 2, int(flat.shape[0] * train_fraction)))
        z = (flat[split - 1:split] - mean) @ basis
        prev = flat[split - 2]
        cur = flat[split - 1]
        for t in range(split, flat.shape[0]):
            truth = frames[t]
            z = z @ a
            preds = {
                "persistence": cur.reshape(truth.shape),
                "velocity": np.clip(cur + (cur - prev), 0, 1).reshape(truth.shape),
                "low_rank_dmd": np.clip(mean + z @ basis.T, 0, 1)[0].reshape(truth.shape),
            }
            for method, pred in preds.items():
                out = masked_metrics(pred, truth, seq["masks"][t], seq["context_mask"])
                out.update({
                    "model_name": model_name,
                    "eval_cohort": row["sequence_cohort"],
                    "roi_id": row["roi_id"],
                    "cycleNo": finite_float(row.get("cycleNo")),
                    "source_stem": row.get("source_stem"),
                    "validation_label": row.get("validation_label"),
                    "validation_score": finite_float(row.get("validation_score")),
                    "method": method,
                    "eval_step": int(t - split),
                    "frame_index": int(seq["frame_indices"][t]) if t < len(seq["frame_indices"]) else int(t),
                    "mask_fallback_used": int(seq["mask_rows"].iloc[t]["fallback_used"]),
                    "accepted_area_fraction": finite_float(seq["mask_rows"].iloc[t]["accepted_area_fraction"]),
                })
                rows.append(out)
            prev, cur = cur, flat[t]
    return pd.DataFrame(rows)


def aggregate_per_roi(frames: pd.DataFrame) -> pd.DataFrame:
    per_roi = frames.groupby(["model_name", "eval_cohort", "roi_id", "cycleNo", "method"], dropna=False).agg({
        "particle_mse": ["mean", "median", "max"],
        "particle_mae": ["mean", "median"],
        "nonparticle_mse": ["mean", "median"],
        "particle_to_nonparticle_mse_ratio": ["mean", "median"],
        "particle_mse_fraction_of_full": ["mean", "median"],
        "mask_fallback_used": "mean",
        "accepted_area_fraction": "median",
        "validation_score": "first",
    }).reset_index()
    per_roi.columns = ["_".join(c).strip("_") for c in per_roi.columns.to_flat_index()]
    return per_roi


def add_ratios(per_roi: pd.DataFrame) -> pd.DataFrame:
    pivot = per_roi.pivot_table(
        index=["model_name", "eval_cohort", "roi_id", "cycleNo"],
        columns="method",
        values="particle_mse_mean",
        aggfunc="first",
    ).reset_index()
    pivot["dmd_particle_mse_ratio_vs_persistence"] = pivot["low_rank_dmd"] / pivot["persistence"]
    pivot["velocity_particle_mse_ratio_vs_persistence"] = pivot["velocity"] / pivot["persistence"]
    return per_roi.merge(
        pivot[["model_name", "eval_cohort", "roi_id", "cycleNo", "dmd_particle_mse_ratio_vs_persistence", "velocity_particle_mse_ratio_vs_persistence"]],
        how="left",
        on=["model_name", "eval_cohort", "roi_id", "cycleNo"],
    )


def domain_shift_table(per_roi: pd.DataFrame) -> pd.DataFrame:
    dmd = per_roi[per_roi["method"] == "low_rank_dmd"].copy()
    rows = []
    for eval_cohort, grp in dmd.groupby("eval_cohort"):
        internal_name = f"{eval_cohort}_internal"
        internal = grp[grp["model_name"] == internal_name]
        for model_name, sub in grp.groupby("model_name"):
            out = {
                "eval_cohort": eval_cohort,
                "model_name": model_name,
                "n_roi": int(sub["roi_id"].nunique()),
                "median_particle_mse": safe_median(sub["particle_mse_mean"]),
                "median_particle_to_nonparticle_ratio": safe_median(sub["particle_to_nonparticle_mse_ratio_mean"]),
                "median_dmd_ratio_vs_persistence": safe_median(sub["dmd_particle_mse_ratio_vs_persistence"]),
                "median_mask_fallback": safe_median(sub["mask_fallback_used_mean"]),
            }
            if not internal.empty and model_name != internal_name:
                cmp = safe_mwu(sub["particle_mse_mean"], internal["particle_mse_mean"])
                out.update({
                    "internal_baseline_model": internal_name,
                    "mwu_p_vs_internal": cmp["p_value"],
                    "median_particle_mse_shift_vs_internal": cmp["median_diff_a_minus_b"],
                    "median_particle_mse_ratio_vs_internal": out["median_particle_mse"] / safe_median(internal["particle_mse_mean"]),
                })
            else:
                out.update({
                    "internal_baseline_model": internal_name,
                    "mwu_p_vs_internal": np.nan,
                    "median_particle_mse_shift_vs_internal": np.nan,
                    "median_particle_mse_ratio_vs_internal": np.nan,
                })
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["eval_cohort", "median_particle_mse"])


def correlation_table(per_roi: pd.DataFrame) -> pd.DataFrame:
    dmd = per_roi[per_roi["method"] == "low_rank_dmd"].copy()
    rows = []
    for (model_name, eval_cohort), grp in dmd.groupby(["model_name", "eval_cohort"]):
        for x, y in [
            ("validation_score_first", "particle_mse_mean"),
            ("validation_score_first", "dmd_particle_mse_ratio_vs_persistence"),
            ("accepted_area_fraction_median", "particle_mse_mean"),
            ("mask_fallback_used_mean", "particle_mse_mean"),
            ("cycleNo", "particle_mse_mean"),
        ]:
            tmp = grp[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(tmp) < 6 or tmp[x].nunique() < 2 or tmp[y].nunique() < 2:
                continue
            rho, p = spearmanr(tmp[x], tmp[y])
            rows.append({
                "model_name": model_name,
                "eval_cohort": eval_cohort,
                "x": x,
                "y": y,
                "n": int(len(tmp)),
                "spearman_rho": float(rho),
                "p_value": float(p),
            })
    return pd.DataFrame(rows).sort_values("p_value", na_position="last") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-manifest", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/selected_roi_sequences/selected_roi_sequence_manifest.csv")
    parser.add_argument("--transfer-manifest", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_roi_sequences/selected_roi_sequence_manifest.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cross_cohort_rollout_transfer_audit")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    selected_manifest = read_manifest(Path(args.selected_manifest), "selected")
    transfer_manifest = read_manifest(Path(args.transfer_manifest), "transfer_ranked")
    selected = load_sequences(selected_manifest)
    transfer = load_sequences(transfer_manifest)
    if not selected or not transfer:
        raise RuntimeError("Both selected and transfer-ranked sequence cohorts must be readable")

    cohorts = {"selected": selected, "transfer_ranked": transfer}
    pooled = selected + transfer
    models = {}
    for name, seqs in [("selected_internal", selected), ("transfer_ranked_internal", transfer), ("pooled", pooled)]:
        x_train, y_train = split_pairs(seqs, args.train_fraction)
        models[name] = linear_fit(x_train, y_train, args.rank, args.ridge)

    frame_tables = []
    for model_name, model in models.items():
        for eval_name, seqs in cohorts.items():
            frame_tables.append(evaluate_model(seqs, model, model_name, args.train_fraction))
    frames = pd.concat(frame_tables, ignore_index=True)
    per_roi = add_ratios(aggregate_per_roi(frames))
    shift = domain_shift_table(per_roi)
    corr = correlation_table(per_roi)

    frames.to_csv(out / "cross_cohort_rollout_frame_metrics.csv", index=False)
    per_roi.to_csv(out / "cross_cohort_rollout_per_roi_metrics.csv", index=False)
    shift.to_csv(out / "cross_cohort_rollout_domain_shift.csv", index=False)
    corr.to_csv(out / "cross_cohort_rollout_correlations.csv", index=False)

    difficult = (
        per_roi[(per_roi["method"] == "low_rank_dmd") & (per_roi["eval_cohort"] == "transfer_ranked")]
        .sort_values("particle_mse_mean", ascending=False)
        .head(12)
    )
    summary = {
        "selected_manifest": str(args.selected_manifest),
        "transfer_manifest": str(args.transfer_manifest),
        "n_selected_roi": int(len(selected)),
        "n_transfer_ranked_roi": int(len(transfer)),
        "rank": int(args.rank),
        "train_fraction": float(args.train_fraction),
        "model_spectral_radius": {name: float(model["spectral_radius"][0]) for name, model in models.items()},
        "domain_shift": shift.to_dict("records"),
        "top_correlations": corr.head(18).to_dict("records") if not corr.empty else [],
        "top_transfer_ranked_difficult_rois": difficult.to_dict("records"),
        "guardrail": (
            "Cross-cohort low-rank rollout transfer compares interpretable linear dynamics across automatic ROI cohorts. "
            "It is evidence about video-domain generalization and difficult particle-local dynamics, not manual QC, calibrated diffusion, or a deployable learned video predictor."
        ),
    }
    with (out / "cross_cohort_rollout_transfer_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out / "README.md").open("w") as f:
        f.write("# Cross-Cohort Rollout Transfer Audit\n\n")
        f.write("Fits low-rank masked ROI rollout dynamics on selected, transfer-ranked, and pooled cohorts, then evaluates each model on both cohorts.\n\n")
        f.write(f"- Selected ROI sequences: {summary['n_selected_roi']}\n")
        f.write(f"- Transfer-ranked ROI sequences: {summary['n_transfer_ranked_roi']}\n")
        f.write(f"- Rank: {summary['rank']}\n\n")
        f.write("Outputs:\n\n")
        f.write("- `cross_cohort_rollout_frame_metrics.csv`: held-out frame metrics by model, cohort, ROI, and method.\n")
        f.write("- `cross_cohort_rollout_per_roi_metrics.csv`: ROI-method aggregates and DMD/velocity ratios versus persistence.\n")
        f.write("- `cross_cohort_rollout_domain_shift.csv`: model-transfer summaries versus within-cohort DMD baselines.\n")
        f.write("- `cross_cohort_rollout_correlations.csv`: links between transfer errors, cycle, validation score, and mask behavior.\n")
        f.write("- `cross_cohort_rollout_transfer_summary.json`: compact summary.\n")

    print(json.dumps({
        "n_selected_roi": summary["n_selected_roi"],
        "n_transfer_ranked_roi": summary["n_transfer_ranked_roi"],
        "domain_shift": summary["domain_shift"],
    }, indent=2))


if __name__ == "__main__":
    main()
