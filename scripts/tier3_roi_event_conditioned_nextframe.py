#!/usr/bin/env python3
"""Fast event-conditioned ROI next-frame model for selected NMC crops.

This script trains a PCA-latent ridge model on early frames of selected ROI
sequences. Inputs are the current latent frame, latent velocity, cycle identity,
time, and validation score; targets are next-frame latent residuals. It evaluates
held-out teacher-forced predictions and recursive rollouts against persistence
and velocity baselines, then exports residual descriptors for degradation mode
analysis.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from skimage.metrics import structural_similarity as ssim


def finite_float(x, default=np.nan) -> float:
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def load_sequences(manifest_path: str, model_stride: int) -> List[Dict[str, object]]:
    manifest = pd.read_csv(manifest_path)
    cycles = sorted(float(c) for c in manifest["cycleNo"].dropna().unique())
    cycle_code = {c: i / max(1, len(cycles) - 1) for i, c in enumerate(cycles)}
    seqs: List[Dict[str, object]] = []
    for _, row in manifest.iterrows():
        data = np.load(row["npz_path"])
        frames = data["frames_norm"].astype(np.float32)
        if model_stride > 1:
            frames = frames[:, ::model_stride, ::model_stride].copy()
        seqs.append({
            "row": row,
            "frames": frames,
            "flat": frames.reshape(frames.shape[0], -1),
            "frame_indices": data["frame_indices"].astype(int),
            "cycle_code": float(cycle_code[float(row["cycleNo"])]),
            "validation_score": finite_float(row.get("validation_score"), 0.0),
        })
    return seqs


def split_index(n_frames: int, train_fraction: float) -> int:
    return max(4, min(n_frames - 3, int(n_frames * train_fraction)))


def fit_latent_model(seqs: List[Dict[str, object]], train_fraction: float, rank: int, alpha: float) -> Tuple[PCA, Ridge, pd.DataFrame]:
    train_frames = []
    for seq in seqs:
        split = split_index(seq["flat"].shape[0], train_fraction)
        train_frames.append(seq["flat"][:split])
    x_frames = np.concatenate(train_frames, axis=0)
    n_components = int(min(rank, x_frames.shape[0] - 1, x_frames.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=29)
    pca.fit(x_frames)
    features, targets = [], []
    for seq in seqs:
        flat = seq["flat"]
        split = split_index(flat.shape[0], train_fraction)
        z = pca.transform(flat[:split])
        denom = max(1, flat.shape[0] - 1)
        vscore = seq["validation_score"] / 10.0
        for t in range(1, split - 1):
            feat = np.concatenate([
                z[t],
                z[t] - z[t - 1],
                np.array([seq["cycle_code"], t / denom, vscore], dtype=float),
            ])
            features.append(feat)
            targets.append(z[t + 1] - z[t])
    features_np = np.asarray(features, dtype=float)
    targets_np = np.asarray(targets, dtype=float)
    model = Ridge(alpha=alpha)
    model.fit(features_np, targets_np)
    train_pred = model.predict(features_np)
    train_latent_mse = float(np.mean((train_pred - targets_np) ** 2))
    meta = pd.DataFrame([{
        "n_train_pairs": int(len(features_np)),
        "rank": int(n_components),
        "alpha": float(alpha),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "train_latent_delta_mse": train_latent_mse,
    }])
    return pca, model, meta


def image_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    pred = np.clip(pred, 0, 1)
    truth = np.clip(truth, 0, 1)
    diff = pred - truth
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    psnr = float(10.0 * math.log10(1.0 / mse)) if mse > 0 else float("inf")
    try:
        sim = float(ssim(truth, pred, data_range=1.0))
    except Exception:
        sim = np.nan
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": sim}


def latent_feature(z_prev: np.ndarray, z_cur: np.ndarray, cycle_code: float, time_code: float, validation_score: float) -> np.ndarray:
    return np.concatenate([z_cur, z_cur - z_prev, np.array([cycle_code, time_code, validation_score / 10.0], dtype=float)])


def decode_latent(pca: PCA, z: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return np.clip(pca.inverse_transform(z.reshape(1, -1))[0].reshape(shape), 0, 1).astype(np.float32)


def evaluate_sequence(seq: Dict[str, object], pca: PCA, model: Ridge, train_fraction: float) -> Tuple[pd.DataFrame, Dict[str, object], np.ndarray]:
    row = seq["row"]
    frames = seq["frames"]
    flat = seq["flat"]
    frame_indices = seq["frame_indices"]
    split = split_index(len(frames), train_fraction)
    z = pca.transform(flat)
    denom = max(1, len(frames) - 1)
    rows = []
    for t in range(split, len(frames) - 1):
        truth = frames[t + 1]
        feat = latent_feature(z[t - 1], z[t], seq["cycle_code"], t / denom, seq["validation_score"])
        pred_delta = model.predict(feat.reshape(1, -1))[0]
        preds = {
            "persistence": frames[t],
            "velocity": np.clip(frames[t] + (frames[t] - frames[t - 1]), 0, 1),
            "event_pca_ridge": decode_latent(pca, z[t] + pred_delta, frames.shape[1:]),
        }
        for method, pred in preds.items():
            out = image_metrics(pred, truth)
            out.update({
                "roi_id": row["roi_id"],
                "cycleNo": float(row["cycleNo"]),
                "source_stem": row["source_stem"],
                "method": method,
                "mode": "teacher_forced",
                "eval_step": int(t - split),
                "frame_index": int(frame_indices[t + 1]),
                "validation_score": finite_float(row.get("validation_score")),
            })
            rows.append(out)
    prev_z = z[split - 2].copy()
    cur_z = z[split - 1].copy()
    states = {
        "persistence": frames[split - 1].copy(),
        "velocity": frames[split - 1].copy(),
        "event_pca_ridge": decode_latent(pca, cur_z, frames.shape[1:]),
    }
    prev_states = {k: frames[split - 2].copy() for k in states}
    ridge_rollout = []
    residual_energy = []
    for step, t in enumerate(range(split, len(frames))):
        truth = frames[t]
        feat = latent_feature(prev_z, cur_z, seq["cycle_code"], (t - 1) / denom, seq["validation_score"])
        pred_delta = model.predict(feat.reshape(1, -1))[0]
        next_z = cur_z + pred_delta
        next_states = {
            "persistence": states["persistence"],
            "velocity": np.clip(states["velocity"] + (states["velocity"] - prev_states["velocity"]), 0, 1),
            "event_pca_ridge": decode_latent(pca, next_z, frames.shape[1:]),
        }
        for method, pred in next_states.items():
            out = image_metrics(pred, truth)
            out.update({
                "roi_id": row["roi_id"],
                "cycleNo": float(row["cycleNo"]),
                "source_stem": row["source_stem"],
                "method": method,
                "mode": "recursive_rollout",
                "eval_step": int(step),
                "frame_index": int(frame_indices[t]),
                "validation_score": finite_float(row.get("validation_score")),
            })
            rows.append(out)
        ridge_rollout.append(next_states["event_pca_ridge"])
        residual_energy.append(float(np.mean((next_states["event_pca_ridge"] - truth) ** 2)))
        prev_z, cur_z = cur_z, next_z
        for method in states:
            prev_states[method] = states[method]
            states[method] = next_states[method]
    residual = np.asarray(residual_energy, dtype=float)
    feature = {
        "roi_id": row["roi_id"],
        "cycleNo": float(row["cycleNo"]),
        "validation_score": finite_float(row.get("validation_score")),
        "rollout_residual_energy_mean": float(np.nanmean(residual)),
        "rollout_residual_energy_last": float(residual[-1]) if residual.size else np.nan,
        "rollout_residual_energy_slope_per_step": float(np.polyfit(np.arange(len(residual)), residual, 1)[0]) if residual.size > 2 else np.nan,
        "truth_roi_delta_tail": float(np.mean(frames[-1] - frames[split - 1])),
        "ridge_rollout_delta_tail": float(np.mean(ridge_rollout[-1] - ridge_rollout[0])) if ridge_rollout else np.nan,
    }
    return pd.DataFrame(rows), feature, np.stack(ridge_rollout, axis=0)


def save_rollout_preview(seq: Dict[str, object], rollout: np.ndarray, out_png: str, train_fraction: float) -> None:
    frames = seq["frames"]
    split = split_index(len(frames), train_fraction)
    truth = frames[split:split + len(rollout)]
    picks = [0, len(rollout) // 2, len(rollout) - 1]
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.2))
    for col, idx in enumerate(picks):
        true = truth[idx]
        pred = rollout[idx]
        diff = pred - true
        axes[0, col].imshow(true, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"truth +{idx}", fontsize=9)
        axes[1, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("PCA ridge", fontsize=9)
        vmax = max(0.03, float(np.nanpercentile(np.abs(diff), 99)))
        axes[2, col].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[2, col].set_title("pred - truth", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(str(seq["row"]["roi_id"]), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_summary_plot(per_roi: pd.DataFrame, residual_features: pd.DataFrame, meta: pd.DataFrame, out_png: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sub = per_roi[per_roi["mode"] == "recursive_rollout"]
    for method in sub["method"].drop_duplicates():
        g = sub[sub["method"] == method].groupby("cycleNo")["mse"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(g["cycleNo"], g["mean"], yerr=g["std"].fillna(0), marker="o", label=method)
    axes[0].set_title("recursive rollout MSE", fontsize=9)
    axes[0].set_xlabel("cycle")
    axes[0].legend(fontsize=7)
    axes[1].scatter(residual_features["truth_roi_delta_tail"], residual_features["rollout_residual_energy_mean"], c=residual_features["cycleNo"], cmap="viridis", s=60)
    axes[1].set_xlabel("truth tail mean delta")
    axes[1].set_ylabel("ridge rollout residual energy")
    axes[1].set_title("residual descriptors", fontsize=9)
    axes[2].bar(["PCA var", "train MSE"], [meta["pca_explained_variance_ratio_sum"].iloc[0], meta["train_latent_delta_mse"].iloc[0]])
    axes[2].set_title("model fit", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/selected_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_event_conditioned_nextframe")
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--model-stride", type=int, default=2)
    args = parser.parse_args()

    manifest_path = os.path.join(args.roi_dir, "selected_roi_sequence_manifest.csv")
    seqs = load_sequences(manifest_path, args.model_stride)
    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "rollout_previews")
    os.makedirs(preview_dir, exist_ok=True)
    pca, model, model_meta = fit_latent_model(seqs, args.train_fraction, args.rank, args.alpha)
    metric_tables = []
    features = []
    for seq in seqs:
        metrics_df, feature, rollout = evaluate_sequence(seq, pca, model, args.train_fraction)
        metric_tables.append(metrics_df)
        features.append(feature)
        save_rollout_preview(seq, rollout, os.path.join(preview_dir, f"{seq['row']['roi_id']}_pca_ridge_rollout.png"), args.train_fraction)
    metrics_all = pd.concat(metric_tables, ignore_index=True)
    residual_features = pd.DataFrame(features)
    per_roi = metrics_all.groupby(["roi_id", "cycleNo", "mode", "method"], as_index=False).agg({
        "mse": "mean",
        "mae": "mean",
        "psnr": "mean",
        "ssim": "mean",
        "validation_score": "first",
    })
    cycle_method = per_roi.groupby(["cycleNo", "mode", "method"], as_index=False).agg({
        "mse": ["mean", "std"],
        "mae": ["mean", "std"],
        "ssim": ["mean", "std"],
        "psnr": ["mean", "std"],
    })
    cycle_method.columns = ["_".join(c).strip("_") for c in cycle_method.columns.to_flat_index()]
    residual_cycle = residual_features.groupby("cycleNo", as_index=False).agg({
        "rollout_residual_energy_mean": "mean",
        "rollout_residual_energy_last": "mean",
        "rollout_residual_energy_slope_per_step": "mean",
        "truth_roi_delta_tail": "mean",
        "ridge_rollout_delta_tail": "mean",
    })

    paths = {
        "frame_metrics": os.path.join(args.out_dir, "roi_event_model_frame_metrics.csv"),
        "per_roi_metrics": os.path.join(args.out_dir, "roi_event_model_per_roi_metrics.csv"),
        "cycle_method_summary": os.path.join(args.out_dir, "roi_event_model_cycle_method_summary.csv"),
        "residual_features": os.path.join(args.out_dir, "rollout_residual_degradation_features.csv"),
        "residual_cycle_summary": os.path.join(args.out_dir, "rollout_residual_cycle_summary.csv"),
        "model_meta": os.path.join(args.out_dir, "model_fit_summary.csv"),
        "plot": os.path.join(args.out_dir, "roi_event_model_summary.png"),
    }
    metrics_all.to_csv(paths["frame_metrics"], index=False)
    per_roi.to_csv(paths["per_roi_metrics"], index=False)
    cycle_method.to_csv(paths["cycle_method_summary"], index=False)
    residual_features.to_csv(paths["residual_features"], index=False)
    residual_cycle.to_csv(paths["residual_cycle_summary"], index=False)
    model_meta.to_csv(paths["model_meta"], index=False)
    save_summary_plot(per_roi, residual_features, model_meta, paths["plot"])
    summary = {
        "roi_dir": args.roi_dir,
        "n_roi_sequences": int(len(seqs)),
        "n_metric_rows": int(len(metrics_all)),
        "rank": int(model_meta["rank"].iloc[0]),
        "alpha": float(args.alpha),
        "model_stride": int(args.model_stride),
        "train_fraction": float(args.train_fraction),
        "pca_explained_variance_ratio_sum": float(model_meta["pca_explained_variance_ratio_sum"].iloc[0]),
        "train_latent_delta_mse": float(model_meta["train_latent_delta_mse"].iloc[0]),
        "cycle_method_summary": cycle_method.to_dict(orient="records"),
        "residual_cycle_summary": residual_cycle.to_dict(orient="records"),
        "guardrail": "PCA-ridge event-conditioned model on 11 automatic ROI crops; use as an interpretable baseline/residual descriptor, not a final general video model.",
        "plot_path": paths["plot"],
    }
    summary_path = os.path.join(args.out_dir, "roi_event_model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# ROI Event-Conditioned Next-Frame Model\n\n")
        f.write("PCA-latent ridge next-frame model conditioned on cycle identity, normalized time, and ROI validation score.\n\n")
        f.write("Outputs compare teacher-forced and recursive rollout metrics against persistence and velocity baselines, plus rollout residual degradation features.\n")
    print(json.dumps(summary, indent=2, sort_keys=True)[:8000])


if __name__ == "__main__":
    main()
