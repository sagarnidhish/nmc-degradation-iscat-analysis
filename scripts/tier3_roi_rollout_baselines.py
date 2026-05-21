#!/usr/bin/env python3
"""Particle-ROI next-frame and rollout baselines for NMC event sequences.

This evaluates simple, interpretable dynamics on selected particle-region crops:

* persistence: next frame equals current frame
* velocity: next frame equals current plus the previous frame difference
* low-rank DMD/PCA: fit a linear latent dynamics model on early frames and
  recursively roll it forward through the held-out tail of each ROI sequence

The point is not to claim a final video model. It is to establish particle-only
model inputs, baseline rollout difficulty, and interpretable latent dynamics
that can be compared across degradation event cycles.
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
from skimage.metrics import structural_similarity as ssim


def load_sequences(manifest_path: str) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    manifest = pd.read_csv(manifest_path)
    seqs = []
    for _, row in manifest.iterrows():
        data = np.load(row["npz_path"])
        frames = data["frames_norm"].astype(np.float32)
        seqs.append({"row": row, "frames": frames.reshape(frames.shape[0], -1), "shape": frames.shape[1:]})
    return manifest, seqs


def split_pairs(seqs: List[Dict[str, object]], train_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for seq in seqs:
        frames = seq["frames"]
        split = max(3, min(frames.shape[0] - 2, int(frames.shape[0] * train_fraction)))
        xs.append(frames[:split - 1])
        ys.append(frames[1:split])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fit_low_rank_dmd(x_train: np.ndarray, y_train: np.ndarray, rank: int, ridge: float) -> Dict[str, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    x0 = x_train - mean
    u, s, vt = np.linalg.svd(x0, full_matrices=False)
    k = int(min(rank, vt.shape[0]))
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


def metrics(pred: np.ndarray, truth: np.ndarray, shape: Tuple[int, int]) -> Dict[str, float]:
    diff = pred - truth
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    psnr = float(10.0 * math.log10(1.0 / mse)) if mse > 0 else float("inf")
    try:
        sim = float(ssim(truth.reshape(shape), pred.reshape(shape), data_range=1.0))
    except Exception:
        sim = np.nan
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": sim}


def evaluate_sequence(seq: Dict[str, object], model: Dict[str, np.ndarray], train_fraction: float) -> Tuple[pd.DataFrame, np.ndarray]:
    row = seq["row"]
    frames = seq["frames"]
    shape = seq["shape"]
    split = max(3, min(frames.shape[0] - 2, int(frames.shape[0] * train_fraction)))
    basis = model["basis"]
    mean = model["mean"]
    a = model["a"]

    start = frames[split - 1:split]
    z = (start - mean) @ basis
    prev = frames[split - 2]
    cur = frames[split - 1]
    dmd_rollout = []
    rows = []
    for t in range(split, frames.shape[0]):
        truth = frames[t]
        persistence = cur
        velocity = np.clip(cur + (cur - prev), 0, 1)
        z = z @ a
        dmd = decode(z, model)[0]
        dmd_rollout.append(dmd.reshape(shape))
        for name, pred in [("persistence", persistence), ("velocity", velocity), ("low_rank_dmd", dmd)]:
            out = metrics(pred, truth, shape)
            out.update({
                "roi_id": row["roi_id"],
                "cycleNo": float(row["cycleNo"]),
                "source_stem": row["source_stem"],
                "method": name,
                "eval_step": int(t - split),
                "frame_index": int(np.load(row["npz_path"])["frame_indices"][t]),
                "validation_score": float(row["validation_score"]),
            })
            rows.append(out)
        prev, cur = cur, truth
    return pd.DataFrame(rows), np.stack(dmd_rollout, axis=0)


def latent_summary(seqs: List[Dict[str, object]], model: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    basis = model["basis"]
    mean = model["mean"]
    for seq in seqs:
        row = seq["row"]
        z = (seq["frames"] - mean) @ basis
        dz = np.diff(z, axis=0)
        rows.append({
            "roi_id": row["roi_id"],
            "cycleNo": float(row["cycleNo"]),
            "validation_score": float(row["validation_score"]),
            "latent_path_length": float(np.sum(np.linalg.norm(dz, axis=1))),
            "latent_mean_step": float(np.mean(np.linalg.norm(dz, axis=1))),
            "latent_net_displacement": float(np.linalg.norm(z[-1] - z[0])),
            "latent_component0_delta": float(z[-1, 0] - z[0, 0]) if z.shape[1] else np.nan,
            "latent_component1_delta": float(z[-1, 1] - z[0, 1]) if z.shape[1] > 1 else np.nan,
            "latent_component0_first": float(z[0, 0]) if z.shape[1] else np.nan,
            "latent_component0_last": float(z[-1, 0]) if z.shape[1] else np.nan,
        })
    return pd.DataFrame(rows)


def save_rollout_preview(seq: Dict[str, object], dmd_rollout: np.ndarray, out_png: str, train_fraction: float) -> None:
    frames = seq["frames"].reshape((-1,) + seq["shape"])
    split = max(3, min(frames.shape[0] - 2, int(frames.shape[0] * train_fraction)))
    truth = frames[split:split + len(dmd_rollout)]
    picks = [0, len(dmd_rollout) // 2, len(dmd_rollout) - 1]
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.2))
    for col, idx in enumerate(picks):
        pred = dmd_rollout[idx]
        true = truth[idx]
        diff = pred - true
        axes[0, col].imshow(true, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"truth +{idx}", fontsize=9)
        axes[1, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("DMD rollout", fontsize=9)
        vmax = max(0.03, float(np.nanpercentile(np.abs(diff), 99)))
        axes[2, col].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[2, col].set_title("pred - truth", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(str(seq["row"]["roi_id"]), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_cycle_plot(per_roi: pd.DataFrame, latent: pd.DataFrame, out_png: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dmd = per_roi[per_roi["method"] == "low_rank_dmd"]
    methods = list(per_roi["method"].drop_duplicates())
    for method in methods:
        sub = per_roi[per_roi["method"] == method]
        grouped = sub.groupby("cycleNo")["mse"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(grouped["cycleNo"], grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", label=method)
    axes[0].set_xlabel("cycle")
    axes[0].set_ylabel("held-out rollout MSE")
    axes[0].legend(fontsize=8)
    axes[0].set_title("ROI rollout baselines")
    axes[1].scatter(latent["latent_component0_delta"], latent["latent_net_displacement"], c=latent["cycleNo"], cmap="viridis", s=60)
    for _, row in latent.iterrows():
        axes[1].text(row["latent_component0_delta"], row["latent_net_displacement"], str(int(row["cycleNo"])), fontsize=7)
    axes[1].set_xlabel("latent component 0 delta")
    axes[1].set_ylabel("latent net displacement")
    axes[1].set_title("DMD/PCA latent movement")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/selected_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_rollout_baselines")
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    args = parser.parse_args()

    manifest_path = os.path.join(args.roi_dir, "selected_roi_sequence_manifest.csv")
    manifest, seqs = load_sequences(manifest_path)
    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "rollout_previews")
    os.makedirs(preview_dir, exist_ok=True)

    x_train, y_train = split_pairs(seqs, args.train_fraction)
    model = fit_low_rank_dmd(x_train, y_train, args.rank, args.ridge)
    metric_tables = []
    preview_paths = []
    for seq in seqs:
        metrics_df, rollout = evaluate_sequence(seq, model, args.train_fraction)
        metric_tables.append(metrics_df)
        preview_path = os.path.join(preview_dir, f"{seq['row']['roi_id']}_dmd_rollout.png")
        save_rollout_preview(seq, rollout, preview_path, args.train_fraction)
        preview_paths.append(preview_path)
    metrics_all = pd.concat(metric_tables, ignore_index=True)
    per_roi = metrics_all.groupby(["roi_id", "cycleNo", "method"], as_index=False).agg({
        "mse": "mean",
        "mae": "mean",
        "psnr": "mean",
        "ssim": "mean",
        "validation_score": "first",
    })
    latent = latent_summary(seqs, model)
    cycle_method = per_roi.groupby(["cycleNo", "method"], as_index=False).agg({
        "mse": ["mean", "std"],
        "mae": ["mean", "std"],
        "ssim": ["mean", "std"],
        "psnr": ["mean", "std"],
    })
    cycle_method.columns = ["_".join(c).strip("_") for c in cycle_method.columns.to_flat_index()]

    metrics_path = os.path.join(args.out_dir, "roi_rollout_frame_metrics.csv")
    per_roi_path = os.path.join(args.out_dir, "roi_rollout_per_roi_metrics.csv")
    cycle_path = os.path.join(args.out_dir, "roi_rollout_cycle_method_summary.csv")
    latent_path = os.path.join(args.out_dir, "roi_latent_dynamics_summary.csv")
    metrics_all.to_csv(metrics_path, index=False)
    per_roi.to_csv(per_roi_path, index=False)
    cycle_method.to_csv(cycle_path, index=False)
    latent.to_csv(latent_path, index=False)
    plot_path = os.path.join(args.out_dir, "roi_rollout_cycle_summary.png")
    save_cycle_plot(per_roi, latent, plot_path)

    summary = {
        "roi_dir": args.roi_dir,
        "n_roi_sequences": int(len(seqs)),
        "n_frame_metric_rows": int(len(metrics_all)),
        "rank": args.rank,
        "train_fraction": args.train_fraction,
        "dmd_spectral_radius": float(model["spectral_radius"][0]),
        "cycle_method_summary": cycle_method.to_dict(orient="records"),
        "latent_cycle_summary": latent.groupby("cycleNo").agg({
            "latent_path_length": "mean",
            "latent_mean_step": "mean",
            "latent_net_displacement": "mean",
            "latent_component0_delta": "mean",
        }).reset_index().to_dict(orient="records"),
        "guardrail": "Low-rank DMD is an interpretable ROI-only rollout baseline, not a final neural video model or calibrated diffusion estimator.",
        "plot_path": plot_path,
        "preview_count": len(preview_paths),
    }
    summary_path = os.path.join(args.out_dir, "roi_rollout_baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# ROI Rollout Baselines\n\n")
        f.write("Particle-region-only next-frame and recursive rollout baselines on selected NMC event ROIs.\n\n")
        f.write("Methods: persistence, velocity extrapolation, and low-rank DMD/PCA latent dynamics.\n\n")
        f.write("These are interpretable baselines for model difficulty and latent motion, not final calibrated physics estimates.\n")
    for path in [metrics_path, per_roi_path, cycle_path, latent_path, summary_path, plot_path]:
        print(f"Saved: {path}")
    print(json.dumps(summary, indent=2, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
