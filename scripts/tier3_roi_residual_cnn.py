#!/usr/bin/env python3
"""Train/evaluate a small particle-ROI residual CNN against persistence.

The selected ROI sequences evolve slowly, so raw next-frame MSE is dominated by
the persistence baseline. This experiment explicitly predicts residuals
``next_frame - current_frame`` from particle-region-only crops, then evaluates
whether learned residuals improve next-frame prediction and residual sign/energy
relative to predicting zero residual.

It uses leave-one-ROI-out evaluation so every test ROI is unseen during training.
The model is intentionally small and fast; this is a baseline and diagnostic,
not a final production video model.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ResidualCNN(nn.Module):
    def __init__(self, hidden: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def load_manifest(roi_dir: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    manifest = pd.read_csv(os.path.join(roi_dir, "selected_roi_sequence_manifest.csv"))
    data = {}
    for _, row in manifest.iterrows():
        npz = np.load(row["npz_path"])
        data[str(row["roi_id"])] = npz["frames_norm"].astype(np.float32)
    return manifest, data


def make_pairs(manifest: pd.DataFrame, data: Dict[str, np.ndarray], roi_ids: List[str], stride: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs, ys, rows = [], [], []
    for roi_id in roi_ids:
        frames = data[roi_id]
        row = manifest[manifest["roi_id"] == roi_id].iloc[0]
        for t in range(0, frames.shape[0] - 1, stride):
            cur = frames[t]
            nxt = frames[t + 1]
            xs.append(cur[None, ...])
            ys.append((nxt - cur)[None, ...])
            rows.append({
                "roi_id": roi_id,
                "cycleNo": float(row["cycleNo"]),
                "pair_index": int(t),
                "validation_score": float(row["validation_score"]),
            })
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), pd.DataFrame(rows)


def split_train_val(x: np.ndarray, y: np.ndarray, val_fraction: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, args) -> Tuple[ResidualCNN, List[Dict[str, float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ResidualCNN(args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    history = []
    best_state = None
    best_val = float("inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).detach().cpu())
        train_loss = float(np.mean(losses)) if losses else np.nan
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def frame_metrics(cur: np.ndarray, truth_res: np.ndarray, pred_res: np.ndarray) -> Dict[str, float]:
    truth = np.clip(cur + truth_res, 0, 1)
    persistence = cur
    pred = np.clip(cur + pred_res, 0, 1)
    zero_res = np.zeros_like(truth_res)
    persist_err = persistence - truth
    pred_err = pred - truth
    res_err_zero = zero_res - truth_res
    res_err_pred = pred_res - truth_res
    mse_p = float(np.mean(persist_err ** 2))
    mse_m = float(np.mean(pred_err ** 2))
    res_mse_zero = float(np.mean(res_err_zero ** 2))
    res_mse_model = float(np.mean(res_err_pred ** 2))
    mae_p = float(np.mean(np.abs(persist_err)))
    mae_m = float(np.mean(np.abs(pred_err)))
    try:
        ssim_p = float(ssim(truth[0], persistence[0], data_range=1.0))
        ssim_m = float(ssim(truth[0], pred[0], data_range=1.0))
    except Exception:
        ssim_p = np.nan
        ssim_m = np.nan
    sign_mask = np.abs(truth_res) > np.percentile(np.abs(truth_res), 75)
    if np.any(sign_mask):
        sign_acc = float(np.mean(np.sign(pred_res[sign_mask]) == np.sign(truth_res[sign_mask])))
    else:
        sign_acc = np.nan
    return {
        "persistence_mse": mse_p,
        "model_mse": mse_m,
        "mse_improvement": mse_p - mse_m,
        "relative_mse_improvement": (mse_p - mse_m) / mse_p if mse_p > 0 else np.nan,
        "persistence_mae": mae_p,
        "model_mae": mae_m,
        "residual_mse_zero": res_mse_zero,
        "residual_mse_model": res_mse_model,
        "relative_residual_mse_improvement": (res_mse_zero - res_mse_model) / res_mse_zero if res_mse_zero > 0 else np.nan,
        "persistence_ssim": ssim_p,
        "model_ssim": ssim_m,
        "residual_sign_accuracy_top_quartile": sign_acc,
        "truth_residual_energy": float(np.mean(truth_res ** 2)),
        "pred_residual_energy": float(np.mean(pred_res ** 2)),
    }


def evaluate_model(model: ResidualCNN, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, cpu: bool) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 64):
            xb = torch.from_numpy(x[i:i + 64]).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    rows = []
    for i in range(x.shape[0]):
        out = frame_metrics(x[i], y[i], pred[i])
        out.update(meta.iloc[i].to_dict())
        rows.append(out)
    return pd.DataFrame(rows)


def save_preview(model: ResidualCNN, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, out_png: str, cpu: bool) -> None:
    if x.shape[0] == 0:
        return
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model = model.to(device)
    model.eval()
    picks = np.linspace(0, x.shape[0] - 1, min(4, x.shape[0]), dtype=int)
    fig, axes = plt.subplots(len(picks), 5, figsize=(11, 2.4 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    with torch.no_grad():
        pred = model(torch.from_numpy(x[picks]).to(device)).detach().cpu().numpy()
    for row_idx, i in enumerate(picks):
        cur = x[i, 0]
        truth_res = y[i, 0]
        pred_res = pred[row_idx, 0]
        truth = np.clip(cur + truth_res, 0, 1)
        pred_frame = np.clip(cur + pred_res, 0, 1)
        diff = pred_frame - truth
        panels = [cur, truth, pred_frame, truth_res, diff]
        titles = ["current", "truth next", "model next", "truth residual", "model error"]
        for ax, img, title in zip(axes[row_idx], panels, titles):
            if "residual" in title or "error" in title:
                vmax = max(0.02, float(np.nanpercentile(np.abs(img), 99)))
                ax.imshow(img, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(str(meta.iloc[i]["roi_id"]), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/selected_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/roi_residual_cnn")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--eval-mode", choices=["leave_one_roi", "leave_one_cycle"], default="leave_one_roi")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)

    manifest, data = load_manifest(args.roi_dir)
    roi_ids = list(manifest["roi_id"].astype(str))
    all_metrics = []
    histories = []
    if args.eval_mode == "leave_one_cycle":
        folds = []
        for cyc in sorted(manifest["cycleNo"].astype(float).unique()):
            test_ids = manifest.loc[manifest["cycleNo"].astype(float) == cyc, "roi_id"].astype(str).tolist()
            train_ids = [r for r in roi_ids if r not in test_ids]
            folds.append((f"cycle_{int(cyc)}", train_ids, test_ids))
    else:
        folds = [(test_roi, [r for r in roi_ids if r != test_roi], [test_roi]) for test_roi in roi_ids]

    for fold_id, train_ids, test_ids in folds:
        x_all, y_all, _ = make_pairs(manifest, data, train_ids, args.stride)
        x_train, y_train, x_val, y_val = split_train_val(x_all, y_all, args.val_fraction)
        model, hist = train_model(x_train, y_train, x_val, y_val, args)
        for h in hist:
            h["fold_id"] = fold_id
        histories.extend(hist)
        x_test, y_test, meta_test = make_pairs(manifest, data, test_ids, args.stride)
        metrics = evaluate_model(model, x_test, y_test, meta_test, args.cpu)
        metrics["fold_id"] = fold_id
        all_metrics.append(metrics)
        save_preview(model, x_test, y_test, meta_test, os.path.join(preview_dir, f"{fold_id}_residual_cnn.png"), args.cpu)
        print(f"evaluated {fold_id}: rel_mse_improvement={metrics['relative_mse_improvement'].mean():.4g}")

    frame_metrics_df = pd.concat(all_metrics, ignore_index=True)
    per_roi = frame_metrics_df.groupby(["roi_id", "cycleNo"], as_index=False).agg({
        "persistence_mse": "mean",
        "model_mse": "mean",
        "mse_improvement": "mean",
        "relative_mse_improvement": "mean",
        "persistence_mae": "mean",
        "model_mae": "mean",
        "relative_residual_mse_improvement": "mean",
        "residual_sign_accuracy_top_quartile": "mean",
        "truth_residual_energy": "mean",
        "pred_residual_energy": "mean",
        "validation_score": "first",
    })
    cycle = per_roi.groupby("cycleNo", as_index=False).agg({
        "roi_id": "count",
        "persistence_mse": "mean",
        "model_mse": "mean",
        "relative_mse_improvement": "mean",
        "relative_residual_mse_improvement": "mean",
        "residual_sign_accuracy_top_quartile": "mean",
        "truth_residual_energy": "mean",
        "pred_residual_energy": "mean",
    }).rename(columns={"roi_id": "n_roi"})

    frame_path = os.path.join(args.out_dir, "roi_residual_cnn_frame_metrics.csv")
    per_roi_path = os.path.join(args.out_dir, "roi_residual_cnn_per_roi_metrics.csv")
    cycle_path = os.path.join(args.out_dir, "roi_residual_cnn_cycle_summary.csv")
    hist_path = os.path.join(args.out_dir, "roi_residual_cnn_training_history.csv")
    frame_metrics_df.to_csv(frame_path, index=False)
    per_roi.to_csv(per_roi_path, index=False)
    cycle.to_csv(cycle_path, index=False)
    pd.DataFrame(histories).to_csv(hist_path, index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(cycle["cycleNo"].astype(str), cycle["relative_mse_improvement"])
    ax[0].axhline(0, color="k", linewidth=0.8)
    ax[0].set_ylabel("relative MSE improvement vs persistence")
    ax[0].set_xlabel("cycle")
    ax[0].set_title("Residual CNN next-frame improvement")
    ax[1].scatter(per_roi["truth_residual_energy"], per_roi["relative_residual_mse_improvement"], c=per_roi["cycleNo"], cmap="viridis", s=65)
    ax[1].axhline(0, color="k", linewidth=0.8)
    ax[1].set_xlabel("truth residual energy")
    ax[1].set_ylabel("relative residual MSE improvement")
    ax[1].set_title("Residual prediction quality")
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, "roi_residual_cnn_summary.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "roi_dir": args.roi_dir,
        "n_roi": int(len(roi_ids)),
        "n_frame_pairs": int(len(frame_metrics_df)),
        "eval_mode": args.eval_mode,
        "device": "cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
        "cycle_summary": cycle.to_dict(orient="records"),
        "overall_relative_mse_improvement": float(per_roi["relative_mse_improvement"].mean()),
        "overall_relative_residual_mse_improvement": float(per_roi["relative_residual_mse_improvement"].mean()),
        "guardrail": "Residual CNN is a small ROI-only next-frame baseline. Positive improvement over persistence is required before considering larger neural rollout models.",
        "plot_path": plot_path,
    }
    summary_path = os.path.join(args.out_dir, "roi_residual_cnn_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# ROI Residual CNN\n\n")
        f.write("Leave-one-ROI-out residual next-frame predictor on selected particle-region NMC crops.\n\n")
        f.write("The model predicts `next - current`; persistence is the zero-residual baseline.\n")
    for path in [frame_path, per_roi_path, cycle_path, hist_path, summary_path, plot_path]:
        print(f"Saved: {path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
