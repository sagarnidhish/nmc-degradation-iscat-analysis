#!/usr/bin/env python3
# File: scripts/tier3_multimodal_transformer.py
# Tier 3.2: Multimodal Temporal Fusion Transformer (optical + echem tokens)
# Predicts probability of entering accelerated fade in next K cycles
# Reads: derived/particle_intensity_normalized.csv, derived/echem_per_cycle.csv,
#        derived/ts2vec/encoder_weights.pt (if available)
# Writes: derived/transformer/model_weights.pt, derived/transformer/predictions.csv,
#         derived/plots/transformer_*.png

import argparse
import json
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


ACCEL_FADE_CYCLE = 80   # accelerated fade typically begins around cycle 80-100
FORECAST_HORIZON = 10   # predict if fade accelerates within next K cycles


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def find_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def build_multimodal_sequence(
    particle_df: pd.DataFrame,
    echem_df: pd.DataFrame,
    context_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (N_windows, context_len, n_features) input tensor and (N_windows,) binary labels.
    label=1 if capacity will drop by >5% relative in next FORECAST_HORIZON cycles.
    Returns: X (N, T, F), mask (N, T), y (N,)
    """
    norm_cols = [c for c in particle_df.columns if c.endswith("_norm")]
    echem_cols = ["capacity_mAh", "coulombic_efficiency_pct", "V_max", "V_min"]
    echem_cols = [c for c in echem_cols if c in echem_df.columns]

    # Merge on cycle
    echem_df = echem_df.copy()
    echem_df["cycleNo"] = pd.to_numeric(echem_df["cycleNo"], errors="coerce")
    particle_df = particle_df.copy()
    particle_df["cycleNo"] = pd.to_numeric(particle_df["cycleNo"], errors="coerce")

    merged = particle_df.merge(echem_df[["cycleNo"] + echem_cols].drop_duplicates("cycleNo"),
                                on="cycleNo", how="left").sort_values("cycleNo").reset_index(drop=True)

    feature_cols = norm_cols + echem_cols
    feat_mat = merged[feature_cols].to_numpy(dtype=float)
    cycles = merged["cycleNo"].to_numpy(dtype=float)

    # Normalise each feature column
    scaler = StandardScaler()
    feat_mat_norm = scaler.fit_transform(np.nan_to_num(feat_mat, nan=0.0))

    X_list, mask_list, y_list = [], [], []
    cap = merged["capacity_mAh"].to_numpy(dtype=float) if "capacity_mAh" in merged.columns else None

    for i in range(context_len, len(merged)):
        window = feat_mat_norm[i - context_len:i]
        valid = np.all(np.isfinite(feat_mat[i - context_len:i]), axis=1)
        mask = valid.astype(float)

        # Label: does capacity drop >5% relative within next FORECAST_HORIZON cycles?
        label = 0
        if cap is not None:
            c0 = cap[i]
            future = cap[i:min(i + FORECAST_HORIZON, len(cap))]
            future_finite = future[np.isfinite(future)]
            if np.isfinite(c0) and c0 > 0 and future_finite.size > 0:
                label = int(future_finite.min() < 0.95 * c0)

        X_list.append(window)
        mask_list.append(mask)
        y_list.append(label)

    if not X_list:
        return np.zeros((0, context_len, len(feature_cols))), np.zeros((0, context_len)), np.zeros(0)

    return (np.array(X_list, dtype=np.float32),
            np.array(mask_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


class TemporalFusionTransformer(nn.Module):
    """Simplified TFT: multi-head attention over context window + feed-forward head."""
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=d_model * 4,
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, F)
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x) + self.pos_enc(pos)
        key_padding_mask = (mask == 0) if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        pooled = h.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


def train_transformer(
    X_tr: np.ndarray, mask_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, mask_val: np.ndarray, y_val: np.ndarray,
    d_model: int = 64, nhead: int = 4, n_layers: int = 3,
    epochs: int = 200, lr: float = 5e-4, device: str = "cpu"
) -> Tuple[TemporalFusionTransformer, List[float], List[float]]:
    n_feat = X_tr.shape[2]
    model = TemporalFusionTransformer(n_feat, d_model, nhead, n_layers).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                               dtype=torch.float32, device=device)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Mt = torch.tensor(mask_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    Mv = torch.tensor(mask_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_losses, val_losses = [], []
    batch_size = min(32, len(X_tr))

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(Xt.size(0))
        ep_losses = []
        for s in range(0, Xt.size(0), batch_size):
            bi = idx[s:s + batch_size]
            pred = model(Xt[bi], Mt[bi])
            loss = F.binary_cross_entropy(pred, yt[bi], reduction="none")
            w = torch.where(yt[bi] == 1, pos_weight.expand_as(yt[bi]), torch.ones_like(yt[bi]))
            loss = (loss * w).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(float(loss.item()))
        sched.step()
        train_losses.append(float(np.mean(ep_losses)))

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = F.binary_cross_entropy(model(Xv, Mv), yv).item()
            val_losses.append(float(vl))
            if ep % 50 == 0:
                print(f"  Epoch {ep}: train_loss={train_losses[-1]:.4f} val_loss={vl:.4f}")

    return model, train_losses, val_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = os.path.join(args.derived_dir, "transformer")
    plots_dir = os.path.join(args.derived_dir, "plots")
    ensure_dirs(out_dir, plots_dir)

    particle_path = find_path([os.path.join(args.derived_dir, "particle_intensity_normalized.csv")])
    echem_path = find_path([os.path.join(args.derived_dir, "echem_per_cycle.csv")])
    if not particle_path or not echem_path:
        print("Warning: Required inputs not found — run Tier 1 first.")
        return

    particle_df = pd.read_csv(particle_path)
    echem_df = pd.read_csv(echem_path)

    X, mask, y = build_multimodal_sequence(particle_df, echem_df, context_len=args.context_len)
    if X.shape[0] < 10:
        print(f"Warning: Only {X.shape[0]} windows — need more cycles. Insufficient data for transformer training.")
        return

    n_events = int(y.sum())
    print(f"Windows: {X.shape[0]}, Events: {n_events} ({n_events/X.shape[0]*100:.1f}%), Features: {X.shape[2]}")

    # Temporal train/val split (no random shuffle — preserves temporal order)
    split = int(0.8 * X.shape[0])
    X_tr, mask_tr, y_tr = X[:split], mask[:split], y[:split]
    X_val, mask_val, y_val = X[split:], mask[split:], y[split:]

    if y_val.sum() == 0:
        print("Warning: No events in validation set — metrics will be degenerate.")

    model, train_losses, val_losses = train_transformer(
        X_tr, mask_tr, y_tr, X_val, mask_val, y_val,
        d_model=args.d_model, nhead=args.nhead, n_layers=args.n_layers,
        epochs=args.epochs, lr=args.lr, device=device
    )

    model.eval()
    with torch.no_grad():
        val_preds = model(
            torch.tensor(X_val, dtype=torch.float32, device=device),
            torch.tensor(mask_val, dtype=torch.float32, device=device)
        ).cpu().numpy()

    metrics: dict = {"n_windows": int(X.shape[0]), "n_events": n_events,
                     "context_len": args.context_len, "forecast_horizon": FORECAST_HORIZON}
    if y_val.sum() > 0 and (y_val == 0).sum() > 0:
        metrics["val_auroc"] = float(roc_auc_score(y_val, val_preds))
        metrics["val_auprc"] = float(average_precision_score(y_val, val_preds))
        print(f"Val AUROC: {metrics['val_auroc']:.4f}  AUPRC: {metrics['val_auprc']:.4f}")
    else:
        print("Warning: Cannot compute AUROC — need both classes in val set.")

    torch.save({
        "state_dict": model.state_dict(),
        "config": {"n_features": X.shape[2], "d_model": args.d_model,
                   "nhead": args.nhead, "n_layers": args.n_layers,
                   "context_len": args.context_len},
        "metrics": metrics,
    }, os.path.join(out_dir, "model_weights.pt"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # All predictions
    with torch.no_grad():
        all_preds = model(
            torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device)
        ).cpu().numpy()
    pred_df = pd.DataFrame({"window_idx": np.arange(X.shape[0]),
                             "true_label": y, "predicted_prob": all_preds})
    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(train_losses, label="Train", linewidth=1.5)
    if val_losses:
        axes[0].plot(np.linspace(0, len(train_losses)-1, len(val_losses)), val_losses, label="Val", linewidth=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Transformer Training Loss"); axes[0].legend(); axes[0].grid(alpha=0.25)

    axes[1].plot(all_preds, label="P(accel. fade)", color="#d62728", linewidth=1.5, alpha=0.8)
    event_idx = np.where(y == 1)[0]
    axes[1].scatter(event_idx, all_preds[event_idx], marker="x", color="black", s=50, zorder=5, label="True event")
    axes[1].axvline(split, color="gray", linestyle="--", linewidth=1.5, label="Train/val split")
    axes[1].set_xlabel("Window index (temporal)")
    axes[1].set_ylabel("P(accelerated fade in next 10 cycles)")
    axes[1].set_title("Transformer fade-acceleration forecast")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "transformer_results.png"), dpi=200)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
