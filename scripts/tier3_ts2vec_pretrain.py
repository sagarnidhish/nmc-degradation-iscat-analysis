#!/usr/bin/env python3
# File: scripts/tier3_ts2vec_pretrain.py
# Tier 3.1: Self-supervised TS2Vec-style contrastive pretraining on particle traces
# Pretrain on all available traces (full cycle range), then export encoder for downstream use
# Reads: derived/particle_intensity_normalized.csv + HDF5 average_intensity streams
# Writes: derived/ts2vec/encoder_weights.pt, derived/ts2vec/pretrain_metrics.json

import argparse
import json
import os
from typing import List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def find_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def discover_h5_files(base_dir: str) -> List[str]:
    results = []
    skip = {"derived", ".git"}
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for fn in files:
            if fn.lower().endswith((".h5", ".hdf5")):
                results.append(os.path.join(root, fn))
    return sorted(results)


def load_avg_intensity_from_h5(path: str, max_frames: int = 5000) -> Optional[np.ndarray]:
    """Load average_intensity (never loads movie). Subsample to max_frames."""
    try:
        with h5py.File(path, "r") as f:
            if "average_intensity" not in f:
                return None
            arr = np.asarray(f["average_intensity"][:], dtype=float)
            if arr.ndim == 2:
                arr = arr[0, :]
            else:
                arr = arr.reshape(-1)
            if arr.size == 0:
                return None
            if arr.size > max_frames:
                idx = np.linspace(0, arr.size - 1, max_frames, dtype=int)
                arr = arr[idx]
            # Normalise to zero mean unit std
            std = np.nanstd(arr)
            if std > 0:
                arr = (arr - np.nanmean(arr)) / std
            arr = np.where(np.isfinite(arr), arr, 0.0)
            return arr.astype(np.float32)
    except Exception:
        return None


def collect_traces(particle_path: Optional[str], h5_base: str, max_h5: int = 5,
                   target_len: int = 128) -> np.ndarray:
    """Collect and resample traces to uniform target_len. Returns (N, target_len)."""
    traces = []

    if particle_path:
        df = pd.read_csv(particle_path)
        df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
        df = df.sort_values("cycleNo")
        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        for col in norm_cols:
            vals = df[col].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if finite.sum() < 10:
                continue
            x = np.arange(len(vals))
            vals = np.interp(x, x[finite], vals[finite]).astype(np.float32)
            vals = np.interp(np.linspace(0, len(vals)-1, target_len), np.arange(len(vals)), vals)
            vals = (vals - vals.mean()) / (vals.std() + 1e-8)
            traces.append(vals)

    # Load frame-level avg intensity traces from HDF5 (subsampled)
    h5_files = discover_h5_files(h5_base)[:max_h5]
    for fp in h5_files:
        arr = load_avg_intensity_from_h5(fp, max_frames=target_len * 10)
        if arr is None:
            continue
        # Resample to target_len
        arr_rs = np.interp(np.linspace(0, len(arr)-1, target_len), np.arange(len(arr)), arr)
        traces.append(arr_rs.astype(np.float32))

    if not traces:
        return np.zeros((0, target_len), dtype=np.float32)
    return np.stack(traces)


class DilatedTCNBlock(nn.Module):
    """Dilated causal conv block used in TS2Vec-style encoder."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation,
                              padding=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.conv(x)))


class TS2VecEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 64, depth: int = 6, out_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        dilations = [2 ** i for i in range(depth)]
        self.blocks = nn.Sequential(*[DilatedTCNBlock(hidden, d) for d in dilations])
        self.output_proj = nn.Conv1d(hidden, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T) -> (B, out_dim, T)
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.output_proj(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pool temporal dim -> (B, out_dim) representation."""
        return self.forward(x).mean(dim=-1)


def hierarchical_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
                                   temporal_unit: int = 0) -> torch.Tensor:
    """
    TS2Vec hierarchical contrastive loss over temporal dimension.
    z1, z2: (B, D, T)
    """
    loss = 0.0
    d = 0
    while z1.size(-1) > 1:
        if d >= temporal_unit:
            loss += instance_contrastive(z1, z2)
        z1 = F.avg_pool1d(z1, kernel_size=2, stride=2, ceil_mode=True)
        z2 = F.avg_pool1d(z2, kernel_size=2, stride=2, ceil_mode=True)
        d += 1
    loss += instance_contrastive(z1, z2)
    return loss / (d + 1)


def instance_contrastive(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent loss: z1 and z2 are paired views, each (B, D, T). Positive = same-index pair."""
    B = z1.size(0)
    # Pool temporal dim -> (B, D)
    z1p = F.normalize(z1.mean(dim=-1), dim=1)
    z2p = F.normalize(z2.mean(dim=-1), dim=1)
    # Concatenate: [z1; z2] -> (2B, D)
    z = torch.cat([z1p, z2p], dim=0)
    sim = torch.mm(z, z.T) / temperature
    # Mask self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))
    # Positive for i (in z1 half) is i+B (in z2 half) and vice versa
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def crop_and_mask(x: torch.Tensor, crop_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two overlapping random crops from trace x: (B, 1, T)."""
    T = x.size(-1)
    crop_len = max(2, int(T * crop_ratio))
    s1 = np.random.randint(0, T - crop_len + 1)
    s2 = np.random.randint(0, T - crop_len + 1)
    noise = torch.randn_like(x) * 0.01
    x1 = (x + noise)[:, :, s1:s1 + crop_len]
    x2 = (x + noise)[:, :, s2:s2 + crop_len]
    # Pad back to T
    x1 = F.pad(x1, (0, T - crop_len))
    x2 = F.pad(x2, (0, T - crop_len))
    return x1, x2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-h5", type=int, default=5, help="Max HDF5 files for avg_intensity traces")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = os.path.join(args.derived_dir, "ts2vec")
    plots_dir = os.path.join(args.derived_dir, "plots")
    ensure_dirs(out_dir, plots_dir)

    particle_path = find_path([os.path.join(args.derived_dir, "particle_intensity_normalized.csv")])
    traces = collect_traces(particle_path, args.base_dir, max_h5=args.max_h5, target_len=args.target_len)

    if traces.shape[0] < 2:
        print(f"Warning: Only {traces.shape[0]} traces found — need at least 2 for contrastive training.")
        return
    print(f"Training on {traces.shape[0]} traces of length {traces.shape[1]}")

    model = TS2VecEncoder(in_channels=1, hidden=args.hidden, depth=args.depth, out_dim=args.out_dim)
    model = model.to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    X = torch.tensor(traces[:, np.newaxis, :], dtype=torch.float32)
    losses = []

    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(X.size(0))
        batch_losses = []
        for start in range(0, X.size(0), args.batch_size):
            batch = X[idx[start:start + args.batch_size]].to(device)
            x1, x2 = crop_and_mask(batch)
            z1 = model(x1)
            z2 = model(x2)
            loss = hierarchical_contrastive_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss.item()))
        sched.step()
        epoch_loss = float(np.mean(batch_losses))
        losses.append(epoch_loss)
        if ep % 50 == 0:
            print(f"  Epoch {ep}/{args.epochs}: loss={epoch_loss:.4f}")

    # Save encoder
    encoder_path = os.path.join(out_dir, "encoder_weights.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"hidden": args.hidden, "depth": args.depth, "out_dim": args.out_dim,
                   "target_len": args.target_len, "in_channels": 1},
        "epochs_trained": args.epochs,
        "final_loss": losses[-1],
    }, encoder_path)
    print(f"Saved: {encoder_path}")

    # Extract final representations
    model.eval()
    with torch.no_grad():
        Z = model.encode(X.to(device)).cpu().numpy()
    np.save(os.path.join(out_dir, "trace_embeddings.npy"), Z)

    metrics = {"final_loss": float(losses[-1]), "n_traces": int(traces.shape[0]),
               "trace_len": int(traces.shape[1]), "epochs": args.epochs}
    with open(os.path.join(out_dir, "pretrain_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=1.5, color="#1f77b4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title("TS2Vec Pretraining Loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ts2vec_loss.png"), dpi=200)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
