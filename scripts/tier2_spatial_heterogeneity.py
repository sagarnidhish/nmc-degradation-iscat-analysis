#!/usr/bin/env python3
# File: scripts/tier2_spatial_heterogeneity.py
# Tier 2.2: Spatial heterogeneity of particle degradation
# Method: 1D CNN encoder on intensity traces -> latent vectors -> HDBSCAN + Moran's I
# Reads: derived/particle_intensity_normalized.csv, derived/h5_inventory.csv
# Writes: derived/spatial/cluster_assignments.csv, derived/spatial/morans_i.json, derived/plots/spatial_*.png

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def find_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


class TraceEncoder(nn.Module):
    """1D CNN encoder: (B, 1, T) -> latent vector of dim latent_dim."""
    def __init__(self, seq_len: int, latent_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, latent_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ContrastiveTraceModel(nn.Module):
    def __init__(self, seq_len: int, latent_dim: int = 32, proj_dim: int = 16):
        super().__init__()
        self.encoder = TraceEncoder(seq_len, latent_dim)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encode(x))


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return nn.functional.cross_entropy(sim, labels)


def augment_trace(trace: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    """Simple augmentation: Gaussian noise + small dropout."""
    noisy = trace + torch.randn_like(trace) * noise_std
    mask = torch.rand_like(trace) > 0.1
    return noisy * mask


def train_encoder(traces: np.ndarray, latent_dim: int = 32, epochs: int = 200,
                  lr: float = 1e-3, device: str = "cpu") -> ContrastiveTraceModel:
    T = traces.shape[1]
    model = ContrastiveTraceModel(T, latent_dim).to(device).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X = torch.tensor(traces[:, np.newaxis, :], dtype=torch.float32, device=device)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z1 = model.project(augment_trace(X))
        z2 = model.project(augment_trace(X))
        loss = nt_xent_loss(z1, z2)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 50 == 0:
            print(f"  Encoder epoch {ep}: loss={loss.item():.4f}")

    return model


def morans_i(values: np.ndarray, coords: np.ndarray, k: int = 3) -> Tuple[float, float]:
    """Compute Moran's I from (N,2) coordinates and (N,) values. Returns (I, permutation p-value)."""
    N = len(values)
    if N < 4:
        return np.nan, np.nan

    # Build k-nearest-neighbour weight matrix
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(coords)
    W = np.zeros((N, N))
    _, idx = nbrs.kneighbors(coords)
    for i, row in enumerate(idx):
        for j in row[1:]:
            W[i, j] = 1.0
            W[j, i] = 1.0
    W_row = W.sum(axis=1, keepdims=True)
    W_row[W_row == 0] = 1.0
    W_norm = W / W_row

    z = values - values.mean()
    I = float(N / W.sum() * (z @ W @ z) / (z @ z))

    # Permutation test
    n_perm = 999
    perm_I = np.array([
        float(N / W.sum() * (np.random.permutation(z) @ W @ np.random.permutation(z)) / (z @ z))
        for _ in range(n_perm)
    ])
    p_val = float((np.abs(perm_I) >= np.abs(I)).mean())
    return I, p_val


def hdbscan_cluster(Z: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    """HDBSCAN clustering — falls back to KMeans if hdbscan not available."""
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        return clusterer.fit_predict(Z)
    except ImportError:
        from sklearn.cluster import KMeans
        n_clusters = max(2, min(5, len(Z) // 2))
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(Z)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    spatial_dir = os.path.join(args.derived_dir, "spatial")
    plots_dir = os.path.join(args.derived_dir, "plots")
    ensure_dirs(spatial_dir, plots_dir)

    particle_path = find_path([os.path.join(args.derived_dir, "particle_intensity_normalized.csv")])
    if not particle_path:
        print("Warning: particle_intensity_normalized.csv not found.")
        return
    df = pd.read_csv(particle_path)
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    df = df.sort_values("cycleNo").reset_index(drop=True)

    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    if not norm_cols:
        print("Warning: No _norm columns found.")
        return

    # Build (n_particles, n_cycles) matrix — pad/interpolate NaNs
    traces = []
    particle_names = []
    for col in norm_cols:
        t = df[col].to_numpy(dtype=float)
        finite = np.isfinite(t)
        if finite.sum() < 5:
            continue
        # Linear interpolation of NaNs
        x = np.arange(len(t))
        t_interp = np.interp(x, x[finite], t[finite])
        traces.append(t_interp)
        particle_names.append(col)

    if len(traces) < 2:
        print("Warning: Fewer than 2 valid particle traces.")
        return

    traces = np.array(traces, dtype=np.float32)

    # Degradation rate per particle: slope of normalized intensity vs cycle
    cycles = df["cycleNo"].to_numpy(dtype=float)
    deg_rates = []
    for t in traces:
        finite = np.isfinite(t)
        if finite.sum() >= 3:
            slope = float(np.polyfit(cycles[finite], t[finite], 1)[0])
        else:
            slope = np.nan
        deg_rates.append(slope)
    deg_rates = np.array(deg_rates)

    # Synthetic spatial coordinates from particle index (real coords need stage_position per particle)
    # We use a 1D layout as placeholder; replace with real XY from h5 stage_position if available
    n_particles = len(traces)
    coords = np.column_stack([np.arange(n_particles), np.zeros(n_particles)]).astype(float)
    print(f"Note: Using 1D synthetic coords. Replace with per-particle XY from HDF5 stage_position for real spatial analysis.")

    print(f"Training contrastive encoder on {n_particles} particle traces, T={traces.shape[1]}")
    model = train_encoder(traces, latent_dim=args.latent_dim, epochs=args.epochs, device=device)
    model.eval()
    with torch.no_grad():
        X = torch.tensor(traces[:, np.newaxis, :], dtype=torch.float32, device=device)
        Z = model.encode(X).cpu().numpy()

    # PCA for visualisation
    pca = PCA(n_components=min(2, Z.shape[1]))
    Z_2d = pca.fit_transform(Z)

    labels = hdbscan_cluster(Z, min_cluster_size=2)

    # Moran's I on degradation rates with 1D coords
    valid_mask = np.isfinite(deg_rates)
    if valid_mask.sum() >= 4:
        I, p_val = morans_i(deg_rates[valid_mask], coords[valid_mask])
    else:
        I, p_val = np.nan, np.nan

    # Save results
    cluster_df = pd.DataFrame({
        "particle": particle_names,
        "degradation_rate_per_cycle": deg_rates,
        "cluster_label": labels,
        "latent_pc1": Z_2d[:, 0],
        "latent_pc2": Z_2d[:, 1] if Z_2d.shape[1] > 1 else 0.0,
    })
    cluster_df.to_csv(os.path.join(spatial_dir, "cluster_assignments.csv"), index=False)

    morans_result = {"morans_I": float(I) if np.isfinite(I) else None,
                     "p_value_permutation": float(p_val) if np.isfinite(p_val) else None,
                     "n_particles": n_particles,
                     "note": "coords are 1D synthetic; replace with real XY from stage_position"}
    with open(os.path.join(spatial_dir, "morans_i.json"), "w") as f:
        json.dump(morans_result, f, indent=2)
    print(f"Moran's I={I:.4f}  p={p_val:.4f}")

    # Plot latent space
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(Z_2d[:, 0], Z_2d[:, 1] if Z_2d.shape[1] > 1 else Z_2d[:, 0],
                         c=labels, cmap="tab10", s=80, alpha=0.9)
    for i, name in enumerate(particle_names):
        axes[0].annotate(name.replace("_norm", ""), (Z_2d[i, 0], Z_2d[i, 1] if Z_2d.shape[1] > 1 else Z_2d[i, 0]),
                         fontsize=7, alpha=0.7)
    axes[0].set_title("Particle Clusters in Latent Space (PCA)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(sc, ax=axes[0], label="Cluster")

    bar_colors = ["#d62728" if r < 0 else "#1f77b4" for r in deg_rates]
    axes[1].bar(range(n_particles), deg_rates, color=bar_colors)
    axes[1].set_xticks(range(n_particles))
    axes[1].set_xticklabels([n.replace("_norm", "") for n in particle_names], rotation=45, ha="right")
    axes[1].set_ylabel("Degradation rate (Δ norm. intensity/cycle)")
    axes[1].set_title(f"Per-particle degradation rates\nMoran's I={I:.3f}, p={p_val:.3f}")
    axes[1].grid(alpha=0.2, axis="y")
    axes[1].axhline(0, color="k", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "spatial_heterogeneity.png"), dpi=200)
    plt.close(fig)
    print(f"Saved outputs to {spatial_dir}")


if __name__ == "__main__":
    main()
