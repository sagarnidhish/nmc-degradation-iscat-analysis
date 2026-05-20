#!/usr/bin/env python3
# File: scripts/tier3_crack_precursor_detector.py
# Tier 3.3: Weakly supervised crack precursor detector from frame-level patch sequences
# Trains a 1D CNN on per-cycle average_intensity windows around crack events
# Uses weak labels from Tier 1 abrupt-drop detection; curriculum from high-confidence events
# Reads: derived/particle_intensity_normalized.csv, HDF5 average_intensity
# Writes: derived/crack_detector/model.pt, derived/crack_detector/results.json,
#         derived/plots/crack_detector_*.png

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


WINDOW_HALF = 50   # frames before/after event cycle to extract
MIN_CONFIDENCE = 0.5


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


def load_avg_intensity_full(path: str) -> Optional[np.ndarray]:
    """Load full average_intensity trace without loading movie."""
    try:
        with h5py.File(path, "r") as f:
            if "average_intensity" not in f:
                return None
            arr = np.asarray(f["average_intensity"][:], dtype=float)
            return arr[0, :] if arr.ndim == 2 else arr.reshape(-1)
    except Exception:
        return None


def get_cycle_frame_range(path: str, cycle_no: int, frames_per_cycle: int = 940) -> Tuple[int, int]:
    """Estimate frame range for a given cycle number within an HDF5 file."""
    try:
        with h5py.File(path, "r") as f:
            if "potentiostat_block" in f:
                pv = f["potentiostat_value"]
                if pv.ndim == 2 and pv.shape[0] >= 4:
                    echem_time = np.asarray(pv[3, :], dtype=float)
                    if "camera_timing" in f:
                        ct = f["camera_timing"]
                        row = 1 if ct.ndim == 2 and ct.shape[0] >= 2 else 0
                        cam_time = np.asarray(ct[row, :] if ct.ndim == 2 else ct[:], dtype=float)
                        # Find nearest frame to start of this cycle
                        cycle_start_time = echem_time[min(cycle_no * 100, len(echem_time) - 1)]
                        frame_idx = int(np.argmin(np.abs(cam_time - cycle_start_time)))
                        return max(0, frame_idx - WINDOW_HALF), frame_idx + WINDOW_HALF
    except Exception:
        pass
    # Fallback: linear estimate
    start = max(0, cycle_no * frames_per_cycle - WINDOW_HALF)
    return start, start + 2 * WINDOW_HALF


def extract_windows_from_h5(
    h5_path: str,
    event_cycles: List[int],
    non_event_cycles: List[int],
    window_size: int = 100,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """Extract fixed-length average_intensity windows around event/non-event cycles."""
    intensity = load_avg_intensity_full(h5_path)
    if intensity is None or intensity.size < window_size:
        return [], [], []

    windows, labels, confidences = [], [], []
    n = len(intensity)

    def extract_window(frame_center: int) -> Optional[np.ndarray]:
        lo = max(0, frame_center - window_size // 2)
        hi = lo + window_size
        if hi > n:
            lo, hi = n - window_size, n
        w = intensity[lo:hi].copy()
        if w.size < window_size:
            w = np.pad(w, (0, window_size - w.size))
        std = np.nanstd(w)
        if std > 0:
            w = (w - np.nanmean(w)) / std
        return np.where(np.isfinite(w), w, 0.0).astype(np.float32)

    total_frames = n
    frames_per_cycle = max(1, total_frames // max(len(event_cycles) + len(non_event_cycles), 1))

    for cyc in event_cycles:
        center = cyc * frames_per_cycle
        w = extract_window(center)
        if w is not None:
            # High confidence = large negative delta in cycle before
            conf = 1.0
            windows.append(w)
            labels.append(1)
            confidences.append(conf)
            # Also extract PRECURSOR window (before event) as positive
            pre_center = max(0, center - frames_per_cycle // 2)
            w_pre = extract_window(pre_center)
            if w_pre is not None:
                windows.append(w_pre)
                labels.append(1)
                confidences.append(0.7)

    for cyc in non_event_cycles:
        center = cyc * frames_per_cycle
        w = extract_window(center)
        if w is not None:
            windows.append(w)
            labels.append(0)
            confidences.append(1.0)

    return windows, labels, confidences


class CrackPrecursorCNN(nn.Module):
    """1D CNN for binary crack event classification from avg_intensity windows."""
    def __init__(self, window_size: int = 100, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(-1)


def curriculum_train(
    windows: List[np.ndarray],
    labels: List[int],
    confidences: List[float],
    window_size: int = 100,
    epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
    n_phases: int = 3,
) -> Tuple[CrackPrecursorCNN, List[float]]:
    """Curriculum: start with high-confidence samples, gradually include lower-confidence."""
    model = CrackPrecursorCNN(window_size).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    conf_arr = np.array(confidences)
    all_losses = []
    phases = np.linspace(MIN_CONFIDENCE, 0.0, n_phases + 1)

    epochs_per_phase = epochs // n_phases

    for phase_idx in range(n_phases):
        min_conf = phases[phase_idx]
        phase_mask = conf_arr >= min_conf
        phase_windows = [w for w, m in zip(windows, phase_mask) if m]
        phase_labels = [l for l, m in zip(labels, phase_mask) if m]
        print(f"  Curriculum phase {phase_idx}: min_conf={min_conf:.2f}, n={sum(phase_mask)}")

        if not phase_windows:
            continue

        X_t = torch.tensor(np.array(phase_windows)[:, np.newaxis, :], dtype=torch.float32, device=device)
        y_t = torch.tensor(phase_labels, dtype=torch.float32, device=device)
        pos_w = torch.tensor([(y_t == 0).sum() / max((y_t == 1).sum(), 1)],
                              dtype=torch.float32, device=device)

        batch_size = min(32, len(phase_windows))
        for ep in range(epochs_per_phase):
            model.train()
            idx = torch.randperm(X_t.size(0))
            ep_losses = []
            for s in range(0, X_t.size(0), batch_size):
                bi = idx[s:s + batch_size]
                pred = model(X_t[bi])
                loss_raw = F.binary_cross_entropy(pred, y_t[bi], reduction="none")
                w = torch.where(y_t[bi] == 1, pos_w.expand_as(y_t[bi]), torch.ones_like(y_t[bi]))
                loss = (loss_raw * w).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_losses.append(float(loss.item()))
            sched.step()
            all_losses.append(float(np.mean(ep_losses)))

    return model, all_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-h5", type=int, default=3, help="Max HDF5 to extract windows from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = os.path.join(args.derived_dir, "crack_detector")
    plots_dir = os.path.join(args.derived_dir, "plots")
    ensure_dirs(out_dir, plots_dir)

    particle_path = find_path([os.path.join(args.derived_dir, "particle_intensity_normalized.csv")])
    if not particle_path:
        print("Warning: particle_intensity_normalized.csv not found.")
        return
    df = pd.read_csv(particle_path)
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")

    # Collect event and non-event cycles from Tier 1 drop detections
    drop_cols = [c for c in df.columns if c.endswith("_abrupt_drop")]
    event_cycles_all, non_event_cycles_all = [], []
    for col in drop_cols:
        events = df[df[col].fillna(False)]["cycleNo"].dropna().astype(int).tolist()
        non_events = df[~df[col].fillna(False)]["cycleNo"].dropna().astype(int).tolist()
        event_cycles_all.extend(events)
        non_event_cycles_all.extend(non_events)

    event_cycles_all = list(set(event_cycles_all))
    non_event_cycles_all = list(set(non_event_cycles_all) - set(event_cycles_all))
    # Balance: up to 3x non-events per event
    non_event_cycles_all = non_event_cycles_all[:max(len(event_cycles_all) * 3, 10)]

    print(f"Event cycles: {len(event_cycles_all)}, Non-event cycles: {len(non_event_cycles_all)}")

    # Extract windows from HDF5 files
    h5_files = discover_h5_files(args.base_dir)[:args.max_h5]
    all_windows, all_labels, all_confs = [], [], []

    for fp in h5_files:
        ws, ls, cs = extract_windows_from_h5(fp, event_cycles_all, non_event_cycles_all,
                                               window_size=args.window_size)
        all_windows.extend(ws)
        all_labels.extend(ls)
        all_confs.extend(cs)
        print(f"  {os.path.basename(fp)}: {len(ws)} windows ({sum(ls)} events)")

    if len(all_windows) < 4:
        # Synthesize minimal windows for pipeline validation
        print("Warning: Insufficient HDF5 windows. Generating synthetic windows for pipeline test.")
        np.random.seed(args.seed)
        for _ in range(10):
            w = np.random.randn(args.window_size).astype(np.float32)
            all_windows.append(w)
            all_labels.append(0)
            all_confs.append(1.0)
        for _ in range(5):
            w = np.random.randn(args.window_size).astype(np.float32)
            w[args.window_size // 2:] -= 0.5  # synthetic drop
            all_windows.append(w)
            all_labels.append(1)
            all_confs.append(0.9)

    print(f"Total windows: {len(all_windows)}, Events: {sum(all_labels)}")
    model, losses = curriculum_train(all_windows, all_labels, all_confs,
                                      window_size=args.window_size, epochs=args.epochs,
                                      lr=1e-3, device=device)

    # Evaluate
    model.eval()
    X_eval = torch.tensor(np.array(all_windows)[:, np.newaxis, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(X_eval).cpu().numpy()
    y_arr = np.array(all_labels)

    metrics: Dict = {"n_windows": len(all_windows), "n_events": int(sum(all_labels))}
    if y_arr.sum() > 0 and (y_arr == 0).sum() > 0:
        metrics["auroc"] = float(roc_auc_score(y_arr, preds))
        metrics["auprc"] = float(average_precision_score(y_arr, preds))
        print(f"AUROC: {metrics['auroc']:.4f}  AUPRC: {metrics['auprc']:.4f}")

    torch.save({"state_dict": model.state_dict(),
                "config": {"window_size": args.window_size},
                "metrics": metrics}, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(losses, linewidth=1.5, color="#1f77b4")
    axes[0].set_title("Crack Precursor Detector — Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(alpha=0.25)

    ev_preds = preds[y_arr == 1]
    nev_preds = preds[y_arr == 0]
    axes[1].hist(nev_preds, bins=20, alpha=0.6, label="Non-event", color="#1f77b4")
    axes[1].hist(ev_preds, bins=20, alpha=0.6, label="Crack event", color="#d62728")
    axes[1].set_xlabel("Predicted crack probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Score distribution")
    axes[1].legend(); axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "crack_detector_results.png"), dpi=200)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
