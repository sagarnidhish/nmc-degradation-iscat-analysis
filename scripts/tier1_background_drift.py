#!/usr/bin/env python3
# File: scripts/tier1_background_drift.py
# Tier 1: background intensity drift analysis from average_intensity + camera_timing
# Selects one representative file per session, fits linear drift, marks cycle boundaries
# Writes: derived/plots/background_drift_{filename}.png

import argparse
import os
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_output_dirs(out_dir: str) -> str:
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def discover_h5_files(base_dir: str) -> List[str]:
    results = []
    skip = {"derived", ".git", "__pycache__"}
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for fn in files:
            if fn.lower().endswith((".h5", ".hdf5")):
                results.append(os.path.join(root, fn))
    return sorted(results)


def session_of(base_dir: str, path: str) -> str:
    rel = os.path.relpath(path, base_dir)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else "root"


def pick_representatives(base_dir: str, files: List[str]) -> List[str]:
    by_sess: Dict[str, List[str]] = {}
    for fp in files:
        by_sess.setdefault(session_of(base_dir, fp), []).append(fp)
    # pick middle-ish file from each session to get representative cycling data
    selected = []
    for sess in sorted(by_sess.keys()):
        group = sorted(by_sess[sess])
        selected.append(group[len(group) // 2])
    return selected


def get_block_boundaries(grp: h5py.Group) -> np.ndarray:
    # Look for potentiostat_block or echem_dataframe to find cycle/block change times
    time_keys = ["block0_values", "axis1"]
    block_keys = ["potentiostat_block"]

    # Try potentiostat_value row 3 (time) directly from parent file — not available here
    # Use echem_dataframe/block0_values column 0 as time proxy
    for k in time_keys:
        if k in grp:
            try:
                arr = np.asarray(grp[k][:], dtype=float)
                if arr.ndim == 2:
                    arr = arr[:, 0]
                arr = arr[np.isfinite(arr)]
                if arr.size > 10:
                    # Find rough cycle boundaries by evenly splitting
                    return np.linspace(arr[0], arr[-1], 12)[1:-1]
            except Exception:
                continue
    return np.array([])


def load_intensity_and_time(path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    try:
        with h5py.File(path, "r") as f:
            if "average_intensity" not in f:
                print(f"  Warning: no average_intensity in {os.path.basename(path)}")
                return None

            avg = np.asarray(f["average_intensity"][:], dtype=float)
            intensity = avg[0, :] if avg.ndim == 2 else avg.reshape(-1)

            if "camera_timing" in f and isinstance(f["camera_timing"], h5py.Dataset):
                ct = f["camera_timing"]
                row = 1 if ct.ndim == 2 and ct.shape[0] >= 2 else 0
                t = np.asarray(ct[row, :] if ct.ndim == 2 else ct[:], dtype=float)
            else:
                t = np.arange(intensity.size, dtype=float)

            n = min(intensity.size, t.size)
            intensity, t = intensity[:n], t[:n]

            # Get cycle count from potentiostat_value shape
            n_cycles = 0
            boundary_times = np.array([])
            if "potentiostat_value" in f:
                try:
                    pv = f["potentiostat_value"]
                    # row 3 = time axis for echem
                    if pv.ndim == 2 and pv.shape[0] >= 4:
                        echem_t = np.asarray(pv[3, :], dtype=float)
                        echem_t = echem_t[np.isfinite(echem_t)]
                    elif pv.ndim == 2 and pv.shape[0] >= 1:
                        echem_t = np.asarray(pv[0, :], dtype=float)
                        echem_t = echem_t[np.isfinite(echem_t)]
                    else:
                        echem_t = np.array([])

                    if "potentiostat_block" in f:
                        blocks = f["potentiostat_block"]
                        if blocks.ndim == 2:
                            raw = np.asarray(blocks[0, :])
                        else:
                            raw = np.asarray(blocks[:])
                        decoded = np.array([
                            x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x)
                            for x in raw
                        ])
                        m = min(decoded.size, echem_t.size)
                        if m >= 2:
                            ch = np.where(decoded[1:m] != decoded[:m-1])[0] + 1
                            if ch.size > 0 and echem_t.size > 0:
                                # align echem time to camera time
                                t0_offset = t[0] - echem_t[0] if echem_t.size > 0 else 0.0
                                boundary_times = echem_t[ch] + t0_offset
                                n_cycles = int(ch.size + 1)
                except Exception:
                    pass

            if n_cycles == 0:
                # Rough estimate: assume ~10s/frame, typical ~940 frames/cycle
                fps = 1.0 / np.median(np.diff(t[np.isfinite(t)][:200])) if t.size > 10 else 0.1
                n_cycles = max(1, int(n / max(1, fps * 94)))  # 940 frames @ C/2

            return intensity, t, boundary_times, n_cycles
    except Exception as exc:
        print(f"  Warning: {path}: {exc}")
        return None


def plot_drift(path: str, plots_dir: str) -> None:
    payload = load_intensity_and_time(path)
    if payload is None:
        return

    intensity, t, boundaries, n_cycles = payload
    mask = np.isfinite(intensity) & np.isfinite(t)
    if mask.sum() < 2:
        print(f"  Warning: Insufficient data in {os.path.basename(path)}")
        return

    x = t[mask]
    y = intensity[mask]
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    x_rel = x - x[0]

    coeff = np.polyfit(x_rel, y, deg=1)
    y_fit = np.polyval(coeff, x_rel)

    total_drift_pct = ((y_fit[-1] - y_fit[0]) / y_fit[0] * 100.0) if y_fit[0] != 0 else np.nan
    drift_per_cycle = total_drift_pct / n_cycles if (n_cycles > 0 and np.isfinite(total_drift_pct)) else np.nan

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_rel, y, linewidth=1.2, color="#1f77b4", alpha=0.85, label="Avg intensity")
    ax.plot(x_rel, y_fit, linewidth=2.2, color="#d62728", linestyle="--", label="Linear fit")
    for bt in boundaries:
        ax.axvline(x=bt - x[0], color="gray", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Time relative to start (s)")
    ax.set_ylabel("Average Frame Intensity (a.u.)")
    ax.set_title(f"Background Drift: {os.path.basename(path)}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    txt_lines = [f"Slope: {coeff[0]:.4g} a.u./s"]
    if np.isfinite(total_drift_pct):
        txt_lines.append(f"Total drift: {total_drift_pct:.3f}%")
    if np.isfinite(drift_per_cycle):
        txt_lines.append(f"Drift rate: {drift_per_cycle:.4f}%/cycle")
    txt_lines.append(f"Est. cycles: {n_cycles}")
    ax.text(0.02, 0.98, "\n".join(txt_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"))

    fig.tight_layout()
    safe_name = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
    out_png = os.path.join(plots_dir, f"background_drift_{safe_name}.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  {safe_name}: slope={coeff[0]:.4g} a.u./s  total_drift={total_drift_pct:.3f}%  rate={drift_per_cycle:.4f}%/cycle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/scratch/<account>/<username>/Alek_Jiho")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    args = parser.parse_args()

    plots_dir = ensure_output_dirs(args.out_dir)
    files = discover_h5_files(args.base_dir)
    if not files:
        print(f"Warning: No HDF5 files found under {args.base_dir}")
        return

    selected = pick_representatives(args.base_dir, files)
    print(f"Selected {len(selected)} representative files (one per session):")
    for fp in selected:
        print(f"  {os.path.relpath(fp, args.base_dir)}")

    for fp in selected:
        plot_drift(fp, plots_dir)


if __name__ == "__main__":
    main()
