#!/usr/bin/env python3
# File: scripts/tier1_particle_intensity_eda.py
# Tier 1 EDA: particle intensity traces, normalization, abrupt-drop detection
# Reads: exampleParticles.csv, cycleFrames.csv, (optional) derived/echem_per_cycle.csv
# Writes: derived/particle_intensity_normalized.csv, derived/plots/particle_traces.png

import argparse
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def ensure_output_dirs(out_dir: str) -> str:
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def normalize_to_cycle2(series: pd.Series, cycle_series: pd.Series) -> Tuple[pd.Series, float]:
    baseline = np.nan
    mask = cycle_series == 2
    if mask.any():
        vals = series[mask].dropna()
        if not vals.empty:
            baseline = float(vals.iloc[0])
    if not np.isfinite(baseline) or baseline == 0:
        vals = series.dropna()
        if not vals.empty:
            baseline = float(vals.iloc[0])
    if not np.isfinite(baseline) or baseline == 0:
        return pd.Series(np.nan, index=series.index), np.nan
    return series.astype(float) / baseline, baseline


def detect_abrupt_drops(norm_series: pd.Series) -> Tuple[pd.Series, float]:
    delta = norm_series.diff()
    arr = delta.to_numpy(dtype=float)
    sigma = float(np.nanstd(arr))
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(False, index=norm_series.index), np.nan
    threshold = -2.0 * sigma
    return (delta < threshold).fillna(False), threshold


def read_cycle_frames_csv(path: str) -> pd.DataFrame:
    """Read cycle frame counts from either headered or legacy headerless CSV."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "cycleNo" in df.columns:
        return df

    raw = pd.read_csv(path, encoding="utf-8-sig", header=None)
    if raw.shape[1] < 2:
        raise ValueError("cycleFrames.csv must have at least two columns")
    raw = raw.iloc[:, :2].copy()
    raw.columns = ["cycleNo", "n_frames"]
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--echem-per-cycle-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    args = parser.parse_args()

    plots_dir = ensure_output_dirs(args.out_dir)

    particles_path = resolve_existing_path([
        args.particles_csv,
        "/scratch/<account>/<username>/Alek_Jiho/exampleParticles.csv",
        "/path/to/alek_jiho_nmc_deg/echemDF_full/exampleParticles.csv",
    ])
    frames_path = resolve_existing_path([
        args.cycle_frames_csv,
        "/scratch/<account>/<username>/Alek_Jiho/cycleFrames.csv",
        "/path/to/alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv",
    ])
    echem_path = resolve_existing_path([
        args.echem_per_cycle_csv,
        os.path.join(args.out_dir, "echem_per_cycle.csv"),
    ])

    if not particles_path:
        print("Warning: exampleParticles.csv not found.")
        return
    if not frames_path:
        print("Warning: cycleFrames.csv not found.")
        return

    try:
        particles_df = pd.read_csv(particles_path, encoding="utf-8-sig")
        frames_df = read_cycle_frames_csv(frames_path)
    except Exception as exc:
        print(f"Warning: Could not read input CSVs: {exc}")
        return

    if "cycleNo" not in particles_df.columns:
        print("Warning: 'cycleNo' missing from exampleParticles.csv.")
        return

    particle_cols = [c for c in particles_df.columns if c.lower().startswith("particle")]
    if not particle_cols:
        print("Warning: No particle columns found.")
        return

    particles_df["cycleNo"] = pd.to_numeric(particles_df["cycleNo"], errors="coerce")
    particles_df = particles_df.dropna(subset=["cycleNo"]).sort_values("cycleNo").reset_index(drop=True)
    for col in particle_cols:
        particles_df[col] = pd.to_numeric(particles_df[col], errors="coerce")

    frames_df["cycleNo"] = pd.to_numeric(frames_df["cycleNo"], errors="coerce")
    frame_cols = [c for c in frames_df.columns if c.lower().startswith("particle")]
    frames_df = frames_df.rename(columns={c: f"{c}_frames" for c in frame_cols})
    merged = particles_df.merge(frames_df, on="cycleNo", how="left")

    event_summary = []
    for pcol in particle_cols:
        norm_vals, baseline = normalize_to_cycle2(merged[pcol], merged["cycleNo"])
        merged[f"{pcol}_norm"] = norm_vals
        events, threshold = detect_abrupt_drops(merged[f"{pcol}_norm"])
        merged[f"{pcol}_abrupt_drop"] = events
        drop_cycles = merged.loc[events, "cycleNo"].tolist()
        event_summary.append((pcol, baseline, threshold, drop_cycles))

    out_csv = os.path.join(args.out_dir, "particle_intensity_normalized.csv")
    merged.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    cap_df = None
    if echem_path:
        try:
            cap_df = pd.read_csv(echem_path)
            cap_df["cycleNo"] = pd.to_numeric(cap_df["cycleNo"], errors="coerce")
            cap_df["capacity_mAh"] = pd.to_numeric(cap_df["capacity_mAh"], errors="coerce")
            cap_df = cap_df.dropna(subset=["cycleNo", "capacity_mAh"]).sort_values("cycleNo")
        except Exception as exc:
            print(f"Warning: Could not read echem CSV: {exc}")
            cap_df = None

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax1 = plt.subplots(figsize=(11, 6))

    for idx, pcol in enumerate(particle_cols):
        c = colors[idx % len(colors)]
        ax1.plot(merged["cycleNo"], merged[f"{pcol}_norm"],
                 marker="o", markersize=4, linewidth=1.7, color=c, label=f"{pcol} (norm)")
        ev_mask = merged[f"{pcol}_abrupt_drop"].fillna(False)
        if ev_mask.any():
            ax1.scatter(merged.loc[ev_mask, "cycleNo"], merged.loc[ev_mask, f"{pcol}_norm"],
                        marker="x", s=60, color=c, linewidths=1.8, zorder=5)

    ax1.set_xlabel("Cycle Number")
    ax1.set_ylabel("Normalized Intensity (Cycle 2 = 1.0)")
    ax1.set_title("Particle Intensity Traces — ✕ marks candidate cracking events")
    ax1.grid(alpha=0.25)

    if cap_df is not None and not cap_df.empty:
        ax2 = ax1.twinx()
        ax2.plot(cap_df["cycleNo"], cap_df["capacity_mAh"],
                 color="black", linestyle="--", linewidth=2.0, label="Capacity (mAh)", alpha=0.7)
        ax2.set_ylabel("Capacity (mAh)")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    else:
        ax1.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_png = os.path.join(plots_dir, "particle_traces.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")

    for pcol, baseline, threshold, drop_cycles in event_summary:
        print(f"{pcol}: baseline={baseline:.4g}  drop_threshold(Δ)<{threshold:.4g}  events@cycles={drop_cycles}")


if __name__ == "__main__":
    main()
