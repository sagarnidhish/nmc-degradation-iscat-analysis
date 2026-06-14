#!/usr/bin/env python3
# File: scripts/tier1_echem_eda.py
# Tier 1 EDA: electrochemistry per-cycle metrics + dQ/dV heatmap
# Reads: echemDF_full.csv (11M rows, chunked)
# Writes: derived/echem_per_cycle.csv, derived/plots/capacity_fade.png, derived/plots/dqdv_heatmap.png

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def resolve_existing_path(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def ensure_output_dirs(out_dir: str) -> str:
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _safe_savgol(y: np.ndarray, preferred_window: int, polyorder: int) -> np.ndarray:
    n = y.size
    if n < 5:
        return y
    window = min(preferred_window, n if n % 2 == 1 else n - 1)
    if window < 5:
        return y
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
        if window > n:
            window = n if n % 2 == 1 else n - 1
    if window < 5 or window <= polyorder:
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")


def process_cycle(
    cycle_no: float,
    cdf: pd.DataFrame,
    preferred_window: int,
    polyorder: int,
) -> Tuple[Dict, Optional[Tuple]]:
    if cdf.empty:
        return {}, None

    time_col = "Computer time (s)" if "Computer time (s)" in cdf.columns else "Time (s)"
    cols_needed = [c for c in [time_col, "Potential (V)", "Current (mA)"] if c in cdf.columns]
    cdf = cdf[cols_needed].dropna()
    if cdf.empty or "Potential (V)" not in cdf.columns or "Current (mA)" not in cdf.columns:
        return {}, None

    cdf = cdf.sort_values(time_col, kind="mergesort")
    t = cdf[time_col].to_numpy(dtype=float)
    v = cdf["Potential (V)"].to_numpy(dtype=float)
    i_ma = cdf["Current (mA)"].to_numpy(dtype=float)

    dt = np.diff(t, prepend=t[0])
    dt[~np.isfinite(dt)] = 0.0
    dt[dt < 0] = 0.0

    q_charge_mAh = float(np.nansum(np.clip(i_ma, 0, None) * dt) / 3600.0)
    q_discharge_mAh = float(-np.nansum(np.clip(i_ma, None, 0) * dt) / 3600.0)
    capacity_mAh = q_discharge_mAh if q_discharge_mAh > 0 else q_charge_mAh
    ce_pct = (q_discharge_mAh / q_charge_mAh * 100.0) if q_charge_mAh > 0 else np.nan

    metrics = {
        "cycleNo": cycle_no,
        "charge_capacity_mAh": q_charge_mAh,
        "discharge_capacity_mAh": q_discharge_mAh,
        "capacity_mAh": capacity_mAh,
        "coulombic_efficiency_pct": ce_pct,
        "V_min": float(np.nanmin(v)) if v.size else np.nan,
        "V_max": float(np.nanmax(v)) if v.size else np.nan,
        "n_points": int(v.size),
    }

    if v.size < 15 or (np.nanmax(v) - np.nanmin(v)) < 1e-6:
        return metrics, None

    q_abs = np.cumsum(np.abs(i_ma) * dt) / 3600.0
    v_s = _safe_savgol(v, preferred_window, polyorder)
    q_s = _safe_savgol(q_abs, preferred_window, polyorder)

    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.gradient(q_s, v_s, edge_order=1)

    mask = np.isfinite(v_s) & np.isfinite(dqdv)
    if not np.any(mask):
        return metrics, None
    return metrics, (cycle_no, v_s[mask], dqdv[mask])


def make_capacity_plot(metrics_df: pd.DataFrame, out_png: str) -> None:
    if metrics_df.empty:
        return
    x = metrics_df["cycleNo"].to_numpy(dtype=float)
    cap = metrics_df["capacity_mAh"].to_numpy(dtype=float)
    ce = metrics_df["coulombic_efficiency_pct"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, cap, marker="o", linewidth=1.8, markersize=4, color="#1f77b4")
    axes[0].set_ylabel("Capacity (mAh)")
    axes[0].set_title("Capacity Fade vs Cycle")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, ce, marker="o", linewidth=1.8, markersize=4, color="#d62728")
    axes[1].set_xlabel("Cycle Number")
    axes[1].set_ylabel("Coulombic Efficiency (%)")
    axes[1].set_title("Coulombic Efficiency vs Cycle")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")


def make_dqdv_heatmap(
    profiles: list,
    metrics_df: pd.DataFrame,
    voltage_bins: int,
    out_png: str,
) -> None:
    if not profiles or metrics_df.empty:
        print("Warning: No dQ/dV data; skipping heatmap.")
        return
    v_min = float(pd.to_numeric(metrics_df["V_min"], errors="coerce").min())
    v_max = float(pd.to_numeric(metrics_df["V_max"], errors="coerce").max())
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        print("Warning: Invalid voltage range; skipping heatmap.")
        return

    cycles = metrics_df["cycleNo"].to_numpy(dtype=float)
    bins = np.linspace(v_min, v_max, voltage_bins + 1)
    heatmap = np.full((cycles.size, voltage_bins), np.nan)
    row_map = {float(c): i for i, c in enumerate(cycles)}

    for cyc, v, dqdv in profiles:
        row = row_map.get(float(cyc))
        if row is None:
            continue
        bidx = np.digitize(v, bins) - 1
        valid = (bidx >= 0) & (bidx < voltage_bins) & np.isfinite(dqdv)
        for b in np.unique(bidx[valid]):
            heatmap[row, b] = np.nanmean(dqdv[valid][bidx[valid] == b])

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [bins[0], bins[-1], float(np.nanmax(cycles)), float(np.nanmin(cycles))]
    im = ax.imshow(heatmap, aspect="auto", extent=extent, cmap="viridis")
    fig.colorbar(im, ax=ax, label="dQ/dV (mAh/V)")
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Cycle Number")
    ax.set_title("dQ/dV Heatmap by Cycle")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--echem-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--sgolay-window", type=int, default=31)
    parser.add_argument("--sgolay-polyorder", type=int, default=3)
    parser.add_argument("--voltage-bins", type=int, default=200)
    args = parser.parse_args()

    plots_dir = ensure_output_dirs(args.out_dir)

    csv_path = resolve_existing_path([
        args.echem_csv.strip() if args.echem_csv else "",
        "/scratch/<account>/<username>/Alek_Jiho/alek_jiho_nmc_deg/echemDF_full/echemDF_full.csv",
        "/scratch/<account>/<username>/Alek_Jiho/echemDF_full/echemDF_full.csv",
        "/scratch/<account>/<username>/Alek_Jiho/echemDF_full.csv",
        "/path/to/alek_jiho_nmc_deg/echemDF_full/echemDF_full.csv",
    ])

    # Fallback: concatenate individual per-session electrochemistry CSVs from dataset dirs
    if not csv_path:
        import glob as _glob
        indiv_csvs = sorted(_glob.glob(
            "/scratch/<account>/<username>/Alek_Jiho/**/*electrochemistry.csv", recursive=True
        ))
        if indiv_csvs:
            print(f"echemDF_full.csv not found. Merging {len(indiv_csvs)} individual echem CSVs...")
            dfs = []
            for i, p in enumerate(indiv_csvs):
                try:
                    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
                    # Infer cycleNo from filename number prefix (e.g. "2_c2_x14..." -> crude proxy)
                    if "cycleNo" not in df.columns:
                        df["cycleNo"] = float(i * 10)
                    df["addrs"] = p
                    dfs.append(df)
                except Exception as exc:
                    print(f"  Warning: {p}: {exc}")
            if dfs:
                merged_csv = os.path.join(args.out_dir, "echem_merged.csv")
                pd.concat(dfs, ignore_index=True).to_csv(merged_csv, index=False)
                csv_path = merged_csv
                print(f"Wrote merged echem: {merged_csv}")

    if not csv_path:
        print("Warning: No echem CSV found (echemDF_full.csv or individual *electrochemistry.csv).")
        return
    print(f"Reading: {csv_path}")

    usecols = ["Computer time (s)", "Time (s)", "Potential (V)", "Current (mA)", "cycleNo"]
    metrics_records = []
    dqdv_profiles = []
    pending = pd.DataFrame()

    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=args.chunksize,
            usecols=lambda c: c in usecols,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except Exception as exc:
        print(f"Warning: Could not open CSV: {exc}")
        return

    for chunk_idx, chunk in enumerate(reader, 1):
        if chunk.empty or "Potential (V)" not in chunk.columns or "Current (mA)" not in chunk.columns:
            continue
        if "cycleNo" not in chunk.columns:
            chunk["cycleNo"] = float(chunk_idx)
        chunk["cycleNo"] = pd.to_numeric(chunk["cycleNo"], errors="coerce")
        chunk = chunk.dropna(subset=["cycleNo", "Potential (V)", "Current (mA)"])
        if chunk.empty:
            continue

        data = pd.concat([pending, chunk], ignore_index=True)
        last_cycle = data["cycleNo"].iloc[-1]
        complete = data[data["cycleNo"] != last_cycle]
        pending = data[data["cycleNo"] == last_cycle].copy()

        for cyc, cdf in complete.groupby("cycleNo", sort=False):
            rec, profile = process_cycle(float(cyc), cdf, args.sgolay_window, args.sgolay_polyorder)
            if rec:
                metrics_records.append(rec)
            if profile is not None:
                dqdv_profiles.append(profile)

        if chunk_idx % 5 == 0:
            print(f"  chunk {chunk_idx} done...")

    for cyc, cdf in pending.groupby("cycleNo", sort=False):
        rec, profile = process_cycle(float(cyc), cdf, args.sgolay_window, args.sgolay_polyorder)
        if rec:
            metrics_records.append(rec)
        if profile is not None:
            dqdv_profiles.append(profile)

    if not metrics_records:
        print("Warning: No cycle records computed.")
        return

    metrics_df = pd.DataFrame(metrics_records).sort_values("cycleNo").reset_index(drop=True)
    out_csv = os.path.join(args.out_dir, "echem_per_cycle.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    make_capacity_plot(metrics_df, os.path.join(plots_dir, "capacity_fade.png"))
    make_dqdv_heatmap(dqdv_profiles, metrics_df, args.voltage_bins, os.path.join(plots_dir, "dqdv_heatmap.png"))


if __name__ == "__main__":
    main()
