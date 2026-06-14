#!/usr/bin/env python3
# File: scripts/tier2_optical_echem_coupling.py
# Tier 2.4: Optical–dQ/dV coupling — which optical features LEAD dQ/dV peak shifts
# Method: lagged cross-correlation, canonical correlation analysis, lead-lag regression
# Reads: derived/particle_intensity_normalized.csv, derived/echem_per_cycle.csv
# Writes: derived/coupling/lag_correlations.csv, derived/coupling/results.json, derived/plots/coupling_*.png

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def find_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def compute_dqdv_features(echem_df: pd.DataFrame) -> pd.DataFrame:
    """Extract dQ/dV peak position, FWHM, amplitude per cycle."""
    # If echem_per_cycle already has these, pass through; else compute from capacity changes
    needed = ["cycleNo", "capacity_mAh", "V_max", "V_min"]
    if not all(c in echem_df.columns for c in needed):
        return pd.DataFrame()

    rows = []
    cycles = sorted(echem_df["cycleNo"].dropna().unique())

    for cyc in cycles:
        row = echem_df[echem_df["cycleNo"] == cyc].iloc[0]
        cap = float(row["capacity_mAh"]) if pd.notna(row["capacity_mAh"]) else np.nan
        v_max = float(row["V_max"]) if pd.notna(row["V_max"]) else np.nan
        v_min = float(row["V_min"]) if pd.notna(row["V_min"]) else np.nan
        v_range = v_max - v_min if (np.isfinite(v_max) and np.isfinite(v_min)) else np.nan
        dqdv_proxy = cap / v_range if (np.isfinite(cap) and np.isfinite(v_range) and v_range > 0) else np.nan

        rows.append({
            "cycleNo": cyc,
            "dqdv_proxy": dqdv_proxy,
            "V_max": v_max,
            "V_min": v_min,
            "V_range": v_range,
            "capacity_mAh": cap,
        })

    df = pd.DataFrame(rows).sort_values("cycleNo").reset_index(drop=True)

    # Smoothed peak proxy trend and FWHM proxy (V_range change)
    if df["dqdv_proxy"].notna().sum() >= 7:
        smooth = savgol_filter(df["dqdv_proxy"].fillna(df["dqdv_proxy"].median()),
                               window_length=min(7, len(df) if len(df) % 2 == 1 else len(df) - 1),
                               polyorder=2)
        df["dqdv_smooth"] = smooth
        df["dqdv_gradient"] = np.gradient(smooth)
    else:
        df["dqdv_smooth"] = df["dqdv_proxy"]
        df["dqdv_gradient"] = np.nan

    return df


def extract_optical_features(particle_df: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle optical summary features from all particles."""
    norm_cols = [c for c in particle_df.columns if c.endswith("_norm")]
    if not norm_cols:
        return pd.DataFrame()

    rows = []
    cycles = sorted(particle_df["cycleNo"].dropna().unique())
    for cyc in cycles:
        row = particle_df[particle_df["cycleNo"] == cyc]
        vals = []
        for col in norm_cols:
            v = pd.to_numeric(row[col], errors="coerce").values
            if v.size > 0 and np.isfinite(v[0]):
                vals.append(float(v[0]))

        if not vals:
            continue
        feat = {
            "cycleNo": cyc,
            "opt_mean": float(np.mean(vals)),
            "opt_std": float(np.std(vals)),
            "opt_CoV": float(np.std(vals) / np.mean(vals)) if np.mean(vals) != 0 else np.nan,
            "opt_min": float(np.min(vals)),
            "opt_max": float(np.max(vals)),
            "opt_range": float(np.max(vals) - np.min(vals)),
        }
        rows.append(feat)
    return pd.DataFrame(rows).sort_values("cycleNo").reset_index(drop=True)


def lagged_spearman(x: np.ndarray, y: np.ndarray, max_lag: int = 10) -> pd.DataFrame:
    """Compute Spearman correlation between x[:-lag] and y[lag:] for lag in 0..max_lag."""
    results = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x_, y_ = x, y
        else:
            x_, y_ = x[:-lag], y[lag:]
        finite = np.isfinite(x_) & np.isfinite(y_)
        if finite.sum() < 5:
            results.append({"lag": lag, "rho": np.nan, "pval": np.nan})
            continue
        rho, pval = spearmanr(x_[finite], y_[finite])
        results.append({"lag": lag, "rho": float(rho), "pval": float(pval)})
    return pd.DataFrame(results)


def run_cca(X: np.ndarray, Y: np.ndarray, n_components: int = 1) -> Tuple[float, np.ndarray, np.ndarray]:
    """Canonical correlation between X and Y. Returns first canonical correlation + loadings."""
    finite = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    if finite.sum() < max(X.shape[1] + Y.shape[1] + 2, 5):
        return np.nan, np.zeros(X.shape[1]), np.zeros(Y.shape[1])
    Xf, Yf = X[finite], Y[finite]
    scaler_x = StandardScaler().fit(Xf)
    scaler_y = StandardScaler().fit(Yf)
    Xs = scaler_x.transform(Xf)
    Ys = scaler_y.transform(Yf)
    n_comp = min(n_components, Xs.shape[1], Ys.shape[1])
    try:
        cca = CCA(n_components=n_comp)
        cca.fit(Xs, Ys)
        Xt, Yt = cca.transform(Xs, Ys)
        rho = float(np.corrcoef(Xt[:, 0], Yt[:, 0])[0, 1])
        return rho, cca.x_weights_[:, 0], cca.y_weights_[:, 0]
    except Exception as exc:
        print(f"  CCA failed: {exc}")
        return np.nan, np.zeros(X.shape[1]), np.zeros(Y.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--max-lag", type=int, default=10)
    args = parser.parse_args()

    coupling_dir = os.path.join(args.derived_dir, "coupling")
    plots_dir = os.path.join(args.derived_dir, "plots")
    ensure_dirs(coupling_dir, plots_dir)

    particle_path = find_path([os.path.join(args.derived_dir, "particle_intensity_normalized.csv")])
    echem_path = find_path([os.path.join(args.derived_dir, "echem_per_cycle.csv")])

    if not particle_path or not echem_path:
        print("Warning: Required input files not found.")
        return

    particle_df = pd.read_csv(particle_path)
    particle_df["cycleNo"] = pd.to_numeric(particle_df["cycleNo"], errors="coerce")
    echem_df = pd.read_csv(echem_path)
    echem_df["cycleNo"] = pd.to_numeric(echem_df["cycleNo"], errors="coerce")

    dqdv_df = compute_dqdv_features(echem_df)
    opt_df = extract_optical_features(particle_df)

    if dqdv_df.empty or opt_df.empty:
        print("Warning: Could not compute dQ/dV or optical features.")
        return

    # Merge on cycle
    merged = opt_df.merge(dqdv_df, on="cycleNo", how="inner").sort_values("cycleNo")
    print(f"Merged: {len(merged)} cycles with both optical and dQ/dV features.")

    opt_feature_cols = ["opt_mean", "opt_std", "opt_CoV", "opt_min", "opt_max", "opt_range"]
    dqdv_target_cols = ["dqdv_smooth", "dqdv_gradient", "V_range", "capacity_mAh"]

    opt_feature_cols = [c for c in opt_feature_cols if c in merged.columns]
    dqdv_target_cols = [c for c in dqdv_target_cols if c in merged.columns]

    # Lagged Spearman correlations for each optical-dQ/dV pair
    lag_records = []
    for opt_col in opt_feature_cols:
        for dqdv_col in dqdv_target_cols:
            x = merged[opt_col].to_numpy(dtype=float)
            y = merged[dqdv_col].to_numpy(dtype=float)
            lag_df = lagged_spearman(x, y, max_lag=args.max_lag)
            lag_df["optical_feature"] = opt_col
            lag_df["dqdv_feature"] = dqdv_col
            lag_records.append(lag_df)

    all_lags = pd.concat(lag_records, ignore_index=True)
    all_lags.to_csv(os.path.join(coupling_dir, "lag_correlations.csv"), index=False)

    # Best lag per pair
    best_lags = (all_lags.dropna(subset=["rho"])
                 .sort_values("pval")
                 .groupby(["optical_feature", "dqdv_feature"])
                 .first()
                 .reset_index())
    best_lags.to_csv(os.path.join(coupling_dir, "best_lags.csv"), index=False)

    # CCA
    X_mat = merged[opt_feature_cols].to_numpy(dtype=float)
    Y_mat = merged[dqdv_target_cols].to_numpy(dtype=float)
    cca_rho, x_weights, y_weights = run_cca(X_mat, Y_mat)
    print(f"CCA first canonical correlation: {cca_rho:.4f}")

    results = {
        "cca_first_canonical_rho": float(cca_rho) if np.isfinite(cca_rho) else None,
        "cca_x_weights": {col: float(w) for col, w in zip(opt_feature_cols, x_weights)},
        "cca_y_weights": {col: float(w) for col, w in zip(dqdv_target_cols, y_weights)},
        "best_lags_summary": best_lags[["optical_feature", "dqdv_feature", "lag", "rho", "pval"]].to_dict(orient="records"),
    }
    with open(os.path.join(coupling_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: lag correlation heatmap for each dQ/dV target
    fig, axes = plt.subplots(1, len(dqdv_target_cols), figsize=(4 * len(dqdv_target_cols), 5))
    if len(dqdv_target_cols) == 1:
        axes = [axes]

    for ax, dqdv_col in zip(axes, dqdv_target_cols):
        subset = all_lags[all_lags["dqdv_feature"] == dqdv_col].pivot(
            index="optical_feature", columns="lag", values="rho"
        )
        if subset.empty:
            continue
        im = ax.imshow(subset.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=[-0.5, args.max_lag + 0.5, -0.5, len(opt_feature_cols) - 0.5])
        ax.set_yticks(range(len(subset.index)))
        ax.set_yticklabels(subset.index, fontsize=8)
        ax.set_xlabel("Optical lag (cycles)")
        ax.set_title(f"Spearman ρ: optical→{dqdv_col}")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Optical features leading dQ/dV changes (lag=0 means same cycle)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "coupling_lag_heatmap.png"), dpi=200)
    plt.close(fig)

    # Plot: best-lag rho bar chart
    if not best_lags.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{r['optical_feature']}→{r['dqdv_feature']} (lag={r['lag']})" for _, r in best_lags.iterrows()]
        rhos = best_lags["rho"].to_numpy(dtype=float)
        colors = ["#d62728" if r > 0 else "#1f77b4" for r in rhos]
        ax.barh(range(len(labels)), rhos, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Best Spearman ρ at optimal lag")
        ax.set_title("Optical–dQ/dV coupling: best lag correlations")
        ax.axvline(0, color="k", linewidth=0.8)
        ax.grid(alpha=0.2, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "coupling_best_lags.png"), dpi=200)
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
