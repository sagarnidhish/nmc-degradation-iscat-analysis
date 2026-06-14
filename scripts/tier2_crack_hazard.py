#!/usr/bin/env python3
# File: scripts/tier2_crack_hazard.py
# Tier 2.3: Discrete-time survival model for crack-like intensity drop events
# Predicts per-cycle hazard of an abrupt drop event given optical+echem covariates
# Reads: derived/particle_intensity_normalized.csv, derived/echem_per_cycle.csv
# Writes: derived/hazard/hazard_results.json, derived/plots/hazard_*.png

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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def find_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def build_survival_dataset(
    particle_df: pd.DataFrame,
    echem_df: pd.DataFrame,
    window: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build discrete-time survival dataset.
    Each row = (particle, cycle) with:
      - event=1 if abrupt drop at this cycle
      - time-varying covariates: rolling optical stats + echem features
    """
    norm_cols = [c for c in particle_df.columns if c.endswith("_norm")]
    drop_cols = [c for c in particle_df.columns if c.endswith("_abrupt_drop")]

    rows = []
    for norm_col, drop_col in zip(norm_cols, drop_cols):
        particle_id = norm_col.replace("_norm", "")
        cycles = particle_df["cycleNo"].to_numpy(dtype=float)
        intensity = particle_df[norm_col].to_numpy(dtype=float)
        events = particle_df[drop_col].fillna(False).to_numpy(dtype=bool)

        for i, (cyc, ev) in enumerate(zip(cycles, events)):
            if not np.isfinite(cyc):
                continue

            # Rolling window features from past `window` cycles
            lo = max(0, i - window)
            past = intensity[lo:i]
            past_finite = past[np.isfinite(past)]

            feat = {
                "particle": particle_id,
                "cycleNo": cyc,
                "event": int(ev),
                "intensity_current": float(intensity[i]) if np.isfinite(intensity[i]) else 0.0,
                "intensity_rolling_mean": float(np.nanmean(past_finite)) if past_finite.size else 0.0,
                "intensity_rolling_std": float(np.nanstd(past_finite)) if past_finite.size else 0.0,
                "intensity_trend": float(np.polyfit(np.arange(len(past_finite)), past_finite, 1)[0])
                    if past_finite.size >= 3 else 0.0,
                "cycle_normalized": float(cyc / max(cycles.max(), 1)),
            }
            rows.append(feat)

    surv_df = pd.DataFrame(rows)

    # Merge echem features
    echem_feats = ["capacity_mAh", "coulombic_efficiency_pct", "V_min", "V_max"]
    echem_available = [c for c in echem_feats if c in echem_df.columns]
    if echem_available:
        echem_df["cycleNo"] = pd.to_numeric(echem_df["cycleNo"], errors="coerce")
        echem_merge = echem_df[["cycleNo"] + echem_available].drop_duplicates("cycleNo")
        for col in echem_available:
            echem_merge[col] = pd.to_numeric(echem_merge[col], errors="coerce")
        surv_df = surv_df.merge(echem_merge, on="cycleNo", how="left")

    feature_cols = [c for c in surv_df.columns if c not in ["particle", "cycleNo", "event"]]
    return surv_df, feature_cols


class DiscreteHazardNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_hazard_model(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 300, lr: float = 1e-3, device: str = "cpu"
) -> DiscreteHazardNet:
    # Positive class weight to handle imbalance (crack events are rare)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCELoss()

    model = DiscreteHazardNet(X_tr.shape[1]).to(device).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    # Weighted BCE manually
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xt)
        weights = torch.where(yt == 1, pos_weight.expand_as(yt), torch.ones_like(yt))
        loss = -(weights * (yt * torch.log(pred + 1e-8) + (1 - yt) * torch.log(1 - pred + 1e-8))).mean()
        loss.backward()
        opt.step()
        sched.step()

    return model


def cox_effect_sizes(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Simple univariate log-odds proxy for each feature (proper Cox via lifelines if available)."""
    results = []
    for i, fname in enumerate(feature_names):
        x = X[:, i]
        finite = np.isfinite(x)
        if finite.sum() < 10:
            results.append({"feature": fname, "log_odds": np.nan, "p_proxy": np.nan})
            continue
        x_f, y_f = x[finite], y[finite]
        # Biserial correlation as proxy
        from scipy.stats import pointbiserialr
        try:
            corr, pval = pointbiserialr(y_f, x_f)
            results.append({"feature": fname, "log_odds": float(corr), "p_proxy": float(pval)})
        except Exception:
            results.append({"feature": fname, "log_odds": np.nan, "p_proxy": np.nan})
    return pd.DataFrame(results).sort_values("p_proxy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    hazard_dir = os.path.join(args.derived_dir, "hazard")
    plots_dir = os.path.join(args.derived_dir, "plots")
    ensure_dirs(hazard_dir, plots_dir)

    particle_path = find_path([os.path.join(args.derived_dir, "particle_intensity_normalized.csv")])
    echem_path = find_path([os.path.join(args.derived_dir, "echem_per_cycle.csv")])
    if not particle_path:
        print("Warning: particle_intensity_normalized.csv not found.")
        return

    particle_df = pd.read_csv(particle_path)
    particle_df["cycleNo"] = pd.to_numeric(particle_df["cycleNo"], errors="coerce")
    echem_df = pd.read_csv(echem_path) if echem_path else pd.DataFrame()
    if not echem_df.empty:
        echem_df["cycleNo"] = pd.to_numeric(echem_df["cycleNo"], errors="coerce")

    surv_df, feature_cols = build_survival_dataset(particle_df, echem_df, window=args.window)
    surv_df[feature_cols] = surv_df[feature_cols].fillna(0)

    n_events = int(surv_df["event"].sum())
    print(f"Survival dataset: {len(surv_df)} rows, {n_events} events ({n_events/len(surv_df)*100:.2f}%)")

    X_all = surv_df[feature_cols].to_numpy(dtype=float)
    y_all = surv_df["event"].to_numpy(dtype=float)

    # Leave-one-particle-out CV
    particles = surv_df["particle"].unique()
    all_preds, all_true = [], []
    fold_aucs = []

    for p in particles:
        mask_val = surv_df["particle"] == p
        mask_tr = ~mask_val
        if mask_tr.sum() < 20 or mask_val.sum() < 2:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[mask_tr])
        X_val = scaler.transform(X_all[mask_val])
        y_tr = y_all[mask_tr]
        y_val_arr = y_all[mask_val]

        model = train_hazard_model(X_tr, y_tr, X_val, y_val_arr, epochs=args.epochs, device=device)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_true.extend(y_val_arr.tolist())
        if y_val_arr.sum() > 0:
            fold_aucs.append(float(roc_auc_score(y_val_arr, preds)))

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    overall_auc = float(roc_auc_score(all_true, all_preds)) if all_true.sum() > 0 else np.nan

    # Covariate effect sizes
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X_all)
    effect_df = cox_effect_sizes(X_scaled, y_all, feature_cols)
    effect_df.to_csv(os.path.join(hazard_dir, "covariate_effects.csv"), index=False)

    # Hazard curve over cycle time
    scaler_hc = StandardScaler()
    scaler_hc.fit(X_all)
    model_full = train_hazard_model(
        scaler_hc.transform(X_all), y_all,
        scaler_hc.transform(X_all), y_all,
        epochs=args.epochs, device=device
    )
    model_full.eval()
    with torch.no_grad():
        hazard_all = model_full(torch.tensor(scaler_hc.transform(X_all), dtype=torch.float32, device=device)).cpu().numpy()
    surv_df["predicted_hazard"] = hazard_all
    surv_df.to_csv(os.path.join(hazard_dir, "hazard_predictions.csv"), index=False)

    results = {
        "overall_auc": overall_auc,
        "fold_aucs": fold_aucs,
        "mean_fold_auc": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "n_events": n_events,
        "n_rows": len(surv_df),
    }
    with open(os.path.join(hazard_dir, "hazard_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Overall AUC: {overall_auc:.4f}  |  Mean fold AUC: {results['mean_fold_auc']:.4f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Hazard by cycle
    grp = surv_df.groupby("cycleNo")["predicted_hazard"].mean().reset_index()
    axes[0].plot(grp["cycleNo"], grp["predicted_hazard"], color="#d62728", linewidth=2)
    axes[0].set_xlabel("Cycle Number")
    axes[0].set_ylabel("Mean predicted hazard")
    axes[0].set_title("Crack Hazard over Cycling")
    axes[0].grid(alpha=0.25)

    # Covariate effects
    top_eff = effect_df.dropna(subset=["log_odds"]).head(10)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_eff["log_odds"]]
    axes[1].barh(top_eff["feature"][::-1], top_eff["log_odds"][::-1], color=colors[::-1])
    axes[1].set_xlabel("Point-biserial correlation with crack event")
    axes[1].set_title("Covariate effects on crack hazard")
    axes[1].axvline(0, color="k", linewidth=0.8)
    axes[1].grid(alpha=0.2, axis="x")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "hazard_analysis.png"), dpi=200)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
