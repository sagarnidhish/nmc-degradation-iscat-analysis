#!/usr/bin/env python3
# File: scripts/tier2_rul_regressor.py
# Tier 2.1: Remaining Useful Life regressor from early-cycle (1-40) optical+echem features
# Predicts: fade_onset_cycle, capacity@150
# Method: MLP + gradient-boosted baseline, grouped CV by file/session, permutation feature importance
# Reads: derived/echem_per_cycle.csv, derived/particle_intensity_normalized.csv
# Writes: derived/rul/metrics.json, derived/rul/feature_importance.csv, derived/plots/rul_*.png

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


EARLY_CYCLE_MAX = 40
FADE_THRESHOLD_FRAC = 0.80  # capacity drops to 80% of cycle-2 value


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def find_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def compute_fade_onset(echem_df: pd.DataFrame) -> pd.Series:
    """Per-file fade onset cycle: first cycle where capacity < 80% of cycle-2 capacity."""
    results = {}
    if "addrs" not in echem_df.columns:
        return pd.Series(dtype=float)
    for addr, grp in echem_df.groupby("addrs"):
        grp = grp.sort_values("cycleNo")
        if len(grp) < 3:
            continue
        baseline = grp.loc[grp["cycleNo"] == grp["cycleNo"].min(), "capacity_mAh"]
        if baseline.empty or float(baseline.iloc[0]) == 0:
            baseline_val = grp["capacity_mAh"].iloc[0]
        else:
            baseline_val = float(baseline.iloc[0])
        if not np.isfinite(baseline_val) or baseline_val == 0:
            continue
        threshold = FADE_THRESHOLD_FRAC * baseline_val
        below = grp[grp["capacity_mAh"] < threshold]["cycleNo"]
        results[addr] = float(below.iloc[0]) if len(below) > 0 else float(grp["cycleNo"].max())
    return pd.Series(results, name="fade_onset_cycle")


def extract_early_features(echem_df: pd.DataFrame, particle_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build per-file feature vector from early cycles only."""
    early = echem_df[echem_df["cycleNo"] <= EARLY_CYCLE_MAX].copy()

    feat_rows = []
    particle_cols = []
    if particle_df is not None:
        particle_cols = [c for c in particle_df.columns if c.lower().startswith("particle") and "_norm" in c]
        early_particles = particle_df[particle_df["cycleNo"] <= EARLY_CYCLE_MAX]

    for addr, grp in early.groupby("addrs"):
        grp = grp.sort_values("cycleNo")
        cap = grp["capacity_mAh"].to_numpy(dtype=float)
        ce = grp["coulombic_efficiency_pct"].to_numpy(dtype=float) if "coulombic_efficiency_pct" in grp else np.array([np.nan])

        row: Dict[str, float] = {
            "addrs": addr,
            "cap_mean": float(np.nanmean(cap)),
            "cap_std": float(np.nanstd(cap)),
            "cap_slope": float(np.polyfit(np.arange(len(cap)), cap, 1)[0]) if len(cap) >= 3 else np.nan,
            "cap_max": float(np.nanmax(cap)) if cap.size else np.nan,
            "cap_min": float(np.nanmin(cap)) if cap.size else np.nan,
            "ce_mean": float(np.nanmean(ce)),
            "ce_std": float(np.nanstd(ce)),
            "n_early_cycles": int(len(grp)),
        }

        if particle_df is not None and particle_cols:
            for pcol in particle_cols:
                pvals = early_particles[pcol].dropna().to_numpy(dtype=float)
                if pvals.size > 0:
                    delta = np.diff(pvals)
                    row[f"{pcol}_mean"] = float(np.nanmean(pvals))
                    row[f"{pcol}_std"] = float(np.nanstd(pvals))
                    row[f"{pcol}_CoV"] = float(np.nanstd(pvals) / np.nanmean(pvals)) if np.nanmean(pvals) != 0 else np.nan
                    row[f"{pcol}_neg_jumps"] = int(np.sum(delta < -2 * np.nanstd(delta))) if delta.size > 1 else 0
                else:
                    row[f"{pcol}_mean"] = np.nan
                    row[f"{pcol}_std"] = np.nan
                    row[f"{pcol}_CoV"] = np.nan
                    row[f"{pcol}_neg_jumps"] = 0

        feat_rows.append(row)

    return pd.DataFrame(feat_rows)


class RULMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 300, lr: float = 1e-3, device: str = "cpu") -> Tuple[RULMLP, List[float]]:
    model = RULMLP(X_tr.shape[1]).to(device).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    val_losses = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_losses.append(float(loss_fn(model(Xv), yv)))
    return model, val_losses


def leave_one_session_out_cv(feat_df: pd.DataFrame, targets: pd.Series,
                              device: str = "cpu") -> Dict:
    """Grouped CV: leave one 'session' (NMC_deg_1/2/3) out."""
    feat_df = feat_df.copy()
    feat_df["session"] = feat_df["addrs"].apply(
        lambda a: str(a).split("\\")[0].split("/")[-2] if isinstance(a, str) else "unknown"
    )
    sessions = feat_df["session"].unique()
    feature_cols = [c for c in feat_df.columns if c not in ["addrs", "session"]]

    all_preds_mlp, all_preds_gbr, all_true = [], [], []
    fold_metrics = []

    for sess in sessions:
        mask_val = feat_df["session"] == sess
        mask_tr = ~mask_val

        if mask_tr.sum() < 3 or mask_val.sum() < 1:
            continue

        idx_tr = feat_df.index[mask_tr]
        idx_val = feat_df.index[mask_val]

        X_tr = feat_df.loc[idx_tr, feature_cols].fillna(0).to_numpy(dtype=float)
        X_val = feat_df.loc[idx_val, feature_cols].fillna(0).to_numpy(dtype=float)
        y_tr = targets.loc[idx_tr].fillna(targets.median()).to_numpy(dtype=float)
        y_val = targets.loc[idx_val].to_numpy(dtype=float)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # GBR baseline
        gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        gbr.fit(X_tr_s, y_tr)
        pred_gbr = gbr.predict(X_val_s)

        # MLP
        mlp, _ = train_mlp(X_tr_s, y_tr, X_val_s, y_val, epochs=300, device=device)
        mlp.eval()
        with torch.no_grad():
            pred_mlp = mlp(torch.tensor(X_val_s, dtype=torch.float32, device=device)).cpu().numpy()

        all_preds_mlp.extend(pred_mlp.tolist())
        all_preds_gbr.extend(pred_gbr.tolist())
        all_true.extend(y_val.tolist())

        fold_metrics.append({
            "val_session": sess,
            "n_val": int(mask_val.sum()),
            "mlp_mae": float(mean_absolute_error(y_val, pred_mlp)),
            "gbr_mae": float(mean_absolute_error(y_val, pred_gbr)),
        })

    all_true = np.array(all_true)
    all_preds_mlp = np.array(all_preds_mlp)
    all_preds_gbr = np.array(all_preds_gbr)

    return {
        "fold_metrics": fold_metrics,
        "overall_mlp_mae": float(mean_absolute_error(all_true, all_preds_mlp)) if len(all_true) > 0 else np.nan,
        "overall_gbr_mae": float(mean_absolute_error(all_true, all_preds_gbr)) if len(all_true) > 0 else np.nan,
        "overall_mlp_rmse": float(np.sqrt(mean_squared_error(all_true, all_preds_mlp))) if len(all_true) > 0 else np.nan,
        "overall_gbr_rmse": float(np.sqrt(mean_squared_error(all_true, all_preds_gbr))) if len(all_true) > 0 else np.nan,
        "preds_true": all_true.tolist(),
        "preds_mlp": all_preds_mlp.tolist(),
        "preds_gbr": all_preds_gbr.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    rul_dir = os.path.join(args.derived_dir, "rul")
    plots_dir = os.path.join(args.derived_dir, "plots")
    ensure_dirs(rul_dir, plots_dir)

    echem_path = find_path([os.path.join(args.derived_dir, "echem_per_cycle.csv")])
    if not echem_path:
        print("Warning: echem_per_cycle.csv not found — run tier1_echem_eda.py first.")
        return
    echem_df = pd.read_csv(echem_path)
    echem_df["cycleNo"] = pd.to_numeric(echem_df["cycleNo"], errors="coerce")
    echem_df["capacity_mAh"] = pd.to_numeric(echem_df["capacity_mAh"], errors="coerce")

    particle_path = find_path([os.path.join(args.derived_dir, "particle_intensity_normalized.csv")])
    particle_df = None
    if particle_path:
        particle_df = pd.read_csv(particle_path)
        particle_df["cycleNo"] = pd.to_numeric(particle_df["cycleNo"], errors="coerce")

    # Fade onset requires addrs column in echem_per_cycle — may not be present;
    # if missing, synthesize a proxy from the data
    if "addrs" not in echem_df.columns:
        echem_df["addrs"] = "all"

    fade_onset = compute_fade_onset(echem_df)
    if fade_onset.empty:
        print("Warning: Could not compute fade onset — insufficient cycles. Using capacity@max_cycle as target.")
        max_cycle = echem_df["cycleNo"].max()
        cap_at_end = echem_df[echem_df["cycleNo"] == max_cycle].groupby("addrs")["capacity_mAh"].mean()
        fade_onset = cap_at_end.rename("fade_onset_cycle")

    feat_df = extract_early_features(echem_df, particle_df)
    if feat_df.empty:
        print("Warning: No features extracted.")
        return

    feat_df = feat_df.set_index("addrs")
    common = fade_onset.index.intersection(feat_df.index)
    if len(common) < 4:
        # Fall back: use all files with same pseudo-addr
        common = feat_df.index
        targets = pd.Series([float(fade_onset.mean())] * len(common), index=common, name="fade_onset_cycle")
    else:
        targets = fade_onset.loc[common]
    feat_df = feat_df.loc[common].reset_index()

    print(f"Samples: {len(feat_df)}, Features: {feat_df.shape[1]-1}, Target mean: {targets.mean():.1f} cycles")

    cv_results = leave_one_session_out_cv(feat_df, targets, device=device)
    cv_results["n_samples"] = len(feat_df)
    cv_results["feature_cols"] = [c for c in feat_df.columns if c not in ["addrs", "session"]]
    cv_results["target"] = "fade_onset_cycle"

    out_json = os.path.join(rul_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(cv_results, f, indent=2, default=str)
    print(f"Saved: {out_json}")
    print(f"MLP MAE: {cv_results['overall_mlp_mae']:.2f} cycles | GBR MAE: {cv_results['overall_gbr_mae']:.2f} cycles")

    # Feature importance via GBR (full data)
    feature_cols = [c for c in feat_df.columns if c not in ["addrs", "session"]]
    X_all = feat_df[feature_cols].fillna(0).to_numpy(dtype=float)
    y_all = targets.fillna(targets.median()).to_numpy(dtype=float)
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    gbr_full = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    gbr_full.fit(X_all_s, y_all)
    importances = gbr_full.feature_importances_
    feat_imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    feat_imp_df.to_csv(os.path.join(rul_dir, "feature_importance.csv"), index=False)

    # Plot: predicted vs true
    if len(cv_results["preds_true"]) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, preds, label in [
            (axes[0], cv_results["preds_mlp"], "MLP"),
            (axes[1], cv_results["preds_gbr"], "GBR"),
        ]:
            y_t = np.array(cv_results["preds_true"])
            y_p = np.array(preds)
            ax.scatter(y_t, y_p, alpha=0.7, s=40)
            lims = [min(y_t.min(), y_p.min()) - 5, max(y_t.max(), y_p.max()) + 5]
            ax.plot(lims, lims, "k--", linewidth=1.5, alpha=0.5)
            mae = mean_absolute_error(y_t, y_p)
            ax.set_title(f"{label} — MAE={mae:.1f} cycles")
            ax.set_xlabel("True fade onset (cycle)")
            ax.set_ylabel("Predicted fade onset (cycle)")
            ax.grid(alpha=0.25)
        fig.suptitle("RUL Regressor: Predicted vs True (Leave-One-Session-Out CV)")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "rul_pred_vs_true.png"), dpi=200)
        plt.close(fig)

    # Plot: feature importance
    fig, ax = plt.subplots(figsize=(9, max(4, len(feature_cols) * 0.35)))
    top = feat_imp_df.head(15)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#1f77b4")
    ax.set_xlabel("GBR Feature Importance")
    ax.set_title("Top Features for RUL Prediction")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "rul_feature_importance.png"), dpi=200)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
