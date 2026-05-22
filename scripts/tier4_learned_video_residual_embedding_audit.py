#!/usr/bin/env python3
"""Learn label-free residual-CNN ROI video embeddings and audit weak labels.

The prior masked-video embedding audit used PCA over masked video bins plus
hand-crafted traces. This script trains a small self-supervised convolutional
encoder-decoder to predict next-frame residuals from current ROI frames, then
summarizes its latent channels and residual errors per ROI. The downstream
models use leave-one-cycle splits and weak labels only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ResidualEncoder(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(channels, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def read_frames(npz_path: Path, downsample: int) -> np.ndarray:
    z = np.load(npz_path)
    key = "frames_norm" if "frames_norm" in z else "frames"
    frames = np.asarray(z[key], dtype=np.float32)
    if downsample > 1:
        frames = frames[:, ::downsample, ::downsample].copy()
    return np.clip(frames, 0.0, 1.0)


def load_manifests(derived: Path) -> pd.DataFrame:
    specs = [
        ("selected_event_control", derived / "multi_cycle_roi_sequences" / "selected_roi_sequence_manifest.csv"),
        ("transfer_ranked", derived / "transfer_ranked_roi_sequences" / "selected_roi_sequence_manifest.csv"),
        ("balanced_future", derived / "balanced_future_roi_sequences" / "selected_roi_sequence_manifest.csv"),
    ]
    frames = []
    for cohort, path in specs:
        if path.exists():
            df = pd.read_csv(path)
            df["embedding_cohort"] = cohort
            df["manifest_path"] = str(path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No ROI sequence manifests found")
    return pd.concat(frames, ignore_index=True, sort=False)


def build_pair_arrays(manifest: pd.DataFrame, downsample: int, pair_stride: int, max_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    rng = np.random.default_rng(seed)
    for _, row in manifest.iterrows():
        frames = read_frames(Path(str(row["npz_path"])), downsample)
        for t in range(0, frames.shape[0] - 1, pair_stride):
            xs.append(frames[t][None, :, :])
            ys.append((frames[t + 1] - frames[t])[None, :, :])
    x = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    if len(x) > max_pairs:
        idx = rng.choice(len(x), size=max_pairs, replace=False)
        x = x[idx]
        y = y[idx]
    return x, y


def train_model(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> Tuple[ResidualEncoder, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(x))
    n_val = max(128, int(0.15 * len(idx)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    model = ResidualEncoder(args.channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_x = torch.from_numpy(x[val_idx]).to(device)
    val_y = torch.from_numpy(y[val_idx]).to(device)
    history = []
    best_state = None
    best_val = float("inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        train_loss = float(np.mean(losses)) if losses else np.nan
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return np.nan
    x = np.linspace(0.0, 1.0, len(values))[mask]
    return float(np.polyfit(x, values[mask], 1)[0])


def extract_features(model: ResidualEncoder, manifest: pd.DataFrame, downsample: int, cpu: bool) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model = model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for _, row in manifest.iterrows():
            frames = read_frames(Path(str(row["npz_path"])), downsample)
            x = torch.from_numpy(frames[:, None, :, :]).to(device)
            latent = model.latent(x).detach().cpu().numpy()
            latent_ch = latent.mean(axis=(2, 3))
            pred_res = []
            for start in range(0, frames.shape[0] - 1, 64):
                xb = torch.from_numpy(frames[start:start + 64, None, :, :]).to(device)
                pred_res.append(model(xb).detach().cpu().numpy()[:, 0])
            pred_res_arr = np.concatenate(pred_res, axis=0)[: frames.shape[0] - 1]
            true_res = frames[1:] - frames[:-1]
            err = pred_res_arr - true_res
            zero_err = -true_res
            mse = np.mean(err * err, axis=(1, 2))
            zero_mse = np.mean(zero_err * zero_err, axis=(1, 2))
            res_energy = np.mean(true_res * true_res, axis=(1, 2))
            pred_energy = np.mean(pred_res_arr * pred_res_arr, axis=(1, 2))
            first = slice(0, max(3, len(mse) // 3))
            last = slice(len(mse) - max(3, len(mse) // 3), len(mse))
            out: Dict[str, Any] = {
                "embedding_row_id": f"{row.get('embedding_cohort', 'cohort')}::{row.get('roi_id')}",
                "roi_id": row.get("roi_id"),
                "embedding_cohort": row.get("embedding_cohort"),
                "cycleNo": row.get("cycleNo"),
                "source_stem": row.get("source_stem", ""),
                "cohort_role": row.get("cohort_role", ""),
                "selection_subrole": row.get("selection_subrole", ""),
                "future_any_drop_within_8cycles": row.get("future_any_drop_within_8cycles", np.nan),
                "future_any_drop_within_16cycles": row.get("future_any_drop_within_16cycles", np.nan),
                "any_abrupt_drop": row.get("any_abrupt_drop", np.nan),
                "transferred_masked_residual_signature": row.get("transferred_masked_residual_signature", np.nan),
                "learned_residual_mse_mean": float(np.mean(mse)),
                "learned_residual_mse_first": float(np.mean(mse[first])),
                "learned_residual_mse_last": float(np.mean(mse[last])),
                "learned_residual_mse_last_minus_first": float(np.mean(mse[last]) - np.mean(mse[first])),
                "learned_residual_mse_slope": slope(mse),
                "zero_residual_mse_mean": float(np.mean(zero_mse)),
                "learned_vs_zero_residual_mse_delta": float(np.mean(zero_mse) - np.mean(mse)),
                "learned_vs_zero_residual_mse_relative": float((np.mean(zero_mse) - np.mean(mse)) / np.mean(zero_mse)) if np.mean(zero_mse) > 0 else np.nan,
                "true_residual_energy_mean": float(np.mean(res_energy)),
                "true_residual_energy_slope": slope(res_energy),
                "pred_residual_energy_mean": float(np.mean(pred_energy)),
                "pred_residual_energy_slope": slope(pred_energy),
            }
            for c in range(latent_ch.shape[1]):
                trace = latent_ch[:, c]
                out[f"learned_latent_ch{c + 1:02d}_mean"] = float(np.mean(trace))
                out[f"learned_latent_ch{c + 1:02d}_std"] = float(np.std(trace))
                out[f"learned_latent_ch{c + 1:02d}_slope"] = slope(trace)
                out[f"learned_latent_ch{c + 1:02d}_last_minus_first"] = float(np.mean(trace[last]) - np.mean(trace[first]))
            rows.append(out)
    return pd.DataFrame(rows)


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 12) -> List[str]:
    out = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            out.append(col)
    return out


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    learned_latent = available_numeric(df, [c for c in df.columns if c.startswith("learned_latent_")])
    learned_residual = available_numeric(df, [c for c in df.columns if c.startswith("learned_residual_") or c.startswith("true_residual_") or c.startswith("pred_residual_")])
    handcrafted_scalar = available_numeric(df, [
        c for c in df.columns
        if c.startswith("particle_") or c.startswith("particle_vs_context_") or c.startswith("mask_")
    ])
    pca_video = available_numeric(df, [c for c in df.columns if c.startswith("video_embed_pc")])
    return {
        "learned_latent": learned_latent,
        "learned_residual": learned_residual,
        "learned_all": sorted(set(learned_latent + learned_residual)),
        "handcrafted_scalar": handcrafted_scalar,
        "pca_video": pca_video,
        "learned_plus_handcrafted": sorted(set(learned_latent + learned_residual + handcrafted_scalar)),
    }


def class_model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed),
    )


def reg_model() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def loo_classification(df: pd.DataFrame, features: List[str], target: str, seed: int) -> pd.DataFrame:
    rows = []
    features = [c for c in features if c != target]
    use = df[["embedding_row_id", "roi_id", "cycleNo", target] + features].copy()
    y = pd.to_numeric(use[target], errors="coerce")
    valid = y.isin([0, 1])
    cycles = sorted(pd.to_numeric(use.loc[valid, "cycleNo"], errors="coerce").dropna().unique())
    for cycle in cycles:
        test = valid & (pd.to_numeric(use["cycleNo"], errors="coerce") == cycle)
        train = valid & ~test
        meta = use.loc[test, ["embedding_row_id", "roi_id", "cycleNo", target]].rename(columns={target: "observed"}).copy()
        if train.sum() < 12 or y[train].nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
        else:
            model = class_model(seed)
            model.fit(use.loc[train, features], y[train].astype(int))
            meta["predicted_probability"] = model.predict_proba(use.loc[test, features])[:, 1]
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def class_metrics(pred: pd.DataFrame, feature_set: str, target: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    row: Dict[str, Any] = {
        "task": "classification",
        "feature_set": feature_set,
        "target": target,
        "n_eval": int(len(tmp)),
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_positive": int(y.sum()) if len(y) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def loo_regression(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    rows = []
    features = [c for c in features if c != target]
    use = df[["embedding_row_id", "roi_id", "cycleNo", target] + features].copy()
    y = pd.to_numeric(use[target], errors="coerce")
    valid = y.notna()
    cycles = sorted(pd.to_numeric(use.loc[valid, "cycleNo"], errors="coerce").dropna().unique())
    for cycle in cycles:
        test = valid & (pd.to_numeric(use["cycleNo"], errors="coerce") == cycle)
        train = valid & ~test
        meta = use.loc[test, ["embedding_row_id", "roi_id", "cycleNo", target]].rename(columns={target: "observed"}).copy()
        if train.sum() < 12 or y[train].nunique() < 2:
            meta["predicted"] = np.nan
            meta["status"] = "skipped_train_size_or_variance"
        else:
            model = reg_model()
            model.fit(use.loc[train, features], y[train].astype(float))
            meta["predicted"] = model.predict(use.loc[test, features])
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def reg_metrics(pred: pd.DataFrame, feature_set: str, target: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted"])
    y = pd.to_numeric(tmp["observed"], errors="coerce")
    p = pd.to_numeric(tmp["predicted"], errors="coerce")
    row: Dict[str, Any] = {
        "task": "regression",
        "feature_set": feature_set,
        "target": target,
        "n_eval": int(len(tmp)),
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "r2": np.nan,
        "mae": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and y.nunique(dropna=True) >= 2:
        row["r2"] = float(r2_score(y, p))
        row["mae"] = float(mean_absolute_error(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def score_null_auc(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int) -> Dict[str, Any]:
    if not np.isfinite(observed_auc) or n_perm <= 0:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce").to_numpy()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        yy = y.copy()
        rng.shuffle(yy)
        if len(np.unique(yy)) == 2:
            vals.append(float(roc_auc_score(yy, p)))
    if not vals:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    arr = np.asarray(vals)
    return {
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)),
        "null_auc_mean": float(np.mean(arr)),
        "null_auc_p95": float(np.quantile(arr, 0.95)),
    }


def metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in sorted(metrics["target"].dropna().unique()):
        for task in sorted(metrics.loc[metrics["target"] == target, "task"].dropna().unique()):
            sub = metrics[(metrics["target"] == target) & (metrics["task"] == task)]
            for base_name in ["pca_video", "handcrafted_scalar"]:
                base = sub[sub["feature_set"] == base_name]
                if base.empty:
                    continue
                for comp_name in ["learned_latent", "learned_all", "learned_plus_handcrafted"]:
                    comp = sub[sub["feature_set"] == comp_name]
                    if comp.empty:
                        continue
                    b = base.iloc[0]
                    c = comp.iloc[0]
                    rows.append({
                        "task": task,
                        "target": target,
                        "comparison": f"{comp_name}_minus_{base_name}",
                        "delta_roc_auc": c.get("roc_auc") - b.get("roc_auc") if pd.notna(c.get("roc_auc")) and pd.notna(b.get("roc_auc")) else np.nan,
                        "delta_average_precision": c.get("average_precision") - b.get("average_precision") if pd.notna(c.get("average_precision")) and pd.notna(b.get("average_precision")) else np.nan,
                        "delta_r2": c.get("r2") - b.get("r2") if pd.notna(c.get("r2")) and pd.notna(b.get("r2")) else np.nan,
                        "delta_spearman_rho": c.get("spearman_rho") - b.get("spearman_rho") if pd.notna(c.get("spearman_rho")) and pd.notna(b.get("spearman_rho")) else np.nan,
                        "base_metric": b.get("roc_auc") if task == "classification" else b.get("r2"),
                        "comparison_metric": c.get("roc_auc") if task == "classification" else c.get("r2"),
                        "base_spearman_rho": b.get("spearman_rho"),
                        "comparison_spearman_rho": c.get("spearman_rho"),
                    })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["delta_roc_auc", "delta_spearman_rho"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/learned_video_residual_embedding_audit")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--pair-stride", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=9000)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifests(derived)
    x, y = build_pair_arrays(manifest, args.downsample, args.pair_stride, args.max_pairs, args.seed)
    model, history = train_model(x, y, args)
    features = extract_features(model, manifest, args.downsample, args.cpu)

    masked_features = pd.read_csv(derived / "masked_video_embedding_audit" / "masked_video_embedding_features.csv")
    keep_cols = ["embedding_row_id"] + [
        c for c in masked_features.columns
        if c.startswith("particle_") or c.startswith("particle_vs_context_") or c.startswith("mask_") or c.startswith("video_embed_pc")
    ]
    joined = features.merge(masked_features[keep_cols], on="embedding_row_id", how="left", suffixes=("", "_handcrafted"))
    feature_sets = {k: v for k, v in build_feature_sets(joined).items() if v}

    class_targets = [
        t for t in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
        if t in joined.columns and pd.to_numeric(joined[t], errors="coerce").isin([0, 1]).sum() >= 16
    ]
    reg_targets = available_numeric(joined, [
        "transferred_masked_residual_signature",
        "learned_residual_mse_mean",
        "learned_residual_mse_last_minus_first",
        "true_residual_energy_mean",
    ], min_nonnull=16)

    metrics = []
    predictions = []
    nulls = []
    for target in class_targets:
        y_target = pd.to_numeric(joined[target], errors="coerce")
        if y_target[y_target.isin([0, 1])].nunique() < 2:
            continue
        for feature_set, cols in feature_sets.items():
            pred = loo_classification(joined, cols, target, args.seed)
            pred["task"] = "classification"
            pred["target"] = target
            pred["feature_set"] = feature_set
            predictions.append(pred)
            met = class_metrics(pred, feature_set, target)
            metrics.append(met)
            if feature_set in {"learned_latent", "learned_all", "learned_plus_handcrafted", "handcrafted_scalar", "pca_video"}:
                null = score_null_auc(pred, met.get("roc_auc", np.nan), args.seed, args.n_permutation)
                null.update({"task": "classification", "target": target, "feature_set": feature_set, "observed_roc_auc": met.get("roc_auc", np.nan)})
                nulls.append(null)
    for target in reg_targets:
        for feature_set, cols in feature_sets.items():
            pred = loo_regression(joined, cols, target)
            pred["task"] = "regression"
            pred["target"] = target
            pred["feature_set"] = feature_set
            predictions.append(pred)
            metrics.append(reg_metrics(pred, feature_set, target))

    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    null_df = pd.DataFrame(nulls)
    if not metrics_df.empty and not null_df.empty:
        metrics_df = metrics_df.merge(
            null_df[["task", "target", "feature_set", "empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]],
            on=["task", "target", "feature_set"],
            how="left",
        )
    delta_df = metric_deltas(metrics_df) if not metrics_df.empty else pd.DataFrame()

    paths = {
        "features": out / "learned_video_residual_embedding_features.csv",
        "training_history": out / "learned_video_residual_training_history.csv",
        "metrics": out / "learned_video_residual_embedding_metrics.csv",
        "predictions": out / "learned_video_residual_embedding_predictions.csv",
        "deltas": out / "learned_video_residual_embedding_feature_set_deltas.csv",
        "permutation_null": out / "learned_video_residual_embedding_permutation_null.csv",
        "summary": out / "learned_video_residual_embedding_summary.json",
    }
    joined.to_csv(paths["features"], index=False)
    history.to_csv(paths["training_history"], index=False)
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)

    top_class = metrics_df[metrics_df["task"] == "classification"].sort_values(["roc_auc", "average_precision"], ascending=[False, False]).head(30) if not metrics_df.empty else pd.DataFrame()
    top_reg = metrics_df[metrics_df["task"] == "regression"].sort_values(["spearman_rho", "r2"], ascending=[False, False]).head(30) if not metrics_df.empty else pd.DataFrame()
    top_delta = delta_df.sort_values(["delta_roc_auc", "delta_spearman_rho"], ascending=[False, False]).head(30) if not delta_df.empty else pd.DataFrame()
    train_summary = {
        "n_pairs_used": int(len(x)),
        "downsample": int(args.downsample),
        "pair_stride": int(args.pair_stride),
        "channels": int(args.channels),
        "n_epochs_run": int(len(history)),
        "best_val_loss": float(history["val_loss"].min()) if not history.empty else np.nan,
        "final_train_loss": float(history["train_loss"].iloc[-1]) if not history.empty else np.nan,
    }
    summary = clean_json({
        "n_embedding_rows": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "embedding_cohort_counts": joined["embedding_cohort"].value_counts(dropna=False).to_dict(),
        "training": train_summary,
        "classification_targets": class_targets,
        "regression_targets": reg_targets,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "top_classification_metrics": top_class.to_dict("records"),
        "top_regression_metrics": top_reg.to_dict("records"),
        "top_feature_set_deltas": top_delta.to_dict("records"),
        "guardrail": "The residual CNN is trained label-free on automatic ROI crops and evaluated only through weak cycle labels. Learned embeddings support representation design and review prioritization, not deployable prediction, manual particle/front labels, or calibrated diffusion.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    readme = [
        "# Learned Video Residual Embedding Audit",
        "",
        "Self-supervised residual-CNN encoder over ROI videos. It predicts next-frame residuals label-free, then audits learned latent/residual descriptors under leave-one-cycle weak-label splits.",
        "",
        f"- Rows: {summary['n_embedding_rows']}",
        f"- Cycles: {summary['n_cycles']}",
        f"- Training: {summary['training']}",
        f"- Feature set sizes: {summary['feature_set_sizes']}",
        "",
        f"Guardrail: {summary['guardrail']}",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
