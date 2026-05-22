#!/usr/bin/env python3
"""Fit a label-free temporal residual dictionary for ROI videos.

This is a faster alternative to a neural residual encoder on the current CPU
environment. It learns a PCA dictionary on next-frame residual fields from all
ROI crops without labels, summarizes each ROI by residual-dictionary coefficient
trajectories, then evaluates those descriptors under leave-one-cycle weak-label
splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


def load_manifests(derived: Path) -> pd.DataFrame:
    specs = [
        ("selected_event_control", derived / "multi_cycle_roi_sequences" / "selected_roi_sequence_manifest.csv"),
        ("transfer_ranked", derived / "transfer_ranked_roi_sequences" / "selected_roi_sequence_manifest.csv"),
        ("balanced_future", derived / "balanced_future_roi_sequences" / "selected_roi_sequence_manifest.csv"),
    ]
    parts = []
    for cohort, path in specs:
        if path.exists():
            df = pd.read_csv(path)
            df["embedding_cohort"] = cohort
            parts.append(df)
    if not parts:
        raise FileNotFoundError("No ROI sequence manifests found")
    return pd.concat(parts, ignore_index=True, sort=False)


def read_frames(path: Path, downsample: int) -> np.ndarray:
    z = np.load(path)
    key = "frames_norm" if "frames_norm" in z else "frames"
    frames = np.asarray(z[key], dtype=np.float32)
    if downsample > 1:
        frames = frames[:, ::downsample, ::downsample].copy()
    return np.clip(frames, 0.0, 1.0)


def slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return np.nan
    x = np.linspace(0.0, 1.0, len(values))[mask]
    return float(np.polyfit(x, values[mask], 1)[0])


def sample_residuals(manifest: pd.DataFrame, downsample: int, stride: int, max_samples: int, seed: int) -> np.ndarray:
    rows = []
    for idx, (_, row) in enumerate(manifest.iterrows(), start=1):
        if idx % 10 == 0:
            print(f"sample_residuals {idx}/{len(manifest)}", flush=True)
        frames = read_frames(Path(str(row["npz_path"])), downsample)
        residuals = frames[1:] - frames[:-1]
        rows.append(residuals[::stride].reshape(-1, residuals.shape[1] * residuals.shape[2]))
    x = np.concatenate(rows, axis=0).astype(np.float32)
    if len(x) > max_samples:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    return x


def extract_features(manifest: pd.DataFrame, pca: PCA, downsample: int) -> pd.DataFrame:
    rows = []
    for idx, (_, row) in enumerate(manifest.iterrows(), start=1):
        if idx % 10 == 0:
            print(f"extract_features {idx}/{len(manifest)}", flush=True)
        frames = read_frames(Path(str(row["npz_path"])), downsample)
        residuals = frames[1:] - frames[:-1]
        flat = residuals.reshape(residuals.shape[0], -1)
        coeff = pca.transform(flat)
        recon = pca.inverse_transform(coeff)
        err = flat - recon
        residual_energy = np.mean(flat * flat, axis=1)
        recon_energy = np.mean(recon * recon, axis=1)
        err_mse = np.mean(err * err, axis=1)
        first = slice(0, max(3, len(err_mse) // 3))
        last = slice(len(err_mse) - max(3, len(err_mse) // 3), len(err_mse))
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
            "residual_energy_mean": float(np.mean(residual_energy)),
            "residual_energy_slope": slope(residual_energy),
            "residual_energy_last_minus_first": float(np.mean(residual_energy[last]) - np.mean(residual_energy[first])),
            "dictionary_recon_energy_mean": float(np.mean(recon_energy)),
            "dictionary_recon_error_mse_mean": float(np.mean(err_mse)),
            "dictionary_recon_error_mse_slope": slope(err_mse),
            "dictionary_recon_error_last_minus_first": float(np.mean(err_mse[last]) - np.mean(err_mse[first])),
        }
        for i in range(coeff.shape[1]):
            c = coeff[:, i]
            out[f"resdict_pc{i + 1:02d}_mean"] = float(np.mean(c))
            out[f"resdict_pc{i + 1:02d}_std"] = float(np.std(c))
            out[f"resdict_pc{i + 1:02d}_slope"] = slope(c)
            out[f"resdict_pc{i + 1:02d}_last_minus_first"] = float(np.mean(c[last]) - np.mean(c[first]))
        rows.append(out)
    return pd.DataFrame(rows)


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 12) -> List[str]:
    keep = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    residual_dictionary = available_numeric(df, [c for c in df.columns if c.startswith("resdict_") or c.startswith("dictionary_") or c.startswith("residual_energy_")])
    handcrafted_scalar = available_numeric(df, [c for c in df.columns if c.startswith("particle_") or c.startswith("particle_vs_context_") or c.startswith("mask_")])
    pca_video = available_numeric(df, [c for c in df.columns if c.startswith("video_embed_pc")])
    return {
        "residual_dictionary": residual_dictionary,
        "handcrafted_scalar": handcrafted_scalar,
        "pca_video": pca_video,
        "residual_dictionary_plus_handcrafted": sorted(set(residual_dictionary + handcrafted_scalar)),
    }


def class_model(seed: int) -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.25, solver="liblinear", random_state=seed))


def reg_model() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def loo_classification(df: pd.DataFrame, features: List[str], target: str, seed: int) -> pd.DataFrame:
    rows = []
    use = df[["embedding_row_id", "roi_id", "cycleNo", target] + [c for c in features if c != target]].copy()
    y = pd.to_numeric(use[target], errors="coerce")
    valid = y.isin([0, 1])
    for cycle in sorted(pd.to_numeric(use.loc[valid, "cycleNo"], errors="coerce").dropna().unique()):
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
    row: Dict[str, Any] = {"task": "classification", "feature_set": feature_set, "target": target, "n_eval": int(len(tmp)), "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0, "n_positive": int(y.sum()) if len(y) else 0, "roc_auc": np.nan, "average_precision": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    if len(tmp) >= 8 and y.nunique() == 2:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def loo_regression(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    rows = []
    use = df[["embedding_row_id", "roi_id", "cycleNo", target] + [c for c in features if c != target]].copy()
    y = pd.to_numeric(use[target], errors="coerce")
    valid = y.notna()
    for cycle in sorted(pd.to_numeric(use.loc[valid, "cycleNo"], errors="coerce").dropna().unique()):
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
    row: Dict[str, Any] = {"task": "regression", "feature_set": feature_set, "target": target, "n_eval": int(len(tmp)), "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0, "r2": np.nan, "mae": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
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
    return {"n_permutation": int(len(arr)), "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)), "null_auc_mean": float(np.mean(arr)), "null_auc_p95": float(np.quantile(arr, 0.95))}


def metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in sorted(metrics["target"].dropna().unique()):
        for task in sorted(metrics.loc[metrics["target"] == target, "task"].dropna().unique()):
            sub = metrics[(metrics["target"] == target) & (metrics["task"] == task)]
            for base_name in ["pca_video", "handcrafted_scalar"]:
                base = sub[sub["feature_set"] == base_name]
                comp = sub[sub["feature_set"] == "residual_dictionary"]
                combo = sub[sub["feature_set"] == "residual_dictionary_plus_handcrafted"]
                for comp_name, cdf in [("residual_dictionary", comp), ("residual_dictionary_plus_handcrafted", combo)]:
                    if base.empty or cdf.empty:
                        continue
                    b = base.iloc[0]
                    c = cdf.iloc[0]
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
                    })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["delta_roc_auc", "delta_spearman_rho"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/residual_dictionary_embedding_audit")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--max-rois", type=int, default=0, help="Optional deterministic cap on ROI videos for quick bounded runs; 0 uses all.")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifests(derived)
    if args.max_rois and len(manifest) > args.max_rois:
        manifest = (
            manifest.assign(_cycle_rank=manifest.groupby("embedding_cohort").cumcount())
            .sort_values(["_cycle_rank", "embedding_cohort", "cycleNo", "roi_id"], na_position="last")
            .head(args.max_rois)
            .drop(columns=["_cycle_rank"])
            .reset_index(drop=True)
        )
    print(f"loaded_manifest rows={len(manifest)} cohorts={manifest['embedding_cohort'].value_counts().to_dict()}", flush=True)
    x = sample_residuals(manifest, args.downsample, args.stride, args.max_samples, args.seed)
    print(f"sampled_residual_matrix shape={x.shape}", flush=True)
    n_components = int(min(args.rank, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.seed)
    pca.fit(x)
    print(f"fit_pca components={n_components} evr={float(np.sum(pca.explained_variance_ratio_)):.4f}", flush=True)
    features = extract_features(manifest, pca, args.downsample)
    print(f"extracted_features rows={len(features)}", flush=True)
    masked = pd.read_csv(derived / "masked_video_embedding_audit" / "masked_video_embedding_features.csv")
    keep = ["embedding_row_id"] + [c for c in masked.columns if c.startswith("particle_") or c.startswith("particle_vs_context_") or c.startswith("mask_") or c.startswith("video_embed_pc")]
    table = features.merge(masked[keep], on="embedding_row_id", how="left")
    feature_sets = {k: v for k, v in build_feature_sets(table).items() if v}

    metrics, predictions, nulls = [], [], []
    class_targets = [t for t in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"] if t in table.columns and pd.to_numeric(table[t], errors="coerce").isin([0, 1]).sum() >= 16]
    reg_targets = available_numeric(table, ["transferred_masked_residual_signature", "residual_energy_mean", "dictionary_recon_error_mse_mean"], min_nonnull=16)
    for target in class_targets:
        y = pd.to_numeric(table[target], errors="coerce")
        if y[y.isin([0, 1])].nunique() < 2:
            continue
        for name, cols in feature_sets.items():
            pred = loo_classification(table, cols, target, args.seed)
            pred["task"] = "classification"; pred["target"] = target; pred["feature_set"] = name
            predictions.append(pred)
            met = class_metrics(pred, name, target)
            metrics.append(met)
            if name in {"residual_dictionary", "residual_dictionary_plus_handcrafted", "pca_video", "handcrafted_scalar"}:
                null = score_null_auc(pred, met.get("roc_auc", np.nan), args.seed, args.n_permutation)
                null.update({"task": "classification", "target": target, "feature_set": name, "observed_roc_auc": met.get("roc_auc", np.nan)})
                nulls.append(null)
    for target in reg_targets:
        for name, cols in feature_sets.items():
            pred = loo_regression(table, cols, target)
            pred["task"] = "regression"; pred["target"] = target; pred["feature_set"] = name
            predictions.append(pred)
            metrics.append(reg_metrics(pred, name, target))
    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    null_df = pd.DataFrame(nulls)
    if not metrics_df.empty and not null_df.empty:
        metrics_df = metrics_df.merge(null_df[["task", "target", "feature_set", "empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]], on=["task", "target", "feature_set"], how="left")
    delta_df = metric_deltas(metrics_df) if not metrics_df.empty else pd.DataFrame()

    paths = {
        "features": out / "residual_dictionary_embedding_features.csv",
        "metrics": out / "residual_dictionary_embedding_metrics.csv",
        "predictions": out / "residual_dictionary_embedding_predictions.csv",
        "deltas": out / "residual_dictionary_embedding_feature_set_deltas.csv",
        "permutation_null": out / "residual_dictionary_embedding_permutation_null.csv",
        "summary": out / "residual_dictionary_embedding_summary.json",
    }
    table.to_csv(paths["features"], index=False)
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)
    top_class = metrics_df[metrics_df["task"] == "classification"].sort_values(["roc_auc", "average_precision"], ascending=[False, False]).head(30) if not metrics_df.empty else pd.DataFrame()
    top_reg = metrics_df[metrics_df["task"] == "regression"].sort_values(["spearman_rho", "r2"], ascending=[False, False]).head(30) if not metrics_df.empty else pd.DataFrame()
    summary = clean_json({
        "n_embedding_rows": int(len(table)),
        "n_cycles": int(table["cycleNo"].nunique()),
        "embedding_cohort_counts": table["embedding_cohort"].value_counts(dropna=False).to_dict(),
        "dictionary": {"rank": n_components, "downsample": args.downsample, "stride": args.stride, "n_residual_samples": int(len(x)), "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_))},
        "classification_targets": class_targets,
        "regression_targets": reg_targets,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "top_classification_metrics": top_class.to_dict("records"),
        "top_regression_metrics": top_reg.to_dict("records"),
        "top_feature_set_deltas": delta_df.head(30).to_dict("records") if not delta_df.empty else [],
        "guardrail": "The residual dictionary is label-free and uses automatic ROI crops. It is a fast temporal-residual representation audit for model design and review prioritization, not a deployable detector, manual front label, or calibrated diffusion measurement.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "README.md").write_text(
        "# Residual Dictionary Embedding Audit\n\n"
        "Learns a PCA dictionary on next-frame residual fields from ROI videos and audits residual-coefficient descriptors under leave-one-cycle weak-label splits.\n\n"
        f"- Rows: {summary['n_embedding_rows']}\n"
        f"- Cycles: {summary['n_cycles']}\n"
        f"- Dictionary: {summary['dictionary']}\n"
        f"- Feature sets: {summary['feature_set_sizes']}\n\n"
        f"Guardrail: {summary['guardrail']}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
