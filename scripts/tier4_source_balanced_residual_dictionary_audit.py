#!/usr/bin/env python3
"""Label-free residual dictionary audit for source-balanced ROI crops.

This learns PCA bases on next-frame residual fields from the source-balanced
particle-region crop tensors, summarizes each ROI by coefficient trajectories,
and tests whether those dynamics descriptors add future-drop signal beyond
automatic mask/front scalar proxies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]


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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def read_frames(npz_path: Path, downsample: int) -> np.ndarray:
    z = np.load(npz_path)
    frames = np.asarray(z["frames_norm"] if "frames_norm" in z else z["frames"], dtype=np.float32)
    if downsample > 1:
        frames = frames[:, ::downsample, ::downsample].copy()
    return np.clip(frames, 0.0, 1.0)


def slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 3 or np.nanstd(values[mask]) <= 1e-12:
        return np.nan
    x = np.linspace(0.0, 1.0, len(values))[mask]
    return float(np.polyfit(x, values[mask], 1)[0])


def source_eta2(series: pd.Series, sources: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.notna() & sources.notna()
    vals = vals[valid]
    src = sources[valid]
    if vals.nunique() < 2 or src.nunique() < 2:
        return np.nan
    overall = vals.mean()
    total = float(((vals - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in vals.groupby(src):
        between += len(sub) * float((sub.mean() - overall) ** 2)
    return between / total


def orient_auc(y: pd.Series, x: pd.Series) -> Tuple[float, float, str]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
        return np.nan, np.nan, "NA"
    direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
    score = xx if direction == "higher_in_positive" else -xx
    return float(roc_auc_score(yy, score)), float(average_precision_score(yy, score)), direction


def feature_tests(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target in [t for t in TARGETS if t in df.columns]:
        y = numeric(df, target)
        for feat in features:
            x = numeric(df, feat)
            valid = y.isin([0, 1]) & x.notna()
            yy = y[valid].astype(int)
            xx = x[valid]
            auc, ap, direction = orient_auc(y, x)
            rho = sp = p_mwu = np.nan
            med_pos = med_neg = np.nan
            if valid.sum() >= 8 and yy.nunique() == 2 and xx.nunique() > 1:
                signed_x = xx if direction == "higher_in_positive" else -xx
                rho, sp = spearmanr(yy, signed_x)
                pos = xx[yy == 1]
                neg = xx[yy == 0]
                med_pos = float(pos.median())
                med_neg = float(neg.median())
                try:
                    _, p_mwu = mannwhitneyu(pos, neg, alternative="two-sided")
                except ValueError:
                    p_mwu = np.nan
            rows.append({
                "target": target,
                "feature": feat,
                "n": int(valid.sum()),
                "n_positive": int(yy.sum()) if len(yy) else 0,
                "direction": direction,
                "oriented_auc": auc,
                "average_precision": ap,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(sp) if np.isfinite(sp) else np.nan,
                "mwu_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
                "median_positive": med_pos,
                "median_negative": med_neg,
                "median_positive_minus_negative": med_pos - med_neg if np.isfinite(med_pos) and np.isfinite(med_neg) else np.nan,
                "source_eta2": source_eta2(x, df["source_stem"]),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "oriented_auc", "average_precision"], ascending=[True, False, False])
    return out


def sample_residuals(manifest: pd.DataFrame, downsample: int, stride: int, max_samples: int, seed: int) -> np.ndarray:
    rows = []
    for idx, (_, row) in enumerate(manifest.iterrows(), start=1):
        if idx % 16 == 0:
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
    rows: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(manifest.iterrows(), start=1):
        if idx % 16 == 0:
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
        out: Dict[str, Any] = row.to_dict()
        out.update({
            "residual_energy_mean": float(np.mean(residual_energy)),
            "residual_energy_p95": float(np.percentile(residual_energy, 95)),
            "residual_energy_slope": slope(residual_energy),
            "residual_energy_last_minus_first": float(np.mean(residual_energy[last]) - np.mean(residual_energy[first])),
            "dictionary_recon_energy_mean": float(np.mean(recon_energy)),
            "dictionary_recon_error_mse_mean": float(np.mean(err_mse)),
            "dictionary_recon_error_mse_slope": slope(err_mse),
            "dictionary_recon_error_last_minus_first": float(np.mean(err_mse[last]) - np.mean(err_mse[first])),
        })
        for i in range(coeff.shape[1]):
            c = coeff[:, i]
            out[f"resdict_pc{i + 1:02d}_mean"] = float(np.mean(c))
            out[f"resdict_pc{i + 1:02d}_std"] = float(np.std(c))
            out[f"resdict_pc{i + 1:02d}_slope"] = slope(c)
            out[f"resdict_pc{i + 1:02d}_last_minus_first"] = float(np.mean(c[last]) - np.mean(c[first]))
        rows.append(out)
    return pd.DataFrame(rows)


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 16) -> List[str]:
    keep = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    residual = available_numeric(df, [c for c in df.columns if c.startswith("resdict_") or c.startswith("dictionary_") or c.startswith("residual_energy_")])
    mask_front = available_numeric(df, [
        "mask_base_area_fraction", "mask_area_fraction_median", "mask_area_fraction_iqr", "mask_area_fraction_slope",
        "mask_centroid_path_px", "mask_centroid_max_step_px", "mask_centroid_drift_px",
        "masked_minus_background_mean_median", "masked_minus_background_mean_slope",
        "front_radius_q60_median_px", "front_radius_q60_delta_px", "front_radius_q60_slope_px_per_norm_time",
        "front_radius_q70_median_px", "front_radius_q70_delta_px", "front_radius_q70_slope_px_per_norm_time",
        "front_radius_q80_median_px", "front_radius_q80_delta_px", "front_radius_q80_slope_px_per_norm_time",
        "front_radius_q70_positive_step_fraction",
        "front_gradient_peak_radius_median_px", "front_gradient_peak_radius_slope_px_per_norm_time",
        "apparent_diffusion_q70_px2_per_norm_time", "apparent_diffusion_q70_um2_per_norm_time",
        "roi_norm_mean_delta_last_minus_first", "object_area_ds_px", "object_mean_residual", "object_mean_abs_z",
    ])
    return {
        "residual_dictionary": residual,
        "mask_front_scalar": mask_front,
        "residual_dictionary_plus_mask_front": sorted(set(residual + mask_front)),
    }


def class_model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.2, solver="liblinear", random_state=seed),
    )


def grouped_predictions(df: pd.DataFrame, features: List[str], target: str, group_col: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = numeric(df, target)
    valid = y.isin([0, 1]) & df[group_col].notna()
    for group in sorted(df.loc[valid, group_col].dropna().unique()):
        test = valid & (df[group_col] == group)
        train = valid & ~test
        meta = df.loc[test, ["roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
        meta[group_col] = group
        if train.sum() < 16 or y[train].nunique() < 2 or y[test].nunique() < 1:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
        else:
            model = class_model(seed)
            model.fit(df.loc[train, features], y[train].astype(int))
            meta["predicted_probability"] = model.predict_proba(df.loc[test, features])[:, 1]
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    out = pd.DataFrame(rows)
    out["target"] = target
    out["group_col"] = group_col
    return out


def prediction_metrics(pred: pd.DataFrame, feature_set: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    row: Dict[str, Any] = {
        "feature_set": feature_set,
        "target": pred["target"].iloc[0] if len(pred) else "",
        "group_col": pred["group_col"].iloc[0] if len(pred) else "",
        "n_eval": int(len(tmp)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "n_groups": int(tmp[pred["group_col"].iloc[0]].nunique()) if len(tmp) and pred["group_col"].iloc[0] in tmp else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2 and p.nunique() > 1:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_roi_sequences")
    parser.add_argument("--mask-front-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_mask_front_sanity_audit")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_residual_dictionary_audit")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=9000)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    roi_dir = Path(args.roi_dir)
    mask_front_dir = Path(args.mask_front_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(roi_dir / "selected_roi_sequence_manifest.csv")
    mask_front_path = mask_front_dir / "source_balanced_mask_front_features.csv"
    if mask_front_path.exists():
        mask_front = pd.read_csv(mask_front_path)
        drop_cols = [c for c in mask_front.columns if c in manifest.columns and c != "roi_id"]
        manifest = manifest.merge(mask_front.drop(columns=drop_cols), on="roi_id", how="left")

    samples = sample_residuals(manifest, args.downsample, args.stride, args.max_samples, args.seed)
    pca = PCA(n_components=args.n_components, random_state=args.seed)
    pca.fit(samples)
    features = extract_features(manifest, pca, args.downsample)
    feature_sets = build_feature_sets(features)
    all_test_features = sorted(set(sum(feature_sets.values(), [])))
    tests = feature_tests(features, all_test_features)

    pred_rows: List[pd.DataFrame] = []
    metrics: List[Dict[str, Any]] = []
    for name, cols in feature_sets.items():
        if not cols:
            continue
        for target in TARGETS:
            if target not in features.columns:
                continue
            for group_col in ["cycleNo", "source_stem"]:
                pred = grouped_predictions(features, cols, target, group_col, args.seed)
                pred["feature_set"] = name
                pred_rows.append(pred)
                metrics.append(prediction_metrics(pred, name))
    predictions = pd.concat(pred_rows, ignore_index=True, sort=False) if pred_rows else pd.DataFrame()
    metric_df = pd.DataFrame(metrics)
    if not metric_df.empty:
        metric_df = metric_df.sort_values(["target", "group_col", "roc_auc", "average_precision"], ascending=[True, True, False, False])

    cycle_aggs = {
        "roi_id": "count",
        "future_any_drop_within_8cycles": "max",
        "future_any_drop_within_16cycles": "max",
    }
    for col in all_test_features:
        if col in features.columns:
            cycle_aggs[col] = "mean"
    cycle = features.groupby(["cycleNo", "source_stem"], as_index=False).agg(cycle_aggs).rename(columns={"roi_id": "n_roi"})
    cycle_tests = feature_tests(cycle, [c for c in all_test_features if c in cycle.columns])

    paths = {
        "features": out / "source_balanced_residual_dictionary_features.csv",
        "feature_tests": out / "source_balanced_residual_dictionary_feature_tests.csv",
        "cycle_summary": out / "source_balanced_residual_dictionary_cycle_summary.csv",
        "cycle_tests": out / "source_balanced_residual_dictionary_cycle_tests.csv",
        "predictions": out / "source_balanced_residual_dictionary_predictions.csv",
        "metrics": out / "source_balanced_residual_dictionary_metrics.csv",
        "summary": out / "source_balanced_residual_dictionary_summary.json",
    }
    features.to_csv(paths["features"], index=False)
    tests.to_csv(paths["feature_tests"], index=False)
    cycle.to_csv(paths["cycle_summary"], index=False)
    cycle_tests.to_csv(paths["cycle_tests"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    metric_df.to_csv(paths["metrics"], index=False)

    top_metrics = metric_df.head(24).to_dict(orient="records") if not metric_df.empty else []
    top_tests = tests.head(24).to_dict(orient="records") if not tests.empty else []
    top_cycle_tests = cycle_tests.head(16).to_dict(orient="records") if not cycle_tests.empty else []
    summary = {
        "n_roi_sequences": int(len(features)),
        "n_cycles": int(features["cycleNo"].nunique()),
        "n_sources": int(features["source_stem"].nunique()),
        "future8_positive_sequences": int(numeric(features, "future_any_drop_within_8cycles").sum()),
        "future16_positive_sequences": int(numeric(features, "future_any_drop_within_16cycles").sum()),
        "downsample": args.downsample,
        "stride": args.stride,
        "n_components": args.n_components,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "top_metrics": top_metrics,
        "top_roi_feature_tests": top_tests,
        "top_cycle_feature_tests": top_cycle_tests,
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Residual dictionary bases are label-free PCA summaries of automatic source-balanced ROI crops. They are useful for ranking dynamics hypotheses, not a trained deployable predictor or calibrated physics model.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Source-Balanced Residual Dictionary Audit",
        "",
        f"ROI sequences: {summary['n_roi_sequences']} across {summary['n_cycles']} cycles and {summary['n_sources']} sources.",
        f"Future8/future16 positive ROI sequences: {summary['future8_positive_sequences']} / {summary['future16_positive_sequences']}.",
        f"PCA components: {args.n_components}; explained residual variance sum: {summary['pca_explained_variance_ratio_sum']:.3f}.",
        "",
        "## Top Grouped Metrics",
    ]
    for row in top_metrics[:10]:
        readme.append(
            f"- {row.get('group_col')} {row.get('target')} {row.get('feature_set')}: "
            f"AUC={row.get('roc_auc'):.3f}, AP={row.get('average_precision'):.3f}, n={row.get('n_eval')}"
        )
    readme.extend(["", "## Top Scalar Feature Tests"])
    for row in top_tests[:10]:
        readme.append(
            f"- {row.get('target')} {row.get('feature')}: AUC={row.get('oriented_auc'):.3f}, "
            f"AP={row.get('average_precision'):.3f}, eta2={row.get('source_eta2'):.3f}"
        )
    readme.extend(["", "## Guardrail", summary["guardrail"], ""])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()
