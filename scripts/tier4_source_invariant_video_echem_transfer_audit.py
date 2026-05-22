#!/usr/bin/env python3
"""Source-invariant video/echem transfer audit.

Prior leave-source audits showed that source/movie composition dominates weak
future-drop labels. This audit asks a stronger question than source weighting:
can we remove feature directions that are explicitly predictable from training
source identity while retaining future16 signal?

Methods are evaluated under leave-source splits only:
- raw: standard scaled features.
- source_mean_resid_k: subtract global train mean and project away the top k
  principal directions of train-source mean offsets.
- source_confound_filter_q: remove the q fraction of features with the largest
  train-source eta-squared before fitting.

The held-out source contributes no labels and no fitted statistics except its
feature values transformed by training-only projections/scalers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 12) -> List[str]:
    keep: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        vals = pd.to_numeric(series, errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def add_context_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pieces = [out]
    for col in ["embedding_cohort", "cohort_role", "selection_subrole"]:
        if col not in out.columns:
            continue
        vals = out[col].fillna("missing").astype(str)
        add = {}
        for val in vals.value_counts().head(10).index:
            safe = "".join(ch if ch.isalnum() else "_" for ch in val.lower()).strip("_")
            add[f"{col}__{safe}"] = (vals == val).astype(float)
        if add:
            pieces.append(pd.DataFrame(add, index=out.index))
    combined = pd.concat(pieces, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()].copy()


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    acquisition = available_numeric(
        df,
        [
            "cycle_index_rank",
            "frames_percentile",
            "n_frames",
            "height",
            "width",
            "particle_prior_area_fraction",
            "mask_prior_area_fraction",
            "mask_fallback_frame_fraction",
            "mask_low_area_fraction",
            "mask_high_area_fraction",
            "mask_fragmented_fraction",
            "mask_centroid_jump_fraction",
            "mask_candidate_area_cv",
            "mask_accepted_area_cv",
            "mask_accepted_area_fraction_median",
            "mask_accepted_area_fraction_iqr",
            "mask_accepted_centroid_path_px",
            "mask_accepted_centroid_max_step_px",
            "mask_candidate_centroid_max_jump_px",
            "mask_mask_instability_score",
        ]
        + [c for c in df.columns if c.startswith("embedding_cohort__") or c.startswith("cohort_role__") or c.startswith("selection_subrole__")],
    )
    echem = available_numeric(
        df,
        [
            "capacity_mAh",
            "capacity_fade_from_first_mAh",
            "capacity_fraction_of_first",
            "coulombic_efficiency_pct",
            "coulombic_inefficiency_pct",
            "shape_V_range",
            "shape_V_mean",
            "shape_V_std",
            "shape_I_abs_mean_mA",
            "shape_I_pos_fraction",
            "shape_I_neg_fraction",
            "shape_charge_mAh_abs",
            "shape_charge_mAh_pos",
            "shape_charge_mAh_neg_abs",
            "charge_discharge_capacity_gap_mAh",
            "charge_discharge_capacity_abs_gap_mAh",
            "signed_charge_fraction",
            "shape_dVdt_abs_p95",
            "shape_dVdt_sign_consistency",
            "all_dq_abs_lowV_frac",
            "all_dq_abs_midV_frac",
            "all_dq_abs_highV_frac",
            "all_dq_abs_peak_voltage",
            "all_dq_abs_peak_frac",
            "all_dq_abs_entropy",
            "pos_dq_abs_lowV_frac",
            "pos_dq_abs_midV_frac",
            "pos_dq_abs_highV_frac",
            "pos_dq_abs_peak_voltage",
            "pos_dq_abs_peak_frac",
            "pos_dq_abs_entropy",
            "neg_dq_abs_lowV_frac",
            "neg_dq_abs_midV_frac",
            "neg_dq_abs_highV_frac",
            "neg_dq_abs_peak_voltage",
            "neg_dq_abs_peak_frac",
            "neg_dq_abs_entropy",
            "voltage_peak_hysteresis_proxy",
            "highV_charge_discharge_imbalance",
            "midV_charge_discharge_imbalance",
            "lowV_charge_discharge_imbalance",
            "dqdv_peak_concentration",
            "dqdv_entropy_asymmetry",
            "dqdv_integral_asymmetry",
            "echem_outlier_score",
            "echem_regime_pc1",
            "echem_regime_pc2",
            "echem_regime_pc3",
            "echem_regime_pc4",
            "cycle_state_pc1",
            "cycle_state_pc2",
            "cycle_state_pc3",
            "cycle_state_pc4",
            "state_step_norm",
            "axis_step",
        ],
    )
    video_scalar = available_numeric(df, [c for c in df.columns if c.startswith("particle_") or c.startswith("particle_vs_context_")])
    video_embedding = available_numeric(df, [c for c in df.columns if c.startswith("video_embed_pc")])
    return {
        "acquisition_context": acquisition,
        "echem_regime": echem,
        "video_all": sorted(set(video_scalar + video_embedding)),
        "video_plus_echem": sorted(set(video_scalar + video_embedding + echem)),
    }


def source_eta2(x: pd.DataFrame, sources: pd.Series) -> pd.Series:
    vals = x.apply(pd.to_numeric, errors="coerce")
    overall = vals.mean(axis=0)
    total = ((vals - overall) ** 2).sum(axis=0)
    between = pd.Series(0.0, index=vals.columns)
    for _, sub in vals.groupby(sources):
        if len(sub) == 0:
            continue
        between += len(sub) * (sub.mean(axis=0) - overall) ** 2
    eta = between / total.replace(0, np.nan)
    return eta.fillna(0.0)


def source_mean_basis(x_scaled: np.ndarray, sources: pd.Series, max_k: int) -> np.ndarray:
    src_vals = []
    for source in sorted(sources.dropna().unique()):
        rows = np.asarray(sources == source)
        if rows.sum() > 0:
            src_vals.append(x_scaled[rows].mean(axis=0))
    if len(src_vals) < 2:
        return np.zeros((x_scaled.shape[1], 0))
    means = np.vstack(src_vals)
    means = means - means.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(means, full_matrices=False)
    k = min(max_k, vt.shape[0], x_scaled.shape[1])
    return vt[:k].T if k > 0 else np.zeros((x_scaled.shape[1], 0))


def transform_fold(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_sources: pd.Series,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"n_features_before": int(train_x.shape[1])}
    if method.startswith("source_confound_filter_"):
        frac = float(method.rsplit("_", 1)[-1])
        eta = source_eta2(train_x, train_sources)
        n_drop = int(np.floor(len(eta) * frac))
        keep = eta.sort_values(ascending=False).index[n_drop:].tolist() if n_drop > 0 else eta.index.tolist()
        train_x = train_x[keep]
        test_x = test_x[keep]
        meta.update({"filter_fraction": frac, "n_dropped_source_confounded_features": n_drop, "n_features_after": len(keep)})
    else:
        meta["n_features_after"] = int(train_x.shape[1])

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(imputer.fit_transform(train_x))
    xte = scaler.transform(imputer.transform(test_x))

    if method.startswith("source_mean_resid_"):
        k = int(method.rsplit("_", 1)[-1])
        basis = source_mean_basis(xtr, train_sources.reset_index(drop=True), k)
        if basis.shape[1] > 0:
            xtr = xtr - xtr @ basis @ basis.T
            xte = xte - xte @ basis @ basis.T
        meta.update({"removed_source_mean_components": int(basis.shape[1])})
    else:
        meta.setdefault("removed_source_mean_components", 0)
    return xtr, xte, meta


def fit_predict(train_x: np.ndarray, train_y: pd.Series, test_x: np.ndarray, seed: int) -> np.ndarray:
    model = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed)
    model.fit(train_x, train_y.astype(int))
    return model.predict_proba(test_x)[:, 1]


def leave_source_predictions(df: pd.DataFrame, cols: List[str], target: str, feature_set: str, method: str, seed: int) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    features = df[cols].apply(pd.to_numeric, errors="coerce")
    for source in sorted(df.loc[valid, "source_stem"].dropna().unique()):
        test = valid & (df["source_stem"] == source)
        train = valid & ~test
        meta = df.loc[test, ["embedding_row_id", "roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
        fold_meta: Dict[str, Any] = {}
        if train.sum() < 12 or y[train].nunique() < 2 or test.sum() == 0:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
        else:
            xtr, xte, fold_meta = transform_fold(features.loc[train], features.loc[test], df.loc[train, "source_stem"], method)
            meta["predicted_probability"] = fit_predict(xtr, y[train], xte, seed)
            meta["status"] = "ok"
        meta["heldout_source"] = source
        meta["feature_set"] = feature_set
        meta["method"] = method
        meta["target"] = target
        for key, value in fold_meta.items():
            meta[key] = value
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def metric_row(pred: pd.DataFrame, feature_set: str, method: str, target: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    out: Dict[str, Any] = {
        "feature_set": feature_set,
        "method": method,
        "target": target,
        "n_eval": int(len(tmp)),
        "n_sources": int(tmp["heldout_source"].nunique()) if "heldout_source" in tmp else 0,
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_positive": int(y.sum()) if len(y) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "mean_features_after": float(tmp.get("n_features_after", pd.Series(dtype=float)).mean()) if "n_features_after" in tmp else np.nan,
        "mean_removed_components": float(tmp.get("removed_source_mean_components", pd.Series(dtype=float)).mean()) if "removed_source_mean_components" in tmp else np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
        rho, pval = spearmanr(y, p)
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(pval)
    return out


def permutation_null(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int) -> Dict[str, Any]:
    if not np.isfinite(observed_auc):
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce").to_numpy()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        yy = y.copy()
        rng.shuffle(yy)
        if len(np.unique(yy)) == 2:
            vals.append(float(roc_auc_score(yy, p)))
    arr = np.asarray(vals)
    if len(arr) == 0:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    return {
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)),
        "null_auc_mean": float(np.mean(arr)),
        "null_auc_p95": float(np.quantile(arr, 0.95)),
    }


def source_shift_summary(df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    cols = feature_sets.get("video_plus_echem", [])
    vals = df[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=df.index)
    med = vals.median(axis=0)
    std = vals.std(axis=0).replace(0, np.nan)
    z = (vals - med) / std
    rows = []
    for source, sub in df.groupby("source_stem"):
        y16 = pd.to_numeric(sub.get("future_any_drop_within_16cycles"), errors="coerce")
        labeled = y16.isin([0, 1])
        rows.append({
            "source_stem": source,
            "n_rows": int(len(sub)),
            "n_cycles": int(sub["cycleNo"].nunique()),
            "future16_labeled": int(labeled.sum()),
            "future16_positive_fraction": float((y16[labeled] == 1).mean()) if labeled.any() else None,
            "mean_abs_video_echem_z_shift": float(z.loc[sub.index].mean(axis=0).abs().mean()) if len(cols) else None,
        })
    return pd.DataFrame(rows).sort_values(["future16_labeled", "n_rows"], ascending=[False, False])


def deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in sorted(metrics["target"].dropna().unique()):
        sub = metrics[metrics["target"] == target]
        acq = sub[(sub["feature_set"] == "acquisition_context") & (sub["method"] == "raw")]
        for _, row in sub.iterrows():
            base = sub[(sub["feature_set"] == row["feature_set"]) & (sub["method"] == "raw")]
            if not base.empty and row["method"] != "raw":
                b = base.iloc[0]
                rows.append({
                    "target": target,
                    "feature_set": row["feature_set"],
                    "method": row["method"],
                    "comparison": f"{row['feature_set']}_{row['method']}_minus_same_raw",
                    "delta_roc_auc": row["roc_auc"] - b["roc_auc"] if pd.notna(row["roc_auc"]) and pd.notna(b["roc_auc"]) else np.nan,
                    "delta_average_precision": row["average_precision"] - b["average_precision"] if pd.notna(row["average_precision"]) and pd.notna(b["average_precision"]) else np.nan,
                    "delta_spearman_rho": row["spearman_rho"] - b["spearman_rho"] if pd.notna(row["spearman_rho"]) and pd.notna(b["spearman_rho"]) else np.nan,
                    "comparison_metric": row["roc_auc"],
                    "base_metric": b["roc_auc"],
                })
            if not acq.empty and row["feature_set"] == "video_plus_echem":
                a = acq.iloc[0]
                rows.append({
                    "target": target,
                    "feature_set": row["feature_set"],
                    "method": row["method"],
                    "comparison": f"video_plus_echem_{row['method']}_minus_acquisition_raw",
                    "delta_roc_auc": row["roc_auc"] - a["roc_auc"] if pd.notna(row["roc_auc"]) and pd.notna(a["roc_auc"]) else np.nan,
                    "delta_average_precision": row["average_precision"] - a["average_precision"] if pd.notna(row["average_precision"]) and pd.notna(a["average_precision"]) else np.nan,
                    "delta_spearman_rho": row["spearman_rho"] - a["spearman_rho"] if pd.notna(row["spearman_rho"]) and pd.notna(a["spearman_rho"]) else np.nan,
                    "comparison_metric": row["roc_auc"],
                    "base_metric": a["roc_auc"],
                })
    return pd.DataFrame(rows).sort_values(["target", "delta_roc_auc"], ascending=[True, False]) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_invariant_video_echem_transfer_audit")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--n-permutation", type=int, default=500)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = add_context_indicators(read_csv(derived / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_joined.csv"))
    feature_sets = build_feature_sets(df)
    targets = [
        t for t in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
        if t in df.columns and pd.to_numeric(df[t], errors="coerce").isin([0, 1]).sum() >= 16
    ]
    methods = ["raw", "source_mean_resid_1", "source_mean_resid_2", "source_mean_resid_4", "source_mean_resid_8", "source_confound_filter_0.10", "source_confound_filter_0.25", "source_confound_filter_0.50"]

    preds = []
    metric_rows = []
    null_rows = []
    for target in targets:
        for feature_set, cols in feature_sets.items():
            for method in methods:
                pred = leave_source_predictions(df, cols, target, feature_set, method, args.seed)
                preds.append(pred)
                met = metric_row(pred, feature_set, method, target)
                metric_rows.append(met)
                if target == "future_any_drop_within_16cycles" and feature_set in {"acquisition_context", "echem_regime", "video_all", "video_plus_echem"}:
                    null = permutation_null(pred, met["roc_auc"], args.seed, args.n_permutation)
                    null.update({"target": target, "feature_set": feature_set, "method": method, "observed_roc_auc": met["roc_auc"]})
                    null_rows.append(null)

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        metrics_df = metrics_df.merge(
            null_df[["target", "feature_set", "method", "empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]],
            on=["target", "feature_set", "method"], how="left",
        )
    delta_df = deltas(metrics_df)
    source_df = source_shift_summary(df, feature_sets)

    paths = {
        "metrics": out / "source_invariant_video_echem_metrics.csv",
        "predictions": out / "source_invariant_video_echem_predictions.csv",
        "deltas": out / "source_invariant_video_echem_deltas.csv",
        "permutation_null": out / "source_invariant_video_echem_permutation_null.csv",
        "source_summary": out / "source_invariant_source_summary.csv",
        "summary": out / "source_invariant_video_echem_summary.json",
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)
    source_df.to_csv(paths["source_summary"], index=False)

    sorted_metrics = metrics_df.sort_values(["target", "roc_auc", "average_precision"], ascending=[True, False, False])
    summary = clean_json({
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "targets": targets,
        "methods": methods,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "top_metrics": sorted_metrics.to_dict("records"),
        "top_deltas": delta_df.to_dict("records") if not delta_df.empty else [],
        "source_summary": source_df.to_dict("records"),
        "guardrail": "Source-invariant projections and source-confound filters are trained without held-out source labels, but labels remain source-composition imbalanced. These results test robustness for review prioritization only; they do not validate source-transferable warning, causal degradation mechanisms, manual QC labels, or calibrated diffusion.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Source-Invariant Video/Echem Transfer Audit\n\n"
        "Tests whether removing train-source mean directions or source-confounded features rescues leave-source weak future-drop prediction.\n\n"
        f"- Rows: {summary['n_rows']}\n"
        f"- Cycles: {summary['n_cycles']}\n"
        f"- Sources: {summary['n_sources']}\n"
        f"- Feature sets: {summary['feature_set_sizes']}\n"
        f"- Methods: {summary['methods']}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
