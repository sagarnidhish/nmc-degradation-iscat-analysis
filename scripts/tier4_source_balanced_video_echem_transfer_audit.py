#!/usr/bin/env python3
"""Source-balanced transfer audit for video/echem weak warnings.

The acquisition-residualized audit showed future16 video+echem signal under
leave-cycle splits but not leave-source splits. This audit probes whether that
failure is driven by source/movie composition by comparing raw source holdouts
with source/class weighted fitting and within-source feature normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def add_context_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["embedding_cohort", "cohort_role", "selection_subrole"]:
        if col not in out.columns:
            continue
        vals = out[col].fillna("missing").astype(str)
        for val in vals.value_counts().head(10).index:
            safe = "".join(ch if ch.isalnum() else "_" for ch in val.lower()).strip("_")
            out[f"{col}__{safe}"] = (vals == val).astype(float)
    return out


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


def within_source_rank(df: pd.DataFrame, cols: List[str], source_col: str = "source_stem") -> pd.DataFrame:
    ranked_cols = {}
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        ranked_cols[col] = vals.groupby(df[source_col]).rank(pct=True, method="average")
    return pd.DataFrame(ranked_cols, index=df.index)


def train_weights(df: pd.DataFrame, target: str) -> np.ndarray:
    y = pd.to_numeric(df[target], errors="coerce").astype(int)
    source_counts = df["source_stem"].value_counts().to_dict()
    class_counts = y.value_counts().to_dict()
    weights = []
    for idx, row in df.iterrows():
        source_w = 1.0 / max(1, source_counts.get(row["source_stem"], 1))
        class_w = 1.0 / max(1, class_counts.get(int(y.loc[idx]), 1))
        weights.append(source_w * class_w)
    arr = np.asarray(weights, dtype=float)
    return arr / np.nanmean(arr)


def fit_predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    weights: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(imputer.fit_transform(train_x))
    xte = scaler.transform(imputer.transform(test_x))
    model = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed)
    if weights is not None:
        model.fit(xtr, train_y.astype(int), sample_weight=weights)
    else:
        model.fit(xtr, train_y.astype(int))
    return model.predict_proba(xte)[:, 1]


def leave_source_predictions(df: pd.DataFrame, cols: List[str], target: str, feature_set: str, mode: str, seed: int) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    feature_matrix = within_source_rank(df, cols) if "source_rank" in mode else df[cols].apply(pd.to_numeric, errors="coerce")
    for source in sorted(df.loc[valid, "source_stem"].dropna().unique()):
        test = valid & (df["source_stem"] == source)
        train = valid & ~test
        meta = df.loc[test, ["embedding_row_id", "roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
        if train.sum() < 12 or y[train].nunique() < 2 or test.sum() == 0:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
        else:
            weights = train_weights(df.loc[train, ["source_stem", target]], target) if "weighted" in mode else None
            meta["predicted_probability"] = fit_predict(feature_matrix.loc[train], y[train], feature_matrix.loc[test], weights, seed)
            meta["status"] = "ok"
        meta["heldout_source"] = source
        meta["feature_set"] = feature_set
        meta["mode"] = mode
        meta["target"] = target
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def metric_row(pred: pd.DataFrame, feature_set: str, mode: str, target: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    out: Dict[str, Any] = {
        "feature_set": feature_set,
        "mode": mode,
        "target": target,
        "n_eval": int(len(tmp)),
        "n_sources": int(tmp["heldout_source"].nunique()) if "heldout_source" in tmp else 0,
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_positive": int(y.sum()) if len(y) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
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


def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, sub in df.groupby("source_stem"):
        row = {"source_stem": source, "n_rows": int(len(sub)), "n_cycles": int(sub["cycleNo"].nunique())}
        for target in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]:
            if target in sub.columns:
                y = pd.to_numeric(sub[target], errors="coerce")
                labeled = y.isin([0, 1])
                row[f"{target}_labeled"] = int(labeled.sum())
                row[f"{target}_positive"] = int((y == 1).sum())
                row[f"{target}_negative"] = int((y == 0).sum())
                row[f"{target}_positive_fraction"] = float((y[labeled] == 1).mean()) if labeled.any() else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_rows", "source_stem"], ascending=[False, True])


def deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("video_plus_echem", "source_rank_weighted", "video_plus_echem", "raw_unweighted"),
        ("video_plus_echem", "source_rank_weighted", "acquisition_context", "raw_unweighted"),
        ("video_plus_echem", "source_rank_weighted", "echem_regime", "raw_unweighted"),
        ("video_plus_echem", "source_rank_weighted", "video_all", "raw_unweighted"),
        ("video_plus_echem", "raw_weighted", "video_plus_echem", "raw_unweighted"),
        ("video_plus_echem", "source_rank_unweighted", "video_plus_echem", "raw_unweighted"),
    ]
    for target in sorted(metrics["target"].dropna().unique()):
        sub = metrics[metrics["target"] == target]
        for cf, cm, bf, bm in comparisons:
            c = sub[(sub["feature_set"] == cf) & (sub["mode"] == cm)]
            b = sub[(sub["feature_set"] == bf) & (sub["mode"] == bm)]
            if c.empty or b.empty:
                continue
            cr, br = c.iloc[0], b.iloc[0]
            rows.append({
                "target": target,
                "comparison": f"{cf}_{cm}_minus_{bf}_{bm}",
                "delta_roc_auc": cr["roc_auc"] - br["roc_auc"] if pd.notna(cr["roc_auc"]) and pd.notna(br["roc_auc"]) else np.nan,
                "delta_average_precision": cr["average_precision"] - br["average_precision"] if pd.notna(cr["average_precision"]) and pd.notna(br["average_precision"]) else np.nan,
                "delta_spearman_rho": cr["spearman_rho"] - br["spearman_rho"] if pd.notna(cr["spearman_rho"]) and pd.notna(br["spearman_rho"]) else np.nan,
                "comparison_metric": cr["roc_auc"],
                "base_metric": br["roc_auc"],
                "comparison_spearman_rho": cr["spearman_rho"],
                "base_spearman_rho": br["spearman_rho"],
            })
    return pd.DataFrame(rows).sort_values(["target", "delta_roc_auc"], ascending=[True, False]) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_video_echem_transfer_audit")
    parser.add_argument("--seed", type=int, default=67)
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
    modes = ["raw_unweighted", "raw_weighted", "source_rank_unweighted", "source_rank_weighted"]

    preds = []
    metric_rows = []
    null_rows = []
    for target in targets:
        for feature_set, cols in feature_sets.items():
            for mode in modes:
                pred = leave_source_predictions(df, cols, target, feature_set, mode, args.seed)
                preds.append(pred)
                met = metric_row(pred, feature_set, mode, target)
                metric_rows.append(met)
                if target == "future_any_drop_within_16cycles" and feature_set in {"acquisition_context", "echem_regime", "video_all", "video_plus_echem"}:
                    null = permutation_null(pred, met["roc_auc"], args.seed, args.n_permutation)
                    null.update({"target": target, "feature_set": feature_set, "mode": mode, "observed_roc_auc": met["roc_auc"]})
                    null_rows.append(null)

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        metrics_df = metrics_df.merge(
            null_df[["target", "feature_set", "mode", "empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]],
            on=["target", "feature_set", "mode"], how="left",
        )
    delta_df = deltas(metrics_df)
    source_df = source_summary(df)

    paths = {
        "source_summary": out / "source_balanced_source_summary.csv",
        "metrics": out / "source_balanced_video_echem_metrics.csv",
        "predictions": out / "source_balanced_video_echem_predictions.csv",
        "deltas": out / "source_balanced_video_echem_deltas.csv",
        "permutation_null": out / "source_balanced_video_echem_permutation_null.csv",
        "summary": out / "source_balanced_video_echem_summary.json",
    }
    source_df.to_csv(paths["source_summary"], index=False)
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)

    sorted_metrics = metrics_df.sort_values(["target", "roc_auc", "average_precision"], ascending=[True, False, False])
    summary = clean_json({
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "targets": targets,
        "modes": modes,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "source_summary": source_df.to_dict("records"),
        "top_metrics": sorted_metrics.to_dict("records"),
        "top_deltas": delta_df.to_dict("records") if not delta_df.empty else [],
        "guardrail": "Leave-source labels are highly source-composition imbalanced. Source weighting and within-source rank normalization test domain robustness and review-prioritization only; they do not validate source-transferable warning, causal degradation mechanisms, manual QC labels, or calibrated diffusion.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Source-Balanced Video/Echem Transfer Audit\n\n"
        "Tests whether source/class balancing and within-source feature ranks rescue leave-source future-drop prediction.\n\n"
        f"- Rows: {summary['n_rows']}\n"
        f"- Cycles: {summary['n_cycles']}\n"
        f"- Sources: {summary['n_sources']}\n"
        f"- Feature sets: {summary['feature_set_sizes']}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
