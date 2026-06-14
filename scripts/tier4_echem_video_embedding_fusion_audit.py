#!/usr/bin/env python3
"""Test whether masked video embeddings add signal beyond echem context.

The masked-video embedding audit extracts label-free particle-region video
descriptors. The echem regime atlas extracts cycle-level electrochemical state.
This audit fuses them and compares feature families under leave-one-cycle-out
models, asking which signals remain useful when the other modality is present.
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 12) -> List[str]:
    out = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            out.append(col)
    return out


def one_series(df: pd.DataFrame, col: str) -> pd.Series:
    value = df[col]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value


def add_cohort_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["embedding_cohort", "cohort_role", "selection_subrole"]:
        if col not in out.columns:
            continue
        vals = out[col].fillna("missing").astype(str)
        top = vals.value_counts().head(8).index
        for val in top:
            safe = "".join(ch if ch.isalnum() else "_" for ch in val.lower()).strip("_")
            out[f"{col}__{safe}"] = (vals == val).astype(float)
    return out


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    acquisition = available_numeric(df, [
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
    ] + [c for c in df.columns if c.startswith("embedding_cohort__") or c.startswith("cohort_role__")])
    echem = available_numeric(df, [
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
        "echem_shape_missing",
        "echem_ce_extreme_or_missing",
    ])
    video_scalar = available_numeric(df, [
        c for c in df.columns
        if c.startswith("particle_") or c.startswith("particle_vs_context_")
    ])
    video_embedding = available_numeric(df, [c for c in df.columns if c.startswith("video_embed_pc")])
    return {
        "acquisition_context": acquisition,
        "echem_regime": echem,
        "video_scalar": video_scalar,
        "video_embedding": video_embedding,
        "video_all": sorted(set(video_scalar + video_embedding)),
        "video_plus_echem": sorted(set(video_scalar + video_embedding + echem)),
        "video_plus_echem_acquisition": sorted(set(video_scalar + video_embedding + echem + acquisition)),
    }


def classifier(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed),
    )


def regressor() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def leave_cycle_classification(df: pd.DataFrame, features: List[str], target: str, seed: int) -> pd.DataFrame:
    rows = []
    features = [c for c in features if c != target]
    use = df[["embedding_row_id", "roi_id", "cycleNo", target] + features].copy()
    y = pd.to_numeric(one_series(use, target), errors="coerce")
    valid = y.isin([0, 1])
    cycles = sorted(pd.to_numeric(use.loc[valid, "cycleNo"], errors="coerce").dropna().unique())
    for cycle in cycles:
        test_mask = valid & (pd.to_numeric(use["cycleNo"], errors="coerce") == cycle)
        train_mask = valid & ~test_mask
        y_train = y[train_mask].astype(int)
        meta = use.loc[test_mask, ["embedding_row_id", "roi_id", "cycleNo", target]].rename(columns={target: "observed"}).copy()
        if len(meta) == 0:
            continue
        if train_mask.sum() < 12 or y_train.nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
        else:
            model = classifier(seed)
            model.fit(use.loc[train_mask, features], y_train)
            meta["predicted_probability"] = model.predict_proba(use.loc[test_mask, features])[:, 1]
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def class_metrics(pred: pd.DataFrame, feature_set: str, target: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    row: Dict[str, Any] = {
        "task": "classification",
        "feature_set": feature_set,
        "target": target,
        "n_eval": int(len(tmp)),
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_positive": int(y.sum()) if len(y) else 0,
        "positive_rate": float(y.mean()) if len(y) else np.nan,
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


def leave_cycle_regression(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    rows = []
    features = [c for c in features if c != target]
    use = df[["embedding_row_id", "roi_id", "cycleNo", target] + features].copy()
    y = pd.to_numeric(one_series(use, target), errors="coerce")
    valid = y.notna()
    cycles = sorted(pd.to_numeric(use.loc[valid, "cycleNo"], errors="coerce").dropna().unique())
    for cycle in cycles:
        test_mask = valid & (pd.to_numeric(use["cycleNo"], errors="coerce") == cycle)
        train_mask = valid & ~test_mask
        meta = use.loc[test_mask, ["embedding_row_id", "roi_id", "cycleNo", target]].rename(columns={target: "observed"}).copy()
        if len(meta) == 0:
            continue
        if train_mask.sum() < 12 or y[train_mask].nunique() < 2:
            meta["predicted"] = np.nan
            meta["status"] = "skipped_train_size_or_variance"
        else:
            model = regressor()
            model.fit(use.loc[train_mask, features], y[train_mask].astype(float))
            meta["predicted"] = model.predict(use.loc[test_mask, features])
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def regression_metrics(pred: pd.DataFrame, feature_set: str, target: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted"]).copy()
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
        return {"n_permutation": int(0), "empirical_p_ge_observed": np.nan}
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce").to_numpy()
    if len(y) < 8 or len(np.unique(y)) < 2:
        return {"n_permutation": int(0), "empirical_p_ge_observed": np.nan}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_perm):
        yy = y.copy()
        rng.shuffle(yy)
        if len(np.unique(yy)) == 2:
            aucs.append(float(roc_auc_score(yy, p)))
    if not aucs:
        return {"n_permutation": int(0), "empirical_p_ge_observed": np.nan}
    arr = np.asarray(aucs)
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
            bases = {
                "acquisition_context": sub[sub["feature_set"] == "acquisition_context"],
                "echem_regime": sub[sub["feature_set"] == "echem_regime"],
                "video_all": sub[sub["feature_set"] == "video_all"],
            }
            for base_name, base in bases.items():
                if base.empty:
                    continue
                for comp_name in ["video_all", "video_plus_echem", "video_plus_echem_acquisition"]:
                    comp = sub[sub["feature_set"] == comp_name]
                    if comp.empty or comp_name == base_name:
                        continue
                    b = base.iloc[0]
                    c = comp.iloc[0]
                    rows.append({
                        "task": task,
                        "target": target,
                        "comparison": f"{comp_name}_minus_{base_name}",
                        "delta_roc_auc": (c.get("roc_auc") - b.get("roc_auc")) if pd.notna(c.get("roc_auc")) and pd.notna(b.get("roc_auc")) else np.nan,
                        "delta_average_precision": (c.get("average_precision") - b.get("average_precision")) if pd.notna(c.get("average_precision")) and pd.notna(b.get("average_precision")) else np.nan,
                        "delta_r2": (c.get("r2") - b.get("r2")) if pd.notna(c.get("r2")) and pd.notna(b.get("r2")) else np.nan,
                        "delta_spearman_rho": (c.get("spearman_rho") - b.get("spearman_rho")) if pd.notna(c.get("spearman_rho")) and pd.notna(b.get("spearman_rho")) else np.nan,
                        "base_metric": b.get("roc_auc") if task == "classification" else b.get("r2"),
                        "comparison_metric": c.get("roc_auc") if task == "classification" else c.get("r2"),
                        "base_spearman_rho": b.get("spearman_rho"),
                        "comparison_spearman_rho": c.get("spearman_rho"),
                    })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sort_cols = ["delta_roc_auc", "delta_spearman_rho"] if "delta_roc_auc" in out else ["delta_spearman_rho"]
    return out.sort_values(sort_cols, ascending=[False] * len(sort_cols))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/echem_video_embedding_fusion_audit")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-permutation", type=int, default=1000)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    embeddings = read_csv(derived / "masked_video_embedding_audit" / "masked_video_embedding_features.csv")
    echem = read_csv(derived / "echem_optical_regime_atlas" / "echem_optical_regime_cycle_table.csv")
    roi_physics = read_csv(derived / "balanced_future_roi_physics_audit" / "balanced_future_roi_physics_joined.csv")
    if embeddings.empty:
        raise FileNotFoundError("Need masked_video_embedding_audit/masked_video_embedding_features.csv")
    if echem.empty:
        raise FileNotFoundError("Need echem_optical_regime_atlas/echem_optical_regime_cycle_table.csv")

    echem_cols = [c for c in echem.columns if c != "source_stem"]
    table = embeddings.merge(echem[echem_cols], on="cycleNo", how="left", suffixes=("", "_echem"))
    if not roi_physics.empty:
        physics_cols = [
            "roi_id",
            "cycleNo",
            "transferred_masked_residual_signature",
            "threshold_robust_phase_score",
            "phase_slope_abs_median_per_s",
            "diffusion_proxy_abs_median_um2_per_s",
            "persistence_particle_mse_fraction_of_full_mean",
            "velocity_particle_mse_fraction_of_full_mean",
            "low_rank_dmd_particle_mse_fraction_of_full_mean",
            "object_mean_residual",
        ]
        physics_cols = [c for c in physics_cols if c in roi_physics.columns]
        table = table.merge(roi_physics[physics_cols], on=["roi_id", "cycleNo"], how="left", suffixes=("", "_physics"))
    table = add_cohort_indicators(table)

    feature_sets = {k: v for k, v in build_feature_sets(table).items() if v}
    class_targets = [
        "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles",
        "any_abrupt_drop",
    ]
    class_targets = [
        t for t in class_targets
        if t in table.columns and pd.to_numeric(table[t], errors="coerce").isin([0, 1]).sum() >= 16
        and pd.to_numeric(table[t], errors="coerce")[pd.to_numeric(table[t], errors="coerce").isin([0, 1])].nunique() == 2
    ]
    reg_targets = available_numeric(table, [
        "transferred_masked_residual_signature",
        "threshold_robust_phase_score",
        "phase_slope_abs_median_per_s",
        "diffusion_proxy_abs_median_um2_per_s",
        "persistence_particle_mse_fraction_of_full_mean",
        "velocity_particle_mse_fraction_of_full_mean",
        "low_rank_dmd_particle_mse_fraction_of_full_mean",
        "object_mean_residual",
        "cross_modal_consensus_score",
        "particle_norm_cv",
    ], min_nonnull=16)

    metrics: List[Dict[str, Any]] = []
    predictions = []
    nulls = []
    for target in class_targets:
        for feature_set, cols in feature_sets.items():
            pred = leave_cycle_classification(table, cols, target, args.seed)
            pred["task"] = "classification"
            pred["target"] = target
            pred["feature_set"] = feature_set
            predictions.append(pred)
            met = class_metrics(pred, feature_set, target)
            metrics.append(met)
            if feature_set in {"acquisition_context", "echem_regime", "video_all", "video_plus_echem", "video_plus_echem_acquisition"}:
                null = score_null_auc(pred, met.get("roc_auc", np.nan), args.seed, args.n_permutation)
                null.update({"task": "classification", "target": target, "feature_set": feature_set, "observed_roc_auc": met.get("roc_auc", np.nan)})
                nulls.append(null)
    for target in reg_targets:
        for feature_set, cols in feature_sets.items():
            pred = leave_cycle_regression(table, cols, target)
            pred["task"] = "regression"
            pred["target"] = target
            pred["feature_set"] = feature_set
            predictions.append(pred)
            metrics.append(regression_metrics(pred, feature_set, target))

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
        "joined": out / "echem_video_embedding_fusion_joined.csv",
        "metrics": out / "echem_video_embedding_fusion_metrics.csv",
        "predictions": out / "echem_video_embedding_fusion_predictions.csv",
        "deltas": out / "echem_video_embedding_fusion_feature_set_deltas.csv",
        "permutation_null": out / "echem_video_embedding_fusion_permutation_null.csv",
        "summary": out / "echem_video_embedding_fusion_summary.json",
    }
    table.to_csv(paths["joined"], index=False)
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)

    top_class = metrics_df[metrics_df["task"] == "classification"].sort_values(["roc_auc", "average_precision"], ascending=[False, False]).head(30) if not metrics_df.empty else pd.DataFrame()
    top_reg = metrics_df[metrics_df["task"] == "regression"].sort_values(["spearman_rho", "r2"], ascending=[False, False]).head(30) if not metrics_df.empty else pd.DataFrame()
    top_delta = delta_df.sort_values(["delta_roc_auc", "delta_spearman_rho"], ascending=[False, False]).head(30) if not delta_df.empty else pd.DataFrame()
    summary = clean_json({
        "n_embedding_rows": int(len(table)),
        "n_cycles": int(table["cycleNo"].nunique()),
        "embedding_cohort_counts": table["embedding_cohort"].value_counts(dropna=False).to_dict() if "embedding_cohort" in table.columns else {},
        "classification_targets": class_targets,
        "regression_targets": reg_targets,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "top_classification_metrics": top_class.to_dict("records"),
        "top_regression_metrics": top_reg.to_dict("records"),
        "top_feature_set_deltas": top_delta.to_dict("records"),
        "guardrail": "This is a grouped weak-label fusion audit over automatic masked-video embeddings, echem descriptors, and automatic ROI physics. It supports representation design and review prioritization, not deployable warning, manual particle/front labels, or calibrated diffusion claims.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    readme = [
        "# Echem Video Embedding Fusion Audit",
        "",
        "Fuses masked particle-video embeddings with cycle-level echem regime descriptors and compares feature families under leave-one-cycle-out models.",
        "",
        f"- Rows: {summary['n_embedding_rows']}",
        f"- Cycles: {summary['n_cycles']}",
        f"- Classification targets: {', '.join(class_targets) or 'none'}",
        f"- Regression targets: {', '.join(reg_targets) or 'none'}",
        f"- Feature set sizes: {summary['feature_set_sizes']}",
        "",
        f"Guardrail: {summary['guardrail']}",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
