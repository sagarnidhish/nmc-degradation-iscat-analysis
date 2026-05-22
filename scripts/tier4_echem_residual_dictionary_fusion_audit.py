#!/usr/bin/env python3
"""Fuse label-free residual-dictionary video features with echem regime context.

The residual dictionary audit showed that next-frame residual bases beat raw
video PCA features for weak future-drop readout but not hand-crafted scalar
particle descriptors. This audit asks whether residual-basis features still add
signal after conditioning on acquisition/context and electrochemical regime
features from the echem optical atlas.
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
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 16) -> List[str]:
    keep: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def one_hot_context(df: pd.DataFrame) -> pd.DataFrame:
    cats = [c for c in ["embedding_cohort", "cohort_role", "selection_subrole"] if c in df.columns]
    if not cats:
        return df
    dummies = pd.get_dummies(df[cats].fillna("missing").astype(str), prefix=cats, dummy_na=False)
    out = pd.concat([df, dummies], axis=1)
    return out


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    context = available_numeric(df, ["cycleNo"] + [c for c in df.columns if c.startswith("embedding_cohort_") or c.startswith("cohort_role_") or c.startswith("selection_subrole_")])
    echem = available_numeric(
        df,
        [
            c for c in df.columns
            if c.startswith("shape_") or c.startswith("all_dq") or c.startswith("pos_dq") or c.startswith("neg_dq")
            or c.startswith("cycle_state_") or c.startswith("echem_regime_") or c in {
                "frames_percentile", "n_frames", "capacity_mAh", "coulombic_efficiency_pct",
                "capacity_fade_from_first_mAh", "capacity_fraction_of_first", "coulombic_inefficiency_pct",
                "charge_discharge_capacity_gap_mAh", "charge_discharge_capacity_abs_gap_mAh",
                "echem_outlier_score", "cycle_index_rank", "state_step_norm", "axis_step",
            }
        ],
    )
    resdict = available_numeric(df, [c for c in df.columns if c.startswith("resdict_") or c.startswith("dictionary_") or c.startswith("residual_energy_")])
    handcrafted = available_numeric(df, [c for c in df.columns if c.startswith("particle_") or c.startswith("particle_vs_context_") or c.startswith("mask_")])
    pca_video = available_numeric(df, [c for c in df.columns if c.startswith("video_embed_pc")])
    out = {
        "acquisition_context": context,
        "echem_regime": sorted(set(echem)),
        "residual_dictionary": resdict,
        "handcrafted_scalar": handcrafted,
        "pca_video": pca_video,
        "residual_dictionary_plus_echem": sorted(set(resdict + echem + context)),
        "residual_dictionary_plus_handcrafted": sorted(set(resdict + handcrafted)),
        "residual_dictionary_handcrafted_echem": sorted(set(resdict + handcrafted + echem + context)),
        "pca_video_plus_echem": sorted(set(pca_video + echem + context)),
        "handcrafted_plus_echem": sorted(set(handcrafted + echem + context)),
    }
    return {k: v for k, v in out.items() if v}


def class_model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed),
    )


def reg_model() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=2.0))


def loo_classification(df: pd.DataFrame, features: List[str], target: str, seed: int) -> pd.DataFrame:
    rows = []
    features = [c for c in features if c not in {target, "embedding_row_id", "roi_id", "cycleNo"}]
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
        "task": "classification", "feature_set": feature_set, "target": target,
        "n_eval": int(len(tmp)), "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_positive": int(y.sum()) if len(y) else 0, "roc_auc": np.nan, "average_precision": np.nan,
        "spearman_rho": np.nan, "spearman_p": np.nan,
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
    features = [c for c in features if c not in {target, "embedding_row_id", "roi_id", "cycleNo"}]
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
        "task": "regression", "feature_set": feature_set, "target": target,
        "n_eval": int(len(tmp)), "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "r2": np.nan, "mae": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan,
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
    comparisons = [
        ("residual_dictionary_plus_echem", "residual_dictionary"),
        ("residual_dictionary_plus_echem", "echem_regime"),
        ("residual_dictionary_handcrafted_echem", "residual_dictionary_plus_handcrafted"),
        ("residual_dictionary_handcrafted_echem", "handcrafted_plus_echem"),
        ("residual_dictionary_handcrafted_echem", "pca_video_plus_echem"),
        ("pca_video_plus_echem", "pca_video"),
        ("handcrafted_plus_echem", "handcrafted_scalar"),
    ]
    for target in sorted(metrics["target"].dropna().unique()):
        for task in sorted(metrics.loc[metrics["target"] == target, "task"].dropna().unique()):
            sub = metrics[(metrics["target"] == target) & (metrics["task"] == task)]
            for comp_name, base_name in comparisons:
                base = sub[sub["feature_set"] == base_name]
                comp = sub[sub["feature_set"] == comp_name]
                if base.empty or comp.empty:
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
    return pd.DataFrame(rows).sort_values(["delta_roc_auc", "delta_spearman_rho"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_residual_dictionary_fusion_audit")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--n-permutation", type=int, default=500)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    residual = read_csv(derived / "residual_dictionary_embedding_audit" / "residual_dictionary_embedding_features.csv")
    echem = read_csv(derived / "echem_optical_regime_atlas" / "echem_optical_regime_cycle_table.csv")
    echem = echem.drop_duplicates("cycleNo", keep="first")
    joined = residual.merge(echem, on="cycleNo", how="left", suffixes=("", "_echem"))
    joined = one_hot_context(joined)
    feature_sets = build_feature_sets(joined)

    class_targets = [
        t for t in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
        if t in joined.columns and pd.to_numeric(joined[t], errors="coerce").isin([0, 1]).sum() >= 16
    ]
    reg_targets = available_numeric(joined, [
        "transferred_masked_residual_signature", "residual_energy_mean", "dictionary_recon_error_mse_mean",
        "cross_modal_consensus_score", "particle_norm_cv", "state_step_norm",
    ], min_nonnull=16)

    metrics, predictions, nulls = [], [], []
    for target in class_targets:
        y = pd.to_numeric(joined[target], errors="coerce")
        if y[y.isin([0, 1])].nunique() < 2:
            continue
        for name, cols in feature_sets.items():
            pred = loo_classification(joined, cols, target, args.seed)
            pred["task"] = "classification"; pred["target"] = target; pred["feature_set"] = name
            predictions.append(pred)
            met = class_metrics(pred, name, target)
            metrics.append(met)
            if name in {"acquisition_context", "echem_regime", "residual_dictionary", "handcrafted_scalar", "pca_video", "residual_dictionary_plus_echem", "residual_dictionary_handcrafted_echem", "handcrafted_plus_echem", "pca_video_plus_echem"}:
                null = score_null_auc(pred, met.get("roc_auc", np.nan), args.seed, args.n_permutation)
                null.update({"task": "classification", "target": target, "feature_set": name, "observed_roc_auc": met.get("roc_auc", np.nan)})
                nulls.append(null)
    for target in reg_targets:
        for name, cols in feature_sets.items():
            pred = loo_regression(joined, cols, target)
            pred["task"] = "regression"; pred["target"] = target; pred["feature_set"] = name
            predictions.append(pred)
            metrics.append(reg_metrics(pred, name, target))

    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    null_df = pd.DataFrame(nulls)
    if not metrics_df.empty and not null_df.empty:
        metrics_df = metrics_df.merge(
            null_df[["task", "target", "feature_set", "empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]],
            on=["task", "target", "feature_set"], how="left",
        )
    delta_df = metric_deltas(metrics_df) if not metrics_df.empty else pd.DataFrame()

    paths = {
        "joined": out / "echem_residual_dictionary_joined_features.csv",
        "metrics": out / "echem_residual_dictionary_metrics.csv",
        "predictions": out / "echem_residual_dictionary_predictions.csv",
        "deltas": out / "echem_residual_dictionary_feature_set_deltas.csv",
        "permutation_null": out / "echem_residual_dictionary_permutation_null.csv",
        "summary": out / "echem_residual_dictionary_summary.json",
    }
    joined.to_csv(paths["joined"], index=False)
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)

    if not metrics_df.empty and "task" in metrics_df.columns:
        class_sub = metrics_df[metrics_df["task"] == "classification"].copy()
        reg_sub = metrics_df[metrics_df["task"] == "regression"].copy()
    else:
        class_sub = pd.DataFrame()
        reg_sub = pd.DataFrame()
    top_class = class_sub.sort_values(["roc_auc", "average_precision"], ascending=[False, False]).head(40) if {"roc_auc", "average_precision"}.issubset(class_sub.columns) and not class_sub.empty else pd.DataFrame()
    top_reg = reg_sub.sort_values(["spearman_rho", "r2"], ascending=[False, False]).head(40) if {"spearman_rho", "r2"}.issubset(reg_sub.columns) and not reg_sub.empty else pd.DataFrame()
    top_delta = delta_df.head(40) if not delta_df.empty else pd.DataFrame()
    summary = clean_json({
        "n_rows": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "embedding_cohort_counts": joined["embedding_cohort"].value_counts(dropna=False).to_dict() if "embedding_cohort" in joined else {},
        "classification_targets": class_targets,
        "regression_targets": reg_targets,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "top_classification_metrics": top_class.to_dict("records"),
        "top_regression_metrics": top_reg.to_dict("records"),
        "top_feature_set_deltas": top_delta.to_dict("records"),
        "guardrail": "Fusion uses weak cycle labels, automatic ROI crops, and echem/acquisition covariates. It tests representation conditioning and review prioritization, not deployable warning, manual particle/front labels, causal echem mechanism, or calibrated diffusion.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "README.md").write_text(
        "# Echem-Conditioned Residual Dictionary Fusion Audit\n\n"
        "Fuses label-free residual-dictionary video descriptors with echem regime and acquisition/context features under leave-one-cycle weak-label evaluation.\n\n"
        f"- Rows: {summary['n_rows']}\n"
        f"- Cycles: {summary['n_cycles']}\n"
        f"- Feature sets: {summary['feature_set_sizes']}\n\n"
        f"Guardrail: {summary['guardrail']}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
