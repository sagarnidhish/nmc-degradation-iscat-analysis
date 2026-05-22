#!/usr/bin/env python3
"""Acquisition-residualized video/echem warning audit.

The echem-video fusion audit found useful future16 signal, but future8 labels
were dominated by acquisition/context. This follow-up tests whether video/echem
features still carry weak future-drop signal after removing linear acquisition
context inside each held-out cycle/source fold.
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
from sklearn.metrics import average_precision_score, roc_auc_score
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


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 12) -> List[str]:
    out: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            out.append(col)
    return out


def add_cohort_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"source_stem", "embedding_cohort", "cohort_role"}.issubset(out.columns):
        out["source_cohort_key"] = (
            out["source_stem"].fillna("missing").astype(str)
            + "::"
            + out["embedding_cohort"].fillna("missing").astype(str)
            + "::"
            + out["cohort_role"].fillna("missing").astype(str)
        )
    for col in ["embedding_cohort", "cohort_role", "selection_subrole", "source_stem"]:
        if col not in out.columns:
            continue
        vals = out[col].fillna("missing").astype(str)
        top = vals.value_counts().head(12).index
        for val in top:
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
        + [
            c
            for c in df.columns
            if c.startswith("embedding_cohort__")
            or c.startswith("cohort_role__")
            or c.startswith("selection_subrole__")
            or c.startswith("source_stem__")
        ],
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
            "echem_shape_missing",
            "echem_ce_extreme_or_missing",
        ],
    )
    video_scalar = available_numeric(
        df,
        [
            c
            for c in df.columns
            if c.startswith("particle_") or c.startswith("particle_vs_context_")
        ],
    )
    video_embedding = available_numeric(df, [c for c in df.columns if c.startswith("video_embed_pc")])
    return {
        "acquisition_context": acquisition,
        "echem_regime": echem,
        "video_scalar": video_scalar,
        "video_embedding": video_embedding,
        "video_all": sorted(set(video_scalar + video_embedding)),
        "video_plus_echem": sorted(set(video_scalar + video_embedding + echem)),
    }


def classifier(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed),
    )


def residualize_train_test(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    ctx_train: pd.DataFrame,
    ctx_test: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_train = x_train.apply(pd.to_numeric, errors="coerce")
    numeric_test = x_test.apply(pd.to_numeric, errors="coerce")
    keep = [
        col for col in numeric_train.columns
        if numeric_train[col].notna().sum() >= 8 and numeric_train[col].nunique(dropna=True) >= 2
    ]
    if not keep:
        return pd.DataFrame(index=x_train.index), pd.DataFrame(index=x_test.index)

    y_imputer = SimpleImputer(strategy="median")
    y_train = y_imputer.fit_transform(numeric_train[keep])
    y_test = y_imputer.transform(numeric_test[keep])

    ctx_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    ctx_train_arr = ctx_pipe.fit_transform(ctx_train)
    ctx_test_arr = ctx_pipe.transform(ctx_test)

    model = Ridge(alpha=2.0)
    model.fit(ctx_train_arr, y_train, sample_weight=sample_weight)
    train_res = pd.DataFrame(
        y_train - model.predict(ctx_train_arr),
        index=x_train.index,
        columns=[f"{col}_acq_resid" for col in keep],
    )
    test_res = pd.DataFrame(
        y_test - model.predict(ctx_test_arr),
        index=x_test.index,
        columns=[f"{col}_acq_resid" for col in keep],
    )
    return train_res, test_res


def cycle_balanced_weights(df: pd.DataFrame, train_mask: pd.Series) -> np.ndarray:
    if "cycleNo" not in df.columns:
        return np.ones(int(train_mask.sum()), dtype=float)
    cycles = df.loc[train_mask, "cycleNo"].fillna("missing").astype(str)
    counts = cycles.value_counts()
    weights = cycles.map(lambda c: 1.0 / float(counts[c])).to_numpy(dtype=float)
    mean = float(np.mean(weights)) if len(weights) else 1.0
    if mean > 0:
        weights = weights / mean
    return weights


def grouped_predictions(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    group_col: str,
    seed: int,
    mode: str,
    acquisition_cols: List[str],
) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1])
    groups = df.loc[valid, group_col].dropna().unique()
    groups = sorted(groups)
    feature_cols = [c for c in feature_cols if c in df.columns and c != target]
    acquisition_cols = [c for c in acquisition_cols if c in df.columns]
    for group in groups:
        test = valid & (df[group_col] == group)
        train = valid & ~test
        meta = df.loc[test, ["embedding_row_id", "roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
        meta["heldout_group"] = group
        if train.sum() < 12 or y[train].nunique() < 2 or test.sum() == 0:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
            rows.extend(meta.to_dict("records"))
            continue
        sample_weight = cycle_balanced_weights(df, train) if mode.endswith("_cycle_balanced") else None
        if mode == "raw":
            x_train = df.loc[train, feature_cols]
            x_test = df.loc[test, feature_cols]
        elif mode in {"acquisition_residualized", "acquisition_residualized_cycle_balanced"}:
            if not acquisition_cols:
                meta["predicted_probability"] = np.nan
                meta["status"] = "skipped_no_acquisition_context"
                rows.extend(meta.to_dict("records"))
                continue
            x_train, x_test = residualize_train_test(
                df.loc[train, feature_cols],
                df.loc[test, feature_cols],
                df.loc[train, acquisition_cols],
                df.loc[test, acquisition_cols],
                sample_weight=sample_weight,
            )
            if x_train.shape[1] == 0:
                meta["predicted_probability"] = np.nan
                meta["status"] = "skipped_no_residual_features"
                rows.extend(meta.to_dict("records"))
                continue
        else:
            raise ValueError(mode)
        model = classifier(seed)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["logisticregression__sample_weight"] = sample_weight
        model.fit(x_train, y[train].astype(int), **fit_kwargs)
        meta["predicted_probability"] = model.predict_proba(x_test)[:, 1]
        meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def metrics(pred: pd.DataFrame, feature_set: str, target: str, group_col: str, mode: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    row: Dict[str, Any] = {
        "feature_set": feature_set,
        "target": target,
        "group_col": group_col,
        "mode": mode,
        "n_eval": int(len(tmp)),
        "n_groups": int(tmp["heldout_group"].nunique()) if "heldout_group" in tmp else 0,
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


def permutation_null(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int) -> Dict[str, Any]:
    if not np.isfinite(observed_auc) or n_perm <= 0:
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
    if not vals:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    arr = np.asarray(vals)
    return {
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)),
        "null_auc_mean": float(np.mean(arr)),
        "null_auc_p95": float(np.quantile(arr, 0.95)),
    }


def deltas(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("video_plus_echem", "acquisition_context", "raw"),
        ("video_plus_echem", "echem_regime", "raw"),
        ("video_plus_echem", "video_all", "raw"),
        ("video_plus_echem", "acquisition_context", "acquisition_residualized"),
        ("echem_regime", "acquisition_context", "acquisition_residualized"),
        ("video_all", "acquisition_context", "acquisition_residualized"),
        ("video_plus_echem", "video_all", "acquisition_residualized"),
        ("video_plus_echem", "echem_regime", "acquisition_residualized"),
        ("video_plus_echem", "acquisition_context", "acquisition_residualized_cycle_balanced"),
        ("echem_regime", "acquisition_context", "acquisition_residualized_cycle_balanced"),
        ("video_all", "acquisition_context", "acquisition_residualized_cycle_balanced"),
        ("video_plus_echem", "video_all", "acquisition_residualized_cycle_balanced"),
        ("video_plus_echem", "echem_regime", "acquisition_residualized_cycle_balanced"),
    ]
    for target in sorted(metrics_df["target"].dropna().unique()):
        for group_col in sorted(metrics_df["group_col"].dropna().unique()):
            sub = metrics_df[(metrics_df["target"] == target) & (metrics_df["group_col"] == group_col)]
            for comp, base, mode in comparisons:
                comp_row = sub[(sub["feature_set"] == comp) & (sub["mode"] == mode)]
                base_mode = mode if mode == "raw" else "raw"
                base_row = sub[(sub["feature_set"] == base) & (sub["mode"] == base_mode)]
                if comp_row.empty or base_row.empty:
                    continue
                c = comp_row.iloc[0]
                b = base_row.iloc[0]
                rows.append({
                    "target": target,
                    "group_col": group_col,
                    "comparison": f"{comp}_{mode}_minus_{base}_{base_mode}",
                    "delta_roc_auc": c["roc_auc"] - b["roc_auc"] if pd.notna(c["roc_auc"]) and pd.notna(b["roc_auc"]) else np.nan,
                    "delta_average_precision": c["average_precision"] - b["average_precision"] if pd.notna(c["average_precision"]) and pd.notna(b["average_precision"]) else np.nan,
                    "delta_spearman_rho": c["spearman_rho"] - b["spearman_rho"] if pd.notna(c["spearman_rho"]) and pd.notna(b["spearman_rho"]) else np.nan,
                    "comparison_metric": c["roc_auc"],
                    "base_metric": b["roc_auc"],
                    "comparison_spearman_rho": c["spearman_rho"],
                    "base_spearman_rho": b["spearman_rho"],
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target", "group_col", "delta_roc_auc"], ascending=[True, True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/acquisition_residualized_video_echem_warning")
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--n-permutation", type=int, default=500)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined_path = derived / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_joined.csv"
    df = add_cohort_indicators(read_csv(joined_path))
    feature_sets = build_feature_sets(df)
    acquisition_cols = feature_sets["acquisition_context"]
    targets = [
        t for t in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
        if t in df.columns and pd.to_numeric(df[t], errors="coerce").isin([0, 1]).sum() >= 16
    ]
    group_cols = [c for c in ["cycleNo", "source_stem", "source_cohort_key"] if c in df.columns]

    prediction_frames = []
    metric_rows = []
    null_rows = []
    for target in targets:
        for group_col in group_cols:
            for feature_set, cols in feature_sets.items():
                modes = (
                    ["raw"]
                    if feature_set == "acquisition_context"
                    else ["raw", "acquisition_residualized", "acquisition_residualized_cycle_balanced"]
                )
                for mode in modes:
                    pred = grouped_predictions(df, cols, target, group_col, args.seed, mode, acquisition_cols)
                    pred["target"] = target
                    pred["group_col"] = group_col
                    pred["feature_set"] = feature_set
                    pred["mode"] = mode
                    prediction_frames.append(pred)
                    met = metrics(pred, feature_set, target, group_col, mode)
                    metric_rows.append(met)
                    if (
                        target == "future_any_drop_within_16cycles"
                        and feature_set in {"acquisition_context", "echem_regime", "video_all", "video_plus_echem"}
                        and mode in {"raw", "acquisition_residualized", "acquisition_residualized_cycle_balanced"}
                    ):
                        null = permutation_null(pred, met["roc_auc"], args.seed, args.n_permutation)
                        null.update({
                            "target": target,
                            "group_col": group_col,
                            "feature_set": feature_set,
                            "mode": mode,
                            "observed_roc_auc": met["roc_auc"],
                        })
                        null_rows.append(null)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    null_df = pd.DataFrame(null_rows)
    if not metrics_df.empty and not null_df.empty:
        metrics_df = metrics_df.merge(
            null_df[["target", "group_col", "feature_set", "mode", "empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]],
            on=["target", "group_col", "feature_set", "mode"],
            how="left",
        )
    delta_df = deltas(metrics_df) if not metrics_df.empty else pd.DataFrame()

    paths = {
        "metrics": out / "acquisition_residualized_video_echem_metrics.csv",
        "predictions": out / "acquisition_residualized_video_echem_predictions.csv",
        "deltas": out / "acquisition_residualized_video_echem_deltas.csv",
        "permutation_null": out / "acquisition_residualized_video_echem_permutation_null.csv",
        "summary": out / "acquisition_residualized_video_echem_summary.json",
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)

    top = metrics_df.sort_values(["target", "group_col", "roc_auc", "average_precision"], ascending=[True, True, False, False])
    strict_future16 = metrics_df[
        (metrics_df["target"] == "future_any_drop_within_16cycles")
        & (metrics_df["feature_set"] == "video_plus_echem")
        & (metrics_df["mode"] == "acquisition_residualized_cycle_balanced")
    ].sort_values(["group_col"])
    strict_deltas = delta_df[
        (delta_df["target"] == "future_any_drop_within_16cycles")
        & delta_df["comparison"].str.startswith("video_plus_echem_acquisition_residualized_cycle_balanced", na=False)
    ] if not delta_df.empty else pd.DataFrame()
    summary = clean_json({
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()) if "source_stem" in df.columns else None,
        "targets": targets,
        "group_cols": group_cols,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "top_metrics": top.to_dict("records"),
        "top_deltas": delta_df.head(60).to_dict("records") if not delta_df.empty else [],
        "strict_cycle_balanced_source_cohort_metrics": strict_future16.to_dict("records"),
        "strict_cycle_balanced_source_cohort_deltas": strict_deltas.to_dict("records"),
        "guardrail": "This audit residualizes candidate video/echem features against acquisition/context inside each held-out fold. It tests context-resistant weak-label signal for prioritizing next analyses, not deployable warning, causal mechanism, manual QC labels, or calibrated diffusion.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Acquisition-Residualized Video/Echem Warning Audit\n\n"
        "Tests whether video/echem features retain future-drop signal after fold-local acquisition/context residualization.\n\n"
        f"- Rows: {summary['n_rows']}\n"
        f"- Cycles: {summary['n_cycles']}\n"
        f"- Sources: {summary['n_sources']}\n"
        f"- Feature sets: {summary['feature_set_sizes']}\n"
        f"- Strict future16 metrics: {len(summary['strict_cycle_balanced_source_cohort_metrics'])} cycle-balanced residualized video+echem rows\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
