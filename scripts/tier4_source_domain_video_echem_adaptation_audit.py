#!/usr/bin/env python3
"""Audit source-domain transfer for future16 video/echem weak labels.

The acquisition-residualized video/echem audit found a useful leave-cycle
future16 signal but poor leave-source transfer. This follow-up quantifies source
domain shift and tests simple unlabeled target-source adaptations: source
centering and CORAL covariance alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "future_any_drop_within_16cycles"


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
    seen = set()
    for col in cols:
        if col in seen or col not in df.columns:
            continue
        seen.add(col)
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            out.append(col)
    return out


def add_source_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["embedding_cohort", "cohort_role", "selection_subrole", "source_stem"]:
        if col not in out.columns:
            continue
        vals = out[col].fillna("missing").astype(str)
        for val in vals.value_counts().head(16).index:
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
            c
            for c in df.columns
            if c.startswith(("shape_", "all_dq", "pos_dq", "neg_dq", "cycle_state_", "echem_regime_"))
            or c
            in {
                "capacity_mAh",
                "capacity_fade_from_first_mAh",
                "capacity_fraction_of_first",
                "coulombic_efficiency_pct",
                "coulombic_inefficiency_pct",
                "charge_discharge_capacity_gap_mAh",
                "charge_discharge_capacity_abs_gap_mAh",
                "signed_charge_fraction",
                "voltage_peak_hysteresis_proxy",
                "highV_charge_discharge_imbalance",
                "midV_charge_discharge_imbalance",
                "lowV_charge_discharge_imbalance",
                "dqdv_peak_concentration",
                "dqdv_entropy_asymmetry",
                "dqdv_integral_asymmetry",
                "echem_outlier_score",
                "state_step_norm",
                "axis_step",
                "echem_shape_missing",
                "echem_ce_extreme_or_missing",
            }
        ],
    )
    video = available_numeric(
        df,
        [
            c
            for c in df.columns
            if c.startswith(("particle_", "particle_vs_context_", "video_embed_pc"))
        ],
    )
    return {
        "acquisition_context": acquisition,
        "video_all": video,
        "echem_regime": echem,
        "video_plus_echem": sorted(set(video + echem)),
    }


def imputed_numeric(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(train), imp.transform(test)


def residualize_against_context(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_ctx: pd.DataFrame,
    test_ctx: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    x_train, x_test = imputed_numeric(train_x, test_x)
    ctx_train, ctx_test = imputed_numeric(train_ctx, test_ctx)
    scaler = StandardScaler()
    ctx_train = scaler.fit_transform(ctx_train)
    ctx_test = scaler.transform(ctx_test)
    model = Ridge(alpha=2.0)
    model.fit(ctx_train, x_train)
    return x_train - model.predict(ctx_train), x_test - model.predict(ctx_test)


def source_center(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return train_x - train_x.mean(axis=0, keepdims=True), test_x - test_x.mean(axis=0, keepdims=True)


def coral_train_to_target(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_center = train_x - train_x.mean(axis=0, keepdims=True)
    test_center = test_x - test_x.mean(axis=0, keepdims=True)
    if train_x.shape[0] < 3 or test_x.shape[0] < 3:
        return train_center, test_center
    cov_train = np.cov(train_center, rowvar=False) + np.eye(train_x.shape[1]) * 1e-3
    cov_test = np.cov(test_center, rowvar=False) + np.eye(test_x.shape[1]) * 1e-3
    whiten = fractional_matrix_power(cov_train, -0.5).real
    color = fractional_matrix_power(cov_test, 0.5).real
    aligned_train = train_center @ whiten @ color
    return aligned_train, test_center


def fit_predict(train_x: np.ndarray, train_y: pd.Series, test_x: np.ndarray, seed: int) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed),
    )
    model.fit(train_x, train_y.astype(int))
    return model.predict_proba(test_x)[:, 1]


def loo_source_predictions(
    df: pd.DataFrame,
    features: List[str],
    acquisition: List[str],
    feature_set: str,
    method: str,
    seed: int,
) -> pd.DataFrame:
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    rows = []
    for source in sorted(df.loc[valid, "source_stem"].unique()):
        test = valid & (df["source_stem"] == source)
        train = valid & ~test
        meta = df.loc[test, ["embedding_row_id", "roi_id", "cycleNo", "source_stem", TARGET]].rename(columns={TARGET: "observed"}).copy()
        meta["heldout_source"] = source
        if train.sum() < 12 or y[train].nunique() < 2 or test.sum() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_class_or_test_size"
            rows.extend(meta.to_dict("records"))
            continue
        if method.startswith("acq_resid"):
            train_x, test_x = residualize_against_context(
                df.loc[train, features],
                df.loc[test, features],
                df.loc[train, acquisition],
                df.loc[test, acquisition],
            )
        else:
            train_x, test_x = imputed_numeric(df.loc[train, features], df.loc[test, features])
        if method.endswith("source_centered"):
            train_x, test_x = source_center(train_x, test_x)
        elif method.endswith("coral"):
            train_x, test_x = coral_train_to_target(train_x, test_x)
        pred = fit_predict(train_x, y[train], test_x, seed)
        meta["predicted_probability"] = pred
        meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    out = pd.DataFrame(rows)
    out["feature_set"] = feature_set
    out["method"] = method
    out["target"] = TARGET
    return out


def metrics(pred: pd.DataFrame) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    row: Dict[str, Any] = {
        "feature_set": pred["feature_set"].iloc[0] if "feature_set" in pred and len(pred) else None,
        "method": pred["method"].iloc[0] if "method" in pred and len(pred) else None,
        "target": TARGET,
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
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def permutation_null(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int) -> Dict[str, Any]:
    if not np.isfinite(observed_auc):
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce").to_numpy()
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_perm):
        yy = y.copy()
        rng.shuffle(yy)
        if len(np.unique(yy)) == 2:
            aucs.append(float(roc_auc_score(yy, p)))
    arr = np.asarray(aucs)
    return {
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)) if len(arr) else np.nan,
        "null_auc_mean": float(np.mean(arr)) if len(arr) else np.nan,
        "null_auc_p95": float(np.quantile(arr, 0.95)) if len(arr) else np.nan,
    }


def source_summary(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    x = df.loc[valid, feature_cols].apply(pd.to_numeric, errors="coerce")
    x_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=feature_cols, index=x.index)
    global_mean = x_imp.mean()
    global_std = x_imp.std(ddof=0).replace(0, np.nan)
    for source, sub in df.loc[valid].groupby("source_stem"):
        idx = sub.index
        z_mean_abs = ((x_imp.loc[idx].mean() - global_mean) / global_std).abs().replace([np.inf, -np.inf], np.nan).mean()
        rows.append({
            "source_stem": source,
            "n_rows": int(len(sub)),
            "n_cycles": int(sub["cycleNo"].nunique()),
            "n_positive_future16": int(pd.to_numeric(sub[TARGET], errors="coerce").sum()),
            "future16_positive_fraction": float(pd.to_numeric(sub[TARGET], errors="coerce").mean()),
            "cycle_min": float(pd.to_numeric(sub["cycleNo"], errors="coerce").min()),
            "cycle_max": float(pd.to_numeric(sub["cycleNo"], errors="coerce").max()),
            "mean_abs_feature_z_shift": float(z_mean_abs),
        })
    return pd.DataFrame(rows).sort_values(["future16_positive_fraction", "n_rows"], ascending=[False, False])


def metric_deltas(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = metrics_df[(metrics_df["feature_set"] == "acquisition_context") & (metrics_df["method"] == "raw")]
    if baseline.empty:
        return pd.DataFrame()
    base = baseline.iloc[0]
    for _, row in metrics_df.iterrows():
        rows.append({
            "feature_set": row["feature_set"],
            "method": row["method"],
            "delta_auc_vs_acquisition_raw": row["roc_auc"] - base["roc_auc"] if pd.notna(row["roc_auc"]) and pd.notna(base["roc_auc"]) else np.nan,
            "delta_ap_vs_acquisition_raw": row["average_precision"] - base["average_precision"] if pd.notna(row["average_precision"]) and pd.notna(base["average_precision"]) else np.nan,
            "delta_rho_vs_acquisition_raw": row["spearman_rho"] - base["spearman_rho"] if pd.notna(row["spearman_rho"]) and pd.notna(base["spearman_rho"]) else np.nan,
            "roc_auc": row["roc_auc"],
            "average_precision": row["average_precision"],
            "spearman_rho": row["spearman_rho"],
        })
    return pd.DataFrame(rows).sort_values(["delta_auc_vs_acquisition_raw", "roc_auc"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_domain_video_echem_adaptation_audit")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--n-permutation", type=int, default=500)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = add_source_indicators(read_csv(derived / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_joined.csv"))
    feature_sets = build_feature_sets(df)
    acquisition = feature_sets["acquisition_context"]
    source_df = source_summary(df, feature_sets["video_plus_echem"])

    methods = ["raw", "source_centered", "coral", "acq_resid", "acq_resid_source_centered", "acq_resid_coral"]
    prediction_frames = []
    metric_rows = []
    null_rows = []
    for feature_set in ["acquisition_context", "video_all", "echem_regime", "video_plus_echem"]:
        fs_methods = ["raw"] if feature_set == "acquisition_context" else methods
        for method in fs_methods:
            pred = loo_source_predictions(df, feature_sets[feature_set], acquisition, feature_set, method, args.seed)
            prediction_frames.append(pred)
            met = metrics(pred)
            metric_rows.append(met)
            if feature_set in {"acquisition_context", "video_plus_echem", "video_all"}:
                null = permutation_null(pred, met["roc_auc"], args.seed, args.n_permutation)
                null.update({"feature_set": feature_set, "method": method, "target": TARGET, "observed_roc_auc": met["roc_auc"]})
                null_rows.append(null)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    null_df = pd.DataFrame(null_rows)
    if not metrics_df.empty and not null_df.empty:
        metrics_df = metrics_df.merge(
            null_df[["feature_set", "method", "target", "empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]],
            on=["feature_set", "method", "target"],
            how="left",
        )
    delta_df = metric_deltas(metrics_df)

    paths = {
        "metrics": out / "source_domain_video_echem_metrics.csv",
        "predictions": out / "source_domain_video_echem_predictions.csv",
        "deltas": out / "source_domain_video_echem_deltas.csv",
        "permutation_null": out / "source_domain_video_echem_permutation_null.csv",
        "source_summary": out / "source_domain_video_echem_source_summary.csv",
        "summary": out / "source_domain_video_echem_summary.json",
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)
    source_df.to_csv(paths["source_summary"], index=False)

    summary = clean_json({
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "target": TARGET,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "top_metrics": metrics_df.sort_values(["roc_auc", "average_precision"], ascending=[False, False]).to_dict("records"),
        "top_deltas": delta_df.head(30).to_dict("records") if not delta_df.empty else [],
        "source_summary": source_df.to_dict("records"),
        "guardrail": "This is an unlabeled target-source adaptation audit over weak labels and automatic ROI descriptors. Source centering/CORAL use held-out source feature distributions but not held-out labels. Results diagnose domain shift; they are not deployable warnings or causal physics proof.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Source-Domain Video/Echem Adaptation Audit\n\n"
        "Tests whether source centering or CORAL-style alignment rescues leave-source future16 video/echem transfer.\n\n"
        f"- Rows: {summary['n_rows']}\n"
        f"- Cycles: {summary['n_cycles']}\n"
        f"- Sources: {summary['n_sources']}\n"
        f"- Feature sets: {summary['feature_set_sizes']}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
