#!/usr/bin/env python3
"""Benchmark future8 optical-physics signal after acquisition/echem controls.

The project has repeatedly found that short-horizon future8 labels are easy to
predict from acquisition/context. This audit asks a narrower question:

Do video/optical physics descriptors add source-held-out future8 signal after
acquisition residualization, cycle-balanced fitting, and echem-context baselines?

It is a guardrail benchmark. Passing it would justify stronger short-horizon
video-physics follow-up. Failing it keeps future8 as a context/echem-dominated
warning label rather than a physical mechanism readout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

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


def finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def numeric_cols(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 16) -> list[str]:
    out: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            out.append(col)
    return out


def add_context_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
        for val in vals.value_counts().head(16).index:
            safe = "".join(ch if ch.isalnum() else "_" for ch in val.lower()).strip("_") or "missing"
            out[f"{col}__{safe}"] = (vals == val).astype(float)
    return out


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    acquisition = numeric_cols(
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
    echem = numeric_cols(
        df,
        [
            "capacity_mAh",
            "capacity_fade_from_first_mAh",
            "capacity_fraction_of_first",
            "coulombic_efficiency_pct",
            "coulombic_inefficiency_pct",
            "charge_discharge_capacity_gap_mAh",
            "charge_discharge_capacity_abs_gap_mAh",
            "signed_charge_fraction",
            "shape_V_range",
            "shape_V_mean",
            "shape_V_std",
            "shape_I_abs_mean_mA",
            "shape_I_pos_fraction",
            "shape_I_neg_fraction",
            "shape_dVdt_abs_p95",
            "shape_dVdt_sign_consistency",
            "all_dq_abs_lowV_frac",
            "all_dq_abs_midV_frac",
            "all_dq_abs_highV_frac",
            "all_dq_abs_peak_voltage",
            "all_dq_abs_peak_frac",
            "all_dq_abs_entropy",
            "voltage_peak_hysteresis_proxy",
            "highV_charge_discharge_imbalance",
            "midV_charge_discharge_imbalance",
            "lowV_charge_discharge_imbalance",
            "dqdv_peak_concentration",
            "dqdv_entropy_asymmetry",
            "dqdv_integral_asymmetry",
            "cycle_state_pc1",
            "cycle_state_pc2",
            "cycle_state_pc3",
            "cycle_state_pc4",
            "state_step_norm",
            "axis_step",
            "echem_outlier_score",
            "echem_regime_pc1",
            "echem_regime_pc2",
            "echem_regime_pc3",
            "echem_regime_pc4",
            "echem_shape_missing",
            "echem_ce_extreme_or_missing",
        ],
    )
    trace = numeric_cols(
        df,
        [
            c
            for c in df.columns
            if c.startswith("particle_")
            or c.startswith("particle_vs_context_")
            or c.startswith("video_embed_pc")
            or c in {"transferred_masked_residual_signature", "roi_transferred_masked_residual_signature"}
        ],
    )
    front_physics = numeric_cols(
        df,
        [
            c
            for c in df.columns
            if any(
                key in c
                for key in [
                    "phase_slope",
                    "radius2_slope",
                    "diffusion_proxy",
                    "threshold_robust",
                    "q70_",
                    "front_",
                    "masked_residual",
                ]
            )
        ],
    )
    rollout = numeric_cols(
        df,
        [
            c
            for c in df.columns
            if c.startswith("persistence_")
            or c.startswith("velocity_")
            or c.startswith("low_rank_dmd_")
            or "rollout" in c
        ],
    )
    optical_physics = sorted(set(trace + front_physics + rollout))
    return {
        "acquisition_context": acquisition,
        "echem_context": echem,
        "optical_trace_video": trace,
        "front_transport_proxy": front_physics,
        "rollout_difficulty": rollout,
        "optical_physics_all": optical_physics,
        "optical_physics_plus_echem": sorted(set(optical_physics + echem)),
    }


def model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", C=0.20, solver="liblinear", random_state=seed),
    )


def cycle_balanced_weights(df: pd.DataFrame, train_mask: pd.Series) -> np.ndarray:
    if "cycleNo" not in df.columns:
        return np.ones(int(train_mask.sum()), dtype=float)
    cycles = df.loc[train_mask, "cycleNo"].fillna("missing").astype(str)
    counts = cycles.value_counts()
    weights = cycles.map(lambda c: 1.0 / float(counts[c])).to_numpy(dtype=float)
    mean = float(np.mean(weights)) if len(weights) else 1.0
    return weights / mean if mean > 0 else weights


def residualize(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    ctx_train: pd.DataFrame,
    ctx_test: pd.DataFrame,
    sample_weight: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    xtr = x_train.apply(pd.to_numeric, errors="coerce")
    xte = x_test.apply(pd.to_numeric, errors="coerce")
    keep = [c for c in xtr.columns if xtr[c].notna().sum() >= 8 and xtr[c].nunique(dropna=True) >= 2]
    if not keep:
        return pd.DataFrame(index=x_train.index), pd.DataFrame(index=x_test.index)
    y_imp = SimpleImputer(strategy="median")
    ytr = y_imp.fit_transform(xtr[keep])
    yte = y_imp.transform(xte[keep])
    ctx_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    ctr = ctx_pipe.fit_transform(ctx_train)
    cte = ctx_pipe.transform(ctx_test)
    ridge = Ridge(alpha=2.0)
    ridge.fit(ctr, ytr, sample_weight=sample_weight)
    return (
        pd.DataFrame(ytr - ridge.predict(ctr), index=x_train.index, columns=[f"{c}_resid" for c in keep]),
        pd.DataFrame(yte - ridge.predict(cte), index=x_test.index, columns=[f"{c}_resid" for c in keep]),
    )


def grouped_oof(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    group_col: str,
    mode_name: str,
    acquisition_cols: list[str],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1])
    features = [c for c in features if c in df.columns]
    acquisition_cols = [c for c in acquisition_cols if c in df.columns]
    for group in sorted(df.loc[valid, group_col].dropna().unique()):
        test = valid & df[group_col].eq(group)
        train = valid & ~test
        meta_cols = [
            c
            for c in ["embedding_row_id", "roi_id", "cycleNo", "source_stem", "embedding_cohort", "cohort_role", target]
            if c in df.columns
        ]
        meta = df.loc[test, meta_cols].rename(columns={target: "observed"}).copy()
        meta["heldout_group"] = group
        if train.sum() < 12 or test.sum() == 0 or y[train].nunique() < 2 or not features:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_features"
            rows.extend(meta.to_dict("records"))
            continue
        weights = cycle_balanced_weights(df, train) if mode_name.endswith("cycle_balanced") else None
        if mode_name == "raw":
            x_train = df.loc[train, features]
            x_test = df.loc[test, features]
        elif mode_name in {"acquisition_residualized", "acquisition_residualized_cycle_balanced"}:
            if not acquisition_cols:
                meta["predicted_probability"] = np.nan
                meta["status"] = "skipped_no_acquisition_context"
                rows.extend(meta.to_dict("records"))
                continue
            x_train, x_test = residualize(
                df.loc[train, features],
                df.loc[test, features],
                df.loc[train, acquisition_cols],
                df.loc[test, acquisition_cols],
                weights,
            )
            if x_train.shape[1] == 0:
                meta["predicted_probability"] = np.nan
                meta["status"] = "skipped_no_residual_features"
                rows.extend(meta.to_dict("records"))
                continue
        else:
            raise ValueError(mode_name)
        clf = model(seed)
        fit_kwargs = {}
        if weights is not None:
            fit_kwargs["logisticregression__sample_weight"] = weights
        clf.fit(x_train, y[train].astype(int), **fit_kwargs)
        meta["predicted_probability"] = clf.predict_proba(x_test)[:, 1]
        meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def metric_row(pred: pd.DataFrame, target: str, group_col: str, feature_set: str, mode_name: str) -> dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    row: dict[str, Any] = {
        "target": target,
        "group_col": group_col,
        "feature_set": feature_set,
        "mode": mode_name,
        "n_eval": int(len(tmp)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_sources": int(tmp["source_stem"].nunique()) if "source_stem" in tmp else 0,
        "n_groups": int(tmp["heldout_group"].nunique()) if "heldout_group" in tmp else 0,
        "roc_auc": np.nan,
        "oriented_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2:
        auc = float(roc_auc_score(y, p))
        row["roc_auc"] = auc
        row["oriented_auc"] = float(max(auc, 1.0 - auc))
        row["average_precision"] = float(average_precision_score(y, p))
        rho, pval = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(pval)
    return row


def permutation_null(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int, strata_col: str | None = None) -> dict[str, Any]:
    if not np.isfinite(observed_auc) or n_perm <= 0:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y0 = pd.to_numeric(tmp["observed"], errors="coerce").astype(int).to_numpy()
    prob = pd.to_numeric(tmp["predicted_probability"], errors="coerce").to_numpy()
    strata = tmp[strata_col].fillna("missing").astype(str).to_numpy() if strata_col and strata_col in tmp.columns else None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        y = y0.copy()
        if strata is None:
            rng.shuffle(y)
        else:
            for val in np.unique(strata):
                idx = np.flatnonzero(strata == val)
                if len(idx) > 1 and len(np.unique(y[idx])) > 1:
                    y[idx] = rng.permutation(y[idx])
        if len(np.unique(y)) == 2:
            vals.append(float(roc_auc_score(y, prob)))
    if not vals:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    arr = np.asarray(vals)
    return {
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)),
        "null_auc_mean": float(np.mean(arr)),
        "null_auc_p95": float(np.quantile(arr, 0.95)),
        "strata_col": strata_col or "global",
    }


def add_comparison_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_specs = [
        ("acquisition_context", "raw", "acquisition_raw"),
        ("echem_context", "raw", "echem_raw"),
        ("echem_context", "acquisition_residualized_cycle_balanced", "echem_residualized_cycle_balanced"),
        ("optical_physics_all", "raw", "optical_raw"),
    ]
    for _, row in metrics.iterrows():
        if row["feature_set"] not in {"optical_physics_all", "optical_physics_plus_echem", "front_transport_proxy", "rollout_difficulty"}:
            continue
        sub = metrics[(metrics["target"] == row["target"]) & (metrics["group_col"] == row["group_col"])]
        for base_feature, base_mode, label in base_specs:
            base = sub[(sub["feature_set"] == base_feature) & (sub["mode"] == base_mode)]
            if base.empty:
                continue
            b = base.iloc[0]
            rows.append(
                {
                    "target": row["target"],
                    "group_col": row["group_col"],
                    "feature_set": row["feature_set"],
                    "mode": row["mode"],
                    "baseline": label,
                    "delta_roc_auc": row["roc_auc"] - b["roc_auc"] if pd.notna(row["roc_auc"]) and pd.notna(b["roc_auc"]) else np.nan,
                    "delta_average_precision": row["average_precision"] - b["average_precision"] if pd.notna(row["average_precision"]) and pd.notna(b["average_precision"]) else np.nan,
                    "delta_spearman_rho": row["spearman_rho"] - b["spearman_rho"] if pd.notna(row["spearman_rho"]) and pd.notna(b["spearman_rho"]) else np.nan,
                    "candidate_roc_auc": row["roc_auc"],
                    "baseline_roc_auc": b["roc_auc"],
                    "candidate_average_precision": row["average_precision"],
                    "baseline_average_precision": b["average_precision"],
                }
            )
    return pd.DataFrame(rows)


def decide(metrics: pd.DataFrame, deltas: pd.DataFrame) -> dict[str, Any]:
    def pick(feature_set: str, mode_name: str, group_col: str) -> dict[str, Any]:
        sub = metrics[
            (metrics["target"] == "future_any_drop_within_8cycles")
            & (metrics["feature_set"] == feature_set)
            & (metrics["mode"] == mode_name)
            & (metrics["group_col"] == group_col)
        ]
        return sub.iloc[0].to_dict() if not sub.empty else {}

    strict_source = pick("optical_physics_all", "acquisition_residualized_cycle_balanced", "source_stem")
    strict_source_cohort = pick("optical_physics_all", "acquisition_residualized_cycle_balanced", "source_cohort_key")
    fused_source_cohort = pick("optical_physics_plus_echem", "acquisition_residualized_cycle_balanced", "source_cohort_key")
    echem_source_cohort = pick("echem_context", "acquisition_residualized_cycle_balanced", "source_cohort_key")
    acq_source_cohort = pick("acquisition_context", "raw", "source_cohort_key")

    delta_vs_echem = np.nan
    if not deltas.empty:
        sub = deltas[
            (deltas["target"] == "future_any_drop_within_8cycles")
            & (deltas["group_col"] == "source_cohort_key")
            & (deltas["feature_set"] == "optical_physics_plus_echem")
            & (deltas["mode"] == "acquisition_residualized_cycle_balanced")
            & (deltas["baseline"] == "echem_residualized_cycle_balanced")
        ]
        if not sub.empty:
            delta_vs_echem = finite(sub.iloc[0].get("delta_roc_auc"))

    strict_pass = (
        finite(strict_source_cohort.get("roc_auc")) >= 0.65
        and finite(strict_source_cohort.get("empirical_p_ge_observed_source_stratified")) <= 0.05
        and finite(strict_source.get("roc_auc")) >= 0.60
    )
    fused_incremental = (
        finite(fused_source_cohort.get("roc_auc")) >= 0.65
        and finite(fused_source_cohort.get("empirical_p_ge_observed_source_stratified")) <= 0.05
        and np.isfinite(delta_vs_echem)
        and delta_vs_echem >= 0.03
    )
    return {
        "future8_video_physics_status": "supported" if strict_pass else "not_supported_after_controls",
        "future8_fused_video_echem_incremental_status": "supported" if fused_incremental else "not_incremental_over_echem_context",
        "strict_optical_source_stem_metric": clean_json(strict_source),
        "strict_optical_source_cohort_metric": clean_json(strict_source_cohort),
        "fused_source_cohort_metric": clean_json(fused_source_cohort),
        "echem_source_cohort_metric": clean_json(echem_source_cohort),
        "acquisition_source_cohort_metric": clean_json(acq_source_cohort),
        "fused_minus_echem_residualized_source_cohort_delta_auc": clean_json(delta_vs_echem),
        "decision_rule": "Support requires source-cohort residualized/cycle-balanced optical AUC >= 0.65 with source-stratified permutation p <= 0.05 and leave-source AUC >= 0.60. Fused video+echem must also improve over residualized echem by >= 0.03 AUC.",
    }


def write_readme(out: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Residualized Future8 Video-Physics Benchmark",
        "",
        "This benchmark tests whether short-horizon future8 weak labels carry source-robust video/optical-physics signal after acquisition residualization and echem-context comparison.",
        "",
        f"- Rows/cycles/sources: {summary.get('n_rows')} / {summary.get('n_cycles')} / {summary.get('n_sources')}",
        f"- Future8 optical status: `{summary.get('decision', {}).get('future8_video_physics_status')}`",
        f"- Fused video+echem incremental status: `{summary.get('decision', {}).get('future8_fused_video_echem_incremental_status')}`",
        "",
        "Outputs:",
        "- `residualized_future8_video_physics_metrics.csv`",
        "- `residualized_future8_video_physics_predictions.csv`",
        "- `residualized_future8_video_physics_deltas.csv`",
        "- `residualized_future8_video_physics_summary.json`",
        "",
        f"Guardrail: {summary.get('guardrail')}",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/residualized_future8_video_physics_benchmark")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--n-permutation", type=int, default=500)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = add_context_indicators(read_csv(derived / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_joined.csv"))
    target = "future_any_drop_within_8cycles"
    feature_sets = build_feature_sets(df)
    group_cols = [c for c in ["cycleNo", "source_stem", "source_cohort_key"] if c in df.columns]
    acquisition_cols = feature_sets["acquisition_context"]

    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for group_col in group_cols:
        for feature_set, cols in feature_sets.items():
            modes = ["raw"] if feature_set == "acquisition_context" else ["raw", "acquisition_residualized", "acquisition_residualized_cycle_balanced"]
            for mode_name in modes:
                pred = grouped_oof(df, cols, target, group_col, mode_name, acquisition_cols, args.seed)
                pred["target"] = target
                pred["group_col"] = group_col
                pred["feature_set"] = feature_set
                pred["mode"] = mode_name
                predictions.append(pred)
                met = metric_row(pred, target, group_col, feature_set, mode_name)
                null_global = permutation_null(pred, finite(met["roc_auc"]), args.seed, args.n_permutation)
                null_source = permutation_null(pred, finite(met["roc_auc"]), args.seed + 11, args.n_permutation, "source_stem")
                met.update(
                    {
                        "empirical_p_ge_observed_global": null_global.get("empirical_p_ge_observed"),
                        "null_auc_p95_global": null_global.get("null_auc_p95"),
                        "empirical_p_ge_observed_source_stratified": null_source.get("empirical_p_ge_observed"),
                        "null_auc_p95_source_stratified": null_source.get("null_auc_p95"),
                        "n_permutation": args.n_permutation,
                    }
                )
                metric_rows.append(met)

    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["group_col", "roc_auc", "average_precision"], ascending=[True, False, False])
    deltas_df = add_comparison_deltas(metrics_df)
    if not deltas_df.empty:
        deltas_df = deltas_df.sort_values(["group_col", "feature_set", "mode", "baseline"])
    decision = decide(metrics_df, deltas_df)

    paths = {
        "metrics": out / "residualized_future8_video_physics_metrics.csv",
        "predictions": out / "residualized_future8_video_physics_predictions.csv",
        "deltas": out / "residualized_future8_video_physics_deltas.csv",
        "summary": out / "residualized_future8_video_physics_summary.json",
        "readme": out / "README.md",
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    deltas_df.to_csv(paths["deltas"], index=False)

    summary = clean_json(
        {
            "n_rows": int(len(df)),
            "n_cycles": int(df["cycleNo"].nunique()),
            "n_sources": int(df["source_stem"].nunique()) if "source_stem" in df.columns else None,
            "target": target,
            "group_cols": group_cols,
            "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
            "decision": decision,
            "top_metrics": metrics_df.head(24).to_dict("records"),
            "key_deltas": deltas_df[
                (deltas_df["target"] == target)
                & (deltas_df["group_col"].isin(["source_stem", "source_cohort_key"]))
                & (deltas_df["feature_set"].isin(["optical_physics_all", "optical_physics_plus_echem"]))
                & (deltas_df["mode"] == "acquisition_residualized_cycle_balanced")
            ].to_dict("records")
            if not deltas_df.empty
            else [],
            "guardrail": "Future8 labels are treated as weak warning labels. This benchmark supports a video-physics claim only if optical features survive source/source-cohort holdout, acquisition residualization, cycle balancing, source-stratified permutation, and echem-context comparison.",
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_readme(out, summary)


if __name__ == "__main__":
    main()
