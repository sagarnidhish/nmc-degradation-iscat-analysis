#!/usr/bin/env python3
"""Signed optical-loss mechanism audit for NMC ROI degradation.

Several prior audits converged on the same guarded interpretation: weak future
degradation positives often look like optical loss or contraction rather than a
clean expanding diffusion front. This script turns that into explicit,
interpretable axes and tests them under source-aware weak-label checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def robust_z(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    med = vals.median()
    mad = (vals - med).abs().median()
    if not np.isfinite(mad) or mad <= 1e-12:
        std = vals.std(ddof=0)
        scale = std if np.isfinite(std) and std > 1e-12 else 1.0
    else:
        scale = 1.4826 * mad
    return (vals - med) / scale


def available_cols(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 24) -> List[str]:
    keep = []
    for col in cols:
        vals = numeric(df, col)
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 3:
            keep.append(col)
    return keep


def source_eta2(series: pd.Series, sources: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.notna() & sources.notna()
    vals = vals[valid]
    src = sources[valid]
    if vals.nunique() < 2 or src.nunique() < 2:
        return float("nan")
    overall = vals.mean()
    total = float(((vals - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in vals.groupby(src):
        between += len(sub) * float((sub.mean() - overall) ** 2)
    return between / total


def add_axis(df: pd.DataFrame, name: str, oriented_terms: Dict[str, float]) -> List[str]:
    cols = available_cols(df, oriented_terms.keys())
    zcols = []
    for col in cols:
        zname = f"{name}__z__{col}"
        df[zname] = robust_z(numeric(df, col) * oriented_terms[col])
        zcols.append(zname)
    df[name] = df[zcols].mean(axis=1) if zcols else np.nan
    return cols


def build_axes(df: pd.DataFrame) -> Dict[str, List[str]]:
    definitions: Dict[str, Dict[str, float]] = {
        "signed_optical_loss_axis": {
            "particle_vs_context_mean_diff_positive_fraction": -1.0,
            "particle_std_diff_positive_fraction": -1.0,
            "particle_mean_diff_positive_fraction": -1.0,
            "particle_gradient_diff_positive_fraction": -1.0,
            "roi_norm_mean_delta_last_minus_first": -1.0,
            "particle_prior_area_fraction": -1.0,
        },
        "front_contraction_axis": {
            "radius2_slope_median_px2_per_s": -1.0,
            "radius2_slope_positive_fraction": -1.0,
            "diffusion_proxy_median_um2_per_s": -1.0,
            "phase_slope_positive_fraction": -1.0,
            "threshold_robust_diffusion_score": 1.0,
        },
        "rollout_difficulty_axis": {
            "transferred_masked_residual_signature": 1.0,
            "persistence_particle_mse_fraction_of_full_mean": 1.0,
            "low_rank_dmd_particle_mse_fraction_of_full_mean": 1.0,
            "velocity_particle_mse_fraction_of_full_mean": 1.0,
            "object_mean_abs_z": 1.0,
        },
        "echem_degraded_state_axis": {
            "capacity_fraction_of_first": -1.0,
            "shape_charge_mAh_neg_abs": -1.0,
            "shape_V_q10": -1.0,
            "shape_V_q05": -1.0,
            "neg_dq_abs_lowV_frac": 1.0,
            "dqdv_integral_asymmetry": 1.0,
            "cycle_state_pc3": 1.0,
            "cycle_state_pc4": -1.0,
        },
    }
    used: Dict[str, List[str]] = {}
    for axis, terms in definitions.items():
        used[axis] = add_axis(df, axis, terms)
    df["combined_loss_mechanism_axis"] = df[
        ["signed_optical_loss_axis", "front_contraction_axis", "rollout_difficulty_axis", "echem_degraded_state_axis"]
    ].mean(axis=1)
    return used


def axis_tests(df: pd.DataFrame, axes: List[str], targets: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target in targets:
        y = numeric(df, target)
        for axis in axes:
            x = numeric(df, axis)
            valid = y.isin([0, 1]) & x.notna()
            if valid.sum() < 12 or y[valid].nunique() < 2:
                continue
            yy = y[valid].astype(int)
            xx = x[valid]
            pos = xx[yy == 1]
            neg = xx[yy == 0]
            direction = "higher_in_positive" if pos.median() >= neg.median() else "lower_in_positive"
            score = xx if direction == "higher_in_positive" else -xx
            try:
                auc = float(roc_auc_score(yy, score))
                ap = float(average_precision_score(yy, score))
            except ValueError:
                auc = np.nan
                ap = np.nan
            rho, rho_p = spearmanr(yy, score)
            try:
                mw_p = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
            except ValueError:
                mw_p = np.nan
            rows.append(
                {
                    "target": target,
                    "axis": axis,
                    "n": int(valid.sum()),
                    "n_positive": int(yy.sum()),
                    "median_positive": float(pos.median()),
                    "median_negative": float(neg.median()),
                    "median_positive_minus_negative": float(pos.median() - neg.median()),
                    "direction": direction,
                    "oriented_auc": auc,
                    "average_precision": ap,
                    "spearman_rho_oriented": float(rho) if np.isfinite(rho) else np.nan,
                    "spearman_p_oriented": float(rho_p) if np.isfinite(rho_p) else np.nan,
                    "mannwhitney_p": mw_p,
                    "source_eta2": source_eta2(xx, df["source_stem"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "oriented_auc"], ascending=[True, False])


def leave_source_axis_models(df: pd.DataFrame, feature_sets: Dict[str, List[str]], targets: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    preds: List[pd.DataFrame] = []
    for target in targets:
        y = numeric(df, target)
        valid = y.isin([0, 1]) & df["source_stem"].notna()
        for set_name, cols in feature_sets.items():
            fold_parts = []
            for source in sorted(df.loc[valid, "source_stem"].dropna().unique()):
                test = valid & (df["source_stem"] == source)
                train = valid & ~test
                meta = df.loc[test, ["embedding_row_id", "roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
                if train.sum() < 12 or y[train].nunique() < 2:
                    meta["predicted_probability"] = np.nan
                    meta["status"] = "skipped"
                else:
                    imp = SimpleImputer(strategy="median")
                    scaler = StandardScaler()
                    xtr = scaler.fit_transform(imp.fit_transform(df.loc[train, cols]))
                    xte = scaler.transform(imp.transform(df.loc[test, cols]))
                    model = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.35, solver="liblinear", random_state=101)
                    model.fit(xtr, y[train].astype(int))
                    meta["predicted_probability"] = model.predict_proba(xte)[:, 1]
                    meta["status"] = "ok"
                meta["target"] = target
                meta["feature_set"] = set_name
                meta["features"] = ";".join(cols)
                meta["heldout_source"] = source
                fold_parts.append(meta)
            pred = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
            preds.append(pred)
            tmp = pred.dropna(subset=["observed", "predicted_probability"])
            yy = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
            pp = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
            out = {
                "target": target,
                "feature_set": set_name,
                "features": ";".join(cols),
                "n_eval": int(len(tmp)),
                "n_positive": int(yy.sum()) if len(yy) else 0,
                "n_sources": int(tmp["heldout_source"].nunique()) if len(tmp) else 0,
                "roc_auc": np.nan,
                "average_precision": np.nan,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
            }
            if len(tmp) >= 12 and yy.nunique() == 2:
                out["roc_auc"] = float(roc_auc_score(yy, pp))
                out["average_precision"] = float(average_precision_score(yy, pp))
                rho, pval = spearmanr(yy, pp)
                out["spearman_rho"] = float(rho)
                out["spearman_p"] = float(pval)
            rows.append(out)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def mechanism_modes(df: pd.DataFrame, axes: List[str], k: int) -> pd.DataFrame:
    cols = [c for c in axes if numeric(df, c).notna().sum() >= 12]
    if len(cols) < 2 or len(df) < k:
        return pd.DataFrame()
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imp.fit_transform(df[cols]))
    labels = KMeans(n_clusters=k, n_init=100, random_state=121).fit_predict(x)
    out = df[["embedding_row_id", "roi_id", "cycleNo", "source_stem"] + cols].copy()
    out["mechanism_cluster"] = labels
    cluster_means = out.groupby("mechanism_cluster")[cols].mean()
    cluster_names = {}
    for cluster, row in cluster_means.iterrows():
        dominant = row.sort_values(ascending=False).index[0]
        if dominant == "signed_optical_loss_axis":
            label = "optical_loss_sign_dominant"
        elif dominant == "front_contraction_axis":
            label = "front_contraction_dominant"
        elif dominant == "rollout_difficulty_axis":
            label = "rollout_difficulty_dominant"
        elif dominant == "echem_degraded_state_axis":
            label = "echem_degraded_state_dominant"
        else:
            label = "mixed_loss_mechanism"
        cluster_names[cluster] = label
    out["mechanism_label"] = out["mechanism_cluster"].map(cluster_names)
    return out


def summarize_modes(mode_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    if mode_df.empty:
        return pd.DataFrame()
    joined = mode_df.merge(
        source_df[["embedding_row_id", "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "transferred_masked_residual_signature"]],
        on="embedding_row_id",
        how="left",
    )
    rows = []
    axis_cols = [c for c in ["signed_optical_loss_axis", "front_contraction_axis", "rollout_difficulty_axis", "echem_degraded_state_axis", "combined_loss_mechanism_axis"] if c in joined]
    for label, sub in joined.groupby("mechanism_label"):
        row = {
            "mechanism_label": label,
            "n_rows": int(len(sub)),
            "n_cycles": int(sub["cycleNo"].nunique()),
            "n_sources": int(sub["source_stem"].nunique()),
            "future8_rate": float(pd.to_numeric(sub["future_any_drop_within_8cycles"], errors="coerce").mean()),
            "future16_rate": float(pd.to_numeric(sub["future_any_drop_within_16cycles"], errors="coerce").mean()),
            "median_transferred_masked_residual_signature": float(pd.to_numeric(sub["transferred_masked_residual_signature"], errors="coerce").median()),
        }
        for axis in axis_cols:
            row[f"mean_{axis}"] = float(pd.to_numeric(sub[axis], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["future16_rate", "n_rows"], ascending=[False, False])


def top_candidates(df: pd.DataFrame, axes: List[str]) -> pd.DataFrame:
    cols = ["embedding_row_id", "roi_id", "cycleNo", "source_stem", "future_any_drop_within_8cycles", "future_any_drop_within_16cycles"] + axes
    out = df[[c for c in cols if c in df.columns]].copy()
    return out.sort_values("combined_loss_mechanism_axis", ascending=False).head(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/signed_optical_loss_mechanism_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = read_csv(derived / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_joined.csv")
    axes_used = build_axes(df)
    axes = [
        "signed_optical_loss_axis",
        "front_contraction_axis",
        "rollout_difficulty_axis",
        "echem_degraded_state_axis",
        "combined_loss_mechanism_axis",
    ]
    eval_mask = numeric(df, "future_any_drop_within_16cycles").isin([0, 1]) & df["source_stem"].notna()
    eval_df = df.loc[eval_mask].copy()

    tests = axis_tests(eval_df, axes, [t for t in TARGETS if t in eval_df.columns])
    feature_sets = {
        "optical_loss_only": ["signed_optical_loss_axis"],
        "front_contraction_only": ["front_contraction_axis"],
        "rollout_difficulty_only": ["rollout_difficulty_axis"],
        "echem_degraded_only": ["echem_degraded_state_axis"],
        "optical_plus_front": ["signed_optical_loss_axis", "front_contraction_axis"],
        "optical_plus_rollout": ["signed_optical_loss_axis", "rollout_difficulty_axis"],
        "all_loss_mechanism_axes": axes,
    }
    model_metrics, predictions = leave_source_axis_models(eval_df, feature_sets, [t for t in TARGETS if t in eval_df.columns])
    modes = mechanism_modes(eval_df, axes, 4)
    mode_summary = summarize_modes(modes, eval_df)
    candidates = top_candidates(eval_df, axes)

    source_summary = (
        eval_df.groupby("source_stem")
        .agg(
            n_rows=("embedding_row_id", "size"),
            n_cycles=("cycleNo", "nunique"),
            future16_rate=("future_any_drop_within_16cycles", "mean"),
            future8_rate=("future_any_drop_within_8cycles", "mean"),
            median_signed_optical_loss_axis=("signed_optical_loss_axis", "median"),
            median_front_contraction_axis=("front_contraction_axis", "median"),
            median_rollout_difficulty_axis=("rollout_difficulty_axis", "median"),
            median_echem_degraded_state_axis=("echem_degraded_state_axis", "median"),
            median_combined_loss_mechanism_axis=("combined_loss_mechanism_axis", "median"),
        )
        .reset_index()
        .sort_values("median_combined_loss_mechanism_axis", ascending=False)
    )

    paths = {
        "axis_scores": out / "signed_optical_loss_axis_scores.csv",
        "axis_tests": out / "signed_optical_loss_axis_tests.csv",
        "axis_model_metrics": out / "signed_optical_loss_axis_model_metrics.csv",
        "axis_model_predictions": out / "signed_optical_loss_axis_model_predictions.csv",
        "mechanism_modes": out / "signed_optical_loss_mechanism_modes.csv",
        "mechanism_mode_summary": out / "signed_optical_loss_mechanism_mode_summary.csv",
        "top_candidates": out / "signed_optical_loss_top_candidates.csv",
        "source_summary": out / "signed_optical_loss_source_summary.csv",
        "summary": out / "signed_optical_loss_mechanism_summary.json",
    }
    score_cols = ["embedding_row_id", "roi_id", "cycleNo", "source_stem", "future_any_drop_within_8cycles", "future_any_drop_within_16cycles"] + axes
    df[[c for c in score_cols if c in df.columns]].to_csv(paths["axis_scores"], index=False)
    tests.to_csv(paths["axis_tests"], index=False)
    model_metrics.to_csv(paths["axis_model_metrics"], index=False)
    predictions.to_csv(paths["axis_model_predictions"], index=False)
    modes.to_csv(paths["mechanism_modes"], index=False)
    mode_summary.to_csv(paths["mechanism_mode_summary"], index=False)
    candidates.to_csv(paths["top_candidates"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)

    top_future16_tests = tests[tests["target"] == "future_any_drop_within_16cycles"].head(10)
    top_models = model_metrics.sort_values(["target", "roc_auc", "average_precision"], ascending=[True, False, False]).head(20)
    summary = clean_json(
        {
            "n_rows": int(len(df)),
            "n_eval_rows": int(len(eval_df)),
            "n_cycles": int(eval_df["cycleNo"].nunique()),
            "n_sources": int(eval_df["source_stem"].nunique()),
            "axes": axes,
            "axis_input_features": axes_used,
            "top_future16_axis_tests": top_future16_tests.to_dict("records"),
            "top_axis_model_metrics": top_models.to_dict("records"),
            "mechanism_mode_summary": mode_summary.to_dict("records"),
            "top_candidates": candidates.head(15).to_dict("records"),
            "source_summary": source_summary.to_dict("records"),
            "guardrail": "Signed optical-loss axes are computed from automatic ROI/video/echem descriptors and weak future labels. They support mechanism triage for optical loss/contraction versus front expansion, but they do not validate manual particle identity, front masks, calibrated diffusion, or deployable warnings.",
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Signed Optical-Loss Mechanism Audit\n\n"
        "Builds signed optical-loss, front-contraction, rollout-difficulty, and echem-state axes for source-aware mechanism triage.\n\n"
        f"- Rows/eval rows: {summary['n_rows']} / {summary['n_eval_rows']}\n"
        f"- Cycles/sources: {summary['n_cycles']} / {summary['n_sources']}\n"
        f"- Axes: {', '.join(axes)}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
