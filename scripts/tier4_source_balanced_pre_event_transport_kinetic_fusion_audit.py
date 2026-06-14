#!/usr/bin/env python3
"""Fuse source-balanced pre-event transport, rollout, front, and kinetic evidence.

This is a guardrail audit for particle-local physics descriptors. It joins the
manual-QC decision ledger with masked rollout and optical-flow transport outputs,
then asks whether transport proxies add source-aware pre-event signal beyond the
existing phase/front/kinetic descriptors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["near_vs_any_non_near", "near_vs_post_control", "near_vs_mid_pre", "near_vs_far_pre"]

FLOW_FEATURES = [
    "abs_radial_flow_mean",
    "abs_radial_flow_q90",
    "particle_flow_mag_mean",
    "particle_flow_mag_q90",
    "curl_mean",
    "curl_q90",
    "flow_acceleration_mean",
    "apparent_transport_instability_score",
]
ROLLOUT_FEATURES = [
    "persistence_particle_mse",
    "particle_vs_context_persistence_mse_ratio",
    "pixel_linear_particle_mse_gain_vs_persistence",
    "radial_profile_trend_particle_mse_gain_vs_persistence",
    "best_nonpersistence_particle_gain",
]
KINETIC_FEATURES = [
    "kinetic_evidence_score",
    "masked_minus_bg_slope",
    "masked_mean_total_variation",
    "q55_phase_fraction_delta",
    "q55_phase_fraction_slope",
    "q65_phase_fraction_slope",
    "q75_phase_fraction_slope",
]
FRONT_FEATURES = [
    "front_evidence_score",
    "front_kinetic_concordance_score",
    "front_consensus_score",
    "front_residual_outward_z_mean",
    "front_raw_outward_z_mean",
    "front_radius2_slope_px2_per_norm_time",
    "strict_qc_priority_score",
]
QC_FEATURES = [
    "qc_evidence_score",
    "visual_review_score",
    "visual_front_plausibility_score",
    "visual_sanity_score",
    "manual_qc_decision_score",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def finite_float(x: Any) -> float | None:
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def robust_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.median(skipna=True)
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    scale = (q3 - q1) / 1.349 if np.isfinite(q3 - q1) and q3 > q1 else x.std(skipna=True)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (x - med) / scale


def source_residual(df: pd.DataFrame, col: str) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    return x - x.groupby(df["source_stem"]).transform("mean")


def available(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def score_from_features(df: pd.DataFrame, cols: Sequence[str], prefix: str, residual: bool = False) -> pd.Series:
    zs = []
    used = []
    for col in available(df, cols):
        x = source_residual(df, col) if residual else pd.to_numeric(df[col], errors="coerce")
        z = robust_z(x)
        zs.append(z)
        used.append(col)
    if not zs:
        return pd.Series(np.nan, index=df.index, name=prefix)
    return pd.concat(zs, axis=1).mean(axis=1).rename(prefix)


def labels_for_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    if target == "near_vs_any_non_near":
        out["target_label"] = (out.event_relative_bin == "near_pre_event_1_8").astype(int)
    elif target == "near_vs_post_control":
        out = out[out.event_relative_bin.isin(["near_pre_event_1_8", "post_event_1_16", "no_near_event_control"])].copy()
        out["target_label"] = (out.event_relative_bin == "near_pre_event_1_8").astype(int)
    elif target == "near_vs_mid_pre":
        out = out[out.event_relative_bin.isin(["near_pre_event_1_8", "mid_pre_event_9_16"])].copy()
        out["target_label"] = (out.event_relative_bin == "near_pre_event_1_8").astype(int)
    elif target == "near_vs_far_pre":
        out = out[out.event_relative_bin.isin(["near_pre_event_1_8", "far_pre_event_17_32"])].copy()
        out["target_label"] = (out.event_relative_bin == "near_pre_event_1_8").astype(int)
    else:
        out = out.iloc[0:0].copy()
        out["target_label"] = []
    return out


def auc_ap(y: np.ndarray, score: np.ndarray) -> Tuple[float, float, str]:
    ok = np.isfinite(score)
    y = y[ok].astype(int)
    score = score[ok].astype(float)
    if len(y) < 4 or len(np.unique(y)) < 2:
        return np.nan, np.nan, "NA"
    auc = float(roc_auc_score(y, score))
    direction = "higher_in_positive"
    oriented_score = score
    oriented_auc = auc
    if auc < 0.5:
        oriented_auc = 1.0 - auc
        oriented_score = -score
        direction = "lower_in_positive"
    ap = float(average_precision_score(y, oriented_score))
    return oriented_auc, ap, direction


def source_stratified_perm_p(sub: pd.DataFrame, score_col: str, observed_auc: float, n_perm: int = 500, seed: int = 20260522) -> float:
    if not np.isfinite(observed_auc):
        return np.nan
    rng = np.random.default_rng(seed)
    y = sub["target_label"].to_numpy(dtype=int)
    score = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    groups = sub.loc[ok, "source_stem"].to_numpy()
    if len(np.unique(y)) < 2:
        return np.nan
    obs = abs(observed_auc - 0.5)
    hits = 0
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) > 1:
                yp[idx] = rng.permutation(yp[idx])
        if len(np.unique(yp)) < 2:
            continue
        auc = roc_auc_score(yp, score)
        if abs(auc - 0.5) >= obs - 1e-12:
            hits += 1
    return float((hits + 1) / (n_perm + 1))


def test_score(df: pd.DataFrame, target: str, score_col: str, n_perm: int) -> Dict[str, Any] | None:
    sub = labels_for_target(df, target)
    sub = sub[np.isfinite(pd.to_numeric(sub[score_col], errors="coerce"))].copy()
    if len(sub) < 8 or sub.target_label.nunique() < 2:
        return None
    y = sub.target_label.to_numpy(dtype=int)
    x = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)
    auc, ap, direction = auc_ap(y, x)
    oriented_x = x if direction != "lower_in_positive" else -x
    pos = oriented_x[y == 1]
    neg = oriented_x[y == 0]
    try:
        mwu_p = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    except Exception:
        mwu_p = np.nan
    try:
        rho, sp = stats.spearmanr(y, oriented_x)
    except Exception:
        rho, sp = np.nan, np.nan
    return {
        "target": target,
        "score": score_col,
        "n": int(len(sub)),
        "n_positive": int(y.sum()),
        "direction": direction,
        "oriented_auc": auc,
        "average_precision": ap,
        "median_positive_minus_negative_oriented": float(np.nanmedian(pos) - np.nanmedian(neg)),
        "mwu_p": mwu_p,
        "spearman_rho_oriented": finite_float(rho),
        "spearman_p": finite_float(sp),
        "source_stratified_permutation_p": source_stratified_perm_p(sub, score_col, auc, n_perm=n_perm),
    }


def leave_source_model(df: pd.DataFrame, target: str, feature_set: str, features: Sequence[str]) -> Dict[str, Any]:
    sub = labels_for_target(df, target)
    feats = [f for f in features if f in sub.columns]
    sub = sub.dropna(subset=["source_stem"]).copy()
    if len(feats) == 0 or len(sub) < 12 or sub.target_label.nunique() < 2:
        return {"target": target, "feature_set": feature_set, "status": "insufficient_data", "n_features": len(feats)}
    preds = np.full(len(sub), np.nan)
    groups = sub.source_stem.to_numpy()
    X_all = sub[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_all = sub.target_label.to_numpy(dtype=int)
    n_folds = 0
    skipped = 0
    for g in np.unique(groups):
        train = groups != g
        test = groups == g
        if y_all[train].min() == y_all[train].max() or y_all[test].min() == y_all[test].max():
            skipped += 1
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        )
        model.fit(X_all[train], y_all[train])
        preds[test] = model.predict_proba(X_all[test])[:, 1]
        n_folds += 1
    ok = np.isfinite(preds)
    if ok.sum() < 8 or len(np.unique(y_all[ok])) < 2:
        return {"target": target, "feature_set": feature_set, "status": "insufficient_eval", "n_features": len(feats), "n_scored": int(ok.sum()), "n_folds": n_folds, "n_skipped_folds": skipped}
    auc = float(roc_auc_score(y_all[ok], preds[ok]))
    ap = float(average_precision_score(y_all[ok], preds[ok]))
    rho, sp = stats.spearmanr(y_all[ok], preds[ok])
    return {
        "target": target,
        "feature_set": feature_set,
        "status": "ok",
        "n_features": len(feats),
        "features": feats,
        "n_eval": int(ok.sum()),
        "n_positive": int(y_all[ok].sum()),
        "n_sources_eval": int(len(np.unique(groups[ok]))),
        "n_folds": int(n_folds),
        "n_skipped_folds": int(skipped),
        "roc_auc": auc,
        "average_precision": ap,
        "spearman_rho": finite_float(rho),
        "spearman_p": finite_float(sp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_transport_kinetic_fusion_audit")
    parser.add_argument("--n-perm", type=int, default=500)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manual = read_csv(derived / "source_balanced_pre_event_manual_qc_decision_packet" / "source_balanced_pre_event_manual_qc_decision_queue.csv")
    flow = read_csv(derived / "source_balanced_pre_event_optical_flow_transport_audit" / "source_balanced_pre_event_optical_flow_transport_per_roi.csv")
    rollout = read_csv(derived / "source_balanced_pre_event_masked_rollout_benchmark" / "source_balanced_pre_event_masked_rollout_per_roi.csv")
    if manual.empty:
        raise FileNotFoundError("manual QC decision queue is required")

    flow_keep = ["roi_id"] + available(flow, FLOW_FEATURES + ["mask_fraction", "particle_context_flow_ratio"])
    rollout_keep = ["roi_id"] + available(rollout, ROLLOUT_FEATURES + ["history_mask_fraction"])
    df = manual.merge(flow[flow_keep], on="roi_id", how="left", suffixes=("", "_flow")) if not flow.empty else manual.copy()
    df = df.merge(rollout[rollout_keep], on="roi_id", how="left", suffixes=("", "_rollout")) if not rollout.empty else df

    df["transport_score_raw"] = score_from_features(df, FLOW_FEATURES, "transport_score_raw", residual=False)
    df["transport_score_source_residual"] = score_from_features(df, FLOW_FEATURES, "transport_score_source_residual", residual=True)
    df["rollout_difficulty_score_raw"] = score_from_features(df, ROLLOUT_FEATURES, "rollout_difficulty_score_raw", residual=False)
    df["rollout_difficulty_score_source_residual"] = score_from_features(df, ROLLOUT_FEATURES, "rollout_difficulty_score_source_residual", residual=True)
    df["kinetic_score_raw"] = score_from_features(df, KINETIC_FEATURES, "kinetic_score_raw", residual=False)
    df["kinetic_score_source_residual"] = score_from_features(df, KINETIC_FEATURES, "kinetic_score_source_residual", residual=True)
    df["front_score_raw"] = score_from_features(df, FRONT_FEATURES, "front_score_raw", residual=False)
    df["front_score_source_residual"] = score_from_features(df, FRONT_FEATURES, "front_score_source_residual", residual=True)
    df["qc_score_raw"] = score_from_features(df, QC_FEATURES, "qc_score_raw", residual=False)

    component_raw = ["transport_score_raw", "rollout_difficulty_score_raw", "kinetic_score_raw", "front_score_raw"]
    component_sr = ["transport_score_source_residual", "rollout_difficulty_score_source_residual", "kinetic_score_source_residual", "front_score_source_residual"]
    df["transport_rollout_fusion_score"] = pd.concat([robust_z(df[c]) for c in component_raw[:2]], axis=1).mean(axis=1)
    df["transport_kinetic_front_fusion_score"] = pd.concat([robust_z(df[c]) for c in component_raw], axis=1).mean(axis=1)
    df["source_guarded_transport_kinetic_front_score"] = pd.concat([robust_z(df[c]) for c in component_sr], axis=1).mean(axis=1)
    df["manual_qc_augmented_fusion_score"] = pd.concat([robust_z(df["transport_kinetic_front_fusion_score"]), robust_z(df["qc_score_raw"])], axis=1).mean(axis=1)

    score_cols = component_raw + component_sr + [
        "transport_rollout_fusion_score",
        "transport_kinetic_front_fusion_score",
        "source_guarded_transport_kinetic_front_score",
        "manual_qc_augmented_fusion_score",
    ]
    tests: List[Dict[str, Any]] = []
    for target in TARGETS:
        for score_col in score_cols:
            res = test_score(df, target, score_col, args.n_perm)
            if res is not None:
                tests.append(res)
    tests_df = pd.DataFrame(tests).sort_values(["mwu_p", "oriented_auc"], ascending=[True, False]) if tests else pd.DataFrame()

    feature_sets = {
        "transport_only": available(df, FLOW_FEATURES),
        "rollout_only": available(df, ROLLOUT_FEATURES),
        "kinetic_front": available(df, KINETIC_FEATURES + FRONT_FEATURES),
        "transport_plus_kinetic_front": available(df, FLOW_FEATURES + KINETIC_FEATURES + FRONT_FEATURES),
        "all_core_no_visual": available(df, FLOW_FEATURES + ROLLOUT_FEATURES + KINETIC_FEATURES + FRONT_FEATURES),
    }
    model_rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        for name, feats in feature_sets.items():
            model_rows.append(leave_source_model(df, target, name, feats))
    model_df = pd.DataFrame(model_rows)

    rank_cols = ["transport_kinetic_front_fusion_score", "source_guarded_transport_kinetic_front_score", "transport_score_source_residual", "manual_qc_decision_score"]
    for col in rank_cols:
        if col not in df.columns:
            df[col] = np.nan
    df["fusion_review_priority_score"] = pd.concat([robust_z(df[c]) for c in rank_cols], axis=1).mean(axis=1)
    top_cols = [
        "roi_id", "cycleNo", "source_stem", "event_relative_bin", "manual_qc_rank", "manual_qc_action_tier",
        "fusion_review_priority_score", "transport_kinetic_front_fusion_score", "source_guarded_transport_kinetic_front_score",
        "transport_score_raw", "transport_score_source_residual", "rollout_difficulty_score_raw", "kinetic_score_raw", "front_score_raw",
        "abs_radial_flow_mean", "persistence_particle_mse", "kinetic_evidence_score", "front_evidence_score", "front_kinetic_tier",
        "frame_strip_png", "mask_overlay_png", "kymograph_png",
    ]
    ranked = df[[c for c in top_cols if c in df.columns]].sort_values("fusion_review_priority_score", ascending=False)

    paths = {
        "per_roi_csv": out / "source_balanced_pre_event_transport_kinetic_fusion_per_roi.csv",
        "event_tests_csv": out / "source_balanced_pre_event_transport_kinetic_fusion_event_tests.csv",
        "leave_source_models_csv": out / "source_balanced_pre_event_transport_kinetic_fusion_leave_source_models.csv",
        "ranked_candidates_csv": out / "source_balanced_pre_event_transport_kinetic_fusion_ranked_candidates.csv",
        "summary_json": out / "source_balanced_pre_event_transport_kinetic_fusion_summary.json",
        "readme": out / "README.md",
    }
    df.to_csv(paths["per_roi_csv"], index=False)
    tests_df.to_csv(paths["event_tests_csv"], index=False)
    model_df.to_csv(paths["leave_source_models_csv"], index=False)
    ranked.to_csv(paths["ranked_candidates_csv"], index=False)

    best_by_target = []
    if not tests_df.empty:
        for target, g in tests_df.groupby("target", sort=False):
            best_by_target.append(g.iloc[0].to_dict())
    model_ok = model_df[model_df.get("status", "") == "ok"].copy() if not model_df.empty else pd.DataFrame()
    top_models = model_ok.sort_values(["roc_auc", "average_precision"], ascending=False).head(16).to_dict("records") if not model_ok.empty else []
    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "event_relative_bin_counts": df["event_relative_bin"].value_counts().to_dict(),
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "score_columns": score_cols,
        "best_event_tests_by_target": best_by_target,
        "top_event_tests": tests_df.head(16).to_dict("records") if not tests_df.empty else [],
        "top_leave_source_models": top_models,
        "top_ranked_candidates": ranked.head(16).to_dict("records"),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Fusion scores join automatic transport, rollout, front, and kinetic descriptors from history-derived particle ROI crops. They are ranking and source-aware hypothesis-generation tools, not manual QC labels, calibrated phase-boundary velocities, diffusion coefficients, or causal degradation proof.",
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    readme = [
        "# Source-Balanced Pre-Event Transport/Kinetic Fusion Audit",
        "",
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Event bins: {summary['event_relative_bin_counts']}",
        f"- Feature set sizes: {summary['feature_set_sizes']}",
        f"- Top event test: {summary['top_event_tests'][0] if summary['top_event_tests'] else 'none'}",
        f"- Top leave-source model: {summary['top_leave_source_models'][0] if summary['top_leave_source_models'] else 'none'}",
        "",
        "Outputs:",
    ]
    readme += [f"- `{Path(v).name}`" for k, v in paths.items() if k not in {"summary_json", "readme"}]
    readme += ["", "Guardrail:", summary["guardrail"]]
    paths["readme"].write_text("\n".join(readme) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
