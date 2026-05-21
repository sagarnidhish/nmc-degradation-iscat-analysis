#!/usr/bin/env python3
"""Audit empirical uncertainty calibration of ROI-only rollout residuals.

The deterministic rollout baselines already show that persistence is hard to
beat on raw pixels. This script asks a different AI/physics question: do
empirical error bands widen where optical phase-transition proxies say they
should, and are event/control ROIs differently calibrated?

It uses only particle-region ROI crops through the existing frame-level rollout
metrics. The output is a calibration and triage audit, not a probabilistic
neural video model.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def finite_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_mwu(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    aa = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    bb = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(aa) == 0 or len(bb) == 0:
        return {"n_a": int(len(aa)), "n_b": int(len(bb)), "median_a": np.nan, "median_b": np.nan, "median_diff": np.nan, "p_value": np.nan}
    try:
        _, p = mannwhitneyu(aa, bb, alternative="two-sided")
    except Exception:
        p = np.nan
    return {
        "n_a": int(len(aa)),
        "n_b": int(len(bb)),
        "median_a": float(aa.median()),
        "median_b": float(bb.median()),
        "median_diff": float(aa.median() - bb.median()),
        "p_value": float(p) if np.isfinite(p) else np.nan,
    }


def conformal_quantile(values: pd.Series, alpha: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if vals.size == 0:
        return np.nan
    # Finite-sample conformal rank, clipped for tiny calibration cohorts.
    q = np.ceil((vals.size + 1) * (1.0 - alpha)) / vals.size
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(vals, q, method="higher"))


def add_transition_context(frame_metrics: pd.DataFrame, phase: pd.DataFrame, mobility: pd.DataFrame) -> pd.DataFrame:
    df = frame_metrics.copy()
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    df["eval_step"] = pd.to_numeric(df["eval_step"], errors="coerce")
    max_step = df.groupby(["roi_id", "method"])["eval_step"].transform("max").replace(0, np.nan)
    df["eval_step_fraction"] = df["eval_step"] / max_step

    phase_cols = [
        "roi_id",
        "cycleNo",
        "cohort_role",
        "event_reference_cycle",
        "is_event_roi",
        "mode_label",
        "is_event_enriched_mode",
        "q60_time_of_max_abs_rate_frac",
        "q70_time_of_max_abs_rate_frac",
        "q80_time_of_max_abs_rate_frac",
        "q60_max_abs_rate_per_s",
        "q70_max_abs_rate_per_s",
        "q80_max_abs_rate_per_s",
        "q70_transformed_fraction_delta",
        "roi_norm_total_variation",
        "duration_s",
    ]
    keep_phase = [c for c in phase_cols if c in phase.columns]
    phase_small = phase[keep_phase].drop_duplicates(["roi_id", "cycleNo"])
    df = df.merge(phase_small, how="left", on=["roi_id", "cycleNo"], suffixes=("", "_phase"))

    mobility_cols = [
        "roi_id",
        "cycleNo",
        "front_quality_score",
        "front_radius_slope_r2",
        "apparent_diffusion_proxy_ds_px2_per_frame",
        "mean_drop_frac",
        "evidence_score",
        "high_fraction_slope_per_s",
        "first_last_corr",
        "cumulative_abs_first_last",
    ]
    keep_mob = [c for c in mobility_cols if c in mobility.columns]
    mob_small = mobility[keep_mob].drop_duplicates(["roi_id", "cycleNo"])
    df = df.merge(mob_small, how="left", on=["roi_id", "cycleNo"], suffixes=("", "_mobility"))

    transition_cols = [c for c in ["q60_time_of_max_abs_rate_frac", "q70_time_of_max_abs_rate_frac", "q80_time_of_max_abs_rate_frac"] if c in df.columns]
    if transition_cols:
        df["transition_center_frac"] = df[transition_cols].apply(pd.to_numeric, errors="coerce").median(axis=1)
    else:
        df["transition_center_frac"] = np.nan
    df["transition_distance_frac"] = (df["eval_step_fraction"] - df["transition_center_frac"]).abs()
    df["near_transition"] = df["transition_distance_frac"] <= 0.15
    df["early_rollout"] = df["eval_step_fraction"] <= 0.33
    df["late_rollout"] = df["eval_step_fraction"] >= 0.67
    df["error_scalar"] = pd.to_numeric(df["mae"], errors="coerce")
    df["mse_error_scalar"] = pd.to_numeric(df["mse"], errors="coerce")
    df["event_reference_cycle"] = pd.to_numeric(df["event_reference_cycle"], errors="coerce")
    return df


def loo_conformal(df: pd.DataFrame, alphas: List[float]) -> pd.DataFrame:
    rows = []
    refs = sorted(pd.to_numeric(df["event_reference_cycle"], errors="coerce").dropna().unique())
    methods = sorted(df["method"].dropna().unique())
    for ref in refs:
        for method in methods:
            train = df[(df["event_reference_cycle"] != ref) & (df["method"] == method)]
            test = df[(df["event_reference_cycle"] == ref) & (df["method"] == method)]
            if test.empty:
                continue
            for alpha in alphas:
                q_global = conformal_quantile(train["error_scalar"], alpha)
                q_near = conformal_quantile(train.loc[train["near_transition"], "error_scalar"], alpha)
                q_far = conformal_quantile(train.loc[~train["near_transition"], "error_scalar"], alpha)
                for group_name, group in [
                    ("all", test),
                    ("near_transition", test[test["near_transition"]]),
                    ("far_from_transition", test[~test["near_transition"]]),
                    ("event_roi", test[test["is_event_roi"] == 1]),
                    ("control_roi", test[test["is_event_roi"] == 0]),
                ]:
                    if group.empty:
                        continue
                    local_q = q_near if group_name == "near_transition" and np.isfinite(q_near) else q_far if group_name == "far_from_transition" and np.isfinite(q_far) else q_global
                    err = pd.to_numeric(group["error_scalar"], errors="coerce").dropna()
                    rows.append({
                        "heldout_event_reference_cycle": float(ref),
                        "method": method,
                        "alpha": float(alpha),
                        "target_coverage": float(1.0 - alpha),
                        "group": group_name,
                        "n": int(len(err)),
                        "global_quantile": q_global,
                        "local_quantile": local_q,
                        "coverage_global": float((err <= q_global).mean()) if len(err) and np.isfinite(q_global) else np.nan,
                        "coverage_local": float((err <= local_q).mean()) if len(err) and np.isfinite(local_q) else np.nan,
                        "mean_excess_over_global": float(np.maximum(err - q_global, 0).mean()) if len(err) and np.isfinite(q_global) else np.nan,
                        "mean_excess_over_local": float(np.maximum(err - local_q, 0).mean()) if len(err) and np.isfinite(local_q) else np.nan,
                        "median_error": float(err.median()) if len(err) else np.nan,
                    })
    return pd.DataFrame(rows)


def summarize_coverage(cal: pd.DataFrame) -> pd.DataFrame:
    if cal.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in cal.groupby(["method", "alpha", "group"], dropna=False):
        method, alpha, group = keys
        weights = pd.to_numeric(grp["n"], errors="coerce").fillna(0)
        total = float(weights.sum())
        def wavg(col: str) -> float:
            vals = pd.to_numeric(grp[col], errors="coerce")
            ok = vals.notna() & (weights > 0)
            if not ok.any():
                return np.nan
            return float(np.average(vals[ok], weights=weights[ok]))
        rows.append({
            "method": method,
            "alpha": float(alpha),
            "target_coverage": float(1.0 - alpha),
            "group": group,
            "n": int(total),
            "coverage_global_weighted": wavg("coverage_global"),
            "coverage_local_weighted": wavg("coverage_local"),
            "coverage_global_gap": wavg("coverage_global") - (1.0 - alpha) if np.isfinite(wavg("coverage_global")) else np.nan,
            "coverage_local_gap": wavg("coverage_local") - (1.0 - alpha) if np.isfinite(wavg("coverage_local")) else np.nan,
            "mean_excess_over_global": wavg("mean_excess_over_global"),
            "mean_excess_over_local": wavg("mean_excess_over_local"),
            "median_error_weighted": wavg("median_error"),
        })
    return pd.DataFrame(rows).sort_values(["alpha", "method", "group"])


def transition_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, grp in df.groupby("method"):
        near = grp[grp["near_transition"]]["error_scalar"]
        far = grp[~grp["near_transition"]]["error_scalar"]
        out = safe_mwu(near, far)
        out.update({"method": method, "contrast": "near_transition_minus_far", "feature": "mae"})
        rows.append(out)

        event = grp[grp["is_event_roi"] == 1]["error_scalar"]
        control = grp[grp["is_event_roi"] == 0]["error_scalar"]
        out = safe_mwu(event, control)
        out.update({"method": method, "contrast": "event_minus_control", "feature": "mae"})
        rows.append(out)

        late = grp[grp["late_rollout"]]["error_scalar"]
        early = grp[grp["early_rollout"]]["error_scalar"]
        out = safe_mwu(late, early)
        out.update({"method": method, "contrast": "late_minus_early_rollout", "feature": "mae"})
        rows.append(out)
    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


def roi_table(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    q90 = cal[(cal["alpha"] == 0.1) & (cal["group"] == "all")]
    q90 = q90[["heldout_event_reference_cycle", "method", "global_quantile"]].rename(columns={"heldout_event_reference_cycle": "event_reference_cycle", "global_quantile": "loo_q90_mae"})
    tmp = df.merge(q90, how="left", on=["event_reference_cycle", "method"])
    tmp["undercovered_q90"] = tmp["error_scalar"] > tmp["loo_q90_mae"]
    agg = tmp.groupby(["roi_id", "cycleNo", "method"], dropna=False).agg({
        "error_scalar": ["mean", "median", "max"],
        "undercovered_q90": "mean",
        "near_transition": "mean",
        "transition_distance_frac": "median",
        "cohort_role": "first",
        "is_event_roi": "first",
        "event_reference_cycle": "first",
        "mode_label": "first",
        "is_event_enriched_mode": "first",
        "q70_max_abs_rate_per_s": "first",
        "q70_transformed_fraction_delta": "first",
        "roi_norm_total_variation": "first",
        "front_quality_score": "first",
        "high_fraction_slope_per_s": "first",
        "first_last_corr": "first",
        "cumulative_abs_first_last": "first",
    }).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    agg = agg.rename(columns={
        "error_scalar_mean": "mae_mean",
        "error_scalar_median": "mae_median",
        "error_scalar_max": "mae_max",
        "undercovered_q90_mean": "q90_undercoverage_rate",
        "near_transition_mean": "near_transition_fraction",
        "transition_distance_frac_median": "transition_distance_frac_median",
    })
    score_parts = []
    for col in ["q90_undercoverage_rate", "mae_max", "near_transition_fraction", "roi_norm_total_variation_first"]:
        if col in agg.columns:
            score_parts.append(pd.to_numeric(agg[col], errors="coerce").rank(pct=True))
    agg["calibration_review_priority"] = np.sum(score_parts, axis=0) if score_parts else np.nan
    return agg.sort_values("calibration_review_priority", ascending=False, na_position="last")


def correlation_table(roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x_cols = [c for c in [
        "q90_undercoverage_rate",
        "mae_mean",
        "mae_max",
        "near_transition_fraction",
    ] if c in roi.columns]
    y_cols = [c for c in [
        "q70_max_abs_rate_per_s_first",
        "q70_transformed_fraction_delta_first",
        "roi_norm_total_variation_first",
        "front_quality_score_first",
        "high_fraction_slope_per_s_first",
        "first_last_corr_first",
        "cumulative_abs_first_last_first",
    ] if c in roi.columns]
    for method, grp in roi.groupby("method"):
        for x in x_cols:
            for y in y_cols:
                pair = grp[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(pair) < 5:
                    rho, p = np.nan, np.nan
                else:
                    rho, p = spearmanr(pair[x], pair[y])
                rows.append({
                    "method": method,
                    "x_calibration_feature": x,
                    "y_physics_feature": y,
                    "n": int(len(pair)),
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                })
    return pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False, na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/probabilistic_rollout_calibration")
    args = parser.parse_args()

    derived = args.derived_dir
    frame_metrics = pd.read_csv(os.path.join(derived, "multi_cycle_roi_rollout_baselines", "roi_rollout_frame_metrics.csv"))
    phase = pd.read_csv(os.path.join(derived, "phase_kinetics_avrami", "phase_kinetics_avrami_roi_table.csv"))
    mobility = pd.read_csv(os.path.join(derived, "multi_cycle_roi_mobility", "multi_cycle_roi_mobility_descriptors.csv"))

    frame = add_transition_context(frame_metrics, phase, mobility)
    cal = loo_conformal(frame, [0.1, 0.05])
    coverage = summarize_coverage(cal)
    tests = transition_tests(frame)
    roi = roi_table(frame, cal)
    corr = correlation_table(roi)

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {
        "frame_table": os.path.join(args.out_dir, "probabilistic_rollout_frame_table.csv"),
        "loo_calibration": os.path.join(args.out_dir, "probabilistic_rollout_loo_calibration.csv"),
        "coverage_summary": os.path.join(args.out_dir, "probabilistic_rollout_coverage_summary.csv"),
        "transition_tests": os.path.join(args.out_dir, "probabilistic_rollout_transition_tests.csv"),
        "roi_table": os.path.join(args.out_dir, "probabilistic_rollout_roi_table.csv"),
        "physics_correlations": os.path.join(args.out_dir, "probabilistic_rollout_physics_correlations.csv"),
    }
    frame.to_csv(paths["frame_table"], index=False)
    cal.to_csv(paths["loo_calibration"], index=False)
    coverage.to_csv(paths["coverage_summary"], index=False)
    tests.to_csv(paths["transition_tests"], index=False)
    roi.to_csv(paths["roi_table"], index=False)
    corr.to_csv(paths["physics_correlations"], index=False)

    top_undercoverage = roi.head(12).to_dict(orient="records")
    summary = {
        "n_frame_rows": int(len(frame)),
        "n_roi_method_rows": int(len(roi)),
        "n_roi": int(frame["roi_id"].nunique()),
        "n_event_reference_cycles": int(pd.to_numeric(frame["event_reference_cycle"], errors="coerce").nunique()),
        "methods": sorted(frame["method"].dropna().unique().tolist()),
        "near_transition_frame_fraction": float(frame["near_transition"].mean()),
        "coverage_summary": coverage.to_dict(orient="records"),
        "top_transition_error_tests": tests.head(12).to_dict(orient="records"),
        "top_calibration_physics_correlations": corr.head(12).to_dict(orient="records"),
        "top_undercovered_roi_method_rows": top_undercoverage,
        "guardrail": "Empirical residual quantiles are a calibration audit for ROI-only rollout baselines. They are not a generative uncertainty model, not calibrated diffusion, and not manual QC.",
        "outputs": paths,
    }
    summary_path = os.path.join(args.out_dir, "probabilistic_rollout_calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Probabilistic Rollout Calibration\n\n")
        f.write("Audits empirical conformal-style residual bands for particle-ROI rollout baselines.\n\n")
        f.write("The analysis leaves one event-reference cycle out, estimates MAE quantiles from the remaining ROI-frame residuals, and checks coverage in all, near-transition, far-from-transition, event, and control strata.\n\n")
        f.write("Use this as uncertainty and review-prioritization evidence, not as a final probabilistic video model or calibrated transport estimate.\n")
    print(json.dumps(summary, indent=2, sort_keys=True)[:8000])


if __name__ == "__main__":
    main()
