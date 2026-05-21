#!/usr/bin/env python3
"""Align masked particle-rollout residual timing with phase-kinetic transitions.

This audit asks whether prediction errors scored only inside the accepted
particle support peak near automatic optical phase-transition times. It uses
the existing masked rollout frame metrics and phase-kinetics descriptors; no
new video model is trained here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
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


def finite_float(value: Any, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_mwu(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    aa = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    bb = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(aa) < 2 or len(bb) < 2:
        return {"n_a": int(len(aa)), "n_b": int(len(bb)), "median_a": np.nan, "median_b": np.nan, "median_diff_a_minus_b": np.nan, "p_value": np.nan}
    try:
        _, p = mannwhitneyu(aa, bb, alternative="two-sided")
    except Exception:
        p = np.nan
    return {
        "n_a": int(len(aa)),
        "n_b": int(len(bb)),
        "median_a": float(aa.median()),
        "median_b": float(bb.median()),
        "median_diff_a_minus_b": float(aa.median() - bb.median()),
        "p_value": float(p) if np.isfinite(p) else np.nan,
    }


def permutation_p_median(values: np.ndarray, observed: float, n_perm: int, seed: int) -> float:
    if not np.isfinite(observed) or len(values) < 4:
        return np.nan
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return np.nan
    null = []
    # Distances to random transition centers under uniform transition timing.
    for _ in range(n_perm):
        random_centers = rng.uniform(0, 1, vals.size)
        null.append(float(np.nanmedian(np.abs(vals - random_centers))))
    arr = np.asarray(null, dtype=float)
    # Small distance is evidence for alignment.
    return float((np.sum(arr <= observed) + 1) / (arr.size + 1))


def add_eval_fraction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["eval_step"] = pd.to_numeric(out["eval_step"], errors="coerce")
    max_step = out.groupby(["roi_id", "method"])["eval_step"].transform("max")
    out["eval_step_fraction"] = out["eval_step"] / max_step.replace(0, np.nan)
    return out


def transition_table(kinetics: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "roi_id",
        "cycleNo",
        "cohort_role",
        "event_reference_cycle",
        "mode_label",
        "is_event_enriched_mode",
        "duration_s",
        "roi_norm_total_variation",
        "q60_time_of_max_abs_rate_frac",
        "q70_time_of_max_abs_rate_frac",
        "q80_time_of_max_abs_rate_frac",
        "q60_max_abs_rate_per_s",
        "q70_max_abs_rate_per_s",
        "q80_max_abs_rate_per_s",
        "q70_transformed_fraction_delta",
        "q80_transformed_fraction_delta",
    ]
    out = kinetics[[c for c in keep if c in kinetics.columns]].copy()
    time_cols = [c for c in ["q60_time_of_max_abs_rate_frac", "q70_time_of_max_abs_rate_frac", "q80_time_of_max_abs_rate_frac"] if c in out.columns]
    out["transition_center_frac"] = out[time_cols].apply(pd.to_numeric, errors="coerce").median(axis=1) if time_cols else np.nan
    out["transition_iqr_frac"] = (
        out[time_cols].apply(pd.to_numeric, errors="coerce").quantile(0.75, axis=1)
        - out[time_cols].apply(pd.to_numeric, errors="coerce").quantile(0.25, axis=1)
    ) if time_cols else np.nan
    return out.drop_duplicates(["roi_id", "cycleNo"])


def per_roi_timing(frame: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    for (roi_id, method), grp in frame.groupby(["roi_id", "method"], dropna=False):
        g = grp.sort_values("eval_step_fraction").copy()
        err = pd.to_numeric(g["particle_mse"], errors="coerce")
        frac = pd.to_numeric(g["eval_step_fraction"], errors="coerce")
        trans = finite_float(g["transition_center_frac"].iloc[0])
        near = (frac - trans).abs() <= 0.15 if np.isfinite(trans) else pd.Series(False, index=g.index)
        far = ~near
        if err.notna().sum() == 0:
            continue
        peak_idx = err.idxmax()
        weights = err.clip(lower=0).fillna(0)
        weighted_center = float(np.average(frac.fillna(0), weights=weights)) if weights.sum() > 0 else np.nan
        total = float(weights.sum()) if np.isfinite(weights.sum()) else np.nan
        near_sum = float(weights[near].sum()) if near.any() else 0.0
        near_median = float(err[near].median()) if near.any() else np.nan
        far_median = float(err[far].median()) if far.any() else np.nan
        try:
            _, paired_p = wilcoxon(
                err[near].reset_index(drop=True).iloc[: min(near.sum(), far.sum())],
                err[far].reset_index(drop=True).iloc[: min(near.sum(), far.sum())],
                zero_method="wilcox",
                alternative="two-sided",
            ) if near.sum() >= 3 and far.sum() >= 3 else (np.nan, np.nan)
        except Exception:
            paired_p = np.nan
        rows.append({
            "roi_id": roi_id,
            "cycleNo": finite_float(g["cycleNo"].iloc[0]),
            "cohort_role": g["cohort_role"].iloc[0] if "cohort_role" in g else "",
            "event_reference_cycle": finite_float(g["event_reference_cycle"].iloc[0]) if "event_reference_cycle" in g else np.nan,
            "method": method,
            "n_eval_frames": int(len(g)),
            "transition_center_frac": trans,
            "transition_iqr_frac": finite_float(g["transition_iqr_frac"].iloc[0]) if "transition_iqr_frac" in g else np.nan,
            "residual_peak_step_fraction": finite_float(frac.loc[peak_idx]),
            "residual_peak_particle_mse": finite_float(err.loc[peak_idx]),
            "residual_weighted_center_frac": weighted_center,
            "peak_distance_to_transition_frac": abs(finite_float(frac.loc[peak_idx]) - trans) if np.isfinite(trans) else np.nan,
            "weighted_center_distance_to_transition_frac": abs(weighted_center - trans) if np.isfinite(weighted_center) and np.isfinite(trans) else np.nan,
            "near_transition_residual_fraction": near_sum / total if np.isfinite(total) and total > 0 else np.nan,
            "near_transition_particle_mse_median": near_median,
            "far_transition_particle_mse_median": far_median,
            "near_minus_far_particle_mse_median": near_median - far_median if np.isfinite(near_median) and np.isfinite(far_median) else np.nan,
            "near_far_wilcoxon_p": float(paired_p) if np.isfinite(paired_p) else np.nan,
            "particle_to_nonparticle_mse_ratio_median": finite_float(pd.to_numeric(g["particle_to_nonparticle_mse_ratio"], errors="coerce").median()),
            "mask_fallback_fraction_eval": finite_float(pd.to_numeric(g["mask_fallback_used"], errors="coerce").mean()) if "mask_fallback_used" in g else np.nan,
            "alignment_permutation_p": permutation_p_median(np.asarray([finite_float(frac.loc[peak_idx])]), abs(finite_float(frac.loc[peak_idx]) - trans) if np.isfinite(trans) else np.nan, n_perm, seed + len(rows)),
        })
    return pd.DataFrame(rows)


def event_control_tests(per_roi: pd.DataFrame) -> pd.DataFrame:
    features = [
        "peak_distance_to_transition_frac",
        "weighted_center_distance_to_transition_frac",
        "near_transition_residual_fraction",
        "near_minus_far_particle_mse_median",
        "particle_to_nonparticle_mse_ratio_median",
    ]
    rows = []
    for method, grp in per_roi.groupby("method"):
        for feat in features:
            out = safe_mwu(grp[grp["cohort_role"] == "event"][feat], grp[grp["cohort_role"] == "control"][feat])
            out.update({"method": method, "feature": feat, "contrast": "event_minus_control"})
            rows.append(out)
    return pd.DataFrame(rows).sort_values("p_value", na_position="last")


def alignment_tests(per_roi: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    for method, grp in per_roi.groupby("method"):
        for feature in ["peak_distance_to_transition_frac", "weighted_center_distance_to_transition_frac"]:
            vals = pd.to_numeric(grp[feature], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size < 5:
                continue
            observed = float(np.nanmedian(vals))
            rng = np.random.default_rng(seed + len(rows))
            null = []
            peaks = pd.to_numeric(grp["residual_peak_step_fraction" if feature.startswith("peak") else "residual_weighted_center_frac"], errors="coerce").dropna().to_numpy(dtype=float)
            for _ in range(n_perm):
                null.append(float(np.nanmedian(np.abs(peaks - rng.uniform(0, 1, len(peaks))))))
            null_arr = np.asarray(null, dtype=float)
            rows.append({
                "method": method,
                "distance_feature": feature,
                "n_roi": int(vals.size),
                "median_distance_to_transition": observed,
                "null_median_distance_mean": float(null_arr.mean()),
                "null_median_distance_p05": float(np.percentile(null_arr, 5)),
                "empirical_p_distance_le_observed": float((np.sum(null_arr <= observed) + 1) / (len(null_arr) + 1)),
            })
    return pd.DataFrame(rows).sort_values("empirical_p_distance_le_observed", na_position="last")


def correlation_tests(per_roi: pd.DataFrame) -> pd.DataFrame:
    x_cols = [
        "peak_distance_to_transition_frac",
        "weighted_center_distance_to_transition_frac",
        "near_transition_residual_fraction",
        "near_minus_far_particle_mse_median",
        "particle_to_nonparticle_mse_ratio_median",
    ]
    y_cols = [c for c in [
        "roi_norm_total_variation",
        "q70_max_abs_rate_per_s",
        "q80_max_abs_rate_per_s",
        "q70_transformed_fraction_delta",
        "q80_transformed_fraction_delta",
        "transition_iqr_frac",
    ] if c in per_roi.columns]
    rows = []
    for method, grp in per_roi.groupby("method"):
        for x in x_cols:
            for y in y_cols:
                tmp = grp[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(tmp) < 6 or tmp[x].nunique() < 2 or tmp[y].nunique() < 2:
                    continue
                rho, p = spearmanr(tmp[x], tmp[y])
                rows.append({"method": method, "x": x, "y": y, "n": int(len(tmp)), "spearman_rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("p_value", na_position="last") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-frame", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_roi_rollout_audit/masked_roi_rollout_frame_metrics.csv")
    parser.add_argument("--phase-kinetics", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/phase_kinetics_avrami/phase_kinetics_avrami_roi_table.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/masked_residual_transition_timing")
    parser.add_argument("--n-permutation", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = add_eval_fraction(pd.read_csv(args.masked_frame))
    trans = transition_table(pd.read_csv(args.phase_kinetics))
    frame = frame.merge(trans, how="left", on=["roi_id", "cycleNo"], suffixes=("", "_kinetics"))
    for col in ["cohort_role", "event_reference_cycle"]:
        alt = f"{col}_kinetics"
        if alt in frame.columns:
            frame[col] = frame[col].fillna(frame[alt]) if col in frame.columns else frame[alt]
    per_roi = per_roi_timing(frame, args.n_permutation, args.seed)
    # Add phase-kinetic context to per-ROI rows for correlations.
    context_cols = [c for c in trans.columns if c not in {"cohort_role", "event_reference_cycle"}]
    per_roi = per_roi.merge(trans[context_cols].drop_duplicates(["roi_id", "cycleNo"]), how="left", on=["roi_id", "cycleNo"], suffixes=("", "_ctx"))
    tests = event_control_tests(per_roi)
    align = alignment_tests(per_roi, args.n_permutation, args.seed)
    corr = correlation_tests(per_roi)
    ranked = per_roi.sort_values(["near_transition_residual_fraction", "residual_peak_particle_mse"], ascending=False, na_position="last")

    per_roi.to_csv(out / "masked_residual_transition_timing_per_roi.csv", index=False)
    tests.to_csv(out / "masked_residual_transition_timing_event_control_tests.csv", index=False)
    align.to_csv(out / "masked_residual_transition_timing_alignment_tests.csv", index=False)
    corr.to_csv(out / "masked_residual_transition_timing_correlations.csv", index=False)
    ranked.to_csv(out / "masked_residual_transition_timing_ranked.csv", index=False)

    summary = {
        "n_roi_method_rows": int(len(per_roi)),
        "n_roi": int(per_roi["roi_id"].nunique()),
        "n_permutation": int(args.n_permutation),
        "top_alignment_tests": clean_json(align.head(10).to_dict("records")) if not align.empty else [],
        "top_event_control_tests": clean_json(tests.head(12).to_dict("records")) if not tests.empty else [],
        "top_correlations": clean_json(corr.head(12).to_dict("records")) if not corr.empty else [],
        "top_near_transition_residual_rois": clean_json(ranked.head(12).to_dict("records")),
        "guardrail": "Automatic phase-kinetic transition timing and masked rollout residual timing audit; not manual front annotation or calibrated transport.",
    }
    with (out / "masked_residual_transition_timing_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2)
    with (out / "README.md").open("w") as f:
        f.write("# Masked Residual Transition Timing\n\n")
        f.write("Aligns masked particle-rollout residual timing with automatic phase-kinetic transition times.\n\n")
        f.write(f"- ROI-method rows: {summary['n_roi_method_rows']}\n")
        f.write(f"- ROI count: {summary['n_roi']}\n")
        f.write(f"- Permutations per alignment null: {summary['n_permutation']}\n\n")
        f.write("Outputs:\n\n")
        f.write("- `masked_residual_transition_timing_per_roi.csv`: per ROI-method timing features.\n")
        f.write("- `masked_residual_transition_timing_alignment_tests.csv`: peak/weighted-center transition alignment nulls.\n")
        f.write("- `masked_residual_transition_timing_event_control_tests.csv`: event/control timing contrasts.\n")
        f.write("- `masked_residual_transition_timing_correlations.csv`: timing links to kinetic descriptors.\n")
        f.write("- `masked_residual_transition_timing_ranked.csv`: ROI-method rows ranked by near-transition residual concentration.\n")

    print(json.dumps(clean_json({
        "n_roi": summary["n_roi"],
        "top_alignment_tests": summary["top_alignment_tests"][:5],
        "top_event_control_tests": summary["top_event_control_tests"][:5],
    }), indent=2))


if __name__ == "__main__":
    main()
