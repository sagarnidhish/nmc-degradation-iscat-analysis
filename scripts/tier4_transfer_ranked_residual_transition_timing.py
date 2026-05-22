#!/usr/bin/env python3
"""Residual-transition timing audit for transfer-ranked NMC ROI crops.

This extends the masked residual timing idea to the transfer-ranked warning
cohort. It computes optical phase-kinetic transition centers directly from the
transfer-ranked ROI movies, joins held-out masked rollout frame errors, and
tests whether residual timing aligns with phase-transition timing or future
drop labels.

All outputs are automatic optical/rollout proxies; no manual front labels or
calibrated diffusion coefficients are claimed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from tier4_phase_kinetics_avrami import clean_json, sequence_kinetics
from tier4_masked_residual_transition_timing import (
    add_eval_fraction,
    alignment_tests,
    correlation_tests,
    per_roi_timing,
    transition_table,
)


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_mwu(a: Iterable[float], b: Iterable[float]) -> Dict[str, float]:
    aa = pd.to_numeric(pd.Series(a), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    bb = pd.to_numeric(pd.Series(b), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "n_positive": int(len(aa)),
        "n_negative": int(len(bb)),
        "median_positive": float(aa.median()) if not aa.empty else np.nan,
        "median_negative": float(bb.median()) if not bb.empty else np.nan,
        "median_positive_minus_negative": float(aa.median() - bb.median()) if not aa.empty and not bb.empty else np.nan,
        "mannwhitney_p": np.nan,
        "abs_oriented_auc": np.nan,
    }
    if len(aa) and len(bb):
        try:
            _, p = mannwhitneyu(aa, bb, alternative="two-sided")
            out["mannwhitney_p"] = float(p)
        except Exception:
            pass
        pos = aa.to_numpy(dtype=float)
        neg = bb.to_numpy(dtype=float)
        auc = float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
        out["abs_oriented_auc"] = abs(auc - 0.5) + 0.5
    return out


def permutation_p(pos: np.ndarray, neg: np.ndarray, observed: float, n_perm: int, rng: np.random.Generator) -> float:
    if not np.isfinite(observed) or len(pos) == 0 or len(neg) == 0:
        return np.nan
    pooled = np.concatenate([pos, neg])
    n_pos = len(pos)
    null = []
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        null.append(float(np.nanmedian(shuffled[:n_pos]) - np.nanmedian(shuffled[n_pos:])))
    arr = np.asarray(null, dtype=float)
    return float((np.sum(np.abs(arr) >= abs(observed)) + 1) / (arr.size + 1))


def target_tests(per_roi: pd.DataFrame, targets: List[str], features: List[str], n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for method, grp in per_roi.groupby("method"):
        for target in targets:
            y = pd.to_numeric(grp.get(target), errors="coerce")
            for feature in features:
                x = pd.to_numeric(grp.get(feature), errors="coerce").replace([np.inf, -np.inf], np.nan)
                mask = y.isin([0, 1]) & x.notna()
                pos = x[mask & (y == 1)].to_numpy(dtype=float)
                neg = x[mask & (y == 0)].to_numpy(dtype=float)
                out = safe_mwu(pos, neg)
                out.update({
                    "method": method,
                    "target": target,
                    "feature": feature,
                    "permutation_p_abs_median_diff": permutation_p(pos, neg, out["median_positive_minus_negative"], n_perm, rng),
                })
                rows.append(out)
    return pd.DataFrame(rows).sort_values(["permutation_p_abs_median_diff", "mannwhitney_p"], na_position="last")


def timing_target_correlations(per_roi: pd.DataFrame, targets: List[str], features: List[str]) -> pd.DataFrame:
    rows = []
    for method, grp in per_roi.groupby("method"):
        for x_name in features:
            x = pd.to_numeric(grp.get(x_name), errors="coerce")
            for target in targets:
                y = pd.to_numeric(grp.get(target), errors="coerce")
                mask = np.isfinite(x) & np.isfinite(y)
                if int(mask.sum()) < 6 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                    rho, p_value = np.nan, np.nan
                else:
                    rho, p_value = spearmanr(x[mask], y[mask])
                rows.append({
                    "method": method,
                    "x": x_name,
                    "target": target,
                    "n": int(mask.sum()),
                    "spearman_rho": finite_float(rho),
                    "p_value": finite_float(p_value),
                })
    out = pd.DataFrame(rows)
    out["abs_rho"] = pd.to_numeric(out["spearman_rho"], errors="coerce").abs()
    return out.sort_values(["p_value", "abs_rho"], ascending=[True, False], na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_residual_transition_timing")
    parser.add_argument("--n-permutation", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(derived / "transfer_ranked_roi_sequences" / "selected_roi_sequence_manifest.csv")
    manifest["cohort_role"] = manifest.get("cohort_role", "transfer_warning")
    manifest["event_reference_cycle"] = pd.to_numeric(manifest.get("event_reference_cycle", manifest["cycleNo"]), errors="coerce").fillna(pd.to_numeric(manifest["cycleNo"], errors="coerce"))
    labels = pd.read_csv(derived / "transfer_ranked_front_physics_audit" / "transfer_ranked_front_physics_joined.csv")
    label_cols = [
        "roi_id",
        "transfer_rank",
        "transferred_masked_residual_signature",
        "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles",
        "any_abrupt_drop",
        "threshold_robust_phase_score",
        "threshold_robust_diffusion_score",
        "diffusion_proxy_median_um2_per_s",
        "radius2_slope_median_px2_per_s",
        "persistence_particle_mse_mean",
        "low_rank_dmd_particle_mse_mean",
    ]
    labels = labels[[c for c in label_cols if c in labels.columns]].drop_duplicates("roi_id")

    kinetics_rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        if isinstance(row.get("npz_path"), str) and Path(row["npz_path"]).exists():
            kinetics_rows.append(sequence_kinetics(row))
    kinetics = pd.DataFrame(kinetics_rows)
    kinetics = kinetics.merge(labels, how="left", on="roi_id")

    frame = add_eval_fraction(pd.read_csv(derived / "transfer_ranked_masked_roi_rollout_audit" / "masked_roi_rollout_frame_metrics.csv"))
    frame = frame.merge(transition_table(kinetics), how="left", on=["roi_id", "cycleNo"], suffixes=("", "_kinetics"))
    per_roi = per_roi_timing(frame, args.n_permutation, args.seed)
    per_roi = per_roi.merge(labels, how="left", on="roi_id")

    align = alignment_tests(per_roi, args.n_permutation, args.seed)
    corr = correlation_tests(per_roi)
    features = [
        "peak_distance_to_transition_frac",
        "weighted_center_distance_to_transition_frac",
        "near_transition_residual_fraction",
        "near_minus_far_particle_mse_median",
        "particle_to_nonparticle_mse_ratio_median",
        "residual_peak_particle_mse",
    ]
    targets = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "any_abrupt_drop"]
    target = target_tests(per_roi, targets, features, args.n_permutation, args.seed)
    target_corr = timing_target_correlations(per_roi, targets, features)

    kinetics_path = out / "transfer_ranked_phase_kinetics_roi_table.csv"
    per_roi_path = out / "transfer_ranked_residual_transition_timing_per_roi.csv"
    align_path = out / "transfer_ranked_residual_transition_alignment_tests.csv"
    corr_path = out / "transfer_ranked_residual_transition_correlations.csv"
    target_path = out / "transfer_ranked_residual_transition_target_tests.csv"
    target_corr_path = out / "transfer_ranked_residual_transition_target_correlations.csv"
    kinetics.to_csv(kinetics_path, index=False)
    per_roi.to_csv(per_roi_path, index=False)
    align.to_csv(align_path, index=False)
    corr.to_csv(corr_path, index=False)
    target.to_csv(target_path, index=False)
    target_corr.to_csv(target_corr_path, index=False)

    summary = {
        "n_roi": int(kinetics["roi_id"].nunique()) if "roi_id" in kinetics else 0,
        "n_roi_method_rows": int(len(per_roi)),
        "n_permutation": int(args.n_permutation),
        "target_positive_counts": {
            target_name: int(pd.to_numeric(labels.get(target_name), errors="coerce").fillna(0).astype(int).sum())
            for target_name in targets if target_name in labels.columns
        },
        "top_alignment_tests": clean_json(align.head(12).to_dict("records")),
        "top_timing_target_tests": clean_json(target.head(18).to_dict("records")),
        "top_timing_target_correlations": clean_json(target_corr.head(18).to_dict("records")),
        "top_transition_correlations": clean_json(corr.head(18).to_dict("records")) if not corr.empty else [],
        "top_near_transition_residual_rois": clean_json(
            per_roi.sort_values(["near_transition_residual_fraction", "residual_peak_particle_mse"], ascending=False)
            .head(12)
            .to_dict("records")
        ),
        "outputs": {
            "phase_kinetics": str(kinetics_path),
            "per_roi_timing": str(per_roi_path),
            "alignment_tests": str(align_path),
            "transition_correlations": str(corr_path),
            "target_tests": str(target_path),
            "target_correlations": str(target_corr_path),
        },
        "guardrail": (
            "Transfer-ranked transition timing uses automatic phase-fraction kinetics and masked rollout residuals from warning-ranked ROI crops. "
            "It tests temporal alignment and future-label association only; it is not manual phase-boundary annotation or calibrated transport."
        ),
    }
    with (out / "transfer_ranked_residual_transition_timing_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)
    with (out / "README.md").open("w") as f:
        f.write("# Transfer-Ranked Residual Transition Timing\n\n")
        f.write("Aligns masked rollout residual timing with automatic phase-kinetic transition centers for transfer-ranked ROI crops.\n\n")
        f.write(f"- ROI rows: {summary['n_roi']}\n")
        f.write(f"- ROI-method rows: {summary['n_roi_method_rows']}\n")
        f.write(f"- Permutations: {summary['n_permutation']}\n")
        f.write("Guardrail: automatic optical timing proxies only; no manual fronts or calibrated diffusion.\n")
    print(json.dumps(clean_json({
        "n_roi": summary["n_roi"],
        "n_roi_method_rows": summary["n_roi_method_rows"],
        "top_alignment_tests": summary["top_alignment_tests"][:3],
        "top_timing_target_tests": summary["top_timing_target_tests"][:3],
    }), indent=2))


if __name__ == "__main__":
    main()
