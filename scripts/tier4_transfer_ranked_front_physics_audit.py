#!/usr/bin/env python3
"""Audit front/phase proxies for transfer-ranked warning ROI crops.

This joins the transfer-ranked ROI reconstruction table, exported ROI sequence
manifest, threshold-robust front proxies, and masked rollout residuals. The
goal is to check whether warning-ranked cycles have direct video-backed
phase/front descriptors that align with future-drop labels or residual
difficulty, while keeping diffusion-like values explicitly guarded.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def clean_records(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                out[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                out[key] = None if not np.isfinite(value) else float(value)
            else:
                out[key] = value
        cleaned.append(out)
    return cleaned


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    value = df[column] if column in df.columns else pd.Series(np.nan, index=df.index)
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


def add_roi_id(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "roi_id" not in out.columns:
        out["roi_id"] = out.apply(
            lambda r: f"cycle{int(float(r['cycleNo']))}_rank{int(float(r['front_candidate_rank']))}_obj{int(float(r['object_candidate_rank']))}",
            axis=1,
        )
    return out


def pivot_rollout(rollout: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "particle_mse_mean",
        "particle_to_nonparticle_mse_ratio_mean",
        "particle_mse_fraction_of_full_mean",
        "particle_mask_area_fraction_mean",
        "mask_fallback_used_mean",
    ]
    pieces = []
    for method, grp in rollout.groupby("method", dropna=False):
        cols = [c for c in keep if c in grp.columns]
        if not cols:
            continue
        renamed = grp[["roi_id"] + cols].copy()
        renamed = renamed.rename(columns={c: f"{method}_{c}" for c in cols})
        pieces.append(renamed)
    if not pieces:
        return pd.DataFrame({"roi_id": []})
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="roi_id", how="outer")
    return out


def binary_tests(df: pd.DataFrame, features: List[str], targets: List[str], rng: np.random.Generator, n_perm: int) -> pd.DataFrame:
    rows = []
    for target in unique(targets):
        y = numeric_series(df, target)
        for feature in unique(features):
            x = numeric_series(df, feature)
            mask = y.isin([0, 1]) & np.isfinite(x)
            pos = x[mask & (y == 1)].to_numpy(dtype=float)
            neg = x[mask & (y == 0)].to_numpy(dtype=float)
            if len(pos) and len(neg):
                med_diff = float(np.nanmedian(pos) - np.nanmedian(neg))
                auc = float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
                try:
                    _, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
                except Exception:
                    p_value = np.nan
                pooled = np.concatenate([pos, neg])
                n_pos = len(pos)
                perm = []
                for _ in range(n_perm):
                    shuffled = rng.permutation(pooled)
                    perm.append(float(np.nanmedian(shuffled[:n_pos]) - np.nanmedian(shuffled[n_pos:])))
                perm = np.asarray(perm, dtype=float)
                perm_p = float((np.sum(np.abs(perm) >= abs(med_diff)) + 1) / (len(perm) + 1)) if len(perm) else np.nan
            else:
                med_diff = np.nan
                auc = np.nan
                p_value = np.nan
                perm_p = np.nan
            rows.append({
                "target": target,
                "feature": feature,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                "median_positive_minus_negative": med_diff,
                "mannwhitney_p": finite_float(p_value),
                "abs_oriented_auc": abs(auc - 0.5) + 0.5 if np.isfinite(auc) else np.nan,
                "permutation_p_abs_median_diff": perm_p,
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["permutation_p_abs_median_diff", "mannwhitney_p"], na_position="last")


def correlations(df: pd.DataFrame, xs: List[str], ys: List[str]) -> pd.DataFrame:
    rows = []
    for x_name in unique(xs):
        x = numeric_series(df, x_name)
        for y_name in unique(ys):
            if x_name == y_name:
                continue
            y = numeric_series(df, y_name)
            mask = np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) >= 4 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
                rho, p_value = spearmanr(x[mask], y[mask])
            else:
                rho, p_value = np.nan, np.nan
            rows.append({
                "x": x_name,
                "y": y_name,
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
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/transfer_ranked_front_physics_audit")
    parser.add_argument("--n-permutation", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    roi = add_roi_id(read_csv(derived / "transfer_ranked_roi_reconstruction" / "transfer_ranked_roi_table.csv"))
    manifest = read_csv(derived / "transfer_ranked_roi_sequences" / "selected_roi_sequence_manifest.csv")
    front = read_csv(derived / "transfer_ranked_threshold_robust_fronts" / "threshold_robust_front_summary.csv")
    front_group = read_csv(derived / "transfer_ranked_threshold_robust_fronts" / "threshold_robust_front_group_summary.csv")
    rollout = read_csv(derived / "transfer_ranked_masked_roi_rollout_audit" / "masked_roi_rollout_per_roi_metrics.csv")

    roi_keep = [
        "roi_id", "cycleNo", "source_stem", "local_cycle_index", "front_candidate_rank",
        "object_candidate_rank", "validation_score", "transfer_rank",
        "transferred_masked_residual_signature", "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles", "any_abrupt_drop", "object_mean_abs_z",
        "object_mean_residual",
    ]
    manifest_keep = [
        "roi_id", "roi_mean_delta_last_minus_first", "roi_norm_mean_delta_last_minus_first",
        "stage_drift_xy_sampled", "first_frame_index", "last_frame_index",
    ]
    joined = front.merge(roi[[c for c in roi_keep if c in roi.columns]], on="roi_id", how="left", suffixes=("", "_recon"))
    joined = joined.merge(manifest[[c for c in manifest_keep if c in manifest.columns]], on="roi_id", how="left")
    joined = joined.merge(pivot_rollout(rollout), on="roi_id", how="left")

    front_features = [
        "phase_slope_median_per_s",
        "phase_slope_abs_median_per_s",
        "phase_slope_positive_fraction",
        "radius2_slope_median_px2_per_s",
        "radius2_slope_positive_fraction",
        "diffusion_proxy_median_um2_per_s",
        "diffusion_proxy_abs_median_um2_per_s",
        "threshold_robust_phase_score",
        "threshold_robust_diffusion_score",
        "q70_phase_slope_bootstrap_p50",
        "q70_radius2_slope_bootstrap_p50_px2_per_s",
        "roi_norm_mean_delta_last_minus_first",
        "low_rank_dmd_particle_mse_mean",
        "low_rank_dmd_particle_to_nonparticle_mse_ratio_mean",
        "persistence_particle_mse_mean",
    ]
    front_features = unique([c for c in front_features if c in joined.columns])
    targets = [
        "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles",
        "any_abrupt_drop",
    ]
    context = [
        "transferred_masked_residual_signature",
        "transfer_rank",
        "object_mean_abs_z",
        "object_mean_residual",
        "roi_mean_delta_last_minus_first",
        "roi_norm_mean_delta_last_minus_first",
        "low_rank_dmd_particle_mse_mean",
        "low_rank_dmd_particle_to_nonparticle_mse_ratio_mean",
        "persistence_particle_mse_mean",
    ]
    context = unique([c for c in context if c in joined.columns])

    rng = np.random.default_rng(args.random_state)
    target_tests = binary_tests(joined, front_features, targets, rng, args.n_permutation)
    corr = correlations(joined, front_features, context + targets)

    cycle_numeric = unique([
        c for c in front_features + context + targets
        if c in joined.columns
    ])
    cycle = joined.groupby("cycleNo", dropna=False)[cycle_numeric].mean(numeric_only=True).reset_index()
    cycle = cycle.merge(
        front_group[[
            c for c in [
                "cycleNo", "n_roi", "phase_slope_median_per_s", "phase_slope_abs_median_per_s",
                "diffusion_proxy_median_um2_per_s", "diffusion_proxy_abs_median_um2_per_s",
                "threshold_robust_phase_score", "threshold_robust_diffusion_score"
            ] if c in front_group.columns
        ]].rename(columns={c: f"cycle_group_{c}" for c in front_group.columns if c != "cycleNo"}),
        on="cycleNo",
        how="left",
    )
    cycle_target_tests = binary_tests(cycle, [c for c in front_features if c in cycle.columns], targets, rng, args.n_permutation)
    cycle_corr = correlations(cycle, [c for c in front_features if c in cycle.columns], [c for c in context + targets if c in cycle.columns])

    joined["front_physics_review_score"] = (
        pd.to_numeric(joined.get("threshold_robust_phase_score"), errors="coerce").rank(pct=True)
        + pd.to_numeric(joined.get("diffusion_proxy_abs_median_um2_per_s"), errors="coerce").rank(pct=True)
        + pd.to_numeric(joined.get("low_rank_dmd_particle_mse_mean"), errors="coerce").rank(pct=True)
        + pd.to_numeric(joined.get("transferred_masked_residual_signature"), errors="coerce").rank(pct=True)
    )
    review_cols = [
        "roi_id", "cycleNo", "transfer_rank", "transferred_masked_residual_signature",
        "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "any_abrupt_drop",
        "threshold_robust_phase_score", "phase_slope_median_per_s",
        "diffusion_proxy_median_um2_per_s", "threshold_robust_diffusion_score",
        "low_rank_dmd_particle_mse_mean", "roi_norm_mean_delta_last_minus_first",
        "front_physics_review_score",
    ]
    top_review = joined[[c for c in review_cols if c in joined.columns]].sort_values("front_physics_review_score", ascending=False)

    joined_path = out / "transfer_ranked_front_physics_joined.csv"
    target_path = out / "transfer_ranked_front_physics_target_tests.csv"
    corr_path = out / "transfer_ranked_front_physics_correlations.csv"
    cycle_path = out / "transfer_ranked_front_physics_cycle_summary.csv"
    cycle_target_path = out / "transfer_ranked_front_physics_cycle_target_tests.csv"
    cycle_corr_path = out / "transfer_ranked_front_physics_cycle_correlations.csv"
    review_path = out / "transfer_ranked_front_physics_review_priority.csv"
    joined.to_csv(joined_path, index=False)
    target_tests.to_csv(target_path, index=False)
    corr.to_csv(corr_path, index=False)
    cycle.to_csv(cycle_path, index=False)
    cycle_target_tests.to_csv(cycle_target_path, index=False)
    cycle_corr.to_csv(cycle_corr_path, index=False)
    top_review.to_csv(review_path, index=False)

    summary = {
        "n_roi": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique(dropna=True)),
        "n_permutation": int(args.n_permutation),
        "target_positive_counts": {
            target: int(pd.to_numeric(joined.get(target), errors="coerce").fillna(0).astype(int).sum())
            for target in targets if target in joined.columns
        },
        "cycle_target_positive_counts": {
            target: int((pd.to_numeric(cycle.get(target), errors="coerce") > 0).sum())
            for target in targets if target in cycle.columns
        },
        "top_target_tests": clean_records(target_tests.head(12).to_dict("records")),
        "top_correlations": clean_records(corr.head(12).to_dict("records")),
        "top_cycle_target_tests": clean_records(cycle_target_tests.head(12).to_dict("records")),
        "top_cycle_correlations": clean_records(cycle_corr.head(12).to_dict("records")),
        "top_review_rois": clean_records(top_review.head(12).to_dict("records")),
        "top_cycle_summary": clean_records(
            cycle.sort_values("threshold_robust_phase_score", ascending=False, na_position="last").head(12).to_dict("records")
        ),
        "outputs": {
            "joined": str(joined_path),
            "target_tests": str(target_path),
            "correlations": str(corr_path),
            "cycle_summary": str(cycle_path),
            "cycle_target_tests": str(cycle_target_path),
            "cycle_correlations": str(cycle_corr_path),
            "review_priority": str(review_path),
        },
        "guardrail": (
            "Transfer-ranked front descriptors are automatic optical phase/radius proxies from ROI crops. "
            "Diffusion-like values are apparent front-motion descriptors, not calibrated transport coefficients, "
            "and the cohort is warning-ranked rather than an event/control design."
        ),
    }
    with (out / "transfer_ranked_front_physics_audit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (out / "README.md").open("w") as f:
        f.write("# Transfer-Ranked Front Physics Audit\n\n")
        f.write("Joins transfer-ranked ROI crops to threshold-robust phase/front proxies, masked rollout residuals, and future-drop labels.\n\n")
        f.write(f"- ROI rows: {summary['n_roi']}\n")
        f.write(f"- Cycles: {summary['n_cycles']}\n")
        f.write("Guardrail: apparent optical-front proxies only; no calibrated diffusion claim.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
