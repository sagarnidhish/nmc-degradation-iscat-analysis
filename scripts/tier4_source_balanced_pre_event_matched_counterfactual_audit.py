#!/usr/bin/env python3
"""Matched counterfactual audit for source-balanced pre-event ROI physics.

The source-balanced pre-event audits nominate front/mask/apparent-diffusion
features as useful review signals, but several effects remain source and context
structured. This audit builds explicit nearest-neighbor counterfactual pairs for
near-pre-event ROI rows against far-pre-event and post/control rows using only
baseline/context features, then tests paired differences in physics descriptors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRE_NEAR = "near_pre_event_1_8"
PRE_FAR = "far_pre_event_17_32"
POST_CONTROL = {"post_event_1_16", "no_near_event_control"}
COMPARISONS = [
    ("near_vs_far_pre", {PRE_NEAR}, {PRE_FAR}),
    ("near_vs_post_control", {PRE_NEAR}, POST_CONTROL),
]
MATCH_SCHEMES = ["same_source", "source_penalized_global"]

CONTEXT_CANDIDATES = [
    "cycleNo",
    "local_cycle_index",
    "object_x_full_approx",
    "object_y_full_approx",
    "object_area_ds_px",
    "object_mean_abs_z",
    "crop_x0",
    "crop_y0",
    "roi_mean_first",
    "roi_norm_mean_first",
    "mask_base_area_fraction",
]
OUTCOME_CANDIDATES = [
    "front_radius_q60_slope_px_per_norm_time",
    "front_radius_q70_slope_px_per_norm_time",
    "front_radius_q80_slope_px_per_norm_time",
    "front_radius_q60_median_px",
    "front_radius_q70_median_px",
    "front_radius_q80_median_px",
    "masked_minus_background_mean_slope",
    "masked_minus_background_mean_median",
    "mask_area_fraction_slope",
    "mask_centroid_path_px",
    "front_gradient_peak_radius_slope_px_per_norm_time",
    "apparent_diffusion_q70_px2_per_norm_time",
    "apparent_diffusion_q70_um2_per_norm_time",
    "roi_norm_mean_delta_last_minus_first",
    "temporal_energy_mean",
    "frame_diff_mse_mean",
    "spatial_std_slope",
]


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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def merge_inputs(readout: pd.DataFrame, sequence: pd.DataFrame, review: pd.DataFrame | None) -> pd.DataFrame:
    seq_cols = [c for c in sequence.columns if c not in readout.columns or c == "roi_id"]
    out = readout.merge(sequence[seq_cols], on="roi_id", how="left")
    if review is not None and not review.empty:
        keep = [
            c for c in [
                "roi_id",
                "pre_event_review_rank",
                "pre_event_review_score",
                "review_reason",
                "si_clean_physics_prob",
                "si_near_far_physics_prob",
                "si_clean_intensity_guardrail_prob",
            ] if c in review.columns
        ]
        out = out.merge(review[keep], on="roi_id", how="left")
    return out


def available_numeric(df: pd.DataFrame, cols: Sequence[str], min_count: int = 12) -> List[str]:
    out = []
    for col in cols:
        if col in df.columns:
            vals = numeric(df, col)
            if vals.notna().sum() >= min_count and vals.nunique(dropna=True) > 1:
                out.append(col)
    return out


def robust_scale(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    scaled = pd.DataFrame(index=frame.index)
    for col in cols:
        x = numeric(frame, col)
        med = x.median()
        mad = (x - med).abs().median()
        scale = 1.4826 * mad if pd.notna(mad) and mad > 1e-12 else x.std()
        if pd.isna(scale) or scale <= 1e-12:
            scaled[col] = 0.0
        else:
            scaled[col] = ((x - med) / scale).clip(-8, 8).fillna(0.0)
    return scaled


def nearest_matches(
    df: pd.DataFrame,
    treated_idx: Sequence[int],
    control_idx: Sequence[int],
    context_cols: Sequence[str],
    scheme: str,
) -> pd.DataFrame:
    if not treated_idx or not control_idx:
        return pd.DataFrame()
    scaled = robust_scale(df.loc[list(treated_idx) + list(control_idx)], context_cols)
    src = df["source_stem"].astype(str)
    rows = []
    for tidx in treated_idx:
        pool = list(control_idx)
        if scheme == "same_source":
            pool = [c for c in pool if src.loc[c] == src.loc[tidx]]
        if not pool:
            continue
        tv = scaled.loc[tidx, context_cols].to_numpy(dtype=float)
        cv = scaled.loc[pool, context_cols].to_numpy(dtype=float)
        d = np.sqrt(np.nanmean((cv - tv) ** 2, axis=1))
        if scheme == "source_penalized_global":
            d = d + np.array([0.75 if src.loc[c] != src.loc[tidx] else 0.0 for c in pool])
        best_pos = int(np.nanargmin(d))
        cidx = pool[best_pos]
        rows.append({
            "treated_index": int(tidx),
            "control_index": int(cidx),
            "treated_roi_id": df.loc[tidx, "roi_id"],
            "control_roi_id": df.loc[cidx, "roi_id"],
            "treated_source": src.loc[tidx],
            "control_source": src.loc[cidx],
            "same_source": bool(src.loc[tidx] == src.loc[cidx]),
            "treated_cycleNo": clean_json(df.loc[tidx, "cycleNo"]),
            "control_cycleNo": clean_json(df.loc[cidx, "cycleNo"]),
            "treated_event_bin": df.loc[tidx, "event_relative_bin"],
            "control_event_bin": df.loc[cidx, "event_relative_bin"],
            "match_distance": float(d[best_pos]),
        })
    return pd.DataFrame(rows)


def signflip_p(diff: pd.Series, rng: np.random.Generator, n_perm: int) -> float:
    x = pd.to_numeric(diff, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 6 or np.allclose(x, 0):
        return np.nan
    obs = abs(float(np.mean(x)))
    null = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(x))
        null.append(abs(float(np.mean(x * signs))))
    return float((sum(v >= obs for v in null) + 1) / (len(null) + 1))


def paired_tests(df: pd.DataFrame, pairs: pd.DataFrame, outcomes: Sequence[str], rng: np.random.Generator, n_perm: int) -> pd.DataFrame:
    rows = []
    for feature in outcomes:
        t = numeric(df, feature).loc[pairs["treated_index"].to_numpy()].reset_index(drop=True)
        c = numeric(df, feature).loc[pairs["control_index"].to_numpy()].reset_index(drop=True)
        diff = t - c
        valid = diff.notna()
        x = diff[valid]
        if len(x) >= 6 and x.nunique(dropna=True) > 1:
            try:
                _, p_w = wilcoxon(x, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                p_w = np.nan
            p_sf = signflip_p(x, rng, n_perm)
            pos_frac = float((x > 0).mean())
            mean_diff = float(x.mean())
            median_diff = float(x.median())
            med_t = float(t[valid].median())
            med_c = float(c[valid].median())
        else:
            p_w = p_sf = pos_frac = mean_diff = median_diff = med_t = med_c = np.nan
        rows.append({
            "feature": feature,
            "n_pairs": int(valid.sum()),
            "treated_median": med_t,
            "control_median": med_c,
            "mean_treated_minus_control": mean_diff,
            "median_treated_minus_control": median_diff,
            "positive_difference_fraction": pos_frac,
            "wilcoxon_p": float(p_w) if np.isfinite(p_w) else np.nan,
            "signflip_mean_abs_p": p_sf,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_median_difference"] = out["median_treated_minus_control"].abs()
        out = out.sort_values(["signflip_mean_abs_p", "abs_median_difference"], ascending=[True, False])
    return out


def source_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    out = pairs.groupby(["treated_source", "control_source", "same_source"], dropna=False).agg(
        n_pairs=("treated_roi_id", "count"),
        median_match_distance=("match_distance", "median"),
    ).reset_index().sort_values(["n_pairs", "median_match_distance"], ascending=[False, True])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_matched_counterfactual_audit")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    readout = read_csv(derived / "source_balanced_pre_event_readout_audit" / "source_balanced_pre_event_readout_features.csv")
    sequence = read_csv(derived / "source_balanced_pre_event_sequence_audit" / "source_balanced_pre_event_sequence_features.csv")
    review_path = derived / "source_balanced_pre_event_review_packet" / "source_balanced_pre_event_review_ranked_candidates.csv"
    review = read_csv(review_path) if review_path.exists() else None
    df = merge_inputs(readout, sequence, review).reset_index(drop=True)

    context_cols = available_numeric(df, CONTEXT_CANDIDATES, min_count=24)
    outcomes = available_numeric(df, OUTCOME_CANDIDATES, min_count=12)
    bins = df["event_relative_bin"].astype(str)

    all_pairs = []
    all_tests = []
    all_source = []
    for comparison, treat_bins, control_bins in COMPARISONS:
        treated_idx = df.index[bins.isin(treat_bins)].tolist()
        control_idx = df.index[bins.isin(control_bins)].tolist()
        for scheme in MATCH_SCHEMES:
            pairs = nearest_matches(df, treated_idx, control_idx, context_cols, scheme)
            if pairs.empty:
                continue
            pairs.insert(0, "comparison", comparison)
            pairs.insert(1, "match_scheme", scheme)
            tests = paired_tests(df, pairs, outcomes, rng, args.n_perm)
            tests.insert(0, "comparison", comparison)
            tests.insert(1, "match_scheme", scheme)
            src = source_summary(pairs)
            if not src.empty:
                src.insert(0, "comparison", comparison)
                src.insert(1, "match_scheme", scheme)
            all_pairs.append(pairs)
            all_tests.append(tests)
            all_source.append(src)

    pair_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    test_df = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()
    source_df = pd.concat(all_source, ignore_index=True) if all_source else pd.DataFrame()

    pair_path = out / "source_balanced_pre_event_matched_pairs.csv"
    test_path = out / "source_balanced_pre_event_matched_feature_tests.csv"
    source_path = out / "source_balanced_pre_event_matched_source_summary.csv"
    summary_path = out / "source_balanced_pre_event_matched_counterfactual_summary.json"
    pair_df.to_csv(pair_path, index=False)
    test_df.to_csv(test_path, index=False)
    source_df.to_csv(source_path, index=False)

    best_rows = []
    physics_terms = ("front_", "apparent_diffusion", "masked_minus_background", "mask_")
    if not test_df.empty:
        for (comparison, scheme), sub in test_df.groupby(["comparison", "match_scheme"]):
            phys = sub[sub["feature"].astype(str).map(lambda x: any(term in x for term in physics_terms))]
            top = phys.sort_values(["signflip_mean_abs_p", "abs_median_difference"], ascending=[True, False]).head(8)
            best_rows.extend(top.to_dict("records"))

    pair_counts = pair_df.groupby(["comparison", "match_scheme"]).size().to_dict() if not pair_df.empty else {}
    same_source_frac = pair_df.groupby(["comparison", "match_scheme"])["same_source"].mean().to_dict() if not pair_df.empty else {}
    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "context_features": context_cols,
        "outcome_features": outcomes,
        "n_permutations": int(args.n_perm),
        "event_relative_bin_counts": clean_json(bins.value_counts().to_dict()),
        "pair_counts": clean_json({"|".join(map(str, k)): int(v) for k, v in pair_counts.items()}),
        "same_source_pair_fraction": clean_json({"|".join(map(str, k)): float(v) for k, v in same_source_frac.items()}),
        "top_physics_matched_tests": clean_json(best_rows[:16]),
        "outputs": {
            "pairs": str(pair_path),
            "feature_tests": str(test_path),
            "source_summary": str(source_path),
            "summary": str(summary_path),
        },
        "guardrail": "Nearest-neighbor matching uses automatic context/baseline descriptors only and controls observed confounding, not unobserved acquisition or particle-identity effects. Pairs reuse controls and all ROI masks/fronts are automatic. Results prioritize manual QC and physics follow-up; they do not validate phase boundaries, calibrated diffusion coefficients, degradation causality, or deployable warnings.",
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")

    lines = [
        "# Source-Balanced Pre-Event Matched Counterfactual Audit",
        "",
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Context match features: {', '.join(context_cols)}",
        f"- Pair counts: {summary['pair_counts']}",
        "",
        "## Top Matched Physics Tests",
    ]
    for row in best_rows[:10]:
        lines.append(
            f"- {row['comparison']} {row['match_scheme']} {row['feature']}: n={row['n_pairs']}, median diff={row['median_treated_minus_control']:.4g}, sign-flip p={row['signflip_mean_abs_p']:.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
