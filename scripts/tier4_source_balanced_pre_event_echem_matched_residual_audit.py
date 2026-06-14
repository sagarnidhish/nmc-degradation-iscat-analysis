#!/usr/bin/env python3
"""Echem-conditioned matched residual audit for pre-event front signals.

The echem/front coupling audit showed that raw front, mask, and kymograph
features are strongly structured by source, acquisition context, and cycle-level
electrochemistry. This follow-up asks whether near-pre rows still differ from
matched controls after pairing on those same context/echem descriptors and
testing source+echem residual optical/front targets.
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
PRE_MID = "mid_pre_event_9_16"
PRE_FAR = "far_pre_event_17_32"
POST_CONTROL = {"post_event_1_16", "no_near_event_control"}

COMPARISONS = [
    ("near_vs_far_pre", {PRE_NEAR}, {PRE_FAR}),
    ("near_vs_mid_pre", {PRE_NEAR}, {PRE_MID}),
    ("near_vs_post_control", {PRE_NEAR}, POST_CONTROL),
    ("near_vs_any_non_near", {PRE_NEAR}, {PRE_MID, PRE_FAR, *POST_CONTROL}),
]

MATCH_SCHEMES = [
    "same_source",
    "source_penalized_echem_context",
    "source_class_penalized_echem_context",
]

CONTEXT_CANDIDATES = [
    "cycleNo",
    "local_cycle_index",
    "expansion_cycle_rank",
    "object_candidate_rank",
    "object_x_full_approx",
    "object_y_full_approx",
    "object_area_ds_px",
    "object_mean_abs_z",
    "crop_x0",
    "crop_y0",
    "n_frames",
    "first_frame_index",
    "last_frame_index",
    "roi_norm_mean_first",
    "stage_drift_xy_sampled",
    "mask_base_area_fraction",
]

ECHEM_CANDIDATES = [
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
    "echem_regime_pc1",
    "echem_regime_pc2",
    "echem_regime_pc3",
    "echem_regime_pc4",
    "shape_V_range",
    "shape_V_mean",
    "shape_V_std",
    "shape_I_abs_mean_mA",
    "shape_I_pos_fraction",
    "shape_I_neg_fraction",
    "shape_charge_mAh_abs",
    "shape_dVdt_abs_p95",
    "shape_dVdt_sign_consistency",
    "all_dq_abs_lowV_frac",
    "all_dq_abs_midV_frac",
    "all_dq_abs_highV_frac",
    "all_dq_abs_peak_voltage",
    "all_dq_abs_peak_frac",
    "all_dq_abs_entropy",
    "pos_dq_abs_peak_voltage",
    "pos_dq_abs_peak_frac",
    "neg_dq_abs_peak_voltage",
    "neg_dq_abs_peak_frac",
]

TARGET_BASES = [
    "front_radius_q60_slope_px_per_norm_time",
    "front_radius_q70_slope_px_per_norm_time",
    "front_radius_q80_slope_px_per_norm_time",
    "masked_minus_background_mean_slope",
    "masked_minus_background_mean_median",
    "mask_area_fraction_slope",
    "mask_centroid_path_px",
    "front_gradient_peak_radius_slope_px_per_norm_time",
    "apparent_diffusion_q70_um2_per_norm_time",
    "front_radius_slope_px_per_norm_time",
    "front_radius2_slope_px2_per_norm_time",
    "front_radius_monotonic_fraction",
    "front_gradient_strength_median",
    "front_gradient_coherence",
    "phase_fraction_slope_per_norm_time",
    "kymograph_temporal_energy",
    "radial_profile_last_minus_first_l1",
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_count: int = 16) -> List[str]:
    out = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = numeric(df, col)
        if vals.notna().sum() >= min_count and vals.nunique(dropna=True) > 1:
            out.append(col)
    return out


def source_class(source: str) -> str:
    text = str(source)
    if "HighHighCOV" in text:
        return "HighHighCOV"
    if "HighCOV" in text:
        return "HighCOV"
    return "standard"


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
    match_cols: Sequence[str],
    scheme: str,
) -> pd.DataFrame:
    if not treated_idx or not control_idx:
        return pd.DataFrame()
    idx = list(treated_idx) + list(control_idx)
    scaled = robust_scale(df.loc[idx], match_cols)
    src = df["source_stem"].astype(str)
    src_class = src.map(source_class)
    rows = []
    for tidx in treated_idx:
        pool = list(control_idx)
        if scheme == "same_source":
            pool = [c for c in pool if src.loc[c] == src.loc[tidx]]
        if not pool:
            continue
        tv = scaled.loc[tidx, match_cols].to_numpy(dtype=float)
        cv = scaled.loc[pool, match_cols].to_numpy(dtype=float)
        dist = np.sqrt(np.nanmean((cv - tv) ** 2, axis=1))
        if scheme == "source_penalized_echem_context":
            dist = dist + np.array([0.75 if src.loc[c] != src.loc[tidx] else 0.0 for c in pool])
        elif scheme == "source_class_penalized_echem_context":
            dist = dist + np.array([0.5 if src_class.loc[c] != src_class.loc[tidx] else 0.0 for c in pool])
        best_pos = int(np.nanargmin(dist))
        cidx = pool[best_pos]
        rows.append({
            "treated_index": int(tidx),
            "control_index": int(cidx),
            "treated_roi_id": df.loc[tidx, "roi_id"],
            "control_roi_id": df.loc[cidx, "roi_id"],
            "treated_source": src.loc[tidx],
            "control_source": src.loc[cidx],
            "treated_source_class": src_class.loc[tidx],
            "control_source_class": src_class.loc[cidx],
            "same_source": bool(src.loc[tidx] == src.loc[cidx]),
            "same_source_class": bool(src_class.loc[tidx] == src_class.loc[cidx]),
            "treated_cycleNo": clean_json(df.loc[tidx, "cycleNo"]),
            "control_cycleNo": clean_json(df.loc[cidx, "cycleNo"]),
            "treated_event_bin": df.loc[tidx, "event_relative_bin"],
            "control_event_bin": df.loc[cidx, "event_relative_bin"],
            "match_distance": float(dist[best_pos]),
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


def paired_tests(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    outcomes: Sequence[str],
    rng: np.random.Generator,
    n_perm: int,
) -> pd.DataFrame:
    rows = []
    for feature in outcomes:
        t = numeric(df, feature).loc[pairs["treated_index"].to_numpy()].reset_index(drop=True)
        c = numeric(df, feature).loc[pairs["control_index"].to_numpy()].reset_index(drop=True)
        diff = t - c
        valid = diff.notna()
        x = diff[valid]
        row: Dict[str, Any] = {
            "feature": feature,
            "base_feature": feature.replace("_source_echem_context_residual", "").replace("_source_context_residual", ""),
            "scale": (
                "source_echem_context_residual"
                if feature.endswith("_source_echem_context_residual")
                else "source_context_residual" if feature.endswith("_source_context_residual") else "raw"
            ),
            "n_pairs": int(valid.sum()),
            "treated_median": np.nan,
            "control_median": np.nan,
            "mean_treated_minus_control": np.nan,
            "median_treated_minus_control": np.nan,
            "positive_difference_fraction": np.nan,
            "wilcoxon_p": np.nan,
            "signflip_mean_abs_p": np.nan,
        }
        if len(x) >= 6 and x.nunique(dropna=True) > 1:
            try:
                _, p_w = wilcoxon(x, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                p_w = np.nan
            row.update({
                "treated_median": float(t[valid].median()),
                "control_median": float(c[valid].median()),
                "mean_treated_minus_control": float(x.mean()),
                "median_treated_minus_control": float(x.median()),
                "positive_difference_fraction": float((x > 0).mean()),
                "wilcoxon_p": float(p_w) if np.isfinite(p_w) else np.nan,
                "signflip_mean_abs_p": signflip_p(x, rng, n_perm),
            })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_median_difference"] = out["median_treated_minus_control"].abs()
        out = out.sort_values(["signflip_mean_abs_p", "abs_median_difference"], ascending=[True, False])
    return out


def source_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    return (
        pairs.groupby(
            ["treated_source", "control_source", "same_source", "same_source_class"],
            dropna=False,
        )
        .agg(n_pairs=("treated_roi_id", "count"), median_match_distance=("match_distance", "median"))
        .reset_index()
        .sort_values(["n_pairs", "median_match_distance"], ascending=[False, True])
    )


def outcome_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for base in TARGET_BASES:
        for suffix in ("_source_echem_context_residual", "_source_context_residual", ""):
            col = f"{base}{suffix}"
            if col in df.columns:
                vals = numeric(df, col)
                if vals.notna().sum() >= 16 and vals.nunique(dropna=True) > 1:
                    cols.append(col)
    return cols


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Pre-Event Echem-Matched Residual Audit",
        "",
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Match features: {summary['n_match_features']} ({summary['n_context_features']} context, {summary['n_echem_features']} echem)",
        f"- Pair counts: {summary['pair_counts']}",
        "",
        "## Top Residual Matched Tests",
    ]
    for row in summary.get("top_source_echem_residual_matched_tests", [])[:10]:
        lines.append(
            f"- {row['comparison']} {row['match_scheme']} {row['base_feature']}: "
            f"n={row['n_pairs']}, median diff={row['median_treated_minus_control']:.4g}, "
            f"sign-flip p={row['signflip_mean_abs_p']:.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument(
        "--out-dir",
        default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_echem_matched_residual_audit",
    )
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    joined = read_csv(
        derived
        / "source_balanced_pre_event_echem_front_coupling_audit"
        / "source_balanced_pre_event_echem_front_joined.csv"
    ).reset_index(drop=True)
    context_cols = available_numeric(joined, CONTEXT_CANDIDATES, min_count=24)
    echem_cols = available_numeric(joined, ECHEM_CANDIDATES, min_count=24)
    match_cols = sorted(set(context_cols + echem_cols))
    outcomes = outcome_columns(joined)
    bins = joined["event_relative_bin"].astype(str)

    all_pairs = []
    all_tests = []
    all_source = []
    for comparison, treat_bins, control_bins in COMPARISONS:
        treated_idx = joined.index[bins.isin(treat_bins)].tolist()
        control_idx = joined.index[bins.isin(control_bins)].tolist()
        for scheme in MATCH_SCHEMES:
            pairs = nearest_matches(joined, treated_idx, control_idx, match_cols, scheme)
            if pairs.empty:
                continue
            pairs.insert(0, "comparison", comparison)
            pairs.insert(1, "match_scheme", scheme)
            tests = paired_tests(joined, pairs, outcomes, rng, args.n_perm)
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

    pair_path = out / "source_balanced_pre_event_echem_matched_pairs.csv"
    test_path = out / "source_balanced_pre_event_echem_matched_feature_tests.csv"
    source_path = out / "source_balanced_pre_event_echem_matched_source_summary.csv"
    summary_path = out / "source_balanced_pre_event_echem_matched_summary.json"
    pair_df.to_csv(pair_path, index=False)
    test_df.to_csv(test_path, index=False)
    source_df.to_csv(source_path, index=False)

    residual_tests = pd.DataFrame()
    if not test_df.empty:
        residual_tests = test_df[test_df["scale"] == "source_echem_context_residual"].copy()
    top_resid = (
        residual_tests.dropna(subset=["signflip_mean_abs_p"])
        .sort_values(["signflip_mean_abs_p", "abs_median_difference"], ascending=[True, False])
        .head(24)
        .to_dict("records")
        if not residual_tests.empty
        else []
    )
    pair_counts = pair_df.groupby(["comparison", "match_scheme"]).size().to_dict() if not pair_df.empty else {}
    same_source_frac = pair_df.groupby(["comparison", "match_scheme"])["same_source"].mean().to_dict() if not pair_df.empty else {}
    same_class_frac = pair_df.groupby(["comparison", "match_scheme"])["same_source_class"].mean().to_dict() if not pair_df.empty else {}

    summary: Dict[str, Any] = {
        "n_rows": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "n_sources": int(joined["source_stem"].nunique()),
        "event_relative_bin_counts": clean_json(bins.value_counts().to_dict()),
        "context_features": context_cols,
        "echem_features": echem_cols,
        "match_features": match_cols,
        "outcome_features": outcomes,
        "n_context_features": int(len(context_cols)),
        "n_echem_features": int(len(echem_cols)),
        "n_match_features": int(len(match_cols)),
        "n_outcome_features": int(len(outcomes)),
        "n_permutations": int(args.n_perm),
        "pair_counts": clean_json({"|".join(map(str, k)): int(v) for k, v in pair_counts.items()}),
        "same_source_pair_fraction": clean_json({"|".join(map(str, k)): float(v) for k, v in same_source_frac.items()}),
        "same_source_class_pair_fraction": clean_json({"|".join(map(str, k)): float(v) for k, v in same_class_frac.items()}),
        "top_source_echem_residual_matched_tests": clean_json(top_resid),
        "outputs": {
            "pairs": str(pair_path),
            "feature_tests": str(test_path),
            "source_summary": str(source_path),
            "summary": str(summary_path),
        },
        "guardrail": (
            "This audit pairs automatic ROI rows on observed source, acquisition, baseline, and cycle-level echem "
            "descriptors, then tests automatic source+echem residual front/kymograph proxies. Matching reuses controls "
            "and cannot resolve unobserved acquisition effects, particle identity, manual front validity, calibrated "
            "diffusion, phase-boundary mechanism, or degradation causality."
        ),
    }
    summary = clean_json(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
