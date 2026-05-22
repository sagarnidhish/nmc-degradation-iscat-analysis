#!/usr/bin/env python3
"""Echem/source-class matched far-control audit for pre-event front signals.

The raw source lattice shows no same-source near-pre versus far-pre controls.
This audit uses the next defensible control design: match each near-pre ROI to
far-pre controls by source/acquisition class plus cycle-level electrochemical
state and crop context, then test whether front/mask/kymograph descriptors
remain directionally organized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


NEAR = {"near_pre_event_1_8"}
FAR = {"far_pre_event_17_32"}

MATCH_SCHEMES: List[Tuple[str, bool, bool]] = [
    ("same_source_class_echem_context", True, True),
    ("same_source_class_context_only", True, False),
    ("global_echem_context", False, True),
]

ECHEM_CANDIDATES = [
    "capacity_mAh",
    "capacity_fade_from_first_mAh",
    "capacity_fraction_of_first",
    "coulombic_efficiency_pct",
    "charge_discharge_capacity_abs_gap_mAh",
    "voltage_peak_hysteresis_proxy",
    "highV_charge_discharge_imbalance",
    "midV_charge_discharge_imbalance",
    "lowV_charge_discharge_imbalance",
    "echem_outlier_score",
    "echem_regime_pc1",
    "echem_regime_pc2",
    "echem_regime_pc3",
    "echem_regime_pc4",
    "shape_V_mean",
    "shape_V_range",
    "shape_I_abs_mean_mA",
    "all_dq_abs_highV_frac",
    "all_dq_abs_midV_frac",
    "all_dq_abs_lowV_frac",
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
    "roi_norm_mean_first",
    "stage_drift_xy_sampled",
    "mask_base_area_fraction",
]

OUTCOME_CANDIDATES = [
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
    "front_radius_slope_r2",
    "front_radius2_slope_r2",
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


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_count: int = 12) -> List[str]:
    out = []
    for col in cols:
        if col in df.columns:
            vals = numeric(df, col)
            if vals.notna().sum() >= min_count and vals.nunique(dropna=True) >= 2:
                out.append(col)
    return out


def source_class(source_stem: Any) -> str:
    text = str(source_stem)
    if "HighHighCOV" in text:
        return "HighHighCOV"
    if "HighCOV" in text:
        return "HighCOV"
    return "nominal"


def robust_scaled(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
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


def merge_inputs(derived: Path) -> pd.DataFrame:
    readout = read_csv(
        derived / "source_balanced_pre_event_readout_audit" / "source_balanced_pre_event_readout_features.csv"
    )
    kymo = read_csv(
        derived / "source_balanced_pre_event_radial_kymograph_audit" / "source_balanced_pre_event_radial_kymograph_features.csv"
    )
    echem = read_csv(derived / "echem_optical_regime_atlas" / "echem_optical_regime_cycle_table.csv")
    kymo_cols = ["roi_id"] + [c for c in OUTCOME_CANDIDATES if c in kymo.columns and c not in readout.columns]
    joined = readout.merge(kymo[kymo_cols], on="roi_id", how="left")
    echem_cols = ["cycleNo"] + [c for c in ECHEM_CANDIDATES if c in echem.columns]
    joined = joined.merge(echem[echem_cols], on="cycleNo", how="left", suffixes=("", "_echem"))
    joined["source_class"] = joined["source_stem"].map(source_class)
    return joined.reset_index(drop=True)


def nearest_pairs(df: pd.DataFrame, feature_cols: Sequence[str], same_class: bool) -> pd.DataFrame:
    bins = df["event_relative_bin"].astype(str)
    near_idx = df.index[bins.isin(NEAR)].tolist()
    far_idx = df.index[bins.isin(FAR)].tolist()
    if not near_idx or not far_idx or not feature_cols:
        return pd.DataFrame()
    scaled = robust_scaled(df, feature_cols)
    rows = []
    for tidx in near_idx:
        candidates = far_idx
        if same_class:
            cls = df.loc[tidx, "source_class"]
            candidates = [i for i in candidates if df.loc[i, "source_class"] == cls]
        if not candidates:
            continue
        tv = scaled.loc[tidx, feature_cols].to_numpy(dtype=float)
        cv = scaled.loc[candidates, feature_cols].to_numpy(dtype=float)
        d = np.sqrt(np.nanmean((cv - tv) ** 2, axis=1))
        best_pos = int(np.nanargmin(d))
        cidx = candidates[best_pos]
        rows.append({
            "treated_index": int(tidx),
            "control_index": int(cidx),
            "treated_roi_id": df.loc[tidx, "roi_id"],
            "control_roi_id": df.loc[cidx, "roi_id"],
            "treated_source_stem": df.loc[tidx, "source_stem"],
            "control_source_stem": df.loc[cidx, "source_stem"],
            "treated_source_class": df.loc[tidx, "source_class"],
            "control_source_class": df.loc[cidx, "source_class"],
            "same_source": bool(df.loc[tidx, "source_stem"] == df.loc[cidx, "source_stem"]),
            "treated_cycleNo": clean_json(df.loc[tidx, "cycleNo"]),
            "control_cycleNo": clean_json(df.loc[cidx, "cycleNo"]),
            "match_distance": float(d[best_pos]),
            "cycle_delta_treated_minus_control": clean_json(numeric(df, "cycleNo").loc[tidx] - numeric(df, "cycleNo").loc[cidx]),
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
                _, p_w = wilcoxon(x, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                p_w = np.nan
            p_sf = signflip_p(x, rng, n_perm)
            row = {
                "feature": feature,
                "n_pairs": int(valid.sum()),
                "treated_median": float(t[valid].median()),
                "control_median": float(c[valid].median()),
                "median_treated_minus_control": float(x.median()),
                "mean_treated_minus_control": float(x.mean()),
                "positive_difference_fraction": float((x > 0).mean()),
                "wilcoxon_p": float(p_w) if np.isfinite(p_w) else np.nan,
                "signflip_mean_abs_p": p_sf,
            }
        else:
            row = {
                "feature": feature,
                "n_pairs": int(valid.sum()),
                "treated_median": np.nan,
                "control_median": np.nan,
                "median_treated_minus_control": np.nan,
                "mean_treated_minus_control": np.nan,
                "positive_difference_fraction": np.nan,
                "wilcoxon_p": np.nan,
                "signflip_mean_abs_p": np.nan,
            }
        rows.append(row)
    out = pd.DataFrame(rows)
    out["abs_median_difference"] = out["median_treated_minus_control"].abs()
    return out.sort_values(["signflip_mean_abs_p", "abs_median_difference"], ascending=[True, False])


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Pre-Event Echem-Matched Far-Control Audit",
        "",
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Echem match features: {summary['echem_features']}",
        f"- Context match features: {summary['context_features']}",
        f"- Pair counts: {summary['pair_counts']}",
        "",
        "## Top Paired Near-vs-Far Tests",
    ]
    for row in summary.get("top_paired_tests", [])[:10]:
        lines.append(
            f"- {row['match_scheme']} {row['feature']}: n={row['n_pairs']}, "
            f"median near-far diff={row['median_treated_minus_control']:.4g}, "
            f"positive fraction={row['positive_difference_fraction']:.3f}, p={row['signflip_mean_abs_p']:.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument(
        "--out-dir",
        default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_echem_matched_far_control_audit",
    )
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = merge_inputs(derived)
    context = available_numeric(df, CONTEXT_CANDIDATES, min_count=16)
    echem = available_numeric(df, ECHEM_CANDIDATES, min_count=16)
    outcomes = available_numeric(df, OUTCOME_CANDIDATES, min_count=8)

    all_pairs = []
    all_tests = []
    for scheme, same_class, use_echem in MATCH_SCHEMES:
        features = context + (echem if use_echem else [])
        pairs = nearest_pairs(df, features, same_class=same_class)
        if pairs.empty:
            continue
        pairs.insert(0, "match_scheme", scheme)
        tests = paired_tests(df, pairs, outcomes, rng, args.n_perm)
        tests.insert(0, "match_scheme", scheme)
        all_pairs.append(pairs)
        all_tests.append(tests)

    pair_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    test_df = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()
    coverage_rows = []
    for source, sub in df.groupby(["source_class", "source_stem"]):
        bins = sub["event_relative_bin"].astype(str).value_counts().to_dict()
        coverage_rows.append({
            "source_class": source[0],
            "source_stem": source[1],
            "near_rows": int(bins.get("near_pre_event_1_8", 0)),
            "far_rows": int(bins.get("far_pre_event_17_32", 0)),
            "mid_rows": int(bins.get("mid_pre_event_9_16", 0)),
            "post_control_rows": int(bins.get("post_event_1_16", 0) + bins.get("no_near_event_control", 0)),
        })
    coverage_df = pd.DataFrame(coverage_rows).sort_values(["source_class", "near_rows", "far_rows"], ascending=[True, False, False])

    pair_path = out / "source_balanced_pre_event_echem_matched_far_pairs.csv"
    test_path = out / "source_balanced_pre_event_echem_matched_far_tests.csv"
    coverage_path = out / "source_balanced_pre_event_echem_matched_far_coverage.csv"
    summary_path = out / "source_balanced_pre_event_echem_matched_far_summary.json"
    pair_df.to_csv(pair_path, index=False)
    test_df.to_csv(test_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)

    physics_terms = ("front_", "apparent_diffusion", "masked_minus_background", "mask_", "kymograph")
    top = []
    if not test_df.empty:
        phys = test_df[test_df["feature"].astype(str).map(lambda x: any(term in x for term in physics_terms))]
        top = phys.sort_values(["signflip_mean_abs_p", "abs_median_difference"], ascending=[True, False]).head(24).to_dict("records")
    pair_counts = pair_df.groupby("match_scheme").size().to_dict() if not pair_df.empty else {}
    pair_source_counts = pair_df.groupby("match_scheme")["control_source_stem"].nunique().to_dict() if not pair_df.empty else {}

    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "event_relative_bin_counts": clean_json(df["event_relative_bin"].astype(str).value_counts().to_dict()),
        "source_class_counts": clean_json(df["source_class"].value_counts().to_dict()),
        "context_features": context,
        "echem_features": echem,
        "outcome_features": outcomes,
        "n_permutations": int(args.n_perm),
        "pair_counts": clean_json({str(k): int(v) for k, v in pair_counts.items()}),
        "control_source_counts": clean_json({str(k): int(v) for k, v in pair_source_counts.items()}),
        "top_paired_tests": clean_json(top),
        "outputs": {
            "pairs": str(pair_path),
            "feature_tests": str(test_path),
            "coverage": str(coverage_path),
            "summary": str(summary_path),
        },
        "guardrail": "This is a cross-source far-control stress test because no same-source near-vs-far lattice exists. Source-class and echem/context matching reduce obvious confounding but cannot prove particle identity, causality, calibrated diffusion, or validated phase-boundary tracking.",
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out, clean_json(summary))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
