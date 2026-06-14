#!/usr/bin/env python3
"""Same-source event-ladder audit for source-balanced pre-event ROI physics.

The matched counterfactual audit showed that near-pre versus far-pre matching
had no same-source pairs. This audit keeps the pairing strictly within source
and asks a narrower question: when a source has near-pre rows plus any available
mid-pre, post-event, or no-near-event controls, do particle-region descriptors
move consistently along that local event-relative ladder?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

PRE_NEAR = {"near_pre_event_1_8"}
PRE_MID = {"mid_pre_event_9_16"}
PRE_FAR = {"far_pre_event_17_32"}
POST_CONTROL = {"post_event_1_16", "no_near_event_control"}

COMPARISONS: List[Tuple[str, Set[str], Set[str]]] = [
    ("near_vs_mid_pre_same_source", PRE_NEAR, PRE_MID),
    ("near_vs_far_pre_same_source", PRE_NEAR, PRE_FAR),
    ("near_vs_post_control_same_source", PRE_NEAR, POST_CONTROL),
    ("near_vs_any_non_near_same_source", PRE_NEAR, PRE_MID | PRE_FAR | POST_CONTROL),
]

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


def available_numeric(df: pd.DataFrame, cols: Sequence[str], min_count: int = 8) -> List[str]:
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


def nearest_same_source_pairs(
    df: pd.DataFrame,
    treat_bins: Set[str],
    control_bins: Set[str],
    context_cols: Sequence[str],
) -> pd.DataFrame:
    bins = df["event_relative_bin"].astype(str)
    rows = []
    for source, sub in df.groupby("source_stem"):
        treated = sub.index[bins.loc[sub.index].isin(treat_bins)].tolist()
        controls = sub.index[bins.loc[sub.index].isin(control_bins)].tolist()
        if not treated or not controls:
            continue
        scaled = robust_scale(df.loc[treated + controls], context_cols)
        for tidx in treated:
            tv = scaled.loc[tidx, context_cols].to_numpy(dtype=float)
            cv = scaled.loc[controls, context_cols].to_numpy(dtype=float)
            d = np.sqrt(np.nanmean((cv - tv) ** 2, axis=1))
            best_pos = int(np.nanargmin(d))
            cidx = controls[best_pos]
            rows.append({
                "treated_index": int(tidx),
                "control_index": int(cidx),
                "treated_roi_id": df.loc[tidx, "roi_id"],
                "control_roi_id": df.loc[cidx, "roi_id"],
                "source_stem": source,
                "treated_cycleNo": clean_json(df.loc[tidx, "cycleNo"]),
                "control_cycleNo": clean_json(df.loc[cidx, "cycleNo"]),
                "treated_event_bin": df.loc[tidx, "event_relative_bin"],
                "control_event_bin": df.loc[cidx, "event_relative_bin"],
                "match_distance": float(d[best_pos]),
                "cycle_delta_treated_minus_control": clean_json(
                    numeric(df, "cycleNo").loc[tidx] - numeric(df, "cycleNo").loc[cidx]
                ),
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


def paired_feature_tests(
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
        if len(x) >= 6 and x.nunique(dropna=True) > 1:
            try:
                _, p_w = wilcoxon(x, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                p_w = np.nan
            p_sf = signflip_p(x, rng, n_perm)
            row = {
                "feature": feature,
                "n_pairs": int(valid.sum()),
                "treated_median": float(t[valid].median()),
                "control_median": float(c[valid].median()),
                "mean_treated_minus_control": float(x.mean()),
                "median_treated_minus_control": float(x.median()),
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
                "mean_treated_minus_control": np.nan,
                "median_treated_minus_control": np.nan,
                "positive_difference_fraction": np.nan,
                "wilcoxon_p": np.nan,
                "signflip_mean_abs_p": np.nan,
            }
        rows.append(row)
    out = pd.DataFrame(rows)
    out["abs_median_difference"] = out["median_treated_minus_control"].abs()
    return out.sort_values(["signflip_mean_abs_p", "abs_median_difference"], ascending=[True, False])


def source_ladder(df: pd.DataFrame) -> pd.DataFrame:
    bins = df["event_relative_bin"].astype(str)
    rows = []
    for source, sub in df.groupby("source_stem"):
        counts = bins.loc[sub.index].value_counts().to_dict()
        rows.append({
            "source_stem": source,
            "n_rows": int(len(sub)),
            "near_pre_rows": int(counts.get("near_pre_event_1_8", 0)),
            "mid_pre_rows": int(counts.get("mid_pre_event_9_16", 0)),
            "far_pre_rows": int(counts.get("far_pre_event_17_32", 0)),
            "post_control_rows": int(counts.get("post_event_1_16", 0) + counts.get("no_near_event_control", 0)),
            "has_near_mid_ladder": bool(counts.get("near_pre_event_1_8", 0) and counts.get("mid_pre_event_9_16", 0)),
            "has_near_far_ladder": bool(counts.get("near_pre_event_1_8", 0) and counts.get("far_pre_event_17_32", 0)),
            "has_near_post_ladder": bool(
                counts.get("near_pre_event_1_8", 0)
                and (counts.get("post_event_1_16", 0) + counts.get("no_near_event_control", 0))
            ),
        })
    return pd.DataFrame(rows).sort_values(["near_pre_rows", "mid_pre_rows", "post_control_rows"], ascending=False)


def within_source_clock_tests(df: pd.DataFrame, outcomes: Sequence[str]) -> pd.DataFrame:
    rows = []
    x = numeric(df, "cycles_to_next_event")
    for feature in outcomes:
        vals = numeric(df, feature)
        residuals = []
        clocks = []
        for _, sub in df.groupby("source_stem"):
            idx = sub.index
            if x.loc[idx].notna().sum() < 3 or vals.loc[idx].notna().sum() < 3:
                continue
            residuals.extend((vals.loc[idx] - vals.loc[idx].median()).tolist())
            clocks.extend(x.loc[idx].tolist())
        tmp = pd.DataFrame({"clock": clocks, "value": residuals}).dropna()
        if len(tmp) >= 8 and tmp["clock"].nunique() > 2 and tmp["value"].nunique() > 2:
            rho, p = spearmanr(-tmp["clock"], tmp["value"])
        else:
            rho, p = np.nan, np.nan
        rows.append({
            "feature": feature,
            "n_rows": int(len(tmp)),
            "within_source_residual_spearman_rho_vs_event_proximity": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_p": float(p) if np.isfinite(p) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        "within_source_residual_spearman_rho_vs_event_proximity",
        key=lambda s: s.abs(),
        ascending=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_same_source_ladder_audit")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = read_csv(derived / "source_balanced_pre_event_readout_audit" / "source_balanced_pre_event_readout_features.csv").reset_index(drop=True)
    context_cols = available_numeric(df, CONTEXT_CANDIDATES, min_count=16)
    outcomes = available_numeric(df, OUTCOME_CANDIDATES, min_count=8)

    all_pairs = []
    all_tests = []
    for comparison, treated_bins, control_bins in COMPARISONS:
        pairs = nearest_same_source_pairs(df, treated_bins, control_bins, context_cols)
        if pairs.empty:
            continue
        pairs.insert(0, "comparison", comparison)
        tests = paired_feature_tests(df, pairs, outcomes, rng, args.n_perm)
        tests.insert(0, "comparison", comparison)
        all_pairs.append(pairs)
        all_tests.append(tests)

    pair_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    test_df = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()
    ladder_df = source_ladder(df)
    clock_df = within_source_clock_tests(df, outcomes)

    pair_path = out / "source_balanced_pre_event_same_source_pairs.csv"
    test_path = out / "source_balanced_pre_event_same_source_feature_tests.csv"
    ladder_path = out / "source_balanced_pre_event_same_source_ladder_coverage.csv"
    clock_path = out / "source_balanced_pre_event_same_source_clock_tests.csv"
    summary_path = out / "source_balanced_pre_event_same_source_ladder_summary.json"
    pair_df.to_csv(pair_path, index=False)
    test_df.to_csv(test_path, index=False)
    ladder_df.to_csv(ladder_path, index=False)
    clock_df.to_csv(clock_path, index=False)

    physics_terms = ("front_", "apparent_diffusion", "masked_minus_background", "mask_")
    top_paired = []
    if not test_df.empty:
        for comparison, sub in test_df.groupby("comparison"):
            phys = sub[sub["feature"].astype(str).map(lambda x: any(term in x for term in physics_terms))]
            top_paired.extend(
                phys.sort_values(["signflip_mean_abs_p", "abs_median_difference"], ascending=[True, False])
                .head(8)
                .to_dict("records")
            )
    top_clocks = clock_df[
        clock_df["feature"].astype(str).map(lambda x: any(term in x for term in physics_terms))
    ].head(12).to_dict("records")

    pair_counts = pair_df.groupby("comparison").size().to_dict() if not pair_df.empty else {}
    comparison_source_counts = (
        pair_df.groupby("comparison")["source_stem"].nunique().to_dict() if not pair_df.empty else {}
    )
    ladder_counts = {
        "sources_with_near_rows": int((ladder_df["near_pre_rows"] > 0).sum()),
        "sources_with_near_mid_ladder": int(ladder_df["has_near_mid_ladder"].sum()),
        "sources_with_near_far_ladder": int(ladder_df["has_near_far_ladder"].sum()),
        "sources_with_near_post_ladder": int(ladder_df["has_near_post_ladder"].sum()),
    }

    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "context_features": context_cols,
        "outcome_features": outcomes,
        "n_permutations": int(args.n_perm),
        "event_relative_bin_counts": clean_json(df["event_relative_bin"].astype(str).value_counts().to_dict()),
        "ladder_source_counts": ladder_counts,
        "pair_counts": clean_json({str(k): int(v) for k, v in pair_counts.items()}),
        "comparison_source_counts": clean_json({str(k): int(v) for k, v in comparison_source_counts.items()}),
        "top_same_source_paired_tests": clean_json(top_paired[:20]),
        "top_within_source_clock_tests": clean_json(top_clocks),
        "outputs": {
            "pairs": str(pair_path),
            "feature_tests": str(test_path),
            "ladder_coverage": str(ladder_path),
            "clock_tests": str(clock_path),
            "summary": str(summary_path),
        },
        "guardrail": "This audit is strictly within-source but opportunistic: sources have incomplete event-relative ladders, pairs reuse controls, and all particle masks/fronts are automatic. It tests whether observed near-pre signals survive same-source local controls; it does not validate particle identity, calibrated diffusion, phase-boundary motion, causality, or deployable warnings.",
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")

    lines = [
        "# Source-Balanced Pre-Event Same-Source Ladder Audit",
        "",
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Ladder source counts: {summary['ladder_source_counts']}",
        f"- Pair counts: {summary['pair_counts']}",
        "",
        "## Top Same-Source Paired Physics Tests",
    ]
    for row in top_paired[:10]:
        lines.append(
            f"- {row['comparison']} {row['feature']}: n={row['n_pairs']}, median near-control diff={row['median_treated_minus_control']:.4g}, sign-flip p={row['signflip_mean_abs_p']:.4g}"
        )
    lines += ["", "## Top Within-Source Clock Tests"]
    for row in top_clocks[:8]:
        lines.append(
            f"- {row['feature']}: n={row['n_rows']}, rho={row['within_source_residual_spearman_rho_vs_event_proximity']:.4g}, p={row['spearman_p']:.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
