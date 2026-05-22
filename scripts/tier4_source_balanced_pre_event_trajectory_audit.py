#!/usr/bin/env python3
"""Event-distance trajectory audit for source-balanced pre-event ROI features.

The readout audit tests whether particle-region features separate event-relative
bins. This companion audit asks a stricter physics-facing question: do front,
mask, diffusion-like, and rollout proxies change monotonically as sampled cycles
approach an abrupt optical event? Rows are first aggregated to one row per
source/cycle/event-distance so that the two ROI proposals per cycle do not
double-count evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

PRE_BINS = ["far_pre_event_17_32", "mid_pre_event_9_16", "near_pre_event_1_8"]
BIN_ORDER = {name: i for i, name in enumerate(PRE_BINS)}
TRANSFORMS = ["raw", "source_residual", "event_residual", "within_source_rank"]


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


def source_eta2(values: pd.Series, sources: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce")
    valid = x.notna() & sources.notna()
    x = x[valid]
    src = sources[valid].astype(str)
    if len(x) < 4 or x.nunique() < 2 or src.nunique() < 2:
        return np.nan
    total = float(((x - x.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in x.groupby(src):
        between += len(sub) * float((sub.mean() - x.mean()) ** 2)
    return between / total


def transform_feature(values: pd.Series, sources: pd.Series, event_cycles: pd.Series, transform: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    src = sources.astype(str)
    evt = event_cycles.astype(str)
    if transform == "raw":
        return x
    if transform == "source_residual":
        return x - x.groupby(src).transform("mean")
    if transform == "event_residual":
        return x - x.groupby(evt).transform("mean")
    if transform == "within_source_rank":
        return x.groupby(src).rank(pct=True) - 0.5
    raise ValueError(transform)


def permutation_p_abs_rho(y: pd.Series, x: pd.Series, sources: pd.Series, observed: float, n_perm: int, seed: int) -> float:
    valid = y.notna() & x.notna() & sources.notna()
    yy = y[valid].to_numpy(dtype=float)
    xx = x[valid].to_numpy(dtype=float)
    src = sources[valid].astype(str).to_numpy()
    if len(yy) < 8 or len(np.unique(yy)) < 2 or len(np.unique(xx)) < 2 or not np.isfinite(observed):
        return np.nan
    rng = np.random.default_rng(seed)
    hits = 0
    groups = {s: np.flatnonzero(src == s) for s in np.unique(src)}
    for _ in range(n_perm):
        yp = yy.copy()
        for idx in groups.values():
            if len(idx) > 1:
                yp[idx] = rng.permutation(yp[idx])
        rho, _ = spearmanr(yp, xx)
        if np.isfinite(rho) and abs(float(rho)) >= abs(observed):
            hits += 1
    return float((hits + 1) / (n_perm + 1))


def feature_columns(df: pd.DataFrame) -> List[str]:
    blocked = {
        "roi_id", "npz_path", "preview_png", "source_stem", "event_relative_bin",
        "selection_reason", "validation_label", "cohort_role",
        "clean_pre_1_8_vs_post_control", "clean_pre_1_16_vs_post_control",
        "clean_pre_1_32_vs_post_control", "near_pre_vs_far_pre", "post_event_vs_control",
    }
    keys = [
        "roi_norm_mean", "raw_roi_mean", "temporal_energy", "persistence", "velocity",
        "masked_minus_background", "front_", "mask_", "apparent_diffusion", "object_",
    ]
    cols = []
    for col in df.columns:
        if col in blocked:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 16 and vals.nunique(dropna=True) >= 2 and any(k in col for k in keys):
            cols.append(col)
    return cols


def aggregate_cycles(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    pre = df[df["event_relative_bin"].isin(PRE_BINS)].copy()
    pre["cycleNo"] = numeric(pre, "cycleNo")
    pre["cycles_to_next_event"] = numeric(pre, "cycles_to_next_event")
    pre["event_cycle"] = pre["cycleNo"] + pre["cycles_to_next_event"]
    pre["proximity_score"] = 33.0 - pre["cycles_to_next_event"]
    pre["event_bin_order"] = pre["event_relative_bin"].map(BIN_ORDER).astype(float)
    group_cols = [
        "source_stem", "event_cycle", "cycleNo", "event_relative_bin",
        "cycles_to_next_event", "proximity_score", "event_bin_order",
    ]
    agg = {feature: "mean" for feature in features}
    agg["roi_id"] = "count"
    out = pre.groupby(group_cols, dropna=False).agg(agg).reset_index()
    return out.rename(columns={"roi_id": "n_roi_proposals"})


def feature_tests(cycles: pd.DataFrame, features: Iterable[str], n_perm: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = numeric(cycles, "proximity_score")
    sources = cycles["source_stem"].astype(str)
    event_cycles = numeric(cycles, "event_cycle")
    for feature_index, feature in enumerate(features):
        raw = numeric(cycles, feature)
        for transform in TRANSFORMS:
            x = transform_feature(raw, sources, event_cycles, transform)
            valid = y.notna() & x.notna()
            yy = y[valid]
            xx = x[valid]
            if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
                rho = p_spearman = slope = intercept = np.nan
            else:
                rho, p_spearman = spearmanr(yy, xx)
                slope, intercept = np.polyfit(yy.to_numpy(dtype=float), xx.to_numpy(dtype=float), 1)
            p_perm = permutation_p_abs_rho(
                y, x, sources, float(rho) if np.isfinite(rho) else np.nan, n_perm, seed=20260522 + feature_index
            )
            bin_medians = {}
            groups_for_kruskal = []
            for bin_name in PRE_BINS:
                vals = x[cycles["event_relative_bin"].eq(bin_name)]
                vals = vals[pd.notna(vals)]
                bin_medians[f"median_{bin_name}"] = float(vals.median()) if len(vals) else np.nan
                if len(vals) >= 2:
                    groups_for_kruskal.append(vals.to_numpy(dtype=float))
            try:
                _, p_kruskal = kruskal(*groups_for_kruskal) if len(groups_for_kruskal) >= 2 else (np.nan, np.nan)
            except ValueError:
                p_kruskal = np.nan
            near = bin_medians["median_near_pre_event_1_8"]
            far = bin_medians["median_far_pre_event_17_32"]
            rows.append({
                "feature": feature,
                "transform": transform,
                "n_cycle_rows": int(valid.sum()),
                "n_sources": int(sources[valid].nunique()),
                "n_event_cycles": int(event_cycles[valid].nunique()),
                "spearman_rho_vs_event_proximity": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(p_spearman) if np.isfinite(p_spearman) else np.nan,
                "within_source_permutation_abs_rho_p": p_perm,
                "linear_slope_per_proximity_cycle": float(slope) if np.isfinite(slope) else np.nan,
                "direction": "increases_toward_event" if np.isfinite(rho) and rho > 0 else "decreases_toward_event" if np.isfinite(rho) and rho < 0 else "NA",
                "source_eta2_after_transform": source_eta2(x, sources),
                "raw_source_eta2": source_eta2(raw, sources),
                "kruskal_bin_p": float(p_kruskal) if np.isfinite(p_kruskal) else np.nan,
                "near_minus_far_median": near - far if np.isfinite(near) and np.isfinite(far) else np.nan,
                **bin_medians,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["transform", "within_source_permutation_abs_rho_p", "spearman_p", "spearman_rho_vs_event_proximity"],
            ascending=[True, True, True, False],
        )
    return out


def full_event_deltas(cycles: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for event_cycle, group in cycles.groupby("event_cycle", dropna=False):
        for feature in features:
            vals = numeric(group, feature)
            near = vals[group["event_relative_bin"].eq("near_pre_event_1_8")]
            mid = vals[group["event_relative_bin"].eq("mid_pre_event_9_16")]
            far = vals[group["event_relative_bin"].eq("far_pre_event_17_32")]
            if len(near) == 0 or len(far) == 0:
                continue
            rows.append({
                "event_cycle": event_cycle,
                "n_sources": int(group["source_stem"].nunique()),
                "source_stems": ";".join(sorted(group["source_stem"].astype(str).unique())),
                "feature": feature,
                "n_cycle_rows": int(len(group)),
                "has_mid_bin": bool(len(mid) > 0),
                "near_median": float(near.median()),
                "mid_median": float(mid.median()) if len(mid) else np.nan,
                "far_median": float(far.median()),
                "near_minus_far": float(near.median() - far.median()),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["feature", "event_cycle"])
    return out


def summarize_bins(cycles: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    agg: Dict[str, Any] = {
        "cycleNo": "count",
        "source_stem": pd.Series.nunique,
        "event_cycle": pd.Series.nunique,
        "n_roi_proposals": "sum",
    }
    for feature in features[:12]:
        agg[feature] = "median"
    out = cycles.groupby("event_relative_bin", dropna=False).agg(agg).reset_index()
    return out.rename(columns={"cycleNo": "n_cycle_rows", "source_stem": "n_sources", "event_cycle": "n_event_cycles"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_readout_audit")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_trajectory_audit")
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    readout_dir = Path(args.readout_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(readout_dir / "source_balanced_pre_event_readout_features.csv")
    features = feature_columns(df)
    cycles = aggregate_cycles(df, features)
    tests = feature_tests(cycles, features, args.n_permutations)
    deltas = full_event_deltas(cycles, features)
    bin_summary = summarize_bins(cycles, features)

    top_source_residual = tests[
        tests["transform"].eq("source_residual")
        & tests["spearman_rho_vs_event_proximity"].notna()
    ].sort_values(["within_source_permutation_abs_rho_p", "spearman_p"], ascending=[True, True]).head(12)
    top_physics_toward_event = tests[
        tests["transform"].isin(["source_residual", "event_residual"])
        & tests["direction"].eq("increases_toward_event")
        & tests["feature"].str.contains("front_|mask_|masked_minus_background|apparent_diffusion", regex=True)
    ].sort_values(["within_source_permutation_abs_rho_p", "spearman_p"], ascending=[True, True]).head(12)

    cycles.to_csv(out_dir / "source_balanced_pre_event_trajectory_cycle_features.csv", index=False)
    tests.to_csv(out_dir / "source_balanced_pre_event_trajectory_feature_tests.csv", index=False)
    deltas.to_csv(out_dir / "source_balanced_pre_event_trajectory_full_event_deltas.csv", index=False)
    bin_summary.to_csv(out_dir / "source_balanced_pre_event_trajectory_bin_summary.csv", index=False)

    summary = {
        "input_readout_dir": str(readout_dir),
        "n_roi_rows_input": int(len(df)),
        "n_pre_event_roi_rows": int(df["event_relative_bin"].isin(PRE_BINS).sum()),
        "n_cycle_rows": int(len(cycles)),
        "n_sources": int(cycles["source_stem"].nunique()),
        "n_event_cycles": int(cycles["event_cycle"].nunique()),
        "n_features_tested": int(len(features)),
        "n_permutations": int(args.n_permutations),
        "bin_summary": clean_json(bin_summary.to_dict("records")),
        "top_source_residual_event_distance_tests": clean_json(top_source_residual.to_dict("records")),
        "top_physics_toward_event_tests": clean_json(top_physics_toward_event.to_dict("records")),
        "full_event_delta_rows": int(len(deltas)),
        "full_event_cycles_with_near_and_far": int(deltas["event_cycle"].nunique()) if not deltas.empty else 0,
        "guardrail": (
            "Cycle-level event-distance trajectories reduce duplicate ROI counting and test monotonic pre-event organization, "
            "but automatic crops/masks and sparse full far-mid-near event trajectories mean these are physics triage signals, "
            "not calibrated phase boundaries, diffusion coefficients, or causal degradation forecasts."
        ),
    }
    (out_dir / "source_balanced_pre_event_trajectory_summary.json").write_text(
        json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "README.md").write_text(
        "# Source-Balanced Pre-Event Trajectory Audit\n\n"
        "Cycle-level event-distance tests for the source-balanced pre-event ROI cohort. "
        "The audit aggregates the two ROI proposals per sampled cycle, then tests whether "
        "front/mask/diffusion-like and rollout proxies vary monotonically as cycles approach "
        "abrupt optical event cycles.\n\n"
        "Key files:\n\n"
        "- `source_balanced_pre_event_trajectory_cycle_features.csv`: one row per source/cycle/event-distance.\n"
        "- `source_balanced_pre_event_trajectory_feature_tests.csv`: Spearman/slope/permutation tests versus event proximity.\n"
        "- `source_balanced_pre_event_trajectory_full_event_deltas.csv`: near-minus-far deltas for source/event groups containing both bins.\n"
        "- `source_balanced_pre_event_trajectory_summary.json`: compact machine-readable summary.\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
