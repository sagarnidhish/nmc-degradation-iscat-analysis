#!/usr/bin/env python3
"""Temporal directionality audit for source-balanced pre-event ROI readouts.

This audit asks whether automatic pre-event ROI features strengthen as sampled
cycles approach abrupt degradation events, and whether that pre-event clock is
stronger than a post-event washout clock after source normalization. It uses
compact feature CSVs from the pre-event sequence/readout audits; ROI tensors
remain on Isambard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PRE_BINS = {"near_pre_event_1_8", "mid_pre_event_9_16", "far_pre_event_17_32"}
POST_BIN = "post_event_1_16"
CONTROL_BIN = "no_near_event_control"
TRANSFORMS = ["raw", "source_residual", "within_source_rank"]


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


def transform_feature(values: pd.Series, sources: pd.Series, transform: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    src = sources.astype(str)
    if transform == "raw":
        return x
    if transform == "source_residual":
        return x - x.groupby(src).transform("mean")
    if transform == "within_source_rank":
        return x.groupby(src).rank(pct=True) - 0.5
    raise ValueError(transform)


def oriented_auc(y: pd.Series, x: pd.Series) -> Tuple[float, float, str, float, float, float]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
        return np.nan, np.nan, "NA", np.nan, np.nan, np.nan
    direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
    score = xx if direction == "higher_in_positive" else -xx
    auc = float(roc_auc_score(yy, score))
    ap = float(average_precision_score(yy, score))
    try:
        _, p_mwu = mannwhitneyu(xx[yy == 1], xx[yy == 0], alternative="two-sided")
    except ValueError:
        p_mwu = np.nan
    return auc, ap, direction, float(xx[yy == 1].median()), float(xx[yy == 0].median()), float(p_mwu)


def permute_within_source(values: pd.Series, sources: pd.Series, rng: np.random.Generator) -> pd.Series:
    out = values.copy()
    for _, idx in values.groupby(sources.astype(str)).groups.items():
        idx = list(idx)
        arr = out.loc[idx].to_numpy(copy=True)
        rng.shuffle(arr)
        out.loc[idx] = arr
    return out


def spearman_test(clock: pd.Series, feature: pd.Series, sources: pd.Series, n_perm: int, rng: np.random.Generator) -> Dict[str, Any]:
    valid = clock.notna() & feature.notna()
    if valid.sum() < 8 or clock[valid].nunique() < 2 or feature[valid].nunique() < 2:
        return {"n": int(valid.sum()), "rho": np.nan, "p": np.nan, "perm_p_abs": np.nan}
    rho, p = spearmanr(clock[valid], feature[valid])
    null = []
    for _ in range(n_perm):
        shuffled = permute_within_source(clock[valid].copy(), sources[valid], rng)
        rr, _ = spearmanr(shuffled, feature[valid])
        if np.isfinite(rr):
            null.append(abs(float(rr)))
    perm_p = (sum(v >= abs(float(rho)) for v in null) + 1) / (len(null) + 1) if null else np.nan
    return {"n": int(valid.sum()), "rho": float(rho), "p": float(p), "perm_p_abs": float(perm_p)}


def auc_permutation(y: pd.Series, x: pd.Series, sources: pd.Series, n_perm: int, rng: np.random.Generator) -> float:
    auc, _, direction, _, _, _ = oriented_auc(y, x)
    if not np.isfinite(auc):
        return np.nan
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(float)
    xx = x[valid]
    null = []
    for _ in range(n_perm):
        shuffled = permute_within_source(yy.copy(), sources[valid], rng)
        aa, _, _, _, _, _ = oriented_auc(shuffled, xx)
        if np.isfinite(aa):
            null.append(float(aa))
    return (sum(v >= float(auc) for v in null) + 1) / (len(null) + 1) if null else np.nan


def feature_columns(df: pd.DataFrame) -> List[str]:
    blocked = {
        "roi_id", "npz_path", "preview_png", "source_stem", "event_relative_bin",
        "selection_reason", "validation_label", "cohort_role",
        "future_any_drop_within_8cycles", "future_any_drop_within_16cycles",
        "future_event_within_32cycles", "past_event_within_16cycles", "any_abrupt_drop",
        "clean_pre_1_8_vs_post_control", "clean_pre_1_16_vs_post_control",
        "clean_pre_1_32_vs_post_control", "near_pre_vs_far_pre", "post_event_vs_control",
        "target_near_pre_vs_rest", "target_pre16_clean_vs_post_control", "target_any_pre_vs_post_control",
    }
    keys = [
        "roi_norm_mean", "raw_roi_mean", "temporal_energy", "persistence", "velocity",
        "masked_minus_background", "front_", "mask_", "apparent_diffusion", "object_",
        "frame_diff", "spatial_std", "bright_fraction", "dark_fraction",
    ]
    cols = []
    for col in df.columns:
        if col in blocked:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 16 and vals.nunique(dropna=True) >= 2 and any(k in col for k in keys):
            cols.append(col)
    return cols


def merge_inputs(readout: pd.DataFrame, sequence: pd.DataFrame) -> pd.DataFrame:
    seq_extra = [c for c in sequence.columns if c not in readout.columns or c == "roi_id"]
    return readout.merge(sequence[seq_extra], on="roi_id", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-features", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_readout_audit/source_balanced_pre_event_readout_features.csv")
    parser.add_argument("--sequence-features", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_sequence_audit/source_balanced_pre_event_sequence_features.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_directionality_audit")
    parser.add_argument("--n-perm", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    readout = pd.read_csv(args.readout_features)
    sequence = pd.read_csv(args.sequence_features)
    df = merge_inputs(readout, sequence)
    bins = df["event_relative_bin"].astype(str)
    sources = df["source_stem"].astype(str)
    features = feature_columns(df)

    df["pre_clock_closer_to_event"] = np.where(bins.isin(PRE_BINS), -numeric(df, "cycles_to_next_event"), np.nan)
    df["post_clock_farther_from_event"] = np.where(bins.eq(POST_BIN), numeric(df, "cycles_since_prev_event"), np.nan)
    df["near_pre_vs_far_pre"] = np.where(bins.eq("near_pre_event_1_8"), 1, np.where(bins.eq("far_pre_event_17_32"), 0, np.nan))
    df["clean_pre8_vs_post_control"] = np.where(bins.eq("near_pre_event_1_8"), 1, np.where(bins.isin({POST_BIN, CONTROL_BIN}), 0, np.nan))
    df["post_vs_control"] = np.where(bins.eq(POST_BIN), 1, np.where(bins.eq(CONTROL_BIN), 0, np.nan))

    rows: List[Dict[str, Any]] = []
    for feature in features:
        raw = numeric(df, feature)
        for transform in TRANSFORMS:
            x = transform_feature(raw, sources, transform)
            pre_clock = spearman_test(numeric(df, "pre_clock_closer_to_event"), x, sources, args.n_perm, rng)
            post_clock = spearman_test(numeric(df, "post_clock_farther_from_event"), x, sources, args.n_perm, rng)
            for target in ["near_pre_vs_far_pre", "clean_pre8_vs_post_control", "post_vs_control"]:
                y = numeric(df, target)
                auc, ap, direction, med_pos, med_neg, p_mwu = oriented_auc(y, x)
                rows.append({
                    "feature": feature,
                    "transform": transform,
                    "target": target,
                    "n": int((y.isin([0, 1]) & x.notna()).sum()),
                    "n_positive": int(y[y.isin([0, 1])].sum()),
                    "oriented_auc": auc,
                    "average_precision": ap,
                    "direction": direction,
                    "median_positive": med_pos,
                    "median_negative": med_neg,
                    "median_positive_minus_negative": med_pos - med_neg if np.isfinite(med_pos) and np.isfinite(med_neg) else np.nan,
                    "mwu_p": p_mwu,
                    "within_source_perm_p": np.nan,
                    "source_eta2_after_transform": source_eta2(x, sources),
                    "raw_source_eta2": source_eta2(raw, sources),
                    "pre_clock_n": pre_clock["n"],
                    "pre_clock_rho": pre_clock["rho"],
                    "pre_clock_p": pre_clock["p"],
                    "pre_clock_perm_p_abs": pre_clock["perm_p_abs"],
                    "post_clock_n": post_clock["n"],
                    "post_clock_rho": post_clock["rho"],
                    "post_clock_p": post_clock["p"],
                    "post_clock_perm_p_abs": post_clock["perm_p_abs"],
                    "abs_pre_minus_abs_post_rho": abs(pre_clock["rho"]) - abs(post_clock["rho"]) if np.isfinite(pre_clock["rho"]) and np.isfinite(post_clock["rho"]) else np.nan,
                })

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(
        ["target", "transform", "oriented_auc", "pre_clock_perm_p_abs", "average_precision"],
        ascending=[True, True, False, True, False],
    )
    clock_rows = (
        metrics[["feature", "transform", "pre_clock_n", "pre_clock_rho", "pre_clock_p", "pre_clock_perm_p_abs",
                 "post_clock_n", "post_clock_rho", "post_clock_p", "post_clock_perm_p_abs",
                 "abs_pre_minus_abs_post_rho", "source_eta2_after_transform", "raw_source_eta2"]]
        .drop_duplicates(["feature", "transform"])
        .sort_values(["pre_clock_perm_p_abs", "pre_clock_rho"], ascending=[True, False])
    )

    best_by_target = []
    for _, sub in metrics.groupby(["target", "transform"], dropna=False):
        best_by_target.extend(sub.head(1).to_dict("records"))
    best_pre_clock = clock_rows.sort_values(["pre_clock_perm_p_abs", "pre_clock_rho"], ascending=[True, False]).head(12)
    best_asymmetry = clock_rows.sort_values(["abs_pre_minus_abs_post_rho", "pre_clock_perm_p_abs"], ascending=[False, True]).head(12)

    paths = {
        "merged_features": out / "source_balanced_pre_event_directionality_features.csv",
        "metrics": out / "source_balanced_pre_event_directionality_metrics.csv",
        "clock_metrics": out / "source_balanced_pre_event_directionality_clock_metrics.csv",
        "best_by_target": out / "source_balanced_pre_event_directionality_best_by_target.csv",
        "summary": out / "source_balanced_pre_event_directionality_summary.json",
    }
    df.to_csv(paths["merged_features"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    clock_rows.to_csv(paths["clock_metrics"], index=False)
    pd.DataFrame(best_by_target).to_csv(paths["best_by_target"], index=False)

    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "event_relative_bin_counts": clean_json(bins.value_counts().to_dict()),
        "n_features_tested": int(len(features)),
        "n_permutations": int(args.n_perm),
        "best_by_target_transform": clean_json(best_by_target),
        "best_pre_event_clock_features": clean_json(best_pre_clock.to_dict("records")),
        "best_pre_vs_post_clock_asymmetry": clean_json(best_asymmetry.to_dict("records")),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Temporal directionality uses automatic pre-event ROI crops, weak event-relative bins, and within-source clock permutations. AUC rows remain descriptive readouts. This tests whether optical/front/rollout proxies are ordered around event time; it does not validate causal precursors, particle identity, calibrated phase boundaries, or diffusion coefficients.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text(
        "# Source-Balanced Pre-Event Directionality Audit\n\n"
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}\n"
        f"- Event-bin counts: {summary['event_relative_bin_counts']}\n"
        f"- Features tested: {summary['n_features_tested']}\n"
        f"- Within-source permutations: {summary['n_permutations']}\n\n"
        "## Guardrail\n\n"
        f"{summary['guardrail']}\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
