#!/usr/bin/env python3
"""Event-relative readout for the source-balanced pre-event ROI cohort.

This audit uses the newly exported pre-event/post-event/control particle-region
ROI tensors and their rollout/mask-front feature tables. It asks whether
automatic ROI-only optical, front, mask, and rollout proxies separate clean
pre-event windows from post-event washout and no-near-event controls after
basic source residualization.
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
CONTROL_BINS = {"post_event_1_16", "no_near_event_control"}
TARGETS = [
    "clean_pre_1_8_vs_post_control",
    "clean_pre_1_16_vs_post_control",
    "clean_pre_1_32_vs_post_control",
    "near_pre_vs_far_pre",
    "post_event_vs_control",
]
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
    means = x.groupby(src).transform("mean")
    if transform == "raw":
        return x
    if transform == "source_residual":
        return x - means
    if transform == "within_source_rank":
        return x.groupby(src).rank(pct=True) - 0.5
    raise ValueError(transform)


def oriented_metrics(y: pd.Series, x: pd.Series) -> Tuple[float, float, str, float, float, float, float, float]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
        return np.nan, np.nan, "NA", np.nan, np.nan, np.nan, np.nan, np.nan
    direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
    score = xx if direction == "higher_in_positive" else -xx
    auc = float(roc_auc_score(yy, score))
    ap = float(average_precision_score(yy, score))
    rho, sp = spearmanr(yy, score)
    pos = xx[yy == 1]
    neg = xx[yy == 0]
    try:
        _, p_mwu = mannwhitneyu(pos, neg, alternative="two-sided")
    except ValueError:
        p_mwu = np.nan
    return auc, ap, direction, float(rho), float(sp), float(p_mwu), float(pos.median()), float(neg.median())


def add_event_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = out["event_relative_bin"].astype(str)
    out["clean_pre_1_8_vs_post_control"] = np.where(
        bins.eq("near_pre_event_1_8"), 1, np.where(bins.isin(CONTROL_BINS), 0, np.nan)
    )
    out["clean_pre_1_16_vs_post_control"] = np.where(
        bins.isin({"near_pre_event_1_8", "mid_pre_event_9_16"}), 1, np.where(bins.isin(CONTROL_BINS), 0, np.nan)
    )
    out["clean_pre_1_32_vs_post_control"] = np.where(
        bins.isin(PRE_BINS), 1, np.where(bins.isin(CONTROL_BINS), 0, np.nan)
    )
    out["near_pre_vs_far_pre"] = np.where(
        bins.eq("near_pre_event_1_8"), 1, np.where(bins.eq("far_pre_event_17_32"), 0, np.nan)
    )
    out["post_event_vs_control"] = np.where(
        bins.eq("post_event_1_16"), 1, np.where(bins.eq("no_near_event_control"), 0, np.nan)
    )
    return out


def merge_features(rollout: pd.DataFrame, mask_front: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in mask_front.columns if c in rollout.columns and c != "roi_id"]
    return rollout.merge(mask_front.drop(columns=drop_cols), on="roi_id", how="left")


def feature_columns(df: pd.DataFrame) -> List[str]:
    blocked = {
        "roi_id", "npz_path", "preview_png", "source_stem", "event_relative_bin",
        "selection_reason", "validation_label", "cohort_role",
        "future_any_drop_within_8cycles", "future_any_drop_within_16cycles",
        "future_event_within_32cycles", "past_event_within_16cycles", "any_abrupt_drop",
    } | set(TARGETS)
    cols = []
    keys = [
        "roi_norm_mean", "raw_roi_mean", "temporal_energy", "persistence", "velocity",
        "masked_minus_background", "front_", "mask_", "apparent_diffusion", "object_",
    ]
    for col in df.columns:
        if col in blocked:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 16 and vals.nunique(dropna=True) >= 2 and any(k in col for k in keys):
            cols.append(col)
    return cols


def audit(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    for target in TARGETS:
        y = numeric(df, target)
        for feature in features:
            raw = numeric(df, feature)
            for transform in TRANSFORMS:
                x = transform_feature(raw, sources, transform)
                auc, ap, direction, rho, sp, p_mwu, med_pos, med_neg = oriented_metrics(y, x)
                valid = y.isin([0, 1]) & x.notna()
                rows.append({
                    "target": target,
                    "feature": feature,
                    "transform": transform,
                    "n": int(valid.sum()),
                    "n_positive": int(y[valid].sum()) if valid.any() else 0,
                    "direction": direction,
                    "oriented_auc": auc,
                    "average_precision": ap,
                    "spearman_rho": rho,
                    "spearman_p": sp,
                    "mwu_p": p_mwu,
                    "median_positive": med_pos,
                    "median_negative": med_neg,
                    "median_positive_minus_negative": med_pos - med_neg if np.isfinite(med_pos) and np.isfinite(med_neg) else np.nan,
                    "source_eta2_after_transform": source_eta2(x, sources),
                    "raw_source_eta2": source_eta2(raw, sources),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "transform", "oriented_auc", "average_precision"], ascending=[True, True, False, False])
    return out


def top_by_group(df: pd.DataFrame, group_cols: List[str], n: int = 1) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    rows = []
    for _, sub in df.groupby(group_cols, dropna=False):
        rows.extend(sub.sort_values(["oriented_auc", "average_precision"], ascending=False).head(n).to_dict("records"))
    return rows


def summarize_bins(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    agg = {
        "roi_id": "count",
        "cycleNo": pd.Series.nunique,
        "source_stem": pd.Series.nunique,
        "future_any_drop_within_8cycles": "sum",
        "future_any_drop_within_16cycles": "sum",
    }
    for feature in features[:16]:
        agg[feature] = "median"
    out = df.groupby("event_relative_bin", dropna=False).agg(agg).reset_index()
    out = out.rename(columns={"roi_id": "n_roi", "cycleNo": "n_cycles", "source_stem": "n_sources"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_sequence_rollout_audit")
    parser.add_argument("--mask-front-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_mask_front_audit")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_readout_audit")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rollout = pd.read_csv(Path(args.rollout_dir) / "source_balanced_sequence_rollout_features.csv")
    mask_front = pd.read_csv(Path(args.mask_front_dir) / "source_balanced_mask_front_features.csv")
    merged = add_event_targets(merge_features(rollout, mask_front))
    features = feature_columns(merged)
    metrics = audit(merged, features)
    best = top_by_group(metrics, ["target", "transform"], 1)
    bin_summary = summarize_bins(merged, features)

    paths = {
        "merged_features": out / "source_balanced_pre_event_readout_features.csv",
        "event_bin_summary": out / "source_balanced_pre_event_readout_bin_summary.csv",
        "metrics": out / "source_balanced_pre_event_readout_metrics.csv",
        "best_by_target_transform": out / "source_balanced_pre_event_readout_best_by_target_transform.csv",
        "summary": out / "source_balanced_pre_event_readout_summary.json",
    }
    merged.to_csv(paths["merged_features"], index=False)
    bin_summary.to_csv(paths["event_bin_summary"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    pd.DataFrame(best).to_csv(paths["best_by_target_transform"], index=False)

    primary = [
        r for r in best
        if r.get("target") in {"clean_pre_1_8_vs_post_control", "clean_pre_1_16_vs_post_control", "clean_pre_1_32_vs_post_control"}
        and r.get("transform") == "source_residual"
    ]
    primary = sorted(primary, key=lambda r: (r.get("oriented_auc") or -np.inf, r.get("average_precision") or -np.inf), reverse=True)
    summary = {
        "n_rows": int(len(merged)),
        "n_cycles": int(merged["cycleNo"].nunique()),
        "n_sources": int(merged["source_stem"].nunique()),
        "event_relative_bin_counts": clean_json(merged["event_relative_bin"].value_counts().to_dict()),
        "n_features_tested": int(len(features)),
        "transforms": TRANSFORMS,
        "best_by_target_transform": clean_json(best),
        "best_source_residual_clean_pre_readouts": clean_json(primary),
        "event_bin_summary": clean_json(bin_summary.to_dict("records")),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Event-relative readouts use automatic ROI crops and weak event-distance bins. They test pre/post/control organization of optical/front/rollout proxies, not manual particle identity, causal mechanism, calibrated phase boundaries, or diffusion coefficients.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text(
        "# Source-Balanced Pre-Event Readout Audit\n\n"
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}\n"
        f"- Event-bin counts: {summary['event_relative_bin_counts']}\n"
        f"- Features tested: {summary['n_features_tested']}\n\n"
        "## Guardrail\n\n"
        f"{summary['guardrail']}\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
