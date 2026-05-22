#!/usr/bin/env python3
"""Build a consensus manual-QC queue for source-balanced pre-event candidates.

Several pre-event audits now nominate overlapping but not identical ROI crops:
source-invariant review scores, radial-kymograph front tracks, echem-conditioned
residual front proxies, and matched-control tests. This script combines those
signals into one guarded candidate table for manual review. It assigns no labels
and makes no calibrated diffusion or phase-boundary claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


DIRECT_SIGNAL_COLS = [
    "pre_event_review_score",
    "si_clean_physics_prob",
    "si_near_far_physics_prob",
    "front_source_residual_score",
    "apparent_diffusion_score",
    "contrast_slope_score",
    "spatial_clock_score",
    "front_radius_slope_px_per_norm_time",
    "front_radius2_slope_px2_per_norm_time",
    "front_radius_q60_slope_px_per_norm_time",
    "front_radius_q70_slope_px_per_norm_time",
    "apparent_diffusion_q70_um2_per_norm_time",
    "masked_minus_background_mean_slope",
    "mask_centroid_path_px",
    "kymograph_temporal_energy",
    "front_radius_slope_px_per_norm_time_source_echem_context_residual",
    "front_radius2_slope_px2_per_norm_time_source_echem_context_residual",
    "apparent_diffusion_q70_um2_per_norm_time_source_echem_context_residual",
    "masked_minus_background_mean_slope_source_echem_context_residual",
    "kymograph_temporal_energy_source_echem_context_residual",
]

MATCH_FEATURE_TERMS = ("front_", "apparent_diffusion", "masked_minus_background", "mask_", "kymograph")


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


def robust_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad if pd.notna(mad) and mad > 1e-12 else x.std()
    if pd.isna(scale) or scale <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((x - med) / scale).clip(-6, 6).fillna(0.0)


def minmax_rank_score(series: pd.Series, higher_better: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 2 or x.nunique(dropna=True) < 2:
        return pd.Series(0.0, index=series.index)
    rank = x.rank(pct=True)
    if not higher_better:
        rank = 1.0 - rank
    return rank.fillna(0.0)


def merge_inputs(derived: Path) -> pd.DataFrame:
    review = read_csv(
        derived / "source_balanced_pre_event_review_packet" / "source_balanced_pre_event_review_ranked_candidates.csv"
    )
    joined = read_csv(
        derived / "source_balanced_pre_event_echem_front_coupling_audit" / "source_balanced_pre_event_echem_front_joined.csv"
    )
    keep = ["roi_id"] + [c for c in DIRECT_SIGNAL_COLS if c in joined.columns and c not in review.columns]
    out = review.merge(joined[keep], on="roi_id", how="left")
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def add_matched_support(df: pd.DataFrame, derived: Path) -> pd.DataFrame:
    df = df.copy()
    support = pd.DataFrame(index=df["roi_id"].astype(str).values)
    support["matched_positive_support_count"] = 0.0
    support["matched_positive_support_weight"] = 0.0
    support["matched_positive_support_details"] = ""
    roi_to_idx = {str(r): i for i, r in enumerate(df["roi_id"].astype(str))}

    joined = df.set_index("roi_id", drop=False)

    configs = [
        (
            derived / "source_balanced_pre_event_echem_matched_residual_audit" / "source_balanced_pre_event_echem_matched_feature_tests.csv",
            derived / "source_balanced_pre_event_echem_matched_residual_audit" / "source_balanced_pre_event_echem_matched_pairs.csv",
            True,
        ),
        (
            derived / "source_balanced_pre_event_echem_matched_far_control_audit" / "source_balanced_pre_event_echem_matched_far_tests.csv",
            derived / "source_balanced_pre_event_echem_matched_far_control_audit" / "source_balanced_pre_event_echem_matched_far_pairs.csv",
            False,
        ),
    ]

    details: Dict[str, List[str]] = {str(r): [] for r in df["roi_id"].astype(str)}
    for test_path, pair_path, has_comparison in configs:
        if not test_path.exists() or not pair_path.exists():
            continue
        tests = pd.read_csv(test_path)
        pairs = pd.read_csv(pair_path)
        if tests.empty or pairs.empty:
            continue
        p_col = "signflip_mean_abs_p"
        tests[p_col] = pd.to_numeric(tests[p_col], errors="coerce")
        tests["median_treated_minus_control"] = pd.to_numeric(tests["median_treated_minus_control"], errors="coerce")
        feature_col = "base_feature" if "base_feature" in tests.columns else "feature"
        tests = tests[
            (tests[p_col].notna())
            & (tests[p_col] <= 0.05)
            & (tests["median_treated_minus_control"] > 0)
            & tests[feature_col].astype(str).map(lambda x: any(term in x for term in MATCH_FEATURE_TERMS))
        ].copy()
        tests = tests.sort_values([p_col, "median_treated_minus_control"], ascending=[True, False]).head(40)
        for _, test in tests.iterrows():
            feature = str(test[feature_col])
            if feature not in joined.columns:
                continue
            sub = pairs[pairs["match_scheme"].astype(str) == str(test["match_scheme"])].copy()
            if has_comparison and "comparison" in pairs.columns:
                sub = sub[sub["comparison"].astype(str) == str(test["comparison"])]
            if sub.empty:
                continue
            weight = float(-np.log10(max(float(test[p_col]), 1e-6)))
            for _, pair in sub.iterrows():
                tid = str(pair.get("treated_roi_id", ""))
                cid = str(pair.get("control_roi_id", ""))
                if tid not in roi_to_idx or cid not in joined.index:
                    continue
                tv = pd.to_numeric(pd.Series([joined.loc[tid, feature]]), errors="coerce").iloc[0]
                cv = pd.to_numeric(pd.Series([joined.loc[cid, feature]]), errors="coerce").iloc[0]
                if pd.notna(tv) and pd.notna(cv) and tv > cv:
                    support.loc[tid, "matched_positive_support_count"] += 1.0
                    support.loc[tid, "matched_positive_support_weight"] += weight
                    label = f"{test.get('comparison', 'near_vs_far')}|{test['match_scheme']}|{feature}|p={float(test[p_col]):.3g}"
                    details[tid].append(label)
    support["matched_positive_support_details"] = [";".join(details[idx][:8]) for idx in support.index]
    return df.merge(support.reset_index(names="roi_id"), on="roi_id", how="left")


def build_queue(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DIRECT_SIGNAL_COLS:
        if col in out.columns:
            out[f"{col}_rank_score"] = minmax_rank_score(out[col])
    out["matched_positive_support_weight_rank_score"] = minmax_rank_score(out["matched_positive_support_weight"])
    out["matched_positive_support_count_rank_score"] = minmax_rank_score(out["matched_positive_support_count"])
    components = [
        ("pre_event_review_score_rank_score", 1.4),
        ("si_clean_physics_prob_rank_score", 1.0),
        ("si_near_far_physics_prob_rank_score", 1.0),
        ("front_radius2_slope_px2_per_norm_time_rank_score", 1.1),
        ("front_radius_slope_px_per_norm_time_rank_score", 1.0),
        ("front_radius_slope_px_per_norm_time_source_echem_context_residual_rank_score", 1.2),
        ("front_radius2_slope_px2_per_norm_time_source_echem_context_residual_rank_score", 1.2),
        ("apparent_diffusion_q70_um2_per_norm_time_source_echem_context_residual_rank_score", 0.8),
        ("kymograph_temporal_energy_rank_score", 0.6),
        ("matched_positive_support_weight_rank_score", 1.4),
        ("matched_positive_support_count_rank_score", 0.8),
    ]
    score = pd.Series(0.0, index=out.index)
    total_w = 0.0
    for col, weight in components:
        if col in out.columns:
            score += weight * pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            total_w += weight
    out["consensus_review_score"] = score / total_w if total_w else 0.0
    bins = out["event_relative_bin"].astype(str)
    out["review_priority_tier"] = "routine_review"
    out.loc[(bins == "near_pre_event_1_8") & (out["consensus_review_score"] >= out["consensus_review_score"].quantile(0.90)), "review_priority_tier"] = "immediate_near_pre_front_qc"
    out.loc[(out["matched_positive_support_count"] >= 8) & (out["consensus_review_score"] >= out["consensus_review_score"].quantile(0.75)), "review_priority_tier"] = "matched_support_front_qc"
    out.loc[(bins != "near_pre_event_1_8") & (out["consensus_review_score"] >= out["consensus_review_score"].quantile(0.92)), "review_priority_tier"] = "high_scoring_control_qc"
    out["consensus_review_rank"] = out["consensus_review_score"].rank(method="first", ascending=False).astype(int)
    reason_cols = [
        "event_relative_bin",
        "review_reason",
        "matched_positive_support_count",
        "front_radius2_slope_px2_per_norm_time",
        "front_radius_slope_px_per_norm_time_source_echem_context_residual",
        "apparent_diffusion_q70_um2_per_norm_time_source_echem_context_residual",
    ]
    reasons = []
    for _, row in out.iterrows():
        bits = []
        for col in reason_cols:
            if col in out.columns and pd.notna(row.get(col)):
                val = row.get(col)
                if isinstance(val, (int, float, np.floating)):
                    bits.append(f"{col}={float(val):.3g}")
                else:
                    bits.append(f"{col}={val}")
        reasons.append("; ".join(bits))
    out["consensus_review_reason"] = reasons
    return out.sort_values("consensus_review_rank")


def write_readme(out_dir: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Pre-Event Consensus Review Queue",
        "",
        f"- Candidate rows: {summary['n_candidates']}",
        f"- Cycles/sources: {summary['n_cycles']} / {summary['n_sources']}",
        f"- Priority tier counts: {summary['priority_tier_counts']}",
        "",
        "## Top Candidates",
    ]
    for row in summary.get("top_candidates", [])[:10]:
        lines.append(
            f"- rank {row['consensus_review_rank']} {row['roi_id']} {row['event_relative_bin']} "
            f"score={row['consensus_review_score']:.3f}, tier={row['review_priority_tier']}, "
            f"matched_support={row['matched_positive_support_count']:.0f}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument(
        "--out-dir",
        default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_consensus_review_queue",
    )
    args = parser.parse_args()
    derived = Path(args.derived_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merge_inputs(derived)
    df = add_matched_support(df, derived)
    queue = build_queue(df)

    keep = [
        "consensus_review_rank",
        "roi_id",
        "cycleNo",
        "source_stem",
        "event_relative_bin",
        "cycles_to_next_event",
        "review_priority_tier",
        "consensus_review_score",
        "pre_event_review_score",
        "pre_event_review_rank",
        "review_reason",
        "si_clean_physics_prob",
        "si_near_far_physics_prob",
        "front_radius_slope_px_per_norm_time",
        "front_radius2_slope_px2_per_norm_time",
        "front_radius_slope_px_per_norm_time_source_echem_context_residual",
        "front_radius2_slope_px2_per_norm_time_source_echem_context_residual",
        "apparent_diffusion_q70_um2_per_norm_time_source_echem_context_residual",
        "kymograph_temporal_energy",
        "matched_positive_support_count",
        "matched_positive_support_weight",
        "matched_positive_support_details",
        "consensus_review_reason",
        "npz_path",
    ]
    keep = [c for c in keep if c in queue.columns]
    queue_out = queue[keep].copy()

    queue_path = out_dir / "source_balanced_pre_event_consensus_review_queue.csv"
    top_path = out_dir / "source_balanced_pre_event_consensus_top40.csv"
    summary_path = out_dir / "source_balanced_pre_event_consensus_review_summary.json"
    queue_out.to_csv(queue_path, index=False)
    queue_out.head(40).to_csv(top_path, index=False)

    top = queue_out.head(20).to_dict("records")
    summary = {
        "n_candidates": int(len(queue_out)),
        "n_cycles": int(queue_out["cycleNo"].nunique()),
        "n_sources": int(queue_out["source_stem"].nunique()),
        "priority_tier_counts": clean_json(queue_out["review_priority_tier"].value_counts().to_dict()),
        "event_relative_bin_counts": clean_json(queue_out["event_relative_bin"].value_counts().to_dict()),
        "top_candidates": clean_json(top),
        "outputs": {
            "queue": str(queue_path),
            "top40": str(top_path),
            "summary": str(summary_path),
        },
        "guardrail": "This is a manual-QC prioritization queue built from automatic proxy evidence. It assigns no labels and does not validate particle identity, front masks, calibrated diffusion, phase-boundary tracking, degradation causality, or deployable warnings.",
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out_dir, summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
