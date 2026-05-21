#!/usr/bin/env python3
"""Rank ROI/manual-QC candidates using cycle-level precursor evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PRECURSOR_WINDOWS = {"pre16_to_pre9", "pre8_to_pre5", "pre4_to_pre1"}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
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
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def robust_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        sd = x.std()
        if not np.isfinite(sd) or sd == 0:
            return x * 0
        return (x - x.mean()) / sd
    return 0.6745 * (x - med) / mad


def minmax(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = x.min()
    hi = x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return x * 0
    return (x - lo) / (hi - lo)


def build_event_precursor_scores(derived: Path, max_tests: int) -> pd.DataFrame:
    tests = read_csv(derived / "particle_event_precursor_atlas" / "particle_event_precursor_window_tests.csv")
    features = read_csv(derived / "particle_event_precursor_atlas" / "particle_event_precursor_window_features.csv")
    if tests.empty or features.empty:
        return pd.DataFrame()

    tests = tests[tests["window"].isin(PRECURSOR_WINDOWS)].copy()
    tests = tests[pd.to_numeric(tests["mannwhitney_p"], errors="coerce").lt(0.05)].copy()
    tests = tests.sort_values("mannwhitney_p").head(max_tests)
    rows = []
    for _, test in tests.iterrows():
        sub = features[
            features["anchor_type"].eq("event")
            & features["window"].eq(test["window"])
            & features["feature"].eq(test["feature"])
        ].copy()
        if sub.empty or test["statistic"] not in sub.columns:
            continue
        control_median = float(test["control_median"])
        direction = 1.0 if float(test["event_minus_control_median"]) >= 0 else -1.0
        p = max(float(test["mannwhitney_p"]), 1e-12)
        weight = -np.log10(p)
        for _, r in sub.iterrows():
            val = pd.to_numeric(pd.Series([r[test["statistic"]]]), errors="coerce").iloc[0]
            if not np.isfinite(val):
                continue
            rows.append({
                "event_reference_cycle": float(r["anchor_cycle"]),
                "anchor_id": r["anchor_id"],
                "window": test["window"],
                "feature": test["feature"],
                "statistic": test["statistic"],
                "value": float(val),
                "control_median": control_median,
                "direction": direction,
                "mannwhitney_p": float(test["mannwhitney_p"]),
                "weight": float(weight),
                "signed_severity": float(direction * (float(val) - control_median) * weight),
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame()
    agg = (
        detail.groupby("event_reference_cycle")
        .agg(
            precursor_tests_used=("signed_severity", "count"),
            precursor_signed_severity_mean=("signed_severity", "mean"),
            precursor_signed_severity_sum=("signed_severity", "sum"),
            precursor_min_p=("mannwhitney_p", "min"),
        )
        .reset_index()
    )
    top_terms = (
        detail.sort_values(["event_reference_cycle", "mannwhitney_p"])
        .groupby("event_reference_cycle")
        .head(4)
        .assign(term=lambda d: d["window"].astype(str) + ":" + d["feature"].astype(str) + ":" + d["statistic"].astype(str))
        .groupby("event_reference_cycle")["term"]
        .apply(lambda x: ";".join(x))
        .reset_index(name="top_precursor_terms")
    )
    return agg.merge(top_terms, on="event_reference_cycle", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/precursor_informed_roi_review")
    parser.add_argument("--max-precursor-tests", type=int, default=10)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manual = read_csv(derived / "manual_qc_label_workbook" / "manual_qc_label_template.csv")
    if manual.empty:
        raise FileNotFoundError("manual_qc_label_template.csv is required")
    event_prec = build_event_precursor_scores(derived, args.max_precursor_tests)
    modes = read_csv(derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv")
    fronts = read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")
    kinetics = read_csv(derived / "phase_kinetics_avrami" / "phase_kinetics_avrami_roi_table.csv")
    graph = read_csv(derived / "spatiotemporal_degradation_graph" / "spatiotemporal_graph_nodes.csv")

    df = manual.copy()
    for col in ["cycleNo", "event_reference_cycle", "combined_review_priority_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if not event_prec.empty:
        df = df.merge(event_prec, on="event_reference_cycle", how="left")
    else:
        df["precursor_signed_severity_mean"] = np.nan
        df["top_precursor_terms"] = ""

    keep_mode = [c for c in ["roi_id", "mode_label", "mode_review_priority", "is_event_enriched_mode", "rollout_mobility_difficulty_score"] if c in modes.columns]
    if keep_mode:
        df = df.merge(modes[keep_mode].drop_duplicates("roi_id"), on="roi_id", how="left", suffixes=("", "_mode"))
    keep_front = [c for c in ["roi_id", "phase_slope_positive_fraction_protocol_residual", "threshold_robust_phase_score_protocol_residual", "diffusion_proxy_abs_median_um2_per_s_protocol_residual"] if c in fronts.columns]
    if keep_front:
        df = df.merge(fronts[keep_front].drop_duplicates("roi_id"), on="roi_id", how="left", suffixes=("", "_front"))
    keep_kin = [c for c in ["roi_id", "q70_fraction_delta", "q70_logistic_k_per_s", "q70_avrami_r2", "roi_norm_max_abs_rate_per_s", "roi_norm_rate_sign_consistency"] if c in kinetics.columns]
    if keep_kin:
        df = df.merge(kinetics[keep_kin].drop_duplicates("roi_id"), on="roi_id", how="left")
    keep_graph = [c for c in ["roi_id", "xy_region", "front_positive_residual_binary", "is_event_enriched_mode"] if c in graph.columns]
    if keep_graph:
        df = df.merge(graph[keep_graph].drop_duplicates("roi_id"), on="roi_id", how="left", suffixes=("", "_graph"))

    df["cycle_offset_from_event_reference"] = pd.to_numeric(df["cycleNo"], errors="coerce") - pd.to_numeric(df["event_reference_cycle"], errors="coerce")
    df["is_pre_event_cycle_window"] = df["cycle_offset_from_event_reference"].between(-16, -1, inclusive="both")
    df["is_event_cycle"] = df["cycle_offset_from_event_reference"].eq(0)
    df["is_post_event_cycle_window"] = df["cycle_offset_from_event_reference"].between(1, 8, inclusive="both")

    components: Dict[str, pd.Series] = {}
    components["manual_priority"] = minmax(df.get("combined_review_priority_score", pd.Series(index=df.index, dtype=float)))
    components["precursor_context"] = minmax(df.get("precursor_signed_severity_mean", pd.Series(index=df.index, dtype=float)).fillna(0))
    components["mode_priority"] = minmax(df.get("mode_review_priority", pd.Series(index=df.index, dtype=float)).fillna(df.get("mode_review_priority", pd.Series(dtype=float)).median() if "mode_review_priority" in df else 0))
    components["front_direction_abs"] = minmax(pd.to_numeric(df.get("phase_slope_positive_fraction_protocol_residual", pd.Series(index=df.index, dtype=float)), errors="coerce").abs())
    components["front_phase_score_abs"] = minmax(pd.to_numeric(df.get("threshold_robust_phase_score_protocol_residual", pd.Series(index=df.index, dtype=float)), errors="coerce").abs())
    components["kinetic_rate_abs"] = minmax(pd.to_numeric(df.get("roi_norm_max_abs_rate_per_s", pd.Series(index=df.index, dtype=float)), errors="coerce").abs())
    components["event_cycle_bonus"] = df["is_event_cycle"].astype(float)
    components["pre_event_bonus"] = df["is_pre_event_cycle_window"].astype(float) * 0.5

    weights = {
        "manual_priority": 2.0,
        "precursor_context": 1.4,
        "mode_priority": 1.2,
        "front_direction_abs": 1.0,
        "front_phase_score_abs": 0.8,
        "kinetic_rate_abs": 0.6,
        "event_cycle_bonus": 0.8,
        "pre_event_bonus": 0.5,
    }
    score = pd.Series(0.0, index=df.index)
    for name, comp in components.items():
        df[f"score_component_{name}"] = comp.fillna(0)
        score = score + weights[name] * comp.fillna(0)
    df["precursor_informed_review_score"] = score

    reasons: List[str] = []
    for _, row in df.iterrows():
        parts = []
        if row.get("is_event_cycle"):
            parts.append("event-cycle ROI")
        if row.get("is_pre_event_cycle_window"):
            parts.append("pre-event-window ROI")
        if pd.notna(row.get("precursor_signed_severity_mean")):
            parts.append("high precursor-context cycle")
        if str(row.get("mode_label", "")).startswith("optical_brightening"):
            parts.append("event-enriched residual mode")
        if abs(float(row.get("phase_slope_positive_fraction_protocol_residual", 0) or 0)) > 0.08:
            parts.append("large front-direction residual")
        if pd.notna(row.get("q70_avrami_r2")):
            parts.append("kinetic proxy available")
        reasons.append(";".join(parts) if parts else "baseline manual-QC candidate")
    df["precursor_review_reason"] = reasons
    df["precursor_review_tier"] = pd.cut(
        df["precursor_informed_review_score"].rank(method="first", ascending=False),
        bins=[0, 12, 30, np.inf],
        labels=["high", "medium", "routine"],
        include_lowest=True,
    ).astype(str)
    df = df.sort_values("precursor_informed_review_score", ascending=False)

    paths = {
        "ranked_manifest": out / "precursor_informed_roi_review_manifest.csv",
        "event_precursor_scores": out / "event_precursor_context_scores.csv",
        "summary": out / "precursor_informed_roi_review_summary.json",
    }
    df.to_csv(paths["ranked_manifest"], index=False)
    event_prec.to_csv(paths["event_precursor_scores"], index=False)

    tier_counts = df["precursor_review_tier"].value_counts().to_dict()
    top_rows = df.head(15).to_dict("records")
    summary = {
        "n_review_candidates": int(len(df)),
        "n_event_candidates": int(df["cohort_role"].eq("event").sum()) if "cohort_role" in df.columns else None,
        "n_control_candidates": int(df["cohort_role"].eq("control").sum()) if "cohort_role" in df.columns else None,
        "precursor_event_cycles_scored": event_prec.to_dict("records") if not event_prec.empty else [],
        "precursor_review_tier_counts": tier_counts,
        "top_precursor_informed_candidates": top_rows,
        "score_weights": weights,
        "guardrail": "This is a review-prioritization manifest. It combines automatic ROI/front/mode/kinetic descriptors with cycle-level precursor context to decide what to inspect first; it does not assign manual QC labels or validate diffusion/front claims.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Precursor-Informed ROI Review",
        "",
        "Ranked manual-QC candidate manifest combining ROI physics, front proxies, kinetics, graph context, and cycle-level precursor severity.",
        "",
        f"- Review candidates: {summary['n_review_candidates']}",
        f"- Event/control candidates: {summary['n_event_candidates']} / {summary['n_control_candidates']}",
        f"- Tier counts: {tier_counts}",
        "",
        "## Top Candidates",
    ]
    for row in top_rows[:10]:
        lines.append(
            f"- {row.get('roi_id')} ({row.get('cohort_role')}, cycle {row.get('cycleNo')}): score={row.get('precursor_informed_review_score'):.3f}, tier={row.get('precursor_review_tier')}, reason={row.get('precursor_review_reason')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "out_dir": str(out),
        "n_review_candidates": summary["n_review_candidates"],
        "tier_counts": tier_counts,
        "top_candidates": [
            {
                "roi_id": r.get("roi_id"),
                "score": r.get("precursor_informed_review_score"),
                "tier": r.get("precursor_review_tier"),
            }
            for r in top_rows[:5]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
