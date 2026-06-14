#!/usr/bin/env python3
"""Cycle-collapsed null audit for ROI trace-fusion associations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


DEFAULT_TARGETS = [
    "phase_slope_positive_fraction_protocol_residual",
    "phase_slope_median_per_s_protocol_residual",
    "threshold_robust_phase_score_protocol_residual",
    "q70_logistic_k_per_s",
    "q70_transformed_fraction_delta",
    "is_event_enriched_mode",
    "mode_review_priority",
]


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


def rank_corr(rx: np.ndarray, ry: np.ndarray) -> float:
    x = rx - rx.mean()
    y = ry - ry.mean()
    denom = np.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom == 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, rng: np.random.Generator, n_perm: int) -> Dict[str, float]:
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    vals = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        vals[i] = rank_corr(rx, rng.permutation(ry))
    arr = vals[np.isfinite(vals)]
    if arr.size == 0:
        return {"empirical_p_abs_ge_observed": np.nan, "null_p95_abs_rho": np.nan, "null_mean_abs_rho": np.nan}
    return {
        "empirical_p_abs_ge_observed": float((np.sum(np.abs(arr) >= abs(observed)) + 1) / (arr.size + 1)),
        "null_p95_abs_rho": float(np.quantile(np.abs(arr), 0.95)),
        "null_mean_abs_rho": float(np.mean(np.abs(arr))),
    }


def spearman_rows(df: pd.DataFrame, predictors: List[str], targets: List[str], n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pred in predictors:
        for target in targets:
            if pred not in df.columns or target not in df.columns:
                continue
            sub = df[[pred, target]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 7 or sub[pred].nunique() < 3 or sub[target].nunique() < 3:
                continue
            x = sub[pred].to_numpy(dtype=float)
            y = sub[target].to_numpy(dtype=float)
            rho, asym_p = spearmanr(x, y)
            if not np.isfinite(rho):
                continue
            null = permutation_p(x, y, float(rho), rng, n_perm)
            rows.append({
                "predictor": pred,
                "target": target,
                "n_cycle_points": int(len(sub)),
                "rho": float(rho),
                "asymptotic_p": float(asym_p),
                **null,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["empirical_p_abs_ge_observed", "asymptotic_p", "predictor", "target"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/roi_trace_fusion_cycle_null")
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--max-predictors", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined_path = derived / "roi_trace_fusion_audit" / "roi_trace_fusion_joined.csv"
    if not joined_path.exists():
        raise FileNotFoundError(f"Run tier4_roi_trace_fusion_audit.py first; missing {joined_path}")
    joined = pd.read_csv(joined_path)
    fusion_summary_path = derived / "roi_trace_fusion_audit" / "roi_trace_fusion_audit_summary.json"
    focus_predictors: List[str] = []
    if fusion_summary_path.exists():
        with fusion_summary_path.open() as f:
            fusion_summary = json.load(f)
        for key in ["top_precursor_context_residual_spearman", "top_precursor_event_enriched_mode_tests"]:
            for row in fusion_summary.get(key, [])[:10]:
                pred = row.get("predictor")
                if pred and pred in joined.columns and pred not in focus_predictors:
                    focus_predictors.append(pred)
    core_terms = [
        "particle_norm_mean",
        "particle_norm_std",
        "particle_norm_range",
        "particle_norm_cv",
        "mean_delta_prev",
        "mean_abs_delta_prev",
        "max_abs_delta_prev",
        "delta_std_across_particles",
        "capacity_mAh",
        "coulombic_efficiency_pct",
        "V_max",
        "n_frames",
        "frames_percentile",
    ]
    for lag in [2, 4, 8, 16]:
        for term in core_terms:
            col = f"trace_lag{lag}_{term}"
            if col in joined.columns and col not in focus_predictors:
                focus_predictors.append(col)
    predictors = [
        c for c in focus_predictors
        if "lag0" not in c and "any_abrupt_drop" not in c and "drop_count" not in c and "synchronized_drop" not in c
    ]
    predictors = predictors[: args.max_predictors]
    targets = [c for c in DEFAULT_TARGETS if c in joined.columns]
    numeric_cols = ["cycleNo", "event_reference_cycle", "n_frames_percentile", "V_mean", "I_mean_mA"] + predictors + targets
    numeric_cols = [c for c in numeric_cols if c in joined.columns]
    work = joined[numeric_cols].copy()
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    cycle = (
        work.groupby("cycleNo", dropna=False)
        .median(numeric_only=True)
        .reset_index()
    )
    ref_map = work.groupby("cycleNo", dropna=False)["event_reference_cycle"].median()
    cycle["event_reference_cycle"] = cycle["cycleNo"].map(ref_map)

    centered = cycle.copy()
    center_cols = [c for c in predictors + targets if c in centered.columns]
    centered[center_cols] = centered[center_cols] - centered.groupby("event_reference_cycle")[center_cols].transform("mean")

    raw_tests = spearman_rows(cycle, predictors, targets, args.n_permutation, args.seed)
    centered_tests = spearman_rows(centered, predictors, targets, args.n_permutation, args.seed + 1)

    paths = {
        "cycle_collapsed_table": out / "roi_trace_cycle_collapsed_table.csv",
        "reference_centered_table": out / "roi_trace_reference_centered_table.csv",
        "cycle_spearman_tests": out / "roi_trace_cycle_spearman_tests.csv",
        "reference_centered_tests": out / "roi_trace_reference_centered_spearman_tests.csv",
        "summary": out / "roi_trace_fusion_cycle_null_summary.json",
    }
    cycle.to_csv(paths["cycle_collapsed_table"], index=False)
    centered.to_csv(paths["reference_centered_table"], index=False)
    raw_tests.to_csv(paths["cycle_spearman_tests"], index=False)
    centered_tests.to_csv(paths["reference_centered_tests"], index=False)

    robust_raw = raw_tests[raw_tests["empirical_p_abs_ge_observed"].lt(0.05)].to_dict("records") if not raw_tests.empty else []
    robust_centered = centered_tests[centered_tests["empirical_p_abs_ge_observed"].lt(0.05)].to_dict("records") if not centered_tests.empty else []
    summary = {
        "n_roi_rows": int(len(joined)),
        "n_cycle_points": int(cycle["cycleNo"].nunique()),
        "n_event_reference_cycles": int(cycle["event_reference_cycle"].nunique()),
        "n_predictors_tested": int(len(predictors)),
        "targets": targets,
        "n_permutation": int(args.n_permutation),
        "top_cycle_collapsed_tests": raw_tests.head(20).to_dict("records") if not raw_tests.empty else [],
        "top_reference_centered_tests": centered_tests.head(20).to_dict("records") if not centered_tests.empty else [],
        "robust_cycle_collapsed_tests": robust_raw,
        "robust_reference_centered_tests": robust_centered,
        "guardrail": "This audit collapses repeated ROI rows to one median point per cycle before testing trace/front associations. It is deliberately conservative for the 52-ROI cohort; surviving tests are stronger evidence, while lost tests indicate cycle-clustering sensitivity.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# ROI Trace Fusion Cycle Null",
        "",
        "Cycle-collapsed permutation audit for lagged trace/front associations.",
        "",
        f"- ROI rows collapsed: {summary['n_roi_rows']} -> {summary['n_cycle_points']} cycle points",
        f"- Event-reference cycles: {summary['n_event_reference_cycles']}",
        f"- Predictors tested: {summary['n_predictors_tested']}",
        f"- Permutations per test: {summary['n_permutation']}",
        "",
        "## Top Cycle-Collapsed Tests",
    ]
    for row in summary["top_cycle_collapsed_tests"][:10]:
        lines.append(
            f"- {row.get('predictor')} vs {row.get('target')}: rho={row.get('rho'):.3g}, empirical p={row.get('empirical_p_abs_ge_observed'):.3g}, n={row.get('n_cycle_points')}"
        )
    lines += ["", "## Top Reference-Centered Tests"]
    for row in summary["top_reference_centered_tests"][:10]:
        lines.append(
            f"- {row.get('predictor')} vs {row.get('target')}: rho={row.get('rho'):.3g}, empirical p={row.get('empirical_p_abs_ge_observed'):.3g}, n={row.get('n_cycle_points')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(clean_json({
        "out_dir": str(out),
        "n_cycle_points": summary["n_cycle_points"],
        "top_cycle_collapsed_tests": summary["top_cycle_collapsed_tests"][:5],
        "top_reference_centered_tests": summary["top_reference_centered_tests"][:5],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
