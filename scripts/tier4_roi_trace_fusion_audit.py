#!/usr/bin/env python3
"""Fuse cycle-level particle trace state onto ROI/front degradation descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


TRACE_FEATURES = [
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
    "future_any_drop_within_8cycles",
    "future_sync2_drop_within_8cycles",
    "drop_count",
    "any_abrupt_drop",
    "synchronized_drop_2plus",
]

TARGETS = [
    "is_event_roi",
    "is_event_enriched_mode",
    "phase_slope_positive_fraction_protocol_residual",
    "phase_slope_median_per_s_protocol_residual",
    "threshold_robust_phase_score_protocol_residual",
    "diffusion_proxy_median_um2_per_s_protocol_residual",
    "q60_logistic_k_per_s",
    "q70_logistic_k_per_s",
    "q70_transformed_fraction_delta",
    "q80_transformed_fraction_delta",
    "mode_review_priority",
]

CONTEXT = ["cycleNo", "n_frames_percentile", "V_mean", "I_mean_mA"]


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
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return x * 0
    return (x - x.mean()) / sd


def residualize(df: pd.DataFrame, value_col: str, context_cols: Iterable[str]) -> pd.Series:
    cols = [c for c in context_cols if c in df.columns]
    work_cols = [value_col] + cols
    work = df[work_cols].apply(pd.to_numeric, errors="coerce")
    valid = work.notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if int(valid.sum()) < max(6, len(cols) + 3):
        return out
    y = work.loc[valid, value_col].to_numpy(dtype=float)
    if len(cols):
        x = np.column_stack([np.ones(len(y))] + [zscore(work.loc[valid, c]).to_numpy(dtype=float) for c in cols])
    else:
        x = np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    out.loc[valid] = y - x @ coef
    return out


def add_trace_lags(roi: pd.DataFrame, trace: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    out = roi.copy()
    trace = trace.copy()
    trace["cycleNo"] = pd.to_numeric(trace["cycleNo"], errors="coerce")
    available = [c for c in TRACE_FEATURES if c in trace.columns]
    keyed = trace.set_index("cycleNo")
    for lag in lags:
        lookup_cycle = pd.to_numeric(out["cycleNo"], errors="coerce") - float(lag)
        for col in available:
            out[f"trace_lag{lag}_{col}"] = lookup_cycle.map(keyed[col])
    return out


def add_prediction_lags(roi: pd.DataFrame, pred: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    out = roi.copy()
    if pred.empty:
        return out
    p = pred.pivot_table(index="cycleNo", columns="target", values="predicted_probability", aggfunc="mean")
    p.columns = [f"trace_predprob_{c}" for c in p.columns]
    for lag in lags:
        lookup_cycle = pd.to_numeric(out["cycleNo"], errors="coerce") - float(lag)
        for col in p.columns:
            out[f"lag{lag}_{col}"] = lookup_cycle.map(p[col])
    return out


def spearman_table(df: pd.DataFrame, predictors: List[str], targets: List[str], context_cols: List[str]) -> pd.DataFrame:
    rows = []
    for pred in predictors:
        for target in targets:
            if pred not in df.columns or target not in df.columns:
                continue
            sub = df[[pred, target] + [c for c in context_cols if c in df.columns]].copy()
            raw = sub[[pred, target]].apply(pd.to_numeric, errors="coerce")
            valid = raw.notna().all(axis=1)
            if int(valid.sum()) >= 8 and raw.loc[valid, pred].nunique() > 1 and raw.loc[valid, target].nunique() > 1:
                rho, p = spearmanr(raw.loc[valid, pred], raw.loc[valid, target])
                rows.append({
                    "predictor": pred,
                    "target": target,
                    "association_type": "raw",
                    "n": int(valid.sum()),
                    "rho": float(rho),
                    "p_value": float(p),
                })
            pred_resid = residualize(sub, pred, context_cols)
            target_resid = residualize(sub, target, context_cols)
            rvalid = pred_resid.notna() & target_resid.notna()
            if int(rvalid.sum()) >= 8 and pred_resid.loc[rvalid].nunique() > 1 and target_resid.loc[rvalid].nunique() > 1:
                rho, p = spearmanr(pred_resid.loc[rvalid], target_resid.loc[rvalid])
                rows.append({
                    "predictor": pred,
                    "target": target,
                    "association_type": "context_residual",
                    "n": int(rvalid.sum()),
                    "rho": float(rho),
                    "p_value": float(p),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["p_value", "association_type", "predictor", "target"]).reset_index(drop=True)


def binary_tests(df: pd.DataFrame, predictors: List[str], labels: List[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        if label not in df.columns:
            continue
        y = pd.to_numeric(df[label], errors="coerce")
        for pred in predictors:
            if pred not in df.columns:
                continue
            x = pd.to_numeric(df[pred], errors="coerce")
            pos = x[y.eq(1)].dropna().to_numpy(dtype=float)
            neg = x[y.eq(0)].dropna().to_numpy(dtype=float)
            if len(pos) >= 3 and len(neg) >= 3:
                _, p = mannwhitneyu(pos, neg, alternative="two-sided")
                rows.append({
                    "label": label,
                    "predictor": pred,
                    "n_positive": int(len(pos)),
                    "n_negative": int(len(neg)),
                    "positive_median": float(np.median(pos)),
                    "negative_median": float(np.median(neg)),
                    "positive_minus_negative_median": float(np.median(pos) - np.median(neg)),
                    "mannwhitney_p": float(p),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["mannwhitney_p", "label", "predictor"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_trace_fusion_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mode = read_csv(derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv")
    front = read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")
    kinetics = read_csv(derived / "phase_kinetics_avrami" / "phase_kinetics_avrami_roi_table.csv")
    trace = read_csv(derived / "particle_trace_physics_audit" / "particle_trace_cycle_features.csv")
    pred = read_csv(derived / "particle_trace_physics_audit" / "particle_trace_future_drop_classifier_predictions.csv")

    keep_front = ["roi_id"] + [c for c in front.columns if c != "roi_id" and c.endswith("_protocol_residual")]
    keep_kinetics = ["roi_id"] + [c for c in kinetics.columns if c != "roi_id" and c not in mode.columns]
    joined = mode.merge(front[keep_front], on="roi_id", how="left", suffixes=("", "_frontdup"))
    joined = joined.merge(kinetics[keep_kinetics], on="roi_id", how="left", suffixes=("", "_kindup"))
    joined = add_trace_lags(joined, trace, lags=[0, 2, 4, 8, 16])
    joined = add_prediction_lags(joined, pred, lags=[0, 2, 4, 8, 16])

    predictors = [
        c
        for c in joined.columns
        if c.startswith("trace_lag") or c.startswith("lag") and "trace_predprob" in c
    ]
    tests = spearman_table(joined, predictors, [t for t in TARGETS if t in joined.columns], CONTEXT)
    btests = binary_tests(joined, predictors, ["is_event_roi", "is_event_enriched_mode"])

    paths = {
        "joined": out / "roi_trace_fusion_joined.csv",
        "spearman_tests": out / "roi_trace_fusion_spearman_tests.csv",
        "binary_tests": out / "roi_trace_fusion_binary_tests.csv",
        "summary": out / "roi_trace_fusion_audit_summary.json",
    }
    joined.to_csv(paths["joined"], index=False)
    tests.to_csv(paths["spearman_tests"], index=False)
    btests.to_csv(paths["binary_tests"], index=False)

    binary_targets = {"is_event_roi", "is_event_enriched_mode"}
    noncausal_terms = ("trace_lag0_", "lag0_trace_predprob", "any_abrupt_drop", "drop_count", "synchronized_drop")
    top_raw = tests[tests["association_type"].eq("raw")].head(20).to_dict("records") if not tests.empty else []
    top_resid = tests[tests["association_type"].eq("context_residual")].head(20).to_dict("records") if not tests.empty else []
    focus_resid = pd.DataFrame()
    if not tests.empty:
        focus_resid = tests[
            tests["association_type"].eq("context_residual")
            & ~tests["target"].isin(binary_targets)
            & ~tests["predictor"].map(lambda x: any(term in str(x) for term in noncausal_terms))
        ].copy()
    focus_binary = pd.DataFrame()
    if not btests.empty:
        focus_binary = btests[
            btests["label"].eq("is_event_enriched_mode")
            & ~btests["predictor"].map(lambda x: any(term in str(x) for term in noncausal_terms))
        ].copy()
    top_focus_resid = focus_resid.head(20).to_dict("records") if not focus_resid.empty else []
    top_binary = btests.head(20).to_dict("records") if not btests.empty else []
    top_focus_binary = focus_binary.head(20).to_dict("records") if not focus_binary.empty else []
    summary = {
        "n_roi_rows": int(len(joined)),
        "n_event_roi": int(pd.to_numeric(joined.get("is_event_roi", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "n_event_enriched_mode": int(pd.to_numeric(joined.get("is_event_enriched_mode", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "n_predictors": int(len(predictors)),
        "context_residualized_against": [c for c in CONTEXT if c in joined.columns],
        "top_raw_spearman": top_raw,
        "top_context_residual_spearman": top_resid,
        "top_precursor_context_residual_spearman": top_focus_resid,
        "top_binary_tests": top_binary,
        "top_precursor_event_enriched_mode_tests": top_focus_binary,
        "guardrail": "Trace lags are cycle-level four-particle/echem summaries attached to selected ROI rows by cycle number. Associations are useful for linking global precursor state to ROI/front outcomes, but rows are not independent within cycle and this does not prove localized causality.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# ROI Trace Fusion Audit",
        "",
        "Cycle-level four-particle trace state joined onto ROI/front/kinetic residual descriptors.",
        "",
        f"- ROI rows: {summary['n_roi_rows']}",
        f"- Event ROI rows: {summary['n_event_roi']}",
        f"- Event-enriched mode rows: {summary['n_event_enriched_mode']}",
        f"- Trace lag predictors: {summary['n_predictors']}",
        "",
        "## Top Precursor Context-Residual Associations",
    ]
    for row in top_focus_resid[:10]:
        lines.append(
            f"- {row.get('predictor')} vs {row.get('target')}: rho={row.get('rho'):.3g}, p={row.get('p_value'):.3g}, n={row.get('n')}"
        )
    lines += ["", "## Top Event-Enriched Mode Precursor Tests"]
    for row in top_focus_binary[:10]:
        lines.append(
            f"- {row.get('label')} {row.get('predictor')}: median diff={row.get('positive_minus_negative_median'):.3g}, p={row.get('mannwhitney_p'):.3g}"
        )
    lines += ["", "## Full Audit Note", "", "Full CSV tables also include lag-0/event-label checks, which are useful sanity checks but not interpreted as precursor evidence."]
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(clean_json({
        "out_dir": str(out),
        "n_roi_rows": summary["n_roi_rows"],
        "top_precursor_context_residual_spearman": top_focus_resid[:5],
        "top_precursor_event_enriched_mode_tests": top_focus_binary[:5],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
