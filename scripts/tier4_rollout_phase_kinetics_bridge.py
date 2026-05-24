#!/usr/bin/env python3
"""Bridge ROI rollout difficulty to optical phase-kinetics descriptors.

This joins the source-balanced rollout audit with the source-balanced
pre-event phase-kinetics audit on ROI identity and source/cycle context.

Goal:
- identify whether rollout residuals and temporal prediction difficulty track
  phase-fraction kinetics, Avrami/logistic parameters, and local front motion;
- separate raw correlations from source-level confounding by residualizing
  within source.

Guardrail:
- This is an association audit over automatic particle-only crops and weak
  labels. It nominates physics-facing hypotheses for follow-up, but it is not a
  calibrated diffusion estimate or manual phase-boundary annotation.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


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
    except Exception:
        pass
    return value


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def source_residual(series: pd.Series, sources: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    src = sources.astype(str)
    means = x.groupby(src).transform("mean")
    return x - means


def orient_pair(y: pd.Series, x: pd.Series) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    out = {
        "n": int(valid.sum()),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "direction": "NA",
        "oriented_auc": np.nan,
        "average_precision": np.nan,
        "mwu_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "median_positive": np.nan,
        "median_negative": np.nan,
        "median_positive_minus_negative": np.nan,
    }
    if len(yy) >= 8 and yy.nunique() == 2 and xx.nunique() > 1:
        direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
        score = xx if direction == "higher_in_positive" else -xx
        out["direction"] = direction
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            out["oriented_auc"] = float(roc_auc_score(yy, score))
            out["average_precision"] = float(average_precision_score(yy, score))
        except Exception:
            pass
        pos = xx[yy == 1]
        neg = xx[yy == 0]
        try:
            out["mwu_p"] = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
        except Exception:
            pass
        rho, p = spearmanr(yy, score)
        if np.isfinite(rho):
            out["spearman_rho"] = float(rho)
        if np.isfinite(p):
            out["spearman_p"] = float(p)
        out["median_positive"] = float(pos.median())
        out["median_negative"] = float(neg.median())
        out["median_positive_minus_negative"] = out["median_positive"] - out["median_negative"]
    return out


def top_columns(cols: Iterable[str], patterns: List[str]) -> List[str]:
    out = []
    for c in cols:
        if any(re.search(p, c) for p in patterns):
            out.append(c)
    return out


def correlation_table(df: pd.DataFrame, x_cols: List[str], y_cols: List[str], residualize: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    src = df["source_stem"].astype(str)
    for x in x_cols:
        if x not in df.columns:
            continue
        xv = numeric(df, x)
        if residualize:
            xv = source_residual(xv, src)
        for y in y_cols:
            if y not in df.columns:
                continue
            yv = numeric(df, y)
            if residualize:
                yv = source_residual(yv, src)
            valid = xv.notna() & yv.notna()
            if valid.sum() < 8 or xv[valid].nunique() < 3 or yv[valid].nunique() < 3:
                continue
            rho, p = spearmanr(xv[valid], yv[valid])
            rows.append({
                "x": x,
                "y": y,
                "n": int(valid.sum()),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "abs_rho": float(abs(rho)) if np.isfinite(rho) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_rho", "p_value"], ascending=[False, True])
    return out


def event_tests(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if target not in df.columns:
        return pd.DataFrame()
    y = numeric(df, target)
    for feat in features:
        if feat not in df.columns:
            continue
        x = numeric(df, feat)
        valid = y.isin([0, 1]) & x.notna()
        yy = y[valid].astype(int)
        xx = x[valid]
        if len(yy) < 8 or yy.nunique() < 2:
            continue
        pos = xx[yy == 1]
        neg = xx[yy == 0]
        try:
            p = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
        except Exception:
            p = np.nan
        rows.append({
            "target": target,
            "feature": feat,
            "n": int(valid.sum()),
            "median_positive": float(pos.median()),
            "median_negative": float(neg.median()),
            "median_positive_minus_negative": float(pos.median() - neg.median()),
            "mannwhitney_p": p,
            **orient_pair(yy, xx),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["oriented_auc", "mannwhitney_p"], ascending=[False, True])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rollout-csv",
        default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_sequence_rollout_audit/source_balanced_sequence_rollout_features.csv",
    )
    ap.add_argument(
        "--phase-csv",
        default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_phase_kinetics_audit/source_balanced_pre_event_phase_kinetics_features.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/rollout_phase_kinetics_bridge",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout = pd.read_csv(args.rollout_csv)
    phase = pd.read_csv(args.phase_csv)
    merged = rollout.merge(
        phase,
        on=["roi_id", "cycleNo", "source_stem", "local_cycle_index", "expansion_cycle_rank", "object_candidate_rank"],
        how="inner",
        suffixes=("_rollout", "_phase"),
    )

    rollout_feature_cols = top_columns(
        merged.columns,
        [
            r"^persistence_mse_",
            r"^velocity_mse_",
            r"^velocity_minus_persistence_mse$",
            r"^temporal_energy_",
            r"^roi_norm_mean_",
            r"^raw_roi_mean_delta_last_minus_first$",
            r"^stage_drift_xy_recomputed$",
        ],
    )
    phase_feature_cols = top_columns(
        merged.columns,
        [
            r"masked_minus_bg",
            r"phase_fraction",
            r"avrami",
            r"logistic",
            r"max_abs_rate",
            r"total_variation",
            r"rate_sign_consistency",
            r"front_gradient",
        ],
    )
    label_cols = [c for c in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"] if c in merged.columns]

    corr = correlation_table(merged, rollout_feature_cols, phase_feature_cols, residualize=False)
    resid_corr = correlation_table(merged, rollout_feature_cols, phase_feature_cols, residualize=True)
    event_rows = []
    for target in label_cols:
        event_rows.append(event_tests(merged, rollout_feature_cols + phase_feature_cols, target))
    event_tests_df = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()

    source_summary = (
        merged.groupby("source_stem", dropna=False)
        .agg(
            n_roi=("roi_id", "count"),
            n_cycles=("cycleNo", "nunique"),
            future8_rate=("future_any_drop_within_8cycles", "mean") if "future_any_drop_within_8cycles" in merged.columns else ("roi_id", "count"),
            future16_rate=("future_any_drop_within_16cycles", "mean") if "future_any_drop_within_16cycles" in merged.columns else ("roi_id", "count"),
        )
        .reset_index()
    )

    merged.to_csv(out_dir / "rollout_phase_kinetics_bridge_merged.csv", index=False)
    corr.to_csv(out_dir / "rollout_phase_kinetics_bridge_correlations.csv", index=False)
    resid_corr.to_csv(out_dir / "rollout_phase_kinetics_bridge_source_residual_correlations.csv", index=False)
    event_tests_df.to_csv(out_dir / "rollout_phase_kinetics_bridge_event_tests.csv", index=False)
    source_summary.to_csv(out_dir / "rollout_phase_kinetics_bridge_source_summary.csv", index=False)

    summary = {
        "n_merged_rows": int(len(merged)),
        "n_sources": int(merged["source_stem"].nunique()) if not merged.empty else 0,
        "n_roi": int(merged["roi_id"].nunique()) if not merged.empty else 0,
        "rollout_feature_count": int(len(rollout_feature_cols)),
        "phase_feature_count": int(len(phase_feature_cols)),
        "top_raw_correlation": corr.head(1).to_dict("records")[0] if not corr.empty else {},
        "top_source_residual_correlation": resid_corr.head(1).to_dict("records")[0] if not resid_corr.empty else {},
        "top_event_test": event_tests_df.head(1).to_dict("records")[0] if not event_tests_df.empty else {},
        "guardrail": "This bridge audit links rollout residuals to optical phase-kinetics descriptors for follow-up hypothesis ranking; it does not validate manual phase boundaries or calibrated diffusion coefficients.",
        "outputs": {
            "merged": str(out_dir / "rollout_phase_kinetics_bridge_merged.csv"),
            "correlations": str(out_dir / "rollout_phase_kinetics_bridge_correlations.csv"),
            "source_residual_correlations": str(out_dir / "rollout_phase_kinetics_bridge_source_residual_correlations.csv"),
            "event_tests": str(out_dir / "rollout_phase_kinetics_bridge_event_tests.csv"),
            "source_summary": str(out_dir / "rollout_phase_kinetics_bridge_source_summary.csv"),
        },
    }
    (out_dir / "rollout_phase_kinetics_bridge_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Rollout / Phase Kinetics Bridge Audit",
                "",
                f"- Merged ROI rows: {summary['n_merged_rows']}",
                f"- Sources: {summary['n_sources']}",
                f"- ROI: {summary['n_roi']}",
                "",
                "## Top Associations",
                f"- Raw top correlation: {summary['top_raw_correlation']}",
                f"- Source-residual top correlation: {summary['top_source_residual_correlation']}",
                f"- Top event test: {summary['top_event_test']}",
                "",
                "## Guardrail",
                "",
                summary["guardrail"],
                "",
            ]
        )
    )

    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

