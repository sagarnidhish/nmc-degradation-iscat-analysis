#!/usr/bin/env python3
"""Consensus front-propagation audit for source-balanced pre-event ROIs.

Several individual front and apparent-diffusion proxies remain review-worthy
after source/echem conditioning, but single-feature readouts are fragile. This
audit asks a stricter question: do near-pre rows show internally coherent front
motion across radius quantiles, radius-squared slope, monotonicity, gradient
coherence, and matched residual evidence?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


PRE_NEAR = "near_pre_event_1_8"
PRE_MID = "mid_pre_event_9_16"
PRE_FAR = "far_pre_event_17_32"
POST_CONTROL = {"post_event_1_16", "no_near_event_control"}

COMPARISONS = [
    ("near_vs_far_pre", {PRE_NEAR}, {PRE_FAR}),
    ("near_vs_mid_pre", {PRE_NEAR}, {PRE_MID}),
    ("near_vs_post_control", {PRE_NEAR}, POST_CONTROL),
    ("near_vs_any_non_near", {PRE_NEAR}, {PRE_MID, PRE_FAR, *POST_CONTROL}),
]

OUTWARD_BASES = [
    "front_radius_q60_slope_px_per_norm_time",
    "front_radius_q70_slope_px_per_norm_time",
    "front_radius_q80_slope_px_per_norm_time",
    "front_radius_slope_px_per_norm_time",
    "front_radius2_slope_px2_per_norm_time",
    "apparent_diffusion_q70_um2_per_norm_time",
]

QC_BASES = [
    "front_radius_slope_r2",
    "front_radius2_slope_r2",
    "front_radius_monotonic_fraction",
    "front_gradient_coherence",
]

REVIEW_COLS = [
    "roi_id",
    "cycleNo",
    "source_stem",
    "event_relative_bin",
    "cycles_to_next_event",
    "object_candidate_rank",
    "object_x_full_approx",
    "object_y_full_approx",
    "front_consensus_score",
    "front_residual_outward_z_mean",
    "front_raw_outward_z_mean",
    "front_quantile_positive_fraction",
    "front_q_slope_cv",
    "front_q_slope_range_px_per_norm_time",
    "front_q_slope_mean_px_per_norm_time",
    "front_q_slope_mean_source_echem_residual",
    "front_quality_score",
    "front_radius_slope_px_per_norm_time",
    "front_radius2_slope_px2_per_norm_time",
    "apparent_diffusion_q70_um2_per_norm_time",
    "front_radius_slope_r2",
    "front_radius2_slope_r2",
    "front_gradient_coherence",
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


def robust_z(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    med = vals.median()
    mad = (vals - med).abs().median()
    scale = 1.4826 * mad if pd.notna(mad) and mad > 1e-12 else vals.std()
    if pd.isna(scale) or scale <= 1e-12:
        return pd.Series(0.0, index=x.index)
    return ((vals - med) / scale).clip(-6, 6)


def available_cols(df: pd.DataFrame, cols: Iterable[str], min_count: int = 16) -> List[str]:
    out = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = numeric(df, col)
        if vals.notna().sum() >= min_count and vals.nunique(dropna=True) > 1:
            out.append(col)
    return out


def add_consensus_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_cols = available_cols(out, OUTWARD_BASES)
    resid_cols = available_cols(out, [f"{c}_source_echem_context_residual" for c in OUTWARD_BASES])
    q_raw = [c for c in OUTWARD_BASES[:3] if c in out.columns]
    q_resid = [f"{c}_source_echem_context_residual" for c in OUTWARD_BASES[:3] if f"{c}_source_echem_context_residual" in out.columns]

    for col in raw_cols + resid_cols + QC_BASES:
        if col in out.columns:
            out[f"{col}_robust_z"] = robust_z(numeric(out, col))

    if raw_cols:
        out["front_raw_outward_z_mean"] = out[[f"{c}_robust_z" for c in raw_cols]].mean(axis=1)
    else:
        out["front_raw_outward_z_mean"] = np.nan
    if resid_cols:
        out["front_residual_outward_z_mean"] = out[[f"{c}_robust_z" for c in resid_cols]].mean(axis=1)
    else:
        out["front_residual_outward_z_mean"] = np.nan

    if q_raw:
        q = out[q_raw].apply(pd.to_numeric, errors="coerce")
        out["front_quantile_positive_fraction"] = (q > 0).mean(axis=1)
        out["front_q_slope_mean_px_per_norm_time"] = q.mean(axis=1)
        out["front_q_slope_range_px_per_norm_time"] = q.max(axis=1) - q.min(axis=1)
        denom = q.abs().mean(axis=1).replace(0, np.nan)
        out["front_q_slope_cv"] = q.std(axis=1) / denom
    else:
        out["front_quantile_positive_fraction"] = np.nan
        out["front_q_slope_mean_px_per_norm_time"] = np.nan
        out["front_q_slope_range_px_per_norm_time"] = np.nan
        out["front_q_slope_cv"] = np.nan

    if q_resid:
        out["front_q_slope_mean_source_echem_residual"] = out[q_resid].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        out["front_q_slope_mean_source_echem_residual"] = np.nan

    qc_z_cols = [f"{c}_robust_z" for c in QC_BASES if f"{c}_robust_z" in out.columns]
    out["front_quality_score"] = out[qc_z_cols].mean(axis=1) if qc_z_cols else np.nan
    out["front_consistency_penalty"] = robust_z(numeric(out, "front_q_slope_cv")).fillna(0.0)
    out["front_consensus_score"] = (
        numeric(out, "front_residual_outward_z_mean").fillna(0.0)
        + 0.5 * numeric(out, "front_raw_outward_z_mean").fillna(0.0)
        + 0.5 * numeric(out, "front_quality_score").fillna(0.0)
        + robust_z(numeric(out, "front_quantile_positive_fraction")).fillna(0.0)
        - 0.25 * numeric(out, "front_consistency_penalty").fillna(0.0)
    )
    return out


def event_tests(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    bins = df["event_relative_bin"].astype(str)
    for comparison, treat_bins, control_bins in COMPARISONS:
        tmask = bins.isin(treat_bins)
        cmask = bins.isin(control_bins)
        for feature in features:
            treat = numeric(df, feature).loc[tmask].dropna()
            control = numeric(df, feature).loc[cmask].dropna()
            row: Dict[str, Any] = {
                "comparison": comparison,
                "feature": feature,
                "n_treated": int(len(treat)),
                "n_control": int(len(control)),
                "treated_median": np.nan,
                "control_median": np.nan,
                "treated_minus_control_median": np.nan,
                "mannwhitney_p": np.nan,
                "roc_auc_treated_high": np.nan,
                "average_precision_treated_high": np.nan,
            }
            if len(treat) >= 4 and len(control) >= 4 and pd.concat([treat, control]).nunique() > 1:
                row["treated_median"] = float(treat.median())
                row["control_median"] = float(control.median())
                row["treated_minus_control_median"] = float(treat.median() - control.median())
                _, p_val = mannwhitneyu(treat, control, alternative="two-sided")
                y = np.array([1] * len(treat) + [0] * len(control))
                score = np.concatenate([treat.to_numpy(dtype=float), control.to_numpy(dtype=float)])
                row["mannwhitney_p"] = float(p_val)
                row["roc_auc_treated_high"] = float(roc_auc_score(y, score))
                row["average_precision_treated_high"] = float(average_precision_score(y, score))
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_median_difference"] = out["treated_minus_control_median"].abs()
        out = out.sort_values(["comparison", "mannwhitney_p", "abs_median_difference"], ascending=[True, True, False])
    return out


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


def matched_tests(df: pd.DataFrame, pairs: pd.DataFrame, features: Sequence[str], rng: np.random.Generator, n_perm: int) -> pd.DataFrame:
    rows = []
    if pairs.empty:
        return pd.DataFrame()
    for (comparison, scheme), sub in pairs.groupby(["comparison", "match_scheme"], dropna=False):
        for feature in features:
            t = numeric(df, feature).loc[sub["treated_index"].to_numpy()].reset_index(drop=True)
            c = numeric(df, feature).loc[sub["control_index"].to_numpy()].reset_index(drop=True)
            diff = t - c
            valid = diff.notna()
            x = diff[valid]
            row: Dict[str, Any] = {
                "comparison": comparison,
                "match_scheme": scheme,
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
            if len(x) >= 6 and x.nunique(dropna=True) > 1:
                try:
                    _, p_w = wilcoxon(x, alternative="two-sided", zero_method="wilcox")
                except ValueError:
                    p_w = np.nan
                row.update({
                    "treated_median": float(t[valid].median()),
                    "control_median": float(c[valid].median()),
                    "median_treated_minus_control": float(x.median()),
                    "mean_treated_minus_control": float(x.mean()),
                    "positive_difference_fraction": float((x > 0).mean()),
                    "wilcoxon_p": float(p_w) if np.isfinite(p_w) else np.nan,
                    "signflip_mean_abs_p": signflip_p(x, rng, n_perm),
                })
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_median_difference"] = out["median_treated_minus_control"].abs()
        out = out.sort_values(["signflip_mean_abs_p", "abs_median_difference"], ascending=[True, False])
    return out


def clock_tests(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    pre = df[df["event_relative_bin"].astype(str).isin({PRE_NEAR, PRE_MID, PRE_FAR})].copy()
    if "cycles_to_next_event" not in pre.columns:
        return pd.DataFrame()
    proximity = -numeric(pre, "cycles_to_next_event")
    for feature in features:
        vals = numeric(pre, feature)
        mask = vals.notna() & proximity.notna()
        if mask.sum() < 12 or vals[mask].nunique() < 2 or proximity[mask].nunique() < 2:
            continue
        rho, p_val = spearmanr(proximity[mask], vals[mask])
        rows.append({
            "feature": feature,
            "n_pre_rows": int(mask.sum()),
            "spearman_rho_vs_event_proximity": float(rho),
            "spearman_p": float(p_val),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["spearman_p", "spearman_rho_vs_event_proximity"], ascending=[True, False]) if not out.empty else out


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Pre-Event Front Consensus Audit",
        "",
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Consensus features tested: {', '.join(summary['consensus_features'])}",
        "",
        "## Top Event-Bin Consensus Tests",
    ]
    for row in summary.get("top_event_tests", [])[:8]:
        lines.append(
            f"- {row['comparison']} {row['feature']}: n={row['n_treated']} vs {row['n_control']}, "
            f"median diff={row['treated_minus_control_median']:.4g}, AUC={row['roc_auc_treated_high']:.3f}, "
            f"p={row['mannwhitney_p']:.4g}"
        )
    lines += ["", "## Top Matched Consensus Tests"]
    for row in summary.get("top_matched_tests", [])[:8]:
        lines.append(
            f"- {row['comparison']} {row['match_scheme']} {row['feature']}: n={row['n_pairs']}, "
            f"median diff={row['median_treated_minus_control']:.4g}, sign-flip p={row['signflip_mean_abs_p']:.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument(
        "--out-dir",
        default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_front_consensus_audit",
    )
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    joined = read_csv(
        derived
        / "source_balanced_pre_event_echem_front_coupling_audit"
        / "source_balanced_pre_event_echem_front_joined.csv"
    ).reset_index(drop=True)
    pairs_path = (
        derived
        / "source_balanced_pre_event_echem_matched_residual_audit"
        / "source_balanced_pre_event_echem_matched_pairs.csv"
    )
    pairs = read_csv(pairs_path) if pairs_path.exists() else pd.DataFrame()
    scored = add_consensus_features(joined)
    consensus_features = available_cols(scored, [
        "front_consensus_score",
        "front_residual_outward_z_mean",
        "front_raw_outward_z_mean",
        "front_quantile_positive_fraction",
        "front_q_slope_mean_source_echem_residual",
        "front_q_slope_mean_px_per_norm_time",
        "front_q_slope_cv",
        "front_quality_score",
    ])

    event_df = event_tests(scored, consensus_features)
    matched_df = matched_tests(scored, pairs, consensus_features, rng, args.n_perm)
    clock_df = clock_tests(scored, consensus_features)

    ranked_cols = [c for c in REVIEW_COLS if c in scored.columns]
    ranked = scored.sort_values("front_consensus_score", ascending=False)[ranked_cols].head(64)

    feature_path = out / "source_balanced_pre_event_front_consensus_features.csv"
    event_path = out / "source_balanced_pre_event_front_consensus_event_tests.csv"
    matched_path = out / "source_balanced_pre_event_front_consensus_matched_tests.csv"
    clock_path = out / "source_balanced_pre_event_front_consensus_clock_tests.csv"
    ranked_path = out / "source_balanced_pre_event_front_consensus_ranked_candidates.csv"
    summary_path = out / "source_balanced_pre_event_front_consensus_summary.json"
    scored.to_csv(feature_path, index=False)
    event_df.to_csv(event_path, index=False)
    matched_df.to_csv(matched_path, index=False)
    clock_df.to_csv(clock_path, index=False)
    ranked.to_csv(ranked_path, index=False)

    top_event = event_df.dropna(subset=["mannwhitney_p"]).head(24).to_dict("records") if not event_df.empty else []
    top_matched = (
        matched_df.dropna(subset=["signflip_mean_abs_p"]).head(24).to_dict("records")
        if not matched_df.empty
        else []
    )
    top_clock = clock_df.head(16).to_dict("records") if not clock_df.empty else []
    top_ranked = ranked.head(12).to_dict("records") if not ranked.empty else []

    summary: Dict[str, Any] = {
        "n_rows": int(len(scored)),
        "n_cycles": int(scored["cycleNo"].nunique()),
        "n_sources": int(scored["source_stem"].nunique()),
        "event_relative_bin_counts": clean_json(scored["event_relative_bin"].astype(str).value_counts().to_dict()),
        "consensus_features": consensus_features,
        "n_consensus_features": int(len(consensus_features)),
        "n_matched_pairs": int(len(pairs)),
        "top_event_tests": clean_json(top_event),
        "top_matched_tests": clean_json(top_matched),
        "top_clock_tests": clean_json(top_clock),
        "top_ranked_candidates": clean_json(top_ranked),
        "outputs": {
            "features": str(feature_path),
            "event_tests": str(event_path),
            "matched_tests": str(matched_path),
            "clock_tests": str(clock_path),
            "ranked_candidates": str(ranked_path),
            "summary": str(summary_path),
        },
        "guardrail": (
            "The consensus score combines automatic radial-front, kymograph, and source+echem residual proxies. "
            "It tests internal coherence and prioritizes manual QC, but it is still not calibrated diffusion, "
            "validated phase-boundary tracking, particle-identity proof, causal degradation evidence, or a deployable warning model."
        ),
    }
    summary = clean_json(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
