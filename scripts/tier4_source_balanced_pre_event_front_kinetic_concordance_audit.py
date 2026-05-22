#!/usr/bin/env python3
"""Source-balanced pre-event front/kinetic concordance audit.

This audit joins the source-balanced pre-event phase-kinetics, front-consensus,
visual-QC, and strict gate outputs. It asks whether masked optical kinetics and
front-like evidence point to the same ROI candidates, then ranks review
candidates without assigning manual labels or calibrated diffusion claims.
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


TARGETS = {
    "near_vs_any_non_near": ("near_pre_event_1_8", None),
    "near_vs_mid_pre": ("near_pre_event_1_8", "mid_pre_event_9_16"),
    "near_vs_far_pre": ("near_pre_event_1_8", "far_pre_event_17_32"),
    "near_vs_post_control": ("near_pre_event_1_8", {"post_event_1_16", "no_near_event_control"}),
}
KINETIC_FEATURES = [
    "masked_minus_bg_slope",
    "q55_phase_fraction_delta",
    "q55_phase_fraction_slope",
    "q65_phase_fraction_slope",
    "q75_phase_fraction_slope",
    "q75_logistic_amp",
    "q75_avrami_n",
]
FRONT_FEATURES = [
    "front_consensus_score",
    "front_residual_outward_z_mean",
    "front_raw_outward_z_mean",
    "front_quantile_positive_fraction",
    "front_radius2_slope_px2_per_norm_time",
    "front_radius2_slope_px2_per_norm_time_source_echem_context_residual",
]
QC_FEATURES = [
    "visual_sanity_score",
    "visual_review_score",
    "strict_qc_priority_score",
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


def robust_z(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = float(np.nanmedian(x)) if x.notna().any() else np.nan
    mad = float(np.nanmedian(np.abs(x - med))) if np.isfinite(med) else np.nan
    if not np.isfinite(mad) or mad <= 1e-12:
        sd = float(np.nanstd(x))
        mad = sd / 1.4826 if np.isfinite(sd) and sd > 1e-12 else np.nan
    if not np.isfinite(mad) or mad <= 1e-12:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return ((x - med) / (1.4826 * mad)).clip(-6, 6)


def row_mean(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    have = [c for c in cols if c in df.columns]
    if not have:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[have].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def source_residual(values: pd.Series, sources: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    return x - x.groupby(sources.astype(str)).transform("mean")


def target_series(df: pd.DataFrame, positive: Any, negative: Any) -> pd.Series:
    bins = df["event_relative_bin"].astype(str)
    pos_set = positive if isinstance(positive, set) else {positive}
    y = pd.Series(np.nan, index=df.index)
    y.loc[bins.isin(pos_set)] = 1
    if negative is None:
        y.loc[~bins.isin(pos_set)] = 0
    else:
        neg_set = negative if isinstance(negative, set) else {negative}
        y.loc[bins.isin(neg_set)] = 0
    return y


def event_tests(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    for target, (positive, negative) in TARGETS.items():
        y = target_series(df, positive, negative)
        for feature in features:
            for transform in ["raw", "source_residual"]:
                x = numeric(df, feature)
                if transform == "source_residual":
                    x = source_residual(x, sources)
                valid = y.isin([0, 1]) & x.notna()
                yy = y[valid].astype(int)
                xx = x[valid]
                if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
                    continue
                direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
                score = xx if direction == "higher_in_positive" else -xx
                try:
                    _, p_mwu = mannwhitneyu(xx[yy == 1], xx[yy == 0], alternative="two-sided")
                except ValueError:
                    p_mwu = np.nan
                rows.append({
                    "target": target,
                    "feature": feature,
                    "transform": transform,
                    "n": int(len(yy)),
                    "n_positive": int(yy.sum()),
                    "direction": direction,
                    "oriented_auc": float(roc_auc_score(yy, score)),
                    "average_precision": float(average_precision_score(yy, score)),
                    "median_positive_minus_negative": float(xx[yy == 1].median() - xx[yy == 0].median()),
                    "mwu_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mwu_p", "oriented_auc", "average_precision"], ascending=[True, False, False])
    return out


def correlations(df: pd.DataFrame, x_cols: Iterable[str], y_cols: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for x_col in x_cols:
        x = numeric(df, x_col)
        for y_col in y_cols:
            y = numeric(df, y_col)
            valid = x.notna() & y.notna()
            if valid.sum() < 8 or x[valid].nunique() < 3 or y[valid].nunique() < 3:
                continue
            stat = spearmanr(x[valid], y[valid])
            rows.append({
                "x": x_col,
                "y": y_col,
                "n": int(valid.sum()),
                "spearman_rho": float(stat.statistic) if np.isfinite(stat.statistic) else np.nan,
                "spearman_p": float(stat.pvalue) if np.isfinite(stat.pvalue) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.assign(abs_rho=out["spearman_rho"].abs()).sort_values(["spearman_p", "abs_rho"], ascending=[True, False])
    return out


def top_items(df: pd.DataFrame, n: int = 12) -> List[Dict[str, Any]]:
    return df.head(n).drop(columns=["abs_rho"], errors="ignore").to_dict("records") if not df.empty else []


def compact_candidates(df: pd.DataFrame, n: int = 20) -> List[Dict[str, Any]]:
    cols = [
        "roi_id",
        "cycleNo",
        "source_stem",
        "event_relative_bin",
        "cycles_to_next_event",
        "consensus_review_rank",
        "consensus_review_score",
        "front_kinetic_concordance_score",
        "kinetic_evidence_score",
        "front_evidence_score",
        "qc_evidence_score",
        "front_kinetic_product",
        "front_kinetic_tier",
        "masked_minus_bg_slope",
        "q75_phase_fraction_slope",
        "front_consensus_score",
        "front_radius2_slope_px2_per_norm_time_source_echem_context_residual",
        "visual_sanity_score",
        "visual_review_score",
        "strict_qc_priority_score",
        "manual_front_review_gate",
        "automatic_diffusion_claim_gate",
    ]
    have = [c for c in cols if c in df.columns]
    return df.head(n)[have].to_dict("records") if have else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_front_kinetic_concordance_audit")
    args = parser.parse_args()
    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kinetics = read_csv(derived / "source_balanced_pre_event_phase_kinetics_audit" / "source_balanced_pre_event_phase_kinetics_features.csv")
    front = read_csv(derived / "source_balanced_pre_event_front_consensus_audit" / "source_balanced_pre_event_front_consensus_ranked_candidates.csv")
    strict = read_csv(derived / "source_balanced_pre_event_strict_qc_gated_front_audit" / "source_balanced_pre_event_strict_qc_gated_front_candidates.csv")
    visual = read_csv(derived / "source_balanced_pre_event_visual_qc_modes" / "source_balanced_pre_event_visual_qc_modes.csv")

    front_keep = ["roi_id"] + [c for c in FRONT_FEATURES if c in front.columns]
    strict_keep = ["roi_id", "manual_front_review_gate", "automatic_diffusion_claim_gate", "strict_qc_priority_score", "n_passed_component_gates"]
    strict_keep = [c for c in strict_keep if c in strict.columns]
    visual_keep = ["roi_id", "visual_qc_tier", "visual_mode_name", "visual_front_plausibility_score", "visual_artifact_risk_score"]
    visual_keep = [c for c in visual_keep if c in visual.columns]
    df = kinetics.merge(front[front_keep], on="roi_id", how="left", suffixes=("", "_front"))
    df = df.merge(strict[strict_keep], on="roi_id", how="left", suffixes=("", "_strict"))
    df = df.merge(visual[visual_keep], on="roi_id", how="left", suffixes=("", "_visual"))
    if "manual_front_review_gate" not in df.columns:
        df["manual_front_review_gate"] = False
    if "automatic_diffusion_claim_gate" not in df.columns:
        df["automatic_diffusion_claim_gate"] = False
    df["manual_front_review_gate"] = df["manual_front_review_gate"].fillna(False).astype(bool)
    df["automatic_diffusion_claim_gate"] = df["automatic_diffusion_claim_gate"].fillna(False).astype(bool)

    for col in KINETIC_FEATURES + FRONT_FEATURES + QC_FEATURES:
        if col in df.columns:
            df[f"z__{col}"] = robust_z(numeric(df, col))
    kinetic_z_cols = [f"z__{c}" for c in KINETIC_FEATURES if f"z__{c}" in df.columns]
    front_z_cols = [f"z__{c}" for c in FRONT_FEATURES if f"z__{c}" in df.columns]
    qc_z_cols = [f"z__{c}" for c in QC_FEATURES if f"z__{c}" in df.columns]
    df["kinetic_evidence_score"] = row_mean(df, kinetic_z_cols)
    df["front_evidence_score"] = row_mean(df, front_z_cols)
    df["qc_evidence_score"] = row_mean(df, qc_z_cols)
    df["front_kinetic_product"] = df["kinetic_evidence_score"] * df["front_evidence_score"]
    df["front_kinetic_concordance_score"] = (
        0.40 * df["kinetic_evidence_score"].clip(lower=-2)
        + 0.35 * df["front_evidence_score"].clip(lower=-2)
        + 0.15 * df["qc_evidence_score"].fillna(0).clip(lower=-2)
        + 0.10 * df["front_kinetic_product"].clip(lower=-2, upper=6)
    )
    df["front_kinetic_concordance_score"] = df["front_kinetic_concordance_score"].replace([np.inf, -np.inf], np.nan)
    near = df["event_relative_bin"].astype(str).eq("near_pre_event_1_8")
    df["front_kinetic_tier"] = "routine_or_low_concordance"
    df.loc[(df["kinetic_evidence_score"] > 0.5) & (df["front_evidence_score"] > 0.5), "front_kinetic_tier"] = "front_kinetic_concordant"
    df.loc[near & (df["kinetic_evidence_score"] > 0.5) & (df["front_evidence_score"] > 0.5), "front_kinetic_tier"] = "near_pre_front_kinetic_review"
    df.loc[df["manual_front_review_gate"], "front_kinetic_tier"] = "strict_gate_manual_front_review"
    df.loc[(df["kinetic_evidence_score"] > 0.75) & (df["front_evidence_score"] <= 0), "front_kinetic_tier"] = "kinetic_only_guardrail"
    df.loc[(df["front_evidence_score"] > 0.75) & (df["kinetic_evidence_score"] <= 0), "front_kinetic_tier"] = "front_only_guardrail"
    ranked = df.sort_values(["front_kinetic_concordance_score", "consensus_review_score"], ascending=[False, False])

    score_features = [
        "front_kinetic_concordance_score",
        "kinetic_evidence_score",
        "front_evidence_score",
        "qc_evidence_score",
        "front_kinetic_product",
    ]
    tests = event_tests(df[df["kinetics_status"].eq("ok")], score_features)
    corr = correlations(df, KINETIC_FEATURES + score_features, FRONT_FEATURES + QC_FEATURES + ["cycles_to_next_event", "capacity_fraction_of_first", "shape_V_mean"])
    source_summary = (
        df.groupby("source_stem", dropna=False)
        .agg(
            n_roi=("roi_id", "count"),
            near_pre=("event_relative_bin", lambda s: int((s.astype(str) == "near_pre_event_1_8").sum())),
            median_concordance=("front_kinetic_concordance_score", "median"),
            median_kinetic=("kinetic_evidence_score", "median"),
            median_front=("front_evidence_score", "median"),
            n_manual_front_gate=("manual_front_review_gate", "sum"),
        )
        .reset_index()
        .sort_values(["median_concordance", "n_roi"], ascending=[False, False])
    )

    paths = {
        "ranked_candidates": out / "source_balanced_pre_event_front_kinetic_concordance_ranked_candidates.csv",
        "event_tests": out / "source_balanced_pre_event_front_kinetic_concordance_event_tests.csv",
        "correlations": out / "source_balanced_pre_event_front_kinetic_concordance_correlations.csv",
        "source_summary": out / "source_balanced_pre_event_front_kinetic_concordance_source_summary.csv",
        "summary": out / "source_balanced_pre_event_front_kinetic_concordance_summary.json",
    }
    ranked.to_csv(paths["ranked_candidates"], index=False)
    tests.to_csv(paths["event_tests"], index=False)
    corr.to_csv(paths["correlations"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)

    summary = {
        "n_rows": int(len(df)),
        "n_ok": int(df["kinetics_status"].eq("ok").sum()) if "kinetics_status" in df else int(len(df)),
        "n_sources": int(df["source_stem"].nunique()),
        "tier_counts": ranked["front_kinetic_tier"].value_counts().to_dict(),
        "top_candidates": compact_candidates(ranked, 20),
        "top_event_tests": top_items(tests, 20),
        "top_correlations": top_items(corr, 20),
        "source_summary": source_summary.to_dict("records"),
        "guardrail": "Front/kinetic concordance is an automatic review-prioritization score joining optical phase kinetics, front proxies, visual QC, and strict gates. It does not assign manual labels, validate phase-boundary motion, calibrate diffusion, or prove degradation causality.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Source-Balanced Pre-Event Front/Kinetic Concordance Audit",
        "",
        "Ranks source-balanced pre-event ROI candidates by agreement between particle-mask optical kinetics, front-consensus evidence, visual QC, and strict front gates.",
        "",
        f"- Rows: {summary['n_rows']}",
        f"- Loaded kinetic rows: {summary['n_ok']}",
        f"- Sources: {summary['n_sources']}",
        f"- Tier counts: {summary['tier_counts']}",
        "",
        "## Top Candidates",
    ]
    for row in summary["top_candidates"][:8]:
        lines.append(
            f"- {row.get('roi_id')} {row.get('event_relative_bin')}: score={row.get('front_kinetic_concordance_score'):.3f}, kinetic={row.get('kinetic_evidence_score'):.3f}, front={row.get('front_evidence_score'):.3f}, tier={row.get('front_kinetic_tier')}"
        )
    lines += ["", "## Top Event Tests"]
    for row in summary["top_event_tests"][:6]:
        lines.append(
            f"- {row.get('target')} {row.get('transform')} {row.get('feature')}: AUC={row.get('oriented_auc'):.3f}, median diff={row.get('median_positive_minus_negative'):.3g}, p={row.get('mwu_p'):.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out), "n_rows": summary["n_rows"], "tier_counts": summary["tier_counts"], "top_candidates": summary["top_candidates"][:3]}, indent=2))


if __name__ == "__main__":
    main()
