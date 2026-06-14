#!/usr/bin/env python3
"""Test whether particle-region descriptors intensify as events approach.

Existing audits show near-pre-event ROIs can be ranked against controls. This
audit asks a stricter temporal-physics question: within pre-event particle
regions, do automatic transport/front/kinetic descriptors increase as the next
drop gets closer?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


SCORE_FEATURES = [
    "transport_mechanism_score",
    "transport_raw_score",
    "transport_source_residual_score",
    "front_kinetic_score",
    "observable_tail_score",
    "qc_review_score",
    "abs_radial_flow_mean",
    "abs_radial_flow_q90",
    "particle_flow_mag_mean",
    "particle_flow_mag_q90",
    "curl_mean",
    "curl_q90",
    "apparent_transport_instability_score",
    "front_evidence_score",
    "kinetic_evidence_score",
    "front_kinetic_concordance_score",
    "front_consensus_score",
    "front_radius2_slope_px2_per_norm_time",
]

PRE_EVENT_BINS = {
    "near_pre_event_1_8",
    "mid_pre_event_9_16",
    "far_pre_event_17_32",
}


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


def rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    idx = np.where(ok)[0]
    order = idx[np.argsort(values[ok], kind="mergesort")]
    sorted_values = values[order]
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x = x[ok]
    y = y[ok]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rank_average(x), rank_average(y))


def permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 20260522) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4:
        return float("nan")
    obs = abs(spearman(x, y))
    count = 1
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if abs(spearman(x, yp)) >= obs - 1e-12:
            count += 1
    return float(count / (n_perm + 1))


def sign_flip_p(values: Iterable[float]) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return float("nan")
    obs = abs(vals.sum())
    total = 2 ** len(vals)
    count = 0
    for mask in range(total):
        signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(len(vals))], dtype=float)
        if abs((vals * signs).sum()) >= obs - 1e-12:
            count += 1
    return float(count / total)


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.unique(x[ok]).size < 2:
        return float("nan")
    x = x[ok]
    y = y[ok]
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    if denom <= 0:
        return float("nan")
    return float(((x - xm) * (y - y.mean())).sum() / denom)


def source_center(df: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    return values - values.groupby(df["source_stem"]).transform("median")


def bin_name(days: float) -> str:
    if days <= 2:
        return "pre_1_2"
    if days <= 4:
        return "pre_3_4"
    if days <= 8:
        return "pre_5_8"
    if days <= 16:
        return "pre_9_16"
    return "pre_17_32"


def feature_tests(pre: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    proximity = pd.to_numeric(pre["event_proximity_score"], errors="coerce").to_numpy(dtype=float)
    for feature in features:
        vals = pd.to_numeric(pre[feature], errors="coerce").to_numpy(dtype=float)
        centered = source_center(pre, feature).to_numpy(dtype=float)
        source_slopes = []
        source_rhos = []
        for _, grp in pre.groupby("source_stem"):
            if len(grp) < 4 or grp["cycles_to_next_event"].nunique() < 2:
                continue
            sx = pd.to_numeric(grp["event_proximity_score"], errors="coerce").to_numpy(dtype=float)
            sy = pd.to_numeric(grp[feature], errors="coerce").to_numpy(dtype=float)
            source_slopes.append(slope(sx, sy))
            source_rhos.append(spearman(sx, sy))
        rows.append({
            "feature": feature,
            "n_rows": int(np.isfinite(vals).sum()),
            "n_sources": int(pre.loc[np.isfinite(vals), "source_stem"].nunique()),
            "raw_spearman_rho": spearman(proximity, vals),
            "raw_permutation_p": permutation_p(proximity, vals),
            "source_centered_spearman_rho": spearman(proximity, centered),
            "source_centered_permutation_p": permutation_p(proximity, centered),
            "n_sources_with_slopes": int(np.isfinite(source_slopes).sum()),
            "median_source_slope": float(np.nanmedian(source_slopes)) if source_slopes else None,
            "positive_source_slope_fraction": float(np.nanmean(np.asarray(source_slopes) > 0)) if source_slopes else None,
            "source_slope_sign_flip_p": sign_flip_p(source_slopes),
            "median_source_spearman": float(np.nanmedian(source_rhos)) if source_rhos else None,
        })
    return pd.DataFrame(rows).sort_values(
        ["source_centered_spearman_rho", "raw_spearman_rho"], ascending=[False, False]
    )


def bin_summary(pre: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature in features:
        for b, grp in pre.groupby("pre_event_distance_bin"):
            rows.append({
                "feature": feature,
                "pre_event_distance_bin": b,
                "n_rows": int(len(grp)),
                "n_sources": int(grp["source_stem"].nunique()),
                "median_value": float(pd.to_numeric(grp[feature], errors="coerce").median()),
                "median_source_centered_value": float(source_center(pre, feature).loc[grp.index].median()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/pre_event_temporal_dose_response_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dossier = pd.read_csv(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_dossier.csv")
    dossier = dossier.copy()
    dossier["cycles_to_next_event"] = pd.to_numeric(dossier["cycles_to_next_event"], errors="coerce")
    pre = dossier[dossier["event_relative_bin"].isin(PRE_EVENT_BINS)].copy()
    pre = pre[pre["cycles_to_next_event"].between(1, 32, inclusive="both")].copy()
    pre["event_proximity_score"] = -pre["cycles_to_next_event"]
    pre["inverse_cycles_to_next_event"] = 1.0 / pre["cycles_to_next_event"]
    pre["pre_event_distance_bin"] = pre["cycles_to_next_event"].map(bin_name)

    features = [f for f in SCORE_FEATURES if f in pre.columns and pd.api.types.is_numeric_dtype(pre[f])]
    tests = feature_tests(pre, features)
    bins = bin_summary(pre, features)
    source_rows = []
    for source, grp in pre.groupby("source_stem"):
        row = {
            "source_stem": source,
            "n_pre_rows": int(len(grp)),
            "n_cycles": int(grp["cycleNo"].nunique()),
            "min_cycles_to_next_event": float(grp["cycles_to_next_event"].min()),
            "max_cycles_to_next_event": float(grp["cycles_to_next_event"].max()),
        }
        for feature in ["transport_mechanism_score", "front_kinetic_score", "qc_review_score"]:
            if feature in grp.columns:
                row[f"{feature}_slope_vs_proximity"] = slope(
                    grp["event_proximity_score"].to_numpy(dtype=float),
                    pd.to_numeric(grp[feature], errors="coerce").to_numpy(dtype=float),
                )
                row[f"{feature}_spearman_vs_proximity"] = spearman(
                    grp["event_proximity_score"].to_numpy(dtype=float),
                    pd.to_numeric(grp[feature], errors="coerce").to_numpy(dtype=float),
                )
        source_rows.append(row)
    source_summary = pd.DataFrame(source_rows).sort_values("n_pre_rows", ascending=False)

    paths = {
        "feature_tests": out / "pre_event_temporal_dose_response_feature_tests.csv",
        "bin_summary": out / "pre_event_temporal_dose_response_bin_summary.csv",
        "source_summary": out / "pre_event_temporal_dose_response_source_summary.csv",
        "summary": out / "pre_event_temporal_dose_response_summary.json",
        "readme": out / "README.md",
    }
    tests.to_csv(paths["feature_tests"], index=False)
    bins.to_csv(paths["bin_summary"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)

    top_centered = tests.sort_values("source_centered_spearman_rho", ascending=False).head(10).to_dict("records")
    top_source_slope = tests.sort_values(["positive_source_slope_fraction", "median_source_slope"], ascending=[False, False]).head(10).to_dict("records")
    key_features = tests[tests["feature"].isin(["transport_mechanism_score", "front_kinetic_score", "qc_review_score"])].to_dict("records")
    summary = clean_json({
        "overall_status": "pre_event_temporal_dose_response_ready",
        "n_input_rows": int(len(dossier)),
        "n_pre_event_rows": int(len(pre)),
        "n_pre_event_cycles": int(pre["cycleNo"].nunique()),
        "n_pre_event_sources": int(pre["source_stem"].nunique()),
        "distance_bin_counts": pre["pre_event_distance_bin"].value_counts().sort_index().to_dict(),
        "features_tested": features,
        "key_feature_tests": key_features,
        "top_source_centered_tests": top_centered,
        "top_source_slope_tests": top_source_slope,
        "guardrail": "Temporal dose-response tests use automatic particle-region optical/front descriptors and event-relative labels. They support precursor-ranking hypotheses only, not causal mechanisms, calibrated phase-boundary velocities, or diffusion coefficients.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Pre-Event Temporal Dose-Response Audit",
        "",
        "Tests whether particle-region descriptors increase as the next event approaches.",
        "",
        f"- Pre-event rows: {summary['n_pre_event_rows']}",
        f"- Pre-event cycles: {summary['n_pre_event_cycles']}",
        f"- Pre-event sources: {summary['n_pre_event_sources']}",
        "",
        "## Key Feature Tests",
        "",
    ]
    for row in key_features:
        lines.append(
            f"- {row['feature']}: source-centered rho {row['source_centered_spearman_rho']:.3f}, "
            f"raw rho {row['raw_spearman_rho']:.3f}, median source slope {row['median_source_slope']:.3g}, "
            f"positive source slope fraction {row['positive_source_slope_fraction']:.3f}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    paths["readme"].write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
