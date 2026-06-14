#!/usr/bin/env python3
"""Couple source-normalized residual dictionary features to optical physics proxies.

This audit follows the source-balanced residual dictionary work by asking whether
source-robust residual dynamics candidates align with crop-local mask/front and
apparent-diffusion proxies, or whether they behave like isolated reconstruction
artifacts. All correlations can be evaluated raw, source-residualized, within
source ranked, or within source z-scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
TRANSFORMS = ["raw", "source_residual", "within_source_rank", "within_source_z"]
PRIMARY_RESIDUAL_CANDIDATES = [
    "dictionary_recon_error_last_minus_first",
    "dictionary_recon_error_mse_slope",
    "resdict_pc01_mean",
    "resdict_pc02_slope",
    "resdict_pc09_slope",
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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def source_eta2(series: pd.Series, sources: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.notna() & sources.notna()
    vals = vals[valid]
    src = sources[valid]
    if vals.nunique() < 2 or src.nunique() < 2:
        return np.nan
    overall = vals.mean()
    total = float(((vals - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in vals.groupby(src):
        between += len(sub) * float((sub.mean() - overall) ** 2)
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
    if transform == "within_source_z":
        stds = x.groupby(src).transform("std").replace(0, np.nan)
        return (x - means) / stds
    raise ValueError(transform)


def oriented_target_metrics(y: pd.Series, x: pd.Series) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
        return {"target_auc": np.nan, "target_ap": np.nan, "target_direction": "NA"}
    direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
    score = xx if direction == "higher_in_positive" else -xx
    return {
        "target_auc": float(roc_auc_score(yy, score)),
        "target_ap": float(average_precision_score(yy, score)),
        "target_direction": direction,
        "target_median_positive": float(xx[yy == 1].median()),
        "target_median_negative": float(xx[yy == 0].median()),
        "target_median_positive_minus_negative": float(xx[yy == 1].median() - xx[yy == 0].median()),
    }


def residual_feature_columns(cols: Iterable[str]) -> List[str]:
    out = []
    for col in cols:
        if col.startswith("resdict_") or col.startswith("residual_energy_") or col.startswith("dictionary_recon_"):
            out.append(col)
    return out


def physics_feature_columns(cols: Iterable[str]) -> List[str]:
    keys = [
        "mask_",
        "masked_minus_background",
        "front_radius_",
        "front_gradient_",
        "apparent_diffusion",
        "roi_norm_mean_delta",
        "stage_drift",
    ]
    out = []
    for col in cols:
        if any(col.startswith(k) for k in keys):
            out.append(col)
    return out


def correlation_rows(df: pd.DataFrame, residual_features: List[str], physics_features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    y16 = numeric(df, "future_any_drop_within_16cycles")
    for transform in TRANSFORMS:
        transformed_residuals = {feat: transform_feature(numeric(df, feat), sources, transform) for feat in residual_features}
        transformed_physics = {feat: transform_feature(numeric(df, feat), sources, transform) for feat in physics_features}
        target_cache = {
            feat: oriented_target_metrics(y16, vals)
            for feat, vals in {**transformed_residuals, **transformed_physics}.items()
        }
        for residual in residual_features:
            rx = transformed_residuals[residual]
            for physics in physics_features:
                px = transformed_physics[physics]
                valid = rx.notna() & px.notna()
                if valid.sum() < 8 or rx[valid].nunique() < 2 or px[valid].nunique() < 2:
                    rho, pval = np.nan, np.nan
                else:
                    rho, pval = spearmanr(rx[valid], px[valid])
                rt = target_cache[residual]
                pt = target_cache[physics]
                rows.append({
                    "transform": transform,
                    "residual_feature": residual,
                    "physics_feature": physics,
                    "n": int(valid.sum()),
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "spearman_p": float(pval) if np.isfinite(pval) else np.nan,
                    "abs_spearman_rho": abs(float(rho)) if np.isfinite(rho) else np.nan,
                    "residual_source_eta2": source_eta2(rx, sources),
                    "physics_source_eta2": source_eta2(px, sources),
                    "residual_future16_auc": rt.get("target_auc"),
                    "residual_future16_ap": rt.get("target_ap"),
                    "residual_future16_direction": rt.get("target_direction"),
                    "physics_future16_auc": pt.get("target_auc"),
                    "physics_future16_ap": pt.get("target_ap"),
                    "physics_future16_direction": pt.get("target_direction"),
                    "target_aligned": rt.get("target_direction") == pt.get("target_direction") and rt.get("target_direction") != "NA",
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["transform", "abs_spearman_rho", "residual_future16_auc"], ascending=[True, False, False])
    return out


def top_records(df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
    return df.head(n).to_dict(orient="records") if not df.empty else []


def pick_top(df: pd.DataFrame, query: str, sort_cols: List[str], n: int) -> List[Dict[str, Any]]:
    sub = df.query(query).copy() if not df.empty else df
    if sub.empty:
        return []
    return sub.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(n).to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_residual_dictionary_audit")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_residual_physics_coupling_audit")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_dir / "source_balanced_residual_dictionary_features.csv")
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 8]
    residual_features = residual_feature_columns(numeric_cols)
    physics_features = physics_feature_columns(numeric_cols)
    correlations = correlation_rows(df, residual_features, physics_features)

    candidate_rows = correlations[correlations["residual_feature"].isin([c for c in PRIMARY_RESIDUAL_CANDIDATES if c in residual_features])].copy()
    source_residual_pairs = correlations[correlations["transform"] == "source_residual"].copy()
    source_residual_aligned = source_residual_pairs[
        (source_residual_pairs["target_aligned"])
        & (source_residual_pairs["residual_future16_auc"] >= 0.58)
        & (source_residual_pairs["physics_future16_auc"] >= 0.58)
    ].copy()

    paths = {
        "correlations": out / "source_balanced_residual_physics_correlations.csv",
        "candidate_correlations": out / "source_balanced_primary_residual_candidate_correlations.csv",
        "target_aligned_pairs": out / "source_balanced_residual_physics_target_aligned_pairs.csv",
        "summary": out / "source_balanced_residual_physics_coupling_summary.json",
    }
    correlations.to_csv(paths["correlations"], index=False)
    candidate_rows.to_csv(paths["candidate_correlations"], index=False)
    source_residual_aligned.sort_values(
        ["abs_spearman_rho", "residual_future16_auc", "physics_future16_auc"], ascending=False
    ).to_csv(paths["target_aligned_pairs"], index=False)

    best_by_transform = []
    for transform, sub in correlations.groupby("transform"):
        row = sub.sort_values(["abs_spearman_rho", "residual_future16_auc"], ascending=False).iloc[0].to_dict()
        row["transform"] = transform
        best_by_transform.append(row)

    best_source_residual_candidate = pick_top(
        candidate_rows,
        "transform == 'source_residual'",
        ["abs_spearman_rho", "residual_future16_auc"],
        8,
    )
    best_source_residual_aligned = top_records(
        source_residual_aligned.sort_values(["abs_spearman_rho", "residual_future16_auc", "physics_future16_auc"], ascending=False),
        12,
    )

    primary_candidate = correlations[
        (correlations["transform"] == "source_residual")
        & (correlations["residual_feature"] == "dictionary_recon_error_last_minus_first")
    ].sort_values(["abs_spearman_rho", "physics_future16_auc"], ascending=False)

    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()) if "cycleNo" in df.columns else None,
        "n_sources": int(df["source_stem"].nunique()),
        "n_residual_features": int(len(residual_features)),
        "n_physics_features": int(len(physics_features)),
        "transforms": TRANSFORMS,
        "primary_residual_candidates_present": [c for c in PRIMARY_RESIDUAL_CANDIDATES if c in residual_features],
        "best_by_transform": best_by_transform,
        "best_source_residual_primary_candidate_correlations": best_source_residual_candidate,
        "best_source_residual_target_aligned_pairs": best_source_residual_aligned,
        "dictionary_recon_error_last_minus_first_source_residual_top_correlations": top_records(primary_candidate, 12),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Residual-physics coupling is a source-normalized correlation audit over automatic crop-local optical proxies. It can prioritize follow-up mechanisms, but it does not calibrate diffusion coefficients or prove phase-boundary physics without manual/QC and physical scale validation.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Source-Balanced Residual-Physics Coupling Audit",
        "",
        f"Rows/residual features/physics proxies/sources: {summary['n_rows']} / {summary['n_residual_features']} / {summary['n_physics_features']} / {summary['n_sources']}",
        "",
        "## Top Source-Residual Couplings for Primary Candidates",
    ]
    for row in best_source_residual_candidate[:6]:
        readme.append(
            f"- {row.get('residual_feature')} vs {row.get('physics_feature')}: rho={row.get('spearman_rho'):.3f}, "
            f"residual AUC={row.get('residual_future16_auc'):.3f}, physics AUC={row.get('physics_future16_auc'):.3f}"
        )
    readme.extend(["", "## Guardrail", summary["guardrail"], ""])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()
