#!/usr/bin/env python3
"""Source-normalization audit for source-balanced residual dictionary features.

The source-balanced residual dictionary audit found useful next-frame residual
PC descriptors, but grouped readouts failed across held-out source movies. This
script asks which residual-dynamics descriptors survive simple source removal:
source residuals, within-source z scores, and within-source ranks.
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

TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
TRANSFORMS = ["raw", "source_mean_only", "source_residual", "within_source_z", "within_source_rank"]


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
    if transform == "source_mean_only":
        return means
    if transform == "source_residual":
        return x - means
    if transform == "within_source_z":
        stds = x.groupby(src).transform("std").replace(0, np.nan)
        return (x - means) / stds
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


def feature_family(feature: str) -> str:
    if feature.startswith("resdict_") or feature.startswith("dictionary_") or feature.startswith("residual_energy_"):
        return "residual_dictionary"
    if any(k in feature for k in ["mask_", "masked_", "front_", "apparent_diffusion", "roi_norm_mean_delta"]):
        return "mask_front_scalar"
    if feature.startswith("object_"):
        return "object_reconstruction"
    return "other"


def feature_columns(df: pd.DataFrame) -> List[str]:
    blocked = {
        "roi_id", "source_stem", "npz_path", "preview_png", "selection_reason", "validation_label",
        "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "any_abrupt_drop",
    }
    cols = []
    for col in df.columns:
        if col in blocked:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 8 and vals.nunique(dropna=True) >= 2:
            cols.append(col)
    residual = [c for c in cols if feature_family(c) == "residual_dictionary"]
    mask_front = [c for c in cols if feature_family(c) == "mask_front_scalar"]
    object_recon = [c for c in cols if feature_family(c) == "object_reconstruction"]
    return residual + mask_front + object_recon


def audit(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    for target in [t for t in TARGETS if t in df.columns]:
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
                    "feature_family": feature_family(feature),
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
        out = out.sort_values(
            ["target", "transform", "oriented_auc", "average_precision"],
            ascending=[True, True, False, False],
        )
    return out


def best_by(metrics: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for group, sub in metrics.groupby(keys, dropna=False):
        sub = sub.sort_values(["oriented_auc", "average_precision"], ascending=False)
        row = sub.iloc[0].to_dict()
        if not isinstance(group, tuple):
            group = (group,)
        for key, value in zip(keys, group):
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_dictionary_audit")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_residual_dictionary_source_residual_audit")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    features_df = pd.read_csv(in_dir / "source_balanced_residual_dictionary_features.csv")
    feats = feature_columns(features_df)
    metrics = audit(features_df, feats)
    best_transform = best_by(metrics, ["target", "transform"])
    best_family = best_by(metrics, ["target", "transform", "feature_family"])

    paths = {
        "metrics": out / "source_balanced_residual_dictionary_source_residual_metrics.csv",
        "best_by_transform": out / "source_balanced_residual_dictionary_source_residual_best_by_transform.csv",
        "best_by_family": out / "source_balanced_residual_dictionary_source_residual_best_by_family.csv",
        "summary": out / "source_balanced_residual_dictionary_source_residual_summary.json",
    }
    metrics.to_csv(paths["metrics"], index=False)
    best_transform.to_csv(paths["best_by_transform"], index=False)
    best_family.to_csv(paths["best_by_family"], index=False)

    def pick(target: str, transform: str, family: str | None = None) -> Dict[str, Any]:
        table = best_family if family is not None else best_transform
        sub = table[(table["target"] == target) & (table["transform"] == transform)]
        if family is not None:
            sub = sub[sub["feature_family"] == family]
        return sub.iloc[0].to_dict() if len(sub) else {}

    summary = {
        "n_rows": int(len(features_df)),
        "n_cycles": int(features_df["cycleNo"].nunique()) if "cycleNo" in features_df.columns else None,
        "n_sources": int(features_df["source_stem"].nunique()),
        "n_features_tested": int(len(feats)),
        "feature_family_counts": pd.Series([feature_family(c) for c in feats]).value_counts().to_dict(),
        "transforms": TRANSFORMS,
        "future16_raw_best": pick("future_any_drop_within_16cycles", "raw"),
        "future16_raw_residual_dictionary_best": pick("future_any_drop_within_16cycles", "raw", "residual_dictionary"),
        "future16_source_residual_best": pick("future_any_drop_within_16cycles", "source_residual"),
        "future16_source_residual_residual_dictionary_best": pick("future_any_drop_within_16cycles", "source_residual", "residual_dictionary"),
        "future16_within_source_rank_best": pick("future_any_drop_within_16cycles", "within_source_rank"),
        "future16_within_source_rank_residual_dictionary_best": pick("future_any_drop_within_16cycles", "within_source_rank", "residual_dictionary"),
        "best_by_transform": best_transform.to_dict(orient="records"),
        "best_by_family": best_family.to_dict(orient="records"),
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Source-normalized residual dictionary tests are in-cohort weak-label audits. They can identify source-robust residual dynamics candidates for follow-up, but they do not prove source-transferable prediction or calibrated phase/diffusion physics.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Source-Balanced Residual Dictionary Source-Residual Audit",
        "",
        f"Rows/features/sources: {summary['n_rows']} / {summary['n_features_tested']} / {summary['n_sources']}",
        f"Feature families: {summary['feature_family_counts']}",
        "",
        "## Future16 Best Residual Dictionary Features",
    ]
    for label, row in [
        ("raw", summary["future16_raw_residual_dictionary_best"]),
        ("source_residual", summary["future16_source_residual_residual_dictionary_best"]),
        ("within_source_rank", summary["future16_within_source_rank_residual_dictionary_best"]),
    ]:
        readme.append(
            f"- {label}: {row.get('feature', 'NA')} AUC={row.get('oriented_auc', np.nan):.3f}, "
            f"AP={row.get('average_precision', np.nan):.3f}, eta2={row.get('source_eta2_after_transform', np.nan):.3f}"
        )
    readme.extend(["", "## Guardrail", summary["guardrail"], ""])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()
