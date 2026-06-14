#!/usr/bin/env python3
"""Source-robustness audit for source-balanced mask/front descriptors.

The source-balanced mask/front sanity audit found useful crop-local optical and
front proxies, but several top associations were source structured. This script
tests whether those descriptors still separate weak future-drop labels after
simple source residualization and within-source normalization.
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
    if transform == "raw":
        return x
    means = x.groupby(src).transform("mean")
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


def feature_columns(df: pd.DataFrame) -> List[str]:
    blocked = {
        "roi_id", "source_stem", "npz_path", "preview_png", "selection_reason", "validation_label",
        "future_any_drop_within_8cycles", "future_any_drop_within_16cycles", "any_abrupt_drop",
    }
    cols = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce")):
            if pd.to_numeric(df[col], errors="coerce").notna().sum() >= 8:
                cols.append(col)
    preferred = [c for c in cols if any(k in c for k in ["mask_", "front_", "apparent_diffusion", "roi_norm_mean_delta"])]
    other = [c for c in cols if c not in preferred]
    return preferred + other


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
        out = out.sort_values(["target", "transform", "oriented_auc", "average_precision"], ascending=[True, True, False, False])
    return out


def best_by_transform(metrics: pd.DataFrame) -> List[Dict[str, Any]]:
    if metrics.empty:
        return []
    rows = []
    for (target, transform), sub in metrics.groupby(["target", "transform"], dropna=False):
        sub = sub.sort_values(["oriented_auc", "average_precision"], ascending=False)
        if not sub.empty:
            rows.append(sub.iloc[0].to_dict())
    rows = sorted(rows, key=lambda r: (str(r.get("target")), str(r.get("transform"))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_mask_front_sanity_audit")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_mask_front_source_residual_audit")
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(in_dir / "source_balanced_mask_front_features.csv")
    feats = feature_columns(features_df)
    metrics = audit(features_df, feats)
    best = best_by_transform(metrics)

    paths = {
        "metrics": out / "source_balanced_mask_front_source_residual_metrics.csv",
        "best_by_transform": out / "source_balanced_mask_front_source_residual_best_by_transform.csv",
        "summary": out / "source_balanced_mask_front_source_residual_summary.json",
    }
    metrics.to_csv(paths["metrics"], index=False)
    pd.DataFrame(best).to_csv(paths["best_by_transform"], index=False)

    future16_raw = next((r for r in best if r.get("target") == "future_any_drop_within_16cycles" and r.get("transform") == "raw"), {})
    future16_resid = next((r for r in best if r.get("target") == "future_any_drop_within_16cycles" and r.get("transform") == "source_residual"), {})
    future16_rank = next((r for r in best if r.get("target") == "future_any_drop_within_16cycles" and r.get("transform") == "within_source_rank"), {})
    summary = {
        "n_rows": int(len(features_df)),
        "n_cycles": int(features_df["cycleNo"].nunique()) if "cycleNo" in features_df.columns else None,
        "n_sources": int(features_df["source_stem"].nunique()),
        "n_features_tested": int(len(feats)),
        "transforms": TRANSFORMS,
        "best_by_transform": best,
        "future16_raw_best": future16_raw,
        "future16_source_residual_best": future16_resid,
        "future16_within_source_rank_best": future16_rank,
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Source residualization tests whether automatic mask/front proxies survive source structure. Passing this audit would still be weak-label, automatic-mask evidence; failing it means the feature is useful mainly for QC/source triage.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Source-Balanced Mask/Front Source-Residual Audit",
        "",
        f"Rows/features/sources: {summary['n_rows']} / {summary['n_features_tested']} / {summary['n_sources']}",
        "",
        "## Future16 Best Transforms",
    ]
    for label, row in [
        ("raw", future16_raw),
        ("source_residual", future16_resid),
        ("within_source_rank", future16_rank),
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
