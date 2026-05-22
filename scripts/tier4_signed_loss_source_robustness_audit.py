#!/usr/bin/env python3
"""Source-robustness audit for signed optical-loss axes.

The signed optical-loss mechanism audit found highly predictive axes, but the
strongest optical axis was source-structured. This script separates source-level
and within-source evidence using rank/centering transforms, source-macro AUCs,
leave-one-source influence, and within-source label permutations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
AXES = [
    "signed_optical_loss_axis",
    "front_contraction_axis",
    "rollout_difficulty_axis",
    "echem_degraded_state_axis",
    "combined_loss_mechanism_axis",
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


def source_eta2(series: pd.Series, sources: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.notna() & sources.notna()
    vals = vals[valid]
    src = sources[valid]
    if vals.nunique() < 2 or src.nunique() < 2:
        return float("nan")
    overall = vals.mean()
    total = float(((vals - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in vals.groupby(src):
        between += len(sub) * float((sub.mean() - overall) ** 2)
    return between / total


def orient_score(y: pd.Series, x: pd.Series) -> tuple[pd.Series, str]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    if len(yy) == 0 or yy.nunique() < 2:
        return x, "NA"
    pos_med = xx[yy == 1].median()
    neg_med = xx[yy == 0].median()
    if pos_med >= neg_med:
        return x, "higher_in_positive"
    return -x, "lower_in_positive"


def metric(y: pd.Series, x: pd.Series, sources: pd.Series, target: str, axis: str, transform: str) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & x.notna() & sources.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    score, direction = orient_score(yy, xx)
    ss = score[valid]
    out: Dict[str, Any] = {
        "target": target,
        "axis": axis,
        "transform": transform,
        "n": int(len(yy)),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "n_sources": int(sources[valid].nunique()) if len(yy) else 0,
        "direction": direction,
        "oriented_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "source_eta2": source_eta2(xx, sources[valid]) if len(yy) else np.nan,
        "source_macro_auc_mean": np.nan,
        "source_macro_auc_n": 0,
        "source_macro_auc_min": np.nan,
    }
    if len(yy) >= 8 and yy.nunique() == 2 and ss.nunique() > 1:
        out["oriented_auc"] = float(roc_auc_score(yy, ss))
        out["average_precision"] = float(average_precision_score(yy, ss))
        rho, pval = spearmanr(yy, ss)
        out["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
        out["spearman_p"] = float(pval) if np.isfinite(pval) else np.nan
    per_source = []
    for _, idx in sources[valid].groupby(sources[valid]).groups.items():
        sy = yy.loc[idx]
        sx = ss.loc[idx]
        if len(sy) >= 4 and sy.nunique() == 2 and sx.nunique() > 1:
            try:
                per_source.append(float(roc_auc_score(sy, sx)))
            except ValueError:
                pass
    if per_source:
        arr = np.asarray(per_source)
        out["source_macro_auc_mean"] = float(arr.mean())
        out["source_macro_auc_n"] = int(len(arr))
        out["source_macro_auc_min"] = float(arr.min())
    return out


def within_source_rank(series: pd.Series, sources: pd.Series) -> pd.Series:
    return series.groupby(sources).rank(pct=True, method="average")


def within_source_centered_z(series: pd.Series, sources: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for _, idx in sources.groupby(sources).groups.items():
        sub = vals.loc[idx]
        sd = sub.std(ddof=0)
        if np.isfinite(sd) and sd > 1e-12:
            out.loc[idx] = (sub - sub.mean()) / sd
        else:
            out.loc[idx] = 0.0
    return out


def source_mean_value(series: pd.Series, sources: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.groupby(sources).transform("mean")


def source_residual(series: pd.Series, sources: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals - vals.groupby(sources).transform("mean") + vals.mean()


def permutation_p(y: pd.Series, score: pd.Series, sources: pd.Series, observed_auc: float, seed: int, n_perm: int, within_source: bool) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & score.notna() & sources.notna()
    yy = y[valid].astype(int).copy()
    ss = score[valid]
    src = sources[valid]
    if not np.isfinite(observed_auc) or len(yy) < 8 or yy.nunique() < 2:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan, "null_auc_mean": np.nan, "null_auc_p95": np.nan}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        yp = yy.copy()
        if within_source:
            for _, idx in src.groupby(src).groups.items():
                arr = yp.loc[idx].to_numpy().copy()
                rng.shuffle(arr)
                yp.loc[idx] = arr
        else:
            arr = yp.to_numpy().copy()
            rng.shuffle(arr)
            yp.iloc[:] = arr
        if yp.nunique() == 2:
            try:
                vals.append(float(roc_auc_score(yp, ss)))
            except ValueError:
                pass
    arr = np.asarray(vals)
    if len(arr) == 0:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan, "null_auc_mean": np.nan, "null_auc_p95": np.nan}
    return {
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)),
        "null_auc_mean": float(arr.mean()),
        "null_auc_p95": float(np.quantile(arr, 0.95)),
    }


def balanced_bootstrap(y: pd.Series, score: pd.Series, sources: pd.Series, seed: int, n_boot: int) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & score.notna() & sources.notna()
    yy = y[valid].astype(int)
    ss = score[valid]
    src = sources[valid]
    groups = [idx.to_numpy() for _, idx in src.groupby(src).groups.items() if len(idx) > 0]
    if not groups:
        return {"n_bootstrap": 0}
    n_per_source = min(len(g) for g in groups)
    if n_per_source < 1:
        return {"n_bootstrap": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        picked = np.concatenate([rng.choice(g, size=n_per_source, replace=True) for g in groups])
        by = yy.loc[picked]
        bs = ss.loc[picked]
        if by.nunique() == 2 and bs.nunique() > 1:
            try:
                vals.append(float(roc_auc_score(by, bs)))
            except ValueError:
                pass
    arr = np.asarray(vals)
    if len(arr) == 0:
        return {"n_bootstrap": 0, "balanced_auc_mean": np.nan, "balanced_auc_p05": np.nan, "balanced_auc_p95": np.nan, "n_per_source": int(n_per_source)}
    return {
        "n_bootstrap": int(len(arr)),
        "n_per_source": int(n_per_source),
        "balanced_auc_mean": float(arr.mean()),
        "balanced_auc_p05": float(np.quantile(arr, 0.05)),
        "balanced_auc_p95": float(np.quantile(arr, 0.95)),
    }


def influence_rows(df: pd.DataFrame, axes: List[str], targets: List[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        y = numeric(df, target)
        for axis in axes:
            x = numeric(df, axis)
            full = metric(y, x, df["source_stem"], target, axis, "raw")
            full_auc = full.get("oriented_auc", np.nan)
            score, _ = orient_score(y, x)
            for source in sorted(df["source_stem"].dropna().unique()):
                keep = df["source_stem"] != source
                sub = metric(y[keep], x[keep], df.loc[keep, "source_stem"], target, axis, "drop_source")
                rows.append({
                    "target": target,
                    "axis": axis,
                    "dropped_source": source,
                    "full_oriented_auc": full_auc,
                    "drop_source_oriented_auc": sub.get("oriented_auc"),
                    "delta_auc_minus_full": sub.get("oriented_auc") - full_auc if np.isfinite(sub.get("oriented_auc", np.nan)) and np.isfinite(full_auc) else np.nan,
                    "n_after_drop": sub.get("n"),
                    "n_positive_after_drop": sub.get("n_positive"),
                })
    return pd.DataFrame(rows).sort_values(["target", "axis", "delta_auc_minus_full"])


def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, sub in df.groupby("source_stem"):
        row = {
            "source_stem": src,
            "n_rows": int(len(sub)),
            "n_cycles": int(sub["cycleNo"].nunique()),
        }
        for target in TARGETS:
            if target in sub.columns:
                row[f"{target}_rate"] = float(pd.to_numeric(sub[target], errors="coerce").mean())
                row[f"{target}_n_positive"] = int(pd.to_numeric(sub[target], errors="coerce").fillna(0).sum())
        for axis in AXES:
            if axis in sub.columns:
                row[f"median_{axis}"] = float(pd.to_numeric(sub[axis], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/signed_loss_source_robustness_audit")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_csv(Path(args.derived_dir) / "signed_optical_loss_mechanism_audit" / "signed_optical_loss_axis_scores.csv")
    axes = [a for a in AXES if a in df.columns]
    targets = [t for t in TARGETS if t in df.columns]
    df = df[df["source_stem"].notna()].copy()

    transformed = df.copy()
    transforms: Dict[str, Dict[str, pd.Series]] = {}
    for axis in axes:
        transforms[axis] = {
            "raw": numeric(df, axis),
            "within_source_rank": within_source_rank(numeric(df, axis), df["source_stem"]),
            "within_source_centered_z": within_source_centered_z(numeric(df, axis), df["source_stem"]),
            "source_residual": source_residual(numeric(df, axis), df["source_stem"]),
            "source_mean_only": source_mean_value(numeric(df, axis), df["source_stem"]),
        }
        for name, vals in transforms[axis].items():
            transformed[f"{axis}__{name}"] = vals

    metric_rows: List[Dict[str, Any]] = []
    null_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    for target in targets:
        y = numeric(df, target)
        for axis in axes:
            for transform_name, vals in transforms[axis].items():
                row = metric(y, vals, df["source_stem"], target, axis, transform_name)
                score, _ = orient_score(y, vals)
                if target == "future_any_drop_within_16cycles" and transform_name in {"raw", "within_source_rank", "within_source_centered_z", "source_residual", "source_mean_only"}:
                    global_null = permutation_p(y, score, df["source_stem"], row["oriented_auc"], args.seed, args.n_permutation, within_source=False)
                    within_null = permutation_p(y, score, df["source_stem"], row["oriented_auc"], args.seed + 1, args.n_permutation, within_source=True)
                    null_rows.append({"target": target, "axis": axis, "transform": transform_name, "null_type": "global_label_shuffle", **global_null})
                    null_rows.append({"target": target, "axis": axis, "transform": transform_name, "null_type": "within_source_label_shuffle", **within_null})
                boot = balanced_bootstrap(y, score, df["source_stem"], args.seed + 2, args.n_bootstrap)
                bootstrap_rows.append({"target": target, "axis": axis, "transform": transform_name, **boot})
                metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows).sort_values(["target", "axis", "transform"])
    nulls = pd.DataFrame(null_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    influence = influence_rows(df, axes, targets)
    sources = source_summary(df)

    # Merge compact robustness columns for summary ranking.
    robust = metrics.merge(bootstraps, on=["target", "axis", "transform"], how="left")
    if not nulls.empty:
        null_wide = nulls.pivot_table(index=["target", "axis", "transform"], columns="null_type", values="empirical_p_ge_observed", aggfunc="first").reset_index()
        robust = robust.merge(null_wide, on=["target", "axis", "transform"], how="left")

    paths = {
        "transformed_scores": out / "signed_loss_source_robustness_scores.csv",
        "metrics": out / "signed_loss_source_robustness_metrics.csv",
        "permutation_nulls": out / "signed_loss_source_robustness_permutation_nulls.csv",
        "balanced_bootstrap": out / "signed_loss_source_robustness_balanced_bootstrap.csv",
        "source_influence": out / "signed_loss_source_robustness_source_influence.csv",
        "source_summary": out / "signed_loss_source_robustness_source_summary.csv",
        "summary": out / "signed_loss_source_robustness_summary.json",
    }
    transformed.to_csv(paths["transformed_scores"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    nulls.to_csv(paths["permutation_nulls"], index=False)
    bootstraps.to_csv(paths["balanced_bootstrap"], index=False)
    influence.to_csv(paths["source_influence"], index=False)
    sources.to_csv(paths["source_summary"], index=False)

    future16 = robust[robust["target"] == "future_any_drop_within_16cycles"].copy()
    key_rows = future16[future16["axis"].isin(["signed_optical_loss_axis", "combined_loss_mechanism_axis", "echem_degraded_state_axis"])]
    top_influence = influence[influence["target"] == "future_any_drop_within_16cycles"].sort_values("delta_auc_minus_full").head(20)
    summary = clean_json({
        "n_rows": int(len(df)),
        "n_labeled_future16": int(numeric(df, "future_any_drop_within_16cycles").isin([0, 1]).sum()) if "future_any_drop_within_16cycles" in df else 0,
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "axes": axes,
        "targets": targets,
        "transforms": ["raw", "within_source_rank", "within_source_centered_z", "source_residual", "source_mean_only"],
        "key_future16_metrics": key_rows.sort_values(["axis", "oriented_auc"], ascending=[True, False]).to_dict("records"),
        "top_future16_metrics": future16.sort_values("oriented_auc", ascending=False).head(30).to_dict("records"),
        "largest_negative_source_influence": top_influence.to_dict("records"),
        "source_summary": sources.to_dict("records"),
        "guardrail": "Source robustness transforms distinguish within-source signal from source-level composition. High raw AUC with high source-mean-only AUC or weak within-source permutation evidence should be treated as source/context-sensitive review evidence, not a source-independent detector.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Signed Loss Source Robustness Audit\n\n"
        "Tests signed optical-loss axes under within-source rank/centering, source-mean-only, source-balanced bootstrap, source-macro AUC, and label-permutation controls.\n\n"
        f"- Rows: {summary['n_rows']}\n- Future16 labeled rows: {summary['n_labeled_future16']}\n- Cycles: {summary['n_cycles']}\n- Sources: {summary['n_sources']}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
