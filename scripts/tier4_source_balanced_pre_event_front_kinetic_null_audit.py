#!/usr/bin/env python3
"""Source-aware null audit for pre-event front/kinetic concordance.

The front/kinetic concordance score is useful only if it is not simply a
source-composition artifact. This audit takes the concordance ranked table and
tests raw, source-residual, within-source-rank, and source-balanced bootstrap
readouts against event-relative targets, then compares them with source-
stratified label permutations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
FEATURES = [
    "front_kinetic_concordance_score",
    "kinetic_evidence_score",
    "front_evidence_score",
    "qc_evidence_score",
    "front_kinetic_product",
    "masked_minus_bg_slope",
    "q75_phase_fraction_slope",
    "front_radius2_slope_px2_per_norm_time_source_echem_context_residual",
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


def source_residual(x: pd.Series, sources: pd.Series) -> pd.Series:
    return x - x.groupby(sources.astype(str)).transform("mean")


def within_source_rank(x: pd.Series, sources: pd.Series) -> pd.Series:
    return x.groupby(sources.astype(str)).rank(pct=True, method="average") - 0.5


def oriented_metrics(y: pd.Series, x: pd.Series) -> Dict[str, Any] | None:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid].astype(float)
    if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
        return None
    direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
    score = xx if direction == "higher_in_positive" else -xx
    try:
        _, p_mwu = mannwhitneyu(xx[yy == 1], xx[yy == 0], alternative="two-sided")
    except ValueError:
        p_mwu = np.nan
    return {
        "n": int(len(yy)),
        "n_positive": int(yy.sum()),
        "direction": direction,
        "oriented_auc": float(roc_auc_score(yy, score)),
        "average_precision": float(average_precision_score(yy, score)),
        "median_positive_minus_negative": float(xx[yy == 1].median() - xx[yy == 0].median()),
        "mwu_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
    }


def permute_within_sources(y: pd.Series, sources: pd.Series, rng: np.random.Generator) -> pd.Series:
    yp = y.copy()
    for _, idx in y.groupby(sources.astype(str)).groups.items():
        vals = yp.loc[idx].to_numpy()
        finite = np.isfinite(vals)
        vals2 = vals.copy()
        vals2[finite] = rng.permutation(vals[finite])
        yp.loc[idx] = vals2
    return yp


def source_balanced_auc(y: pd.Series, x: pd.Series, sources: pd.Series, rng: np.random.Generator, n_boot: int) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & x.notna() & sources.notna()
    sub = pd.DataFrame({"y": y[valid].astype(int), "x": x[valid].astype(float), "source": sources[valid].astype(str)})
    eligible = []
    for src, g in sub.groupby("source"):
        if g["y"].nunique() == 2:
            eligible.append((src, g.index.to_numpy()))
    if len(eligible) < 2:
        return {"n_eligible_sources": len(eligible), "bootstrap_auc_median": np.nan, "bootstrap_auc_p10": np.nan, "bootstrap_auc_p90": np.nan}
    aucs = []
    for _ in range(n_boot):
        rows = []
        chosen = rng.choice(len(eligible), size=len(eligible), replace=True)
        for j in chosen:
            _, idx = eligible[j]
            rows.extend(rng.choice(idx, size=len(idx), replace=True).tolist())
        b = sub.loc[rows]
        if b["y"].nunique() < 2 or b["x"].nunique() < 2:
            continue
        direction = "higher" if b.loc[b["y"] == 1, "x"].median() >= b.loc[b["y"] == 0, "x"].median() else "lower"
        score = b["x"] if direction == "higher" else -b["x"]
        aucs.append(float(roc_auc_score(b["y"], score)))
    if not aucs:
        return {"n_eligible_sources": len(eligible), "bootstrap_auc_median": np.nan, "bootstrap_auc_p10": np.nan, "bootstrap_auc_p90": np.nan}
    arr = np.asarray(aucs, dtype=float)
    return {
        "n_eligible_sources": len(eligible),
        "bootstrap_auc_median": float(np.nanmedian(arr)),
        "bootstrap_auc_p10": float(np.nanpercentile(arr, 10)),
        "bootstrap_auc_p90": float(np.nanpercentile(arr, 90)),
    }


def run_tests(df: pd.DataFrame, features: Iterable[str], n_perm: int, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    for target, (positive, negative) in TARGETS.items():
        y = target_series(df, positive, negative)
        for feature in features:
            x0 = numeric(df, feature)
            transforms = {
                "raw": x0,
                "source_residual": source_residual(x0, sources),
                "within_source_rank": within_source_rank(x0, sources),
            }
            for transform, x in transforms.items():
                obs = oriented_metrics(y, x)
                if obs is None:
                    continue
                null_auc = []
                valid = y.isin([0, 1]) & x.notna()
                for _ in range(n_perm):
                    yp = permute_within_sources(y[valid], sources[valid], rng)
                    perm = oriented_metrics(yp, x[valid])
                    if perm is not None:
                        null_auc.append(perm["oriented_auc"])
                null_arr = np.asarray(null_auc, dtype=float)
                perm_p = float((np.sum(null_arr >= obs["oriented_auc"]) + 1) / (len(null_arr) + 1)) if len(null_arr) else np.nan
                boot = source_balanced_auc(y, x, sources, rng, n_boot)
                rows.append({
                    "target": target,
                    "feature": feature,
                    "transform": transform,
                    **obs,
                    "n_permutations": int(len(null_arr)),
                    "source_stratified_perm_p_auc_ge_observed": perm_p,
                    "null_auc_median": float(np.nanmedian(null_arr)) if len(null_arr) else np.nan,
                    "null_auc_p95": float(np.nanpercentile(null_arr, 95)) if len(null_arr) else np.nan,
                    **boot,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["source_stratified_perm_p_auc_ge_observed", "oriented_auc", "average_precision"],
            ascending=[True, False, False],
        )
    return out


def proximity_tests(df: pd.DataFrame, features: Iterable[str], n_perm: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed + 13)
    pre = df[df["event_relative_bin"].astype(str).isin(["near_pre_event_1_8", "mid_pre_event_9_16", "far_pre_event_17_32"])].copy()
    if pre.empty:
        return pd.DataFrame(rows)
    proximity = -numeric(pre, "cycles_to_next_event")
    sources = pre["source_stem"].astype(str)
    for feature in features:
        for transform, x in {
            "raw": numeric(pre, feature),
            "source_residual": source_residual(numeric(pre, feature), sources),
            "within_source_rank": within_source_rank(numeric(pre, feature), sources),
        }.items():
            valid = x.notna() & proximity.notna()
            if valid.sum() < 8 or x[valid].nunique() < 3 or proximity[valid].nunique() < 3:
                continue
            obs = spearmanr(x[valid], proximity[valid])
            null = []
            for _ in range(n_perm):
                yp = permute_within_sources(proximity[valid], sources[valid], rng)
                stat = spearmanr(x[valid], yp)
                if np.isfinite(stat.statistic):
                    null.append(abs(float(stat.statistic)))
            null_arr = np.asarray(null, dtype=float)
            p = float((np.sum(null_arr >= abs(obs.statistic)) + 1) / (len(null_arr) + 1)) if len(null_arr) else np.nan
            rows.append({
                "feature": feature,
                "transform": transform,
                "n": int(valid.sum()),
                "spearman_rho_vs_event_proximity": float(obs.statistic),
                "spearman_p": float(obs.pvalue),
                "source_stratified_perm_p_abs_rho": p,
                "null_abs_rho_p95": float(np.nanpercentile(null_arr, 95)) if len(null_arr) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["source_stratified_perm_p_abs_rho", "spearman_p"])
    return out


def top_items(df: pd.DataFrame, n: int = 12) -> List[Dict[str, Any]]:
    return df.head(n).to_dict("records") if not df.empty else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_front_kinetic_null_audit")
    parser.add_argument("--n-perm", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()
    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ranked = read_csv(derived / "source_balanced_pre_event_front_kinetic_concordance_audit" / "source_balanced_pre_event_front_kinetic_concordance_ranked_candidates.csv")
    features = [f for f in FEATURES if f in ranked.columns and numeric(ranked, f).notna().sum() >= 8]
    print(json.dumps({"phase": "run_tests_start", "n_features": len(features), "n_perm": args.n_perm, "n_boot": args.n_boot}), flush=True)
    tests = run_tests(ranked, features, args.n_perm, args.n_boot, args.seed)
    print(json.dumps({"phase": "run_tests_done", "n_rows": int(len(tests))}), flush=True)
    prox = proximity_tests(ranked, features, args.n_perm, args.seed)
    print(json.dumps({"phase": "proximity_done", "n_rows": int(len(prox))}), flush=True)
    source_summary = (
        ranked.groupby("source_stem", dropna=False)
        .agg(
            n_roi=("roi_id", "count"),
            near_pre=("event_relative_bin", lambda s: int((s.astype(str) == "near_pre_event_1_8").sum())),
            median_concordance=("front_kinetic_concordance_score", "median"),
            median_kinetic=("kinetic_evidence_score", "median"),
            median_front=("front_evidence_score", "median"),
        )
        .reset_index()
        .sort_values(["near_pre", "median_concordance"], ascending=[False, False])
    )

    paths = {
        "null_tests": out / "source_balanced_pre_event_front_kinetic_null_tests.csv",
        "proximity_tests": out / "source_balanced_pre_event_front_kinetic_proximity_tests.csv",
        "source_summary": out / "source_balanced_pre_event_front_kinetic_null_source_summary.csv",
        "summary": out / "source_balanced_pre_event_front_kinetic_null_summary.json",
    }
    tests.to_csv(paths["null_tests"], index=False)
    prox.to_csv(paths["proximity_tests"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)
    summary = {
        "n_rows": int(len(ranked)),
        "n_sources": int(ranked["source_stem"].nunique()),
        "n_features_tested": int(len(features)),
        "n_permutations": int(args.n_perm),
        "n_bootstrap": int(args.n_boot),
        "top_null_tests": top_items(tests, 20),
        "top_proximity_tests": top_items(prox, 20),
        "source_summary": source_summary.to_dict("records"),
        "guardrail": "Source-aware null tests audit whether front/kinetic concordance survives source-stratified label shuffles and source-balanced resampling. Passing rows remain automatic review-prioritization evidence, not manual labels, causal mechanisms, deployable warnings, or calibrated diffusion coefficients.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Source-Balanced Pre-Event Front/Kinetic Null Audit",
        "",
        "Source-aware null tests for the front/kinetic concordance score and component features.",
        "",
        f"- Rows: {summary['n_rows']}",
        f"- Sources: {summary['n_sources']}",
        f"- Features tested: {summary['n_features_tested']}",
        f"- Source-stratified permutations per test: {summary['n_permutations']}",
        f"- Source-balanced bootstraps per test: {summary['n_bootstrap']}",
        "",
        "## Top Null Tests",
    ]
    for row in summary["top_null_tests"][:8]:
        if summary["n_bootstrap"] > 0 and row.get("bootstrap_auc_median") is not None:
            boot_txt = f", boot AUC median={row.get('bootstrap_auc_median'):.3f}"
        else:
            boot_txt = ", bootstrap disabled"
        lines.append(
            f"- {row.get('target')} {row.get('transform')} {row.get('feature')}: AUC={row.get('oriented_auc'):.3f}, perm p={row.get('source_stratified_perm_p_auc_ge_observed'):.4g}{boot_txt}"
        )
    lines += ["", "## Top Proximity Tests"]
    for row in summary["top_proximity_tests"][:6]:
        lines.append(
            f"- {row.get('transform')} {row.get('feature')}: rho={row.get('spearman_rho_vs_event_proximity'):.3f}, perm p={row.get('source_stratified_perm_p_abs_rho'):.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out), "top_null_tests": summary["top_null_tests"][:3], "top_proximity_tests": summary["top_proximity_tests"][:3]}, indent=2))


if __name__ == "__main__":
    main()
