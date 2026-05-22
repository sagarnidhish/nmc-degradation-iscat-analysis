#!/usr/bin/env python3
"""Sparse invariant physics-rule discovery for NMC ROI degradation.

This audit searches for small, interpretable if/then rules that separate weak
future degradation labels under leave-source evaluation. It is deliberately not
a deployable warning model: rules are used as review-prioritization hypotheses
and are stress-tested for source concentration, acquisition confounding, and
automatic-mask/front guardrails.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


TARGET = "future_any_drop_within_16cycles"


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


def numeric_cols(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 24) -> List[str]:
    keep: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 3:
            keep.append(col)
    return keep


def candidate_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    particle = numeric_cols(
        df,
        [c for c in df.columns if c.startswith(("particle_norm_", "particle_mean_", "particle_std_", "particle_gradient_", "particle_vs_context_"))],
    )
    front = numeric_cols(
        df,
        [
            "phase_slope_median_per_s",
            "phase_slope_abs_median_per_s",
            "phase_slope_positive_fraction",
            "radius2_slope_median_px2_per_s",
            "radius2_slope_positive_fraction",
            "diffusion_proxy_median_um2_per_s",
            "diffusion_proxy_abs_median_um2_per_s",
            "threshold_robust_phase_score",
            "threshold_robust_diffusion_score",
            "q70_phase_slope_bootstrap_p50",
            "q70_radius2_slope_bootstrap_p50_px2_per_s",
        ],
    )
    rollout = numeric_cols(
        df,
        [
            "transferred_masked_residual_signature",
            "persistence_particle_mse_fraction_of_full_mean",
            "low_rank_dmd_particle_mse_fraction_of_full_mean",
            "velocity_particle_mse_fraction_of_full_mean",
            "persistence_particle_to_nonparticle_mse_ratio_mean",
            "low_rank_dmd_particle_to_nonparticle_mse_ratio_mean",
            "velocity_particle_to_nonparticle_mse_ratio_mean",
            "object_mean_residual",
            "object_mean_abs_z",
            "roi_norm_mean_delta_last_minus_first",
        ],
    )
    echem = numeric_cols(
        df,
        [c for c in df.columns if c.startswith(("shape_", "all_dq_", "pos_dq_", "neg_dq_", "cycle_state_", "echem_regime_"))]
        + [
            "capacity_fraction_of_first",
            "coulombic_inefficiency_pct",
            "charge_discharge_capacity_abs_gap_mAh",
            "voltage_peak_hysteresis_proxy",
            "dqdv_entropy_asymmetry",
            "dqdv_integral_asymmetry",
            "degradation_state_axis",
            "state_step_norm",
            "axis_step",
            "frames_percentile",
            "cycle_index_rank",
        ],
    )
    acquisition = numeric_cols(
        df,
        [
            "cycleNo",
            "local_cycle_index",
            "n_frames",
            "n_frames_echem",
            "frames_percentile",
            "cycle_gap",
            "first_frame_index",
            "last_frame_index",
            "balanced_cycle_rank",
        ],
    )
    all_physics = sorted(set(particle + front + rollout + echem))
    return {
        "particle_optical": particle,
        "front_rollout": sorted(set(front + rollout)),
        "echem_state": echem,
        "all_interpretable": all_physics,
        "acquisition_context": acquisition,
    }


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


def oriented_feature_tests(df: pd.DataFrame, cols: List[str], target: str) -> pd.DataFrame:
    y = pd.to_numeric(df[target], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce")
        valid = y.isin([0, 1]) & x.notna()
        if valid.sum() < 20 or y[valid].nunique() < 2:
            continue
        yy = y[valid].astype(int)
        xx = x[valid]
        pos = xx[yy == 1]
        neg = xx[yy == 0]
        direction = "high" if pos.median() >= neg.median() else "low"
        score = xx if direction == "high" else -xx
        try:
            auc = float(roc_auc_score(yy, score))
            ap = float(average_precision_score(yy, score))
        except ValueError:
            auc = float("nan")
            ap = float("nan")
        try:
            mw = mannwhitneyu(pos, neg, alternative="two-sided")
            pval = float(mw.pvalue)
        except ValueError:
            pval = float("nan")
        rho, rho_p = spearmanr(yy, score)
        rows.append(
            {
                "feature": col,
                "direction": direction,
                "n": int(valid.sum()),
                "n_positive": int(yy.sum()),
                "median_positive": float(pos.median()),
                "median_negative": float(neg.median()),
                "oriented_auc": auc,
                "average_precision": ap,
                "mannwhitney_p": pval,
                "spearman_rho_oriented": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p_oriented": float(rho_p) if np.isfinite(rho_p) else np.nan,
                "source_eta2": source_eta2(x, df["source_stem"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["oriented_auc", "average_precision"], ascending=[False, False])


def threshold_from_train(x: pd.Series, direction: str, quantile: float) -> float:
    q = quantile if direction == "high" else (1.0 - quantile)
    return float(pd.to_numeric(x, errors="coerce").quantile(q))


def apply_rule(df: pd.DataFrame, terms: List[Tuple[str, str]], thresholds: Dict[str, float]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for feature, direction in terms:
        vals = pd.to_numeric(df[feature], errors="coerce")
        thr = thresholds[feature]
        if direction == "high":
            mask &= vals >= thr
        else:
            mask &= vals <= thr
    return mask.fillna(False)


def metric_from_binary(y: pd.Series, pred: pd.Series) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & pred.notna()
    yy = y[valid].astype(int)
    pp = pred[valid].astype(int)
    n = int(len(yy))
    positives = int(yy.sum())
    covered = int(pp.sum())
    covered_pos = int(((pp == 1) & (yy == 1)).sum())
    uncovered_pos = int(((pp == 0) & (yy == 1)).sum())
    covered_neg = int(((pp == 1) & (yy == 0)).sum())
    uncovered_neg = int(((pp == 0) & (yy == 0)).sum())
    precision = covered_pos / covered if covered else np.nan
    recall = covered_pos / positives if positives else np.nan
    base_rate = positives / n if n else np.nan
    lift = precision / base_rate if covered and base_rate else np.nan
    try:
        _, fisher_p = fisher_exact([[covered_pos, covered_neg], [uncovered_pos, uncovered_neg]], alternative="greater")
    except ValueError:
        fisher_p = np.nan
    auc = float(roc_auc_score(yy, pp)) if n >= 8 and yy.nunique() == 2 and pp.nunique() == 2 else np.nan
    return {
        "n_eval": n,
        "n_positive": positives,
        "n_covered": covered,
        "n_covered_positive": covered_pos,
        "coverage": covered / n if n else np.nan,
        "precision": precision,
        "recall": recall,
        "base_rate": base_rate,
        "lift": lift,
        "binary_auc": auc,
        "fisher_p_greater": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
    }


def leave_source_rule_eval(
    df: pd.DataFrame,
    terms: List[Tuple[str, str]],
    target: str,
    quantile: float,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    fold_rows: List[pd.DataFrame] = []
    source_rows: List[Dict[str, Any]] = []
    for source in sorted(df.loc[valid, "source_stem"].dropna().unique()):
        train = valid & (df["source_stem"] != source)
        test = valid & (df["source_stem"] == source)
        thresholds = {feature: threshold_from_train(df.loc[train, feature], direction, quantile) for feature, direction in terms}
        pred = apply_rule(df.loc[test], terms, thresholds)
        fold = df.loc[test, ["embedding_row_id", "roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
        fold["rule_hit"] = pred.to_numpy(dtype=bool)
        fold["heldout_source"] = source
        for feature, threshold in thresholds.items():
            fold[f"threshold__{feature}"] = threshold
        fold_rows.append(fold)
        met = metric_from_binary(pd.to_numeric(fold["observed"], errors="coerce"), fold["rule_hit"])
        met.update({"heldout_source": source, "n_cycles": int(fold["cycleNo"].nunique())})
        source_rows.append(met)
    pred_df = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame()
    source_df = pd.DataFrame(source_rows)
    pooled = metric_from_binary(pd.to_numeric(pred_df["observed"], errors="coerce"), pred_df["rule_hit"]) if not pred_df.empty else {}
    pooled.update(
        {
            "n_sources": int(pred_df["heldout_source"].nunique()) if not pred_df.empty else 0,
            "n_sources_with_hits": int((source_df.get("n_covered", pd.Series(dtype=int)) > 0).sum()) if not source_df.empty else 0,
            "n_sources_with_positive_hits": int((source_df.get("n_covered_positive", pd.Series(dtype=int)) > 0).sum()) if not source_df.empty else 0,
            "min_source_precision_with_hits": float(source_df.loc[source_df["n_covered"] > 0, "precision"].min()) if not source_df.empty and (source_df["n_covered"] > 0).any() else np.nan,
            "median_source_precision_with_hits": float(source_df.loc[source_df["n_covered"] > 0, "precision"].median()) if not source_df.empty and (source_df["n_covered"] > 0).any() else np.nan,
        }
    )
    return pooled, pred_df, source_df


def discover_rules(df: pd.DataFrame, tests: pd.DataFrame, target: str, quantile: float, max_pairs: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top = tests[(tests["oriented_auc"].notna()) & (tests["source_eta2"].fillna(0) <= 0.90)].head(max_pairs).copy()
    candidates: List[Tuple[str, List[Tuple[str, str]]]] = []
    for _, row in top.iterrows():
        candidates.append((row["feature"], [(row["feature"], row["direction"])]))
    for (_, a), (_, b) in itertools.combinations(top.iterrows(), 2):
        if a["feature"] == b["feature"]:
            continue
        name = f"{a['feature']}__AND__{b['feature']}"
        candidates.append((name, [(a["feature"], a["direction"]), (b["feature"], b["direction"])]))

    rule_rows: List[Dict[str, Any]] = []
    all_preds: List[pd.DataFrame] = []
    all_sources: List[pd.DataFrame] = []
    for rule_name, terms in candidates:
        pooled, pred, source = leave_source_rule_eval(df, terms, target, quantile)
        max_eta = max(source_eta2(df[feature], df["source_stem"]) for feature, _ in terms)
        row = dict(pooled)
        row.update(
            {
                "rule_name": rule_name,
                "n_terms": len(terms),
                "terms": " AND ".join([f"{direction}({feature})" for feature, direction in terms]),
                "features": ";".join([feature for feature, _ in terms]),
                "directions": ";".join([direction for _, direction in terms]),
                "quantile": quantile,
                "max_feature_source_eta2": float(max_eta) if np.isfinite(max_eta) else np.nan,
            }
        )
        rule_rows.append(row)
        if not pred.empty:
            pred["rule_name"] = rule_name
            pred["terms"] = row["terms"]
            all_preds.append(pred)
        if not source.empty:
            source["rule_name"] = rule_name
            source["terms"] = row["terms"]
            all_sources.append(source)
    rules = pd.DataFrame(rule_rows)
    if not rules.empty:
        rules["priority_score"] = (
            rules["lift"].fillna(0)
            * np.sqrt(rules["n_covered"].clip(lower=0).fillna(0))
            * (rules["n_sources_with_positive_hits"].fillna(0) / rules["n_sources"].replace(0, np.nan).fillna(1))
            * (1.0 - rules["max_feature_source_eta2"].fillna(1.0).clip(0, 1) * 0.25)
        )
        rules = rules.sort_values(["priority_score", "precision", "recall"], ascending=[False, False, False])
    return (
        rules,
        pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(),
        pd.concat(all_sources, ignore_index=True) if all_sources else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/invariant_physics_rule_discovery")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--max-pair-features", type=int, default=18)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = read_csv(derived / "echem_video_embedding_fusion_audit" / "echem_video_embedding_fusion_joined.csv")
    families = candidate_features(df)
    target = args.target
    y = pd.to_numeric(df[target], errors="coerce")
    eval_df = df[y.isin([0, 1]) & df["source_stem"].notna()].copy()
    feature_tests = oriented_feature_tests(eval_df, families["all_interpretable"], target)
    rule_df, pred_df, source_df = discover_rules(eval_df, feature_tests, target, args.quantile, args.max_pair_features)

    paths = {
        "feature_screen": out / "invariant_rule_feature_screen.csv",
        "rules": out / "invariant_physics_rules.csv",
        "predictions": out / "invariant_physics_rule_predictions.csv",
        "source_support": out / "invariant_physics_rule_source_support.csv",
        "summary": out / "invariant_physics_rule_summary.json",
    }
    feature_tests.to_csv(paths["feature_screen"], index=False)
    rule_df.to_csv(paths["rules"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    source_df.to_csv(paths["source_support"], index=False)

    top_rules = rule_df.head(20).to_dict("records") if not rule_df.empty else []
    top_features = feature_tests.head(30).to_dict("records") if not feature_tests.empty else []
    best = rule_df.iloc[0].to_dict() if not rule_df.empty else {}
    summary = clean_json(
        {
            "n_rows": int(len(df)),
            "n_eval_rows": int(len(eval_df)),
            "n_cycles": int(eval_df["cycleNo"].nunique()) if "cycleNo" in eval_df else 0,
            "n_sources": int(eval_df["source_stem"].nunique()) if "source_stem" in eval_df else 0,
            "target": target,
            "positive_rate": float(pd.to_numeric(eval_df[target], errors="coerce").mean()) if len(eval_df) else None,
            "quantile": args.quantile,
            "feature_family_sizes": {k: len(v) for k, v in families.items()},
            "n_candidate_rules": int(len(rule_df)),
            "best_rule": best,
            "top_rules": top_rules,
            "top_oriented_features": top_features,
            "guardrail": "Rules use automatic ROI masks, weak future labels, and data-derived thresholds under leave-source evaluation. They are sparse review-prioritization hypotheses only; source/outcome imbalance, acquisition coupling, missing manual QC, and uncalibrated diffusion/front proxies remain hard claim limits.",
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Invariant Physics Rule Discovery\n\n"
        "Sparse leave-source if/then rule mining over interpretable NMC particle, front, rollout, and echem descriptors.\n\n"
        f"- Evaluation rows: {summary['n_eval_rows']}\n"
        f"- Cycles/sources: {summary['n_cycles']} / {summary['n_sources']}\n"
        f"- Target: {target}\n"
        f"- Candidate rules: {summary['n_candidate_rules']}\n"
        f"- Best rule: {best.get('terms', 'none')}\n"
        f"- Best precision/recall/lift: {best.get('precision')} / {best.get('recall')} / {best.get('lift')}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
