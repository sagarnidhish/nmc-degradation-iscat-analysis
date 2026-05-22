#!/usr/bin/env python3
"""Source-normalized echem/optical residual audit.

Signed optical-loss axes are predictive but source-structured. The source
robustness audit showed the echem-degraded axis survives source residualization
better than optical rank features. This script asks whether source-normalized
echem state and source-normalized optical-loss axes provide complementary weak
future-drop evidence, and emits guarded review-prioritization rules/candidates.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
BASE_AXES = [
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


def orient_score(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, str]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    if len(yy) < 2 or yy.nunique() < 2:
        return x, "NA"
    if xx[yy == 1].median() >= xx[yy == 0].median():
        return x, "higher_in_positive"
    return -x, "lower_in_positive"


def metric(y: pd.Series, x: pd.Series, sources: pd.Series, target: str, feature_set: str, mode: str) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & x.notna() & sources.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    score, direction = orient_score(yy, xx)
    ss = score[valid]
    out: Dict[str, Any] = {
        "target": target,
        "feature_set": feature_set,
        "mode": mode,
        "n": int(len(yy)),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "n_sources": int(sources[valid].nunique()) if len(yy) else 0,
        "direction": direction,
        "oriented_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "source_eta2": source_eta2(xx, sources[valid]) if len(yy) else np.nan,
    }
    if len(yy) >= 8 and yy.nunique() == 2 and ss.nunique() > 1:
        out["oriented_auc"] = float(roc_auc_score(yy, ss))
        out["average_precision"] = float(average_precision_score(yy, ss))
        rho, p = spearmanr(yy, ss)
        out["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
        out["spearman_p"] = float(p) if np.isfinite(p) else np.nan
    return out


def leave_source_model(df: pd.DataFrame, cols: List[str], target: str, feature_set: str, mode: str, seed: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    y = numeric(df, target)
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    rows = []
    for source in sorted(df.loc[valid, "source_stem"].dropna().unique()):
        test = valid & (df["source_stem"] == source)
        train = valid & ~test
        meta = df.loc[test, ["embedding_row_id", "roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
        if train.sum() < 12 or y[train].nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped"
        else:
            imp = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            xtr = scaler.fit_transform(imp.fit_transform(df.loc[train, cols]))
            xte = scaler.transform(imp.transform(df.loc[test, cols]))
            model = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.25, solver="liblinear", random_state=seed)
            model.fit(xtr, y[train].astype(int))
            meta["predicted_probability"] = model.predict_proba(xte)[:, 1]
            meta["status"] = "ok"
        meta["target"] = target
        meta["feature_set"] = feature_set
        meta["mode"] = mode
        meta["features"] = ";".join(cols)
        meta["heldout_source"] = source
        rows.extend(meta.to_dict("records"))
    pred = pd.DataFrame(rows)
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    yy = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    pp = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    out: Dict[str, Any] = {
        "target": target,
        "feature_set": feature_set,
        "mode": mode,
        "features": ";".join(cols),
        "n_eval": int(len(tmp)),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "n_sources": int(tmp["heldout_source"].nunique()) if len(tmp) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and yy.nunique() == 2 and pp.nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(yy, pp))
        out["average_precision"] = float(average_precision_score(yy, pp))
        rho, p = spearmanr(yy, pp)
        out["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
        out["spearman_p"] = float(p) if np.isfinite(p) else np.nan
    return out, pred


def permutation_p(y: pd.Series, score: pd.Series, sources: pd.Series, observed_auc: float, seed: int, n_perm: int, within_source: bool) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & score.notna() & sources.notna()
    yy = y[valid].astype(int).copy()
    ss = score[valid]
    src = sources[valid]
    if len(yy) < 8 or yy.nunique() < 2 or not np.isfinite(observed_auc):
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
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
            vals.append(float(roc_auc_score(yp, ss)))
    arr = np.asarray(vals)
    if len(arr) == 0:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan}
    return {
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)),
        "null_auc_mean": float(arr.mean()),
        "null_auc_p95": float(np.quantile(arr, 0.95)),
    }


def rule_metrics(df: pd.DataFrame, target: str, echem_col: str, optical_col: str) -> pd.DataFrame:
    y = numeric(df, target)
    e = numeric(df, echem_col)
    o = numeric(df, optical_col)
    valid = y.isin([0, 1]) & e.notna() & o.notna()
    rows = []
    quantiles = [0.50, 0.60, 0.70, 0.75]
    for eq, oq in itertools.product(quantiles, quantiles):
        eth = e[valid].quantile(eq)
        oth = o[valid].quantile(oq)
        for name, mask in {
            f"echem_ge_q{int(eq*100)}": e >= eth,
            f"optical_ge_q{int(oq*100)}": o >= oth,
            f"echem_ge_q{int(eq*100)}_AND_optical_ge_q{int(oq*100)}": (e >= eth) & (o >= oth),
            f"echem_ge_q{int(eq*100)}_OR_optical_ge_q{int(oq*100)}": (e >= eth) | (o >= oth),
        }.items():
            m = valid & mask
            yy = y[valid].astype(int)
            pred = mask[valid].astype(int)
            covered = int(pred.sum())
            if covered == 0:
                continue
            covered_pos = int(((pred == 1) & (yy == 1)).sum())
            uncovered_pos = int(((pred == 0) & (yy == 1)).sum())
            covered_neg = int(((pred == 1) & (yy == 0)).sum())
            uncovered_neg = int(((pred == 0) & (yy == 0)).sum())
            precision = covered_pos / covered if covered else np.nan
            recall = covered_pos / int(yy.sum()) if yy.sum() else np.nan
            base = float(yy.mean())
            try:
                _, fisher_p = fisher_exact([[covered_pos, covered_neg], [uncovered_pos, uncovered_neg]], alternative="greater")
            except ValueError:
                fisher_p = np.nan
            try:
                auc = float(roc_auc_score(yy, pred)) if pred.nunique() > 1 and yy.nunique() == 2 else np.nan
            except ValueError:
                auc = np.nan
            rows.append({
                "target": target,
                "rule": name,
                "echem_quantile": eq,
                "optical_quantile": oq,
                "n_eval": int(valid.sum()),
                "n_covered": covered,
                "n_covered_positive": covered_pos,
                "precision": precision,
                "recall": recall,
                "base_rate": base,
                "lift": precision / base if base else np.nan,
                "binary_auc": auc,
                "fisher_p_greater": float(fisher_p) if np.isfinite(fisher_p) else np.nan,
                "n_sources_with_hits": int(df.loc[m, "source_stem"].nunique()),
                "n_sources_with_positive_hits": int(df.loc[m & (y == 1), "source_stem"].nunique()),
            })
    return pd.DataFrame(rows).sort_values(["fisher_p_greater", "lift", "n_sources_with_positive_hits"], ascending=[True, False, False])


def candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "embedding_row_id", "roi_id", "cycleNo", "source_stem", "future_any_drop_within_8cycles", "future_any_drop_within_16cycles",
        "echem_degraded_state_axis__source_residual", "signed_optical_loss_axis__source_residual", "combined_loss_mechanism_axis__source_residual",
        "echem_degraded_state_axis__within_source_rank", "signed_optical_loss_axis__within_source_rank",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out["source_residual_echem_optical_score"] = numeric(out, "echem_degraded_state_axis__source_residual") + numeric(out, "signed_optical_loss_axis__source_residual")
    return out.sort_values("source_residual_echem_optical_score", ascending=False).head(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_optical_source_residual_audit")
    parser.add_argument("--seed", type=int, default=149)
    parser.add_argument("--n-permutation", type=int, default=500)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_csv(derived / "signed_loss_source_robustness_audit" / "signed_loss_source_robustness_scores.csv")

    targets = [t for t in TARGETS if t in df.columns]
    feature_sets = {
        "echem_source_residual": ["echem_degraded_state_axis__source_residual"],
        "optical_source_residual": ["signed_optical_loss_axis__source_residual"],
        "front_source_residual": ["front_contraction_axis__source_residual"],
        "echem_plus_optical_source_residual": ["echem_degraded_state_axis__source_residual", "signed_optical_loss_axis__source_residual"],
        "echem_plus_optical_rank": ["echem_degraded_state_axis__within_source_rank", "signed_optical_loss_axis__within_source_rank"],
        "echem_plus_optical_centered_z": ["echem_degraded_state_axis__within_source_centered_z", "signed_optical_loss_axis__within_source_centered_z"],
        "echem_optical_front_residual": ["echem_degraded_state_axis__source_residual", "signed_optical_loss_axis__source_residual", "front_contraction_axis__source_residual"],
        "all_source_residual_axes": [f"{axis}__source_residual" for axis in BASE_AXES],
        "raw_axes_guardrail": BASE_AXES,
        "source_mean_only_guardrail": [f"{axis}__source_mean_only" for axis in BASE_AXES],
    }
    feature_sets = {k: [c for c in v if c in df.columns] for k, v in feature_sets.items()}

    direct_rows = []
    null_rows = []
    for target in targets:
        y = numeric(df, target)
        for name, cols in feature_sets.items():
            if len(cols) == 1:
                x = numeric(df, cols[0])
            else:
                x = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            row = metric(y, x, df["source_stem"], target, name, "direct_mean_score")
            score, _ = orient_score(y, x)
            if target == "future_any_drop_within_16cycles":
                null_rows.append({"target": target, "feature_set": name, "null_type": "global_label_shuffle", **permutation_p(y, score, df["source_stem"], row["oriented_auc"], args.seed, args.n_permutation, False)})
                null_rows.append({"target": target, "feature_set": name, "null_type": "within_source_label_shuffle", **permutation_p(y, score, df["source_stem"], row["oriented_auc"], args.seed + 1, args.n_permutation, True)})
            direct_rows.append(row)

    model_rows = []
    pred_frames = []
    for target in targets:
        for name, cols in feature_sets.items():
            if not cols:
                continue
            row, pred = leave_source_model(df, cols, target, name, "leave_source_logistic", args.seed)
            model_rows.append(row)
            pred_frames.append(pred)

    direct = pd.DataFrame(direct_rows).sort_values(["target", "oriented_auc"], ascending=[True, False])
    nulls = pd.DataFrame(null_rows)
    models = pd.DataFrame(model_rows).sort_values(["target", "roc_auc"], ascending=[True, False])
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    rules = pd.concat([rule_metrics(df, t, "echem_degraded_state_axis__source_residual", "signed_optical_loss_axis__source_residual") for t in targets], ignore_index=True)
    candidates = candidate_table(df)

    paths = {
        "direct_metrics": out / "echem_optical_source_residual_direct_metrics.csv",
        "model_metrics": out / "echem_optical_source_residual_model_metrics.csv",
        "predictions": out / "echem_optical_source_residual_predictions.csv",
        "permutation_nulls": out / "echem_optical_source_residual_permutation_nulls.csv",
        "rules": out / "echem_optical_source_residual_rules.csv",
        "top_candidates": out / "echem_optical_source_residual_top_candidates.csv",
        "summary": out / "echem_optical_source_residual_summary.json",
    }
    direct.to_csv(paths["direct_metrics"], index=False)
    models.to_csv(paths["model_metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    nulls.to_csv(paths["permutation_nulls"], index=False)
    rules.to_csv(paths["rules"], index=False)
    candidates.to_csv(paths["top_candidates"], index=False)

    future16_direct = direct[direct["target"] == "future_any_drop_within_16cycles"]
    future16_models = models[models["target"] == "future_any_drop_within_16cycles"]
    summary = clean_json({
        "n_rows": int(len(df)),
        "n_labeled_future16": int(numeric(df, "future_any_drop_within_16cycles").isin([0, 1]).sum()) if "future_any_drop_within_16cycles" in df else 0,
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "feature_sets": feature_sets,
        "top_future16_direct_metrics": future16_direct.head(20).to_dict("records"),
        "top_future16_model_metrics": future16_models.head(20).to_dict("records"),
        "top_direct_metrics": direct.head(30).to_dict("records"),
        "top_model_metrics": models.head(30).to_dict("records"),
        "top_rules": rules.head(30).to_dict("records"),
        "top_candidates": candidates.head(20).to_dict("records"),
        "guardrail": "Source-residual and within-source-rank transforms use unlabeled source distribution information and weak future labels. They test whether echem state contextualizes optical loss after source normalization; they are not deployable warning models or causal mechanism proof.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Echem/Optical Source-Residual Audit\n\n"
        "Tests whether source-normalized echem degraded state and optical-loss axes provide complementary future-drop evidence.\n\n"
        f"- Rows: {summary['n_rows']}\n- Future16 labeled rows: {summary['n_labeled_future16']}\n- Cycles: {summary['n_cycles']}\n- Sources: {summary['n_sources']}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )


if __name__ == "__main__":
    main()
