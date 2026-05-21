#!/usr/bin/env python3
"""Rolling-origin cycle hazard warning audit for NMC photometry.

This experiment asks whether cycle-level particle trace, echem-shape, and
state-space descriptors can warn of an abrupt particle-drop event before it
appears. It is intentionally cycle-level: no video ROI/front claim is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LEAK_PATTERNS = ("future_", "cycles_to_next", "abrupt_drop", "synchronized_drop", "drop_count")


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
        return pd.DataFrame()
    return pd.read_csv(path)


def feature_group(col: str) -> str:
    if col.startswith("cycle_state_pc") or col in {"degradation_state_axis", "state_step_norm", "axis_step"}:
        return "cycle_state"
    if col.startswith("shape_") or col.startswith(("all_dq", "pos_dq", "neg_dq")):
        return "echem_shape"
    if col.startswith("particle") or "delta" in col or col in {"mean_abs_delta_prev", "max_abs_delta_prev", "particle_norm_range", "particle_norm_cv"}:
        return "particle_trace"
    if col in {"capacity_mAh", "charge_capacity_mAh", "discharge_capacity_mAh", "coulombic_efficiency_pct", "V_min", "V_max"}:
        return "cycle_echem"
    if "frame" in col or col in {"n_frames", "frames_percentile", "cycle_gap", "n_points", "echem_shape_points", "echem_shape_duration_s"}:
        return "acquisition_context"
    return "other"


def usable_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for col in df.columns:
        if col == "cycleNo" or any(p in col for p in LEAK_PATTERNS):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() < 20 or vals.nunique(dropna=True) < 3:
            continue
        groups.setdefault(feature_group(col), []).append(col)
    return groups


def model_for(n_train: int, n_features: int, seed: int):
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if n_features > 12 and n_train > 20:
        steps.append(PCA(n_components=min(10, n_features, max(2, n_train - 2)), random_state=seed))
    steps.append(LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5))
    return make_pipeline(*steps)


def rolling_origin_predictions(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    purge_cycles: int,
    min_train: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ordered = df.sort_values("cycleNo").reset_index(drop=True)
    y_all = pd.to_numeric(ordered[target], errors="coerce")
    cycles = pd.to_numeric(ordered["cycleNo"], errors="coerce")
    for i, row in ordered.iterrows():
        y_test = y_all.iloc[i]
        if y_test not in {0, 1} or not np.isfinite(cycles.iloc[i]):
            continue
        train_mask = (cycles <= cycles.iloc[i] - purge_cycles) & y_all.isin([0, 1])
        train_idx = np.flatnonzero(train_mask.to_numpy())
        y_train = y_all.iloc[train_idx].astype(int).to_numpy()
        base_row: Dict[str, Any] = {
            "cycleNo": float(cycles.iloc[i]),
            "target": target,
            "observed": int(y_test),
            "n_train": int(len(train_idx)),
            "n_positive_train": int(y_train.sum()) if len(y_train) else 0,
            "purge_cycles": int(purge_cycles),
        }
        if len(train_idx) < min_train or len(np.unique(y_train)) < 2:
            base_row.update({"predicted_probability": np.nan, "status": "skipped_train_class_or_size"})
            rows.append(base_row)
            continue
        x_train = ordered.loc[train_idx, features].apply(pd.to_numeric, errors="coerce")
        x_test = ordered.loc[[i], features].apply(pd.to_numeric, errors="coerce")
        model = model_for(len(train_idx), len(features), seed + i)
        model.fit(x_train, y_train)
        prob = float(model.predict_proba(x_test)[:, 1][0])
        base_row.update({"predicted_probability": prob, "status": "evaluated"})
        rows.append(base_row)
    return pd.DataFrame(rows)


def summarize_predictions(pred: pd.DataFrame, feature_set: str, n_features: int) -> Dict[str, Any]:
    ev = pred[pred["status"].eq("evaluated")].copy()
    out: Dict[str, Any] = {
        "feature_set": feature_set,
        "n_features": int(n_features),
        "n_evaluated_cycles": int(len(ev)),
        "n_positive": int(ev["observed"].sum()) if not ev.empty else 0,
    }
    if ev.empty or ev["observed"].nunique() < 2:
        out.update({"roc_auc": np.nan, "average_precision": np.nan, "brier_score": np.nan, "balanced_accuracy_top_rate": np.nan})
        return out
    y = ev["observed"].astype(int).to_numpy()
    p = ev["predicted_probability"].to_numpy(float)
    positive_rate = float(y.mean())
    threshold = float(np.nanquantile(p, max(0.0, 1.0 - positive_rate)))
    pred_label = (p >= threshold).astype(int)
    out.update({
        "positive_rate": positive_rate,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "top_rate_threshold": threshold,
        "balanced_accuracy_top_rate": float(balanced_accuracy_score(y, pred_label)),
    })
    return out


def warning_lead_table(pred: pd.DataFrame, cycles_with_events: Iterable[float], horizons: Iterable[int]) -> pd.DataFrame:
    ev = pred[pred["status"].eq("evaluated")].copy()
    if ev.empty:
        return pd.DataFrame()
    y = ev["observed"].astype(int).to_numpy()
    p = ev["predicted_probability"].to_numpy(float)
    threshold = float(np.nanquantile(p, max(0.0, 1.0 - float(y.mean())))) if y.sum() else np.nan
    rows = []
    for event_cycle in sorted(float(x) for x in cycles_with_events):
        for h in horizons:
            window = ev[(ev["cycleNo"] < event_cycle) & (ev["cycleNo"] >= event_cycle - h)]
            rows.append({
                "event_cycle": event_cycle,
                "lookback_horizon_cycles": int(h),
                "n_prior_predictions": int(len(window)),
                "max_prior_probability": float(window["predicted_probability"].max()) if len(window) else np.nan,
                "mean_prior_probability": float(window["predicted_probability"].mean()) if len(window) else np.nan,
                "warning_threshold": threshold,
                "hit": bool(len(window) and np.nanmax(window["predicted_probability"]) >= threshold) if np.isfinite(threshold) else False,
                "best_warning_cycle": float(window.loc[window["predicted_probability"].idxmax(), "cycleNo"]) if len(window) else np.nan,
                "lead_cycles": float(event_cycle - window.loc[window["predicted_probability"].idxmax(), "cycleNo"]) if len(window) else np.nan,
            })
    return pd.DataFrame(rows)


def permutation_null(df: pd.DataFrame, features: List[str], target: str, observed_auc: float, purge: int, min_train: int, n_perm: int, seed: int) -> Dict[str, Any]:
    if not np.isfinite(observed_auc):
        return {"target": target, "n_permutations": 0, "empirical_p_ge_observed": None}
    rng = np.random.default_rng(seed)
    valid = pd.to_numeric(df[target], errors="coerce").isin([0, 1]).to_numpy()
    y = pd.to_numeric(df.loc[valid, target], errors="coerce").astype(int).to_numpy()
    aucs = []
    for i in range(n_perm):
        work = df.copy()
        shuffled = y.copy()
        rng.shuffle(shuffled)
        work.loc[valid, "_perm_target"] = shuffled
        pred = rolling_origin_predictions(work, features, "_perm_target", purge, min_train, seed + 1000 + i)
        summ = summarize_predictions(pred, "permuted", len(features))
        auc = summ.get("roc_auc")
        if auc is not None and np.isfinite(auc):
            aucs.append(float(auc))
    arr = np.asarray(aucs, dtype=float)
    return {
        "target": target,
        "observed_auc": float(observed_auc),
        "n_permutations": int(len(arr)),
        "null_mean_auc": float(arr.mean()) if len(arr) else None,
        "null_p95_auc": float(np.percentile(arr, 95)) if len(arr) else None,
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)) if len(arr) else None,
    }


def probability_context_correlations(pred: pd.DataFrame, df: pd.DataFrame, context_cols: List[str]) -> pd.DataFrame:
    ev = pred[pred["status"].eq("evaluated")][["cycleNo", "predicted_probability"]].copy()
    joined = ev.merge(df[["cycleNo"] + context_cols], on="cycleNo", how="left")
    rows = []
    for col in context_cols:
        tmp = joined[["predicted_probability", col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(tmp) >= 8 and tmp[col].nunique() > 2:
            r = spearmanr(tmp["predicted_probability"], tmp[col])
            rows.append({"feature": col, "n": int(len(tmp)), "spearman_rho": float(r.statistic), "p_value": float(r.pvalue)})
    return pd.DataFrame(rows).sort_values("p_value", na_position="last") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_hazard_warning_audit")
    parser.add_argument("--target", default="future_any_drop_within_8cycles")
    parser.add_argument("--purge-cycles", type=int, default=8)
    parser.add_argument("--min-train", type=int, default=24)
    parser.add_argument("--n-permutation", type=int, default=500)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = read_csv(derived / "cycle_state_space_transition_audit" / "cycle_state_space_table.csv")
    if table.empty:
        raise FileNotFoundError("cycle_state_space_table.csv is required")
    table = table.sort_values("cycleNo").reset_index(drop=True)
    groups = usable_features(table)
    feature_sets: Dict[str, List[str]] = {
        "particle_trace": groups.get("particle_trace", []),
        "echem_shape_cycle": groups.get("echem_shape", []) + groups.get("cycle_echem", []),
        "particle_trace_echem_no_acquisition": groups.get("particle_trace", []) + groups.get("echem_shape", []) + groups.get("cycle_echem", []),
        "particle_trace_echem_with_acquisition": groups.get("particle_trace", []) + groups.get("echem_shape", []) + groups.get("cycle_echem", []) + groups.get("acquisition_context", []),
        "combined_with_cycle_state": groups.get("particle_trace", []) + groups.get("echem_shape", []) + groups.get("cycle_echem", []) + groups.get("cycle_state", []),
    }
    feature_sets = {k: sorted(set(v)) for k, v in feature_sets.items() if v}

    all_predictions = []
    summaries = []
    for name, feats in feature_sets.items():
        pred = rolling_origin_predictions(table, feats, args.target, args.purge_cycles, args.min_train, args.seed)
        pred.insert(0, "feature_set", name)
        all_predictions.append(pred)
        summaries.append(summarize_predictions(pred, name, len(feats)))
    pred_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    summary_df = pd.DataFrame(summaries).sort_values("roc_auc", ascending=False, na_position="last")

    best_set = str(summary_df.iloc[0]["feature_set"]) if not summary_df.empty else ""
    best_features = feature_sets.get(best_set, [])
    best_pred = pred_df[pred_df["feature_set"].eq(best_set)].copy() if not pred_df.empty else pd.DataFrame()
    event_cycles = table.loc[pd.to_numeric(table.get("any_abrupt_drop"), errors="coerce").eq(1), "cycleNo"].dropna().to_numpy(float)
    lead = warning_lead_table(best_pred, event_cycles, [4, 8, 16]) if not best_pred.empty else pd.DataFrame()

    ablation_rows = []
    best_auc = float(summary_df.iloc[0]["roc_auc"]) if not summary_df.empty and np.isfinite(summary_df.iloc[0]["roc_auc"]) else np.nan
    for group in sorted({feature_group(f) for f in best_features}):
        feats = [f for f in best_features if feature_group(f) != group]
        if len(feats) < 2:
            continue
        pred = rolling_origin_predictions(table, feats, args.target, args.purge_cycles, args.min_train, args.seed + len(ablation_rows) + 7)
        summ = summarize_predictions(pred, f"remove_{group}", len(feats))
        summ["removed_group"] = group
        summ["auc_drop_vs_best"] = best_auc - summ.get("roc_auc", np.nan) if np.isfinite(best_auc) else np.nan
        ablation_rows.append(summ)
    ablation = pd.DataFrame(ablation_rows).sort_values("auc_drop_vs_best", ascending=False, na_position="last") if ablation_rows else pd.DataFrame()

    null = permutation_null(table, best_features, args.target, best_auc, args.purge_cycles, args.min_train, args.n_permutation, args.seed) if best_features else {}
    context_cols = [c for c in ["cycle_state_pc1", "cycle_state_pc2", "degradation_state_axis", "mean_abs_delta_prev", "particle_norm_range", "capacity_mAh", "coulombic_efficiency_pct", "shape_V_q95", "frames_percentile"] if c in table.columns]
    corr = probability_context_correlations(best_pred, table, context_cols) if not best_pred.empty else pd.DataFrame()

    pred_df.to_csv(out / "cycle_hazard_warning_predictions.csv", index=False)
    summary_df.to_csv(out / "cycle_hazard_warning_feature_set_summary.csv", index=False)
    lead.to_csv(out / "cycle_hazard_warning_lead_time.csv", index=False)
    ablation.to_csv(out / "cycle_hazard_warning_group_ablation.csv", index=False)
    corr.to_csv(out / "cycle_hazard_warning_probability_correlations.csv", index=False)

    lead_summary = {}
    if not lead.empty:
        lead_summary = (
            lead.groupby("lookback_horizon_cycles", dropna=False)
            .agg(n_events=("event_cycle", "count"), hit_rate=("hit", "mean"), median_max_probability=("max_prior_probability", "median"), median_lead_cycles=("lead_cycles", "median"))
            .reset_index()
            .to_dict("records")
        )
    summary = {
        "target": args.target,
        "n_cycles": int(len(table)),
        "n_event_cycles": int(len(event_cycles)),
        "event_cycles": [float(x) for x in event_cycles],
        "purge_cycles": int(args.purge_cycles),
        "min_train": int(args.min_train),
        "feature_group_counts": {k: len(v) for k, v in groups.items()},
        "feature_set_summary": clean_json(summary_df.to_dict("records")),
        "best_feature_set": best_set,
        "permutation_null": clean_json(null),
        "lead_time_summary": clean_json(lead_summary),
        "top_group_ablation": clean_json(ablation.head(10).to_dict("records")) if not ablation.empty else [],
        "top_probability_correlations": clean_json(corr.head(10).to_dict("records")) if not corr.empty else [],
        "guardrail": "Rolling-origin cycle-level warning audit using particle trace/echem/state descriptors. It tests early-warning covariates, not localized ROI fronts, manual degradation labels, or calibrated diffusion.",
    }
    with (out / "cycle_hazard_warning_audit_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2)
    with (out / "README.md").open("w") as f:
        f.write("# Cycle Hazard Warning Audit\n\n")
        f.write("Rolling-origin warning audit for future abrupt particle drops using cycle-level photometry/echem descriptors.\n\n")
        f.write(f"- Cycles: {summary['n_cycles']}\n")
        f.write(f"- Event cycles: {summary['event_cycles']}\n")
        f.write(f"- Target: {summary['target']}\n")
        f.write(f"- Best feature set: {summary['best_feature_set']}\n\n")
        f.write("Outputs:\n\n")
        f.write("- `cycle_hazard_warning_predictions.csv`: rolling-origin probabilities by cycle and feature set.\n")
        f.write("- `cycle_hazard_warning_feature_set_summary.csv`: AUC/AP/Brier summaries.\n")
        f.write("- `cycle_hazard_warning_lead_time.csv`: pre-event warning hits and lead cycles.\n")
        f.write("- `cycle_hazard_warning_group_ablation.csv`: feature-group ablation summary for the best model.\n")
        f.write("- `cycle_hazard_warning_probability_correlations.csv`: warning probability links to interpretable cycle descriptors.\n")
    print(json.dumps(clean_json({"best_feature_set": best_set, "feature_set_summary": summary["feature_set_summary"][:3], "permutation_null": null, "lead_time_summary": lead_summary}), indent=2))


if __name__ == "__main__":
    main()
