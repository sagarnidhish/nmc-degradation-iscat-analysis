#!/usr/bin/env python3
"""Particle-trace physics audit across the larger NMC cycle table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PARTICLES = [f"particle{i}" for i in range(4)]
NORM_COLS = [f"{p}_norm" for p in PARTICLES]
DROP_COLS = [f"{p}_abrupt_drop" for p in PARTICLES]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
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


def add_trace_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("cycleNo").copy()
    for col in NORM_COLS:
        vals = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_delta_prev"] = vals.diff()
        out[f"{col}_abs_delta_prev"] = out[f"{col}_delta_prev"].abs()
        out[f"{col}_rolling3_slope"] = vals.rolling(3, min_periods=3).apply(
            lambda x: float(np.polyfit(np.arange(len(x)), x, 1)[0]), raw=False
        )
        out[f"{col}_rolling5_mean"] = vals.rolling(5, min_periods=2).mean()
        out[f"{col}_rolling5_std"] = vals.rolling(5, min_periods=2).std()
    norm = out[NORM_COLS].apply(pd.to_numeric, errors="coerce")
    delta = out[[f"{c}_delta_prev" for c in NORM_COLS]].apply(pd.to_numeric, errors="coerce")
    out["particle_norm_mean"] = norm.mean(axis=1)
    out["particle_norm_std"] = norm.std(axis=1)
    out["particle_norm_min"] = norm.min(axis=1)
    out["particle_norm_max"] = norm.max(axis=1)
    out["particle_norm_range"] = out["particle_norm_max"] - out["particle_norm_min"]
    out["particle_norm_cv"] = out["particle_norm_std"] / out["particle_norm_mean"].replace(0, np.nan).abs()
    out["mean_delta_prev"] = delta.mean(axis=1)
    out["mean_abs_delta_prev"] = delta.abs().mean(axis=1)
    out["max_abs_delta_prev"] = delta.abs().max(axis=1)
    out["delta_std_across_particles"] = delta.std(axis=1)
    out["drop_count"] = out[DROP_COLS].astype(bool).sum(axis=1)
    out["any_abrupt_drop"] = out["drop_count"].gt(0).astype(int)
    out["synchronized_drop_2plus"] = out["drop_count"].ge(2).astype(int)
    out["synchronized_drop_3plus"] = out["drop_count"].ge(3).astype(int)
    out["cycle_gap"] = pd.to_numeric(out["cycleNo"], errors="coerce").diff()
    out["frames_percentile"] = pd.to_numeric(out["n_frames"], errors="coerce").rank(pct=True)
    return out


def add_future_targets(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    out = df.sort_values("cycleNo").copy()
    cycles = out["cycleNo"].to_numpy(dtype=float)
    event = out["any_abrupt_drop"].to_numpy(dtype=int)
    sync2 = out["synchronized_drop_2plus"].to_numpy(dtype=int)
    for h in horizons:
        future_event = []
        future_sync = []
        cycles_to_event = []
        for c in cycles:
            mask = (cycles > c) & (cycles <= c + h)
            ev_cycles = cycles[mask & event.astype(bool)]
            sync_cycles = cycles[mask & sync2.astype(bool)]
            future_event.append(int(len(ev_cycles) > 0))
            future_sync.append(int(len(sync_cycles) > 0))
            cycles_to_event.append(float(ev_cycles[0] - c) if len(ev_cycles) else np.nan)
        out[f"future_any_drop_within_{h}cycles"] = future_event
        out[f"future_sync2_drop_within_{h}cycles"] = future_sync
        out[f"cycles_to_next_drop_within_{h}"] = cycles_to_event
    return out


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded_prefixes = ("future_", "cycles_to_next")
    excluded = {"cycleNo", "addrs", *DROP_COLS}
    features: List[str] = []
    for col in df.columns:
        if col in excluded or col.startswith(excluded_prefixes):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 12 and vals.nunique(dropna=True) > 2:
            features.append(col)
    return features


def block_id(cycle: pd.Series, block_width: int = 20) -> pd.Series:
    return (pd.to_numeric(cycle, errors="coerce") // block_width).astype("Int64")


def leave_block_out_classifier(df: pd.DataFrame, features: List[str], target: str) -> Dict[str, Any]:
    rows = []
    pred_rows = []
    blocks = block_id(df["cycleNo"])
    x = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce").astype(int)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
    ])
    for blk in sorted(blocks.dropna().unique()):
        test = blocks.eq(blk)
        train = ~test
        if y[train].nunique() < 2 or y[test].nunique() < 2 or int(test.sum()) < 4:
            continue
        model.fit(x[train], y[train])
        prob = model.predict_proba(x[test])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "heldout_cycle_block": int(blk),
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "event_rate_test": float(y[test].mean()),
            "roc_auc": float(roc_auc_score(y[test], prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
        })
        pred_rows.extend({
            "cycleNo": float(c),
            "heldout_cycle_block": int(blk),
            "target": target,
            "observed": int(obs),
            "predicted_probability": float(pr),
            "prediction": int(pp),
        } for c, obs, pr, pp in zip(df.loc[test, "cycleNo"], y[test], prob, pred))
    fold_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)
    summary = {
        "target": target,
        "n_folds": int(len(fold_df)),
        "mean_roc_auc": float(fold_df["roc_auc"].mean()) if not fold_df.empty else np.nan,
        "mean_balanced_accuracy": float(fold_df["balanced_accuracy"].mean()) if not fold_df.empty else np.nan,
    }
    return {"folds": fold_df, "predictions": pred_df, "summary": summary}


def permutation_null_classifier(df: pd.DataFrame, features: List[str], target: str, observed_auc: float, n_perm: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    aucs = []
    y = pd.to_numeric(df[target], errors="coerce").astype(int).to_numpy()
    for i in range(n_perm):
        permuted = df.copy()
        shuffled = y.copy()
        rng.shuffle(shuffled)
        permuted["_perm_target"] = shuffled
        result = leave_block_out_classifier(permuted, features, "_perm_target")["summary"]
        auc = result["mean_roc_auc"]
        if np.isfinite(auc):
            aucs.append(float(auc))
    arr = np.asarray(aucs, dtype=float)
    return {
        "target": target,
        "observed_mean_auc": float(observed_auc) if np.isfinite(observed_auc) else None,
        "n_permutations": int(len(arr)),
        "null_mean_auc": float(arr.mean()) if len(arr) else None,
        "null_p95_auc": float(np.percentile(arr, 95)) if len(arr) else None,
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)) if len(arr) and np.isfinite(observed_auc) else None,
    }


def cluster_states(df: pd.DataFrame, features: List[str], max_k: int, seed: int) -> Dict[str, Any]:
    x = df[features].apply(pd.to_numeric, errors="coerce")
    z = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]).fit_transform(x)
    rows = []
    labels_by_k = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=50, random_state=seed + k)
        labels = km.fit_predict(z)
        sil = silhouette_score(z, labels) if len(set(labels)) > 1 else np.nan
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
        labels_by_k[k] = labels
    select = pd.DataFrame(rows).sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    chosen_k = int(select["k"])
    out = df[["cycleNo", "any_abrupt_drop", "synchronized_drop_2plus", "capacity_mAh", "coulombic_efficiency_pct"]].copy()
    out["trace_state_cluster"] = labels_by_k[chosen_k]
    return {
        "selection": pd.DataFrame(rows),
        "assignments": out,
        "chosen_k": chosen_k,
        "best_silhouette": float(select["silhouette"]),
    }


def group_tests(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target], errors="coerce").astype(int)
    for feat in features:
        x = pd.to_numeric(df[feat], errors="coerce")
        pos = x[y.eq(1) & x.notna()].to_numpy(dtype=float)
        neg = x[y.eq(0) & x.notna()].to_numpy(dtype=float)
        if len(pos) >= 3 and len(neg) >= 3:
            _, p = mannwhitneyu(pos, neg, alternative="two-sided")
            rows.append({
                "target": target,
                "feature": feat,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                "median_positive": float(np.median(pos)),
                "median_negative": float(np.median(neg)),
                "median_pos_minus_neg": float(np.median(pos) - np.median(neg)),
                "mannwhitney_p": float(p),
            })
    cols = [
        "target",
        "feature",
        "n_positive",
        "n_negative",
        "median_positive",
        "median_negative",
        "median_pos_minus_neg",
        "mannwhitney_p",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(["mannwhitney_p", "feature"])


def correlations(df: pd.DataFrame, left: Iterable[str], right: Iterable[str]) -> pd.DataFrame:
    rows = []
    for a in left:
        x = pd.to_numeric(df[a], errors="coerce")
        for b in right:
            y = pd.to_numeric(df[b], errors="coerce")
            valid = x.notna() & y.notna()
            if int(valid.sum()) >= 8 and x[valid].nunique() > 2 and y[valid].nunique() > 2:
                r = spearmanr(x[valid], y[valid])
                rows.append({
                    "left_feature": a,
                    "right_feature": b,
                    "n": int(valid.sum()),
                    "rho": float(r.statistic),
                    "p_value": float(r.pvalue),
                })
    return pd.DataFrame(rows).sort_values(["p_value", "left_feature", "right_feature"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_trace_physics_audit")
    parser.add_argument("--n-permutation", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    traces = pd.read_csv(derived / "particle_intensity_normalized.csv")
    echem = pd.read_csv(derived / "echem_per_cycle.csv")
    traces["cycleNo"] = pd.to_numeric(traces["cycleNo"], errors="coerce")
    echem["cycleNo"] = pd.to_numeric(echem["cycleNo"], errors="coerce")
    df = add_future_targets(add_trace_features(traces), horizons=[4, 8, 16])
    df = df.merge(echem, on="cycleNo", how="left", suffixes=("", "_echem"))

    features = numeric_feature_columns(df)
    no_lead_features = [f for f in features if not f.endswith("_abrupt_drop") and f not in {"any_abrupt_drop", "synchronized_drop_2plus", "synchronized_drop_3plus", "drop_count"}]
    echem_features = ["capacity_mAh", "coulombic_efficiency_pct", "V_min", "V_max", "n_points", "n_frames", "frames_percentile"]
    trace_features = [f for f in no_lead_features if f not in echem_features and not f.startswith("particle") or f in ["particle_norm_mean", "particle_norm_std", "particle_norm_range", "particle_norm_cv"]]

    cluster_feature_cols = [
        c for c in [
            "particle_norm_mean",
            "particle_norm_std",
            "particle_norm_range",
            "particle_norm_cv",
            "mean_delta_prev",
            "mean_abs_delta_prev",
            "max_abs_delta_prev",
            "delta_std_across_particles",
            "capacity_mAh",
            "coulombic_efficiency_pct",
            "frames_percentile",
        ] if c in df.columns
    ]
    clusters = cluster_states(df.dropna(subset=["cycleNo"]).copy(), cluster_feature_cols, max_k=6, seed=args.seed)
    assignments = clusters["assignments"]
    cluster_summary = (
        assignments.groupby("trace_state_cluster", dropna=False)
        .agg(
            n_cycles=("cycleNo", "count"),
            cycle_min=("cycleNo", "min"),
            cycle_max=("cycleNo", "max"),
            abrupt_drop_rate=("any_abrupt_drop", "mean"),
            sync2_drop_rate=("synchronized_drop_2plus", "mean"),
            median_capacity=("capacity_mAh", "median"),
            median_ce=("coulombic_efficiency_pct", "median"),
        )
        .reset_index()
        .sort_values(["abrupt_drop_rate", "sync2_drop_rate"], ascending=[False, False])
    )

    event_feature_cols = list(dict.fromkeys(cluster_feature_cols + [c for c in echem_features if c in df.columns]))
    event_tests = pd.concat([
        group_tests(df, event_feature_cols, "any_abrupt_drop"),
        group_tests(df, event_feature_cols, "synchronized_drop_2plus"),
    ], ignore_index=True)
    echem_corr = correlations(df, ["capacity_mAh", "coulombic_efficiency_pct", "V_min", "V_max"], [
        "particle_norm_mean", "particle_norm_std", "particle_norm_range", "mean_delta_prev", "mean_abs_delta_prev", "max_abs_delta_prev"
    ])

    classifier_targets = ["future_any_drop_within_8cycles", "future_sync2_drop_within_8cycles"]
    clf_summaries = []
    clf_folds = []
    clf_preds = []
    null_rows = []
    clf_features = [c for c in list(dict.fromkeys(cluster_feature_cols + [f for f in echem_features if f in df.columns])) if c not in {"n_points"}]
    for target in classifier_targets:
        if df[target].nunique() < 2:
            continue
        result = leave_block_out_classifier(df, clf_features, target)
        clf_summaries.append(result["summary"])
        if not result["folds"].empty:
            clf_folds.append(result["folds"])
        if not result["predictions"].empty:
            clf_preds.append(result["predictions"])
        null_rows.append(permutation_null_classifier(df, clf_features, target, result["summary"]["mean_roc_auc"], args.n_permutation, args.seed + len(null_rows)))

    paths = {
        "cycle_features": out / "particle_trace_cycle_features.csv",
        "cluster_selection": out / "particle_trace_cluster_selection.csv",
        "cluster_assignments": out / "particle_trace_cluster_assignments.csv",
        "cluster_summary": out / "particle_trace_cluster_summary.csv",
        "event_feature_tests": out / "particle_trace_event_feature_tests.csv",
        "echem_correlations": out / "particle_trace_echem_correlations.csv",
        "classifier_summary": out / "particle_trace_future_drop_classifier_summary.csv",
        "classifier_folds": out / "particle_trace_future_drop_classifier_folds.csv",
        "classifier_predictions": out / "particle_trace_future_drop_classifier_predictions.csv",
        "classifier_null": out / "particle_trace_future_drop_classifier_null.csv",
        "summary": out / "particle_trace_physics_audit_summary.json",
    }
    df.to_csv(paths["cycle_features"], index=False)
    clusters["selection"].to_csv(paths["cluster_selection"], index=False)
    assignments.to_csv(paths["cluster_assignments"], index=False)
    cluster_summary.to_csv(paths["cluster_summary"], index=False)
    event_tests.to_csv(paths["event_feature_tests"], index=False)
    echem_corr.to_csv(paths["echem_correlations"], index=False)
    pd.DataFrame(clf_summaries).to_csv(paths["classifier_summary"], index=False)
    pd.concat(clf_folds, ignore_index=True).to_csv(paths["classifier_folds"], index=False) if clf_folds else pd.DataFrame().to_csv(paths["classifier_folds"], index=False)
    pd.concat(clf_preds, ignore_index=True).to_csv(paths["classifier_predictions"], index=False) if clf_preds else pd.DataFrame().to_csv(paths["classifier_predictions"], index=False)
    pd.DataFrame(null_rows).to_csv(paths["classifier_null"], index=False)

    summary = {
        "n_cycle_rows": int(len(df)),
        "cycle_min": float(df["cycleNo"].min()),
        "cycle_max": float(df["cycleNo"].max()),
        "n_particles": int(len(PARTICLES)),
        "n_any_drop_cycles": int(df["any_abrupt_drop"].sum()),
        "n_sync2_drop_cycles": int(df["synchronized_drop_2plus"].sum()),
        "n_sync3_drop_cycles": int(df["synchronized_drop_3plus"].sum()),
        "drop_cycles": df.loc[df["any_abrupt_drop"].eq(1), ["cycleNo", "drop_count"]].to_dict("records"),
        "chosen_trace_state_k": clusters["chosen_k"],
        "trace_state_best_silhouette": clusters["best_silhouette"],
        "top_trace_state_clusters": cluster_summary.head(8).to_dict("records"),
        "top_event_feature_tests": event_tests.head(12).to_dict("records") if not event_tests.empty else [],
        "top_echem_correlations": echem_corr.head(12).to_dict("records") if not echem_corr.empty else [],
        "future_drop_classifier": clf_summaries,
        "future_drop_classifier_null": null_rows,
        "guardrail": "This audit uses the larger four-particle cycle intensity table, not video ROI masks. It tests cycle-level photometry/echem physics hypotheses and early-warning signals, but cannot localize phase fronts or validate diffusion without ROI/video QC.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Particle Trace Physics Audit",
        "",
        "Cycle-level audit over the larger normalized four-particle intensity table.",
        "",
        f"- Cycle rows: {summary['n_cycle_rows']}",
        f"- Cycle range: {summary['cycle_min']:.0f}-{summary['cycle_max']:.0f}",
        f"- Any-drop cycles: {summary['n_any_drop_cycles']}",
        f"- Synchronized 2+ drop cycles: {summary['n_sync2_drop_cycles']}",
        f"- Chosen trace-state k: {summary['chosen_trace_state_k']} (silhouette={summary['trace_state_best_silhouette']:.3f})",
        "",
        "## Top Event Feature Tests",
    ]
    for row in summary["top_event_feature_tests"][:8]:
        lines.append(
            f"- {row.get('target')} {row.get('feature')}: median pos-neg={row.get('median_pos_minus_neg'):.3g}, p={row.get('mannwhitney_p'):.3g}"
        )
    lines += ["", "## Future Drop Classifier"]
    for row in clf_summaries:
        lines.append(
            f"- {row.get('target')}: folds={row.get('n_folds')}, mean AUC={row.get('mean_roc_auc'):.3f}, balanced accuracy={row.get('mean_balanced_accuracy'):.3f}"
        )
    for row in null_rows:
        lines.append(
            f"- null {row.get('target')}: observed AUC={row.get('observed_mean_auc'):.3f}, null p95={row.get('null_p95_auc'):.3f}, empirical p={row.get('empirical_p_ge_observed'):.4f}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "out_dir": str(out),
        "n_cycle_rows": summary["n_cycle_rows"],
        "drop_cycles": summary["drop_cycles"],
        "chosen_k": summary["chosen_trace_state_k"],
        "classifier": clf_summaries,
        "null": null_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
