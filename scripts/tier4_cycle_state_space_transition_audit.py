#!/usr/bin/env python3
"""Cycle-level degradation state-space and transition audit for NMC photometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDE_PATTERNS = (
    "future_",
    "cycles_to_next",
    "abrupt_drop",
    "synchronized_drop",
    "drop_count",
)


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


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for col in df.columns:
        if col == "cycleNo":
            continue
        if any(p in col for p in EXCLUDE_PATTERNS):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.to_numeric(df[col], errors="coerce").notna().sum() >= 8:
                out.append(col)
    return out


def feature_group(col: str) -> str:
    if col.startswith("particle") or "particle_norm" in col or "delta" in col:
        return "particle_trace"
    if col.startswith("shape_") or "_dq_" in col or "dqdv" in col:
        return "within_cycle_echem_shape"
    if col in {"capacity_mAh", "charge_capacity_mAh", "discharge_capacity_mAh", "coulombic_efficiency_pct", "V_min", "V_max"}:
        return "cycle_echem_summary"
    if "frame" in col or col in {"n_frames", "frames_percentile", "cycle_gap", "n_points"}:
        return "acquisition_protocol"
    return "other"


def orient_axis(axis: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, str, float | None]:
    candidates = [
        "future_any_drop_within_8cycles",
        "future_sync2_drop_within_8cycles",
        "any_abrupt_drop",
        "mean_abs_delta_prev",
    ]
    best_name = "unoriented"
    best_rho: float | None = None
    best_abs = -np.inf
    for col in candidates:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        mask = np.isfinite(axis) & y.notna()
        if mask.sum() < 8 or y[mask].nunique() < 2:
            continue
        rho = float(spearmanr(axis[mask], y[mask]).statistic)
        if np.isfinite(rho) and abs(rho) > best_abs:
            best_abs = abs(rho)
            best_name = col
            best_rho = rho
    if best_rho is not None and best_rho < 0:
        return -axis, best_name, -best_rho
    return axis, best_name, best_rho


def permutation_p(obs: float, null: np.ndarray) -> float:
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))


def binary_tests(df: pd.DataFrame, features: Iterable[str], target: str, rng: np.random.Generator, n_perm: int) -> pd.DataFrame:
    rows = []
    if target not in df.columns:
        return pd.DataFrame()
    y = pd.to_numeric(df[target], errors="coerce")
    for feature in features:
        x = pd.to_numeric(df.get(feature), errors="coerce")
        mask = x.notna() & y.notna() & y.isin([0, 1])
        if mask.sum() < 8 or y[mask].nunique() < 2:
            continue
        pos = x[mask & y.eq(1)].to_numpy(float)
        neg = x[mask & y.eq(0)].to_numpy(float)
        if len(pos) < 2 or len(neg) < 2:
            continue
        obs = float(np.nanmedian(pos) - np.nanmedian(neg))
        pooled = x[mask].to_numpy(float)
        labels = y[mask].to_numpy(int)
        null = []
        for _ in range(n_perm):
            shuf = rng.permutation(labels)
            null.append(float(np.nanmedian(pooled[shuf == 1]) - np.nanmedian(pooled[shuf == 0])))
        null_arr = np.asarray(null)
        rows.append(
            {
                "target": target,
                "feature": feature,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                "median_positive": float(np.nanmedian(pos)),
                "median_negative": float(np.nanmedian(neg)),
                "median_positive_minus_negative": obs,
                "mannwhitney_p": float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue),
                "permutation_p": permutation_p(obs, null_arr),
                "null_p95_abs": float(np.nanpercentile(np.abs(null_arr), 95)),
            }
        )
    return pd.DataFrame(rows).sort_values("permutation_p") if rows else pd.DataFrame()


def cv_logistic(df: pd.DataFrame, feature_cols: List[str], target: str, seed: int) -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.isin([0, 1])
    yv = y[mask].astype(int).to_numpy()
    if len(np.unique(yv)) < 2 or min(np.bincount(yv)) < 2:
        return pd.DataFrame()
    X = df.loc[mask, feature_cols].apply(pd.to_numeric, errors="coerce")
    n_splits = min(5, int(min(np.bincount(yv))))
    if n_splits < 2:
        return pd.DataFrame()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, yv), start=1):
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5),
        )
        pipe.fit(X.iloc[tr], yv[tr])
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(
            {
                "target": target,
                "fold": fold,
                "n_test": int(len(te)),
                "n_positive_test": int(yv[te].sum()),
                "roc_auc": float(roc_auc_score(yv[te], prob)) if len(np.unique(yv[te])) > 1 else np.nan,
                "balanced_accuracy": float(balanced_accuracy_score(yv[te], pred)),
            }
        )
    return pd.DataFrame(rows)


def temporal_holdout_logistic(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    seed: int,
    horizon_cycles: int = 8,
    n_blocks: int = 5,
) -> pd.DataFrame:
    if target not in df.columns:
        return pd.DataFrame()
    ordered = df.sort_values("cycleNo").reset_index(drop=True)
    y = pd.to_numeric(ordered[target], errors="coerce")
    valid = y.isin([0, 1])
    ordered = ordered.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    if len(ordered) < n_blocks * 4 or y.nunique() < 2:
        return pd.DataFrame()

    index_blocks = np.array_split(np.arange(len(ordered)), n_blocks)
    rows = []
    for fold, test_idx in enumerate(index_blocks[1:], start=1):
        test_start_cycle = float(ordered.loc[test_idx[0], "cycleNo"])
        train_mask = pd.to_numeric(ordered["cycleNo"], errors="coerce") <= (test_start_cycle - horizon_cycles)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()
        row: Dict[str, Any] = {
            "target": target,
            "fold": fold,
            "train_cycle_max": float(ordered.loc[train_idx, "cycleNo"].max()) if len(train_idx) else np.nan,
            "test_cycle_min": test_start_cycle,
            "test_cycle_max": float(ordered.loc[test_idx[-1], "cycleNo"]),
            "purge_horizon_cycles": int(horizon_cycles),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_positive_train": int(y_train.sum()) if len(y_train) else 0,
            "n_positive_test": int(y_test.sum()),
        }
        if len(train_idx) < 12 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            row.update({"roc_auc": np.nan, "balanced_accuracy": np.nan, "status": "skipped_class_or_size"})
            rows.append(row)
            continue
        X_train = ordered.loc[train_idx, feature_cols].apply(pd.to_numeric, errors="coerce")
        X_test = ordered.loc[test_idx, feature_cols].apply(pd.to_numeric, errors="coerce")
        n_components = min(8, len(feature_cols), max(2, len(train_idx) - 1))
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=n_components, random_state=seed),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5),
        )
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        row.update(
            {
                "roc_auc": float(roc_auc_score(y_test, prob)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "mean_predicted_probability": float(np.mean(prob)),
                "status": "evaluated",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/cycle_state_space_transition_audit")
    parser.add_argument("--n-permutation", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    trace = read_csv(derived / "particle_trace_physics_audit" / "particle_trace_cycle_features.csv")
    shape = read_csv(derived / "within_cycle_echem_shape_audit" / "within_cycle_echem_shape_features.csv")
    if trace.empty:
        raise FileNotFoundError("particle_trace_cycle_features.csv is required.")
    df = trace.copy()
    if not shape.empty:
        df = df.merge(shape, on="cycleNo", how="left", suffixes=("", "_shape"))
    df = df.sort_values("cycleNo").reset_index(drop=True)

    features = numeric_feature_columns(df)
    X_raw = df[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_raw)
    X = scaler.fit_transform(X_imp)
    n_pcs = min(8, X.shape[1], max(2, X.shape[0] - 1))
    pca = PCA(n_components=n_pcs, random_state=args.seed)
    pcs = pca.fit_transform(X)
    for i in range(n_pcs):
        df[f"cycle_state_pc{i+1}"] = pcs[:, i]
    axis, orient_target, orient_rho = orient_axis(pcs[:, 0].copy(), df)
    df["degradation_state_axis"] = axis
    df["degradation_axis_oriented_to"] = orient_target

    k_rows = []
    for k in range(2, min(8, len(df) - 1)):
        km = KMeans(n_clusters=k, n_init=50, random_state=args.seed)
        labels = km.fit_predict(pcs[:, : min(4, n_pcs)])
        sil = float(silhouette_score(pcs[:, : min(4, n_pcs)], labels)) if len(set(labels)) > 1 else np.nan
        k_rows.append({"k": k, "silhouette": sil, "inertia": float(km.inertia_)})
    k_df = pd.DataFrame(k_rows).sort_values("silhouette", ascending=False)
    chosen_k = int(k_df.iloc[0]["k"]) if not k_df.empty else 2
    km = KMeans(n_clusters=chosen_k, n_init=100, random_state=args.seed)
    df["cycle_state_cluster"] = km.fit_predict(pcs[:, : min(4, n_pcs)])

    diff = np.vstack([np.full((1, min(4, n_pcs)), np.nan), np.diff(pcs[:, : min(4, n_pcs)], axis=0)])
    df["state_step_norm"] = np.sqrt(np.nansum(diff**2, axis=1))
    df.loc[0, "state_step_norm"] = np.nan
    df["axis_step"] = pd.to_numeric(df["degradation_state_axis"], errors="coerce").diff()

    event_cols = [
        "any_abrupt_drop",
        "synchronized_drop_2plus",
        "future_any_drop_within_4cycles",
        "future_any_drop_within_8cycles",
        "future_sync2_drop_within_8cycles",
        "future_any_drop_within_16cycles",
    ]
    state_rows = []
    for state, g in df.groupby("cycle_state_cluster"):
        row = {
            "cycle_state_cluster": int(state),
            "n_cycles": int(len(g)),
            "cycle_min": float(g["cycleNo"].min()),
            "cycle_max": float(g["cycleNo"].max()),
            "median_degradation_axis": float(g["degradation_state_axis"].median()),
            "median_state_step_norm": float(g["state_step_norm"].median(skipna=True)),
        }
        for col in event_cols:
            if col in g.columns:
                row[f"{col}_rate"] = float(pd.to_numeric(g[col], errors="coerce").mean())
        state_rows.append(row)
    state_summary = pd.DataFrame(state_rows).sort_values("median_degradation_axis", ascending=False)

    transitions = []
    for prev, cur in zip(df.iloc[:-1].to_dict("records"), df.iloc[1:].to_dict("records")):
        transitions.append(
            {
                "cycleNo_prev": prev["cycleNo"],
                "cycleNo_next": cur["cycleNo"],
                "state_prev": int(prev["cycle_state_cluster"]),
                "state_next": int(cur["cycle_state_cluster"]),
                "transition": f"{int(prev['cycle_state_cluster'])}->{int(cur['cycle_state_cluster'])}",
                "cycle_gap": cur.get("cycle_gap", np.nan),
                "state_step_norm": cur.get("state_step_norm", np.nan),
                "axis_step": cur.get("axis_step", np.nan),
                "next_any_abrupt_drop": cur.get("any_abrupt_drop", np.nan),
                "next_future_any_drop_within_8cycles": cur.get("future_any_drop_within_8cycles", np.nan),
            }
        )
    trans_df = pd.DataFrame(transitions)
    transition_summary = (
        trans_df.groupby("transition", dropna=False)
        .agg(
            n_transitions=("transition", "count"),
            median_state_step_norm=("state_step_norm", "median"),
            median_axis_step=("axis_step", "median"),
            next_drop_rate=("next_any_abrupt_drop", "mean"),
            next_future8_rate=("next_future_any_drop_within_8cycles", "mean"),
        )
        .reset_index()
        .sort_values(["next_future8_rate", "median_state_step_norm"], ascending=False)
    )

    tests = binary_tests(
        df,
        [
            "degradation_state_axis",
            "state_step_norm",
            "axis_step",
            "cycle_state_pc1",
            "cycle_state_pc2",
            "particle_norm_range",
            "mean_abs_delta_prev",
            "shape_V_q95",
            "neg_dq_abs_peak_frac",
            "all_dq_abs_entropy",
            "capacity_mAh",
            "coulombic_efficiency_pct",
        ],
        "future_any_drop_within_8cycles",
        rng,
        args.n_permutation,
    )

    cv_features = [c for c in ["degradation_state_axis", "state_step_norm", "cycle_state_pc1", "cycle_state_pc2", "cycle_state_pc3", "particle_norm_range", "mean_abs_delta_prev", "capacity_mAh", "coulombic_efficiency_pct"] if c in df.columns]
    folds = cv_logistic(df, cv_features, "future_any_drop_within_8cycles", args.seed)
    temporal_folds = temporal_holdout_logistic(df, features, "future_any_drop_within_8cycles", args.seed)
    classifier_summary: Dict[str, Any] = {}
    if not folds.empty:
        evaluated_temporal = temporal_folds[temporal_folds.get("status").eq("evaluated")] if not temporal_folds.empty else pd.DataFrame()
        classifier_summary = {
            "target": "future_any_drop_within_8cycles",
            "n_folds": int(len(folds)),
            "mean_roc_auc": float(folds["roc_auc"].mean(skipna=True)),
            "mean_balanced_accuracy": float(folds["balanced_accuracy"].mean(skipna=True)),
            "features": cv_features,
            "temporal_holdout": {
                "n_blocks": int(len(temporal_folds)),
                "n_evaluated_blocks": int(len(evaluated_temporal)),
                "purge_horizon_cycles": 8,
                "mean_roc_auc": float(evaluated_temporal["roc_auc"].mean(skipna=True)) if not evaluated_temporal.empty else None,
                "mean_balanced_accuracy": float(evaluated_temporal["balanced_accuracy"].mean(skipna=True)) if not evaluated_temporal.empty else None,
                "features": features,
                "note": "Expanding-origin chronological blocks; train rows within 8 cycle numbers of each test block are purged and PCA is refit inside each training window.",
            },
        }

    corr_rows = []
    for target in ["cycleNo", "capacity_mAh", "coulombic_efficiency_pct", "frames_percentile"]:
        if target not in df.columns:
            continue
        for feat in ["degradation_state_axis", "state_step_norm", "cycle_state_pc1", "cycle_state_pc2"]:
            x = pd.to_numeric(df[feat], errors="coerce")
            y = pd.to_numeric(df[target], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() >= 8 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
                res = spearmanr(x[mask], y[mask])
                corr_rows.append({"feature": feat, "target": target, "rho": float(res.statistic), "p_value": float(res.pvalue), "n": int(mask.sum())})
    corr_df = pd.DataFrame(corr_rows).sort_values("p_value") if corr_rows else pd.DataFrame()

    loadings = []
    for pc_i in range(min(4, n_pcs)):
        comp = pca.components_[pc_i]
        for idx in np.argsort(np.abs(comp))[::-1][:12]:
            loadings.append(
                {
                    "pc": f"cycle_state_pc{pc_i+1}",
                    "feature": features[idx],
                    "feature_group": feature_group(features[idx]),
                    "loading": float(comp[idx]),
                    "abs_loading": float(abs(comp[idx])),
                }
            )
    loadings_df = pd.DataFrame(loadings)

    state_table_path = out / "cycle_state_space_table.csv"
    df.to_csv(state_table_path, index=False)
    k_path = out / "cycle_state_cluster_selection.csv"
    k_df.to_csv(k_path, index=False)
    state_summary_path = out / "cycle_state_cluster_summary.csv"
    state_summary.to_csv(state_summary_path, index=False)
    transition_path = out / "cycle_state_transition_summary.csv"
    transition_summary.to_csv(transition_path, index=False)
    tests_path = out / "cycle_state_future_drop_tests.csv"
    tests.to_csv(tests_path, index=False)
    folds_path = out / "cycle_state_future_drop_classifier_folds.csv"
    folds.to_csv(folds_path, index=False)
    temporal_folds_path = out / "cycle_state_future_drop_temporal_holdout.csv"
    temporal_folds.to_csv(temporal_folds_path, index=False)
    corr_path = out / "cycle_state_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    loadings_path = out / "cycle_state_pc_loadings.csv"
    loadings_df.to_csv(loadings_path, index=False)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(df["cycle_state_pc1"], df["cycle_state_pc2"], c=df["future_any_drop_within_8cycles"], cmap="coolwarm", s=55, edgecolor="black", linewidth=0.3)
        ax.plot(df["cycle_state_pc1"], df["cycle_state_pc2"], color="0.7", linewidth=0.8, zorder=0)
        ax.set_xlabel("cycle_state_pc1")
        ax.set_ylabel("cycle_state_pc2")
        ax.set_title("Cycle State Space: future drop within 8 cycles")
        fig.colorbar(sc, ax=ax, label="future_any_drop_within_8cycles")
        fig.tight_layout()
        fig.savefig(out / "cycle_state_space_pc1_pc2.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass

    summary = {
        "n_cycles": int(len(df)),
        "n_trace_cycles": int(len(trace)),
        "n_echem_shape_cycles_joined": int(df["echem_shape_points"].notna().sum()) if "echem_shape_points" in df else 0,
        "n_features_used": int(len(features)),
        "feature_group_counts": clean_json(pd.Series([feature_group(c) for c in features]).value_counts().to_dict()),
        "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_],
        "degradation_axis_oriented_to": orient_target,
        "degradation_axis_orientation_rho": orient_rho,
        "chosen_k": chosen_k,
        "best_silhouette": float(k_df.iloc[0]["silhouette"]) if not k_df.empty else None,
        "future_drop_classifier": clean_json(classifier_summary),
        "top_future_drop_tests": clean_json(tests.head(10).to_dict("records")) if not tests.empty else [],
        "top_cycle_state_correlations": clean_json(corr_df.head(10).to_dict("records")) if not corr_df.empty else [],
        "top_state_clusters": clean_json(state_summary.head(8).to_dict("records")),
        "top_transitions": clean_json(transition_summary.head(8).to_dict("records")),
        "top_pc_loadings": clean_json(loadings_df.head(24).to_dict("records")),
        "guardrail": "Cycle state-space clusters use four-particle trace summaries and echem-shape descriptors at cycle resolution. They are degradation-state hypotheses and early-warning covariates, not localized ROI/front validation or calibrated diffusion measurements.",
        "outputs": {
            "cycle_state_table": str(state_table_path),
            "cluster_selection": str(k_path),
            "cluster_summary": str(state_summary_path),
            "transition_summary": str(transition_path),
            "future_drop_tests": str(tests_path),
            "classifier_folds": str(folds_path),
            "temporal_holdout": str(temporal_folds_path),
            "correlations": str(corr_path),
            "pc_loadings": str(loadings_path),
            "summary": str(out / "cycle_state_space_transition_audit_summary.json"),
        },
    }
    with (out / "cycle_state_space_transition_audit_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)

    lines = [
        "# Cycle State-Space Transition Audit",
        "",
        "Cycle-level degradation-state audit using four-particle photometry traces joined to within-cycle echem-shape descriptors.",
        "",
        f"- Cycles scored: {summary['n_cycles']}",
        f"- Features used: {summary['n_features_used']}",
        f"- Joined echem-shape cycles: {summary['n_echem_shape_cycles_joined']}",
        f"- Chosen state clusters: {summary['chosen_k']} (silhouette {summary['best_silhouette']:.3f})",
        f"- Degradation axis oriented to: {summary['degradation_axis_oriented_to']} (rho {summary['degradation_axis_orientation_rho']})",
    ]
    if classifier_summary:
        lines.append(f"- Future-drop classifier mean AUC: {classifier_summary['mean_roc_auc']:.3f}; balanced accuracy {classifier_summary['mean_balanced_accuracy']:.3f}")
        temporal = classifier_summary.get("temporal_holdout", {})
        if temporal.get("mean_roc_auc") is not None:
            lines.append(
                f"- Temporal holdout mean AUC: {temporal['mean_roc_auc']:.3f}; balanced accuracy {temporal['mean_balanced_accuracy']:.3f} "
                f"across {temporal['n_evaluated_blocks']} evaluated blocks"
            )
    lines += ["", "## Top Future-Drop Associations", ""]
    for row in summary["top_future_drop_tests"][:6]:
        lines.append(
            f"- {row['feature']}: positive-negative median {row['median_positive_minus_negative']:.3f}, permutation p={row['permutation_p']:.3g}"
        )
    lines += ["", "## Top State Clusters", ""]
    for row in summary["top_state_clusters"][:5]:
        lines.append(
            f"- state {row['cycle_state_cluster']}: n={row['n_cycles']}, cycles {row['cycle_min']:.0f}-{row['cycle_max']:.0f}, future8 rate={row.get('future_any_drop_within_8cycles_rate')}"
        )
    lines += ["", "## Interpretation", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
