#!/usr/bin/env python3
"""Leave-source multimodal pre-event predictor for source-balanced ROI crops.

The phase-kinetics audit found strong masked optical kinetics but also high
source structure. This follow-up asks the practical modeling question: do
masked kinetics add predictive value to existing echem, front/residual,
consensus, and visual-QC features under leave-source evaluation?

The output is a model-comparison audit, not a deployable warning model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


TARGET_SPECS = {
    "near_vs_post_control": ({"near_pre_event_1_8"}, {"post_event_1_16", "no_near_event_control"}),
    "near_vs_far_pre": ({"near_pre_event_1_8"}, {"far_pre_event_17_32"}),
    "clean_pre16_vs_post_control": ({"near_pre_event_1_8", "mid_pre_event_9_16"}, {"post_event_1_16", "no_near_event_control"}),
}
METHODS = ["raw_standard", "source_mean_resid_2", "source_confound_filter_0.25"]


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


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 16) -> List[str]:
    out = []
    for col in cols:
        vals = numeric(df, col)
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            out.append(col)
    return out


def source_eta2(values: pd.DataFrame, sources: pd.Series) -> pd.Series:
    vals = values.apply(pd.to_numeric, errors="coerce")
    src = sources.astype(str)
    out = {}
    for col in vals.columns:
        x = vals[col]
        valid = x.notna() & src.notna()
        xv = x[valid]
        sv = src[valid]
        if len(xv) < 4 or xv.nunique(dropna=True) < 2 or sv.nunique(dropna=True) < 2:
            out[col] = 0.0
            continue
        mean = float(xv.mean())
        total = float(((xv - mean) ** 2).sum())
        if total <= 0:
            out[col] = 0.0
            continue
        between = 0.0
        for _, sub in xv.groupby(sv):
            between += len(sub) * float((sub.mean() - mean) ** 2)
        out[col] = float(np.clip(between / total, 0.0, 1.0))
    return pd.Series(out, dtype=float)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = out["event_relative_bin"].astype(str)
    for target, (pos, neg) in TARGET_SPECS.items():
        out[target] = np.nan
        out.loc[bins.isin(pos), target] = 1
        out.loc[bins.isin(neg), target] = 0
    return out


def feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    echem = [
        "capacity_fraction_of_first",
        "coulombic_inefficiency_pct",
        "echem_regime_pc1",
        "echem_regime_pc2",
        "shape_V_mean",
        "shape_I_abs_mean_mA",
        "all_dq_abs_entropy",
    ]
    residual_front = [
        c for c in df.columns
        if c.endswith("_source_echem_context_residual")
        and any(k in c for k in ["front", "mask", "phase_fraction", "kymograph", "diffusion", "background"])
    ]
    consensus_qc = [
        "consensus_review_score",
        "matched_positive_support_count",
        "visual_sanity_score",
        "visual_review_score",
        "visual_front_plausibility_score",
    ]
    kinetics = [
        c for c in df.columns
        if (
            c.startswith("masked_")
            or c.startswith("q55_")
            or c.startswith("q65_")
            or c.startswith("q75_")
            or c.startswith("particle_centroid_")
            or c.startswith("stable_particle_")
            or c.startswith("frame_mask_")
        )
        and "variation_to_net" not in c
        and not c.endswith("_direction")
    ]
    object_context = [
        "object_x_full_approx",
        "object_y_full_approx",
        "object_area_ds_px",
        "object_mean_residual",
        "object_mean_abs_z",
        "stage_drift_xy_sampled",
    ]
    families = {
        "echem_context": echem,
        "front_echem_residual": residual_front,
        "consensus_visual_qc": consensus_qc,
        "phase_kinetics": kinetics,
        "no_kinetics_signal": sorted(set(echem + residual_front + consensus_qc)),
        "kinetics_plus_front_echem": sorted(set(kinetics + residual_front + echem)),
        "all_signal": sorted(set(echem + residual_front + consensus_qc + kinetics)),
        "object_context_guardrail": object_context,
    }
    return {name: available_numeric(df, cols) for name, cols in families.items() if available_numeric(df, cols)}


def source_mean_basis(x_scaled: np.ndarray, sources: pd.Series, max_k: int) -> np.ndarray:
    src = sources.reset_index(drop=True).astype(str)
    means = []
    for source in sorted(src.unique()):
        rows = np.asarray(src == source)
        if rows.sum():
            means.append(x_scaled[rows].mean(axis=0))
    if len(means) < 2:
        return np.zeros((x_scaled.shape[1], 0))
    mat = np.vstack(means)
    mat -= mat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(mat, full_matrices=False)
    k = min(max_k, vt.shape[0], x_scaled.shape[1])
    return vt[:k].T if k > 0 else np.zeros((x_scaled.shape[1], 0))


def transform_fold(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_sources: pd.Series,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"n_features_before": int(train_x.shape[1]), "n_features_after": int(train_x.shape[1])}
    if method.startswith("source_confound_filter_") and train_x.shape[1] > 1:
        frac = float(method.rsplit("_", 1)[-1])
        eta = source_eta2(train_x, train_sources)
        n_drop = min(int(np.floor(len(eta) * frac)), len(eta) - 1)
        keep = eta.sort_values(ascending=False).index[n_drop:].tolist() if n_drop > 0 else eta.index.tolist()
        train_x = train_x[keep]
        test_x = test_x[keep]
        meta.update({"n_dropped_source_confounded_features": int(n_drop), "n_features_after": int(len(keep))})
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(imputer.fit_transform(train_x))
    xte = scaler.transform(imputer.transform(test_x))
    if method.startswith("source_mean_resid_"):
        k = int(method.rsplit("_", 1)[-1])
        basis = source_mean_basis(xtr, train_sources, k)
        if basis.shape[1] > 0:
            xtr = xtr - xtr @ basis @ basis.T
            xte = xte - xte @ basis @ basis.T
        meta["removed_source_mean_components"] = int(basis.shape[1])
    return xtr, xte, meta


def leave_source_predictions(df: pd.DataFrame, cols: List[str], target: str, family: str, method: str, seed: int) -> pd.DataFrame:
    y = numeric(df, target)
    valid = y.isin([0, 1]) & df["source_stem"].notna()
    features = df[cols].apply(pd.to_numeric, errors="coerce")
    rows: List[Dict[str, Any]] = []
    for source in sorted(df.loc[valid, "source_stem"].astype(str).unique()):
        test = valid & df["source_stem"].astype(str).eq(source)
        train = valid & ~test
        meta_cols = [c for c in ["roi_id", "cycleNo", "source_stem", "event_relative_bin", target] if c in df.columns]
        meta = df.loc[test, meta_cols].rename(columns={target: "observed"}).copy()
        fold_meta: Dict[str, Any] = {}
        if train.sum() < 16 or y[train].nunique() < 2 or len(cols) == 0:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped"
        else:
            xtr, xte, fold_meta = transform_fold(features.loc[train], features.loc[test], df.loc[train, "source_stem"], method)
            model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.2, solver="liblinear", random_state=seed)
            model.fit(xtr, y[train].astype(int))
            meta["predicted_probability"] = model.predict_proba(xte)[:, 1]
            meta["status"] = "ok"
        meta["heldout_source"] = source
        meta["feature_family"] = family
        meta["method"] = method
        meta["target"] = target
        for key, val in fold_meta.items():
            meta[key] = val
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def metric_row(pred: pd.DataFrame, family: str, method: str, target: str, cols: List[str], raw_eta: pd.Series) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    if tmp.empty:
        return {"target": target, "feature_family": family, "method": method, "n_features": len(cols), "n_eval": 0}
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    rho = pval = np.nan
    if len(tmp) >= 8 and y.nunique() == 2:
        stat = spearmanr(y, p)
        rho, pval = stat.statistic, stat.pvalue
    return {
        "target": target,
        "feature_family": family,
        "method": method,
        "n_features": int(len(cols)),
        "n_eval": int(len(tmp)),
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_sources": int(tmp["heldout_source"].nunique()) if "heldout_source" in tmp else 0,
        "n_positive": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if len(tmp) >= 8 and y.nunique() == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p)) if len(tmp) >= 8 and y.nunique() == 2 else np.nan,
        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        "spearman_p": float(pval) if np.isfinite(pval) else np.nan,
        "mean_raw_source_eta2": float(raw_eta.reindex(cols).mean()) if cols else np.nan,
        "max_raw_source_eta2": float(raw_eta.reindex(cols).max()) if cols else np.nan,
    }


def family_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    comparisons = [
        ("all_signal", "no_kinetics_signal"),
        ("kinetics_plus_front_echem", "front_echem_residual"),
        ("phase_kinetics", "front_echem_residual"),
    ]
    for target in metrics["target"].dropna().unique():
        for method in metrics["method"].dropna().unique():
            sub = metrics[(metrics["target"] == target) & (metrics["method"] == method)]
            for richer, base in comparisons:
                r = sub[sub["feature_family"] == richer]
                b = sub[sub["feature_family"] == base]
                if r.empty or b.empty:
                    continue
                rr = r.iloc[0]
                bb = b.iloc[0]
                rows.append({
                    "target": target,
                    "method": method,
                    "richer_family": richer,
                    "baseline_family": base,
                    "roc_auc_delta": float(rr.get("roc_auc", np.nan) - bb.get("roc_auc", np.nan)),
                    "average_precision_delta": float(rr.get("average_precision", np.nan) - bb.get("average_precision", np.nan)),
                    "richer_auc": float(rr.get("roc_auc", np.nan)),
                    "baseline_auc": float(bb.get("roc_auc", np.nan)),
                    "richer_ap": float(rr.get("average_precision", np.nan)),
                    "baseline_ap": float(bb.get("average_precision", np.nan)),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "method", "roc_auc_delta"], ascending=[True, True, False])
    return out


def top_univariate(df: pd.DataFrame, features: List[str], target: str, raw_eta: pd.Series, n: int = 32) -> List[Dict[str, Any]]:
    y = numeric(df, target)
    rows: List[Dict[str, Any]] = []
    for feature in features:
        x = numeric(df, feature)
        valid = y.isin([0, 1]) & x.notna()
        yy = y[valid].astype(int)
        xx = x[valid]
        if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
            continue
        direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
        score = xx if direction == "higher_in_positive" else -xx
        rows.append({
            "target": target,
            "feature": feature,
            "direction": direction,
            "n": int(len(yy)),
            "n_positive": int(yy.sum()),
            "oriented_auc": float(roc_auc_score(yy, score)),
            "average_precision": float(average_precision_score(yy, score)),
            "median_positive_minus_negative": float(xx[yy == 1].median() - xx[yy == 0].median()),
            "raw_source_eta2": float(raw_eta.get(feature, np.nan)),
        })
    return sorted(rows, key=lambda r: (r["oriented_auc"], r["average_precision"]), reverse=True)[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_phase_kinetics_audit/source_balanced_pre_event_phase_kinetics_features.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_multimodal_predictor")
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = add_targets(read_csv(Path(args.features)))
    families = feature_families(df)
    all_features = sorted(set(sum(families.values(), [])))
    raw_eta = source_eta2(df[all_features], df["source_stem"]) if all_features else pd.Series(dtype=float)

    preds = []
    metrics = []
    for target in TARGET_SPECS:
        for family, cols in families.items():
            for method in METHODS:
                pred = leave_source_predictions(df, cols, target, family, method, args.seed)
                preds.append(pred)
                metrics.append(metric_row(pred, family, method, target, cols, raw_eta))
    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["target", "roc_auc", "average_precision"], ascending=[True, False, False])
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    deltas = family_deltas(metrics_df)
    uni_rows = []
    for target in TARGET_SPECS:
        uni_rows.extend(top_univariate(df, all_features, target, raw_eta))
    uni_df = pd.DataFrame(uni_rows)

    paths = {
        "metrics": out / "source_balanced_pre_event_multimodal_metrics.csv",
        "predictions": out / "source_balanced_pre_event_multimodal_predictions.csv",
        "family_deltas": out / "source_balanced_pre_event_multimodal_family_deltas.csv",
        "univariate": out / "source_balanced_pre_event_multimodal_univariate.csv",
        "summary": out / "source_balanced_pre_event_multimodal_summary.json",
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    deltas.to_csv(paths["family_deltas"], index=False)
    uni_df.to_csv(paths["univariate"], index=False)

    best_by_target = []
    for target, sub in metrics_df.groupby("target", dropna=False):
        best_by_target.extend(sub.head(8).to_dict("records"))
    best_deltas = deltas.sort_values(["roc_auc_delta", "average_precision_delta"], ascending=False).head(20).to_dict("records") if not deltas.empty else []
    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "event_relative_bin_counts": df["event_relative_bin"].astype(str).value_counts().to_dict(),
        "targets": list(TARGET_SPECS.keys()),
        "methods": METHODS,
        "feature_family_sizes": {k: len(v) for k, v in families.items()},
        "best_by_target": best_by_target,
        "best_family_deltas": best_deltas,
        "top_univariate": uni_rows[:32],
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Leave-source multimodal pre-event models use automatic ROI crops, weak event-relative labels, and analysis-time source-confound transforms. They test whether masked phase kinetics add source-heldout signal to echem/front/QC features; they are not deployable warnings, manual labels, causal mechanisms, calibrated phase-boundaries, or diffusion coefficients.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = [
        "# Source-Balanced Pre-Event Multimodal Predictor",
        "",
        "Leave-source logistic model comparison over echem, front/residual, consensus/QC, and masked phase-kinetic feature families.",
        "",
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Feature family sizes: {summary['feature_family_sizes']}",
        "",
        "## Top Models",
    ]
    for row in best_by_target[:10]:
        lines.append(
            f"- {row.get('target')} {row.get('feature_family')} {row.get('method')}: AUC={row.get('roc_auc'):.3f}, AP={row.get('average_precision'):.3f}, n={row.get('n_eval')}"
        )
    lines += ["", "## Kinetics Deltas"]
    for row in best_deltas[:8]:
        lines.append(
            f"- {row.get('target')} {row.get('method')} {row.get('richer_family')} vs {row.get('baseline_family')}: dAUC={row.get('roc_auc_delta'):.3f}, dAP={row.get('average_precision_delta'):.3f}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out), "best_by_target": best_by_target[:6], "best_family_deltas": best_deltas[:6]}, indent=2))


if __name__ == "__main__":
    main()
