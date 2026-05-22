#!/usr/bin/env python3
"""Echem/context-conditioned residual-dictionary video objective audit.

The prior fusion audit appended echem context to residual-dictionary features.
This script asks a stricter question: after predicting residual-dictionary
features from echem/acquisition context inside each held-out split, do the
remaining video residual modes still carry useful weak-label or physics signal?
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]


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


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 12) -> List[str]:
    out: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            out.append(col)
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["embedding_cohort", "cohort_role", "selection_subrole", "source_stem", "shape_block_mode"]:
        if col not in out.columns:
            continue
        vals = out[col].fillna("missing").astype(str)
        for val in vals.value_counts().head(12).index:
            safe = "".join(ch if ch.isalnum() else "_" for ch in val.lower()).strip("_")
            out[f"{col}__{safe}"] = (vals == val).astype(float)
    return out


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    residual = available_numeric(
        df,
        [
            c for c in df.columns
            if c.startswith("resdict_")
            or c.startswith("residual_energy_")
            or c.startswith("dictionary_recon_")
        ],
    )
    handcrafted = available_numeric(
        df,
        [
            c for c in df.columns
            if c.startswith("particle_")
            or c.startswith("particle_vs_context_")
            or c.startswith("mask_")
        ],
    )
    echem = available_numeric(
        df,
        [
            c for c in df.columns
            if c.startswith("shape_")
            or c.startswith("all_dq")
            or c.startswith("pos_dq")
            or c.startswith("neg_dq")
            or c.startswith("dqdv")
            or c.startswith("cycle_state_")
            or c.startswith("echem_regime_")
            or c in {
                "capacity_mAh",
                "capacity_fraction_of_first",
                "capacity_fade_from_first_mAh",
                "coulombic_efficiency_pct",
                "coulombic_inefficiency_pct",
                "charge_discharge_capacity_gap_mAh",
                "charge_discharge_capacity_abs_gap_mAh",
                "signed_charge_fraction",
                "state_step_norm",
                "axis_step",
                "echem_outlier_score",
            }
        ],
    )
    context = available_numeric(
        df,
        [
            "cycleNo",
            "n_frames",
            "frames_percentile",
            "cycle_gap",
            "n_points",
            "echem_shape_points",
            "echem_shape_duration_s",
            "particle_prior_area_fraction",
            "mask_prior_area_fraction",
            "mask_fallback_frame_fraction",
            "mask_fragmented_fraction",
            "mask_mask_instability_score",
        ]
        + [
            c for c in df.columns
            if c.startswith("embedding_cohort__")
            or c.startswith("cohort_role__")
            or c.startswith("selection_subrole__")
            or c.startswith("source_stem__")
            or c.startswith("shape_block_mode__")
        ],
    )
    return {
        "residual_dictionary_raw": residual,
        "handcrafted_scalar": handcrafted,
        "echem_context": sorted(set(echem + context)),
        "conditioning_context": sorted(set(echem + context)),
        "raw_residual_plus_echem_context": sorted(set(residual + echem + context)),
        "handcrafted_plus_echem_context": sorted(set(handcrafted + echem + context)),
    }


def class_model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=0.2, solver="liblinear", random_state=seed),
    )


def ridge_model(alpha: float = 2.0) -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def split_masks(df: pd.DataFrame, split: str) -> List[Tuple[str, pd.Series, pd.Series]]:
    if split == "leave_cycle":
        keys = sorted(pd.to_numeric(df["cycleNo"], errors="coerce").dropna().unique())
        return [(str(int(k)), pd.to_numeric(df["cycleNo"], errors="coerce").ne(k), pd.to_numeric(df["cycleNo"], errors="coerce").eq(k)) for k in keys]
    if split == "leave_source":
        keys = sorted(df["source_stem"].dropna().astype(str).unique())
        return [(k, df["source_stem"].astype(str).ne(k), df["source_stem"].astype(str).eq(k)) for k in keys]
    raise ValueError(split)


def conditioned_residual_features(
    df: pd.DataFrame,
    residual_cols: List[str],
    conditioning_cols: List[str],
    split: str,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    residual_cols = [c for c in residual_cols if c in df.columns]
    conditioning_cols = [c for c in conditioning_cols if c in df.columns and c not in residual_cols]
    pred = pd.DataFrame(index=df.index)
    resid = pd.DataFrame(index=df.index)
    raw_y = df[residual_cols].apply(pd.to_numeric, errors="coerce")
    for fold_key, train, test in split_masks(df, split):
        if test.sum() == 0 or train.sum() < 12:
            continue
        y_train = raw_y.loc[train]
        keep = [c for c in residual_cols if y_train[c].notna().sum() >= 8 and y_train[c].nunique(dropna=True) >= 2]
        if not keep:
            continue
        model = ridge_model(alpha)
        model.fit(df.loc[train, conditioning_cols], y_train[keep])
        yhat = pd.DataFrame(model.predict(df.loc[test, conditioning_cols]), index=df.index[test], columns=keep)
        for col in keep:
            pred.loc[test, f"{col}__echem_pred_{split}"] = yhat[col]
            resid.loc[test, f"{col}__echem_conditioned_resid_{split}"] = raw_y.loc[test, col] - yhat[col]
    return pred, resid


def predict_labels(df: pd.DataFrame, features: List[str], target: str, split: str, seed: int) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1])
    features = [c for c in features if c in df.columns and c != target]
    for fold_key, train_all, test_all in split_masks(df, split):
        train = train_all & valid
        test = test_all & valid
        meta = df.loc[test, ["embedding_row_id", "roi_id", "cycleNo", "source_stem", target]].rename(columns={target: "observed"}).copy()
        meta["split"] = split
        meta["fold_key"] = fold_key
        if test.sum() == 0:
            continue
        if train.sum() < 12 or y[train].nunique() < 2 or not features:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_class_or_features"
        else:
            model = class_model(seed)
            model.fit(df.loc[train, features], y[train].astype(int))
            meta["predicted_probability"] = model.predict_proba(df.loc[test, features])[:, 1]
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def class_metrics(pred: pd.DataFrame, feature_set: str, target: str, split: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce")
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    row: Dict[str, Any] = {
        "task": "classification",
        "split": split,
        "feature_set": feature_set,
        "target": target,
        "n_eval": int(len(tmp)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_sources": int(tmp["source_stem"].nunique()) if "source_stem" in tmp else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2 and p.nunique() > 1:
        row["roc_auc"] = float(roc_auc_score(y.astype(int), p))
        row["average_precision"] = float(average_precision_score(y.astype(int), p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
        row["spearman_p"] = float(sp) if np.isfinite(sp) else np.nan
    return row


def permutation_null(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce").to_numpy()
    if not np.isfinite(observed_auc) or len(y) < 8 or len(np.unique(y)) < 2:
        return {"n_permutation": 0, "empirical_p_ge_observed": np.nan, "null_auc_mean": np.nan, "null_auc_p95": np.nan}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        yy = y.copy()
        rng.shuffle(yy)
        if len(np.unique(yy)) == 2:
            vals.append(float(roc_auc_score(yy, p)))
    arr = np.asarray(vals)
    return {
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)),
        "null_auc_mean": float(arr.mean()) if len(arr) else np.nan,
        "null_auc_p95": float(np.quantile(arr, 0.95)) if len(arr) else np.nan,
    }


def residual_context_fit_metrics(df: pd.DataFrame, pred_cols: List[str], residual_cols: List[str], split: str) -> pd.DataFrame:
    rows = []
    for raw_col in residual_cols:
        pred_col = f"{raw_col}__echem_pred_{split}"
        resid_col = f"{raw_col}__echem_conditioned_resid_{split}"
        if pred_col not in df.columns or resid_col not in df.columns:
            continue
        y = pd.to_numeric(df[raw_col], errors="coerce")
        p = pd.to_numeric(df[pred_col], errors="coerce")
        r = pd.to_numeric(df[resid_col], errors="coerce")
        mask = y.notna() & p.notna()
        if mask.sum() < 8:
            continue
        rho, sp = spearmanr(y[mask], p[mask]) if y[mask].nunique() > 1 and p[mask].nunique() > 1 else (np.nan, np.nan)
        rows.append({
            "split": split,
            "feature": raw_col,
            "n": int(mask.sum()),
            "r2": float(r2_score(y[mask], p[mask])) if y[mask].nunique() > 1 else np.nan,
            "mae": float(mean_absolute_error(y[mask], p[mask])),
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_p": float(sp) if np.isfinite(sp) else np.nan,
            "residual_std": float(r[mask].std()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["r2", "spearman_rho"], ascending=False) if not out.empty else out


def metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("conditioned_residual_dictionary", "residual_dictionary_raw"),
        ("conditioned_residual_plus_echem_context", "echem_context"),
        ("conditioned_residual_plus_echem_context", "raw_residual_plus_echem_context"),
        ("conditioned_residual_plus_handcrafted_echem", "handcrafted_plus_echem_context"),
        ("conditioned_residual_plus_handcrafted_echem", "conditioned_residual_plus_echem_context"),
    ]
    for split in sorted(metrics["split"].dropna().unique()):
        for target in sorted(metrics["target"].dropna().unique()):
            sub = metrics[(metrics["split"] == split) & (metrics["target"] == target)]
            for comp, base in comparisons:
                c = sub[sub["feature_set"] == comp]
                b = sub[sub["feature_set"] == base]
                if c.empty or b.empty:
                    continue
                cr, br = c.iloc[0], b.iloc[0]
                rows.append({
                    "split": split,
                    "target": target,
                    "comparison": f"{comp}_minus_{base}",
                    "delta_roc_auc": float(cr.get("roc_auc", np.nan) - br.get("roc_auc", np.nan)) if pd.notna(cr.get("roc_auc")) and pd.notna(br.get("roc_auc")) else np.nan,
                    "delta_average_precision": float(cr.get("average_precision", np.nan) - br.get("average_precision", np.nan)) if pd.notna(cr.get("average_precision")) and pd.notna(br.get("average_precision")) else np.nan,
                    "delta_spearman_rho": float(cr.get("spearman_rho", np.nan) - br.get("spearman_rho", np.nan)) if pd.notna(cr.get("spearman_rho")) and pd.notna(br.get("spearman_rho")) else np.nan,
                    "comparison_auc": cr.get("roc_auc"),
                    "base_auc": br.get("roc_auc"),
                })
    out = pd.DataFrame(rows)
    return out.sort_values("delta_roc_auc", ascending=False) if not out.empty else out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_conditioned_residual_dictionary")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--n-permutation", type=int, default=300)
    parser.add_argument("--ridge-alpha", type=float, default=3.0)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = read_csv(derived / "echem_residual_dictionary_fusion_audit" / "echem_residual_dictionary_joined_features.csv")
    df = add_indicators(df)
    feature_sets = build_feature_sets(df)
    residual_cols = feature_sets["residual_dictionary_raw"]
    conditioning_cols = feature_sets["conditioning_context"]
    handcrafted = feature_sets["handcrafted_scalar"]

    conditioned_frames = [df.copy()]
    fit_metric_frames = []
    for split in ["leave_cycle", "leave_source"]:
        pred, resid = conditioned_residual_features(df, residual_cols, conditioning_cols, split, args.ridge_alpha)
        conditioned_frames.append(pred)
        conditioned_frames.append(resid)
    conditioned = pd.concat(conditioned_frames, axis=1)
    conditioned = conditioned.loc[:, ~conditioned.columns.duplicated()].copy()

    split_feature_sets: Dict[str, Dict[str, List[str]]] = {}
    for split in ["leave_cycle", "leave_source"]:
        resid_cols = [c for c in conditioned.columns if c.endswith(f"__echem_conditioned_resid_{split}")]
        pred_cols = [c for c in conditioned.columns if c.endswith(f"__echem_pred_{split}")]
        fit_metric_frames.append(residual_context_fit_metrics(conditioned, pred_cols, residual_cols, split))
        split_feature_sets[split] = {
            "residual_dictionary_raw": residual_cols,
            "conditioned_residual_dictionary": resid_cols,
            "echem_context": conditioning_cols,
            "raw_residual_plus_echem_context": sorted(set(residual_cols + conditioning_cols)),
            "conditioned_residual_plus_echem_context": sorted(set(resid_cols + conditioning_cols)),
            "handcrafted_plus_echem_context": sorted(set(handcrafted + conditioning_cols)),
            "conditioned_residual_plus_handcrafted_echem": sorted(set(resid_cols + handcrafted + conditioning_cols)),
        }

    pred_frames = []
    metric_rows = []
    null_rows = []
    for split, sets in split_feature_sets.items():
        for target in [t for t in TARGETS if t in conditioned.columns]:
            for name, cols in sets.items():
                if not cols:
                    continue
                pred = predict_labels(conditioned, cols, target, split, args.seed)
                if pred.empty:
                    continue
                pred.insert(0, "feature_set", name)
                pred_frames.append(pred)
                row = class_metrics(pred, name, target, split)
                metric_rows.append(row)
                if name in {"residual_dictionary_raw", "conditioned_residual_dictionary", "echem_context", "conditioned_residual_plus_echem_context", "conditioned_residual_plus_handcrafted_echem"}:
                    null_rows.append({"split": split, "feature_set": name, "target": target, **permutation_null(pred, row.get("roc_auc", np.nan), args.seed + 17, args.n_permutation)})

    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows).sort_values(["target", "split", "roc_auc"], ascending=[True, True, False]) if metric_rows else pd.DataFrame()
    nulls = pd.DataFrame(null_rows)
    if not metrics.empty and not nulls.empty:
        metrics = metrics.merge(nulls, on=["split", "feature_set", "target"], how="left")
    deltas = metric_deltas(metrics) if not metrics.empty else pd.DataFrame()
    fit_metrics = pd.concat(fit_metric_frames, ignore_index=True) if fit_metric_frames else pd.DataFrame()

    paths = {
        "features": out / "echem_conditioned_residual_dictionary_features.csv",
        "context_fit_metrics": out / "echem_conditioned_residual_dictionary_context_fit_metrics.csv",
        "predictions": out / "echem_conditioned_residual_dictionary_predictions.csv",
        "metrics": out / "echem_conditioned_residual_dictionary_metrics.csv",
        "deltas": out / "echem_conditioned_residual_dictionary_deltas.csv",
        "permutation_null": out / "echem_conditioned_residual_dictionary_permutation_null.csv",
        "summary": out / "echem_conditioned_residual_dictionary_summary.json",
    }
    conditioned.to_csv(paths["features"], index=False)
    fit_metrics.to_csv(paths["context_fit_metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    deltas.to_csv(paths["deltas"], index=False)
    nulls.to_csv(paths["permutation_null"], index=False)

    summary = clean_json({
        "n_rows": int(len(conditioned)),
        "n_cycles": int(conditioned["cycleNo"].nunique()),
        "n_sources": int(conditioned["source_stem"].nunique()),
        "residual_dictionary_features": residual_cols,
        "n_residual_features": int(len(residual_cols)),
        "n_conditioning_features": int(len(conditioning_cols)),
        "feature_set_sizes": {split: {k: len(v) for k, v in sets.items()} for split, sets in split_feature_sets.items()},
        "top_metrics": metrics.head(40).to_dict("records") if not metrics.empty else [],
        "top_deltas": deltas.head(30).to_dict("records") if not deltas.empty else [],
        "top_context_fit_metrics": fit_metrics.head(20).to_dict("records") if not fit_metrics.empty else [],
        "guardrail": "Echem-conditioned residual dictionary features are split-specific residuals from echem/acquisition predictions of label-free residual bases. They test whether video residual modes add signal beyond measured context, not deployable warning, manual QC, causal mechanism, or calibrated diffusion.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Echem-Conditioned Residual Dictionary",
        "",
        "Split-specific residual-dictionary features after subtracting echem/acquisition-predicted residual bases.",
        "",
        f"- Rows: {summary['n_rows']}",
        f"- Cycles: {summary['n_cycles']}",
        f"- Sources: {summary['n_sources']}",
        f"- Residual dictionary features: {summary['n_residual_features']}",
        f"- Conditioning features: {summary['n_conditioning_features']}",
        "",
        "## Top Metrics",
        "",
    ]
    for row in summary["top_metrics"][:12]:
        lines.append(
            f"- {row.get('split')} {row.get('target')} {row.get('feature_set')}: AUC {row.get('roc_auc')}, AP {row.get('average_precision')}, p={row.get('empirical_p_ge_observed')}"
        )
    lines += ["", "## Deltas", ""]
    for row in summary["top_deltas"][:10]:
        lines.append(
            f"- {row.get('split')} {row.get('target')} {row.get('comparison')}: delta AUC {row.get('delta_roc_auc')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
