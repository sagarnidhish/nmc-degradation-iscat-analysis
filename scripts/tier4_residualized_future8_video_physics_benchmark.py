#!/usr/bin/env python3
"""Benchmark video-physics signal after acquisition/context residualization.

The echem residual-dictionary fusion audit showed that future-8 weak labels are
strongly acquisition/context structured. This audit asks the narrower question:
do residual-dictionary, hand-crafted particle, or video-PCA descriptors retain
predictive signal after conditioning on acquisition and echem context?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ID_COLS = {"embedding_row_id", "roi_id", "cycleNo"}
TARGET_PATTERNS = (
    "future_",
    "past_",
    "cycles_to_",
    "cycles_since_",
    "any_abrupt_drop",
    "synchronized_drop",
    "synchronized_multimodal",
    "multimodal_outlier",
    "consensus_class",
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
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def is_forbidden_feature(col: str) -> bool:
    if col in ID_COLS:
        return True
    return any(col.startswith(p) for p in TARGET_PATTERNS)


def add_context_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    cats = [c for c in ["embedding_cohort", "cohort_role", "selection_subrole", "source_stem"] if c in df.columns]
    if not cats:
        return df
    dummies = pd.get_dummies(df[cats].fillna("missing").astype(str), prefix=cats, dummy_na=False)
    dummies = dummies[[c for c in dummies.columns if c not in df.columns]]
    return pd.concat([df, dummies], axis=1)


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 16) -> List[str]:
    keep: List[str] = []
    seen = set()
    for col in cols:
        if col in seen or col not in df.columns or is_forbidden_feature(col):
            continue
        seen.add(col)
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        vals = pd.to_numeric(series, errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def build_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    context_prefixes = ("embedding_cohort_", "cohort_role_", "selection_subrole_", "source_stem_")
    context = available_numeric(
        df,
        [
            "frames_percentile",
            "n_frames",
            "cycle_index_rank",
            "echem_shape_missing",
            "echem_ce_extreme_or_missing",
            *[c for c in df.columns if c.startswith(context_prefixes)],
        ],
    )
    echem = available_numeric(
        df,
        [
            c
            for c in df.columns
            if c.startswith(("shape_", "all_dq", "pos_dq", "neg_dq", "cycle_state_", "echem_regime_"))
            or c
            in {
                "capacity_mAh",
                "capacity_fraction_of_first",
                "capacity_fade_from_first_mAh",
                "coulombic_efficiency_pct",
                "coulombic_inefficiency_pct",
                "charge_discharge_capacity_gap_mAh",
                "charge_discharge_capacity_abs_gap_mAh",
                "echem_outlier_score",
                "state_step_norm",
                "axis_step",
                "signed_charge_fraction",
                "voltage_peak_hysteresis_proxy",
                "highV_charge_discharge_imbalance",
                "midV_charge_discharge_imbalance",
                "lowV_charge_discharge_imbalance",
                "dqdv_peak_concentration",
                "dqdv_entropy_asymmetry",
                "dqdv_integral_asymmetry",
            }
        ],
    )
    residual = available_numeric(
        df,
        [c for c in df.columns if c.startswith(("resdict_", "dictionary_", "residual_energy_"))],
    )
    handcrafted = available_numeric(
        df,
        [
            c
            for c in df.columns
            if c.startswith(("particle_", "mask_", "roi_", "phase_", "radius2_", "diffusion_", "threshold_robust_"))
            or c.endswith("_particle_mse_fraction_of_full_mean")
        ],
    )
    pca_video = available_numeric(df, [c for c in df.columns if c.startswith("video_embed_pc")])
    video_all = sorted(set(residual + handcrafted + pca_video))
    return {
        "acquisition_context": context,
        "echem_context": sorted(set(context + echem)),
        "residual_dictionary_raw": residual,
        "handcrafted_particle_raw": handcrafted,
        "video_pca_raw": pca_video,
        "all_video_raw": video_all,
        "context_plus_residual_dictionary": sorted(set(context + residual)),
        "context_plus_handcrafted_particle": sorted(set(context + handcrafted)),
        "context_plus_video_pca": sorted(set(context + pca_video)),
        "context_plus_all_video": sorted(set(context + video_all)),
        "echem_context_plus_all_video": sorted(set(context + echem + video_all)),
    }


def class_model(seed: int, c_value: float = 0.20) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=c_value, solver="liblinear", random_state=seed),
    )


def reg_model(alpha: float = 5.0) -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def valid_binary_rows(df: pd.DataFrame, target: str) -> pd.Series:
    y = pd.to_numeric(df[target], errors="coerce")
    return y.isin([0, 1])


def score_predictions(pred: pd.DataFrame, feature_set: str, target: str, task: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    row: Dict[str, Any] = {
        "task": task,
        "target": target,
        "feature_set": feature_set,
        "n_eval": int(len(tmp)),
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "n_positive": int(y.sum()) if len(y) else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def loo_classification(df: pd.DataFrame, features: Sequence[str], target: str, seed: int) -> pd.DataFrame:
    features = [c for c in features if c in df.columns and not is_forbidden_feature(c)]
    base_cols = ["embedding_row_id", "roi_id", "cycleNo", target]
    use = df[base_cols + features].copy()
    y = pd.to_numeric(use[target], errors="coerce")
    valid = y.isin([0, 1])
    rows: List[Dict[str, Any]] = []
    for cycle in sorted(pd.to_numeric(use.loc[valid, "cycleNo"], errors="coerce").dropna().unique()):
        test = valid & (pd.to_numeric(use["cycleNo"], errors="coerce") == cycle)
        train = valid & ~test
        meta = use.loc[test, base_cols].rename(columns={target: "observed"}).copy()
        if not features or train.sum() < 12 or y[train].nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_class_or_features"
        else:
            model = class_model(seed)
            model.fit(use.loc[train, features], y[train].astype(int))
            meta["predicted_probability"] = model.predict_proba(use.loc[test, features])[:, 1]
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def residualize_matrix(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_context: pd.DataFrame,
    test_context: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_cols: Dict[str, np.ndarray] = {}
    test_cols: Dict[str, np.ndarray] = {}
    for col in train_x.columns:
        out_col = f"{col}__ctx_resid"
        model = reg_model(alpha=5.0)
        y = pd.to_numeric(train_x[col], errors="coerce")
        if y.notna().sum() < 12 or y.nunique(dropna=True) < 2:
            train_cols[out_col] = np.zeros(len(train_x), dtype=float)
            test_cols[out_col] = np.zeros(len(test_x), dtype=float)
            continue
        model.fit(train_context, y)
        train_cols[out_col] = y.to_numpy(dtype=float) - model.predict(train_context)
        test_y = pd.to_numeric(test_x[col], errors="coerce")
        test_cols[out_col] = test_y.to_numpy(dtype=float) - model.predict(test_context)
    return pd.DataFrame(train_cols, index=train_x.index), pd.DataFrame(test_cols, index=test_x.index)

def loo_residualized_video(
    df: pd.DataFrame,
    context_features: Sequence[str],
    video_features: Sequence[str],
    target: str,
    seed: int,
    include_context_logit: bool,
) -> pd.DataFrame:
    context_features = [c for c in context_features if c in df.columns and not is_forbidden_feature(c)]
    video_features = [c for c in video_features if c in df.columns and not is_forbidden_feature(c)]
    base_cols = ["embedding_row_id", "roi_id", "cycleNo", target]
    use = df[base_cols + context_features + video_features].copy()
    y = pd.to_numeric(use[target], errors="coerce")
    valid = y.isin([0, 1])
    rows: List[Dict[str, Any]] = []
    for cycle in sorted(pd.to_numeric(use.loc[valid, "cycleNo"], errors="coerce").dropna().unique()):
        test = valid & (pd.to_numeric(use["cycleNo"], errors="coerce") == cycle)
        train = valid & ~test
        meta = use.loc[test, base_cols].rename(columns={target: "observed"}).copy()
        if (
            not context_features
            or not video_features
            or train.sum() < 12
            or y[train].nunique() < 2
        ):
            meta["predicted_probability"] = np.nan
            meta["context_probability"] = np.nan
            meta["status"] = "skipped_train_size_class_or_features"
            rows.extend(meta.to_dict("records"))
            continue

        context_clf = class_model(seed)
        context_clf.fit(use.loc[train, context_features], y[train].astype(int))
        train_context_prob = context_clf.predict_proba(use.loc[train, context_features])[:, 1]
        test_context_prob = context_clf.predict_proba(use.loc[test, context_features])[:, 1]

        train_res, test_res = residualize_matrix(
            use.loc[train, video_features],
            use.loc[test, video_features],
            use.loc[train, context_features],
            use.loc[test, context_features],
        )
        if include_context_logit:
            train_model_x = train_res.copy()
            test_model_x = test_res.copy()
            train_model_x["context_logit"] = logit(np.clip(train_context_prob, 1e-4, 1 - 1e-4))
            test_model_x["context_logit"] = logit(np.clip(test_context_prob, 1e-4, 1 - 1e-4))
        else:
            train_model_x = train_res
            test_model_x = test_res

        model = class_model(seed)
        model.fit(train_model_x, y[train].astype(int))
        meta["predicted_probability"] = model.predict_proba(test_model_x)[:, 1]
        meta["context_probability"] = test_context_prob
        meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    return pd.DataFrame(rows)


def context_residual_feature_tests(
    df: pd.DataFrame,
    context_features: Sequence[str],
    video_features: Sequence[str],
    target: str,
    seed: int,
) -> pd.DataFrame:
    context_features = [c for c in context_features if c in df.columns and not is_forbidden_feature(c)]
    video_features = [c for c in video_features if c in df.columns and not is_forbidden_feature(c)]
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1])
    rows: List[Dict[str, Any]] = []
    all_parts: List[pd.DataFrame] = []
    for cycle in sorted(pd.to_numeric(df.loc[valid, "cycleNo"], errors="coerce").dropna().unique()):
        test = valid & (pd.to_numeric(df["cycleNo"], errors="coerce") == cycle)
        train = valid & ~test
        if not context_features or not video_features or train.sum() < 12 or y[train].nunique() < 2:
            continue
        context_clf = class_model(seed)
        context_clf.fit(df.loc[train, context_features], y[train].astype(int))
        test_context_prob = context_clf.predict_proba(df.loc[test, context_features])[:, 1]
        train_res, test_res = residualize_matrix(
            df.loc[train, video_features],
            df.loc[test, video_features],
            df.loc[train, context_features],
            df.loc[test, context_features],
        )
        part = test_res.copy()
        part["cycleNo"] = df.loc[test, "cycleNo"].to_numpy()
        part["observed"] = y[test].to_numpy(dtype=float)
        part["context_probability"] = test_context_prob
        part["label_residual"] = part["observed"] - part["context_probability"]
        all_parts.append(part)
    if not all_parts:
        return pd.DataFrame()
    residual_table = pd.concat(all_parts, axis=0, ignore_index=True)
    label_resid = pd.to_numeric(residual_table["label_residual"], errors="coerce")
    observed = pd.to_numeric(residual_table["observed"], errors="coerce")
    for col in [c for c in residual_table.columns if c.endswith("__ctx_resid")]:
        vals = pd.to_numeric(residual_table[col], errors="coerce")
        ok = vals.notna() & label_resid.notna()
        if ok.sum() < 12 or vals[ok].nunique() < 2:
            continue
        rho, sp = spearmanr(vals[ok], label_resid[ok])
        auc = np.nan
        ok_auc = vals.notna() & observed.isin([0, 1])
        if ok_auc.sum() >= 8 and observed[ok_auc].nunique() == 2:
            auc = float(roc_auc_score(observed[ok_auc].astype(int), vals[ok_auc]))
            auc = max(auc, 1.0 - auc)
        rows.append(
            {
                "target": target,
                "feature": col.removesuffix("__ctx_resid"),
                "n_eval": int(ok.sum()),
                "spearman_abs_rho_vs_context_label_residual": abs(float(rho)),
                "spearman_rho_vs_context_label_residual": float(rho),
                "spearman_p": float(sp),
                "direction_free_auc_vs_label": auc,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["spearman_abs_rho_vs_context_label_residual", "direction_free_auc_vs_label"],
        ascending=[False, False],
    )


def permutation_null(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int, mode: str) -> Dict[str, Any]:
    if not np.isfinite(observed_auc) or n_perm <= 0:
        return {"null_mode": mode, "n_permutation": 0, "empirical_p_ge_observed": np.nan}
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int).to_numpy()
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce").to_numpy()
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    groups = tmp["cycleNo"].to_numpy() if mode == "cycle_block" else np.asarray(["all"] * len(tmp))
    for _ in range(n_perm):
        yy = y.copy()
        if mode == "cycle_block":
            unique_groups = np.array(sorted(pd.unique(groups)))
            shuffled_groups = unique_groups.copy()
            rng.shuffle(shuffled_groups)
            mapping = dict(zip(unique_groups, shuffled_groups))
            for idx, grp in enumerate(groups):
                source = groups == mapping[grp]
                if source.sum() == 1:
                    yy[idx] = y[source][0]
                else:
                    yy[idx] = rng.choice(y[source])
        else:
            rng.shuffle(yy)
        if len(np.unique(yy)) == 2:
            vals.append(float(roc_auc_score(yy, p)))
    if not vals:
        return {"null_mode": mode, "n_permutation": 0, "empirical_p_ge_observed": np.nan}
    arr = np.asarray(vals)
    return {
        "null_mode": mode,
        "n_permutation": int(len(arr)),
        "empirical_p_ge_observed": float((np.sum(arr >= observed_auc) + 1) / (len(arr) + 1)),
        "null_auc_mean": float(np.mean(arr)),
        "null_auc_p95": float(np.quantile(arr, 0.95)),
    }


def metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("context_plus_residual_dictionary", "acquisition_context"),
        ("context_plus_handcrafted_particle", "acquisition_context"),
        ("context_plus_video_pca", "acquisition_context"),
        ("context_plus_all_video", "acquisition_context"),
        ("echem_context_plus_all_video", "echem_context"),
        ("residualized_residual_dictionary_plus_context_logit", "acquisition_context"),
        ("residualized_handcrafted_particle_plus_context_logit", "acquisition_context"),
        ("residualized_video_pca_plus_context_logit", "acquisition_context"),
        ("residualized_all_video_plus_context_logit", "acquisition_context"),
        ("residual_dictionary_raw", "acquisition_context"),
        ("handcrafted_particle_raw", "acquisition_context"),
        ("video_pca_raw", "acquisition_context"),
        ("all_video_raw", "acquisition_context"),
    ]
    rows: List[Dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for target in sorted(metrics["target"].dropna().unique()):
        sub = metrics[metrics["target"] == target]
        for comp_name, base_name in comparisons:
            comp = sub[sub["feature_set"] == comp_name]
            base = sub[sub["feature_set"] == base_name]
            if comp.empty or base.empty:
                continue
            c = comp.iloc[0]
            b = base.iloc[0]
            rows.append(
                {
                    "target": target,
                    "comparison": f"{comp_name}_minus_{base_name}",
                    "base_roc_auc": b.get("roc_auc"),
                    "comparison_roc_auc": c.get("roc_auc"),
                    "delta_roc_auc": c.get("roc_auc") - b.get("roc_auc")
                    if pd.notna(c.get("roc_auc")) and pd.notna(b.get("roc_auc"))
                    else np.nan,
                    "base_average_precision": b.get("average_precision"),
                    "comparison_average_precision": c.get("average_precision"),
                    "delta_average_precision": c.get("average_precision") - b.get("average_precision")
                    if pd.notna(c.get("average_precision")) and pd.notna(b.get("average_precision"))
                    else np.nan,
                    "base_spearman_rho": b.get("spearman_rho"),
                    "comparison_spearman_rho": c.get("spearman_rho"),
                    "delta_spearman_rho": c.get("spearman_rho") - b.get("spearman_rho")
                    if pd.notna(c.get("spearman_rho")) and pd.notna(b.get("spearman_rho"))
                    else np.nan,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["delta_roc_auc", "delta_spearman_rho"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/acquisition_residualized_video_physics_benchmark")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-permutation", type=int, default=500)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = read_csv(derived / "echem_residual_dictionary_fusion_audit" / "echem_residual_dictionary_joined_features.csv")
    joined = add_context_dummies(joined)
    feature_groups = {k: v for k, v in build_feature_groups(joined).items() if v}
    context = feature_groups.get("acquisition_context", [])
    video_groups = {
        "residual_dictionary": feature_groups.get("residual_dictionary_raw", []),
        "handcrafted_particle": feature_groups.get("handcrafted_particle_raw", []),
        "video_pca": feature_groups.get("video_pca_raw", []),
        "all_video": feature_groups.get("all_video_raw", []),
    }
    targets = [
        t
        for t in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
        if t in joined.columns and valid_binary_rows(joined, t).sum() >= 16
    ]

    metrics: List[Dict[str, Any]] = []
    predictions: List[pd.DataFrame] = []
    nulls: List[Dict[str, Any]] = []
    feature_tests: List[pd.DataFrame] = []

    for target in targets:
        y = pd.to_numeric(joined[target], errors="coerce")
        if y[y.isin([0, 1])].nunique() < 2:
            continue
        for name, cols in feature_groups.items():
            pred = loo_classification(joined, cols, target, args.seed)
            pred["task"] = "classification"
            pred["target"] = target
            pred["feature_set"] = name
            predictions.append(pred)
            met = score_predictions(pred, name, target, "classification")
            metrics.append(met)
            if name in {
                "acquisition_context",
                "echem_context",
                "all_video_raw",
                "context_plus_all_video",
                "echem_context_plus_all_video",
            }:
                for mode in ["global", "cycle_block"]:
                    null = permutation_null(pred, met.get("roc_auc", np.nan), args.seed, args.n_permutation, mode)
                    null.update({"target": target, "feature_set": name, "observed_roc_auc": met.get("roc_auc", np.nan)})
                    nulls.append(null)

        for short_name, cols in video_groups.items():
            if not cols:
                continue
            for include_context_logit in [False, True]:
                name = f"residualized_{short_name}" + ("_plus_context_logit" if include_context_logit else "")
                pred = loo_residualized_video(joined, context, cols, target, args.seed, include_context_logit)
                pred["task"] = "classification"
                pred["target"] = target
                pred["feature_set"] = name
                predictions.append(pred)
                met = score_predictions(pred, name, target, "classification")
                metrics.append(met)
                for mode in ["global", "cycle_block"]:
                    null = permutation_null(pred, met.get("roc_auc", np.nan), args.seed, args.n_permutation, mode)
                    null.update({"target": target, "feature_set": name, "observed_roc_auc": met.get("roc_auc", np.nan)})
                    nulls.append(null)

            tests = context_residual_feature_tests(joined, context, cols, target, args.seed)
            if not tests.empty:
                tests["feature_group"] = short_name
                feature_tests.append(tests)

    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    null_df = pd.DataFrame(nulls)
    tests_df = pd.concat(feature_tests, ignore_index=True) if feature_tests else pd.DataFrame()
    delta_df = metric_deltas(metrics_df)
    if not metrics_df.empty and not null_df.empty:
        null_values = ["empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]
        null_wide = (
            null_df.pivot_table(
                index=["target", "feature_set"],
                columns="null_mode",
                values=null_values,
                aggfunc="first",
            )
            .reset_index()
        )
        null_wide.columns = [
            f"{value}_{mode}" if mode else value
            for value, mode in null_wide.columns.to_flat_index()
        ]
        metrics_df = metrics_df.merge(null_wide, on=["target", "feature_set"], how="left")

    paths = {
        "metrics": out / "acquisition_residualized_metrics.csv",
        "predictions": out / "acquisition_residualized_predictions.csv",
        "feature_tests": out / "acquisition_residualized_feature_tests.csv",
        "deltas": out / "acquisition_residualized_feature_set_deltas.csv",
        "permutation_null": out / "acquisition_residualized_permutation_null.csv",
        "summary": out / "acquisition_residualized_summary.json",
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    tests_df.to_csv(paths["feature_tests"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)

    top_metrics = (
        metrics_df.sort_values(["roc_auc", "average_precision"], ascending=[False, False]).head(50)
        if {"roc_auc", "average_precision"}.issubset(metrics_df.columns) and not metrics_df.empty
        else pd.DataFrame()
    )
    top_deltas = delta_df.head(40) if not delta_df.empty else pd.DataFrame()
    top_tests = (
        tests_df.sort_values(["spearman_abs_rho_vs_context_label_residual", "direction_free_auc_vs_label"], ascending=[False, False]).head(50)
        if {"spearman_abs_rho_vs_context_label_residual", "direction_free_auc_vs_label"}.issubset(tests_df.columns) and not tests_df.empty
        else pd.DataFrame()
    )
    summary = clean_json(
        {
            "n_rows": int(len(joined)),
            "n_cycles": int(joined["cycleNo"].nunique()) if "cycleNo" in joined else 0,
            "targets": targets,
            "feature_group_sizes": {k: len(v) for k, v in feature_groups.items()},
            "residualized_video_group_sizes": {k: len(v) for k, v in video_groups.items()},
            "top_metrics": top_metrics.to_dict("records"),
            "top_feature_set_deltas": top_deltas.to_dict("records"),
            "top_context_residual_feature_tests": top_tests.to_dict("records"),
            "guardrail": (
                "This is a weak-label, leave-one-cycle benchmark over automatically selected ROI embeddings. "
                "A strong acquisition-context score is treated as design/context structure, not a deployable warning model. "
                "Residualized video scores test whether particle-region video descriptors add signal after context conditioning."
            ),
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "README.md").write_text(
        "# Acquisition-Residualized Video Physics Benchmark\n\n"
        "Leave-one-cycle benchmark for future weak-label signal after acquisition/context conditioning.\n\n"
        f"- Rows: {summary['n_rows']}\n"
        f"- Cycles: {summary['n_cycles']}\n"
        f"- Targets: {', '.join(targets)}\n"
        f"- Feature groups: {summary['feature_group_sizes']}\n\n"
        f"Guardrail: {summary['guardrail']}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
