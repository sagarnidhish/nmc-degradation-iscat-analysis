#!/usr/bin/env python3
"""Test whether echem regime features add optical-degradation prediction signal.

The echem optical regime atlas is descriptive. This audit turns it into a
controlled model comparison: acquisition/frame-count context versus echem
regime descriptors versus their combination, evaluated on cycle-level optical
targets with blocked cycle-CV and rolling-origin block splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


def add_quantile_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for source, target in [
        ("cross_modal_consensus_score", "high_cross_modal_consensus_q75"),
        ("particle_norm_cv", "high_particle_norm_cv_q75"),
        ("shape_dVdt_abs_p95", "high_echem_dvdt_q75"),
        ("roi_phase_slope_abs_median_per_s", "high_roi_phase_slope_abs_q75"),
        ("state_state_step_norm", "high_state_step_norm_q75"),
    ]:
        if source in out.columns:
            vals = pd.to_numeric(out[source], errors="coerce")
            if vals.notna().sum() >= 12:
                q = vals.quantile(0.75)
                out[target] = (vals >= q).astype(float)
                out.loc[vals.isna(), target] = np.nan
    return out


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 12) -> List[str]:
    keep = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    acquisition = available_numeric(df, [
        "cycle_index_rank",
        "n_frames",
        "frames_percentile",
        "n_points",
        "echem_shape_points",
        "echem_shape_missing",
        "echem_ce_extreme_or_missing",
    ])
    echem_base = available_numeric(df, [
        "capacity_mAh",
        "capacity_fade_from_first_mAh",
        "capacity_fraction_of_first",
        "coulombic_efficiency_pct",
        "coulombic_inefficiency_pct",
        "shape_V_range",
        "shape_V_mean",
        "shape_V_std",
        "shape_I_abs_mean_mA",
        "shape_I_pos_fraction",
        "shape_I_neg_fraction",
        "shape_charge_mAh_abs",
        "shape_charge_mAh_pos",
        "shape_charge_mAh_neg_abs",
        "charge_discharge_capacity_gap_mAh",
        "charge_discharge_capacity_abs_gap_mAh",
        "signed_charge_fraction",
        "shape_dVdt_abs_p95",
        "shape_dVdt_sign_consistency",
        "all_dq_abs_lowV_frac",
        "all_dq_abs_midV_frac",
        "all_dq_abs_highV_frac",
        "all_dq_abs_peak_voltage",
        "all_dq_abs_peak_frac",
        "all_dq_abs_entropy",
        "pos_dq_abs_lowV_frac",
        "pos_dq_abs_midV_frac",
        "pos_dq_abs_highV_frac",
        "pos_dq_abs_peak_voltage",
        "pos_dq_abs_peak_frac",
        "pos_dq_abs_entropy",
        "neg_dq_abs_lowV_frac",
        "neg_dq_abs_midV_frac",
        "neg_dq_abs_highV_frac",
        "neg_dq_abs_peak_voltage",
        "neg_dq_abs_peak_frac",
        "neg_dq_abs_entropy",
        "voltage_peak_hysteresis_proxy",
        "highV_charge_discharge_imbalance",
        "midV_charge_discharge_imbalance",
        "lowV_charge_discharge_imbalance",
        "dqdv_peak_concentration",
        "dqdv_entropy_asymmetry",
        "dqdv_integral_asymmetry",
        "echem_outlier_score",
        "echem_regime_pc1",
        "echem_regime_pc2",
        "echem_regime_pc3",
        "echem_regime_pc4",
    ])
    state = available_numeric(df, [
        "cycle_state_pc1",
        "cycle_state_pc2",
        "cycle_state_pc3",
        "cycle_state_pc4",
        "degradation_state_axis",
        "state_step_norm",
        "axis_step",
    ])
    sets = {
        "acquisition_context": acquisition,
        "echem_regime": echem_base,
        "echem_plus_acquisition": sorted(set(echem_base + acquisition)),
        "cycle_state_upper_bound": sorted(set(state + acquisition)),
    }
    return {k: v for k, v in sets.items() if v}


def model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.25, random_state=seed, solver="liblinear"),
    )


def metric_row(pred: pd.DataFrame, split: str, feature_set: str, target: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = tmp["observed"].astype(int).to_numpy()
    p = tmp["predicted_probability"].to_numpy(dtype=float)
    row: Dict[str, Any] = {
        "split": split,
        "feature_set": feature_set,
        "target": target,
        "n_eval": int(len(tmp)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "positive_rate": float(np.mean(y)) if len(y) else np.nan,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "balanced_accuracy_at_0p5": np.nan,
        "brier": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 6 and len(np.unique(y)) == 2:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        row["balanced_accuracy_at_0p5"] = float(balanced_accuracy_score(y, p >= 0.5))
        row["brier"] = float(brier_score_loss(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def cycle_block_predictions(df: pd.DataFrame, features: List[str], target: str, seed: int, n_blocks: int = 5) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    use = df[["cycleNo", target] + features].copy().sort_values("cycleNo").reset_index(drop=True)
    use[target] = pd.to_numeric(use[target], errors="coerce")
    valid_idx = list(use.index[use[target].isin([0, 1])])
    if len(valid_idx) < n_blocks:
        n_blocks = max(2, len(valid_idx))
    blocks = np.array_split(np.asarray(valid_idx), n_blocks)
    for block_id, test_idx_arr in enumerate(blocks, start=1):
        test_idx = [int(i) for i in test_idx_arr]
        test_set = set(test_idx)
        train_idx = [i for i in valid_idx if i not in test_set]
        y_train = use.loc[train_idx, target].astype(int).to_numpy()
        status = "ok"
        clf = None
        if len(train_idx) < 8 or len(np.unique(y_train)) < 2:
            status = "skipped_train_class_or_size"
        else:
            clf = model(seed + block_id)
            clf.fit(use.loc[train_idx, features], y_train)
        for idx in test_idx:
            row = {
                "cycleNo": float(use.loc[idx, "cycleNo"]),
                "observed": int(use.loc[idx, target]),
                "block_id": int(block_id),
                "n_train": int(len(train_idx)),
                "n_positive_train": int(y_train.sum()) if len(y_train) else 0,
                "status": status,
            }
            row["predicted_probability"] = np.nan if clf is None else float(clf.predict_proba(use.loc[[idx], features])[0, 1])
            rows.append(row)
    return pd.DataFrame(rows)


def rolling_origin_block_predictions(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    seed: int,
    purge_cycles: int,
    min_train: int,
    n_test_blocks: int = 5,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    use = df[["cycleNo", target] + features].copy().sort_values("cycleNo").reset_index(drop=True)
    use[target] = pd.to_numeric(use[target], errors="coerce")
    valid_idx = list(use.index[use[target].isin([0, 1])])
    eligible = [i for i in valid_idx if i >= min_train]
    if not eligible:
        return pd.DataFrame()
    blocks = np.array_split(np.asarray(eligible), min(n_test_blocks, len(eligible)))
    cycles = pd.to_numeric(use["cycleNo"], errors="coerce")
    for block_id, test_idx_arr in enumerate(blocks, start=1):
        test_idx = [int(i) for i in test_idx_arr]
        min_test_cycle = float(cycles.iloc[test_idx].min())
        train_mask = (cycles <= min_test_cycle - purge_cycles) & use[target].isin([0, 1])
        train_idx = list(use.index[train_mask])
        y_train = use.loc[train_idx, target].astype(int).to_numpy()
        status = "ok"
        clf = None
        if len(train_idx) < min_train or len(np.unique(y_train)) < 2:
            status = "skipped_train_class_or_size"
        else:
            clf = model(seed + 100 + block_id)
            clf.fit(use.loc[train_idx, features], y_train)
        for idx in test_idx:
            row = {
                "cycleNo": float(use.loc[idx, "cycleNo"]),
                "observed": int(use.loc[idx, target]),
                "block_id": int(block_id),
                "n_train": int(len(train_idx)),
                "n_positive_train": int(y_train.sum()) if len(y_train) else 0,
                "status": status,
            }
            row["predicted_probability"] = np.nan if clf is None else float(clf.predict_proba(use.loc[[idx], features])[0, 1])
            rows.append(row)
    return pd.DataFrame(rows)

def score_permutation_null(pred: pd.DataFrame, observed_auc: float, seed: int, n_perm: int) -> Dict[str, Any]:
    if not np.isfinite(observed_auc):
        return {"empirical_p_ge_observed": np.nan, "n_permutation": int(n_perm)}
    rng = np.random.default_rng(seed)
    tmp = pred.dropna(subset=["observed", "predicted_probability"]).copy()
    y = pd.to_numeric(tmp["observed"], errors="coerce")
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    valid = y.isin([0, 1]) & p.notna()
    y = y[valid].to_numpy(dtype=int)
    p = p[valid].to_numpy(dtype=float)
    if len(y) < 6 or len(np.unique(y)) < 2:
        return {"empirical_p_ge_observed": np.nan, "n_permutation": int(n_perm)}
    aucs = []
    for _ in range(n_perm):
        shuffled = y.copy()
        rng.shuffle(shuffled)
        if len(np.unique(shuffled)) == 2:
            aucs.append(float(roc_auc_score(shuffled, p)))
    if not aucs:
        return {"empirical_p_ge_observed": np.nan, "n_permutation": int(n_perm)}
    auc_arr = np.asarray(aucs)
    return {
        "empirical_p_ge_observed": float((np.sum(auc_arr >= observed_auc) + 1) / (len(auc_arr) + 1)),
        "n_permutation": int(len(auc_arr)),
        "null_auc_mean": float(np.mean(auc_arr)),
        "null_auc_p95": float(np.quantile(auc_arr, 0.95)),
    }


def coefficient_table(df: pd.DataFrame, features: List[str], target: str, feature_set: str, seed: int) -> pd.DataFrame:
    use = df[features + [target]].copy()
    use[target] = pd.to_numeric(use[target], errors="coerce")
    use = use[use[target].isin([0, 1])].copy()
    if len(use) < 10 or use[target].nunique() < 2:
        return pd.DataFrame()
    clf = model(seed)
    clf.fit(use[features], use[target].astype(int))
    coefs = clf.named_steps["logisticregression"].coef_[0]
    return pd.DataFrame({
        "target": target,
        "feature_set": feature_set,
        "feature": features,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_conditioned_optical_predictor")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-permutation", type=int, default=100)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = read_csv(derived / "echem_optical_regime_atlas" / "echem_optical_regime_cycle_table.csv")
    if table.empty:
        raise FileNotFoundError("Need echem_optical_regime_atlas/echem_optical_regime_cycle_table.csv")
    table = add_quantile_targets(table)
    feature_sets = build_feature_sets(table)
    targets = [
        "future_any_drop_within_8cycles",
        "high_cross_modal_consensus_q75",
        "high_particle_norm_cv_q75",
        "high_roi_phase_slope_abs_q75",
        "high_state_step_norm_q75",
    ]
    targets = [t for t in targets if t in table.columns and pd.to_numeric(table[t], errors="coerce").isin([0, 1]).sum() >= 10]

    rolling_targets = {"future_any_drop_within_8cycles", "high_cross_modal_consensus_q75", "high_particle_norm_cv_q75"}
    predictions = []
    metrics = []
    nulls = []
    coeffs = []
    for target in targets:
        y = pd.to_numeric(table[target], errors="coerce")
        if y.isin([0, 1]).sum() < 10 or y[y.isin([0, 1])].nunique() < 2:
            continue
        for feature_set, features in feature_sets.items():
            for split in ["cycle_block_cv", "rolling_origin_block"]:
                if split == "rolling_origin_block" and (target not in rolling_targets or feature_set == "cycle_state_upper_bound"):
                    continue
                if split == "cycle_block_cv":
                    pred = cycle_block_predictions(table, features, target, args.seed, n_blocks=5)
                else:
                    pred = rolling_origin_block_predictions(table, features, target, args.seed, purge_cycles=2, min_train=16, n_test_blocks=5)
                pred["target"] = target
                pred["feature_set"] = feature_set
                pred["split"] = split
                predictions.append(pred)
                met = metric_row(pred, split, feature_set, target)
                metrics.append(met)
                if split == "cycle_block_cv" and feature_set in {"acquisition_context", "echem_regime", "echem_plus_acquisition"}:
                    null = score_permutation_null(pred, met.get("roc_auc", np.nan), args.seed, args.n_permutation)
                    null.update({"split": split, "feature_set": feature_set, "target": target, "observed_roc_auc": met.get("roc_auc", np.nan), "null_mode": "heldout_score_label_shuffle"})
                    nulls.append(null)
            coef = coefficient_table(table, features, target, feature_set, args.seed)
            if not coef.empty:
                coeffs.append(coef.head(20))

    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics)
    null_df = pd.DataFrame(nulls)
    coeff_df = pd.concat(coeffs, ignore_index=True) if coeffs else pd.DataFrame()
    if not metrics_df.empty and not null_df.empty:
        for col in ["empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]:
            if col not in null_df.columns:
                null_df[col] = np.nan
        metrics_df = metrics_df.merge(
            null_df[["target", "feature_set", "split", "empirical_p_ge_observed", "null_auc_mean", "null_auc_p95", "n_permutation"]],
            on=["target", "feature_set", "split"],
            how="left",
        )
    deltas = []
    for target in targets:
        for split in ["cycle_block_cv", "rolling_origin_block"]:
            sub = metrics_df[(metrics_df["target"] == target) & (metrics_df["split"] == split)]
            if sub.empty:
                continue
            base = sub[sub["feature_set"] == "acquisition_context"]
            combo = sub[sub["feature_set"] == "echem_plus_acquisition"]
            echem = sub[sub["feature_set"] == "echem_regime"]
            if not base.empty and not combo.empty:
                deltas.append({
                    "target": target,
                    "split": split,
                    "comparison": "echem_plus_acquisition_minus_acquisition",
                    "delta_roc_auc": float(combo.iloc[0]["roc_auc"] - base.iloc[0]["roc_auc"]) if pd.notna(combo.iloc[0]["roc_auc"]) and pd.notna(base.iloc[0]["roc_auc"]) else np.nan,
                    "delta_average_precision": float(combo.iloc[0]["average_precision"] - base.iloc[0]["average_precision"]) if pd.notna(combo.iloc[0]["average_precision"]) and pd.notna(base.iloc[0]["average_precision"]) else np.nan,
                    "base_roc_auc": base.iloc[0]["roc_auc"],
                    "comparison_roc_auc": combo.iloc[0]["roc_auc"],
                })
            if not base.empty and not echem.empty:
                deltas.append({
                    "target": target,
                    "split": split,
                    "comparison": "echem_regime_minus_acquisition",
                    "delta_roc_auc": float(echem.iloc[0]["roc_auc"] - base.iloc[0]["roc_auc"]) if pd.notna(echem.iloc[0]["roc_auc"]) and pd.notna(base.iloc[0]["roc_auc"]) else np.nan,
                    "delta_average_precision": float(echem.iloc[0]["average_precision"] - base.iloc[0]["average_precision"]) if pd.notna(echem.iloc[0]["average_precision"]) and pd.notna(base.iloc[0]["average_precision"]) else np.nan,
                    "base_roc_auc": base.iloc[0]["roc_auc"],
                    "comparison_roc_auc": echem.iloc[0]["roc_auc"],
                })
    delta_df = pd.DataFrame(deltas).sort_values(["split", "delta_roc_auc"], ascending=[True, False]) if deltas else pd.DataFrame()

    paths = {
        "metrics": out / "echem_conditioned_optical_predictor_metrics.csv",
        "predictions": out / "echem_conditioned_optical_predictor_predictions.csv",
        "permutation_null": out / "echem_conditioned_optical_predictor_permutation_null.csv",
        "deltas": out / "echem_conditioned_optical_predictor_feature_set_deltas.csv",
        "coefficients": out / "echem_conditioned_optical_predictor_coefficients.csv",
        "summary": out / "echem_conditioned_optical_predictor_summary.json",
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    null_df.to_csv(paths["permutation_null"], index=False)
    delta_df.to_csv(paths["deltas"], index=False)
    coeff_df.to_csv(paths["coefficients"], index=False)

    best = metrics_df.sort_values(["split", "roc_auc", "average_precision"], ascending=[True, False, False]).head(30)
    best_deltas = delta_df.sort_values("delta_roc_auc", ascending=False).head(20) if not delta_df.empty else pd.DataFrame()
    summary = {
        "n_cycles": int(len(table)),
        "targets": targets,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "top_metrics": best.to_dict("records"),
        "top_feature_set_deltas": best_deltas.to_dict("records") if not best_deltas.empty else [],
        "top_coefficients": coeff_df.head(50).to_dict("records") if not coeff_df.empty else [],
        "guardrail": "This is a cycle-level weak-label model comparison with blocked cycle-CV and rolling-origin block splits. Echem-regime gains show conditional association, not deployable prediction, causal mechanism, calibrated dQ/dV, or validated front/diffusion physics. Rare 2-4 positive targets are excluded from the model-comparison table, and the permutation null shuffles labels against held-out prediction scores rather than retraining every permutation.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Echem-Conditioned Optical Predictor",
        "",
        "Controlled cycle-level comparison of acquisition context versus echem regime descriptors for weak optical degradation targets.",
        "",
        f"- Cycles: {summary['n_cycles']}",
        f"- Targets: {', '.join(targets)}",
        f"- Feature sets: {summary['feature_set_sizes']}",
        "",
        "## Top Metrics",
    ]
    for row in summary["top_metrics"][:12]:
        lines.append(
            f"- {row.get('split')} {row.get('target')} {row.get('feature_set')}: "
            f"AUC={row.get('roc_auc'):.3g}, AP={row.get('average_precision'):.3g}, "
            f"n={row.get('n_eval')}, positives={row.get('n_positive')}"
        )
    lines += ["", "## Top Echem Feature-Set Gains"]
    for row in summary["top_feature_set_deltas"][:10]:
        lines.append(
            f"- {row.get('split')} {row.get('target')} {row.get('comparison')}: "
            f"delta AUC={row.get('delta_roc_auc'):.3g}, base={row.get('base_roc_auc'):.3g}, compare={row.get('comparison_roc_auc'):.3g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json({
        "out_dir": str(out),
        "top_metrics": summary["top_metrics"][:5],
        "top_feature_set_deltas": summary["top_feature_set_deltas"][:5],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
