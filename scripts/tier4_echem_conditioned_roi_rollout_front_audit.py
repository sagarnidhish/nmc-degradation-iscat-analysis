#!/usr/bin/env python3
"""Ask whether echem regime features explain ROI rollout/front difficulty.

This audit joins the echem optical regime atlas to the balanced future ROI
physics table, then compares acquisition/context features with echem-regime
features for continuous ROI-level rollout and front-motion targets. Evaluation
uses leave-one-cycle-out splits so multiple ROIs from the same cycle are never
split across train/test.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
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
        "stage_drift_xy_sampled",
        "front_candidate_rank",
        "object_candidate_rank",
    ])
    echem = available_numeric(df, [
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
    return {
        "acquisition_context": acquisition,
        "echem_regime": echem,
        "echem_plus_acquisition": sorted(set(acquisition + echem)),
    }


def target_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "low_rank_dmd_particle_mse_fraction_of_full_mean",
        "low_rank_dmd_particle_to_nonparticle_mse_ratio_mean",
        "persistence_particle_mse_fraction_of_full_mean",
        "velocity_particle_mse_fraction_of_full_mean",
        "transferred_masked_residual_signature",
        "phase_slope_abs_median_per_s",
        "threshold_robust_phase_score",
        "diffusion_proxy_abs_median_um2_per_s",
        "radius2_slope_median_px2_per_s",
        "phase_slope_positive_fraction",
        "roi_norm_mean_delta_last_minus_first",
        "object_mean_residual",
    ]
    return available_numeric(df, candidates, min_nonnull=10)


def ridge_model(alpha: float = 1.0) -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def leave_cycle_predictions(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    use = df[["roi_id", "cycleNo", target] + features].copy()
    use[target] = pd.to_numeric(use[target], errors="coerce")
    use = use[use[target].notna()].copy()
    cycles = sorted(pd.to_numeric(use["cycleNo"], errors="coerce").dropna().unique())
    for cycle in cycles:
        test_mask = pd.to_numeric(use["cycleNo"], errors="coerce") == cycle
        train = use[~test_mask].copy()
        test = use[test_mask].copy()
        row_meta = test[["roi_id", "cycleNo", target]].copy()
        if len(train) < 12 or train[target].nunique(dropna=True) < 2:
            row_meta["predicted"] = np.nan
            row_meta["status"] = "skipped_train_size_or_variance"
        else:
            model = ridge_model()
            model.fit(train[features], train[target].astype(float))
            row_meta["predicted"] = model.predict(test[features])
            row_meta["status"] = "ok"
        row_meta = row_meta.rename(columns={target: "observed"})
        rows.extend(row_meta.to_dict("records"))
    return pd.DataFrame(rows)


def regression_metrics(pred: pd.DataFrame, feature_set: str, target: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted"]).copy()
    row: Dict[str, Any] = {
        "feature_set": feature_set,
        "target": target,
        "n_eval": int(len(tmp)),
        "n_cycles": int(tmp["cycleNo"].nunique()) if "cycleNo" in tmp else 0,
        "r2": np.nan,
        "mae": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and tmp["observed"].nunique(dropna=True) >= 2:
        y = tmp["observed"].astype(float).to_numpy()
        p = tmp["predicted"].astype(float).to_numpy()
        row["r2"] = float(r2_score(y, p))
        row["mae"] = float(mean_absolute_error(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def spearman_table(df: pd.DataFrame, xs: List[str], ys: List[str], family: str) -> pd.DataFrame:
    rows = []
    for x in xs:
        xv = pd.to_numeric(df[x], errors="coerce")
        for y in ys:
            yv = pd.to_numeric(df[y], errors="coerce")
            mask = xv.notna() & yv.notna()
            if mask.sum() < 10 or xv[mask].nunique() < 2 or yv[mask].nunique() < 2:
                continue
            rho, pval = spearmanr(xv[mask], yv[mask])
            rows.append({
                "family": family,
                "feature": x,
                "target": y,
                "n": int(mask.sum()),
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "abs_spearman_rho": abs(float(rho)),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["abs_spearman_rho", "n"], ascending=[False, False])


def residual_correlations(df: pd.DataFrame, acquisition: List[str], echem: List[str], targets: List[str]) -> pd.DataFrame:
    rows = []
    if not acquisition or not echem:
        return pd.DataFrame()
    for target in targets:
        use = df[acquisition + echem + [target]].copy()
        use[target] = pd.to_numeric(use[target], errors="coerce")
        valid = use[target].notna()
        if valid.sum() < 12 or use.loc[valid, target].nunique(dropna=True) < 2:
            continue
        model = ridge_model()
        model.fit(use.loc[valid, acquisition], use.loc[valid, target].astype(float))
        resid = use.loc[valid, target].astype(float) - model.predict(use.loc[valid, acquisition])
        for feature in echem:
            xv = pd.to_numeric(use.loc[valid, feature], errors="coerce")
            mask = xv.notna() & resid.notna()
            if mask.sum() < 10 or xv[mask].nunique() < 2 or resid[mask].nunique() < 2:
                continue
            rho, pval = spearmanr(xv[mask], resid[mask])
            rows.append({
                "target": target,
                "echem_feature": feature,
                "n": int(mask.sum()),
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "abs_spearman_rho": abs(float(rho)),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["abs_spearman_rho", "n"], ascending=[False, False])


def write_readme(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Echem-Conditioned ROI Rollout/Front Audit",
        "",
        "This audit joins cycle-level electrochemical regime descriptors to balanced future ROI physics rows.",
        "It compares acquisition/context features with echem-regime features for continuous ROI rollout and front-motion targets under leave-one-cycle-out regression.",
        "",
        f"- ROI rows: {summary['n_roi_rows']}",
        f"- Cycles: {summary['n_cycles']}",
        f"- Targets: {', '.join(summary['targets'])}",
        f"- Feature set sizes: {summary['feature_set_sizes']}",
        "",
        f"Guardrail: {summary['guardrail']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/echem_conditioned_roi_rollout_front_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    echem = read_csv(derived / "echem_optical_regime_atlas" / "echem_optical_regime_cycle_table.csv")
    roi = read_csv(derived / "balanced_future_roi_physics_audit" / "balanced_future_roi_physics_joined.csv")
    if echem.empty:
        raise FileNotFoundError("Need echem_optical_regime_atlas/echem_optical_regime_cycle_table.csv")
    if roi.empty:
        raise FileNotFoundError("Need balanced_future_roi_physics_audit/balanced_future_roi_physics_joined.csv")

    echem_cols = [c for c in echem.columns if c != "source_stem"]
    joined = roi.merge(echem[echem_cols], on="cycleNo", how="left", suffixes=("", "_echem"))
    features = build_feature_sets(joined)
    targets = target_columns(joined)

    correlations = []
    if features.get("echem_regime"):
        correlations.append(spearman_table(joined, features["echem_regime"], targets, "echem_regime_raw"))
    if features.get("acquisition_context"):
        correlations.append(spearman_table(joined, features["acquisition_context"], targets, "acquisition_context_raw"))
    corr_df = pd.concat([x for x in correlations if not x.empty], ignore_index=True) if correlations else pd.DataFrame()
    residual_df = residual_correlations(joined, features.get("acquisition_context", []), features.get("echem_regime", []), targets)

    metrics = []
    predictions = []
    for target in targets:
        for feature_set, cols in features.items():
            if not cols:
                continue
            pred = leave_cycle_predictions(joined, cols, target)
            pred["target"] = target
            pred["feature_set"] = feature_set
            predictions.append(pred)
            metrics.append(regression_metrics(pred, feature_set, target))
    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    deltas = []
    for target in targets:
        sub = metrics_df[metrics_df["target"] == target]
        base = sub[sub["feature_set"] == "acquisition_context"]
        for feature_set in ["echem_regime", "echem_plus_acquisition"]:
            comp = sub[sub["feature_set"] == feature_set]
            if base.empty or comp.empty:
                continue
            b = base.iloc[0]
            c = comp.iloc[0]
            deltas.append({
                "target": target,
                "comparison": f"{feature_set}_minus_acquisition",
                "delta_r2": float(c["r2"] - b["r2"]) if pd.notna(c["r2"]) and pd.notna(b["r2"]) else np.nan,
                "delta_spearman_rho": float(c["spearman_rho"] - b["spearman_rho"]) if pd.notna(c["spearman_rho"]) and pd.notna(b["spearman_rho"]) else np.nan,
                "delta_mae": float(c["mae"] - b["mae"]) if pd.notna(c["mae"]) and pd.notna(b["mae"]) else np.nan,
                "base_r2": b["r2"],
                "comparison_r2": c["r2"],
                "base_spearman_rho": b["spearman_rho"],
                "comparison_spearman_rho": c["spearman_rho"],
            })
    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_df = delta_df.sort_values(["delta_spearman_rho", "delta_r2"], ascending=[False, False])

    joined.to_csv(out / "echem_conditioned_roi_rollout_front_joined.csv", index=False)
    corr_df.to_csv(out / "echem_conditioned_roi_rollout_front_correlations.csv", index=False)
    residual_df.to_csv(out / "echem_conditioned_roi_rollout_front_residual_correlations.csv", index=False)
    metrics_df.to_csv(out / "echem_conditioned_roi_rollout_front_model_metrics.csv", index=False)
    pred_df.to_csv(out / "echem_conditioned_roi_rollout_front_predictions.csv", index=False)
    delta_df.to_csv(out / "echem_conditioned_roi_rollout_front_feature_set_deltas.csv", index=False)

    top_metrics = metrics_df.sort_values(["spearman_rho", "r2"], ascending=[False, False]).head(30) if not metrics_df.empty else pd.DataFrame()
    top_deltas = delta_df.head(30) if not delta_df.empty else pd.DataFrame()
    top_resid = residual_df.head(30) if not residual_df.empty else pd.DataFrame()
    summary = {
        "n_roi_rows": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "targets": targets,
        "feature_set_sizes": {k: len(v) for k, v in features.items()},
        "top_model_metrics": top_metrics.to_dict("records"),
        "top_feature_set_deltas": top_deltas.to_dict("records"),
        "top_residual_correlations": top_resid.to_dict("records"),
        "guardrail": "ROI rows are automatic, clustered by cycle, and front/diffusion variables are proxy measurements; use this as a weak-label explanatory audit, not calibrated electrochemical mechanism proof.",
    }
    summary = clean_json(summary)
    (out / "echem_conditioned_roi_rollout_front_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(out / "README.md", summary)


if __name__ == "__main__":
    main()
