#!/usr/bin/env python3
"""Couple source-balanced pre-event front/kymograph signals to echem context.

The source-balanced pre-event packet found near-pre optical/front candidates,
while follow-up matching showed that some signals are sensitive to source and
context. This audit joins those ROI descriptors to the cycle-level
electrochemical regime atlas and asks two guarded questions:

1. Which cycle-level echem descriptors correlate with the pre-event front and
   kymograph proxies?
2. Which near-pre event-bin effects remain after residualizing against source,
   acquisition context, and echem descriptors?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ECHEM_CANDIDATES = [
    "capacity_mAh",
    "capacity_fade_from_first_mAh",
    "capacity_fraction_of_first",
    "coulombic_efficiency_pct",
    "coulombic_inefficiency_pct",
    "charge_discharge_capacity_gap_mAh",
    "charge_discharge_capacity_abs_gap_mAh",
    "signed_charge_fraction",
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
    "shape_V_range",
    "shape_V_mean",
    "shape_V_std",
    "shape_I_abs_mean_mA",
    "shape_I_pos_fraction",
    "shape_I_neg_fraction",
    "shape_charge_mAh_abs",
    "shape_dVdt_abs_p95",
    "shape_dVdt_sign_consistency",
    "all_dq_abs_lowV_frac",
    "all_dq_abs_midV_frac",
    "all_dq_abs_highV_frac",
    "all_dq_abs_peak_voltage",
    "all_dq_abs_peak_frac",
    "all_dq_abs_entropy",
    "pos_dq_abs_peak_voltage",
    "pos_dq_abs_peak_frac",
    "neg_dq_abs_peak_voltage",
    "neg_dq_abs_peak_frac",
]

CONTEXT_CANDIDATES = [
    "cycleNo",
    "local_cycle_index",
    "expansion_cycle_rank",
    "object_candidate_rank",
    "object_x_full_approx",
    "object_y_full_approx",
    "object_area_ds_px",
    "object_mean_abs_z",
    "crop_x0",
    "crop_y0",
    "n_frames",
    "first_frame_index",
    "last_frame_index",
    "roi_norm_mean_first",
    "stage_drift_xy_sampled",
    "mask_base_area_fraction",
]

TARGET_CANDIDATES = [
    "front_radius_q60_slope_px_per_norm_time",
    "front_radius_q70_slope_px_per_norm_time",
    "front_radius_q80_slope_px_per_norm_time",
    "masked_minus_background_mean_slope",
    "masked_minus_background_mean_median",
    "mask_area_fraction_slope",
    "mask_centroid_path_px",
    "front_gradient_peak_radius_slope_px_per_norm_time",
    "apparent_diffusion_q70_um2_per_norm_time",
    "spatial_std_slope",
    "frame_diff_mse_slope",
    "front_radius_slope_px_per_norm_time",
    "front_radius2_slope_px2_per_norm_time",
    "front_radius_slope_r2",
    "front_radius2_slope_r2",
    "front_radius_monotonic_fraction",
    "front_gradient_strength_median",
    "front_gradient_coherence",
    "phase_fraction_slope_per_norm_time",
    "phase_fraction_slope_r2",
    "kymograph_temporal_energy",
    "radial_profile_last_minus_first_l1",
]

COMPARISONS: List[Tuple[str, Sequence[str], Sequence[str]]] = [
    ("near_pre_vs_far_pre", ["near_pre_event_1_8"], ["far_pre_event_17_32"]),
    (
        "clean_pre_1_8_vs_post_control",
        ["near_pre_event_1_8"],
        ["post_event_1_16", "no_near_event_control"],
    ),
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


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_count: int = 16) -> List[str]:
    keep = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = numeric(df, col)
        if vals.notna().sum() >= min_count and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def one_hot_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)
    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    values = df[cols].fillna("__missing__").astype(str)
    arr = enc.fit_transform(values)
    names = enc.get_feature_names_out(cols)
    return pd.DataFrame(arr, columns=names, index=df.index)


def design_matrix(df: pd.DataFrame, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        x[col] = numeric(df, col)
    cats = one_hot_frame(df, categorical_cols)
    if not cats.empty:
        x = pd.concat([x, cats], axis=1)
    return x


def residualize(df: pd.DataFrame, target: str, x: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
    y = numeric(df, target)
    mask = y.notna()
    resid = pd.Series(np.nan, index=df.index, dtype=float)
    meta: Dict[str, Any] = {
        "target": target,
        "n": int(mask.sum()),
        "variance_explained_by_model": np.nan,
        "ridge_alpha": np.nan,
    }
    if mask.sum() < 12 or y[mask].nunique(dropna=True) < 2 or x.shape[1] == 0:
        return resid, meta
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 25))),
    ])
    model.fit(x.loc[mask], y.loc[mask].astype(float))
    pred = pd.Series(model.predict(x.loc[mask]), index=x.index[mask])
    resid.loc[mask] = y.loc[mask] - pred
    var_y = float(np.var(y.loc[mask]))
    var_r = float(np.var(resid.loc[mask]))
    meta["variance_explained_by_model"] = float(1.0 - var_r / var_y) if var_y > 0 else np.nan
    meta["ridge_alpha"] = float(model.named_steps["ridge"].alpha_)
    return resid, meta


def effect_tests(df: pd.DataFrame, features: Sequence[str], value_suffix: str) -> pd.DataFrame:
    rows = []
    bins = df["event_relative_bin"].astype(str)
    for comparison, treat_bins, control_bins in COMPARISONS:
        treat_mask = bins.isin(treat_bins)
        control_mask = bins.isin(control_bins)
        for feature in features:
            col = feature if value_suffix == "raw" else f"{feature}_{value_suffix}"
            if col not in df.columns:
                continue
            treat = numeric(df, col).loc[treat_mask].dropna()
            control = numeric(df, col).loc[control_mask].dropna()
            row: Dict[str, Any] = {
                "comparison": comparison,
                "scale": value_suffix,
                "feature": feature,
                "value_column": col,
                "n_treated": int(len(treat)),
                "n_control": int(len(control)),
                "treated_median": np.nan,
                "control_median": np.nan,
                "treated_minus_control_median": np.nan,
                "mannwhitney_p": np.nan,
                "roc_auc_treated_high": np.nan,
                "average_precision_treated_high": np.nan,
            }
            if len(treat) >= 4 and len(control) >= 4 and pd.concat([treat, control]).nunique() > 1:
                row["treated_median"] = float(treat.median())
                row["control_median"] = float(control.median())
                row["treated_minus_control_median"] = float(treat.median() - control.median())
                _, p_val = mannwhitneyu(treat, control, alternative="two-sided")
                row["mannwhitney_p"] = float(p_val)
                y = np.array([1] * len(treat) + [0] * len(control))
                score = np.concatenate([treat.to_numpy(dtype=float), control.to_numpy(dtype=float)])
                row["roc_auc_treated_high"] = float(roc_auc_score(y, score))
                row["average_precision_treated_high"] = float(average_precision_score(y, score))
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_median_difference"] = out["treated_minus_control_median"].abs()
    return out.sort_values(["comparison", "mannwhitney_p", "abs_median_difference"], ascending=[True, True, False])


def spearman_table(df: pd.DataFrame, echem_cols: Sequence[str], targets: Sequence[str]) -> pd.DataFrame:
    rows = []
    for echem_col in echem_cols:
        x = numeric(df, echem_col)
        for target in targets:
            y = numeric(df, target)
            mask = x.notna() & y.notna()
            if mask.sum() < 12 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                continue
            rho, p_val = spearmanr(x[mask], y[mask])
            rows.append({
                "echem_feature": echem_col,
                "target": target,
                "n": int(mask.sum()),
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "abs_spearman_rho": abs(float(rho)),
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["abs_spearman_rho", "n"], ascending=[False, False]) if not out.empty else out


def merge_inputs(derived: Path) -> pd.DataFrame:
    readout = read_csv(
        derived / "source_balanced_pre_event_readout_audit" / "source_balanced_pre_event_readout_features.csv"
    )
    kymo = read_csv(
        derived / "source_balanced_pre_event_radial_kymograph_audit" / "source_balanced_pre_event_radial_kymograph_features.csv"
    )
    echem = read_csv(derived / "echem_optical_regime_atlas" / "echem_optical_regime_cycle_table.csv")

    kymo_extra = [
        "roi_id",
        "front_radius_slope_px_per_norm_time",
        "front_radius_slope_r2",
        "front_radius2_slope_px2_per_norm_time",
        "front_radius2_slope_r2",
        "front_radius_monotonic_fraction",
        "front_gradient_strength_median",
        "front_gradient_coherence",
        "phase_fraction_slope_per_norm_time",
        "phase_fraction_slope_r2",
        "kymograph_temporal_energy",
        "radial_profile_last_minus_first_l1",
    ]
    kymo_extra = [c for c in kymo_extra if c in kymo.columns]
    joined = readout.merge(kymo[kymo_extra], on="roi_id", how="left")
    echem_cols = ["cycleNo"] + [c for c in ECHEM_CANDIDATES if c in echem.columns]
    return joined.merge(echem[echem_cols], on="cycleNo", how="left", suffixes=("", "_echem"))


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    top_raw = summary.get("top_raw_effects", [])[:6]
    top_resid = summary.get("top_echem_source_residual_effects", [])[:6]
    top_corr = summary.get("top_echem_correlations", [])[:6]
    lines = [
        "# Source-Balanced Pre-Event Echem/Front Coupling Audit",
        "",
        f"- ROI rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"- Echem features used: {summary['n_echem_features']}",
        f"- Optical/front targets used: {summary['n_targets']}",
        "",
        "## Top Raw Event-Bin Effects",
    ]
    for row in top_raw:
        lines.append(
            f"- {row['comparison']} {row['feature']}: n={row['n_treated']} vs {row['n_control']}, "
            f"median diff={row['treated_minus_control_median']:.4g}, p={row['mannwhitney_p']:.4g}, "
            f"AUC={row['roc_auc_treated_high']:.3f}"
        )
    lines += ["", "## Top Source+Echem Residual Event-Bin Effects"]
    for row in top_resid:
        lines.append(
            f"- {row['comparison']} {row['feature']}: n={row['n_treated']} vs {row['n_control']}, "
            f"residual median diff={row['treated_minus_control_median']:.4g}, p={row['mannwhitney_p']:.4g}, "
            f"AUC={row['roc_auc_treated_high']:.3f}"
        )
    lines += ["", "## Top Echem/Optical Correlations"]
    for row in top_corr:
        lines.append(
            f"- {row['echem_feature']} vs {row['target']}: n={row['n']}, "
            f"rho={row['spearman_rho']:.3f}, p={row['p_value']:.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument(
        "--out-dir",
        default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_echem_front_coupling_audit",
    )
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = merge_inputs(derived).reset_index(drop=True)
    echem_cols = available_numeric(joined, ECHEM_CANDIDATES, min_count=24)
    context_cols = available_numeric(joined, CONTEXT_CANDIDATES, min_count=24)
    targets = available_numeric(joined, TARGET_CANDIDATES, min_count=24)

    source_x = design_matrix(joined, context_cols, ["source_stem"])
    echem_x = design_matrix(joined, sorted(set(context_cols + echem_cols)), ["source_stem"])

    residual_meta = []
    for target in targets:
        joined[f"{target}_source_context_residual"], meta = residualize(joined, target, source_x)
        meta["residual_model"] = "source_plus_acquisition_context"
        residual_meta.append(meta)
        joined[f"{target}_source_echem_context_residual"], meta = residualize(joined, target, echem_x)
        meta["residual_model"] = "source_plus_acquisition_context_plus_echem"
        residual_meta.append(meta)

    raw_tests = effect_tests(joined, targets, "raw")
    source_resid_tests = effect_tests(joined, targets, "source_context_residual")
    echem_resid_tests = effect_tests(joined, targets, "source_echem_context_residual")
    all_tests = pd.concat([raw_tests, source_resid_tests, echem_resid_tests], ignore_index=True)
    corr = spearman_table(joined, echem_cols, targets)
    fit = pd.DataFrame(residual_meta).sort_values(["target", "residual_model"])

    joined_path = out / "source_balanced_pre_event_echem_front_joined.csv"
    corr_path = out / "source_balanced_pre_event_echem_front_correlations.csv"
    tests_path = out / "source_balanced_pre_event_echem_front_eventbin_tests.csv"
    fit_path = out / "source_balanced_pre_event_echem_front_residual_fits.csv"
    summary_path = out / "source_balanced_pre_event_echem_front_summary.json"
    joined.to_csv(joined_path, index=False)
    corr.to_csv(corr_path, index=False)
    all_tests.to_csv(tests_path, index=False)
    fit.to_csv(fit_path, index=False)

    top_raw = raw_tests.dropna(subset=["mannwhitney_p"]).head(20).to_dict("records") if not raw_tests.empty else []
    top_source = (
        source_resid_tests.dropna(subset=["mannwhitney_p"]).head(20).to_dict("records")
        if not source_resid_tests.empty
        else []
    )
    top_echem = (
        echem_resid_tests.dropna(subset=["mannwhitney_p"]).head(20).to_dict("records")
        if not echem_resid_tests.empty
        else []
    )
    top_corr = corr.head(20).to_dict("records") if not corr.empty else []
    top_fit = fit.sort_values("variance_explained_by_model", ascending=False).head(20).to_dict("records")

    summary: Dict[str, Any] = {
        "n_rows": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "n_sources": int(joined["source_stem"].nunique()),
        "event_relative_bin_counts": clean_json(joined["event_relative_bin"].astype(str).value_counts().to_dict()),
        "echem_features": echem_cols,
        "context_features": context_cols,
        "targets": targets,
        "n_echem_features": int(len(echem_cols)),
        "n_context_features": int(len(context_cols)),
        "n_targets": int(len(targets)),
        "top_raw_effects": clean_json(top_raw),
        "top_source_context_residual_effects": clean_json(top_source),
        "top_echem_source_residual_effects": clean_json(top_echem),
        "top_echem_correlations": clean_json(top_corr),
        "top_residual_fits": clean_json(top_fit),
        "outputs": {
            "joined": str(joined_path),
            "correlations": str(corr_path),
            "eventbin_tests": str(tests_path),
            "residual_fits": str(fit_path),
            "summary": str(summary_path),
        },
        "guardrail": (
            "Cycle-level echem descriptors are joined to ROI-level automatic front/kymograph proxies, "
            "so rows are clustered by cycle and source. Residualization is an explanatory stress test, "
            "not causal evidence, calibrated diffusion, validated phase-boundary tracking, or a deployable warning model."
        ),
    }
    summary = clean_json(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
