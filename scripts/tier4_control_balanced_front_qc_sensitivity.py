#!/usr/bin/env python3
"""Compare front/phase sensitivity under original and control-balanced QC panels."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


RAW_FEATURES = [
    "phase_slope_median_per_s",
    "phase_slope_positive_fraction",
    "phase_slope_abs_median_per_s",
    "threshold_robust_phase_score",
    "radius2_slope_median_px2_per_s",
    "diffusion_proxy_median_um2_per_s",
    "diffusion_proxy_abs_median_um2_per_s",
    "threshold_robust_diffusion_score",
]

RESIDUAL_FEATURES = [
    "phase_slope_median_per_s_protocol_residual",
    "phase_slope_positive_fraction_protocol_residual",
    "phase_slope_abs_median_per_s_protocol_residual",
    "threshold_robust_phase_score_protocol_residual",
    "radius2_slope_median_px2_per_s_protocol_residual",
    "diffusion_proxy_median_um2_per_s_protocol_residual",
    "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    "threshold_robust_diffusion_score_protocol_residual",
]

FOCUS_FEATURES = [
    "phase_slope_positive_fraction_protocol_residual",
    "threshold_robust_phase_score_protocol_residual",
    "phase_slope_median_per_s",
    "threshold_robust_phase_score",
    "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    "diffusion_proxy_abs_median_um2_per_s",
]


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_json(value):
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


def bootstrap_ci(event: np.ndarray, control: np.ndarray, rng: np.random.Generator, n_boot: int) -> Dict[str, float]:
    if len(event) < 2 or len(control) < 2:
        return {"bootstrap_p05": np.nan, "bootstrap_p50": np.nan, "bootstrap_p95": np.nan}
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        e = rng.choice(event, size=len(event), replace=True)
        c = rng.choice(control, size=len(control), replace=True)
        diffs[i] = float(np.nanmedian(e) - np.nanmedian(c))
    p05, p50, p95 = np.nanpercentile(diffs, [5, 50, 95])
    return {"bootstrap_p05": float(p05), "bootstrap_p50": float(p50), "bootstrap_p95": float(p95)}


def permutation_p(event: np.ndarray, control: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    if len(event) < 2 or len(control) < 2:
        return np.nan
    observed = float(np.nanmedian(event) - np.nanmedian(control))
    pooled = np.concatenate([event, control])
    n_event = len(event)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(np.nanmedian(perm[:n_event]) - np.nanmedian(perm[n_event:]))
        if abs(diff) >= abs(observed):
            count += 1
    return float((count + 1) / (n_perm + 1))


def test_features(df: pd.DataFrame, features: Iterable[str], stratum: str, rng: np.random.Generator, n_boot: int, n_perm: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature in features:
        if feature not in df.columns:
            continue
        event = to_num(df.loc[df["cohort_role"] == "event", feature]).dropna().to_numpy(dtype=float)
        control = to_num(df.loc[df["cohort_role"] == "control", feature]).dropna().to_numpy(dtype=float)
        if len(event) >= 2 and len(control) >= 2:
            stat, p_value = mannwhitneyu(event, control, alternative="two-sided")
            ci = bootstrap_ci(event, control, rng, n_boot)
            perm_p = permutation_p(event, control, rng, n_perm)
            event_median = float(np.nanmedian(event))
            control_median = float(np.nanmedian(control))
            event_mean = float(np.nanmean(event))
            control_mean = float(np.nanmean(control))
        else:
            stat = p_value = perm_p = np.nan
            ci = {"bootstrap_p05": np.nan, "bootstrap_p50": np.nan, "bootstrap_p95": np.nan}
            event_median = control_median = event_mean = control_mean = np.nan
        rows.append({
            "stratum": stratum,
            "feature": feature,
            "n_event": int(len(event)),
            "n_control": int(len(control)),
            "event_median": event_median,
            "control_median": control_median,
            "median_event_minus_control": float(event_median - control_median) if np.isfinite(event_median) and np.isfinite(control_median) else np.nan,
            "event_mean": event_mean,
            "control_mean": control_mean,
            "mean_event_minus_control": float(event_mean - control_mean) if np.isfinite(event_mean) and np.isfinite(control_mean) else np.nan,
            "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
            "mannwhitney_p": float(p_value) if np.isfinite(p_value) else np.nan,
            "permutation_median_p": perm_p,
            **ci,
        })
    return pd.DataFrame(rows)


def add_manifest_flags(df: pd.DataFrame, manifest: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    keep = [
        "roi_id",
        "auto_review_flags",
        "manual_qc_status",
        "n_components",
        "largest_component_fraction",
        "edge_touch_fraction",
        "q70_first_fraction",
        "q70_last_fraction",
        "q70_fraction_delta",
    ]
    if "in_original_qc_package" in manifest.columns:
        keep.append("in_original_qc_package")
    if "selection_group" in manifest.columns:
        keep.append("selection_group")
    keep = [c for c in keep if c in manifest.columns]
    if not keep:
        out[f"{prefix}_selected"] = False
        return out
    m = manifest[keep].drop_duplicates("roi_id").copy()
    m = m.rename(columns={c: f"{prefix}_{c}" for c in m.columns if c != "roi_id"})
    out = out.merge(m, on="roi_id", how="left")
    flags = out.get(f"{prefix}_auto_review_flags", pd.Series("", index=out.index)).fillna("").astype(str)
    out[f"{prefix}_selected"] = flags.ne("")
    out[f"{prefix}_fragmented_q70_mask"] = flags.str.contains("fragmented_q70_mask", regex=False)
    out[f"{prefix}_diffusion_ci_crosses_zero"] = flags.str.contains("diffusion_ci_crosses_zero", regex=False)
    out[f"{prefix}_active_control"] = flags.str.contains("active_control", regex=False)
    out[f"{prefix}_already_in_primary_qc"] = flags.str.contains("already_in_primary_qc", regex=False)
    out[f"{prefix}_no_auto_flags"] = flags.eq("none")
    return out


def add_front_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"q70_phase_slope_bootstrap_p05", "q70_phase_slope_bootstrap_p95"}.issubset(out.columns):
        p05 = to_num(out["q70_phase_slope_bootstrap_p05"])
        p95 = to_num(out["q70_phase_slope_bootstrap_p95"])
        out["q70_phase_ci_excludes_zero"] = (p05 > 0) | (p95 < 0)
        out["q70_phase_ci_positive"] = p05 > 0
    else:
        out["q70_phase_ci_excludes_zero"] = False
        out["q70_phase_ci_positive"] = False
    out["thresholds_finite"] = to_num(out["thresholds_finite"]) if "thresholds_finite" in out.columns else np.nan
    out["front_quality_score"] = to_num(out["front_quality_score"]) if "front_quality_score" in out.columns else np.nan
    return out


def load_joined(derived: str) -> pd.DataFrame:
    front = pd.read_csv(os.path.join(derived, "multi_cycle_threshold_robust_fronts", "threshold_robust_front_summary.csv"))
    residuals = pd.read_csv(os.path.join(derived, "protocol_conditioned_front_effects", "protocol_conditioned_front_residuals.csv"))
    residual_keep = ["roi_id"] + [c for c in RESIDUAL_FEATURES if c in residuals.columns]
    df = front.merge(residuals[residual_keep], on="roi_id", how="left")
    df = add_front_flags(df)

    original_path = os.path.join(derived, "roi_front_qc_package", "roi_front_qc_manifest.csv")
    balanced_path = os.path.join(derived, "control_balanced_front_qc_package", "control_balanced_front_qc_manifest.csv")
    original = pd.read_csv(original_path) if os.path.exists(original_path) else pd.DataFrame({"roi_id": []})
    balanced = pd.read_csv(balanced_path) if os.path.exists(balanced_path) else pd.DataFrame({"roi_id": []})
    df = add_manifest_flags(df, original, "original_qc")
    df = add_manifest_flags(df, balanced, "balanced_qc")
    return df


def stratum_summary_row(df: pd.DataFrame, name: str) -> Dict[str, object]:
    flags_cols = [c for c in df.columns if c.endswith("_fragmented_q70_mask") or c.endswith("_no_auto_flags")]
    row = {
        "stratum": name,
        "n_roi": int(len(df)),
        "n_event": int((df["cohort_role"] == "event").sum()) if len(df) else 0,
        "n_control": int((df["cohort_role"] == "control").sum()) if len(df) else 0,
        "event_fraction": float((df["cohort_role"] == "event").mean()) if len(df) else np.nan,
        "front_quality_median": float(np.nanmedian(to_num(df["front_quality_score"]))) if len(df) else np.nan,
        "q70_phase_ci_positive_fraction": float(df["q70_phase_ci_positive"].mean()) if len(df) else np.nan,
    }
    for col in flags_cols:
        row[f"{col}_fraction"] = float(df[col].mean()) if len(df) else np.nan
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/control_balanced_front_qc_sensitivity")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutation", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    df = load_joined(args.derived_dir)
    max_thresholds = int(np.nanmax(to_num(df["thresholds_finite"]))) if "thresholds_finite" in df.columns else 0
    fq = to_num(df["front_quality_score"])
    fq_median = float(np.nanmedian(fq))

    strata = {
        "all_front_rois": pd.Series(True, index=df.index),
        "complete_threshold_sweep": to_num(df["thresholds_finite"]).ge(max_thresholds),
        "front_quality_top_half": fq.ge(fq_median),
        "q70_phase_ci_excludes_zero": df["q70_phase_ci_excludes_zero"],
        "q70_phase_ci_positive": df["q70_phase_ci_positive"],
        "original_qc_selected": df["original_qc_selected"],
        "original_qc_not_fragmented": df["original_qc_selected"] & ~df["original_qc_fragmented_q70_mask"],
        "original_qc_no_auto_flags": df["original_qc_no_auto_flags"],
        "balanced_qc_selected": df["balanced_qc_selected"],
        "balanced_qc_new_only": df["balanced_qc_selected"] & ~df["balanced_qc_already_in_primary_qc"],
        "balanced_qc_not_fragmented": df["balanced_qc_selected"] & ~df["balanced_qc_fragmented_q70_mask"],
        "balanced_qc_no_auto_flags": df["balanced_qc_no_auto_flags"],
        "balanced_qc_not_fragmented_no_active_control": df["balanced_qc_selected"] & ~df["balanced_qc_fragmented_q70_mask"] & ~df["balanced_qc_active_control"],
    }

    rng = np.random.default_rng(args.seed)
    all_features = RAW_FEATURES + RESIDUAL_FEATURES
    stratum_rows = []
    test_tables = []
    for name, mask in strata.items():
        sdf = df.loc[mask.fillna(False)].copy()
        stratum_rows.append(stratum_summary_row(sdf, name))
        if len(sdf):
            test_tables.append(test_features(sdf, all_features, name, rng, args.n_bootstrap, args.n_permutation))

    stratum_summary = pd.DataFrame(stratum_rows)
    tests = pd.concat(test_tables, ignore_index=True) if test_tables else pd.DataFrame()
    if not tests.empty:
        tests = tests.sort_values(["stratum", "mannwhitney_p", "feature"], na_position="last")
    focus = tests[tests["feature"].isin(FOCUS_FEATURES)].copy() if not tests.empty else pd.DataFrame()

    positive_residual = focus[focus["feature"] == "phase_slope_positive_fraction_protocol_residual"] if not focus.empty else pd.DataFrame()
    robust_positive_strata = []
    if not positive_residual.empty:
        usable = positive_residual[(positive_residual["n_event"] >= 3) & (positive_residual["n_control"] >= 3)]
        robust_positive_strata = usable[
            (usable["median_event_minus_control"] > 0)
            & (usable["bootstrap_p05"] > 0)
            & (usable["permutation_median_p"] < 0.1)
        ]["stratum"].tolist()

    os.makedirs(args.out_dir, exist_ok=True)
    joined_path = os.path.join(args.out_dir, "control_balanced_front_qc_sensitivity_joined.csv")
    strata_path = os.path.join(args.out_dir, "control_balanced_front_qc_sensitivity_strata.csv")
    tests_path = os.path.join(args.out_dir, "control_balanced_front_qc_sensitivity_tests.csv")
    focus_path = os.path.join(args.out_dir, "control_balanced_front_qc_sensitivity_focus_tests.csv")
    summary_path = os.path.join(args.out_dir, "control_balanced_front_qc_sensitivity_summary.json")
    df.to_csv(joined_path, index=False)
    stratum_summary.to_csv(strata_path, index=False)
    tests.to_csv(tests_path, index=False)
    focus.to_csv(focus_path, index=False)

    summary = {
        "source_fronts": os.path.join(args.derived_dir, "multi_cycle_threshold_robust_fronts", "threshold_robust_front_summary.csv"),
        "source_residuals": os.path.join(args.derived_dir, "protocol_conditioned_front_effects", "protocol_conditioned_front_residuals.csv"),
        "source_original_qc_manifest": os.path.join(args.derived_dir, "roi_front_qc_package", "roi_front_qc_manifest.csv"),
        "source_balanced_qc_manifest": os.path.join(args.derived_dir, "control_balanced_front_qc_package", "control_balanced_front_qc_manifest.csv"),
        "n_roi": int(len(df)),
        "n_event_roi": int((df["cohort_role"] == "event").sum()),
        "n_control_roi": int((df["cohort_role"] == "control").sum()),
        "n_bootstrap": args.n_bootstrap,
        "n_permutation": args.n_permutation,
        "strata": stratum_summary.to_dict("records"),
        "focus_tests": focus.to_dict("records") if not focus.empty else [],
        "robust_positive_phase_residual_strata": robust_positive_strata,
        "interpretation": (
            "Control-balanced QC sensitivity compares the original event-heavy review panel with the augmented "
            "event/control-balanced panel. It audits selection bias in phase-front directionality, but remains an "
            "automatic candidate review and not a manual accept/reject QC result."
        ),
        "outputs": {
            "joined": joined_path,
            "strata": strata_path,
            "tests": tests_path,
            "focus_tests": focus_path,
            "summary": summary_path,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    readme = [
        "# Control-Balanced Front QC Sensitivity",
        "",
        "Compares NMC phase/front event-control effects under the original front-QC package and the control-balanced augmentation.",
        "",
        f"- ROI rows: {summary['n_roi']}",
        f"- Event/control: {summary['n_event_roi']} / {summary['n_control_roi']}",
        f"- Robust positive phase-residual strata: {', '.join(robust_positive_strata) if robust_positive_strata else 'none'}",
        "",
        "Guardrail: this tests automatic panel selection and mask flags only. It does not replace manual front-mask QC or validate calibrated diffusion.",
    ]
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("\n".join(readme) + "\n")

    print(json.dumps({
        "out_dir": args.out_dir,
        "n_roi": summary["n_roi"],
        "robust_positive_phase_residual_strata": robust_positive_strata,
    }, indent=2))


if __name__ == "__main__":
    main()
