#!/usr/bin/env python3
"""Sensitivity tests for automatic NMC front/phase metrics under QC filters.

The threshold-front and protocol-conditioned-front analyses found event/control
front-direction effects, but many review-panel masks are fragmented. This script
asks whether the conclusions survive automatic quality strata before manual QC:
complete threshold sweeps, high front-quality score, q70 phase CI excluding
zero, and the review-panel subsets with and without fragmentation flags.
"""

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
    diffs = []
    for _ in range(n_boot):
        e = rng.choice(event, size=len(event), replace=True)
        c = rng.choice(control, size=len(control), replace=True)
        diffs.append(float(np.nanmedian(e) - np.nanmedian(c)))
    qs = np.nanpercentile(diffs, [5, 50, 95])
    return {"bootstrap_p05": float(qs[0]), "bootstrap_p50": float(qs[1]), "bootstrap_p95": float(qs[2])}


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
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        event = to_num(df.loc[df["cohort_role"] == "event", feat]).dropna().to_numpy(dtype=float)
        control = to_num(df.loc[df["cohort_role"] == "control", feat]).dropna().to_numpy(dtype=float)
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
            "feature": feat,
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


def add_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flags = out.get("auto_review_flags", pd.Series("", index=out.index)).fillna("").astype(str)
    out["review_selected"] = flags.ne("")
    out["review_fragmented_q70_mask"] = flags.str.contains("fragmented_q70_mask", regex=False)
    out["review_diffusion_ci_crosses_zero"] = flags.str.contains("diffusion_ci_crosses_zero", regex=False)
    out["review_active_control"] = flags.str.contains("active_control", regex=False)
    out["review_no_auto_flags"] = flags.eq("none")
    out["thresholds_finite"] = to_num(out["thresholds_finite"]) if "thresholds_finite" in out.columns else np.nan
    out["front_quality_score"] = to_num(out["front_quality_score"]) if "front_quality_score" in out.columns else np.nan
    if {"q70_phase_slope_bootstrap_p05", "q70_phase_slope_bootstrap_p95"}.issubset(out.columns):
        p05 = to_num(out["q70_phase_slope_bootstrap_p05"])
        p95 = to_num(out["q70_phase_slope_bootstrap_p95"])
        out["q70_phase_ci_excludes_zero"] = (p05 > 0) | (p95 < 0)
        out["q70_phase_ci_positive"] = p05 > 0
        out["q70_phase_ci_negative"] = p95 < 0
    else:
        out["q70_phase_ci_excludes_zero"] = False
        out["q70_phase_ci_positive"] = False
        out["q70_phase_ci_negative"] = False
    if {"q70_radius2_slope_bootstrap_p05_px2_per_s", "q70_radius2_slope_bootstrap_p95_px2_per_s"}.issubset(out.columns):
        r05 = to_num(out["q70_radius2_slope_bootstrap_p05_px2_per_s"])
        r95 = to_num(out["q70_radius2_slope_bootstrap_p95_px2_per_s"])
        out["q70_radius_ci_excludes_zero"] = (r05 > 0) | (r95 < 0)
        out["q70_radius_ci_crosses_zero"] = (r05 <= 0) & (r95 >= 0)
    else:
        out["q70_radius_ci_excludes_zero"] = False
        out["q70_radius_ci_crosses_zero"] = False
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/front_qc_sensitivity")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutation", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    derived = args.derived_dir
    front = pd.read_csv(os.path.join(derived, "multi_cycle_threshold_robust_fronts", "threshold_robust_front_summary.csv"))
    residuals = pd.read_csv(os.path.join(derived, "protocol_conditioned_front_effects", "protocol_conditioned_front_residuals.csv"))
    manifest_path = os.path.join(derived, "roi_front_qc_package", "roi_front_qc_manifest.csv")
    manifest = pd.read_csv(manifest_path) if os.path.exists(manifest_path) else pd.DataFrame({"roi_id": []})

    residual_keep = ["roi_id"] + [c for c in RESIDUAL_FEATURES if c in residuals.columns]
    review_keep = [
        "roi_id",
        "qc_priority_score",
        "n_components",
        "largest_component_fraction",
        "edge_touch_fraction",
        "q70_first_fraction",
        "q70_last_fraction",
        "q70_fraction_delta",
        "auto_review_flags",
        "manual_qc_status",
    ]
    review_keep = [c for c in review_keep if c in manifest.columns]
    df = front.merge(residuals[residual_keep], on="roi_id", how="left")
    if review_keep:
        df = df.merge(manifest[review_keep], on="roi_id", how="left")
    df = add_qc_flags(df)

    max_thresholds = int(np.nanmax(to_num(df["thresholds_finite"]))) if "thresholds_finite" in df.columns else 0
    fq = to_num(df["front_quality_score"])
    fq_median = float(np.nanmedian(fq))
    fq_q75 = float(np.nanpercentile(fq, 75))

    strata = {
        "all_front_rois": pd.Series(True, index=df.index),
        "complete_threshold_sweep": to_num(df["thresholds_finite"]).ge(max_thresholds),
        "front_quality_top_half": fq.ge(fq_median),
        "front_quality_top_quartile": fq.ge(fq_q75),
        "q70_phase_ci_excludes_zero": df["q70_phase_ci_excludes_zero"],
        "q70_phase_ci_positive": df["q70_phase_ci_positive"],
        "review_panel_selected": df["review_selected"],
        "review_panel_not_fragmented": df["review_selected"] & ~df["review_fragmented_q70_mask"],
        "review_panel_no_auto_flags": df["review_no_auto_flags"],
    }

    rng = np.random.default_rng(args.seed)
    test_tables = []
    stratum_rows = []
    features = RAW_FEATURES + RESIDUAL_FEATURES
    for name, mask in strata.items():
        sdf = df.loc[mask.fillna(False)].copy()
        stratum_rows.append({
            "stratum": name,
            "n_roi": int(len(sdf)),
            "n_event": int((sdf["cohort_role"] == "event").sum()) if len(sdf) else 0,
            "n_control": int((sdf["cohort_role"] == "control").sum()) if len(sdf) else 0,
            "event_fraction": float((sdf["cohort_role"] == "event").mean()) if len(sdf) else np.nan,
            "front_quality_median": float(np.nanmedian(to_num(sdf["front_quality_score"]))) if len(sdf) else np.nan,
            "phase_ci_positive_fraction": float(sdf["q70_phase_ci_positive"].mean()) if len(sdf) else np.nan,
            "fragmented_review_fraction": float(sdf["review_fragmented_q70_mask"].mean()) if len(sdf) else np.nan,
        })
        if len(sdf):
            test_tables.append(test_features(sdf, features, name, rng, args.n_bootstrap, args.n_permutation))

    stratum_summary = pd.DataFrame(stratum_rows)
    tests = pd.concat(test_tables, ignore_index=True) if test_tables else pd.DataFrame()
    if not tests.empty:
        tests = tests.sort_values(["stratum", "mannwhitney_p", "feature"], na_position="last")

    focus_features = {
        "phase_slope_positive_fraction_protocol_residual",
        "phase_slope_median_per_s",
        "threshold_robust_phase_score",
        "diffusion_proxy_abs_median_um2_per_s",
        "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    }
    focus = tests[tests["feature"].isin(focus_features)].copy() if not tests.empty else pd.DataFrame()
    if not focus.empty:
        focus["direction_matches_all"] = np.sign(focus["median_event_minus_control"]) == np.sign(
            focus.loc[
                (focus["stratum"] == "all_front_rois") & (focus["feature"].isin(focus_features)),
                ["feature", "median_event_minus_control"],
            ].set_index("feature")["median_event_minus_control"].reindex(focus["feature"]).to_numpy()
        )

    os.makedirs(args.out_dir, exist_ok=True)
    joined_path = os.path.join(args.out_dir, "front_qc_sensitivity_joined.csv")
    strata_path = os.path.join(args.out_dir, "front_qc_sensitivity_strata.csv")
    tests_path = os.path.join(args.out_dir, "front_qc_sensitivity_tests.csv")
    focus_path = os.path.join(args.out_dir, "front_qc_sensitivity_focus_tests.csv")
    df.to_csv(joined_path, index=False)
    stratum_summary.to_csv(strata_path, index=False)
    tests.to_csv(tests_path, index=False)
    focus.to_csv(focus_path, index=False)

    top_by_stratum = []
    for name in strata:
        if tests.empty:
            continue
        top_by_stratum.extend(tests[tests["stratum"] == name].head(6).to_dict("records"))

    positive_residual = focus[focus["feature"] == "phase_slope_positive_fraction_protocol_residual"] if not focus.empty else pd.DataFrame()
    robust_positive_strata = []
    if not positive_residual.empty:
        usable = positive_residual[(positive_residual["n_event"] >= 3) & (positive_residual["n_control"] >= 3)]
        robust_positive_strata = usable[
            (usable["median_event_minus_control"] > 0) &
            (usable["bootstrap_p05"] > 0)
        ]["stratum"].tolist()

    summary = {
        "source_fronts": os.path.join(derived, "multi_cycle_threshold_robust_fronts", "threshold_robust_front_summary.csv"),
        "source_residuals": os.path.join(derived, "protocol_conditioned_front_effects", "protocol_conditioned_front_residuals.csv"),
        "source_qc_manifest": manifest_path,
        "n_roi": int(len(df)),
        "n_event_roi": int((df["cohort_role"] == "event").sum()),
        "n_control_roi": int((df["cohort_role"] == "control").sum()),
        "max_thresholds_finite": max_thresholds,
        "front_quality_median": fq_median,
        "front_quality_q75": fq_q75,
        "n_bootstrap": args.n_bootstrap,
        "n_permutation": args.n_permutation,
        "strata": stratum_summary.to_dict("records"),
        "top_tests_by_stratum": top_by_stratum,
        "focus_tests": focus.to_dict("records") if not focus.empty else [],
        "robust_positive_phase_residual_strata": robust_positive_strata,
        "interpretation": (
            "Automatic QC-filtered sensitivity tests evaluate whether phase/front event-control effects persist "
            "when restricting to higher-quality or reviewable masks. Passing these filters is not a substitute for manual QC."
        ),
        "outputs": {
            "joined": joined_path,
            "strata": strata_path,
            "tests": tests_path,
            "focus_tests": focus_path,
            "summary": os.path.join(args.out_dir, "front_qc_sensitivity_summary.json"),
        },
    }
    with open(summary["outputs"]["summary"], "w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    readme = [
        "# Front QC Sensitivity",
        "",
        "Automatic quality-stratified tests for NMC phase/front metrics.",
        "",
        f"- ROI rows: {summary['n_roi']}",
        f"- Event/control: {summary['n_event_roi']} / {summary['n_control_roi']}",
        f"- Complete-threshold count: {int(stratum_summary.loc[stratum_summary['stratum'] == 'complete_threshold_sweep', 'n_roi'].iloc[0])}",
        f"- High-quality median cutoff: {fq_median:.3f}",
        f"- Positive phase-residual robust strata: {', '.join(robust_positive_strata) if robust_positive_strata else 'none'}",
        "",
        "Guardrail: automatic QC strata reduce obvious mask artifacts, but publication-scale front or diffusion claims still require manual review labels.",
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
