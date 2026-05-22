#!/usr/bin/env python3
"""Audit HDF5 camera timebase provenance for apparent front/diffusion proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


STRICT_DT_MAX_TO_MEDIAN = 1.25
ROI_H5_RATIO_TOL = 0.02
Q70 = 0.70


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def records(df: pd.DataFrame, n: int = 20) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.head(n).to_json(orient="records"))


def feature_test(df: pd.DataFrame, feature: str, target: str) -> Dict[str, Any]:
    if feature not in df or target not in df:
        return {"feature": feature, "target": target, "status": "missing_column"}
    x = to_num(df[feature])
    y = to_num(df[target])
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid].astype(int)
    if y.nunique() != 2:
        return {"feature": feature, "target": target, "status": "single_class", "n": int(valid.sum())}
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) < 3 or len(neg) < 3:
        return {"feature": feature, "target": target, "status": "too_few_rows", "n_positive": int(len(pos)), "n_negative": int(len(neg))}
    p = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    try:
        auc = float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))
    except ZeroDivisionError:
        auc = np.nan
    return {
        "feature": feature,
        "target": target,
        "status": "ok",
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "median_positive": float(pos.median()),
        "median_negative": float(neg.median()),
        "median_positive_minus_negative": float(pos.median() - neg.median()),
        "positive_fraction_positive": float((pos > 0).mean()),
        "positive_fraction_negative": float((neg > 0).mean()),
        "mannwhitney_p": p,
        "roc_auc_positive_high": auc,
        "abs_oriented_auc": float(max(auc, 1.0 - auc)) if np.isfinite(auc) else None,
    }


def correlations(df: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for x_col, y_col in pairs:
        if x_col not in df or y_col not in df:
            continue
        x = to_num(df[x_col])
        y = to_num(df[y_col])
        valid = x.notna() & y.notna()
        if valid.sum() < 10 or x[valid].nunique() < 3 or y[valid].nunique() < 3:
            continue
        rho, p = spearmanr(x[valid], y[valid])
        rows.append(
            {
                "x": x_col,
                "y": y_col,
                "n": int(valid.sum()),
                "spearman_rho": float(rho),
                "abs_rho": float(abs(rho)),
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_rho", "p_value"], ascending=[False, True])
    return out


def scenario_table(q70: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = {
        "all_q70_rows": q70,
        "strict_h5_timebase_sources": q70[q70["source_strict_timebase"]],
        "pause_heavy_sources": q70[~q70["source_strict_timebase"]],
        "roi_elapsed_h5_aligned": q70[q70["roi_h5_elapsed_aligned"]],
        "strict_source_and_roi_aligned": q70[q70["source_strict_timebase"] & q70["roi_h5_elapsed_aligned"]],
    }
    for name, df in scenarios.items():
        test = feature_test(df, "apparent_D_h5median_px0p096_um2_per_s", "future_any_drop_within_8cycles")
        rows.append(
            {
                "scenario": name,
                "n_rows": int(len(df)),
                "n_roi": int(df["roi_id"].nunique()) if "roi_id" in df else 0,
                "n_sources": int(df["source_stem"].nunique()) if "source_stem" in df else 0,
                "median_dt_max_to_median": float(to_num(df["h5_dt_max_to_median_ratio"]).median()) if len(df) else np.nan,
                "median_roi_elapsed_to_h5": float(to_num(df["roi_elapsed_to_h5_median_ratio"]).median()) if len(df) else np.nan,
                "median_D_um2_per_s": float(to_num(df["apparent_D_h5median_px0p096_um2_per_s"]).median()) if len(df) else np.nan,
                "median_abs_D_um2_per_s": float(to_num(df["apparent_D_h5median_abs_um2_per_s"]).median()) if len(df) else np.nan,
                "positive_D_fraction": float((to_num(df["apparent_D_h5median_px0p096_um2_per_s"]) > 0).mean()) if len(df) else np.nan,
                "future8_median_positive_minus_negative": test.get("median_positive_minus_negative"),
                "future8_mannwhitney_p": test.get("mannwhitney_p"),
                "future8_abs_oriented_auc": test.get("abs_oriented_auc"),
                "test_status": test.get("status"),
            }
        )
    return pd.DataFrame(rows)


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# HDF5 Timebase Provenance Audit",
        "",
        "Source-level and ROI-level audit of camera timing used to express front and apparent-diffusion proxies per second.",
        "",
        f"- q70 ROI rows: {summary['n_q70_roi_rows']}",
        f"- Sources: {summary['n_sources']}",
        f"- Strict timebase sources: {summary['n_strict_timebase_sources']}",
        f"- Pause-heavy sources: {summary['n_pause_heavy_sources']}",
        f"- ROI/HDF5 elapsed-aligned rows: {summary['n_roi_elapsed_h5_aligned']} / {summary['n_q70_roi_rows']}",
        f"- Median ROI/HDF5 elapsed ratio: {summary['median_roi_elapsed_to_h5_ratio']:.6f}",
        f"- Max source dt max/median ratio: {summary['max_source_dt_max_to_median_ratio']:.3f}",
        f"- Timebase status: {summary['timebase_status']}",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Outputs",
        "",
        "- `hdf5_timebase_source_summary.csv`",
        "- `hdf5_timebase_roi_q70_table.csv`",
        "- `hdf5_timebase_scenario_table.csv`",
        "- `hdf5_timebase_correlations.csv`",
        "- `hdf5_timebase_provenance_summary.json`",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/hdf5_timebase_provenance_audit")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = ensure_dir(Path(args.out_dir))

    joined = pd.read_csv(derived / "apparent_diffusion_calibration_bounds" / "apparent_diffusion_calibration_joined.csv")
    h5 = pd.read_csv(derived / "calibration_metadata_audit" / "h5_calibration_metadata.csv")
    diffusion_scores = pd.read_csv(derived / "diffusion_physics_consistency_audit" / "diffusion_physics_consistency_roi_scores.csv")

    q70 = joined[np.isclose(to_num(joined["threshold_quantile"]), Q70)].copy()
    q70["h5_dt_median_s"] = to_num(q70["sampled_timing_dt_median_s"])
    q70["h5_dt_min_s"] = to_num(q70["sampled_timing_dt_min_s"])
    q70["h5_dt_max_s"] = to_num(q70["sampled_timing_dt_max_s"])
    q70["h5_dt_max_to_median_ratio"] = to_num(q70["h5_dt_max_to_median_ratio"])
    q70["roi_elapsed_to_h5_median_ratio"] = to_num(q70["roi_elapsed_to_h5_median_ratio"])
    q70["source_strict_timebase"] = q70["h5_dt_max_to_median_ratio"].le(STRICT_DT_MAX_TO_MEDIAN)
    q70["roi_h5_elapsed_aligned"] = (q70["roi_elapsed_to_h5_median_ratio"] - 1.0).abs().le(ROI_H5_RATIO_TOL)
    q70["timebase_gate"] = q70["source_strict_timebase"] & q70["roi_h5_elapsed_aligned"]

    score_cols = [
        "roi_id",
        "automatic_diffusion_physics_consistent",
        "publication_ready_diffusion_candidate",
        "gate_h5_timing_stable",
        "gate_q70_positive_ci",
        "gate_fit_quality",
        "gate_positive_expansion",
        "physics_consistency_score",
        "median_radius2_fit_r2",
        "positive_D_fraction",
    ]
    q70 = q70.merge(diffusion_scores[[c for c in score_cols if c in diffusion_scores.columns]], on="roi_id", how="left", suffixes=("", "_physics"))

    source = (
        q70.groupby("source_stem", dropna=False)
        .agg(
            n_q70_roi=("roi_id", "nunique"),
            n_cycles=("cycleNo", "nunique"),
            future8_positive_fraction=("future_any_drop_within_8cycles", "mean"),
            h5_dt_median_s=("h5_dt_median_s", "median"),
            h5_dt_min_s=("h5_dt_min_s", "median"),
            h5_dt_max_s=("h5_dt_max_s", "median"),
            h5_dt_max_to_median_ratio=("h5_dt_max_to_median_ratio", "median"),
            roi_elapsed_to_h5_median_ratio=("roi_elapsed_to_h5_median_ratio", "median"),
            roi_elapsed_to_h5_max_abs_error=("roi_elapsed_to_h5_median_ratio", lambda x: float(np.nanmax(np.abs(to_num(x) - 1.0)))),
            strict_timebase_fraction=("source_strict_timebase", "mean"),
            roi_h5_elapsed_aligned_fraction=("roi_h5_elapsed_aligned", "mean"),
            median_D_um2_per_s=("apparent_D_h5median_px0p096_um2_per_s", "median"),
            median_abs_D_um2_per_s=("apparent_D_h5median_abs_um2_per_s", "median"),
            positive_D_fraction=("apparent_D_h5median_px0p096_um2_per_s", lambda x: float((to_num(x) > 0).mean())),
        )
        .reset_index()
    )
    source["source_strict_timebase"] = source["h5_dt_max_to_median_ratio"].le(STRICT_DT_MAX_TO_MEDIAN)
    source = source.sort_values(["source_strict_timebase", "h5_dt_max_to_median_ratio", "source_stem"], ascending=[True, False, True])

    h5_keep = [
        "relative_path",
        "movie_shape",
        "camera_timing_shape",
        "sampled_timing_duration_s",
        "sampled_timing_dt_median_s",
        "sampled_timing_dt_min_s",
        "sampled_timing_dt_max_s",
        "sampled_timing_fps_median",
        "camera_timing_unit_inference",
    ]
    h5_small = h5[[c for c in h5_keep if c in h5.columns]].copy()
    h5_small["source_stem"] = h5_small["relative_path"].astype(str).str.rsplit("/", n=1).str[-1].str.replace(".hdf5", "", regex=False)
    h5_small["dt_max_to_median_ratio"] = to_num(h5_small["sampled_timing_dt_max_s"]) / to_num(h5_small["sampled_timing_dt_median_s"])
    h5_small["strict_timebase"] = h5_small["dt_max_to_median_ratio"].le(STRICT_DT_MAX_TO_MEDIAN)
    h5_small = h5_small.sort_values(["strict_timebase", "dt_max_to_median_ratio", "source_stem"], ascending=[True, False, True])

    scenarios = scenario_table(q70)
    corr = correlations(
        q70,
        [
            ("h5_dt_max_to_median_ratio", "cycleNo"),
            ("h5_dt_max_to_median_ratio", "future_any_drop_within_8cycles"),
            ("h5_dt_max_to_median_ratio", "future_any_drop_within_16cycles"),
            ("h5_dt_max_to_median_ratio", "transferred_masked_residual_signature"),
            ("roi_elapsed_to_h5_median_ratio", "transferred_masked_residual_signature"),
            ("roi_elapsed_to_h5_median_ratio", "validation_score_recon"),
            ("h5_dt_max_to_median_ratio", "apparent_D_h5median_abs_um2_per_s"),
        ],
    )

    n_sources = int(source["source_stem"].nunique())
    n_strict = int(source["source_strict_timebase"].sum())
    n_pause = int((~source["source_strict_timebase"]).sum())
    strict_rows = int(q70["timebase_gate"].sum())
    if n_pause and strict_rows < len(q70):
        status = "mixed_timebase_pause_heavy_sources_present"
        interpretation = (
            "ROI elapsed times align closely with HDF5 median frame spacing, but several source files contain large "
            "camera-timing gaps relative to their 10 s median frame interval. Per-second apparent front/diffusion "
            "values are usable as median-timebase proxies, while calibrated diffusion claims should either exclude "
            "pause-heavy sources or model dropped/paused acquisition intervals explicitly."
        )
    elif n_strict == n_sources and strict_rows == len(q70):
        status = "hdf5_timebase_consistent_for_q70_roi_set"
        interpretation = (
            "All q70 ROI rows pass strict source timebase and ROI/HDF5 elapsed checks. This clears the timing "
            "provenance component for these apparent proxies, but spatial calibration, manual front QC, and front "
            "identity gates still govern diffusion wording."
        )
    else:
        status = "timebase_alignment_unresolved"
        interpretation = (
            "The q70 ROI set has incomplete or inconsistent HDF5 timing alignment. Keep front and diffusion-like "
            "readouts in frame-normalized proxy units until source timing is reconciled."
        )

    summary = {
        "strict_dt_max_to_median_threshold": STRICT_DT_MAX_TO_MEDIAN,
        "roi_h5_elapsed_ratio_tolerance": ROI_H5_RATIO_TOL,
        "n_h5_files": int(len(h5_small)),
        "n_q70_roi_rows": int(len(q70)),
        "n_sources": n_sources,
        "n_strict_timebase_sources": n_strict,
        "n_pause_heavy_sources": n_pause,
        "n_roi_elapsed_h5_aligned": int(q70["roi_h5_elapsed_aligned"].sum()),
        "n_q70_rows_passing_full_timebase_gate": strict_rows,
        "median_dt_median_s": float(q70["h5_dt_median_s"].median()),
        "median_roi_elapsed_to_h5_ratio": float(q70["roi_elapsed_to_h5_median_ratio"].median()),
        "max_source_dt_max_to_median_ratio": float(source["h5_dt_max_to_median_ratio"].max()),
        "pause_heavy_sources": source.loc[~source["source_strict_timebase"], "source_stem"].astype(str).tolist(),
        "strict_timebase_sources": source.loc[source["source_strict_timebase"], "source_stem"].astype(str).tolist(),
        "scenario_summary": records(scenarios, 10),
        "top_timebase_correlations": records(corr, 12),
        "top_pause_heavy_sources": records(source[~source["source_strict_timebase"]], 12),
        "timebase_status": status,
        "interpretation": interpretation,
        "guardrail": "This audit validates timing provenance only. It does not validate spatial calibration, manual front identity, q70 confidence intervals, or material diffusion coefficients.",
        "outputs": {
            "source_summary": str(out / "hdf5_timebase_source_summary.csv"),
            "roi_q70_table": str(out / "hdf5_timebase_roi_q70_table.csv"),
            "h5_file_table": str(out / "hdf5_timebase_h5_file_table.csv"),
            "scenario_table": str(out / "hdf5_timebase_scenario_table.csv"),
            "correlations": str(out / "hdf5_timebase_correlations.csv"),
            "summary": str(out / "hdf5_timebase_provenance_summary.json"),
            "readme": str(out / "README.md"),
        },
    }

    source.to_csv(out / "hdf5_timebase_source_summary.csv", index=False)
    q70.to_csv(out / "hdf5_timebase_roi_q70_table.csv", index=False)
    h5_small.to_csv(out / "hdf5_timebase_h5_file_table.csv", index=False)
    scenarios.to_csv(out / "hdf5_timebase_scenario_table.csv", index=False)
    corr.to_csv(out / "hdf5_timebase_correlations.csv", index=False)
    with (out / "hdf5_timebase_provenance_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)
    write_readme(out, clean_json(summary))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
