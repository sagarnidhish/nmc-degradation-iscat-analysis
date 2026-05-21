#!/usr/bin/env python3
"""Summarize timing-aware mobility descriptors for the multi-cycle ROI cohort."""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from tier3_roi_phase_boundary_mobility import timing_seconds, trace_features


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def add_roi_id(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["roi_id"] = out.apply(
        lambda r: f"cycle{int(float(r['cycleNo']))}_rank{int(r['front_candidate_rank'])}_obj{int(r['object_candidate_rank'])}",
        axis=1,
    )
    return out


def descriptor_rows(manifest_path: str, cohort_table_path: str, root: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    cohort = add_roi_id(pd.read_csv(cohort_table_path))
    meta_cols = [
        "roi_id",
        "cohort_role",
        "event_reference_cycle",
        "validation_score",
        "front_quality_score",
        "front_radius_slope_r2",
        "apparent_diffusion_proxy_ds_px2_per_frame",
        "n_event_particles",
        "mean_drop_frac",
        "evidence_score",
        "degradation_mode_hypothesis",
    ]
    meta_cols = [c for c in meta_cols if c in cohort.columns]
    merged = manifest.merge(cohort[meta_cols].drop_duplicates("roi_id"), how="left", on="roi_id")

    rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        with np.load(row["npz_path"]) as data:
            features = trace_features(data["frames_norm"])
            frame_indices = data["frame_indices"] if "frame_indices" in data else np.array([], dtype=int)
        timing = timing_seconds(root, str(row.get("source_stem", "")), frame_indices)
        seconds_per_sample = timing["seconds_per_sample"]
        if np.isfinite(seconds_per_sample) and seconds_per_sample > 0:
            for col in [
                "high_fraction_slope_per_frame",
                "low_fraction_slope_per_frame",
                "mid_fraction_slope_per_frame",
                "high_radius2_slope_px2_per_frame",
                "low_radius2_slope_px2_per_frame",
                "interface_density_slope_per_frame",
            ]:
                features[col.replace("_per_frame", "_per_s")] = features[col] / seconds_per_sample
        out = {
            "roi_id": row["roi_id"],
            "cycleNo": float(row["cycleNo"]),
            "source_stem": row.get("source_stem", ""),
            "cohort_role": row.get("cohort_role", ""),
            "is_event_roi": int(row.get("cohort_role", "") == "event"),
            "event_reference_cycle": finite_float(row.get("event_reference_cycle")),
            "validation_score": finite_float(row.get("validation_score")),
            "front_quality_score": finite_float(row.get("front_quality_score")),
            "front_radius_slope_r2": finite_float(row.get("front_radius_slope_r2")),
            "apparent_diffusion_proxy_ds_px2_per_frame": finite_float(row.get("apparent_diffusion_proxy_ds_px2_per_frame")),
            "n_event_particles": finite_float(row.get("n_event_particles")),
            "mean_drop_frac": finite_float(row.get("mean_drop_frac")),
            "evidence_score": finite_float(row.get("evidence_score")),
            "degradation_mode_hypothesis": row.get("degradation_mode_hypothesis", ""),
            "roi_mean_delta": finite_float(row.get("roi_norm_mean_delta_last_minus_first")),
            **timing,
        }
        out.update(features)
        rows.append(out)
    return pd.DataFrame(rows)


def feature_tests(df: pd.DataFrame, features: List[str], group_col: str = "") -> pd.DataFrame:
    rows = []
    groups = [(np.nan, df)] if not group_col else list(df.groupby(group_col, dropna=False))
    for group_value, grp in groups:
        event = grp[grp["is_event_roi"] == 1]
        control = grp[grp["is_event_roi"] == 0]
        for feat in features:
            a = pd.to_numeric(event[feat], errors="coerce").dropna()
            b = pd.to_numeric(control[feat], errors="coerce").dropna()
            if len(a) and len(b):
                try:
                    stat, p = mannwhitneyu(a, b, alternative="two-sided")
                except Exception:
                    stat, p = np.nan, np.nan
            else:
                stat, p = np.nan, np.nan
            rows.append({
                "group": group_col or "all",
                "group_value": group_value,
                "feature": feat,
                "event_mean": float(a.mean()) if len(a) else np.nan,
                "control_mean": float(b.mean()) if len(b) else np.nan,
                "event_minus_control": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
                "n_event": int(len(a)),
                "n_control": int(len(b)),
                "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["group", "p_value"], na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_sequences/selected_roi_sequence_manifest.csv")
    parser.add_argument("--cohort-table", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_cohort/multi_cycle_roi_table.csv")
    parser.add_argument("--root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/multi_cycle_roi_mobility")
    args = parser.parse_args()

    df = descriptor_rows(args.manifest, args.cohort_table, args.root)
    features = [
        "roi_mean_delta",
        "high_fraction_slope_per_s",
        "mid_fraction_slope_per_s",
        "high_radius2_delta_px2",
        "high_radius2_slope_px2_per_s",
        "interface_density_delta",
        "centroid_path_px",
        "first_last_corr",
        "cumulative_abs_first_last",
        "temporal_diff_energy",
    ]
    tests_all = feature_tests(df, features)
    tests_by_event = feature_tests(df, features, "event_reference_cycle")
    group = df.groupby(["cohort_role", "event_reference_cycle", "cycleNo"], dropna=False).agg({
        "roi_id": "count",
        "roi_mean_delta": "mean",
        "high_fraction_slope_per_s": "mean",
        "high_radius2_slope_px2_per_s": "mean",
        "interface_density_delta": "mean",
        "first_last_corr": "mean",
        "cumulative_abs_first_last": "mean",
    }).reset_index().rename(columns={"roi_id": "n_roi"})
    ranked = df.copy()
    ranked["multicycle_mobility_score"] = (
        ranked["cumulative_abs_first_last"].rank(pct=True)
        + ranked["centroid_path_px"].rank(pct=True)
        - ranked["first_last_corr"].rank(pct=True)
        + ranked["high_fraction_slope_per_s"].rank(pct=True)
    )
    ranked = ranked.sort_values("multicycle_mobility_score", ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    desc_path = os.path.join(args.out_dir, "multi_cycle_roi_mobility_descriptors.csv")
    tests_path = os.path.join(args.out_dir, "multi_cycle_roi_mobility_feature_tests.csv")
    tests_by_event_path = os.path.join(args.out_dir, "multi_cycle_roi_mobility_by_event_tests.csv")
    group_path = os.path.join(args.out_dir, "multi_cycle_roi_mobility_group_summary.csv")
    ranked_path = os.path.join(args.out_dir, "multi_cycle_roi_mobility_ranked.csv")
    df.to_csv(desc_path, index=False)
    tests_all.to_csv(tests_path, index=False)
    tests_by_event.to_csv(tests_by_event_path, index=False)
    group.to_csv(group_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    summary = {
        "n_roi": int(len(df)),
        "n_event_roi": int((df["is_event_roi"] == 1).sum()),
        "n_control_roi": int((df["is_event_roi"] == 0).sum()),
        "event_reference_cycles": sorted(float(x) for x in df["event_reference_cycle"].dropna().unique()),
        "cycle_counts": group.to_dict(orient="records"),
        "top_overall_feature_tests": tests_all.head(10).to_dict(orient="records"),
        "top_by_event_feature_tests": tests_by_event.groupby("group_value", dropna=False).head(5).to_dict(orient="records"),
        "top_ranked_rois": ranked.head(12)[[
            "roi_id",
            "cohort_role",
            "cycleNo",
            "event_reference_cycle",
            "multicycle_mobility_score",
            "roi_mean_delta",
            "high_fraction_slope_per_s",
            "high_radius2_slope_px2_per_s",
            "first_last_corr",
            "cumulative_abs_first_last",
        ]].to_dict(orient="records"),
        "guardrail": "Automatic multi-cycle ROI cohort; use for hypothesis ranking and model inputs, not manual particle annotations.",
        "outputs": {
            "descriptors": desc_path,
            "feature_tests": tests_path,
            "by_event_tests": tests_by_event_path,
            "group_summary": group_path,
            "ranked": ranked_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "multi_cycle_roi_mobility_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Multi-Cycle ROI Mobility\n\n")
        f.write("Timing-aware mobility descriptors for automatically selected event/control ROIs across cycles 60, 86, 116, and 156.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
