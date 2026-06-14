#!/usr/bin/env python3
"""Build a guarded weak-label benchmark manifest for ROI video degradation modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


EVENT_MODE_LABELS = {
    "optical_brightening_decorrelating_rollout_hard_front_positive",
}

POSITIVE_TIERS = {
    "cross_modal_high_priority",
    "cross_modal_review_priority",
    "front_kinetic_consistent",
    "rollout_mode_consistent",
}


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


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    med = vals.median(skipna=True)
    mad = (vals - med).abs().median(skipna=True)
    scale = 1.4826 * mad if pd.notna(mad) and mad > 1e-12 else vals.std(skipna=True)
    if pd.isna(scale) or scale <= 1e-12:
        return vals * 0.0
    return (vals - med) / scale


def semicolon_join(parts: List[str]) -> str:
    return ";".join([p for p in parts if p])


def assign_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["physics_consistency_score", "precursor_informed_review_score", "mode_review_priority", "mask_instability_score"]:
        if col in out.columns:
            out[f"{col}_robust_z"] = zscore(out[col])

    mask_q90 = float(pd.to_numeric(out.get("mask_instability_score"), errors="coerce").quantile(0.90))
    physics_q25 = float(pd.to_numeric(out.get("physics_consistency_score"), errors="coerce").quantile(0.25))
    stable_mask = pd.to_numeric(out.get("mask_instability_score"), errors="coerce").le(mask_q90)
    no_severe_auto_flags = ~out.get("auto_review_flags", "").fillna("").astype(str).str.contains("fragmented|edge_touch|threshold_sign_unstable", case=False, regex=True)
    event_mode = out.get("mode_label", "").fillna("").isin(EVENT_MODE_LABELS)
    positive_tier = out.get("physics_consistency_tier", "").fillna("").isin(POSITIVE_TIERS)
    event_role = out.get("cohort_role", "").fillna("").eq("event")
    control_role = out.get("cohort_role", "").fillna("").eq("control")
    low_consistency = pd.to_numeric(out.get("physics_consistency_score"), errors="coerce").le(physics_q25)

    labels = []
    binary = []
    confidence = []
    exclusions = []
    for idx, row in out.iterrows():
        reasons: List[str] = []
        exclude: List[str] = []
        if not bool(stable_mask.loc[idx]):
            exclude.append("top_decile_mask_instability")
        if not bool(no_severe_auto_flags.loc[idx]):
            exclude.append("auto_artifact_flag")
        if row.get("manual_qc_status", "pending") != "pending":
            reasons.append(f"manual_qc_status_{row.get('manual_qc_status')}")

        if bool(event_role.loc[idx]) and bool(positive_tier.loc[idx]) and bool(event_mode.loc[idx]) and not exclude:
            labels.append("weak_event_enriched_front_mode")
            binary.append(1)
            confidence.append("high")
            reasons += ["event_role", "positive_physics_tier", "event_enriched_mode", "stable_auto_mask"]
        elif bool(control_role.loc[idx]) and bool(low_consistency.loc[idx]) and not bool(event_mode.loc[idx]) and not exclude:
            labels.append("weak_low_consistency_control")
            binary.append(0)
            confidence.append("medium")
            reasons += ["control_role", "bottom_quartile_physics_consistency", "not_event_enriched_mode", "stable_auto_mask"]
        elif bool(positive_tier.loc[idx]) or bool(event_mode.loc[idx]):
            labels.append("review_positive_uncertain")
            binary.append(np.nan)
            confidence.append("review_only")
            reasons += ["positive_or_event_enriched_evidence"]
        elif bool(control_role.loc[idx]):
            labels.append("review_control_uncertain")
            binary.append(np.nan)
            confidence.append("review_only")
            reasons += ["control_role_no_high_confidence_negative"]
        else:
            labels.append("review_uncertain")
            binary.append(np.nan)
            confidence.append("review_only")
            reasons += ["insufficient_consensus"]

        out.loc[idx, "weak_label_reason"] = semicolon_join(reasons)
        exclusions.append(semicolon_join(exclude))

    out["weak_label_class"] = labels
    out["weak_binary_degradation_mode_label"] = binary
    out["weak_label_confidence"] = confidence
    out["weak_label_exclusion_reason"] = exclusions
    out["is_trainable_weak_label"] = out["weak_binary_degradation_mode_label"].isin([0, 1]).astype(int)
    return out


def make_leave_reference_splits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref = pd.to_numeric(df["event_reference_cycle"], errors="coerce").fillna(pd.to_numeric(df["cycleNo"], errors="coerce"))
    refs = sorted([float(x) for x in ref.dropna().unique()])
    trainable = df["is_trainable_weak_label"].eq(1)
    for fold, holdout_ref in enumerate(refs, start=1):
        test_mask = ref.eq(holdout_ref)
        train_mask = ~test_mask
        row = {
            "fold": fold,
            "holdout_event_reference_cycle": holdout_ref,
            "n_train_rows": int(train_mask.sum()),
            "n_test_rows": int(test_mask.sum()),
            "n_trainable_train": int((train_mask & trainable).sum()),
            "n_trainable_test": int((test_mask & trainable).sum()),
            "n_positive_train": int((train_mask & trainable & df["weak_binary_degradation_mode_label"].eq(1)).sum()),
            "n_negative_train": int((train_mask & trainable & df["weak_binary_degradation_mode_label"].eq(0)).sum()),
            "n_positive_test": int((test_mask & trainable & df["weak_binary_degradation_mode_label"].eq(1)).sum()),
            "n_negative_test": int((test_mask & trainable & df["weak_binary_degradation_mode_label"].eq(0)).sum()),
        }
        row["trainable_fold_status"] = (
            "usable_binary_fold"
            if min(row["n_positive_train"], row["n_negative_train"], row["n_positive_test"], row["n_negative_test"]) > 0
            else "weak_label_class_missing_in_train_or_test"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/weak_label_degradation_benchmark")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seq = read_csv(derived / "multi_cycle_roi_sequences" / "selected_roi_sequence_manifest.csv")
    phys = read_csv(derived / "physics_consistency_claim_matrix" / "physics_consistency_claim_matrix.csv")
    review = read_csv(derived / "precursor_informed_roi_review" / "precursor_informed_roi_review_manifest.csv")
    masks = read_csv(derived / "particle_mask_stability_audit" / "particle_mask_stability_per_roi.csv")
    bridge = read_csv(derived / "cycle_state_roi_bridge" / "cycle_state_roi_bridge_joined.csv")
    rollout = read_csv(derived / "probabilistic_rollout_calibration" / "probabilistic_rollout_roi_table.csv")

    if seq.empty or phys.empty:
        raise FileNotFoundError("selected ROI sequence manifest and physics consistency matrix are required.")

    cols = ["roi_id", "cycleNo", "event_reference_cycle", "cohort_role", "physics_consistency_tier", "claim_readiness", "physics_consistency_score", "front_direction_score", "optical_change_score", "rollout_residual_score", "kinetic_transition_score", "mode_taxonomy_score", "mode_label", "manual_qc_status", "auto_review_flags"]
    df = phys[[c for c in cols if c in phys.columns]].copy()
    seq_keep = ["roi_id", "npz_path", "preview_png", "source_stem", "front_candidate_rank", "object_candidate_rank", "validation_score", "validation_label", "n_frames", "stage_drift_xy_sampled", "object_x_full_approx", "object_y_full_approx"]
    df = df.merge(seq[[c for c in seq_keep if c in seq.columns]], on="roi_id", how="left")

    review_keep = ["roi_id", "review_sources", "review_priority_tier", "combined_review_priority_score", "precursor_informed_review_score", "precursor_review_tier", "precursor_review_reason", "primary_qc_png", "control_balanced_qc_png", "rollout_preview_path"]
    if not review.empty:
        df = df.merge(review[[c for c in review_keep if c in review.columns]], on="roi_id", how="left", suffixes=("", "_review"))
    mask_keep = ["roi_id", "mask_instability_score", "fallback_frame_fraction", "accepted_area_cv", "accepted_area_fraction_iqr", "accepted_centroid_path_px"]
    if not masks.empty:
        df = df.merge(masks[[c for c in mask_keep if c in masks.columns]], on="roi_id", how="left")
    bridge_keep = ["roi_id", "cycle_state_pc1", "cycle_state_pc2", "cycle_state_pc3", "cycle_state_cluster", "future_any_drop_within_8cycles"]
    if not bridge.empty:
        df = df.merge(bridge[[c for c in bridge_keep if c in bridge.columns]], on="roi_id", how="left")
    if not rollout.empty and {"roi_id", "method"}.issubset(rollout.columns):
        dmd = rollout[rollout["method"].eq("low_rank_dmd")].copy()
        keep = [c for c in ["roi_id", "q90_undercoverage_rate", "mae_mean", "mae_max"] if c in dmd.columns]
        df = df.merge(dmd[keep].rename(columns={c: f"dmd_{c}" for c in keep if c != "roi_id"}), on="roi_id", how="left")

    df = assign_weak_labels(df)
    splits = make_leave_reference_splits(df)

    label_counts = df["weak_label_class"].value_counts(dropna=False).to_dict()
    confidence_counts = df["weak_label_confidence"].value_counts(dropna=False).to_dict()
    trainable = df[df["is_trainable_weak_label"].eq(1)]
    leakage = {
        "split_strategy": "leave_event_reference_cycle_out",
        "n_folds": int(len(splits)),
        "n_usable_binary_folds": int(splits["trainable_fold_status"].eq("usable_binary_fold").sum()) if not splits.empty else 0,
        "folds_with_missing_class": splits[splits["trainable_fold_status"].ne("usable_binary_fold")].to_dict("records") if not splits.empty else [],
        "warning": "Weak labels are consensus-derived review targets, not ground truth. Use splits to avoid event-reference leakage; do not report deployment metrics from these labels.",
    }
    summary = {
        "n_roi_rows": int(len(df)),
        "n_trainable_weak_label_rows": int(len(trainable)),
        "n_positive_weak_labels": int(trainable["weak_binary_degradation_mode_label"].eq(1).sum()),
        "n_negative_weak_labels": int(trainable["weak_binary_degradation_mode_label"].eq(0).sum()),
        "label_counts": clean_json(label_counts),
        "confidence_counts": clean_json(confidence_counts),
        "leakage_audit": clean_json(leakage),
        "top_positive_training_rows": clean_json(df[df["weak_binary_degradation_mode_label"].eq(1)].sort_values("physics_consistency_score", ascending=False).head(12).to_dict("records")),
        "top_negative_training_rows": clean_json(df[df["weak_binary_degradation_mode_label"].eq(0)].sort_values("physics_consistency_score", ascending=True).head(12).to_dict("records")),
        "guardrail": "This benchmark contains weak consensus labels for model development and review prioritization only. It is not a manual-QC label set and must not be used to claim validated degradation modes or calibrated diffusion.",
        "outputs": {
            "manifest": str(out / "weak_label_degradation_benchmark_manifest.csv"),
            "splits": str(out / "weak_label_leave_reference_splits.csv"),
            "summary": str(out / "weak_label_degradation_benchmark_summary.json"),
        },
    }

    df.to_csv(out / "weak_label_degradation_benchmark_manifest.csv", index=False)
    splits.to_csv(out / "weak_label_leave_reference_splits.csv", index=False)
    with (out / "weak_label_degradation_benchmark_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)

    lines = [
        "# Weak-Label Degradation Benchmark",
        "",
        "Consensus weak-label manifest for ROI video model development and degradation-mode review.",
        "",
        f"- ROI rows: {summary['n_roi_rows']}",
        f"- Trainable weak-label rows: {summary['n_trainable_weak_label_rows']}",
        f"- Positive/negative weak labels: {summary['n_positive_weak_labels']} / {summary['n_negative_weak_labels']}",
        f"- Leave-reference folds: {leakage['n_folds']} ({leakage['n_usable_binary_folds']} usable binary folds)",
        f"- Label counts: {summary['label_counts']}",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
