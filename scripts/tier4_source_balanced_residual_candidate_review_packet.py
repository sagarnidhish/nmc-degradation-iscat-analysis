#!/usr/bin/env python3
"""Create a review packet for source-balanced residual-dynamics candidates.

This script does not assign manual labels. It ranks source-balanced ROI crops for
human review by combining the best source-normalized residual readout, the
source-residual scalar candidate, crop-local contrast/front proxies, and existing
preview paths. The goal is to make the next manual/QC pass concrete and auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BEST_FEATURE_SET = "dictionary_recon_error_last_minus_first_source_residual"
TARGET = "future_any_drop_within_16cycles"
GROUP_COL = "source_stem"


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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def source_residual(series: pd.Series, sources: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x - x.groupby(sources.astype(str)).transform("mean")


def robust_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad <= 0:
        std = x.std()
        return (x - x.mean()) / std if np.isfinite(std) and std > 0 else x * np.nan
    return 0.6745 * (x - med) / mad


def bounded_score(series: pd.Series) -> pd.Series:
    z = robust_z(series).clip(-4, 4)
    return 1.0 / (1.0 + np.exp(-z))


def load_predictions(readout_dir: Path) -> pd.DataFrame:
    pred = pd.read_csv(readout_dir / "source_balanced_residual_dictionary_normalized_readout_predictions.csv")
    pred = pred[
        (pred["target"] == TARGET)
        & (pred["group_col"] == GROUP_COL)
        & (pred["feature_set"] == BEST_FEATURE_SET)
        & (pred["status"] == "ok")
    ].copy()
    return pred[["roi_id", "predicted_probability", "observed"]].rename(
        columns={"predicted_probability": "source_heldout_future16_probability", "observed": "future16_label"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_residual_dictionary_audit")
    parser.add_argument("--readout-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_residual_dictionary_normalized_readout")
    parser.add_argument("--coupling-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_residual_physics_coupling_audit")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_residual_candidate_review_packet")
    args = parser.parse_args()

    residual_dir = Path(args.residual_dir)
    readout_dir = Path(args.readout_dir)
    coupling_dir = Path(args.coupling_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(residual_dir / "source_balanced_residual_dictionary_features.csv")
    pred = load_predictions(readout_dir)
    rows = features.merge(pred, on="roi_id", how="left")
    sources = rows["source_stem"].astype(str)

    rows["dictionary_recon_error_last_minus_first_source_residual"] = source_residual(
        numeric(rows, "dictionary_recon_error_last_minus_first"), sources
    )
    rows["masked_minus_background_mean_slope_source_residual"] = source_residual(
        numeric(rows, "masked_minus_background_mean_slope"), sources
    )
    rows["front_radius_q80_slope_source_residual"] = source_residual(
        numeric(rows, "front_radius_q80_slope_px_per_norm_time"), sources
    )
    rows["front_radius_q60_slope_source_residual"] = source_residual(
        numeric(rows, "front_radius_q60_slope_px_per_norm_time"), sources
    )
    rows["apparent_diffusion_q70_source_residual"] = source_residual(
        numeric(rows, "apparent_diffusion_q70_um2_per_norm_time"), sources
    )

    rows["prediction_priority_score"] = rows["source_heldout_future16_probability"].rank(pct=True)
    rows["recon_residual_priority_score"] = bounded_score(rows["dictionary_recon_error_last_minus_first_source_residual"])
    rows["contrast_residual_priority_score"] = bounded_score(rows["masked_minus_background_mean_slope_source_residual"])
    rows["front_motion_magnitude_score"] = bounded_score(rows["front_radius_q80_slope_source_residual"].abs())
    rows["diffusion_guardrail_score"] = bounded_score(rows["apparent_diffusion_q70_source_residual"].abs())

    rows["review_priority_score"] = (
        0.38 * rows["prediction_priority_score"].fillna(0.5)
        + 0.28 * rows["recon_residual_priority_score"].fillna(0.5)
        + 0.22 * rows["contrast_residual_priority_score"].fillna(0.5)
        + 0.12 * rows["front_motion_magnitude_score"].fillna(0.5)
    )
    rows["review_tier"] = pd.cut(
        rows["review_priority_score"],
        bins=[-np.inf, 0.55, 0.70, 0.82, np.inf],
        labels=["routine", "standard_review", "high_priority", "immediate_manual_qc"],
    ).astype(str)
    rows["review_reason"] = "source_residual_reconstruction_error;source_heldout_future16_readout;contrast_coupling"
    rows.loc[rows["front_motion_magnitude_score"] > 0.75, "review_reason"] += ";front_motion_check"
    rows.loc[rows["diffusion_guardrail_score"] > 0.75, "review_reason"] += ";diffusion_guardrail"
    rows["manual_qc_status"] = "pending"
    rows["manual_qc_instruction"] = "Review preview/NPZ crop for particle identity, stable mask, phase-front plausibility, and artifact risk; do not treat automatic scores as labels."

    review_cols = [
        "roi_id", "cycleNo", "source_stem", "future16_label", "source_heldout_future16_probability",
        "review_priority_score", "review_tier", "review_reason", "manual_qc_status",
        "dictionary_recon_error_last_minus_first_source_residual", "masked_minus_background_mean_slope_source_residual",
        "front_radius_q80_slope_source_residual", "front_radius_q60_slope_source_residual",
        "apparent_diffusion_q70_source_residual", "preview_png", "npz_path", "manual_qc_instruction",
    ]
    review = rows[review_cols].sort_values("review_priority_score", ascending=False).reset_index(drop=True)
    review.insert(0, "review_rank", np.arange(1, len(review) + 1))

    top_review = review.head(32).copy()
    immediate = review[review["review_tier"] == "immediate_manual_qc"].head(16).copy()
    source_summary = review.groupby("source_stem", as_index=False).agg(
        n_candidates=("roi_id", "count"),
        max_review_priority_score=("review_priority_score", "max"),
        median_review_priority_score=("review_priority_score", "median"),
        future16_positive_rate=("future16_label", "mean"),
    ).sort_values("max_review_priority_score", ascending=False)
    tier_counts = review["review_tier"].value_counts().to_dict()

    aligned = pd.read_csv(coupling_dir / "source_balanced_residual_physics_target_aligned_pairs.csv")
    aligned_top = aligned.head(8).to_dict(orient="records") if not aligned.empty else []

    paths = {
        "review_queue": out / "source_balanced_residual_candidate_review_queue.csv",
        "top_review": out / "source_balanced_residual_candidate_top32.csv",
        "immediate_manual_qc": out / "source_balanced_residual_candidate_immediate_manual_qc.csv",
        "source_summary": out / "source_balanced_residual_candidate_source_summary.csv",
        "summary": out / "source_balanced_residual_candidate_review_summary.json",
    }
    review.to_csv(paths["review_queue"], index=False)
    top_review.to_csv(paths["top_review"], index=False)
    immediate.to_csv(paths["immediate_manual_qc"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)

    summary = {
        "n_candidates": int(len(review)),
        "n_sources": int(review["source_stem"].nunique()),
        "n_cycles": int(review["cycleNo"].nunique()),
        "review_tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "top_review_candidates": top_review.head(12).to_dict(orient="records"),
        "immediate_manual_qc_candidates": immediate.head(12).to_dict(orient="records"),
        "source_summary": source_summary.head(16).to_dict(orient="records"),
        "target_aligned_residual_physics_pairs": aligned_top,
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "This packet ranks automatic source-balanced ROI crops for human review. It keeps manual_qc_status pending and does not assign particle identity, front validity, diffusion validity, or degradation labels.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Source-Balanced Residual Candidate Review Packet",
        "",
        f"Candidates/sources/cycles: {summary['n_candidates']} / {summary['n_sources']} / {summary['n_cycles']}",
        f"Review tiers: {summary['review_tier_counts']}",
        "",
        "## Top Review Candidates",
    ]
    for row in summary["top_review_candidates"][:8]:
        readme.append(
            f"- rank {row.get('review_rank')}: {row.get('roi_id')} score={row.get('review_priority_score'):.3f}, "
            f"prob={row.get('source_heldout_future16_probability'):.3f}, tier={row.get('review_tier')}, cycle={row.get('cycleNo')}"
        )
    readme.extend(["", "## Guardrail", summary["guardrail"], ""])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()
