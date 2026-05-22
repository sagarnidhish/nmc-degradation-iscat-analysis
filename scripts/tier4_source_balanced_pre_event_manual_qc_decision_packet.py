#!/usr/bin/env python3
"""Build a manual-QC decision packet for source-balanced pre-event ROIs.

This packet consolidates the source-balanced front/kinetic concordance audit,
source-aware null tests, strict front gates, and rendered visual assets into an
operational queue for manual review. It does not assign manual labels and does
not promote any automatic diffusion claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


OUTPUT_DIRNAME = "source_balanced_pre_event_manual_qc_decision_packet"


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


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def minmax_score(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if not x.notna().any():
        return pd.Series(0.0, index=x.index)
    lo = float(x.min())
    hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.0, index=x.index)
    return ((x - lo) / (hi - lo)).fillna(0.0).clip(0, 1)


def first_existing(df: pd.DataFrame, cols: Iterable[str], default: Any = "") -> pd.Series:
    out = pd.Series(default, index=df.index, dtype=object)
    for col in cols:
        if col in df.columns:
            value = df[col]
            out = out.where(out.astype(str).ne(str(default)) & out.notna(), value)
    return out


def event_priority(bin_name: Any) -> float:
    mapping = {
        "near_pre_event_1_8": 1.0,
        "mid_pre_event_9_16": 0.7,
        "far_pre_event_17_32": 0.45,
        "post_event_1_16": 0.2,
        "no_near_event_control": 0.1,
    }
    return mapping.get(str(bin_name), 0.25)


def null_support_table(null_tests: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    support: Dict[str, Dict[str, Any]] = {}
    if null_tests.empty:
        return support
    tests = null_tests.copy()
    tests["p"] = numeric(tests, "source_stratified_perm_p_auc_ge_observed")
    tests = tests[(tests.get("transform", "") == "raw") & tests["p"].notna()]
    for feature, group in tests.groupby("feature"):
        best = group.sort_values(["p", "oriented_auc"], ascending=[True, False]).iloc[0].to_dict()
        support[str(feature)] = clean_json(best)
    return support


def support_tags(row: pd.Series, null_support: Dict[str, Dict[str, Any]]) -> str:
    tags: List[str] = []
    if "masked_minus_bg_slope" in null_support and row.get("kinetic_evidence_score", 0) >= 0.5:
        tags.append("kinetic_source_null_support_directional")
    if "kinetic_evidence_score" in null_support and row.get("kinetic_evidence_score", 0) >= 0.5:
        tags.append("kinetic_composite_null_support_directional")
    if "front_kinetic_concordance_score" in null_support and row.get("front_kinetic_concordance_score", 0) >= 0.5:
        tags.append("concordance_null_support_directional")
    if boolish(row.get("manual_front_review_gate", False)):
        tags.append("strict_manual_front_gate_pass")
    if boolish(row.get("automatic_diffusion_claim_gate", False)):
        tags.append("automatic_diffusion_claim_gate_pass")
    else:
        tags.append("diffusion_claim_blocked")
    if str(row.get("visual_sanity_flag", "")).lower() == "uncertain_auto":
        tags.append("visual_sanity_uncertain")
    if str(row.get("visual_qc_tier", "")).lower() == "front_plausible_followup":
        tags.append("front_plausible_followup")
    if row.get("source_fraction_in_top40", 0) >= 0.25:
        tags.append("source_concentration_watch")
    return ";".join(tags)


def action_tier(row: pd.Series) -> str:
    fk_tier = str(row.get("front_kinetic_tier", ""))
    event_bin = str(row.get("event_relative_bin", ""))
    rendered = boolish(row.get("has_visual_assets", False))
    manual_gate = boolish(row.get("manual_front_review_gate", False))
    high_concordance = row.get("front_kinetic_concordance_score", 0) >= 0.9
    kinetic = row.get("kinetic_evidence_score", 0)
    front = row.get("front_evidence_score", 0)

    if manual_gate:
        return "review_strict_front_gate"
    if event_bin == "near_pre_event_1_8" and fk_tier == "near_pre_front_kinetic_review":
        return "review_front_and_kinetics_first"
    if event_bin == "near_pre_event_1_8" and high_concordance and rendered:
        return "review_front_and_kinetics_first"
    if fk_tier == "kinetic_only_guardrail" or (kinetic >= 0.9 and front < 0.4):
        return "review_kinetic_only_guardrail"
    if fk_tier == "front_only_guardrail" or (front >= 0.9 and kinetic < 0.4):
        return "review_front_only_guardrail"
    if fk_tier == "front_kinetic_concordant" and event_bin != "near_pre_event_1_8":
        return "context_control_concordant"
    return "routine_or_low_concordance"


def review_question(row: pd.Series) -> str:
    tier = row.get("manual_qc_action_tier", "")
    if tier == "review_strict_front_gate":
        return "Can a human confirm a coherent outward/front-like trace in the strip, overlay, and radial kymograph?"
    if tier == "review_front_and_kinetics_first":
        return "Do masked optical kinetics and the radial/front view describe the same localized pre-event change?"
    if tier == "review_kinetic_only_guardrail":
        return "Is the kinetic signal particle-local and non-artifactual despite weak front evidence?"
    if tier == "review_front_only_guardrail":
        return "Is the front-like morphology real, or driven by mask/crop/illumination artifacts without kinetic support?"
    if tier == "context_control_concordant":
        return "Does this non-near concordant example represent baseline morphology that should temper pre-event claims?"
    return "No immediate manual decision required unless needed as a low-priority comparator."


def compact_records(df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
    cols = [
        "manual_qc_rank",
        "roi_id",
        "cycleNo",
        "source_stem",
        "event_relative_bin",
        "cycles_to_next_event",
        "manual_qc_action_tier",
        "manual_qc_decision_score",
        "front_kinetic_concordance_score",
        "kinetic_evidence_score",
        "front_evidence_score",
        "qc_evidence_score",
        "consensus_review_score",
        "strict_qc_priority_score",
        "visual_sanity_score",
        "visual_review_score",
        "front_kinetic_tier",
        "manual_front_review_gate",
        "automatic_diffusion_claim_gate",
        "evidence_tags",
        "review_question",
        "frame_strip_png",
        "mask_overlay_png",
        "kymograph_png",
    ]
    have = [c for c in cols if c in df.columns]
    return clean_json(df.head(n)[have].to_dict("records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir) if args.out_dir else derived / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)

    concordance = read_csv(
        derived
        / "source_balanced_pre_event_front_kinetic_concordance_audit"
        / "source_balanced_pre_event_front_kinetic_concordance_ranked_candidates.csv"
    )
    null_tests = read_csv(
        derived
        / "source_balanced_pre_event_front_kinetic_null_audit"
        / "source_balanced_pre_event_front_kinetic_null_tests.csv"
    )
    proximity = read_csv(
        derived
        / "source_balanced_pre_event_front_kinetic_null_audit"
        / "source_balanced_pre_event_front_kinetic_proximity_tests.csv",
        required=False,
    )
    visual = read_csv(
        derived
        / "source_balanced_pre_event_consensus_visual_packet"
        / "source_balanced_pre_event_consensus_visual_assets.csv",
        required=False,
    )
    strict = read_csv(
        derived
        / "source_balanced_pre_event_strict_qc_gated_front_audit"
        / "source_balanced_pre_event_strict_qc_gated_front_candidates.csv",
        required=False,
    )

    df = concordance.copy()
    if not visual.empty:
        visual_cols = [
            "roi_id",
            "render_status",
            "frame_strip_png",
            "mask_overlay_png",
            "kymograph_png",
            "n_frames_loaded",
            "frame_height",
            "frame_width",
        ]
        df = df.merge(visual[[c for c in visual_cols if c in visual.columns]], on="roi_id", how="left")
    else:
        for col in ["render_status", "frame_strip_png", "mask_overlay_png", "kymograph_png"]:
            df[col] = ""

    if not strict.empty:
        strict_cols = [
            "roi_id",
            "front_trace_r2_mode",
            "front_direction_agreement_gate",
            "stable_mask_gate",
            "centroid_stability_gate",
        ]
        add = [c for c in strict_cols if c in strict.columns and c not in df.columns]
        if add:
            df = df.merge(strict[["roi_id"] + add], on="roi_id", how="left")

    top40_sources = df.head(args.top_n)["source_stem"].astype(str).value_counts(normalize=True).to_dict()
    df["source_fraction_in_top40"] = df["source_stem"].astype(str).map(top40_sources).fillna(0.0)
    frame_strip = first_existing(df, ["frame_strip_png"], "")
    df["has_visual_assets"] = (
        frame_strip.notna()
        & frame_strip.astype(str).str.strip().str.len().gt(0)
        & ~frame_strip.astype(str).str.strip().str.lower().isin({"nan", "none", "null"})
    )
    df["event_priority"] = df["event_relative_bin"].map(event_priority)

    score = (
        0.32 * minmax_score(numeric(df, "front_kinetic_concordance_score"))
        + 0.18 * minmax_score(numeric(df, "consensus_review_score"))
        + 0.16 * minmax_score(numeric(df, "strict_qc_priority_score", 0))
        + 0.12 * minmax_score(numeric(df, "kinetic_evidence_score"))
        + 0.10 * minmax_score(numeric(df, "front_evidence_score"))
        + 0.07 * df["event_priority"]
        + 0.05 * df["has_visual_assets"].astype(float)
    )
    df["manual_qc_decision_score"] = score

    null_support = null_support_table(null_tests)
    df["manual_qc_action_tier"] = df.apply(action_tier, axis=1)
    action_bonus = {
        "review_strict_front_gate": 0.35,
        "review_front_and_kinetics_first": 0.25,
        "review_kinetic_only_guardrail": 0.12,
        "review_front_only_guardrail": 0.10,
        "context_control_concordant": 0.05,
        "routine_or_low_concordance": 0.0,
    }
    df["manual_qc_decision_score"] = df["manual_qc_decision_score"] + df["manual_qc_action_tier"].map(action_bonus).fillna(0)
    df["evidence_tags"] = df.apply(lambda r: support_tags(r, null_support), axis=1)
    df["review_question"] = df.apply(review_question, axis=1)
    df = df.sort_values(
        ["manual_qc_decision_score", "front_kinetic_concordance_score", "consensus_review_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "manual_qc_rank", np.arange(1, len(df) + 1))

    action_counts = df["manual_qc_action_tier"].value_counts().to_dict()
    source_summary = (
        df.groupby("source_stem", dropna=False)
        .agg(
            n_rows=("roi_id", "count"),
            n_top40=("manual_qc_rank", lambda s: int((s <= args.top_n).sum())),
            best_rank=("manual_qc_rank", "min"),
            mean_decision_score=("manual_qc_decision_score", "mean"),
            n_visual_assets=("has_visual_assets", "sum"),
        )
        .reset_index()
        .sort_values(["best_rank", "n_top40"], ascending=[True, False])
    )
    tier_summary = (
        df.groupby("manual_qc_action_tier", dropna=False)
        .agg(
            n_rows=("roi_id", "count"),
            n_top40=("manual_qc_rank", lambda s: int((s <= args.top_n).sum())),
            median_score=("manual_qc_decision_score", "median"),
            n_visual_assets=("has_visual_assets", "sum"),
            n_manual_front_gate=("manual_front_review_gate", lambda s: int(pd.Series(s).map(boolish).sum())),
            n_diffusion_gate=("automatic_diffusion_claim_gate", lambda s: int(pd.Series(s).map(boolish).sum())),
        )
        .reset_index()
        .sort_values(["n_top40", "median_score"], ascending=[False, False])
    )
    asset_manifest = df[df["has_visual_assets"]].copy()
    asset_cols = [
        "manual_qc_rank",
        "roi_id",
        "manual_qc_action_tier",
        "event_relative_bin",
        "source_stem",
        "frame_strip_png",
        "mask_overlay_png",
        "kymograph_png",
        "review_question",
    ]

    queue_path = out / "source_balanced_pre_event_manual_qc_decision_queue.csv"
    top_path = out / f"source_balanced_pre_event_manual_qc_top{args.top_n}.csv"
    source_path = out / "source_balanced_pre_event_manual_qc_source_summary.csv"
    tier_path = out / "source_balanced_pre_event_manual_qc_action_summary.csv"
    asset_path = out / "source_balanced_pre_event_manual_qc_visual_asset_manifest.csv"
    summary_path = out / "source_balanced_pre_event_manual_qc_decision_summary.json"
    readme_path = out / "README.md"

    df.to_csv(queue_path, index=False)
    df.head(args.top_n).to_csv(top_path, index=False)
    source_summary.to_csv(source_path, index=False)
    tier_summary.to_csv(tier_path, index=False)
    asset_manifest[[c for c in asset_cols if c in asset_manifest.columns]].to_csv(asset_path, index=False)

    top_null = null_tests.head(8).to_dict("records") if not null_tests.empty else []
    top_proximity = proximity.head(5).to_dict("records") if not proximity.empty else []
    summary = {
        "n_rows": int(len(df)),
        "n_sources": int(df["source_stem"].nunique()),
        "n_visual_asset_rows": int(df["has_visual_assets"].sum()),
        "top_n": int(args.top_n),
        "action_counts": clean_json(action_counts),
        "top40_action_counts": clean_json(df.head(args.top_n)["manual_qc_action_tier"].value_counts().to_dict()),
        "top40_source_counts": clean_json(df.head(args.top_n)["source_stem"].astype(str).value_counts().to_dict()),
        "n_manual_front_review_gate": int(df["manual_front_review_gate"].map(boolish).sum()) if "manual_front_review_gate" in df.columns else 0,
        "n_automatic_diffusion_claim_gate": int(df["automatic_diffusion_claim_gate"].map(boolish).sum()) if "automatic_diffusion_claim_gate" in df.columns else 0,
        "top_candidates": compact_records(df, 20),
        "top_null_tests": clean_json(top_null),
        "top_proximity_tests": clean_json(top_proximity),
        "null_support_by_feature": clean_json(null_support),
        "outputs": {
            "decision_queue_csv": str(queue_path),
            "top_review_csv": str(top_path),
            "source_summary_csv": str(source_path),
            "action_summary_csv": str(tier_path),
            "visual_asset_manifest_csv": str(asset_path),
            "summary_json": str(summary_path),
            "readme": str(readme_path),
        },
        "guardrail": (
            "Manual-QC decision packet only prioritizes review. It carries source-null and visual-QC evidence forward, "
            "but it does not create manual labels, validate fronts, calibrate diffusion, or make source-invariant causal claims."
        ),
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")

    lines = [
        "# Source-Balanced Pre-Event Manual-QC Decision Packet",
        "",
        "This packet merges front/kinetic concordance, source-stratified null evidence, strict front gates, and rendered visual assets into a manual-review queue.",
        "",
        "## Key Counts",
        "",
        f"- ROI rows: {len(df)}",
        f"- Sources: {df['source_stem'].nunique()}",
        f"- Rows with rendered visual assets: {int(df['has_visual_assets'].sum())}",
        f"- Manual front-review gate rows: {summary['n_manual_front_review_gate']}",
        f"- Automatic diffusion-claim gate rows: {summary['n_automatic_diffusion_claim_gate']}",
        f"- Action counts: {action_counts}",
        "",
        "## Outputs",
        "",
        f"- Decision queue: `{queue_path.name}`",
        f"- Top review queue: `{top_path.name}`",
        f"- Visual asset manifest: `{asset_path.name}`",
        f"- Source summary: `{source_path.name}`",
        f"- Action summary: `{tier_path.name}`",
        f"- JSON summary: `{summary_path.name}`",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    readme_path.write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
