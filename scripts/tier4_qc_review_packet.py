#!/usr/bin/env python3
"""Build a review-ready QC packet for top NMC ROI/front candidates."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def norm_roi_id(text: str) -> str:
    return str(text).replace("_front", "_rank")


def find_first(paths: List[Path], roi_id: str) -> str:
    for path in paths:
        if roi_id in path.name:
            return str(path)
    return ""


def front_style_id(roi_id: str) -> str:
    # cycle116_rank1_obj6 -> cycle116_front1_obj6
    parts = roi_id.split("_")
    if len(parts) < 3:
        return roi_id
    return "_".join([parts[0], parts[1].replace("rank", "front"), parts[2]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/qc_review_packet")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mobility = pd.read_csv(derived / "multi_cycle_rollout_mobility_coupling" / "multi_cycle_rollout_mobility_ranked.csv")
    fronts = pd.read_csv(derived / "multi_cycle_threshold_robust_fronts" / "threshold_robust_front_summary.csv")
    echem = pd.read_csv(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_joined.csv")
    conditioned = pd.read_csv(derived / "protocol_conditioned_roi_effects" / "protocol_conditioned_roi_residuals.csv")
    front_conditioned = pd.read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")

    table = mobility.merge(fronts, on=["roi_id", "cycleNo", "cohort_role", "event_reference_cycle"], how="outer", suffixes=("", "_front"))
    table = table.merge(echem[["roi_id", "n_frames_percentile", "V_mean", "I_mean_mA"]], on="roi_id", how="left")
    keep_cond = [c for c in conditioned.columns if c == "roi_id" or c.endswith("_protocol_residual")]
    table = table.merge(conditioned[keep_cond], on="roi_id", how="left")
    keep_front_cond = [c for c in front_conditioned.columns if c == "roi_id" or c.endswith("_protocol_residual")]
    table = table.merge(front_conditioned[keep_front_cond], on="roi_id", how="left", suffixes=("", "_front_residual"))

    score_cols = {
        "rollout_mobility_difficulty_score": 1.0,
        "threshold_robust_phase_score": 1.0,
        "threshold_robust_diffusion_score": 0.5,
        "latent_net_displacement": 0.75,
        "cumulative_abs_first_last": 0.75,
        "phase_slope_positive_fraction_protocol_residual": 1.0,
        "roi_norm_mean_delta_protocol_residual": 0.75,
        "high_fraction_delta_protocol_residual": 0.75,
    }
    table["qc_priority_score"] = 0.0
    for col, weight in score_cols.items():
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce").abs()
        if values.notna().sum() < 2:
            continue
        pct = values.rank(pct=True)
        table["qc_priority_score"] += weight * pct.fillna(0.0)

    table["qc_reason"] = ""
    table.loc[table.get("cohort_role", "") == "event", "qc_reason"] += "event_roi;"
    for col, label in [
        ("rollout_mobility_difficulty_score", "hard_rollout"),
        ("threshold_robust_phase_score", "phase_front"),
        ("threshold_robust_diffusion_score", "diffusion_proxy"),
        ("phase_slope_positive_fraction_protocol_residual", "conditioned_front_direction"),
        ("roi_norm_mean_delta_protocol_residual", "conditioned_optical_shift"),
    ]:
        if col in table.columns:
            cutoff = pd.to_numeric(table[col], errors="coerce").abs().quantile(0.8)
            table.loc[pd.to_numeric(table[col], errors="coerce").abs() >= cutoff, "qc_reason"] += f"{label};"

    remote_preview_roots = [
        derived / "multi_cycle_roi_sequences" / "previews",
        derived / "selected_roi_sequences" / "previews",
        derived / "control_roi_sequences" / "previews",
        derived / "selected_front_roi_tracking" / "crop_previews",
        derived / "multi_cycle_roi_rollout_baselines" / "rollout_previews",
        derived / "selected_front_roi_tracking" / "plots",
    ]
    all_files = []
    for root in remote_preview_roots:
        if root.exists():
            all_files.extend([p for p in root.rglob("*") if p.is_file()])

    preview_files = [p for p in all_files if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    records = []
    for _, row in table.sort_values("qc_priority_score", ascending=False).head(args.top_n).iterrows():
        roi_id = str(row["roi_id"])
        front_id = front_style_id(roi_id)
        rec = row.to_dict()
        rec["roi_preview_path"] = find_first(preview_files, roi_id)
        rec["front_crop_preview_path"] = find_first(preview_files, front_id + "_crop_preview")
        rec["front_tracking_plot_path"] = find_first(preview_files, front_id + "_front_tracking")
        rec["rollout_preview_path"] = find_first(preview_files, roi_id + "_dmd_rollout")
        rec["manual_qc_status"] = "pending"
        rec["manual_qc_decision"] = ""
        rec["manual_qc_notes"] = ""
        records.append(rec)

    review = pd.DataFrame(records)
    selected_cols = [
        "roi_id",
        "cycleNo",
        "cohort_role",
        "event_reference_cycle",
        "qc_priority_score",
        "qc_reason",
        "manual_qc_status",
        "manual_qc_decision",
        "manual_qc_notes",
        "rollout_mobility_difficulty_score",
        "threshold_robust_phase_score",
        "threshold_robust_diffusion_score",
        "phase_slope_median_per_s",
        "diffusion_proxy_median_um2_per_s",
        "phase_slope_positive_fraction_protocol_residual",
        "roi_norm_mean_delta_protocol_residual",
        "high_fraction_delta_protocol_residual",
        "first_last_corr",
        "latent_net_displacement",
        "n_frames_percentile",
        "roi_preview_path",
        "front_crop_preview_path",
        "front_tracking_plot_path",
        "rollout_preview_path",
    ]
    review = review[[c for c in selected_cols if c in review.columns]]
    review_path = out / "qc_review_manifest.csv"
    review.to_csv(review_path, index=False)

    summary = {
        "n_candidates": int(len(review)),
        "n_event_candidates": int((review["cohort_role"] == "event").sum()) if "cohort_role" in review else 0,
        "n_control_candidates": int((review["cohort_role"] == "control").sum()) if "cohort_role" in review else 0,
        "top_candidates": review.head(12).to_dict("records"),
        "guardrail": "This is a review packet, not manual QC. Use manual_qc_decision/status columns to record accepted/rejected particle/front masks before publication-scale diffusion claims.",
        "outputs": {"manifest": str(review_path), "report": str(out / "QC_REVIEW_PACKET.md")},
    }
    with (out / "qc_review_packet_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    lines = [
        "# QC Review Packet",
        "",
        "Prioritized ROI/front candidates for manual review before treating phase-front or diffusion proxies as physical estimates.",
        "",
        f"- Candidates: {len(review)}",
        f"- Event candidates: {summary['n_event_candidates']}",
        f"- Control candidates: {summary['n_control_candidates']}",
        "",
        "## Top Candidates",
        "",
    ]
    for _, row in review.head(15).iterrows():
        lines.append(
            f"- {row['roi_id']} ({row['cohort_role']}, cycle {row['cycleNo']}): score {row['qc_priority_score']:.3f}; {row['qc_reason']}"
        )
    lines += [
        "",
        "## Review Instructions",
        "",
        "For each candidate, inspect available ROI preview, front crop, tracking plot, and rollout preview paths. Set `manual_qc_decision` to `accept`, `reject`, or `uncertain`, and record particle/front-mask concerns in `manual_qc_notes`.",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    (out / "QC_REVIEW_PACKET.md").write_text("\n".join(lines) + "\n")
    (out / "README.md").write_text("# QC Review Packet\\n\\nSee `QC_REVIEW_PACKET.md` and `qc_review_manifest.csv`.\\n")


if __name__ == "__main__":
    main()
