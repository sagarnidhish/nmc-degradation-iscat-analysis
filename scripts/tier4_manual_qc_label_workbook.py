#!/usr/bin/env python3
"""Build a single manual-QC label workbook for NMC ROI/front review.

This consolidates the primary visual front-QC package, the control-balanced
augmentation, and the ranked QC review packet into one deduplicated table. The
output is a label template: it preserves existing manual labels if present, but
never invents accept/reject decisions.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SOURCE_SPECS = [
    ("primary_visual_qc", "roi_front_qc_package", "roi_front_qc_manifest.csv", "roi_front_qc_index.html"),
    ("control_balanced_visual_qc", "control_balanced_front_qc_package", "control_balanced_front_qc_manifest.csv", "control_balanced_front_qc_index.html"),
    ("ranked_review_packet", "qc_review_packet", "qc_review_manifest.csv", "QC_REVIEW_PACKET.md"),
]

BASE_COLUMNS = [
    "roi_id",
    "cohort_role",
    "cycleNo",
    "event_reference_cycle",
    "review_sources",
    "review_priority_tier",
    "combined_review_priority_score",
    "recommended_review_reason",
    "manual_qc_status",
    "manual_qc_decision",
    "manual_qc_notes",
    "manual_particle_identity_ok",
    "manual_front_mask_ok",
    "manual_diffusion_interpretable",
    "manual_reviewer",
    "manual_review_date",
]

METRIC_COLUMNS = [
    "auto_review_flags",
    "mode_label",
    "degradation_mode_hypothesis",
    "qc_priority_score",
    "control_balance_priority_score",
    "mode_review_priority",
    "front_quality_score",
    "rollout_mobility_difficulty_score",
    "threshold_robust_phase_score",
    "phase_slope_positive_fraction_protocol_residual",
    "diffusion_proxy_abs_median_um2_per_s",
    "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    "q70_phase_ci_positive",
    "q70_radius_ci_crosses_zero",
    "n_components",
    "largest_component_fraction",
    "edge_touch_fraction",
    "q70_fraction_delta",
    "n_frames_percentile",
    "V_mean",
    "I_mean_mA",
]

PATH_COLUMNS = [
    "primary_qc_png",
    "control_balanced_qc_png",
    "roi_preview_path",
    "front_crop_preview_path",
    "front_tracking_plot_path",
    "rollout_preview_path",
]


def clean_json(value: Any) -> Any:
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


def read_source(derived: Path, source_name: str, folder: str, manifest: str, index_name: str) -> pd.DataFrame:
    path = derived / folder / manifest
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "roi_id" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["review_source"] = source_name
    df["source_manifest_path"] = str(path)
    df["source_index_path"] = str(derived / folder / index_name)
    if "qc_png" in df.columns:
        out_col = "primary_qc_png" if source_name == "primary_visual_qc" else "control_balanced_qc_png"
        df[out_col] = df["qc_png"]
    if "manual_qc_status" not in df.columns:
        df["manual_qc_status"] = "pending"
    if "manual_qc_decision" not in df.columns:
        df["manual_qc_decision"] = ""
    if "manual_qc_notes" not in df.columns:
        df["manual_qc_notes"] = ""
    return df


def finite_number(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def first_nonempty(values: pd.Series) -> Any:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return value
        elif not pd.isna(value):
            return value
    return ""


def combine_strings(values: pd.Series, sep: str = ";") -> str:
    parts: List[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        for part in value.split(sep):
            part = part.strip()
            if part and part not in parts:
                parts.append(part)
    return sep.join(parts)


def priority_reason(row: pd.Series) -> str:
    reasons: List[str] = []
    flags = str(row.get("auto_review_flags", ""))
    if row.get("cohort_role") == "event":
        reasons.append("event_roi")
    else:
        reasons.append("control_roi")
    if "primary_visual_qc" in str(row.get("review_sources", "")):
        reasons.append("primary_visual_panel")
    if "control_balanced_visual_qc" in str(row.get("review_sources", "")):
        reasons.append("control_balanced_panel")
    if "none" in flags.split(";"):
        reasons.append("no_auto_flags")
    if "fragmented_q70_mask" in flags:
        reasons.append("fragmentation_check")
    if "diffusion_ci_crosses_zero" in flags:
        reasons.append("diffusion_ci_check")
    if finite_number(row.get("mode_review_priority")) >= 3:
        reasons.append("high_residual_mode_priority")
    if abs(finite_number(row.get("phase_slope_positive_fraction_protocol_residual"))) >= 0.08:
        reasons.append("large_conditioned_front_sign_residual")
    if finite_number(row.get("rollout_mobility_difficulty_score")) >= 2.5:
        reasons.append("hard_rollout_mobility")
    return ";".join(dict.fromkeys(reasons))


def combine_sources(frames: List[pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat([f for f in frames if not f.empty], ignore_index=True, sort=False)
    if all_df.empty:
        return pd.DataFrame(columns=BASE_COLUMNS + METRIC_COLUMNS + PATH_COLUMNS)

    for col in ["qc_priority_score", "control_balance_priority_score", "mode_review_priority", "front_quality_score", "rollout_mobility_difficulty_score", "threshold_robust_phase_score", "phase_slope_positive_fraction_protocol_residual"]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    rows: List[Dict[str, Any]] = []
    for roi_id, group in all_df.groupby("roi_id", sort=False):
        row: Dict[str, Any] = {"roi_id": roi_id}
        row["review_sources"] = combine_strings(group["review_source"])
        row["cohort_role"] = first_nonempty(group.get("cohort_role", pd.Series(dtype=object)))
        row["cycleNo"] = first_nonempty(group.get("cycleNo", pd.Series(dtype=object)))
        row["event_reference_cycle"] = first_nonempty(group.get("event_reference_cycle", pd.Series(dtype=object)))
        for col in METRIC_COLUMNS + PATH_COLUMNS:
            if col in group.columns:
                if col == "auto_review_flags":
                    row[col] = combine_strings(group[col])
                elif col.endswith("path") or col.endswith("png") or col in {"mode_label", "degradation_mode_hypothesis"}:
                    row[col] = first_nonempty(group[col])
                else:
                    values = pd.to_numeric(group[col], errors="coerce")
                    row[col] = float(values.max(skipna=True)) if values.notna().any() else first_nonempty(group[col])
        score_terms = []
        source_bonus = 0.5 * len(str(row["review_sources"]).split(";"))
        for col, weight in [
            ("qc_priority_score", 1.0),
            ("control_balance_priority_score", 0.7),
            ("mode_review_priority", 0.8),
            ("front_quality_score", 0.5),
            ("rollout_mobility_difficulty_score", 0.6),
            ("threshold_robust_phase_score", 0.5),
            ("phase_slope_positive_fraction_protocol_residual", 1.0),
        ]:
            score_terms.append(weight * abs(finite_number(row.get(col))))
        row["combined_review_priority_score"] = float(source_bonus + sum(score_terms))
        row["manual_qc_status"] = first_nonempty(group.get("manual_qc_status", pd.Series(dtype=object))) or "pending"
        row["manual_qc_decision"] = first_nonempty(group.get("manual_qc_decision", pd.Series(dtype=object)))
        row["manual_qc_notes"] = first_nonempty(group.get("manual_qc_notes", pd.Series(dtype=object)))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["recommended_review_reason"] = out.apply(priority_reason, axis=1)
    out = out.sort_values("combined_review_priority_score", ascending=False).reset_index(drop=True)
    n = len(out)
    if n:
        high_cut = max(1, int(np.ceil(0.25 * n)))
        med_cut = max(high_cut + 1, int(np.ceil(0.60 * n)))
        tiers = []
        for idx in range(n):
            if idx < high_cut:
                tiers.append("high")
            elif idx < med_cut:
                tiers.append("medium")
            else:
                tiers.append("routine")
        out["review_priority_tier"] = tiers
    for col in ["manual_particle_identity_ok", "manual_front_mask_ok", "manual_diffusion_interpretable", "manual_reviewer", "manual_review_date"]:
        if col not in out.columns:
            out[col] = ""
    keep = [c for c in BASE_COLUMNS + METRIC_COLUMNS + PATH_COLUMNS if c in out.columns]
    return out[keep]


def write_html(path: Path, label_table: pd.DataFrame) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>NMC Manual QC Label Workbook</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} table{border-collapse:collapse;width:100%;margin-bottom:28px;} th,td{border:1px solid #bbb;padding:4px;font-size:12px;vertical-align:top;} img{max-width:440px;} .high{background:#fff4d6;} .medium{background:#eef6ff;} .routine{background:#f7f7f7;} code{font-size:11px;}</style>",
        "</head><body>",
        "<h1>NMC Manual QC Label Workbook</h1>",
        "<p>Deduplicated label template merging primary visual QC, control-balanced visual QC, and ranked QC packet sources. Fill the manual columns in the CSV; this page is for inspection only.</p>",
        "<h2>Review Columns</h2>",
        "<ul><li><code>manual_qc_decision</code>: accept, reject, or uncertain.</li><li><code>manual_particle_identity_ok</code>: yes/no/uncertain.</li><li><code>manual_front_mask_ok</code>: yes/no/uncertain.</li><li><code>manual_diffusion_interpretable</code>: yes/no/uncertain; use yes only for fronts suitable for apparent transport analysis.</li></ul>",
    ]
    for tier, group in label_table.groupby("review_priority_tier", sort=False):
        lines.append(f"<h2>{html.escape(str(tier)).title()} Priority ({len(group)})</h2>")
        lines.append("<table><tr><th>ROI</th><th>Role</th><th>Cycle</th><th>Sources</th><th>Reason</th><th>Flags</th><th>Primary Panel</th><th>Control-Balanced Panel</th></tr>")
        for _, row in group.iterrows():
            primary = str(row.get("primary_qc_png", ""))
            balanced = str(row.get("control_balanced_qc_png", ""))
            primary_img = f"<img src='{html.escape(primary)}'>" if primary else ""
            balanced_img = f"<img src='{html.escape(balanced)}'>" if balanced else ""
            lines.append(
                f"<tr class='{html.escape(str(tier))}'>"
                f"<td><code>{html.escape(str(row.get('roi_id', '')))}</code></td>"
                f"<td>{html.escape(str(row.get('cohort_role', '')))}</td>"
                f"<td>{html.escape(str(row.get('cycleNo', '')))}</td>"
                f"<td>{html.escape(str(row.get('review_sources', '')))}</td>"
                f"<td>{html.escape(str(row.get('recommended_review_reason', '')))}</td>"
                f"<td>{html.escape(str(row.get('auto_review_flags', '')))}</td>"
                f"<td>{primary_img}</td>"
                f"<td>{balanced_img}</td>"
                "</tr>"
            )
        lines.append("</table>")
    lines += ["</body></html>"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/manual_qc_label_workbook")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = [read_source(derived, *spec) for spec in SOURCE_SPECS]
    label_table = combine_sources(frames)
    label_path = out / "manual_qc_label_template.csv"
    label_table.to_csv(label_path, index=False)

    review_state = label_table.copy()
    if "manual_qc_status" in review_state.columns:
        status_counts = review_state["manual_qc_status"].fillna("missing").replace("", "missing").value_counts().to_dict()
    else:
        status_counts = {}
    role_counts = label_table.get("cohort_role", pd.Series(dtype=object)).value_counts().to_dict()
    tier_counts = label_table.get("review_priority_tier", pd.Series(dtype=object)).value_counts().to_dict()
    source_counts: Dict[str, int] = {}
    for sources in label_table.get("review_sources", pd.Series(dtype=object)).fillna(""):
        for source in str(sources).split(";"):
            if source:
                source_counts[source] = source_counts.get(source, 0) + 1

    html_path = out / "manual_qc_label_workbook.html"
    write_html(html_path, label_table)

    summary = {
        "n_unique_roi": int(len(label_table)),
        "role_counts": clean_json(role_counts),
        "review_priority_tier_counts": clean_json(tier_counts),
        "manual_qc_status_counts": clean_json(status_counts),
        "source_counts": clean_json(source_counts),
        "n_with_primary_qc_panel": int(label_table.get("primary_qc_png", pd.Series(dtype=object)).astype(str).str.len().gt(0).sum()) if not label_table.empty else 0,
        "n_with_control_balanced_qc_panel": int(label_table.get("control_balanced_qc_png", pd.Series(dtype=object)).astype(str).str.len().gt(0).sum()) if not label_table.empty else 0,
        "top_review_rows": clean_json(label_table.head(12).to_dict("records")),
        "guardrail": "This workbook is a manual-label template. It deduplicates review candidates and preserves pending status, but it does not assign accept/reject labels or validate diffusion.",
        "outputs": {
            "label_template": str(label_path),
            "html_workbook": str(html_path),
            "summary": str(out / "manual_qc_label_workbook_summary.json"),
        },
    }
    with (out / "manual_qc_label_workbook_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    readme = [
        "# Manual QC Label Workbook",
        "",
        "Deduplicated label template for NMC ROI/front manual review.",
        "",
        f"- Unique ROI candidates: {summary['n_unique_roi']}",
        f"- Role counts: {summary['role_counts']}",
        f"- Priority tiers: {summary['review_priority_tier_counts']}",
        "",
        "Fill `manual_qc_label_template.csv`; do not treat this package as completed QC labels.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(json.dumps({"out_dir": str(out), "n_unique_roi": summary["n_unique_roi"], "role_counts": summary["role_counts"]}, indent=2))


if __name__ == "__main__":
    main()
