#!/usr/bin/env python3
"""Build a blinded manual-QC workbook for source-balanced pre-event candidates.

The manual-QC visual packet is still ordered by automatic evidence. This script
creates a stable randomized/blinded review table with image paths and rubric
fields separated from the hidden key containing event bins, sources, cycles, and
automatic action tiers. It never assigns manual labels.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


OUTPUT_DIRNAME = "source_balanced_pre_event_manual_qc_blind_workbook"
DEFAULT_SEED = 20260522

REVIEW_FIELDS = [
    "reviewer_id",
    "review_date",
    "particle_identity_ok",
    "particle_identity_confidence_0_3",
    "front_like_motion_present",
    "front_motion_confidence_0_3",
    "front_direction",
    "phase_boundary_interpretable",
    "kinetic_signal_particle_local",
    "artifact_or_drift_issue",
    "diffusion_interpretation_allowed",
    "overall_manual_decision",
    "review_notes",
]

RUBRIC: Dict[str, str] = {
    "particle_identity_ok": "yes/no/uncertain: crop contains a stable particle-region signal throughout the sequence.",
    "particle_identity_confidence_0_3": "0 none, 1 weak, 2 plausible, 3 strong.",
    "front_like_motion_present": "yes/no/uncertain: strip and radial kymograph show coherent spatial boundary movement.",
    "front_motion_confidence_0_3": "0 none, 1 weak, 2 plausible, 3 strong.",
    "front_direction": "outward/inward/oscillatory/none/uncertain.",
    "phase_boundary_interpretable": "yes/no/uncertain: front-like motion could be discussed as a phase-boundary proxy.",
    "kinetic_signal_particle_local": "yes/no/uncertain: intensity dynamics are localized to the particle mask rather than background/drift.",
    "artifact_or_drift_issue": "none/minor/major/uncertain.",
    "diffusion_interpretation_allowed": "yes/no/uncertain: yes only if particle identity and front trace are strong enough for follow-up calibration.",
    "overall_manual_decision": "accept_front_proxy/reject_artifact/kinetic_only/front_only/context_control/uncertain.",
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
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def local_asset_path(remote_path: Any, packet_dir: Path) -> str:
    if not isinstance(remote_path, str) or not remote_path.strip():
        return ""
    name = Path(remote_path).name
    return str(packet_dir / "visual_assets" / name)


def rel_asset_path(remote_path: Any) -> str:
    if not isinstance(remote_path, str) or not remote_path.strip():
        return ""
    return f"../source_balanced_pre_event_manual_qc_visual_packet/visual_assets/{Path(remote_path).name}"


def build_blinded_tables(assets: pd.DataFrame, seed: int, derived: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    table = assets.copy()
    if "manual_visual_render_status" in table.columns:
        table = table[table["manual_visual_render_status"].eq("ok")].copy()
    table = table.sort_values("manual_qc_rank").reset_index(drop=True)
    perm = rng.permutation(len(table))
    table = table.iloc[perm].reset_index(drop=True)
    table.insert(0, "blind_review_id", [f"SBPRE-BLIND-{i + 1:03d}" for i in range(len(table))])
    table.insert(1, "blind_order", np.arange(1, len(table) + 1))

    visual_packet_dir = derived / "source_balanced_pre_event_manual_qc_visual_packet"
    table["frame_strip_png_remote"] = table["manual_frame_strip_png"].fillna("")
    table["mask_overlay_png_remote"] = table["manual_mask_overlay_png"].fillna("")
    table["kymograph_png_remote"] = table["manual_kymograph_png"].fillna("")
    table["frame_strip_png_relative"] = table["manual_frame_strip_png"].map(rel_asset_path)
    table["mask_overlay_png_relative"] = table["manual_mask_overlay_png"].map(rel_asset_path)
    table["kymograph_png_relative"] = table["manual_kymograph_png"].map(rel_asset_path)
    table["frame_strip_png_local_equivalent"] = table["manual_frame_strip_png"].map(lambda p: local_asset_path(p, visual_packet_dir))
    table["mask_overlay_png_local_equivalent"] = table["manual_mask_overlay_png"].map(lambda p: local_asset_path(p, visual_packet_dir))
    table["kymograph_png_local_equivalent"] = table["manual_kymograph_png"].map(lambda p: local_asset_path(p, visual_packet_dir))

    visible_cols = [
        "blind_review_id",
        "blind_order",
        "frame_strip_png_relative",
        "mask_overlay_png_relative",
        "kymograph_png_relative",
        "frame_strip_png_remote",
        "mask_overlay_png_remote",
        "kymograph_png_remote",
        "frame_strip_png_local_equivalent",
        "mask_overlay_png_local_equivalent",
        "kymograph_png_local_equivalent",
    ]
    workbook = table[visible_cols].copy()
    for field in REVIEW_FIELDS:
        workbook[field] = ""

    key_cols = [
        "blind_review_id",
        "blind_order",
        "manual_qc_rank",
        "roi_id",
        "cycleNo",
        "source_stem",
        "event_relative_bin",
        "cycles_to_next_event",
        "manual_qc_action_tier",
        "manual_qc_decision_score",
        "front_kinetic_tier",
        "evidence_tags",
        "review_question",
        "front_kinetic_concordance_score",
        "kinetic_evidence_score",
        "front_evidence_score",
        "qc_evidence_score",
        "manual_front_review_gate",
        "automatic_diffusion_claim_gate",
        "visual_sanity_score",
        "visual_review_score",
        "strict_qc_priority_score",
        "manual_frame_strip_png",
        "manual_mask_overlay_png",
        "manual_kymograph_png",
    ]
    key = table[[c for c in key_cols if c in table.columns]].copy()
    return workbook, key


def write_html(workbook: pd.DataFrame, out_path: Path) -> None:
    rows: List[str] = []
    for _, row in workbook.iterrows():
        cells = []
        for col, label in [
            ("frame_strip_png_relative", "strip"),
            ("mask_overlay_png_relative", "mask"),
            ("kymograph_png_relative", "kymograph"),
        ]:
            src = row.get(col, "")
            if isinstance(src, str) and src.strip():
                cells.append(f'<td><img src="{html.escape(src)}" alt="{label}" loading="lazy"></td>')
            else:
                cells.append("<td>missing</td>")
        rows.append(
            "<section>"
            f"<h2>{html.escape(str(row['blind_review_id']))}</h2>"
            "<table><tr>" + "".join(cells) + "</tr></table>"
            "<p>Fill labels in the CSV workbook. Keep the hidden key closed until labels are frozen.</p>"
            "</section>"
        )
    rubric_items = "\n".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in RUBRIC.items())
    text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Source-Balanced Pre-Event Blinded Manual QC Workbook</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 1px solid #ccc; padding: 18px 0; }}
table {{ border-collapse: collapse; width: 100%; }}
td {{ width: 33%; vertical-align: top; padding: 6px; border: 1px solid #ddd; }}
img {{ max-width: 100%; height: auto; }}
h1, h2 {{ margin-bottom: 8px; }}
.rubric {{ background: #f7f7f7; padding: 12px 16px; }}
</style>
</head>
<body>
<h1>Source-Balanced Pre-Event Blinded Manual QC Workbook</h1>
<p>Review IDs are randomized and hide event timing, source, cycle, and automatic action tier. Use the CSV for labels.</p>
<div class="rubric"><h2>Rubric</h2><ul>{rubric_items}</ul></div>
{''.join(rows)}
</body>
</html>
"""
    out_path.write_text(text, encoding="utf-8")


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Pre-Event Blinded Manual-QC Workbook",
        "",
        "This packet randomizes the top rendered manual-QC candidates and separates reviewer-visible image paths from the hidden event/source/action key.",
        "",
        "## Counts",
        "",
        f"- Candidate rows: {summary['n_blinded_rows']}",
        f"- Sources in hidden key: {summary['n_sources_hidden_key']}",
        f"- Event bins in hidden key: {summary['event_relative_bin_counts_hidden_key']}",
        f"- Action tiers in hidden key: {summary['action_tier_counts_hidden_key']}",
        "",
        "## Files",
        "",
        "- `source_balanced_pre_event_manual_qc_blinded_workbook.csv`: reviewer-facing label sheet.",
        "- `source_balanced_pre_event_manual_qc_blinded_key.csv`: hidden key with event/source/action metadata.",
        "- `source_balanced_pre_event_manual_qc_blinded_review.html`: image review page using relative paths.",
        "- `source_balanced_pre_event_manual_qc_rubric.json`: explicit label schema.",
        "- `source_balanced_pre_event_manual_qc_blind_summary.json`: machine-readable summary.",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir) if args.out_dir else derived / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)

    assets = read_csv(
        derived
        / "source_balanced_pre_event_manual_qc_visual_packet"
        / "source_balanced_pre_event_manual_qc_visual_assets.csv"
    )
    workbook, key = build_blinded_tables(assets, args.seed, derived)

    workbook_path = out / "source_balanced_pre_event_manual_qc_blinded_workbook.csv"
    key_path = out / "source_balanced_pre_event_manual_qc_blinded_key.csv"
    html_path = out / "source_balanced_pre_event_manual_qc_blinded_review.html"
    rubric_path = out / "source_balanced_pre_event_manual_qc_rubric.json"
    summary_path = out / "source_balanced_pre_event_manual_qc_blind_summary.json"

    workbook.to_csv(workbook_path, index=False)
    key.to_csv(key_path, index=False)
    write_html(workbook, html_path)
    rubric_path.write_text(json.dumps(RUBRIC, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "seed": int(args.seed),
        "n_input_rows": int(len(assets)),
        "n_blinded_rows": int(len(workbook)),
        "n_sources_hidden_key": int(key["source_stem"].nunique()) if "source_stem" in key.columns else 0,
        "event_relative_bin_counts_hidden_key": clean_json(key["event_relative_bin"].value_counts().to_dict()) if "event_relative_bin" in key.columns else {},
        "action_tier_counts_hidden_key": clean_json(key["manual_qc_action_tier"].value_counts().to_dict()) if "manual_qc_action_tier" in key.columns else {},
        "first_blind_ids": clean_json(workbook["blind_review_id"].head(8).tolist()),
        "review_fields": REVIEW_FIELDS,
        "rubric": RUBRIC,
        "outputs": {
            "blinded_workbook_csv": str(workbook_path),
            "hidden_key_csv": str(key_path),
            "review_html": str(html_path),
            "rubric_json": str(rubric_path),
            "summary_json": str(summary_path),
            "readme": str(out / "README.md"),
        },
        "guardrail": (
            "This workbook randomizes and blinds manual review but does not assign labels. "
            "Do not open/use the hidden key until reviewer decisions are frozen; diffusion claims remain blocked unless manual labels support particle identity, front mask quality, and front-motion interpretability."
        ),
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out, clean_json(summary))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
