#!/usr/bin/env python3
"""Render visual assets for the source-balanced manual-QC decision queue.

The decision packet ranks strict-front, front/kinetic, front-only, kinetic-only,
and non-near concordant control rows. This script renders the top queue rows
directly from their particle-region crop tensors so every review tier has a
consistent strip, mask overlay, radial kymograph, and contact-sheet entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from tier4_source_balanced_pre_event_consensus_visual_packet import (
    clean_json,
    load_frames,
    make_contact_sheet,
    render_frame_strip,
    render_kymograph,
    render_mask_overlay,
)


OUTPUT_DIRNAME = "source_balanced_pre_event_manual_qc_visual_packet"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Pre-Event Manual-QC Visual Packet",
        "",
        "This packet renders the manual-QC decision queue, not just the earlier consensus top candidates.",
        "",
        "## Key Counts",
        "",
        f"- Queue rows: {summary['n_queue_rows']}",
        f"- Requested/rendered: {summary['top_n']} / {summary['n_rendered']}",
        f"- Sources rendered: {summary['n_sources_rendered']}",
        f"- Action tiers rendered: {summary['action_tier_counts_rendered']}",
        f"- Event bins rendered: {summary['event_relative_bin_counts_rendered']}",
        f"- Contact sheet: `{summary.get('contact_sheet')}`",
        "",
        "## Top Rendered Candidates",
        "",
    ]
    for row in summary.get("rendered_candidates", [])[:12]:
        lines.append(
            f"- rank {row.get('manual_qc_rank')} {row.get('roi_id')} "
            f"{row.get('manual_qc_action_tier')} {row.get('event_relative_bin')} "
            f"score={row.get('manual_qc_decision_score'):.3f}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir) if args.out_dir else derived / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "visual_assets"
    assets.mkdir(parents=True, exist_ok=True)

    queue = read_csv(
        derived
        / "source_balanced_pre_event_manual_qc_decision_packet"
        / "source_balanced_pre_event_manual_qc_decision_queue.csv"
    )
    top = queue.sort_values("manual_qc_rank").head(args.top_n).copy()

    rendered_rows: List[Dict[str, Any]] = []
    contact_paths: List[Path] = []
    contact_labels: List[str] = []
    for _, row in top.iterrows():
        roi_id = str(row["roi_id"])
        rank = int(row["manual_qc_rank"])
        action = str(row.get("manual_qc_action_tier", "unknown"))
        frames = load_frames(str(row.get("npz_path", "")))
        rec = row.to_dict()
        if frames is None:
            rec["manual_visual_render_status"] = "missing_or_invalid_npz"
            rendered_rows.append(rec)
            continue
        label = (
            f"rank {rank} {action} {row.get('event_relative_bin')} "
            f"score={float(row.get('manual_qc_decision_score', np.nan)):.3f}"
        )
        safe = f"rank{rank:03d}_{action}_{roi_id}".replace("/", "_")
        strip = render_frame_strip(frames, assets / f"{safe}_strip.png", label)
        overlay = render_mask_overlay(frames, assets / f"{safe}_mask_overlay.png", label)
        kymo = render_kymograph(frames, assets / f"{safe}_kymograph.png", label)
        rec.update(
            {
                "manual_visual_render_status": "ok" if strip and overlay and kymo else "render_failed",
                "manual_frame_strip_png": strip,
                "manual_mask_overlay_png": overlay,
                "manual_kymograph_png": kymo,
                "manual_n_frames_loaded": int(frames.shape[0]),
                "manual_frame_height": int(frames.shape[1]),
                "manual_frame_width": int(frames.shape[2]),
            }
        )
        rendered_rows.append(rec)
        if strip:
            contact_paths.append(Path(strip))
            contact_labels.append(f"rank {rank} {action}")

    rendered = pd.DataFrame(rendered_rows)
    rendered_path = out / "source_balanced_pre_event_manual_qc_visual_assets.csv"
    rendered.to_csv(rendered_path, index=False)
    contact = make_contact_sheet(contact_paths, contact_labels, out / "manual_qc_visual_contact_sheet.png")

    rendered_ok = (
        rendered[rendered["manual_visual_render_status"].eq("ok")]
        if "manual_visual_render_status" in rendered.columns
        else pd.DataFrame()
    )
    summary = {
        "n_queue_rows": int(len(queue)),
        "top_n": int(args.top_n),
        "n_rendered": int(len(rendered_ok)),
        "n_sources_rendered": int(rendered_ok["source_stem"].nunique()) if not rendered_ok.empty else 0,
        "action_tier_counts_rendered": clean_json(rendered_ok["manual_qc_action_tier"].value_counts().to_dict()) if not rendered_ok.empty else {},
        "event_relative_bin_counts_rendered": clean_json(rendered_ok["event_relative_bin"].value_counts().to_dict()) if not rendered_ok.empty else {},
        "rendered_candidates": clean_json(rendered.head(20).to_dict("records")),
        "contact_sheet": contact,
        "outputs": {
            "visual_assets": str(assets),
            "asset_table": str(rendered_path),
            "contact_sheet": contact,
            "summary": str(out / "source_balanced_pre_event_manual_qc_visual_summary.json"),
            "readme": str(out / "README.md"),
        },
        "guardrail": (
            "This packet renders automatic particle-crop visualizations for manual inspection. It does not assign labels, "
            "validate particle identity, validate front masks, calibrate diffusion, prove phase-boundary motion, or establish causality."
        ),
    }
    summary_path = out / "source_balanced_pre_event_manual_qc_visual_summary.json"
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out, clean_json(summary))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
