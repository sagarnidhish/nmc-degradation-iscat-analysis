#!/usr/bin/env python3
"""Build a visual review bundle for top precursor-informed ROI candidates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


IMAGE_COLUMNS = [
    "primary_qc_png",
    "control_balanced_qc_png",
    "roi_preview_path",
    "rollout_preview_path",
    "front_tracking_plot_path",
    "front_crop_preview_path",
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_name(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def copy_if_present(src_value: Any, dest_dir: Path, prefix: str) -> Optional[str]:
    if pd.isna(src_value) or not str(src_value).strip():
        return None
    src = Path(str(src_value))
    if not src.exists() or not src.is_file():
        return None
    suffix = src.suffix if src.suffix else ".png"
    dest = dest_dir / f"{prefix}{suffix}"
    shutil.copy2(src, dest)
    return str(dest)


def make_contact_sheet(image_paths: List[Path], labels: List[str], out_path: Path, thumb_width: int = 360) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    if not image_paths:
        return None
    thumbs = []
    label_height = 40
    for path, label in zip(image_paths, labels):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        scale = thumb_width / max(1, img.width)
        thumb_height = max(1, int(img.height * scale))
        img = img.resize((thumb_width, thumb_height))
        canvas = Image.new("RGB", (thumb_width, thumb_height + label_height), "white")
        canvas.paste(img, (0, label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 6), label[:70], fill=(0, 0, 0))
        thumbs.append(canvas)
    if not thumbs:
        return None
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), "white")
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * thumb_width
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x, y))
    sheet.save(out_path)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/precursor_review_visual_bundle")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "assets"
    assets.mkdir(exist_ok=True)

    manifest = read_csv(derived / "precursor_informed_roi_review" / "precursor_informed_roi_review_manifest.csv")
    manifest = manifest.sort_values("precursor_informed_review_score", ascending=False).head(args.top_n).copy()

    rows: List[Dict[str, Any]] = []
    contact_images: List[Path] = []
    contact_labels: List[str] = []
    for rank, (_, row) in enumerate(manifest.iterrows(), start=1):
        roi_id = str(row.get("roi_id"))
        roi_dir = assets / f"{rank:02d}_{safe_name(roi_id)}"
        roi_dir.mkdir(exist_ok=True)
        copied: Dict[str, Optional[str]] = {}
        for col in IMAGE_COLUMNS:
            copied[col] = copy_if_present(row.get(col), roi_dir, col)
        primary = copied.get("primary_qc_png") or copied.get("control_balanced_qc_png") or copied.get("roi_preview_path")
        if primary:
            contact_images.append(Path(primary))
            contact_labels.append(f"{rank}. {roi_id} score={row.get('precursor_informed_review_score'):.3g}")
        rows.append({
            "visual_rank": rank,
            "roi_id": roi_id,
            "cohort_role": row.get("cohort_role"),
            "cycleNo": row.get("cycleNo"),
            "event_reference_cycle": row.get("event_reference_cycle"),
            "precursor_review_tier": row.get("precursor_review_tier"),
            "precursor_informed_review_score": row.get("precursor_informed_review_score"),
            "precursor_review_reason": row.get("precursor_review_reason"),
            "auto_review_flags": row.get("auto_review_flags"),
            "manual_qc_status": row.get("manual_qc_status"),
            **{f"copied_{k}": v for k, v in copied.items()},
        })

    index = pd.DataFrame(rows)
    index_path = out / "visual_review_bundle_index.csv"
    index.to_csv(index_path, index=False)
    sheet_path = make_contact_sheet(contact_images, contact_labels, out / "top_candidate_contact_sheet.png")

    summary = {
        "n_ranked_candidates": int(len(index)),
        "n_candidates_with_visual_asset": int(index[[f"copied_{c}" for c in IMAGE_COLUMNS]].notna().any(axis=1).sum()),
        "contact_sheet": sheet_path,
        "top_candidates": index.head(8).to_dict("records"),
        "guardrail": "This bundle copies existing automatic QC/preview assets for manual inspection. It does not create labels, adjudicate particle identity, or validate diffusion/front interpretability.",
        "outputs": {
            "index": str(index_path),
            "assets": str(assets),
            "contact_sheet": sheet_path,
            "summary": str(out / "precursor_review_visual_bundle_summary.json"),
        },
    }
    with (out / "precursor_review_visual_bundle_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Precursor Review Visual Bundle",
        "",
        "Inspection bundle for the top precursor-informed ROI candidates.",
        "",
        f"- Ranked candidates included: {summary['n_ranked_candidates']}",
        f"- Candidates with at least one visual asset: {summary['n_candidates_with_visual_asset']}",
        f"- Contact sheet: `{sheet_path}`" if sheet_path else "- Contact sheet: not created; Pillow unavailable or no image assets found.",
        "",
        "## Top Candidates",
    ]
    for item in summary["top_candidates"]:
        lines.append(
            f"- {item.get('visual_rank')}. {item.get('roi_id')} ({item.get('cohort_role')}, cycle {item.get('cycleNo')}): score={item.get('precursor_informed_review_score'):.3f}, tier={item.get('precursor_review_tier')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json({
        "out_dir": str(out),
        "n_ranked_candidates": summary["n_ranked_candidates"],
        "n_candidates_with_visual_asset": summary["n_candidates_with_visual_asset"],
        "contact_sheet": sheet_path,
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
