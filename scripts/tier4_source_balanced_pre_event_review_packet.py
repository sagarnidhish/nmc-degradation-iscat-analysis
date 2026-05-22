#!/usr/bin/env python3
"""Build a pre-event ROI review packet from source-invariant evidence.

The source-balanced pre-event audits nominate front/mask/apparent-diffusion
features as the most useful source-handled mechanism family. This script turns
those scalar/model outputs into a compact manual-QC packet: ranked ROI rows,
small frame-strip PNGs rendered from remote NPZ tensors, and a contact sheet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


TARGET_MODELS = [
    ("clean_pre8_vs_post_control", "physics_front_combo", "source_mean_resid_2", "si_clean_physics_prob"),
    ("near_pre_vs_far_pre", "physics_front_combo", "source_residual", "si_near_far_physics_prob"),
    ("clean_pre8_vs_post_control", "roi_intensity", "source_mean_resid_2", "si_clean_intensity_guardrail_prob"),
]


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


def percentile(series: pd.Series, high_good: bool = True) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    ranks = vals.rank(pct=True)
    if not high_good:
        ranks = 1.0 - ranks
    return ranks.fillna(0.5)


def robust_z(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    med = vals.median()
    mad = (vals - med).abs().median()
    scale = 1.4826 * mad if pd.notna(mad) and mad > 1e-12 else vals.std()
    if pd.isna(scale) or scale <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((vals - med) / scale).clip(-5, 5).fillna(0.0)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def model_prediction_columns(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, family, method, name in TARGET_MODELS:
        sub = pred[
            pred["target"].astype(str).eq(target)
            & pred["feature_family"].astype(str).eq(family)
            & pred["method"].astype(str).eq(method)
            & pred["status"].astype(str).eq("ok")
        ][["roi_id", "predicted_probability"]].copy()
        sub = sub.rename(columns={"predicted_probability": name})
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["roi_id"])
    out = rows[0]
    for sub in rows[1:]:
        out = out.merge(sub, on="roi_id", how="outer")
    return out


def render_candidate_strip(npz_path: str, out_path: Path, label: str) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    try:
        data = np.load(npz_path)
        frames = data["frames_norm"] if "frames_norm" in data else data["frames"]
    except Exception:
        return None
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3 or frames.shape[0] < 2:
        return None
    idx = np.linspace(0, frames.shape[0] - 1, 5).round().astype(int)
    panels: List[np.ndarray] = [frames[i] for i in idx]
    panels.append(frames[-1] - frames[0])
    thumbs = []
    labels = [f"f{i}" for i in idx] + ["last-first"]
    for arr, name in zip(panels, labels):
        arr = np.asarray(arr, dtype=float)
        lo, hi = np.nanpercentile(arr, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            img = np.zeros(arr.shape, dtype=np.uint8)
        else:
            img = np.clip((arr - lo) / (hi - lo), 0, 1)
            img = (255 * img).astype(np.uint8)
        pil = Image.fromarray(img).convert("RGB").resize((96, 96))
        canvas = Image.new("RGB", (96, 116), "white")
        canvas.paste(pil, (0, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), name[:16], fill=(0, 0, 0))
        thumbs.append(canvas)
    header_h = 38
    sheet = Image.new("RGB", (len(thumbs) * 96, 116 + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), label[:110], fill=(0, 0, 0))
    for j, thumb in enumerate(thumbs):
        sheet.paste(thumb, (j * 96, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def make_contact_sheet(image_paths: Iterable[Path], labels: Iterable[str], out_path: Path) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    thumbs = []
    for path, label in zip(image_paths, labels):
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        scale = 320 / max(1, img.width)
        img = img.resize((320, max(1, int(img.height * scale))))
        canvas = Image.new("RGB", (320, img.height + 32), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), label[:72], fill=(0, 0, 0))
        canvas.paste(img, (0, 32))
        thumbs.append(canvas)
    if not thumbs:
        return None
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * 320, rows * cell_h), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 320, (i // cols) * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_review_packet")
    parser.add_argument("--top-n", type=int, default=24)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "candidate_frame_strips"
    assets.mkdir(exist_ok=True)

    readout = read_csv(derived / "source_balanced_pre_event_readout_audit" / "source_balanced_pre_event_readout_features.csv")
    seq = read_csv(derived / "source_balanced_pre_event_sequence_audit" / "source_balanced_pre_event_sequence_features.csv")
    pred = read_csv(derived / "source_balanced_pre_event_source_invariant_audit" / "source_balanced_pre_event_source_invariant_predictions.csv")
    pred_cols = model_prediction_columns(pred)

    seq_cols = [
        "roi_id", "spatial_std_slope", "frame_diff_mse_slope", "target_near_pre_vs_rest",
        "target_pre16_clean_vs_post_control", "target_any_pre_vs_post_control",
    ]
    seq_cols = [c for c in seq_cols if c in seq.columns]
    df = readout.merge(seq[seq_cols], on="roi_id", how="left").merge(pred_cols, on="roi_id", how="left")

    near_weight = {
        "near_pre_event_1_8": 1.0,
        "mid_pre_event_9_16": 0.65,
        "far_pre_event_17_32": 0.35,
        "post_event_1_16": -0.25,
        "no_near_event_control": -0.10,
    }
    df["event_bin_priority"] = df["event_relative_bin"].map(near_weight).fillna(0.0)
    df["front_source_residual_score"] = percentile(df.get("front_radius_q60_slope_px_per_norm_time", pd.Series(index=df.index)), True)
    df["apparent_diffusion_score"] = percentile(df.get("apparent_diffusion_q70_um2_per_norm_time", pd.Series(index=df.index)), True)
    df["contrast_slope_score"] = percentile(df.get("masked_minus_background_mean_slope", pd.Series(index=df.index)), True)
    df["spatial_clock_score"] = percentile(df.get("spatial_std_slope", pd.Series(index=df.index)), True)
    for col in ["si_clean_physics_prob", "si_near_far_physics_prob", "si_clean_intensity_guardrail_prob"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pre_event_review_score"] = (
        1.20 * df["si_clean_physics_prob"].fillna(0.5)
        + 1.10 * df["si_near_far_physics_prob"].fillna(0.5)
        + 0.85 * df["front_source_residual_score"]
        + 0.75 * df["apparent_diffusion_score"]
        + 0.60 * df["contrast_slope_score"]
        + 0.45 * df["spatial_clock_score"]
        + 0.80 * df["event_bin_priority"]
        - 0.35 * df["si_clean_intensity_guardrail_prob"].fillna(0.5)
    )
    df["review_reason"] = np.select(
        [
            df["event_relative_bin"].eq("near_pre_event_1_8") & (df["si_clean_physics_prob"].fillna(0) >= 0.65),
            df["event_relative_bin"].eq("near_pre_event_1_8"),
            df["event_relative_bin"].eq("mid_pre_event_9_16"),
        ],
        [
            "near_pre_high_source_invariant_front_score",
            "near_pre_front_diffusion_review",
            "mid_pre_followup_front_diffusion_review",
        ],
        default="context_or_guardrail_row",
    )
    df["manual_qc_status"] = "pending_review"
    ranked = df.sort_values("pre_event_review_score", ascending=False).reset_index(drop=True)
    ranked["pre_event_review_rank"] = np.arange(1, len(ranked) + 1)

    asset_rows = []
    contact_paths: List[Path] = []
    contact_labels: List[str] = []
    for _, row in ranked.head(args.top_n).iterrows():
        roi_id = str(row["roi_id"])
        rank = int(row["pre_event_review_rank"])
        label = f"{rank}. {roi_id} {row['event_relative_bin']} score={row['pre_event_review_score']:.2f}"
        strip_path = assets / f"{rank:02d}_{roi_id}.png"
        rendered = render_candidate_strip(str(row.get("npz_path", "")), strip_path, label)
        asset_rows.append({
            "pre_event_review_rank": rank,
            "roi_id": roi_id,
            "event_relative_bin": row.get("event_relative_bin"),
            "cycleNo": row.get("cycleNo"),
            "source_stem": row.get("source_stem"),
            "pre_event_review_score": row.get("pre_event_review_score"),
            "review_reason": row.get("review_reason"),
            "frame_strip_png": rendered,
        })
        if rendered:
            contact_paths.append(Path(rendered))
            contact_labels.append(label)
    contact_sheet = make_contact_sheet(contact_paths, contact_labels, out / "pre_event_review_contact_sheet.png")

    ranked_path = out / "source_balanced_pre_event_review_ranked_candidates.csv"
    asset_path = out / "source_balanced_pre_event_review_assets.csv"
    summary_path = out / "source_balanced_pre_event_review_packet_summary.json"
    ranked.to_csv(ranked_path, index=False)
    pd.DataFrame(asset_rows).to_csv(asset_path, index=False)

    tier_counts = ranked["review_reason"].value_counts().to_dict()
    summary = {
        "n_candidates": int(len(ranked)),
        "n_cycles": int(ranked["cycleNo"].nunique()),
        "n_sources": int(ranked["source_stem"].nunique()),
        "top_n_rendered": int(args.top_n),
        "n_rendered_frame_strips": int(sum(pd.notna(r.get("frame_strip_png")) for r in asset_rows)),
        "event_relative_bin_counts": clean_json(ranked["event_relative_bin"].value_counts().to_dict()),
        "review_reason_counts": clean_json(tier_counts),
        "top_candidates": clean_json(ranked.head(12)[[
            "pre_event_review_rank", "roi_id", "cycleNo", "source_stem", "event_relative_bin",
            "pre_event_review_score", "review_reason", "si_clean_physics_prob",
            "si_near_far_physics_prob", "front_radius_q60_slope_px_per_norm_time",
            "apparent_diffusion_q70_um2_per_norm_time", "masked_minus_background_mean_slope",
        ]].to_dict("records")),
        "contact_sheet": contact_sheet,
        "outputs": {
            "ranked_candidates": str(ranked_path),
            "asset_index": str(asset_path),
            "frame_strips": str(assets),
            "contact_sheet": contact_sheet,
            "summary": str(summary_path),
        },
        "guardrail": "This packet ranks automatic source-balanced pre-event ROI crops for manual review using weak event-relative labels and source-invariant/front-proxy scores. Frame strips are rendered from automatic fixed crops. No manual labels, particle identities, phase-boundary validation, calibrated diffusion coefficients, or causal precursor claims are assigned.",
    }
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n")
    lines = [
        "# Source-Balanced Pre-Event Review Packet",
        "",
        f"- Candidates ranked: {summary['n_candidates']}",
        f"- Top frame strips rendered: {summary['n_rendered_frame_strips']} / {summary['top_n_rendered']}",
        f"- Contact sheet: `{contact_sheet}`" if contact_sheet else "- Contact sheet: unavailable",
        f"- Review reason counts: {summary['review_reason_counts']}",
        "",
        "## Top Candidates",
    ]
    for row in summary["top_candidates"][:8]:
        lines.append(
            f"- {row['pre_event_review_rank']}. {row['roi_id']} ({row['event_relative_bin']}, cycle {row['cycleNo']}): score={row['pre_event_review_score']:.3f}, reason={row['review_reason']}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
