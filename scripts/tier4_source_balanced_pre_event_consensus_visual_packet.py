#!/usr/bin/env python3
"""Render visual QC packet for top consensus pre-event candidates.

The consensus review queue ranks automatic ROI/front candidates, but scalar
rankings are not enough for physics claims. This packet renders compact visual
evidence for the top queue rows: frame strips, a crop/mask overlay, radial
kymograph heatmap, front trace plot, and a contact sheet. It assigns no labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def load_frames(npz_path: str) -> Optional[np.ndarray]:
    try:
        data = np.load(npz_path)
        frames = data["frames_norm"] if "frames_norm" in data else data["frames"]
    except Exception:
        return None
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3 or frames.shape[0] < 2:
        return None
    return frames


def norm_u8(arr: np.ndarray, lohi: Optional[Tuple[float, float]] = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if lohi is None:
        lo, hi = np.nanpercentile(arr, [2, 98])
    else:
        lo, hi = lohi
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (255 * np.clip((arr - lo) / (hi - lo), 0, 1)).astype(np.uint8)


def render_frame_strip(frames: np.ndarray, out_path: Path, label: str) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    idx = np.linspace(0, frames.shape[0] - 1, 5).round().astype(int)
    panels = [frames[i] for i in idx] + [frames[-1] - frames[0]]
    names = [f"f{i}" for i in idx] + ["last-first"]
    lohi = tuple(np.nanpercentile(frames, [2, 98]))
    thumbs = []
    for arr, name in zip(panels, names):
        u8 = norm_u8(arr, None if name == "last-first" else lohi)
        img = Image.fromarray(u8).convert("RGB").resize((112, 112))
        canvas = Image.new("RGB", (112, 134), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), name, fill=(0, 0, 0))
        canvas.paste(img, (0, 22))
        thumbs.append(canvas)
    header_h = 42
    sheet = Image.new("RGB", (len(thumbs) * 112, 134 + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), label[:125], fill=(0, 0, 0))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, (i * 112, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def stable_mask(frames: np.ndarray) -> np.ndarray:
    base = np.nanmedian(frames[: max(2, min(5, len(frames)))], axis=0)
    lo, hi = np.nanpercentile(base, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones(base.shape, dtype=bool)
    high = base >= np.nanpercentile(base, 70)
    low = base <= np.nanpercentile(base, 30)
    mask = high if high.mean() < 0.65 else low
    if mask.mean() < 0.02 or mask.mean() > 0.85:
        yy, xx = np.indices(base.shape)
        cy, cx = (np.array(base.shape) - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = rr <= 0.38 * min(base.shape)
    return mask.astype(bool)


def render_mask_overlay(frames: np.ndarray, out_path: Path, label: str) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    img_arr = np.nanmedian(frames, axis=0)
    mask = stable_mask(frames)
    rgb = Image.fromarray(norm_u8(img_arr)).convert("RGB").resize((240, 240))
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask.astype(np.uint8) * 120)).resize(rgb.size)
    red = Image.new("RGBA", rgb.size, (255, 30, 30, 0))
    red.putalpha(mask_img)
    overlay.alpha_composite(red)
    rgb = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
    canvas = Image.new("RGB", (240, 280), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), label[:90], fill=(0, 0, 0))
    draw.text((5, 22), f"stable-mask fraction={mask.mean():.3f}", fill=(0, 0, 0))
    canvas.paste(rgb, (0, 40))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def radial_kymograph(frames: np.ndarray, n_bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    mask = stable_mask(frames)
    yy, xx = np.indices(frames.shape[1:])
    if mask.any():
        cy = float(yy[mask].mean())
        cx = float(xx[mask].mean())
    else:
        cy, cx = (np.array(frames.shape[1:]) - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.linspace(0, float(rr.max()) + 1e-6, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    kymo = np.full((frames.shape[0], n_bins), np.nan, dtype=float)
    for i, frame in enumerate(frames):
        for b in range(n_bins):
            pix = frame[(rr >= bins[b]) & (rr < bins[b + 1])]
            if pix.size:
                kymo[i, b] = np.nanmean(pix)
    return kymo, centers


def front_trace_from_kymo(kymo: np.ndarray, centers: np.ndarray) -> np.ndarray:
    trace = np.full(kymo.shape[0], np.nan, dtype=float)
    for i, row in enumerate(kymo):
        if np.isfinite(row).sum() < 4:
            continue
        smooth = pd.Series(row).interpolate(limit_direction="both").rolling(3, center=True, min_periods=1).mean().to_numpy()
        grad = np.abs(np.gradient(smooth))
        trace[i] = centers[int(np.nanargmax(grad))]
    return trace


def render_kymograph(frames: np.ndarray, out_path: Path, title: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    kymo, centers = radial_kymograph(frames)
    trace = front_trace_from_kymo(kymo, centers)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=120)
    lo, hi = np.nanpercentile(kymo, [2, 98])
    axes[0].imshow(kymo.T, aspect="auto", origin="lower", cmap="viridis", vmin=lo, vmax=hi)
    axes[0].plot(np.arange(len(trace)), np.interp(trace, centers, np.arange(len(centers))), color="white", lw=1)
    axes[0].set_title("radial kymograph")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("radius bin")
    axes[1].plot(trace, color="#1f77b4", lw=1.5)
    axes[1].set_title("front proxy")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("radius px")
    fig.suptitle(title[:95], fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def make_contact_sheet(paths: Iterable[Path], labels: Iterable[str], out_path: Path) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    thumbs = []
    for path, label in zip(paths, labels):
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        scale = 360 / max(1, img.width)
        img = img.resize((360, max(1, int(img.height * scale))))
        canvas = Image.new("RGB", (360, img.height + 28), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), label[:75], fill=(0, 0, 0))
        canvas.paste(img, (0, 28))
        thumbs.append(canvas)
    if not thumbs:
        return None
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * 360, rows * cell_h), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 360, (i // cols) * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Pre-Event Consensus Visual Packet",
        "",
        f"- Queue candidates considered: {summary['n_queue_rows']}",
        f"- Visual candidates requested/rendered: {summary['top_n']} / {summary['n_rendered']}",
        f"- Contact sheet: `{summary.get('contact_sheet')}`" if summary.get("contact_sheet") else "- Contact sheet: unavailable",
        "",
        "## Top Rendered Candidates",
    ]
    for row in summary.get("rendered_candidates", [])[:10]:
        lines.append(
            f"- rank {row['consensus_review_rank']} {row['roi_id']} {row['event_relative_bin']} "
            f"score={row['consensus_review_score']:.3f}, strip={row.get('frame_strip_png')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument(
        "--out-dir",
        default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_consensus_visual_packet",
    )
    parser.add_argument("--top-n", type=int, default=24)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "visual_assets"
    assets.mkdir(parents=True, exist_ok=True)

    queue = read_csv(
        derived
        / "source_balanced_pre_event_consensus_review_queue"
        / "source_balanced_pre_event_consensus_review_queue.csv"
    )
    top = queue.sort_values("consensus_review_rank").head(args.top_n).copy()

    rendered_rows: List[Dict[str, Any]] = []
    contact_paths: List[Path] = []
    contact_labels: List[str] = []
    for _, row in top.iterrows():
        roi_id = str(row["roi_id"])
        rank = int(row["consensus_review_rank"])
        frames = load_frames(str(row.get("npz_path", "")))
        rec = row.to_dict()
        if frames is None:
            rec["render_status"] = "missing_or_invalid_npz"
            rendered_rows.append(rec)
            continue
        label = f"rank {rank} {roi_id} {row.get('event_relative_bin')} score={float(row.get('consensus_review_score', np.nan)):.3f}"
        safe = f"rank{rank:03d}_{roi_id}".replace("/", "_")
        strip = render_frame_strip(frames, assets / f"{safe}_strip.png", label)
        overlay = render_mask_overlay(frames, assets / f"{safe}_mask_overlay.png", label)
        kymo = render_kymograph(frames, assets / f"{safe}_kymograph.png", label)
        rec.update({
            "render_status": "ok" if strip else "render_failed",
            "frame_strip_png": strip,
            "mask_overlay_png": overlay,
            "kymograph_png": kymo,
            "n_frames_loaded": int(frames.shape[0]),
            "frame_height": int(frames.shape[1]),
            "frame_width": int(frames.shape[2]),
        })
        rendered_rows.append(rec)
        if strip:
            contact_paths.append(Path(strip))
            contact_labels.append(f"rank {rank} {row.get('event_relative_bin')}")

    rendered = pd.DataFrame(rendered_rows)
    rendered_path = out / "source_balanced_pre_event_consensus_visual_assets.csv"
    rendered.to_csv(rendered_path, index=False)
    contact = make_contact_sheet(contact_paths, contact_labels, out / "consensus_visual_contact_sheet.png")

    rendered_ok = rendered[rendered["render_status"].eq("ok")] if "render_status" in rendered.columns else pd.DataFrame()
    summary = {
        "n_queue_rows": int(len(queue)),
        "top_n": int(args.top_n),
        "n_rendered": int(len(rendered_ok)),
        "n_sources_rendered": int(rendered_ok["source_stem"].nunique()) if not rendered_ok.empty else 0,
        "event_relative_bin_counts_rendered": clean_json(rendered_ok["event_relative_bin"].value_counts().to_dict()) if not rendered_ok.empty else {},
        "rendered_candidates": clean_json(rendered.head(20).to_dict("records")),
        "contact_sheet": contact,
        "outputs": {
            "visual_assets": str(assets),
            "asset_table": str(rendered_path),
            "contact_sheet": contact,
            "summary": str(out / "source_balanced_pre_event_consensus_visual_summary.json"),
        },
        "guardrail": (
            "This packet renders automatic crop/front visualizations for manual inspection. It does not assign labels, "
            "validate particle identity, validate front masks, calibrate diffusion, prove phase-boundary motion, or establish causality."
        ),
    }
    summary_path = out / "source_balanced_pre_event_consensus_visual_summary.json"
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out, clean_json(summary))
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
