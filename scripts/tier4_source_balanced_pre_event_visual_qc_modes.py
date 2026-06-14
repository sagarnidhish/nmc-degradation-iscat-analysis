#!/usr/bin/env python3
"""Score and cluster consensus visual candidates for front/manual-QC review.

The consensus visual packet renders candidates but intentionally assigns no
labels. This audit keeps that guardrail: it computes automatic visual/front
plausibility descriptors from each rendered ROI sequence, ranks likely coherent
fronts versus likely artifacts, and groups candidates into review modes. The
outputs are review priorities, not accepted/rejected labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REMOTE_ROOT = Path("/scratch/<account>/<username>/Alek_Jiho")
DERIVED = REMOTE_ROOT / "derived"
VISUAL_DIR = DERIVED / "source_balanced_pre_event_consensus_visual_packet"
OUT_DIR = DERIVED / "source_balanced_pre_event_visual_qc_modes"


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


def load_frames(path: str) -> Optional[np.ndarray]:
    try:
        data = np.load(path)
        arr = data["frames_norm"] if "frames_norm" in data else data["frames"]
    except Exception:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3 or arr.shape[0] < 8:
        return None
    return arr


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


def radial_kymograph(frames: np.ndarray, mask: np.ndarray, n_bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
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


def front_trace(kymo: np.ndarray, centers: np.ndarray) -> np.ndarray:
    trace = np.full(kymo.shape[0], np.nan, dtype=float)
    for i, row in enumerate(kymo):
        if np.isfinite(row).sum() < 4:
            continue
        smooth = pd.Series(row).interpolate(limit_direction="both").rolling(3, center=True, min_periods=1).mean().to_numpy()
        grad = np.abs(np.gradient(smooth))
        trace[i] = centers[int(np.nanargmax(grad))]
    return trace


def robust_slope(y: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    x = np.linspace(0.0, 1.0, len(y))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan
    x = x[ok]
    y = y[ok]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    direction_fraction = float(np.mean(np.sign(np.diff(y)) == np.sign(slope))) if len(y) > 2 and slope != 0 else 0.0
    return float(slope), float(r2), direction_fraction


def score_row(row: pd.Series) -> Dict[str, Any]:
    frames = load_frames(str(row.get("npz_path", "")))
    if frames is None:
        return {"visual_qc_status": "load_failed"}
    mask = stable_mask(frames)
    kymo, centers = radial_kymograph(frames, mask)
    trace = front_trace(kymo, centers)
    slope, r2, direction_fraction = robust_slope(trace)
    trace_span = float(np.nanpercentile(trace, 90) - np.nanpercentile(trace, 10)) if np.isfinite(trace).any() else np.nan
    trace_jitter = float(np.nanmedian(np.abs(np.diff(trace)))) if np.isfinite(trace).sum() > 2 else np.nan
    inside = frames[:, mask]
    outside = frames[:, ~mask] if (~mask).any() else frames.reshape(frames.shape[0], -1)
    masked_mean = np.nanmean(inside, axis=1)
    bg_mean = np.nanmean(outside, axis=1)
    contrast = masked_mean - bg_mean
    contrast_slope, contrast_r2, contrast_direction_fraction = robust_slope(contrast)
    frame_delta = np.nanmean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))
    temporal_snr = abs(float(np.nanmedian(frame_delta))) / (float(np.nanmedian(np.abs(frame_delta - np.nanmedian(frame_delta)))) + 1e-9)
    kymo_energy = float(np.nanmean(np.abs(np.diff(kymo, axis=0))))
    mask_fraction = float(mask.mean())
    mask_ok = 0.04 <= mask_fraction <= 0.65
    front_score = (
        0.30 * np.clip(r2, 0, 1)
        + 0.22 * np.clip(direction_fraction, 0, 1)
        + 0.16 * np.clip(abs(slope) / 8.0, 0, 1)
        + 0.12 * np.clip(trace_span / 12.0, 0, 1)
        + 0.10 * np.clip(temporal_snr / 12.0, 0, 1)
        + 0.10 * float(mask_ok)
    )
    artifact_risk = (
        0.35 * float(not mask_ok)
        + 0.25 * np.clip((trace_jitter or 0.0) / 8.0, 0, 1)
        + 0.20 * np.clip(kymo_energy / 0.02, 0, 1)
        + 0.20 * float(np.nan_to_num(r2) < 0.05)
    )
    consensus = float(row.get("consensus_review_score", 0) or 0)
    support = float(row.get("matched_positive_support_count", 0) or 0)
    support_norm = np.clip(support / 60.0, 0, 1)
    review_score = 0.48 * front_score + 0.34 * consensus + 0.18 * support_norm - 0.18 * artifact_risk
    if review_score >= 0.62 and artifact_risk < 0.45:
        tier = "front_plausible_priority"
    elif artifact_risk >= 0.55:
        tier = "artifact_risk_review"
    elif review_score >= 0.48:
        tier = "front_plausible_followup"
    else:
        tier = "routine_or_low_front_plausibility"
    return {
        "visual_qc_status": "ok",
        "mask_fraction": mask_fraction,
        "front_trace_slope_px_per_norm_time": slope,
        "front_trace_r2": r2,
        "front_trace_direction_fraction": direction_fraction,
        "front_trace_span_px": trace_span,
        "front_trace_jitter_px": trace_jitter,
        "masked_minus_background_slope": contrast_slope,
        "masked_minus_background_r2": contrast_r2,
        "masked_minus_background_direction_fraction": contrast_direction_fraction,
        "frame_delta_temporal_snr": temporal_snr,
        "kymograph_delta_energy": kymo_energy,
        "visual_front_plausibility_score": float(front_score),
        "visual_artifact_risk_score": float(artifact_risk),
        "visual_review_score": float(review_score),
        "visual_qc_tier": tier,
    }


def add_clusters(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "visual_front_plausibility_score",
        "visual_artifact_risk_score",
        "front_trace_slope_px_per_norm_time",
        "front_trace_r2",
        "front_trace_direction_fraction",
        "front_trace_span_px",
        "masked_minus_background_slope",
        "frame_delta_temporal_snr",
        "kymograph_delta_energy",
        "consensus_review_score",
        "matched_positive_support_count",
    ]
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    scale = x.std(axis=0).replace(0, 1.0)
    z = (x - x.mean(axis=0)) / scale
    n_clusters = min(4, max(2, len(df) // 5))
    labels = None
    try:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=n_clusters, n_init=50, random_state=17).fit_predict(z)
    except Exception:
        order = pd.qcut(df["visual_review_score"].rank(method="first"), q=n_clusters, labels=False)
        labels = np.asarray(order, dtype=int)
    df = df.copy()
    df["visual_mode_cluster"] = labels.astype(int)
    summaries = []
    for cluster, g in df.groupby("visual_mode_cluster"):
        summaries.append(
            {
                "cluster": int(cluster),
                "n": int(len(g)),
                "median_review_score": float(g["visual_review_score"].median()),
                "median_front_score": float(g["visual_front_plausibility_score"].median()),
                "median_artifact_risk": float(g["visual_artifact_risk_score"].median()),
                "top_tier": str(g["visual_qc_tier"].mode().iloc[0]),
            }
        )
    names = {}
    for s in summaries:
        if s["median_artifact_risk"] >= 0.55:
            name = "artifact-risk / unstable-front review"
        elif s["median_front_score"] >= 0.55 and s["median_review_score"] >= 0.58:
            name = "coherent outward-front priority"
        elif s["median_front_score"] >= 0.40:
            name = "moderate front-like followup"
        else:
            name = "low front-plausibility context"
        names[s["cluster"]] = name
    df["visual_mode_name"] = df["visual_mode_cluster"].map(names)
    return df


def make_contact_sheet(df: pd.DataFrame, out_path: Path, max_items: int = 24) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    panels = []
    for _, row in df.head(max_items).iterrows():
        strip = Path(str(row.get("frame_strip_png", "")))
        kymo = Path(str(row.get("kymograph_png", "")))
        overlay = Path(str(row.get("mask_overlay_png", "")))
        imgs = []
        for path in [strip, overlay, kymo]:
            if path.exists():
                try:
                    imgs.append(Image.open(path).convert("RGB").resize((220, 120)))
                except Exception:
                    pass
        if not imgs:
            continue
        canvas = Image.new("RGB", (660, 168), "white")
        draw = ImageDraw.Draw(canvas)
        label = (
            f"rank {int(row.get('consensus_review_rank', 0))} "
            f"mode {int(row.get('visual_mode_cluster', -1))} "
            f"{row.get('visual_qc_tier', 'NA')} "
            f"score={row.get('visual_review_score', np.nan):.3f}"
        )
        draw.text((5, 5), label[:105], fill=(0, 0, 0))
        draw.text((5, 22), str(row.get("roi_id", "NA"))[:105], fill=(0, 0, 0))
        for i, img in enumerate(imgs[:3]):
            canvas.paste(img, (i * 220, 48))
        panels.append(canvas)
    if not panels:
        return None
    cols = 1
    rows = len(panels)
    sheet = Image.new("RGB", (660 * cols, 168 * rows), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, i * 168))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def write_readme(summary: Dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Source-Balanced Pre-Event Visual QC Modes",
        "",
        "Automatic visual/front plausibility audit for the consensus review packet.",
        "This is a prioritization layer for manual QC, not a manual label set.",
        "",
        f"- Candidates scored: {summary['n_scored']}",
        f"- Priority-tier counts: {summary['visual_qc_tier_counts']}",
        f"- Mode counts: {summary['visual_mode_counts']}",
        f"- Top candidate: {summary['top_candidate'].get('roi_id', 'NA')} ({summary['top_candidate'].get('visual_qc_tier', 'NA')}, score {summary['top_candidate'].get('visual_review_score', 'NA')})",
        "",
        "## Outputs",
        "",
        f"- `source_balanced_pre_event_visual_qc_modes.csv`",
        f"- `source_balanced_pre_event_visual_qc_mode_summary.csv`",
        f"- `visual_qc_priority_contact_sheet.png`",
        f"- `source_balanced_pre_event_visual_qc_modes_summary.json`",
        "",
        f"Guardrail: {summary['guardrail']}",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asset_path = VISUAL_DIR / "source_balanced_pre_event_consensus_visual_assets.csv"
    df = pd.read_csv(asset_path)
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        rec.update(score_row(row))
        rows.append(rec)
    scored = pd.DataFrame(rows)
    scored = scored[scored["visual_qc_status"].eq("ok")].copy()
    scored = add_clusters(scored)
    scored = scored.sort_values(["visual_review_score", "consensus_review_score"], ascending=False)
    scored.to_csv(OUT_DIR / "source_balanced_pre_event_visual_qc_modes.csv", index=False)
    mode_summary = (
        scored.groupby(["visual_mode_cluster", "visual_mode_name"], dropna=False)
        .agg(
            n=("roi_id", "size"),
            median_visual_review_score=("visual_review_score", "median"),
            median_front_plausibility_score=("visual_front_plausibility_score", "median"),
            median_artifact_risk_score=("visual_artifact_risk_score", "median"),
            near_pre_count=("event_relative_bin", lambda s: int(s.astype(str).str.contains("near_pre").sum())),
            matched_support_front_qc_count=("review_priority_tier", lambda s: int((s == "matched_support_front_qc").sum())),
        )
        .reset_index()
        .sort_values("median_visual_review_score", ascending=False)
    )
    mode_summary.to_csv(OUT_DIR / "source_balanced_pre_event_visual_qc_mode_summary.csv", index=False)
    contact_sheet = make_contact_sheet(scored, OUT_DIR / "visual_qc_priority_contact_sheet.png")
    top = scored.head(1).to_dict("records")[0] if len(scored) else {}
    summary = {
        "n_input_candidates": int(len(df)),
        "n_scored": int(len(scored)),
        "visual_qc_tier_counts": clean_json(scored["visual_qc_tier"].value_counts().to_dict()),
        "visual_mode_counts": clean_json(scored["visual_mode_name"].value_counts().to_dict()),
        "top_candidate": clean_json(top),
        "top_candidates": clean_json(scored.head(12).to_dict("records")),
        "mode_summary": clean_json(mode_summary.to_dict("records")),
        "outputs": {
            "scored_csv": str(OUT_DIR / "source_balanced_pre_event_visual_qc_modes.csv"),
            "mode_summary_csv": str(OUT_DIR / "source_balanced_pre_event_visual_qc_mode_summary.csv"),
            "contact_sheet": contact_sheet,
            "summary": str(OUT_DIR / "source_balanced_pre_event_visual_qc_modes_summary.json"),
        },
        "guardrail": "Automatic visual/front plausibility scores and clusters are review priorities only. They do not validate particle identity, front masks, manual QC, calibrated diffusion, phase-boundary motion, or causality.",
    }
    (OUT_DIR / "source_balanced_pre_event_visual_qc_modes_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    write_readme(summary, OUT_DIR)


if __name__ == "__main__":
    main()
