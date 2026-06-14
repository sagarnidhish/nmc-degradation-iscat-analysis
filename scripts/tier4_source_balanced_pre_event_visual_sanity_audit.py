#!/usr/bin/env python3
"""Automatic sanity metrics for consensus pre-event visual candidates.

The visual packet makes top candidates inspectable, but it still needs a
machine-readable QC layer before any physical interpretation. This audit scores
the rendered consensus candidates for crop/mask/front sanity using only the ROI
tensors: stable-mask size and edge contact, centroid drift, focus/blur proxy,
temporal contrast, radial-front trace fit, and source concentration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


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


def frame_mask(frame: np.ndarray, prior: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(frame, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return prior.copy()
    high = frame >= np.nanpercentile(frame, 70)
    low = frame <= np.nanpercentile(frame, 30)
    cand = high if high.mean() < 0.65 else low
    if cand.mean() < 0.02 or cand.mean() > 0.85:
        return prior.copy()
    return cand.astype(bool)


def centroid(mask: np.ndarray) -> Tuple[float, float]:
    yy, xx = np.indices(mask.shape)
    if not mask.any():
        return np.nan, np.nan
    return float(yy[mask].mean()), float(xx[mask].mean())


def edge_fraction(mask: np.ndarray) -> float:
    if not mask.any():
        return np.nan
    edge = np.zeros(mask.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return float((mask & edge).sum() / mask.sum())


def laplacian_variance(frame: np.ndarray) -> float:
    arr = np.asarray(frame, dtype=float)
    lap = (
        -4 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )
    return float(np.nanvar(lap[1:-1, 1:-1]))


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
            vals = frame[(rr >= bins[b]) & (rr < bins[b + 1])]
            if vals.size:
                kymo[i, b] = np.nanmean(vals)
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


def slope_fit(y: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(y, dtype=float)
    x = np.linspace(0, 1, len(vals))
    mask = np.isfinite(vals)
    if mask.sum() < 6 or np.nanstd(vals[mask]) <= 1e-12:
        return {"slope": np.nan, "r2": np.nan, "monotonic_fraction": np.nan}
    coef = np.polyfit(x[mask], vals[mask], 1)
    pred = np.polyval(coef, x[mask])
    ss_res = float(np.sum((vals[mask] - pred) ** 2))
    ss_tot = float(np.sum((vals[mask] - np.mean(vals[mask])) ** 2))
    diffs = np.diff(vals[mask])
    return {
        "slope": float(coef[0]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "monotonic_fraction": float((diffs >= 0).mean()) if len(diffs) else np.nan,
    }


def robust_rank(series: pd.Series, high_good: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 2 or x.nunique(dropna=True) < 2:
        return pd.Series(0.5, index=series.index)
    out = x.rank(pct=True)
    if not high_good:
        out = 1.0 - out
    return out.fillna(0.5)


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mask_fraction_score"] = 1.0 - (numeric(out, "stable_mask_fraction") - 0.25).abs().clip(0, 0.25) / 0.25
    out["mask_fraction_score"] = out["mask_fraction_score"].clip(0, 1).fillna(0.0)
    out["edge_score"] = 1.0 - numeric(out, "stable_mask_edge_fraction").clip(0, 0.2) / 0.2
    out["edge_score"] = out["edge_score"].clip(0, 1).fillna(0.0)
    out["centroid_stability_score"] = 1.0 - numeric(out, "centroid_path_px").clip(0, 40) / 40
    out["centroid_stability_score"] = out["centroid_stability_score"].clip(0, 1).fillna(0.0)
    out["focus_score"] = robust_rank(out["focus_laplacian_var_median"], True)
    out["front_fit_score"] = robust_rank(out["front_trace_r2"], True)
    out["front_monotonic_score"] = pd.to_numeric(out["front_trace_monotonic_fraction"], errors="coerce").fillna(0.5).clip(0, 1)
    out["visual_sanity_score"] = (
        0.20 * out["mask_fraction_score"]
        + 0.15 * out["edge_score"]
        + 0.20 * out["centroid_stability_score"]
        + 0.15 * out["focus_score"]
        + 0.20 * out["front_fit_score"]
        + 0.10 * out["front_monotonic_score"]
    )
    out["visual_sanity_flag"] = "reviewable_auto"
    out.loc[out["visual_sanity_score"] < 0.45, "visual_sanity_flag"] = "artifact_risk_auto"
    out.loc[
        (out["visual_sanity_score"] >= 0.45) & (out["visual_sanity_score"] < 0.60),
        "visual_sanity_flag",
    ] = "uncertain_auto"
    return out


def audit_row(row: pd.Series) -> Dict[str, Any]:
    frames = load_frames(str(row.get("npz_path", "")))
    rec = row.to_dict()
    if frames is None:
        rec["audit_status"] = "missing_or_invalid_npz"
        return rec
    base_mask = stable_mask(frames)
    masks = [frame_mask(frame, base_mask) for frame in frames]
    cents = np.array([centroid(m) for m in masks], dtype=float)
    good = np.isfinite(cents).all(axis=1)
    if good.sum() >= 2:
        steps = np.sqrt(np.sum(np.diff(cents[good], axis=0) ** 2, axis=1))
        centroid_path = float(np.sum(steps))
        centroid_max_step = float(np.max(steps)) if len(steps) else 0.0
    else:
        centroid_path = np.nan
        centroid_max_step = np.nan
    focus = np.array([laplacian_variance(f) for f in frames], dtype=float)
    kymo, centers = radial_kymograph(frames, base_mask)
    trace = front_trace(kymo, centers)
    fit = slope_fit(trace)
    rec.update({
        "audit_status": "ok",
        "stable_mask_fraction": float(base_mask.mean()),
        "stable_mask_edge_fraction": edge_fraction(base_mask),
        "centroid_path_px": centroid_path,
        "centroid_max_step_px": centroid_max_step,
        "focus_laplacian_var_median": float(np.nanmedian(focus)),
        "focus_laplacian_var_iqr": float(np.nanpercentile(focus, 75) - np.nanpercentile(focus, 25)),
        "temporal_abs_delta_mean": float(np.nanmean(np.abs(frames[-1] - frames[0]))),
        "temporal_abs_diff_mean": float(np.nanmean(np.abs(np.diff(frames, axis=0)))),
        "front_trace_slope_px_per_norm_time": fit["slope"],
        "front_trace_r2": fit["r2"],
        "front_trace_monotonic_fraction": fit["monotonic_fraction"],
        "front_trace_first_px": float(trace[np.isfinite(trace)][0]) if np.isfinite(trace).any() else np.nan,
        "front_trace_last_px": float(trace[np.isfinite(trace)][-1]) if np.isfinite(trace).any() else np.nan,
        "n_frames_audited": int(frames.shape[0]),
    })
    return rec


def event_tests(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    bins = df["event_relative_bin"].astype(str)
    near = bins.eq("near_pre_event_1_8")
    for comparison, control_mask in [
        ("near_vs_rendered_non_near", ~near),
        ("near_vs_rendered_post_control", bins.isin(["post_event_1_16", "no_near_event_control"])),
    ]:
        for feature in features:
            t = numeric(df, feature).loc[near].dropna()
            c = numeric(df, feature).loc[control_mask].dropna()
            row = {
                "comparison": comparison,
                "feature": feature,
                "n_near": int(len(t)),
                "n_control": int(len(c)),
                "near_median": np.nan,
                "control_median": np.nan,
                "near_minus_control_median": np.nan,
                "mannwhitney_p": np.nan,
            }
            if len(t) >= 4 and len(c) >= 3 and pd.concat([t, c]).nunique() > 1:
                _, p_val = mannwhitneyu(t, c, alternative="two-sided")
                row.update({
                    "near_median": float(t.median()),
                    "control_median": float(c.median()),
                    "near_minus_control_median": float(t.median() - c.median()),
                    "mannwhitney_p": float(p_val),
                })
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["comparison", "mannwhitney_p"]) if not out.empty else out


def write_readme(out: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Pre-Event Visual Sanity Audit",
        "",
        f"- Audited/rendered candidates: {summary['n_audited']} / {summary['n_ok']}",
        f"- Visual sanity flags: {summary['visual_sanity_flag_counts']}",
        f"- Sources represented: {summary['n_sources']}",
        "",
        "## Top Reviewable Candidates",
    ]
    for row in summary.get("top_reviewable_candidates", [])[:10]:
        lines.append(
            f"- rank {row['consensus_review_rank']} {row['roi_id']} {row['event_relative_bin']} "
            f"sanity={row['visual_sanity_score']:.3f}, front_r2={row.get('front_trace_r2')}, "
            f"mask_frac={row.get('stable_mask_fraction')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument(
        "--out-dir",
        default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_visual_sanity_audit",
    )
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    assets = read_csv(
        derived
        / "source_balanced_pre_event_consensus_visual_packet"
        / "source_balanced_pre_event_consensus_visual_assets.csv"
    )
    rows = [audit_row(row) for _, row in assets.iterrows()]
    df = score_rows(pd.DataFrame(rows))
    tests = event_tests(df[df["audit_status"].eq("ok")], [
        "visual_sanity_score",
        "stable_mask_fraction",
        "stable_mask_edge_fraction",
        "centroid_path_px",
        "focus_laplacian_var_median",
        "front_trace_slope_px_per_norm_time",
        "front_trace_r2",
        "front_trace_monotonic_fraction",
    ])
    source_summary = (
        df.groupby("source_stem", dropna=False)
        .agg(
            n_candidates=("roi_id", "count"),
            median_visual_sanity_score=("visual_sanity_score", "median"),
            n_reviewable=("visual_sanity_flag", lambda x: int((x == "reviewable_auto").sum())),
            n_near_pre=("event_relative_bin", lambda x: int((x.astype(str) == "near_pre_event_1_8").sum())),
        )
        .reset_index()
        .sort_values(["n_candidates", "median_visual_sanity_score"], ascending=[False, False])
    )

    table_path = out / "source_balanced_pre_event_visual_sanity_metrics.csv"
    tests_path = out / "source_balanced_pre_event_visual_sanity_event_tests.csv"
    source_path = out / "source_balanced_pre_event_visual_sanity_source_summary.csv"
    summary_path = out / "source_balanced_pre_event_visual_sanity_summary.json"
    df.to_csv(table_path, index=False)
    tests.to_csv(tests_path, index=False)
    source_summary.to_csv(source_path, index=False)

    ok = df[df["audit_status"].eq("ok")].copy()
    top_reviewable = (
        ok.sort_values(["visual_sanity_score", "consensus_review_score"], ascending=[False, False])
        .head(16)
        .to_dict("records")
    )
    top_artifact = (
        ok.sort_values(["visual_sanity_score", "consensus_review_score"], ascending=[True, False])
        .head(12)
        .to_dict("records")
    )
    summary: Dict[str, Any] = {
        "n_audited": int(len(df)),
        "n_ok": int(len(ok)),
        "n_sources": int(ok["source_stem"].nunique()) if not ok.empty else 0,
        "event_relative_bin_counts": clean_json(ok["event_relative_bin"].astype(str).value_counts().to_dict()) if not ok.empty else {},
        "visual_sanity_flag_counts": clean_json(ok["visual_sanity_flag"].value_counts().to_dict()) if not ok.empty else {},
        "median_visual_sanity_score": float(ok["visual_sanity_score"].median()) if not ok.empty else None,
        "source_summary": clean_json(source_summary.to_dict("records")),
        "top_reviewable_candidates": clean_json(top_reviewable),
        "top_artifact_risk_candidates": clean_json(top_artifact),
        "event_tests": clean_json(tests.to_dict("records")),
        "outputs": {
            "metrics": str(table_path),
            "event_tests": str(tests_path),
            "source_summary": str(source_path),
            "summary": str(summary_path),
        },
        "guardrail": (
            "Visual sanity metrics are automatic crop/front quality proxies. They can prioritize manual QC and flag "
            "artifact risk, but they are not manual labels, particle-identity validation, phase-boundary validation, "
            "calibrated diffusion, or degradation causality evidence."
        ),
    }
    summary = clean_json(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
