#!/usr/bin/env python3
"""Strict QC gate for pre-event front/diffusion candidate claims.

This audit combines the consensus review queue, automatic visual-sanity metrics,
and the visual-QC mode audit into explicit gates. It does not accept/reject
ROIs manually. Its purpose is to separate candidates worth manual front review
from candidates that remain ineligible for phase-boundary or diffusion claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


DERIVED = Path("/scratch/<account>/<username>/Alek_Jiho/derived")
OUT = DERIVED / "source_balanced_pre_event_strict_qc_gated_front_audit"


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


def num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def merge_inputs() -> pd.DataFrame:
    visual_assets = read_csv(
        DERIVED
        / "source_balanced_pre_event_consensus_visual_packet"
        / "source_balanced_pre_event_consensus_visual_assets.csv"
    )
    sanity = read_csv(
        DERIVED
        / "source_balanced_pre_event_visual_sanity_audit"
        / "source_balanced_pre_event_visual_sanity_metrics.csv"
    )
    modes = read_csv(
        DERIVED
        / "source_balanced_pre_event_visual_qc_modes"
        / "source_balanced_pre_event_visual_qc_modes.csv"
    )

    sanity_cols = [
        "roi_id",
        "visual_sanity_score",
        "visual_sanity_flag",
        "stable_mask_fraction",
        "stable_mask_edge_fraction",
        "centroid_path_px",
        "focus_laplacian_var_median",
        "front_trace_slope_px_per_norm_time",
        "front_trace_r2",
        "front_trace_monotonic_fraction",
    ]
    mode_cols = [
        "roi_id",
        "visual_review_score",
        "visual_qc_tier",
        "visual_mode_name",
        "visual_front_plausibility_score",
        "visual_artifact_risk_score",
        "front_trace_slope_px_per_norm_time",
        "front_trace_r2",
        "front_trace_direction_fraction",
        "front_trace_span_px",
        "front_trace_jitter_px",
        "masked_minus_background_slope",
    ]
    df = visual_assets.merge(
        sanity[[c for c in sanity_cols if c in sanity.columns]],
        on="roi_id",
        how="left",
        suffixes=("", "_sanity"),
    )
    df = df.merge(
        modes[[c for c in mode_cols if c in modes.columns]],
        on="roi_id",
        how="left",
        suffixes=("_sanity", "_mode"),
    )
    return df


def add_gates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stable_mask_fraction = num(out, "stable_mask_fraction")
    edge_fraction = num(out, "stable_mask_edge_fraction")
    centroid_path = num(out, "centroid_path_px")
    sanity_score = num(out, "visual_sanity_score")
    visual_review_score = num(out, "visual_review_score")
    artifact = num(out, "visual_artifact_risk_score")
    front_score = num(out, "visual_front_plausibility_score")
    sanity_r2 = num(out, "front_trace_r2_sanity")
    mode_r2 = num(out, "front_trace_r2_mode")
    sanity_slope = num(out, "front_trace_slope_px_per_norm_time_sanity")
    mode_slope = num(out, "front_trace_slope_px_per_norm_time_mode")
    consensus_slope = num(out, "front_radius_slope_px_per_norm_time")
    consensus_r2proxy = num(out, "front_radius2_slope_px2_per_norm_time")

    out["stable_mask_gate"] = stable_mask_fraction.between(0.08, 0.55) & (edge_fraction <= 0.06)
    out["centroid_stability_gate"] = centroid_path <= 40.0
    out["visual_sanity_gate"] = (sanity_score >= 0.60) & out["visual_sanity_flag"].astype(str).eq("reviewable_auto")
    out["visual_mode_gate"] = (
        out["visual_qc_tier"].astype(str).eq("front_plausible_followup")
        & (visual_review_score >= 0.48)
        & (artifact <= 0.25)
        & (front_score >= 0.24)
    )
    out["front_trace_fit_gate"] = pd.concat([sanity_r2, mode_r2], axis=1).max(axis=1) >= 0.05
    slope_agreement = (
        np.sign(mode_slope.fillna(sanity_slope))
        == np.sign(consensus_slope)
    )
    out["front_direction_agreement_gate"] = slope_agreement & consensus_slope.notna() & (consensus_slope.abs() > 0.05)
    out["manual_front_review_gate"] = (
        out["stable_mask_gate"]
        & out["centroid_stability_gate"]
        & out["visual_sanity_gate"]
        & out["visual_mode_gate"]
        & out["front_trace_fit_gate"]
    )

    # Diffusion claims need much stronger evidence than manual-review triage:
    # coherent front fit, directional agreement between independent traces and
    # scalar consensus, positive outward radius^2 signal, low artifact risk, and
    # near-pre context. These are automatic gates, not publication validation.
    out["automatic_diffusion_claim_gate"] = (
        out["manual_front_review_gate"]
        & out["front_direction_agreement_gate"]
        & (pd.concat([sanity_r2, mode_r2], axis=1).max(axis=1) >= 0.50)
        & (consensus_r2proxy > 0)
        & (artifact <= 0.10)
        & out["event_relative_bin"].astype(str).eq("near_pre_event_1_8")
    )
    gate_cols = [
        "stable_mask_gate",
        "centroid_stability_gate",
        "visual_sanity_gate",
        "visual_mode_gate",
        "front_trace_fit_gate",
        "front_direction_agreement_gate",
        "manual_front_review_gate",
        "automatic_diffusion_claim_gate",
    ]
    out["n_passed_component_gates"] = out[gate_cols[:-2]].sum(axis=1)
    out["strict_qc_priority_score"] = (
        0.22 * sanity_score.fillna(0)
        + 0.22 * visual_review_score.fillna(0)
        + 0.18 * front_score.fillna(0)
        + 0.16 * pd.concat([sanity_r2, mode_r2], axis=1).max(axis=1).fillna(0).clip(0, 1)
        + 0.14 * out["front_direction_agreement_gate"].astype(float)
        + 0.08 * (1.0 - artifact.fillna(1).clip(0, 1))
    )
    return out.sort_values(["manual_front_review_gate", "strict_qc_priority_score"], ascending=[False, False])


def make_contact_sheet(df: pd.DataFrame, out_path: Path) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    rows = df.sort_values("strict_qc_priority_score", ascending=False).head(12)
    panels = []
    for _, row in rows.iterrows():
        imgs = []
        for col in ["frame_strip_png", "mask_overlay_png", "kymograph_png"]:
            path = Path(str(row.get(col, "")))
            if path.exists():
                try:
                    imgs.append(Image.open(path).convert("RGB").resize((220, 120)))
                except Exception:
                    pass
        if not imgs:
            continue
        canvas = Image.new("RGB", (660, 172), "white")
        draw = ImageDraw.Draw(canvas)
        label = (
            f"rank {int(row.get('consensus_review_rank', 0))} "
            f"manual_gate={int(bool(row.get('manual_front_review_gate', False)))} "
            f"diff_gate={int(bool(row.get('automatic_diffusion_claim_gate', False)))} "
            f"score={row.get('strict_qc_priority_score', np.nan):.3f}"
        )
        draw.text((5, 5), label, fill=(0, 0, 0))
        draw.text((5, 23), str(row.get("roi_id", "NA"))[:110], fill=(0, 0, 0))
        for i, img in enumerate(imgs[:3]):
            canvas.paste(img, (220 * i, 52))
        panels.append(canvas)
    if not panels:
        return None
    sheet = Image.new("RGB", (660, 172 * len(panels)), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, 172 * i))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def summarize_failures(df: pd.DataFrame, gates: Iterable[str]) -> Dict[str, int]:
    return {gate: int((~df[gate].astype(bool)).sum()) for gate in gates}


def write_readme(summary: Dict[str, Any]) -> None:
    lines = [
        "# Source-Balanced Pre-Event Strict QC-Gated Front Audit",
        "",
        "Automatic stop/go gates for front and diffusion interpretation. These gates prioritize manual review and explicitly prevent automatic diffusion claims.",
        "",
        f"- Candidates scored: {summary['n_candidates']}",
        f"- Manual front-review candidates: {summary['n_manual_front_review_candidates']}",
        f"- Automatic diffusion-claim candidates: {summary['n_automatic_diffusion_claim_candidates']}",
        f"- Gate pass counts: {summary['gate_pass_counts']}",
        "",
        "## Top Manual-Review Candidates",
    ]
    for row in summary.get("top_manual_front_review_candidates", [])[:8]:
        lines.append(
            f"- rank {row.get('consensus_review_rank')} {row.get('roi_id')} "
            f"score={row.get('strict_qc_priority_score'):.3f}, "
            f"sanity={row.get('visual_sanity_score'):.3f}, visual_qc={row.get('visual_review_score'):.3f}"
        )
    lines += [
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_gates(merge_inputs())
    gates = [
        "stable_mask_gate",
        "centroid_stability_gate",
        "visual_sanity_gate",
        "visual_mode_gate",
        "front_trace_fit_gate",
        "front_direction_agreement_gate",
        "manual_front_review_gate",
        "automatic_diffusion_claim_gate",
    ]
    df.to_csv(OUT / "source_balanced_pre_event_strict_qc_gated_front_candidates.csv", index=False)
    gate_table = pd.DataFrame(
        [
            {
                "gate": gate,
                "n_pass": int(df[gate].astype(bool).sum()),
                "n_fail": int((~df[gate].astype(bool)).sum()),
                "fraction_pass": float(df[gate].astype(bool).mean()),
            }
            for gate in gates
        ]
    )
    gate_table.to_csv(OUT / "source_balanced_pre_event_strict_qc_gate_counts.csv", index=False)
    manual = df[df["manual_front_review_gate"]].copy()
    diffusion = df[df["automatic_diffusion_claim_gate"]].copy()
    contact_sheet = make_contact_sheet(df, OUT / "strict_qc_front_review_contact_sheet.png")
    summary = {
        "n_candidates": int(len(df)),
        "n_manual_front_review_candidates": int(len(manual)),
        "n_automatic_diffusion_claim_candidates": int(len(diffusion)),
        "gate_pass_counts": clean_json(gate_table.set_index("gate")["n_pass"].to_dict()),
        "gate_fail_counts": summarize_failures(df, gates),
        "top_manual_front_review_candidates": clean_json(
            manual.sort_values("strict_qc_priority_score", ascending=False).head(12).to_dict("records")
        ),
        "top_strict_qc_ranked_candidates": clean_json(
            df.sort_values("strict_qc_priority_score", ascending=False).head(12).to_dict("records")
        ),
        "diffusion_claim_candidates": clean_json(diffusion.to_dict("records")),
        "outputs": {
            "candidates": str(OUT / "source_balanced_pre_event_strict_qc_gated_front_candidates.csv"),
            "gate_counts": str(OUT / "source_balanced_pre_event_strict_qc_gate_counts.csv"),
            "contact_sheet": contact_sheet,
            "summary": str(OUT / "source_balanced_pre_event_strict_qc_gated_front_summary.json"),
        },
        "guardrail": (
            "Strict QC gates are automatic review-prioritization gates. They do not create manual labels, validate "
            "particle identity or front masks, calibrate diffusion, prove phase-boundary motion, or establish causality. "
            "Rows failing the diffusion gate must not be used for material diffusion coefficient claims."
        ),
    }
    (OUT / "source_balanced_pre_event_strict_qc_gated_front_summary.json").write_text(
        json.dumps(clean_json(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(summary)


if __name__ == "__main__":
    main()
