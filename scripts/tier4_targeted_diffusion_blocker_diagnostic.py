#!/usr/bin/env python3
"""Targeted diagnostic for diffusion-blocker candidates.

The diffusion readiness audit identifies the nearest candidates to a calibrated
diffusion claim, but it intentionally stays at go/no-go accounting. This script
rechecks those candidate ROIs against threshold-sweep front metrics and local
same-source/cycle reference groups to decide whether the blocker is likely a
single-ROI remeasurement problem, a threshold-fit problem, or a broader
source/context pattern.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


THRESHOLD_DIRS = [
    "balanced_future_threshold_robust_fronts",
    "transfer_ranked_threshold_robust_fronts",
    "multi_cycle_threshold_robust_fronts",
]

TARGET_ACTIONS = {
    "manual_qc_diffusion_blocker_followup",
    "manual_qc_transfer_ranked_roi",
    "manual_qc_transport_mechanism_roi",
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
        return pd.DataFrame()
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def percentile_rank(values: pd.Series, value: Any) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(x) or len(vals) == 0:
        return float("nan")
    return float((np.sum(vals < x) + 0.5 * np.sum(vals == x)) / len(vals))


def first_value(df: pd.DataFrame, col: str, default: Any = np.nan) -> Any:
    if col not in df.columns or df.empty:
        return default
    vals = df[col].dropna()
    if vals.empty:
        return default
    return vals.iloc[0]


def load_thresholds(derived: Path) -> pd.DataFrame:
    frames = []
    for name in THRESHOLD_DIRS:
        path = derived / name / "threshold_sweep_per_roi.csv"
        df = read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["threshold_cohort"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["roi_id"] = out["roi_id"].astype(str)
    out["cycleNo"] = num(out, "cycleNo")
    out["source_stem"] = out.get("source_stem", pd.Series("", index=out.index)).astype(str)
    return out


def per_roi_threshold_summary(th: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for roi_id, grp in th.groupby("roi_id", dropna=False):
        d = num(grp, "apparent_diffusion_proxy_um2_per_s")
        r2 = num(grp, "radius2_slope_r2")
        radius_slope = num(grp, "radius2_slope_px2_per_s")
        phase_slope = num(grp, "phase_fraction_slope_per_s")
        q = num(grp, "threshold_quantile")
        positive = d > 0
        best_pos = grp.loc[(positive & r2.notna()).fillna(False)].copy()
        if not best_pos.empty:
            best_pos = best_pos.sort_values("radius2_slope_r2", ascending=False).iloc[0]
        else:
            best_pos = pd.Series(dtype=object)
        q70 = grp.iloc[(q - 0.70).abs().argsort().iloc[:1]] if q.notna().any() else pd.DataFrame()
        q80 = grp.iloc[(q - 0.80).abs().argsort().iloc[:1]] if q.notna().any() else pd.DataFrame()
        rows.append(
            {
                "roi_id": roi_id,
                "cycleNo": first_value(grp, "cycleNo"),
                "source_stem": first_value(grp, "source_stem", ""),
                "cohort_role": first_value(grp, "cohort_role", ""),
                "threshold_cohort": first_value(grp, "threshold_cohort", ""),
                "n_threshold_rows": int(len(grp)),
                "threshold_min": float(q.min()) if q.notna().any() else np.nan,
                "threshold_max": float(q.max()) if q.notna().any() else np.nan,
                "median_D_um2_per_s": float(d.median()) if d.notna().any() else np.nan,
                "q70_D_um2_per_s": first_value(q70, "apparent_diffusion_proxy_um2_per_s"),
                "q80_D_um2_per_s": first_value(q80, "apparent_diffusion_proxy_um2_per_s"),
                "positive_D_fraction": float(positive.mean()) if d.notna().any() else np.nan,
                "median_abs_D_um2_per_s": float(d.abs().median()) if d.notna().any() else np.nan,
                "threshold_D_iqr": float(d.quantile(0.75) - d.quantile(0.25)) if d.notna().sum() >= 3 else np.nan,
                "median_radius2_slope_px2_per_s": float(radius_slope.median()) if radius_slope.notna().any() else np.nan,
                "q70_radius2_slope_px2_per_s": first_value(q70, "radius2_slope_px2_per_s"),
                "median_radius2_fit_r2": float(r2.median()) if r2.notna().any() else np.nan,
                "q70_radius2_fit_r2": first_value(q70, "radius2_slope_r2"),
                "max_positive_fit_r2": first_value(best_pos.to_frame().T, "radius2_slope_r2"),
                "best_positive_threshold_quantile": first_value(best_pos.to_frame().T, "threshold_quantile"),
                "best_positive_D_um2_per_s": first_value(best_pos.to_frame().T, "apparent_diffusion_proxy_um2_per_s"),
                "median_phase_slope_per_s": float(phase_slope.median()) if phase_slope.notna().any() else np.nan,
                "phase_slope_positive_fraction": float((phase_slope > 0).mean()) if phase_slope.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_reference_percentiles(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    metrics = [
        "median_D_um2_per_s",
        "q70_D_um2_per_s",
        "median_radius2_fit_r2",
        "q70_radius2_fit_r2",
        "positive_D_fraction",
        "median_phase_slope_per_s",
    ]
    for metric in metrics:
        out[f"{metric}_all_percentile"] = [percentile_rank(out[metric], v) for v in out[metric]]
        out[f"{metric}_same_source_percentile"] = [
            percentile_rank(out.loc[out["source_stem"].eq(src), metric], v)
            for src, v in zip(out["source_stem"], out[metric])
        ]
        out[f"{metric}_same_cycle_percentile"] = [
            percentile_rank(out.loc[out["cycleNo"].eq(cyc), metric], v)
            for cyc, v in zip(out["cycleNo"], out[metric])
        ]
    return out


def target_roi_ids(readiness: pd.DataFrame, queue: pd.DataFrame) -> set[str]:
    ids: set[str] = set()
    if not readiness.empty:
        ranked = readiness.sort_values("review_priority", ascending=False).head(20)
        ids |= set(ranked["roi_id"].dropna().astype(str))
        ids |= set(ranked.get("selected_roi_id", pd.Series(dtype=str)).dropna().astype(str))
    if not queue.empty:
        action = queue.get("recommended_action", pd.Series("", index=queue.index)).astype(str)
        ranked = queue.loc[action.isin(TARGET_ACTIONS)].copy()
        if "priority_score" in ranked.columns:
            ranked = ranked.sort_values("priority_score", ascending=False)
        ids |= set(ranked.head(40)["roi_id"].dropna().astype(str))
    return ids


def classify(row: pd.Series) -> tuple[str, str]:
    blockers = str(row.get("blockers", ""))
    max_r2 = row.get("max_positive_fit_r2")
    q70_r2 = row.get("q70_radius2_fit_r2")
    same_source_d = row.get("median_D_um2_per_s_same_source_percentile")
    pos_frac = row.get("positive_D_fraction")
    auto_ok = bool(row.get("automatic_physics_consistent", False))

    fit_blocked = (pd.isna(max_r2) or max_r2 < 0.5) and (pd.isna(q70_r2) or q70_r2 < 0.35)
    if "q70 positive" in blockers and auto_ok and not fit_blocked:
        return "remeasure_q70_ci_and_manual_front_qc", "nearest candidate; internal gates mostly pass but q70 CI/manual publication gate blocks"
    if auto_ok and pd.notna(same_source_d) and same_source_d >= 0.75 and pd.notna(pos_frac) and pos_frac >= 0.7:
        return "manual_front_qc_candidate_specific", "positive threshold behavior is high within source; manual front trace review is the next blocker"
    if fit_blocked:
        return "fit_quality_blocked_retrack_front", "best positive radius2 fits remain weak; needs manual/retracked front boundary before diffusion wording"
    if pd.notna(same_source_d) and same_source_d < 0.6:
        return "source_context_common_not_candidate_specific", "diffusion proxy is not exceptional within its source; avoid candidate-specific diffusion claim"
    return "guarded_review_only", "use as optical-front proxy review row only"


def build_diagnostic(derived: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    readiness = read_csv(derived / "diffusion_claim_readiness_audit" / "diffusion_claim_readiness_candidates.csv")
    queue = read_csv(derived / "targeted_densification_qc_plan" / "targeted_manual_qc_roi_queue.csv")
    physics = read_csv(derived / "diffusion_physics_consistency_audit" / "diffusion_physics_consistency_roi_scores.csv")
    thresholds = load_thresholds(derived)
    if thresholds.empty:
        raise FileNotFoundError("No threshold_sweep_per_roi.csv files found")

    per_roi = add_reference_percentiles(per_roi_threshold_summary(thresholds))
    ids = target_roi_ids(readiness, queue)
    diag = per_roi.loc[per_roi["roi_id"].isin(ids)].copy()

    readiness_cols = [
        "roi_id",
        "cycleNo",
        "source_stem",
        "cohort_role",
        "candidate_source",
        "automatic_physics_consistent",
        "publication_ready",
        "gate_count",
        "h5_dt_max_to_median_ratio",
        "manual_qc_status",
        "blockers",
        "review_priority",
    ]
    if not readiness.empty:
        rsmall = readiness[[c for c in readiness_cols if c in readiness.columns]].copy()
        rename = {c: f"{c}_readiness" for c in ["cycleNo", "source_stem", "cohort_role"] if c in rsmall.columns}
        rsmall = rsmall.rename(columns=rename)
        diag = diag.merge(rsmall, on="roi_id", how="left")
    physics_cols = [
        "roi_id",
        "cycleNo",
        "source_stem",
        "cohort_role",
        "q70_radius2_slope_bootstrap_p05_px2_per_s",
        "q70_radius2_slope_bootstrap_p95_px2_per_s",
        "gate_q70_positive_ci",
        "gate_h5_timing_stable",
        "diffusion_physics_gate_count",
        "physics_consistency_score",
    ]
    if not physics.empty:
        psmall = physics[[c for c in physics_cols if c in physics.columns]].copy()
        rename = {c: f"{c}_physics" for c in ["cycleNo", "source_stem", "cohort_role"] if c in psmall.columns}
        psmall = psmall.rename(columns=rename)
        diag = diag.merge(psmall, on="roi_id", how="left")

    if not queue.empty:
        qcols = [
            "roi_id",
            "cycleNo",
            "source_stem",
            "cohort_role",
            "origin",
            "recommended_action",
            "priority_score",
            "has_visual_assets",
            "event_relative_bin",
            "cycles_to_next_event",
        ]
        qsmall = queue[[c for c in qcols if c in queue.columns]].drop_duplicates("roi_id")
        rename = {c: f"{c}_queue" for c in ["cycleNo", "source_stem", "cohort_role"] if c in qsmall.columns}
        qsmall = qsmall.rename(columns=rename)
        diag = diag.merge(qsmall, on="roi_id", how="left")

    for col in ["source_stem", "cohort_role"]:
        if col not in diag.columns:
            diag[col] = ""
        base = diag[col].astype(str)
        for suffix in ["_readiness", "_physics", "_queue"]:
            alt = f"{col}{suffix}"
            if alt in diag.columns:
                alt_vals = diag[alt].astype(str)
                base = base.where(base.ne("") & base.ne("nan"), alt_vals)
        diag[col] = base.where(base.ne("nan"), "")
    if "cycleNo" in diag.columns:
        base_cycle = pd.to_numeric(diag["cycleNo"], errors="coerce")
        for alt in ["cycleNo_readiness", "cycleNo_physics", "cycleNo_queue"]:
            if alt in diag.columns:
                base_cycle = base_cycle.fillna(pd.to_numeric(diag[alt], errors="coerce"))
        diag["cycleNo"] = base_cycle

    diag = add_reference_percentiles(diag)

    actions = diag.apply(classify, axis=1)
    diag["diagnostic_action"] = [a for a, _ in actions]
    diag["diagnostic_reason"] = [b for _, b in actions]
    diag["targeted_remeasurement_score"] = (
        pd.to_numeric(diag.get("review_priority"), errors="coerce").fillna(0.0)
        + 10.0 * pd.to_numeric(diag.get("automatic_physics_consistent"), errors="coerce").fillna(0.0)
        + 5.0 * pd.to_numeric(diag.get("positive_D_fraction"), errors="coerce").fillna(0.0)
        + 3.0 * pd.to_numeric(diag.get("median_D_um2_per_s_same_source_percentile"), errors="coerce").fillna(0.0)
        + 2.0 * pd.to_numeric(diag.get("max_positive_fit_r2"), errors="coerce").fillna(0.0)
    )
    diag = diag.sort_values("targeted_remeasurement_score", ascending=False)

    target_thresholds = thresholds.loc[thresholds["roi_id"].isin(set(diag["roi_id"].astype(str)))].copy()
    queue_cols = [
        "roi_id",
        "cycleNo",
        "source_stem",
        "cohort_role",
        "diagnostic_action",
        "targeted_remeasurement_score",
        "median_D_um2_per_s",
        "q70_D_um2_per_s",
        "positive_D_fraction",
        "q70_radius2_fit_r2",
        "max_positive_fit_r2",
        "best_positive_threshold_quantile",
        "q70_radius2_slope_bootstrap_p05_px2_per_s",
        "q70_radius2_slope_bootstrap_p95_px2_per_s",
        "blockers",
        "diagnostic_reason",
    ]
    remeasure = diag[[c for c in queue_cols if c in diag.columns]].head(20)
    return diag, target_thresholds, remeasure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/targeted_diffusion_blocker_diagnostic")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    diagnostic, threshold_variants, remeasure = build_diagnostic(derived)
    action_counts = diagnostic["diagnostic_action"].value_counts().rename_axis("diagnostic_action").reset_index(name="n")

    paths = {
        "diagnostic_table": out / "targeted_diffusion_candidate_diagnostic.csv",
        "threshold_variants": out / "targeted_diffusion_threshold_variants.csv",
        "remeasurement_queue": out / "targeted_diffusion_remeasurement_queue.csv",
        "action_counts": out / "targeted_diffusion_diagnostic_action_counts.csv",
        "summary": out / "targeted_diffusion_blocker_diagnostic_summary.json",
        "readme": out / "README.md",
    }
    diagnostic.to_csv(paths["diagnostic_table"], index=False)
    threshold_variants.to_csv(paths["threshold_variants"], index=False)
    remeasure.to_csv(paths["remeasurement_queue"], index=False)
    action_counts.to_csv(paths["action_counts"], index=False)

    top = diagnostic.head(12).to_dict("records")
    summary = clean_json(
        {
            "overall_status": "targeted_diffusion_blocker_diagnostic_ready",
            "n_target_candidates_with_thresholds": int(len(diagnostic)),
            "n_threshold_variant_rows": int(len(threshold_variants)),
            "action_counts": action_counts.to_dict("records"),
            "top_remeasurement_candidates": top,
            "nearest_diffusion_candidate": top[0] if top else {},
            "guardrail": "This diagnostic ranks diffusion-blocker follow-up candidates using existing automatic threshold/front tables. It does not accept manual labels, relax readiness gates, or emit calibrated diffusion coefficients.",
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Targeted Diffusion Blocker Diagnostic",
        "",
        "Rechecks diffusion-readiness candidates against threshold-sweep front metrics and same-source/cycle reference groups.",
        "",
        f"- Target candidates with threshold rows: {summary['n_target_candidates_with_thresholds']}",
        f"- Threshold variant rows: {summary['n_threshold_variant_rows']}",
        "",
        "## Diagnostic Actions",
        "",
    ]
    for row in summary["action_counts"]:
        lines.append(f"- {row['diagnostic_action']}: {row['n']}")
    lines += ["", "## Top Remeasurement Candidates", ""]
    for row in top[:8]:
        lines.append(
            f"- {row.get('roi_id')}: action {row.get('diagnostic_action')}, score "
            f"{row.get('targeted_remeasurement_score'):.3f}, positive-D fraction "
            f"{row.get('positive_D_fraction')}, q70 R2 {row.get('q70_radius2_fit_r2')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    paths["readme"].write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
