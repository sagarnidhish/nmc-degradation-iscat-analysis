#!/usr/bin/env python3
"""Bridge rollout difficulty to phase kinetics at the source/cycle level.

The first bridge attempt matched exact ROI ids and produced zero overlap because
the rollout and phase-kinetics audits were computed on different but related ROI
exports. This version aggregates both audits by source_stem and cycleNo and
then measures associations on that shared source/cycle grid.

Guardrail:
- This is still an association audit over automatic particle-only crops. It is
  intended for hypothesis ranking, not calibrated diffusion or manual phase
  labels.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
    except Exception:
        pass
    return value


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def group_numeric(df: pd.DataFrame, group_cols: List[str], cols: List[str]) -> pd.DataFrame:
    work = df[group_cols + cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    grouped = work.groupby(group_cols, dropna=False).median().reset_index()
    return grouped


def corr_table(df: pd.DataFrame, x_cols: List[str], y_cols: List[str]) -> pd.DataFrame:
    rows = []
    for x in x_cols:
        if x not in df.columns:
            continue
        xv = numeric(df, x)
        for y in y_cols:
            if y not in df.columns:
                continue
            yv = numeric(df, y)
            valid = xv.notna() & yv.notna()
            if valid.sum() < 6 or xv[valid].nunique() < 3 or yv[valid].nunique() < 3:
                continue
            rho, p = spearmanr(xv[valid], yv[valid])
            rows.append({
                "x": x,
                "y": y,
                "n": int(valid.sum()),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "abs_rho": float(abs(rho)) if np.isfinite(rho) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_rho", "p_value"], ascending=[False, True])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-csv", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_sequence_rollout_audit/source_balanced_sequence_rollout_features.csv")
    ap.add_argument("--phase-csv", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_phase_kinetics_audit/source_balanced_pre_event_phase_kinetics_features.csv")
    ap.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/rollout_phase_kinetics_bridge_v2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout = pd.read_csv(args.rollout_csv)
    phase = pd.read_csv(args.phase_csv)
    group_cols = ["source_stem", "cycleNo"]
    rollout_cols = [
        c for c in rollout.columns
        if c.startswith("persistence_mse_")
        or c.startswith("velocity_mse_")
        or c.startswith("temporal_energy_")
        or c.startswith("roi_norm_mean_")
        or c in ["raw_roi_mean_delta_last_minus_first", "stage_drift_xy_recomputed"]
    ]
    phase_cols = [
        c for c in phase.columns
        if "masked_minus_bg" in c
        or "phase_fraction" in c
        or "avrami" in c
        or "logistic" in c
        or "front_gradient" in c
        or "max_abs_rate" in c
        or "total_variation" in c
    ]
    rollout_grp = group_numeric(rollout, group_cols, rollout_cols)
    phase_grp = group_numeric(phase, group_cols, phase_cols)
    merged = rollout_grp.merge(phase_grp, on=group_cols, how="inner", suffixes=("_rollout", "_phase"))

    corr = corr_table(merged, rollout_cols, phase_cols)
    source_summary = merged.groupby("source_stem", dropna=False).agg(
        n_cycles=("cycleNo", "nunique"),
        n_rows=("cycleNo", "count"),
    ).reset_index()

    merged.to_csv(out_dir / "rollout_phase_kinetics_bridge_v2_merged.csv", index=False)
    corr.to_csv(out_dir / "rollout_phase_kinetics_bridge_v2_correlations.csv", index=False)
    source_summary.to_csv(out_dir / "rollout_phase_kinetics_bridge_v2_source_summary.csv", index=False)

    summary = {
        "n_merged_rows": int(len(merged)),
        "n_sources": int(merged["source_stem"].nunique()) if not merged.empty else 0,
        "n_group_rows_rollout": int(len(rollout_grp)),
        "n_group_rows_phase": int(len(phase_grp)),
        "rollout_feature_count": int(len(rollout_cols)),
        "phase_feature_count": int(len(phase_cols)),
        "top_correlation": corr.head(1).to_dict("records")[0] if not corr.empty else {},
        "guardrail": "Source/cycle-level association audit for physics-follow-up ranking, not diffusion calibration or manual phase labeling.",
        "outputs": {
            "merged": str(out_dir / "rollout_phase_kinetics_bridge_v2_merged.csv"),
            "correlations": str(out_dir / "rollout_phase_kinetics_bridge_v2_correlations.csv"),
            "source_summary": str(out_dir / "rollout_phase_kinetics_bridge_v2_source_summary.csv"),
        },
    }
    (out_dir / "rollout_phase_kinetics_bridge_v2_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    (out_dir / "README.md").write_text(
        "\n".join([
            "# Rollout / Phase Kinetics Bridge v2",
            "",
            f"- Merged source/cycle rows: {summary['n_merged_rows']}",
            f"- Sources: {summary['n_sources']}",
            "",
            f"- Top correlation: {summary['top_correlation']}",
            "",
            "## Guardrail",
            "",
            summary["guardrail"],
            "",
        ]) + "\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

