#!/usr/bin/env python3
"""Physics atlas for echem-conditioned residual-dictionary modes.

The echem-conditioned residual dictionary audit shows that split-specific video
residual modes can recover future16 signal after subtracting echem/acquisition
predictable structure. This script asks what those residual modes look like
physically by correlating each conditioned residual basis with front, diffusion,
rollout, particle, mask, echem, degradation, and consensus descriptors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS = ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"]
SPLITS = ["leave_cycle", "leave_source"]

EXCLUDE_SUBSTRINGS = [
    "__echem_pred_",
    "__echem_conditioned_resid_",
]

CATEGORY_PATTERNS = {
    "front_phase_diffusion": ["phase", "radius2", "diffusion", "threshold_robust", "front_motion", "front_"],
    "rollout_prediction": ["persistence_", "low_rank_dmd", "velocity_", "rollout", "mse_", "residual_energy", "dictionary_recon"],
    "particle_optical": ["particle_mean", "particle_std", "particle_gradient", "particle_vs_context", "roi_mean", "roi_norm", "object_mean"],
    "mask_qc": ["mask_", "stage_drift", "fragmented", "centroid", "area_fraction"],
    "echem_state": ["capacity", "coulombic", "dqdv", "dq_", "shape_", "cycle_state", "echem_", "state_step", "axis_step", "signed_charge"],
    "degradation_trace": ["future_", "drop", "degradation", "hazard", "consensus", "integrated_", "trace_"],
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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def source_center(series: pd.Series, sources: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals - vals.groupby(sources).transform("mean")


def source_eta2(series: pd.Series, sources: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    valid = vals.notna() & sources.notna()
    vals = vals[valid]
    src = sources[valid]
    if vals.nunique() < 2 or src.nunique() < 2:
        return np.nan
    overall = vals.mean()
    total = float(((vals - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in vals.groupby(src):
        between += len(sub) * float((sub.mean() - overall) ** 2)
    return between / total


def spearman_pair(a: pd.Series, b: pd.Series) -> Tuple[float, float, int]:
    valid = a.notna() & b.notna()
    aa = a[valid]
    bb = b[valid]
    if len(aa) < 20 or aa.nunique() < 2 or bb.nunique() < 2:
        return np.nan, np.nan, int(len(aa))
    rho, p = spearmanr(aa, bb)
    return (float(rho) if np.isfinite(rho) else np.nan, float(p) if np.isfinite(p) else np.nan, int(len(aa)))


def orient_score(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, str]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
        return x, "NA"
    if xx[yy == 1].median() >= xx[yy == 0].median():
        return x, "higher_in_positive"
    return -x, "lower_in_positive"


def target_metric(df: pd.DataFrame, residual_col: str, target: str) -> Dict[str, Any]:
    y = numeric(df, target)
    x = numeric(df, residual_col)
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    score, direction = orient_score(yy, xx)
    score = score[valid]
    row: Dict[str, Any] = {
        "residual_feature": residual_col,
        "target": target,
        "n": int(len(yy)),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "direction": direction,
        "oriented_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "source_eta2": source_eta2(xx, df.loc[valid, "source_stem"]) if len(yy) else np.nan,
    }
    if len(yy) >= 8 and yy.nunique() == 2 and score.nunique() > 1:
        row["oriented_auc"] = float(roc_auc_score(yy, score))
        row["average_precision"] = float(average_precision_score(yy, score))
        rho, p = spearmanr(yy, score)
        row["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
        row["spearman_p"] = float(p) if np.isfinite(p) else np.nan
    return row


def category_for(col: str) -> str:
    low = col.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(pat in low for pat in patterns):
            return category
    return "other_numeric"


def candidate_physics_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if any(s in col for s in EXCLUDE_SUBSTRINGS):
            continue
        if col in {"cycleNo"} or col in TARGETS:
            continue
        vals = numeric(df, col)
        if vals.notna().sum() < 30 or vals.nunique(dropna=True) < 3:
            continue
        category = category_for(col)
        if category != "other_numeric":
            cols.append(col)
    return cols


def residual_base(col: str, split: str) -> str:
    suffix = f"__echem_conditioned_resid_{split}"
    return col[:-len(suffix)] if col.endswith(suffix) else col


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/conditioned_residual_physics_atlas")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_csv(derived / "echem_conditioned_residual_dictionary" / "echem_conditioned_residual_dictionary_features.csv")
    physics_cols = candidate_physics_columns(df)

    alignment_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    mode_rows: List[Dict[str, Any]] = []

    for split in SPLITS:
        residual_cols = [c for c in df.columns if c.endswith(f"__echem_conditioned_resid_{split}")]
        for rcol in residual_cols:
            rx = numeric(df, rcol)
            rcenter = source_center(rx, df["source_stem"])
            for target in [t for t in TARGETS if t in df.columns]:
                row = target_metric(df, rcol, target)
                row["split"] = split
                row["residual_base"] = residual_base(rcol, split)
                target_rows.append(row)
            best_by_category: Dict[str, Dict[str, Any]] = {}
            for pcol in physics_cols:
                px = numeric(df, pcol)
                rho, p, n = spearman_pair(rx, px)
                crho, cp, cn = spearman_pair(rcenter, source_center(px, df["source_stem"]))
                if not np.isfinite(rho) and not np.isfinite(crho):
                    continue
                category = category_for(pcol)
                row = {
                    "split": split,
                    "residual_feature": rcol,
                    "residual_base": residual_base(rcol, split),
                    "physics_feature": pcol,
                    "physics_category": category,
                    "n": n,
                    "spearman_rho": rho,
                    "spearman_p": p,
                    "source_centered_n": cn,
                    "source_centered_rho": crho,
                    "source_centered_p": cp,
                    "abs_spearman_rho": abs(rho) if np.isfinite(rho) else np.nan,
                    "abs_source_centered_rho": abs(crho) if np.isfinite(crho) else np.nan,
                    "residual_source_eta2": source_eta2(rx, df["source_stem"]),
                    "physics_source_eta2": source_eta2(px, df["source_stem"]),
                }
                alignment_rows.append(row)
                score = row["abs_source_centered_rho"] if np.isfinite(row["abs_source_centered_rho"]) else row["abs_spearman_rho"]
                prev = best_by_category.get(category)
                prev_score = -np.inf if prev is None else prev.get("category_score", -np.inf)
                if np.isfinite(score) and score > prev_score:
                    row2 = dict(row)
                    row2["category_score"] = score
                    best_by_category[category] = row2
            tsub = [r for r in target_rows if r["residual_feature"] == rcol]
            tsub16 = [z for z in tsub if z.get("target") == "future_any_drop_within_16cycles"]
            target_pool = tsub16 if tsub16 else tsub
            best_target = max(target_pool, key=lambda z: z.get("oriented_auc") if np.isfinite(z.get("oriented_auc", np.nan)) else -np.inf) if target_pool else {}
            interpretable_phys = [v for v in best_by_category.values() if v.get("physics_category") != "degradation_trace"]
            best_phys = max(interpretable_phys, key=lambda z: z.get("category_score", -np.inf)) if interpretable_phys else {}
            mode_rows.append({
                "split": split,
                "residual_feature": rcol,
                "residual_base": residual_base(rcol, split),
                "residual_source_eta2": source_eta2(rx, df["source_stem"]),
                "best_target": best_target.get("target"),
                "best_target_auc": best_target.get("oriented_auc"),
                "best_target_ap": best_target.get("average_precision"),
                "best_target_direction": best_target.get("direction"),
                "best_physics_feature": best_phys.get("physics_feature"),
                "best_physics_category": best_phys.get("physics_category"),
                "best_physics_rho": best_phys.get("spearman_rho"),
                "best_physics_source_centered_rho": best_phys.get("source_centered_rho"),
                "n_physics_categories_with_abs_centered_rho_ge_0p30": int(sum(
                    1 for v in best_by_category.values()
                    if np.isfinite(v.get("abs_source_centered_rho", np.nan)) and v["abs_source_centered_rho"] >= 0.30
                )),
            })

    alignment = pd.DataFrame(alignment_rows)
    targets = pd.DataFrame(target_rows)
    modes = pd.DataFrame(mode_rows)
    if not alignment.empty:
        alignment = alignment.sort_values(["abs_source_centered_rho", "abs_spearman_rho"], ascending=False)
    if not targets.empty:
        targets = targets.sort_values(["target", "split", "oriented_auc"], ascending=[True, True, False])
    if not modes.empty:
        modes = modes.sort_values(["best_target_auc", "n_physics_categories_with_abs_centered_rho_ge_0p30"], ascending=False)

    category_summary = pd.DataFrame()
    if not alignment.empty:
        category_summary = (
            alignment.assign(strong_centered=lambda d: pd.to_numeric(d["abs_source_centered_rho"], errors="coerce") >= 0.30)
            .groupby(["split", "physics_category"], dropna=False)
            .agg(
                n_pairs=("physics_feature", "count"),
                n_strong_centered=("strong_centered", "sum"),
                max_abs_source_centered_rho=("abs_source_centered_rho", "max"),
                max_abs_raw_rho=("abs_spearman_rho", "max"),
            )
            .reset_index()
            .sort_values(["split", "max_abs_source_centered_rho"], ascending=[True, False])
        )

    paths = {
        "alignment": out / "conditioned_residual_physics_alignments.csv",
        "target_tests": out / "conditioned_residual_target_tests.csv",
        "mode_summary": out / "conditioned_residual_mode_summary.csv",
        "category_summary": out / "conditioned_residual_category_summary.csv",
        "summary": out / "conditioned_residual_physics_atlas_summary.json",
    }
    alignment.to_csv(paths["alignment"], index=False)
    targets.to_csv(paths["target_tests"], index=False)
    modes.to_csv(paths["mode_summary"], index=False)
    category_summary.to_csv(paths["category_summary"], index=False)

    summary = clean_json({
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "n_physics_features": int(len(physics_cols)),
        "physics_category_counts": pd.Series([category_for(c) for c in physics_cols]).value_counts().to_dict(),
        "n_conditioned_residual_modes_per_split": {s: int(sum(c.endswith(f"__echem_conditioned_resid_{s}") for c in df.columns)) for s in SPLITS},
        "top_source_centered_alignments": alignment.head(30).to_dict("records") if not alignment.empty else [],
        "top_interpretable_source_centered_alignments": alignment[alignment["physics_category"] != "degradation_trace"].head(30).to_dict("records") if not alignment.empty else [],
        "top_target_tests": targets.head(30).to_dict("records") if not targets.empty else [],
        "top_modes": modes.head(30).to_dict("records") if not modes.empty else [],
        "category_summary": category_summary.head(30).to_dict("records") if not category_summary.empty else [],
        "guardrail": "Conditioned residual modes are split-specific, label-free video residual features; source-centered correlations reduce source/acquisition confounding but do not prove causal physics, calibrated diffusion, or deployable warning performance.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Conditioned Residual Physics Atlas",
        "",
        "Maps echem-conditioned residual-dictionary modes onto interpretable physics, rollout, particle, mask, echem, and degradation descriptors.",
        "",
        f"- Rows: {summary['n_rows']}",
        f"- Cycles: {summary['n_cycles']}",
        f"- Sources: {summary['n_sources']}",
        f"- Physics descriptor columns screened: {summary['n_physics_features']}",
        f"- Conditioned residual modes per split: {summary['n_conditioned_residual_modes_per_split']}",
        "",
        "## Top Source-Centered Alignments",
        "",
    ]
    for row in summary["top_interpretable_source_centered_alignments"][:12]:
        lines.append(
            f"- {row.get('split')} {row.get('residual_base')} vs {row.get('physics_feature')} ({row.get('physics_category')}): centered rho {row.get('source_centered_rho')}, raw rho {row.get('spearman_rho')}"
        )
    lines += ["", "## Top Target Tests", ""]
    for row in summary["top_target_tests"][:12]:
        lines.append(
            f"- {row.get('split')} {row.get('target')} {row.get('residual_base')}: AUC {row.get('oriented_auc')}, AP {row.get('average_precision')}, direction {row.get('direction')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
