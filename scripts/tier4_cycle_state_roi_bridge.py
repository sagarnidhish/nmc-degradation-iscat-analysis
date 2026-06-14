#!/usr/bin/env python3
"""Bridge cycle-level degradation state-space signals to ROI/front physics rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


CYCLE_FEATURES = [
    "cycle_state_pc1",
    "cycle_state_pc2",
    "cycle_state_pc3",
    "cycle_state_pc4",
    "degradation_state_axis",
    "state_step_norm",
    "axis_step",
    "future_any_drop_within_8cycles",
    "cycle_state_cluster",
]

ROI_TARGETS = [
    "physics_consistency_score",
    "front_direction_score",
    "optical_change_score",
    "rollout_residual_score",
    "kinetic_transition_score",
    "precursor_context_score",
    "mode_taxonomy_score",
    "phase_slope_positive_fraction_protocol_residual_shape_residual",
    "threshold_robust_phase_score_protocol_residual_shape_residual",
    "diffusion_proxy_median_um2_per_s_protocol_residual_shape_residual",
    "diffusion_proxy_abs_median_um2_per_s_protocol_residual_shape_residual",
    "q70_transformed_fraction_delta_shape_residual",
    "mode_review_priority_shape_residual",
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def permutation_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_perm: int) -> tuple[float, float, float]:
    res = spearmanr(x, y)
    rho = float(res.statistic)
    p_asym = float(res.pvalue)
    if not np.isfinite(rho):
        return np.nan, np.nan, np.nan
    null = []
    for _ in range(n_perm):
        null.append(float(spearmanr(rng.permutation(x), y).statistic))
    null_arr = np.asarray(null)
    p_perm = float((np.sum(np.abs(null_arr) >= abs(rho)) + 1) / (len(null_arr) + 1))
    return rho, p_asym, p_perm


def spearman_tests(df: pd.DataFrame, predictors: Iterable[str], targets: Iterable[str], rng: np.random.Generator, n_perm: int, scope: str) -> pd.DataFrame:
    rows = []
    for pred in predictors:
        for target in targets:
            if pred not in df.columns or target not in df.columns:
                continue
            x = pd.to_numeric(df[pred], errors="coerce")
            y = pd.to_numeric(df[target], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 8 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                continue
            rho, p_asym, p_perm = permutation_spearman(x[mask].to_numpy(float), y[mask].to_numpy(float), rng, n_perm)
            rows.append(
                {
                    "scope": scope,
                    "predictor": pred,
                    "target": target,
                    "rho": rho,
                    "spearman_p": p_asym,
                    "permutation_p": p_perm,
                    "n": int(mask.sum()),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["permutation_p", "spearman_p"]) if not out.empty else out


def residualize_by_group(df: pd.DataFrame, cols: Iterable[str], group_col: str) -> pd.DataFrame:
    out = df.copy()
    if group_col not in out.columns:
        return out
    for col in cols:
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        med = vals.groupby(out[group_col]).transform("median")
        out[f"{col}_ref_centered"] = vals - med
    return out


def cycle_collapse(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    keep = ["cycleNo"] + [c for c in cols if c in df.columns]
    sub = df[keep].copy()
    for col in keep:
        if col != "cycleNo":
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
    return sub.groupby("cycleNo", as_index=False).median(numeric_only=True)


def binary_state_tests(df: pd.DataFrame, target_cols: Iterable[str], binary_col: str, rng: np.random.Generator, n_perm: int) -> pd.DataFrame:
    rows = []
    if binary_col not in df.columns:
        return pd.DataFrame()
    y = pd.to_numeric(df[binary_col], errors="coerce")
    for target in target_cols:
        if target not in df.columns:
            continue
        x = pd.to_numeric(df[target], errors="coerce")
        mask = x.notna() & y.isin([0, 1])
        if mask.sum() < 8 or y[mask].nunique() < 2:
            continue
        pos = x[mask & y.eq(1)].to_numpy(float)
        neg = x[mask & y.eq(0)].to_numpy(float)
        if len(pos) < 2 or len(neg) < 2:
            continue
        obs = float(np.nanmedian(pos) - np.nanmedian(neg))
        pooled = x[mask].to_numpy(float)
        labels = y[mask].to_numpy(int)
        null = []
        for _ in range(n_perm):
            shuf = rng.permutation(labels)
            null.append(float(np.nanmedian(pooled[shuf == 1]) - np.nanmedian(pooled[shuf == 0])))
        null_arr = np.asarray(null)
        rows.append(
            {
                "binary_predictor": binary_col,
                "target": target,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                "median_positive": float(np.nanmedian(pos)),
                "median_negative": float(np.nanmedian(neg)),
                "median_positive_minus_negative": obs,
                "mannwhitney_p": float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue),
                "permutation_p": float((np.sum(np.abs(null_arr) >= abs(obs)) + 1) / (len(null_arr) + 1)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["permutation_p", "mannwhitney_p"]) if not out.empty else out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/cycle_state_roi_bridge")
    parser.add_argument("--n-permutation", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cycle_state = read_csv(derived / "cycle_state_space_transition_audit" / "cycle_state_space_table.csv")
    physics = read_csv(derived / "physics_consistency_claim_matrix" / "physics_consistency_claim_matrix.csv")
    residuals = read_csv(derived / "echem_shape_conditioned_roi_front_effects" / "echem_shape_conditioned_residuals.csv")
    if cycle_state.empty or physics.empty:
        raise FileNotFoundError("cycle_state_space_table.csv and physics_consistency_claim_matrix.csv are required.")

    cycle_cols = ["cycleNo"] + [c for c in CYCLE_FEATURES if c in cycle_state.columns]
    joined = physics.merge(cycle_state[cycle_cols], on="cycleNo", how="left", suffixes=("", "_cycle_state"))
    if not residuals.empty:
        res_cols = ["roi_id"] + [c for c in ROI_TARGETS if c in residuals.columns and c not in joined.columns]
        joined = joined.merge(residuals[res_cols], on="roi_id", how="left")

    available_targets = [c for c in ROI_TARGETS if c in joined.columns]
    available_predictors = [c for c in CYCLE_FEATURES if c in joined.columns]
    joined["is_future_drop_cycle_state"] = pd.to_numeric(joined.get("future_any_drop_within_8cycles"), errors="coerce")
    joined["is_cross_modal_priority"] = joined.get("physics_consistency_tier", "").isin(["cross_modal_high_priority", "cross_modal_review_priority"]).astype(int)

    row_tests = spearman_tests(joined, available_predictors, available_targets, rng, args.n_permutation, "roi_row")

    centered_cols = available_predictors + available_targets
    centered = residualize_by_group(joined, centered_cols, "event_reference_cycle")
    centered_predictors = [f"{c}_ref_centered" for c in available_predictors if f"{c}_ref_centered" in centered.columns]
    centered_targets = [f"{c}_ref_centered" for c in available_targets if f"{c}_ref_centered" in centered.columns]
    centered_tests = spearman_tests(centered, centered_predictors, centered_targets, rng, args.n_permutation, "event_reference_centered")

    collapsed = cycle_collapse(joined, available_predictors + available_targets + ["is_future_drop_cycle_state", "is_cross_modal_priority"])
    collapsed_tests = spearman_tests(collapsed, available_predictors, available_targets, rng, args.n_permutation, "cycle_collapsed")
    future_state_tests = binary_state_tests(collapsed, available_targets + ["is_cross_modal_priority"], "is_future_drop_cycle_state", rng, args.n_permutation)

    cluster_rows = []
    if "cycle_state_cluster" in joined.columns:
        for cluster, g in joined.groupby("cycle_state_cluster", dropna=True):
            row: Dict[str, Any] = {
                "cycle_state_cluster": int(cluster),
                "n_roi": int(len(g)),
                "n_cycles": int(g["cycleNo"].nunique()),
                "event_fraction": float(g.get("cohort_role", pd.Series(dtype=str)).eq("event").mean()),
                "cross_modal_priority_fraction": float(g["is_cross_modal_priority"].mean()),
            }
            for target in available_targets[:8]:
                row[f"median_{target}"] = float(pd.to_numeric(g[target], errors="coerce").median())
            cluster_rows.append(row)
    cluster_summary = pd.DataFrame(cluster_rows).sort_values("cross_modal_priority_fraction", ascending=False) if cluster_rows else pd.DataFrame()

    joined_path = out / "cycle_state_roi_bridge_joined.csv"
    row_tests_path = out / "cycle_state_roi_bridge_row_tests.csv"
    centered_tests_path = out / "cycle_state_roi_bridge_reference_centered_tests.csv"
    collapsed_path = out / "cycle_state_roi_bridge_cycle_collapsed.csv"
    collapsed_tests_path = out / "cycle_state_roi_bridge_cycle_collapsed_tests.csv"
    future_state_tests_path = out / "cycle_state_roi_bridge_future_state_tests.csv"
    cluster_summary_path = out / "cycle_state_roi_bridge_cluster_summary.csv"
    joined.to_csv(joined_path, index=False)
    row_tests.to_csv(row_tests_path, index=False)
    centered_tests.to_csv(centered_tests_path, index=False)
    collapsed.to_csv(collapsed_path, index=False)
    collapsed_tests.to_csv(collapsed_tests_path, index=False)
    future_state_tests.to_csv(future_state_tests_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)

    top_row = row_tests.head(10).to_dict("records") if not row_tests.empty else []
    top_centered = centered_tests.head(10).to_dict("records") if not centered_tests.empty else []
    top_collapsed = collapsed_tests.head(10).to_dict("records") if not collapsed_tests.empty else []
    top_future = future_state_tests.head(10).to_dict("records") if not future_state_tests.empty else []
    summary = {
        "n_roi_rows": int(len(joined)),
        "n_cycles": int(joined["cycleNo"].nunique()),
        "n_cycle_state_joined": int(joined[available_predictors].notna().any(axis=1).sum()) if available_predictors else 0,
        "n_targets": int(len(available_targets)),
        "n_predictors": int(len(available_predictors)),
        "top_row_tests": clean_json(top_row),
        "top_reference_centered_tests": clean_json(top_centered),
        "top_cycle_collapsed_tests": clean_json(top_collapsed),
        "top_future_state_tests": clean_json(top_future),
        "cluster_summary": clean_json(cluster_summary.to_dict("records")) if not cluster_summary.empty else [],
        "guardrail": "Cycle-state to ROI/front bridge joins cycle-level state coordinates to selected automatic ROI rows. Row-level associations are not independent within cycle; reference-centered and cycle-collapsed tests are the stricter evidence. This does not create manual QC labels or calibrated diffusion claims.",
        "outputs": {
            "joined": str(joined_path),
            "row_tests": str(row_tests_path),
            "reference_centered_tests": str(centered_tests_path),
            "cycle_collapsed": str(collapsed_path),
            "cycle_collapsed_tests": str(collapsed_tests_path),
            "future_state_tests": str(future_state_tests_path),
            "cluster_summary": str(cluster_summary_path),
            "summary": str(out / "cycle_state_roi_bridge_summary.json"),
        },
    }
    with (out / "cycle_state_roi_bridge_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)

    lines = [
        "# Cycle State to ROI/Front Bridge",
        "",
        "Joins cycle-level state-space coordinates to ROI/front physics-consistency and echem-shape-conditioned residual targets.",
        "",
        f"- ROI rows joined: {summary['n_roi_rows']}",
        f"- Cycles represented: {summary['n_cycles']}",
        f"- Cycle-state predictors: {summary['n_predictors']}",
        f"- ROI/front targets: {summary['n_targets']}",
        "",
        "## Strongest Associations",
        "",
    ]
    for row in top_row[:5]:
        lines.append(
            f"- row {row['predictor']} vs {row['target']}: rho={row['rho']:.3f}, permutation p={row['permutation_p']:.3g}, n={row['n']}"
        )
    lines += ["", "## Stricter Checks", ""]
    for row in top_collapsed[:5]:
        lines.append(
            f"- cycle-collapsed {row['predictor']} vs {row['target']}: rho={row['rho']:.3f}, permutation p={row['permutation_p']:.3g}, n={row['n']}"
        )
    for row in top_future[:5]:
        lines.append(
            f"- future-drop state {row['target']}: median positive-negative {row['median_positive_minus_negative']:.3f}, permutation p={row['permutation_p']:.3g}"
        )
    lines += ["", "## Interpretation", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
