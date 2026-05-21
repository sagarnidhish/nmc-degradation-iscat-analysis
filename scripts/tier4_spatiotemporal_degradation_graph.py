#!/usr/bin/env python3
"""Spatiotemporal graph audit for NMC ROI degradation physics signals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score


EVENT_MODE = "optical_brightening_decorrelating_rollout_hard_front_positive"
TARGETS = [
    "is_event_roi",
    "is_event_enriched_mode",
    "front_positive_residual_binary",
]
CONTINUOUS_TARGETS = [
    "mode_review_priority",
    "phase_slope_positive_fraction_protocol_residual",
    "threshold_robust_phase_score_protocol_residual",
    "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
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


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_nodes(derived: Path) -> pd.DataFrame:
    echem = pd.read_csv(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_joined.csv")
    modes = pd.read_csv(derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv")
    front = pd.read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")
    mode_keep = [
        "roi_id",
        "mode_label",
        "mode_review_priority",
        "mode_pc1",
        "mode_pc2",
        "rollout_mobility_difficulty_score",
    ]
    front_keep = [
        "roi_id",
        "phase_slope_positive_fraction_protocol_residual",
        "phase_slope_median_per_s_protocol_residual",
        "threshold_robust_phase_score_protocol_residual",
        "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    ]
    base_keep = [
        "roi_id",
        "cycleNo",
        "cohort_role",
        "event_reference_cycle",
        "source_stem",
        "object_x_full_approx",
        "object_y_full_approx",
        "n_frames_percentile",
        "V_mean",
        "I_mean_mA",
        "front_quality_score",
        "degradation_mode_hypothesis",
    ]
    nodes = echem[[c for c in base_keep if c in echem.columns]].drop_duplicates("roi_id")
    nodes = nodes.merge(modes[[c for c in mode_keep if c in modes.columns]].drop_duplicates("roi_id"), on="roi_id", how="left")
    nodes = nodes.merge(front[[c for c in front_keep if c in front.columns]].drop_duplicates("roi_id"), on="roi_id", how="left")
    for col in [
        "cycleNo",
        "event_reference_cycle",
        "object_x_full_approx",
        "object_y_full_approx",
        "mode_review_priority",
        "phase_slope_positive_fraction_protocol_residual",
        "threshold_robust_phase_score_protocol_residual",
        "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    ]:
        if col in nodes.columns:
            nodes[col] = to_num(nodes[col])
    nodes["is_event_roi"] = nodes["cohort_role"].eq("event").astype(int)
    nodes["is_event_enriched_mode"] = nodes["mode_label"].eq(EVENT_MODE).astype(int)
    nodes["front_positive_residual_binary"] = nodes["phase_slope_positive_fraction_protocol_residual"].gt(0).astype(int)
    x = to_num(nodes["object_x_full_approx"])
    y = to_num(nodes["object_y_full_approx"])
    nodes["xy_region"] = pd.cut(x, bins=4, labels=False, duplicates="drop").astype("Int64").astype(str) + "_" + pd.cut(y, bins=4, labels=False, duplicates="drop").astype("Int64").astype(str)
    return nodes


def pair_distance(a: pd.Series, b: pd.Series) -> float:
    dx = float(a["object_x_full_approx"]) - float(b["object_x_full_approx"])
    dy = float(a["object_y_full_approx"]) - float(b["object_y_full_approx"])
    return math.sqrt(dx * dx + dy * dy)


def build_edges(nodes: pd.DataFrame, k: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    clean = nodes.dropna(subset=["object_x_full_approx", "object_y_full_approx", "cycleNo", "event_reference_cycle"]).copy()
    for _, a in clean.iterrows():
        others = clean[clean["roi_id"].ne(a["roi_id"])].copy()
        if others.empty:
            continue
        others["distance_px"] = others.apply(lambda b: pair_distance(a, b), axis=1)
        same_ref = others[others["event_reference_cycle"].eq(a["event_reference_cycle"])].sort_values("distance_px").head(k)
        same_cycle = others[others["cycleNo"].eq(a["cycleNo"])].sort_values("distance_px").head(k)
        prev_cycle = others[
            others["event_reference_cycle"].eq(a["event_reference_cycle"])
            & others["cycleNo"].lt(a["cycleNo"])
        ].sort_values(["cycleNo", "distance_px"], ascending=[False, True]).head(k)
        next_cycle = others[
            others["event_reference_cycle"].eq(a["event_reference_cycle"])
            & others["cycleNo"].gt(a["cycleNo"])
        ].sort_values(["cycleNo", "distance_px"], ascending=[True, True]).head(k)
        for edge_type, frame in [
            ("same_reference_spatial_knn", same_ref),
            ("same_cycle_spatial_knn", same_cycle),
            ("previous_cycle_spatial_knn", prev_cycle),
            ("next_cycle_spatial_knn", next_cycle),
        ]:
            for rank, (_, b) in enumerate(frame.iterrows(), start=1):
                rows.append({
                    "src_roi_id": a["roi_id"],
                    "dst_roi_id": b["roi_id"],
                    "edge_type": edge_type,
                    "neighbor_rank": rank,
                    "src_cycle": float(a["cycleNo"]),
                    "dst_cycle": float(b["cycleNo"]),
                    "cycle_delta_dst_minus_src": float(b["cycleNo"] - a["cycleNo"]),
                    "event_reference_cycle": float(a["event_reference_cycle"]),
                    "distance_px": float(b["distance_px"]),
                    "same_xy_region": bool(a.get("xy_region") == b.get("xy_region")),
                })
    edges = pd.DataFrame(rows).drop_duplicates(["src_roi_id", "dst_roi_id", "edge_type"])
    return edges


def attach_edge_values(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    src = nodes.set_index("roi_id")
    out = edges.copy()
    for target in TARGETS + CONTINUOUS_TARGETS + ["mode_label", "cohort_role", "xy_region"]:
        if target not in nodes.columns:
            continue
        out[f"src_{target}"] = out["src_roi_id"].map(src[target])
        out[f"dst_{target}"] = out["dst_roi_id"].map(src[target])
    for target in TARGETS:
        out[f"{target}_same"] = out[f"src_{target}"].eq(out[f"dst_{target}"])
        out[f"{target}_both_positive"] = out[f"src_{target}"].eq(1) & out[f"dst_{target}"].eq(1)
    return out


def permute_within(values: pd.Series, groups: pd.Series, rng: np.random.Generator) -> pd.Series:
    out = values.copy()
    for _, idx in groups.groupby(groups).groups.items():
        vals = out.loc[idx].to_numpy(copy=True)
        rng.shuffle(vals)
        out.loc[idx] = vals
    return out


def edge_homophily_tests(edges: pd.DataFrame, nodes: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = nodes.set_index("roi_id")["event_reference_cycle"]
    for edge_type, sub in edges.groupby("edge_type"):
        for target in TARGETS:
            if f"src_{target}" not in sub.columns:
                continue
            src = to_num(sub[f"src_{target}"])
            dst = to_num(sub[f"dst_{target}"])
            valid = src.notna() & dst.notna()
            if valid.sum() < 4:
                continue
            observed_same = float((src[valid].astype(int).to_numpy() == dst[valid].astype(int).to_numpy()).mean())
            observed_both = float(((src[valid] == 1) & (dst[valid] == 1)).mean())
            same_null = []
            both_null = []
            node_vals = nodes.set_index("roi_id")[target]
            for _ in range(n_perm):
                perm = permute_within(node_vals, groups, rng)
                ps = sub.loc[valid, "src_roi_id"].map(perm).astype(int)
                pdst = sub.loc[valid, "dst_roi_id"].map(perm).astype(int)
                same_null.append(float((ps.to_numpy() == pdst.to_numpy()).mean()))
                both_null.append(float(((ps == 1) & (pdst == 1)).mean()))
            same_arr = np.asarray(same_null)
            both_arr = np.asarray(both_null)
            rows.append({
                "edge_type": edge_type,
                "target": target,
                "n_edges": int(valid.sum()),
                "observed_same_fraction": observed_same,
                "null_same_mean": float(same_arr.mean()),
                "null_same_p95": float(np.percentile(same_arr, 95)),
                "empirical_p_same_ge_observed": float((np.sum(same_arr >= observed_same) + 1) / (len(same_arr) + 1)),
                "observed_both_positive_fraction": observed_both,
                "null_both_positive_mean": float(both_arr.mean()),
                "empirical_p_both_ge_observed": float((np.sum(both_arr >= observed_both) + 1) / (len(both_arr) + 1)),
            })
    return pd.DataFrame(rows).sort_values(["empirical_p_same_ge_observed", "edge_type", "target"])


def continuous_neighbor_tests(edges: pd.DataFrame, nodes: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = nodes.set_index("roi_id")["event_reference_cycle"]
    for edge_type, sub in edges.groupby("edge_type"):
        for target in CONTINUOUS_TARGETS:
            if f"src_{target}" not in sub.columns:
                continue
            src = to_num(sub[f"src_{target}"])
            dst = to_num(sub[f"dst_{target}"])
            valid = src.notna() & dst.notna()
            if valid.sum() < 6 or src[valid].nunique() < 3 or dst[valid].nunique() < 3:
                continue
            rho = spearmanr(src[valid], dst[valid]).statistic
            node_vals = nodes.set_index("roi_id")[target]
            null = []
            for _ in range(n_perm):
                perm = permute_within(node_vals, groups, rng)
                ps = sub.loc[valid, "src_roi_id"].map(perm)
                pdst = sub.loc[valid, "dst_roi_id"].map(perm)
                val = spearmanr(ps, pdst).statistic
                if np.isfinite(val):
                    null.append(float(val))
            arr = np.asarray(null)
            rows.append({
                "edge_type": edge_type,
                "target": target,
                "n_edges": int(valid.sum()),
                "spearman_src_dst": float(rho) if np.isfinite(rho) else np.nan,
                "null_mean": float(arr.mean()) if len(arr) else np.nan,
                "null_p95": float(np.percentile(arr, 95)) if len(arr) else np.nan,
                "null_p05": float(np.percentile(arr, 5)) if len(arr) else np.nan,
                "empirical_p_abs_ge_observed": float((np.sum(np.abs(arr) >= abs(rho)) + 1) / (len(arr) + 1)) if len(arr) and np.isfinite(rho) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["empirical_p_abs_ge_observed", "edge_type", "target"])


def temporal_lag_tests(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = edges[edges["edge_type"].eq("previous_cycle_spatial_knn") & edges["neighbor_rank"].eq(1)].copy()
    if sub.empty:
        return pd.DataFrame()
    for target in TARGETS:
        src = to_num(sub[f"src_{target}"])
        prev = to_num(sub[f"dst_{target}"])
        valid = src.notna() & prev.notna()
        if valid.sum() < 6 or src[valid].nunique() < 2 or prev[valid].nunique() < 2:
            continue
        try:
            auc = roc_auc_score(src[valid].astype(int), prev[valid].astype(float))
        except ValueError:
            auc = np.nan
        rows.append({
            "target": target,
            "n_lag_pairs": int(valid.sum()),
            "previous_nearest_positive_rate_when_current_positive": float(prev[valid & src.eq(1)].mean()) if (valid & src.eq(1)).sum() else np.nan,
            "previous_nearest_positive_rate_when_current_negative": float(prev[valid & src.eq(0)].mean()) if (valid & src.eq(0)).sum() else np.nan,
            "roc_auc_previous_neighbor_predicts_current": float(auc) if np.isfinite(auc) else np.nan,
        })
    for target in CONTINUOUS_TARGETS:
        src = to_num(sub[f"src_{target}"])
        prev = to_num(sub[f"dst_{target}"])
        valid = src.notna() & prev.notna()
        if valid.sum() < 6 or src[valid].nunique() < 3 or prev[valid].nunique() < 3:
            continue
        rho = spearmanr(prev[valid], src[valid]).statistic
        rows.append({
            "target": target,
            "n_lag_pairs": int(valid.sum()),
            "spearman_previous_nearest_vs_current": float(rho) if np.isfinite(rho) else np.nan,
        })
    return pd.DataFrame(rows)


def distance_gradient_tests(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = edges[edges["edge_type"].eq("same_reference_spatial_knn")].copy()
    for target in TARGETS:
        if f"{target}_both_positive" not in sub.columns:
            continue
        pos = sub.loc[sub[f"{target}_both_positive"], "distance_px"].dropna().to_numpy(dtype=float)
        other = sub.loc[~sub[f"{target}_both_positive"], "distance_px"].dropna().to_numpy(dtype=float)
        if len(pos) >= 2 and len(other) >= 2:
            _, p = mannwhitneyu(pos, other, alternative="two-sided")
            rows.append({
                "target": target,
                "n_positive_positive_edges": int(len(pos)),
                "n_other_edges": int(len(other)),
                "median_distance_positive_positive": float(np.median(pos)),
                "median_distance_other": float(np.median(other)),
                "median_distance_pp_minus_other": float(np.median(pos) - np.median(other)),
                "mannwhitney_p": float(p),
            })
    return pd.DataFrame(rows).sort_values("mannwhitney_p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/spatiotemporal_degradation_graph")
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(derived)
    edges = attach_edge_values(build_edges(nodes, args.k_neighbors), nodes)
    homophily = edge_homophily_tests(edges, nodes, args.n_permutation, args.seed)
    continuous = continuous_neighbor_tests(edges, nodes, args.n_permutation, args.seed + 1)
    lag = temporal_lag_tests(edges)
    distance = distance_gradient_tests(edges)

    node_path = out / "spatiotemporal_graph_nodes.csv"
    edge_path = out / "spatiotemporal_graph_edges.csv"
    homophily_path = out / "spatiotemporal_graph_homophily_tests.csv"
    continuous_path = out / "spatiotemporal_graph_continuous_tests.csv"
    lag_path = out / "spatiotemporal_graph_lag_tests.csv"
    distance_path = out / "spatiotemporal_graph_distance_gradient_tests.csv"
    nodes.to_csv(node_path, index=False)
    edges.to_csv(edge_path, index=False)
    homophily.to_csv(homophily_path, index=False)
    continuous.to_csv(continuous_path, index=False)
    lag.to_csv(lag_path, index=False)
    distance.to_csv(distance_path, index=False)

    top_homophily = homophily.head(12).to_dict("records") if not homophily.empty else []
    top_continuous = continuous.head(12).to_dict("records") if not continuous.empty else []
    summary = {
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "k_neighbors": int(args.k_neighbors),
        "n_permutation": int(args.n_permutation),
        "event_mode": EVENT_MODE,
        "node_target_rates": {t: float(nodes[t].mean()) for t in TARGETS if t in nodes.columns},
        "edge_counts": edges["edge_type"].value_counts().to_dict() if not edges.empty else {},
        "top_homophily_tests": top_homophily,
        "top_continuous_neighbor_tests": top_continuous,
        "temporal_lag_tests": lag.to_dict("records") if not lag.empty else [],
        "distance_gradient_tests": distance.to_dict("records") if not distance.empty else [],
        "guardrail": "Spatiotemporal graph tests use automatic ROI coordinates and automatic residual labels on a selected 52-ROI cohort. They test clustering/propagation hypotheses for review prioritization, not causal material degradation mechanisms.",
        "outputs": {
            "nodes": str(node_path),
            "edges": str(edge_path),
            "homophily_tests": str(homophily_path),
            "continuous_tests": str(continuous_path),
            "lag_tests": str(lag_path),
            "distance_gradient_tests": str(distance_path),
            "summary": str(out / "spatiotemporal_degradation_graph_summary.json"),
        },
    }
    with (out / "spatiotemporal_degradation_graph_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Spatiotemporal Degradation Graph",
        "",
        "Nearest-neighbor graph audit for NMC ROI degradation modes, front residuals, cycles, and particle regions.",
        "",
        f"- Nodes: {summary['n_nodes']}",
        f"- Edges: {summary['n_edges']}",
        f"- k-neighbors: {summary['k_neighbors']}",
        "",
        "## Top Homophily Tests",
    ]
    for row in top_homophily[:8]:
        lines.append(
            f"- {row.get('edge_type')} {row.get('target')}: same={row.get('observed_same_fraction'):.3f}, null_mean={row.get('null_same_mean'):.3f}, p={row.get('empirical_p_same_ge_observed'):.4f}"
        )
    lines += ["", "## Top Continuous Neighbor Tests"]
    for row in top_continuous[:8]:
        lines.append(
            f"- {row.get('edge_type')} {row.get('target')}: rho={row.get('spearman_src_dst'):.3f}, null_mean={row.get('null_mean'):.3f}, p={row.get('empirical_p_abs_ge_observed'):.4f}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "out_dir": str(out),
        "n_nodes": summary["n_nodes"],
        "n_edges": summary["n_edges"],
        "top_homophily": top_homophily[:3],
        "top_continuous": top_continuous[:3],
    }, indent=2))


if __name__ == "__main__":
    main()
