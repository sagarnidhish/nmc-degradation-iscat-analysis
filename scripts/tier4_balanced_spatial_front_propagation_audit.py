#!/usr/bin/env python3
"""
Balanced-cohort spatial/temporal propagation audit for ROI front physics.

Builds spatial kNN edges within source videos for the balanced future-drop ROI
cohort and tests whether front-motion / rollout-residual descriptors propagate
to nearby particle regions in later observed cycles. This complements pooled
future-label classifiers by asking a more physical question: do neighboring
regions carry coherent front/residual state forward in time?

The audit is conservative: ROI identities are automatic, only three ROI rows
exist per selected cycle, and edges connect reconstructed candidates rather than
tracked particles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score


FEATURES = [
    "radius2_slope_median_px2_per_s",
    "diffusion_proxy_median_um2_per_s",
    "q70_radius2_slope_bootstrap_p50_px2_per_s",
    "q70_radius2_slope_bootstrap_p95_px2_per_s",
    "phase_slope_positive_fraction",
    "threshold_robust_phase_score",
    "threshold_robust_diffusion_score",
    "persistence_particle_mse_fraction_of_full_mean",
    "velocity_particle_mse_fraction_of_full_mean",
    "mask_instability_score",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def make_nodes(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "roi_id",
        "cycleNo",
        "source_stem",
        "local_cycle_index",
        "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles",
        "any_abrupt_drop",
        "object_x_full_approx",
        "object_y_full_approx",
        "x_bin",
        "y_bin",
        "xy_region",
        "transferred_masked_residual_signature",
        "validation_score_recon",
    ] + [c for c in FEATURES if c in df.columns]
    nodes = df[[c for c in keep if c in df.columns]].drop_duplicates("roi_id").copy()
    for col in [
        "cycleNo",
        "local_cycle_index",
        "future_any_drop_within_8cycles",
        "future_any_drop_within_16cycles",
        "any_abrupt_drop",
        "object_x_full_approx",
        "object_y_full_approx",
    ] + [c for c in FEATURES if c in nodes.columns]:
        if col in nodes:
            nodes[col] = to_num(nodes[col])
    nodes = nodes.dropna(subset=["source_stem", "cycleNo", "object_x_full_approx", "object_y_full_approx"]).reset_index(drop=True)
    nodes["node_index"] = np.arange(len(nodes))
    return nodes


def spatial_dist(a: pd.Series, b: pd.DataFrame) -> np.ndarray:
    dx = b["object_x_full_approx"].to_numpy(float) - float(a["object_x_full_approx"])
    dy = b["object_y_full_approx"].to_numpy(float) - float(a["object_y_full_approx"])
    return np.sqrt(dx * dx + dy * dy)


def build_edges(nodes: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    edges = []
    for _, src in nodes.iterrows():
        same_source = nodes[nodes["source_stem"].eq(src["source_stem"]) & nodes["node_index"].ne(src["node_index"])].copy()
        if same_source.empty:
            continue

        edge_specs = []
        same_cycle = same_source[same_source["cycleNo"].eq(src["cycleNo"])]
        if not same_cycle.empty:
            edge_specs.append(("same_cycle_spatial_knn", same_cycle, 0.0))

        future_cycles = same_source[same_source["cycleNo"].gt(src["cycleNo"])]["cycleNo"]
        if not future_cycles.empty:
            next_cycle = future_cycles.min()
            edge_specs.append(("next_observed_cycle_spatial_knn", same_source[same_source["cycleNo"].eq(next_cycle)], float(next_cycle - src["cycleNo"])))

        previous_cycles = same_source[same_source["cycleNo"].lt(src["cycleNo"])]["cycleNo"]
        if not previous_cycles.empty:
            prev_cycle = previous_cycles.max()
            edge_specs.append(("previous_observed_cycle_spatial_knn", same_source[same_source["cycleNo"].eq(prev_cycle)], float(src["cycleNo"] - prev_cycle)))

        for edge_type, candidates, lag in edge_specs:
            candidates = candidates.copy()
            candidates["distance_px"] = spatial_dist(src, candidates)
            candidates = candidates.sort_values(["distance_px", "node_index"]).head(k)
            for rank, (_, dst) in enumerate(candidates.iterrows(), start=1):
                row = {
                    "edge_type": edge_type,
                    "src_roi_id": src["roi_id"],
                    "dst_roi_id": dst["roi_id"],
                    "source_stem": src["source_stem"],
                    "src_cycleNo": float(src["cycleNo"]),
                    "dst_cycleNo": float(dst["cycleNo"]),
                    "cycle_lag": lag,
                    "spatial_rank": rank,
                    "distance_px": float(dst["distance_px"]),
                    "distance_um_96nm": float(dst["distance_px"]) * 0.096,
                    "src_future8": int(src["future_any_drop_within_8cycles"]) if pd.notna(src.get("future_any_drop_within_8cycles")) else np.nan,
                    "dst_future8": int(dst["future_any_drop_within_8cycles"]) if pd.notna(dst.get("future_any_drop_within_8cycles")) else np.nan,
                    "same_future8": int(src.get("future_any_drop_within_8cycles") == dst.get("future_any_drop_within_8cycles"))
                    if pd.notna(src.get("future_any_drop_within_8cycles")) and pd.notna(dst.get("future_any_drop_within_8cycles"))
                    else np.nan,
                }
                for feature in [c for c in FEATURES if c in nodes.columns]:
                    row[f"src_{feature}"] = src.get(feature)
                    row[f"dst_{feature}"] = dst.get(feature)
                    if pd.notna(src.get(feature)) and pd.notna(dst.get(feature)):
                        row[f"abs_delta_{feature}"] = abs(float(src.get(feature)) - float(dst.get(feature)))
                    else:
                        row[f"abs_delta_{feature}"] = np.nan
                edges.append(row)
    return pd.DataFrame(edges)


def empirical_p_ge(observed: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return np.nan
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def empirical_p_abs_ge(observed: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return np.nan
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))


def homophily_tests(edges: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    valid_all = edges.dropna(subset=["src_future8", "dst_future8"])
    for edge_type, g in valid_all.groupby("edge_type"):
        if g.empty:
            continue
        same = (g["src_future8"].astype(int).to_numpy() == g["dst_future8"].astype(int).to_numpy()).astype(float)
        both_pos = ((g["src_future8"].astype(int).to_numpy() == 1) & (g["dst_future8"].astype(int).to_numpy() == 1)).astype(float)
        obs_same = float(same.mean())
        obs_both = float(both_pos.mean())
        dst = g["dst_future8"].astype(int).to_numpy()
        src = g["src_future8"].astype(int).to_numpy()
        null_same = []
        null_both = []
        for _ in range(n_perm):
            p_dst = rng.permutation(dst)
            null_same.append(float((src == p_dst).mean()))
            null_both.append(float(((src == 1) & (p_dst == 1)).mean()))
        rows.append(
            {
                "edge_type": edge_type,
                "n_edges": int(len(g)),
                "observed_same_future8_fraction": obs_same,
                "observed_both_future8_positive_fraction": obs_both,
                "null_same_mean": float(np.mean(null_same)),
                "null_same_p95": float(np.quantile(null_same, 0.95)),
                "empirical_p_same_ge_observed": empirical_p_ge(obs_same, np.asarray(null_same)),
                "null_both_positive_mean": float(np.mean(null_both)),
                "null_both_positive_p95": float(np.quantile(null_both, 0.95)),
                "empirical_p_both_ge_observed": empirical_p_ge(obs_both, np.asarray(null_both)),
            }
        )
    return pd.DataFrame(rows).sort_values("empirical_p_same_ge_observed") if rows else pd.DataFrame(rows)


def feature_autocorrelation_tests(edges: pd.DataFrame, features: list[str], n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for edge_type, g in edges.groupby("edge_type"):
        for feature in features:
            src_col = f"src_{feature}"
            dst_col = f"dst_{feature}"
            if src_col not in g or dst_col not in g:
                continue
            valid = g[src_col].notna() & g[dst_col].notna()
            if valid.sum() < 8 or g.loc[valid, src_col].nunique() < 3 or g.loc[valid, dst_col].nunique() < 3:
                continue
            src = g.loc[valid, src_col].to_numpy(float)
            dst = g.loc[valid, dst_col].to_numpy(float)
            rho, p = spearmanr(src, dst)
            null = []
            for _ in range(n_perm):
                p_dst = rng.permutation(dst)
                null.append(float(spearmanr(src, p_dst).statistic))
            rows.append(
                {
                    "edge_type": edge_type,
                    "feature": feature,
                    "n_edges": int(valid.sum()),
                    "spearman_src_dst": float(rho),
                    "spearman_p": float(p),
                    "null_mean": float(np.nanmean(null)),
                    "null_p05": float(np.nanquantile(null, 0.05)),
                    "null_p95": float(np.nanquantile(null, 0.95)),
                    "empirical_p_abs_ge_observed": empirical_p_abs_ge(float(rho), np.asarray(null)),
                }
            )
    return pd.DataFrame(rows).sort_values(["empirical_p_abs_ge_observed", "edge_type"]) if rows else pd.DataFrame(rows)


def lag_feature_label_tests(edges: pd.DataFrame, features: list[str], n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    lag_edges = edges[edges["edge_type"].eq("next_observed_cycle_spatial_knn")].copy()
    if lag_edges.empty:
        return pd.DataFrame(rows)
    for feature in features:
        src_col = f"src_{feature}"
        if src_col not in lag_edges:
            continue
        valid = lag_edges[src_col].notna() & lag_edges["dst_future8"].notna()
        y = lag_edges.loc[valid, "dst_future8"].astype(int).to_numpy()
        x = lag_edges.loc[valid, src_col].to_numpy(float)
        if len(y) < 10 or len(np.unique(y)) != 2 or len(np.unique(x)) < 3:
            continue
        auc = float(roc_auc_score(y, x))
        oriented_auc = max(auc, 1.0 - auc)
        null = []
        for _ in range(n_perm):
            p_y = rng.permutation(y)
            try:
                n_auc = roc_auc_score(p_y, x)
                null.append(max(n_auc, 1.0 - n_auc))
            except ValueError:
                continue
        rows.append(
            {
                "edge_type": "next_observed_cycle_spatial_knn",
                "feature": feature,
                "n_edges": int(len(y)),
                "n_dst_positive": int(y.sum()),
                "n_dst_negative": int((1 - y).sum()),
                "auc_src_feature_predicts_dst_future8": auc,
                "oriented_auc": float(oriented_auc),
                "null_oriented_auc_mean": float(np.mean(null)) if null else np.nan,
                "null_oriented_auc_p95": float(np.quantile(null, 0.95)) if null else np.nan,
                "empirical_p_ge_observed": empirical_p_ge(float(oriented_auc), np.asarray(null)),
            }
        )
    return pd.DataFrame(rows).sort_values("empirical_p_ge_observed") if rows else pd.DataFrame(rows)


def distance_gradient_tests(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = edges.dropna(subset=["src_future8", "dst_future8", "distance_px"]).copy()
    for edge_type, g in valid.groupby("edge_type"):
        both = g[(g["src_future8"].astype(int) == 1) & (g["dst_future8"].astype(int) == 1)]["distance_um_96nm"]
        other = g[~((g["src_future8"].astype(int) == 1) & (g["dst_future8"].astype(int) == 1))]["distance_um_96nm"]
        if len(both) < 3 or len(other) < 3:
            continue
        rows.append(
            {
                "edge_type": edge_type,
                "n_both_future8_positive_edges": int(len(both)),
                "n_other_edges": int(len(other)),
                "median_distance_both_positive_um": float(both.median()),
                "median_distance_other_um": float(other.median()),
                "median_distance_both_positive_minus_other_um": float(both.median() - other.median()),
                "mannwhitney_p": float(mannwhitneyu(both, other, alternative="two-sided").pvalue),
            }
        )
    return pd.DataFrame(rows).sort_values("mannwhitney_p") if rows else pd.DataFrame(rows)


def records(df: pd.DataFrame, n: int) -> list[dict]:
    if df.empty:
        return []
    return json.loads(df.head(n).to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/balanced_spatial_front_propagation_audit")
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--n-permutation", type=int, default=1000)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = ensure_dir(Path(args.out_dir))
    joined = pd.read_csv(derived / "balanced_future_context_region_audit" / "balanced_future_context_region_joined.csv")
    nodes = make_nodes(joined)
    features = [f for f in FEATURES if f in nodes.columns]
    edges = build_edges(nodes, args.k_neighbors)

    homophily = homophily_tests(edges, args.n_permutation, seed=11)
    autocorr = feature_autocorrelation_tests(edges, features, args.n_permutation, seed=13)
    lag_label = lag_feature_label_tests(edges, features, args.n_permutation, seed=17)
    distance = distance_gradient_tests(edges)

    source_cycle = (
        nodes.groupby(["source_stem", "cycleNo"], as_index=False)
        .agg(
            n_roi=("roi_id", "count"),
            future8_positive_fraction=("future_any_drop_within_8cycles", "mean"),
            x_median=("object_x_full_approx", "median"),
            y_median=("object_y_full_approx", "median"),
            radius2_median=("radius2_slope_median_px2_per_s", "median"),
            diffusion_median=("diffusion_proxy_median_um2_per_s", "median"),
            persistence_fraction_median=("persistence_particle_mse_fraction_of_full_mean", "median"),
            mask_instability_median=("mask_instability_score", "median"),
        )
        .sort_values(["source_stem", "cycleNo"])
    )

    edge_counts = edges["edge_type"].value_counts().sort_index().to_dict() if not edges.empty else {}
    summary = {
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_cycles": int(nodes["cycleNo"].nunique()),
        "n_source_stems": int(nodes["source_stem"].nunique()),
        "k_neighbors": int(args.k_neighbors),
        "n_permutation": int(args.n_permutation),
        "edge_counts": {str(k): int(v) for k, v in edge_counts.items()},
        "node_future8_positive_fraction": float(nodes["future_any_drop_within_8cycles"].mean()),
        "top_homophily_tests": records(homophily, 12),
        "top_feature_autocorrelation_tests": records(autocorr, 20),
        "top_lag_feature_label_tests": records(lag_label, 20),
        "distance_gradient_tests": records(distance, 12),
        "guardrail": "Balanced spatial front propagation audit uses automatic ROI coordinates and reconstructed candidates, not tracked particle identities. Edges test spatial/temporal coherence for hypothesis ranking, not causal propagation.",
    }

    nodes.to_csv(out / "balanced_spatial_front_nodes.csv", index=False)
    edges.to_csv(out / "balanced_spatial_front_edges.csv", index=False)
    source_cycle.to_csv(out / "balanced_spatial_front_source_cycle_summary.csv", index=False)
    homophily.to_csv(out / "balanced_spatial_front_homophily_tests.csv", index=False)
    autocorr.to_csv(out / "balanced_spatial_front_feature_autocorrelation_tests.csv", index=False)
    lag_label.to_csv(out / "balanced_spatial_front_lag_feature_label_tests.csv", index=False)
    distance.to_csv(out / "balanced_spatial_front_distance_gradient_tests.csv", index=False)
    with (out / "balanced_spatial_front_propagation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (out / "README.md").open("w") as f:
        f.write("# Balanced Spatial Front Propagation Audit\n\n")
        f.write("Spatial kNN graph over balanced future-drop ROI candidates, testing front/residual coherence within source videos and across adjacent observed cycles.\n\n")
        f.write(f"- Nodes: {summary['n_nodes']}\n")
        f.write(f"- Edges: {summary['n_edges']}\n")
        f.write(f"- Edge counts: {summary['edge_counts']}\n")
        f.write("\nGuardrail: automatic reconstructed ROI candidates; use as spatial hypothesis ranking, not particle-tracked causal propagation.\n")


if __name__ == "__main__":
    main()
