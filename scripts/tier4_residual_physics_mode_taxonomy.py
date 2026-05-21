#!/usr/bin/env python3
"""Protocol-adjusted ROI physics mode taxonomy for NMC degradation candidates.

This combines optical/rollout residuals, threshold-front residuals, rollout
mobility, and QC-priority context into an interpretable mode table. The goal is
not a final supervised detector; it is a compact degradation-mode hypothesis
taxonomy after the strongest protocol/echem confounders have been accounted for.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "roi_norm_mean_delta_protocol_residual",
    "high_fraction_delta_protocol_residual",
    "low_fraction_delta_protocol_residual",
    "first_last_corr_protocol_residual",
    "cumulative_abs_norm_change_protocol_residual",
    "latent_net_displacement_protocol_residual",
    "dmd_minus_persistence_mse_protocol_residual",
    "phase_slope_positive_fraction_protocol_residual",
    "phase_slope_median_per_s_protocol_residual",
    "threshold_robust_phase_score_protocol_residual",
    "diffusion_proxy_median_um2_per_s_protocol_residual",
    "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    "rollout_mobility_difficulty_score",
]


def load_joined(derived: Path) -> pd.DataFrame:
    echem = pd.read_csv(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_joined.csv")
    roi_resid = pd.read_csv(derived / "protocol_conditioned_roi_effects" / "protocol_conditioned_roi_residuals.csv")
    front_resid = pd.read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")
    mobility = pd.read_csv(derived / "multi_cycle_rollout_mobility_coupling" / "multi_cycle_rollout_mobility_ranked.csv")
    qc = pd.read_csv(derived / "qc_review_packet" / "qc_review_manifest.csv")

    base_cols = [
        "roi_id",
        "cycleNo",
        "cohort_role",
        "event_reference_cycle",
        "degradation_mode_hypothesis",
        "n_frames_percentile",
        "V_mean",
        "I_mean_mA",
    ]
    df = echem[[c for c in base_cols if c in echem.columns]].drop_duplicates("roi_id")
    df = df.merge(roi_resid.drop(columns=[c for c in ["cohort_role", "event_reference_cycle"] if c in roi_resid.columns]), on="roi_id", how="left")
    df = df.merge(front_resid.drop(columns=[c for c in ["cohort_role", "event_reference_cycle", "cycleNo"] if c in front_resid.columns]), on="roi_id", how="left")
    df = df.merge(
        mobility[[
            c for c in [
                "roi_id",
                "rollout_mobility_difficulty_score",
                "latent_path_length",
                "latent_net_displacement",
                "first_last_corr",
                "cumulative_abs_first_last",
            ]
            if c in mobility.columns
        ]].drop_duplicates("roi_id"),
        on="roi_id",
        how="left",
    )
    df = df.merge(
        qc[[
            c for c in [
                "roi_id",
                "qc_priority_score",
                "qc_reason",
                "manual_qc_status",
                "manual_qc_decision",
            ]
            if c in qc.columns
        ]].drop_duplicates("roi_id"),
        on="roi_id",
        how="left",
    )
    df["is_event_roi"] = (df["cohort_role"] == "event").astype(int)
    return df


def choose_k(x: np.ndarray, random_state: int) -> Dict[str, object]:
    rows = []
    best = {"k": 2, "silhouette": -1.0}
    for k in range(2, min(6, len(x) - 1)):
        labels = KMeans(n_clusters=k, n_init=50, random_state=random_state).fit_predict(x)
        score = float(silhouette_score(x, labels)) if len(np.unique(labels)) > 1 else np.nan
        rows.append({"k": k, "silhouette": score})
        if np.isfinite(score) and score > best["silhouette"]:
            best = {"k": k, "silhouette": score}
    return {"best": best, "scores": rows}


def z(row: pd.Series, feature: str) -> float:
    return float(row.get(f"{feature}_z_mean", 0) or 0)


def name_mode(row: pd.Series, threshold: float = 0.4) -> str:
    parts = []
    if z(row, "high_fraction_delta_protocol_residual") > threshold and z(row, "roi_norm_mean_delta_protocol_residual") > threshold:
        parts.append("optical_brightening")
    if z(row, "high_fraction_delta_protocol_residual") < -threshold and z(row, "roi_norm_mean_delta_protocol_residual") < -threshold:
        parts.append("optical_loss")
    if z(row, "first_last_corr_protocol_residual") < -threshold or z(row, "cumulative_abs_norm_change_protocol_residual") > threshold:
        parts.append("decorrelating")
    if z(row, "latent_net_displacement_protocol_residual") > threshold or z(row, "rollout_mobility_difficulty_score") > threshold:
        parts.append("rollout_hard")
    if z(row, "phase_slope_positive_fraction_protocol_residual") > threshold or z(row, "threshold_robust_phase_score_protocol_residual") > threshold:
        parts.append("front_positive")
    if z(row, "phase_slope_positive_fraction_protocol_residual") < -threshold or z(row, "threshold_robust_phase_score_protocol_residual") < -threshold:
        parts.append("front_negative")
    if z(row, "diffusion_proxy_abs_median_um2_per_s_protocol_residual") > threshold:
        parts.append("high_apparent_front_proxy")
    return "_".join(parts[:4]) if parts else "near_baseline_or_context_like"


def cluster_stability(x: np.ndarray, k: int, random_state: int) -> Dict[str, object]:
    reference = KMeans(n_clusters=k, n_init=100, random_state=random_state).fit_predict(x)
    rows = []
    for seed in range(random_state + 1, random_state + 21):
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
        rows.append({"seed": seed, "adjusted_rand_index": float(adjusted_rand_score(reference, labels))})
    values = [r["adjusted_rand_index"] for r in rows]
    return {
        "n_repeats": len(rows),
        "mean_adjusted_rand_index": float(np.mean(values)),
        "min_adjusted_rand_index": float(np.min(values)),
        "max_adjusted_rand_index": float(np.max(values)),
        "rows": rows,
    }


def mode_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_event = int((df["cohort_role"] == "event").sum())
    total_control = int((df["cohort_role"] == "control").sum())
    for mode, grp in df.groupby("mode_label"):
        event = int((grp["cohort_role"] == "event").sum())
        control = int((grp["cohort_role"] == "control").sum())
        _, p = fisher_exact([[event, control], [total_event - event, total_control - control]], alternative="two-sided")
        rows.append({
            "mode_label": mode,
            "n_roi": int(len(grp)),
            "n_event": event,
            "n_control": control,
            "event_fraction": float(event / len(grp)) if len(grp) else np.nan,
            "fisher_p_value": float(p),
            "top_cycles": ";".join(str(x) for x in grp["cycleNo"].value_counts().head(4).index.tolist()),
        })
    return pd.DataFrame(rows).sort_values(["event_fraction", "n_roi"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/residual_physics_mode_taxonomy")
    parser.add_argument("--random-state", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_joined(derived)
    features = [f for f in FEATURES if f in df.columns]
    x_raw = df[features].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    x = pipe.fit_transform(x_raw)
    k_info = choose_k(x, args.random_state)
    k = int(k_info["best"]["k"])
    stability = cluster_stability(x, k, args.random_state)
    labels = KMeans(n_clusters=k, n_init=100, random_state=args.random_state).fit_predict(x)
    pca = PCA(n_components=2, random_state=args.random_state)
    xy = pca.fit_transform(x)

    out_df = df.copy()
    out_df["mode_cluster"] = labels
    out_df["mode_pc1"] = xy[:, 0]
    out_df["mode_pc2"] = xy[:, 1]
    scaled = pd.DataFrame(x, columns=[f"{f}_z" for f in features])
    for col in scaled.columns:
        out_df[col] = scaled[col].to_numpy()

    centroids = []
    for cluster, grp in out_df.groupby("mode_cluster"):
        row = {"mode_cluster": int(cluster), "n_roi": int(len(grp))}
        for f in features:
            row[f] = float(pd.to_numeric(grp[f], errors="coerce").mean())
            row[f"{f}_z_mean"] = float(pd.to_numeric(grp[f"{f}_z"], errors="coerce").mean())
        row["mode_label"] = name_mode(pd.Series(row))
        row["event_fraction"] = float((grp["cohort_role"] == "event").mean())
        row["control_fraction"] = float((grp["cohort_role"] == "control").mean())
        row["mean_qc_priority_score"] = float(pd.to_numeric(grp.get("qc_priority_score"), errors="coerce").mean())
        centroids.append(row)
    centroid_df = pd.DataFrame(centroids).sort_values(["event_fraction", "mean_qc_priority_score"], ascending=False)
    label_map = dict(zip(centroid_df["mode_cluster"], centroid_df["mode_label"]))
    out_df["mode_label"] = out_df["mode_cluster"].map(label_map)

    # A mechanism score for review triage, not a calibrated probability.
    out_df["mode_review_priority"] = (
        out_df["qc_priority_score"].fillna(0).rank(pct=True)
        + out_df["rollout_mobility_difficulty_score"].fillna(0).rank(pct=True)
        + out_df["cumulative_abs_norm_change_protocol_residual"].abs().fillna(0).rank(pct=True)
        + out_df["phase_slope_positive_fraction_protocol_residual"].abs().fillna(0).rank(pct=True)
    )
    out_df = out_df.sort_values("mode_review_priority", ascending=False)
    enrichment = mode_enrichment(out_df)

    rows_path = out / "residual_physics_mode_assignments.csv"
    centroid_path = out / "residual_physics_mode_centroids.csv"
    enrichment_path = out / "residual_physics_mode_enrichment.csv"
    out_df.to_csv(rows_path, index=False)
    centroid_df.to_csv(centroid_path, index=False)
    enrichment.to_csv(enrichment_path, index=False)

    summary = {
        "n_roi": int(len(out_df)),
        "n_event_roi": int((out_df["cohort_role"] == "event").sum()),
        "n_control_roi": int((out_df["cohort_role"] == "control").sum()),
        "features": features,
        "chosen_k": k,
        "cluster_selection": k_info,
        "cluster_stability": stability,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "mode_enrichment": enrichment.to_dict("records"),
        "top_mode_centroids": centroid_df.to_dict("records"),
        "top_review_rois": out_df.head(15)[[
            "roi_id",
            "cohort_role",
            "cycleNo",
            "mode_label",
            "mode_review_priority",
            "qc_priority_score",
            "rollout_mobility_difficulty_score",
            "degradation_mode_hypothesis",
        ]].to_dict("records"),
        "guardrail": "Mode labels are protocol-adjusted computational hypotheses from automatic ROI candidates; they require QC labels before being treated as biological/material degradation modes.",
        "outputs": {
            "assignments": str(rows_path),
            "centroids": str(centroid_path),
            "enrichment": str(enrichment_path),
        },
    }
    with (out / "residual_physics_mode_taxonomy_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    lines = [
        "# Residual Physics Mode Taxonomy",
        "",
        "Protocol-adjusted degradation-mode hypotheses from optical residuals, rollout/latent descriptors, threshold-front residuals, and QC-priority context.",
        "",
        f"- ROI rows: {len(out_df)}",
        f"- Event ROI: {summary['n_event_roi']}",
        f"- Control ROI: {summary['n_control_roi']}",
        f"- Selected k: {k}",
        f"- Best silhouette: {k_info['best']['silhouette']:.3f}",
        f"- Mean seed-stability ARI: {stability['mean_adjusted_rand_index']:.3f}",
        "",
        "## Modes",
        "",
    ]
    for _, row in enrichment.iterrows():
        lines.append(
            f"- {row['mode_label']}: n={int(row['n_roi'])}, event fraction={row['event_fraction']:.2f}, p={row['fisher_p_value']:.3g}, cycles={row['top_cycles']}"
        )
    lines += [
        "",
        "## Top Review ROIs",
        "",
    ]
    for row in summary["top_review_rois"][:12]:
        lines.append(
            f"- {row['roi_id']} ({row['cohort_role']}, cycle {row['cycleNo']}): {row['mode_label']}, priority={row['mode_review_priority']:.3f}"
        )
    lines += [
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
