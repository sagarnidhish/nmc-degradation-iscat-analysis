#!/usr/bin/env python3
"""Joint ROI physics/degradation-mode synthesis for selected NMC event ROIs.

Combines selected-ROI image physics descriptors, event-conditioned rollout
residuals, residual-CNN negative-baseline metrics, selected front tracking, and
cycle-level event/echem evidence into one ROI table. The goal is to rank and
cluster degradation behavior using interpretable physics-facing features rather
than raw video loss alone.
"""

import argparse
import json
import os
import re
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def finite_float(x, default=np.nan) -> float:
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def roi_key_from_id(roi_id: str) -> str:
    # Supports cycle86_rank5_obj8 and cycle86_front5_obj8.
    m = re.search(r"cycle(\d+)_(?:rank|front)(\d+)_obj(\d+)", str(roi_id))
    if not m:
        return str(roi_id)
    return f"cycle{int(m.group(1))}_front{int(m.group(2))}_obj{int(m.group(3))}"


def add_roi_key(df: pd.DataFrame, id_col: str = "roi_id") -> pd.DataFrame:
    out = df.copy()
    if id_col in out:
        out["roi_key"] = out[id_col].map(roi_key_from_id)
    elif {"cycleNo", "front_candidate_rank", "object_candidate_rank"}.issubset(out.columns):
        out["roi_key"] = out.apply(lambda r: f"cycle{int(float(r['cycleNo']))}_front{int(r['front_candidate_rank'])}_obj{int(r['object_candidate_rank'])}", axis=1)
    return out


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mu = vals.mean()
    sigma = vals.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(vals)), index=series.index)
    return (vals - mu) / sigma


def choose_k(features: np.ndarray, max_k: int = 4) -> Dict[str, object]:
    rows = []
    best = {"k": 1, "silhouette": np.nan, "labels": np.zeros(features.shape[0], dtype=int)}
    n = features.shape[0]
    for k in range(2, min(max_k, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=50, random_state=29)
        labels = km.fit_predict(features)
        sil = float(silhouette_score(features, labels)) if len(set(labels)) > 1 else np.nan
        rows.append({"k": k, "silhouette": sil, "inertia": float(km.inertia_)})
        if not np.isfinite(best["silhouette"]) or sil > best["silhouette"]:
            best = {"k": k, "silhouette": sil, "labels": labels, "model": km}
    if "model" not in best:
        best["model"] = KMeans(n_clusters=1, n_init=1, random_state=29).fit(features)
    best["model_selection"] = rows
    return best


def name_modes(df: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    tmp = df.copy()
    tmp["cluster"] = labels
    rows = []
    for cluster, grp in tmp.groupby("cluster"):
        rows.append({
            "cluster": int(cluster),
            "score": finite_float(grp.get("joint_degradation_score", pd.Series([0])).mean(), 0),
            "residual": finite_float(grp["rollout_residual_energy_mean"].mean(), 0),
            "front": finite_float(grp["radius2_slope_full_px2_per_s"].mean(), 0),
            "loss": finite_float(grp["roi_mean_delta_corrected"].mean(), 0),
            "active": finite_float(grp["active_fraction_last"].mean(), 0),
        })
    if not rows:
        return {}
    by_score = sorted(rows, key=lambda r: r["score"], reverse=True)
    names: Dict[int, str] = {}
    for rank, row in enumerate(by_score):
        if rank == 0:
            if row["front"] < 0 and row["loss"] < 0:
                name = "highest_score_contracting_optical_loss"
            elif row["active"] >= np.median([r["active"] for r in rows]):
                name = "highest_score_active_front"
            else:
                name = "highest_score_mixed_residual"
        elif row["residual"] < np.median([r["residual"] for r in rows]):
            name = "lower_residual_comparison_mode"
        elif row["active"] >= np.median([r["active"] for r in rows]):
            name = "active_front_comparison_mode"
        else:
            name = "contracting_comparison_mode"
        names[row["cluster"]] = name
    return names


def save_plot(df: pd.DataFrame, out_png: str) -> None:
    plot_df = df.copy()
    feature_cols = [
        "rollout_residual_energy_mean",
        "radius2_slope_full_px2_per_s",
        "active_fraction_last",
        "roi_mean_delta_corrected",
        "cumulative_abs_change",
        "evidence_score",
    ]
    vals = plot_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    if vals.shape[0] > 1 and vals.shape[1] > 1:
        coords = PCA(n_components=2, random_state=29).fit_transform(StandardScaler().fit_transform(vals))
    else:
        coords = np.zeros((len(plot_df), 2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=plot_df["cycleNo"], cmap="viridis", s=70)
    for _, row in plot_df.iterrows():
        idx = int(row.name)
        axes[0].text(coords[idx, 0], coords[idx, 1], str(row["roi_short"]), fontsize=7)
    axes[0].set_title("joint ROI feature PCA", fontsize=9)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[0], label="cycle")
    for cycle, grp in plot_df.groupby("cycleNo"):
        axes[1].scatter(grp["radius2_slope_full_px2_per_s"], grp["rollout_residual_energy_mean"], label=f"cycle {int(cycle)}", s=70)
    axes[1].axvline(0, color="0.7", lw=1)
    axes[1].set_xlabel("front radius^2 slope (px^2/s)")
    axes[1].set_ylabel("rollout residual energy")
    axes[1].set_title("front motion vs model residual", fontsize=9)
    axes[1].legend(fontsize=8)
    counts = plot_df.groupby(["cycleNo", "joint_mode_name"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=axes[2])
    axes[2].set_title("joint mode counts", fontsize=9)
    axes[2].set_xlabel("cycle")
    axes[2].set_ylabel("ROI count")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/roi_joint_physics_degradation_modes")
    args = parser.parse_args()

    derived = args.derived_dir
    phys = add_roi_key(read_csv(os.path.join(derived, "roi_physics_descriptors", "roi_physics_descriptors.csv")))
    event_resid = add_roi_key(read_csv(os.path.join(derived, "roi_event_conditioned_nextframe", "rollout_residual_degradation_features.csv")))
    cnn = add_roi_key(read_csv(os.path.join(derived, "roi_residual_cnn_fast", "roi_residual_cnn_per_roi_metrics.csv")))
    front = add_roi_key(read_csv(os.path.join(derived, "selected_front_roi_tracking", "selected_front_roi_tracking_summary.csv")))
    event_evidence = read_csv(os.path.join(derived, "integrated_event_evidence", "integrated_event_evidence.csv"))
    echem = read_csv(os.path.join(derived, "event_echem_coupling", "echem_cycle_summary.csv"))
    local = read_csv(os.path.join(derived, "protocol_local_window_scan", "local_echem_window_features.csv"))

    if phys.empty or event_resid.empty:
        raise SystemExit("missing required ROI physics or residual feature tables")
    df = phys.merge(event_resid.drop(columns=["cycleNo", "validation_score"], errors="ignore"), on="roi_key", how="left", suffixes=("", "_event"))
    df = df.merge(cnn.drop(columns=["cycleNo", "validation_score"], errors="ignore"), on="roi_key", how="left", suffixes=("", "_cnn"))
    front_cols = [c for c in front.columns if c not in {"cycleNo", "source_stem", "validation_score"}]
    df = df.merge(front[front_cols], on="roi_key", how="left", suffixes=("", "_front"))
    if not event_evidence.empty:
        ev_cols = ["cycleNo", "mean_drop_frac", "n_event_particles", "global_frame_percentile", "evidence_score", "degradation_mode_hypothesis"]
        df = df.merge(event_evidence[[c for c in ev_cols if c in event_evidence.columns]], on="cycleNo", how="left")
    if not echem.empty:
        e_cols = ["cycleNo", "V_mean", "I_mean_mA", "I_abs_mean_mA", "V_range", "block_mode"]
        df = df.merge(echem[[c for c in e_cols if c in echem.columns]], on="cycleNo", how="left")
    if not local.empty:
        l_cols = [c for c in ["cycleNo", "local_window_role", "local_cycle_offset", "echem_points_percentile"] if c in local.columns]
        if l_cols:
            center = local[pd.to_numeric(local.get("local_cycle_offset", np.nan), errors="coerce") == 0] if "local_cycle_offset" in local else local
            center = center.drop_duplicates("cycleNo")
            df = df.merge(center[l_cols], on="cycleNo", how="left")

    for col in [
        "rollout_residual_energy_mean",
        "rollout_residual_energy_last",
        "radius2_slope_full_px2_per_s",
        "active_fraction_last",
        "roi_mean_delta_corrected",
        "cumulative_abs_change",
        "high_fraction_slope_per_frame",
        "apparent_diffusion_proxy_px2_per_frame",
        "relative_mse_improvement",
        "evidence_score",
        "mean_drop_frac",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["contraction_strength"] = np.clip(-pd.to_numeric(df["radius2_slope_full_px2_per_s"], errors="coerce"), 0, None)
    df["optical_loss_strength"] = np.clip(-pd.to_numeric(df["roi_mean_delta_corrected"], errors="coerce"), 0, None)
    df["ai_residual_strength"] = pd.to_numeric(df["rollout_residual_energy_mean"], errors="coerce")
    df["persistence_advantage_strength"] = np.clip(-pd.to_numeric(df["relative_mse_improvement"], errors="coerce"), 0, None)
    df["joint_degradation_score"] = (
        zscore(df["ai_residual_strength"]).fillna(0)
        + zscore(df["contraction_strength"]).fillna(0)
        + 0.7 * zscore(df["optical_loss_strength"]).fillna(0)
        + 0.5 * zscore(df["active_fraction_last"]).fillna(0)
        + 0.5 * zscore(df["evidence_score"]).fillna(0)
    )
    df["roi_short"] = df["roi_key"].str.replace("cycle", "c", regex=False).str.replace("_front", "f", regex=False).str.replace("_obj", "o", regex=False)

    feature_cols = [
        "rollout_residual_energy_mean",
        "rollout_residual_energy_slope_per_step",
        "radius2_slope_full_px2_per_s",
        "active_fraction_last",
        "roi_mean_delta_corrected",
        "cumulative_abs_change",
        "high_fraction_slope_per_frame",
        "apparent_diffusion_proxy_px2_per_frame",
        "relative_mse_improvement",
        "joint_degradation_score",
    ]
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True)).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    choice = choose_k(scaled, max_k=4)
    df["joint_mode_cluster"] = choice["labels"]
    names = name_modes(df, choice["labels"])
    df["joint_mode_name"] = df["joint_mode_cluster"].map(names)
    df = df.sort_values("joint_degradation_score", ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    joint_path = os.path.join(args.out_dir, "roi_joint_physics_degradation_modes.csv")
    df.to_csv(joint_path, index=False)
    mode_summary = df.groupby(["joint_mode_cluster", "joint_mode_name"], as_index=False).agg({
        "roi_key": "count",
        "cycleNo": "mean",
        "joint_degradation_score": "mean",
        "rollout_residual_energy_mean": "mean",
        "radius2_slope_full_px2_per_s": "mean",
        "roi_mean_delta_corrected": "mean",
        "active_fraction_last": "mean",
        "cumulative_abs_change": "mean",
    }).rename(columns={"roi_key": "n_roi", "cycleNo": "mean_cycle"})
    mode_path = os.path.join(args.out_dir, "joint_mode_summary.csv")
    mode_summary.to_csv(mode_path, index=False)
    cycle_summary = df.groupby("cycleNo", as_index=False).agg({
        "roi_key": "count",
        "joint_degradation_score": "mean",
        "rollout_residual_energy_mean": "mean",
        "radius2_slope_full_px2_per_s": "mean",
        "roi_mean_delta_corrected": "mean",
        "active_fraction_last": "mean",
        "mean_drop_frac": "first",
        "evidence_score": "first",
    }).rename(columns={"roi_key": "n_roi"})
    cycle_path = os.path.join(args.out_dir, "joint_cycle_summary.csv")
    cycle_summary.to_csv(cycle_path, index=False)
    corr_cols = ["cycleNo", "mean_drop_frac", "evidence_score"] + feature_cols
    corr = df[[c for c in corr_cols if c in df.columns]].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    corr_path = os.path.join(args.out_dir, "joint_feature_spearman_correlations.csv")
    corr.to_csv(corr_path)
    model_path = os.path.join(args.out_dir, "joint_mode_model_selection.csv")
    pd.DataFrame(choice["model_selection"]).to_csv(model_path, index=False)
    plot_path = os.path.join(args.out_dir, "roi_joint_physics_degradation_modes.png")
    save_plot(df.reset_index(drop=True), plot_path)

    summary = {
        "n_roi": int(len(df)),
        "feature_columns": feature_cols,
        "selected_k": int(choice["k"]),
        "silhouette": finite_float(choice["silhouette"]),
        "mode_names": {str(k): v for k, v in names.items()},
        "cycle_summary": cycle_summary.to_dict(orient="records"),
        "top_ranked_rois": df.head(8)[["roi_key", "cycleNo", "joint_degradation_score", "joint_mode_name", "rollout_residual_energy_mean", "radius2_slope_full_px2_per_s", "roi_mean_delta_corrected", "active_fraction_last"]].to_dict(orient="records"),
        "guardrail": "Joint modes combine automatic ROI/model/front proxies. They rank degradation hypotheses but are not calibrated diffusion coefficients or manual annotations.",
        "outputs": {
            "joint_table": joint_path,
            "mode_summary": mode_path,
            "cycle_summary": cycle_path,
            "correlations": corr_path,
            "plot": plot_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "roi_joint_physics_degradation_modes_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Joint ROI Physics Degradation Modes\n\n")
        f.write("Combines ROI physics descriptors, front tracking, rollout residuals, residual-CNN guardrail metrics, and cycle-level event/echem evidence.\n\n")
        f.write("The resulting modes are automatic hypothesis rankings, not manual annotations or calibrated diffusion coefficients.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
