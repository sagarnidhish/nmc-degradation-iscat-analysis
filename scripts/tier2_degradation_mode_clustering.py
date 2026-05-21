#!/usr/bin/env python3
"""Cluster particle-cycle degradation modes from optical and echem features."""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-table", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_targets/particle_event_training_table.csv")
    parser.add_argument("--cycle-table", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_echem_coupling/event_echem_cycle_table.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/degradation_mode_clustering")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    events = pd.read_csv(args.event_table)
    cycles = pd.read_csv(args.cycle_table)
    events["cycleNo"] = pd.to_numeric(events["cycleNo"], errors="coerce")
    cycles["cycleNo"] = pd.to_numeric(cycles["cycleNo"], errors="coerce")
    merged = events.merge(cycles, on="cycleNo", how="left", suffixes=("", "_cycle"))

    feature_cols = [
        "norm_intensity", "delta_norm", "trailing_mean_norm", "trailing_std_norm", "trailing_min_delta", "trailing_slope_norm",
        "n_frames_percentile", "V_mean", "V_range", "I_mean_mA", "I_abs_mean_mA", "V_mean_delta", "I_mean_delta",
    ]
    feature_cols = [c for c in feature_cols if c in merged.columns]
    Xdf = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True)).fillna(0.0)
    X = StandardScaler().fit_transform(Xdf.to_numpy(dtype=float))

    candidates = []
    best = None
    for k in range(args.k_min, min(args.k_max, len(merged) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=args.seed, n_init=50)
        labels = model.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else np.nan
        candidates.append({"k": k, "silhouette": sil, "inertia": float(model.inertia_)})
        if best is None or (np.isfinite(sil) and sil > best["silhouette"]):
            best = {"k": k, "silhouette": sil, "labels": labels, "model": model}
    if best is None:
        raise SystemExit("No clustering candidate could be fit")

    merged["degradation_mode"] = best["labels"]
    mode_summary = []
    for mode, g in merged.groupby("degradation_mode"):
        row = {
            "degradation_mode": int(mode),
            "n_rows": int(len(g)),
            "event_rate": float(pd.to_numeric(g["is_abrupt_drop"], errors="coerce").fillna(0).mean()),
            "mean_cycle": float(g["cycleNo"].mean()),
        }
        for col in feature_cols:
            row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
        mode_summary.append(row)

    # Human-readable labels based on dominant optical behavior.
    label_map = {}
    for row in mode_summary:
        mode = row["degradation_mode"]
        if row.get("event_rate", 0) > 0.05 or row.get("trailing_min_delta_mean", 0) < -0.10:
            label = "abrupt_drop_risk"
        elif row.get("trailing_slope_norm_mean", 0) < -0.01:
            label = "gradual_decline"
        elif row.get("delta_norm_mean", 0) > 0.02:
            label = "recovery_or_brightening"
        else:
            label = "stable_or_slow_drift"
        label_map[int(mode)] = label
    merged["degradation_mode_label"] = merged["degradation_mode"].map(label_map)

    os.makedirs(args.out_dir, exist_ok=True)
    merged.to_csv(os.path.join(args.out_dir, "particle_cycle_degradation_modes.csv"), index=False)
    pd.DataFrame(mode_summary).to_csv(os.path.join(args.out_dir, "degradation_mode_summary.csv"), index=False)
    pd.DataFrame(candidates).to_csv(os.path.join(args.out_dir, "kmeans_model_selection.csv"), index=False)
    summary = {
        "event_table": args.event_table,
        "cycle_table": args.cycle_table,
        "feature_cols": feature_cols,
        "best_k": int(best["k"]),
        "best_silhouette": float(best["silhouette"]),
        "label_map": label_map,
        "interpretation": "Clusters are exploratory degradation modes. They require raw ROI/video validation before being treated as physical mechanisms.",
    }
    with open(os.path.join(args.out_dir, "degradation_mode_clustering_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Degradation Mode Clustering\n\nExploratory particle-cycle clustering into optical/echem degradation modes.\n")
    print(f"[done] wrote degradation mode clustering to {args.out_dir}; best_k={best['k']}")


if __name__ == "__main__":
    main()
