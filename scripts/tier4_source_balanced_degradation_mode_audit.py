#!/usr/bin/env python3
"""Unsupervised degradation-state modes from source-balanced ROI residual physics.

This audit clusters source-residualized particle-region residual/front/contrast
features, then tests how the resulting modes distribute across pre-event,
post-event, and far-from-event cycle neighborhoods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "masked_minus_background_mean_slope",
    "masked_minus_background_mean_median",
    "roi_norm_mean_delta_last_minus_first",
    "front_radius_q60_slope_px_per_norm_time",
    "front_radius_q70_slope_px_per_norm_time",
    "front_radius_q80_slope_px_per_norm_time",
    "front_radius_q70_positive_step_fraction",
    "front_gradient_peak_radius_slope_px_per_norm_time",
    "mask_area_fraction_slope",
    "mask_centroid_drift_px",
    "residual_energy_slope",
    "residual_energy_last_minus_first",
    "dictionary_recon_error_mse_slope",
    "dictionary_recon_error_last_minus_first",
    "resdict_pc01_mean",
    "resdict_pc02_slope",
    "resdict_pc09_slope",
    "resdict_pc10_mean",
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


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def source_residual(values: pd.Series, sources: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    return x - x.groupby(sources.astype(str)).transform("mean")


def load_event_cycles(path: Path) -> np.ndarray:
    events = pd.read_csv(path)
    return np.sort(pd.to_numeric(events["cycleNo"], errors="coerce").dropna().unique().astype(float))


def append_event_context(df: pd.DataFrame, event_cycles: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    cycles = numeric(out, "cycleNo").to_numpy(dtype=float)
    rows = []
    for cycle in cycles:
        deltas = event_cycles - cycle
        future_pos = deltas[deltas > 0]
        past_pos = -deltas[deltas < 0]
        to_next = float(future_pos.min()) if len(future_pos) else np.nan
        since_prev = float(past_pos.min()) if len(past_pos) else np.nan
        if np.any(deltas == 0):
            phase = "current_event"
        elif np.isfinite(to_next) and to_next <= 8:
            phase = "pre_event_8"
        elif np.isfinite(to_next) and to_next <= 16:
            phase = "pre_event_16"
        elif np.isfinite(since_prev) and since_prev <= 8:
            phase = "post_event_8"
        elif np.isfinite(since_prev) and since_prev <= 16:
            phase = "post_event_16"
        else:
            phase = "far_from_event"
        rows.append({
            "cycles_to_next_event": to_next,
            "cycles_since_prev_event": since_prev,
            "event_neighborhood_phase": phase,
            "future8": int(np.isfinite(to_next) and to_next <= 8),
            "future16": int(np.isfinite(to_next) and to_next <= 16),
            "past8": int(np.isfinite(since_prev) and since_prev <= 8),
            "past16": int(np.isfinite(since_prev) and since_prev <= 16),
        })
    return pd.concat([out, pd.DataFrame(rows, index=out.index)], axis=1)


def available_features(df: pd.DataFrame, features: Iterable[str]) -> List[str]:
    out = []
    for feature in features:
        if feature in df.columns:
            vals = numeric(df, feature)
            if vals.notna().sum() >= 24 and vals.nunique(dropna=True) >= 2:
                out.append(feature)
    return out


def transformed_feature_table(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    sources = df["source_stem"].astype(str)
    table = {}
    for feature in features:
        table[f"{feature}__source_residual"] = source_residual(numeric(df, feature), sources)
    return pd.DataFrame(table, index=df.index)


def choose_k(x: np.ndarray, seed: int) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in range(2, min(7, len(x) - 1)):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=50).fit_predict(x)
        score = silhouette_score(x, labels) if len(set(labels)) > 1 else np.nan
        rows.append({"k": k, "silhouette": float(score)})
    scores = pd.DataFrame(rows)
    if scores.empty:
        return 2, scores
    best = scores.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best["k"]), scores


def feature_family(feature: str) -> str:
    raw = feature.replace("__source_residual", "")
    if raw.startswith("front_") or raw.startswith("mask_area") or raw.startswith("mask_centroid"):
        return "front_mask_geometry"
    if raw.startswith("masked_") or raw.startswith("roi_norm"):
        return "optical_contrast"
    if raw.startswith("dictionary_") or raw.startswith("resdict_") or raw.startswith("residual_energy"):
        return "residual_dictionary"
    return "other"


def label_mode(top_features: List[str]) -> str:
    families = pd.Series([feature_family(f) for f in top_features]).value_counts()
    primary = str(families.index[0]) if len(families) else "mixed"
    if primary == "optical_contrast":
        return "source_residual_optical_contrast_state"
    if primary == "front_mask_geometry":
        return "source_residual_front_geometry_state"
    if primary == "residual_dictionary":
        return "source_residual_temporal_dictionary_state"
    return "source_residual_mixed_state"


def cluster_feature_summary(assignments: pd.DataFrame, transformed: pd.DataFrame) -> pd.DataFrame:
    joined = pd.concat([assignments[["mode"]], transformed], axis=1)
    global_mean = transformed.mean()
    global_std = transformed.std().replace(0, np.nan)
    rows = []
    for mode, sub in joined.groupby("mode"):
        means = sub[transformed.columns].mean()
        z = ((means - global_mean) / global_std).sort_values(key=lambda s: s.abs(), ascending=False)
        for rank, (feature, value) in enumerate(z.head(8).items(), start=1):
            rows.append({
                "mode": int(mode),
                "rank": rank,
                "feature": feature,
                "feature_family": feature_family(feature),
                "mode_mean_source_residual": float(means[feature]),
                "mode_vs_global_z": float(value) if np.isfinite(value) else np.nan,
            })
    return pd.DataFrame(rows)


def enrichment_rows(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = ["future8", "future16", "past8", "past16"]
    phases = sorted(assignments["event_neighborhood_phase"].dropna().unique())
    for mode in sorted(assignments["mode"].unique()):
        in_mode = assignments["mode"].eq(mode)
        for label in labels:
            positive = numeric(assignments, label).eq(1)
            table = [
                [int((in_mode & positive).sum()), int((in_mode & ~positive).sum())],
                [int((~in_mode & positive).sum()), int((~in_mode & ~positive).sum())],
            ]
            try:
                odds, pval = fisher_exact(table)
            except ValueError:
                odds, pval = np.nan, np.nan
            rows.append({
                "mode": int(mode),
                "label": label,
                "mode_positive": table[0][0],
                "mode_total": int(in_mode.sum()),
                "outside_positive": table[1][0],
                "outside_total": int((~in_mode).sum()),
                "mode_fraction": table[0][0] / max(1, int(in_mode.sum())),
                "outside_fraction": table[1][0] / max(1, int((~in_mode).sum())),
                "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
                "fisher_p": float(pval) if np.isfinite(pval) else np.nan,
                "enrichment_type": "binary_label",
            })
        for phase in phases:
            positive = assignments["event_neighborhood_phase"].eq(phase)
            table = [
                [int((in_mode & positive).sum()), int((in_mode & ~positive).sum())],
                [int((~in_mode & positive).sum()), int((~in_mode & ~positive).sum())],
            ]
            try:
                odds, pval = fisher_exact(table)
            except ValueError:
                odds, pval = np.nan, np.nan
            rows.append({
                "mode": int(mode),
                "label": phase,
                "mode_positive": table[0][0],
                "mode_total": int(in_mode.sum()),
                "outside_positive": table[1][0],
                "outside_total": int((~in_mode).sum()),
                "mode_fraction": table[0][0] / max(1, int(in_mode.sum())),
                "outside_fraction": table[1][0] / max(1, int((~in_mode).sum())),
                "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
                "fisher_p": float(pval) if np.isfinite(pval) else np.nan,
                "enrichment_type": "event_phase",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["fisher_p", "mode_fraction"], ascending=[True, False])
    return out


def mode_sequence(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, sub in assignments.sort_values(["source_stem", "cycleNo"]).groupby("source_stem"):
        modes = sub["mode_label"].tolist()
        cycles = sub["cycleNo"].tolist()
        for i in range(len(sub) - 1):
            rows.append({
                "source_stem": source,
                "cycle_from": cycles[i],
                "cycle_to": cycles[i + 1],
                "mode_from": modes[i],
                "mode_to": modes[i + 1],
                "changed": modes[i] != modes[i + 1],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_residual_dictionary_audit")
    parser.add_argument("--event-cycles", default="/scratch/<account>/<username>/Alek_Jiho/derived/particle_event_targets/particle_abrupt_events.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_degradation_mode_audit")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.feature_dir) / "source_balanced_residual_dictionary_features.csv")
    event_cycles = load_event_cycles(Path(args.event_cycles))
    df = append_event_context(df, event_cycles)
    features = available_features(df, FEATURES)
    transformed = transformed_feature_table(df, features)
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    x = pipeline.fit_transform(transformed)
    k, k_scores = choose_k(x, args.seed)
    kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=100).fit(x)
    pca = PCA(n_components=min(3, x.shape[1]), random_state=args.seed)
    pcs = pca.fit_transform(x)

    assignments = df[[
        "roi_id", "cycleNo", "source_stem", "npz_path", "preview_png",
        "cycles_to_next_event", "cycles_since_prev_event", "event_neighborhood_phase",
        "future8", "future16", "past8", "past16",
    ]].copy()
    assignments["mode"] = kmeans.labels_
    assignments["mode_distance"] = np.linalg.norm(x - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    for i in range(pcs.shape[1]):
        assignments[f"mode_pc{i + 1}"] = pcs[:, i]

    feature_summary = cluster_feature_summary(assignments, transformed)
    mode_labels = {}
    for mode, sub in feature_summary.groupby("mode"):
        top = sub.sort_values("rank")["feature"].head(4).tolist()
        mode_labels[int(mode)] = label_mode(top)
    assignments["mode_label"] = assignments["mode"].map(mode_labels)

    cluster_rows = []
    for mode, sub in assignments.groupby("mode"):
        phase_counts = sub["event_neighborhood_phase"].value_counts().to_dict()
        top_features = feature_summary[feature_summary["mode"] == mode].sort_values("rank").head(5)
        cluster_rows.append({
            "mode": int(mode),
            "mode_label": mode_labels[int(mode)],
            "n_roi": int(len(sub)),
            "n_cycles": int(sub["cycleNo"].nunique()),
            "n_sources": int(sub["source_stem"].nunique()),
            "future8_fraction": float(sub["future8"].mean()),
            "future16_fraction": float(sub["future16"].mean()),
            "past16_fraction": float(sub["past16"].mean()),
            "median_cycles_to_next_event": float(pd.to_numeric(sub["cycles_to_next_event"], errors="coerce").median()) if sub["cycles_to_next_event"].notna().any() else np.nan,
            "median_cycles_since_prev_event": float(pd.to_numeric(sub["cycles_since_prev_event"], errors="coerce").median()) if sub["cycles_since_prev_event"].notna().any() else np.nan,
            "phase_counts": phase_counts,
            "top_features": top_features[["feature", "mode_vs_global_z"]].to_dict(orient="records"),
        })
    cluster_summary = pd.DataFrame(cluster_rows).sort_values("mode")
    enrichment = enrichment_rows(assignments)
    sequence = mode_sequence(assignments)
    representatives = assignments.sort_values("mode_distance").groupby("mode", as_index=False).head(4)

    paths = {
        "assignments": out / "source_balanced_degradation_mode_assignments.csv",
        "cluster_summary": out / "source_balanced_degradation_mode_cluster_summary.csv",
        "feature_summary": out / "source_balanced_degradation_mode_feature_summary.csv",
        "enrichment": out / "source_balanced_degradation_mode_event_enrichment.csv",
        "sequence": out / "source_balanced_degradation_mode_source_sequence.csv",
        "representatives": out / "source_balanced_degradation_mode_representatives.csv",
        "k_scores": out / "source_balanced_degradation_mode_k_scores.csv",
        "summary": out / "source_balanced_degradation_mode_summary.json",
    }
    assignments.to_csv(paths["assignments"], index=False)
    cluster_summary.to_csv(paths["cluster_summary"], index=False)
    feature_summary.to_csv(paths["feature_summary"], index=False)
    enrichment.to_csv(paths["enrichment"], index=False)
    sequence.to_csv(paths["sequence"], index=False)
    representatives.to_csv(paths["representatives"], index=False)
    k_scores.to_csv(paths["k_scores"], index=False)

    top_enrichment = enrichment.head(24).to_dict(orient="records") if not enrichment.empty else []
    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "event_cycles": [float(c) for c in event_cycles],
        "features_used": features,
        "n_features_used": int(len(features)),
        "chosen_k": int(k),
        "k_scores": k_scores.to_dict(orient="records"),
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "top_enrichment": top_enrichment,
        "representatives": representatives.head(24).to_dict(orient="records"),
        "source_mode_transition_count": int(len(sequence)),
        "source_mode_change_fraction": float(sequence["changed"].mean()) if len(sequence) else np.nan,
        "outputs": {k: str(v) for k, v in paths.items()},
        "guardrail": "Modes are unsupervised source-residual clusters of automatic particle-region optical/residual proxies. They organize degradation-state hypotheses and review candidates, but they do not prove calibrated phase boundaries, diffusion coefficients, or causal event mechanisms.",
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8")

    readme = [
        "# Source-Balanced Degradation Mode Audit",
        "",
        f"Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}",
        f"Chosen k: {summary['chosen_k']}; source-mode change fraction: {summary['source_mode_change_fraction']:.3f}",
        "",
        "## Modes",
    ]
    for row in summary["cluster_summary"]:
        readme.append(
            f"- mode {row['mode']} {row['mode_label']}: n={row['n_roi']}, sources={row['n_sources']}, "
            f"future16={row['future16_fraction']:.3f}, past16={row['past16_fraction']:.3f}, phases={row['phase_counts']}"
        )
    readme.extend(["", "## Top Enrichment"])
    for row in top_enrichment[:8]:
        readme.append(
            f"- mode {row['mode']} {row['label']}: fraction={row['mode_fraction']:.3f} vs {row['outside_fraction']:.3f}, p={row['fisher_p']:.3g}"
        )
    readme.extend(["", "## Guardrail", summary["guardrail"], ""])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2))


if __name__ == "__main__":
    main()
