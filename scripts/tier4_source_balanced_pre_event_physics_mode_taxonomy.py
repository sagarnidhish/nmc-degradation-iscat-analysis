#!/usr/bin/env python3
"""Source-balanced pre-event physics-mode taxonomy.

This audit clusters source-normalized particle-region front, mask, diffusion-
like, and heterogeneity descriptors from the source-balanced pre-event cohort.
The goal is to ask whether the pre-event signal is a repeatable mode/state
rather than only a target-specific classifier. Clusters remain automatic
hypothesis labels, not manual degradation modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PRE_BINS = {"near_pre_event_1_8", "mid_pre_event_9_16", "far_pre_event_17_32"}
PHYSICS_KEYS = (
    "masked_minus_background",
    "mask_",
    "front_",
    "apparent_diffusion",
    "spatial_std",
    "frame_diff",
    "temporal_energy",
)


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


def source_eta2(values: pd.Series, sources: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce")
    valid = x.notna() & sources.notna()
    x = x[valid]
    src = sources[valid].astype(str)
    if len(x) < 4 or x.nunique() < 2 or src.nunique() < 2:
        return np.nan
    total = float(((x - x.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, sub in x.groupby(src):
        between += len(sub) * float((sub.mean() - x.mean()) ** 2)
    return between / total


def feature_columns(df: pd.DataFrame) -> List[str]:
    blocked = {
        "roi_id",
        "source_stem",
        "event_relative_bin",
        "npz_path",
        "preview_png",
        "selection_reason",
        "validation_label",
        "cohort_role",
        "clean_pre_1_8_vs_post_control",
        "clean_pre_1_16_vs_post_control",
        "clean_pre_1_32_vs_post_control",
        "near_pre_vs_far_pre",
        "post_event_vs_control",
    }
    cols: List[str] = []
    for col in df.columns:
        if col in blocked:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 24 and vals.nunique(dropna=True) >= 3 and any(k in col for k in PHYSICS_KEYS):
            cols.append(col)
    return cols


def source_residualize(df: pd.DataFrame, cols: Iterable[str], sources: pd.Series) -> pd.DataFrame:
    vals = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    means = vals.groupby(sources.astype(str)).transform("mean")
    return vals - means


def add_event_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = out["event_relative_bin"].astype(str)
    out["is_pre_event"] = bins.isin(PRE_BINS).astype(int)
    out["is_near_pre"] = bins.eq("near_pre_event_1_8").astype(int)
    out["is_mid_pre"] = bins.eq("mid_pre_event_9_16").astype(int)
    out["is_far_pre"] = bins.eq("far_pre_event_17_32").astype(int)
    out["is_post_event"] = bins.eq("post_event_1_16").astype(int)
    out["is_control"] = bins.eq("no_near_event_control").astype(int)
    out["pre_event_clock_closer"] = np.where(bins.isin(PRE_BINS), -numeric(out, "cycles_to_next_event"), np.nan)
    out["event_cycle"] = numeric(out, "cycleNo") + numeric(out, "cycles_to_next_event")
    return out


def choose_k(x: np.ndarray, min_k: int, max_k: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    upper = min(max_k, max(min_k, x.shape[0] - 1))
    for k in range(min_k, upper + 1):
        model = KMeans(n_clusters=k, n_init=50, random_state=seed)
        labels = model.fit_predict(x)
        counts = pd.Series(labels).value_counts()
        sil = silhouette_score(x, labels) if len(counts) > 1 and counts.min() >= 2 else np.nan
        rows.append({
            "k": k,
            "silhouette": float(sil) if np.isfinite(sil) else np.nan,
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "inertia": float(model.inertia_),
        })
    return pd.DataFrame(rows).sort_values(["silhouette", "min_cluster_size"], ascending=[False, False])


def cluster_labels(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=100, random_state=seed).fit_predict(x)


def label_mode(row: pd.Series, top_features: List[str]) -> str:
    joined = " ".join(top_features)
    parts = []
    if "apparent_diffusion" in joined:
        parts.append("diffusion_proxy")
    if "front_radius" in joined or "front_gradient" in joined or "front_" in joined:
        parts.append("front_geometry")
    if "masked_minus_background" in joined or "mask_" in joined:
        parts.append("mask_contrast")
    if "spatial_std" in joined or "frame_diff" in joined or "temporal_energy" in joined:
        parts.append("video_heterogeneity")
    if row.get("near_pre_fraction", 0) >= 0.35:
        parts.append("near_pre_enriched")
    if row.get("post_event_fraction", 0) >= 0.35:
        parts.append("post_event_enriched")
    return "_".join(parts[:4]) if parts else "mixed_source_residual_state"


def mode_summary(assignments: pd.DataFrame, residual: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    global_means = residual.mean(axis=0)
    global_stds = residual.std(axis=0).replace(0, np.nan)
    for mode, sub in assignments.groupby("mode"):
        idx = sub.index
        z = ((residual.loc[idx].mean(axis=0) - global_means) / global_stds).replace([np.inf, -np.inf], np.nan)
        top = z.abs().sort_values(ascending=False).head(6)
        row = {
            "mode": int(mode),
            "n_roi": int(len(sub)),
            "n_cycles": int(sub["cycleNo"].nunique()),
            "n_sources": int(sub["source_stem"].nunique()),
            "near_pre_fraction": float(sub["is_near_pre"].mean()),
            "mid_pre_fraction": float(sub["is_mid_pre"].mean()),
            "far_pre_fraction": float(sub["is_far_pre"].mean()),
            "post_event_fraction": float(sub["is_post_event"].mean()),
            "control_fraction": float(sub["is_control"].mean()),
            "median_cycles_to_next_event": float(numeric(sub, "cycles_to_next_event").median()) if numeric(sub, "cycles_to_next_event").notna().any() else np.nan,
            "median_pre_event_clock_closer": float(numeric(sub, "pre_event_clock_closer").median()) if numeric(sub, "pre_event_clock_closer").notna().any() else np.nan,
            "top_loading_features": ";".join(top.index.astype(str).tolist()),
            "top_loading_z": ";".join(f"{v:.3f}" for v in top.to_list()),
        }
        row["mode_label"] = label_mode(pd.Series(row), top.index.astype(str).tolist())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["near_pre_fraction", "n_roi"], ascending=[False, False])


def enrichment_tests(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    labels = {
        "near_pre": "is_near_pre",
        "pre_event": "is_pre_event",
        "post_event": "is_post_event",
        "control": "is_control",
    }
    for mode, sub in assignments.groupby("mode"):
        in_mode = assignments["mode"].eq(mode)
        for label, col in labels.items():
            a = int((in_mode & assignments[col].eq(1)).sum())
            b = int((in_mode & assignments[col].eq(0)).sum())
            c = int((~in_mode & assignments[col].eq(1)).sum())
            d = int((~in_mode & assignments[col].eq(0)).sum())
            try:
                odds, p = fisher_exact([[a, b], [c, d]])
            except ValueError:
                odds, p = np.nan, np.nan
            rows.append({
                "mode": int(mode),
                "label": label,
                "mode_positive": a,
                "mode_total": a + b,
                "outside_positive": c,
                "outside_total": c + d,
                "mode_fraction": a / (a + b) if (a + b) else np.nan,
                "outside_fraction": c / (c + d) if (c + d) else np.nan,
                "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
                "fisher_p": float(p) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["fisher_p", "mode_fraction"], ascending=[True, False])


def clock_tests(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    clock = numeric(assignments, "pre_event_clock_closer")
    for mode, sub in assignments.groupby("mode"):
        in_mode = assignments["mode"].eq(mode)
        x1 = clock[in_mode & clock.notna()]
        x0 = clock[~in_mode & clock.notna()]
        if len(x1) >= 2 and len(x0) >= 2:
            try:
                _, p_mwu = mannwhitneyu(x1, x0, alternative="two-sided")
            except ValueError:
                p_mwu = np.nan
        else:
            p_mwu = np.nan
        groups = [
            clock[assignments["event_relative_bin"].eq(bin_name) & in_mode & clock.notna()].to_numpy(dtype=float)
            for bin_name in ["far_pre_event_17_32", "mid_pre_event_9_16", "near_pre_event_1_8"]
        ]
        groups = [g for g in groups if len(g) >= 2]
        try:
            _, p_kruskal = kruskal(*groups) if len(groups) >= 2 else (np.nan, np.nan)
        except ValueError:
            p_kruskal = np.nan
        rows.append({
            "mode": int(mode),
            "n_pre_rows": int(len(x1)),
            "median_pre_clock_in_mode": float(x1.median()) if len(x1) else np.nan,
            "median_pre_clock_outside": float(x0.median()) if len(x0) else np.nan,
            "median_in_minus_out": float(x1.median() - x0.median()) if len(x1) and len(x0) else np.nan,
            "mwu_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
            "within_mode_prebin_kruskal_p": float(p_kruskal) if np.isfinite(p_kruskal) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["mwu_p", "median_in_minus_out"], ascending=[True, False])


def representatives(assignments: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for mode, sub in assignments.groupby("mode"):
        idx = sub.index.to_numpy()
        center = x[idx].mean(axis=0)
        d = np.linalg.norm(x[idx] - center, axis=1)
        for order, pos in enumerate(np.argsort(d)[:4], start=1):
            row = sub.iloc[pos].to_dict()
            rows.append({
                "mode": int(mode),
                "mode_rank": order,
                "roi_id": row.get("roi_id"),
                "cycleNo": row.get("cycleNo"),
                "source_stem": row.get("source_stem"),
                "event_relative_bin": row.get("event_relative_bin"),
                "cycles_to_next_event": row.get("cycles_to_next_event"),
                "cycles_since_prev_event": row.get("cycles_since_prev_event"),
                "distance_to_mode_center": float(d[pos]),
            })
    return pd.DataFrame(rows).sort_values(["mode", "mode_rank"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_directionality_audit/source_balanced_pre_event_directionality_features.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_physics_mode_taxonomy")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = add_event_context(pd.read_csv(args.features).loc[:, lambda d: ~d.columns.duplicated()].copy())
    features = feature_columns(df)
    residual = source_residualize(df, features, df["source_stem"])
    eta_rows = [{"feature": f, "raw_source_eta2": source_eta2(df[f], df["source_stem"]), "residual_source_eta2": source_eta2(residual[f], df["source_stem"])} for f in features]
    eta_df = pd.DataFrame(eta_rows).sort_values("raw_source_eta2", ascending=False)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(residual))
    k_scores = choose_k(x, args.min_k, args.max_k, args.seed)
    chosen_k = int(k_scores.iloc[0]["k"]) if not k_scores.empty else 2
    labels = cluster_labels(x, chosen_k, args.seed)

    assignments = df.copy()
    assignments["mode"] = labels
    summary = mode_summary(assignments, residual, features)
    enrich = enrichment_tests(assignments)
    clocks = clock_tests(assignments)
    reps = representatives(assignments, x)

    assignments.to_csv(out / "source_balanced_pre_event_physics_mode_assignments.csv", index=False)
    summary.to_csv(out / "source_balanced_pre_event_physics_mode_summary.csv", index=False)
    enrich.to_csv(out / "source_balanced_pre_event_physics_mode_enrichment.csv", index=False)
    clocks.to_csv(out / "source_balanced_pre_event_physics_mode_clock_tests.csv", index=False)
    reps.to_csv(out / "source_balanced_pre_event_physics_mode_representatives.csv", index=False)
    eta_df.to_csv(out / "source_balanced_pre_event_physics_mode_feature_source_eta.csv", index=False)
    k_scores.to_csv(out / "source_balanced_pre_event_physics_mode_k_scores.csv", index=False)

    payload = {
        "input_features": str(args.features),
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "n_features_used": int(len(features)),
        "chosen_k": chosen_k,
        "k_scores": clean_json(k_scores.to_dict("records")),
        "mode_summary": clean_json(summary.to_dict("records")),
        "top_enrichment": clean_json(enrich.head(20).to_dict("records")),
        "clock_tests": clean_json(clocks.to_dict("records")),
        "representatives": clean_json(reps.to_dict("records")),
        "top_source_confounded_features": clean_json(eta_df.head(16).to_dict("records")),
        "outputs": {
            "assignments": str(out / "source_balanced_pre_event_physics_mode_assignments.csv"),
            "summary": str(out / "source_balanced_pre_event_physics_mode_summary.csv"),
            "enrichment": str(out / "source_balanced_pre_event_physics_mode_enrichment.csv"),
            "clock_tests": str(out / "source_balanced_pre_event_physics_mode_clock_tests.csv"),
            "representatives": str(out / "source_balanced_pre_event_physics_mode_representatives.csv"),
            "feature_source_eta": str(out / "source_balanced_pre_event_physics_mode_feature_source_eta.csv"),
            "k_scores": str(out / "source_balanced_pre_event_physics_mode_k_scores.csv"),
        },
        "guardrail": (
            "Pre-event physics modes are unsupervised clusters of automatic source-residual ROI features. "
            "Mode enrichment and event-clock tests nominate repeatable front/diffusion/heterogeneity states for review; "
            "they are not manual degradation labels, calibrated phase boundaries, diffusion coefficients, or causal forecasts."
        ),
    }
    (out / "source_balanced_pre_event_physics_mode_summary.json").write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n"
    )
    (out / "README.md").write_text(
        "# Source-Balanced Pre-Event Physics Mode Taxonomy\n\n"
        f"- Rows/cycles/sources: {payload['n_rows']} / {payload['n_cycles']} / {payload['n_sources']}\n"
        f"- Features used: {payload['n_features_used']}\n"
        f"- Chosen k: {payload['chosen_k']}\n\n"
        "## Guardrail\n\n"
        f"{payload['guardrail']}\n"
    )
    print(json.dumps(clean_json(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
