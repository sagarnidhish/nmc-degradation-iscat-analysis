#!/usr/bin/env python3
"""Build particle-masked video embeddings across ROI cohorts.

This audit treats existing ROI tensors as particle-region videos, builds a
history-derived particle prior for each tensor, extracts label-free temporal and
spatial-temporal descriptors, and then tests whether those descriptors carry
weak future-drop or event/control signal under grouped cycle splits.

The embeddings are for hypothesis ranking and cohort comparison. They are not
manual particle labels, calibrated diffusion estimates, or a deployable detector.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tier4_particle_mask_stability_audit import build_prior_mask, trace_masks


def clean_value(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_value(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    return x


def finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    val = df[col]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return pd.to_numeric(val, errors="coerce")


def robust_scale_trace(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad if np.isfinite(mad) and mad > 1e-9 else np.nanstd(x)
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - med) / scale).astype(np.float32)


def slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = np.linspace(0.0, 1.0, len(y))[mask]
    return float(np.polyfit(x, y[mask], 1)[0])


def trace_stats(prefix: str, trace: np.ndarray) -> Dict[str, float]:
    trace = np.asarray(trace, dtype=float)
    diff = np.diff(trace)
    first = trace[: max(3, len(trace) // 3)]
    last = trace[-max(3, len(trace) // 3) :]
    return {
        f"{prefix}_mean": finite(np.nanmean(trace)),
        f"{prefix}_std": finite(np.nanstd(trace)),
        f"{prefix}_slope": slope(trace),
        f"{prefix}_first_mean": finite(np.nanmean(first)),
        f"{prefix}_last_mean": finite(np.nanmean(last)),
        f"{prefix}_last_minus_first": finite(np.nanmean(last) - np.nanmean(first)),
        f"{prefix}_diff_abs_mean": finite(np.nanmean(np.abs(diff))) if diff.size else np.nan,
        f"{prefix}_diff_q10": finite(np.nanquantile(diff, 0.10)) if diff.size else np.nan,
        f"{prefix}_diff_q90": finite(np.nanquantile(diff, 0.90)) if diff.size else np.nan,
        f"{prefix}_diff_positive_fraction": finite(np.nanmean(diff > 0)) if diff.size else np.nan,
    }


def average_pool(frame: np.ndarray, out_size: int = 12) -> np.ndarray:
    h, w = frame.shape
    if h % out_size == 0 and w % out_size == 0:
        return frame.reshape(out_size, h // out_size, out_size, w // out_size).mean(axis=(1, 3))
    zoom = (out_size / h, out_size / w)
    return ndi.zoom(frame, zoom=zoom, order=1)


def spatial_temporal_vector(frames: np.ndarray, prior: np.ndarray, n_time_bins: int, spatial_size: int) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    masked = np.where(prior[None, :, :], frames, np.nan)
    fill = np.nanmedian(masked, axis=(1, 2))
    fill = np.where(np.isfinite(fill), fill, 0.0)
    pieces = []
    for idxs in np.array_split(np.arange(frames.shape[0]), n_time_bins):
        bin_fill = float(np.nanmean(fill[idxs])) if len(idxs) else 0.0
        stack = np.where(np.isfinite(masked[idxs]), masked[idxs], bin_fill)
        img = stack.mean(axis=0)
        img = robust_scale_trace(img.ravel()).reshape(img.shape)
        pieces.append(average_pool(img, spatial_size).ravel())
    return np.concatenate(pieces).astype(np.float32)


def load_manifest(path: Path, cohort: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["embedding_cohort"] = cohort
    df["manifest_path"] = str(path)
    return df


def read_frames(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path)
    key = "frames_norm" if "frames_norm" in z else "frames"
    frames = np.asarray(z[key], dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(f"Expected 3D frames in {npz_path}, got {frames.shape}")
    return frames


def extract_one(row: pd.Series, n_time_bins: int, spatial_size: int) -> Tuple[Dict[str, Any], np.ndarray]:
    npz_path = Path(str(row["npz_path"]))
    frames = read_frames(npz_path)
    prior = build_prior_mask(frames)
    frame_df, mask_summary = trace_masks(frames)
    prior_area = float(prior.sum())
    masked = np.where(prior[None, :, :], frames, np.nan)
    masked_mean = np.nanmean(masked, axis=(1, 2))
    masked_std = np.nanstd(masked, axis=(1, 2))
    outside = np.where(~prior[None, :, :], frames, np.nan)
    outside_mean = np.nanmean(outside, axis=(1, 2))
    contrast = masked_mean - outside_mean
    grad_energy = []
    for frame in masked:
        fill = float(np.nanmedian(frame)) if np.isfinite(frame).any() else 0.0
        ff = np.where(np.isfinite(frame), frame, fill)
        gy, gx = np.gradient(ff)
        grad_energy.append(float(np.nanmean(np.sqrt(gx * gx + gy * gy)[prior])))
    grad_energy = np.asarray(grad_energy, dtype=float)
    vec = spatial_temporal_vector(frames, prior, n_time_bins, spatial_size)
    label_text = str(row.get("validation_label", "")).lower()
    inferred_role = row.get("cohort_role", "")
    if pd.isna(inferred_role) or not str(inferred_role).strip():
        if "control" in label_text:
            inferred_role = "control"
        elif "event" in label_text:
            inferred_role = "event"
    features: Dict[str, Any] = {
        "embedding_row_id": f"{row.get('embedding_cohort', 'cohort')}::{row.get('roi_id', npz_path.stem)}",
        "roi_id": row.get("roi_id", npz_path.stem),
        "embedding_cohort": row.get("embedding_cohort", ""),
        "cycleNo": finite(row.get("cycleNo")),
        "source_stem": row.get("source_stem", ""),
        "cohort_role": inferred_role,
        "selection_subrole": row.get("selection_subrole", ""),
        "validation_label": row.get("validation_label", ""),
        "future_any_drop_within_8cycles": finite(row.get("future_any_drop_within_8cycles")),
        "future_any_drop_within_16cycles": finite(row.get("future_any_drop_within_16cycles")),
        "any_abrupt_drop": finite(row.get("any_abrupt_drop")),
        "npz_path": str(npz_path),
        "n_frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "particle_prior_area_fraction": prior_area / float(frames.shape[1] * frames.shape[2]),
    }
    features.update({f"mask_{k}": v for k, v in mask_summary.items()})
    features.update(trace_stats("particle_mean", masked_mean))
    features.update(trace_stats("particle_std", masked_std))
    features.update(trace_stats("particle_vs_context_mean", contrast))
    features.update(trace_stats("particle_gradient", grad_energy))
    return features, vec


def add_embedding_components(feature_df: pd.DataFrame, vectors: np.ndarray, n_components: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X = np.asarray(vectors, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)
    n_comp = int(min(n_components, X.shape[0] - 1, X.shape[1]))
    if n_comp < 1:
        return feature_df, {"n_embedding_components": 0, "explained_variance_ratio": []}
    pca = PCA(n_components=n_comp, random_state=seed)
    emb = pca.fit_transform(X)
    out = feature_df.copy()
    for i in range(n_comp):
        out[f"video_embed_pc{i+1:02d}"] = emb[:, i]
    return out, {
        "n_embedding_components": n_comp,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def grouped_oof(df: pd.DataFrame, features: List[str], target: str, group_col: str, seed: int) -> Dict[str, Any]:
    y = numeric(df, target).to_numpy(float)
    groups = df[group_col].to_numpy()
    X = df[features]
    pred = np.full(len(df), np.nan, dtype=float)
    for group in pd.unique(groups):
        test = groups == group
        train = ~test
        ok_train = train & np.isin(y, [0, 1])
        ok_test = test & np.isin(y, [0, 1])
        if len(np.unique(y[ok_train])) < 2 or not ok_test.any():
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        model.fit(X.loc[ok_train], y[ok_train].astype(int))
        pred[ok_test] = model.predict_proba(X.loc[ok_test])[:, 1]
    mask = np.isfinite(pred) & np.isin(y, [0, 1])
    y_eval = y[mask].astype(int)
    if len(np.unique(y_eval)) == 2:
        auc = float(roc_auc_score(y_eval, pred[mask]))
        ap = float(average_precision_score(y_eval, pred[mask]))
    else:
        auc = ap = np.nan
    return {
        "target": target,
        "group_col": group_col,
        "feature_set": "masked_video_embedding_trace",
        "n_features": int(len(features)),
        "n_scored": int(mask.sum()),
        "n_positive_scored": int((y_eval == 1).sum()) if len(y_eval) else 0,
        "n_negative_scored": int((y_eval == 0).sum()) if len(y_eval) else 0,
        "pooled_oof_roc_auc": finite(auc),
        "pooled_oof_average_precision": finite(ap),
    }


def label_permutation_null(df: pd.DataFrame, features: List[str], target: str, group_col: str, n_perm: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed = grouped_oof(df, features, target, group_col, seed)
    y = numeric(df, target).to_numpy(float)
    groups = df[group_col].to_numpy()
    valid_groups = pd.DataFrame({"group": groups, "y": y}).dropna()
    group_label = valid_groups.groupby("group")["y"].first()
    aucs = []
    for i in range(n_perm):
        shuffled_group_label = pd.Series(rng.permutation(group_label.to_numpy()), index=group_label.index)
        y_perm = np.array([shuffled_group_label.get(g, np.nan) for g in groups], dtype=float)
        tmp = df.copy()
        tmp[target] = y_perm
        metric = grouped_oof(tmp, features, target, group_col, seed + i + 1)
        auc = metric.get("pooled_oof_roc_auc")
        if auc is not None and np.isfinite(float(auc)):
            aucs.append(float(auc))
    obs_auc = observed.get("pooled_oof_roc_auc")
    if aucs and obs_auc is not None and np.isfinite(float(obs_auc)):
        p = float((np.sum(np.asarray(aucs) >= float(obs_auc)) + 1) / (len(aucs) + 1))
    else:
        p = np.nan
    return {
        "target": target,
        "group_col": group_col,
        "n_permutation": int(n_perm),
        "n_valid_permutation": int(len(aucs)),
        "observed_auc": observed.get("pooled_oof_roc_auc"),
        "observed_average_precision": observed.get("pooled_oof_average_precision"),
        "null_auc_mean": finite(np.nanmean(aucs)) if aucs else None,
        "null_auc_p95": finite(np.nanquantile(aucs, 0.95)) if aucs else None,
        "empirical_p_ge_observed": finite(p),
    }


def binary_feature_tests(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    rows = []
    y = numeric(df, target)
    for feature in features:
        x = numeric(df, feature)
        mask = y.isin([0, 1]) & np.isfinite(x)
        pos = x[mask & (y == 1)].to_numpy(float)
        neg = x[mask & (y == 0)].to_numpy(float)
        if len(pos) and len(neg):
            try:
                _, p = mannwhitneyu(pos, neg, alternative="two-sided")
            except Exception:
                p = np.nan
            auc = float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
            diff = float(np.nanmedian(pos) - np.nanmedian(neg))
        else:
            p = auc = diff = np.nan
        rows.append({
            "target": target,
            "feature": feature,
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "median_positive_minus_negative": finite(diff),
            "oriented_auc": finite(abs(auc - 0.5) + 0.5) if np.isfinite(auc) else None,
            "mannwhitney_p": finite(p),
        })
    return pd.DataFrame(rows).sort_values(["mannwhitney_p", "oriented_auc"], ascending=[True, False], na_position="last")


def cluster_summary(df: pd.DataFrame, features: List[str], n_clusters: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df[features].copy()
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Xs = pipe.fit_transform(X)
    k = int(min(n_clusters, max(2, len(df) // 8)))
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(Xs)
    out = df[["embedding_row_id", "roi_id", "embedding_cohort", "cycleNo", "cohort_role", "future_any_drop_within_8cycles"]].copy()
    out["video_embedding_cluster"] = labels
    out["cluster_distance"] = np.linalg.norm(Xs - km.cluster_centers_[labels], axis=1)
    summaries = []
    for label, grp in out.groupby("video_embedding_cluster"):
        summaries.append({
            "video_embedding_cluster": int(label),
            "n_roi": int(len(grp)),
            "cohort_counts": grp["embedding_cohort"].value_counts(dropna=False).to_dict(),
            "future8_positive_fraction": finite(pd.to_numeric(grp["future_any_drop_within_8cycles"], errors="coerce").mean()),
            "median_cycleNo": finite(pd.to_numeric(grp["cycleNo"], errors="coerce").median()),
            "prototype_roi": grp.sort_values("cluster_distance").iloc[0]["embedding_row_id"],
        })
    return out, pd.DataFrame(summaries).sort_values("video_embedding_cluster")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--n-time-bins", type=int, default=8)
    parser.add_argument("--spatial-size", type=int, default=12)
    parser.add_argument("--n-permutation", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohorts = {
        "selected_event_control": derived / "multi_cycle_roi_sequences" / "selected_roi_sequence_manifest.csv",
        "transfer_ranked": derived / "transfer_ranked_roi_sequences" / "selected_roi_sequence_manifest.csv",
        "balanced_future": derived / "balanced_future_roi_sequences" / "selected_roi_sequence_manifest.csv",
    }
    manifests = []
    missing = []
    for cohort, path in cohorts.items():
        if path.exists():
            manifests.append(load_manifest(path, cohort))
        else:
            missing.append(str(path))
    if not manifests:
        raise FileNotFoundError("No ROI sequence manifests found")
    manifest = pd.concat(manifests, ignore_index=True, sort=False)

    feature_rows: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []
    failures = []
    for _, row in manifest.iterrows():
        try:
            feat, vec = extract_one(row, args.n_time_bins, args.spatial_size)
        except Exception as exc:
            failures.append({"roi_id": row.get("roi_id"), "embedding_cohort": row.get("embedding_cohort"), "error": str(exc)})
            continue
        feature_rows.append(feat)
        vectors.append(vec)
    if not feature_rows:
        raise RuntimeError("No ROI embeddings could be extracted")
    feature_df = pd.DataFrame(feature_rows)
    feature_df, pca_summary = add_embedding_components(feature_df, np.vstack(vectors), args.n_components, args.seed)

    embedding_cols = [c for c in feature_df.columns if c.startswith("video_embed_pc")]
    trace_cols = [
        c for c in feature_df.columns
        if c.startswith("particle_") or c.startswith("mask_")
    ]
    feature_cols = embedding_cols + [c for c in trace_cols if pd.api.types.is_numeric_dtype(feature_df[c])]

    metrics = []
    balanced = feature_df[feature_df["embedding_cohort"] == "balanced_future"].copy()
    if numeric(balanced, "future_any_drop_within_8cycles").isin([0, 1]).sum() >= 8:
        metrics.append(grouped_oof(balanced, feature_cols, "future_any_drop_within_8cycles", "cycleNo", args.seed))
        perm_null = label_permutation_null(balanced, feature_cols, "future_any_drop_within_8cycles", "cycleNo", args.n_permutation, args.seed)
    else:
        perm_null = {}
    selected = feature_df[feature_df["embedding_cohort"] == "selected_event_control"].copy()
    if "cohort_role" in selected.columns:
        selected["event_vs_control"] = selected["cohort_role"].astype(str).str.lower().map({"event": 1, "control": 0})
        if numeric(selected, "event_vs_control").isin([0, 1]).sum() >= 8:
            metrics.append(grouped_oof(selected, feature_cols, "event_vs_control", "cycleNo", args.seed))
    metrics_df = pd.DataFrame(metrics)

    tests = []
    if not balanced.empty:
        tests.append(binary_feature_tests(balanced, feature_cols, "future_any_drop_within_8cycles"))
    if "event_vs_control" in selected.columns:
        tests.append(binary_feature_tests(selected, feature_cols, "event_vs_control"))
    tests_df = pd.concat(tests, ignore_index=True, sort=False) if tests else pd.DataFrame()

    cluster_assign, cluster_summ = cluster_summary(feature_df, feature_cols, 8, args.seed)

    corrs = []
    for x in feature_cols:
        for y in ["cycleNo", "future_any_drop_within_8cycles", "mask_mask_instability_score", "particle_mean_last_minus_first"]:
            if x == y:
                continue
            xs = numeric(feature_df, x)
            ys = numeric(feature_df, y)
            mask = np.isfinite(xs) & np.isfinite(ys)
            if mask.sum() >= 6 and xs[mask].nunique() > 1 and ys[mask].nunique() > 1:
                rho, p = spearmanr(xs[mask], ys[mask])
                corrs.append({"x": x, "y": y, "n": int(mask.sum()), "spearman_rho": finite(rho), "p_value": finite(p), "abs_rho": finite(abs(rho))})
    corr_df = pd.DataFrame(corrs).sort_values(["p_value", "abs_rho"], ascending=[True, False], na_position="last") if corrs else pd.DataFrame()

    feature_df.to_csv(out_dir / "masked_video_embedding_features.csv", index=False)
    metrics_df.to_csv(out_dir / "masked_video_embedding_target_metrics.csv", index=False)
    tests_df.to_csv(out_dir / "masked_video_embedding_feature_tests.csv", index=False)
    cluster_assign.to_csv(out_dir / "masked_video_embedding_cluster_assignments.csv", index=False)
    cluster_summ.to_csv(out_dir / "masked_video_embedding_cluster_summary.csv", index=False)
    corr_df.to_csv(out_dir / "masked_video_embedding_correlations.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "masked_video_embedding_failures.csv", index=False)

    summary = {
        "n_manifest_rows": int(len(manifest)),
        "n_embedding_rows": int(len(feature_df)),
        "cohort_counts": feature_df["embedding_cohort"].value_counts(dropna=False).to_dict(),
        "missing_manifests": missing,
        "n_failures": int(len(failures)),
        "particle_region_method": "history-derived central particle prior plus stability/fallback audit from tier4_particle_mask_stability_audit; embeddings aggregate only prior-masked pixels",
        "pca_summary": pca_summary,
        "target_metrics": metrics_df.to_dict("records"),
        "balanced_future_label_permutation_null": perm_null,
        "top_feature_tests": tests_df.head(16).to_dict("records") if not tests_df.empty else [],
        "cluster_summary": cluster_summ.to_dict("records"),
        "top_correlations": corr_df.head(16).to_dict("records") if not corr_df.empty else [],
        "guardrail": "Masked video embeddings are unsupervised ROI-video descriptors for review prioritization and physics hypothesis generation. Labels remain weak/manual-QC-pending; diffusion claims remain apparent optical proxies.",
    }
    (out_dir / "masked_video_embedding_audit_summary.json").write_text(json.dumps(clean_value(summary), indent=2))
    (out_dir / "README.md").write_text(
        "# Masked Video Embedding Audit\n\n"
        "Particle-prior masked temporal/spatial-temporal embeddings across selected event/control, transfer-ranked, and balanced future ROI tensors. "
        "Outputs are compact CSV/JSON summaries; heavy NPZ tensors remain in the source derived folders.\n"
    )
    print(json.dumps(clean_value({
        "n_embedding_rows": summary["n_embedding_rows"],
        "cohort_counts": summary["cohort_counts"],
        "target_metrics": summary["target_metrics"],
        "permutation_null": summary["balanced_future_label_permutation_null"],
    }), indent=2))


if __name__ == "__main__":
    main()
