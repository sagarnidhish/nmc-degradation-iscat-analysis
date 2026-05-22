#!/usr/bin/env python3
"""Audit event-relative video dynamics in pre-event ROI sequences.

This consumes the source-balanced pre-event ROI sequence export and asks
whether simple particle-only video descriptors separate near/mid/far pre-event
windows from post-event and no-near-event controls. The audit is intentionally
transparent: it uses frame-difference, intensity, heterogeneity, and bright
fraction descriptors, plus grouped leave-source/leave-cycle logistic readouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "intensity": [
        "roi_norm_mean_delta",
        "roi_norm_mean_slope",
        "roi_norm_mean_abs_step_mean",
        "roi_norm_mean_abs_step_p95",
    ],
    "dynamics": [
        "frame_diff_mse_mean",
        "frame_diff_mse_p95",
        "frame_diff_mse_slope",
        "frame_diff_abs_mean",
        "temporal_gradient_signed_mean",
    ],
    "spatial": [
        "spatial_std_mean",
        "spatial_std_slope",
        "bright_fraction_060_delta",
        "bright_fraction_075_delta",
        "dark_fraction_025_delta",
    ],
}
FEATURE_SETS["all_video"] = sorted({c for cols in FEATURE_SETS.values() for c in cols})


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


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or not np.isfinite(y).any():
        return np.nan
    x = np.linspace(0.0, 1.0, len(y))
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def load_npz_features(path: Path) -> Dict[str, float]:
    with np.load(path) as z:
        frames = np.asarray(z["frames_norm"], dtype=np.float32)
        roi_mean = np.asarray(z["roi_norm_mean"], dtype=float)
    diffs = np.diff(frames, axis=0)
    diff_mse = np.mean(diffs * diffs, axis=(1, 2))
    diff_abs = np.mean(np.abs(diffs), axis=(1, 2))
    spatial_std = np.std(frames, axis=(1, 2))
    bright060 = np.mean(frames > 0.60, axis=(1, 2))
    bright075 = np.mean(frames > 0.75, axis=(1, 2))
    dark025 = np.mean(frames < 0.25, axis=(1, 2))
    roi_steps = np.diff(roi_mean)
    return {
        "roi_norm_mean_first": float(roi_mean[0]),
        "roi_norm_mean_last": float(roi_mean[-1]),
        "roi_norm_mean_delta": float(roi_mean[-1] - roi_mean[0]),
        "roi_norm_mean_slope": slope(roi_mean),
        "roi_norm_mean_abs_step_mean": float(np.mean(np.abs(roi_steps))) if len(roi_steps) else np.nan,
        "roi_norm_mean_abs_step_p95": float(np.percentile(np.abs(roi_steps), 95)) if len(roi_steps) else np.nan,
        "frame_diff_mse_mean": float(np.mean(diff_mse)) if len(diff_mse) else np.nan,
        "frame_diff_mse_p95": float(np.percentile(diff_mse, 95)) if len(diff_mse) else np.nan,
        "frame_diff_mse_slope": slope(diff_mse),
        "frame_diff_abs_mean": float(np.mean(diff_abs)) if len(diff_abs) else np.nan,
        "temporal_gradient_signed_mean": float(np.mean(diffs)) if len(diffs) else np.nan,
        "spatial_std_mean": float(np.mean(spatial_std)),
        "spatial_std_slope": slope(spatial_std),
        "bright_fraction_060_delta": float(bright060[-1] - bright060[0]),
        "bright_fraction_075_delta": float(bright075[-1] - bright075[0]),
        "dark_fraction_025_delta": float(dark025[-1] - dark025[0]),
    }


def append_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = out["event_relative_bin"].astype(str)
    out["target_near_pre_vs_rest"] = bins.eq("near_pre_event_1_8").astype(int)
    out["target_pre16_clean_vs_post_control"] = np.nan
    pre16_clean = bins.isin(["near_pre_event_1_8", "mid_pre_event_9_16"]) & pd.to_numeric(out.get("past_event_within_16cycles", 0), errors="coerce").fillna(0).eq(0)
    post_control = bins.isin(["post_event_1_16", "no_near_event_control"])
    out.loc[pre16_clean, "target_pre16_clean_vs_post_control"] = 1
    out.loc[post_control, "target_pre16_clean_vs_post_control"] = 0
    out["target_any_pre_vs_post_control"] = np.nan
    out.loc[bins.isin(["near_pre_event_1_8", "mid_pre_event_9_16", "far_pre_event_17_32"]), "target_any_pre_vs_post_control"] = 1
    out.loc[post_control, "target_any_pre_vs_post_control"] = 0
    return out


def source_transform(values: pd.Series, sources: pd.Series, transform: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    src = sources.astype(str)
    means = x.groupby(src).transform("mean")
    if transform == "raw":
        return x
    if transform == "source_residual":
        return x - means
    if transform == "within_source_rank":
        return x.groupby(src).rank(pct=True) - 0.5
    raise ValueError(transform)


def oriented_metrics(y: pd.Series, x: pd.Series) -> Dict[str, Any]:
    valid = y.isin([0, 1]) & x.notna()
    yy = y[valid].astype(int)
    xx = x[valid]
    out: Dict[str, Any] = {
        "n": int(valid.sum()),
        "n_positive": int(yy.sum()) if len(yy) else 0,
        "direction": "NA",
        "oriented_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "median_positive_minus_negative": np.nan,
        "mannwhitney_p": np.nan,
    }
    if len(yy) >= 8 and yy.nunique() == 2 and xx.nunique() > 1:
        direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
        score = xx if direction == "higher_in_positive" else -xx
        out["direction"] = direction
        out["oriented_auc"] = float(roc_auc_score(yy, score))
        out["average_precision"] = float(average_precision_score(yy, score))
        rho, sp = spearmanr(yy, score)
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(sp)
        out["median_positive_minus_negative"] = float(xx[yy == 1].median() - xx[yy == 0].median())
        try:
            out["mannwhitney_p"] = float(mannwhitneyu(score[yy == 1], score[yy == 0], alternative="two-sided").pvalue)
        except ValueError:
            pass
    return out


def scalar_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    targets = ["target_near_pre_vs_rest", "target_pre16_clean_vs_post_control", "target_any_pre_vs_post_control"]
    for target in targets:
        y = pd.to_numeric(df[target], errors="coerce")
        for feature in FEATURE_SETS["all_video"]:
            raw = pd.to_numeric(df[feature], errors="coerce")
            for transform in ["raw", "source_residual", "within_source_rank"]:
                met = oriented_metrics(y, source_transform(raw, sources, transform))
                rows.append({"target": target, "feature": feature, "transform": transform, **met})
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "oriented_auc", "average_precision"], ascending=[True, False, False])


def model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3, solver="liblinear", random_state=seed),
    )


def grouped_predictions(df: pd.DataFrame, cols: List[str], target: str, group_col: str, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1]) & df[group_col].notna()
    for group in sorted(df.loc[valid, group_col].dropna().unique()):
        test = valid & df[group_col].eq(group)
        train = valid & ~test
        meta_cols = ["roi_id", "cycleNo", "source_stem", "event_relative_bin", target]
        meta = df.loc[test, meta_cols].rename(columns={target: "observed"}).copy()
        meta[group_col] = group
        if train.sum() < 16 or y[train].nunique() < 2:
            meta["predicted_probability"] = np.nan
            meta["status"] = "skipped_train_size_or_class"
        else:
            clf = model(seed)
            clf.fit(df.loc[train, cols], y[train].astype(int))
            meta["predicted_probability"] = clf.predict_proba(df.loc[test, cols])[:, 1]
            meta["status"] = "ok"
        rows.extend(meta.to_dict("records"))
    out = pd.DataFrame(rows)
    out["target"] = target
    out["group_col"] = group_col
    return out


def prediction_metrics(pred: pd.DataFrame, feature_set: str) -> Dict[str, Any]:
    tmp = pred.dropna(subset=["observed", "predicted_probability"])
    y = pd.to_numeric(tmp["observed"], errors="coerce").astype(int)
    p = pd.to_numeric(tmp["predicted_probability"], errors="coerce")
    group_col = pred["group_col"].iloc[0] if len(pred) else ""
    row: Dict[str, Any] = {
        "feature_set": feature_set,
        "target": pred["target"].iloc[0] if len(pred) else "",
        "group_col": group_col,
        "n_eval": int(len(tmp)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "n_groups": int(tmp[group_col].nunique()) if len(tmp) and group_col in tmp else 0,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if len(tmp) >= 8 and y.nunique() == 2 and p.nunique() > 1:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(average_precision_score(y, p))
        rho, sp = spearmanr(y, p)
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_sequence_audit")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    seq_dir = Path(args.sequence_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(seq_dir / "selected_roi_sequence_manifest.csv")
    feature_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        try:
            rec = row.to_dict()
            rec.update(load_npz_features(Path(str(row["npz_path"]))))
            feature_rows.append(rec)
        except Exception as exc:
            failures.append({"roi_id": row.get("roi_id"), "npz_path": row.get("npz_path"), "error": str(exc)})
    features = append_targets(pd.DataFrame(feature_rows))
    tests = scalar_tests(features)

    all_preds: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, Any]] = []
    for target in ["target_near_pre_vs_rest", "target_pre16_clean_vs_post_control", "target_any_pre_vs_post_control"]:
        for group_col in ["source_stem", "cycleNo"]:
            for feature_set, cols in FEATURE_SETS.items():
                pred = grouped_predictions(features, cols, target, group_col, args.seed)
                pred["feature_set"] = feature_set
                all_preds.append(pred)
                metric_rows.append(prediction_metrics(pred, feature_set))
    metrics = pd.DataFrame(metric_rows).sort_values(["target", "roc_auc", "average_precision"], ascending=[True, False, False])
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    bin_summary = features.groupby("event_relative_bin", dropna=False).agg(
        n_roi=("roi_id", "count"),
        n_cycles=("cycleNo", "nunique"),
        n_sources=("source_stem", "nunique"),
        mean_roi_norm_delta=("roi_norm_mean_delta", "mean"),
        mean_frame_diff_mse=("frame_diff_mse_mean", "mean"),
        mean_spatial_std=("spatial_std_mean", "mean"),
        median_cycles_to_next_event=("cycles_to_next_event", "median"),
    ).reset_index()
    source_bin_summary = features.groupby(["source_stem", "event_relative_bin"], dropna=False).agg(
        n_roi=("roi_id", "count"),
        n_cycles=("cycleNo", "nunique"),
        mean_roi_norm_delta=("roi_norm_mean_delta", "mean"),
        mean_frame_diff_mse=("frame_diff_mse_mean", "mean"),
    ).reset_index()

    features.to_csv(out / "source_balanced_pre_event_sequence_features.csv", index=False)
    tests.to_csv(out / "source_balanced_pre_event_sequence_scalar_tests.csv", index=False)
    metrics.to_csv(out / "source_balanced_pre_event_sequence_model_metrics.csv", index=False)
    predictions.to_csv(out / "source_balanced_pre_event_sequence_model_predictions.csv", index=False)
    bin_summary.to_csv(out / "source_balanced_pre_event_sequence_bin_summary.csv", index=False)
    source_bin_summary.to_csv(out / "source_balanced_pre_event_sequence_source_bin_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "source_balanced_pre_event_sequence_feature_failures.csv", index=False)

    top_scalar = tests.head(20).to_dict("records")
    top_metrics = metrics.head(20).to_dict("records")
    summary = {
        "sequence_dir": args.sequence_dir,
        "n_manifest_rows": int(len(manifest)),
        "n_feature_rows": int(len(features)),
        "n_failures": int(len(failures)),
        "n_cycles": int(features["cycleNo"].nunique()) if not features.empty else 0,
        "n_sources": int(features["source_stem"].nunique()) if not features.empty else 0,
        "bin_counts": clean_json(features["event_relative_bin"].value_counts().to_dict()) if not features.empty else {},
        "target_positive_counts": clean_json({c: int(pd.to_numeric(features[c], errors="coerce").fillna(0).sum()) for c in ["target_near_pre_vs_rest", "target_pre16_clean_vs_post_control", "target_any_pre_vs_post_control"]}) if not features.empty else {},
        "top_scalar_tests": clean_json(top_scalar),
        "top_model_metrics": clean_json(top_metrics),
        "bin_summary": clean_json(bin_summary.to_dict("records")),
        "outputs": {
            "features": str(out / "source_balanced_pre_event_sequence_features.csv"),
            "scalar_tests": str(out / "source_balanced_pre_event_sequence_scalar_tests.csv"),
            "model_metrics": str(out / "source_balanced_pre_event_sequence_model_metrics.csv"),
            "model_predictions": str(out / "source_balanced_pre_event_sequence_model_predictions.csv"),
            "bin_summary": str(out / "source_balanced_pre_event_sequence_bin_summary.csv"),
            "source_bin_summary": str(out / "source_balanced_pre_event_sequence_source_bin_summary.csv"),
            "summary": str(out / "source_balanced_pre_event_sequence_audit_summary.json"),
        },
        "guardrail": "Event-relative sequence features are computed from automatic fixed particle crops and labels are derived from abrupt-event cycle proximity. Positive readouts indicate pre/post/control optical-dynamics separation for follow-up QC and modeling, not validated precursors, particle identities, phase boundaries, diffusion coefficients, or causal degradation mechanisms.",
    }
    (out / "source_balanced_pre_event_sequence_audit_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    lines = [
        "# Source-Balanced Pre-Event Sequence Audit",
        "",
        f"- ROI sequences: {summary['n_feature_rows']} / manifest rows {summary['n_manifest_rows']}",
        f"- Cycles/sources: {summary['n_cycles']} / {summary['n_sources']}",
        f"- Bin counts: {summary['bin_counts']}",
        f"- Target positives: {summary['target_positive_counts']}",
        "",
        "## Top Scalar Tests",
        "",
    ]
    for row in top_scalar[:8]:
        lines.append(
            f"- {row.get('target')} {row.get('feature')} {row.get('transform')}: AUC={row.get('oriented_auc')}, AP={row.get('average_precision')}, p={row.get('mannwhitney_p')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
