#!/usr/bin/env python3
"""Compare selected event ROI physics against matched non-event control ROIs.

Uses particle-region tensors exported for synchronized event cycles and nearby
non-event controls. It extracts the same kind of simple temporal/optical
descriptors from both sets, tests event-vs-control differences, and trains a
small transparent classifier as a sanity check for whether the selected event
ROIs are distinguishable from matched particle-like controls.
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def finite_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def slope(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return {"slope": np.nan, "r2": np.nan}
    coef = np.polyfit(x[mask], y[mask], 1)
    pred = coef[0] * x[mask] + coef[1]
    ss_res = float(np.sum((y[mask] - pred) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return {"slope": float(coef[0]), "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan}


def persistence_tail_mse(frames: np.ndarray, train_fraction: float) -> float:
    split = max(2, min(frames.shape[0] - 2, int(frames.shape[0] * train_fraction)))
    if split >= frames.shape[0] - 1:
        return np.nan
    pred = frames[split:-1]
    truth = frames[split + 1:]
    return float(np.mean((pred - truth) ** 2))


def descriptors_for_manifest(manifest_path: str, label: str, control_table: pd.DataFrame, train_fraction: float) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows: List[Dict[str, object]] = []
    control_map = {}
    if not control_table.empty and "control_for_event_cycle" in control_table.columns:
        tmp = control_table.copy()
        tmp["roi_id"] = tmp.apply(lambda r: f"cycle{int(float(r['cycleNo']))}_rank{int(r['front_candidate_rank'])}_obj{int(r['object_candidate_rank'])}", axis=1)
        control_map = dict(zip(tmp["roi_id"], tmp["control_for_event_cycle"]))
    for _, row in manifest.iterrows():
        data = np.load(row["npz_path"])
        frames = data["frames_norm"].astype(np.float32)
        mean_trace = frames.mean(axis=(1, 2))
        diff = np.diff(frames, axis=0)
        lo = np.percentile(frames[: max(3, frames.shape[0] // 8)], 25)
        hi = np.percentile(frames[: max(3, frames.shape[0] // 8)], 75)
        high_fraction = (frames >= hi).mean(axis=(1, 2))
        low_fraction = (frames <= lo).mean(axis=(1, 2))
        mean_fit = slope(mean_trace)
        high_fit = slope(high_fraction)
        low_fit = slope(low_fraction)
        event_cycle = float(row["cycleNo"]) if label == "event" else finite_float(control_map.get(row["roi_id"]), np.nan)
        rows.append({
            "roi_id": row["roi_id"],
            "roi_class": label,
            "is_event_roi": int(label == "event"),
            "cycleNo": float(row["cycleNo"]),
            "event_cycle": event_cycle,
            "source_stem": row["source_stem"],
            "validation_score": finite_float(row.get("validation_score")),
            "roi_mean_first": float(mean_trace[0]),
            "roi_mean_last": float(mean_trace[-1]),
            "roi_mean_delta": float(mean_trace[-1] - mean_trace[0]),
            "roi_mean_slope_per_frame": mean_fit["slope"],
            "roi_mean_slope_r2": mean_fit["r2"],
            "high_fraction_delta": float(high_fraction[-1] - high_fraction[0]),
            "high_fraction_slope_per_frame": high_fit["slope"],
            "high_fraction_slope_r2": high_fit["r2"],
            "low_fraction_delta": float(low_fraction[-1] - low_fraction[0]),
            "low_fraction_slope_per_frame": low_fit["slope"],
            "temporal_diff_energy": float(np.mean(diff ** 2)),
            "cumulative_abs_change": float(np.mean(np.abs(diff))),
            "persistence_tail_mse": persistence_tail_mse(frames, train_fraction),
            "stage_drift_xy_sampled": finite_float(row.get("stage_drift_xy_sampled")),
        })
    return pd.DataFrame(rows)


def feature_tests(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    event = df[df["is_event_roi"] == 1]
    control = df[df["is_event_roi"] == 0]
    for feat in features:
        a = pd.to_numeric(event[feat], errors="coerce").dropna()
        b = pd.to_numeric(control[feat], errors="coerce").dropna()
        if len(a) and len(b):
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        rows.append({
            "feature": feat,
            "event_mean": float(a.mean()) if len(a) else np.nan,
            "control_mean": float(b.mean()) if len(b) else np.nan,
            "event_minus_control": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
            "mannwhitney_u": float(stat) if np.isfinite(stat) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("p_value")


def leave_pair_classifier(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    valid = df.dropna(subset=["event_cycle"]).copy()
    valid["event_cycle"] = valid["event_cycle"].astype(float)
    for holdout in sorted(valid["event_cycle"].dropna().unique()):
        train = valid[valid["event_cycle"] != holdout]
        test = valid[valid["event_cycle"] == holdout]
        if train["is_event_roi"].nunique() < 2 or test.empty:
            continue
        x_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(train[features].median(numeric_only=True)).fillna(0)
        x_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(train[features].median(numeric_only=True)).fillna(0)
        scaler = StandardScaler().fit(x_train)
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=29)
        clf.fit(scaler.transform(x_train), train["is_event_roi"])
        prob = clf.predict_proba(scaler.transform(x_test))[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(test["is_event_roi"], prob) if test["is_event_roi"].nunique() == 2 else np.nan
        rows.append({
            "holdout_event_cycle": float(holdout),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "accuracy": float(accuracy_score(test["is_event_roi"], pred)),
            "roc_auc": float(auc) if np.isfinite(auc) else np.nan,
            "mean_event_probability_for_events": float(np.mean(prob[test["is_event_roi"].to_numpy() == 1])) if np.any(test["is_event_roi"].to_numpy() == 1) else np.nan,
            "mean_event_probability_for_controls": float(np.mean(prob[test["is_event_roi"].to_numpy() == 0])) if np.any(test["is_event_roi"].to_numpy() == 0) else np.nan,
        })
    return pd.DataFrame(rows)


def save_plot(df: pd.DataFrame, tests: pd.DataFrame, out_png: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label, grp in df.groupby("roi_class"):
        axes[0].scatter(grp["roi_mean_delta"], grp["persistence_tail_mse"], label=label, s=60)
    axes[0].set_xlabel("ROI mean delta")
    axes[0].set_ylabel("tail persistence MSE")
    axes[0].legend(fontsize=8)
    axes[0].set_title("event vs control ROI dynamics", fontsize=9)
    for label, grp in df.groupby("roi_class"):
        axes[1].scatter(grp["cumulative_abs_change"], grp["temporal_diff_energy"], label=label, s=60)
    axes[1].set_xlabel("cumulative abs change")
    axes[1].set_ylabel("temporal diff energy")
    axes[1].set_title("temporal activity", fontsize=9)
    top = tests.sort_values("p_value").head(6).copy()
    axes[2].barh(top["feature"], top["event_minus_control"])
    axes[2].axvline(0, color="0.7", lw=1)
    axes[2].set_title("event-control feature shift", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_vs_control_roi_physics")
    parser.add_argument("--train-fraction", type=float, default=0.67)
    args = parser.parse_args()

    derived = args.derived_dir
    control_table = pd.read_csv(os.path.join(derived, "control_roi_selection", "selected_control_rois.csv"))
    event_df = descriptors_for_manifest(os.path.join(derived, "selected_roi_sequences", "selected_roi_sequence_manifest.csv"), "event", control_table, args.train_fraction)
    control_df = descriptors_for_manifest(os.path.join(derived, "control_roi_sequences", "selected_roi_sequence_manifest.csv"), "control", control_table, args.train_fraction)
    df = pd.concat([event_df, control_df], ignore_index=True)
    features = [
        "roi_mean_delta",
        "roi_mean_slope_per_frame",
        "high_fraction_delta",
        "high_fraction_slope_per_frame",
        "low_fraction_delta",
        "low_fraction_slope_per_frame",
        "temporal_diff_energy",
        "cumulative_abs_change",
        "persistence_tail_mse",
        "stage_drift_xy_sampled",
    ]
    tests = feature_tests(df, features)
    classifier = leave_pair_classifier(df, features)
    group_summary = df.groupby(["roi_class", "event_cycle"], dropna=False).agg({
        "roi_id": "count",
        "roi_mean_delta": "mean",
        "temporal_diff_energy": "mean",
        "cumulative_abs_change": "mean",
        "persistence_tail_mse": "mean",
        "stage_drift_xy_sampled": "mean",
    }).reset_index().rename(columns={"roi_id": "n_roi"})

    os.makedirs(args.out_dir, exist_ok=True)
    desc_path = os.path.join(args.out_dir, "event_vs_control_roi_descriptors.csv")
    tests_path = os.path.join(args.out_dir, "event_vs_control_feature_tests.csv")
    clf_path = os.path.join(args.out_dir, "event_vs_control_classifier_leave_pair_out.csv")
    group_path = os.path.join(args.out_dir, "event_vs_control_group_summary.csv")
    plot_path = os.path.join(args.out_dir, "event_vs_control_roi_physics.png")
    df.to_csv(desc_path, index=False)
    tests.to_csv(tests_path, index=False)
    classifier.to_csv(clf_path, index=False)
    group_summary.to_csv(group_path, index=False)
    save_plot(df, tests, plot_path)
    summary = {
        "n_event_roi": int((df["is_event_roi"] == 1).sum()),
        "n_control_roi": int((df["is_event_roi"] == 0).sum()),
        "event_cycles": sorted(float(x) for x in event_df["event_cycle"].dropna().unique()),
        "control_cycles": sorted(float(x) for x in control_df["cycleNo"].dropna().unique()),
        "top_feature_tests": tests.head(8).to_dict(orient="records"),
        "classifier_leave_pair_out": classifier.to_dict(orient="records"),
        "group_summary": group_summary.to_dict(orient="records"),
        "guardrail": "Controls are automatic nearby reconstructed ROIs, not manual matched particles. Treat event-control differences as hypothesis-ranking evidence.",
        "outputs": {
            "descriptors": desc_path,
            "feature_tests": tests_path,
            "classifier": clf_path,
            "group_summary": group_path,
            "plot": plot_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "event_vs_control_roi_physics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Event vs Control ROI Physics\n\n")
        f.write("Compares selected synchronized-event ROI tensors with nearby non-event reconstructed control ROI tensors.\n\n")
        f.write("Controls are automatic matched candidates and should be treated as hypothesis-ranking evidence, not manual labels.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
