#!/usr/bin/env python3
"""Classify NMC event ROIs versus matched controls from particle-only features.

Uses the event/control ROI metrics table built from selected particle-region
tensors. The evaluation is deliberately small and conservative: train on one
event/control cycle pair and test on the other pair. This tests whether
physics-facing ROI descriptors generalize across synchronized events rather
than only separating ROIs within one source movie.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "roi_norm_mean_delta_last_minus_first",
    "roi_mean_delta_last_minus_first",
    "stage_drift_xy_sampled",
    "temporal_diff_energy",
    "cumulative_abs_norm_change",
    "first_last_corr",
    "mean_norm_intensity",
    "std_norm_intensity",
]


def infer_pair_label(row: pd.Series) -> int:
    cycle = float(row["cycleNo"])
    if cycle in (86.0, 88.0):
        return 86
    if cycle in (116.0, 118.0):
        return 116
    return int(cycle)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_control_roi_comparison/event_control_roi_metrics.csv")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_control_roi_classifier")
    parser.add_argument("--c", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    df["label_event"] = (df["cohort"] == "event").astype(int)
    df["pair_label"] = df.apply(infer_pair_label, axis=1)
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(df[FEATURES].median(numeric_only=True)).fillna(0.0)
    y = df["label_event"].to_numpy(dtype=int)

    rows: List[Dict[str, object]] = []
    pred_tables = []
    coefs = []
    for test_pair in sorted(df["pair_label"].unique()):
        train_mask = df["pair_label"] != test_pair
        test_mask = df["pair_label"] == test_pair
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=args.c, penalty="l2", solver="liblinear", class_weight="balanced", random_state=41)),
        ])
        model.fit(x.loc[train_mask], y[train_mask])
        prob = model.predict_proba(x.loc[test_mask])[:, 1]
        pred = (prob >= 0.5).astype(int)
        yt = y[test_mask]
        rows.append({
            "test_pair": int(test_pair),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "accuracy": float(accuracy_score(yt, pred)),
            "roc_auc": safe_auc(yt, prob),
            "average_precision": safe_ap(yt, prob),
            "mean_event_probability_true_event": float(np.mean(prob[yt == 1])) if np.any(yt == 1) else np.nan,
            "mean_event_probability_control": float(np.mean(prob[yt == 0])) if np.any(yt == 0) else np.nan,
        })
        pt = df.loc[test_mask, ["roi_id", "cohort", "cycleNo", "pair_label"]].copy()
        pt["event_probability"] = prob
        pt["predicted_event"] = pred
        pred_tables.append(pt)
        clf = model.named_steps["clf"]
        for feature, coef in zip(FEATURES, clf.coef_[0]):
            coefs.append({"test_pair": int(test_pair), "feature": feature, "coefficient": float(coef)})

    result = pd.DataFrame(rows)
    preds = pd.concat(pred_tables, ignore_index=True)
    coef_df = pd.DataFrame(coefs)
    mean_coef = coef_df.groupby("feature", as_index=False)["coefficient"].mean()
    mean_coef["abs_mean_coefficient"] = mean_coef["coefficient"].abs()
    mean_coef = mean_coef.sort_values("abs_mean_coefficient", ascending=False)

    os.makedirs(args.out_dir, exist_ok=True)
    result_path = os.path.join(args.out_dir, "event_control_roi_classifier_pair_holdout.csv")
    pred_path = os.path.join(args.out_dir, "event_control_roi_classifier_predictions.csv")
    coef_path = os.path.join(args.out_dir, "event_control_roi_classifier_coefficients.csv")
    result.to_csv(result_path, index=False)
    preds.to_csv(pred_path, index=False)
    coef_df.to_csv(coef_path, index=False)
    mean_coef_path = os.path.join(args.out_dir, "event_control_roi_classifier_mean_coefficients.csv")
    mean_coef.to_csv(mean_coef_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for pair, grp in preds.groupby("pair_label"):
        vals = [grp.loc[grp["cohort"] == "event", "event_probability"], grp.loc[grp["cohort"] == "control", "event_probability"]]
        axes[0].boxplot(vals, positions=[pair - 3, pair + 3], widths=4)
    axes[0].set_xticks(sorted(preds["pair_label"].unique()))
    axes[0].set_xlabel("held-out event/control pair")
    axes[0].set_ylabel("predicted event probability")
    axes[0].set_title("pair-holdout classifier")
    axes[1].barh(mean_coef["feature"], mean_coef["coefficient"])
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_title("mean logistic coefficients")
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, "event_control_roi_classifier.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "metrics": args.metrics,
        "n_rows": int(len(df)),
        "n_event": int((df["label_event"] == 1).sum()),
        "n_control": int((df["label_event"] == 0).sum()),
        "features": FEATURES,
        "pair_holdout": result.to_dict(orient="records"),
        "mean_accuracy": float(result["accuracy"].mean()),
        "mean_roc_auc": float(result["roc_auc"].mean()),
        "mean_average_precision": float(result["average_precision"].mean()),
        "top_coefficients": mean_coef.head(6).to_dict(orient="records"),
        "guardrail": "Small pair-holdout classifier on automatic ROIs; use as feature ranking, not definitive validation.",
        "plot_path": plot_path,
    }
    summary_path = os.path.join(args.out_dir, "event_control_roi_classifier_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Event-Control ROI Classifier\n\n")
        f.write("Pair-holdout logistic classifier separating selected event ROIs from matched non-event control ROIs using particle-only descriptors.\n")
    for path in [result_path, pred_path, coef_path, mean_coef_path, summary_path, plot_path]:
        print(f"Saved: {path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
