#!/usr/bin/env python3
"""Compare selected event ROI sequences against matched non-event controls."""

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_manifest(path: str, cohort: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cohort"] = cohort
    return df


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    av = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    bv = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(av) < 2 or len(bv) < 2:
        return np.nan
    pooled = np.sqrt(((len(av) - 1) * av.var(ddof=1) + (len(bv) - 1) * bv.var(ddof=1)) / (len(av) + len(bv) - 2))
    return float((av.mean() - bv.mean()) / pooled) if pooled > 0 else np.nan


def compare_groups(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        event = df[df["cohort"] == "event"][metric]
        control = df[df["cohort"] == "control"][metric]
        try:
            mw = stats.mannwhitneyu(event.dropna(), control.dropna(), alternative="two-sided")
            p = float(mw.pvalue)
        except Exception:
            p = np.nan
        rows.append({
            "metric": metric,
            "event_mean": float(pd.to_numeric(event, errors="coerce").mean()),
            "control_mean": float(pd.to_numeric(control, errors="coerce").mean()),
            "event_median": float(pd.to_numeric(event, errors="coerce").median()),
            "control_median": float(pd.to_numeric(control, errors="coerce").median()),
            "event_minus_control_mean": float(pd.to_numeric(event, errors="coerce").mean() - pd.to_numeric(control, errors="coerce").mean()),
            "cohens_d_event_vs_control": cohens_d(event, control),
            "mannwhitney_p": p,
        })
    return pd.DataFrame(rows)


def load_npz_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in manifest.iterrows():
        npz = np.load(row["npz_path"])
        frames = npz["frames_norm"].astype(np.float32)
        diff = np.diff(frames, axis=0)
        rows.append({
            "roi_id": row["roi_id"],
            "cohort": row["cohort"],
            "cycleNo": float(row["cycleNo"]),
            "control_for_event_cycle": float(row["control_for_event_cycle"]) if "control_for_event_cycle" in row and pd.notna(row["control_for_event_cycle"]) else np.nan,
            "roi_norm_mean_delta_last_minus_first": float(row["roi_norm_mean_delta_last_minus_first"]),
            "roi_mean_delta_last_minus_first": float(row["roi_mean_delta_last_minus_first"]),
            "stage_drift_xy_sampled": float(row["stage_drift_xy_sampled"]),
            "temporal_diff_energy": float(np.mean(diff ** 2)),
            "cumulative_abs_norm_change": float(np.mean(np.abs(frames[-1] - frames[0]))),
            "first_last_corr": float(np.corrcoef(frames[0].ravel(), frames[-1].ravel())[0, 1]),
            "mean_norm_intensity": float(np.mean(frames)),
            "std_norm_intensity": float(np.std(frames)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/selected_roi_sequences")
    parser.add_argument("--control-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/control_roi_sequences")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_control_roi_comparison")
    args = parser.parse_args()

    event = load_manifest(os.path.join(args.event_dir, "selected_roi_sequence_manifest.csv"), "event")
    control = load_manifest(os.path.join(args.control_dir, "selected_roi_sequence_manifest.csv"), "control")
    manifest = pd.concat([event, control], ignore_index=True)
    metrics_df = load_npz_metrics(manifest)
    metrics = [
        "roi_norm_mean_delta_last_minus_first",
        "roi_mean_delta_last_minus_first",
        "stage_drift_xy_sampled",
        "temporal_diff_energy",
        "cumulative_abs_norm_change",
        "first_last_corr",
        "mean_norm_intensity",
        "std_norm_intensity",
    ]
    comparison = compare_groups(metrics_df, metrics)
    cycle = metrics_df.groupby(["cohort", "cycleNo"], as_index=False)[metrics].mean()
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "event_control_roi_metrics.csv")
    comparison_path = os.path.join(args.out_dir, "event_control_roi_metric_comparison.csv")
    cycle_path = os.path.join(args.out_dir, "event_control_roi_cycle_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    cycle.to_csv(cycle_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["roi_norm_mean_delta_last_minus_first", "temporal_diff_energy", "cumulative_abs_norm_change"]):
        vals = [metrics_df.loc[metrics_df["cohort"] == c, metric].to_numpy(dtype=float) for c in ["event", "control"]]
        ax.boxplot(vals, tick_labels=["event", "control"])
        ax.set_title(metric, fontsize=9)
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, "event_control_roi_comparison.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    summary = {
        "event_dir": args.event_dir,
        "control_dir": args.control_dir,
        "n_event_roi": int((metrics_df["cohort"] == "event").sum()),
        "n_control_roi": int((metrics_df["cohort"] == "control").sum()),
        "top_metric_differences": comparison.reindex(comparison["cohens_d_event_vs_control"].abs().sort_values(ascending=False).index).head(5).to_dict(orient="records"),
        "guardrail": "Controls are automatic reconstructed non-event ROIs from adjacent source segments; this is a matched-control screen, not manual annotation.",
        "plot_path": plot_path,
    }
    summary_path = os.path.join(args.out_dir, "event_control_roi_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Event vs Control ROI Comparison\n\n")
        f.write("Compares selected synchronized-event ROI tensors with matched non-event reconstructed control ROI tensors.\n")
    for path in [metrics_path, comparison_path, cycle_path, summary_path, plot_path]:
        print(f"Saved: {path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
