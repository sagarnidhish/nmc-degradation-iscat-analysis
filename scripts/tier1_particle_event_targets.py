#!/usr/bin/env python3
"""Build NMC particle abrupt-event labels and simple precursor baselines.

This turns the example particle intensity traces into a cycle-level event target:
"will this particle show an abrupt intensity drop within the next K cycles?" The
features are intentionally transparent and derived only from prior normalized
intensity history, so they can be used as a lightweight degradation forecasting
baseline before heavier Tier 2/3 models finish.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def resolve_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def read_cycle_frames_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "cycleNo" in df.columns:
        return df
    raw = pd.read_csv(path, encoding="utf-8-sig", header=None)
    if raw.shape[1] < 2:
        raise ValueError("cycleFrames.csv must have at least two columns")
    raw = raw.iloc[:, :2].copy()
    raw.columns = ["cycleNo", "n_frames"]
    return raw


def normalize_to_cycle2(series: pd.Series, cycles: pd.Series) -> Tuple[pd.Series, float]:
    baseline = np.nan
    vals = series[cycles == 2].dropna()
    if not vals.empty:
        baseline = float(vals.iloc[0])
    if not np.isfinite(baseline) or baseline == 0:
        vals = series.dropna()
        baseline = float(vals.iloc[0]) if not vals.empty else np.nan
    if not np.isfinite(baseline) or baseline == 0:
        return pd.Series(np.nan, index=series.index), np.nan
    return series.astype(float) / baseline, baseline


def detect_abrupt_drops(norm: pd.Series, sigma_multiplier: float) -> Tuple[pd.Series, float]:
    delta = norm.diff()
    sigma = float(np.nanstd(delta.to_numpy(dtype=float)))
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(False, index=norm.index), np.nan
    threshold = -abs(sigma_multiplier) * sigma
    return (delta < threshold).fillna(False), threshold


def trailing_slope(values: np.ndarray) -> float:
    good = np.isfinite(values)
    if good.sum() < 2:
        return np.nan
    y = values[good]
    x = np.arange(len(values), dtype=float)[good]
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * (y - y.mean())) / denom)


def make_long_table(particles_df: pd.DataFrame, particle_cols: List[str], horizons: List[int], window: int, sigma_multiplier: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cycles = particles_df["cycleNo"].astype(float)
    rows = []
    event_rows = []
    for pcol in particle_cols:
        raw = pd.to_numeric(particles_df[pcol], errors="coerce")
        norm, baseline = normalize_to_cycle2(raw, cycles)
        events, threshold = detect_abrupt_drops(norm, sigma_multiplier)
        event_cycles = cycles[events].astype(float).tolist()
        for cyc in event_cycles:
            event_rows.append({"particle": pcol, "cycleNo": cyc, "event_type": "abrupt_intensity_drop", "threshold_delta_norm": threshold, "baseline_intensity": baseline})
        delta = norm.diff()
        for i, cyc in enumerate(cycles):
            hist = norm.iloc[max(0, i - window + 1) : i + 1].to_numpy(dtype=float)
            dhist = delta.iloc[max(0, i - window + 1) : i + 1].to_numpy(dtype=float)
            row = {
                "particle": pcol,
                "cycleNo": float(cyc),
                "raw_intensity": float(raw.iloc[i]) if np.isfinite(raw.iloc[i]) else np.nan,
                "norm_intensity": float(norm.iloc[i]) if np.isfinite(norm.iloc[i]) else np.nan,
                "delta_norm": float(delta.iloc[i]) if np.isfinite(delta.iloc[i]) else np.nan,
                "trailing_mean_norm": float(np.nanmean(hist)) if np.isfinite(hist).any() else np.nan,
                "trailing_std_norm": float(np.nanstd(hist)) if np.isfinite(hist).any() else np.nan,
                "trailing_min_delta": float(np.nanmin(dhist)) if np.isfinite(dhist).any() else np.nan,
                "trailing_slope_norm": trailing_slope(hist),
                "is_abrupt_drop": bool(events.iloc[i]),
                "baseline_intensity": baseline,
                "event_delta_threshold": threshold,
            }
            future_cycles = cycles.iloc[i + 1 :].to_numpy(dtype=float)
            future_events = [e for e in event_cycles if e > float(cyc)]
            for h in horizons:
                row["event_within_%d_cycles" % h] = int(any((e - float(cyc)) <= h for e in future_events))
                dist = [e - float(cyc) for e in future_events if e > float(cyc)]
                row["cycles_to_next_event"] = float(min(dist)) if dist else np.nan
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def score_feature_thresholds(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    features = ["delta_norm", "trailing_std_norm", "trailing_min_delta", "trailing_slope_norm", "norm_intensity"]
    rows = []
    particles = sorted(df["particle"].dropna().unique())
    for horizon in horizons:
        label_col = "event_within_%d_cycles" % horizon
        for feature in features:
            vals = pd.to_numeric(df[feature], errors="coerce")
            if vals.notna().sum() < 5 or df[label_col].sum() == 0:
                continue
            # Choose direction from point-biserial sign on all data, then evaluate leave-one-particle-out thresholds.
            y = df[label_col].to_numpy(dtype=float)
            x = vals.to_numpy(dtype=float)
            good = np.isfinite(x)
            if good.sum() < 5 or np.nanstd(x[good]) == 0:
                continue
            direction = -1.0 if np.nanmean(x[(y == 1) & good]) < np.nanmean(x[(y == 0) & good]) else 1.0
            preds_all, y_all = [], []
            for p in particles:
                train = (df["particle"] != p) & vals.notna()
                test = (df["particle"] == p) & vals.notna()
                if train.sum() < 5 or test.sum() == 0:
                    continue
                train_scores = direction * vals[train].to_numpy(dtype=float)
                # Positive labels are rare; use high-risk top quartile from training scores.
                threshold = float(np.nanquantile(train_scores, 0.75))
                test_scores = direction * vals[test].to_numpy(dtype=float)
                pred = (test_scores >= threshold).astype(int)
                preds_all.extend(pred.tolist())
                y_all.extend(df.loc[test, label_col].astype(int).tolist())
            if not y_all:
                continue
            pred = np.asarray(preds_all, dtype=int)
            target = np.asarray(y_all, dtype=int)
            tp = int(((pred == 1) & (target == 1)).sum())
            fp = int(((pred == 1) & (target == 0)).sum())
            fn = int(((pred == 0) & (target == 1)).sum())
            tn = int(((pred == 0) & (target == 0)).sum())
            precision = tp / (tp + fp) if (tp + fp) else np.nan
            recall = tp / (tp + fn) if (tp + fn) else np.nan
            f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan
            rows.append({
                "horizon_cycles": horizon,
                "feature": feature,
                "direction": "higher_risk_when_larger" if direction > 0 else "higher_risk_when_smaller",
                "n_eval": int(len(target)),
                "positive_rate": float(target.mean()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })
    return pd.DataFrame(rows).sort_values(["horizon_cycles", "f1", "recall"], ascending=[True, False, False]) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_targets")
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--sigma-multiplier", type=float, default=2.0)
    args = parser.parse_args()

    particles_path = resolve_existing_path([
        args.particles_csv,
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/exampleParticles.csv",
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg/echemDF_full/exampleParticles.csv",
        "/home/ns2038/Downloads/alek_jiho_nmc_deg/echemDF_full/exampleParticles.csv",
    ])
    frames_path = resolve_existing_path([
        args.cycle_frames_csv,
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/cycleFrames.csv",
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv",
        "/home/ns2038/Downloads/alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv",
    ])
    if not particles_path:
        raise SystemExit("exampleParticles.csv not found")
    if not frames_path:
        raise SystemExit("cycleFrames.csv not found")

    particles = pd.read_csv(particles_path, encoding="utf-8-sig")
    frames = read_cycle_frames_csv(frames_path)
    particles["cycleNo"] = pd.to_numeric(particles["cycleNo"], errors="coerce")
    particles = particles.dropna(subset=["cycleNo"]).sort_values("cycleNo").reset_index(drop=True)
    particle_cols = [c for c in particles.columns if c.lower().startswith("particle")]
    if not particle_cols:
        raise SystemExit("No particle intensity columns found")
    for col in particle_cols:
        particles[col] = pd.to_numeric(particles[col], errors="coerce")

    long_df, events_df = make_long_table(particles, particle_cols, args.horizons, args.window, args.sigma_multiplier)
    score_df = score_feature_thresholds(long_df, args.horizons)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    long_path = os.path.join(out_dir, "particle_event_training_table.csv")
    event_path = os.path.join(out_dir, "particle_abrupt_events.csv")
    score_path = os.path.join(out_dir, "particle_event_feature_baselines.csv")
    long_df.to_csv(long_path, index=False)
    events_df.to_csv(event_path, index=False)
    score_df.to_csv(score_path, index=False)

    summary: Dict[str, object] = {
        "particles_csv": particles_path,
        "cycle_frames_csv": frames_path,
        "n_cycles": int(particles["cycleNo"].nunique()),
        "n_particles": len(particle_cols),
        "n_training_rows": int(len(long_df)),
        "n_events": int(len(events_df)),
        "events_by_particle": events_df.groupby("particle")["cycleNo"].apply(lambda s: [float(x) for x in s]).to_dict() if not events_df.empty else {},
        "horizons": args.horizons,
        "window": args.window,
        "sigma_multiplier": args.sigma_multiplier,
    }
    with open(os.path.join(out_dir, "particle_event_targets_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    lines = [
        "# NMC Particle Event Targets",
        "",
        "Cycle-level abrupt-drop labels and transparent precursor-feature baselines.",
        "",
        "## Event Counts",
        "",
        "| Particle | Event cycles |",
        "|---|---|",
    ]
    for particle, cycles in summary["events_by_particle"].items():
        lines.append("| `%s` | %s |" % (particle, cycles))
    lines.extend(["", "## Best Feature Baselines", "", "| Horizon | Feature | Direction | Precision | Recall | F1 |", "|---:|---|---|---:|---:|---:|"])
    if not score_df.empty:
        for _, row in score_df.groupby("horizon_cycles", as_index=False).head(3).iterrows():
            lines.append("| {horizon_cycles} | `{feature}` | {direction} | {precision:.3f} | {recall:.3f} | {f1:.3f} |".format(**row))
    lines.extend(["", "Files:", "", "- `particle_event_training_table.csv`", "- `particle_abrupt_events.csv`", "- `particle_event_feature_baselines.csv`", "- `particle_event_targets_summary.json`", ""])
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))
    print("[done] wrote %s; events=%d rows=%d" % (out_dir, len(events_df), len(long_df)))


if __name__ == "__main__":
    main()
