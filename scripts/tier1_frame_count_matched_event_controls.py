#!/usr/bin/env python3
"""Frame-count/protocol matched controls for synchronized NMC optical events.

This tests whether cycles with >=2 particle abrupt-drop events remain unusual after
matching on frame count percentile and protocol block. It is a direct artifact
control for the cycles 86/116 synchrony result.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def empirical_p_ge(observed: float, null_values: np.ndarray) -> float:
    return float((np.sum(null_values >= observed) + 1) / (len(null_values) + 1))


def empirical_p_le(observed: float, null_values: np.ndarray) -> float:
    return float((np.sum(null_values <= observed) + 1) / (len(null_values) + 1))


def sample_matched_controls(df: pd.DataFrame, event_rows: pd.DataFrame, n_perm: int, seed: int, frame_band: float) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    usable = df.copy()
    usable["n_frames_percentile"] = pd.to_numeric(usable["n_frames_percentile"], errors="coerce")
    event_rows = event_rows.copy()
    event_rows["n_frames_percentile"] = pd.to_numeric(event_rows["n_frames_percentile"], errors="coerce")

    draws: List[List[float]] = []
    failures = 0
    for _ in range(n_perm):
        chosen = []
        for _, ev in event_rows.iterrows():
            pool = usable[usable["synchronized_event"].astype(int) == 0].copy()
            if pd.notna(ev.get("block_mode")) and str(ev.get("block_mode")) not in ("", "nan"):
                same_block = pool[pool["block_mode"].astype(str) == str(ev.get("block_mode"))]
                if len(same_block) >= 3:
                    pool = same_block
            if pd.notna(ev["n_frames_percentile"]):
                band = frame_band
                matched = pool[(pool["n_frames_percentile"] - ev["n_frames_percentile"]).abs() <= band]
                while len(matched) < 3 and band < 0.50:
                    band *= 1.5
                    matched = pool[(pool["n_frames_percentile"] - ev["n_frames_percentile"]).abs() <= band]
                if len(matched):
                    pool = matched
            if len(pool) == 0:
                failures += 1
                continue
            chosen.append(float(rng.choice(pool["cycleNo"].to_numpy(dtype=float))))
        if len(chosen) == len(event_rows):
            draws.append(chosen)
    arr = np.asarray(draws, dtype=float)
    return {"draws": arr, "failed_event_matches": failures}


def summarize_cycle_set(df: pd.DataFrame, cycles: np.ndarray) -> Dict[str, float]:
    sub = df[df["cycleNo"].isin(cycles)].copy()
    out: Dict[str, float] = {"n_cycles": float(len(sub))}
    for col in ["n_frames", "n_frames_percentile", "V_mean", "I_mean_mA", "I_abs_mean_mA", "duration_s", "echem_points"]:
        if col in sub.columns:
            vals = pd.to_numeric(sub[col], errors="coerce")
            out[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            out[f"{col}_min"] = float(vals.min()) if vals.notna().any() else np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-table", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_echem_coupling/event_echem_cycle_table.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/frame_count_matched_controls")
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--frame-band", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    df = pd.read_csv(args.cycle_table)
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    for col in ["synchronized_event", "any_event", "n_particles_event"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    event_rows = df[df["synchronized_event"] == 1].copy()
    if event_rows.empty:
        raise SystemExit("No synchronized event cycles found")

    observed_cycles = event_rows["cycleNo"].to_numpy(dtype=float)
    observed = summarize_cycle_set(df, observed_cycles)
    matched = sample_matched_controls(df, event_rows, args.n_perm, args.seed, args.frame_band)
    draws = matched["draws"]

    null_rows = []
    for cycles in draws:
        null_rows.append(summarize_cycle_set(df, cycles))
    null_df = pd.DataFrame(null_rows)

    tests = []
    for key, obs_val in observed.items():
        if key == "n_cycles" or key not in null_df.columns or not np.isfinite(obs_val):
            continue
        vals = pd.to_numeric(null_df[key], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        tests.append({
            "metric": key,
            "observed": obs_val,
            "null_mean": float(np.mean(vals)),
            "null_p05": float(np.quantile(vals, 0.05)),
            "null_p50": float(np.quantile(vals, 0.50)),
            "null_p95": float(np.quantile(vals, 0.95)),
            "p_lower_or_equal": empirical_p_le(obs_val, vals),
            "p_higher_or_equal": empirical_p_ge(obs_val, vals),
        })

    # Global unpaired comparison retained as context, not the main matched test.
    unpaired = []
    non_sync = df[df["synchronized_event"] == 0]
    for col in ["n_frames", "n_frames_percentile", "V_mean", "I_mean_mA", "I_abs_mean_mA", "duration_s"]:
        if col not in df.columns:
            continue
        a = pd.to_numeric(event_rows[col], errors="coerce").dropna()
        b = pd.to_numeric(non_sync[col], errors="coerce").dropna()
        if len(a) and len(b) >= 3:
            _stat, p = mannwhitneyu(a, b, alternative="two-sided")
            unpaired.append({"metric": col, "event_mean": float(a.mean()), "non_event_mean": float(b.mean()), "mannwhitney_p": float(p)})

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(tests).to_csv(os.path.join(args.out_dir, "matched_control_p_values.csv"), index=False)
    pd.DataFrame(unpaired).to_csv(os.path.join(args.out_dir, "unpaired_sync_vs_other_tests.csv"), index=False)
    null_df.to_csv(os.path.join(args.out_dir, "matched_null_draw_summaries.csv"), index=False)

    summary = {
        "cycle_table": args.cycle_table,
        "synchronized_event_cycles": observed_cycles.tolist(),
        "observed_summary": observed,
        "n_permutations_requested": args.n_perm,
        "n_matched_draws": int(len(null_df)),
        "failed_event_matches": int(matched["failed_event_matches"]),
        "top_matched_tests": tests[:12],
        "interpretation": "If observed low frame-count percentile remains extreme after matching, event cycles are not explained by frame-count/protocol artifact alone. If it does not, raw ROI QC is mandatory before physical claims.",
    }
    with open(os.path.join(args.out_dir, "frame_count_matched_controls_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Frame-Count Matched Event Controls\n\n")
        f.write("Tests synchronized event cycles against matched non-event cycles.\n\n")
        f.write(f"Observed synchronized cycles: {observed_cycles.tolist()}\n\n")
        f.write("Key output: `matched_control_p_values.csv`.\n")
    print(f"[done] wrote matched controls to {args.out_dir}; draws={len(null_df)}")


if __name__ == "__main__":
    main()
