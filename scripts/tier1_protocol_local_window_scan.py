#!/usr/bin/env python3
"""Protocol-local electrochemistry scan around synchronized optical events."""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def local_features(echem: pd.DataFrame, cycle: float, half_width: int) -> List[Dict[str, float]]:
    rows = []
    for offset in range(-half_width, half_width + 1):
        target = cycle + offset
        sub = echem[echem["cycleNo"] == target]
        if sub.empty:
            continue
        row = sub.iloc[0].to_dict()
        row["event_cycle"] = cycle
        row["offset"] = offset
        rows.append(row)
    return rows


def add_neighbor_deltas(local: pd.DataFrame) -> pd.DataFrame:
    out = local.copy()
    for col in ["V_mean", "V_range", "I_mean_mA", "I_abs_mean_mA", "duration_s", "echem_points"]:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        baseline = out[out["offset"].isin([-2, -1, 1, 2])].groupby("event_cycle")[col].transform("mean")
        out[f"{col}_minus_neighbor_mean"] = out[col] - baseline
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-table", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_echem_coupling/event_echem_cycle_table.csv")
    parser.add_argument("--echem-summary", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_echem_coupling/echem_cycle_summary.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/protocol_local_window_scan")
    parser.add_argument("--half-width", type=int, default=6)
    args = parser.parse_args()

    cycles = pd.read_csv(args.cycle_table)
    echem = pd.read_csv(args.echem_summary)
    cycles["cycleNo"] = pd.to_numeric(cycles["cycleNo"], errors="coerce")
    echem["cycleNo"] = pd.to_numeric(echem["cycleNo"], errors="coerce")
    event_cycles = cycles.loc[pd.to_numeric(cycles["synchronized_event"], errors="coerce").fillna(0).astype(int) == 1, "cycleNo"].to_numpy(dtype=float)
    if len(event_cycles) == 0:
        raise SystemExit("No synchronized event cycles found")

    local_rows: List[Dict[str, float]] = []
    for cyc in event_cycles:
        local_rows.extend(local_features(echem, cyc, args.half_width))
    local = pd.DataFrame(local_rows)
    local = add_neighbor_deltas(local)

    event_center = local[local["offset"] == 0].copy()
    neighbor = local[local["offset"].abs().isin([1, 2, 3])].copy()
    tests = []
    for col in ["V_mean", "V_range", "I_mean_mA", "I_abs_mean_mA", "duration_s", "echem_points", "V_mean_delta", "I_mean_delta", "V_range_delta"]:
        if col not in local.columns:
            continue
        a = pd.to_numeric(event_center[col], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(neighbor[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(a) and len(b) >= 3:
            _stat, p = mannwhitneyu(a, b, alternative="two-sided")
            tests.append({"metric": col, "event_center_mean": float(np.mean(a)), "neighbor_mean": float(np.mean(b)), "event_minus_neighbor": float(np.mean(a) - np.mean(b)), "mannwhitney_p": float(p)})

    os.makedirs(args.out_dir, exist_ok=True)
    local.to_csv(os.path.join(args.out_dir, "local_echem_window_features.csv"), index=False)
    pd.DataFrame(tests).sort_values("mannwhitney_p").to_csv(os.path.join(args.out_dir, "event_center_vs_neighbor_tests.csv"), index=False)
    summary = {
        "event_cycles": event_cycles.tolist(),
        "half_width": args.half_width,
        "n_local_rows": int(len(local)),
        "top_tests": sorted(tests, key=lambda r: r["mannwhitney_p"])[:12],
        "interpretation": "Local event-cycle anomalies suggest protocol/echem forcing candidates; null results mean coarse per-cycle echem is insufficient and raw within-cycle traces should be scanned.",
    }
    with open(os.path.join(args.out_dir, "protocol_local_window_scan_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("# Protocol-Local Echem Window Scan\n\nCompares synchronized event cycles with nearby cycles in echem feature space.\n")
    print(f"[done] wrote protocol-local scan to {args.out_dir}")


if __name__ == "__main__":
    main()
