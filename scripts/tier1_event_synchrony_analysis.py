#!/usr/bin/env python3
"""Test whether NMC particle abrupt-drop events are synchronized across particles.

The immediate scientific question is whether optical degradation events are
independent particle failures or coordinated cell/protocol-linked events. This
script uses the event labels built by tier1_particle_event_targets.py and runs a
small permutation control that preserves each particle's event count while
randomizing event cycles over that particle's observed cycle grid.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def resolve_existing_path(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if path and os.path.exists(path):
            return path
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


def event_count_by_cycle(events: pd.DataFrame) -> pd.DataFrame:
    counts = (
        events.groupby("cycleNo")
        .agg(n_particles_event=("particle", "nunique"), particles=("particle", lambda s: ",".join(sorted(set(map(str, s))))))
        .reset_index()
        .sort_values(["n_particles_event", "cycleNo"], ascending=[False, True])
    )
    return counts


def permutation_max_synchrony(table: pd.DataFrame, events: pd.DataFrame, rng: np.random.Generator, n_perm: int) -> Dict[str, object]:
    particles = sorted(table["particle"].dropna().unique())
    observed_counts = event_count_by_cycle(events)
    observed_max = int(observed_counts["n_particles_event"].max()) if not observed_counts.empty else 0
    observed_cycles = observed_counts.loc[observed_counts["n_particles_event"] == observed_max, "cycleNo"].astype(float).tolist() if observed_max else []

    perm_maxima = np.zeros(n_perm, dtype=int)
    perm_n_multi = np.zeros(n_perm, dtype=int)
    for i in range(n_perm):
        rows = []
        for particle in particles:
            p_cycles = table.loc[table["particle"] == particle, "cycleNo"].dropna().astype(float).unique()
            n_events = int((events["particle"] == particle).sum())
            if n_events == 0 or len(p_cycles) == 0:
                continue
            chosen = rng.choice(p_cycles, size=min(n_events, len(p_cycles)), replace=False)
            rows.extend({"particle": particle, "cycleNo": float(c)} for c in chosen)
        if not rows:
            continue
        p_counts = event_count_by_cycle(pd.DataFrame(rows))
        perm_maxima[i] = int(p_counts["n_particles_event"].max())
        perm_n_multi[i] = int((p_counts["n_particles_event"] >= 2).sum())

    observed_n_multi = int((observed_counts["n_particles_event"] >= 2).sum()) if not observed_counts.empty else 0
    p_max = float((np.sum(perm_maxima >= observed_max) + 1) / (n_perm + 1)) if n_perm else np.nan
    p_multi = float((np.sum(perm_n_multi >= observed_n_multi) + 1) / (n_perm + 1)) if n_perm else np.nan
    return {
        "observed_max_particles_same_cycle": observed_max,
        "observed_max_cycles": observed_cycles,
        "observed_cycles_with_two_or_more_particles": observed_n_multi,
        "permutation_p_max_synchrony": p_max,
        "permutation_p_multi_cycle_count": p_multi,
        "perm_mean_max_particles_same_cycle": float(np.mean(perm_maxima)) if n_perm else np.nan,
        "perm_mean_cycles_with_two_or_more_particles": float(np.mean(perm_n_multi)) if n_perm else np.nan,
    }


def frame_count_context(counts: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    if frames.empty or "n_frames" not in frames.columns:
        return counts
    f = frames.copy()
    f["cycleNo"] = pd.to_numeric(f["cycleNo"], errors="coerce")
    f["n_frames"] = pd.to_numeric(f["n_frames"], errors="coerce")
    f = f.dropna(subset=["cycleNo", "n_frames"])
    if f.empty:
        return counts
    mean = float(f["n_frames"].mean())
    std = float(f["n_frames"].std(ddof=0))
    f["n_frames_z"] = (f["n_frames"] - mean) / std if std > 0 else np.nan
    f["n_frames_percentile"] = f["n_frames"].rank(pct=True)
    return counts.merge(f[["cycleNo", "n_frames", "n_frames_z", "n_frames_percentile"]], on="cycleNo", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_targets")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_synchrony")
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    table_path = os.path.join(args.event_dir, "particle_event_training_table.csv")
    events_path = os.path.join(args.event_dir, "particle_abrupt_events.csv")
    if not os.path.exists(table_path) or not os.path.exists(events_path):
        raise SystemExit("Missing particle event target files; run tier1_particle_event_targets.py first")

    frames_path = resolve_existing_path([
        args.cycle_frames_csv,
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/cycleFrames.csv",
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv",
        "/home/ns2038/Downloads/alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv",
    ])

    table = pd.read_csv(table_path)
    events = pd.read_csv(events_path)
    table["cycleNo"] = pd.to_numeric(table["cycleNo"], errors="coerce")
    events["cycleNo"] = pd.to_numeric(events["cycleNo"], errors="coerce")
    table = table.dropna(subset=["particle", "cycleNo"])
    events = events.dropna(subset=["particle", "cycleNo"])

    counts = event_count_by_cycle(events)
    frames = read_cycle_frames_csv(frames_path) if frames_path else pd.DataFrame()
    counts = frame_count_context(counts, frames)

    rng = np.random.default_rng(args.seed)
    sync = permutation_max_synchrony(table, events, rng, args.n_permutations)

    os.makedirs(args.out_dir, exist_ok=True)
    counts_path = os.path.join(args.out_dir, "event_synchrony_by_cycle.csv")
    summary_path = os.path.join(args.out_dir, "event_synchrony_summary.json")
    readme_path = os.path.join(args.out_dir, "README.md")

    counts.to_csv(counts_path, index=False)
    summary = {
        "event_dir": args.event_dir,
        "cycle_frames_csv": frames_path,
        "n_particles": int(table["particle"].nunique()),
        "n_cycles": int(table["cycleNo"].nunique()),
        "n_events": int(len(events)),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        **sync,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    top = counts.head(10).copy()
    lines = [
        "# NMC Event Synchrony Analysis",
        "",
        "Tests whether abrupt optical intensity drops are synchronized across particles more often than expected when each particle's event count is preserved but event cycles are randomized.",
        "",
        "## Summary",
        "",
        f"- Events: {summary['n_events']} across {summary['n_particles']} particles and {summary['n_cycles']} cycles.",
        f"- Observed maximum particles with an event in the same cycle: {summary['observed_max_particles_same_cycle']} at cycles {summary['observed_max_cycles']}.",
        f"- Permutation p-value for maximum same-cycle synchrony: {summary['permutation_p_max_synchrony']:.6f}.",
        f"- Observed cycles with >=2 particles eventing: {summary['observed_cycles_with_two_or_more_particles']}; permutation p-value: {summary['permutation_p_multi_cycle_count']:.6f}.",
        "",
        "## Top Event Cycles",
        "",
        "| cycleNo | n_particles_event | particles | n_frames | n_frames_percentile |",
        "|---:|---:|---|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            "| {cycleNo:.1f} | {n_particles_event} | `{particles}` | {n_frames} | {pct} |".format(
                cycleNo=float(row["cycleNo"]),
                n_particles_event=int(row["n_particles_event"]),
                particles=row["particles"],
                n_frames="" if pd.isna(row.get("n_frames", np.nan)) else int(row["n_frames"]),
                pct="" if pd.isna(row.get("n_frames_percentile", np.nan)) else f"{float(row['n_frames_percentile']):.3f}",
            )
        )
    lines.extend([
        "",
        "## Interpretation Guardrail",
        "",
        "A low permutation p-value supports coordinated optical event timing, but does not by itself prove a degradation mechanism. The next step is to compare these cycles with electrochemical capacity/dQdV features, protocol changes, and negative controls.",
        "",
    ])
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {counts_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {readme_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
