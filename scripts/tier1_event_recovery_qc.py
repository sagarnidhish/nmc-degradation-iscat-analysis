#!/usr/bin/env python3
"""QC abrupt NMC optical events by pre/post recovery behavior.

The event synchrony analysis found cycles where multiple particles drop together.
This script asks whether those drops are single-cycle transients or persist into
subsequent observed cycles. Persistent drops are more consistent with degradation
than with a one-off imaging/segmentation artifact, though visual QC is still
required.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


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


def parse_addr(addr: str) -> Dict[str, object]:
    s = str(addr)
    m = re.search(r"NMC_degradation_3_160623_Halfthedata\\([^\\]+)chopped\\([^\\]+)_cycle(\d+)\.hdf5", s)
    if not m:
        return {"source_stem": "", "local_cycle_index": np.nan}
    return {"source_stem": m.group(1), "local_cycle_index": int(m.group(3))}


def build_trace_table(particles: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    df = particles.copy()
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    meta = df["addrs"].map(parse_addr).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)
    for col in [c for c in df.columns if c.lower().startswith("particle")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    frames = frames.copy()
    frames["cycleNo"] = pd.to_numeric(frames["cycleNo"], errors="coerce")
    frames["n_frames"] = pd.to_numeric(frames["n_frames"], errors="coerce")
    df = df.merge(frames[["cycleNo", "n_frames"]], on="cycleNo", how="left")
    df["global_frame_percentile"] = df["n_frames"].rank(pct=True)
    df["session_frame_percentile"] = df.groupby("source_stem")["n_frames"].rank(pct=True)
    return df.sort_values("cycleNo").reset_index(drop=True)


def event_recovery_rows(trace: pd.DataFrame, events: pd.DataFrame, pre_window: int, post_window: int) -> pd.DataFrame:
    rows = []
    particle_cols = [c for c in trace.columns if c.lower().startswith("particle") and c != "particles"]
    cycle_values = trace["cycleNo"].to_numpy(dtype=float)
    for _, ev in events.iterrows():
        particle = str(ev["particle"])
        cyc = float(ev["cycleNo"])
        if particle not in particle_cols:
            continue
        idxs = np.where(cycle_values == cyc)[0]
        if len(idxs) == 0:
            continue
        i = int(idxs[0])
        pre = trace.iloc[max(0, i - pre_window):i]
        post = trace.iloc[i + 1:i + 1 + post_window]
        event_val = float(trace.iloc[i][particle]) if pd.notna(trace.iloc[i][particle]) else np.nan
        pre_vals = pd.to_numeric(pre[particle], errors="coerce").dropna().to_numpy(dtype=float)
        post_vals = pd.to_numeric(post[particle], errors="coerce").dropna().to_numpy(dtype=float)
        pre_mean = float(np.mean(pre_vals)) if pre_vals.size else np.nan
        post_mean = float(np.mean(post_vals)) if post_vals.size else np.nan
        next_val = float(post_vals[0]) if post_vals.size else np.nan
        drop_abs = pre_mean - event_val if np.isfinite(pre_mean) and np.isfinite(event_val) else np.nan
        drop_frac = drop_abs / pre_mean if np.isfinite(drop_abs) and pre_mean != 0 else np.nan
        recovery_frac_next = (next_val - event_val) / drop_abs if np.isfinite(next_val) and np.isfinite(drop_abs) and drop_abs != 0 else np.nan
        recovery_frac_post = (post_mean - event_val) / drop_abs if np.isfinite(post_mean) and np.isfinite(drop_abs) and drop_abs != 0 else np.nan
        next_deficit_frac = (pre_mean - next_val) / pre_mean if np.isfinite(pre_mean) and np.isfinite(next_val) and pre_mean != 0 else np.nan
        sustained_next = bool(np.isfinite(recovery_frac_next) and recovery_frac_next < 0.75 and np.isfinite(drop_frac) and drop_frac > 0.05)
        rows.append({
            "particle": particle,
            "cycleNo": cyc,
            "event_value": event_val,
            "pre_mean": pre_mean,
            "next_value": next_val,
            "post_mean": post_mean,
            "drop_abs": drop_abs,
            "drop_frac": drop_frac,
            "recovery_frac_next": recovery_frac_next,
            "recovery_frac_post_mean": recovery_frac_post,
            "next_deficit_frac": next_deficit_frac,
            "sustained_next_cycle": sustained_next,
            "source_stem": trace.iloc[i].get("source_stem", ""),
            "local_cycle_index": trace.iloc[i].get("local_cycle_index", np.nan),
            "n_frames": trace.iloc[i].get("n_frames", np.nan),
            "global_frame_percentile": trace.iloc[i].get("global_frame_percentile", np.nan),
            "session_frame_percentile": trace.iloc[i].get("session_frame_percentile", np.nan),
            "addrs": trace.iloc[i].get("addrs", ""),
        })
    return pd.DataFrame(rows)


def cycle_event_summary(event_rows: pd.DataFrame) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    g = event_rows.groupby("cycleNo", as_index=False).agg(
        n_event_particles=("particle", "nunique"),
        particles=("particle", lambda s: ",".join(sorted(set(map(str, s))))),
        mean_drop_frac=("drop_frac", "mean"),
        median_drop_frac=("drop_frac", "median"),
        mean_recovery_frac_next=("recovery_frac_next", "mean"),
        median_next_deficit_frac=("next_deficit_frac", "median"),
        n_sustained_next_cycle=("sustained_next_cycle", "sum"),
        n_frames=("n_frames", "first"),
        global_frame_percentile=("global_frame_percentile", "first"),
        session_frame_percentile=("session_frame_percentile", "first"),
        source_stem=("source_stem", "first"),
        local_cycle_index=("local_cycle_index", "first"),
    )
    g["synchronized_event"] = (g["n_event_particles"] >= 2).astype(int)
    g["all_event_particles_sustained_next"] = (g["n_sustained_next_cycle"] == g["n_event_particles"]).astype(int)
    return g.sort_values(["n_event_particles", "cycleNo"], ascending=[False, True])


def compare_synchronized_to_single(cycle_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if cycle_summary.empty or cycle_summary["synchronized_event"].nunique() < 2:
        return pd.DataFrame()
    for feat in ["mean_drop_frac", "mean_recovery_frac_next", "median_next_deficit_frac", "n_frames", "global_frame_percentile", "session_frame_percentile"]:
        a = pd.to_numeric(cycle_summary.loc[cycle_summary.synchronized_event == 1, feat], errors="coerce").dropna().to_numpy(float)
        b = pd.to_numeric(cycle_summary.loc[cycle_summary.synchronized_event == 0, feat], errors="coerce").dropna().to_numpy(float)
        if len(a) < 1 or len(b) < 1:
            continue
        pval = np.nan
        if len(a) >= 1 and len(b) >= 1:
            try:
                _, pval = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                pval = np.nan
        rows.append({
            "feature": feat,
            "sync_mean": float(np.mean(a)),
            "single_mean": float(np.mean(b)),
            "sync_n": int(len(a)),
            "single_n": int(len(b)),
            "mannwhitney_p": float(pval) if np.isfinite(pval) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mannwhitney_p", na_position="last") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles-csv", default="")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--event-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/particle_event_targets")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/event_recovery_qc")
    parser.add_argument("--pre-window", type=int, default=2)
    parser.add_argument("--post-window", type=int, default=2)
    args = parser.parse_args()

    particles_path = resolve_existing_path([
        args.particles_csv,
        "/scratch/<account>/<username>/Alek_Jiho/exampleParticles.csv",
        "/scratch/<account>/<username>/Alek_Jiho/alek_jiho_nmc_deg/echemDF_full/exampleParticles.csv",
        "/path/to/alek_jiho_nmc_deg/echemDF_full/exampleParticles.csv",
    ])
    frames_path = resolve_existing_path([
        args.cycle_frames_csv,
        "/scratch/<account>/<username>/Alek_Jiho/cycleFrames.csv",
        "/scratch/<account>/<username>/Alek_Jiho/alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv",
        "/path/to/alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv",
    ])
    events_path = os.path.join(args.event_dir, "particle_abrupt_events.csv")
    if not particles_path or not frames_path:
        raise SystemExit("Missing particles/cycle frames CSV")
    if not os.path.exists(events_path):
        raise SystemExit("Missing particle_abrupt_events.csv; run tier1_particle_event_targets.py first")

    particles = pd.read_csv(particles_path, encoding="utf-8-sig")
    frames = read_cycle_frames_csv(frames_path)
    events = pd.read_csv(events_path)
    events["cycleNo"] = pd.to_numeric(events["cycleNo"], errors="coerce")

    trace = build_trace_table(particles, frames)
    event_rows = event_recovery_rows(trace, events, args.pre_window, args.post_window)
    cycle_summary = cycle_event_summary(event_rows)
    comparisons = compare_synchronized_to_single(cycle_summary)

    os.makedirs(args.out_dir, exist_ok=True)
    trace_path = os.path.join(args.out_dir, "event_trace_context.csv")
    rows_path = os.path.join(args.out_dir, "event_particle_recovery_qc.csv")
    cycle_path = os.path.join(args.out_dir, "event_cycle_recovery_summary.csv")
    comp_path = os.path.join(args.out_dir, "synchronized_vs_single_event_recovery_tests.csv")
    trace.to_csv(trace_path, index=False)
    event_rows.to_csv(rows_path, index=False)
    cycle_summary.to_csv(cycle_path, index=False)
    comparisons.to_csv(comp_path, index=False)

    summary: Dict[str, object] = {
        "particles_csv": particles_path,
        "cycle_frames_csv": frames_path,
        "event_dir": args.event_dir,
        "pre_window": args.pre_window,
        "post_window": args.post_window,
        "n_event_particle_rows": int(len(event_rows)),
        "n_event_cycles": int(cycle_summary["cycleNo"].nunique()) if not cycle_summary.empty else 0,
        "n_synchronized_event_cycles": int(cycle_summary["synchronized_event"].sum()) if not cycle_summary.empty else 0,
        "event_cycle_summary": cycle_summary.to_dict(orient="records") if not cycle_summary.empty else [],
        "synchronized_vs_single_tests": comparisons.to_dict(orient="records") if not comparisons.empty else [],
    }
    summary_path = os.path.join(args.out_dir, "event_recovery_qc_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    readme_path = os.path.join(args.out_dir, "README.md")
    lines = [
        "# NMC Event Recovery QC",
        "",
        "Checks whether abrupt particle-intensity drops persist into subsequent observed cycles or recover immediately.",
        "",
        "## Event Cycle Summary",
        "",
        "| cycleNo | n_event_particles | particles | mean_drop_frac | mean_recovery_frac_next | median_next_deficit_frac | sustained/total | n_frames | global_frame_percentile | source_stem | local_cycle_index |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for _, row in cycle_summary.iterrows():
        lines.append("| {cycleNo:.1f} | {n} | `{particles}` | {drop:.3f} | {rec:.3f} | {deficit:.3f} | {sust}/{n} | {frames} | {fp:.3f} | `{stem}` | {local} |".format(
            cycleNo=float(row["cycleNo"]),
            n=int(row["n_event_particles"]),
            particles=row["particles"],
            drop=float(row["mean_drop_frac"]),
            rec=float(row["mean_recovery_frac_next"]) if pd.notna(row["mean_recovery_frac_next"]) else float("nan"),
            deficit=float(row["median_next_deficit_frac"]) if pd.notna(row["median_next_deficit_frac"]) else float("nan"),
            sust=int(row["n_sustained_next_cycle"]),
            frames="" if pd.isna(row.get("n_frames", np.nan)) else int(row["n_frames"]),
            fp=float(row["global_frame_percentile"]) if pd.notna(row["global_frame_percentile"]) else float("nan"),
            stem=row.get("source_stem", ""),
            local="" if pd.isna(row.get("local_cycle_index", np.nan)) else int(row["local_cycle_index"]),
        ))
    lines.extend(["", "## Interpretation Guardrail", "", "Persistence in the next cycle supports a real degradation-like optical change, but it is not a substitute for visual particle-region QC from the raw frames. Missing chopped HDF5 files mean this pass uses the provided particle trace tables and full-file metadata rather than direct per-particle image crops.", ""])
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))

    for path in [trace_path, rows_path, cycle_path, comp_path, summary_path, readme_path]:
        print(f"Saved: {path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
