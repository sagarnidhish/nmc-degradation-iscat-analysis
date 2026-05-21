#!/usr/bin/env python3
"""Couple synchronized NMC optical events to electrochemical cycle features.

This script scans the large echemDF_full.csv in chunks, builds per-cycle echem
features, and compares cycles with abrupt optical particle events against all
other observed particle cycles. It is deliberately bounded and tabular so it can
run while GPU jobs are pending.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


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


def scan_echem_cycles(csv_path: str, chunksize: int) -> pd.DataFrame:
    usecols = ["Computer time (s)", "Time (s)", "Potential (V)", "Current (mA)", "Block", "cycleNo"]
    records = []
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, usecols=lambda c: c in usecols, encoding="utf-8-sig", low_memory=False), 1):
        if "cycleNo" not in chunk.columns:
            continue
        chunk["cycleNo"] = pd.to_numeric(chunk["cycleNo"], errors="coerce")
        chunk["Potential (V)"] = pd.to_numeric(chunk.get("Potential (V)"), errors="coerce")
        chunk["Current (mA)"] = pd.to_numeric(chunk.get("Current (mA)"), errors="coerce")
        if "Computer time (s)" in chunk.columns:
            time_col = "Computer time (s)"
        else:
            time_col = "Time (s)"
        chunk[time_col] = pd.to_numeric(chunk.get(time_col), errors="coerce")
        chunk = chunk.dropna(subset=["cycleNo", "Potential (V)", "Current (mA)"])
        if chunk.empty:
            continue
        for cyc, cdf in chunk.groupby("cycleNo", sort=False):
            t = cdf[time_col].to_numpy(dtype=float)
            v = cdf["Potential (V)"].to_numpy(dtype=float)
            i = cdf["Current (mA)"].to_numpy(dtype=float)
            if t.size:
                t_sorted = np.sort(t[np.isfinite(t)])
                duration = float(t_sorted[-1] - t_sorted[0]) if t_sorted.size >= 2 else 0.0
            else:
                duration = np.nan
            # Chunk-local summary records are merged later by weighted/aggregate reducers.
            block_mode = ""
            if "Block" in cdf.columns and cdf["Block"].notna().any():
                modes = cdf["Block"].astype(str).value_counts()
                block_mode = str(modes.index[0]) if not modes.empty else ""
            records.append({
                "cycleNo": float(cyc),
                "n_points": int(len(cdf)),
                "duration_s_chunk": duration,
                "V_min": float(np.nanmin(v)),
                "V_max": float(np.nanmax(v)),
                "V_mean_num": float(np.nansum(v)),
                "I_mean_num": float(np.nansum(i)),
                "I_abs_mean_num": float(np.nansum(np.abs(i))),
                "I_pos_points": int(np.sum(i > 0)),
                "I_neg_points": int(np.sum(i < 0)),
                "I_near_zero_points": int(np.sum(np.abs(i) <= 1e-6)),
                "block_mode_chunk": block_mode,
            })
        if chunk_idx % 10 == 0:
            print(f"scanned chunks={chunk_idx} records={len(records)}")
    if not records:
        return pd.DataFrame()
    part = pd.DataFrame(records)
    rows = []
    for cyc, g in part.groupby("cycleNo", sort=True):
        n = float(g["n_points"].sum())
        block = ""
        if g["block_mode_chunk"].notna().any():
            modes = g.loc[g["block_mode_chunk"] != "", "block_mode_chunk"].value_counts()
            block = str(modes.index[0]) if not modes.empty else ""
        rows.append({
            "cycleNo": float(cyc),
            "echem_points": int(n),
            "duration_s": float(g["duration_s_chunk"].sum()),
            "V_min": float(g["V_min"].min()),
            "V_max": float(g["V_max"].max()),
            "V_range": float(g["V_max"].max() - g["V_min"].min()),
            "V_mean": float(g["V_mean_num"].sum() / n) if n else np.nan,
            "I_mean_mA": float(g["I_mean_num"].sum() / n) if n else np.nan,
            "I_abs_mean_mA": float(g["I_abs_mean_num"].sum() / n) if n else np.nan,
            "I_pos_fraction": float(g["I_pos_points"].sum() / n) if n else np.nan,
            "I_neg_fraction": float(g["I_neg_points"].sum() / n) if n else np.nan,
            "I_near_zero_fraction": float(g["I_near_zero_points"].sum() / n) if n else np.nan,
            "block_mode": block,
        })
    out = pd.DataFrame(rows).sort_values("cycleNo").reset_index(drop=True)
    out["V_mean_delta"] = out["V_mean"].diff()
    out["I_mean_delta"] = out["I_mean_mA"].diff()
    out["V_range_delta"] = out["V_range"].diff()
    return out


def build_event_cycle_table(training: pd.DataFrame, events: pd.DataFrame, frames: pd.DataFrame, echem: pd.DataFrame) -> pd.DataFrame:
    all_cycles = pd.DataFrame({"cycleNo": sorted(training["cycleNo"].dropna().astype(float).unique())})
    counts = events.groupby("cycleNo").agg(
        n_particles_event=("particle", "nunique"),
        particles=("particle", lambda s: ",".join(sorted(set(map(str, s)))))
    ).reset_index()
    out = all_cycles.merge(counts, on="cycleNo", how="left")
    out["n_particles_event"] = out["n_particles_event"].fillna(0).astype(int)
    out["particles"] = out["particles"].fillna("")
    out["any_event"] = (out["n_particles_event"] > 0).astype(int)
    out["synchronized_event"] = (out["n_particles_event"] >= 2).astype(int)
    if not frames.empty:
        out = out.merge(frames, on="cycleNo", how="left")
        if "n_frames" in out.columns:
            out["n_frames"] = pd.to_numeric(out["n_frames"], errors="coerce")
            out["n_frames_percentile"] = out["n_frames"].rank(pct=True)
    if not echem.empty:
        out = out.merge(echem, on="cycleNo", how="left")
    return out.sort_values("cycleNo").reset_index(drop=True)


def compare_features(df: pd.DataFrame, label_col: str, features: List[str]) -> pd.DataFrame:
    rows = []
    labels = df[label_col].astype(int)
    for feat in features:
        vals = pd.to_numeric(df.get(feat), errors="coerce")
        a = vals[(labels == 1) & vals.notna()].to_numpy(dtype=float)
        b = vals[(labels == 0) & vals.notna()].to_numpy(dtype=float)
        if len(a) < 1 or len(b) < 3:
            continue
        try:
            _stat, pval = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            pval = np.nan
        rows.append({
            "label": label_col,
            "feature": feat,
            "event_mean": float(np.mean(a)),
            "non_event_mean": float(np.mean(b)),
            "event_median": float(np.median(a)),
            "non_event_median": float(np.median(b)),
            "event_n": int(len(a)),
            "non_event_n": int(len(b)),
            "difference_event_minus_non_event": float(np.mean(a) - np.mean(b)),
            "mannwhitney_p": float(pval) if np.isfinite(pval) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["label", "mannwhitney_p", "feature"], na_position="last") if rows else pd.DataFrame()


def correlate_event_load(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df["n_particles_event"], errors="coerce").to_numpy(dtype=float)
    for feat in features:
        x = pd.to_numeric(df.get(feat), errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 5 or np.nanstd(x[good]) == 0:
            continue
        rho, pval = spearmanr(x[good], y[good])
        rows.append({"feature": feat, "spearman_rho_event_count": float(rho), "pval": float(pval), "n": int(good.sum())})
    return pd.DataFrame(rows).sort_values("pval") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_targets")
    parser.add_argument("--echem-csv", default="")
    parser.add_argument("--cycle-frames-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/event_echem_coupling")
    parser.add_argument("--chunksize", type=int, default=750000)
    args = parser.parse_args()

    echem_path = resolve_existing_path([
        args.echem_csv,
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg/echemDF_full/echemDF_full.csv",
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/echemDF_full/echemDF_full.csv",
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/echemDF_full.csv",
        "/home/ns2038/Downloads/alek_jiho_nmc_deg/echemDF_full/echemDF_full.csv",
    ])
    frames_path = resolve_existing_path([
        args.cycle_frames_csv,
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/cycleFrames.csv",
        "/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv",
        "/home/ns2038/Downloads/alek_jiho_nmc_deg/echemDF_full/cycleFrames.csv",
    ])
    table_path = os.path.join(args.event_dir, "particle_event_training_table.csv")
    events_path = os.path.join(args.event_dir, "particle_abrupt_events.csv")
    if not echem_path:
        raise SystemExit("echemDF_full.csv not found")
    if not os.path.exists(table_path) or not os.path.exists(events_path):
        raise SystemExit("event target files not found")

    print(f"Reading event targets: {args.event_dir}")
    training = pd.read_csv(table_path)
    events = pd.read_csv(events_path)
    training["cycleNo"] = pd.to_numeric(training["cycleNo"], errors="coerce")
    events["cycleNo"] = pd.to_numeric(events["cycleNo"], errors="coerce")
    frames = read_cycle_frames_csv(frames_path) if frames_path else pd.DataFrame()
    if not frames.empty:
        frames["cycleNo"] = pd.to_numeric(frames["cycleNo"], errors="coerce")
        frames["n_frames"] = pd.to_numeric(frames["n_frames"], errors="coerce")

    print(f"Scanning echem CSV: {echem_path}")
    echem = scan_echem_cycles(echem_path, args.chunksize)

    os.makedirs(args.out_dir, exist_ok=True)
    echem_path_out = os.path.join(args.out_dir, "echem_cycle_summary.csv")
    echem.to_csv(echem_path_out, index=False)

    cycle_table = build_event_cycle_table(training, events, frames, echem)
    cycle_path = os.path.join(args.out_dir, "event_echem_cycle_table.csv")
    cycle_table.to_csv(cycle_path, index=False)

    feature_cols = [
        "n_frames", "n_frames_percentile", "echem_points", "duration_s", "V_min", "V_max", "V_range", "V_mean",
        "I_mean_mA", "I_abs_mean_mA", "I_pos_fraction", "I_neg_fraction", "I_near_zero_fraction",
        "V_mean_delta", "I_mean_delta", "V_range_delta",
    ]
    feature_cols = [c for c in feature_cols if c in cycle_table.columns]
    comparisons = pd.concat([
        compare_features(cycle_table, "any_event", feature_cols),
        compare_features(cycle_table, "synchronized_event", feature_cols),
    ], ignore_index=True)
    comp_path = os.path.join(args.out_dir, "event_vs_none_feature_tests.csv")
    comparisons.to_csv(comp_path, index=False)

    correlations = correlate_event_load(cycle_table, feature_cols)
    corr_path = os.path.join(args.out_dir, "event_count_feature_correlations.csv")
    correlations.to_csv(corr_path, index=False)

    sync_cycles = cycle_table[cycle_table["synchronized_event"] == 1].copy()
    summary: Dict[str, object] = {
        "event_dir": args.event_dir,
        "echem_csv": echem_path,
        "cycle_frames_csv": frames_path,
        "n_cycles_with_particle_data": int(cycle_table["cycleNo"].nunique()),
        "n_cycles_with_echem_match": int(cycle_table["echem_points"].notna().sum()) if "echem_points" in cycle_table.columns else 0,
        "n_any_event_cycles": int(cycle_table["any_event"].sum()),
        "n_synchronized_event_cycles": int(cycle_table["synchronized_event"].sum()),
        "synchronized_event_cycles": sync_cycles[["cycleNo", "n_particles_event", "particles"]].to_dict(orient="records"),
        "top_feature_tests": comparisons.head(12).to_dict(orient="records") if not comparisons.empty else [],
        "top_event_count_correlations": correlations.head(12).to_dict(orient="records") if not correlations.empty else [],
    }
    summary_path = os.path.join(args.out_dir, "event_echem_coupling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    readme_path = os.path.join(args.out_dir, "README.md")
    lines = [
        "# NMC Event-Electrochemistry Coupling",
        "",
        "Cycle-level comparison between abrupt optical particle events and electrochemical/protocol features from `echemDF_full.csv`.",
        "",
        "## Matched Data",
        "",
        f"- Particle cycles: {summary['n_cycles_with_particle_data']}",
        f"- Cycles with matching echem summary: {summary['n_cycles_with_echem_match']}",
        f"- Any-event cycles: {summary['n_any_event_cycles']}",
        f"- Synchronized-event cycles: {summary['n_synchronized_event_cycles']}",
        "",
        "## Synchronized Event Cycles",
        "",
        "| cycleNo | n_particles_event | particles | n_frames_percentile | V_mean | I_mean_mA | block_mode |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for _, row in sync_cycles.iterrows():
        lines.append("| {cycleNo:.1f} | {n} | `{particles}` | {fp} | {v} | {i} | `{block}` |".format(
            cycleNo=float(row["cycleNo"]),
            n=int(row["n_particles_event"]),
            particles=row["particles"],
            fp="" if pd.isna(row.get("n_frames_percentile", np.nan)) else f"{float(row['n_frames_percentile']):.3f}",
            v="" if pd.isna(row.get("V_mean", np.nan)) else f"{float(row['V_mean']):.4g}",
            i="" if pd.isna(row.get("I_mean_mA", np.nan)) else f"{float(row['I_mean_mA']):.4g}",
            block=row.get("block_mode", ""),
        ))
    lines.extend(["", "## Strongest Feature Differences", "", "| label | feature | event_mean | non_event_mean | p |", "|---|---|---:|---:|---:|"])
    if not comparisons.empty:
        for _, row in comparisons.head(10).iterrows():
            lines.append("| {label} | `{feature}` | {em:.4g} | {nm:.4g} | {p:.4g} |".format(
                label=row["label"], feature=row["feature"], em=float(row["event_mean"]), nm=float(row["non_event_mean"]), p=float(row["mannwhitney_p"])
            ))
    lines.extend(["", "## Interpretation Guardrail", "", "This is an exploratory cycle-level association analysis. Low p-values are hypothesis-generating only because event cycles are few and cycle numbers/experiment protocol can confound features. Use shuffled-cycle controls and independent sessions before making mechanistic claims.", ""])
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))

    for p in [echem_path_out, cycle_path, comp_path, corr_path, summary_path, readme_path]:
        print(f"Saved: {p}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
