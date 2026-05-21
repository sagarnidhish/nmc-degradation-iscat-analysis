#!/usr/bin/env python3
"""Event-aligned particle-trace precursor atlas for NMC cycle photometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


FEATURES = [
    "particle_norm_mean",
    "particle_norm_std",
    "particle_norm_range",
    "particle_norm_cv",
    "mean_delta_prev",
    "mean_abs_delta_prev",
    "max_abs_delta_prev",
    "delta_std_across_particles",
    "capacity_mAh",
    "coulombic_efficiency_pct",
    "V_max",
    "n_frames",
    "frames_percentile",
]

WINDOWS: Dict[str, Tuple[int, int]] = {
    "pre16_to_pre9": (-16, -9),
    "pre8_to_pre5": (-8, -5),
    "pre4_to_pre1": (-4, -1),
    "event_cycle": (0, 0),
    "post1_to_post8": (1, 8),
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
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


def load_cycle_features(derived: Path) -> pd.DataFrame:
    path = derived / "particle_trace_physics_audit" / "particle_trace_cycle_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run tier4_particle_trace_physics_audit.py first; missing {path}")
    df = pd.read_csv(path)
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    return df.sort_values("cycleNo").reset_index(drop=True)


def event_anchors(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["cycleNo", "drop_count", "synchronized_drop_2plus", "synchronized_drop_3plus", "n_frames", "capacity_mAh", "V_max"]
    anchors = df.loc[df["any_abrupt_drop"].eq(1), [c for c in keep if c in df.columns]].copy()
    anchors["anchor_type"] = "event"
    anchors["anchor_id"] = anchors["cycleNo"].map(lambda c: f"event_cycle{int(c)}")
    return anchors


def candidate_control_anchors(df: pd.DataFrame, event_cycles: Iterable[float], exclusion: int, min_pre: int, min_post: int) -> pd.DataFrame:
    cycles = df["cycleNo"].to_numpy(dtype=float)
    event_arr = np.asarray(list(event_cycles), dtype=float)
    rows = []
    for _, row in df.iterrows():
        c = float(row["cycleNo"])
        if int(row.get("any_abrupt_drop", 0)) == 1:
            continue
        if len(event_arr) and np.min(np.abs(event_arr - c)) <= exclusion:
            continue
        n_pre = int(((cycles < c) & (cycles >= c - 16)).sum())
        n_post = int(((cycles > c) & (cycles <= c + 8)).sum())
        if n_pre < min_pre or n_post < min_post:
            continue
        rows.append(row)
    controls = pd.DataFrame(rows)
    if controls.empty:
        return pd.DataFrame()
    controls["anchor_type"] = "control"
    controls["anchor_id"] = controls["cycleNo"].map(lambda c: f"control_cycle{int(c)}")
    return controls


def standardize(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return x * 0
    return (x - x.mean()) / sd


def match_controls(events: pd.DataFrame, controls: pd.DataFrame, k: int) -> pd.DataFrame:
    if controls.empty:
        return controls
    match_cols = [c for c in ["cycleNo", "n_frames", "capacity_mAh", "V_max"] if c in controls.columns and c in events.columns]
    pooled = pd.concat([events[match_cols], controls[match_cols]], ignore_index=True)
    z = {c: standardize(pooled[c]) for c in match_cols}
    ev_z = pd.DataFrame({c: z[c].iloc[: len(events)].to_numpy() for c in match_cols})
    ctl_z = pd.DataFrame({c: z[c].iloc[len(events):].to_numpy() for c in match_cols}, index=controls.index)
    rows = []
    for ev_idx, ev in events.reset_index(drop=True).iterrows():
        dist = np.zeros(len(controls), dtype=float)
        for c in match_cols:
            dist += (ctl_z[c].to_numpy(dtype=float) - float(ev_z.loc[ev_idx, c])) ** 2
        order = np.argsort(dist)[:k]
        for rank, pos in enumerate(order, start=1):
            ctl = controls.iloc[pos].copy()
            ctl["matched_event_cycle"] = float(ev["cycleNo"])
            ctl["match_rank"] = rank
            ctl["match_distance"] = float(np.sqrt(dist[pos]))
            rows.append(ctl)
    matched = pd.DataFrame(rows).drop_duplicates(["cycleNo", "matched_event_cycle"])
    matched["anchor_type"] = "matched_control"
    matched["anchor_id"] = matched.apply(lambda r: f"control_cycle{int(r['cycleNo'])}_for_event{int(r['matched_event_cycle'])}", axis=1)
    return matched


def build_aligned_rows(df: pd.DataFrame, anchors: pd.DataFrame, max_pre: int, max_post: int) -> pd.DataFrame:
    rows = []
    cols = ["cycleNo", "any_abrupt_drop", "drop_count", "synchronized_drop_2plus"] + [c for c in FEATURES if c in df.columns]
    for _, anchor in anchors.iterrows():
        ac = float(anchor["cycleNo"])
        sub = df[(df["cycleNo"] >= ac - max_pre) & (df["cycleNo"] <= ac + max_post)].copy()
        sub["anchor_cycle"] = ac
        sub["anchor_id"] = anchor["anchor_id"]
        sub["anchor_type"] = anchor["anchor_type"]
        sub["matched_event_cycle"] = anchor.get("matched_event_cycle", np.nan)
        sub["relative_cycle"] = sub["cycleNo"] - ac
        rows.append(sub[["anchor_id", "anchor_type", "anchor_cycle", "matched_event_cycle", "relative_cycle"] + [c for c in cols if c in sub.columns]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def slope_for_window(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return np.nan
    return float(np.polyfit(x[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float), 1)[0])


def aggregate_windows(aligned: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for (anchor_id, anchor_type, anchor_cycle), sub in aligned.groupby(["anchor_id", "anchor_type", "anchor_cycle"], dropna=False):
        for name, (lo, hi) in WINDOWS.items():
            w = sub[(sub["relative_cycle"] >= lo) & (sub["relative_cycle"] <= hi)]
            if w.empty:
                continue
            for feat in features:
                vals = pd.to_numeric(w[feat], errors="coerce")
                rows.append({
                    "anchor_id": anchor_id,
                    "anchor_type": anchor_type,
                    "anchor_cycle": float(anchor_cycle),
                    "window": name,
                    "relative_cycle_min": lo,
                    "relative_cycle_max": hi,
                    "feature": feat,
                    "n_points": int(vals.notna().sum()),
                    "mean_value": float(vals.mean()) if vals.notna().any() else np.nan,
                    "median_value": float(vals.median()) if vals.notna().any() else np.nan,
                    "min_value": float(vals.min()) if vals.notna().any() else np.nan,
                    "max_value": float(vals.max()) if vals.notna().any() else np.nan,
                    "slope_per_cycle": slope_for_window(w["relative_cycle"], vals),
                })
    return pd.DataFrame(rows)


def compare_windows(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, feature, statistic), sub in agg.melt(
        id_vars=["anchor_id", "anchor_type", "anchor_cycle", "window", "feature"],
        value_vars=["mean_value", "median_value", "min_value", "max_value", "slope_per_cycle"],
        var_name="statistic",
        value_name="value",
    ).groupby(["window", "feature", "statistic"], dropna=False):
        event = sub.loc[sub["anchor_type"].eq("event"), "value"].dropna().to_numpy(dtype=float)
        control = sub.loc[sub["anchor_type"].eq("matched_control"), "value"].dropna().to_numpy(dtype=float)
        if len(event) >= 3 and len(control) >= 3:
            _, p = mannwhitneyu(event, control, alternative="two-sided")
            rows.append({
                "window": window,
                "feature": feature,
                "statistic": statistic,
                "n_event_anchors": int(len(event)),
                "n_control_anchors": int(len(control)),
                "event_median": float(np.median(event)),
                "control_median": float(np.median(control)),
                "event_minus_control_median": float(np.median(event) - np.median(control)),
                "mannwhitney_p": float(p),
            })
    return pd.DataFrame(rows).sort_values(["mannwhitney_p", "window", "feature", "statistic"]) if rows else pd.DataFrame()


def trajectory_summary(aligned: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for (atype, rel), sub in aligned.groupby(["anchor_type", "relative_cycle"], dropna=False):
        for feat in features:
            vals = pd.to_numeric(sub[feat], errors="coerce")
            if vals.notna().any():
                rows.append({
                    "anchor_type": atype,
                    "relative_cycle": float(rel),
                    "feature": feat,
                    "n": int(vals.notna().sum()),
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "std": float(vals.std()) if vals.notna().sum() > 1 else np.nan,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/particle_event_precursor_atlas")
    parser.add_argument("--control-k", type=int, default=6)
    parser.add_argument("--event-exclusion-cycles", type=int, default=16)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_cycle_features(derived)
    features = [f for f in FEATURES if f in df.columns]
    events = event_anchors(df)
    controls = candidate_control_anchors(df, events["cycleNo"], args.event_exclusion_cycles, min_pre=3, min_post=2)
    matched = match_controls(events, controls, args.control_k)
    anchors = pd.concat([events, matched], ignore_index=True, sort=False)
    aligned = build_aligned_rows(df, anchors, max_pre=16, max_post=8)
    aggregate = aggregate_windows(aligned, features)
    tests = compare_windows(aggregate)
    traj = trajectory_summary(aligned, features)

    paths = {
        "anchors": out / "particle_event_precursor_anchors.csv",
        "aligned_rows": out / "particle_event_precursor_aligned_rows.csv",
        "window_features": out / "particle_event_precursor_window_features.csv",
        "window_tests": out / "particle_event_precursor_window_tests.csv",
        "trajectory_summary": out / "particle_event_precursor_trajectory_summary.csv",
        "summary": out / "particle_event_precursor_atlas_summary.json",
    }
    anchors.to_csv(paths["anchors"], index=False)
    aligned.to_csv(paths["aligned_rows"], index=False)
    aggregate.to_csv(paths["window_features"], index=False)
    tests.to_csv(paths["window_tests"], index=False)
    traj.to_csv(paths["trajectory_summary"], index=False)

    top_tests = tests.head(20).to_dict("records") if not tests.empty else []
    pre_tests = tests[tests["window"].isin(["pre16_to_pre9", "pre8_to_pre5", "pre4_to_pre1"])].head(20).to_dict("records") if not tests.empty else []
    summary = {
        "n_cycle_rows": int(len(df)),
        "n_event_anchors": int(len(events)),
        "n_candidate_control_anchors": int(len(controls)),
        "n_matched_control_anchors": int(len(matched)),
        "event_cycles": events[["cycleNo", "drop_count", "synchronized_drop_2plus"]].to_dict("records"),
        "features": features,
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "top_window_tests": top_tests,
        "top_precursor_window_tests": pre_tests,
        "guardrail": "Precursor windows are aligned to four detected abrupt-drop cycles and matched non-event anchors from the four-particle cycle table. Results show cycle-level trace precursors for review and hypothesis generation, not localized phase-front motion or calibrated diffusion.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Particle Event Precursor Atlas",
        "",
        "Event-aligned cycle-level precursor windows from the normalized four-particle trace table.",
        "",
        f"- Event anchors: {summary['n_event_anchors']}",
        f"- Candidate control anchors: {summary['n_candidate_control_anchors']}",
        f"- Matched control anchors: {summary['n_matched_control_anchors']}",
        "",
        "## Top Precursor Window Tests",
    ]
    for row in pre_tests[:10]:
        lines.append(
            f"- {row.get('window')} {row.get('feature')} {row.get('statistic')}: event-control={row.get('event_minus_control_median'):.3g}, p={row.get('mannwhitney_p'):.3g}"
        )
    lines += ["", "## Top All Window Tests"]
    for row in top_tests[:10]:
        lines.append(
            f"- {row.get('window')} {row.get('feature')} {row.get('statistic')}: event-control={row.get('event_minus_control_median'):.3g}, p={row.get('mannwhitney_p'):.3g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "out_dir": str(out),
        "n_event_anchors": summary["n_event_anchors"],
        "n_matched_control_anchors": summary["n_matched_control_anchors"],
        "top_precursor_tests": pre_tests[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
