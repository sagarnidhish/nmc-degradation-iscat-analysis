#!/usr/bin/env python3
"""Echem/optical breakpoint audit for NMC degradation cycles.

This audit tests whether synchronized optical degradation cycles coincide with
local shifts in cycle-level electrochemical trajectories.  It deliberately uses
compact cycle-level tables produced by earlier audits, so it can run quickly and
serve as a guardrail against overinterpreting video-only signals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


EVENT_WINDOWS = (4, 8, 12)
DEFAULT_N_PERM = 5000


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
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


def empirical_p_upper(obs: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(obs):
        return np.nan
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def empirical_p_abs(obs: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(obs):
        return np.nan
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))


def robust_scale(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(vals) < 4:
        return np.nan
    q25, q75 = np.nanpercentile(vals, [25, 75])
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 0:
        return float(iqr)
    std = np.nanstd(vals)
    return float(std) if np.isfinite(std) and std > 0 else np.nan


def local_shift(df: pd.DataFrame, center: float, feature: str, window: int) -> Dict[str, Any]:
    cyc = pd.to_numeric(df["cycleNo"], errors="coerce")
    vals = pd.to_numeric(df[feature], errors="coerce")
    pre = vals[(cyc >= center - window) & (cyc < center)].dropna()
    post = vals[(cyc > center) & (cyc <= center + window)].dropna()
    current = vals[cyc == center].dropna()
    scale = robust_scale(vals)
    if len(pre) == 0 or len(post) == 0:
        delta = np.nan
        delta_scaled = np.nan
    else:
        delta = float(post.median() - pre.median())
        delta_scaled = float(delta / scale) if np.isfinite(scale) and scale > 0 else np.nan
    current_minus_pre = np.nan
    if len(pre) and len(current):
        current_minus_pre = float(current.median() - pre.median())
    return {
        "center_cycle": float(center),
        "feature": feature,
        "window_cycles": int(window),
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
        "n_current": int(len(current)),
        "median_pre": float(pre.median()) if len(pre) else np.nan,
        "median_current": float(current.median()) if len(current) else np.nan,
        "median_post": float(post.median()) if len(post) else np.nan,
        "post_minus_pre": delta,
        "post_minus_pre_iqr_scaled": delta_scaled,
        "current_minus_pre": current_minus_pre,
    }


def candidate_centers(df: pd.DataFrame, window: int, event_cycles: Iterable[float]) -> List[float]:
    cycles = sorted(pd.to_numeric(df["cycleNo"], errors="coerce").dropna().unique())
    events = set(float(c) for c in event_cycles)
    centers = []
    for c in cycles:
        if float(c) in events:
            continue
        n_pre = sum((x >= c - window) and (x < c) for x in cycles)
        n_post = sum((x > c) and (x <= c + window) for x in cycles)
        if n_pre >= 2 and n_post >= 2:
            centers.append(float(c))
    return centers


def event_center_tests(df: pd.DataFrame, event_cycles: List[float], features: List[str], rng: np.random.Generator, n_perm: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for window in EVENT_WINDOWS:
        controls = candidate_centers(df, window, event_cycles)
        for feature in features:
            event_shifts = [local_shift(df, c, feature, window) for c in event_cycles]
            valid_events = [r for r in event_shifts if np.isfinite(r["post_minus_pre_iqr_scaled"])]
            if not valid_events:
                continue
            obs = float(np.nanmedian([r["post_minus_pre_iqr_scaled"] for r in valid_events]))
            control_rows = [local_shift(df, c, feature, window) for c in controls]
            control_shifts = np.array([
                r["post_minus_pre_iqr_scaled"] for r in control_rows if r["n_pre"] >= 2 and r["n_post"] >= 2
            ], dtype=float)
            control_shifts = control_shifts[np.isfinite(control_shifts)]
            null = []
            if len(control_shifts):
                draw_size = len(valid_events)
                for _ in range(n_perm):
                    sample = rng.choice(control_shifts, size=draw_size, replace=True)
                    null.append(float(np.nanmedian(sample)))
            null_arr = np.asarray(null, dtype=float)
            rows.append({
                "event_set": "synchronized_drop_2plus",
                "feature": feature,
                "window_cycles": int(window),
                "n_event_centers": int(len(valid_events)),
                "event_cycles": ";".join(str(int(c)) if float(c).is_integer() else str(c) for c in event_cycles),
                "event_median_post_minus_pre": float(np.nanmedian([r["post_minus_pre"] for r in valid_events])),
                "event_median_scaled_shift": obs,
                "control_median_scaled_shift": float(np.nanmedian(control_shifts)) if len(control_shifts) else np.nan,
                "control_p95_abs_scaled_shift": float(np.nanpercentile(np.abs(control_shifts), 95)) if len(control_shifts) else np.nan,
                "empirical_p_abs_vs_control_centers": empirical_p_abs(obs, control_shifts),
                "bootstrap_p_abs_vs_control_centers": empirical_p_abs(obs, null_arr),
                "event_shift_direction": "increase" if obs > 0 else "decrease" if obs < 0 else "flat",
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["bootstrap_p_abs_vs_control_centers", "empirical_p_abs_vs_control_centers", "feature"], na_position="last")


def breakpoint_scan(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cycles = sorted(pd.to_numeric(df["cycleNo"], errors="coerce").dropna().unique())
    for window in EVENT_WINDOWS:
        centers = []
        for c in cycles:
            n_pre = sum((x >= c - window) and (x < c) for x in cycles)
            n_post = sum((x > c) and (x <= c + window) for x in cycles)
            if n_pre >= 2 and n_post >= 2:
                centers.append(float(c))
        for feature in features:
            scale = robust_scale(df[feature])
            for c in centers:
                row = local_shift(df, c, feature, window)
                score = row["post_minus_pre_iqr_scaled"]
                if row["n_pre"] < 2 or row["n_post"] < 2 or not np.isfinite(score):
                    continue
                row.update({
                    "abs_scaled_shift": float(abs(score)),
                    "feature_scale_iqr_or_std": scale,
                    "rank_abs_shift_desc": np.nan,
                })
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_abs_shift_desc"] = out.groupby(["feature", "window_cycles"])["abs_scaled_shift"].rank(ascending=False, method="min")
    out["percentile_abs_shift"] = out.groupby(["feature", "window_cycles"])["abs_scaled_shift"].rank(pct=True)
    return out.sort_values(["abs_scaled_shift", "feature"], ascending=[False, True])


def future_label_tests(df: pd.DataFrame, features: List[str], targets: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target in targets:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce")
        for feature in features:
            x = pd.to_numeric(df[feature], errors="coerce")
            mask = x.notna() & y.isin([0, 1])
            if mask.sum() < 8 or y[mask].nunique() < 2:
                continue
            pos = x[mask & y.eq(1)].to_numpy(float)
            neg = x[mask & y.eq(0)].to_numpy(float)
            if len(pos) < 2 or len(neg) < 2:
                continue
            rho = np.nan
            pval = np.nan
            try:
                res = spearmanr(x[mask], y[mask])
                rho = float(res.statistic)
                pval = float(res.pvalue)
            except Exception:
                pass
            rows.append({
                "target": target,
                "feature": feature,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                "median_positive": float(np.nanmedian(pos)),
                "median_negative": float(np.nanmedian(neg)),
                "median_positive_minus_negative": float(np.nanmedian(pos) - np.nanmedian(neg)),
                "mannwhitney_p": float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue),
                "spearman_rho": rho,
                "spearman_p": pval,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["mannwhitney_p", "feature"])


def add_derivative_features(df: pd.DataFrame, base_features: List[str]) -> tuple[pd.DataFrame, List[str]]:
    out = df.sort_values("cycleNo").copy()
    derivative_features: List[str] = []
    for feature in base_features:
        vals = pd.to_numeric(out[feature], errors="coerce")
        delta_name = f"{feature}_delta_prev"
        roll_name = f"{feature}_rolling5_slope"
        out[delta_name] = vals.diff()
        derivative_features.append(delta_name)
        slopes = []
        cycles = pd.to_numeric(out["cycleNo"], errors="coerce").to_numpy(float)
        arr = vals.to_numpy(float)
        for i in range(len(out)):
            lo = max(0, i - 4)
            xx = cycles[lo : i + 1]
            yy = arr[lo : i + 1]
            mask = np.isfinite(xx) & np.isfinite(yy)
            if mask.sum() >= 3 and np.nanmax(xx[mask]) > np.nanmin(xx[mask]):
                slopes.append(float(np.polyfit(xx[mask], yy[mask], 1)[0]))
            else:
                slopes.append(np.nan)
        out[roll_name] = slopes
        derivative_features.append(roll_name)
    return out, derivative_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/echem_optical_breakpoint_audit")
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    state = read_csv(derived / "cycle_state_space_transition_audit" / "cycle_state_space_table.csv")
    trace = read_csv(derived / "particle_trace_physics_audit" / "particle_trace_cycle_features.csv")
    event = read_csv(derived / "integrated_event_evidence" / "integrated_event_evidence.csv")
    consensus = read_csv(derived / "cross_modal_degradation_consensus" / "cross_modal_consensus_cycle_table.csv")

    if state.empty and trace.empty:
        raise FileNotFoundError("Need cycle_state_space_table or particle_trace_cycle_features")
    df = state.copy() if not state.empty else trace.copy()
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    df = df.dropna(subset=["cycleNo"]).drop_duplicates("cycleNo", keep="first").sort_values("cycleNo")

    if not consensus.empty and "cycleNo" in consensus.columns:
        keep = [c for c in ["cycleNo", "cross_modal_consensus_score", "n_modal_votes", "consensus_class"] if c in consensus.columns]
        df = df.merge(consensus[keep].drop_duplicates("cycleNo"), on="cycleNo", how="left")

    event_cycles = sorted(pd.to_numeric(df.loc[pd.to_numeric(df.get("synchronized_drop_2plus", 0), errors="coerce").fillna(0).gt(0), "cycleNo"], errors="coerce").dropna().unique())
    if not event.empty and "synchronized_event" in event.columns:
        ev2 = pd.to_numeric(event.loc[pd.to_numeric(event["synchronized_event"], errors="coerce").fillna(0).gt(0), "cycleNo"], errors="coerce").dropna().unique()
        event_cycles = sorted(set(float(c) for c in event_cycles).union(float(c) for c in ev2))

    base_features = [
        "capacity_mAh",
        "charge_capacity_mAh",
        "discharge_capacity_mAh",
        "coulombic_efficiency_pct",
        "n_frames",
        "frames_percentile",
        "n_points",
        "shape_V_mean",
        "shape_V_std",
        "shape_V_range",
        "shape_I_abs_mean_mA",
        "shape_dVdt_abs_p95",
        "all_dq_abs_total_mAh",
        "all_dq_abs_lowV_frac",
        "all_dq_abs_midV_frac",
        "all_dq_abs_highV_frac",
        "all_dq_abs_entropy",
        "all_dqdv_abs_integral_proxy",
        "pos_dqdv_abs_integral_proxy",
        "neg_dqdv_abs_integral_proxy",
        "degradation_state_axis",
        "state_step_norm",
        "axis_step",
        "cycle_state_pc1",
        "cycle_state_pc2",
        "mean_abs_delta_prev",
        "max_abs_delta_prev",
        "particle_norm_cv",
    ]
    base_features = [f for f in base_features if f in df.columns and pd.to_numeric(df[f], errors="coerce").notna().sum() >= 8]
    df, derivative_features = add_derivative_features(df, base_features)
    test_features = base_features + derivative_features

    event_tests = event_center_tests(df, event_cycles, test_features, rng, args.n_perm)
    scan = breakpoint_scan(df, test_features)
    future_tests = future_label_tests(df, test_features, ["future_any_drop_within_8cycles", "future_sync2_drop_within_8cycles", "any_abrupt_drop", "synchronized_drop_2plus"])

    event_tests.to_csv(out / "echem_optical_event_centered_breakpoint_tests.csv", index=False)
    scan.to_csv(out / "echem_optical_breakpoint_scan.csv", index=False)
    future_tests.to_csv(out / "echem_optical_future_label_tests.csv", index=False)
    df.to_csv(out / "echem_optical_breakpoint_cycle_table.csv", index=False)

    event_rank_rows = []
    if not scan.empty and event_cycles:
        event_set = set(float(c) for c in event_cycles)
        tmp = scan[scan["center_cycle"].isin(event_set)].copy()
        event_rank_rows = tmp.sort_values(["rank_abs_shift_desc", "feature", "window_cycles"]).head(30).to_dict("records")
        tmp.to_csv(out / "echem_optical_event_cycle_breakpoint_ranks.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "echem_optical_event_cycle_breakpoint_ranks.csv", index=False)

    top_event_tests = event_tests.head(20).to_dict("records") if not event_tests.empty else []
    top_future_tests = future_tests.head(20).to_dict("records") if not future_tests.empty else []
    top_scan = scan.head(20).to_dict("records") if not scan.empty else []
    top_event_rank = event_rank_rows[:20]

    summary = {
        "n_cycles": int(len(df)),
        "n_features_tested": int(len(test_features)),
        "event_cycles": [float(c) for c in event_cycles],
        "n_permutation": int(args.n_perm),
        "top_event_centered_tests": clean_json(top_event_tests),
        "top_future_label_tests": clean_json(top_future_tests),
        "top_global_breakpoints": clean_json(top_scan),
        "event_cycle_breakpoint_ranks": clean_json(top_event_rank),
        "guardrail": "Cycle-level echem/optical breakpoint audit uses compact derived cycle features and weak optical event labels. It tests temporal co-occurrence and trajectory shifts, not causality, manual ROI validation, or calibrated material transport.",
        "outputs": {
            "cycle_table": str(out / "echem_optical_breakpoint_cycle_table.csv"),
            "event_centered_tests": str(out / "echem_optical_event_centered_breakpoint_tests.csv"),
            "breakpoint_scan": str(out / "echem_optical_breakpoint_scan.csv"),
            "future_label_tests": str(out / "echem_optical_future_label_tests.csv"),
            "event_cycle_ranks": str(out / "echem_optical_event_cycle_breakpoint_ranks.csv"),
        },
    }
    with (out / "echem_optical_breakpoint_audit_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2)

    readme = [
        "# Echem Optical Breakpoint Audit",
        "",
        "Tests whether synchronized optical degradation cycles align with local electrochemical trajectory shifts.",
        "",
        f"- Cycles: {summary['n_cycles']}",
        f"- Event cycles: {summary['event_cycles']}",
        f"- Features tested including derivatives: {summary['n_features_tested']}",
        f"- Permutations/bootstrap draws: {summary['n_permutation']}",
        "",
        "Guardrail: this is a cycle-level temporal association audit, not a calibrated diffusion or causal proof.",
    ]
    (out / "README.md").write_text("\n".join(readme).rstrip() + "\n")


if __name__ == "__main__":
    main()
