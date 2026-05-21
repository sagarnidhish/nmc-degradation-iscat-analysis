#!/usr/bin/env python3
"""Within-cycle electrochemical shape audit for NMC optical degradation physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


ROI_TARGETS = [
    "is_event_roi",
    "is_event_enriched_mode",
    "phase_slope_positive_fraction_protocol_residual",
    "threshold_robust_phase_score_protocol_residual",
    "q70_logistic_k_per_s",
    "q70_transformed_fraction_delta",
    "mode_review_priority",
]


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def observed_cycles(derived: Path) -> Set[int]:
    cycles: Set[int] = set()
    for path in [
        derived / "particle_trace_physics_audit" / "particle_trace_cycle_features.csv",
        derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv",
        derived / "phase_kinetics_avrami" / "phase_kinetics_avrami_roi_table.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path, usecols=lambda c: c == "cycleNo")
            cycles.update(int(c) for c in pd.to_numeric(df["cycleNo"], errors="coerce").dropna().unique())
    return cycles


def trapezoid_capacity_mAh(t: np.ndarray, i_mA: np.ndarray) -> float:
    if len(t) < 2:
        return np.nan
    order = np.argsort(t)
    t = t[order]
    i_mA = i_mA[order]
    good = np.isfinite(t) & np.isfinite(i_mA)
    if good.sum() < 2:
        return np.nan
    return float(np.trapz(i_mA[good], t[good]) / 3600.0)


def slope_safe(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.nanstd(x[good]) == 0:
        return np.nan
    return float(np.polyfit(x[good], y[good], 1)[0])


def voltage_binned_capacity_features(t: np.ndarray, v: np.ndarray, i: np.ndarray, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    good = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    if good.sum() < 5:
        return out
    order = np.argsort(t[good])
    tt = t[good][order]
    vv = v[good][order]
    ii = i[good][order]
    dt = np.diff(tt)
    if len(dt) == 0:
        return out
    vmid = 0.5 * (vv[:-1] + vv[1:])
    dq = 0.5 * (ii[:-1] + ii[1:]) * dt / 3600.0
    bins = np.linspace(3.0, 4.25, 26)
    hist, edges = np.histogram(vmid, bins=bins, weights=dq)
    abs_hist, _ = np.histogram(vmid, bins=bins, weights=np.abs(dq))
    centers = 0.5 * (edges[:-1] + edges[1:])
    total_abs = float(np.sum(abs_hist))
    if total_abs > 0:
        frac = abs_hist / total_abs
        out[f"{prefix}_dq_abs_total_mAh"] = total_abs
        out[f"{prefix}_dq_abs_lowV_frac"] = float(frac[centers < 3.55].sum())
        out[f"{prefix}_dq_abs_midV_frac"] = float(frac[(centers >= 3.55) & (centers < 3.95)].sum())
        out[f"{prefix}_dq_abs_highV_frac"] = float(frac[centers >= 3.95].sum())
        idx = int(np.argmax(abs_hist))
        out[f"{prefix}_dq_abs_peak_voltage"] = float(centers[idx])
        out[f"{prefix}_dq_abs_peak_frac"] = float(frac[idx])
        out[f"{prefix}_dq_abs_entropy"] = float(-np.sum(frac[frac > 0] * np.log(frac[frac > 0])))
    vspan = np.nanmax(vv) - np.nanmin(vv)
    out[f"{prefix}_dqdv_abs_integral_proxy"] = float(total_abs / vspan) if np.isfinite(vspan) and vspan > 0 else np.nan
    return out


def summarize_cycle(cyc: int, df: pd.DataFrame) -> Dict[str, Any]:
    d = df.copy()
    tcol = "Computer time (s)" if "Computer time (s)" in d.columns else "Time (s)"
    d[tcol] = pd.to_numeric(d[tcol], errors="coerce")
    d["Potential (V)"] = pd.to_numeric(d["Potential (V)"], errors="coerce")
    d["Current (mA)"] = pd.to_numeric(d["Current (mA)"], errors="coerce")
    d = d.dropna(subset=[tcol, "Potential (V)", "Current (mA)"]).sort_values(tcol)
    t = d[tcol].to_numpy(dtype=float)
    v = d["Potential (V)"].to_numpy(dtype=float)
    i = d["Current (mA)"].to_numpy(dtype=float)
    row: Dict[str, Any] = {"cycleNo": int(cyc), "echem_shape_points": int(len(d))}
    if len(d) < 5:
        return row
    tt = t - np.nanmin(t)
    row.update({
        "echem_shape_duration_s": float(np.nanmax(tt) - np.nanmin(tt)),
        "shape_V_min": float(np.nanmin(v)),
        "shape_V_max": float(np.nanmax(v)),
        "shape_V_range": float(np.nanmax(v) - np.nanmin(v)),
        "shape_V_mean": float(np.nanmean(v)),
        "shape_V_std": float(np.nanstd(v)),
        "shape_I_mean_mA": float(np.nanmean(i)),
        "shape_I_abs_mean_mA": float(np.nanmean(np.abs(i))),
        "shape_I_pos_fraction": float(np.mean(i > 0)),
        "shape_I_neg_fraction": float(np.mean(i < 0)),
        "shape_charge_mAh_signed": trapezoid_capacity_mAh(t, i),
        "shape_charge_mAh_abs": trapezoid_capacity_mAh(t, np.abs(i)),
        "shape_charge_mAh_pos": trapezoid_capacity_mAh(t, np.clip(i, 0, None)),
        "shape_charge_mAh_neg_abs": abs(trapezoid_capacity_mAh(t, np.clip(i, None, 0))),
        "shape_dVdt_slope": slope_safe(tt, v),
        "shape_dIdt_slope": slope_safe(tt, i),
    })
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        row[f"shape_V_q{int(q*100):02d}"] = float(np.nanquantile(v, q))
        row[f"shape_I_q{int(q*100):02d}"] = float(np.nanquantile(i, q))
    dv = np.diff(v)
    dt = np.diff(t)
    good_dt = np.isfinite(dt) & (dt > 0)
    if good_dt.any():
        dvdt = dv[good_dt] / dt[good_dt]
        row["shape_dVdt_abs_median"] = float(np.nanmedian(np.abs(dvdt)))
        row["shape_dVdt_abs_p95"] = float(np.nanquantile(np.abs(dvdt), 0.95))
        row["shape_dVdt_sign_consistency"] = float(abs(np.mean(np.sign(dvdt[np.isfinite(dvdt)])))) if np.isfinite(dvdt).any() else np.nan
    row.update(voltage_binned_capacity_features(t, v, i, "all"))
    pos = i > 0
    neg = i < 0
    if pos.sum() >= 5:
        row.update(voltage_binned_capacity_features(t[pos], v[pos], i[pos], "pos"))
    if neg.sum() >= 5:
        row.update(voltage_binned_capacity_features(t[neg], v[neg], i[neg], "neg"))
    if "Block" in d.columns and d["Block"].notna().any():
        row["shape_block_mode"] = str(d["Block"].astype(str).value_counts().index[0])
        row["shape_block_nunique"] = int(d["Block"].astype(str).nunique())
    return row


def scan_echem_shapes(csv_path: Path, cycles: Set[int], chunksize: int) -> pd.DataFrame:
    usecols = ["Computer time (s)", "Time (s)", "Potential (V)", "Current (mA)", "Block", "cycleNo"]
    rows = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=lambda c: c in usecols, encoding="utf-8-sig", low_memory=False):
        chunk["cycleNo"] = pd.to_numeric(chunk.get("cycleNo"), errors="coerce")
        chunk = chunk[chunk["cycleNo"].isin(cycles)].copy()
        if chunk.empty:
            continue
        rows.append(chunk)
    if not rows:
        return pd.DataFrame()
    all_rows = pd.concat(rows, ignore_index=True)
    out = [summarize_cycle(int(c), g) for c, g in all_rows.groupby("cycleNo", sort=True)]
    return pd.DataFrame(out).sort_values("cycleNo").reset_index(drop=True)


def numeric_cols(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    excluded = set(exclude)
    cols = []
    for c in df.columns:
        if c in excluded:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= 6 and x.nunique(dropna=True) >= 3:
            cols.append(c)
    return cols


def binary_tests(df: pd.DataFrame, features: List[str], label: str) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[label], errors="coerce")
    for feat in features:
        x = pd.to_numeric(df[feat], errors="coerce")
        pos = x[y.eq(1) & x.notna()].to_numpy(dtype=float)
        neg = x[y.eq(0) & x.notna()].to_numpy(dtype=float)
        if len(pos) >= 3 and len(neg) >= 3:
            _, p = mannwhitneyu(pos, neg, alternative="two-sided")
            rows.append({
                "label": label,
                "feature": feat,
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                "positive_median": float(np.median(pos)),
                "negative_median": float(np.median(neg)),
                "positive_minus_negative_median": float(np.median(pos) - np.median(neg)),
                "mannwhitney_p": float(p),
            })
    return pd.DataFrame(rows).sort_values(["mannwhitney_p", "feature"]).reset_index(drop=True) if rows else pd.DataFrame()


def correlation_tests(df: pd.DataFrame, features: List[str], targets: List[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        for target in targets:
            if target not in df.columns:
                continue
            sub = df[[feat, target]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) >= 7 and sub[feat].nunique() >= 3 and sub[target].nunique() >= 3:
                rho, p = spearmanr(sub[feat], sub[target])
                rows.append({"feature": feat, "target": target, "n": int(len(sub)), "rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values(["p_value", "feature", "target"]).reset_index(drop=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--repo-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/alek_jiho_nmc_deg")
    parser.add_argument("--echem-csv", default="")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/within_cycle_echem_shape_audit")
    parser.add_argument("--chunksize", type=int, default=750000)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    repo = Path(args.repo_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    echem_csv = Path(args.echem_csv) if args.echem_csv else repo / "echemDF_full" / "echemDF_full.csv"
    if not echem_csv.exists():
        raise FileNotFoundError(echem_csv)

    cycles = observed_cycles(derived)
    shape = scan_echem_shapes(echem_csv, cycles, args.chunksize)
    trace = read_csv(derived / "particle_trace_physics_audit" / "particle_trace_cycle_features.csv")
    mode = read_csv(derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv")
    kinetics = read_csv(derived / "phase_kinetics_avrami" / "phase_kinetics_avrami_roi_table.csv")
    front = read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")
    keep_kin = ["roi_id"] + [c for c in kinetics.columns if c not in mode.columns and c != "roi_id"]
    roi = mode.merge(kinetics[keep_kin], on="roi_id", how="left")
    front_keep = ["roi_id"] + [c for c in front.columns if c.endswith("_protocol_residual")]
    roi = roi.merge(front[front_keep], on="roi_id", how="left", suffixes=("", "_front"))
    roi = roi.merge(shape, on="cycleNo", how="left")
    cycle = trace.merge(shape, on="cycleNo", how="left")

    shape_features = numeric_cols(shape, ["cycleNo", "echem_shape_points"])
    event_tests = binary_tests(cycle, shape_features, "any_abrupt_drop") if "any_abrupt_drop" in cycle.columns else pd.DataFrame()
    sync_tests = binary_tests(cycle, shape_features, "synchronized_drop_2plus") if "synchronized_drop_2plus" in cycle.columns else pd.DataFrame()
    roi_bin = []
    for label in ["is_event_roi", "is_event_enriched_mode"]:
        if label in roi.columns:
            roi_bin.append(binary_tests(roi, shape_features, label))
    roi_binary = pd.concat([x for x in roi_bin if not x.empty], ignore_index=True) if roi_bin else pd.DataFrame()
    roi_corr = correlation_tests(roi, shape_features, [t for t in ROI_TARGETS if t in roi.columns])
    cycle_corr = correlation_tests(cycle, shape_features, ["mean_delta_prev", "mean_abs_delta_prev", "max_abs_delta_prev", "particle_norm_mean", "capacity_mAh", "coulombic_efficiency_pct"])

    paths = {
        "cycle_shape_features": out / "within_cycle_echem_shape_features.csv",
        "trace_joined": out / "within_cycle_echem_trace_joined.csv",
        "roi_joined": out / "within_cycle_echem_roi_joined.csv",
        "event_tests": out / "within_cycle_echem_event_tests.csv",
        "sync_tests": out / "within_cycle_echem_sync_tests.csv",
        "roi_binary_tests": out / "within_cycle_echem_roi_binary_tests.csv",
        "roi_correlations": out / "within_cycle_echem_roi_correlations.csv",
        "cycle_correlations": out / "within_cycle_echem_cycle_correlations.csv",
        "summary": out / "within_cycle_echem_shape_audit_summary.json",
    }
    shape.to_csv(paths["cycle_shape_features"], index=False)
    cycle.to_csv(paths["trace_joined"], index=False)
    roi.to_csv(paths["roi_joined"], index=False)
    event_tests.to_csv(paths["event_tests"], index=False)
    sync_tests.to_csv(paths["sync_tests"], index=False)
    roi_binary.to_csv(paths["roi_binary_tests"], index=False)
    roi_corr.to_csv(paths["roi_correlations"], index=False)
    cycle_corr.to_csv(paths["cycle_correlations"], index=False)

    summary = {
        "n_observed_cycle_requests": int(len(cycles)),
        "n_echem_shape_cycles": int(len(shape)),
        "n_shape_features": int(len(shape_features)),
        "n_roi_rows": int(len(roi)),
        "top_event_shape_tests": event_tests.head(20).to_dict("records") if not event_tests.empty else [],
        "top_sync_shape_tests": sync_tests.head(20).to_dict("records") if not sync_tests.empty else [],
        "top_roi_binary_shape_tests": roi_binary.head(20).to_dict("records") if not roi_binary.empty else [],
        "top_roi_shape_correlations": roi_corr.head(20).to_dict("records") if not roi_corr.empty else [],
        "top_cycle_shape_correlations": cycle_corr.head(20).to_dict("records") if not cycle_corr.empty else [],
        "guardrail": "Within-cycle echem shape features are computed from raw time/potential/current rows for observed particle/ROI cycles. dQ/dV terms are proxy descriptors from current-time integration over voltage bins, not calibrated electrochemical capacity analysis.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Within-Cycle Echem Shape Audit",
        "",
        "Voltage/current trajectory and dQ/dV-proxy descriptors joined to particle-trace and ROI optical physics outputs.",
        "",
        f"- Observed cycles requested: {summary['n_observed_cycle_requests']}",
        f"- Echem shape cycles found: {summary['n_echem_shape_cycles']}",
        f"- Shape features: {summary['n_shape_features']}",
        f"- ROI rows joined: {summary['n_roi_rows']}",
        "",
        "## Top ROI Shape Correlations",
    ]
    for row in summary["top_roi_shape_correlations"][:10]:
        lines.append(f"- {row.get('feature')} vs {row.get('target')}: rho={row.get('rho'):.3g}, p={row.get('p_value'):.3g}, n={row.get('n')}")
    lines += ["", "## Top Cycle Shape Correlations"]
    for row in summary["top_cycle_shape_correlations"][:10]:
        lines.append(f"- {row.get('feature')} vs {row.get('target')}: rho={row.get('rho'):.3g}, p={row.get('p_value'):.3g}, n={row.get('n')}")
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json({
        "out_dir": str(out),
        "n_echem_shape_cycles": summary["n_echem_shape_cycles"],
        "top_roi_shape_correlations": summary["top_roi_shape_correlations"][:5],
        "top_cycle_shape_correlations": summary["top_cycle_shape_correlations"][:5],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
