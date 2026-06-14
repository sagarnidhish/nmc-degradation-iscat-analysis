#!/usr/bin/env python3
"""Phase-transformation kinetics from cropped NMC particle ROI movies.

This audit treats each particle ROI sequence as a small phase-fraction movie and
fits simple kinetic summaries: threshold phase-fraction trajectories, maximum
front-rate timing, logistic transition sharpness, and Avrami-style transformed
fraction exponents. The goal is not calibrated electrochemical kinetics; it is
to separate gradual, abrupt, reversible, and weak-transition ROI behaviors under
the same particle-region-only guardrails as the rollout analyses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, spearmanr


THRESHOLD_QS = (0.60, 0.70, 0.80)
EVENT_MODE = "optical_brightening_decorrelating_rollout_hard_front_positive"


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


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if len(x) < window:
        return x.astype(float)
    pad = window // 2
    xp = np.pad(x.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(xp, kernel, mode="valid")


def logistic(t: np.ndarray, low: float, amp: float, k: float, t0: float) -> np.ndarray:
    z = np.clip(-k * (t - t0), -60, 60)
    return low + amp / (1.0 + np.exp(z))


def fit_logistic(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    out = {
        "logistic_low": np.nan,
        "logistic_amp": np.nan,
        "logistic_k_per_s": np.nan,
        "logistic_t0_s": np.nan,
        "logistic_r2": np.nan,
    }
    if len(t) < 8 or np.nanmax(y) - np.nanmin(y) < 1e-4:
        return out
    low0 = float(np.nanmin(y))
    amp0 = float(np.nanmax(y) - np.nanmin(y))
    dy = y[-1] - y[0]
    k0 = (1.0 if dy >= 0 else -1.0) / max(float(t[-1] - t[0]), 1.0)
    t00 = float(t[np.nanargmax(np.abs(np.gradient(moving_average(y), t)))])
    try:
        popt, _ = curve_fit(
            logistic,
            t,
            y,
            p0=[low0, amp0 if dy >= 0 else -amp0, k0, t00],
            maxfev=10000,
        )
        pred = logistic(t, *popt)
        ss_res = float(np.nansum((y - pred) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
        out.update({
            "logistic_low": float(popt[0]),
            "logistic_amp": float(popt[1]),
            "logistic_k_per_s": float(popt[2]),
            "logistic_t0_s": float(popt[3]),
            "logistic_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        })
    except Exception:
        pass
    return out


def fit_avrami(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    out = {
        "avrami_direction": "flat",
        "avrami_n": np.nan,
        "avrami_logk": np.nan,
        "avrami_r2": np.nan,
        "transformed_fraction_delta": np.nan,
    }
    if len(t) < 8:
        return out
    delta = float(y[-1] - y[0])
    amp = float(np.nanmax(y) - np.nanmin(y))
    out["transformed_fraction_delta"] = delta
    if amp < 1e-4:
        return out
    direction = "growth" if delta >= 0 else "decay"
    if direction == "growth":
        x = (y - np.nanmin(y)) / amp
    else:
        x = (np.nanmax(y) - y) / amp
    valid = (x > 0.03) & (x < 0.97) & np.isfinite(x) & (t > 0)
    if int(valid.sum()) < 6:
        out["avrami_direction"] = direction
        return out
    xv = np.clip(x[valid], 1e-6, 1 - 1e-6)
    tv = t[valid]
    lhs = np.log(-np.log(1.0 - xv))
    rhs = np.log(np.maximum(tv - float(t[0]) + 1e-6, 1e-6))
    if np.nanstd(rhs) == 0 or np.nanstd(lhs) == 0:
        out["avrami_direction"] = direction
        return out
    n, logk = np.polyfit(rhs, lhs, 1)
    pred = n * rhs + logk
    ss_res = float(np.nansum((lhs - pred) ** 2))
    ss_tot = float(np.nansum((lhs - np.nanmean(lhs)) ** 2))
    out.update({
        "avrami_direction": direction,
        "avrami_n": float(n),
        "avrami_logk": float(logk),
        "avrami_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    })
    return out


def threshold_kinetics(frames: np.ndarray, t: np.ndarray, q: float) -> Dict[str, float]:
    first = frames[0]
    thresh = float(np.nanquantile(first, q))
    frac = (frames >= thresh).mean(axis=(1, 2)).astype(float)
    smooth = moving_average(frac, window=5)
    rate = np.gradient(smooth, t) if len(t) > 2 else np.full_like(smooth, np.nan)
    abs_rate = np.abs(rate)
    imax = int(np.nanargmax(abs_rate)) if np.isfinite(abs_rate).any() else 0
    net = float(smooth[-1] - smooth[0])
    total_var = float(np.nansum(np.abs(np.diff(smooth))))
    prefix = f"q{int(round(q * 100))}"
    out: Dict[str, float] = {
        f"{prefix}_threshold": thresh,
        f"{prefix}_fraction_first": float(frac[0]),
        f"{prefix}_fraction_last": float(frac[-1]),
        f"{prefix}_fraction_delta": net,
        f"{prefix}_fraction_total_variation": total_var,
        f"{prefix}_variation_to_net_abs": float(total_var / max(abs(net), 1e-9)),
        f"{prefix}_max_abs_rate_per_s": float(abs_rate[imax]) if np.isfinite(abs_rate[imax]) else np.nan,
        f"{prefix}_signed_rate_at_max_abs_per_s": float(rate[imax]) if np.isfinite(rate[imax]) else np.nan,
        f"{prefix}_time_of_max_abs_rate_s": float(t[imax]),
        f"{prefix}_time_of_max_abs_rate_frac": float((t[imax] - t[0]) / max(t[-1] - t[0], 1e-9)),
        f"{prefix}_positive_rate_fraction": float((rate > 0).mean()),
        f"{prefix}_negative_rate_fraction": float((rate < 0).mean()),
    }
    for k, v in fit_logistic(t, smooth).items():
        out[f"{prefix}_{k}"] = v
    for k, v in fit_avrami(t, smooth).items():
        out[f"{prefix}_{k}"] = v
    return out


def sequence_kinetics(row: pd.Series) -> Dict[str, Any]:
    npz_path = Path(str(row["npz_path"]))
    z = np.load(npz_path)
    frames = np.asarray(z["frames_norm"], dtype=np.float32)
    n = int(frames.shape[0])
    duration = float(row.get("duration_s", np.nan))
    if not np.isfinite(duration) or duration <= 0:
        duration = float(n - 1)
    t = np.linspace(0.0, duration, n)
    out: Dict[str, Any] = {
        "roi_id": row["roi_id"],
        "cycleNo": row.get("cycleNo"),
        "cohort_role": row.get("cohort_role"),
        "event_reference_cycle": row.get("event_reference_cycle"),
        "mode_label": row.get("mode_label"),
        "is_event_roi": int(row.get("cohort_role") == "event"),
        "is_event_enriched_mode": int(row.get("mode_label") == EVENT_MODE),
        "n_frames": n,
        "duration_s": duration,
    }
    roi_norm = np.asarray(z["roi_norm_mean"], dtype=float) if "roi_norm_mean" in z.files else frames.mean(axis=(1, 2))
    roi_smooth = moving_average(roi_norm, 5)
    roi_rate = np.gradient(roi_smooth, t)
    out.update({
        "roi_norm_delta": float(roi_smooth[-1] - roi_smooth[0]),
        "roi_norm_total_variation": float(np.nansum(np.abs(np.diff(roi_smooth)))),
        "roi_norm_max_abs_rate_per_s": float(np.nanmax(np.abs(roi_rate))),
        "roi_norm_rate_sign_consistency": float(max((roi_rate > 0).mean(), (roi_rate < 0).mean())),
    })
    for q in THRESHOLD_QS:
        out.update(threshold_kinetics(frames, t, q))
    return out


def compare_groups(df: pd.DataFrame, features: Iterable[str], group_col: str, positive_label: Any, label: str) -> pd.DataFrame:
    rows = []
    group = df[group_col]
    for feat in features:
        if feat not in df.columns:
            continue
        vals = to_num(df[feat])
        pos = vals[group.eq(positive_label)].dropna().to_numpy(dtype=float)
        neg = vals[~group.eq(positive_label)].dropna().to_numpy(dtype=float)
        if len(pos) < 2 or len(neg) < 2:
            continue
        _, p = mannwhitneyu(pos, neg, alternative="two-sided")
        rows.append({
            "comparison": label,
            "feature": feat,
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "positive_median": float(np.nanmedian(pos)),
            "negative_median": float(np.nanmedian(neg)),
            "median_positive_minus_negative": float(np.nanmedian(pos) - np.nanmedian(neg)),
            "mannwhitney_p": float(p),
        })
    return pd.DataFrame(rows)


def correlations(df: pd.DataFrame, features: Iterable[str], targets: Iterable[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        x = to_num(df[feat]) if feat in df.columns else pd.Series(dtype=float)
        for target in targets:
            if target not in df.columns:
                continue
            y = to_num(df[target])
            valid = x.notna() & y.notna()
            if int(valid.sum()) < 8 or x[valid].nunique() < 3 or y[valid].nunique() < 3:
                continue
            stat = spearmanr(x[valid], y[valid])
            rows.append({
                "feature": feat,
                "target": target,
                "n": int(valid.sum()),
                "spearman_rho": float(stat.statistic) if np.isfinite(stat.statistic) else np.nan,
                "spearman_p": float(stat.pvalue) if np.isfinite(stat.pvalue) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/phase_kinetics_avrami")
    args = parser.parse_args()
    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    echem = pd.read_csv(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_joined.csv")
    modes = pd.read_csv(derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv")
    front = pd.read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")
    df = echem.merge(modes[["roi_id", "mode_label", "mode_review_priority"]], on="roi_id", how="left")
    keep_front = [
        "roi_id",
        "phase_slope_positive_fraction_protocol_residual",
        "threshold_robust_phase_score_protocol_residual",
        "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    ]
    df = df.merge(front[[c for c in keep_front if c in front.columns]], on="roi_id", how="left")
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        if isinstance(row.get("npz_path"), str) and Path(row["npz_path"]).exists():
            rows.append(sequence_kinetics(row))
    kinetics = pd.DataFrame(rows)
    merged = kinetics.merge(df.drop(columns=[c for c in kinetics.columns if c in df.columns and c != "roi_id"], errors="ignore"), on="roi_id", how="left")

    generated_cols = set(kinetics.columns)
    kinetic_features = [
        c for c in generated_cols
        if any(token in c for token in ["fraction_delta", "fraction_total_variation", "variation_to_net", "max_abs_rate", "time_of_max_abs_rate_frac", "positive_rate_fraction", "logistic_k", "logistic_r2", "avrami_n", "avrami_r2", "roi_norm_"])
        and to_num(merged[c]).notna().sum() >= 8
    ]
    event_tests = compare_groups(merged, kinetic_features, "cohort_role", "event", "event_vs_control")
    mode_tests = compare_groups(merged, kinetic_features, "is_event_enriched_mode", 1, "event_enriched_mode_vs_other")
    corr = correlations(merged, kinetic_features, [
        "phase_slope_positive_fraction_protocol_residual",
        "threshold_robust_phase_score_protocol_residual",
        "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
        "mode_review_priority",
        "n_frames_percentile",
        "V_mean",
    ])
    tests = pd.concat([event_tests, mode_tests], ignore_index=True) if not event_tests.empty or not mode_tests.empty else pd.DataFrame()
    if not tests.empty:
        tests = tests.sort_values(["mannwhitney_p", "comparison", "feature"], na_position="last")
    if not corr.empty:
        corr = corr.sort_values(["spearman_p", "feature"], na_position="last")

    kinetics_path = out / "phase_kinetics_avrami_roi_table.csv"
    tests_path = out / "phase_kinetics_avrami_group_tests.csv"
    corr_path = out / "phase_kinetics_avrami_correlations.csv"
    kinetics.to_csv(kinetics_path, index=False)
    tests.to_csv(tests_path, index=False)
    corr.to_csv(corr_path, index=False)
    summary = {
        "n_roi": int(len(kinetics)),
        "n_kinetic_features": int(len(kinetic_features)),
        "threshold_quantiles": list(THRESHOLD_QS),
        "top_group_tests": tests.head(20).to_dict("records") if not tests.empty else [],
        "top_correlations": corr.head(20).to_dict("records") if not corr.empty else [],
        "guardrail": "Kinetic fits are optical phase-fraction proxies from cropped particle ROIs using provisional timing. Avrami/logistic parameters are descriptive and not calibrated reaction constants.",
        "outputs": {
            "roi_table": str(kinetics_path),
            "group_tests": str(tests_path),
            "correlations": str(corr_path),
            "summary": str(out / "phase_kinetics_avrami_summary.json"),
        },
    }
    with (out / "phase_kinetics_avrami_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Phase Kinetics Avrami Audit",
        "",
        "Optical phase-fraction kinetic summaries from cropped NMC particle ROI movies.",
        "",
        f"- ROI rows: {summary['n_roi']}",
        f"- Kinetic features: {summary['n_kinetic_features']}",
        "",
        "## Top Group Tests",
    ]
    for row in summary["top_group_tests"][:8]:
        lines.append(
            f"- {row.get('comparison')} {row.get('feature')}: median diff={row.get('median_positive_minus_negative'):.3g}, p={row.get('mannwhitney_p'):.4g}"
        )
    lines += ["", "## Top Correlations"]
    for row in summary["top_correlations"][:8]:
        lines.append(
            f"- {row.get('feature')} vs {row.get('target')}: rho={row.get('spearman_rho'):.3f}, p={row.get('spearman_p'):.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out), "n_roi": summary["n_roi"], "top_group_tests": summary["top_group_tests"][:3]}, indent=2))


if __name__ == "__main__":
    main()
