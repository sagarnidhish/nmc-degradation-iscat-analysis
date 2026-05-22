#!/usr/bin/env python3
"""Source-balanced pre-event optical phase-kinetics audit.

This audit complements the front/kymograph visual QC layers with a stricter
particle-region-only kinetic readout. It loads the source-balanced pre-event ROI
tensors, builds a stable particle mask from early frames, falls back to that
prior mask when per-frame threshold masks are unstable, and measures masked
phase-fraction trajectories, logistic/Avrami-style timing, and source/echem
associations against event-relative bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


THRESHOLD_QS = (0.55, 0.65, 0.75)
PRE_BINS = {"near_pre_event_1_8", "mid_pre_event_9_16", "far_pre_event_17_32"}
CONTROL_BINS = {"post_event_1_16", "no_near_event_control"}
TARGETS = {
    "near_vs_any_non_near": ("near_pre_event_1_8", None),
    "near_vs_mid_pre": ("near_pre_event_1_8", "mid_pre_event_9_16"),
    "near_vs_far_pre": ("near_pre_event_1_8", "far_pre_event_17_32"),
    "near_vs_post_control": ("near_pre_event_1_8", CONTROL_BINS),
    "clean_pre_1_16_vs_post_control": ({"near_pre_event_1_8", "mid_pre_event_9_16"}, CONTROL_BINS),
}
ECHEM_CONTEXT_COLS = [
    "capacity_fraction_of_first",
    "coulombic_inefficiency_pct",
    "echem_regime_pc1",
    "echem_regime_pc2",
    "shape_V_mean",
    "shape_I_abs_mean_mA",
    "all_dq_abs_entropy",
]


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).loc[:, lambda d: ~d.columns.duplicated()].copy()


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def load_frames(path: str) -> np.ndarray | None:
    try:
        z = np.load(path)
        frames = z["frames_norm"] if "frames_norm" in z.files else z["frames"]
    except Exception:
        return None
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3 or frames.shape[0] < 8:
        return None
    return frames


def stable_particle_mask(frames: np.ndarray) -> np.ndarray:
    base = np.nanmedian(frames[: min(7, len(frames))], axis=0)
    high = base >= np.nanpercentile(base, 70)
    low = base <= np.nanpercentile(base, 30)
    mask = high if high.mean() < 0.55 else low
    if mask.mean() < 0.02 or mask.mean() > 0.75:
        yy, xx = np.indices(base.shape)
        cy, cx = (np.array(base.shape) - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = rr <= 0.36 * min(base.shape)
    return mask.astype(bool)


def frame_particle_mask(frame: np.ndarray, prior: np.ndarray) -> np.ndarray:
    high = frame >= np.nanpercentile(frame, 70)
    low = frame <= np.nanpercentile(frame, 30)
    cand = high if high.mean() < 0.55 else low
    if cand.mean() < 0.02 or cand.mean() > 0.75:
        return prior.copy()
    overlap = (cand & prior).sum() / max(prior.sum(), 1)
    if overlap < 0.25:
        return prior.copy()
    return cand.astype(bool)


def centroid(mask: np.ndarray) -> Tuple[float, float]:
    yy, xx = np.indices(mask.shape)
    if not mask.any():
        return np.nan, np.nan
    return float(yy[mask].mean()), float(xx[mask].mean())


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
    out = {"logistic_amp": np.nan, "logistic_k": np.nan, "logistic_t0": np.nan, "logistic_r2": np.nan}
    if len(t) < 8 or np.nanmax(y) - np.nanmin(y) < 1e-5:
        return out
    low0 = float(np.nanmin(y))
    amp0 = float(np.nanmax(y) - np.nanmin(y))
    direction = 1.0 if y[-1] >= y[0] else -1.0
    t00 = float(t[int(np.nanargmax(np.abs(np.gradient(moving_average(y), t))))])
    try:
        popt, _ = curve_fit(logistic, t, y, p0=[low0, direction * amp0, direction, t00], maxfev=10000)
        pred = logistic(t, *popt)
        ss_res = float(np.nansum((y - pred) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
        out.update({
            "logistic_amp": float(popt[1]),
            "logistic_k": float(popt[2]),
            "logistic_t0": float(popt[3]),
            "logistic_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        })
    except Exception:
        pass
    return out


def fit_avrami(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    out = {"avrami_n": np.nan, "avrami_logk": np.nan, "avrami_r2": np.nan, "avrami_direction": "flat"}
    amp = float(np.nanmax(y) - np.nanmin(y))
    if len(t) < 8 or amp < 1e-5:
        return out
    direction = "growth" if y[-1] >= y[0] else "decay"
    x = (y - np.nanmin(y)) / amp if direction == "growth" else (np.nanmax(y) - y) / amp
    valid = np.isfinite(x) & (x > 0.03) & (x < 0.97) & (t > 0)
    if valid.sum() < 6:
        out["avrami_direction"] = direction
        return out
    lhs = np.log(-np.log(1.0 - np.clip(x[valid], 1e-6, 1 - 1e-6)))
    rhs = np.log(np.maximum(t[valid] - float(t[0]) + 1e-6, 1e-6))
    if np.nanstd(lhs) <= 0 or np.nanstd(rhs) <= 0:
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


def slope_r2(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(y) < 6 or np.nanstd(y) <= 1e-12:
        return np.nan, np.nan
    coef = np.polyfit(t, y, 1)
    pred = np.polyval(coef, t)
    ss_res = float(np.nansum((y - pred) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    return float(coef[0]), float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def kinetic_features(row: pd.Series) -> Dict[str, Any]:
    frames = load_frames(str(row.get("npz_path", "")))
    rec = row.to_dict()
    if frames is None:
        rec["kinetics_status"] = "missing_or_invalid_npz"
        return rec
    prior = stable_particle_mask(frames)
    masks = np.asarray([frame_particle_mask(frame, prior) for frame in frames], dtype=bool)
    cents = np.asarray([centroid(mask) for mask in masks], dtype=float)
    good = np.isfinite(cents).all(axis=1)
    steps = np.sqrt(np.sum(np.diff(cents[good], axis=0) ** 2, axis=1)) if good.sum() >= 2 else np.asarray([])
    t = np.linspace(0.0, 1.0, frames.shape[0])
    prior_vals = frames[:, prior]
    masked_mean = np.asarray([float(np.nanmean(frame[mask])) if mask.any() else np.nan for frame, mask in zip(frames, masks)])
    outside = np.asarray([float(np.nanmean(frame[~prior])) if (~prior).any() else np.nan for frame in frames])
    masked_bg = masked_mean - outside
    rec.update({
        "kinetics_status": "ok",
        "stable_particle_mask_fraction": float(prior.mean()),
        "frame_mask_fraction_median": float(np.nanmedian(masks.mean(axis=(1, 2)))),
        "frame_mask_fraction_iqr": float(np.nanpercentile(masks.mean(axis=(1, 2)), 75) - np.nanpercentile(masks.mean(axis=(1, 2)), 25)),
        "particle_centroid_path_px": float(np.nansum(steps)) if len(steps) else np.nan,
        "particle_centroid_max_step_px": float(np.nanmax(steps)) if len(steps) else np.nan,
        "masked_mean_delta": float(masked_mean[-1] - masked_mean[0]),
        "masked_mean_total_variation": float(np.nansum(np.abs(np.diff(moving_average(masked_mean))))),
        "masked_minus_bg_delta": float(masked_bg[-1] - masked_bg[0]),
    })
    slope, r2 = slope_r2(t, moving_average(masked_bg))
    rec["masked_minus_bg_slope"] = slope
    rec["masked_minus_bg_slope_r2"] = r2
    for q in THRESHOLD_QS:
        prefix = f"q{int(round(100 * q))}"
        thresh = float(np.nanquantile(prior_vals[: min(7, len(prior_vals))].ravel(), q))
        frac = np.asarray([float((frame[prior] >= thresh).mean()) for frame in frames])
        smooth = moving_average(frac)
        rate = np.gradient(smooth, t)
        imax = int(np.nanargmax(np.abs(rate))) if np.isfinite(rate).any() else 0
        slope, r2 = slope_r2(t, smooth)
        rec.update({
            f"{prefix}_phase_fraction_first": float(frac[0]),
            f"{prefix}_phase_fraction_last": float(frac[-1]),
            f"{prefix}_phase_fraction_delta": float(smooth[-1] - smooth[0]),
            f"{prefix}_phase_fraction_slope": slope,
            f"{prefix}_phase_fraction_slope_r2": r2,
            f"{prefix}_phase_fraction_total_variation": float(np.nansum(np.abs(np.diff(smooth)))),
            f"{prefix}_phase_variation_to_net": float(np.nansum(np.abs(np.diff(smooth))) / max(abs(smooth[-1] - smooth[0]), 1e-9)),
            f"{prefix}_max_abs_rate": float(np.nanmax(np.abs(rate))),
            f"{prefix}_signed_rate_at_max_abs": float(rate[imax]),
            f"{prefix}_time_of_max_abs_rate_frac": float(t[imax]),
            f"{prefix}_positive_rate_fraction": float((rate > 0).mean()),
            f"{prefix}_negative_rate_fraction": float((rate < 0).mean()),
        })
        for k, v in fit_logistic(t, smooth).items():
            rec[f"{prefix}_{k}"] = v
        for k, v in fit_avrami(t, smooth).items():
            rec[f"{prefix}_{k}"] = v
    return rec


def source_residual(values: pd.Series, sources: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    return x - x.groupby(sources.astype(str)).transform("mean")


def target_series(df: pd.DataFrame, positive: Any, negative: Any) -> pd.Series:
    bins = df["event_relative_bin"].astype(str)
    pos_set = positive if isinstance(positive, set) else {positive}
    y = pd.Series(np.nan, index=df.index)
    y.loc[bins.isin(pos_set)] = 1
    if negative is None:
        y.loc[~bins.isin(pos_set)] = 0
    else:
        neg_set = negative if isinstance(negative, set) else {negative}
        y.loc[bins.isin(neg_set)] = 0
    return y


def source_eta2(values: pd.Series, sources: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce")
    valid = x.notna() & sources.notna()
    x = x[valid]
    src = sources[valid].astype(str)
    if len(x) < 4 or x.nunique() < 2 or src.nunique() < 2:
        return np.nan
    total = float(((x - x.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    between = sum(len(sub) * float((sub.mean() - x.mean()) ** 2) for _, sub in x.groupby(src))
    return float(between / total)


def event_tests(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = df["source_stem"].astype(str)
    for target, (positive, negative) in TARGETS.items():
        y = target_series(df, positive, negative)
        for feature in features:
            for transform in ["raw", "source_residual"]:
                x = numeric(df, feature)
                if transform == "source_residual":
                    x = source_residual(x, sources)
                valid = y.isin([0, 1]) & x.notna()
                yy = y[valid].astype(int)
                xx = x[valid]
                if len(yy) < 8 or yy.nunique() < 2 or xx.nunique() < 2:
                    continue
                direction = "higher_in_positive" if xx[yy == 1].median() >= xx[yy == 0].median() else "lower_in_positive"
                score = xx if direction == "higher_in_positive" else -xx
                try:
                    _, p_mwu = mannwhitneyu(xx[yy == 1], xx[yy == 0], alternative="two-sided")
                except ValueError:
                    p_mwu = np.nan
                rows.append({
                    "target": target,
                    "feature": feature,
                    "transform": transform,
                    "n": int(len(yy)),
                    "n_positive": int(yy.sum()),
                    "direction": direction,
                    "oriented_auc": float(roc_auc_score(yy, score)),
                    "average_precision": float(average_precision_score(yy, score)),
                    "median_positive": float(xx[yy == 1].median()),
                    "median_negative": float(xx[yy == 0].median()),
                    "median_positive_minus_negative": float(xx[yy == 1].median() - xx[yy == 0].median()),
                    "mwu_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
                    "source_eta2_after_transform": source_eta2(x, sources),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mwu_p", "oriented_auc", "average_precision"], ascending=[True, False, False])
    return out


def matched_tests(df: pd.DataFrame, features: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    near = df[df["event_relative_bin"].astype(str).eq("near_pre_event_1_8")].copy()
    pairs: List[Dict[str, Any]] = []
    pair_id = 0
    for control_bin in ["mid_pre_event_9_16", "far_pre_event_17_32", "post_event_1_16", "no_near_event_control"]:
        controls_all = df[df["event_relative_bin"].astype(str).eq(control_bin)].copy()
        for _, row in near.iterrows():
            controls = controls_all[controls_all["source_stem"].astype(str).eq(str(row.get("source_stem")))].copy()
            scheme = "same_source"
            if controls.empty:
                controls = controls_all.copy()
                scheme = "global_echem_context"
            if controls.empty:
                continue
            dist = (numeric(controls, "cycleNo") - float(row.get("cycleNo", np.nan))).abs().fillna(9999.0)
            for col in ECHEM_CONTEXT_COLS:
                if col in df.columns and pd.notna(row.get(col)):
                    vals = numeric(controls, col)
                    scale = float(vals.std()) if np.isfinite(vals.std()) and vals.std() > 1e-12 else 1.0
                    dist = dist + 0.25 * ((vals - float(row[col])).abs() / scale).fillna(2.0)
            ctrl = controls.loc[dist.sort_values().index[0]]
            pair_id += 1
            base = {
                "pair_id": pair_id,
                "match_scheme": scheme,
                "control_bin": control_bin,
                "near_roi_id": row.get("roi_id"),
                "control_roi_id": ctrl.get("roi_id"),
                "near_source_stem": row.get("source_stem"),
                "control_source_stem": ctrl.get("source_stem"),
                "near_cycleNo": row.get("cycleNo"),
                "control_cycleNo": ctrl.get("cycleNo"),
            }
            for feature in features:
                base[f"near__{feature}"] = row.get(feature)
                base[f"control__{feature}"] = ctrl.get(feature)
                base[f"diff__{feature}"] = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([ctrl.get(feature)]), errors="coerce").iloc[0]
            pairs.append(base)
    pair_df = pd.DataFrame(pairs)
    rows: List[Dict[str, Any]] = []
    for (scheme, control_bin), sub in pair_df.groupby(["match_scheme", "control_bin"], dropna=False):
        for feature in features:
            vals = pd.to_numeric(sub.get(f"diff__{feature}"), errors="coerce").dropna()
            if len(vals) < 4:
                continue
            n_pos = int((vals > 0).sum())
            n_neg = int((vals < 0).sum())
            # Two-sided sign-flip permutation on mean absolute signed differences.
            rng = np.random.default_rng(12345)
            signs = rng.choice([-1, 1], size=(2000, len(vals)))
            null = np.abs(signs @ vals.to_numpy(dtype=float) / len(vals))
            obs = abs(float(vals.mean()))
            p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
            rows.append({
                "match_scheme": scheme,
                "control_bin": control_bin,
                "feature": feature,
                "n_pairs": int(len(vals)),
                "median_near_minus_control": float(vals.median()),
                "mean_near_minus_control": float(vals.mean()),
                "positive_fraction": float(n_pos / max(n_pos + n_neg, 1)),
                "signflip_mean_abs_p": p,
            })
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests = tests.sort_values(["signflip_mean_abs_p", "n_pairs"], ascending=[True, False])
    return pair_df, tests


def correlations(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    targets = ["cycles_to_next_event", "capacity_fraction_of_first", "shape_V_mean", "echem_regime_pc1", "front_consensus_score", "visual_sanity_score", "visual_review_score"]
    rows = []
    for feature in features:
        x = numeric(df, feature)
        for target in targets:
            if target not in df.columns:
                continue
            y = numeric(df, target)
            valid = x.notna() & y.notna()
            if valid.sum() < 8 or x[valid].nunique() < 3 or y[valid].nunique() < 3:
                continue
            stat = spearmanr(x[valid], y[valid])
            rows.append({
                "feature": feature,
                "target": target,
                "n": int(valid.sum()),
                "spearman_rho": float(stat.statistic) if np.isfinite(stat.statistic) else np.nan,
                "spearman_p": float(stat.pvalue) if np.isfinite(stat.pvalue) else np.nan,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["spearman_p", "feature"], na_position="last")
    return out


def top_items(df: pd.DataFrame, n: int = 12) -> List[Dict[str, Any]]:
    return df.head(n).to_dict("records") if not df.empty else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_phase_kinetics_audit")
    args = parser.parse_args()
    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = read_csv(derived / "source_balanced_pre_event_roi_sequences" / "selected_roi_sequence_manifest.csv")
    echem_front = read_csv(derived / "source_balanced_pre_event_echem_front_coupling_audit" / "source_balanced_pre_event_echem_front_joined.csv")
    consensus = read_csv(derived / "source_balanced_pre_event_consensus_review_queue" / "source_balanced_pre_event_consensus_review_queue.csv")
    visual_sanity = read_csv(derived / "source_balanced_pre_event_visual_sanity_audit" / "source_balanced_pre_event_visual_sanity_metrics.csv")
    visual_qc = read_csv(derived / "source_balanced_pre_event_visual_qc_modes" / "source_balanced_pre_event_visual_qc_modes.csv")

    keep_echem = ["roi_id"] + [c for c in echem_front.columns if c in ECHEM_CONTEXT_COLS or c.endswith("_source_echem_context_residual")]
    df = manifest.merge(echem_front[keep_echem], on="roi_id", how="left")
    df = df.merge(consensus[["roi_id", "consensus_review_rank", "consensus_review_score", "matched_positive_support_count"]], on="roi_id", how="left")
    df = df.merge(visual_sanity[["roi_id", "visual_sanity_score", "visual_sanity_flag"]], on="roi_id", how="left")
    visual_qc_keep = [
        c for c in [
            "roi_id",
            "visual_review_score",
            "visual_front_plausibility_score",
            "visual_qc_tier",
            "visual_mode",
            "visual_mode_name",
            "visual_mode_cluster",
        ]
        if c in visual_qc.columns
    ]
    df = df.merge(visual_qc[visual_qc_keep], on="roi_id", how="left")

    rows = [kinetic_features(row) for _, row in df.iterrows()]
    kinetics = pd.DataFrame(rows)
    feature_cols = [
        c for c in kinetics.columns
        if any(token in c for token in ["masked_", "phase_fraction", "max_abs_rate", "logistic_", "avrami_", "particle_centroid"])
        and "variation_to_net" not in c
        and numeric(kinetics, c).notna().sum() >= 16
        and numeric(kinetics, c).nunique(dropna=True) >= 2
    ]
    tests = event_tests(kinetics[kinetics["kinetics_status"].eq("ok")], feature_cols)
    pair_df, matched = matched_tests(kinetics[kinetics["kinetics_status"].eq("ok")], feature_cols)
    corr = correlations(kinetics[kinetics["kinetics_status"].eq("ok")], feature_cols)
    source_summary = (
        kinetics[kinetics["kinetics_status"].eq("ok")]
        .groupby("source_stem", dropna=False)
        .agg(
            n_roi=("roi_id", "count"),
            n_cycles=("cycleNo", pd.Series.nunique),
            near_pre=("event_relative_bin", lambda s: int((s.astype(str) == "near_pre_event_1_8").sum())),
            median_q65_delta=("q65_phase_fraction_delta", "median"),
            median_masked_bg_slope=("masked_minus_bg_slope", "median"),
            median_visual_sanity=("visual_sanity_score", "median"),
        )
        .reset_index()
        .sort_values(["n_roi", "near_pre"], ascending=False)
    )

    paths = {
        "features": out / "source_balanced_pre_event_phase_kinetics_features.csv",
        "event_tests": out / "source_balanced_pre_event_phase_kinetics_event_tests.csv",
        "matched_pairs": out / "source_balanced_pre_event_phase_kinetics_matched_pairs.csv",
        "matched_tests": out / "source_balanced_pre_event_phase_kinetics_matched_tests.csv",
        "correlations": out / "source_balanced_pre_event_phase_kinetics_correlations.csv",
        "source_summary": out / "source_balanced_pre_event_phase_kinetics_source_summary.csv",
        "summary": out / "source_balanced_pre_event_phase_kinetics_summary.json",
    }
    kinetics.to_csv(paths["features"], index=False)
    tests.to_csv(paths["event_tests"], index=False)
    pair_df.to_csv(paths["matched_pairs"], index=False)
    matched.to_csv(paths["matched_tests"], index=False)
    corr.to_csv(paths["correlations"], index=False)
    source_summary.to_csv(paths["source_summary"], index=False)

    ok = kinetics[kinetics["kinetics_status"].eq("ok")]
    summary = {
        "n_input": int(len(kinetics)),
        "n_ok": int(len(ok)),
        "n_sources": int(ok["source_stem"].nunique()) if not ok.empty else 0,
        "n_kinetic_features": int(len(feature_cols)),
        "threshold_quantiles": list(THRESHOLD_QS),
        "event_relative_bin_counts": ok["event_relative_bin"].astype(str).value_counts().to_dict(),
        "top_event_tests": top_items(tests, 20),
        "top_matched_tests": top_items(matched, 20),
        "top_correlations": top_items(corr, 20),
        "source_summary": source_summary.to_dict("records"),
        "guardrail": "Source-balanced pre-event phase kinetics are automatic optical phase-fraction proxies inside stable particle masks with prior-mask fallback. Logistic/Avrami values are descriptive timing summaries, not calibrated reaction constants, manual phase labels, validated particle identities, or diffusion coefficients.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Source-Balanced Pre-Event Phase Kinetics Audit",
        "",
        "Particle-region-only optical phase-fraction kinetics for the source-balanced pre-event ROI cohort.",
        "",
        f"- Input ROI rows: {summary['n_input']}",
        f"- Loaded ROI tensors: {summary['n_ok']}",
        f"- Sources: {summary['n_sources']}",
        f"- Kinetic features tested: {summary['n_kinetic_features']}",
        f"- Event-bin counts: {summary['event_relative_bin_counts']}",
        "",
        "## Top Event Tests",
    ]
    for row in summary["top_event_tests"][:8]:
        lines.append(
            f"- {row.get('target')} {row.get('transform')} {row.get('feature')}: AUC={row.get('oriented_auc'):.3f}, median diff={row.get('median_positive_minus_negative'):.3g}, p={row.get('mwu_p'):.4g}"
        )
    lines += ["", "## Top Matched Tests"]
    for row in summary["top_matched_tests"][:8]:
        lines.append(
            f"- {row.get('match_scheme')} near-vs-{row.get('control_bin')} {row.get('feature')}: n={row.get('n_pairs')}, median diff={row.get('median_near_minus_control'):.3g}, p={row.get('signflip_mean_abs_p'):.4g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out), "n_ok": summary["n_ok"], "top_event_tests": summary["top_event_tests"][:3], "top_matched_tests": summary["top_matched_tests"][:3]}, indent=2))


if __name__ == "__main__":
    main()
