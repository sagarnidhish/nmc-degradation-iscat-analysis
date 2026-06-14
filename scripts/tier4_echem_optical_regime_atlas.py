#!/usr/bin/env python3
"""Build an electrochemical regime atlas for optical degradation candidates.

This is a cycle-level bridge between echem trajectory shape and the optical/AI
physics outputs. It treats charge/discharge asymmetry, voltage-bin capacity
allocation, and dQ/dV proxy concentration as regime descriptors, then asks
where synchronized optical drops, future-drop warnings, and cross-modal
degradation candidates sit in that regime space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


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
        return pd.DataFrame()
    return pd.read_csv(path)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def rank01(series: pd.Series, high: bool = True) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() <= 1:
        return pd.Series(np.nan, index=series.index)
    ranks = vals.rank(pct=True, method="average")
    return ranks if high else 1.0 - ranks


def robust_z(frame: pd.DataFrame, cols: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    z = pd.DataFrame(index=frame.index)
    stats: Dict[str, Dict[str, float]] = {}
    for col in cols:
        vals = pd.to_numeric(frame[col], errors="coerce")
        med = vals.median(skipna=True)
        iqr = vals.quantile(0.75) - vals.quantile(0.25)
        if not np.isfinite(iqr) or iqr == 0:
            std = vals.std(skipna=True)
            scale = std if np.isfinite(std) and std > 0 else np.nan
        else:
            scale = iqr / 1.349
        if not np.isfinite(scale) or scale == 0:
            z[col] = np.nan
        else:
            z[col] = (vals - med) / scale
        stats[col] = {"median": float(med) if np.isfinite(med) else np.nan, "scale": float(scale) if np.isfinite(scale) else np.nan}
    return z, stats


def pca_scores(frame: pd.DataFrame, cols: List[str], n_components: int = 3) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    z, _ = robust_z(frame, cols)
    use = z.copy()
    for col in use.columns:
        use[col] = use[col].fillna(use[col].median(skipna=True))
    good_cols = [c for c in use.columns if use[c].notna().sum() >= 8 and use[c].std(skipna=True) > 0]
    if len(good_cols) < 2:
        return pd.DataFrame(index=frame.index), []
    x = use[good_cols].to_numpy(dtype=float)
    x = x - np.nanmean(x, axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    n = min(n_components, vt.shape[0])
    scores = pd.DataFrame(index=frame.index)
    denom = max(x.shape[0] - 1, 1)
    total_var = float(np.sum((s ** 2) / denom))
    loadings = []
    for i in range(n):
        scores[f"echem_regime_pc{i + 1}"] = u[:, i] * s[i]
        comp = vt[i, :]
        order = np.argsort(np.abs(comp))[::-1][:10]
        loadings.append({
            "component": f"echem_regime_pc{i + 1}",
            "explained_variance_fraction": float(((s[i] ** 2) / denom) / total_var) if total_var > 0 else np.nan,
            "top_loadings": [
                {"feature": good_cols[j], "loading": float(comp[j])}
                for j in order
            ],
        })
    return scores, loadings


def tertile_labels(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    labels = pd.Series("missing", index=series.index, dtype=object)
    if vals.notna().sum() < 6 or vals.nunique(dropna=True) < 3:
        return labels
    q1, q2 = vals.quantile([1 / 3, 2 / 3])
    labels.loc[vals <= q1] = "pc1_low"
    labels.loc[(vals > q1) & (vals <= q2)] = "pc1_mid"
    labels.loc[vals > q2] = "pc1_high"
    return labels


def add_derived_echem_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["capacity_fade_from_first_mAh"] = num(out, "capacity_mAh").iloc[0] - num(out, "capacity_mAh")
    out["capacity_fraction_of_first"] = num(out, "capacity_mAh") / num(out, "capacity_mAh").iloc[0]
    out["coulombic_inefficiency_pct"] = 100.0 - num(out, "coulombic_efficiency_pct")
    out["charge_discharge_capacity_gap_mAh"] = num(out, "shape_charge_mAh_pos") - num(out, "shape_charge_mAh_neg_abs")
    out["charge_discharge_capacity_abs_gap_mAh"] = out["charge_discharge_capacity_gap_mAh"].abs()
    out["signed_charge_fraction"] = num(out, "shape_charge_mAh_signed") / num(out, "shape_charge_mAh_abs")
    out["voltage_peak_hysteresis_proxy"] = num(out, "pos_dq_abs_peak_voltage") - num(out, "neg_dq_abs_peak_voltage")
    out["highV_charge_discharge_imbalance"] = num(out, "pos_dq_abs_highV_frac") - num(out, "neg_dq_abs_highV_frac")
    out["midV_charge_discharge_imbalance"] = num(out, "pos_dq_abs_midV_frac") - num(out, "neg_dq_abs_midV_frac")
    out["lowV_charge_discharge_imbalance"] = num(out, "pos_dq_abs_lowV_frac") - num(out, "neg_dq_abs_lowV_frac")
    out["dqdv_peak_concentration"] = pd.concat(
        [num(out, "pos_dq_abs_peak_frac"), num(out, "neg_dq_abs_peak_frac")],
        axis=1,
    ).max(axis=1)
    out["dqdv_entropy_asymmetry"] = num(out, "pos_dq_abs_entropy") - num(out, "neg_dq_abs_entropy")
    out["dqdv_integral_asymmetry"] = num(out, "pos_dqdv_abs_integral_proxy") - num(out, "neg_dqdv_abs_integral_proxy")
    out["cycle_index_rank"] = rank01(num(out, "cycleNo"))
    out["echem_shape_missing"] = num(out, "shape_charge_mAh_abs").isna().astype(int)
    out["echem_ce_extreme_or_missing"] = (
        num(out, "coulombic_efficiency_pct").isna()
        | num(out, "coulombic_efficiency_pct").lt(80)
        | num(out, "coulombic_efficiency_pct").gt(120)
    ).astype(int)
    out["echem_outlier_score"] = pd.concat([
        rank01(out["capacity_fade_from_first_mAh"].abs()),
        rank01(out["coulombic_inefficiency_pct"].abs()),
        rank01(out["charge_discharge_capacity_abs_gap_mAh"]),
        rank01(out["voltage_peak_hysteresis_proxy"].abs()),
        rank01(out["dqdv_peak_concentration"]),
    ], axis=1).mean(axis=1, skipna=True)
    return out


def numeric_features(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    features = []
    for col in candidates:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 12 and vals.nunique(dropna=True) >= 4:
            features.append(col)
    return features


def binary_contrast(df: pd.DataFrame, feature: str, target: str) -> Dict[str, Any]:
    sub = df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 8 or sub[target].nunique() < 2:
        return {}
    pos = sub.loc[sub[target] > 0, feature].to_numpy(dtype=float)
    neg = sub.loc[sub[target] <= 0, feature].to_numpy(dtype=float)
    if len(pos) < 2 or len(neg) < 4:
        return {}
    _, p = mannwhitneyu(pos, neg, alternative="two-sided")
    return {
        "feature": feature,
        "target": target,
        "n": int(len(sub)),
        "n_positive": int(len(pos)),
        "median_positive": float(np.median(pos)),
        "median_negative": float(np.median(neg)),
        "median_positive_minus_negative": float(np.median(pos) - np.median(neg)),
        "mannwhitney_p": float(p),
    }


def correlation(df: pd.DataFrame, feature: str, target: str) -> Dict[str, Any]:
    sub = df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 8 or sub[feature].nunique() < 4 or sub[target].nunique() < 4:
        return {}
    rho, p = spearmanr(sub[feature], sub[target])
    return {"feature": feature, "target": target, "n": int(len(sub)), "spearman_rho": float(rho), "p_value": float(p)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/echem_optical_regime_atlas")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = read_csv(derived / "cycle_state_space_transition_audit" / "cycle_state_space_table.csv")
    consensus = read_csv(derived / "cross_modal_degradation_consensus" / "cross_modal_consensus_cycle_table.csv")
    front = read_csv(derived / "balanced_future_roi_physics_audit" / "balanced_future_cycle_collapsed_features.csv")
    temporal = read_csv(derived / "temporal_directionality_physics_audit" / "temporal_directionality_cycle_summary.csv")
    if state.empty or "cycleNo" not in state.columns:
        raise FileNotFoundError("Need cycle_state_space_transition_audit/cycle_state_space_table.csv")

    df = state.copy()
    df["cycleNo"] = pd.to_numeric(df["cycleNo"], errors="coerce")
    df = df.dropna(subset=["cycleNo"]).drop_duplicates("cycleNo", keep="first").sort_values("cycleNo")
    df["cycleNo"] = df["cycleNo"].astype(int)

    if not consensus.empty:
        keep = [c for c in consensus.columns if c not in df.columns or c == "cycleNo"]
        df = df.merge(consensus[keep], on="cycleNo", how="left", suffixes=("", "_consensus"))
    if not front.empty:
        front_keep = ["cycleNo"] + [c for c in front.columns if c not in df.columns and c != "cycleNo"]
        df = df.merge(front[front_keep], on="cycleNo", how="left")
    if not temporal.empty:
        temporal_keep = ["cycleNo"] + [c for c in temporal.columns if c not in df.columns and c != "cycleNo"]
        df = df.merge(temporal[temporal_keep], on="cycleNo", how="left", suffixes=("", "_temporal"))

    df = add_derived_echem_features(df)

    echem_candidates = [
        "capacity_mAh", "capacity_fade_from_first_mAh", "capacity_fraction_of_first",
        "coulombic_efficiency_pct", "coulombic_inefficiency_pct",
        "shape_V_range", "shape_V_mean", "shape_V_std", "shape_I_abs_mean_mA",
        "shape_I_pos_fraction", "shape_I_neg_fraction", "shape_charge_mAh_abs",
        "shape_charge_mAh_pos", "shape_charge_mAh_neg_abs",
        "charge_discharge_capacity_gap_mAh", "charge_discharge_capacity_abs_gap_mAh",
        "signed_charge_fraction", "shape_dVdt_abs_p95", "shape_dVdt_sign_consistency",
        "all_dq_abs_lowV_frac", "all_dq_abs_midV_frac", "all_dq_abs_highV_frac",
        "all_dq_abs_peak_voltage", "all_dq_abs_peak_frac", "all_dq_abs_entropy",
        "pos_dq_abs_lowV_frac", "pos_dq_abs_midV_frac", "pos_dq_abs_highV_frac",
        "pos_dq_abs_peak_voltage", "pos_dq_abs_peak_frac", "pos_dq_abs_entropy",
        "neg_dq_abs_lowV_frac", "neg_dq_abs_midV_frac", "neg_dq_abs_highV_frac",
        "neg_dq_abs_peak_voltage", "neg_dq_abs_peak_frac", "neg_dq_abs_entropy",
        "voltage_peak_hysteresis_proxy", "highV_charge_discharge_imbalance",
        "midV_charge_discharge_imbalance", "lowV_charge_discharge_imbalance",
        "dqdv_peak_concentration", "dqdv_entropy_asymmetry", "dqdv_integral_asymmetry",
        "cycle_index_rank", "echem_outlier_score",
    ]
    echem_features = numeric_features(df, echem_candidates)
    scores, loadings = pca_scores(df, echem_features, n_components=4)
    df = pd.concat([df, scores], axis=1)
    if "echem_regime_pc1" in df.columns:
        df["echem_pc1_tertile"] = tertile_labels(df["echem_regime_pc1"])
    else:
        df["echem_pc1_tertile"] = "missing"

    df["synchronized_multimodal_candidate"] = (
        df.get("consensus_class", pd.Series("", index=df.index)).astype(str)
        .eq("synchronized_multimodal_degradation_candidate")
        .astype(int)
    )
    df["multimodal_outlier_without_trace_drop"] = (
        df.get("consensus_class", pd.Series("", index=df.index)).astype(str)
        .eq("multimodal_outlier_without_trace_drop")
        .astype(int)
    )
    df["echem_optical_priority_score"] = pd.concat([
        rank01(num(df, "cross_modal_consensus_score")),
        rank01(num(df, "n_modal_votes")),
        rank01(num(df, "echem_outlier_score")),
        rank01(num(df, "state_state_step_norm")),
        rank01(num(df, "future_any_drop_within_8cycles")),
    ], axis=1).mean(axis=1, skipna=True)

    binary_targets = [
        "any_abrupt_drop",
        "synchronized_drop_2plus",
        "future_any_drop_within_8cycles",
        "synchronized_multimodal_candidate",
        "multimodal_outlier_without_trace_drop",
    ]
    continuous_targets = [
        "cross_modal_consensus_score",
        "n_modal_votes",
        "max_abs_delta_prev",
        "particle_norm_cv",
        "state_state_step_norm",
        "roi_phase_slope_abs_median_per_s",
        "roi_diffusion_proxy_abs_median_um2_per_s",
        "roi_transferred_masked_residual_signature",
    ]
    test_features = echem_features + [c for c in scores.columns if c in df.columns]

    binary_rows = []
    for feature in test_features:
        for target in binary_targets:
            if target in df.columns:
                row = binary_contrast(df, feature, target)
                if row:
                    binary_rows.append(row)
    corr_rows = []
    for feature in test_features:
        for target in continuous_targets:
            if target in df.columns:
                row = correlation(df, feature, target)
                if row:
                    corr_rows.append(row)
    binary_tests = pd.DataFrame(binary_rows).sort_values(["mannwhitney_p", "feature", "target"]).reset_index(drop=True) if binary_rows else pd.DataFrame()
    correlations = pd.DataFrame(corr_rows).sort_values(["p_value", "feature", "target"]).reset_index(drop=True) if corr_rows else pd.DataFrame()

    regime_summary = (
        df.groupby("echem_pc1_tertile", dropna=False)
        .agg(
            n_cycles=("cycleNo", "count"),
            median_cycle=("cycleNo", "median"),
            median_capacity_mAh=("capacity_mAh", "median"),
            median_echem_outlier_score=("echem_outlier_score", "median"),
            median_cross_modal_score=("cross_modal_consensus_score", "median"),
            mean_modal_votes=("n_modal_votes", "mean"),
            abrupt_drop_rate=("any_abrupt_drop", "mean"),
            sync_drop_rate=("synchronized_drop_2plus", "mean"),
            future8_rate=("future_any_drop_within_8cycles", "mean"),
            median_frame_percentile=("frames_percentile", "median"),
            echem_ce_extreme_or_missing_rate=("echem_ce_extreme_or_missing", "mean"),
        )
        .reset_index()
        .sort_values(["median_cross_modal_score", "future8_rate"], ascending=False)
    )

    top_cycles = df.sort_values("echem_optical_priority_score", ascending=False).head(20)
    top_cols = [
        "cycleNo", "echem_optical_priority_score", "echem_pc1_tertile",
        "echem_regime_pc1", "echem_regime_pc2", "echem_outlier_score",
        "capacity_mAh", "capacity_fade_from_first_mAh", "coulombic_efficiency_pct",
        "echem_ce_extreme_or_missing",
        "voltage_peak_hysteresis_proxy", "highV_charge_discharge_imbalance",
        "cross_modal_consensus_score", "n_modal_votes", "consensus_class",
        "any_abrupt_drop", "synchronized_drop_2plus", "future_any_drop_within_8cycles",
        "frames_percentile", "state_state_step_norm",
    ]
    top_cycles = top_cycles[[c for c in top_cols if c in top_cycles.columns]]

    paths = {
        "cycle_table": out / "echem_optical_regime_cycle_table.csv",
        "binary_tests": out / "echem_optical_regime_binary_tests.csv",
        "correlations": out / "echem_optical_regime_correlations.csv",
        "regime_summary": out / "echem_optical_regime_summary_by_pc1_tertile.csv",
        "top_cycles": out / "echem_optical_regime_top_cycles.csv",
        "summary": out / "echem_optical_regime_atlas_summary.json",
    }
    df.to_csv(paths["cycle_table"], index=False)
    binary_tests.to_csv(paths["binary_tests"], index=False)
    correlations.to_csv(paths["correlations"], index=False)
    regime_summary.to_csv(paths["regime_summary"], index=False)
    top_cycles.to_csv(paths["top_cycles"], index=False)

    summary = {
        "n_cycles": int(len(df)),
        "n_echem_features": int(len(echem_features)),
        "n_cycles_missing_echem_shape": int(num(df, "echem_shape_missing").sum()),
        "n_cycles_extreme_or_missing_ce": int(num(df, "echem_ce_extreme_or_missing").sum()),
        "pca_loadings": loadings,
        "regime_summary": regime_summary.to_dict("records"),
        "top_binary_tests": binary_tests.head(20).to_dict("records") if not binary_tests.empty else [],
        "top_correlations": correlations.head(20).to_dict("records") if not correlations.empty else [],
        "top_cycles": top_cycles.to_dict("records"),
        "guardrail": "This atlas uses echem shape and dQ/dV-like proxy descriptors to organize optical degradation hypotheses. It is not calibrated dQ/dV, not a mechanistic phase diagram, and does not remove the acquisition/frame-count confounder by itself.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    with paths["summary"].open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True, allow_nan=False)

    lines = [
        "# Echem Optical Regime Atlas",
        "",
        "Cycle-level electrochemical regime descriptors joined to optical/AI degradation evidence.",
        "",
        f"- Cycles: {summary['n_cycles']}",
        f"- Echem regime features: {summary['n_echem_features']}",
        f"- Missing echem-shape cycles: {summary['n_cycles_missing_echem_shape']}",
        f"- Extreme-or-missing CE cycles: {summary['n_cycles_extreme_or_missing_ce']}",
        "",
        "## Regime Summary",
    ]
    for row in summary["regime_summary"]:
        lines.append(
            f"- {row.get('echem_pc1_tertile')}: n={row.get('n_cycles')}, median cycle={row.get('median_cycle'):.3g}, "
            f"median consensus={row.get('median_cross_modal_score'):.3g}, modal votes={row.get('mean_modal_votes'):.3g}, "
            f"event rate={row.get('abrupt_drop_rate'):.3g}, future8 rate={row.get('future8_rate'):.3g}, "
            f"extreme/missing CE rate={row.get('echem_ce_extreme_or_missing_rate'):.3g}"
        )
    lines += ["", "## Top Binary Echem/Optical Tests"]
    for row in summary["top_binary_tests"][:10]:
        lines.append(
            f"- {row.get('feature')} vs {row.get('target')}: median positive-negative "
            f"{row.get('median_positive_minus_negative'):.3g}, p={row.get('mannwhitney_p'):.3g}, n={row.get('n')}"
        )
    lines += ["", "## Top Echem/Optical Correlations"]
    for row in summary["top_correlations"][:10]:
        lines.append(
            f"- {row.get('feature')} vs {row.get('target')}: rho={row.get('spearman_rho'):.3g}, "
            f"p={row.get('p_value'):.3g}, n={row.get('n')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json({
        "out_dir": str(out),
        "n_cycles": summary["n_cycles"],
        "top_binary_tests": summary["top_binary_tests"][:5],
        "top_correlations": summary["top_correlations"][:5],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
