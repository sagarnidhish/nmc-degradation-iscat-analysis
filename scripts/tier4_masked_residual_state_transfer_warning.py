#!/usr/bin/env python3
"""Transfer masked video-residual signatures onto the full cycle-state table.

The masked rollout warning audit has video-backed particle-region residuals for
only the ROI cycles that were exported as crops. This bridge asks whether those
masked residual signatures live on the broader trace/echem cycle-state manifold,
then projects the signature across all trace cycles to test future-drop warning
with more positives.

This is a transfer guardrail, not a replacement for video residual extraction on
every cycle.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score


TARGETS = [
    "any_abrupt_drop",
    "future_any_drop_within_4cycles",
    "future_any_drop_within_8cycles",
    "future_any_drop_within_16cycles",
    "future_sync2_drop_within_8cycles",
]

STATE_FEATURES = [
    "cycle_state_pc1",
    "cycle_state_pc2",
    "cycle_state_pc3",
    "cycle_state_pc4",
    "cycle_state_pc5",
    "cycle_state_pc6",
    "cycle_state_pc7",
    "cycle_state_pc8",
    "degradation_state_axis",
    "state_step_norm",
    "axis_step",
    "frames_percentile",
    "capacity_mAh",
    "coulombic_efficiency_pct",
    "particle_norm_range",
    "particle_norm_cv",
    "mean_abs_delta_prev",
    "max_abs_delta_prev",
    "delta_std_across_particles",
]


def clean_value(x):
    if isinstance(x, dict):
        return {str(k): clean_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_value(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    return x


def zscore(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    med = vals.median()
    mad = (vals - med).abs().median()
    scale = 1.4826 * mad if np.isfinite(mad) and mad > 0 else vals.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(vals)), index=vals.index, dtype=float)
    return (vals - med) / scale


def safe_mwu(pos: Iterable[float], neg: Iterable[float], rng: np.random.Generator, n_perm: int) -> Dict[str, float]:
    a = pd.to_numeric(pd.Series(pos), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.to_numeric(pd.Series(neg), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 2 or len(b) < 2:
        return {
            "n_positive": int(len(a)),
            "n_negative": int(len(b)),
            "median_positive": np.nan,
            "median_negative": np.nan,
            "median_positive_minus_negative": np.nan,
            "mannwhitney_p": np.nan,
            "permutation_p_abs_median_diff": np.nan,
        }
    obs = float(a.median() - b.median())
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        p = np.nan
    pooled = np.concatenate([a.to_numpy(float), b.to_numpy(float)])
    n_a = len(a)
    more = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = float(np.nanmedian(perm[:n_a]) - np.nanmedian(perm[n_a:]))
        if abs(stat) >= abs(obs):
            more += 1
    return {
        "n_positive": int(len(a)),
        "n_negative": int(len(b)),
        "median_positive": float(a.median()),
        "median_negative": float(b.median()),
        "median_positive_minus_negative": obs,
        "mannwhitney_p": float(p) if np.isfinite(p) else np.nan,
        "permutation_p_abs_median_diff": float((more + 1) / (n_perm + 1)),
    }


def residual_feature_candidates(anchor: pd.DataFrame) -> List[str]:
    out = []
    for col in anchor.columns:
        if not any(col.startswith(prefix) for prefix in ["low_rank_dmd_", "persistence_", "velocity_"]):
            continue
        if "accepted_area" in col or "mask_fallback" in col:
            continue
        if not any(tok in col for tok in ["particle_mse", "particle_to_nonparticle", "fraction_of_full", "ratio_vs_persistence"]):
            continue
        vals = pd.to_numeric(anchor[col], errors="coerce")
        if vals.notna().sum() >= 6 and vals.nunique(dropna=True) > 2:
            out.append(col)
    return out


def choose_signature_features(anchor: pd.DataFrame, rng: np.random.Generator, n_perm: int, top_k: int) -> Tuple[pd.DataFrame, pd.Series]:
    rows = []
    y = pd.to_numeric(anchor["any_abrupt_drop"], errors="coerce").fillna(0).astype(int)
    for col in residual_feature_candidates(anchor):
        vals = pd.to_numeric(anchor[col], errors="coerce")
        out = safe_mwu(vals[y == 1], vals[y == 0], rng, n_perm)
        out.update({"feature": col, "orientation": 1.0 if out["median_positive_minus_negative"] >= 0 else -1.0})
        rows.append(out)
    tests = pd.DataFrame(rows).sort_values(["permutation_p_abs_median_diff", "mannwhitney_p"], na_position="last")
    chosen = tests.head(top_k).copy()
    if chosen.empty:
        raise RuntimeError("No usable masked residual features for signature.")
    parts = []
    for _, row in chosen.iterrows():
        parts.append(float(row["orientation"]) * zscore(anchor[str(row["feature"])]))
    signature = pd.concat(parts, axis=1).median(axis=1)
    return tests, signature


def ridge_transfer(anchor: pd.DataFrame, full: pd.DataFrame, target: pd.Series, alpha: float) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
    features = [c for c in STATE_FEATURES if c in full.columns and c in anchor.columns]
    x_anchor = anchor[features].apply(pd.to_numeric, errors="coerce")
    x_full = full[features].apply(pd.to_numeric, errors="coerce")
    med = x_anchor.median()
    x_anchor = x_anchor.fillna(med)
    x_full = x_full.fillna(med)
    mu = x_anchor.mean()
    sd = x_anchor.std(ddof=0).replace(0, 1.0)
    xa = ((x_anchor - mu) / sd).to_numpy(float)
    xf = ((x_full - mu) / sd).to_numpy(float)
    y = pd.to_numeric(target, errors="coerce").to_numpy(float)
    y_mu = float(np.nanmean(y))
    yc = y - y_mu
    lhs = xa.T @ xa + alpha * np.eye(xa.shape[1])
    coef = np.linalg.solve(lhs, xa.T @ yc)
    pred_full = pd.Series(xf @ coef + y_mu, index=full.index, name="transferred_masked_residual_signature")
    loo = []
    for i in range(len(anchor)):
        keep = np.arange(len(anchor)) != i
        lhs_i = xa[keep].T @ xa[keep] + alpha * np.eye(xa.shape[1])
        coef_i = np.linalg.solve(lhs_i, xa[keep].T @ (y[keep] - np.nanmean(y[keep])))
        loo.append(float(xa[i] @ coef_i + np.nanmean(y[keep])))
    pred_anchor_loo = pd.Series(loo, index=anchor.index, name="anchor_loo_transferred_signature")
    coef_rows = [{"feature": f, "coef": float(c)} for f, c in sorted(zip(features, coef), key=lambda x: abs(x[1]), reverse=True)]
    return pred_full, pred_anchor_loo, {"features": features, "coefficients": coef_rows}


def score_tests(df: pd.DataFrame, score_cols: List[str], rng: np.random.Generator, n_perm: int) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
        for score in score_cols:
            vals = pd.to_numeric(df[score], errors="coerce")
            out = safe_mwu(vals[y == 1], vals[y == 0], rng, n_perm)
            tmp = df[[score, target]].copy()
            tmp[score] = pd.to_numeric(tmp[score], errors="coerce")
            tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
            tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
            auc = np.nan
            if len(tmp) >= 6 and tmp[target].nunique() == 2 and tmp[score].nunique() > 1:
                try:
                    auc = roc_auc_score(tmp[target].astype(int), tmp[score])
                    if auc < 0.5:
                        auc = 1.0 - auc
                except Exception:
                    auc = np.nan
            out.update({"target": target, "score": score, "abs_oriented_auc": auc})
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["permutation_p_abs_median_diff", "mannwhitney_p"], na_position="last")


def block_auc(df: pd.DataFrame, score_cols: List[str], target: str) -> pd.DataFrame:
    tmp = df.copy().sort_values("cycleNo").reset_index(drop=True)
    tmp["temporal_block"] = pd.qcut(tmp.index, q=4, labels=False, duplicates="drop")
    rows = []
    for block, grp in tmp.groupby("temporal_block"):
        y = pd.to_numeric(grp[target], errors="coerce")
        for score in score_cols:
            vals = pd.to_numeric(grp[score], errors="coerce")
            use = pd.DataFrame({"y": y, "score": vals}).dropna()
            if len(use) < 6 or use["y"].nunique() < 2 or use["score"].nunique() < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(use["y"].astype(int), use["score"])
                auc = max(float(auc), float(1.0 - auc))
            rows.append({
                "target": target,
                "score": score,
                "temporal_block": int(block),
                "cycle_min": float(grp["cycleNo"].min()),
                "cycle_max": float(grp["cycleNo"].max()),
                "n": int(len(use)),
                "n_positive": int((use["y"] == 1).sum()) if len(use) else 0,
                "abs_oriented_auc": auc,
            })
    return pd.DataFrame(rows)


def correlation_tests(df: pd.DataFrame, score: str, context_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in context_cols:
        if col not in df.columns:
            continue
        tmp = df[[score, col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) < 8 or tmp[score].nunique() < 2 or tmp[col].nunique() < 2:
            continue
        rho, p = spearmanr(tmp[score], tmp[col])
        rows.append({"score": score, "context": col, "n": int(len(tmp)), "rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("p_value", na_position="last") if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-cycle-warning", default="/scratch/<account>/<username>/Alek_Jiho/derived/masked_rollout_cycle_warning/masked_rollout_cycle_warning_joined.csv")
    parser.add_argument("--cycle-state", default="/scratch/<account>/<username>/Alek_Jiho/derived/cycle_state_space_transition_audit/cycle_state_space_table.csv")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/masked_residual_state_transfer_warning")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    parser.add_argument("--n-permutation", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    anchor = pd.read_csv(args.masked_cycle_warning)
    full = pd.read_csv(args.cycle_state)
    anchor["cycleNo"] = pd.to_numeric(anchor["cycleNo"], errors="coerce")
    full["cycleNo"] = pd.to_numeric(full["cycleNo"], errors="coerce")
    anchor = anchor.merge(full[["cycleNo"] + [c for c in STATE_FEATURES if c in full.columns]], how="left", on="cycleNo", suffixes=("", "_full"))

    feature_tests, signature = choose_signature_features(anchor, rng, args.n_permutation, args.top_k)
    anchor["masked_residual_signature"] = signature
    transferred, anchor_loo, fit = ridge_transfer(anchor, full, signature, args.ridge_alpha)
    full["transferred_masked_residual_signature"] = transferred
    anchor["anchor_loo_transferred_signature"] = anchor_loo

    score_cols = ["transferred_masked_residual_signature"]
    for col in ["cycle_state_pc2", "degradation_state_axis", "state_step_norm", "mean_abs_delta_prev", "frames_percentile"]:
        if col in full.columns:
            score_cols.append(col)
    tests = score_tests(full, score_cols, rng, args.n_permutation)
    temporal = block_auc(full, score_cols, "future_any_drop_within_8cycles")
    corr = correlation_tests(full, "transferred_masked_residual_signature", [c for c in STATE_FEATURES + TARGETS if c in full.columns])

    anchor_tmp = anchor[["masked_residual_signature", "anchor_loo_transferred_signature"]].dropna()
    if len(anchor_tmp) >= 6 and anchor_tmp.nunique().min() > 1:
        rho, p = spearmanr(anchor_tmp["masked_residual_signature"], anchor_tmp["anchor_loo_transferred_signature"])
        anchor_loo_summary = {"n_anchor": int(len(anchor_tmp)), "rho": float(rho), "p_value": float(p)}
    else:
        anchor_loo_summary = {"n_anchor": int(len(anchor_tmp)), "rho": np.nan, "p_value": np.nan}

    ranked = full.sort_values("transferred_masked_residual_signature", ascending=False).copy()
    ranked_summary_cols = [
        c for c in [
            "cycleNo",
            "transferred_masked_residual_signature",
            "any_abrupt_drop",
            "future_any_drop_within_4cycles",
            "future_any_drop_within_8cycles",
            "future_any_drop_within_16cycles",
            "cycle_state_pc2",
            "degradation_state_axis",
            "state_step_norm",
            "frames_percentile",
            "mean_abs_delta_prev",
            "particle_norm_range",
            "capacity_mAh",
            "coulombic_efficiency_pct",
        ] if c in ranked.columns
    ]
    anchor.to_csv(out / "masked_residual_state_transfer_anchor_cycles.csv", index=False)
    full.to_csv(out / "masked_residual_state_transfer_full_cycles.csv", index=False)
    feature_tests.to_csv(out / "masked_residual_signature_feature_tests.csv", index=False)
    tests.to_csv(out / "masked_residual_state_transfer_target_tests.csv", index=False)
    temporal.to_csv(out / "masked_residual_state_transfer_temporal_blocks.csv", index=False)
    corr.to_csv(out / "masked_residual_state_transfer_correlations.csv", index=False)
    pd.DataFrame(fit["coefficients"]).to_csv(out / "masked_residual_state_transfer_coefficients.csv", index=False)
    ranked.head(30).to_csv(out / "masked_residual_state_transfer_ranked_cycles.csv", index=False)

    summary = {
        "n_anchor_cycles": int(len(anchor)),
        "n_full_cycles": int(len(full)),
        "n_signature_features": int(min(args.top_k, len(feature_tests))),
        "n_permutation": int(args.n_permutation),
        "anchor_loo_summary": anchor_loo_summary,
        "top_signature_features": feature_tests.head(args.top_k).to_dict("records"),
        "transfer_features": fit["features"],
        "top_transfer_coefficients": fit["coefficients"][:10],
        "top_target_tests": tests.head(15).to_dict("records"),
        "temporal_block_auc": temporal.sort_values(["score", "temporal_block"]).to_dict("records"),
        "top_correlations": corr.head(12).to_dict("records") if not corr.empty else [],
        "top_ranked_cycles": ranked[ranked_summary_cols].head(12).to_dict("records"),
        "guardrail": "Masked residual signature is learned from 11 video-backed ROI cycles and transferred through the cycle-state/echem manifold to 89 cycles; use as hypothesis-ranking evidence, not a deployable warning model or direct video residual measurement for unexported cycles.",
    }
    with (out / "masked_residual_state_transfer_warning_summary.json").open("w") as f:
        json.dump(clean_value(summary), f, indent=2)
    with (out / "README.md").open("w") as f:
        f.write("# Masked Residual State Transfer Warning\n\n")
        f.write("Projects a masked particle-rollout residual signature from video-backed ROI cycles onto the full cycle-state table.\n\n")
        f.write(f"- Anchor video-backed cycles: {summary['n_anchor_cycles']}\n")
        f.write(f"- Full cycle-state rows: {summary['n_full_cycles']}\n")
        f.write(f"- Signature features: {summary['n_signature_features']}\n\n")
        f.write(summary["guardrail"] + "\n")
    print(json.dumps(clean_value({
        "n_anchor_cycles": summary["n_anchor_cycles"],
        "n_full_cycles": summary["n_full_cycles"],
        "anchor_loo_summary": summary["anchor_loo_summary"],
        "top_target_tests": summary["top_target_tests"][:5],
    }), indent=2))


if __name__ == "__main__":
    main()
