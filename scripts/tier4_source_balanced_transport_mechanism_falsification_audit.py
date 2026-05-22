#!/usr/bin/env python3
"""Falsify the source-balanced transport mechanism dossier.

This audit asks whether the mechanism review score remains event-local when the
comparison is restricted within each acquisition source. It intentionally does
not train a classifier. It builds matched near-pre versus control/far/post pairs,
source-median contrasts, and top-k enrichment summaries for the dossier scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


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


def auc_from_scores(y: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * (y_sorted == 1)).sum() / n_pos)


def permutation_pvalue(y: np.ndarray, score: np.ndarray, groups: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    ok = np.isfinite(score)
    y = y[ok].astype(int)
    score = score[ok].astype(float)
    groups = groups[ok].astype(str)
    observed = auc_from_scores(y, score)
    if not np.isfinite(observed):
        return float("nan")
    extreme = 0
    unique_groups = np.unique(groups)
    for _ in range(n_perm):
        yp = y.copy()
        for group in unique_groups:
            idx = np.flatnonzero(groups == group)
            if len(np.unique(yp[idx])) > 1:
                yp[idx] = rng.permutation(yp[idx])
        stat = auc_from_scores(yp, score)
        if np.isfinite(stat) and abs(stat - 0.5) >= abs(observed - 0.5):
            extreme += 1
    return float((extreme + 1) / (n_perm + 1))


def sign_flip_pvalue(deltas: Sequence[float], n_perm: int, rng: np.random.Generator) -> float:
    x = np.asarray(deltas, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    observed = float(np.mean(x))
    extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(x))
        stat = float(np.mean(x * signs))
        if abs(stat) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (n_perm + 1))


def build_pairs(df: pd.DataFrame, control_bins: Iterable[str], label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    controls = set(control_bins)
    near = df[df["event_relative_bin"].astype(str).eq("near_pre_event_1_8")].copy()
    for _, pos in near.iterrows():
        same = df[(df["source_stem"].astype(str) == str(pos["source_stem"])) & df["event_relative_bin"].astype(str).isin(controls)].copy()
        if same.empty:
            continue
        same["cycle_distance"] = (numeric(same, "cycleNo") - float(pos["cycleNo"])).abs()
        same = same.sort_values(["cycle_distance", "transport_review_rank", "roi_id"])
        neg = same.iloc[0]
        base = {
            "pair_set": label,
            "source_stem": pos.get("source_stem"),
            "near_roi_id": pos.get("roi_id"),
            "control_roi_id": neg.get("roi_id"),
            "near_cycle": pos.get("cycleNo"),
            "control_cycle": neg.get("cycleNo"),
            "control_event_bin": neg.get("event_relative_bin"),
            "cycle_distance": neg.get("cycle_distance"),
        }
        for col in SCORE_COLS:
            base[f"near_{col}"] = pos.get(col, np.nan)
            base[f"control_{col}"] = neg.get(col, np.nan)
            base[f"delta_{col}"] = pd.to_numeric(pd.Series([pos.get(col, np.nan)]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([neg.get(col, np.nan)]), errors="coerce").iloc[0]
        rows.append(base)
    return pd.DataFrame(rows)


SCORE_COLS = [
    "transport_mechanism_score",
    "transport_source_residual_score",
    "transport_raw_score",
    "front_kinetic_score",
    "observable_tail_score",
    "qc_review_score",
    "abs_radial_flow_mean_source_residual",
    "particle_flow_mag_mean_source_residual",
    "curl_mean_source_residual",
]

TARGETS = {
    "near_vs_any_non_near": lambda d: d["event_relative_bin"].astype(str).ne("near_pre_event_1_8"),
    "near_vs_post_control": lambda d: d["event_relative_bin"].astype(str).isin(["post_event_1_16", "no_near_event_control"]),
    "near_vs_far_mid_control": lambda d: d["event_relative_bin"].astype(str).isin(["far_pre_event_17_32", "mid_pre_event_9_16", "no_near_event_control"]),
}

PAIR_SETS = {
    "same_source_post_control": ["post_event_1_16", "no_near_event_control"],
    "same_source_far_mid_control": ["far_pre_event_17_32", "mid_pre_event_9_16", "no_near_event_control"],
    "same_source_any_non_near": ["far_pre_event_17_32", "mid_pre_event_9_16", "post_event_1_16", "no_near_event_control"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_transport_mechanism_falsification_audit")
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = read_csv(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_dossier.csv")
    df["cycleNo"] = numeric(df, "cycleNo")
    df["near_pre_flag"] = df["event_relative_bin"].astype(str).eq("near_pre_event_1_8").astype(int)

    event_rows: List[Dict[str, Any]] = []
    for target, neg_mask_fn in TARGETS.items():
        mask = df["near_pre_flag"].eq(1) | neg_mask_fn(df)
        sub = df[mask].copy()
        y = sub["near_pre_flag"].to_numpy(dtype=int)
        groups = sub["source_stem"].astype(str).to_numpy()
        for col in SCORE_COLS:
            if col not in sub.columns:
                continue
            score = numeric(sub, col).to_numpy(dtype=float)
            pos = score[y == 1]
            neg = score[y == 0]
            event_rows.append({
                "target": target,
                "feature": col,
                "n_rows": int(np.isfinite(score).sum()),
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == 0).sum()),
                "n_sources": int(sub["source_stem"].nunique()),
                "auc": auc_from_scores(y, score),
                "average_precision": average_precision(y, score),
                "median_pos": float(np.nanmedian(pos)) if len(pos) else float("nan"),
                "median_neg": float(np.nanmedian(neg)) if len(neg) else float("nan"),
                "median_diff_pos_minus_neg": float(np.nanmedian(pos) - np.nanmedian(neg)) if len(pos) and len(neg) else float("nan"),
                "source_stratified_permutation_p": permutation_pvalue(y, score, groups, args.n_perm, rng),
            })
    event_tests = pd.DataFrame(event_rows).sort_values(["target", "source_stratified_permutation_p", "auc"], ascending=[True, True, False])

    pair_tables = [build_pairs(df, bins, name) for name, bins in PAIR_SETS.items()]
    pairs = pd.concat([p for p in pair_tables if not p.empty], ignore_index=True) if any(not p.empty for p in pair_tables) else pd.DataFrame()
    pair_rows: List[Dict[str, Any]] = []
    if not pairs.empty:
        for pair_set, sub in pairs.groupby("pair_set"):
            for col in SCORE_COLS:
                delta_col = f"delta_{col}"
                if delta_col not in sub.columns:
                    continue
                deltas = numeric(sub, delta_col).dropna().to_numpy(dtype=float)
                if len(deltas) == 0:
                    continue
                pair_rows.append({
                    "pair_set": pair_set,
                    "feature": col,
                    "n_pairs": int(len(deltas)),
                    "n_sources": int(sub.loc[numeric(sub, delta_col).notna(), "source_stem"].nunique()),
                    "median_delta_near_minus_control": float(np.median(deltas)),
                    "mean_delta_near_minus_control": float(np.mean(deltas)),
                    "positive_delta_fraction": float(np.mean(deltas > 0)),
                    "sign_flip_p": sign_flip_pvalue(deltas, args.n_perm, rng),
                })
    pair_tests = pd.DataFrame(pair_rows).sort_values(["pair_set", "sign_flip_p", "median_delta_near_minus_control"], ascending=[True, True, False]) if pair_rows else pd.DataFrame()

    source_rows: List[Dict[str, Any]] = []
    for source, sub in df.groupby("source_stem", dropna=False):
        near = sub[sub["near_pre_flag"].eq(1)]
        non = sub[sub["near_pre_flag"].eq(0)]
        if near.empty or non.empty:
            continue
        row: Dict[str, Any] = {
            "source_stem": source,
            "n_rows": int(len(sub)),
            "n_near": int(len(near)),
            "n_non_near": int(len(non)),
        }
        for col in SCORE_COLS:
            if col in sub.columns:
                row[f"delta_{col}"] = float(np.nanmedian(numeric(near, col)) - np.nanmedian(numeric(non, col)))
        source_rows.append(row)
    source_contrasts = pd.DataFrame(source_rows)

    source_test_rows: List[Dict[str, Any]] = []
    if not source_contrasts.empty:
        for col in SCORE_COLS:
            delta_col = f"delta_{col}"
            if delta_col not in source_contrasts.columns:
                continue
            deltas = numeric(source_contrasts, delta_col).dropna().to_numpy(dtype=float)
            if len(deltas) == 0:
                continue
            source_test_rows.append({
                "feature": col,
                "n_sources": int(len(deltas)),
                "median_source_delta_near_minus_non": float(np.median(deltas)),
                "mean_source_delta_near_minus_non": float(np.mean(deltas)),
                "positive_source_fraction": float(np.mean(deltas > 0)),
                "sign_flip_p": sign_flip_pvalue(deltas, args.n_perm, rng),
            })
    source_tests = pd.DataFrame(source_test_rows).sort_values(["sign_flip_p", "median_source_delta_near_minus_non"], ascending=[True, False]) if source_test_rows else pd.DataFrame()

    top_rows: List[Dict[str, Any]] = []
    ranked = df.sort_values("transport_review_rank")
    for k in [5, 10, 20, 40]:
        sub = ranked.head(k)
        top_source = sub["source_stem"].astype(str).value_counts().head(1)
        top_rows.append({
            "top_k": k,
            "near_pre_fraction": float(sub["near_pre_flag"].mean()),
            "n_sources": int(sub["source_stem"].nunique()),
            "dominant_source": top_source.index[0] if not top_source.empty else None,
            "dominant_source_fraction": float(top_source.iloc[0] / len(sub)) if not top_source.empty else None,
            "n_priority_tiers": int(sub["transport_review_tier"].astype(str).str.startswith("priority").sum()),
            "n_diffusion_claim_candidates": int((~sub["diffusion_claim_blocked"].fillna(True).astype(bool)).sum()) if "diffusion_claim_blocked" in sub.columns else 0,
        })
    topk = pd.DataFrame(top_rows)

    paths = {
        "event_tests": out / "source_balanced_transport_mechanism_falsification_event_tests.csv",
        "matched_pairs": out / "source_balanced_transport_mechanism_falsification_matched_pairs.csv",
        "pair_tests": out / "source_balanced_transport_mechanism_falsification_pair_tests.csv",
        "source_contrasts": out / "source_balanced_transport_mechanism_falsification_source_contrasts.csv",
        "source_tests": out / "source_balanced_transport_mechanism_falsification_source_tests.csv",
        "topk": out / "source_balanced_transport_mechanism_falsification_topk.csv",
        "summary": out / "source_balanced_transport_mechanism_falsification_summary.json",
    }
    event_tests.to_csv(paths["event_tests"], index=False)
    pairs.to_csv(paths["matched_pairs"], index=False)
    pair_tests.to_csv(paths["pair_tests"], index=False)
    source_contrasts.to_csv(paths["source_contrasts"], index=False)
    source_tests.to_csv(paths["source_tests"], index=False)
    topk.to_csv(paths["topk"], index=False)

    lead_event = event_tests[event_tests["feature"].eq("transport_mechanism_score")].head(3).to_dict("records")
    lead_pair = pair_tests[pair_tests["feature"].eq("transport_mechanism_score")].head(3).to_dict("records") if not pair_tests.empty else []
    lead_source = source_tests[source_tests["feature"].eq("transport_mechanism_score")].head(1).to_dict("records") if not source_tests.empty else []
    summary = {
        "n_rows": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_sources": int(df["source_stem"].nunique()),
        "n_near_pre": int(df["near_pre_flag"].sum()),
        "n_matched_pairs": int(len(pairs)),
        "lead_event_tests_for_transport_mechanism_score": clean_json(lead_event),
        "lead_pair_tests_for_transport_mechanism_score": clean_json(lead_pair),
        "lead_source_test_for_transport_mechanism_score": clean_json(lead_source[0] if lead_source else {}),
        "topk_enrichment": clean_json(topk.to_dict("records")),
        "guardrail": "This is a falsification audit for the automatic mechanism review score. Same-source enrichment supports event-local ranking, but the scores remain optical/AI descriptors rather than calibrated transport, phase-boundary velocity, diffusion, or causal proof.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["summary"].write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    (out / "README.md").write_text(
        "# Source-Balanced Transport Mechanism Falsification Audit\n\n"
        "Same-source event-local falsification tests for the transport mechanism dossier.\n\n"
        f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}\n"
        f"- Near-pre rows: {summary['n_near_pre']}\n"
        f"- Matched same-source pairs: {summary['n_matched_pairs']}\n\n"
        f"Guardrail: {summary['guardrail']}\n"
    )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True)[:12000])


if __name__ == "__main__":
    main()
