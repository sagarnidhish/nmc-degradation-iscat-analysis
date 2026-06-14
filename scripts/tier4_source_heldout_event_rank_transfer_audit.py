#!/usr/bin/env python3
"""Audit whether event-local particle ROI ranking transfers to held-out sources.

The mechanism dossier ranks particle-region crops well in aggregate, but top-k
rows can concentrate in one acquisition source. This script trains only feature
orientation and scaling on all other sources, scores the held-out source, and
compares that transferable automatic score with fixed review scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


TARGETS = {
    "near_vs_any_non_near": ("near_pre_flag", None),
    "near_vs_post_control": ("near_pre_flag", {"post_event_0_8", "control_far_from_event"}),
}

FIXED_SCORE_COLUMNS = [
    "transport_mechanism_score",
    "transport_raw_score",
    "front_kinetic_score",
    "qc_review_score",
    "visual_sanity_score",
]

EXCLUDE_SUBSTRINGS = [
    "source_residual",
    "rank",
    "flag",
    "future_any_drop",
    "any_abrupt_drop",
    "cycle",
    "local_cycle_index",
    "cycles_to_next_event",
    "cycles_since_prev_event",
    "score",
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


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_score = score[order]
    i = 0
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    pos_rank_sum = ranks[y == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    hits = np.cumsum(y_sorted)
    precision = hits / (np.arange(len(y_sorted)) + 1.0)
    return float((precision * y_sorted).sum() / y_sorted.sum())


def sign_flip_p(values: List[float]) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return float("nan")
    obs = abs(vals.sum())
    n = len(vals)
    count = 0
    total = 2 ** n
    for mask in range(total):
        signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)], dtype=float)
        if abs((vals * signs).sum()) >= obs - 1e-12:
            count += 1
    return float(count / total)


def empirical_percentile(train_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    train_values = np.asarray(train_values, dtype=float)
    values = np.asarray(values, dtype=float)
    train_values = np.sort(train_values[np.isfinite(train_values)])
    if len(train_values) == 0:
        return np.full(len(values), np.nan)
    left = np.searchsorted(train_values, values, side="left")
    right = np.searchsorted(train_values, values, side="right")
    return (left + right) / (2.0 * len(train_values))


def candidate_features(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if col in FIXED_SCORE_COLUMNS:
            continue
        low = col.lower()
        if any(token in low for token in EXCLUDE_SUBSTRINGS):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            finite = np.isfinite(pd.to_numeric(df[col], errors="coerce")).sum()
            if finite >= 16 and df[col].nunique(dropna=True) >= 5:
                cols.append(col)
    return cols


def orient_features(train: pd.DataFrame, y_col: str, features: List[str], max_features: int) -> pd.DataFrame:
    rows = []
    y = train[y_col].astype(int).to_numpy()
    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=float)
        auc = auc_score(y, values)
        if not np.isfinite(auc):
            continue
        strength = abs(auc - 0.5)
        if strength <= 0:
            continue
        rows.append({
            "feature": feature,
            "train_auc": auc,
            "orientation": 1 if auc >= 0.5 else -1,
            "strength": strength,
            "n_train_finite": int(np.isfinite(values).sum()),
        })
    weights = pd.DataFrame(rows)
    if weights.empty:
        return weights
    return weights.sort_values(["strength", "n_train_finite"], ascending=[False, False]).head(max_features).reset_index(drop=True)


def transferable_score(train: pd.DataFrame, test: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    if weights.empty:
        return np.full(len(test), np.nan)
    parts = []
    for row in weights.itertuples(index=False):
        feature = row.feature
        pct = empirical_percentile(
            pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(test[feature], errors="coerce").to_numpy(dtype=float),
        )
        if row.orientation < 0:
            pct = 1.0 - pct
        parts.append(pct * float(row.strength))
    denom = float(weights["strength"].sum())
    if denom <= 0:
        return np.nanmean(np.vstack(parts), axis=0)
    return np.nansum(np.vstack(parts), axis=0) / denom


def summarize_predictions(df: pd.DataFrame, target: str, score_col: str) -> Dict[str, Any]:
    y = df[target].astype(int).to_numpy()
    s = df[score_col].to_numpy(dtype=float)
    pos = df.loc[df[target].eq(1), score_col]
    neg = df.loc[df[target].eq(0), score_col]
    return {
        "score": score_col,
        "n_rows": int(len(df)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "n_sources": int(df["source_stem"].nunique()),
        "auc": auc_score(y, s),
        "average_precision": average_precision(y, s),
        "median_pos": float(pos.median()) if len(pos) else None,
        "median_neg": float(neg.median()) if len(neg) else None,
        "median_diff_pos_minus_neg": float(pos.median() - neg.median()) if len(pos) and len(neg) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_heldout_event_rank_transfer_audit")
    parser.add_argument("--max-features", type=int, default=12)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dossier = pd.read_csv(derived / "source_balanced_transport_mechanism_dossier" / "source_balanced_transport_mechanism_dossier.csv")
    dossier = dossier.copy()
    dossier["near_pre_flag"] = dossier["near_pre_flag"].astype(int)
    features = candidate_features(dossier)

    all_predictions = []
    all_folds = []
    all_weights = []
    all_score_tests = []

    for target_name, (label_col, controls) in TARGETS.items():
        data = dossier.copy()
        if controls is not None:
            data = data[data["event_relative_bin"].isin(set(controls) | {"near_pre_event_1_8"})].copy()
        data["target_label"] = data[label_col].astype(int)
        eligible_sources = []
        for source, grp in data.groupby("source_stem"):
            if grp["target_label"].nunique() == 2:
                eligible_sources.append(source)
        for source in sorted(eligible_sources):
            train = data[data["source_stem"].ne(source)].copy()
            test = data[data["source_stem"].eq(source)].copy()
            if train["target_label"].nunique() < 2 or test["target_label"].nunique() < 2:
                continue
            weights = orient_features(train, "target_label", features, args.max_features)
            test = test.copy()
            test["transfer_oriented_feature_score"] = transferable_score(train, test, weights)
            for score_col in FIXED_SCORE_COLUMNS:
                if score_col in test.columns:
                    test[f"fixed_{score_col}"] = pd.to_numeric(test[score_col], errors="coerce")
            test["target"] = target_name
            test["heldout_source"] = source
            all_predictions.append(test[[
                "target", "heldout_source", "roi_id", "cycleNo", "source_stem", "event_relative_bin",
                "target_label", "transfer_oriented_feature_score",
                *[f"fixed_{c}" for c in FIXED_SCORE_COLUMNS if c in test.columns],
            ]])
            row = {
                "target": target_name,
                "heldout_source": source,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_test_pos": int(test["target_label"].sum()),
                "n_test_neg": int(len(test) - test["target_label"].sum()),
                "n_selected_features": int(len(weights)),
                "selected_features": "; ".join(weights["feature"].tolist()) if not weights.empty else "",
                "transfer_auc": auc_score(test["target_label"].to_numpy(), test["transfer_oriented_feature_score"].to_numpy()),
                "transfer_ap": average_precision(test["target_label"].to_numpy(), test["transfer_oriented_feature_score"].to_numpy()),
                "transfer_median_delta": float(test.loc[test["target_label"].eq(1), "transfer_oriented_feature_score"].median() - test.loc[test["target_label"].eq(0), "transfer_oriented_feature_score"].median()),
            }
            for score_col in FIXED_SCORE_COLUMNS:
                if score_col in test.columns:
                    row[f"{score_col}_auc"] = auc_score(test["target_label"].to_numpy(), pd.to_numeric(test[score_col], errors="coerce").to_numpy(dtype=float))
            all_folds.append(row)
            if not weights.empty:
                w = weights.copy()
                w["target"] = target_name
                w["heldout_source"] = source
                all_weights.append(w)

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    folds = pd.DataFrame(all_folds)
    weights = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()

    if not predictions.empty:
        for target_name, target_df in predictions.groupby("target"):
            score_cols = ["transfer_oriented_feature_score"] + [c for c in predictions.columns if c.startswith("fixed_")]
            for score_col in score_cols:
                if score_col not in target_df.columns:
                    continue
                summary = summarize_predictions(target_df, "target_label", score_col)
                summary["target"] = target_name
                per_source = []
                for _, fold in folds[folds["target"].eq(target_name)].iterrows():
                    key = "transfer_auc" if score_col == "transfer_oriented_feature_score" else f"{score_col.replace('fixed_', '')}_auc"
                    if key in fold and np.isfinite(fold[key]):
                        per_source.append(float(fold[key]) - 0.5)
                summary["source_auc_minus_half_sign_flip_p"] = sign_flip_p(per_source)
                summary["n_eligible_heldout_sources"] = len(per_source)
                all_score_tests.append(summary)

    score_tests = pd.DataFrame(all_score_tests)
    topk_rows = []
    if not predictions.empty:
        for target_name, target_df in predictions.groupby("target"):
            for score_col in ["transfer_oriented_feature_score", "fixed_transport_mechanism_score", "fixed_qc_review_score"]:
                if score_col not in target_df.columns:
                    continue
                ranked = target_df.sort_values(score_col, ascending=False)
                for k in [5, 10, 20, 40]:
                    top = ranked.head(k)
                    if top.empty:
                        continue
                    source_counts = top["source_stem"].value_counts()
                    topk_rows.append({
                        "target": target_name,
                        "score": score_col,
                        "k": int(k),
                        "n_rows": int(len(top)),
                        "near_pre_fraction": float(top["target_label"].mean()),
                        "n_sources": int(top["source_stem"].nunique()),
                        "max_source_fraction": float(source_counts.max() / len(top)),
                        "dominant_source": str(source_counts.index[0]),
                    })
    topk = pd.DataFrame(topk_rows)

    paths = {
        "predictions": out / "source_heldout_event_rank_transfer_predictions.csv",
        "folds": out / "source_heldout_event_rank_transfer_folds.csv",
        "feature_weights": out / "source_heldout_event_rank_transfer_feature_weights.csv",
        "score_tests": out / "source_heldout_event_rank_transfer_score_tests.csv",
        "topk": out / "source_heldout_event_rank_transfer_topk.csv",
        "summary": out / "source_heldout_event_rank_transfer_summary.json",
        "readme": out / "README.md",
    }
    predictions.to_csv(paths["predictions"], index=False)
    folds.to_csv(paths["folds"], index=False)
    weights.to_csv(paths["feature_weights"], index=False)
    score_tests.to_csv(paths["score_tests"], index=False)
    topk.to_csv(paths["topk"], index=False)

    best = score_tests.sort_values(["target", "auc"], ascending=[True, False]).groupby("target").head(3).to_dict("records") if not score_tests.empty else []
    transfer_rows = score_tests[score_tests["score"].eq("transfer_oriented_feature_score")].to_dict("records") if not score_tests.empty else []
    summary = clean_json({
        "n_input_rows": int(len(dossier)),
        "n_sources": int(dossier["source_stem"].nunique()),
        "n_candidate_features": int(len(features)),
        "n_prediction_rows": int(len(predictions)),
        "n_folds": int(len(folds)),
        "targets": sorted(predictions["target"].unique().tolist()) if not predictions.empty else [],
        "transfer_score_tests": transfer_rows,
        "best_score_tests_by_target": best,
        "topk_summary": topk.to_dict("records"),
        "guardrail": "Held-out-source transfer scores are automatic particle-region ranking diagnostics. They do not add manual labels, calibrated velocities, diffusion coefficients, or deployment validation.",
        "outputs": {k: str(v) for k, v in paths.items()},
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Source-Heldout Event Rank Transfer Audit",
        "",
        "Tests whether automatic particle-region features can rank near-pre-event ROIs in held-out acquisition sources.",
        "",
        f"- Input dossier rows: {summary['n_input_rows']}",
        f"- Sources: {summary['n_sources']}",
        f"- Candidate raw automatic features: {summary['n_candidate_features']}",
        f"- Held-out folds: {summary['n_folds']}",
        "",
        "## Transfer Score Tests",
        "",
    ]
    for row in transfer_rows:
        lines.append(
            f"- {row['target']}: AUC {row['auc']:.3f}, AP {row['average_precision']:.3f}, "
            f"eligible held-out sources {row['n_eligible_heldout_sources']}, sign-flip p {row['source_auc_minus_half_sign_flip_p']:.3g}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    paths["readme"].write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
