#!/usr/bin/env python3
"""Source-held-out predictor on source/cycle-aggregated rollout + phase features.

This fixes the exact-row merge failure in the first predictor attempt by
aggregating both feature tables on the shared source/cycle grid before training.

Targets come from the phase-kinetics cohort, which already carries the
future-event labels. Rollout features are merged in by source/cycle and compared
against phase-only and combined feature families under source-held-out CV.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
    except Exception:
        pass
    return value


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def group_median(df: pd.DataFrame, group_cols: List[str], feature_cols: List[str]) -> pd.DataFrame:
    keep = group_cols + feature_cols
    work = df[keep].copy()
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    return work.groupby(group_cols, dropna=False).median().reset_index()


def make_model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=4000, solver="liblinear", class_weight="balanced", C=0.35, random_state=seed),
    )


def evaluate(df: pd.DataFrame, feature_cols: List[str], target: str, seed: int = 0) -> Dict[str, Any]:
    sub = df[df[target].isin([0, 1])].copy()
    if sub.empty:
        return {"target": target, "feature_count": len(feature_cols), "n": 0}
    X = sub[feature_cols]
    y = sub[target].astype(int).values
    groups = sub["source_stem"].astype(str).values
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return {"target": target, "feature_count": len(feature_cols), "n": int(len(sub))}
    n_splits = min(5, len(uniq))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(sub), np.nan)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = make_model(seed)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    valid = np.isfinite(oof)
    if valid.sum() < 8 or len(np.unique(y[valid])) < 2:
        return {"target": target, "feature_count": len(feature_cols), "n": int(valid.sum())}
    auc = roc_auc_score(y[valid], oof[valid])
    ap = average_precision_score(y[valid], oof[valid])
    return {
        "target": target,
        "feature_count": int(len(feature_cols)),
        "n": int(valid.sum()),
        "oriented_auc": float(max(auc, 1.0 - auc)),
        "average_precision": float(ap),
        "raw_auc": float(auc),
        "positive_rate": float(y[valid].mean()),
        "oof_mean": float(np.mean(oof[valid])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-csv", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_sequence_rollout_audit/source_balanced_sequence_rollout_features.csv")
    ap.add_argument("--phase-csv", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/source_balanced_pre_event_phase_kinetics_audit/source_balanced_pre_event_phase_kinetics_features.csv")
    ap.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/rollout_phase_kinetics_predictor_v2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout = pd.read_csv(args.rollout_csv)
    phase = pd.read_csv(args.phase_csv)
    group_cols = ["source_stem", "cycleNo"]

    rollout_features = [
        c for c in rollout.columns
        if c.startswith("persistence_mse_")
        or c.startswith("velocity_mse_")
        or c.startswith("temporal_energy_")
        or c.startswith("roi_norm_mean_")
        or c in ["raw_roi_mean_delta_last_minus_first", "stage_drift_xy_recomputed"]
    ]
    phase_features = [
        c for c in phase.columns
        if "masked_minus_bg" in c
        or "phase_fraction" in c
        or "avrami" in c
        or "logistic" in c
        or "front_gradient" in c
        or "max_abs_rate" in c
        or "total_variation" in c
    ]

    rollout_grp = group_median(rollout, group_cols, rollout_features)
    phase_grp = group_median(phase, group_cols, phase_features + ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"])
    merged = phase_grp.merge(rollout_grp, on=group_cols, how="inner", suffixes=("_phase", "_rollout"))

    # Preserve the target columns from phase_grp.
    targets = [t for t in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"] if t in merged.columns]
    rollout_features = [c for c in rollout_features if c in merged.columns]
    phase_features = [c for c in phase_features if c in merged.columns]
    combined_features = sorted(set(rollout_features + phase_features))

    rows = []
    for target in targets:
        rows.append({"feature_family": "rollout_only", **evaluate(merged, rollout_features, target, seed=args.seed)})
        rows.append({"feature_family": "phase_only", **evaluate(merged, phase_features, target, seed=args.seed)})
        rows.append({"feature_family": "combined", **evaluate(merged, combined_features, target, seed=args.seed)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "rollout_phase_kinetics_predictor_v2_metrics.csv", index=False)

    summary = {
        "n_rows": int(len(merged)),
        "n_sources": int(merged["source_stem"].nunique()) if not merged.empty else 0,
        "n_targets": int(len(targets)),
        "feature_counts": {"rollout_only": len(rollout_features), "phase_only": len(phase_features), "combined": len(combined_features)},
        "best_rows": metrics.sort_values(["target", "oriented_auc", "average_precision"], ascending=[True, False, False]).head(6).to_dict("records") if not metrics.empty else [],
        "outputs": {"metrics": str(out_dir / "rollout_phase_kinetics_predictor_v2_metrics.csv")},
        "guardrail": "Source/cycle-aggregated weak-label classifier over automatic particle-only features. It is intended to test whether phase kinetics add event-risk signal beyond rollout features, not to deploy warnings.",
    }
    (out_dir / "rollout_phase_kinetics_predictor_v2_summary.json").write_text(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    (out_dir / "README.md").write_text(
        "\n".join([
            "# Rollout / Phase Kinetics Predictor v2",
            "",
            f"- Rows: {summary['n_rows']}",
            f"- Sources: {summary['n_sources']}",
            f"- Targets: {summary['n_targets']}",
            "",
            "## Guardrail",
            "",
            summary["guardrail"],
            "",
        ]) + "\n"
    )
    print(metrics.sort_values(["target", "oriented_auc"], ascending=[True, False]).to_string(index=False) if not metrics.empty else "no metrics")


if __name__ == "__main__":
    main()

