#!/usr/bin/env python3
"""Predict future drop risk from combined rollout and phase-kinetics features.

This is a source-held-out classification audit over the source-balanced
pre-event ROI cohort. It compares:

- rollout-only features,
- phase-kinetics-only features,
- combined rollout + phase-kinetics features.

Goal:
- test whether the phase-kinetics signal improves future-event prediction beyond
  pure ROI rollout difficulty;
- keep the analysis particle-region-only and source-held-out.

Guardrail:
- This is a weak-label ranking/association experiment, not a deployable warning
  system or a calibrated degradation classifier.
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


def make_model(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=4000, solver="liblinear", class_weight="balanced", C=0.35, random_state=seed),
    )


def evaluate(df: pd.DataFrame, feature_cols: List[str], target: str, seed: int = 0) -> Dict[str, Any]:
    sub = df[df[target].isin([0, 1])].copy()
    sub = sub[sub["source_stem"].notna()]
    if sub.empty:
        return {"target": target, "feature_count": len(feature_cols), "n": 0}
    X = sub[feature_cols]
    y = sub[target].astype(int).values
    groups = sub["source_stem"].astype(str).values
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
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
    ap.add_argument("--rollout-csv", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_sequence_rollout_audit/source_balanced_sequence_rollout_features.csv")
    ap.add_argument("--phase-csv", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_phase_kinetics_audit/source_balanced_pre_event_phase_kinetics_features.csv")
    ap.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/rollout_phase_kinetics_predictor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout = pd.read_csv(args.rollout_csv)
    phase = pd.read_csv(args.phase_csv)
    merged = rollout.merge(
        phase,
        on=["roi_id", "cycleNo", "source_stem", "local_cycle_index", "expansion_cycle_rank", "object_candidate_rank"],
        how="inner",
        suffixes=("_rollout", "_phase"),
    )

    rollout_features = [
        c for c in merged.columns
        if c.startswith("persistence_mse_")
        or c.startswith("velocity_mse_")
        or c.startswith("temporal_energy_")
        or c.startswith("roi_norm_mean_")
        or c in ["raw_roi_mean_delta_last_minus_first", "stage_drift_xy_recomputed"]
    ]
    phase_features = [
        c for c in merged.columns
        if "masked_minus_bg" in c
        or "phase_fraction" in c
        or "avrami" in c
        or "logistic" in c
        or "front_gradient" in c
        or "max_abs_rate" in c
        or "total_variation" in c
    ]
    rollout_features = [c for c in rollout_features if c in merged.columns]
    phase_features = [c for c in phase_features if c in merged.columns]
    combined_features = sorted(set(rollout_features + phase_features))

    targets = [t for t in ["future_any_drop_within_8cycles", "future_any_drop_within_16cycles"] if t in merged.columns]
    rows = []
    for target in targets:
        rows.append({"feature_family": "rollout_only", **evaluate(merged, rollout_features, target, seed=args.seed)})
        rows.append({"feature_family": "phase_only", **evaluate(merged, phase_features, target, seed=args.seed)})
        rows.append({"feature_family": "combined", **evaluate(merged, combined_features, target, seed=args.seed)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "rollout_phase_kinetics_predictor_metrics.csv", index=False)
    (out_dir / "rollout_phase_kinetics_predictor_summary.json").write_text(json.dumps(clean_json({
        "n_rows": int(len(merged)),
        "n_sources": int(merged["source_stem"].nunique()) if not merged.empty else 0,
        "n_targets": int(len(targets)),
        "feature_counts": {"rollout_only": len(rollout_features), "phase_only": len(phase_features), "combined": len(combined_features)},
        "best_rows": metrics.sort_values(["target", "oriented_auc", "average_precision"], ascending=[True, False, False]).head(6).to_dict("records"),
        "outputs": {
            "metrics": str(out_dir / "rollout_phase_kinetics_predictor_metrics.csv"),
        },
        "guardrail": "Source-held-out weak-label classifier over automatic particle-only features. Useful for ranking whether phase-kinetics adds event-risk signal beyond rollout residuals, not a deployable predictor.",
    }), indent=2, sort_keys=True))
    (out_dir / "README.md").write_text(
        "\n".join([
            "# Rollout / Phase Kinetics Predictor",
            "",
            f"- Rows: {len(merged)}",
            f"- Sources: {merged['source_stem'].nunique() if not merged.empty else 0}",
            f"- Feature counts: {{'rollout_only': {len(rollout_features)}, 'phase_only': {len(phase_features)}, 'combined': {len(combined_features)}}}",
            "",
            "## Guardrail",
            "",
            "Source-held-out weak-label classifier over automatic particle-only features. Useful for ranking whether phase-kinetics adds event-risk signal beyond rollout residuals, not a deployable predictor.",
        ]) + "\n"
    )
    print(metrics.sort_values(["target", "oriented_auc"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()

