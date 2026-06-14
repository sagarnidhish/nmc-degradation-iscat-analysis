#!/usr/bin/env python3
"""Source-balanced pre-event physical-observable forecast audit.

This tests a narrower and more physical target than pixel rollout: can early
particle-crop observables forecast held-out-tail optical observables under
source holdout, and do they add value beyond acquisition/echem context?

The output is a guarded AI/physics audit. It is not a deployable predictor,
manual particle label, calibrated phase-boundary tracker, or diffusion claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = [
    "tail_particle_mean_delta",
    "tail_particle_minus_background_delta",
    "tail_contrast_delta",
    "tail_front_radius_q70_delta",
    "tail_front_radius2_slope",
    "tail_frame_diff_energy",
]

ECHEM_COLS = [
    "capacity_fraction_of_first",
    "coulombic_inefficiency_pct",
    "echem_regime_pc1",
    "echem_regime_pc2",
    "echem_regime_pc3",
    "echem_regime_pc4",
    "shape_V_mean",
    "shape_V_std",
    "shape_I_abs_mean_mA",
    "shape_dVdt_abs_p95",
    "all_dq_abs_entropy",
    "all_dq_abs_peak_voltage",
]

CONTEXT_NUMERIC = [
    "cycleNo",
    "local_cycle_index",
    "expansion_cycle_rank",
    "object_candidate_rank",
    "object_x_full_approx",
    "object_y_full_approx",
    "object_area_ds_px",
    "object_mean_abs_z",
    "crop_x0",
    "crop_y0",
    "n_frames",
    "first_frame_index",
    "last_frame_index",
    "stage_drift_xy_sampled",
    "cycles_to_next_event",
    "cycles_since_prev_event",
]

CONTEXT_CATEGORICAL = ["event_relative_bin", "selection_reason", "already_in_existing_video_cohort"]


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


def numeric_cols(df: pd.DataFrame, cols: Iterable[str], min_count: int = 12) -> List[str]:
    keep: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_count and vals.nunique(dropna=True) >= 2:
            keep.append(col)
    return keep


def stable_mask(history: np.ndarray) -> np.ndarray:
    base = np.nanmedian(history, axis=0)
    high = base >= np.nanpercentile(base, 70)
    low = base <= np.nanpercentile(base, 30)
    mask = high if 0.03 <= high.mean() <= 0.65 else low
    if mask.mean() < 0.03 or mask.mean() > 0.75:
        yy, xx = np.indices(base.shape)
        cy, cx = (np.array(base.shape) - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = rr <= 0.38 * min(base.shape)
    return mask.astype(bool)


def front_radius(frame: np.ndarray, mask: np.ndarray, q: float = 70.0) -> float:
    yy, xx = np.indices(frame.shape)
    if mask.any():
        cy = float(yy[mask].mean())
        cx = float(xx[mask].mean())
    else:
        cy, cx = (np.array(frame.shape) - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    center = np.nanmedian(frame[mask]) if mask.any() else np.nanmedian(frame)
    bright = frame >= np.nanpercentile(frame, q)
    edge = bright & mask if (bright & mask).sum() >= 4 else bright
    if edge.sum() < 4:
        return np.nan
    # Direction-agnostic apparent front extent: radius of high-quantile optical
    # structure inside the history-defined particle support.
    return float(np.nanmedian(rr[edge]))


def slope(values: Sequence[float]) -> float:
    y = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = np.linspace(0.0, 1.0, len(y))[ok]
    yy = y[ok]
    if np.nanstd(x) == 0:
        return np.nan
    return float(np.polyfit(x, yy, 1)[0])


def frame_observables(frames: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    bg = ~mask
    rows: List[Dict[str, float]] = []
    for i, fr in enumerate(frames):
        part = fr[mask] if mask.any() else fr.ravel()
        back = fr[bg] if bg.any() else fr.ravel()
        rows.append(
            {
                "step": float(i),
                "particle_mean": float(np.nanmean(part)),
                "background_mean": float(np.nanmean(back)),
                "particle_minus_background": float(np.nanmean(part) - np.nanmean(back)),
                "contrast": float(np.nanstd(part)),
                "front_radius_q70": front_radius(fr, mask, 70.0),
            }
        )
    out = pd.DataFrame(rows)
    out["front_radius2_q70"] = out["front_radius_q70"] ** 2
    return out


def extract_one(row: pd.Series, prefix_fraction: float) -> Dict[str, Any]:
    data = np.load(str(row["npz_path"]))
    frames = np.asarray(data["frames_norm"] if "frames_norm" in data else data["frames"], dtype=float)
    if frames.ndim != 3 or frames.shape[0] < 16:
        raise ValueError(f"bad sequence shape {frames.shape}")
    split = int(np.clip(round(frames.shape[0] * prefix_fraction), 8, frames.shape[0] - 4))
    history = frames[:split]
    tail = frames[split:]
    mask = stable_mask(history)
    obs = frame_observables(frames, mask)
    prefix = obs.iloc[:split]
    future = obs.iloc[split:]
    rec: Dict[str, Any] = row.to_dict()
    rec.update(
        {
            "n_frames_loaded": int(frames.shape[0]),
            "prefix_n": int(split),
            "tail_n": int(len(future)),
            "history_mask_fraction": float(mask.mean()),
            "prefix_particle_mean_last": float(prefix["particle_mean"].iloc[-1]),
            "prefix_particle_mean_slope": slope(prefix["particle_mean"]),
            "prefix_particle_minus_background_last": float(prefix["particle_minus_background"].iloc[-1]),
            "prefix_particle_minus_background_slope": slope(prefix["particle_minus_background"]),
            "prefix_contrast_last": float(prefix["contrast"].iloc[-1]),
            "prefix_contrast_slope": slope(prefix["contrast"]),
            "prefix_front_radius_q70_last": float(prefix["front_radius_q70"].iloc[-1]),
            "prefix_front_radius_q70_slope": slope(prefix["front_radius_q70"]),
            "prefix_front_radius2_q70_slope": slope(prefix["front_radius2_q70"]),
            "tail_particle_mean_delta": float(future["particle_mean"].iloc[-1] - prefix["particle_mean"].iloc[-1]),
            "tail_particle_minus_background_delta": float(future["particle_minus_background"].iloc[-1] - prefix["particle_minus_background"].iloc[-1]),
            "tail_contrast_delta": float(future["contrast"].iloc[-1] - prefix["contrast"].iloc[-1]),
            "tail_front_radius_q70_delta": float(future["front_radius_q70"].iloc[-1] - prefix["front_radius_q70"].iloc[-1]),
            "tail_front_radius2_slope": slope(future["front_radius2_q70"]),
            "tail_frame_diff_energy": float(np.nanmean(np.diff(tail, axis=0) ** 2)) if len(tail) > 1 else np.nan,
        }
    )
    rec["persistence_pred_tail_particle_mean_delta"] = 0.0
    rec["persistence_pred_tail_particle_minus_background_delta"] = 0.0
    rec["persistence_pred_tail_contrast_delta"] = 0.0
    rec["persistence_pred_tail_front_radius_q70_delta"] = 0.0
    rec["linear_pred_tail_particle_mean_delta"] = float(rec["prefix_particle_mean_slope"])
    rec["linear_pred_tail_particle_minus_background_delta"] = float(rec["prefix_particle_minus_background_slope"])
    rec["linear_pred_tail_contrast_delta"] = float(rec["prefix_contrast_slope"])
    rec["linear_pred_tail_front_radius_q70_delta"] = float(rec["prefix_front_radius_q70_slope"])
    rec["linear_pred_tail_front_radius2_slope"] = float(rec["prefix_front_radius2_q70_slope"])
    return rec


def build_model(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> Pipeline:
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", num_pipe, list(numeric_features)),
            ("cat", cat_pipe, list(categorical_features)),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])


def grouped_oof(df: pd.DataFrame, target: str, numeric_features: Sequence[str], categorical_features: Sequence[str], group_col: str) -> pd.Series:
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    y = pd.to_numeric(df[target], errors="coerce")
    groups = df[group_col].astype(str)
    for group in sorted(groups.dropna().unique()):
        test = groups == group
        train = ~test & y.notna()
        if train.sum() < 16 or y[train].nunique() < 3:
            continue
        model = build_model(numeric_features, categorical_features)
        model.fit(df.loc[train, list(numeric_features) + list(categorical_features)], y.loc[train])
        pred.loc[test] = model.predict(df.loc[test, list(numeric_features) + list(categorical_features)])
    return pred


def eval_prediction(y: pd.Series, pred: pd.Series, label: str, target: str, group_col: str) -> Dict[str, Any]:
    yy = pd.to_numeric(y, errors="coerce")
    pp = pd.to_numeric(pred, errors="coerce")
    valid = yy.notna() & pp.notna()
    if valid.sum() < 8:
        return {"target": target, "model": label, "group_col": group_col, "n": int(valid.sum())}
    rho = spearmanr(yy[valid], pp[valid])
    return {
        "target": target,
        "model": label,
        "group_col": group_col,
        "n": int(valid.sum()),
        "mae": float(mean_absolute_error(yy[valid], pp[valid])),
        "r2": float(r2_score(yy[valid], pp[valid])),
        "spearman_rho": float(rho.statistic) if np.isfinite(rho.statistic) else np.nan,
        "spearman_p": float(rho.pvalue) if np.isfinite(rho.pvalue) else np.nan,
    }


def source_centered_auc_like(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    # Directional event-relative diagnostic for forecast targets, not model
    # selection. It asks whether target magnitude is organized near events after
    # subtracting source means.
    ybin = df["event_relative_bin"].astype(str).eq("near_pre_event_1_8")
    x = pd.to_numeric(df[target], errors="coerce")
    x = x - x.groupby(df["source_stem"].astype(str)).transform("mean")
    valid = ybin.notna() & x.notna()
    if valid.sum() < 12 or ybin[valid].nunique() < 2:
        return {"target": target, "n": int(valid.sum())}
    pos = x[valid & ybin]
    neg = x[valid & ~ybin]
    direction = "higher_near_pre" if pos.median() >= neg.median() else "lower_near_pre"
    score = x[valid] if direction == "higher_near_pre" else -x[valid]
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "target": target,
        "n": int(valid.sum()),
        "n_near_pre": int(ybin[valid].sum()),
        "direction": direction,
        "oriented_auc": float(roc_auc_score(ybin[valid].astype(int), score)),
        "average_precision": float(average_precision_score(ybin[valid].astype(int), score)),
        "source_centered_median_near_minus_other": float(pos.median() - neg.median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/source_balanced_pre_event_observable_forecast")
    parser.add_argument("--prefix-fraction", type=float, default=0.6)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined_path = derived / "source_balanced_pre_event_echem_front_coupling_audit" / "source_balanced_pre_event_echem_front_joined.csv"
    if joined_path.exists():
        table = pd.read_csv(joined_path).loc[:, lambda d: ~d.columns.duplicated()].copy()
    else:
        table = pd.read_csv(derived / "source_balanced_pre_event_roi_sequences" / "selected_roi_sequence_manifest.csv")

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for _, row in table.iterrows():
        try:
            rows.append(extract_one(row, args.prefix_fraction))
        except Exception as exc:
            failures.append({"roi_id": row.get("roi_id", ""), "error": str(exc)})
    obs = pd.DataFrame(rows)
    fail = pd.DataFrame(failures)

    prefix_features = numeric_cols(
        obs,
        [
            "history_mask_fraction",
            "prefix_particle_mean_last",
            "prefix_particle_mean_slope",
            "prefix_particle_minus_background_last",
            "prefix_particle_minus_background_slope",
            "prefix_contrast_last",
            "prefix_contrast_slope",
            "prefix_front_radius_q70_last",
            "prefix_front_radius_q70_slope",
            "prefix_front_radius2_q70_slope",
        ],
    )
    context_features = numeric_cols(obs, CONTEXT_NUMERIC)
    echem_features = numeric_cols(obs, ECHEM_COLS)
    categorical = [c for c in CONTEXT_CATEGORICAL if c in obs.columns]
    feature_sets = {
        "prefix_observables": (prefix_features, []),
        "context": (context_features, categorical),
        "echem_context": (context_features + echem_features, categorical),
        "prefix_plus_echem_context": (prefix_features + context_features + echem_features, categorical),
    }

    metrics: List[Dict[str, Any]] = []
    preds = obs[["roi_id", "cycleNo", "source_stem", "event_relative_bin"]].copy()
    for target in [t for t in TARGETS if t in obs.columns]:
        y = pd.to_numeric(obs[target], errors="coerce")
        for group_col in ["source_stem", "cycleNo"]:
            groups = obs[group_col].astype(str)
            baseline = y.groupby(groups).transform(lambda s: np.nan)
            global_median = float(y.median()) if y.notna().any() else np.nan
            metrics.append(eval_prediction(y, pd.Series(global_median, index=obs.index), "global_median", target, group_col))
            persist_col = f"persistence_pred_{target}"
            linear_col = f"linear_pred_{target}"
            if persist_col in obs.columns:
                metrics.append(eval_prediction(y, obs[persist_col], "observable_persistence", target, group_col))
            if linear_col in obs.columns:
                metrics.append(eval_prediction(y, obs[linear_col], "prefix_linear_extrapolation", target, group_col))
            for name, (num, cat) in feature_sets.items():
                if not num and not cat:
                    continue
                pred = grouped_oof(obs, target, num, cat, group_col)
                preds[f"{target}__{name}__loo_{group_col}"] = pred
                metrics.append(eval_prediction(y, pred, name, target, group_col))

    metric_df = pd.DataFrame(metrics)
    if not metric_df.empty:
        metric_df = metric_df.sort_values(["target", "group_col", "mae"], na_position="last")
    event_diag = pd.DataFrame([source_centered_auc_like(obs, t) for t in TARGETS if t in obs.columns])
    best = metric_df.sort_values(["group_col", "spearman_rho"], ascending=[True, False], na_position="last").head(24)
    source_best = metric_df[metric_df["group_col"].eq("source_stem")].sort_values("spearman_rho", ascending=False, na_position="last").head(12)
    incremental: List[Dict[str, Any]] = []
    for target in [t for t in TARGETS if t in obs.columns]:
        for group_col in ["source_stem", "cycleNo"]:
            sub = metric_df[(metric_df["target"].eq(target)) & (metric_df["group_col"].eq(group_col))]
            echem = sub[sub["model"].eq("echem_context")]
            fused = sub[sub["model"].eq("prefix_plus_echem_context")]
            if not echem.empty and not fused.empty:
                incremental.append(
                    {
                        "target": target,
                        "group_col": group_col,
                        "delta_spearman_prefix_plus_echem_minus_echem": float(fused.iloc[0].get("spearman_rho", np.nan) - echem.iloc[0].get("spearman_rho", np.nan)),
                        "delta_mae_prefix_plus_echem_minus_echem": float(fused.iloc[0].get("mae", np.nan) - echem.iloc[0].get("mae", np.nan)),
                    }
                )
    inc_df = pd.DataFrame(incremental).sort_values("delta_spearman_prefix_plus_echem_minus_echem", ascending=False, na_position="last")

    obs_path = out / "source_balanced_pre_event_observable_forecast_features.csv"
    metric_path = out / "source_balanced_pre_event_observable_forecast_metrics.csv"
    pred_path = out / "source_balanced_pre_event_observable_forecast_predictions.csv"
    event_path = out / "source_balanced_pre_event_observable_forecast_event_diagnostics.csv"
    inc_path = out / "source_balanced_pre_event_observable_forecast_incremental.csv"
    fail_path = out / "source_balanced_pre_event_observable_forecast_failures.csv"
    obs.to_csv(obs_path, index=False)
    metric_df.to_csv(metric_path, index=False)
    preds.to_csv(pred_path, index=False)
    event_diag.to_csv(event_path, index=False)
    inc_df.to_csv(inc_path, index=False)
    fail.to_csv(fail_path, index=False)

    decision = {
        "status": "guarded_observable_forecast_audit",
        "best_leave_source_metric": source_best.head(1).to_dict(orient="records")[0] if not source_best.empty else {},
        "best_incremental_over_echem": inc_df.head(1).to_dict(orient="records")[0] if not inc_df.empty else {},
        "interpretation": "Prefix observables are useful only if leave-source forecasts beat context/echem baselines and remain interpretable; otherwise they are review descriptors.",
    }
    summary = {
        "n_rows": int(len(obs)),
        "n_failures": int(len(fail)),
        "n_cycles": int(pd.to_numeric(obs.get("cycleNo"), errors="coerce").nunique()) if not obs.empty else 0,
        "n_sources": int(obs.get("source_stem", pd.Series(dtype=str)).astype(str).nunique()) if not obs.empty else 0,
        "prefix_fraction": float(args.prefix_fraction),
        "targets": [t for t in TARGETS if t in obs.columns],
        "feature_set_sizes": {k: {"numeric": len(v[0]), "categorical": len(v[1])} for k, v in feature_sets.items()},
        "decision": decision,
        "top_metrics": best.to_dict(orient="records"),
        "source_heldout_top_metrics": source_best.to_dict(orient="records"),
        "incremental_over_echem": inc_df.head(24).to_dict(orient="records"),
        "event_relative_diagnostics": event_diag.to_dict(orient="records"),
        "guardrail": "This forecasts held-out-tail optical observables from automatic source-balanced particle crops. It does not validate particle identity, phase-boundary motion, diffusion coefficients, or degradation causality.",
        "outputs": {
            "features": str(obs_path),
            "metrics": str(metric_path),
            "predictions": str(pred_path),
            "event_diagnostics": str(event_path),
            "incremental": str(inc_path),
            "failures": str(fail_path),
        },
    }
    with (out / "source_balanced_pre_event_observable_forecast_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)
    with (out / "README.md").open("w") as f:
        f.write("# Source-Balanced Pre-Event Observable Forecast\n\n")
        f.write("Forecasts held-out-tail physical observables from early particle-crop observables, with leave-source and leave-cycle comparisons to context/echem baselines.\n\n")
        f.write(f"- Rows/cycles/sources: {summary['n_rows']} / {summary['n_cycles']} / {summary['n_sources']}\n")
        f.write(f"- Prefix fraction: {summary['prefix_fraction']}\n")
        f.write(f"- Guardrail: {summary['guardrail']}\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True)[:12000])


if __name__ == "__main__":
    main()
