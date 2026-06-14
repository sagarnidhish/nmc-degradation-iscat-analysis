#!/usr/bin/env python3
"""Bridge cycle-state/echem transitions to ROI degradation-mode frequencies.

This audit collapses automatic ROI mode assignments to cycle-level mode
fractions, then asks whether cycle-state and within-cycle echem descriptors
predict held-out cycle mode composition beyond cycle/acquisition context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fmt(value: Any, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


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
        return pd.DataFrame()
    return pd.read_csv(path)


def available_numeric(df: pd.DataFrame, cols: Iterable[str], min_nonnull: int = 8) -> List[str]:
    out: List[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= min_nonnull and vals.nunique(dropna=True) >= 2:
            out.append(col)
    return out


def one_hot_top(df: pd.DataFrame, col: str, top_n: int = 8) -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        return out
    vals = out[col].fillna("missing").astype(str)
    for value in vals.value_counts().head(top_n).index:
        safe = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
        out[f"{col}__{safe}"] = (vals == value).astype(float)
    return out


def build_cycle_mode_table(assignments: pd.DataFrame, cycle_state: pd.DataFrame) -> pd.DataFrame:
    if "mode_label" not in assignments.columns or "cycleNo" not in assignments.columns:
        raise ValueError("mode assignment table must contain cycleNo and mode_label")
    roi = assignments.copy()
    roi["cycleNo"] = pd.to_numeric(roi["cycleNo"], errors="coerce")
    roi = roi.dropna(subset=["cycleNo", "mode_label"])
    roi["cycleNo"] = roi["cycleNo"].astype(int)
    roi["mode_label"] = roi["mode_label"].fillna("missing").astype(str)

    counts = roi.groupby(["cycleNo", "mode_label"]).size().unstack(fill_value=0)
    counts.columns = [f"mode_count__{c}" for c in counts.columns]
    counts = counts.reset_index()
    count_cols = [c for c in counts.columns if c.startswith("mode_count__")]
    counts["n_roi"] = counts[count_cols].sum(axis=1)
    for col in count_cols:
        counts[col.replace("mode_count__", "mode_fraction__")] = counts[col] / counts["n_roi"].replace(0, np.nan)

    extras = []
    for col in ["cohort_role", "event_reference_cycle", "source_stem", "protocol_block_segment", "block_mode", "cycle_bin"]:
        if col in roi.columns:
            if col in {"event_reference_cycle", "protocol_block_segment"}:
                extras.append(roi.groupby("cycleNo")[col].agg(lambda s: pd.to_numeric(s, errors="coerce").median()).rename(col))
            else:
                extras.append(roi.groupby("cycleNo")[col].agg(lambda s: s.astype(str).mode().iloc[0] if not s.astype(str).mode().empty else "missing").rename(col))
    if extras:
        extra_df = pd.concat(extras, axis=1).reset_index()
        counts = counts.merge(extra_df, on="cycleNo", how="left")

    cs = cycle_state.copy()
    cs["cycleNo"] = pd.to_numeric(cs["cycleNo"], errors="coerce")
    cs = cs.dropna(subset=["cycleNo"]).copy()
    cs["cycleNo"] = cs["cycleNo"].astype(int)
    table = counts.merge(cs.drop_duplicates("cycleNo"), on="cycleNo", how="left", suffixes=("", "_cycle_state"))
    for col in ["shape_block_mode", "block_mode", "source_stem", "cohort_role", "cycle_bin"]:
        table = one_hot_top(table, col)
    return table


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    context = available_numeric(
        df,
        [
            "cycleNo",
            "n_frames",
            "frames_percentile",
            "cycle_gap",
            "n_points",
            "echem_shape_points",
            "echem_shape_duration_s",
            "event_reference_cycle",
            "protocol_block_segment",
        ]
        + [
            c
            for c in df.columns
            if c.startswith("shape_block_mode__")
            or c.startswith("block_mode__")
            or c.startswith("source_stem__")
            or c.startswith("cohort_role__")
            or c.startswith("cycle_bin__")
        ],
    )
    cycle_state = available_numeric(
        df,
        [
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
            "cycle_state_cluster",
        ],
    )
    echem = available_numeric(
        df,
        [
            c
            for c in df.columns
            if c.startswith("shape_")
            or c.startswith("all_dq")
            or c.startswith("pos_dq")
            or c.startswith("neg_dq")
            or c.startswith("dqdv")
            or c
            in {
                "capacity_mAh",
                "charge_capacity_mAh",
                "discharge_capacity_mAh",
                "coulombic_efficiency_pct",
                "V_min",
                "V_max",
                "particle_norm_mean",
                "particle_norm_std",
                "mean_abs_delta_prev",
                "delta_std_across_particles",
            }
        ],
    )
    return {
        "context_only": context,
        "cycle_state_only": cycle_state,
        "echem_only": echem,
        "cycle_state_plus_context": sorted(set(cycle_state + context)),
        "echem_plus_context": sorted(set(echem + context)),
        "cycle_state_echem_context": sorted(set(cycle_state + echem + context)),
    }


def ridge_model() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=2.0))


def loo_predictions(df: pd.DataFrame, features: List[str], targets: List[str]) -> pd.DataFrame:
    rows = []
    features = [c for c in features if c not in {"cycleNo"} and c not in targets]
    use = df[["cycleNo"] + features + targets].copy()
    cycles = sorted(pd.to_numeric(use["cycleNo"], errors="coerce").dropna().unique())
    for cycle in cycles:
        test = pd.to_numeric(use["cycleNo"], errors="coerce").eq(cycle)
        train = ~test
        y_train = use.loc[train, targets].apply(pd.to_numeric, errors="coerce")
        if train.sum() < 8 or y_train.notna().sum().sum() == 0:
            continue
        model = ridge_model()
        model.fit(use.loc[train, features], y_train.fillna(0.0))
        pred = model.predict(use.loc[test, features])
        for i, idx in enumerate(use.loc[test].index):
            row: Dict[str, Any] = {"cycleNo": int(use.loc[idx, "cycleNo"])}
            for j, target in enumerate(targets):
                row[f"observed__{target}"] = float(use.loc[idx, target])
                row[f"predicted__{target}"] = float(pred[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


def prediction_metrics(pred: pd.DataFrame, feature_set: str, targets: List[str]) -> pd.DataFrame:
    rows = []
    macro_abs = []
    for target in targets:
        obs_col = f"observed__{target}"
        pred_col = f"predicted__{target}"
        if obs_col not in pred.columns or pred_col not in pred.columns:
            continue
        y = pd.to_numeric(pred[obs_col], errors="coerce")
        p = pd.to_numeric(pred[pred_col], errors="coerce")
        mask = y.notna() & p.notna()
        row = {
            "feature_set": feature_set,
            "target": target,
            "n_eval_cycles": int(mask.sum()),
            "r2": np.nan,
            "mae": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
        if mask.sum() >= 5 and y[mask].nunique(dropna=True) >= 2:
            row["r2"] = float(r2_score(y[mask], p[mask]))
            row["mae"] = float(mean_absolute_error(y[mask], p[mask]))
            rho, sp = spearmanr(y[mask], p[mask])
            row["spearman_rho"] = float(rho)
            row["spearman_p"] = float(sp)
            macro_abs.append(np.abs(y[mask].to_numpy(float) - p[mask].to_numpy(float)))
        rows.append(row)
    if macro_abs:
        joined = np.concatenate(macro_abs)
        rows.append(
            {
                "feature_set": feature_set,
                "target": "macro_mode_fraction",
                "n_eval_cycles": int(len(pred)),
                "r2": np.nan,
                "mae": float(np.mean(joined)),
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
            }
        )
    return pd.DataFrame(rows)


def permutation_null(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    targets: List[str],
    observed_metrics: pd.DataFrame,
    rng: np.random.Generator,
    n_perm: int,
) -> pd.DataFrame:
    rows = []
    obs_macro = {
        row["feature_set"]: row
        for row in observed_metrics[observed_metrics["target"] == "macro_mode_fraction"].to_dict("records")
    }
    for name, features in feature_sets.items():
        if not features or name not in obs_macro:
            continue
        null_mae = []
        for _ in range(n_perm):
            shuffled = df.copy()
            shuffled[targets] = shuffled[targets].sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1_000_000_000))).to_numpy()
            pred = loo_predictions(shuffled, features, targets)
            met = prediction_metrics(pred, name, targets)
            macro = met[met["target"] == "macro_mode_fraction"]
            if not macro.empty and np.isfinite(float(macro.iloc[0]["mae"])):
                null_mae.append(float(macro.iloc[0]["mae"]))
        if null_mae:
            arr = np.asarray(null_mae)
            obs = float(obs_macro[name]["mae"])
            rows.append(
                {
                    "feature_set": name,
                    "observed_macro_mae": obs,
                    "n_permutation": int(len(arr)),
                    "null_mae_mean": float(arr.mean()),
                    "null_mae_p05": float(np.quantile(arr, 0.05)),
                    "empirical_p_le_observed_mae": float((np.sum(arr <= obs) + 1) / (len(arr) + 1)),
                }
            )
    return pd.DataFrame(rows).sort_values("empirical_p_le_observed_mae") if rows else pd.DataFrame()


def state_cluster_mode_summary(df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    if "cycle_state_cluster" not in df.columns:
        return pd.DataFrame()
    rows = []
    for cluster, grp in df.groupby("cycle_state_cluster", dropna=True):
        row: Dict[str, Any] = {
            "cycle_state_cluster": int(cluster),
            "n_cycles": int(len(grp)),
            "total_roi": int(pd.to_numeric(grp["n_roi"], errors="coerce").sum()),
            "median_cycle": float(pd.to_numeric(grp["cycleNo"], errors="coerce").median()),
        }
        for target in targets:
            row[f"mean_{target}"] = float(pd.to_numeric(grp[target], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("total_roi", ascending=False) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/cycle_state_mode_frequency_bridge")
    parser.add_argument("--n-permutation", type=int, default=200)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    assignments = read_csv(derived / "cycle_region_mode_context" / "cycle_region_mode_assignments.csv")
    if assignments.empty:
        assignments = read_csv(derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv")
    cycle_state = read_csv(derived / "cycle_state_space_transition_audit" / "cycle_state_space_table.csv")
    if assignments.empty or cycle_state.empty:
        raise FileNotFoundError("mode assignments and cycle_state_space_table are required")

    table = build_cycle_mode_table(assignments, cycle_state)
    targets = [c for c in table.columns if c.startswith("mode_fraction__")]
    targets = [c for c in targets if pd.to_numeric(table[c], errors="coerce").sum() > 0]
    feature_sets = {k: v for k, v in build_feature_sets(table).items() if v}

    prediction_frames = []
    metric_frames = []
    for name, features in feature_sets.items():
        pred = loo_predictions(table, features, targets)
        if pred.empty:
            continue
        pred.insert(0, "feature_set", name)
        prediction_frames.append(pred)
        metric_frames.append(prediction_metrics(pred, name, targets))
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    nulls = permutation_null(table, feature_sets, targets, metrics, rng, args.n_permutation)
    cluster_summary = state_cluster_mode_summary(table, targets)

    metrics_sorted = metrics.copy()
    if not metrics_sorted.empty:
        metrics_sorted["sort_mae"] = pd.to_numeric(metrics_sorted["mae"], errors="coerce")
        metrics_sorted["sort_rho"] = pd.to_numeric(metrics_sorted["spearman_rho"], errors="coerce").abs()
        metrics_sorted = metrics_sorted.sort_values(["target", "sort_mae", "sort_rho"], ascending=[True, True, False]).drop(columns=["sort_mae", "sort_rho"])

    table_path = out / "cycle_state_mode_frequency_table.csv"
    pred_path = out / "cycle_state_mode_frequency_predictions.csv"
    metrics_path = out / "cycle_state_mode_frequency_model_metrics.csv"
    null_path = out / "cycle_state_mode_frequency_permutation_null.csv"
    cluster_path = out / "cycle_state_mode_frequency_cluster_summary.csv"
    table.to_csv(table_path, index=False)
    predictions.to_csv(pred_path, index=False)
    metrics_sorted.to_csv(metrics_path, index=False)
    nulls.to_csv(null_path, index=False)
    cluster_summary.to_csv(cluster_path, index=False)

    macro = metrics[metrics["target"] == "macro_mode_fraction"].sort_values("mae") if not metrics.empty else pd.DataFrame()
    best_macro = macro.head(1).to_dict("records")[0] if not macro.empty else {}
    context_macro = macro[macro["feature_set"] == "context_only"].head(1).to_dict("records")
    best_vs_context_delta = None
    if best_macro and context_macro:
        best_vs_context_delta = float(context_macro[0]["mae"] - best_macro["mae"])

    summary = {
        "n_cycles": int(table["cycleNo"].nunique()),
        "n_roi_rows": int(pd.to_numeric(table["n_roi"], errors="coerce").sum()),
        "n_mode_targets": int(len(targets)),
        "mode_targets": targets,
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "best_macro_model": clean_json(best_macro),
        "context_macro_model": clean_json(context_macro[0]) if context_macro else {},
        "best_minus_context_macro_mae_reduction": best_vs_context_delta,
        "top_metrics": clean_json(metrics_sorted.head(30).to_dict("records")) if not metrics_sorted.empty else [],
        "permutation_null": clean_json(nulls.to_dict("records")) if not nulls.empty else [],
        "cluster_summary": clean_json(cluster_summary.to_dict("records")) if not cluster_summary.empty else [],
        "guardrail": "Cycle-state mode-frequency bridge predicts automatic ROI mode composition at cycle resolution from cycle/echem descriptors. It is a degradation-mode organization audit, not manual QC, causal proof, or calibrated diffusion validation.",
        "outputs": {
            "cycle_mode_table": str(table_path),
            "predictions": str(pred_path),
            "metrics": str(metrics_path),
            "permutation_null": str(null_path),
            "cluster_summary": str(cluster_path),
            "summary": str(out / "cycle_state_mode_frequency_bridge_summary.json"),
        },
    }
    with (out / "cycle_state_mode_frequency_bridge_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)

    lines = [
        "# Cycle-State Mode-Frequency Bridge",
        "",
        "Cycle-level bridge from echem/cycle-state transition descriptors to automatic ROI degradation-mode composition.",
        "",
        f"- Cycles: {summary['n_cycles']}",
        f"- ROI rows represented: {summary['n_roi_rows']}",
        f"- Mode-frequency targets: {summary['n_mode_targets']}",
        f"- Best macro model: {best_macro.get('feature_set', 'NA')} MAE {best_macro.get('mae', np.nan):.3f}",
        f"- Context macro MAE reduction: {best_vs_context_delta if best_vs_context_delta is not None else np.nan:.3f}",
        "",
        "## Feature Sets",
        "",
    ]
    for name, cols in feature_sets.items():
        lines.append(f"- {name}: {len(cols)} features")
    lines += ["", "## Top Metrics", ""]
    for row in summary["top_metrics"][:12]:
        lines.append(
            f"- {row.get('feature_set')} -> {row.get('target')}: MAE {fmt(row.get('mae'))}, R2 {fmt(row.get('r2'))}, rho {fmt(row.get('spearman_rho'))}"
        )
    lines += ["", "## Permutation Null", ""]
    for row in summary["permutation_null"][:8]:
        lines.append(
            f"- {row.get('feature_set')}: observed macro MAE {fmt(row.get('observed_macro_mae'))}, null mean {fmt(row.get('null_mae_mean'))}, p={fmt(row.get('empirical_p_le_observed_mae'))}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
