#!/usr/bin/env python3
"""Cycle and particle-region context for residual NMC physics modes.

This is a compact follow-up to the residual physics mode taxonomy. It asks
whether the protocol-adjusted modes are concentrated by cycle, event-reference
cycle, coarse particle image region, or remaining electrochemical/protocol axes.
The output is descriptive and audit-oriented, not a deployable classifier.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


EVENT_MODE = "optical_brightening_decorrelating_rollout_hard_front_positive"


def qcut_labels(series: pd.Series, prefix: str, q: int = 3) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        bins = pd.qcut(numeric, q=q, labels=[f"{prefix}{i+1}" for i in range(q)], duplicates="drop")
        return bins.astype("object").fillna(f"{prefix}NA")
    except ValueError:
        return pd.Series([f"{prefix}NA"] * len(series), index=series.index)


def safe_float(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def top_records(df: pd.DataFrame, n: int = 8) -> List[Dict[str, Any]]:
    return df.head(n).replace({np.nan: None}).to_dict("records")


def add_region_context(assignments: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    context_cols = [
        "roi_id",
        "source_stem",
        "object_x_full_approx",
        "object_y_full_approx",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "front_candidate_rank",
        "object_candidate_rank",
        "cycles_from_block_start",
        "cycles_to_block_end",
        "block_fraction_elapsed",
        "block_mode",
        "protocol_block_segment",
        "near_block_boundary_10cycles",
        "V_min",
        "V_max",
        "V_range",
        "I_abs_mean_mA",
        "I_pos_fraction",
        "I_neg_fraction",
        "V_mean_delta",
        "I_mean_delta",
        "V_range_delta",
        "echem_points",
        "duration_s",
    ]
    keep = [c for c in context_cols if c in joined.columns and c not in assignments.columns]
    df = assignments.merge(joined[["roi_id"] + keep].drop_duplicates("roi_id"), on="roi_id", how="left")
    df["x_region"] = qcut_labels(df.get("object_x_full_approx"), "x")
    df["y_region"] = qcut_labels(df.get("object_y_full_approx"), "y")
    df["xy_region"] = df["x_region"].astype(str) + "_" + df["y_region"].astype(str)
    df["cycle_bin"] = qcut_labels(df.get("cycleNo"), "cycle")
    df["is_event_enriched_mode"] = (df["mode_label"] == EVENT_MODE).astype(int)
    df["is_high_apparent_front_proxy_mode"] = df["mode_label"].astype(str).str.contains("front_proxy", regex=False).astype(int)
    return df


def summarize_group(df: pd.DataFrame, group_cols: Iterable[str], out_col: str) -> pd.DataFrame:
    group_cols = list(group_cols)
    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_roi": int(len(grp)),
            "n_event": int((grp["cohort_role"] == "event").sum()),
            "event_fraction": float((grp["cohort_role"] == "event").mean()),
            "event_enriched_mode_fraction": float(grp["is_event_enriched_mode"].mean()),
            "front_proxy_mode_fraction": float(grp["is_high_apparent_front_proxy_mode"].mean()),
            "mean_mode_review_priority": safe_float(pd.to_numeric(grp.get("mode_review_priority"), errors="coerce").mean()),
            "mean_cycle": safe_float(pd.to_numeric(grp.get("cycleNo"), errors="coerce").mean()),
            "mean_x": safe_float(pd.to_numeric(grp.get("object_x_full_approx"), errors="coerce").mean()),
            "mean_y": safe_float(pd.to_numeric(grp.get("object_y_full_approx"), errors="coerce").mean()),
            "mean_V": safe_float(pd.to_numeric(grp.get("V_mean"), errors="coerce").mean()),
            "mean_I_mA": safe_float(pd.to_numeric(grp.get("I_mean_mA"), errors="coerce").mean()),
            "top_modes": ";".join(grp["mode_label"].value_counts().head(3).index.astype(str).tolist()),
            "top_roi_ids": ";".join(grp.sort_values("mode_review_priority", ascending=False)["roi_id"].head(5).astype(str).tolist()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values([out_col, "n_roi"], ascending=[False, False])


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    targets = [
        "is_event_enriched_mode",
        "is_high_apparent_front_proxy_mode",
        "mode_review_priority",
        "mode_pc1",
        "mode_pc2",
    ]
    context = [
        "cycleNo",
        "event_reference_cycle",
        "object_x_full_approx",
        "object_y_full_approx",
        "front_candidate_rank",
        "object_candidate_rank",
        "n_frames_percentile",
        "V_mean",
        "V_range",
        "I_mean_mA",
        "I_abs_mean_mA",
        "I_pos_fraction",
        "cycles_from_block_start",
        "cycles_to_block_end",
        "block_fraction_elapsed",
    ]
    rows = []
    for target in targets:
        y = pd.to_numeric(df.get(target), errors="coerce")
        for feature in context:
            if feature not in df.columns:
                continue
            x = pd.to_numeric(df[feature], errors="coerce")
            mask = x.notna() & y.notna()
            if int(mask.sum()) < 6 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                continue
            rho, p = spearmanr(x[mask], y[mask])
            rows.append({
                "target": target,
                "context_feature": feature,
                "n": int(mask.sum()),
                "rho": safe_float(rho),
                "p_value": safe_float(p),
                "abs_rho": safe_float(abs(rho)),
            })
    return pd.DataFrame(rows).sort_values(["abs_rho", "p_value"], ascending=[False, True])


def categorical_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ["xy_region", "x_region", "y_region", "cycle_bin", "event_reference_cycle", "block_mode", "protocol_block_segment"]:
        if feature not in df.columns:
            continue
        tab = pd.crosstab(df[feature].astype(str), df["is_event_enriched_mode"])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(tab)
        rows.append({
            "feature": feature,
            "test": "chi2_mode_enriched",
            "n_levels": int(tab.shape[0]),
            "statistic": safe_float(chi2),
            "dof": int(dof),
            "p_value": safe_float(p),
            "table": tab.to_json(),
        })
    for feature in ["xy_region", "x_region", "y_region", "cycle_bin", "event_reference_cycle"]:
        if feature not in df.columns:
            continue
        for level, grp in df.groupby(feature, dropna=False):
            inside_pos = int(grp["is_event_enriched_mode"].sum())
            inside_neg = int(len(grp) - inside_pos)
            outside = df.loc[grp.index.symmetric_difference(df.index)]
            outside_pos = int(outside["is_event_enriched_mode"].sum())
            outside_neg = int(len(outside) - outside_pos)
            if min(inside_pos + inside_neg, outside_pos + outside_neg) == 0:
                continue
            _, p = fisher_exact([[inside_pos, inside_neg], [outside_pos, outside_neg]])
            rows.append({
                "feature": feature,
                "level": str(level),
                "test": "fisher_level_vs_rest",
                "n_inside": int(len(grp)),
                "inside_event_enriched_fraction": safe_float(inside_pos / len(grp)),
                "outside_event_enriched_fraction": safe_float(outside_pos / len(outside)) if len(outside) else None,
                "p_value": safe_float(p),
            })
    return pd.DataFrame(rows).sort_values("p_value")


def context_classifier(df: pd.DataFrame) -> Dict[str, Any]:
    features = [
        "cycleNo",
        "object_x_full_approx",
        "object_y_full_approx",
        "front_candidate_rank",
        "object_candidate_rank",
        "n_frames_percentile",
        "V_mean",
        "V_range",
        "I_mean_mA",
        "I_abs_mean_mA",
        "cycles_from_block_start",
        "block_fraction_elapsed",
    ]
    features = [f for f in features if f in df.columns]
    work_cols = list(dict.fromkeys(["is_event_enriched_mode", "cycleNo"] + features))
    work = df[work_cols].copy()
    y = work["is_event_enriched_mode"].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(features) < 2:
        return {"status": "skipped", "reason": "insufficient labels or features"}
    groups = pd.to_numeric(df["cycleNo"], errors="coerce").fillna(-1).to_numpy()
    x = work[features].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)),
    ])
    rows = []
    probs = np.full(len(y), np.nan)
    preds = np.full(len(y), -1)
    logo = LeaveOneGroupOut()
    for train, test in logo.split(x, y, groups):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        pipe.fit(x.iloc[train], y[train])
        p = pipe.predict_proba(x.iloc[test])[:, 1]
        pred = (p >= 0.5).astype(int)
        probs[test] = p
        preds[test] = pred
        rows.append({
            "heldout_cycle": safe_float(groups[test][0]),
            "n_test": int(len(test)),
            "roc_auc": safe_float(roc_auc_score(y[test], p)),
            "balanced_accuracy": safe_float(balanced_accuracy_score(y[test], pred)),
        })
    valid = np.isfinite(probs) & (preds >= 0)
    result = {
        "status": "ok" if rows else "limited",
        "features": features,
        "folds": rows,
        "mean_fold_roc_auc": safe_float(np.mean([r["roc_auc"] for r in rows])) if rows else None,
        "mean_fold_balanced_accuracy": safe_float(np.mean([r["balanced_accuracy"] for r in rows])) if rows else None,
    }
    if valid.sum() and len(np.unique(y[valid])) > 1:
        result["pooled_roc_auc"] = safe_float(roc_auc_score(y[valid], probs[valid]))
        result["pooled_balanced_accuracy"] = safe_float(balanced_accuracy_score(y[valid], preds[valid]))
        result["n_scored"] = int(valid.sum())
    return result


def write_readme(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Cycle/Region Residual Mode Context",
        "",
        "Cycle, echem, protocol, and coarse particle-region context for protocol-adjusted residual physics modes.",
        "",
        f"- ROI rows: {summary['n_roi']}",
        f"- Cycles: {summary['n_cycles']}",
        f"- Event-enriched mode: `{EVENT_MODE}`",
        f"- Event-enriched mode fraction: {summary['event_enriched_mode_fraction']:.3f}",
        "",
        "## Top Cycle Summaries",
        "",
    ]
    for row in summary["top_cycle_summaries"]:
        lines.append(
            f"- cycle {row.get('cycleNo')}: n={row.get('n_roi')}, event-enriched fraction={row.get('event_enriched_mode_fraction'):.3f}, top modes={row.get('top_modes')}"
        )
    lines += ["", "## Strongest Context Correlations", ""]
    for row in summary["top_context_correlations"]:
        lines.append(
            f"- {row.get('target')} vs {row.get('context_feature')}: rho={row.get('rho'):.3f}, p={row.get('p_value'):.3g}, n={row.get('n')}"
        )
    lines += ["", "## Guardrail", "", summary["guardrail"]]
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/<account>/<username>/Alek_Jiho/derived/cycle_region_mode_context")
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv")
    joined = pd.read_csv(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_joined.csv")
    df = add_region_context(assignments, joined)

    cycle_summary = summarize_group(df, ["cycleNo", "cohort_role"], "event_enriched_mode_fraction")
    cycle_all_summary = summarize_group(df, ["cycleNo"], "event_enriched_mode_fraction")
    region_summary = summarize_group(df, ["xy_region"], "event_enriched_mode_fraction")
    ref_region_summary = summarize_group(df, ["event_reference_cycle", "xy_region"], "event_enriched_mode_fraction")
    correlations = spearman_table(df)
    cat_tests = categorical_tests(df)
    classifier = context_classifier(df)

    enriched = df.sort_values("mode_review_priority", ascending=False)
    paths = {
        "assignments": out / "cycle_region_mode_assignments.csv",
        "cycle_summary": out / "cycle_mode_context_summary.csv",
        "cycle_all_summary": out / "cycle_mode_context_all_roles.csv",
        "region_summary": out / "spatial_region_mode_summary.csv",
        "reference_region_summary": out / "event_reference_region_mode_summary.csv",
        "context_correlations": out / "mode_context_spearman.csv",
        "categorical_tests": out / "mode_context_categorical_tests.csv",
    }
    df.to_csv(paths["assignments"], index=False)
    cycle_summary.to_csv(paths["cycle_summary"], index=False)
    cycle_all_summary.to_csv(paths["cycle_all_summary"], index=False)
    region_summary.to_csv(paths["region_summary"], index=False)
    ref_region_summary.to_csv(paths["reference_region_summary"], index=False)
    correlations.to_csv(paths["context_correlations"], index=False)
    cat_tests.to_csv(paths["categorical_tests"], index=False)

    mode_groups = []
    for mode, grp in df.groupby("mode_label"):
        numeric_groups = [pd.to_numeric(g["cycleNo"], errors="coerce").dropna().to_numpy() for _, g in grp.groupby("cohort_role")]
        kw_p = None
        if len(numeric_groups) >= 2 and all(len(g) >= 2 for g in numeric_groups):
            try:
                kw_p = safe_float(kruskal(*numeric_groups).pvalue)
            except ValueError:
                kw_p = None
        mode_groups.append({
            "mode_label": mode,
            "n_roi": int(len(grp)),
            "n_event": int((grp["cohort_role"] == "event").sum()),
            "mean_cycle": safe_float(pd.to_numeric(grp["cycleNo"], errors="coerce").mean()),
            "median_x": safe_float(pd.to_numeric(grp.get("object_x_full_approx"), errors="coerce").median()),
            "median_y": safe_float(pd.to_numeric(grp.get("object_y_full_approx"), errors="coerce").median()),
            "top_xy_regions": ";".join(grp["xy_region"].value_counts().head(4).index.astype(str).tolist()),
            "cycle_by_role_kruskal_p": kw_p,
        })
    mode_context_summary = pd.DataFrame(mode_groups).sort_values(["n_event", "n_roi"], ascending=False)
    mode_context_summary.to_csv(out / "mode_context_summary.csv", index=False)

    summary = {
        "n_roi": int(len(df)),
        "n_cycles": int(df["cycleNo"].nunique()),
        "n_xy_regions": int(df["xy_region"].nunique()),
        "event_enriched_mode": EVENT_MODE,
        "event_enriched_mode_fraction": float(df["is_event_enriched_mode"].mean()),
        "top_cycle_summaries": top_records(cycle_all_summary, 8),
        "top_spatial_region_summaries": top_records(region_summary, 8),
        "top_context_correlations": top_records(correlations, 12),
        "top_categorical_tests": top_records(cat_tests, 12),
        "mode_context_summary": top_records(mode_context_summary, 8),
        "context_only_classifier": classifier,
        "guardrail": "Cycle/region mode context is descriptive and uses automatic ROI coordinates plus automatic residual-mode labels; use it to prioritize manual review, not as proof of spatial mechanism or calibrated degradation physics.",
        "outputs": {k: str(v) for k, v in paths.items()} | {"mode_context_summary": str(out / "mode_context_summary.csv"), "readme": str(out / "README.md")},
    }
    with (out / "cycle_region_mode_context_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_readme(out / "README.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
