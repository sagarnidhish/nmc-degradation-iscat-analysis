#!/usr/bin/env python3
"""ERA-style metric search over real NMC derived analysis outputs.

This is the project-specific implementation of the empirical-software idea:
collect existing model/audit metric tables, normalize their columns, and rank
analysis variants by a guarded scientific score rather than raw AUC alone.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
from agentic_utils import markdown_table, output_root, resolve_root, write_json


METRIC_SPECS = [
    ("echem_conditioned_residual_dictionary", "echem_conditioned_residual_dictionary/echem_conditioned_residual_dictionary_metrics.csv"),
    ("source_invariant_video_echem_transfer", "source_invariant_video_echem_transfer_audit/source_invariant_video_echem_metrics.csv"),
    ("source_balanced_video_echem_transfer", "source_balanced_video_echem_transfer_audit/source_balanced_video_echem_metrics.csv"),
    ("source_domain_video_echem_adaptation", "source_domain_video_echem_adaptation_audit/source_domain_video_echem_metrics.csv"),
    ("echem_video_embedding_fusion", "echem_video_embedding_fusion_audit/echem_video_embedding_fusion_metrics.csv"),
    ("echem_residual_dictionary_fusion", "echem_residual_dictionary_fusion_audit/echem_residual_dictionary_metrics.csv"),
    ("signed_loss_source_robustness", "signed_loss_source_robustness_audit/signed_loss_source_robustness_metrics.csv"),
    ("source_invariant_physical_family", "source_invariant_physical_family_audit/source_invariant_family_metrics.csv"),
    ("source_invariant_interpretable_feature", "source_invariant_interpretable_feature_audit/interpretable_feature_set_metrics.csv"),
    ("balanced_future_context_region", "balanced_future_context_region_audit/balanced_future_context_region_oof_metrics.csv"),
    ("balanced_future_roi_physics", "balanced_future_roi_physics_audit/balanced_future_leave_cycle_oof_metrics.csv"),
    ("temporal_directionality_physics", "temporal_directionality_physics_audit/temporal_directionality_model_metrics.csv"),
    ("cycle_state_mode_frequency", "cycle_state_mode_frequency_bridge/cycle_state_mode_frequency_model_metrics.csv"),
    ("acquisition_residualized_video_echem", "acquisition_residualized_video_echem_warning/acquisition_residualized_video_echem_metrics.csv"),
    ("acquisition_residualized_video_physics", "acquisition_residualized_video_physics_benchmark/acquisition_residualized_metrics.csv"),
]


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def first_value(row: pd.Series, names: List[str], default: Any = "") -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def metric_file_candidates(root: Path, rel: str) -> List[Path]:
    return [
        root / "derived" / rel,
        root / "alek_jiho_nmc_deg" / "derived_local" / rel,
        root / "derived_local" / rel,
    ]


def load_metric_rows(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for source_name, rel in METRIC_SPECS:
        path = next((p for p in metric_file_candidates(root, rel) if p.exists()), None)
        if path is None:
            continue
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            feature_set = str(first_value(row, ["feature_set", "axis", "model", "task", "method"], "unknown"))
            method = str(first_value(row, ["method", "model", "transform", "split"], ""))
            target = str(first_value(row, ["target", "label", "outcome"], ""))
            split = str(first_value(row, ["split", "group_col", "validation", "method"], ""))
            auc = finite(first_value(row, ["roc_auc", "oriented_auc", "pooled_oof_roc_auc", "mean_roc_auc"], float("nan")))
            ap = finite(first_value(row, ["average_precision", "pooled_oof_average_precision", "mean_average_precision"], float("nan")))
            rho = finite(first_value(row, ["spearman_rho"], float("nan")))
            empirical_p = finite(first_value(row, ["empirical_p_ge_observed"], float("nan")))
            null_p95 = finite(first_value(row, ["null_auc_p95"], float("nan")))
            source_eta2 = finite(first_value(row, ["source_eta2"], 0.0), 0.0)
            n_eval = finite(first_value(row, ["n_eval", "n", "n_scored"], 0), 0.0)
            rows.append({
                "source_table": source_name,
                "source_path": str(path),
                "row_index": int(idx),
                "feature_set": feature_set,
                "method": method,
                "target": target,
                "split": split,
                "auc": auc,
                "average_precision": ap,
                "spearman_rho": rho,
                "empirical_p_ge_observed": empirical_p,
                "null_auc_p95": null_p95,
                "source_eta2": source_eta2,
                "n_eval": n_eval,
            })
    return pd.DataFrame(rows)


def leakage_penalty(row: pd.Series) -> float:
    text = " ".join(str(row.get(c, "")).lower() for c in ["source_table", "feature_set", "method", "target", "split"])
    penalty = 0.0
    if "future_any_drop_within_8" in text:
        penalty += 0.08
    if "acquisition_context" in text or "design_context" in text:
        penalty += 0.22
    if "source_mean_only" in text or "raw" == str(row.get("method", "")).lower():
        penalty += 0.05
    if "leave_source" not in text and "source_resid" not in text and "source_invariant" not in text and "source_balanced" not in text:
        penalty += 0.04
    eta = finite(row.get("source_eta2"), 0.0)
    if eta > 0:
        penalty += min(0.25, 0.25 * eta)
    return round(penalty, 4)


def control_bonus(row: pd.Series) -> float:
    text = " ".join(str(row.get(c, "")).lower() for c in ["source_table", "feature_set", "method", "split"])
    bonus = 0.0
    if "leave_source" in text or "source_invariant" in text:
        bonus += 0.08
    if "source_balanced" in text:
        bonus += 0.08
    if "resid" in text or "residualized" in text or "source_residual" in text or "within_source" in text:
        bonus += 0.05
    if "permutation" in text or finite(row.get("empirical_p_ge_observed"), float("nan")) <= 0.05:
        bonus += 0.04
    null_p95 = finite(row.get("null_auc_p95"), float("nan"))
    auc = finite(row.get("auc"), float("nan"))
    if math.isfinite(null_p95) and math.isfinite(auc) and auc > null_p95:
        bonus += 0.05
    return round(bonus, 4)


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["leakage_penalty"] = out.apply(leakage_penalty, axis=1)
    out["control_bonus"] = out.apply(control_bonus, axis=1)
    out["auc_component"] = out["auc"].fillna(0.5)
    out["ap_component"] = out["average_precision"].fillna(0.5)
    out["rho_component"] = out["spearman_rho"].abs().fillna(0.0).clip(upper=1.0)
    out["scientific_score"] = (
        out["auc_component"]
        + 0.35 * out["ap_component"]
        + 0.10 * out["rho_component"]
        + out["control_bonus"]
        - out["leakage_penalty"]
    ).round(4)
    out["promotion_status"] = "candidate"
    out.loc[out["auc_component"] < 0.6, "promotion_status"] = "weak"
    out.loc[out["leakage_penalty"] >= 0.25, "promotion_status"] = "leakage_guarded"
    out.loc[(out["scientific_score"] >= 1.05) & (out["leakage_penalty"] < 0.2), "promotion_status"] = "promote_for_followup"
    return out.sort_values(["scientific_score", "auc_component", "average_precision"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()
    root = resolve_root(args.root)
    out = output_root(root, args.out_dir) / "05_agentic_metric_search"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_metric_rows(root)
    ranked = score_rows(raw)
    ranked.to_csv(out / "agentic_metric_search_ranked.csv", index=False)
    top = ranked.head(args.top_n).to_dict("records") if not ranked.empty else []
    summary = {
        "root": str(root),
        "n_metric_rows": int(len(ranked)),
        "n_source_tables_present": int(ranked["source_table"].nunique()) if not ranked.empty else 0,
        "n_promote_for_followup": int((ranked["promotion_status"] == "promote_for_followup").sum()) if not ranked.empty else 0,
        "top_rows": top[:5],
        "score_definition": "auc + 0.35*AP + 0.10*abs(spearman_rho) + control_bonus - leakage_penalty",
        "guardrail": "High score ranks computational follow-up candidates only; weak labels, source imbalance, acquisition context, automatic masks, and missing manual QC still block physical mechanism claims.",
    }
    write_json(out / "agentic_metric_search_summary.json", summary)
    md = [
        "# Agentic Metric Search",
        "",
        f"Rows scored: {summary['n_metric_rows']}",
        "",
        "## Top Analysis Variants",
        "",
        markdown_table(top, ["source_table", "feature_set", "method", "target", "split", "auc", "average_precision", "leakage_penalty", "control_bonus", "scientific_score", "promotion_status"]),
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
        "",
    ]
    (out / "agentic_metric_search_report.md").write_text("\n".join(md))
    print(f"[done] wrote agentic metric search outputs to {out}")


if __name__ == "__main__":
    main()
