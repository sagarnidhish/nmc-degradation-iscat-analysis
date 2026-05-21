#!/usr/bin/env python3
"""Recompute NMC front/diffusion effects behind manual-QC gates.

This script is intentionally conservative. It reads the deduplicated manual-QC
label workbook and only computes publication-facing front/diffusion statistics
for ROIs whose manual labels explicitly accept particle identity and front-mask
quality. With the current workbook all labels are pending, so the expected output
is a reproducible "no accepted labels yet" audit rather than another automatic
claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


FRONT_FEATURES = [
    "phase_slope_positive_fraction_protocol_residual",
    "phase_slope_negative_fraction_protocol_residual",
    "threshold_robust_phase_score_protocol_residual",
    "phase_slope_median_per_s",
    "threshold_robust_phase_score",
    "diffusion_proxy_abs_median_um2_per_s_protocol_residual",
    "diffusion_proxy_abs_median_um2_per_s",
    "diffusion_proxy_median_um2_per_s",
    "radius2_slope_median_px2_per_s",
]

ACCEPT_VALUES = {"accept", "accepted", "yes", "y", "true", "1"}
UNCERTAIN_VALUES = {"uncertain", "maybe", "review", "needs_review"}
REJECT_VALUES = {"reject", "rejected", "no", "n", "false", "0"}


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


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip().lower()


def is_accept(value: Any) -> bool:
    return norm_text(value) in ACCEPT_VALUES


def label_category(value: Any) -> str:
    text = norm_text(value)
    if not text:
        return "pending"
    if text in ACCEPT_VALUES:
        return "accept"
    if text in REJECT_VALUES:
        return "reject"
    if text in UNCERTAIN_VALUES:
        return "uncertain"
    return "other"


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_inputs(derived: Path) -> pd.DataFrame:
    labels = pd.read_csv(derived / "manual_qc_label_workbook" / "manual_qc_label_template.csv")
    front = pd.read_csv(derived / "multi_cycle_threshold_robust_fronts" / "threshold_robust_front_summary.csv")
    residuals = pd.read_csv(derived / "protocol_conditioned_front_effects" / "protocol_conditioned_front_residuals.csv")
    echem = pd.read_csv(derived / "multi_cycle_roi_echem_coupling" / "multi_cycle_roi_echem_joined.csv")
    modes_path = derived / "residual_physics_mode_taxonomy" / "residual_physics_mode_assignments.csv"
    modes = pd.read_csv(modes_path) if modes_path.exists() else pd.DataFrame({"roi_id": []})

    residual_keep = ["roi_id"] + [c for c in FRONT_FEATURES if c in residuals.columns]
    base = front.merge(residuals[residual_keep], on="roi_id", how="left", suffixes=("", "_resid"))
    echem_keep = [c for c in ["roi_id", "n_frames_percentile", "V_mean", "I_mean_mA", "degradation_mode_hypothesis"] if c in echem.columns]
    base = base.merge(echem[echem_keep].drop_duplicates("roi_id"), on="roi_id", how="left")
    mode_keep = [c for c in ["roi_id", "mode_label", "mode_review_priority"] if c in modes.columns]
    if mode_keep:
        base = base.merge(modes[mode_keep].drop_duplicates("roi_id"), on="roi_id", how="left")

    label_cols = [
        "roi_id",
        "review_priority_tier",
        "review_sources",
        "recommended_review_reason",
        "manual_qc_status",
        "manual_qc_decision",
        "manual_particle_identity_ok",
        "manual_front_mask_ok",
        "manual_diffusion_interpretable",
        "manual_qc_notes",
        "primary_qc_png",
        "control_balanced_qc_png",
    ]
    label_cols = [c for c in label_cols if c in labels.columns]
    joined = base.merge(labels[label_cols].drop_duplicates("roi_id"), on="roi_id", how="left")
    for col in ["manual_qc_status", "manual_qc_decision", "manual_particle_identity_ok", "manual_front_mask_ok", "manual_diffusion_interpretable"]:
        if col not in joined.columns:
            joined[col] = ""
    joined["manual_decision_category"] = joined["manual_qc_decision"].map(label_category)
    joined["manual_particle_identity_accept"] = joined["manual_particle_identity_ok"].map(is_accept)
    joined["manual_front_mask_accept"] = joined["manual_front_mask_ok"].map(is_accept)
    joined["manual_diffusion_interpretable_accept"] = joined["manual_diffusion_interpretable"].map(is_accept)
    joined["manual_front_effect_accepted"] = (
        joined["manual_decision_category"].eq("accept")
        & joined["manual_particle_identity_accept"]
        & joined["manual_front_mask_accept"]
    )
    joined["manual_diffusion_accepted"] = joined["manual_front_effect_accepted"] & joined["manual_diffusion_interpretable_accept"]
    return joined


def permutation_p(event: np.ndarray, control: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    if len(event) < 2 or len(control) < 2:
        return np.nan
    observed = float(np.nanmedian(event) - np.nanmedian(control))
    pooled = np.concatenate([event, control])
    n_event = len(event)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(np.nanmedian(perm[:n_event]) - np.nanmedian(perm[n_event:]))
        if abs(diff) >= abs(observed):
            count += 1
    return float((count + 1) / (n_perm + 1))


def bootstrap_ci(event: np.ndarray, control: np.ndarray, rng: np.random.Generator, n_boot: int) -> Dict[str, float]:
    if len(event) < 2 or len(control) < 2:
        return {"bootstrap_p05": np.nan, "bootstrap_p50": np.nan, "bootstrap_p95": np.nan}
    diffs = []
    for _ in range(n_boot):
        e = rng.choice(event, len(event), replace=True)
        c = rng.choice(control, len(control), replace=True)
        diffs.append(float(np.nanmedian(e) - np.nanmedian(c)))
    p05, p50, p95 = np.nanpercentile(diffs, [5, 50, 95])
    return {"bootstrap_p05": float(p05), "bootstrap_p50": float(p50), "bootstrap_p95": float(p95)}


def test_feature_table(df: pd.DataFrame, features: Iterable[str], stratum: str, rng: np.random.Generator, n_boot: int, n_perm: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature in features:
        if feature not in df.columns:
            continue
        event = to_num(df.loc[df["cohort_role"].eq("event"), feature]).dropna().to_numpy(dtype=float)
        control = to_num(df.loc[df["cohort_role"].eq("control"), feature]).dropna().to_numpy(dtype=float)
        if len(event) >= 2 and len(control) >= 2:
            u, p = mannwhitneyu(event, control, alternative="two-sided")
            ci = bootstrap_ci(event, control, rng, n_boot)
            perm_p = permutation_p(event, control, rng, n_perm)
            event_median = float(np.nanmedian(event))
            control_median = float(np.nanmedian(control))
        else:
            u = p = perm_p = np.nan
            ci = {"bootstrap_p05": np.nan, "bootstrap_p50": np.nan, "bootstrap_p95": np.nan}
            event_median = control_median = np.nan
        rows.append({
            "stratum": stratum,
            "feature": feature,
            "n_event": int(len(event)),
            "n_control": int(len(control)),
            "event_median": event_median,
            "control_median": control_median,
            "median_event_minus_control": float(event_median - control_median) if np.isfinite(event_median) and np.isfinite(control_median) else np.nan,
            "mannwhitney_u": float(u) if np.isfinite(u) else np.nan,
            "mannwhitney_p": float(p) if np.isfinite(p) else np.nan,
            "permutation_median_p": perm_p,
            **ci,
        })
    return pd.DataFrame(rows)


def status_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df.fillna("missing").replace("", "missing").value_counts().to_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived")
    parser.add_argument("--out-dir", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/manual_qc_gated_front_effects")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    derived = Path(args.derived_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = load_inputs(derived)
    accepted_front = joined[joined["manual_front_effect_accepted"]].copy()
    accepted_diffusion = joined[joined["manual_diffusion_accepted"]].copy()
    pending = joined[~joined["manual_front_effect_accepted"]].copy()
    pending = pending.sort_values(["review_priority_tier", "cohort_role", "roi_id"], na_position="last")

    rng = np.random.default_rng(args.seed)
    test_tables = []
    if len(accepted_front):
        test_tables.append(test_feature_table(accepted_front, FRONT_FEATURES, "manual_front_effect_accepted", rng, args.n_bootstrap, args.n_permutation))
    if len(accepted_diffusion):
        diffusion_features = [f for f in FRONT_FEATURES if "diffusion" in f or "radius2" in f]
        test_tables.append(test_feature_table(accepted_diffusion, diffusion_features, "manual_diffusion_interpretable", rng, args.n_bootstrap, args.n_permutation))
    tests = pd.concat(test_tables, ignore_index=True) if test_tables else pd.DataFrame(columns=[
        "stratum", "feature", "n_event", "n_control", "event_median", "control_median",
        "median_event_minus_control", "mannwhitney_u", "mannwhitney_p", "permutation_median_p",
        "bootstrap_p05", "bootstrap_p50", "bootstrap_p95",
    ])

    joined_path = out / "manual_qc_gated_joined.csv"
    pending_path = out / "manual_qc_pending_review_queue.csv"
    tests_path = out / "manual_qc_gated_front_effect_tests.csv"
    joined.to_csv(joined_path, index=False)
    pending_cols = [c for c in [
        "roi_id", "cohort_role", "cycleNo", "event_reference_cycle", "review_priority_tier",
        "review_sources", "recommended_review_reason", "manual_qc_status", "manual_qc_decision",
        "manual_particle_identity_ok", "manual_front_mask_ok", "manual_diffusion_interpretable",
        "primary_qc_png", "control_balanced_qc_png",
    ] if c in pending.columns]
    pending[pending_cols].to_csv(pending_path, index=False)
    tests.to_csv(tests_path, index=False)

    accepted_role_counts = accepted_front.get("cohort_role", pd.Series(dtype=object)).value_counts().to_dict()
    diffusion_role_counts = accepted_diffusion.get("cohort_role", pd.Series(dtype=object)).value_counts().to_dict()
    status = "ready_for_manual_labels" if len(accepted_front) == 0 else "computed_accepted_front_effects"
    if len(accepted_front) > 0 and (accepted_front["cohort_role"].nunique() < 2):
        status = "accepted_labels_missing_one_role"

    summary = {
        "status": status,
        "n_joined_roi": int(len(joined)),
        "manual_qc_status_counts": status_counts(joined["manual_qc_status"]),
        "manual_qc_decision_counts": status_counts(joined["manual_decision_category"]),
        "n_manual_front_effect_accepted": int(len(accepted_front)),
        "accepted_front_role_counts": clean_json(accepted_role_counts),
        "n_manual_diffusion_accepted": int(len(accepted_diffusion)),
        "accepted_diffusion_role_counts": clean_json(diffusion_role_counts),
        "n_pending_or_not_accepted": int(len(pending)),
        "n_effect_tests": int(len(tests)),
        "guardrail": "Only manually accepted particle/front labels are eligible for gated front-effect tests. Current pending labels intentionally produce no accepted-front physics claims.",
        "outputs": {
            "joined": str(joined_path),
            "pending_review_queue": str(pending_path),
            "effect_tests": str(tests_path),
            "summary": str(out / "manual_qc_gated_front_effects_summary.json"),
        },
    }
    with (out / "manual_qc_gated_front_effects_summary.json").open("w") as f:
        json.dump(clean_json(summary), f, indent=2, sort_keys=True)

    lines = [
        "# Manual-QC Gated Front Effects",
        "",
        "Recomputes NMC front/diffusion effect tests only after manual QC labels accept particle identity and front-mask quality.",
        "",
        f"- Status: {summary['status']}",
        f"- Joined ROI rows: {summary['n_joined_roi']}",
        f"- Accepted front-effect rows: {summary['n_manual_front_effect_accepted']}",
        f"- Accepted diffusion-interpretable rows: {summary['n_manual_diffusion_accepted']}",
        f"- Pending/not accepted rows: {summary['n_pending_or_not_accepted']}",
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out), "status": status, "n_accepted": summary["n_manual_front_effect_accepted"]}, indent=2))


if __name__ == "__main__":
    main()
