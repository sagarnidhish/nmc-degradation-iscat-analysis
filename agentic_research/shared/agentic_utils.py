#!/usr/bin/env python3
"""Shared helpers for Alek/Jiho agentic research workflows."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_REMOTE_ROOT = Path("/scratch/u6hp/nsagar.u6hp/Alek_Jiho")


def resolve_root(root: str = "") -> Path:
    candidates = [
        Path(root).expanduser() if root else None,
        DEFAULT_REMOTE_ROOT,
        Path("/home/ns2038/Downloads/alek_jiho_nmc_deg"),
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            if (candidate / "derived").exists() or (candidate / "echemDF_full").exists():
                return candidate
            if (candidate.parent / "derived").exists():
                return candidate.parent
    return Path(root or ".").expanduser().resolve()


def output_root(root: Path, out_dir: str = "") -> Path:
    out = Path(out_dir).expanduser() if out_dir else root / "agentic_research_outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    return val if np.isfinite(val) else default


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def summarize_available_artifacts(root: Path) -> Dict[str, Any]:
    derived = root / "derived"
    repo_derived = root / "alek_jiho_nmc_deg" / "derived_local"
    files = {
        "particle_events": derived / "particle_event_targets" / "particle_abrupt_events.csv",
        "event_training": derived / "particle_event_targets" / "particle_event_training_table.csv",
        "event_baselines": derived / "particle_event_targets" / "particle_event_feature_baselines.csv",
        "event_synchrony": derived / "event_synchrony" / "event_synchrony_summary.json",
        "event_echem": derived / "event_echem_coupling" / "event_echem_coupling_summary.json",
        "event_cycle_table": derived / "event_echem_coupling" / "event_echem_cycle_table.csv",
        "protocol_context": derived / "event_protocol_context" / "event_protocol_context_summary.json",
        "recovery_qc": derived / "event_recovery_qc" / "event_recovery_qc_summary.json",
        "event_candidate_fronts": derived / "event_candidate_fronts" / "event_candidate_fronts_summary.json",
        "validated_front_rois": derived / "validated_front_rois" / "validated_front_rois_summary.json",
        "selected_front_roi_tracking": derived / "selected_front_roi_tracking" / "selected_front_roi_tracking_summary.json",
        "roi_rollout_baselines": derived / "roi_rollout_baselines" / "roi_rollout_baseline_summary.json",
        "roi_event_conditioned_nextframe": derived / "roi_event_conditioned_nextframe" / "roi_event_model_summary.json",
        "roi_residual_cnn_fast": derived / "roi_residual_cnn_fast" / "roi_residual_cnn_summary.json",
        "roi_joint_physics_degradation_modes": derived / "roi_joint_physics_degradation_modes" / "roi_joint_physics_degradation_modes_summary.json",
        "event_vs_control_roi_physics": derived / "event_vs_control_roi_physics" / "event_vs_control_roi_physics_summary.json",
        "control_roi_selection": derived / "control_roi_selection" / "control_roi_selection_summary.json",
        "echem_cycle_summary": derived / "event_echem_coupling" / "echem_cycle_summary.csv",
        "local_particle_events": repo_derived / "particle_event_targets" / "particle_abrupt_events.csv",
    }
    return {
        "root": str(root),
        "derived": str(derived),
        "available": {name: path.exists() for name, path in files.items()},
        "paths": {name: str(path) for name, path in files.items()},
    }


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4g}" if np.isfinite(val) else "")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)
