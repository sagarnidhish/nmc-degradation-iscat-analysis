#!/usr/bin/env python3
"""Co-Scientist-style hypothesis tournament for NMC degradation analyses."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
from agentic_utils import markdown_table, output_root, read_json, resolve_root, summarize_available_artifacts, write_json


def load_hypotheses(path: Path) -> List[Dict[str, Any]]:
    seed = read_json(path)
    return list(seed.get("hypotheses", []))


def score_hypothesis(hyp: Dict[str, Any], artifacts: Dict[str, Any], root: Path) -> Dict[str, Any]:
    available = artifacts["available"]
    support_keys = hyp.get("supports", [])
    support_count = sum(1 for key in support_keys if available.get(key, False))
    support_score = support_count / max(len(support_keys), 1)
    test_count = len(hyp.get("critical_tests", []))
    control_score = min(1.0, test_count / 3.0)
    specificity = 1.0 if any(token in hyp["statement"].lower() for token in ["cycle", "particle", "protocol", "observable", "front"]) else 0.5

    bonus = 0.0
    if hyp["id"] == "H1_sync_events_are_real_degradation":
        sync = read_json(root / "derived" / "event_synchrony" / "event_synchrony_summary.json")
        p = sync.get("permutation_p_max_same_cycle_particles")
        if isinstance(p, (int, float)) and p < 0.01:
            bonus += 1.5
    if hyp["id"] == "H3_event_precursors_are_weak_but_present":
        path = root / "derived" / "particle_event_targets" / "particle_event_feature_baselines.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "f1" in df and df["f1"].notna().any():
                bonus += min(1.0, float(df["f1"].max()) * 2.0)

    tournament_score = 10.0 * support_score + 4.0 * control_score + 2.0 * specificity + bonus
    missing = [key for key in support_keys if not available.get(key, False)]
    return {
        "id": hyp["id"],
        "statement": hyp["statement"],
        "support_score": round(support_score, 3),
        "control_score": round(control_score, 3),
        "tournament_score": round(tournament_score, 3),
        "missing_support": ";".join(missing),
        "critical_next_tests": "; ".join(hyp.get("critical_tests", [])[:3]),
        "skeptical_risk": hyp.get("risk", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--hypotheses", default="")
    args = parser.parse_args()
    root = resolve_root(args.root)
    out = output_root(root, args.out_dir) / "02_hypothesis_tournament"
    out.mkdir(parents=True, exist_ok=True)
    hyp_path = Path(args.hypotheses) if args.hypotheses else Path(__file__).resolve().parents[1] / "shared" / "hypotheses_seed.json"
    artifacts = summarize_available_artifacts(root)
    hypotheses = load_hypotheses(hyp_path)
    rows = [score_hypothesis(h, artifacts, root) for h in hypotheses]
    rows = sorted(rows, key=lambda r: r["tournament_score"], reverse=True)
    pd.DataFrame(rows).to_csv(out / "hypothesis_tournament.csv", index=False)
    write_json(out / "hypothesis_tournament.json", {"root": str(root), "hypotheses": rows})
    md = [
        "# Hypothesis Tournament",
        "",
        "Ranked hypotheses using current derived evidence, required controls, and skeptical risks.",
        "",
        markdown_table(rows, ["id", "tournament_score", "support_score", "statement", "critical_next_tests", "skeptical_risk"]),
        "",
        "## Rule",
        "",
        "High score means the hypothesis is ready for targeted validation, not that it is proven.",
        "",
    ]
    (out / "hypothesis_tournament_report.md").write_text("\n".join(md))
    print(f"[done] wrote hypothesis tournament outputs to {out}")


if __name__ == "__main__":
    main()
