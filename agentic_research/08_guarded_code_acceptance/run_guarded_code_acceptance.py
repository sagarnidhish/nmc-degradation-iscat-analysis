#!/usr/bin/env python3
"""Acceptance gates for generated or agentic experiment scripts."""

import argparse
import py_compile
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
from agentic_utils import markdown_table, output_root, resolve_root, write_json


REQUIRED_TERMS = {
    "grouped_or_split_control": ["leave", "group", "split", "source", "cycle"],
    "null_or_leakage_control": ["null", "leak", "resid", "shuffle", "permutation", "guardrail"],
    "output_schema": ["summary", "csv", "json", "readme", "schema"],
    "destructive_call_patterns": ["unlink(", "rmtree(", "os.remove", "shutil.rmtree", ".remove("],
}


def scan_script(path: Path) -> Dict[str, object]:
    text = path.read_text(errors="replace")
    compile_ok = True
    compile_error = ""
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        compile_ok = False
        compile_error = str(exc)
    lower = text.lower()
    checks = {
        "compile_ok": compile_ok,
        "has_grouped_or_split_control": any(term in lower for term in REQUIRED_TERMS["grouped_or_split_control"]),
        "has_null_or_leakage_control": any(term in lower for term in REQUIRED_TERMS["null_or_leakage_control"]),
        "has_output_schema": any(term in lower for term in REQUIRED_TERMS["output_schema"]),
        "has_destructive_delete_term": any(term in lower for term in REQUIRED_TERMS["destructive_call_patterns"]),
    }
    accepted = (
        checks["compile_ok"]
        and checks["has_grouped_or_split_control"]
        and checks["has_null_or_leakage_control"]
        and checks["has_output_schema"]
        and not checks["has_destructive_delete_term"]
    )
    status = "accepted_for_review" if accepted else "needs_revision"
    if checks["has_destructive_delete_term"]:
        status = "reject_destructive_term"
    return {
        "script_path": str(path),
        "script_name": path.name,
        **checks,
        "acceptance_status": status,
        "compile_error": compile_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--script-dir", default="")
    args = parser.parse_args()
    root = resolve_root(args.root)
    out_base = output_root(root, args.out_dir)
    out = out_base / "08_guarded_code_acceptance"
    out.mkdir(parents=True, exist_ok=True)
    script_dir = Path(args.script_dir) if args.script_dir else out_base / "04_guarded_code_generation" / "generated_scripts"
    scripts: List[Path] = sorted(script_dir.glob("*.py")) if script_dir.exists() else []
    rows = [scan_script(path) for path in scripts]
    df = pd.DataFrame(rows)
    df.to_csv(out / "guarded_code_acceptance.csv", index=False)
    status_counts = df["acceptance_status"].value_counts().to_dict() if not df.empty else {}
    summary = {
        "root": str(root),
        "script_dir": str(script_dir),
        "n_scripts_scanned": int(len(rows)),
        "acceptance_status_counts": status_counts,
        "guardrail": "Acceptance means suitable for human/code review and smoke execution, not scientifically validated.",
    }
    write_json(out / "guarded_code_acceptance_summary.json", summary)
    md = [
        "# Guarded Code Acceptance",
        "",
        f"Scripts scanned: {summary['n_scripts_scanned']}",
        "",
        markdown_table(rows, ["script_name", "compile_ok", "has_grouped_or_split_control", "has_null_or_leakage_control", "has_output_schema", "has_destructive_delete_term", "acceptance_status"]),
        "",
        "## Guardrail",
        "",
        summary["guardrail"],
        "",
    ]
    (out / "guarded_code_acceptance_report.md").write_text("\n".join(md))
    print(f"[done] wrote guarded code acceptance outputs to {out}")


if __name__ == "__main__":
    main()
