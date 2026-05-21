#!/usr/bin/env python3
"""Generate guarded, reviewable script stubs for the top NMC experiments."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
from agentic_utils import output_root, resolve_root, write_json


TEMPLATE = '''#!/usr/bin/env python3
"""Generated guarded experiment stub: {experiment_id}

Question:
    {question}

Safety:
    - Reads derived CSV/JSON summaries only by default.
    - Writes to --out-dir.
    - Does not delete or mutate raw files.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/u6hp/nsagar.u6hp/Alek_Jiho")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else root / "agentic_research_outputs" / "generated_experiments" / "{experiment_id}"
    out.mkdir(parents=True, exist_ok=True)

    # TODO: implement experiment-specific logic after reviewing the generated plan.
    status = {{
        "experiment_id": "{experiment_id}",
        "question": {question_json},
        "status": "stub_generated",
        "safe_to_run": True,
        "smoke": bool(args.smoke),
        "expected_inputs": {inputs_json},
    }}
    with (out / "status.json").open("w") as f:
        json.dump(status, f, indent=2, sort_keys=True)
    pd.DataFrame([status]).to_csv(out / "status.csv", index=False)
    print("[done] generated stub status at", out)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--era-csv", default="")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()
    root = resolve_root(args.root)
    out = output_root(root, args.out_dir) / "04_guarded_code_generation"
    generated = out / "generated_scripts"
    generated.mkdir(parents=True, exist_ok=True)
    era_csv = Path(args.era_csv) if args.era_csv else output_root(root, args.out_dir) / "01_era_experiment_search" / "era_ranked_experiments.csv"
    if not era_csv.exists():
        raise SystemExit(f"ERA ranking not found: {era_csv}")
    ranked = pd.read_csv(era_csv).head(args.top_n)
    manifest_rows = []
    for _, row in ranked.iterrows():
        exp_id = str(row["id"])
        question = str(row["question"])
        inputs = str(row.get("inputs", ""))
        script_path = generated / f"{exp_id}.py"
        import json
        script_path.write_text(TEMPLATE.format(
            experiment_id=exp_id,
            question=question,
            question_json=json.dumps(question),
            inputs_json=json.dumps(inputs),
        ))
        script_path.chmod(0o755)
        manifest_rows.append({
            "experiment_id": exp_id,
            "script_path": str(script_path),
            "smoke_command": f"python {script_path} --root {root} --smoke",
            "question": question,
        })
    pd.DataFrame(manifest_rows).to_csv(out / "guarded_generation_manifest.csv", index=False)
    write_json(out / "guarded_generation_manifest.json", {"generated": manifest_rows})
    md_lines = ["# Guarded Code Generation", "", "Generated reviewable stubs for top-ranked experiments.", ""]
    for m in manifest_rows:
        md_lines.extend([f"## {m['experiment_id']}", "", m["question"], "", f"Smoke: `{m['smoke_command']}`", ""])
    (out / "guarded_code_generation_report.md").write_text("\n".join(md_lines))
    print(f"[done] generated {len(manifest_rows)} guarded scripts in {generated}")


if __name__ == "__main__":
    main()
