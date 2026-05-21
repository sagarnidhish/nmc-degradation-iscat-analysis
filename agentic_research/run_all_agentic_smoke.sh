#!/bin/bash
set -euo pipefail

ROOT="${1:-/scratch/u6hp/nsagar.u6hp/Alek_Jiho}"
OUT="${2:-$ROOT/agentic_research_outputs}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if ! "$PYTHON_BIN" -c "import pandas" >/dev/null 2>&1; then
  if [ -x "$HOME/miniforge3/envs/walrus-env/bin/python" ]; then
    PYTHON_BIN="$HOME/miniforge3/envs/walrus-env/bin/python"
  fi
fi

"$PYTHON_BIN" "$HERE/01_era_experiment_search/run_era_search.py" --root "$ROOT" --out-dir "$OUT"
"$PYTHON_BIN" "$HERE/02_hypothesis_tournament/run_hypothesis_tournament.py" --root "$ROOT" --out-dir "$OUT"
"$PYTHON_BIN" "$HERE/03_closed_loop_analysis/run_closed_loop_analysis.py" --root "$ROOT" --out-dir "$OUT"
"$PYTHON_BIN" "$HERE/04_guarded_code_generation/run_guarded_code_generation.py" --root "$ROOT" --out-dir "$OUT" --top-n 3

echo "[done] agentic smoke workflows wrote outputs to $OUT"
