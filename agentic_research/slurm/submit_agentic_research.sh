#!/bin/bash
#SBATCH --job-name=nmc_agentic
#SBATCH --partition=workq
#SBATCH --account=<slurm-account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/<account>/<username>/Alek_Jiho/derived/logs/%x_%j.out
#SBATCH --error=/scratch/<account>/<username>/Alek_Jiho/derived/logs/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-/scratch/<account>/<username>/Alek_Jiho}"
REPO="${REPO:-$ROOT/alek_jiho_nmc_deg}"
OUT="${OUT:-$ROOT/agentic_research_outputs}"

cd "$REPO"
bash agentic_research/run_all_agentic_smoke.sh "$ROOT" "$OUT"
