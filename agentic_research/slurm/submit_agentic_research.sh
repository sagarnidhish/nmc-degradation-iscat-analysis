#!/bin/bash
#SBATCH --job-name=nmc_agentic
#SBATCH --partition=workq
#SBATCH --account=brics.u6hp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/%x_%j.out
#SBATCH --error=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-/scratch/u6hp/nsagar.u6hp/Alek_Jiho}"
REPO="${REPO:-$ROOT/alek_jiho_nmc_deg}"
OUT="${OUT:-$ROOT/agentic_research_outputs}"

cd "$REPO"
bash agentic_research/run_all_agentic_smoke.sh "$ROOT" "$OUT"
