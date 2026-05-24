#!/bin/bash
#SBATCH --job-name=tier5_nf_sweep_sb
#SBATCH --account=brics.u6hp
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/tier5_nf_sweep_sb_%j.out
#SBATCH --error=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/tier5_nf_sweep_sb_%j.err

set -euo pipefail

WORKDIR="/scratch/u6hp/nsagar.u6hp/Alek_Jiho"
DERIVED="${WORKDIR}/derived"
LOGS="${DERIVED}/logs"
ROOT="${DERIVED}/next_frame_baseline_context_sweep_source_balanced_v1"

mkdir -p "${LOGS}" "${ROOT}"

module load brics/userenv
source "${HOME}/miniforge3/bin/activate"
conda activate walrus-env

for ctx in 2 4 8; do
  OUT="${ROOT}/context_${ctx}"
  mkdir -p "${OUT}"
  python "${WORKDIR}/alek_jiho_nmc_deg/scripts/tier5_next_frame_baseline.py" \
    --manifest-csv "${DERIVED}/source_balanced_roi_sequences/selected_roi_sequence_manifest.csv" \
    --out-dir "${OUT}" \
    --context "${ctx}" \
    --epochs 8 \
    --batch-size 32 \
    --holdout-frac 0.25 \
    --seed 0
done

