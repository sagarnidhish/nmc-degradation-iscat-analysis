#!/bin/bash
#SBATCH --job-name=tier5_timebase_fit_sb
#SBATCH --account=brics.u6hp
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=0
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/tier5_timebase_fit_sb_%j.out
#SBATCH --error=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/tier5_timebase_fit_sb_%j.err

set -euo pipefail

WORKDIR="/scratch/u6hp/nsagar.u6hp/Alek_Jiho"
DERIVED="${WORKDIR}/derived"
LOGS="${DERIVED}/logs"

mkdir -p "${LOGS}" "${DERIVED}/timebase_corrected_front_fit_source_balanced_v1"

module load brics/userenv
source "${HOME}/miniforge3/bin/activate"
conda activate walrus-env

python "${WORKDIR}/alek_jiho_nmc_deg/scripts/tier5_timebase_corrected_front_fit.py" \
  --manifest-csv "${DERIVED}/source_balanced_roi_sequences/selected_roi_sequence_manifest.csv" \
  --out-dir "${DERIVED}/timebase_corrected_front_fit_source_balanced_v1"

