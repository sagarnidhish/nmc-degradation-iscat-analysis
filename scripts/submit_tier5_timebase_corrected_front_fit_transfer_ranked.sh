#!/bin/bash
#SBATCH --job-name=tier5_timebase_fit_xfer
#SBATCH --account=<slurm-account>
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=0
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/<account>/<username>/Alek_Jiho/derived/logs/tier5_timebase_fit_xfer_%j.out
#SBATCH --error=/scratch/<account>/<username>/Alek_Jiho/derived/logs/tier5_timebase_fit_xfer_%j.err

set -euo pipefail

WORKDIR="/scratch/<account>/<username>/Alek_Jiho"
DERIVED="${WORKDIR}/derived"
LOGS="${DERIVED}/logs"

mkdir -p "${LOGS}" "${DERIVED}/timebase_corrected_front_fit_transfer_ranked_v1"

module load brics/userenv
source "${HOME}/miniforge3/bin/activate"
conda activate walrus-env

python "${WORKDIR}/alek_jiho_nmc_deg/scripts/tier5_timebase_corrected_front_fit.py" \
  --manifest-csv "${DERIVED}/transfer_ranked_roi_sequences/selected_roi_sequence_manifest.csv" \
  --out-dir "${DERIVED}/timebase_corrected_front_fit_transfer_ranked_v1"

