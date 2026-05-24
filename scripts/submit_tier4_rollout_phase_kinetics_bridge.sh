#!/bin/bash
#SBATCH --job-name=rollout_phase_bridge
#SBATCH --account=brics.u6hp
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=0
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/rollout_phase_bridge_%j.out
#SBATCH --error=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/rollout_phase_bridge_%j.err

set -euo pipefail

WORKDIR="/scratch/u6hp/nsagar.u6hp/Alek_Jiho"
DERIVED="${WORKDIR}/derived"

module load brics/userenv
source "${HOME}/miniforge3/bin/activate"
conda activate walrus-env

python "${WORKDIR}/alek_jiho_nmc_deg/scripts/tier4_rollout_phase_kinetics_bridge.py" \
  --out-dir "${DERIVED}/rollout_phase_kinetics_bridge"

