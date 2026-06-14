#!/bin/bash
#SBATCH --job-name=rollout_phase_bridge2
#SBATCH --account=<slurm-account>
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=0
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/<account>/<username>/Alek_Jiho/derived/logs/rollout_phase_bridge2_%j.out
#SBATCH --error=/scratch/<account>/<username>/Alek_Jiho/derived/logs/rollout_phase_bridge2_%j.err

set -euo pipefail

WORKDIR="/scratch/<account>/<username>/Alek_Jiho"
module load brics/userenv
source "${HOME}/miniforge3/bin/activate"
conda activate walrus-env

python "${WORKDIR}/alek_jiho_nmc_deg/scripts/tier4_rollout_phase_kinetics_bridge_v2.py"

