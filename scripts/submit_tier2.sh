#!/bin/bash
#SBATCH --job-name=tier2_nmc_ml
#SBATCH --account=brics.u6hp
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sagarnidhish26@gmail.com
#SBATCH --output=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/tier2_%j.out
#SBATCH --error=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/tier2_%j.err

set -euo pipefail

WORKDIR="/scratch/u6hp/nsagar.u6hp/Alek_Jiho"
DERIVED="${WORKDIR}/derived"
LOGS="${DERIVED}/logs"
SCRIPTS="${WORKDIR}/scripts"

mkdir -p "${DERIVED}" "${DERIVED}/plots" "${LOGS}" \
         "${DERIVED}/rul" "${DERIVED}/spatial" \
         "${DERIVED}/hazard" "${DERIVED}/coupling"

module load brics/userenv
source "${HOME}/miniforge3/bin/activate"
conda activate walrus-env

echo "=== Job ${SLURM_JOB_ID} started $(date) ==="
echo "Node: ${SLURMD_NODENAME}  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

cd "${WORKDIR}"

echo "--- T2.1: RUL Regressor ---"
python "${SCRIPTS}/tier2_rul_regressor.py" \
  --derived-dir "${DERIVED}" \
  2>&1 | tee "${LOGS}/rul_${SLURM_JOB_ID}.log"

echo "--- T2.2: Spatial Heterogeneity ---"
python "${SCRIPTS}/tier2_spatial_heterogeneity.py" \
  --derived-dir "${DERIVED}" \
  --epochs 200 \
  2>&1 | tee "${LOGS}/spatial_${SLURM_JOB_ID}.log"

echo "--- T2.3: Crack Hazard Model ---"
python "${SCRIPTS}/tier2_crack_hazard.py" \
  --derived-dir "${DERIVED}" \
  --epochs 300 \
  2>&1 | tee "${LOGS}/hazard_${SLURM_JOB_ID}.log"

echo "--- T2.4: Optical-EChem Coupling ---"
python "${SCRIPTS}/tier2_optical_echem_coupling.py" \
  --derived-dir "${DERIVED}" \
  --max-lag 10 \
  2>&1 | tee "${LOGS}/coupling_${SLURM_JOB_ID}.log"

echo "=== Job ${SLURM_JOB_ID} completed $(date) ==="
