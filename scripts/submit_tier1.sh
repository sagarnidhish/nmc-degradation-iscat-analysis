#!/bin/bash
#SBATCH --job-name=tier1_nmc_eda
#SBATCH --account=brics.u6hp
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sagarnidhish26@gmail.com
#SBATCH --output=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/tier1_%j.out
#SBATCH --error=/scratch/u6hp/nsagar.u6hp/Alek_Jiho/derived/logs/tier1_%j.err

set -euo pipefail

WORKDIR="/scratch/u6hp/nsagar.u6hp/Alek_Jiho"
DERIVED="${WORKDIR}/derived"
LOGS="${DERIVED}/logs"
SCRIPTS="${WORKDIR}/scripts"

mkdir -p "${DERIVED}" "${DERIVED}/plots" "${LOGS}"

module load brics/userenv
source "${HOME}/miniforge3/bin/activate"
conda activate walrus-env

echo "=== Job ${SLURM_JOB_ID} started $(date) ==="
echo "Node: ${SLURMD_NODENAME}"
echo "Python: $(which python)"

# Step 1: HDF5 metadata inventory (no movie loading, ~5 min)
echo "--- Step 1: H5 inventory ---"
python "${SCRIPTS}/tier1_h5_inventory.py" \
  --base-dir "${WORKDIR}" \
  --out-dir "${DERIVED}" \
  2>&1 | tee "${LOGS}/h5_inventory_${SLURM_JOB_ID}.log"

# Step 2: Electrochemistry EDA (chunked CSV, dQ/dV heatmap, ~30 min for 11M rows)
echo "--- Step 2: Echem EDA ---"
python "${SCRIPTS}/tier1_echem_eda.py" \
  --out-dir "${DERIVED}" \
  2>&1 | tee "${LOGS}/echem_eda_${SLURM_JOB_ID}.log"

# Step 3: Particle intensity traces and drop detection
echo "--- Step 3: Particle intensity EDA ---"
python "${SCRIPTS}/tier1_particle_intensity_eda.py" \
  --out-dir "${DERIVED}" \
  2>&1 | tee "${LOGS}/particle_eda_${SLURM_JOB_ID}.log"

# Step 4: Background drift analysis (one file per session)
echo "--- Step 4: Background drift ---"
python "${SCRIPTS}/tier1_background_drift.py" \
  --base-dir "${WORKDIR}" \
  --out-dir "${DERIVED}" \
  2>&1 | tee "${LOGS}/background_drift_${SLURM_JOB_ID}.log"

echo "=== Job ${SLURM_JOB_ID} completed $(date) ==="
echo "Outputs in: ${DERIVED}"
