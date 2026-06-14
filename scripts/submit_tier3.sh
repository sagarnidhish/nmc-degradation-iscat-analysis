#!/bin/bash
#SBATCH --job-name=tier3_nmc_dl
#SBATCH --account=<slurm-account>
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com
#SBATCH --output=/scratch/<account>/<username>/Alek_Jiho/derived/logs/tier3_%j.out
#SBATCH --error=/scratch/<account>/<username>/Alek_Jiho/derived/logs/tier3_%j.err

set -euo pipefail

WORKDIR="/scratch/<account>/<username>/Alek_Jiho"
DERIVED="${WORKDIR}/derived"
LOGS="${DERIVED}/logs"
SCRIPTS="${WORKDIR}/scripts"

mkdir -p "${DERIVED}" "${DERIVED}/plots" "${LOGS}" \
         "${DERIVED}/ts2vec" "${DERIVED}/transformer" "${DERIVED}/crack_detector"

module load brics/userenv
source "${HOME}/miniforge3/bin/activate"
conda activate walrus-env

echo "=== Job ${SLURM_JOB_ID} started $(date) ==="
echo "Node: ${SLURMD_NODENAME}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

cd "${WORKDIR}"

echo "--- T3.1: TS2Vec Pretraining ---"
python "${SCRIPTS}/tier3_ts2vec_pretrain.py" \
  --base-dir "${WORKDIR}" \
  --derived-dir "${DERIVED}" \
  --target-len 128 \
  --hidden 64 \
  --depth 6 \
  --epochs 500 \
  --max-h5 6 \
  2>&1 | tee "${LOGS}/ts2vec_${SLURM_JOB_ID}.log"

echo "--- T3.2: Multimodal Transformer ---"
python "${SCRIPTS}/tier3_multimodal_transformer.py" \
  --derived-dir "${DERIVED}" \
  --context-len 20 \
  --d-model 64 \
  --nhead 4 \
  --n-layers 3 \
  --epochs 300 \
  2>&1 | tee "${LOGS}/transformer_${SLURM_JOB_ID}.log"

echo "--- T3.3: Crack Precursor Detector ---"
python "${SCRIPTS}/tier3_crack_precursor_detector.py" \
  --base-dir "${WORKDIR}" \
  --derived-dir "${DERIVED}" \
  --window-size 100 \
  --epochs 300 \
  --max-h5 5 \
  2>&1 | tee "${LOGS}/crack_detector_${SLURM_JOB_ID}.log"

echo "=== Job ${SLURM_JOB_ID} completed $(date) ==="
