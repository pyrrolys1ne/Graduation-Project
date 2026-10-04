#!/usr/bin/env bash
#SBATCH --job-name=encoder-check
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
if [[ -n "${ENCODER_SLURM_ENV:-}" ]]; then
  common="$(dirname "${ENCODER_SLURM_ENV}")/lib/common.sh"
else
  common="${SLURM_SUBMIT_DIR}/slurm/lib/common.sh"
fi
# shellcheck disable=SC1091
source "${common}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
python -m encoder_sched.runtime --config "${ENCODER_CONFIG}"
python scripts/smoke_gpu.py --config "${ENCODER_CONFIG}"
