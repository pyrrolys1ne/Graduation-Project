#!/usr/bin/env bash
#SBATCH --job-name=encoder-repeated
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

group="${REPEATED_GROUP:-concurrency}"
port="${SERVICE_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
run_dir="${RESULTS_ROOT}/repeated/${group}/${SLURM_JOB_ID}"
mkdir -p "${run_dir}"

workload_args=()
if [[ -n "${REPEATED_WORKLOAD:-}" ]]; then
  workload_args=(--workload "${REPEATED_WORKLOAD}")
fi

python scripts/run_repeated_comparison.py \
  --group "${group}" \
  --config "${ENCODER_CONFIG}" \
  "${workload_args[@]}" \
  --output-dir "${run_dir}/runs" \
  --repeats "${REPEATS:-5}" \
  --concurrency "${REPEATED_CONCURRENCY:-0}" \
  --shards "${REPEATED_SHARDS:-4}" \
  --port "${port}"

echo "重复实验结果: ${run_dir}"
