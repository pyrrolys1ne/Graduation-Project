#!/usr/bin/env bash
#SBATCH --job-name=encoder-service
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

port="${SERVICE_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
host="$(hostname -f 2>/dev/null || hostname)"
mkdir -p "${RESULTS_ROOT}/slurm-services"
endpoint_file="${RESULTS_ROOT}/slurm-services/${SLURM_JOB_ID}.endpoint"
printf 'job_id=%s\nhost=%s\nport=%s\n' "${SLURM_JOB_ID}" "${host}" "${port}" > "${endpoint_file}"

echo "服务节点: ${host}:${port}"
echo "端点记录: ${endpoint_file}"
echo "停止服务: scancel ${SLURM_JOB_ID}"
exec encoder-sched-server --config "${ENCODER_CONFIG}" --host 0.0.0.0 --port "${port}"
