#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENCODER_SLURM_ENV:-${ROOT}/slurm/cluster.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "错误: 未找到 ${ENV_FILE}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

options=(
  --nodes=1
  --ntasks=1
  --gres="${SLURM_GRES:-gpu:1}"
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
  --mem="${SLURM_MEM:-32G}"
  --time="${SLURM_TIME:-04:00:00}"
  --export="ALL,ENCODER_SLURM_ENV=${ENV_FILE}"
)
[[ -n "${SLURM_PARTITION:-}" ]] && options+=(--partition="${SLURM_PARTITION}")
[[ -n "${SLURM_QOS:-}" ]] && options+=(--qos="${SLURM_QOS}")
[[ -n "${SLURM_ACCOUNT:-}" ]] && options+=(--account="${SLURM_ACCOUNT}")

printf -v init_command 'source %q; exec bash --noprofile --norc -i' "${ROOT}/slurm/lib/common.sh"
exec srun "${options[@]}" "$@" --pty bash -lc "${init_command}"
