#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENCODER_SLURM_ENV:-${ROOT}/slurm/cluster.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "错误: 未找到 ${ENV_FILE}" >&2
  echo "请先执行: cp slurm/cluster.env.example slurm/cluster.env" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

kind="${1:-}"
if [[ -z "${kind}" ]]; then
  echo "用法: $0 {bootstrap|cache-model|check|service|experiment|repeated} [sbatch 参数]" >&2
  exit 2
fi
shift
job_script="${ROOT}/slurm/${kind}.slurm"
if [[ ! -f "${job_script}" ]]; then
  echo "错误: 未知作业类型 ${kind}" >&2
  exit 2
fi

mkdir -p "${ROOT}/slurm/logs"
options=(
  --nodes=1
  --ntasks=1
  --gres="${SLURM_GRES:-gpu:1}"
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
  --mem="${SLURM_MEM:-32G}"
  --time="${SLURM_TIME:-04:00:00}"
  --chdir="${ROOT}"
  --output="${ROOT}/slurm/logs/%x-%j.out"
  --error="${ROOT}/slurm/logs/%x-%j.err"
  --export="ALL,ENCODER_SLURM_ENV=${ENV_FILE}"
)
[[ -n "${SLURM_PARTITION:-}" ]] && options+=(--partition="${SLURM_PARTITION}")
[[ -n "${SLURM_QOS:-}" ]] && options+=(--qos="${SLURM_QOS}")
[[ -n "${SLURM_ACCOUNT:-}" ]] && options+=(--account="${SLURM_ACCOUNT}")

exec sbatch "${options[@]}" "$@" "${job_script}"
