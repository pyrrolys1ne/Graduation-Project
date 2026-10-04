#!/usr/bin/env bash

# Shared initialization for every Slurm job. The caller decides whether to use
# `set -e`; this file is also sourced by an interactive shell.

SLURM_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_REPO_ROOT="$(cd "${SLURM_HELPER_DIR}/../.." && pwd)"
ENCODER_SLURM_ENV="${ENCODER_SLURM_ENV:-${SLURM_REPO_ROOT}/slurm/cluster.env}"

# Compute-node non-interactive shells may not define the module function.
if [[ -r /etc/profile ]]; then
  common_had_nounset=0
  if [[ "$-" == *u* ]]; then
    common_had_nounset=1
    set +u
  fi
  # shellcheck disable=SC1091
  . /etc/profile
  if [[ "${common_had_nounset}" == "1" ]]; then
    set -u
  fi
fi

if [[ -f "${ENCODER_SLURM_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ENCODER_SLURM_ENV}"
elif [[ "${ALLOW_MISSING_SLURM_ENV:-0}" != "1" ]]; then
  echo "错误: 未找到 ${ENCODER_SLURM_ENV}" >&2
  echo "请先执行: cp slurm/cluster.env.example slurm/cluster.env" >&2
  return 1 2>/dev/null || exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_REPO_ROOT}}"
STORAGE_ROOT="${STORAGE_ROOT:-${HOME}/.local/share/encoder-sched}"
VENV="${VENV:-${PROJECT_ROOT}/.venv}"
HF_HOME="${HF_HOME:-${STORAGE_ROOT}/huggingface}"
TORCH_HOME="${TORCH_HOME:-${STORAGE_ROOT}/torch}"
DATASET_ROOT="${DATASET_ROOT:-${STORAGE_ROOT}/datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-${STORAGE_ROOT}/results}"
ENCODER_CONFIG="${ENCODER_CONFIG:-${PROJECT_ROOT}/config.yaml}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "错误: PROJECT_ROOT 不存在: ${PROJECT_ROOT}" >&2
  return 1 2>/dev/null || exit 1
fi
cd "${PROJECT_ROOT}"

if [[ "${SLURM_MODULE_PURGE:-1}" == "1" ]]; then
  if type module >/dev/null 2>&1; then
    module purge
  else
    echo "警告: module 命令不可用；请检查 /etc/profile 和集群环境" >&2
  fi
fi

if [[ -n "${SLURM_MODULES:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    echo "错误: 配置了 SLURM_MODULES，但 module 命令不可用" >&2
    return 1 2>/dev/null || exit 1
  fi
  # Module names cannot contain whitespace; intentional word splitting.
  # shellcheck disable=SC2086
  module load ${SLURM_MODULES}
fi

export HF_HOME TORCH_HOME
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${DATASET_ROOT}" "${RESULTS_ROOT}"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR="${SLURM_TMPDIR}/encoder-sched"
else
  export TMPDIR="${STORAGE_ROOT}/tmp/${SLURM_JOB_ID:-login}"
fi
mkdir -p "${TMPDIR}"

if [[ "${REQUIRE_VENV:-1}" == "1" ]]; then
  if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "错误: 虚拟环境不存在: ${VENV}" >&2
    echo "请先提交 bootstrap 作业: slurm/submit.sh bootstrap" >&2
    return 1 2>/dev/null || exit 1
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

echo "[slurm] job=${SLURM_JOB_ID:-none} host=$(hostname) root=${PROJECT_ROOT}"
echo "[slurm] python=$(command -v python3 || true) config=${ENCODER_CONFIG}"
if type module >/dev/null 2>&1; then
  module list 2>&1 || true
fi
