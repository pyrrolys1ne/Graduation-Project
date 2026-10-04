#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ALLOW_MISSING_SLURM_ENV=0
export REQUIRE_VENV=0
# shellcheck disable=SC1091
source "${ROOT}/slurm/lib/common.sh"

link_directory() {
  local link_path="$1"
  local target_path="$2"
  mkdir -p "${target_path}"

  if [[ -L "${link_path}" ]]; then
    local current
    current="$(readlink -f "${link_path}")"
    if [[ "${current}" != "$(readlink -f "${target_path}")" ]]; then
      echo "错误: ${link_path} 已指向 ${current}，不会覆盖" >&2
      return 1
    fi
    return
  fi
  if [[ -e "${link_path}" ]]; then
    if [[ -d "${link_path}" && -z "$(find "${link_path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir "${link_path}"
    else
      echo "保留已有路径 ${link_path}；请自行迁移内容后再创建软链接" >&2
      return
    fi
  fi
  ln -s "${target_path}" "${link_path}"
  echo "已连接 ${link_path} -> ${target_path}"
}

mkdir -p "${PROJECT_ROOT}/data"
link_directory "${PROJECT_ROOT}/data/datasets" "${DATASET_ROOT}"
link_directory "${PROJECT_ROOT}/results" "${RESULTS_ROOT}"

cat <<EOF
存储准备完成。
  HF_HOME=${HF_HOME}
  TORCH_HOME=${TORCH_HOME}
  datasets=${DATASET_ROOT}
  results=${RESULTS_ROOT}

说明：普通 Slurm 用户通常无权执行 mount。集群 home/project 文件系统已由管理员挂载，
本脚本用环境变量和软链接接入持久存储，不使用 sudo 或计算节点本地临时盘保存结果。
EOF
