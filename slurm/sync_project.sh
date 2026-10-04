#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法: $0 <user@login-host> <remote-project-path> [额外 rsync 参数]" >&2
  exit 2
fi

host="$1"
remote_path="$2"
shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "SSH 可能要求密码和二次验证码；本脚本不会保存或绕过二次验证。"
exec rsync -az --info=progress2 \
  --exclude='.git/' \
  --exclude='.venv*/' \
  --exclude='results/' \
  --exclude='slurm/cluster.env' \
  --exclude='__pycache__/' \
  "$@" "${ROOT}/" "${host}:${remote_path%/}/"
