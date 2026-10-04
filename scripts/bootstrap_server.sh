#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-$ROOT/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/config.yaml}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "错误: 服务器部署只支持 Linux" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"需要 Python >= 3.10，当前为 {sys.version.split()[0]}")
PY

command -v nvidia-smi >/dev/null || {
  echo "错误: 找不到 nvidia-smi，请先安装 NVIDIA 驱动" >&2
  exit 1
}
nvidia-smi >/dev/null

"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

if ! "$VENV/bin/python" -c 'import torch' >/dev/null 2>&1; then
  if [[ -z "$TORCH_INDEX_URL" ]]; then
    cat >&2 <<'EOF'
错误: 虚拟环境中没有 PyTorch。
请按 PyTorch 官方矩阵指定与服务器驱动兼容的 wheel，例如：
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 scripts/bootstrap_server.sh
CUDA 13.0 驱动可以运行不高于其支持版本的 PyTorch CUDA runtime；无需强求 wheel 名称为 cu130。
EOF
    exit 1
  fi
  "$VENV/bin/python" -m pip install torch --index-url "$TORCH_INDEX_URL"
fi

"$VENV/bin/python" -m pip install -e "$ROOT[gpu,test,plot]"
"$VENV/bin/encoder-sched-check" --config "$CONFIG_PATH"

echo "服务器环境已就绪: $VENV"
echo "启动命令: $VENV/bin/encoder-sched-server --config $CONFIG_PATH"
