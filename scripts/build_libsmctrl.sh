#!/usr/bin/env bash
# 构建真实 libsmctrl 共享库。
#
# 只克隆上游原仓库并用它自己的 Makefile 构建，不下载来源不明的二进制、
# 也不在本地重新实现该库。库本身只需 gcc -shared（不需要 nvcc）；
# nvcc 只用于上游自带的验证测试程序。
#
# 用法:
#   scripts/build_libsmctrl.sh [--prefix DIR] [--src DIR] [--check-only]
set -euo pipefail

UPSTREAM_URL="${LIBSMCTRL_UPSTREAM:-http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git}"
SRC_DIR="${LIBSMCTRL_SRC:-$HOME/.cache/libsmctrl}"
PREFIX="${LIBSMCTRL_PREFIX:-$HOME/.local/lib/libsmctrl}"
CHECK_ONLY=0
APPLY_PATCH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --src) SRC_DIR="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --no-patch) APPLY_PATCH=0; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

fail() { echo "错误: $*" >&2; exit 1; }

echo "== 前置检查 =="
command -v gcc >/dev/null || fail "缺少 gcc"
echo "  gcc: $(gcc --version | head -1)"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -f "$CUDA_HOME/include/cuda.h" ]]; then
  echo "  cuda.h: $CUDA_HOME/include/cuda.h"
else
  cat >&2 <<EOF
  错误: 找不到 cuda.h（已查找 $CUDA_HOME/include/cuda.h）

  libsmctrl 需要 CUDA 头文件来链接 -lcuda，但**不需要 nvcc**。可选方案：
    1) 安装 CUDA Toolkit（任意与驱动兼容的版本，只用其头文件）
    2) 只装头文件包，例如 nvidia-cuda-dev，并设置 CUDA_HOME 指向其安装位置
    3) 若已装在别处: CUDA_HOME=/path/to/cuda $0
EOF
  exit 1
fi

command -v nvcc >/dev/null && echo "  nvcc: $(nvcc --version | tail -1)" \
  || echo "  nvcc: 缺失（仅影响上游验证测试，不影响构建 .so）"

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "== check-only 结束，未执行构建 =="
  exit 0
fi

echo "== 获取上游源码 =="
if [[ -d "$SRC_DIR/.git" ]]; then
  echo "  复用已有克隆: $SRC_DIR"
  git -C "$SRC_DIR" fetch --quiet --all || true
  git -C "$SRC_DIR" checkout --quiet master || true
  git -C "$SRC_DIR" pull --quiet --ff-only || echo "  警告: 拉取失败，使用本地版本"
else
  rm -rf "$SRC_DIR"
  # 不加 --depth 1：上游的 cgit 走 dumb http transport，不支持浅克隆。
  git clone --quiet "$UPSTREAM_URL" "$SRC_DIR" \
    || fail "克隆失败: $UPSTREAM_URL （网络受限时可手动克隆后用 --src 指定）"
fi
echo "  提交: $(git -C "$SRC_DIR" rev-parse --short HEAD)"

PATCH_FILE="${LIBSMCTRL_PATCH:-$(cd "$(dirname "$0")/.." && pwd)/patches/libsmctrl-thread-mask.patch}"
if [[ "$APPLY_PATCH" == "1" && -f "$PATCH_FILE" ]]; then
  echo "== 应用补丁 =="
  echo "  $PATCH_FILE"
  if grep -q "libsmctrl_set_thread_mask" "$SRC_DIR/libsmctrl.c"; then
    echo "  已是打过补丁的状态，跳过"
  else
    patch -d "$SRC_DIR" -p1 --forward < "$PATCH_FILE" \
      || fail "补丁应用失败；上游可能已变动，请人工核对 $PATCH_FILE"
    echo "  已应用（新增 libsmctrl_set_thread_mask：粘性线程掩码）"
  fi
elif [[ "$APPLY_PATCH" != "1" ]]; then
  echo "== 跳过补丁（--no-patch）=="
fi

echo "== 构建 libsmctrl.so =="
make -C "$SRC_DIR" libsmctrl.so CUDA="$CUDA_HOME"

[[ -f "$SRC_DIR/libsmctrl.so" ]] || fail "构建结束但未生成 libsmctrl.so"

mkdir -p "$PREFIX"
cp "$SRC_DIR/libsmctrl.so" "$PREFIX/"
echo "== 完成 =="
echo "  库文件: $PREFIX/libsmctrl.so"
echo
echo "  在项目中启用:"
echo "    export LIBSMCTRL_PATH=$PREFIX/libsmctrl.so"
echo "  然后确认能力探针（本机驱动为 CUDA 13.x 时探针会明确报告不支持）:"
echo "    python scripts/probe_libsmctrl.py --config config.libsmctrl.example.yaml"
