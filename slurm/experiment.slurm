#!/usr/bin/env bash
#SBATCH --job-name=encoder-experiment
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
run_dir="${RESULTS_ROOT}/slurm/${SLURM_JOB_ID}"
mkdir -p "${run_dir}"
server_log="${run_dir}/server.log"

encoder-sched-server --config "${ENCODER_CONFIG}" --host 127.0.0.1 --port "${port}" \
  >"${server_log}" 2>&1 &
server_pid=$!
cleanup() {
  kill -INT "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "错误: 服务提前退出；日志如下" >&2
    tail -100 "${server_log}" >&2
    exit 1
  fi
  if curl --silent --fail "http://127.0.0.1:${port}/health" >"${run_dir}/health.json"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "错误: 服务在 180 秒内未就绪" >&2
  tail -100 "${server_log}" >&2
  exit 1
fi

python scripts/run_experiment.py \
  --url "http://127.0.0.1:${port}" \
  --workload "${WORKLOAD}" \
  --output "${run_dir}/requests.jsonl" \
  --concurrency "${CLIENT_CONCURRENCY:-8}" \
  --shards "${CLIENT_SHARDS:-1}"

echo "实验结果: ${run_dir}"
