#!/usr/bin/env bash
#SBATCH --job-name=encoder-bootstrap
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
export REQUIRE_VENV=0
if [[ -n "${ENCODER_SLURM_ENV:-}" ]]; then
  common="$(dirname "${ENCODER_SLURM_ENV}")/lib/common.sh"
else
  common="${SLURM_SUBMIT_DIR}/slurm/lib/common.sh"
fi
# shellcheck disable=SC1091
source "${common}"

nvidia-smi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}" \
VENV="${VENV}" \
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}" \
CONFIG_PATH="${ENCODER_CONFIG}" \
  "${PROJECT_ROOT}/scripts/bootstrap_server.sh"

"${VENV}/bin/python" -m pytest -q
