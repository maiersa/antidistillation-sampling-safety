#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-pipeline.sh}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_DIR="${EXP_DIR:-$SCRIPT_DIR/experiments/full_pipeline_$TIMESTAMP}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/pipeline_$TIMESTAMP.log}"
REPO_PYTHON="$SCRIPT_DIR/.venv/bin/python"

mkdir -p "$LOG_DIR"

echo "[pipeline] repo: $SCRIPT_DIR"
echo "[pipeline] pipeline_script=$PIPELINE_SCRIPT"
echo "[pipeline] exp_dir=$EXP_DIR"
echo "[pipeline] log_file=$LOG_FILE"

python -m pip install -U uv ninja packaging psutil
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv sync

if [[ ! -x "$REPO_PYTHON" ]]; then
    echo "[pipeline] missing repo python: $REPO_PYTHON"
    exit 1
fi

uv pip install --python "$REPO_PYTHON" "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

export EXP_DIR
export LOG_DIR
export LOG_FILE
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

bash "$SCRIPT_DIR/$PIPELINE_SCRIPT"