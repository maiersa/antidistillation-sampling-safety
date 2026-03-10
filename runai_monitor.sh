#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_DIR="${EXP_DIR:-$SCRIPT_DIR/experiments/full_pipeline_$TIMESTAMP}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"
MONITOR_LOG="${MONITOR_LOG:-$LOG_DIR/monitor_$TIMESTAMP.log}"
REPO_PYTHON="$SCRIPT_DIR/.venv/bin/python"

INPUT_TRACE_PATHS="${INPUT_TRACE_PATHS:-[\"$EXP_DIR/traces/holdout\"]}"
INPUT_TRACE_PATHS_LIST="${INPUT_TRACE_PATHS_LIST:-}"
TRACE_COLUMNS="${TRACE_COLUMNS:-[trace,trace_af]}"
ONLY_CORRECT="${ONLY_CORRECT:-true}"
MAX_EXAMPLES_PER_TRACE_COLUMN="${MAX_EXAMPLES_PER_TRACE_COLUMN:-null}"
MATCH_EXAMPLES_ACROSS_TRACES="${MATCH_EXAMPLES_ACROSS_TRACES:-false}"

MONITOR_BACKEND="${MONITOR_BACKEND:-transformers}"
MONITOR_MODEL_NAME="${MONITOR_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MONITOR_MODEL_SLUG="${MONITOR_MODEL_SLUG:-qwen2_5_7b_instruct}"
MONITOR_BATCH_SIZE="${MONITOR_BATCH_SIZE:-2}"
MONITOR_MAX_NEW_TOKENS="${MONITOR_MAX_NEW_TOKENS:-384}"
MONITOR_MAX_PROMPT_LENGTH="${MONITOR_MAX_PROMPT_LENGTH:-8192}"
MONITOR_DTYPE="${MONITOR_DTYPE:-bfloat16}"
MONITOR_DEVICE_MAP="${MONITOR_DEVICE_MAP:-null}"
PROMPT_PATH="${PROMPT_PATH:-prompts/monitorability_2510_23966_prompt.txt}"
ALLOW_PROMPT_SCAFFOLD="${ALLOW_PROMPT_SCAFFOLD:-false}"
SAVE_PROMPT_TEXT="${SAVE_PROMPT_TEXT:-true}"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$MONITOR_LOG") 2>&1

if [[ -n "$INPUT_TRACE_PATHS_LIST" ]]; then
    normalized_lines="$(printf '%s\n' "$INPUT_TRACE_PATHS_LIST" | tr ',' '\n' | sed 's/^ *//; s/ *$//' | sed '/^$/d')"
    if [[ -z "$normalized_lines" ]]; then
        echo "[monitor] INPUT_TRACE_PATHS_LIST was provided but no valid paths were found"
        exit 1
    fi

    hydra_list="["
    first_item=true
    while IFS= read -r trace_path; do
        escaped_path="${trace_path//\\/\\\\}"
        escaped_path="${escaped_path//\"/\\\"}"
        if [[ "$first_item" == true ]]; then
            hydra_list+="\"$escaped_path\""
            first_item=false
        else
            hydra_list+=",\"$escaped_path\""
        fi
    done <<< "$normalized_lines"
    hydra_list+="]"
    INPUT_TRACE_PATHS="$hydra_list"
fi

echo "[monitor] repo: $SCRIPT_DIR"
echo "[monitor] exp_dir=$EXP_DIR"
echo "[monitor] monitor_log=$MONITOR_LOG"
echo "[monitor] input_trace_paths=$INPUT_TRACE_PATHS"
echo "[monitor] trace_columns=$TRACE_COLUMNS"
echo "[monitor] max_examples_per_trace_column=$MAX_EXAMPLES_PER_TRACE_COLUMN"
echo "[monitor] match_examples_across_traces=$MATCH_EXAMPLES_ACROSS_TRACES"
echo "[monitor] monitor_backend=$MONITOR_BACKEND"
echo "[monitor] monitor_model_name=$MONITOR_MODEL_NAME"

python -m pip install -U uv ninja packaging psutil
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv sync

if [[ ! -x "$REPO_PYTHON" ]]; then
    echo "[monitor] missing repo python: $REPO_PYTHON"
    exit 1
fi

uv pip install --python "$REPO_PYTHON" "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

export EXP_DIR
export LOG_DIR
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

uv run accelerate launch --config_file acc_config.yaml score_monitor.py \
    exp_dir="$EXP_DIR" \
    input_trace_paths="$INPUT_TRACE_PATHS" \
    trace_columns="$TRACE_COLUMNS" \
    only_correct="$ONLY_CORRECT" \
    max_examples_per_trace_column="$MAX_EXAMPLES_PER_TRACE_COLUMN" \
    match_examples_across_traces="$MATCH_EXAMPLES_ACROSS_TRACES" \
    allow_prompt_scaffold="$ALLOW_PROMPT_SCAFFOLD" \
    save_prompt_text="$SAVE_PROMPT_TEXT" \
    monitor.backend="$MONITOR_BACKEND" \
    monitor.model_name="$MONITOR_MODEL_NAME" \
    monitor.model_slug="$MONITOR_MODEL_SLUG" \
    monitor.batch_size="$MONITOR_BATCH_SIZE" \
    monitor.max_new_tokens="$MONITOR_MAX_NEW_TOKENS" \
    monitor.max_prompt_length="$MONITOR_MAX_PROMPT_LENGTH" \
    monitor.torch_dtype="$MONITOR_DTYPE" \
    monitor.device_map="$MONITOR_DEVICE_MAP" \
    monitor.prompt_path="$PROMPT_PATH" \
    "$@" \
    2>&1 | tee -a "$MONITOR_LOG"