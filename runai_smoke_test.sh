#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_COUNT="${GPU_COUNT:-4}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.3}"
RUN_GENTRACES_SMOKE="${RUN_GENTRACES_SMOKE:-0}"
GENTRACES_PROCESSES="${GENTRACES_PROCESSES:-$GPU_COUNT}"
GENTRACES_MAX_SAMPLES="${GENTRACES_MAX_SAMPLES:-128}"
GENTRACES_MAX_PROMPT_LENGTH="${GENTRACES_MAX_PROMPT_LENGTH:-512}"
EXP_DIR="${EXP_DIR:-$SCRIPT_DIR/experiments/runai_smoke}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/runai_smoke_$(date +%Y%m%d_%H%M%S).log}"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[smoke] repo: $SCRIPT_DIR"
echo "[smoke] gpu_count=$GPU_COUNT"
echo "[smoke] exp_dir=$EXP_DIR"
echo "[smoke] log_file=$LOG_FILE"
echo "[smoke] gentraces_processes=$GENTRACES_PROCESSES"

python -m pip install -U uv ninja packaging psutil
uv sync

REPO_PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$REPO_PYTHON" ]]; then
    echo "[smoke] missing repo python: $REPO_PYTHON"
    exit 1
fi

echo "[smoke] repo_python=$REPO_PYTHON"

uv pip install --python "$REPO_PYTHON" "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

"$REPO_PYTHON" -V
nvidia-smi

"$REPO_PYTHON" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(
        f"gpu[{index}]={torch.cuda.get_device_name(index)} capability={torch.cuda.get_device_capability(index)}"
    )
PY

"$REPO_PYTHON" - <<'PY'
import flash_attn

print(f"flash_attn={flash_attn.__version__}")
PY

cat > /tmp/runai_dist_smoke.py <<'PY'
import os
import socket

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    value = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    if abs(value.item() - expected) > 1e-5:
        raise RuntimeError(f"all_reduce mismatch: got {value.item()} expected {expected}")

    print(
        f"rank={rank} world_size={world_size} host={socket.gethostname()} "
        f"device={torch.cuda.get_device_name(local_rank)} all_reduce_sum={value.item()}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

uv run --python "$REPO_PYTHON" torchrun --standalone --nproc_per_node="$GPU_COUNT" /tmp/runai_dist_smoke.py

"$REPO_PYTHON" - <<'PY'
import torch
from flash_attn import flash_attn_func

device = "cuda"
batch = 2
seqlen = 128
nheads = 8
headdim = 64

q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.float16, requires_grad=True)
k = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.float16, requires_grad=True)
v = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.float16, requires_grad=True)

out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
loss = out.square().mean()
loss.backward()

print(f"flash_attn_forward_backward_ok shape={tuple(out.shape)} grad_norm={q.grad.norm().item():.6f}")
PY

if [[ "$RUN_GENTRACES_SMOKE" == "1" ]]; then
    mkdir -p "$EXP_DIR"
    HYDRA_FULL_ERROR=1 uv run --python "$REPO_PYTHON" accelerate launch --num_processes "$GENTRACES_PROCESSES" gentraces.py \
        hydra.run.dir="$EXP_DIR/metadata/holdout" \
        teacher=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        exp_dir="$EXP_DIR" \
        seed=42 \
        data_split=gsm8k_holdout \
        max_samples="$GENTRACES_MAX_SAMPLES" \
        batch_size=1 \
        max_length=256 \
        max_prompt_length="$GENTRACES_MAX_PROMPT_LENGTH" \
        use_wandb=false \
        trace_name=holdout_smoke
fi

echo "[smoke] completed"