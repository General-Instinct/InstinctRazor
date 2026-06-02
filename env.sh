#!/usr/bin/env bash
# moe-lowbit: source this before any pipeline (`source env.sh`).
#
# It does two essential things:
#   1. Sets PYTHONPATH to all three src/ subdirs. The modules cross-import by bare
#      name (`import moe_quant`, `import vllm_eval`, `import bench8_loaders`, ...),
#      so all of quant/ eval/ distill/ must be importable at once. This is the ONLY
#      packaging change vs. the original flat layout — no source logic was modified.
#   2. Activates the vLLM venv and sets HF + scratch/cache env, all overridable.
#
# Override any of these by exporting them before sourcing, e.g.:
#   MOE_LOWBIT_VENV=/path/to/venv HF_TOKEN=hf_xxx SCRATCH=/big/scratch source env.sh

HERE="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
export MOE_LOWBIT_ROOT="$HERE"

# (1) cross-subdir bare imports — required.
export PYTHONPATH="$HERE/src/quant:$HERE/src/eval:$HERE/src/distill${PYTHONPATH:+:$PYTHONPATH}"

# (2a) venv — point at your vLLM venv (default: the 4xH100 dev box).
: "${MOE_LOWBIT_VENV:=./vllm_venv}"
if [ -f "$MOE_LOWBIT_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$MOE_LOWBIT_VENV/bin/activate"
else
  echo "[env.sh] WARN: venv not found at $MOE_LOWBIT_VENV — using current python. Set MOE_LOWBIT_VENV." >&2
fi

# (2b) HuggingFace — set HF_TOKEN for gated datasets (HLE) / models.
: "${HF_HOME:=$HOME/.cache/huggingface}"
: "${HF_TOKEN:=}"
export HF_HOME HF_TOKEN
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_TRUST_REMOTE_CODE=1 VLLM_LOGGING_LEVEL=WARNING

# (2c) scratch + caches — on a big fast disk (Triton/vLLM compile caches live here).
# The dev box had a broken /tmp accounting bug; keeping scratch off /tmp avoids it.
: "${SCRATCH:=/tmp/moe-lowbit}"
export TMPDIR="$SCRATCH/tmp" TRITON_CACHE_DIR="$SCRATCH/triton" VLLM_CACHE_ROOT="$SCRATCH/vllm"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# (2d) allocator — reduces fragmentation OOMs on long-context vLLM and FSDP training.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[env.sh] MOE_LOWBIT_ROOT=$MOE_LOWBIT_ROOT  PYTHONPATH set (quant:eval:distill)  HF_HOME=$HF_HOME  SCRATCH=$SCRATCH"
