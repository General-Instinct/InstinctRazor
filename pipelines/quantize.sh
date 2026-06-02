#!/usr/bin/env bash
# quantize — BF16 -> 47GB quantized clip. Reproduces the shipped q122_ptq3b_clip exactly.
#
#   bash pipelines/quantize.sh            # the shipped uniform 3b/4b + clip-search recipe
#   PROBE=1 bash pipelines/quantize.sh    # also run the (optional) router/salience probe first
#   OUT=/path/to/out EXPERT_BITS=2.5 bash pipelines/quantize.sh   # override anything
#
# Hardware: 4x H100-80GB (device_map="auto" load, ~72 GiB/GPU). Writes a DEQUANTIZED bf16 checkpoint
# (~229 GB on disk) that vLLM loads to eval the capability ceiling of the ~47 GB packed recipe.
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck disable=SC1091
source "$HERE/env.sh"
# shellcheck disable=SC1091
source "$HERE/configs/clip_122b.env"
cd "$HERE"   # relative results/ paths (probe artifact) resolve under the framework root
Q="$HERE/src/quant"

echo "[quantize] recipe: experts=${EXPERT_BITS}b linears=${LINEAR_BITS}b group=${GROUP} clip_steps=${CLIP_STEPS}"
echo "[quantize] model=$MODEL  ->  out=$OUT"

# (optional, research) probe the router: per-expert freq/wmass/asal salience + load-balance stats.
# Establishes finding F4/F6 (allocation is second-order); NOT part of the shipped uniform recipe.
if [ "${PROBE:-0}" = "1" ]; then
  echo "[quantize] PROBE: router/salience probe -> results/moe_probe_<tag>.json"
  python "$Q/moe_probe.py" --model "$MODEL" --tag "$(basename "$OUT")" --max-mem-gib "$MAX_MEM_GIB"
fi

# the shipped recipe: uniform int3 experts + int4 backbone + per-block MSE clip-search.
python "$Q/quant_save.py" \
  --model "$MODEL" \
  --expert-bits "$EXPERT_BITS" \
  --linear-bits "$LINEAR_BITS" \
  --group "$GROUP" \
  --clip-steps "$CLIP_STEPS" \
  --max-mem-gib "$MAX_MEM_GIB" \
  --out "$OUT"

echo "[quantize] DONE -> $OUT  (eval it with: bash pipelines/eval.sh --model $OUT)"
echo "[quantize] NOTE: this is a fake-quant bf16 checkpoint (capability ceiling). For deployment, pack to"
echo "[quantize]       GGUF/llama.cpp at the matching bit-widths — see docs/MOE_PIPELINE.md."
