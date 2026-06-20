#!/usr/bin/env bash
# smoke_moe — single-GPU cross-family MoE smoke. Proves the ModelAdapter generalization on a real,
# non-Qwen, text-only MoE (OLMoE): its experts now get QUANTIZED via the model_type-keyed adapter, where
# the original Qwen3.5-only code quantized zero of them. (OLMoE under transformers 5.x is FUSED, like
# Qwen3.5; the GenericMoEAdapter additionally covers legacy per-expert-Linear layouts.) ~5-15 min, one GPU.
#
#   bash pipelines/smoke_moe.sh
#   SMOKE_MODEL=allenai/OLMoE-1B-7B-0924 bash pipelines/smoke_moe.sh
#
# Success = STEP 1 reports a large linears count INCLUDING the experts (asserted below), and STEP 2
# reloads the dequant checkpoint in vLLM and writes a results JSON.
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck disable=SC1091
source "$HERE/env.sh"; source "$HERE/configs/smoke_moe.env"
cd "$HERE"
Q="$HERE/src/quant"; E="$HERE/src/eval"

echo "[smoke_moe] === STEP 1/2: quantize SEPARATE-expert MoE $MODEL (${EXPERT_BITS}b experts / ${LINEAR_BITS}b backbone) ==="
LOG="$(mktemp)"
python "$Q/quant_save.py" --model "$MODEL" --expert-bits "$EXPERT_BITS" --linear-bits "$LINEAR_BITS" \
  --group "$GROUP" --clip-steps "$CLIP_STEPS" --max-mem-gib "$MAX_MEM_GIB" --out "$OUT" 2>&1 | tee "$LOG"

# Assert the experts were actually quantized. OLMoE under transformers 5.x is FUSED, so experts show up
# as "expert tensors" (nL x 2); a legacy per-expert-Linear MoE would instead show a large "linears" count.
# Either path proves the adapter routed the experts to the quantizer (the old code did neither -> 0/66).
PTQLINE="$(grep -oE 'quantized [0-9]+ expert tensors, [0-9]+ linears' "$LOG" | tail -1)"
NEXP="$(echo "$PTQLINE" | grep -oE '[0-9]+ expert' | grep -oE '[0-9]+' || echo 0)"
NLIN="$(echo "$PTQLINE" | grep -oE '[0-9]+ linears' | grep -oE '[0-9]+' || echo 0)"
echo "[smoke_moe] apply_ptq: ${NEXP:-0} fused expert tensors + ${NLIN:-0} linears quantized"
if [ "${NEXP:-0}" -lt 1 ] && [ "${NLIN:-0}" -lt 500 ]; then
  echo "[smoke_moe] FAIL: experts not quantized (expert_tensors=${NEXP}, linears=${NLIN})." >&2
  echo "[smoke_moe] The adapter is not routing this model's experts to the quantizer." >&2
  exit 1
fi

echo "[smoke_moe] === STEP 2/2: eval the quantized OLMoE (tp=$TP, benches=$SMOKE_BENCHMARKS, n=$SMOKE_N) ==="
python "$E/vllm_eval8.py" --model "$OUT" --tag smoke_olmoe --benchmarks "$SMOKE_BENCHMARKS" \
  --tp "$TP" --max-tokens "$SMOKE_BUDGET" --max-model-len "$((SMOKE_BUDGET + 2048))" --max-num-seqs 32 --gpu-mem 0.90 \
  --mmlu-n "$SMOKE_N" --math-n "$SMOKE_N" --gpqa-n "$SMOKE_N" --mmmlu-n "$SMOKE_N" --he-n "$SMOKE_N"

echo "[smoke_moe] === SMOKE_MOE OK ===  quantized separate-expert ckpt: $OUT   results: results/vllm_eval/smoke_olmoe.json"
