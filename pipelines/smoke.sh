#!/usr/bin/env bash
# smoke — single-GPU END-TO-END smoke. Verifies the whole framework (load -> PTQ -> dequant-save ->
# vLLM reload -> eval) on a SMALL DENSE model, no 4xH100 / 122B needed. ~10-20 min on one H100/A100/4090.
#
#   bash pipelines/smoke.sh
#   SMOKE_MODEL=Qwen/Qwen3-0.6B bash pipelines/smoke.sh
#
# Exercises the int4 LINEAR quant path + clip-search + the eval harness. A dense model has no fused
# experts, so the int3 EXPERT path is a no-op here (that path is verified at 122B scale; see build logs
# referenced in docs/MOE_PIPELINE.md). Success = both pipeline steps complete and a results JSON is written.
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck disable=SC1091
source "$HERE/env.sh"; source "$HERE/configs/smoke_2b.env"
cd "$HERE"   # relative results/ paths resolve under the framework root
Q="$HERE/src/quant"; E="$HERE/src/eval"

echo "[smoke] === STEP 1/2: quantize $MODEL (${EXPERT_BITS}b experts / ${LINEAR_BITS}b linears, clip ${CLIP_STEPS}) ==="
python "$Q/quant_save.py" --model "$MODEL" --expert-bits "$EXPERT_BITS" --linear-bits "$LINEAR_BITS" \
  --group "$GROUP" --clip-steps "$CLIP_STEPS" --max-mem-gib "$MAX_MEM_GIB" --out "$OUT"

echo "[smoke] === STEP 2/2: eval the quantized smoke model (tp=$TP, benches=$SMOKE_BENCHMARKS, n=$SMOKE_N) ==="
python "$E/vllm_eval8.py" --model "$OUT" --tag smoke_2b --benchmarks "$SMOKE_BENCHMARKS" \
  --tp "$TP" --max-tokens "$SMOKE_BUDGET" --max-model-len "$((SMOKE_BUDGET + 2048))" --max-num-seqs 32 --gpu-mem 0.90 \
  --mmlu-n "$SMOKE_N" --math-n "$SMOKE_N" --gpqa-n "$SMOKE_N" --mmmlu-n "$SMOKE_N" --he-n "$SMOKE_N"

echo "[smoke] === SMOKE OK ===  quantized ckpt: $OUT   results: results/vllm_eval/smoke_2b.json"
echo "[smoke] If both steps printed a number, the framework is wired correctly end-to-end on one GPU."
