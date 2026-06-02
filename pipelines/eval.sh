#!/usr/bin/env bash
# eval — eval a model on the shared harness (the same one that produced results/).
#
#   bash pipelines/eval.sh --model /path/to/q122_ptq3b_clip --tag clip --benchmarks mmlu_pro,gpqa,mmmlu
#   bash pipelines/eval.sh --model google/gemma-4-26b-a4b-it --tag a4b --benchmarks lcb --budget 64k
#   bash pipelines/eval.sh --model Qwen/Qwen3.5-122B-A10B  --tag teacher --benchmarks mmmu_pro,mathvision
#   # any extra args after the known flags are passed through to the underlying tool, e.g. --lcb-n 200
#
# Text benches (mmlu_pro gpqa mmmlu bbeh aime aime2026 hmmt math500 humaneval mbpp lcb hle) -> vllm_eval8.py
# Multimodal benches (mmmu mmmu_pro mathvision medxpert)                                    -> mm_eval.py
# --budget 64k uses the long-context config (math/code/HLE). Gemma models auto-get --enforce-eager.
#
# VALIDATION GATE: before trusting any head-to-head, reproduce the model's OWN official number on that
# benchmark to ~1-2 pt (targets in configs/eval.env). See docs/EVAL_PROTOCOL.md.
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck disable=SC1091
source "$HERE/env.sh"; source "$HERE/configs/eval.env"
cd "$HERE"   # vllm_eval8.py / mm_eval.py write relative results/{vllm_eval,mm}/ — resolve under framework root
E="$HERE/src/eval"

MODEL=""; TAG=""; BENCHMARKS="mmlu_pro,gpqa,mmmlu"; BUDGET="32k"; PASS=()
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --tag) TAG="$2"; shift 2;;
  --benchmarks) BENCHMARKS="$2"; shift 2;;
  --budget) BUDGET="$2"; shift 2;;
  *) PASS+=("$1"); shift;;
esac; done
[ -n "$MODEL" ] || { echo "ERROR: --model required" >&2; exit 2; }
[ -n "$TAG" ] || TAG="$(basename "$MODEL")"

# budget -> vLLM context flags
if [ "$BUDGET" = "64k" ]; then
  MT="$MAX_TOKENS_64K"; ML="$MAX_MODEL_LEN_64K"; MS="$MAX_NUM_SEQS_64K"; GM="$GPU_MEM_64K"
else
  MT="$MAX_TOKENS"; ML="$MAX_MODEL_LEN"; MS="$MAX_NUM_SEQS"; GM="$GPU_MEM"
fi

# enforce-eager: Gemma-4/A4B crash at long context in vLLM 0.22 without it (Qwen does not need it).
EAGER=()
shopt -s nocasematch
[[ "$MODEL" == *gemma* ]] && { EAGER=(--enforce-eager); echo "[eval] gemma model -> --enforce-eager (long-context stability)"; }
shopt -u nocasematch

# split benches into text vs multimodal
TEXT=""; MM=""
IFS=',' read -ra BS <<< "$BENCHMARKS"
for b in "${BS[@]}"; do case "$b" in
  mmmu|mmmu_pro|mathvision|medxpert) MM="${MM:+$MM,}$b";;
  *) TEXT="${TEXT:+$TEXT,}$b";;
esac; done

if [ -n "$TEXT" ]; then
  echo "[eval] TEXT benches=$TEXT  budget=$BUDGET (max_tokens=$MT model_len=$ML)  tag=$TAG"
  python "$E/vllm_eval8.py" --model "$MODEL" --tag "$TAG" --benchmarks "$TEXT" \
    --tp "$TP" --max-tokens "$MT" --max-model-len "$ML" --max-num-seqs "$MS" --gpu-mem "$GM" \
    "${EAGER[@]}" "${PASS[@]}"
fi
if [ -n "$MM" ]; then
  echo "[eval] MULTIMODAL benches=$MM  budget=$MT  tag=mm_$TAG"
  python "$E/mm_eval.py" --model "$MODEL" --tag "mm_$TAG" --benches "$MM" \
    --tp "$TP" --budget "$MT" --n "$N_MM" "${EAGER[@]}" "${PASS[@]}"
fi
echo "[eval] DONE -> results/vllm_eval/$TAG.json ${MM:+and results/mm/mm_$TAG.json}"
