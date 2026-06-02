#!/usr/bin/env bash
# distill — Lightning-OPD on-policy distillation (optional). Closes a recoverable-gap axis (e.g. LiveCodeBench v6)
# by distilling the BF16 teacher into a rank-16 per-expert LoRA on the 3b student, then re-quantizing
# back to ~47 GB. Four phases across TWO venvs:
#   A gen   (vLLM eval venv)  : student rollouts, verified
#   B score (vLLM eval venv)  : BF16 teacher top-k logprobs per token
#   C train (torch2.7 venv)   : FSDP2 4-GPU, reverse-KL, STE per-expert LoRA  <-- torchrun
#   D merge (torch2.7 venv)   : bake LoRA into base + re-quantize -> vLLM-loadable ~47 GB ckpt
#
#   bash pipelines/distill.sh --smoke      # 2-step FSDP smoke first (prints SMOKE OK) — DO THIS FIRST
#   bash pipelines/distill.sh              # full A->B->C->D
#   OUT=/models/q122_opd_code bash pipelines/distill.sh   # override
#
# Each phase is guarded on the prior output; re-running skips completed phases. Train on the GAP'S
# DOMAIN (configs/opd_r1.env note) — for the LCB code gap, point opd_gen at competitive-programming CoT.
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck disable=SC1091
source "$HERE/env.sh"; source "$HERE/configs/opd_r1.env"
cd "$HERE"   # phase outputs (results/stage2/) resolve under the framework root
R="$HERE/src/distill"
: "${TP:=4}"
: "${MOE_LOWBIT_TRAIN_VENV:=/workspace/venv}"   # torch 2.7 + flash-linear-attention FSDP venv
TORCHRUN="$MOE_LOWBIT_TRAIN_VENV/bin/torchrun"
TRAIN_PY="$MOE_LOWBIT_TRAIN_VENV/bin/python"
SMOKE=0; [ "${1:-}" = "--smoke" ] && SMOKE=1
mkdir -p "$S"

gpu_free() { for _ in $(seq 1 30); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "$u" -lt 8000 ] && return 0; sleep 10; done; echo "[distill] WARN: GPUs still busy (${u}MiB)"; }

# --- Phase C smoke: validate the 4-GPU FSDP path on 2 steps before spending on gen/score ---
if [ "$SMOKE" = 1 ]; then
  echo "[distill] FSDP smoke (--smoke 2). Needs the 4 GPUs free."
  gpu_free; sleep 12
  # smoke needs a rollouts+teacher_lp pair; generate a tiny one if absent
  [ -s "$S/rollouts.jsonl" ]   || python "$R/opd_gen.py"   --model "$STU" --out "$S/rollouts.jsonl" --n 8 --k 2 --max-tokens 2048 --tp "$TP"
  [ -s "$S/teacher_lp.npz" ]   || python "$R/opd_score.py" --teacher "$TEA" --rollouts "$S/rollouts.jsonl" --out "$S/teacher_lp.npz" --k "$SCORE_K" --max-len "$SCORE_MAX_LEN" --tp "$TP"
  gpu_free; sleep 12
  "$TORCHRUN" --nproc_per_node=4 --master_port=29512 "$R/opd_train_fsdp.py" \
    --model "$TEA" --rollouts "$S/rollouts.jsonl" --teacher-lp "$S/teacher_lp.npz" \
    --adapter-out "$S/adapter_smoke" --rank "$RANK" --scale "$SCALE" --bits "$BITS" --group "$GROUP" \
    --max-len "$TRAIN_MAX_LEN" --smoke 2
  echo "[distill] smoke complete — look for 'SMOKE OK' above, then run without --smoke."
  exit 0
fi

# --- Phase A (eval venv): student rollouts ---
if [ ! -s "$S/rollouts.jsonl" ]; then
  echo "[distill] A: gen rollouts -> $S/rollouts.jsonl"
  python "$R/opd_gen.py" --model "$STU" --out "$S/rollouts.jsonl" --n "$GEN_N" --k "$GEN_K" --max-tokens "$GEN_MAX_TOKENS" --tp "$TP"
fi
# --- Phase B (eval venv): teacher top-k logprobs ---
if [ ! -s "$S/teacher_lp.npz" ]; then
  echo "[distill] B: teacher score -> $S/teacher_lp.npz"; gpu_free; sleep 8
  python "$R/opd_score.py" --teacher "$TEA" --rollouts "$S/rollouts.jsonl" --out "$S/teacher_lp.npz" --k "$SCORE_K" --max-len "$SCORE_MAX_LEN" --tp "$TP"
fi
# --- Phase C (train venv): FSDP2 reverse-KL train ---
if [ ! -e "$S/adapter_r1_fsdp/adapter.pt" ]; then
  echo "[distill] C: FSDP2 train (torchrun, train venv) -> $S/adapter_r1_fsdp"; gpu_free; sleep 12
  "$TORCHRUN" --nproc_per_node=4 --master_port="$MASTER_PORT" "$R/opd_train_fsdp.py" \
    --model "$TEA" --rollouts "$S/rollouts.jsonl" --teacher-lp "$S/teacher_lp.npz" \
    --adapter-out "$S/adapter_r1_fsdp" --rank "$RANK" --scale "$SCALE" --bits "$BITS" --group "$GROUP" \
    --epochs "$EPOCHS" --accum "$ACCUM" --lr "$LR" --max-len "$TRAIN_MAX_LEN" --reverse "$REVERSE" --tau "$TAU" --ckpt-every 5
fi
# --- Phase D (train venv): bake + re-quantize ---
echo "[distill] D: merge + re-quantize -> $OUT"; gpu_free; sleep 12
"$TRAIN_PY" "$R/merge_adapter.py" --model "$TEA" --adapter "$S/adapter_r1_fsdp" --out "$OUT" \
  --rank "$RANK" --scale "$SCALE" --bits "$BITS" --group "$GROUP" --clip-steps 24
echo "[distill] DONE -> $OUT  (re-eval with: bash pipelines/eval.sh --model $OUT --benchmarks lcb --budget 64k)"
