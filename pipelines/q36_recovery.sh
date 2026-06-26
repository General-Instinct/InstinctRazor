#!/usr/bin/env bash
# Full OPD recovery on Qwen3.6-35B-A3B (the now-unblocked Phase-C path).
# A gen (student rollouts, vLLM) -> B score (BF16 teacher top-k, vLLM) -> C FSDP train (replicated-LoRA,
# --checkpoint 0 dynamic loop) -> D merge+requant -> E eval baseline + recovered (matched 32k budget).
# Each phase guards on its output so a re-run resumes. Goal: does OPD cut the student's TRUNCATION
# (GPQA 43/198, MATH-500 15/120) and lift raw acc toward acc_finished (86.5 / 90.5)?
set -uo pipefail
cd /lambda/nfs/GI-a100-80GB/InstinctRazor
export MOE_LOWBIT_VENV="$PWD/vllm_venv"
source env.sh >/dev/null 2>&1
VLLM=./vllm_venv/bin/python
TRUN=./train_venv/bin/torchrun
TPY=./train_venv/bin/python
S=runs/q36_opd/recovery; mkdir -p "$S" results/campaign_logs
STU=models/q36_ptq3b_clip
TEA=Qwen/Qwen3.6-35B-A3B
REC=models/q36_opd_recovered

gpu_free(){ for _ in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|awk '{s+=$1}END{print s+0}')
  [ "${u:-9999}" -lt 5000 ] && return 0; sleep 10; done
  echo "[recovery] WARN GPUs busy (${u}MiB) — proceeding"; }
say(){ echo "[recovery $(date +%H:%M:%S)] $*"; }

: "${MAXLEN:=2048}"     # train/score sequence cap; long enough for real math CoT (was 1024, mostly truncated)
: "${GENTOK:=1920}"     # student gen cap (+prompt ~ MAXLEN). Long seq is affordable now: expert-body checkpointing
                        # frees the seq-independent ~64GB of reconstructed weights, base stays GPU-resident.

say "A: student rollouts -> $S/rollouts.jsonl (max-tokens $GENTOK)"
if [ ! -s "$S/rollouts.jsonl" ]; then gpu_free
  CUDA_VISIBLE_DEVICES=0,1,2,3 $VLLM src/distill/opd_gen.py --model "$STU" \
    --out "$S/rollouts.jsonl" --n 384 --k 2 --max-tokens "$GENTOK" --tp 4 || { say "A FAILED"; exit 11; }
fi
say "A done: $(wc -l < "$S/rollouts.jsonl") rollouts"

say "B: teacher top-k logprobs -> $S/teacher_lp.npz (max-len $MAXLEN)"
if [ ! -s "$S/teacher_lp.npz" ]; then gpu_free
  CUDA_VISIBLE_DEVICES=0,1,2,3 $VLLM src/distill/opd_score.py --teacher "$TEA" \
    --rollouts "$S/rollouts.jsonl" --out "$S/teacher_lp.npz" --k 20 --max-len "$MAXLEN" --tp 4 || { say "B FAILED"; exit 12; }
fi
say "B done"

say "C: FSDP2 train (replicated-LoRA, expert-body ckpt, GPU-resident base) -> $S/adapter (max-len $MAXLEN)"
if [ ! -e "$S/adapter/adapter.pt" ]; then gpu_free
  export PYTHONPATH="$PWD/src/quant:$PWD/src/eval:$PWD/src/distill"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CUDA_VISIBLE_DEVICES=0,1,2,3 $TRUN --nproc_per_node=4 --master_port=29611 src/distill/opd_train_fsdp.py \
    --model "$TEA" --rollouts "$S/rollouts.jsonl" --teacher-lp "$S/teacher_lp.npz" --adapter-out "$S/adapter" \
    --rank 16 --scale 2.0 --bits 3.0 --group 128 --epochs 2 --accum 8 --lr 5e-5 --max-len "$MAXLEN" \
    --fast 1 --cpu-offload 0 --ckpt-every 10 || { say "C FAILED"; exit 13; }
fi
say "C done: adapter saved"

say "D: merge LoRA + re-quantize -> $REC"
if [ ! -d "$REC" ]; then gpu_free
  $TPY src/distill/merge_adapter.py --model "$TEA" --adapter "$S/adapter" --out "$REC" \
    --rank 16 --scale 2.0 --bits 3.0 --group 128 --clip-steps 24 --max-mem-gib 72 || { say "D FAILED"; exit 14; }
fi
say "D done: recovered ckpt at $REC"

say "E1: baseline re-eval (matched flags) gpqa,math500"
gpu_free
bash pipelines/eval.sh --model "$STU" --tag q36_base_rematch --benchmarks gpqa,math500 || say "E1 eval returned nonzero"
say "E2: recovered eval gpqa,math500"
gpu_free
bash pipelines/eval.sh --model "$REC" --tag q36_recovered --benchmarks gpqa,math500 || say "E2 eval returned nonzero"

say "RECOVERY_PIPELINE_DONE"
