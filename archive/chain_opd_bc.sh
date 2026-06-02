#!/bin/bash
# OPD round B->C ONLY (reuses existing valid rollouts.jsonl from Phase A; Phase A succeeded, ~1h, don't redo).
# score(vLLM teacher, OOM-hardened) -> train(HF device_map reverse-KL). Guards: each phase checks its output.
source ./vllm_venv/bin/activate
export HF_HOME=~/.cache/huggingface HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_TRUST_REMOTE_CODE=1 VLLM_LOGGING_LEVEL=WARNING
export TMPDIR=/tmp/moe-lowbit/tmp TRITON_CACHE_DIR=/tmp/moe-lowbit/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/moe-lowbit/cache/inductor VLLM_CACHE_ROOT=/tmp/moe-lowbit/cache/vllm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ./qwen35_qad
L=./logs; S=results/stage2
TEA=Qwen/Qwen3.5-122B-A10B
OUT=models/q122_opd_r1
rm -f $L/opd_done.txt $L/opd_failed.txt

if [ ! -s $S/rollouts.jsonl ]; then echo "PRECOND_FAILED: no rollouts.jsonl" | tee $L/opd_failed.txt; exit 1; fi

echo "=== PHASE B: teacher scoring (OOM-hardened) $(date +%H:%M) ===" > $L/opd_B.log
python opd_score.py --teacher $TEA --rollouts $S/rollouts.jsonl --out $S/teacher_lp.npz --k 20 --max-len 18432 >> $L/opd_B.log 2>&1
if [ ! -s $S/teacher_lp.npz ]; then echo "PHASE_B_FAILED: no teacher_lp.npz" | tee $L/opd_failed.txt >> $L/opd_B.log; exit 1; fi
echo "B_DONE" >> $L/opd_B.log

echo "=== PHASE C: reverse-KL LoRA train (device_map) $(date +%H:%M) ===" > $L/opd_C.log
python opd_train.py --model $TEA --rollouts $S/rollouts.jsonl --teacher-lp $S/teacher_lp.npz \
  --adapter-out $S/adapter_r1 --out $OUT \
  --rank 16 --scale 2.0 --bits 3.0 --epochs 1 --accum 8 --lr 5e-5 --max-len 8192 --reverse 1 --max-mem-gib 70 >> $L/opd_C.log 2>&1
if [ ! -e $OUT/model.safetensors.index.json ] && [ ! -e $OUT/config.json ]; then
  echo "PHASE_C_FAILED: no model at $OUT" | tee $L/opd_failed.txt >> $L/opd_C.log; exit 1; fi
echo "C_DONE" >> $L/opd_C.log
echo "OPD_R1_DONE" > $L/opd_done.txt
