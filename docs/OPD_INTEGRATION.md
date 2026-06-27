# Combining OPD (on-policy distillation) with our protect-always-on PTQ framework

## Why (diagnosis, not assumption)
Truncation on BBEH/LCB is **genuine token-inefficiency**, not loops: rep-score ≈0.01, 0 loopy outputs; tails
show coherent 200+-step reasoning that overflows the budget. Sampling penalties do NOT help (BBEH acc 29→25→
4.2→37 across base/pp0.3/pp0.5+fp0.3/rep1.1; truncation never drops). 64k budget still truncated (BBEH 9/100,
LCB 20/80). ⇒ our 3b-quantized student needs FAR more reasoning tokens than the BF16 teacher / Gemma (both
trunc=0) for the same problems. Inference tricks can mask truncation (budget-forcing) but can't fix the
underlying inefficiency. **The fix must change the MODEL: teach it to reason as concisely as the teacher.**
That is exactly what OPD does.

## The integration: "QAD-OPD" — on-policy distillation as the recovery stage of our PTQ pipeline
Our pipeline today: BF16 122B → [protect-always-on + per-expert salience + MSE-clip PTQ] → 47GB student.
Add a 4th stage: → [offline on-policy distillation from the BF16 teacher] → recovered 47GB student.

### What trains (memory-feasible — the historical 1.5TB-QAT blocker is avoided)
- **Trainable = low-rank adapters on the quantized routed experts only** (QLoRA-for-recovery / LQER style),
  plus optionally the always-on path. Experts hold 94.6% of params but only 8/256 fire per token, so the
  active gradient/optimizer footprint is ~a 10B-active model — fits 4×H100.
- **STE** (already in moe_quant.ste_quant): forward = fakequant(W+BA, 3b); backward passes through to the
  adapter A,B. After training, merge BA into W and re-quantize at 3b → footprint stays 47GB.
- Backbone (shared expert/attn/DeltaNet/router/embed) frozen at its protected precision.

### Lightning-OPD (offline) to dodge the memory-bound teacher (arXiv 2604.13010)
Full online distillation needs teacher(245GB)+student(245GB BF16-for-grad) co-resident → infeasible. Lightning
OPD makes it offline + 4× cheaper:
1. **Teacher rollout cache (inference only, vLLM, fast):** run BF16 122B on reasoning prompts → store the
   teacher's CONCISE CoT + per-token top-k logits (k≈20). Teacher then leaves memory.
2. **Teacher-consistency:** same BF16 teacher seeds the SFT init and the OPD phase (the paper's key stability
   condition — mismatched teachers degrade).
3. **Student training (teacher absent):**
   - *Seed (off-policy SFT):* CE on the teacher's cached concise CoT tokens — directly teaches brevity.
   - *On-policy OPD:* student samples rollouts on the same prompts; score each student token against the
     cached teacher distribution; minimize **forward-KL(teacher‖student)** per token (dense reward, the OPD
     signal). On-policy (student's own tokens) avoids the exposure bias that pure SFT has, and is what lets a
     student match/exceed the teacher on reasoning (R1-distill, ExOPD 2602.12125).

### Prefix-OPD / OPSD as the no-cache fallback
If caching full teacher logits is too heavy: **OPSD** (self-distillation, no teacher) distills the student's
best-of-N / privileged-prefix (Prefix-OPD: feed a hint, distill the hinted answer into the un-hinted student).
Weaker than teacher-OPD but zero teacher cost — a cheap first probe.

### Data (target the weak axes)
BBEH-style algorithmic/multi-step, competitive code (LCB/CodeContests), GPQA-style science, plus general
R1/OpenThoughts CoT for breadth. Emphasize problems where teacher CoT is much shorter than student CoT
(the token-efficiency gap is the training signal).

## Expected outcome (lit-grounded)
- **Token efficiency ↑ → truncation → 0 naturally** (student inherits teacher's concise CoT). This is the
  goal's real fix, since inference tricks can't.
- **Reasoning/code recovery:** NVFP4-QAD (2601.20088) recovered GPQA 59→62.7 with 0.3–2.5B tokens; Lightning
  OPD hit 69.9% AIME-2024 on Qwen3-8B in ~30 GPU-hr. Our gaps (BBEH −21, LCB −22.5, GPQA −2) are the target.
- **Footprint unchanged (47GB)** — adapters merge + re-quantize; novelty stays ours (fused-MoE protect-always-
  on + salience + clip + **expert-adapter OPD recovery**), distinct from GPTQ/AWQ and from generic QAT.

## Minimal first experiment (cheapest decisive)
1. Cache BF16-teacher CoT+logits on ~2–5k BBEH-style + code prompts (vLLM, hours).
2. Train expert-LoRA via STE with seed-SFT + forward-KL OPD (~1e3 GPU-hr budget; start ~few hundred M tokens).
3. Re-quantize → re-run BBEH/LCB/GPQA with the SAME budget → measure (a) truncation rate and (b) accuracy vs
   the un-recovered student and vs Gemma. Success = trunc↓ toward 0 AND BBEH/LCB close the gap.

## Model-generality scope (multi-family OPD)
OPD/expert-LoRA is now **model-general across fused MoE families**, not Qwen-only. The recovery path is fully
duck-typed:
- `moe_lora._is_experts` detects a fused experts module by its 3D `gate_up_proj`/`down_proj` Parameters (no
  class import; excludes the shared-expert MLP whose proj are Linears). dims (E/H/I), `num_experts`, and
  `act_fn` are read from shapes/attrs, not Qwen-specific names.
- `opd_train_fsdp.wrap_fsdp_opd` wraps experts via `_is_experts` and decoder layers via the `*DecoderLayer`
  class-name suffix — no `Qwen3_5Moe*` imports.
- the model is loaded with the adapter's `auto_model_class` (VLM→ImageTextToText, flat→CausalLM, with fallback).

A family is OPD-capable iff its adapter sets `supports_opd=True`. Enabled for the **Group-A** fused families
whose experts share the reference forward (`gate,up = (x@gate_up_proj[e]).chunk(2); h = act_fn(gate)*up;
out = h@down_proj[e]`, routing computed in the parent and passed as `(hidden_states, top_k_index,
top_k_weights)`): `qwen3_5_moe` (Qwen3.5/3.6), `mixtral`, `qwen2_moe`, `qwen3_moe`, `olmoe`, `qwen3_next`,
`deepseek_v3` (eager experts impl). **Validated:** the OPD FSDP smoke (attach → replicated-LoRA → 2 steps →
nonzero grad → `SMOKE OK`) runs on a non-Qwen `qwen3_moe` (`Qwen3MoeExperts`/`Qwen3MoeDecoderLayer`); and a
per-family attach→forward→backward check produces nonzero LoRA grad on all six — `qwen2_moe`, `mixtral`,
`olmoe`, `deepseek_v3` (despite its dispatcher decorator), and `qwen3_next` (with `use_cache=False`, which the
trainer sets) — confirming each family's parent→experts call signature and expert math match the hook.

Families with a DIFFERENT expert math keep `supports_opd=False` and need a per-family expert-compute:
`gpt_oss` (transposed weights + gate/up bias + interleaved `::2`/`1::2` split + clamp±7 + custom GLU), and
the atypical `dbrx`/`granitemoe`. `moe_lora.attach_expert_lora` raises `NotImplementedError` when the active
adapter is not OPD-capable. Quantization (PTQ + eval) is universal regardless.

## Phase-C FSDP status at 35B (2026-06, smoke-tested on Qwen3.6-35B-A3B, 4×H100)
The 122B never reached a training step (it OOM'd before the blocker could even be characterized). At
35B the smoke gets far enough to isolate the failure mode precisely. Three configurations, in order:

1. **`--checkpoint 1` (gradient checkpointing ON, static expert loop):** `CheckpointError` — the
   dynamic per-expert MoE forward is rerouted on recompute, so the checkpoint's saved/recomputed tensor
   sets diverge. This is the originally-documented Phase-C blocker. **Confirmed at 35B.**
2. **`--checkpoint 0` + static loop:** OOM (the static loop materializes all 256 experts' activations).
3. **`--checkpoint 0` + dynamic loop (`static_loop=bool(args.checkpoint)`), `--max-len 1024`, short
   rollouts:** **clears both prior blockers** — no `CheckpointError`, no OOM. Forward + backward run and
   a real loss is produced (`step0 kl=20.87`). New, deeper failure surfaces:
   **`AssertionError: zero LoRA grad` (`grad(all-shard)=0.000e+00`).**

**Root cause of the zero-grad (the remaining blocker):** under FSDP2 the per-expert LoRA params
(`lora.Bgu`/`lora.Agu`, shape `[E, …]`) are **sharded** (DTensor). `_patched_forward` indexes them
per-expert (`lora.Bgu[e]`); FSDP all-gathers the params for the **forward**, so the loss computes, but
the gradient does not flow back to the sharded LoRA on the **backward** (grad is *exactly* 0, i.e. the
LoRA params are disconnected from the autograd graph after the per-expert index, not merely small). The
base experts survive only because FSDP2 gathers them for forward and they are frozen (no grad needed).

**RESOLVED (2026-06).** The LoRA params are now passed as FSDP `ignored_params` to every wrap that
encloses an experts module, so they stay **replicated (un-sharded)**. `lora.Bgu[e]` then indexes a plain
local full tensor and the gradient accumulates normally; DP correctness is restored by a manual
`all_reduce(AVG)` of the LoRA grads each step (the reduce-scatter FSDP would otherwise do). After wrap
the ignored params are moved to GPU (FSDP leaves them on their CPU origin) and they start bitwise-equal
across ranks (same `torch.manual_seed(0)` at attach). LoRA is ~0.9 GB at rank 16, so replication is cheap.

**Smoke result (Qwen3.6-35B-A3B, 4×H100, `--checkpoint 0 --max-len 1024 --smoke 2`):**
```
step0 kl=20.8763  grad(LoRA,avg)=1.646e+03   <- nonzero (was exactly 0)
step 0/2 kl=20.8763 ema=20.8763
step 1/2 kl=20.8732 ema=20.8760
SMOKE OK
```
All three Phase-C blockers are now cleared at 35B (`CheckpointError` + OOM via `--checkpoint 0` + dynamic
loop; zero-grad via replicated LoRA). OPD trains end-to-end — the 122B never reached a single step. The
full recovery run (gen→train→merge→re-quantize→re-eval to measure truncation/accuracy recovery) is now
runnable; only short-seq memory headroom (no-checkpoint backward holds the un-checkpointed backbone at
the chosen `--max-len`) bounds the sequence length per step.

## Target-set filter (user directive: only pursue benchmarks the TEACHER beats Gemma -> recoverable wins)
Keep ONLY where BF16 teacher > Gemma-4 AND student lags (quant-recoverable):
- **AIME**: teacher 90 > Gemma 86.7/89.2; student 73 → RECOVER (math CoT distillation).
- **GPQA-Diamond**: teacher 88 > Gemma 84.3; student 84 → RECOVER (hypothesis: math reasoning-policy transfers; add science CoT only if not).
DROP (teacher cannot beat Gemma → no recoverable win): BBEH (teacher 62 < 74.4), LiveCodeBench (Qwen official 78.9 < Gemma 80), MMMU/MMMU-Pro (teacher ≈ Gemma). Already-winning (not recovery): MMLU-Pro, MMMLU.
⇒ Stage-2 SFT uses MATH CoT only (1090 ex; code dropped). Eval = AIME + GPQA at trunc=0 vs Gemma.
Teacher CoT cache built: 1314 verified (1090 math kept/1500, 224 code kept — code now unused).

## Recovery results — Qwen3.6-35B-A3B (2026-06, matched 32k eval budget)

First end-to-end OPD recovery runs on the (now-unblocked) FSDP path. Two gen sources, both gen+train at
seq 2048, 2 epochs, rank-16 per-expert LoRA, reverse-KL.

| model | GPQA acc | finished-acc | trunc | MATH-500 acc | finished-acc | trunc |
|-------|---------:|-------------:|------:|-------------:|-------------:|------:|
| teacher (BF16)        | 84.3 | 88.8 | 11/198 | 90.0 | 90.8 |  1/120 |
| baseline (3b student) | 68.7 | 88.8 | 64/198 | 81.7 | 90.1 | 19/120 |
| on-policy OPD         | 61.1 | 90.5 | 72/198 | 56.7 | 93.1 | 48/120 |
| teacher-CoT OPD       | 66.2 | 92.5 | 65/198 | 55.8 | 90.1 | 49/120 |
| complete-traj OPD     | 77.3 | 88.1 | 30/198 | 89.2 | 89.9 |  1/120 |

**Finding: naive OPD lifts per-token reasoning but REGRESSES truncation — because the training rollouts
are themselves mostly truncated.** Both runs raised `finished-acc` (GPQA 88.8→90.5/92.5, the recovered
model is *more accurate when it concludes*) but raised truncation, so raw acc fell. Root cause: at the
1920-token gen budget, **~89% of rollouts were truncated for BOTH the student (686/768) AND the BF16
teacher-with-thinking (683/768)** — the gen budget, not the model, is the binding constraint. Distilling
trajectories that never emit `\boxed{}`/EOS teaches the student to keep generating → verbosity ↑ →
truncation ↑. teacher-CoT partially recovered GPQA truncation (72→65 ≈ baseline 64, highest finished-acc
92.5), confirming the mechanism: more-complete data → less verbosity reinforcement; it failed on MATH only
because math CoT is long and still truncated at 1920.

**Fix (the `--finished-only` lever): train on COMPLETE trajectories.** Generate at a long budget (8–16k;
now affordable — the memory fixes leave headroom to train at seq 4–8k) and filter to rollouts that
concluded (`finish_reason != "length"`, `opd_train_fsdp.py --finished-only 1`). This is the only way to
supply real conclusion/EOS signal. Run it via:
`GEN_MODEL=$TEA GENTOK=8192 MAXLEN=8192 FINISHED_ONLY=1 ... bash pipelines/q36_recovery.sh`.

**RESULT — the fix works.** Generating at 8192 (424/768 finished vs ~80 before) and training only on the
**313/209 concluded** trajectories (capped at seq 3584 for memory) recovered the gap: **MATH-500 81.7→89.2,
truncation 19→1 (teacher level); GPQA 68.7→77.3, truncation 64→30**. The two variants trained on
~89%-truncated rollouts regressed (61.1/56.7) — so the decisive lever is trajectory COMPLETENESS, not the
policy source or quant algorithm. The 3b student's truncation damage is recoverable with on-policy
distillation, provided you distill conclusions rather than cutoffs.
