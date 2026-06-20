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

## Model-generality scope (multi-MoE refactor)
OPD/expert-LoRA is **fused-expert only** (Qwen3.5-122B-A10B and Qwen3.6-35B-A3B, both `model_type:
qwen3_5_moe`). The whole recovery path is built around the fused 3D expert layout — `_patched_forward`
indexes `gate_up_proj[e]` / `down_proj[e]` and the FSDP wrap keys on `Qwen3_5MoeExperts` /
`Qwen3_5MoeDecoderLayer`. Other MoE families are a **non-goal** for OPD (even fused ones like OLMoE, whose
expert module/forward differ): `moe_lora.attach_expert_lora` raises `NotImplementedError` unless the active
adapter sets `supports_opd` — only `Qwen35MoeAdapter` does (see model_adapters.py). The fused PTQ + eval
path is fully model-general; only this recovery stage is Qwen-specific. (OPD Phase C itself is also a separately-documented FSDP blocker —
see the KNOWN BLOCKER in opd_train_fsdp.py — so no new distill runs are gated on this refactor.)

## Target-set filter (user directive: only pursue benchmarks the TEACHER beats Gemma -> recoverable wins)
Keep ONLY where BF16 teacher > Gemma-4 AND student lags (quant-recoverable):
- **AIME**: teacher 90 > Gemma 86.7/89.2; student 73 → RECOVER (math CoT distillation).
- **GPQA-Diamond**: teacher 88 > Gemma 84.3; student 84 → RECOVER (hypothesis: math reasoning-policy transfers; add science CoT only if not).
DROP (teacher cannot beat Gemma → no recoverable win): BBEH (teacher 62 < 74.4), LiveCodeBench (Qwen official 78.9 < Gemma 80), MMMU/MMMU-Pro (teacher ≈ Gemma). Already-winning (not recovery): MMLU-Pro, MMMLU.
⇒ Stage-2 SFT uses MATH CoT only (1090 ex; code dropped). Eval = AIME + GPQA at trunc=0 vs Gemma.
Teacher CoT cache built: 1314 verified (1090 math kept/1500, 224 code kept — code now unused).
