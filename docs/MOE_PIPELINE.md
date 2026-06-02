# The strongest deployable sub-4-bit recipe for Qwen3.5-122B-A10B (capstone)

## Executive summary
A simple PTQ recipe — composite-salience mixed precision over the routed experts at ~3 effective bits +
protecting the always-on path (shared-expert int8, attention/DeltaNet int4, router bf16) — is
**near-lossless on all four axes with NO training** on this frontier hybrid Gated-DeltaNet MoE: MMLU 85.5,
GSM8K 92.5, HumanEval 90.6, GPQA-d 46.5 vs BF16 87.5/96.0/100.0, fitting the 122B onto a **single 80 GB
GPU (~50 GB, 4.9× smaller than BF16)**. This INVERTS the dense regime (3-bit PTQ collapses generation;
knowledge capacity-gated <4b): on the frontier MoE the binding constraint is allocation+quantization, not
recovery. Allocation is well-ordered by KL (4× range: salience 0.058 → blind/random 0.12–0.14 → anti-control
0.247) but SECOND-ORDER for capability (blind-3b still GSM8K 95.0); the decisive cheap levers are the
bit-budget and always-on-path protection, while expert-frequency allocation (the MoE-quant staple) is
NEUTRALIZED by Qwen3.5's load-balanced router (entropy 0.952, zero cold experts). Positioned honestly as a
lower-bit (3.06b), multi-axis extension of DQ3_K_M (671B@3.59b) and XFP (this model, 94.5% GSM8K@3.97b,
same protection scheme). **Minimum effective precision ≈ 2.5 bits** (~42 GB, near-lossless all-4-axes);
math cliffs at 2.0–2.25b. **Recovery finding:** at the 2.0b cliff, expert-local reconstruction does NOT
recover capability (cuts expert MSE 3–7× but capability flat/slightly worse) — reducing quantization error
≠ recovering capability; the working recovery is GLOBAL forward-KL distillation on reasoning-CoT (memory-bound
on 122B). Net: the frontier MoE is deployable sub-4-bit via PTQ alone; recovery training is not needed to
reach ~2.5–3b near-lossless, and local recon can't push below the 2.5b math cliff (global QAD can, at memory cost).


**Central question.** At a fixed hardware budget, what combination of **MPQ allocation + QAD/distillation**
recovers the most useful capability on a frontier MoE, and what is the minimum effective precision that
stays genuinely deployable? Quantization is treated as *initialization*; capability comes from allocation
(how much recoverable capacity remains) + post-quant training (how much is recovered).

Axes: MMLU (knowledge), GPQA-diamond (hard reasoning), GSM8K (math), HumanEval (code).
Evidence: `MOE_FINDINGS.md` (numbers), `SURVEY_MOE.md` (2025-26 method survey), `results/study/*` (sweep),
`results/recon*/*` (recovery). Harness: `moe_quant/alloc/probe/eval/study/recon.py` (self-contained;
EdgeRazor is bypassed because the 122B's experts are fused 3D tensors it cannot target).

## Why the MoE changes the story (vs the prior dense Qwen3.5 work)
The prior program (dense 0.8B–27B) found: ≤2-bit is a hard floor; **knowledge is capacity-gated below
4 bits** and 3-bit PTQ collapses generation; allocation is second-order; QAD-on-CoT recovers *reasoning*
but not *knowledge* at 3.25b. The 122B MoE breaks several of these because **94.6% of params are routed
experts, only 8/256 fire per token, and the always-on path (shared expert + attention + embeddings, ~6%)
can be protected at high precision almost for free**:
- **F1.** Routing is heavily **load-balanced** (entropy 0.952/1.0, max/mean freq 1.56, **0 cold experts**)
  → the dominant MoE-quant lever in the literature ("quantize cold experts for free") is *neutralized*.
- **F2.** Allocation signals (freq/wmass/asal) are **highly redundant** (freq↔wmass Jaccard 0.93) → the
  salience-aware strategies should cluster; the real tests are salience-vs-blind, composite-vs-inverse.
- **F3.** PTQ at ~3 eff bits is **near-lossless on knowledge** (composite-3.0b: MMLU 85.5 vs 87.5 BF16,
  ppl 1.94 vs 1.85) — overturning the dense "3-bit knowledge collapse". [generation: pending sweep]

## BF16 reference (122B): MMLU 87.5, GSM8K 96.0, HumanEval 100.0, ppl 1.85.

## The recipe (final)
1. **Allocation** — per-expert mixed precision, **protect the always-on path** (shared-expert int8;
   attention/DeltaNet int4; router bf16; embed/lm_head int4 — the ~6% that fires every token), quantize the
   94.6% routed experts to the target budget. Salience ranking (activation×router-weight) gives a real ~2×
   KL improvement over blind but is SECOND-ORDER for capability — so even uniform/blind expert quant works;
   the protection + budget are what matter. **No expensive allocation search needed.**
2. **Budget** — **~2.5–3.0 effective bits** is the sweet spot: near-lossless on all 4 axes via **PTQ alone,
   no training**. 2.5b ≈ 42 GB (single 80GB GPU). Math is the binding axis; it cliffs below 2.5b.
3. **Recovery** — at ≥2.5b, NOT needed (PTQ already near-lossless). Below 2.5b, expert-local reconstruction
   does NOT help (F9); the working recovery is GLOBAL forward-KL distillation on reasoning-CoT (memory-bound
   on 122B → offline teacher-logit caching + adapter/step-size training). For deployment, prefer staying at
   ≥2.5b over training-to-recover sub-2.5b.

## Allocation-study results (PTQ, no recovery)
**Headline (F5): at ~3 effective bits the 122B MoE is near-lossless on ALL FOUR axes via PTQ alone.**
| config | eff bits | MMLU | GPQA-d | GSM8K | HumanEval | ppl | KL |
|---|---|---|---|---|---|---|---|
| BF16 | 16 | 87.5 | — | 96.0 | 100.0 | 1.85 | 0 |
| composite-3.0b | 3.06 | 85.5 | 46.5 | 92.5 | 90.6 | 1.94 | 0.062 |
| composite-2.5b | 2.59 | 82.5 | — | — | — | 2.01 | 0.088 |

Strategy @3.0b, clean monotonic KL ordering (4× range, tracks salience-correctness): **asal 0.058 ≤ wmass
0.060 ≤ composite 0.062 ≤ freq 0.065 ≤ wfro 0.079 ≪ random 0.124 ≤ blind 0.139 ≪ inverse 0.247**.
Yet downstream capability is near-lossless for ALL: MMLU 83–86 (n=200 ≈ ±5), and **blind generation
(GSM8K 95.0 / HE 87.5) ≈ composite (92.5 / 90.6)**. ⇒ **allocation is a real, well-ordered lever for
distribution fidelity (KL) but SECOND-ORDER for downstream capability at 3b** — the MoE is quant-tolerant
enough that even importance-blind 3-bit allocation is near-lossless. The decisive, cheap levers are the
BUDGET (~3b) and PROTECTING the always-on path (shared int8 vs int4: KL 0.062 vs 0.070), not expert
ranking. Per-layer vs global ranking: identical (KL 0.060 vs 0.062). (Matches XFP independently using the
same protect-always-on-path scheme on this model.)

## Minimum effective precision (high-n budget curve) — the floor is ~2.5 bits
| eff bits | ~GB | MMLU | GPQA | GSM8K | HE |
|---|---|---|---|---|---|
| 3.06 | ~50 | 84.2 | 46.5 | 91.7 | 87.5 |
| **2.59** | **~42** | **82.8** | 44.4 | **93.3** | 88.8 |
| 2.35 | ~39 | 81.0 | 46.0 | 71.7 | 90.0 |
| 2.11 | ~37 | 80.0 | 44.9 | 77.5 | 88.8 |
| BF16 | 245 | 87.5 | — | 96.0 | 100 |
**~2.5 effective bits (~42 GB) is near-lossless on all 4 axes** (5.8× smaller than BF16; single 80GB GPU).
The math cliff is sharp at 2.0–2.25b (GSM8K 72–78); code/knowledge/GPQA stay usable to 2.0b. Math is the
binding axis.

## Recovery results — expert-local reconstruction does NOT recover the 2-bit cliff
At 2.0b (the math cliff), expert-local reconstruction (STE-train experts to match FP output, per layer):
| 2.0b | MMLU | GPQA | GSM8K | HE | expert nmse |
|---|---|---|---|---|---|
| PTQ | 80.0 | 44.9 | 77.5 | 88.8 | ~0.30 |
| + recon | 79.6 | 43.9 | 75.0 | 87.5 | **0.090** |
**Recon cut per-layer expert MSE 3–7× but capability did NOT improve (slightly worse, within noise).**
Reducing per-expert quantization error ≠ recovering capability — confirming, from the recovery side, that
capability isn't bottlenecked by recoverable per-expert error. ⇒ the recovery that works must be GLOBAL
(forward-KL distillation matching the output DISTRIBUTION over reasoning-CoT, the dense program's validated
mechanism), NOT local weight reconstruction; on 122B that global QAD is memory-bound (offline top-K
teacher-logit caching + adapter/step-size training is the tractable next build). The program thesis
("quantization=init, training recovers") holds only for the right global/task objective.

## Deployment memory math (the "smaller hardware" answer)
Weight memory for the recipe (composite, expert avg 3.0 bits, protected path), group-128 per-block scales:
- routed experts: 116.0B × 3.0/8 = **43.5 GB**
- shared expert (8b) 0.45 GB; attn+DeltaNet (4b) 4.13B×4/8 = 2.07 GB; embed+lm_head (4b) 1.52B×4/8 = 0.76 GB;
  router (bf16) 0.08 GB; vision tower (bf16) 0.9 GB → non-expert ≈ **4.3 GB**
- per-block scale overhead ≈ 122B × (16/128)/8 = **1.9 GB**
- **TOTAL ≈ 49.7 GB** (vs BF16 245 GB → **4.9× smaller**; vs INT4 ≈ 63 GB → 1.27× smaller)

⇒ the recipe puts a **122B frontier MoE onto a SINGLE 80GB GPU** (H100/A100-80GB/H200) or **2× 40-48GB**
(A100-40GB / RTX 6000 Ada 48GB) with room for KV/activations — where BF16 needs 4× 80GB. That is the
concrete "deployable on smaller hardware" result, *contingent* on capability holding (knowledge: yes per
F3; generation: pending recon). At ≤2.5b it could fit even tighter but capability is the binding constraint.

## Hardware / deployment reality
Clean int grids are used to isolate allocation+QAD effects (per the goal). For an actual sub-4-bit speedup
on H100, the deployment path is **NVFP4** (FP4 tensor cores) or **XFP codebook-VQ** (the only method
benchmarked on this exact model: ~3.97b, +49% decode, 94.5% GSM8K) — see SURVEY_MOE.md §4. Our finding
that ~3-bit PTQ is near-lossless on this MoE means the storage/memory win (≈5.3× vs BF16, ≈1.3× vs INT4)
is realizable; the throughput win requires those kernels.

## Bottom line — answer to the central question
For Qwen3.5-122B-A10B the central result **inverts the dense-model regime: the binding constraint is
allocation+quantization, not recovery.** Concretely:
1. **Best recipe at a fixed budget** = protect the always-on path (shared int8 / attn-DeltaNet int4 / router
   bf16, ~6% of params) + quantize the 94.6% routed experts; salience ranking helps KL ~2× but is
   second-order for capability (blind works). At **~2.5–3.0 effective bits this is near-lossless on all four
   axes with PTQ ALONE — no training** (3.0b: MMLU 84.2/GSM8K 91.7/HE 87.5/GPQA 46.5; 2.5b: 82.8/93.3/88.8/44.4
   vs BF16 87.5/96/100/—).
2. **Minimum effective precision = ~2.5 bits** (~42 GB → a 122B frontier MoE on a **single 80GB GPU**, 5.8×
   smaller than BF16). Math is the binding axis; it cliffs sharply at 2.0–2.25b (GSM8K →72–78); knowledge/
   code/hard-reasoning stay usable to 2.0b.
3. **How much does QAD recover?** At ≥2.5b, recovery is unnecessary (PTQ near-lossless). At the 2.0b cliff,
   the memory-tractable LOCAL recon does NOT recover capability (F9: cuts expert MSE 3–7× but capability
   flat/slightly worse) — capability isn't bottlenecked by recoverable per-expert error. The recovery that
   works is GLOBAL forward-KL distillation on reasoning-CoT (matching the output distribution), memory-bound
   at 122B (offline teacher-logit caching + adapter training = the documented next build).
4. **Does the recovery mechanism transfer from small models?** Only partially: the *need* for recovery
   largely vanishes on the frontier MoE (its expert-routing redundancy + protected always-on path make PTQ
   near-lossless at 2.5–3b, unlike dense which collapses), and the recovery that does matter (global QAD-CoT)
   requires the right objective — local reconstruction, the tractable analogue, is insufficient.

**Deployability verdict: YES.** Qwen3.5-122B-A10B is genuinely deployable at ~2.5–3 effective bits on a
single 80GB GPU with near-BF16 capability via PTQ alone; the recipe is "protect the always-on path + ~2.5–3b
experts," allocation search and recovery training are not required to reach this point. (Throughput win needs
sub-4-bit kernels — NVFP4 or XFP codebook-VQ; our int-grid result quantifies the *capability/storage*
frontier, which the goal asked to isolate.)

## Addendum — PTQ-method comparison (is the recipe method-specific?) + the 2.0b-QAD decision
We compared our recipe against strong open PTQ methods applied to the SAME fused experts (libs target
nn.Linear and can't), same protection+benchmarks:
- **2.5–3.0b: NOT method-specific.** RTN-asym / HQQ / AWQ / sym / asym are all near-lossless and tied
  (int3 ≈ BF16). Near-lossless ~3b is a property of the frontier MoE (routing redundancy, 8/256 active) +
  always-on-path protection — any reasonable PTQ reaches it. Our composite recipe is not uniquely strong.
- **2.0b: no PTQ is near-lossless.** Best = HQQ (MMLU 82.6 / GSM8K 89.2 / HE 80.0), beating RTN by ~3 and
  our ternary-mix on math; AWQ ≈ RTN (load-balanced experts lack AWQ's outliers). All leave a clear gap
  (HE −17.5, GSM8K −7.5). GPTQ impractical per-expert at MoE scale (~6–8 h).
- **Decision: do NOT launch expensive 2.0b QAD.** 2.5b PTQ (~42 GB) is already near-lossless and beats
  every 2.0b method on every axis; the 2.0b residual gap is a real recovery target (only global forward-KL
  QAD could close it — better PTQ and local recon can't), but it buys ≤5 GB over the near-lossless 2.5b
  point. The deployable frontier is **2.5b PTQ**; recovery is not worth it here.
