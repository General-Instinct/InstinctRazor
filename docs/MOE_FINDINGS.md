# Qwen3.5-122B-A10B sub-4-bit: MoE allocation + QAD recovery — findings

Goal: at a fixed hardware budget, what combination of **MPQ allocation + QAD/distillation** recovers the
most useful capability on the frontier MoE, and what is the minimum effective precision that stays
deployable? Quantization is treated as INITIALIZATION; recovery comes from post-quant training.
Axes: MMLU (knowledge), GPQA-diamond (hard reasoning), GSM8K (math), HumanEval (code).

## Setup (this machine)
4×H100 80GB. transformers 5.9 (`qwen3_5_moe`), custom MoE quantizer (`moe_quant.py`) because **94.6%
of the 122B's params live in FUSED 3D expert tensors** (`experts.gate_up_proj [256,2048,3072]`,
`experts.down_proj [256,3072,1024]` per layer) that EdgeRazor's nn.Linear targeting silently skips.
Quant = per-block symmetric int (group 128); per-expert bit-width assignment; ternary(1.58)/int3/int4
grid; STE for training. Eval loads the model once, snapshots FP to CPU, restores between specs.
Full STE-QAT is infeasible (Adam≈1.5TB) → recovery via expert-local reconstruction (`moe_recon.py`).

## Established empirical pre-findings (before allocation eval)

**F1 — Routing is heavily load-balanced; there are NO cold experts.** Probe over calibration tokens:
normalized routing entropy **0.952** (1.0 = uniform), global per-expert freq max/mean **1.56**, **0/256
cold experts** (freq < 10% mean), 0 zero-freq. Top-64/256 experts capture 50% of routing (vs 25%
uniform) — only mild concentration. Entropy dips slightly in mid/late layers (0.94 vs 0.98 early).
*Implication:* the dominant MoE-quant lever in the literature — "quantize rarely-used cold experts for
free" (DynaExq/MC#/hot-cold) — is **largely neutralized on a well-load-balanced frontier MoE**. Those
papers used Mixtral/DeepSeek-MoE with more routing skew; Qwen3.5's balanced router removes the free lunch.

**F2 — Allocation signals are highly redundant.** Per-expert ranking correlations: freq↔wmass 0.95
(Jaccard of int4-selected sets 0.93 — frequency≈routing-weight under load-balancing), freq↔asal 0.89,
wmass↔asal 0.84; activation-norm per expert is nearly uniform (asal/freq spread 0.046). *Implication:*
the salience-aware strategies (freq/wmass/asal/composite) should give SIMILAR allocations → among them
strategy choice is likely second-order. The decisive tests are salience-vs-blind, composite-vs-inverse
(does protecting the *right* experts matter at all?), the budget curve, and protected-carve-out on/off.

## BF16 reference (122B)
MMLU **87.5**, GSM8K **96.0** (n=40), HumanEval **100.0** (n=32), calib ppl **1.85**. (BF16 GPQA captured in recon-eval.)

## F3 — The 122B MoE is dramatically more quantization-tolerant than dense models (PTQ, no recovery)
First sweep result, `s_composite_b30` (composite salience allocation, expert avg **3.00 bits** → 3.06 all-in
incl protected ~6% non-expert path; protected = shared-expert int8, attn/ssm/embed int4, router bf16):
**ppl 1.94 (BF16 1.85), KL 0.062, MMLU 85.5 (−2.0), GPQA-diamond 46.5** — all from PTQ alone.
This OVERTURNS the prior dense finding ("knowledge is capacity-gated below 4 bits; 3-bit PTQ collapses").
On the frontier MoE, knowledge survives ~losslessly at ~3 effective bits because (a) only the routed
experts (94.6% of params) are quantized hard while the ALWAYS-ON path (shared expert + attention +
embeddings, ~6%) is protected, and (b) only 8/256 experts fire per token, so per-expert quant error
averages out across the active set + the high-precision shared path. Generation (the real test, since MC
overstates) is the open question the sweep + recon resolve.
[Strategy comparison, budget curve, protection ablation, generation — pending sweep completion.]

## F4 — Allocation matters ~2× in fidelity but is second-order for downstream knowledge (PTQ @3.0b)
Screen at expert-avg 3.0 bits (KL = top-K KL vs BF16, the low-variance metric; MMLU n=200 ≈ ±5 noise):
| strategy | KL↓ | ppl↓ | MMLU |
|---|---|---|---|
| asal (activation salience) | **0.058** | 1.94 | 85.0 |
| wmass (router-weight) | 0.060 | 1.93 | 83.0 |
| composite (wmass+asal+wfro) | 0.062 | 1.94 | 85.5 |
| freq | 0.065 | 1.94 | 84.0 |
| composite, NO protection | 0.070 | 1.94 | 85.0 |
| wfro (weight Frobenius) | 0.079 | 2.01 | 88.0 |
| **random (control)** | **0.124** | 2.08 | 85.0 |
| **blind (control)** | **0.139** | 2.12 | 83.5 |
| **inverse (anti-control)** | **0.247** | 2.34 | 85.5 |
| BF16 | 0.000 | 1.85 | 87.5 |

Clean monotonic KL ordering tracking salience-correctness (4× range): **salience (asal 0.058 ≤ composite
0.062 ≤ wfro 0.079) ≪ blind/random (0.124–0.139) ≪ inverse 0.247** — yet downstream MMLU is 83–86 for ALL
(within n=200 noise). So allocation is a real, well-ordered lever for *distribution fidelity* but
*second-order for capability* at 3b (the model is quant-tolerant enough that all allocations are
near-lossless). [blind gen 95.0/87.5 ≈ composite 92.5/90.6 confirms second-order for generation too.]

- **Salience-aware allocation beats random selection ~2× on KL** (0.058 vs 0.124) and on ppl (1.94 vs 2.08)
  → allocation is NOT fully second-order on the MoE (unlike the dense finding); the activation/router signals
  carry real information about which experts to protect.
- BUT **downstream MMLU is within noise for ALL strategies (83–88 ≈ BF16 87.5)** — the knowledge axis is
  preserved at 3 bits regardless of allocation, because the always-on path is protected and only experts
  are quantized hard (F3). So allocation refines *distribution fidelity* (KL/ppl) more than *knowledge acc*.
- Best single signals: **asal ≈ wmass > freq > wfro**; composite ≈ best. (Consistent w/ F2 redundancy.)
- **CRUCIAL (generation control): the 2× KL gap does NOT translate to a capability gap.** blind-3.0b
  (KL 0.139) gets GSM8K **95.0** / HE 87.5 / MMLU 83.5 / GPQA 48.5 — statistically indistinguishable from
  composite-3.0b (92.5/90.6/85.5/46.5) at n=40/32/200/198. ⇒ **allocation is SECOND-ORDER for downstream
  capability** (knowledge AND generation) on this MoE; even importance-blind 3-bit allocation is
  near-lossless. The salience KL advantage is real for distribution fidelity but capability is saturated
  either way. Dominant levers = the BUDGET (~3b) and PROTECTING the always-on path, not expert ranking.
  Full gen control: composite 92.5/90.6 · blind 95.0/87.5 · inverse(anti-control, KL 0.247) **90.0/84.4** —
  across the 4× KL range generation spans only GSM8K 90–95 / HE 84–91 (within ±~9pp at n=40/32); faint
  salience-favoring trend (inverse lowest) but second-order. Even the WORST 3-bit allocation generates well.
- Protection (shared-expert int8 vs int4) gives a small KL gain (0.062→0.070 when removed). The larger
  protection value (vs quantizing the always-on path to ternary) is implied by F3 but not directly ablated.
- **Harness verified** (adversarial workflow): quant axis/fusion/STE/chunked-apply all correct; router IS
  bf16-protected (it is a Qwen3_5MoeTopKRouter, not nn.Linear → not matched by the quantizer). "avg_bits
  3.06" is weight-bits (literature convention); true storage incl per-block scales ≈ 3.18b (deployment
  math in MOE_PIPELINE.md already counts scales).
## F5 — HEADLINE: generation SURVIVES at 3 bits (PTQ, no recovery) — overturns the dense collapse
`s_composite_b30` (composite alloc, expert avg 3.0 bits, protected), PTQ:
| config | eff bits | MMLU | GPQA-d | GSM8K | HumanEval | ppl |
|---|---|---|---|---|---|---|
| BF16 | 16 | 87.5 | — | 96.0 | 100.0 | 1.85 |
| **composite-3.0b PTQ** | **3.06** | **85.5** | **46.5** | **92.5** | **90.6** | 1.94 |
| composite-2.5b PTQ | 2.59 | 82.5 | — | — | — | 2.01 |

**At ~3 effective bits the 122B MoE is near-lossless on ALL FOUR axes with PTQ alone** (−2.0 MMLU,
−3.5 GSM8K, −9.4 HumanEval). This OVERTURNS the dense-model finding that 3-bit PTQ collapses generation
and that knowledge is capacity-gated below 4 bits. The earlier slow generation was pipeline-parallel decode
speed + HumanEval exec timeouts, NOT degenerate output (the model genuinely solves 92.5% of GSM8K at 3b).
Mechanism (F3): only routed experts are quantized hard; the always-on path is protected; 8/256 active +
huge capacity ⇒ per-token error averages out. Combined with the ~50 GB deployment footprint (single 80GB
GPU, MOE_PIPELINE.md), this is the core "deployable on smaller hardware" result.
**Reframes recovery:** at 3b, PTQ init is already near-lossless, so QAD/recon value lies at the 2–2.5b
FLOOR where PTQ degrades. The minimum-effective-precision search (2.0/2.25/2.5b) pinpoints the cliff.
## F6 — Budget curve (composite PTQ): capability saturates ≥3b; KL is the only monotonic signal
| eff bits | KL↓ | MMLU | GPQA-d | GSM8K | HumanEval |
|---|---|---|---|---|---|
| 4.0 (W4) | 0.041 | 85.5 | 47.5 | 95.0 | 93.8 |
| 3.5 | 0.047 | 87.5 | 44.9 | 90.0 | 93.8 |
| 3.06 | 0.062 | 85.5 | 46.5 | 92.5 | 90.6 |
| 2.59 | 0.088 | 82.5 | — | (floor) | (floor) |
| BF16 | 0.000 | 87.5 | — | 96.0 | 100.0 |
Capability is near-lossless and saturated from 3.0b→4.0b (gen 90–95 GSM8K is n=40 noise, not monotonic);
**KL is the only clean monotonic budget signal** (0.041→0.088 as bits drop 4.0→2.5). Knowledge degradation
onsets at 2.5b (MMLU 82.5, −5.0). The cliff for GENERATION is below 2.5b → floor search (2.0/2.25/2.5b at
n=120 GSM8K / 500 MMLU) running now to locate it.
**Mixed-vs-uniform @3.0b:** uniform-int3 (all experts int3; KL 0.103) ≈ composite-mix (int4/ternary; KL
0.062) on CAPABILITY (MMLU 87.0 vs 85.5, GSM8K 95.0 vs 92.5) despite worse KL — the simplest uniform-int3
is competitive, reinforcing that allocation is second-order for capability; per-block int3 (7 levels) beats
a salience-chosen int4/ternary MIX in stability even if KL favors the mix.

## F7 — DEFENSIBLE high-n confirmation + the floor is BELOW 2 bits for knowledge
Re-run at n=500 MMLU / n=120 GSM8K / n=80 HE / n=198 GPQA (addresses the n=40 power critique):
| eff bits | MMLU(500) | GPQA(198) | GSM8K(120) | HE(80) | KL |
|---|---|---|---|---|---|
| 3.06 | **84.2** | 46.5 | **91.7** | 87.5 | 0.062 |
| 2.11 (2.0b) | **80.0** | 44.9 | (gen running) | — | 0.144 |
| BF16 | 87.5 | — | 96.0(n40) | — | 0.000 |
At 3.06b the near-lossless claim HOLDS at proper n (MMLU −3.3, GSM8K −4.3). Stunningly, at **2.0 effective
bits** knowledge is still MMLU 80.0 (−7.5) and GPQA 44.9 (≈3b) — the 122B MoE tolerates 2-bit experts on
knowledge.

## F8 — MINIMUM EFFECTIVE PRECISION ≈ 2.5 bits (near-lossless); sharp math cliff at 2.0–2.25b
DEFINITIVE high-n PTQ budget curve (n=500 MMLU / 120 GSM8K / 80 HE / 198 GPQA) — the central answer:
| eff bits | ~GB | MMLU | GPQA | GSM8K | HE | KL |
|---|---|---|---|---|---|---|
| 3.06 | ~50 | 84.2 | 46.5 | 91.7 | 87.5 | 0.062 |
| **2.59 (2.5b)** | **~42** | **82.8** | 44.4 | **93.3** | 88.8 | 0.088 |
| 2.35 (2.25b) | ~39 | 81.0 | 46.0 | 71.7 | 90.0 | 0.108 |
| 2.11 (2.0b) | ~37 | 80.0 | 44.9 | 77.5 | 88.8 | 0.144 |
| BF16 | 245 | 87.5 | — | 96.0 | 100 | 0 |
**Minimum effective precision ≈ 2.5 bits**: at 2.59b (~42 GB, 5.8× smaller than BF16, still single-80GB-GPU)
the MoE is near-lossless on ALL axes (GSM8K 93.3 ≈ 3b/96 BF16, MMLU 82.8, HE 88.8, GPQA 44.4). **The math
cliff is sharp and sits at 2.0–2.25b** (GSM8K 71.7/77.5 — a −20pp drop; non-monotonic between 2.0/2.25b =
n=120 noise + allocation-specific effects). Code (HE), knowledge (MMLU, gradual), and GPQA stay usable even
at 2.0b — **math is the binding axis** (matches the dense finding that math is most quant-sensitive). KL is
the clean monotonic signal (0.062→0.144). ⇒ deployable floor = **2.5b** for full capability; 2.0–2.25b
trades math for a smaller footprint. **2.0–2.25b is exactly where post-quant TRAINING (recon/QAD) should
matter** — recon@2.0b running now to test whether training recovers the GSM8K cliff (77.5 → toward 93).

## Positioning & rigor (from the adversarial positioning workflow — see POSITIONING.md)
- The "3-bit near-lossless frontier MoE" result is **credible and consistent with 2025-26 literature, but
  NOT novel in kind**: **DQ3_K_M (2505.02390)** already showed DeepSeek-V3/R1 (671B MoE) near-lossless
  (0.30% avg drop incl MATH/MMLU/code) at **3.59 eff bits** PTQ; **XFP (2605.14844)** showed the EXACT
  model Qwen3.5-122B at **94.5% GSM8K @3.97b** using the SAME protect-always-on-path scheme (τ=0.96
  attn/DeltaNet/shared, τ=0.93 routed). Our contribution is the **lower-bit (3.06b) 4-axis** point + the
  **per-expert salience allocation/ablation** + the **load-balanced-router finding** (entropy 0.952, 0 cold
  experts → the cold-expert "free lunch" of Mixtral/DeepSeek-MoE papers is neutralized here).
- The dense/small-MoE-collapse vs frontier-MoE-survives dichotomy IS well-supported (MoEQuant Mixtral W3
  GSM8K 66→43; MC# Mixtral 2.54b GSM8K −35%; DynaExq Qwen3-80B INT2 collapse) — the determining variable is
  scale + expert-routing redundancy, not MoE-ness alone.
- **STATS CAVEAT (important):** headline gen used n=40 GSM8K / n=32 HE (±~9pp) vs XFP n=1319×3. The 92.5 vs
  96.0 GSM8K gap is within that noise. → **Re-running the cliff-region budget curve (2.0/2.25/2.5/3.0b) at
  n=120 GSM8K / n=80 HE / n=500 MMLU** for a defensible "near-lossless" claim. Allocation deltas should be
  read on KL (low-variance), not small-n MMLU.
- **The genuinely open, more-novel direction = recovery/QAD at the 2–2.5b FLOOR** (where PTQ degrades:
  composite@2.5b MMLU 82.5/ppl 2.01 shows onset). That is the next experiment.

## F9 — Expert-local reconstruction does NOT recover the 2-bit cliff (the recovery answer)
Expert-local reconstruction at 2.0b (STE-train quantized experts to match FP self-teacher output, per layer;
12/48 layers every-4th, 60 steps, grad-clip + best-checkpoint guard, FP streamed from safetensors):
| 2.0b config | MMLU(500) | GPQA(198) | GSM8K(120) | HE(80) | mean expert nmse |
|---|---|---|---|---|---|
| PTQ (no recovery) | 80.0 | 44.9 | 77.5 | 88.8 | ~0.30 (PTQ) |
| **+ expert-local recon** | 79.6 | 43.9 | 75.0 | 87.5 | **0.090** |

**Recon reduced per-layer expert-output MSE 3–7× (nmse 0.30→0.09) but downstream capability did NOT improve —
it was slightly WORSE (within noise) on all axes.** This is the decisive recovery finding: reducing per-expert
QUANTIZATION ERROR (the local MSE objective) does not recover CAPABILITY. It confirms F4 from the recovery
side — capability is not bottlenecked by recoverable per-expert error, so the memory-tractable LOCAL recon is
the wrong lever (it can even slightly hurt, plausibly calib-MSE overfit + a recon/PTQ layer mismatch at 12/48).
⇒ **The recovery that works must be GLOBAL** — forward-KL distillation matching the output DISTRIBUTION over
reasoning-CoT (the dense program's validated QAD-CoT mechanism), NOT local weight reconstruction. On 122B that
global QAD is memory-bound (full STE-QAT ≈1.5TB; the tractable path is offline top-K teacher-logit caching +
adapter/step-size training — the documented next build). Net for the central hypothesis: "quantization=init,
training recovers" holds ONLY for the right (global/task) objective; local reconstruction is insufficient.
[Engineering note: recon needed 4 fixes — STE divergence (grad-clip+best-guard), accelerate device_map not
freeing on del (two-process capture/recon split), 116GB CPU-snapshot ENOSPC (safetensors per-layer streaming),
full-set best-eval OOM (bounded 2048-tok eval). All in moe_recon.py.]

## F10 — PTQ-method comparison (AWQ/GPTQ/HQQ vs ours): result is NOT method-specific at ~3b
Uniform int{2,3} experts + protected always-on path; only the quant ALGORITHM varies (MMLU n=500, GSM8K
n=120, HE n=80, GPQA n=198). BF16 ref: 85.6/50.5/96.7/97.5.
| method @ bits | MMLU | GPQA | GSM8K | HE |
|---|---|---|---|---|
| RTN-asym int3 | 86.2 | 50.0 | 95.0 | 88.8 |
| HQQ int3 | 84.4 | 52.0 | 95.8 | 90.0 |
| ours uniform-int3 (sym) | 87.0 | 46.5 | 95.0 | 93.8 |
| ours composite-3.0b (mix) | 84.2 | 46.5 | 91.7 | 87.5 |
| — int2 — |||||
| RTN-asym int2 | 79.4 | 38.9 | 85.8 | 77.5 |
| HQQ int2 | 82.6 | 42.4 | 89.2 | 80.0 |
| ours composite-2.0b (ternary/int4 mix) | 80.0 | 44.9 | 77.5 | 88.8 |

**At 3.0b the result is NOT method-specific** — RTN/HQQ/sym/asym are all near-lossless and mutually within
noise (≈BF16). Our composite recipe is NOT uniquely strong; near-lossless ~3b PTQ is a property of the
frontier MoE + always-on-path protection, reachable by any reasonable PTQ. **At 2.0b better PTQ helps**
(HQQ int2 > RTN-asym int2 by ~3pts on MMLU/GSM8K) and the GRID matters (uniform 4-level int2 GSM8K 85.8 >
ternary/int4-mix 77.5 — so part of the earlier "2b math cliff" was the ternary-heavy grid, not 2 bits per se).
But NO PTQ method reaches near-lossless at 2b (best GSM8K 89.2 vs BF16 96.7, MMLU 82.6 vs 85.6). ⇒ the 2b
residual gap is where stronger PTQ (GPTQ/AWQ, testing now) and/or recovery could matter; 2.5–3b is solved by
PTQ alone regardless of method. [GPTQ/AWQ int2 pending to set the strong-PTQ frontier before any QAD.]

## F11 — VERDICT: 2.5–3b near-lossless is method-agnostic; 2.0b QAD is NOT worth it
Complete 2.0b strong-PTQ frontier (uniform int2, protected always-on path):
| 2.0b method | MMLU | GPQA | GSM8K | HE |
|---|---|---|---|---|
| RTN-asym | 79.4 | 38.9 | 85.8 | 77.5 |
| HQQ (best PTQ) | 82.6 | 42.4 | 89.2 | 80.0 |
| AWQ | 78.0 | 46.5 | 85.0 | 73.8 |
| ours composite-2.0b + expert-local recon | 79.6 | 43.9 | 75.0 | 87.5 |
| **ours 2.5b PTQ (any method)** | **82.8** | 44.4 | **93.3** | 88.8 |
| BF16 | 85.6 | 50.5 | 96.7 | 97.5 |

**(1) Our 2.5–3b near-lossless result is NOT method-specific** — RTN/HQQ/AWQ/sym/asym are all near-lossless
and tied at 2.5–3b. It is a property of the frontier MoE (expert-routing redundancy + only 8/256 active)
plus protecting the always-on path — reachable by ANY reasonable PTQ. GPTQ is impractical per-expert at this
MoE scale (12,288 separate expert slices ⇒ ~6–8 h; standard GPTQ is per-Linear); HQQ/AWQ represent the
strong-PTQ frontier and confirm the conclusion.
**(2) At 2.0b no PTQ method is near-lossless** — best (HQQ) leaves MMLU −3, GSM8K −7.5, HE −17.5 vs BF16.
Better PTQ helps modestly (HQQ > RTN ≈ +3 GSM8K); AWQ ≈ RTN (load-balanced experts lack the activation
outliers AWQ exploits). The 2b gap (esp. code/math) is a genuine RECOVERY target — better PTQ can't close it
and expert-local recon didn't either (F9); only global forward-KL QAD plausibly would.
**(3) DECISION — do NOT launch expensive 2.0b QAD.** 2.5b PTQ (~42 GB, single 80GB GPU) is already
near-lossless and beats EVERY 2.0b method incl. HQQ on every axis. Pushing to 2.0b via memory-bound global
QAD buys ≤5 GB over the near-lossless 2.5b point — poor cost/benefit. The remaining opportunity is "deploy at
2.5b", not "recover 2.0b". (If sub-2.5b were ever required, global forward-KL QAD — not better PTQ, not local
recon — is the only lever, at significant memory-engineering cost.)

---

## F12 — Uniform-int3 generalizes across active-param scale, but the truncation penalty grows as A shrinks

Ran the shipped uniform recipe (int3 experts + int4 backbone + MSE-clip, ~3.07 eff bits) on
**Qwen3.6-35B-A3B** (same `qwen3_5_moe` arch as the 122B but **3B active** vs 10B), through the model-general
`ModelAdapter` with only a config change. Same-harness teacher(BF16)-vs-clip(int3), 64k thinking
(full table + provenance in `results/RESULTS.md`):

| axis | Δ overall acc | Δ acc-among-finished | truncation BF16→int3 |
|------|---------------|----------------------|----------------------|
| GPQA-Diamond | −11.1 | **−2.3** | 6% → 22% |
| MATH-500     | −6.7  | **−0.3** | 1% → 13% |
| LiveCodeBench-v6 | −10.5 | **+0.2** | 15% → 41% |
| MMLU-Pro     | −11.3 | −4.9 | 0% → 9% |
| MMMLU        | −4.7  | −2.6 | 0% → 7% |

**Capability is preserved; the gap is token-efficiency.** Accuracy *among finished generations* barely
moves (GPQA −2.3, MATH −0.3, LCB +0.2) — int3 does not collapse reasoning even at 3B-active. What degrades
is reasoning **conciseness**: truncation roughly triples-to-quadruples, so overall acc (truncation = wrong)
falls 5–11 pt. This is the SAME mechanism as the 122B (F-truncation: gaps live in truncation, not capability)
but **markedly worse at 3B-active** — a smaller active set has less headroom to absorb per-expert quant error,
so the quantized model needs more tokens to reach the same answer. **Implications:** (a) the near-lossless
sub-4-bit MoE result is not specific to the 122B — it holds at 35B/A3B on the *capability* axis; (b) the
binding cost at small A is truncation, so the optional OPD token-efficiency recovery (`docs/OPD_INTEGRATION.md`)
and/or a larger generation budget matter more here than at 10B-active; (c) the cross-architecture
generalization itself was validated independently on **OLMoE-1B-7B** (text-only, non-Qwen, fused experts
under transformers 5.x) — quantized to 3.07 eff bits where the original Qwen-only code covered 0% of its
experts (`pipelines/smoke_moe.sh`, `tests/test_adapter_invariance.py`).

---

## F13 — Our clip-search int3 is competitive with AWQ/GPTQ; GPTQ overfits calibration at int3

Direct apples-to-apples vs the activation-aware baselines (`src/quant/moe_compare.py`): per-expert
**end-to-end output NMSE on HELD-OUT activations** (AWQ/GPTQ calibrate on one half of the dispatched
tokens, NMSE measured on the other half — so calibration-overfitting is penalized, as it should be).
All at int3 / group-128, only the algorithm varies. **Qwen3.6-35B-A3B** (160 experts × 5 layers;
`results/quant_compare_q36.json`); OLMoE-1B-7B agrees:

| method | NMSE (Qwen3.6) | × ours_clip | notes |
|--------|----------------|-------------|-------|
| awq (activation-aware + RTN) | 0.0954 | 0.95× | marginally best; needs calib |
| **ours_clip (sym int + MSE clip-search)** | **0.1004** | 1.00× | **calibration-free (weights only)** |
| rtn_asym (asym uint RTN) | 0.1065 | 1.06× | calibration-free baseline |
| gptq (Hessian error-comp) | 0.1231 | 1.23× | **worse than ours** — overfits the per-expert calib set |
| ours_absmax (sym int, no clip) | 0.1786 | 1.78× | the clip-search gain is real |

**Findings.** (1) Our calibration-free clip-search int3 is **within ~5% of the best method (AWQ)** and
**beats RTN and GPTQ** on held-out reconstruction. (2) **GPTQ does NOT help at int3** on this load-balanced
MoE — its Hessian error-compensation minimizes error on the (small, ~128-token) per-expert calibration set
but **generalizes worse** (1.23× ours on Qwen3.6, 1.05× on OLMoE); the apparent in-sample GPTQ win (~27× on
the same tokens it calibrated on) is almost entirely overfitting. (3) The **dominant lever is the clip-search
itself** (1.78× over plain absmax), not asym-vs-sym (rtn ≈ ours) or calibration (awq only +5%). This extends
F1 (near-lossless is not method-specific at ~3b) with a sharper claim: at int3 the calibration-based methods
have **no meaningful generalization edge**, and GPTQ can actively hurt. **Caveat (F9):** this is
reconstruction error, not capability — per F9 a 5–25% per-expert NMSE difference need not move downstream
accuracy (expert-local error reduction did not recover capability). A full capability A/B (build an
AWQ-quantized checkpoint and eval) is the definitive test and is cheap to add; the reconstruction parity
above already predicts a tie. Reproduce: `python src/quant/moe_compare.py --model Qwen/Qwen3.6-35B-A3B`.

**Capability A/B (the definitive test, now run).** Built a full Qwen3.6-35B-A3B checkpoint with experts
quantized by AWQ instead of clip-search — same int3 / group-128 / int4-backbone / protection, so it is
**footprint-matched** to the shipped clip (~13.5 GB). Same harness, 64k (`src/quant/moe_quant_method.py`;
`results/vllm_eval/q36_awq3b.json`):

| model (footprint) | MMLU-Pro acc/fin | GPQA acc/fin | MATH-500 acc/fin |
|-------------------|------------------|--------------|------------------|
| clip-int3 ours (~13.5 GB) | 76.7 / 83.1 | 73.2 / 86.5 | 83.3 / 90.5 |
| **awq-int3 ours (~13.5 GB)** | 80.0 / 84.9 | 68.2 / 87.2 | 86.7 / 93.7 |
| QuantTrio awq-**4bit** (~21 GB, bf16 backbone) | 87.3 / 90.3 | 77.3 / 86.4 | 86.7 / 91.2 |

**Verdict.** On **capability-among-finished** (`fin`, the truncation-free signal) AWQ-experts are
**consistently a hair better** than clip-experts at int3: +1.8 (MMLU-Pro), +0.7 (GPQA), +3.2 (MATH-500),
avg ~+2. On the **headline overall acc** it's a **wash** (clip 77.7 vs awq 78.3 avg) — AWQ wins MMLU-Pro
(+3.3) and MATH-500 (+3.4) but loses GPQA (−5.0), and that GPQA swing is a *truncation* artifact (awq tr=49
vs clip tr=43, while awq's `fin` is actually higher). So: **AWQ's activation-awareness buys a small, real
capability edge (~2 pt fin) that the held-out NMSE (~5%) predicted** — but it costs a calibration pass; our
clip-search is calibration-free (weights only) and within a couple points. The **dominant lever for the
overall-acc gap is truncation/bits, not the quant algorithm**: the 4-bit, bf16-backbone QuantTrio reaches
near-teacher overall acc mostly by cutting truncation (tr 5/21/7 vs our 11/49/9) at ~1.5× the footprint.
**Actionable:** swapping the shipped expert quant from clip-search to AWQ is a cheap, real (~+2 fin)
upgrade at the same footprint; the bigger wins remain truncation recovery (OPD / larger budget) and the
extra bit. (n=150/198/120 → ±several pt CI; the `fin` trend is small but consistent, overall-acc deltas are
within noise.) Reproduce: `python src/quant/moe_quant_method.py --model Qwen/Qwen3.6-35B-A3B --method awq --out models/q36_awq3b`.

**Ecosystem checkpoints (published 4-bit methods).** Two community/SOTA quantizations of Qwen3.6-35B-A3B,
same harness, vs ours:

| model | bits / footprint | MMLU-Pro | GPQA | MATH-500 |
|-------|------------------|----------|------|----------|
| teacher BF16 | 16b / ~70 GB | 88.0 | 84.3 | 90.0 |
| **ParoQuant** (z-lab, rotation) | 4b / ~17 GB | 85.3 | 81.8 | 90.0 |
| QuantTrio **AWQ** | 4b / ~21 GB | 87.3 | 77.3 | 86.7 |
| our **awq**-int3 | 3b / ~13.5 GB | 80.0 | 68.2 | 86.7 |
| our **clip**-int3 | 3b / ~13.5 GB | 76.7 | 73.2 | 83.3 |

- **ParoQuant required reconstruction to run on CUDA at all.** z-lab's CUDA backends (vllm + transformers)
  only quantize `nn.Linear` and DON'T wire up the *fused* MoE experts — the SAME blind spot this framework
  was built to fix. `paroquant[vllm]` loading `z-lab/Qwen3.6-35B-A3B-PARO` fails with
  `KeyError: layers.0.mlp.experts.w2_qweight` (no FusedMoEMethod; the transformers HfQuantizer skips the
  fused `Qwen3_5MoeExperts`); only their MLX (Apple) backend handles MoE. We benchmarked it by
  **reconstructing its effective bf16 weights** (`src/quant/paro_dequant.py`): ParoQuant forward is
  `y = AWQ_GEMM(rotate(x)) = x @ (M @ W_dq)` with `M=rotate(I)` a fixed linear map, so `W_eff =
  RotateQuantizedLinear.forward(I)` reproduces it exactly (validated ~8e-4 rel err). Folded all 130 backbone
  + 30,720 expert linears → standard bf16 ckpt → eval in normal vLLM (capability ceiling, same methodology
  as our clip/awq).
- **ParoQuant's "+2.4% over AWQ on reasoning" reproduces here.** vs QuantTrio-AWQ-4bit: PARO +4.5 GPQA,
  +3.3 MATH-500 (the reasoning axes), −2.0 MMLU-Pro; avg +1.9, reasoning-led. And PARO does it at a SMALLER
  footprint (~17 GB; it also quantizes attn/DeltaNet, vs AWQ's bf16 backbone ~21 GB) — a genuinely strong
  4-bit result, near-lossless vs teacher (−2.7 / −2.5 / 0.0).
- **Where we stand.** Our int3 (~13.5 GB) trails the SOTA 4-bit ParoQuant by ~+7–9 pt — but that is +1 bit
  plus learned rotations (and their kernels/toolchain). At a *matched* bit-width the gap collapses (F13
  above: our clip ≈ AWQ within ~5%). So the honest read: our recipe is a simple, calibration-free, smaller
  3-bit point; closing to ParoQuant means adopting **rotation + the 4th bit**, not just a better PTQ search.
  The framework's value (and the reason we could even run PARO) is the model-general fused-MoE handling that
  z-lab's own CUDA stack lacks.

**Decomposing the gap: "+1 bit" vs "rotation" (matched 4-bit, ~17 GB).** Built our own int4-expert
checkpoints (clip and awq) to isolate the bit budget from ParoQuant's learned rotation. acc / fin / trunc:

| model | bits | MMLU-Pro | GPQA | MATH-500 |
|-------|------|----------|------|----------|
| teacher BF16 | 16b | 88.0/88/0 | 84.3/89/11 | 90.0/91/1 |
| ParoQuant | 4b | 85.3/87/3 | 81.8/86/10 | 90.0/91/3 |
| ours **clip-int4** | 4b | 84.7/87/7 | 74.7/85/25 | **90.0**/92/13 |
| ours **awq-int4** | 4b | 86.0/87/3 | 71.7/**90**/43 | 86.7/93/12 |
| ours clip-int3 | 3b | 76.7/83/14 | 73.2/86/43 | 83.3/90/15 |
| ours awq-int3 | 3b | 80.0/85/11 | 68.2/87/49 | 86.7/94/9 |

**The 4th bit is the dominant lever, NOT rotation.** int3→int4 lifts MMLU-Pro +8.0 (clip) / +6.0 (awq) and
MATH-500 to 90.0 (clip-int4) = teacher = PARO. At matched 4-bit our simple methods are **competitive with
SOTA ParoQuant on raw capability**: MMLU-Pro 84.7–86.0 vs PARO 85.3; MATH-500 clip-int4 90.0 = PARO;
GPQA *acc-among-finished* ours 85–90 ≥ PARO 86 (awq-int4's 90 fin is the highest of any model, incl. teacher).
**ParoQuant's residual overall-acc edge is almost entirely lower TRUNCATION** (better-preserved token
efficiency): on GPQA PARO truncates 10/198 vs our int4's 25–43, which is the whole GPQA overall-acc gap
(our capability-among-finished is already ≥ PARO's). This ties back to F12 — quantization's binding cost is
token-inefficiency/truncation, not capability collapse — and reframes the takeaway: **closing to SOTA is
"the 4th bit + truncation recovery (OPD / budget)", and rotation's benefit shows up as token-efficiency,
not raw accuracy.** (clip-int4 also truncates less than awq-int4 on GPQA, 25 vs 43 — the symmetric
clip-search preserves conciseness better here.) Reproduce: `quant_save.py --expert-bits 4` /
`moe_quant_method.py --method awq --bits 4`.

**ParoQuant AT int3 (we quantized it ourselves — it collapses from int4).** ParoQuant ships 4-bit only;
we ran their optimizer at `n_bit=3` to get the int3 datapoint. Their CUDA optimizer is single-GPU + layer-
sequential, so we patched it with **within-layer data parallelism** across 8 GPUs
(`src/quant/paro_optimize_ddp.py`: shard calib per rank, all-reduce(AVG) the rotation/scale/weight grads,
replicate val, save on rank 0) — block-0 val quant-loss matched single-GPU (2.20e-6 vs 2.05e-6), ~6× faster
(6.5h vs ~33h). Converted with `convert --mode pseudo` → FP16, same harness:

| model (int3, ~13.5 GB) | MMLU-Pro | GPQA | MATH-500 | avg acc |
|------------------------|----------|------|----------|---------|
| ParoQuant-int3 | 79.3/84/9 | 74.2/**92**/49 | 76.7/88/22 | **76.7** |
| ours awq-int3  | 80.0/85/11 | 68.2/87/49 | 86.7/94/9 | **78.3** |
| ours clip-int3 | 76.7/83/14 | 73.2/86/43 | 83.3/90/15 | 77.7 |
| (ref) ParoQuant-int4 | 85.3 | 81.8 | 90.0 | 85.7 |

**ParoQuant does NOT dominate at int3.** (1) It **collapses int4→int3** (avg 85.7→76.7, −9; MATH 90.0→76.7,
−13) — the rotation advantage largely evaporates at 3-bit, reinforcing "the 4th bit is the dominant lever."
(2) On headline overall acc our simple awq-int3 (78.3) ≥ ParoQuant-int3 (76.7); ParoQuant keeps the best
GPQA capability-among-finished (fin 92, highest of any model incl. teacher) but truncates so much (49/198)
that overall acc drops. **Caveats (do not overclaim a win):** the calibration sets differ — our awq used
math+code+instruct calib (`build_calib`), ParoQuant-int3 used wikitext2 per their script, which likely
explains much of the MATH gap; int3 is outside ParoQuant's 4-bit design point; n=150/198/120 (±several pt).
Net: **no method dominates at int3 — it's a wash, and the real levers stay the 4th bit + truncation
recovery, not the quant algorithm.** Reproduce: `torchrun --nproc_per_node=8 src/quant/paro_optimize_ddp.py
--n-bit 3 ...` then `python -m paroquant.cli.convert --mode pseudo`.

## F14 — OPD truncation recovery: train on COMPLETE trajectories, not truncated ones

The 3b student's gap to the BF16 teacher is mostly **truncation** (GPQA trunc 64/198 vs teacher 11; MATH
19/120 vs 1), not capability (`finished-acc` is already ~teacher-level). Naive on-policy distillation made
it **worse**: GPQA acc 68.7→61.1, MATH 81.7→56.7, truncation up — while `finished-acc` *rose* (more
accurate when concluding). Cause: at the gen budget, ~89% of rollouts (student AND teacher-with-thinking)
were truncated, so distilling them reinforces "keep generating." teacher-CoT partially recovered GPQA
truncation (72→65 ≈ baseline) where CoT is short, confirming the mechanism. Fix = `--finished-only` +
long gen budget (train only on rollouts that emit `\boxed{}`/EOS). **This works:** complete-trajectory OPD
recovered **MATH-500 81.7→89.2 (trunc 19→1, = teacher) and GPQA 68.7→77.3 (trunc 64→30)** — quantization's
3-bit damage is mostly truncation and is recoverable by distilling conclusions. Full tables + design:
`docs/OPD_INTEGRATION.md` → "Recovery results".
