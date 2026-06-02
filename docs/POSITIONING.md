# Findings: Positioning, Rigor & Threats

## 1. How the result sits vs. SOTA

**Verdict: credible and consistent with the literature, but not novel in kind — an incremental (lower-bit, multi-axis) extension of already-published results.**

Two papers establish the precedent directly:

| Comparable | Model | Eff. bits | Headline | Relevance |
|---|---|---|---|---|
| **DQ3_K_M** (2505.02390) | DeepSeek-V3/R1, 671B | 3.59b | **0.30% avg drop** across AIME/MATH500/GPQA/MBPP(+)/LiveCodeBench/MMLU/CMMLU/C-Eval (V3); 0.34% (R1). MATH500 95.35 vs 95.45, MMLU 91.03 vs 90.99. Beats FP8 on some axes and llama.cpp Q4_K_M. | **Existence proof** that frontier MoE is near-lossless on knowledge+math+code at ~3b via weight-only PTQ. The bar we extend downward. |
| **XFP** (2605.14844) | **Qwen3.5-122B-A10B (identical model)** | 3.97b | **GSM8K 94.49% ±0.57** (strict-match, 3 seeds, full n=1319). Their Marlin-INT4 = 94.62%; implied BF16 122B ref ~95.27%. Reports **GSM8K only** for the 122B (no MMLU/HumanEval/GPQA). | **Direct head-to-head SOTA.** Uses essentially **our** protection scheme independently: two cosine-similarity floors, τ=0.96 (attn/Gated-DeltaNet/shared-expert), τ=0.93 (routed experts) = "protect always-on path, quantize routed experts hard." |

**Where our result sits:** Our 92.5% GSM8K @3.06b (n=40) is ~2pp below XFP's 94.49% @3.97b, but at **~0.9 fewer effective bits** and demonstrated across 4 axes. Frame this as **"comparable GSM8K at a ~0.9-bit lower budget," not "we beat XFP."** Our numbers sit on a sensible bits-vs-accuracy curve between DQ3/XFP (~3.6–4.0b, near-zero drop) and the small-MoE collapse zone (~2.5b, large drop). Nothing contradicts the literature: MMLU 85.5 (−2.0) is consistent with DQ3's near-zero MMLU given we are ~0.5b lower; GSM8K −3.5 is XFP interpolated down.

**The "unlike dense" framing is well-supported.** Small/older MoE *does* collapse at ~3b: MoEQuant (2505.03804) Mixtral-8x7B W3 GSM8K 65.88→43.21 (−34% rel) even with AGQ+EBSS; MC# (2510.10962) Mixtral 2.54b GSM8K 37.67 vs 58.30 (−35%), explicitly says models "cannot maintain near-lossless performance" at ultra-low bits; DynaExq (2511.15015) Qwen3-80B INT2 → 73.09 vs 78.11. The dichotomy is driven by **model scale/quality + expert-routing redundancy** (2604.19884: routing gives "natural redundancy against deadzone pruning" dense lacks), not MoE-ness per se.

**Our genuine, incremental contributions** (frame as these, NOT "first to show frontier MoE survives 3-bit"):
1. Pushing the near-lossless point **~0.5–0.9 effective bits lower (3.06b)** than the two closest comparables on a 122B model.
2. Demonstrating it **simultaneously on all 4 axes** (XFP published GSM8K only; DQ3 is a different/larger model).
3. The **per-expert composite-salience MPQ allocation** + empirical KL ranking (asal ≈ wmass ≈ composite ≈ freq < wfro << random/blind) and the **load-balanced-router finding** (entropy 0.952, 0 cold experts) that neutralizes the "cold-expert free lunch" the Mixtral/DeepSeek-MoE papers rely on.

**Must do before claiming:** (a) acknowledge XFP independently published the protection scheme; (b) **normalize the effective-bit definition** before claiming "lower than XFP/DQ3" — conventions differ (weight-only vs incl. scales/zeros vs incl. KV/activation); our 3.06b is weight-bits incl. ~6% protected path, ~3.18b with scales, vs DQ3 3.59b / XFP 3.97b own conventions; (c) re-verify 2026-range arXiv IDs (XFP 2605.14844, GEMQ 2605.23078) against the live listing — GEMQ's "2.5b Mixtral ~7% MMLU drop" could not be verified here.

## 2. Statistical caveats — concrete CIs and which claims need more n

**Verdict: UNDERPOWERED.** Every headline 3b-vs-BF16 delta is statistically indistinguishable from zero at reported n, including under the correct, more-powerful paired/McNemar framing. This is a failure-to-reject from small samples, **not evidence of equivalence.**

Evals are deterministic greedy decode (do_sample=False), so the *only* uncertainty is finite-item binomial sampling — Wilson CIs capture it exactly, McNemar is valid, and re-running won't change numbers (only larger samples will).

**95% Wilson CIs (point [lo, hi], half-width):**

| Benchmark | n | Acc (count) | 95% CI | ±HW |
|---|---|---|---|---|
| GSM8K 3b | 40 | 92.5% (37/40) | [80.1, 97.4] | ±8.6 |
| HumanEval 3b | 32 | 90.6% (29/32) | [75.8, 96.8] | ±10.5 |
| MMLU 3b | 200 | 85.5% (171/200) | [80.0, 89.7] | ±4.9 |
| MMLU BF16 | 200 | 87.5% (175/200) | [82.2, 91.4] | ±4.6 |
| GPQA-d | 198 | 46.5% (92/198) | [39.7, 53.4] | ±6.9 |
| MMLU 2.5b | 200 | 82.5% (165/200) | [76.6, 87.1] | ±5.3 |

**Are the claimed deltas distinguishable from zero? No.**

| Delta | Two-proportion z (unpaired) | Paired McNemar (best-case discordant split) |
|---|---|---|
| MMLU −2.0 | z=−0.59, p=0.56, CI [−8.7,+4.7] | net −4 items, p=0.125 |
| GSM8K −3.5 | z=−0.67, p=0.50, CI [−13.7,+6.7] | net −1 item, p=1.0 |
| HumanEval −9.4 | z=−1.77, p=0.076, CI [−19.5,+0.7] | net −3 items, p=0.25 |
| **MMLU 2.5b −5.0** | — | **net −10 items, p≈0.002 (favorable split) to ≈0.10 (high discordance)** |

The **only quantization effect with statistical support is the 2.5b MMLU −5.0pt drop** — which reinforces the program's hypothesis that recovery/QAD value lives at the **2–2.5b floor.**

**Which claims need larger n (ranked by weakness):**
1. **GSM8K (n=40) and HumanEval (n=32) are weakest** — ±9 to ±11pt CIs make any sub-10pt "near-lossless" claim indefensible. Must scale up.
2. **Flagship MMLU −2.0** is within noise at n=200 (−2.0 = exactly 4 items; paired p≥0.125).
3. **GPQA-d 46.5%** has **no BF16 baseline captured** — its delta is untestable; bare ±6.9pt point estimate near chance-adjusted floor.

**Required n (α=0.05, 80% power):**
- **MMLU true 2pt gap:** ~4,600/arm unpaired (~9,200 total) OR paired ~1,000 (5% discordant) / ~2,000 (10%) / ~3,900 (20%). **Recommendation: run the full ~14k MMLU test set once per config** — removes sampling noise as the limiting factor.
- **GSM8K true 3.5pt gap:** ~700/arm unpaired → **use full 1,319-item test set** (trivially sufficient).
- **MMLU-2.5b 5pt gap:** ~800/arm.
- **HumanEval:** only 164 problems total; even the full set gives ~±5pt at ~91% and **cannot resolve a sub-5pt code delta** — report pass@1 greedy on all 164 and frame small deltas as bounded-by-resolution.
- **GPQA-diamond:** 198 items is the entire diamond set; ±7pt is a hard ceiling absent a larger reasoning benchmark.

**Always report Wilson CIs + McNemar p-values, capture a paired BF16 baseline on every axis (esp. GPQA), and pin down the true per-axis n from run logs** — the b30 study JSON stores n=32 (HumanEval field); if MMLU was actually n=32 its CI balloons to ±12pt and the −2.0 delta is even less resolvable (paired p=1.0). The 96.0 BF16 GSM8K is not an integer count at n=40/25/80, so the GSM8K n is currently unverifiable from artifacts.

## 3. Ranked next experiments

**E1 — Re-run flagship composite-3.0b AND BF16 with proper statistical power. [highest value, make-or-break]**
GSM8K full 1,319; HumanEval full 164; MMLU full ~14k (or ≥1,000); report Wilson CIs and **paired McNemar** vs BF16. Uses the existing harness. This is the gate for any "near-lossless" claim and closes (or honestly justifies) the ~2pp gap to XFP via the ~0.9-bit budget difference.

**E2 — Generation under allocation (does the KL gap become a decode gap?).**
The salience-vs-random KL gap (0.058 vs 0.124, ~2x, low-variance over ~6k calib tokens) is real but its translation to generation is **untested** — random/blind have no GSM/HE runs. Run GSM8K + HumanEval (n≥200) for **random, blind, and inverse at 3.0b.** Forward-KL over top-256 is insensitive to exactly the tail-token / long-horizon error accumulation that breaks generation, so this directly tests whether allocation is generation-second-order or load-bearing.

**E3 — Measure the PTQ generation cliff at 2.0/2.25/2.5b, then run recon where a real gap exists.**
The recovery/QAD arm is **entirely unexecuted** (results/recon/ absent; only a synthetic nmse 0.26→0.19 self-check). Logical trap: if PTQ is near-lossless at 3b there is no headroom to recover, and at 2.0–2.5b only MMLU/KL are screened (composite-2.5b: MMLU 82.5 n=200, ppl 2.01) with **no generation numbers** — so we don't yet know a generation cliff exists to recover. The recovery claim needs (a) a *measured* PTQ generation cliff at 2.0/2.25/2.5b on GSM8K+HumanEval, then (b) the already-written moe_recon closing it. This is the genuinely open, more-novel direction.

**E4 — Deployment fidelity (convert storage estimate into a measured artifact).**
Current eval is weight-only fake-quant: fakequant() returns dequantized bf16 (no integer storage, no real kernel), **activations are never quantized** (full bf16 activations through dequant-bf16 weights). The ~50GB / 3.06b figure is sound *arithmetic* but "deployable" conflates a storage estimate with a measured artifact, and "3.06 effective bits" is honest only as a tag-average — true storage ~3.18b with group-128 scales, a load-bearing protected ~6% backbone (shared int8 / attn-ssm-embed int4 / router+norms bf16), and a **bimodal expert mix (58.7% int4 + 41.3% ternary 1.58b, no true int3)**. Validate ≥1 real low-bit path (NVFP4/XFP codebook-VQ or a true int-packed kernel) end-to-end on a few layers, **quantize activations**, and measure throughput/latency (currently zero). Also add a **routing-flip-rate measurement under PTQ** — PTQ of experts can flip top-8 routing vs the FP-captured idx, which would undercut both the recon design (router frozen bf16) and the "errors average over 8 active experts" survival mechanism.

**E5 (secondary) — Capture a GPQA BF16 baseline** (bf16.json has no gpqa_acc) so the 46.5 reasoning number has a testable delta; **add multi-seed random-allocation controls** (currently single seed=0, so "salience beats random 2x" is one draw, not a distribution).

**Honest headline to adopt now:** "lowest-bit (expert-avg 3.0b; 1.58/4 mix; 3.06b tag, ~3.18b w/ scales) near-lossless 4-axis weight-only PTQ on Qwen3.5-122B with a protected ~6% always-on backbone, plus a per-expert salience-allocation/ablation study" — pending E1 to make "near-lossless" statistically defensible against the XFP/DQ3 credibility bar.