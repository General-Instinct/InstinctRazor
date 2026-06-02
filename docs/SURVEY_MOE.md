# MoE Sub-4-bit Quantization & Recovery — Survey (2025-26) for Qwen3.5-122B-A10B

> **Target.** Qwen3.5-122B-A10B: 122B total / ~10B active MoE. 256 routed experts + 1 always-on shared expert, top-8 routing, 48 layers (12 sparse full-attention every 4th layer + 36 Gated-DeltaNet linear-attention layers), hidden 3072, moe_intermediate 1024, untied 248320-token VL vocab. Hardware: 4×H100 80GB = 320GB total.
> **Core hypothesis (adopted).** Quantization is *initialization*; capability recovery comes mainly from post-quantization training (forward-KL distillation / QAD / reasoning-CoT data). Full self-distillation QAT is infeasible at 122B on 320GB → recovery must be memory-tractable (block-wise reconstruction, adapter/LoRA distillation, offline-teacher-logit KD).
> **Counting facts that drive every decision below.** 48 layers × 256 experts × 3 linear blocks (gate/up/down) = **36,864 expert weight matrices**. A per-module re-forward search over these is infeasible; every allocation mechanism we pick must be one-pass or closed-form. The shared expert fires on every one of 48 layers per token → its error multiplies ~48× vs a 1-in-256 routed expert. The 36 Gated-DeltaNet layers carry recurrent state that, like SSM PScan, *accumulates* quantization error across the sequence.

---

## 1. Executive Summary — Highest Impact-per-Effort Mechanisms

Ranked for OUR setting. "Effort" weighs implementation complexity and GPU-hours; "Impact" weighs expected accuracy recovery on MMLU/GPQA/GSM8K/HumanEval.

| # | Mechanism | Source | Effort | Why it wins for us |
|---|-----------|--------|--------|--------------------|
| **1** | **Forward-KL QAD from a frozen BF16 self-teacher** (T=1, ~1–3B tokens, LR 1e-5), NOT QAT-with-task-loss. Offline-logit-cached so the teacher need not co-reside with the student. | NV-QAD `2601.20088`, Reasoning-QAT `2601.14888`, UPQ `2506.09104` | Med | This is the single largest accuracy lever. QAT with cross-entropy *destroys* RL-trained reasoning (Nemotron-30B: LiveCodeBench 72.1→62.0, worse than PTQ). Forward-KL recovers near-BF16. Robust to data quality (random tokens ≈ curated). Larger models need *fewer* tokens/param — favorable for 122B. |
| **2** | **Affinity-Guided GPTQ (AGQ)**: replace Hessian `H=XXᵀ` with `H=(X·c)(X·c)ᵀ`, c = router gate score. + **EBSS expert-balanced calibration**. | MoEQuant `2505.03804` | **Low** (5-line GPTQ patch) | With 8/256 routing, a 512-seq random calibration set leaves many experts with ~0 tokens. AGQ+EBSS fixes the sparse-calibration pathology that silently wrecks cold-expert quantization. Cheapest high-value change. |
| **3** | **One-pass composite expert sensitivity → ILP/greedy mixed-precision allocation** at linear-block granularity (see §2 for explicit math). Stack 3 cheap signals: router L2-norm, data-free Hutchinson Hessian trace, gate-weight. | MxMoE `2505.05799`, MoPEQ `2509.02512`, `2604.06515`, GEMQ `2605.23078` | Med | Avoids the infeasible 36,864-matrix re-forward. Load-balanced training flattens frequency → frequency alone is a bad proxy; the composite triangulates. Block-level beats expert-level by 0.21 PPL. |
| **4** | **Protect the always-on path unconditionally.** Shared expert → INT8/FP16; full-attention layers (12) → ≥INT4; **Gated-DeltaNet `ssm_out`/`ssm_in` → ≥INT4 with KLT rotation**; router gate → ≥W8. | QuantMoE-Bench `2406.08155`, Super-Expert `2507.23279`, MambaQuant `2501.13484`, DQ3_K_M `2505.02390` | **Low** | These are negligible-memory carve-outs that prevent catastrophic collapse. `ssm_out` is the empirically-confirmed extreme outlier of this hybrid architecture (Unsloth KL analysis). Super-expert collapse: pruning 3/6144 experts raised PPL 8.7→59.9. |
| **5** | **Router KL-consistency calibration (EA-RCA)** during router quantization: minimize `MSE(W_g, Q(W_g)) + λ·KL_top-k(p_fp‖p_quant)`, λ≈0.1, top-8 only. | EAQuant `2506.13329` | Low | Prevents expert-shift: quantizing the 256-way router flips top-K selection. Cheap calibration-time constraint that protects routing topology. |
| **6** | **MiLo calibration-free low-rank compensators** on INT3 experts: alternating HQQ + truncated-SVD on the weight residual; rank by kurtosis/frequency. +1.4% memory, no training data. | MiLo `2504.02658` | Low | Free PPL recovery (+~1 PPL) that improves the *init* before QAD. Ships W3A16 kernels 1.26× faster than Marlin W4A16. Shared expert → high rank (~512), cold experts → rank 8–32. |
| **7** | **Progressive bit schedule (W4→W3→W2 warm-start)** instead of direct sub-3-bit init. | Bit-by-Bit `2604.07888`, UPQ `2506.09104`, ParetoQ `2502.02631` | Low | INT4-intermediate cuts FP16→INT2 quantization error 43% (0.898→0.516). Lower-bit grids nest in higher grids, so optimal high-bit rounding is a valid low-bit init — prevents loss spikes. |
| **8** | **DynaExq online hot-expert promotion at serving** (EMA hotness, hysteresis, async swaps). | DynaExq `2511.15015` | Med | Recovers ~4.5 pts over static INT2 (Qwen3-80B: 73.09→77.57) at near-zero extra memory by promoting top-~32 experts/layer to INT4. Orthogonal to everything above; deploy last. |

**One-paragraph recipe.** Run Super-Expert + `ssm_out` profiling (1 forward pass) → carve out protected tensors. Build EBSS calibration set. Compute the composite one-pass sensitivity (§2) and solve the block-level ILP to a ~3.0-bit expert average with progressive W4→W3 warm-start, using AGQ-GPTQ + EA-RCA router alignment. Add MiLo compensators. Then run offline-logit forward-KL QAD (§3) on ~1–3B reasoning tokens. Optionally finish with Silver-Bullet DPO and DynaExq serving-time promotion.

---

## 2. MoE Bit Allocation — Concrete One-Pass Algorithm

### 2.1 The hard constraint

We cannot do per-module re-forward search over 36,864 matrices. Every signal below is either **closed-form on weights** (zero forward passes) or **one calibration forward pass**. We allocate at **linear-block granularity** (gate/up/down independently), because MxMoE (`2505.05799`) shows blocks inside one expert have heterogeneous sensitivity (0.21 PPL gain over expert-level), and the ILP is trivially small.

### 2.2 The four sensitivity signals (explicit math)

For each routed expert `e` in layer `l`, and each linear block `b ∈ {gate, up, down}`:

**(A) Router L2-norm proxy — zero cost, no data** (`2604.06515`).
We have no init checkpoint, so use the *final* router row norm as a surrogate for norm-change:
```
norm_e = ‖ W_router[e, :] ‖_2          # one row of the [3072 × 256] gate matrix
```
Counter-intuitive but theoretically grounded rule: **small norm ⇒ rare-feature specialist ⇒ MORE bits**. Rank ascending. Free (just row norms).

**(B) Data-free Hutchinson Hessian trace — seconds, no data** (MoPEQ `2509.02512`).
Using the Frobenius-norm surrogate loss `½‖W‖²_F`, the HVP collapses to identity (`HVP = v`), giving the clean closed form:
```
Tr(H_e,b) ≈ (1/m) Σ_{i=1..m} ‖ W_{e,b} v_i ‖²   ,   v_i ~ N(0, I)   # m = 30 suffices
```
This is just the expected squared output norm — pure weight-space, no calibration data. Sum over the three blocks for an expert-level score; keep per-block for block-level allocation.

**(C) Output-distortion proxy — one calibration forward** (MxMoE `2505.05799`, GEMQ `2605.23078`).
On the EBSS calibration set, for the candidate low-bit scheme `k`:
```
Δ_{e,b,k} = ‖ O_{e,b}(W) − O_{e,b}(Q_k(W)) ‖_2     # block output, tokens routed to e only
```
Because only 8/256 experts fire per token, gathering this is cheap. This is the *objective* term; (A),(B) are *priors/floors*.

**(D) Intra-neuron MaxVar override — zero cost** (`2604.06515`).
```
MaxVar_e = max_r  Var_i( W_{e,gate}[r, i] )
```
Any expert with `MaxVar_e > mean + 2·std` (≈ top 4–5%) is force-promoted one tier regardless of (A)–(C): outlier-concentrated weights blow up at low bit.

### 2.3 Affinity-weighted calibration (so the signals are *correct*)

Whenever (C) or the actual GPTQ runs, weight calibration tokens by router affinity (MoEQuant AGQ `2505.03804`):
```
H_AGQ = (X · diag(√c)) (X · diag(√c))ᵀ = (X·c) Xᵀ      # c_i = gate softmax for token i→expert e
```
For the shared expert, `c ≡ 1`. This is a ~5-line patch to any GPTQ implementation and is the highest-ROI calibration fix for sparse routing.

Build the calibration set with **EBSS** so all 256 experts get coverage:
```
score(S‖v) = −1/(l+1)·(R_S + log P(v|S)) + σ(M,S)/τ ,  τ=1.2, beam w=4
```
where `σ` is the std-dev of expert activation counts. Supplement with **EA-CDB** (`2506.13329`): oversample until every expert sees ≥ `r·k·N/n` tokens (r=2.0) → expect 512–1024 sequences needed for 256 experts.

### 2.4 Composite score and the allocation solver

Combine via rank-fusion (robust to scale differences):
```
S_e,b = w1·rank(−norm_e) + w2·rank(Tr(H_e,b)) + w3·rank(w̄_e)        # w̄_e = mean gate weight
```
Start `w1=w2=w3=1/3`; tune on a small dev set. Then solve a **per-layer ILP** at block granularity (PMQ/MC# `2510.10962`, MxMoE):

```
minimize   Σ_{e,b} Σ_k  Δ_{e,b,k} · x_{e,b,k}                 # x ∈ {0,1}, k ∈ {2,3,4}
subject to Σ_k x_{e,b,k} = 1                  ∀ e,b          # one precision per block
           Σ_{e,b,k} bits_k · size_{e,b} · x_{e,b,k} ≤ Budget_l
           diversity: ≥1 block at 2-bit and ≥1 at 3-bit per layer   # avoid degenerate
           floors (hard):  shared expert → 8-bit; MaxVar experts → +1 tier
```
256×3×3 ≈ 2304 binary vars/layer → solves in ms with PuLP/gurobipy. Target **~3.0-bit average** across routed-expert blocks. If a global view matters, use GEMQ's **progressive loop**: quantize → re-measure Δ on calibration → re-solve → router-fine-tune (50–200 steps) → repeat 2–3×. This corrects cross-layer error interactions a single shot misses.

**Optional refinement (ScaleBITS `2602.17698`):** after init, re-evaluate sensitivity *at the quantized model* `s_i = |g(w^Q)ᵀ Δw_i|` (gradients are non-zero away from the FP optimum) and adjust the top-k blocks. Especially relevant for the 36 Gated-DeltaNet layers where sequential error compounds.

### 2.5 Protected-tensor carve-outs (applied before the ILP)

| Tensor | Bit floor | Reason | Source |
|--------|-----------|--------|--------|
| Shared expert (all 48 layers) | INT8 / FP16 | Fires every token; 48× error accumulation | `2406.08155` |
| Super-experts (top down_proj activation: >P99.5 AND >10% global max) | FP16 / INT8 | Pruning 3/6144 → PPL 8.7→59.9; they create attention sinks | `2507.23279` |
| `ssm_out`, `ssm_in` (36 Gated-DeltaNet layers) | INT4 + KLT rotation | Recurrent-state outlier amplification; confirmed extreme outlier on Qwen3.5-122B | `2501.13484`, `2505.02390` |
| 12 full-attention layers | INT4 | Long-context; errors propagate through KV cache | `2406.08155` |
| Router gate (48 matrices) | W8 + EA-RCA KL | Quantization flips top-K selection (expert-shift) | `2506.13329` |
| First ~2–3 high-magnitude `ffn_down_exps` layers | Q6/Q4 | "Super weights"; DQ3_K_M 1-in-4 elevation pattern | `2505.02390` |

### 2.6 KLT rotation for Gated-DeltaNet (the unique architectural risk)

The delta-rule update `S_t = β_t·S_{t-1} + k_t v_tᵀ` is PScan-like and amplifies per-channel outliers; plain Hadamard cannot equalize channel variance. Apply **KLT-enhanced rotation** offline (MambaQuant `2501.13484`):
```
H_K = K·H   where K = eigenvectors of Cov(calibration activations)   # absorbed into ssm_out weight, zero runtime
```
This is the only directly-adaptable SSM-quant technique; without it sub-4-bit on `ssm_out` causes catastrophic KL blow-up.

---

## 3. Recovery / QAD / Distillation — Memory-Tractable Recipe for 122B on 4×80GB

### 3.1 The memory problem and its resolution

A BF16 122B teacher ≈ 244GB; co-residing teacher + trainable student + optimizer on 320GB is infeasible. **Solution: offline logit caching.** Run the BF16 teacher once with TP=4 across the H100s (it fits at 244GB in inference), and cache **top-K logits (K=256) per token in FP16** for the training corpus. QAD then needs only the compressed student (~46GB at INT3) + streamed logits — fits comfortably.

### 3.2 The objective (this is the crux of the hypothesis)

Use **forward-KL distillation against the frozen BF16 self-teacher**, T=1, *no* next-token cross-entropy:
```
L_QAD = D_KL( p_teacher ‖ p_student ) = Σ_y p_FP16(y|x) · log[ p_FP16(y|x) / p_student(y|x) ]
```
- **Never use QAT-with-task-loss on this model.** Both NV-QAD (`2601.20088`) and Reasoning-QAT (`2601.14888`) show task-loss QAT destroys RL-acquired reasoning; forward-KL preserves the full output distribution and is robust to data source.
- **Optional improvements:** generalized JSD with β=0.5 (UPQ `2506.09104`) avoids forward-KL's mean-seeking and reverse-KL's mode-seeking; CAKLD confidence weighting (BitDistiller `2402.10631`) up-weights high-confidence teacher tokens.

### 3.3 Training config

| Knob | Value | Source / rationale |
|------|-------|--------------------|
| LR | **1e-5** (Qwen3 lineage is RL-trained → use the RL-model rate, not 1e-6) | NV-QAD `2601.20088` |
| Token budget | **~1–3B** (interpolating 0.3B@49B, 2.5B@30B — larger models need fewer tokens/param) | NV-QAD |
| Temperature | T=1 (train), T=0.6 (eval sampling) | NV-QAD |
| Data | Math + code CoT (OpenR1-Math, NuminaMath); **calibration & training domain MUST match** | Reasoning-QAT `2601.14888` |
| Init | Quantized checkpoint from §2 (NOT random rounding; NOT FP16) | UPQ, ParetoQ |
| Grids | LSQ asymmetric for W3/W4; SEQ balanced levels for W2 | ParetoQ `2502.02631` |

**3-bit vs 2-bit regime (ParetoQ `2502.02631`):** at W3, QAT is *compensation* (~10–20% of weights shift) → converges at ~10B tokens, fine-tune from checkpoint with standard LR. At W2 it is *reconstruction* (~40% shift) → needs ~30B tokens, ~3× longer. **Recommendation: target W3 average for experts**; reserve W2 only for the coldest tail.

### 3.4 The three implementation paths (choose by budget)

**Path A — Block-wise reconstruction (EfficientQAT `2407.11062`, Bit-by-Bit `2604.07888`).** Hold one block in BF16 at a time + its low-bit copy; the rest stay quantized. 122B@INT3 ≈ 46GB + one FP16 block ≈ well within 320GB. Block-AP MSE reconstruction (4096 samples, group 64), then E2E-QP step-size-only pass in DDP. Add **OCS** outlier-channel splitting (top 10% by `‖X_i‖₂·max|W_ij|`, depth-scheduled to later layers) and the **W4→W3→W2 progressive nested** objective so one run yields all bit-widths. Est. ~25–40 GPU-h for 122B-active-equivalent.

**Path B — Adapter / LQ-LoRA distillation (`2311.12023`).** Decompose `W = W_q + A·B`; freeze `W_q`, train only rank-64 FP16 adapters via QAD. Gradient flows only through tiny adapters → smallest memory. ~1GB trainable params for 122B@3-bit. Merge `quantize(W_q + AB)` at the end. **Best fit when GPU-hours are tight.**

**Path C — QaRL rollout-aligned RL** (`2604.07853`), only if pushing past KD into GRPO. Apply *real* low-bit GEMM in the training forward (STE backward), matching the quantized vLLM rollout engine, so `π_sampler = π_learner`. Use **TBPO** sequence-level dual clipping (`r_seq = exp((1/L)Σ log-ratio)`) to handle error-token accumulation in long Gated-DeltaNet CoT chains. Requires a KD cold-start first.

### 3.5 Selective quantization during recovery

Keep the 12 full-attention layers and the shared expert at higher precision during QAD (frozen-expert strategy possible: freeze 256 routed experts, fine-tune only attention + shared expert + adapters). Run a per-layer PTQ sensitivity scan (block-MSE-increase vs FP16) and exclude the worst 5–10 layers, mirroring NV-QAD's MoE-Mamba hybrid handling.

### 3.6 Cheap finishers

- **MiLo compensators** (`2504.02658`) before QAD: free init boost, no data.
- **Silver-Bullet DPO** (`2505.11574`) after QAD: detect first-error step in quantized CoT (LLM-ensemble judge, 97.2% localization), build ~500 truncation-based preference pairs, LoRA rank-32 DPO (β=1, LR 1e-6) in 3–5 GPU-min — recovers 70–80% of *residual* gap. Build separate sets for GPQA / GSM8K / HumanEval error modes.
- **Router fine-tuning** (GEMQ): after allocation, fine-tune only the 48 router matrices (~37M params) on 1–2k samples to re-align routing to quantized experts.

---

## 4. Format / Kernel Reality — What Actually Gets a Speedup

Sub-4-bit accuracy is only useful if it deploys. The decisive fact: **decode at batch≈1 is memory-bandwidth-bound**, so fewer weight bytes/token = faster, *if* an efficient dequant kernel exists.

| Format | Group / scale | Hardware | Reality for us | Source |
|--------|---------------|----------|----------------|--------|
| **NVFP4** (E2M1 + E4M3 block scale @16 + FP32 tensor scale) | g=16 | H100/Blackwell FP4 tensor cores | **Highest-fidelity HW format.** 2–3× throughput vs FP8, 1.8× memory. PTQ recovers ~96% BF16; +QAD → near-BF16. Rotations *hurt* at g=16 — use SOAR + Four-over-Six + RaZeR instead. | `2601.20088`, `2605.12245`, `2512.02010`, `2501.04052` |
| **MXFP4** (E2M1 + E8M0 pow-2 scale @32) | g=32 | Open/Blackwell + AMD | ~40% more MSE than NVFP4 from coarse pow-2 scales; **MR-GPTQ Hadamard (k=32) + MBS-Hybrid + OAS closes gap 10%→<1%**. Actually 15% *faster* than NVFP4 (simpler scale decode). Use if AMD/OCP portability matters. | `2509.23202`, `2603.08713` |
| **XFP** adaptive Lloyd codebook | per-128 group | RTX PRO 6000 / H100 (custom CUDA) | **The only method benchmarked directly on Qwen3.5-122B-A10B.** ~3.97 effective bits → **138 tok/s, +49% vs Marlin INT4, 94.49% GSM8K.** Two quality floors map exactly to our arch: τ=0.96 (attention/DeltaNet/shared), τ=0.93 (routed). k=4σ FP16 outlier residual ≤2%. | XFP `2605.14844` |
| **MiLo W3A16** | g per-expert | H100 | Custom 3-bit kernel, zero-waste packing, **1.2–1.26× faster than Marlin W4A16** at batch>1. Ships the deployment path for INT3 + compensators. | `2504.02658` |
| **Marlin INT4 (W4A16)** | g=128 | H100 | The baseline everyone beats; reliable, well-supported. | — |
| Pure VQ (VPTQ/PCDVQ 2-bit) | vector | — | Best 2-bit *accuracy* (PCDVQ `2506.05432`: E8-lattice direction + Lloyd-Max magnitude) but **no HW-native decode** → use VQ-LLM `2503.02236` codebook-cache kernels, or only for the coldest tail where memory dominates. | — |

**Practical call.** For 4×H100, the two live options are **(a) NVFP4 ~4-bit** (FP4 tensor cores, stack Four-over-Six → SOAR CJSO+DSS → RaZeR → AGQ-GPTQ, then QAD) and **(b) XFP ~3.97-bit adaptive codebook** (proven on this exact model, +49% decode). NVFP4 is the safer accuracy/throughput bet given native tensor-core support; XFP is the proven-on-Qwen3.5 sub-INT4 throughput play. Ternary/2-bit win the accuracy-size Pareto frontier but are *less deployable* (ternary needs >90% sparsity; 2-bit needs reconstruction-regime training).

---

## 5. Per-Method Reference Table

| Method | arXiv | Year | MoE? | Bits | Core mechanism we use | Headline result |
|--------|-------|------|------|------|----------------------|-----------------|
| MxMoE | `2505.05799` | 2025 | ✓ | 2–5 | Per-linear-block ILP, `min L^r·T^(1-r)`, Δ=‖Ô−O‖₂ | 2.4× lower PPL vs GPTQ @2.25-bit; 3.4× speedup |
| MC# (PMQ/OTP) | `2510.10962` | 2025 | ✓ | 1–3 | LP `Σ φ^α w^β ε^γ x`; diversity constraint | 2.54-bit Mixtral, 67.5% (−3.8% vs FP16) |
| DynaExq | `2511.15015` | 2025 | ✓ | INT2↔FP16 | EMA hotness + hysteresis, async swap, dual copies | Qwen3-80B 73.09→77.57 |
| EAQuant | `2506.13329` | 2025 | ✓ | W2A4–W4A4 | EA-SA smoothing, EA-RCA router KL, EA-CDB balance | +1.15–13.81% over DuQuant |
| MoPEQ | `2509.02512` | 2025 | ✓ | 2/3/4 | Data-free Hutchinson Tr(H) + K-means(C=3) | <5% loss, beats freq baseline 63/105 |
| EAC-MoE | `2508.01625` | 2025 | ✓ | 2–3 | QESC TopK-MSE calibration; PESF pruning | Mixtral 2.06-bit 66.31% (ACL'25) |
| Gen.-Guarantee MoE | `2604.06515` | 2026 | ✓ | 2–4 | Router-norm Λ_s + MaxVar override; theory bound | 2.75-bit Mixtral 70.01% |
| GEMQ | `2605.23078` | 2026 | ✓ | 2.5–3.5 | Global ILP + progressive loop + router-FT | 2.5-bit Mixtral, 7% MMLU drop (ICML'26) |
| CoopQ/IMPQ | `2509.15455` | 2025 | ✗ | 2–4 | Shapley SPQE + pairwise K, MILP | 20–80% PPL reduction |
| SliM-LLM(+) | `2405.14917` | 2024-25 | ✗ | 1–4 | OBS group salience `w²/[H⁻¹]²`, double-ptr | 48% PPL reduction @2-bit (ICML'25) |
| MixLLM | `2412.14590` | 2024 | ✗ | 4–8 | Global output-channel Fisher salience | <0.2 PPL Δ @4.4-bit on L3.1-70B |
| ScaleBITS | `2602.17698` | 2026 | ✗ | sub-4 | Sensitivity at quantized ref, 16–36 iters | ~0.5–1h H100 allocation |
| EfficientQAT | `2407.11062` | 2024-25 | ✗ | 2–4 | Block-AP + E2E-QP | 70B@W2 −2.93pp MMLU, 41h/A100 |
| Bit-by-Bit | `2604.07888` | 2026 | ✗ | 2–4 | Progressive schedule + OCS + nested | 7B@W2A16 6.50 vs 7.39 PPL |
| BitDistiller | `2402.10631` | 2024 | ✗ | 2–3 | CAKLD confidence-weighted self-distill | Beats EfficientQAT @W2/W3 |
| NV-QAD | `2601.20088` | 2026 | ✓* | NVFP4 | Forward-KL T=1, offline-logit, selective | Nemotron-49B AIME25 45.6 vs 46.0 BF16 |
| UPQ | `2506.09104` | 2025 | ✗ | INT2 | INT4→INT2 warm-start, SEQ, JSD(β=0.5) | L3.2-3B INT2 MMLU 53.2 (vs NTP-QAT 39.2) |
| LLM-QAT | `2305.17888` | 2024 | ✗ | 3–4+KV8 | Data-free self-gen + per-token KV quant | Strong @W3A8 |
| MoEQuant | `2505.03804` | 2025 | ✓ | W3–W4 | EBSS + AGQ `H=(X·c)Xᵀ` | +2.16 Mixtral; +10 HumanEval DeepSeek-MoE |
| MiLo | `2504.02658` | 2025 | ✓ | INT3 | HQQ+SVD compensators, kurtosis rank | Mixtral INT3 MMLU 67.69 vs 59.36 RTN |
| LQ-LoRA | `2311.12023` | 2023-24 | ✗ | 2.75–3 | W=W_q+AB joint decomp + Fisher ILP | 70B@2.75-bit 27GB, MMLU ~67% |
| Reasoning-QAT | `2601.14888` | 2026 | ✗ | 2–3 | GPTQ→fwd-KL→GRPO; domain-aligned calib | Qwen3-4B W2 MATH 4.8→78.27 |
| ParetoQ | `2502.02631` | 2025 | ✗ | 1–4 | Bit-specific grids; compensation vs recon | 3-bit within 1.8pt of FP16 |
| QaRL/TBPO | `2604.07853` | 2026 | ✓* | W4A16 | Rollout-aligned GEMM + seq-level dual clip | Qwen3-30B-A3B 51.2 vs 52.1 BF16 |
| Silver-Bullet | `2505.11574` | 2025 | ✗ | INT4 | First-error DPO, 332 pairs, LoRA-32 | Qwen2.5-3B MATH 29→68 in 3–5min |
| Super-Expert | `2507.23279` | 2025 | ✓ | protect | P99.5 down_proj activation detection | Prune 3/6144 → PPL 8.7→59.9 |
| MambaQuant | `2501.13484` | 2025 | ✗ | W4A8 | KLT+Hadamard variance-aligned rotation | Mamba W8A8 <1% drop |
| DQ3_K_M | `2505.02390` | 2025 | ✓ | 2.8–3.2 | Dynamic Q3/Q4/Q6, 1-in-4 ffn_down protect | DeepSeek-R1 671B 0.34% drop vs FP8 |
| XFP | `2605.14844` | 2026 | ✓ | 2–4 | Quality-floor adaptive Lloyd + 4σ residual | **Qwen3.5-122B 138 tok/s, 94.49% GSM8K** |
| SOAR | `2605.12245` | 2026 | ✗ | NVFP4 | CJSO closed-form scale + DSS decoupled | +3.26 GSM8K Qwen3-8B |
| RaZeR | `2501.04052` | 2025 | ✗ | NVFP4 | Reclaim redundant zero → special value ±5 | 34.6% PPL-loss reduction |
| MR-GPTQ | `2509.23202` | 2025 | ✗ | MX/NVFP4 | Block Hadamard + static reorder | 70B+ recover 98–99% FP16; QuTLASS 3.6× B200 |
| MXFP4-MBS/OAS | `2603.08713` | 2026 | ✗ | MXFP4 | Macro-block scaling + overflow-aware | Gap to NVFP4 10%→<1% |
| Four-over-Six | `2512.02010` | 2025 | ✗ | NVFP4 | Per-block absmax/4 vs /6 by MSE | 30% pretrain loss-gap reduction |
| PCDVQ | `2506.05432` | 2025 | ✗ | 2–2.125 | Polar decouple: E8 direction + Lloyd-Max mag | L2-7B 2-bit PPL 5.81 vs QuIP# 6.19 |
| VQ-LLM | `2503.02236` | 2025 | ✗ | sub-4 | Codebook-cache + fused decode-GEMM | 46% latency reduction |

\* MoE-relevant via hybrid/MoE evaluation though method is architecture-general.

---

## 6. Concrete Experiment Ideas (mapped to mechanisms)

**E1 — Protected-tensor ablation (1 forward pass to set up).** Profile down_proj max-activation per expert (Super-Expert `2507.23279`) and `ssm_out` KL contribution. Quantize everything to INT3, then sweep: (a) no protection, (b) +shared-expert FP16, (c) +super-experts FP16, (d) +`ssm_out` INT4+KLT. Measure ΔMMLU/GSM8K. *Hypothesis:* (d) is mandatory; expect catastrophic collapse without `ssm_out` protection. Cheap, run first.

**E2 — AGQ + EBSS vs vanilla GPTQ (low effort, high signal).** Hold allocation fixed at flat W3. Compare random-512-seq GPTQ vs EBSS-1024 + AGQ `H=(X·c)Xᵀ`. Track per-expert reconstruction error histogram. *Hypothesis:* cold experts' error drops ~19% (EAQuant EA-CDB number), HumanEval +several pts (MoEQuant DeepSeek-MoE >10pt).

**E3 — Composite sensitivity signal value-add.** Allocate to 3.0-bit with (a) frequency only, (b) router-norm only, (c) Hutchinson-Tr(H) only, (d) composite rank-fusion. Validate which best predicts true block-output distortion Δ on held-out calibration. *Hypothesis:* under load-balanced routing, frequency is worst; composite wins (MoPEQ 63/105, `2604.06515`).

**E4 — Block-level vs expert-level granularity.** Re-run E3's best signal at expert-level vs gate/up/down-independent ILP. *Hypothesis:* block-level gains ~0.21 PPL (MxMoE) — confirm it transfers to 256-expert scale.

**E5 — The QAD-vs-QAT crucial test.** From the same INT3 init, run (a) task-loss QAT, (b) forward-KL QAD T=1, (c) JSD(β=0.5), all at LR 1e-5 on 1B math+code tokens. Eval GPQA + AIME-style + LiveCodeBench. *Hypothesis (the whole program's thesis):* QAT degrades reasoning below PTQ; QAD/JSD recovers near-BF16 (NV-QAD `2601.20088`, Reasoning-QAT `2601.14888`).

**E6 — Token-budget scaling curve.** QAD at {0.3, 1, 3}B tokens; fit recovery vs budget. *Hypothesis:* 122B saturates ~1–2B (larger-model favorable scaling), justifying the offline-logit-cache investment.

**E7 — Progressive vs direct init.** Direct FP16→W2 QAD vs W4→W3→W2 nested warm-start (Bit-by-Bit `2604.07888`, UPQ `2506.09104`). *Hypothesis:* ~43% lower init error, no loss spikes, fewer tokens to converge.

**E8 — Adapter (Path B) vs block-wise (Path A) GPU-hour efficiency.** LQ-LoRA rank-64 QAD vs EfficientQAT block-AP+E2E-QP at matched token budget. Measure accuracy-per-GPU-hour. *Decision output:* which path to standardize on.

**E9 — Format bake-off on the real model.** NVFP4 (Four-over-Six→SOAR→RaZeR→AGQ-GPTQ) vs XFP (~3.97-bit adaptive Lloyd) vs Marlin-INT4 baseline — measure tok/s on 4×H100 *and* GSM8K/MMLU at matched memory. *Hypothesis:* XFP ≈ +49% decode (proven on Qwen3.5-122B), NVFP4 best accuracy/throughput via tensor cores.

**E10 — Silver-Bullet finisher.** After best QAD model, run first-error DPO on GPQA/GSM8K/HumanEval-specific failure sets (~500 pairs each). *Hypothesis:* closes 70–80% of residual gap in <1 GPU-h, biggest gains on multi-step arithmetic and Gated-DeltaNet error-accumulation chains.

**E11 — DynaExq serving recovery.** On the static-INT2/INT3 base, enable EMA hot-expert promotion (top-32/layer→INT4, hysteresis θ=20% mean). *Hypothesis:* +~4.5 pts vs static (Qwen3-80B precedent) at near-zero extra memory.

**E12 — Gated-DeltaNet KLT rotation isolation.** With/without KLT on `ssm_out`/`ssm_in` at W3A8. Track per-layer KL divergence during QAD as a diagnostic. *Hypothesis:* rotation is the difference between stable sub-4-bit and divergence on the 36 linear-attention layers (MambaQuant `2501.13484`).

---

*Note on a referenced ID:* several subtopic reports cite the MoE generalization-guarantee paper as `2604.06515` — treat the numeric form as reported by the upstream researchers; verify against arXiv listing before citing externally, as 26xx.xxxxx IDs correspond to 2026 submissions.