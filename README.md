# InstinctRazor

**Sub-4-bit quantization of a frontier MoE — and a reproducible way to prove it didn't lose anything that matters.**

> 📦 **Ready-to-run weights:** the deployable **48 GB IQ3_XXS GGUF** (runs on one 80 GB GPU, or a small card + CPU expert-offload) is on the Hugging Face Hub — **[General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF](https://huggingface.co/General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF)**. The framework in this repo reproduces it from the original BF16. *(The Python package/module namespace is `moe-lowbit`.)*

We take **Qwen3.5-122B-A10B** (a 122B hybrid Gated-DeltaNet MoE, ~245 GB BF16) down to a **~47 GB**
3-bit-expert / 4-bit-backbone checkpoint (`q122_ptq3b_clip`), and show that on a shared, validation-gated
harness it **beats the footprint-matched Gemma-4-26B-A4B** (~52 GB) on knowledge, reasoning, multilingual,
and multimodal-MMMU — tracking the uncompressed BF16 teacher within ~1 point, **with no training**. Where
it doesn't win (LiveCodeBench v6, MATH-Vision — both *truncation*-driven, not capability loss), there's an
optional **Lightning-OPD** on-policy-distillation path.

> The headline isn't just "we compressed it." It's: a 122B MoE *inverts* the dense low-bit regime — at ~3
> effective bits, PTQ is near-lossless because only 8/256 experts fire per token and the ~6% always-on path
> is protected. A 122B model that needed 4×80 GB now fits on **one 80 GB GPU** and out-scores a same-size dense-ish MoE.

## Headline result

~47 GB clip vs the ~52 GB A4B baseline, same harness (vLLM 0.22, TP=4, thinking mode, seed 0). Full
provenance + caveats in [`results/RESULTS.md`](results/RESULTS.md).

| Benchmark | teacher BF16 | **clip (~47 GB)** | A4B (~52 GB) | verdict |
|-----------|-------------|-------------------|--------------|---------|
| MMLU-Pro | 87.6 | **88.5** | 85.6 | ✅ clip ≥ A4B |
| GPQA-Diamond | 83.8 | **84.8** | 79.3 | ✅ clip ≥ A4B |
| MMMLU | 88.8 | **87.2** | 85.4 | ✅ clip ≥ A4B |
| MMMU-Pro | — | **80.8** | 73.8† | ✅ clip ≥ A4B |
| LiveCodeBench v6 | 65.5 | 57.0 (75.7 finished) | 66.0 | ⚠️ recoverable gap (clip truncates 30%) |
| MATH-Vision | — | 70.0 → 77.5 hi-budget | 82.4† | ⚠️ recoverable gap (truncation) |
| HLE (no-tools) | 18.0 | 13.3 | 12.3 | ✅ clip ≥ A4B (below teacher) |
| τ²-Bench | — | — | — | ⏳ no in-tree harness |

†A4B official figure (a same-harness A4B multimodal run was not done). All comparisons are subject to the
**validation gate** — see caveats below and [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md).

## Quickstart

### A. Verify the whole framework on ONE GPU (~15 min, no 122B, no 4×H100)

Runs the *exact same* load → PTQ → dequant-save → vLLM-eval code path on a small dense model:

```bash
python3.12 -m venv vllm_venv && source vllm_venv/bin/activate && pip install -r requirements.txt
MOE_LOWBIT_VENV=$PWD/vllm_venv source env.sh
bash pipelines/smoke.sh        # quantizes Qwen3.5-2B to 4-bit, evals it; prints "SMOKE OK"
```

### B. Reproduce the headline (4×H100-80GB)

Three commands: set up env → build the 47 GB clip from BF16 → eval clip and the A4B baseline.

```bash
# 0. env (once): create the eval venv, then source env.sh (sets PYTHONPATH, HF token, caches)
python3.12 -m venv vllm_venv && source vllm_venv/bin/activate && pip install -r requirements.txt
HF_TOKEN=hf_xxx MOE_LOWBIT_VENV=$PWD/vllm_venv source env.sh

# 1. BF16 -> ~47 GB clip  (recipe in configs/clip_122b.env)
bash pipelines/quantize.sh

# 2. eval clip vs the footprint-matched A4B on the gated axes
bash pipelines/eval.sh --model models/q122_ptq3b_clip --tag clip --benchmarks mmlu_pro,gpqa,mmmlu
bash pipelines/eval.sh --model google/gemma-4-26b-a4b-it --tag a4b  --benchmarks mmlu_pro,gpqa,mmmlu
```

Then read `results/vllm_eval/{clip,a4b}.json` (or regenerate `results/RESULTS.md`). For the long-context
axes add `--budget 64k` (math/code/HLE). **Optional on-policy distillation** for the LCB code gap:

```bash
bash pipelines/distill.sh --smoke   # validate the 4-GPU FSDP path first
bash pipelines/distill.sh           # gen -> score -> FSDP train -> merge+requant -> ~47 GB
```

## Repo layout

```
InstinctRazor/           # (Python module namespace: moe-lowbit)
  env.sh                 # source first: PYTHONPATH(quant:eval:distill) + venv + HF/caches (overridable)
  requirements.txt       # EVAL/quantize venv (torch 2.11, vllm 0.22, transformers 5.9)
  requirements-train.txt # OPD FSDP train venv (torch 2.7 + flash-linear-attention) — separate by necessity
  src/
    quant/   moe_quant.py (fake-quant/STE/clip-search) · quant_save.py (the builder) ·
             moe_probe.py + moe_alloc.py + moe_study.py (the salience research path) · moe_ptq.py
    eval/    vllm_eval8.py (text harness) · mm_eval.py (multimodal) · vllm_eval.py · bench8_loaders.py ·
             moe_eval.py · bench_mm.py
    distill/ opd_gen.py · opd_score.py · opd_train_fsdp.py · merge_adapter.py · moe_lora.py ·
             cache_teacher.py · fsdp_setup.py (reference; live logic is inline in opd_train_fsdp.py)
  pipelines/ smoke.sh · quantize.sh · eval.sh · distill.sh
  configs/   clip_122b.env · smoke_2b.env · eval.env · opd_r1.env (+ README explaining each)
  results/   the result JSONs + RESULTS.md (each number -> its source JSON + command)
  docs/      EVAL_PROTOCOL.md + MOE_FINDINGS/MOE_PIPELINE/LADDER_RESULTS/OPD_INTEGRATION/POSITIONING/SURVEY_MOE
  archive/   quarantined dead-ends (device_map OPD, EP/FSDP smokes, diagnostics) — kept, marked deprecated
```

**Note on imports:** the modules cross-import by bare name (`import moe_quant`, `import vllm_eval`, …),
so `env.sh` puts all three `src/` subdirs on `PYTHONPATH`. That is the *only* packaging change — no source
logic was rewritten.

## The recipe (what actually ships)

```
expert_bits = 3.0   (int3, all 256 routed experts, both gate_up + down blocks)
linear_bits = 4.0   (int4, all non-expert nn.Linear + embed/lm_head)
group = 128 · per-block symmetric · MSE clip-search 16 steps
protected @ bf16:  vision tower · router/gate · all norms
=>  ~3.05 effective bits  ~=  ~47 GB packed  (under A4B's ~52 GB)
```

This is **uniform** quantization, deliberately. We built per-expert salience allocation
(`moe_probe`+`moe_alloc`+`moe_study`) and the study (finding **F4/F6**, `docs/MOE_FINDINGS.md`) showed that
on a load-balanced router (0/256 cold experts) allocation is a real lever for *distribution fidelity* but
**second-order for downstream capability** — uniform 3b already lands near-lossless. So the probe/alloc path
is the *research that justified uniform*, run optionally via `PROBE=1 bash pipelines/quantize.sh`.

## Deployment: GGUF/llama.cpp (the packed, single-GPU-runnable form) — VALIDATED

The fake-quant checkpoint above is the **capability ceiling** (what `eval.sh` measures), stored dequantized.
The **shipped deployable artifact** is a real low-bit GGUF, quantized **from the original BF16** (not the
dequant clip — that would double-quantize) with an **imatrix**, via llama.cpp (`qwen3_5_moe` support merged
upstream, PR #19468):

**Don't want to rebuild? Download it.** The built artifact (+ the `mmproj` vision projector) is on the Hub:
[**General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF**](https://huggingface.co/General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF).
To rebuild it from the original BF16 instead:

```bash
BASE_TYPE=IQ3_XXS bash pipelines/pack_gguf.sh   # convert orig BF16 → imatrix → IQ3_XXS protected
```

**Validated artifact** `q122_orig-IQ3XXS-protected.gguf` — **48.04 GiB** (IQ3_XXS 3.06 bpw experts; protected
path shared int8 / attn int4 / router+SSM f16 / embed+lm_head q8_0; recipe `configs/gguf_tensor_types_iq3.txt`):

| metric | value | vs A4B | vs fake-quant clip |
|--------|-------|--------|--------------------|
| MMLU-Pro | **90.7** (n=150, 0 trunc) | ≥ 85.6 ✅ | tracks (88.5–90) |
| GPQA-D | **80.8** (n=198, 0 trunc) | ≥ 79.3 ✅ | ~4 pt under (84.8) |
| decode tok/s, 1×80 GB H100 | **115.9** (prefill 2541) | — | — |
| decode tok/s, expert-offload (peak ~7.6 GiB VRAM, fits 8 GB to 8k ctx) | **45.7 ± 0.5** (prefill ~154) | — | runs on an 8 GB card + ~48 GiB RAM |

The deployable GGUF **preserves the win** (≥ A4B on both axes) and **runs on one 80 GB GPU at ~116 tok/s** (or
a small card + CPU at ~47 tok/s with all routed experts offloaded). GPQA is ~4 pt below the fake-quant ceiling
— an honest, small i-quant loss, not a collapse. Accuracy is measured by `src/eval/llamacpp_eval8.py`
(llama-server + our exact prompt-build/grading — vLLM can't load this arch's GGUF). Pipeline notes baked in:
`--no-mtp` (no `mtp.*` tensors despite config); `#` comments stripped before `--tensor-type-file`; vision tower
→ separate `*-mmproj-f16.gguf`.

*Superseded fallback:* an earlier `q122_clip-Q3K-protected.gguf` (57 GB) was built from the **dequant clip**
(double-quantized, Q3_K 3.4 bpw, **no imatrix**). The IQ3_XXS-from-original build above replaces it as the
shipped artifact.

## Hardware

- **4× NVIDIA H100-80 GB** (NVLink), driver 580.105.08, CUDA 13.0, ~885 GB CPU RAM. Quantize + eval use
  `device_map="auto"` / vLLM TP=4; OPD training uses FSDP2 across all 4 GPUs via `torchrun`.
- The **packed ~47 GB** recipe is designed to run on a **single 80 GB GPU**; the BF16 teacher needs 4.
- The one-GPU smoke (path A) runs on any ≥16 GB GPU.

## Methodology & roadmap

- **Eval numbers are weight-only fake-quant — a capability *ceiling*, measured at full fidelity.**
  `quant_save.py` bakes the 3b/4b quantization, then saves a *dequantized BF16* checkpoint so vLLM measures
  capability exactly (no low-bit eval kernel). The **deployable artifact is the GGUF/llama.cpp pack**
  (`docs/MOE_PIPELINE.md`) — the ~47 GB on-disk, single-GPU-runnable form, and the one
  [published on the Hub](https://huggingface.co/General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF).
- **Same-harness deltas are the comparison.** Some axes (e.g. LiveCodeBench v6 extraction) read below their
  official absolutes for *all* models; the gated quantity is the same-harness *delta* vs A4B, not the
  absolute. See `docs/EVAL_PROTOCOL.md` for the validation gate.
- **Remaining gaps are an OPD target, not a wall.** Where the clip still trails A4B — code (LCB v6) and
  math / multimodal-math — the loss is largely token-inefficiency introduced by quantization. We close it
  with **OPD (on-policy distillation)**, a separate framework we'll open-source later; `pipelines/distill.sh`
  is the in-tree reference path (gen → score → FSDP train → merge + requant), trained on the gap's own
  domain (e.g. code CoT for LCB).

## Where the numbers come from

Every value in `results/RESULTS.md` maps to a JSON in `results/` and the `eval.sh` command that
produced it. Start with [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) (the validation gate), then
[`docs/MOE_FINDINGS.md`](docs/MOE_FINDINGS.md) (why the recipe is what it is).
