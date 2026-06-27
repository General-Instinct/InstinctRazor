# InstinctRazor

**Sub-4-bit quantization + on-policy recovery for any Hugging Face model.**

Point it at any model — InstinctRazor quantizes it sub-4-bit, then recovers the loss with on-policy
distillation. A new family is one `ModelAdapter`, not a fork.

```bash
./razor --model <any-hf-model> --quant instinct-iq3 --recover opd --eval mmlu_pro,gpqa,math500
```

- **Qwen3.5-122B-A10B** → 47 GB GGUF, runs on one 80 GB GPU
- **Qwen3.6-35B + OPD** → MATH-500 81.7 → 89.2, at the same footprint

[GGUF release on Hugging Face](https://huggingface.co/General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF). Worked examples below.

## Features

- **Any model** — fused-expert MoE, separate-expert MoE, and dense layouts (shape-agnostic quant math)
- **Quantize** — INT3 experts / INT4 backbone / BF16 protected (router, norms, vision); group-128 clip-search, no calibration data
- **Recover (OPD)** — distill the BF16 teacher into a per-expert LoRA, re-quantize; footprint unchanged
- **Compare** — AWQ / GPTQ / RTN / ParoQuant at matched bits
- **Deploy** — llama.cpp GGUF, or a dequantized-BF16 checkpoint for clean eval
- **Eval** — MMLU-Pro, GPQA, MMMLU, MATH-500, AIME, LiveCodeBench, HumanEval, MBPP, HLE, multimodal

## Install

```bash
python3.12 -m venv vllm_venv && source vllm_venv/bin/activate && pip install -r requirements.txt
MOE_LOWBIT_VENV=$PWD/vllm_venv source env.sh
```

That core install is enough to **quantize and evaluate**. Two extra pieces, each needed only for one feature:

**llama.cpp** — the tool that writes `.gguf` files (used by both `razor --quant <gguf-type>` and the deploy
script). Skip it if you only do `--recipe ... --no-gguf` eval. Build once (CPU build is enough):

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=OFF && cmake --build build -j --target llama-quantize llama-server
export LLAMA_CPP=$PWD
```

**train venv** — only for OPD recovery (`--recover opd`); it needs a different torch:

```bash
python3.12 -m venv train_venv && train_venv/bin/pip install -r requirements-train.txt
```

## Usage

```bash
./razor --model Qwen/Qwen3.6-35B-A3B --quant instinct-iq3 --out runs/q36
./razor --model meta-llama/Llama-3.1-8B-Instruct --quant Q4_K_M --eval mmlu_pro,gpqa
./razor --model Qwen/Qwen3.6-35B-A3B --recover opd --student models/q36_ptq3b_clip
./razor --model Qwen/Qwen3-0.6B --quant Q4_K_M --eval mmlu_pro --eval-n 16 --dry-run
```

| Flag | Meaning |
|------|---------|
| `--quant` | GGUF type (`Q4_K_M`, `IQ3_XXS`, …) or InstinctRazor recipe (`instinct-q3`, `instinct-iq3`) → `.gguf` |
| `--recipe` | `clip` / `awq` / `gptq` / `rtn` → dequant-BF16 eval checkpoint (no GGUF) |
| `--recover opd` | recover a quantized `--student` via OPD (`--recover-smoke` = FSDP smoke only) |
| `--eval` | comma-separated benchmarks; `--budget 32k\|64k`, `--eval-n` to cap samples |
| `--dry-run` | print the plan, run nothing |

## Supported models

Same pipeline for all; one adapter (keyed on `config.model_type`) is the only per-model code. Coverage is validated against transformers 5.9 in `tests/test_adapter_coverage.py`.

| Family | `model_type` | Quantize | OPD |
|------|------|:---:|:---:|
| Qwen3.5 / 3.6 MoE | `qwen3_5_moe` | ✅ | ✅ |
| Fused MoE | `mixtral`, `qwen2_moe`, `qwen3_moe`, `olmoe`, `qwen3_next`, `deepseek_v3` | ✅ | ✅ |
| Fused MoE (quant-only) | `deepseek_v2`, `phimoe`, `gpt_oss`, `minimax`, `jamba` | ✅ | hook* |
| Atypical MoE | `dbrx`, `granitemoe` | ✅ | hook* |
| Dense | `llama`, `mistral`, `qwen2`, `qwen3`, `gemma2`, `gemma3`, `phi3` | ✅ | — |
| Anything else | generic fallback | ✅ | hook* |

\* Quantize is universal — transformers 5.x batches every MoE family into fused expert tensors, and the gate/router is auto-protected. OPD recovery works for any fused MoE whose experts share the reference forward (`gate_up_proj`/`down_proj`, `chunk(2)`, routing-in-parent) — validated on `qwen3_moe`. Families with a different expert math (e.g. `gpt_oss`: bias + clamp + interleaved GLU) need a small per-family expert-compute hook; the FSDP training, recipe, merge, and eval are all shared.

## Quantization recipe

The default recipe applied to any model:

| Experts | Backbone | Group | Method | Protected (BF16) | Effective bits |
|------|------|------|------|------|------|
| INT3 | INT4 | 128 | symmetric clip-search | router, norms, vision | ~3.05 |

## Examples

### Quantize — Qwen3.5-122B-A10B

InstinctRazor (~47 GB) vs. footprint-matched Gemma-4-26B-A4B and the BF16 teacher.

| Benchmark | Teacher | InstinctRazor | Gemma-4-26B-A4B |
|------|------|------|------|
| MMLU-Pro | 87.6 | **88.5** | 85.6 |
| GPQA-Diamond | 83.8 | **84.8** | 79.3 |
| MMMLU | 88.8 | **87.2** | 85.4 |
| MMMU-Pro | — | **80.8** | 73.8 |
| LiveCodeBench v6 | 65.5 | 57.0 | 66.0 |
| HLE (no tools) | 18.0 | 13.3 | 12.3 |

Pack the deployable GGUF (reproduces the shipped 122B IQ3_XXS protected artifact — tensor-type recipe + imatrix; needs llama.cpp from [Install](#install)):

```bash
BASE_TYPE=IQ3_XXS bash pipelines/pack_gguf.sh
```

122B IQ3_XXS (48 GiB): MMLU-Pro 90.7, GPQA 80.8, 115.9 tok/s on 1×H100 (45.7 with expert offload).

### Recover — Qwen3.6-35B-A3B

OPD on the 3-bit student (matched 32k eval). The gap is mostly *truncation*; distilling completed teacher trajectories recovers it at no extra footprint.

| | GPQA acc | trunc | MATH-500 acc | trunc |
|------|------|------|------|------|
| baseline (3b) | 68.7 | 64/198 | 81.7 | 19/120 |
| **+ OPD** | **77.3** | **30/198** | **89.2** | **1/120** |
| teacher (BF16) | 84.3 | 11/198 | 90.0 | 1/120 |

## Citation

```bibtex
@software{instinctrazor2026,
  title={InstinctRazor}, author={General Instinct}, year={2026},
  url={https://github.com/General-Instinct/InstinctRazor}
}
```
