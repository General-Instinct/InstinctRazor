# InstinctRazor

**Bring any model. Quantize it sub-4-bit. Recover the loss with on-policy distillation.**

InstinctRazor is a model-general framework: point it at a Hugging Face checkpoint and it runs the **same two
stages** on it — **quantize** to a deployment-ready sub-4-bit artifact, then **recover** the quantization
loss via on-policy distillation (OPD) from the BF16 teacher — leaving the footprint unchanged. Both stages
are driven by one `ModelAdapter` abstraction (keyed on `config.model_type`), so supporting a new family is a
small adapter, never a fork of the pipeline.

```bash
./razor --model <any-hf-model> --quant instinct-iq3 --recover opd --eval mmlu_pro,gpqa,math500
```

Any model in → quantized + OPD-recovered → benchmarked, at the same deployment footprint. The quantizer is
model-agnostic across MoE layouts and dense models alike; the OPD recovery stage is wired first-class for
fused-expert MoE (Qwen3.5 / Qwen3.6 today) and extends to other families through the same adapter interface.

Two results from the framework:
- **Quantize** — Qwen3.5-122B-A10B compressed from ~245 GB to a ~47 GB deployable GGUF that fits on one 80 GB GPU.
- **Recover** — on Qwen3.6-35B-A3B, OPD took the 3-bit student's **MATH-500 81.7 → 89.2** (truncation 19 → 1, matching the teacher) and **GPQA 68.7 → 77.3**, at no extra deployment size.

- Hugging Face: https://huggingface.co/General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF
- Deployable artifact: 48 GB IQ3_XXS GGUF

---

## What the framework does

The two core stages run on **any model you bring in**; the rest support them.

| Stage | What it does |
|-------|-----------|
| **① Quantize** | Any HF model → sub-4-bit: INT3 routed experts + INT4 backbone + BF16 protected path (router / norms / vision). Symmetric per-block (group-128) clip-search; no calibration data. Model-general across fused-expert MoE, separate-expert MoE, and dense layouts (the quant math is shape-agnostic). |
| **② Recover (OPD)** | Quantize, then close the gap: distill the BF16 teacher into a footprint-preserving per-expert LoRA on the quantized student and re-quantize — same deploy size, capability back toward the teacher. FSDP2, 4-GPU; recipe = on-policy rollouts → teacher top-k logprobs → reverse-KL train on *completed* trajectories. |
| Generalize | One adapter per family (`ModelAdapter`, keyed on `config.model_type`) wires both stages — `qwen3_5_moe`, `olmoe`, generic fallback. Adding a model is an adapter, not a fork. |
| Compare | Benchmark our quantization vs AWQ / GPTQ / RTN / ParoQuant at matched bits (`moe_compare.py`, `moe_quant_method.py`). |
| Deploy | llama.cpp GGUF for real low-bit inference, or a dequantized-BF16 *capability-ceiling* checkpoint for clean vLLM eval. |
| Evaluate | Built-in harness: MMLU-Pro, GPQA-Diamond, MMMLU, MATH-500, AIME, LiveCodeBench, HumanEval, MBPP, HLE, and multimodal (MMMU / MMMU-Pro / MATH-Vision). |

---

## Unified CLI (`razor`)

One command takes a **Hugging Face model → quantized `.gguf` → (optional) benchmarked eval**, wrapping the
whole framework (download, the model-general quantizer, llama.cpp convert+quantize, eval harnesses, and the
recovery pipeline) behind a single entrypoint.

```bash
./razor --model Qwen/Qwen3.6-35B-A3B --quant instinct-iq3 --out runs/q36

./razor --model meta-llama/Llama-3.1-8B-Instruct --quant Q4_K_M --eval mmlu_pro,gpqa --budget 32k

./razor --model Qwen/Qwen3-0.6B --quant Q4_K_M --eval mmlu_pro --eval-n 16 --eval-budget 2048

./razor --model Qwen/Qwen3.6-35B-A3B --recipe awq --expert-bits 3 --no-gguf --eval mmlu_pro,gpqa,math500

./razor --model Qwen/Qwen3.6-35B-A3B --recover opd --student models/q36_ptq3b_clip --recover-smoke
```

| Flag | Meaning |
|------|---------|
| `--model` | HF repo id or local path |
| `--quant` | GGUF type (`Q4_K_M`, `IQ3_XXS`, `Q6_K`, …) **or** an InstinctRazor protected recipe (`instinct-q3`, `instinct-iq3`) → emits a `.gguf` |
| `--recipe` | `clip` / `awq` / `gptq` / `rtn` → our PTQ to a dequant-bf16 *capability-ceiling* checkpoint (research/eval) |
| `--recover opd` | recover a quantized `--student` via on-policy distillation from the BF16 teacher (`--recover-smoke` runs just the FSDP blocker test). Needs the train venv (`requirements-train.txt`) |
| `--eval` | comma-separated benchmarks (e.g. `mmlu_pro,gpqa,math500`); omit to skip eval. GGUF → llama.cpp harness, bf16 ckpt → vLLM harness |
| `--budget` | `32k` (knowledge/reasoning) or `64k` (code/math) thinking budget; `--eval-n` / `--eval-budget` for quick runs |
| `--dry-run` | print the full plan, run nothing |

**Prereq for `.gguf`:** a llama.cpp build supporting the model's arch (CPU build is enough to quantize):
`git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build -DGGML_CUDA=OFF && cmake --build build -j --target llama-quantize llama-server`, then `export LLAMA_CPP=$PWD` (or place it at `./llama.cpp`).

---

## Supported models (`ModelAdapter`)

Every model flows through the **same pipeline** — `quantize` for all, `OPD recover` through the same adapter
interface. One adapter per family (keyed on `config.model_type`) is the only per-model code; the pipeline
itself never forks.

| `model_type` | Adapter | Quantize | OPD recover | Examples |
|--------------|---------|:---:|:---:|----------|
| `qwen3_5_moe` | `Qwen35MoeAdapter` | ✅ | ✅ | Qwen3.5-122B-A10B, Qwen3.6-35B-A3B |
| `olmoe` | `OlmoeAdapter` | ✅ | adapter hook* | OLMoE-1B-7B |
| *(fallback)* | `GenericMoEAdapter` | ✅ | adapter hook* | any unrecognized MoE / dense model |

\* Quantization is universal today. OPD is first-class for fused-expert MoE (Qwen3.5/3.6); enabling it for a
new family means implementing that adapter's expert-forward hook (`supports_opd` + the per-expert LoRA path)
— the FSDP training, the distillation recipe, merge, and eval are all shared and model-independent.

The quant math (`_int_perblock` / `fakequant` / `ste_quant` / `apply_ptq`) is shape-agnostic — it groups along
the contraction axis and works identically on 2D `[out,in]` and fused 3D `[E,out,in]` tensors, which is what
makes "any model in" hold for the quantize stage.

---

## Quantization configuration

| Component | Quantization |
|------------|------------|
| Routed experts | INT3 |
| Backbone linear layers | INT4 |
| Group size | 128 |
| Method | Symmetric MSE clip-search (calibration-free) |
| Router / gates | BF16 |
| Norms | BF16 |
| Vision tower | BF16 |
| Effective bits | ~3.05 |

The repository also includes expert-salience analysis and adaptive bit allocation, but in practice uniform
3-bit expert quantization retained downstream performance, so the default recipe stays uniform.

---

## Results

Flagship: **Qwen3.5-122B-A10B**, InstinctRazor (~47 GB) vs. the footprint-matched Gemma-4-26B-A4B and the
BF16 teacher.

| Benchmark | BF16 Teacher | InstinctRazor (~47 GB) | Gemma-4-26B-A4B (~52 GB) |
|-----------|-------------|------------------------|--------------------------|
| MMLU-Pro | 87.6 | **88.5** | 85.6 |
| GPQA-Diamond | 83.8 | **84.8** | 79.3 |
| MMMLU | 88.8 | **87.2** | 85.4 |
| MMMU-Pro | — | **80.8** | 73.8 |
| LiveCodeBench v6 | 65.5 | 57.0 | 66.0 |
| MATH-Vision | — | 70.0 | 82.4 |
| HLE (no tools) | 18.0 | 13.3 | 12.3 |

Most remaining gaps are concentrated in long-form code/math generation, where quantization increases
**truncation** rates rather than degrading capability — which is what the recovery stage targets.
Generalization to **Qwen3.6-35B-A3B** and the cross-method comparison are documented in
`docs/MOE_FINDINGS.md` and `results/RESULTS.md`. Evaluation methodology: `docs/EVAL_PROTOCOL.md`.

---

## Pipelines (manual)

The `razor` CLI orchestrates these; they can also be run directly.

```text
quantize : src/quant/quant_save.py    HF model        -> dequant-BF16 eval ckpt (3b experts / 4b backbone)
gguf     : llama.cpp convert + quantize  ckpt/weights -> deployable .gguf
eval     : src/eval/vllm_eval8.py     ckpt|gguf       -> benchmark JSON (acc / acc_finished / trunc)
recover  : pipelines/q36_recovery.sh  gen -> score -> FSDP train -> merge -> re-quantize -> eval
```

### Smoke test

```bash
python3.12 -m venv vllm_venv && source vllm_venv/bin/activate && pip install -r requirements.txt
MOE_LOWBIT_VENV=$PWD/vllm_venv source env.sh
bash pipelines/smoke.sh
bash pipelines/smoke_moe.sh
```

`smoke.sh` prints `SMOKE OK`; `smoke_moe.sh` runs the separate-expert (OLMoE) quant-sanity check.

### Reproduce the 122B result

```bash
HF_TOKEN=hf_xxx MOE_LOWBIT_VENV=$PWD/vllm_venv source env.sh
bash pipelines/quantize.sh
bash pipelines/eval.sh --model models/q122_ptq3b_clip --tag clip --benchmarks mmlu_pro,gpqa,mmmlu
```

Results are written to `results/vllm_eval/`.

---

## Recovery (on-policy distillation)

For code/math workloads where quantization raises truncation, InstinctRazor includes an optional
distillation stage that distills the BF16 teacher into a footprint-preserving per-expert LoRA, then
re-quantizes — leaving the deployment size unchanged.

```text
generate (student or teacher rollouts) -> score (teacher top-k logprobs)
   -> FSDP2 train (per-expert LoRA, reverse-KL) -> merge + re-quantize -> re-eval
```

The training stage runs **FSDP2 across 4 GPUs** with the frozen base GPU-resident and the per-expert LoRA
replicated (FSDP-ignored); the expert body is checkpointed so the dominant (sequence-independent)
reconstructed-weight memory is recomputed on backward — ~31 GB/GPU at 95% utilization for a 35B-A3B model.
See `docs/OPD_INTEGRATION.md` for the design, the FSDP details, and current findings (including the
on-policy-vs-teacher-CoT trade-off in truncation recovery).

```bash
bash pipelines/distill.sh --smoke
bash pipelines/distill.sh
```

`--smoke` runs the 2-step FSDP blocker check; without it, the full gen → score → train → merge runs.

---

## GGUF deployment

Evaluation checkpoints are saved as dequantized BF16 after quantization-aware processing, so benchmarks
measure capability without kernel-specific low-bit effects. Deployment uses a separate GGUF path built from
the original BF16 weights.

```bash
BASE_TYPE=IQ3_XXS bash pipelines/pack_gguf.sh
```

Produces `q122_orig-IQ3XXS-protected.gguf`.

| Metric | Value (122B IQ3_XXS, 48.04 GiB) |
|----------|---------|
| MMLU-Pro | 90.7 |
| GPQA-Diamond | 80.8 |
| Decode speed (1× H100-80GB) | 115.9 tok/s |
| Decode speed (expert offload) | 45.7 tok/s |

---

## Repository layout

```text
InstinctRazor/
├── razor / razor.py        unified CLI
├── env.sh                  PYTHONPATH for src/{quant,eval,distill}
├── requirements.txt        quant / eval venv (torch + vLLM)
├── requirements-train.txt  FSDP train venv (torch 2.7 + flash-linear-attention)
├── src/
│   ├── quant/    model_adapters, moe_quant, quant_save, moe_probe/alloc/study, moe_compare
│   ├── eval/     vllm_eval8, bench8_loaders, multimodal eval
│   └── distill/  opd_gen, opd_score, opd_train_fsdp, moe_lora, merge_adapter
├── pipelines/    smoke, quantize, eval, pack_gguf, distill, q36_recovery
├── configs/  docs/  results/  archive/
```

---

## Hardware

Primary experiments: 4× NVIDIA H100 80 GB, CUDA 13.0, NVLink, ~885 GB system RAM. The compressed 122B
runs on a single 80 GB GPU (or smaller with expert offload); the smoke tests run on far less.

---

## Documentation

| Document | Description |
|-----------|-------------|
| `docs/EVAL_PROTOCOL.md` | Evaluation methodology + validation gate |
| `docs/MOE_FINDINGS.md` | Quantization study, cross-method comparison, generalization |
| `docs/MOE_PIPELINE.md` | End-to-end quantization pipeline |
| `docs/OPD_INTEGRATION.md` | Distillation/recovery workflow + FSDP details |
| `results/RESULTS.md` | Benchmark provenance |

---

## Citation

```bibtex
@software{instinctrazor2026,
  title={InstinctRazor},
  author={General Instinct},
  year={2026},
  url={https://github.com/General-Instinct/InstinctRazor}
}
```
