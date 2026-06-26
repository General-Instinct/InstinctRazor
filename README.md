# InstinctRazor

**A framework for sub-4-bit quantization — and recovery — of Mixture-of-Experts LLMs.**

InstinctRazor takes any Hugging Face MoE model and produces a deployment-ready low-bit artifact whose
capability stays close to the BF16 teacher. It is **model-general** (a `ModelAdapter` registry keyed on
`config.model_type`), **method-agnostic** (our calibration-free clip-search, plus AWQ / GPTQ / RTN for
comparison), and ships an **evaluation harness** and an optional **on-policy distillation** recovery stage —
all behind one command:

```bash
./razor --model Qwen/Qwen3.6-35B-A3B --quant instinct-iq3 --eval mmlu_pro,gpqa,math500
#        HF model  ───────────────►  quantize  ───────►  .gguf  ───────►  benchmarked
```

The flagship artifact — **Qwen3.5-122B-A10B compressed from ~245 GB to a ~47 GB deployable GGUF** that fits
on a single 80 GB GPU — is one output of this framework, reproducible from the original BF16 checkpoint.

- Hugging Face: https://huggingface.co/General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF
- Deployable artifact: 48 GB IQ3_XXS GGUF

---

## What the framework does

| Stage | Capability |
|-------|-----------|
| **Quantize** | INT3 routed experts + INT4 backbone + BF16 protected path (router / norms / vision). Symmetric per-block (group-128) clip-search; no calibration data required. |
| **Generalize** | One quant/probe/eval core works across MoE families via `ModelAdapter` (`src/quant/model_adapters.py`), keyed on `config.model_type` — fused-expert (Qwen3.5/3.6) and separate-expert (OLMoE, generic) layouts. |
| **Compare** | Drop-in AWQ / GPTQ / RTN / ParoQuant paths to benchmark our quantization against the field at matched bits (`src/quant/moe_compare.py`, `moe_quant_method.py`). |
| **Deploy** | llama.cpp GGUF conversion + quantization for real low-bit inference, or a dequantized-BF16 *capability-ceiling* checkpoint for clean vLLM evaluation. |
| **Evaluate** | Built-in harness: MMLU-Pro, GPQA-Diamond, MMMLU, MATH-500, AIME, LiveCodeBench, HumanEval, MBPP, HLE, and multimodal (MMMU / MMMU-Pro / MATH-Vision). |
| **Recover** *(optional)* | On-policy distillation (Lightning-OPD): close quantization gaps by distilling the BF16 teacher into a footprint-preserving per-expert LoRA, then re-quantizing. FSDP2, 4-GPU. |

---

## Unified CLI (`razor`)

One command takes a **Hugging Face model → quantized `.gguf` → (optional) benchmarked eval**, wrapping the
whole framework (download, the model-general quantizer, llama.cpp convert+quantize, eval harnesses, and the
recovery pipeline) behind a single entrypoint.

```bash
# HF model -> deployable 3-bit GGUF (MoE-aware "InstinctRazor" protected recipe)
./razor --model Qwen/Qwen3.6-35B-A3B --quant instinct-iq3 --out runs/q36

# HF model -> standard llama.cpp 4-bit GGUF, then eval on two benchmarks
./razor --model meta-llama/Llama-3.1-8B-Instruct --quant Q4_K_M \
        --eval mmlu_pro,gpqa --budget 32k

# Quick eval (cap samples) — handy for a smoke
./razor --model Qwen/Qwen3-0.6B --quant Q4_K_M --eval mmlu_pro --eval-n 16 --eval-budget 2048

# Research path: our fake-quant capability-ceiling checkpoint (no GGUF), eval in vLLM
./razor --model Qwen/Qwen3.6-35B-A3B --recipe awq --expert-bits 3 --no-gguf \
        --eval mmlu_pro,gpqa,math500

# Recovery: on-policy distillation of a quantized student from the BF16 teacher (truncation/accuracy recovery)
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

The quant/probe/alloc/eval core is decoupled from model-specific naming via a registry keyed on
`config.model_type`. Adding a new MoE family is a single adapter, not a fork of the pipeline.

| `model_type` | Adapter | Expert layout | OPD recovery | Examples |
|--------------|---------|---------------|:---:|----------|
| `qwen3_5_moe` | `Qwen35MoeAdapter` | fused 3D | ✅ | Qwen3.5-122B-A10B, Qwen3.6-35B-A3B |
| `olmoe` | `OlmoeAdapter` | fused 3D | — | OLMoE-1B-7B |
| *(fallback)* | `GenericMoEAdapter` | separate per-expert `nn.Linear` | — | unrecognized MoE (degrades gracefully) |

The quant math (`_int_perblock` / `fakequant` / `ste_quant` / `apply_ptq`) is shape-agnostic — it groups along
the contraction axis and works identically on 2D `[out,in]` and fused 3D `[E,out,in]` expert tensors.

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
bash pipelines/smoke.sh           # expect: SMOKE OK
bash pipelines/smoke_moe.sh       # separate-expert (OLMoE) quant-sanity
```

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
bash pipelines/distill.sh --smoke    # FSDP blocker smoke (prints SMOKE OK)
bash pipelines/distill.sh            # full gen -> score -> train -> merge
```

---

## GGUF deployment

Evaluation checkpoints are saved as dequantized BF16 after quantization-aware processing, so benchmarks
measure capability without kernel-specific low-bit effects. Deployment uses a separate GGUF path built from
the original BF16 weights.

```bash
BASE_TYPE=IQ3_XXS bash pipelines/pack_gguf.sh        # -> q122_orig-IQ3XXS-protected.gguf
```

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
├── razor / razor.py        # unified CLI
├── env.sh                  # puts src/{quant,eval,distill} on PYTHONPATH (bare imports)
├── requirements.txt        # quant/eval/recovery-A-B venv (torch + vLLM)
├── requirements-train.txt  # FSDP train venv (torch 2.7 + flash-linear-attention)
├── src/
│   ├── quant/   # model_adapters, moe_quant, quant_save, moe_probe/alloc/study, moe_compare
│   ├── eval/    # vllm_eval8, bench8_loaders, multimodal eval
│   └── distill/ # opd_gen, opd_score, opd_train_fsdp, moe_lora, merge_adapter
├── pipelines/   # smoke, quantize, eval, pack_gguf, distill, q36_recovery
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
