# InstinctRazor

**Sub-4-bit quantization for MoE with near-lossless post-training quantization.**

InstinctRazor compresses **Qwen3.5-122B-A10B** (~245 GB BF16) into a **~47 GB** deployment-ready model and no additional training.

The resulting model fits on a single 80 GB GPU and consistently outperforms the footprint-matched **Gemma-4-26B-A4B** across knowledge, reasoning, multilingual, and multimodal benchmarks while remaining close to the original BF16 teacher.

**GGUF release**

- Hugging Face: https://huggingface.co/General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF
- Deployable artifact: 48 GB IQ3_XXS GGUF
- Reproducible from the original BF16 checkpoint using this repository

---

## Results

| Benchmark | BF16 Teacher | InstinctRazor (~47 GB) | Gemma-4-26B-A4B (~52 GB) |
|-----------|-------------|------------------------|--------------------------|
| MMLU-Pro | 87.6 | **88.5** | 85.6 |
| GPQA-Diamond | 83.8 | **84.8** | 79.3 |
| MMMLU | 88.8 | **87.2** | 85.4 |
| MMMU-Pro | — | **80.8** | 73.8 |
| LiveCodeBench v6 | 65.5 | 57.0 | 66.0 |
| MATH-Vision | — | 70.0 | 82.4 |
| HLE (no tools) | 18.0 | 13.3 | 12.3 |

Most benchmark gaps relative to the teacher are concentrated in long-form code and math generation, where quantization increases truncation rates rather than causing large capability degradation.

Evaluation details are documented in `docs/EVAL_PROTOCOL.md`.

---

## Quickstart

### Smoke test

Run the full quantization → save → evaluation pipeline on a small model.

```bash
python3.12 -m venv vllm_venv
source vllm_venv/bin/activate
pip install -r requirements.txt

MOE_LOWBIT_VENV=$PWD/vllm_venv source env.sh

bash pipelines/smoke.sh
```

Expected output:

```text
SMOKE OK
```

### Reproduce the 122B result

```bash
python3.12 -m venv vllm_venv
source vllm_venv/bin/activate
pip install -r requirements.txt

HF_TOKEN=hf_xxx MOE_LOWBIT_VENV=$PWD/vllm_venv source env.sh

bash pipelines/quantize.sh

bash pipelines/eval.sh \
  --model models/q122_ptq3b_clip \
  --tag clip \
  --benchmarks mmlu_pro,gpqa,mmmlu

bash pipelines/eval.sh \
  --model google/gemma-4-26b-a4b-it \
  --tag a4b \
  --benchmarks mmlu_pro,gpqa,mmmlu
```

Results are written to:

```text
results/vllm_eval/
```

---

## Quantization Configuration

| Component | Quantization |
|------------|------------|
| Routed experts | INT3 |
| Backbone linear layers | INT4 |
| Group size | 128 |
| Quantization | Symmetric MSE clipping |
| Router/gates | BF16 |
| Norms | BF16 |
| Vision tower | BF16 |
| Effective bits | ~3.05 |
| Final size | ~47 GB |

The repository includes tooling for expert-level salience analysis and adaptive bit allocation. In practice, uniform 3-bit expert quantization was sufficient to retain downstream performance, so the default recipe remains uniform.

---

## GGUF Deployment

The evaluation checkpoints are saved as dequantized BF16 models after quantization-aware processing so that benchmark results measure capability without kernel-specific low-bit effects.

Deployment uses a separate GGUF conversion path built directly from the original BF16 weights.

Download:

```text
General-Instinct/InstinctRazor-Qwen3.5-122B-A10B-GGUF
```

Or rebuild locally:

```bash
BASE_TYPE=IQ3_XXS bash pipelines/pack_gguf.sh
```

Validated artifact:

```text
q122_orig-IQ3XXS-protected.gguf
```

Characteristics:

| Metric | Value |
|----------|---------|
| Size | 48.04 GiB |
| MMLU-Pro | 90.7 |
| GPQA-Diamond | 80.8 |
| Decode speed (1× H100-80GB) | 115.9 tok/s |
| Decode speed (expert offload) | 45.7 tok/s |

---

## Repository Layout

```text
InstinctRazor/

├── env.sh
├── requirements.txt
├── requirements-train.txt

├── src/
│   ├── quant/
│   ├── eval/
│   └── distill/

├── pipelines/
│   ├── smoke.sh
│   ├── quantize.sh
│   ├── eval.sh
│   └── distill.sh

├── configs/
├── docs/
├── results/
└── archive/
```

---

## Optional Distillation

For code and mathematical reasoning workloads, InstinctRazor includes an optional on-policy distillation pipeline.

```bash
bash pipelines/distill.sh --smoke
bash pipelines/distill.sh
```

Pipeline:

```text
Generate
→ Score
→ FSDP training
→ Merge adapter
→ Re-quantize
```

---

## Hardware

Primary experiments were run on:

- 4× NVIDIA H100 80 GB
- CUDA 13.0
- NVLink
- ~885 GB system RAM

The compressed model is intended to run on:

- 1× H100 80 GB
- Smaller GPUs with expert offloading

The smoke test runs on substantially smaller hardware.

---

## Documentation

| Document | Description |
|-----------|-------------|
| `docs/EVAL_PROTOCOL.md` | Evaluation methodology |
| `docs/MOE_FINDINGS.md` | Quantization study and findings |
| `docs/MOE_PIPELINE.md` | End-to-end quantization pipeline |
| `docs/OPD_INTEGRATION.md` | Distillation workflow |
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
