# configs/

Single source of truth for every recipe. The pipelines `source` these files; every value is
overridable from the environment (`VAR=... bash pipelines/quantize.sh`).

| file | used by | what it pins |
|------|---------|--------------|
| `clip_122b.env` | `quantize.sh` | the **shipped** 122B recipe: uniform 3b experts / 4b backbone, group 128, MSE clip-search 16, protected always-on path |
| `smoke_2b.env`  | `smoke.sh` | single-GPU dense smoke (same code path, linear-quant only) |
| `eval.env`      | `eval.sh` | sampling, 32k/64k budgets, the enforce-eager rule, validation-gate targets |
| `opd_r1.env`    | `distill.sh` | Lightning-OPD on-policy distillation: rank-16 expert-LoRA, reverse-KL, FSDP2 |

## The recipe in one line

```
expert_bits=3.0 (int3, all 256 routed experts) · linear_bits=4.0 (int4, backbone+embed) ·
group=128 · per-block MSE clip-search (16 steps) · protected@bf16: vision tower + router/gate + norms
```

## Why uniform, not salience-aware allocation?

The proposed structure imagined `probe → per-expert alloc → quantize`. We built exactly that
tooling (`src/quant/moe_probe.py` + `moe_alloc.py` + `moe_study.py`) and ran the study
(`results/study/`). The finding (docs/MOE_FINDINGS.md **F4/F6**): on a load-balanced router
(entropy 0.952, 0/256 cold experts) salience allocation is a real lever for *distribution fidelity*
(KL) but **second-order for downstream capability** — blind/uniform 3b already lands near-lossless.
So the **shipped** checkpoint is deliberately uniform (`quant_save.py` with a flat `AllocSpec`), and
the probe/alloc path is the **optional research** that justified it. `quantize.sh` runs the
probe as an optional diagnostic and then the real uniform recipe.

Two protection schemes exist; don't confuse them:
- **shipped clip** (this framework's headline): flat — non-expert linears all 4b, only vision+router+norms at bf16.
- **study/sweep variant** (`moe_alloc.build_spec(protect=True)`): tiered — shared-expert int8, attn/DeltaNet int4,
  embed int4, router bf16. Used in `results/study/`, **not** in the deployed model. (docs/MOE_PIPELINE.md describes this tiered recipe — it is the study recipe, not the shipped one.)
