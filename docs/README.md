# docs/ — index

Start with the top README, then `results/RESULTS.md`, then these in order.

| doc | what it is | read it for |
|-----|-----------|-------------|
| **EVAL_PROTOCOL.md** | the validation-gate protocol | *read before trusting any number* — how a comparison earns trust |
| **MOE_FINDINGS.md** | canonical results log, findings F1–F11 | why the recipe is what it is (esp. **F3–F6**: 3b near-lossless; allocation is 2nd-order; **F9**: recovery must be global, not local) |
| **MOE_PIPELINE.md** | capstone recipe + deployment memory math | the bit budget, the ~47 GB footprint math, the GGUF deployment step |
| **LADDER_RESULTS.md** | win-ladder vs Gemma-4 (the beat-A4B study) | the head-to-head numbers + the LCB bug-fix story |
| **OPD_INTEGRATION.md** | Lightning-OPD on-policy-distillation design | the distill/ path: gen→score→train→merge, what gap it targets |
| **POSITIONING.md** | SOTA positioning + statistical rigor | how this sits vs DQ3_K_M / XFP; the underpowered-n caveats; ranked next experiments |
| **GEMMA_PARETO.md** | the Gemma-4 Pareto study | provenance of the baseline choice. **Note:** an early "NO" verdict is explicitly OVERTURNED later in the same file — only the post-"VERDICT REVERSED" section is current. |
| **SURVEY_MOE.md** | 2025–26 MoE-quant method survey | the literature the recipe draws on |

## Two honesty flags carried from the source docs

1. **MOE_PIPELINE.md describes the *tiered* protection recipe** (shared int8 / attn-DeltaNet int4 /
   embed int4 / router bf16). That is the **study-sweep** recipe (`moe_alloc.build_spec(protect=True)`,
   `results/study/`), **not** the shipped `q122_ptq3b_clip`, which is flat 3b-experts / 4b-everything-else
   (only vision + router + norms at bf16). See `configs/README.md`.
2. **POSITIONING.md** flags that the original near-lossless generation numbers were underpowered
   (n=40 GSM8K / n=32 HumanEval). The later ladder (`LADDER_RESULTS.md`, n=500/200) is the higher-powered
   evidence; treat the small-n F-numbers as directional.

The dense-era predecessor docs (FINDINGS/PIPELINE/RESULTS/SCALING/SURVEY for the 0.8B–27B "EdgeRazor"
program) were intentionally **not** copied here — that regime is *inverted* by the MoE findings and would
mislead. They remain in the original `qwen35_qad/` tree for provenance.
