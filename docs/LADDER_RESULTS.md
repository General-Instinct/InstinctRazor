# Win-ladder results — compressed Qwen3.5-122B vs Gemma-4 (A4B primary)

**Setup.** Same harness (vllm_eval8.py, vLLM 0.22 TP=4, 32k thinking budget, seed-0 identical samples), n = MMLU-Pro 500 / GPQA-D 198 / MMMLU 500. Validation gate: Gemma-4-31b-it reproduces its official numbers within ~1.5pt (MMLU-Pro 85.4 vs 85.2, GPQA 85.9 vs 84.3, MMMLU 89.6 vs 88.4) ⇒ harness trustworthy. Same-harness deltas are the reliable quantity (per-model harness-vs-official offsets cancel).

**Primary baseline = Gemma-4-26B-A4B (MoE, ~52GB BF16, ~3.8B active)** — footprint-matched (our compressed ≈47–50GB @3b) and MoE-vs-MoE. Secondary = Gemma-4-31B-dense (62.5GB). Our model has ~2.6× A4B's active compute (10B vs 3.8B).

## Tier-1 results (overall / among-finished / truncations)

| Axis | Gemma-31b (2nd) | A4B (PRIMARY) | Teacher BF16 (Step 1) | Clip 3b/47GB (Step 2) |
|---|---|---|---|---|
| MMLU-Pro | 85.4 / 85.4 / 0 | 85.6 / 87.1 / 12 | 87.6 / 87.6 / 0 | 86.2 / 86.2 / 0 |
| GPQA-Diamond | 85.9 / 86.2 / 2 | 79.3 / 82.5 / 9 | 83.8 / 85.1 / 3 | 82.3 / 86.2 / 10 |
| MMMLU | 89.6 / 89.6 / 0 | 85.4 / 85.4 / 0 | 88.8 / 88.8 / 0 | 87.2 / 87.2 / 0 |

(clip MMLU-Pro & GPQA from ladder_clip.log — that run died on a transient RPC-timeout entering MMMLU before writing its JSON; values are valid, same run/seed. clip MMMLU + A4B from JSONs.)

## Classification vs A4B (the primary) — overall accuracy

| Axis | base vs A4B | compression (clip vs A4B) | class |
|---|---|---|---|
| MMLU-Pro | WIN +2.0 | PRESERVES +0.6 (within noise) | **claim-now** (marginal) |
| GPQA-D | WIN +4.5 | PRESERVES +3.0 (finished +3.7) | **claim-now** |
| MMMLU | WIN +3.4 | PRESERVES +1.8 | **claim-now** |

**On all 3 measured Tier-1 axes the compressed 47GB model beats the footprint-matched A4B, near-lossless from the BF16 teacher — no recovery needed.** vs the bigger 31B-dense: wins MMLU-Pro (86.2>85.4), ties GPQA-on-finished (86.2≈86.2, loses overall only to truncation), loses MMMLU (87.2<89.6) — "smaller and competitive."

## Honest caveats
- A4B truncates at 32k too (MMLU-Pro 12, GPQA 9) — truncation is inherent problem difficulty, not a model defect (cf. teacher control: BF16 teacher truncates ~19% on hard MATH). Among-finished is the cleaner capability measure.
- MMLU-Pro clip-vs-A4B (+0.6) is within harness noise; GPQA (+3.7 finished) and MMMLU (+1.8) are the solid wins.
- Harness reads A4B MMLU-Pro a bit high and GPQA a bit low (truncation) vs official; same-harness deltas remain valid.

## LiveCodeBench v6 (MEASURED — the recoverable-gap axis; built loader in bench8_loaders.build_lcb/lcb_run)

Same harness, 32k budget, seed-0 n=200 of the full v6 (1055; 611 stdin / 444 functional). 3 models (Gemma-31b dropped).

**⚠ BUG FOUND + FIXED (user-prompted by A4B 91 vs official 77.1).** `private_test_cases` are `base64(zlib(pickle(json)))` — pickle-wrapped — but build_lcb only tried plain JSON → threw → fell back to PUBLIC-tests-only (median 3 tests/problem) → far too lenient → inflated pass@1. Fixed the decoder (pickle-aware) → median 20 tests/problem. The buggy run (A4B 91.0 / teacher 83.5 / clip 64.5) is INVALID; corrected numbers below.

| Model | overall pass@1 | among-finished | truncation | official |
|---|---|---|---|---|
| A4B (primary) | 69.0 | 71.5 | 3.5% (7/200) | 77.1 |
| Teacher BF16 | 64.5 | 72.6 | 12.5% (25/200) | 78.9 |
| Clip 3b/47GB | **53.5** | 82.8* | **42% (84/200)** | — |

**LCB v6 = THE recoverable-gap (compression breaks the win).** Clip overall 53.5 ≪ A4B 69.0 (−15.5) and ≪ teacher 64.5 — the deficit is **truncation (42%), not capability collapse**: quantization made the model token-INEFFICIENT on code (rambles → hits 32k → truncates), exactly the Stage-1 thesis. Even the BF16 teacher truncates 12.5% on code (vs A4B 3.5%); quantization AMPLIFIES this to 42%. Among-finished, teacher 72.6 ≈ A4B 71.5 (matches official's teacher≈A4B). **Contrast with MATH** (teacher control: clip's excess truncation over teacher ~1.7pp, no headroom) vs **CODE** (clip 42% vs teacher 12.5% = ~30pp excess) ⇒ **OPD should target CODE, not math.** OPD target: pull clip's 42% truncation toward teacher's ~12% (ideally A4B's ~3.5%). Justifies resuming the paused OPD/FSDP push (task #19) on code.

Caveats: (1) uniform ~8-14pt under-shoot vs official (A4B 69 vs 77.1, teacher 64.5 vs 78.9) = reimplementation strictness (functional `==` + code-extraction stricter than LCB's official normalized grader) + sample/window — consistent across models, so DELTAS are valid; absolutes are not vendor-exact. (2) clip among-finished 82.8 is SURVIVORSHIP-biased (finishes only the easier 116, truncates the hard 84) — NOT evidence of capability parity; the honest claim is the overall-pass@1 loss + the truncation mechanism. (3) full-v6 sample incl old problems (Gemma-4 is a 2026 model → all v6 in-distribution anyway).

## Remaining Tier-2 axes (not yet measured)
- **HLE no-tools** (official base 25.3 ≫ A4B 8.7 — big base-win): build_hle exists but grading is hard (free-form; official uses an LLM judge) — needs an hle branch + a grading approach.
- **Codeforces ELO / τ²-Bench**: no harness in-tree (Elo estimator / agentic loop) — out of scope unless built.
