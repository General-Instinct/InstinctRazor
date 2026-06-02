# The validation-gate protocol

The single most important rule in this repo:

> **Do not trust a head-to-head on a benchmark until your harness reproduces that model's *own*
> published number on that benchmark, to ~1–2 points.** Until then the harness — not the model — may
> be what you're measuring.

A reimplemented benchmark can be wrong in a dozen quiet ways (prompt template, answer extraction,
grading strictness, generation budget, sample set). Each silently shifts every model's score. If you
compare two models on an ungated harness, you may just be comparing harness artifacts.

## Procedure

1. **Pick the axis** and the official targets (table in `configs/eval.env`; e.g. A4B MMLU-Pro 82.6, LCB-v6 77.1).
2. **Run the baseline you have a public number for** (A4B and/or the BF16 teacher) with
   `pipelines/eval.sh --model <M> --benchmarks <b>`.
3. **Compare to official:**
   - within ~1–2 pt → **gated.** Head-to-heads on this axis are trustworthy.
   - off by more → **not gated.** Diagnose before trusting *any* comparison on it.
4. **Diagnose a failed gate** using the three numbers the harness always reports per axis
   (`*_acc`, `*_acc_finished`, `*_trunc`):
   - **Truncation** (`trunc` high, `finished` ≫ `acc`): generation budget too low / the model rambles.
     Fix: raise the budget (`--budget 64k` for math/code/HLE). This is a *recipe* signal, not a bug.
   - **Extraction / grading strictness** (`trunc` low but `acc` uniformly under official for *all* models):
     the parser/grader is stricter than the vendor's. The **same-harness delta between models is still
     valid**; the **absolute** number is not vendor-exact. Say so explicitly.

## Reporting rules

- Headline uses overall **`acc`** (truncations = wrong). Report `finished` and `trunc` alongside, but
  **never compare `finished` across very different truncation rates** — it's a selection artifact.
- State each axis's gate status (see `results/RESULTS.md` → "Validation-gate status").

## Long-context (64k) config rule (validated on this box, vLLM 0.22)

- `--budget 64k` → `--max-tokens 65536 --max-model-len 66560 --max-num-seqs 32 --gpu-mem 0.88`.
- **Gemma-4 / A4B crash** at long context (NCCL HeartbeatMonitor worker-death) **unless `--enforce-eager`**
  is set (disables CUDA graphs, ~7× slower). `eval.sh` adds it automatically for `*gemma*` models.
- The **Qwen3.5 teacher + clip do NOT need** `--enforce-eager` (fast, no crash) — don't add it to them.

## Worked example (why LCB-v6 is reported but not gated)

Two earlier bugs were fixed (private-test base64(zlib(pickle(json))) decode; missing-imports preamble), but
the harness *still* read ~12 pt under official for **every** model uniformly (A4B 66 vs 77.1; teacher 65.5
vs 78.9) — a uniform offset + low truncation ⇒ extraction/grading strictness, not a per-model effect.

**A third fix (the one that should close the gate)** addresses four strictness sources identified by
root-cause analysis (see `git log` on `bench8_loaders.py` / `vllm_eval8.py`):
1. **Extraction** (`code_of_lcb`): now searches the full output (not just the post-`</think>` answer, so code
   emitted inside `<think>` is recovered) and **salvages an unclosed final fence** from a length-truncated
   generation (previously that returned prose → SyntaxError → false FAIL — the population inflating the
   acc-vs-finished spread).
2. **Functional arg parsing** (`lcb_run`): whole-input JSON first, then per-line (multi-line single-arg inputs
   no longer false-FAIL).
3. **Tolerant compare**: recursive list↔tuple coercion + float abs-tol 1e-6 (matches official semantics).
4. **Per-test SIGALRM timeout** (functional): a slow-but-correct solution is no longer aggregate-TLE'd; test
   cap raised 100→300 to cover the full set (max 103).

**The gate (must pass before any LCB head-to-head is trusted):** on this fixed harness, re-measure A4B and
the Qwen teacher @64k — they must reproduce official **A4B ≈ 77.1** and **Qwen ≈ 78.9** within ~1–2 pt. Until
that re-measure lands, LCB stays **not gated**: report the same-harness clip-vs-A4B *delta* as the
recoverable-gap signal, but do **not** treat the absolutes as vendor-exact. (The fix is validated on synthetic
cases; the gated re-measure is the documented reproduce-step — `bash pipelines/eval.sh --benchmarks lcb
--budget 64k` — and was deliberately not run in the packaging-only pass.)
