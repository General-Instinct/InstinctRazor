#!/usr/bin/env python3
"""FAIR thinking-mode eval via vLLM (fast TP=4 + continuous batching) — the decisive Gemma-vs-122B harness.
Fixes the prior flaws: (1) 32k-token thinking budget (was 2048 -> truncated CoT -> answers never emitted);
(2) thinking-vs-thinking (both models reason); (3) MMLU-Pro not classic MMLU. Reports per-bench accuracy +
truncation rate (finish_reason=='length'). Reuses moe_eval builders for GPQA-Diamond/GSM8K/HumanEval; adds
MMLU-Pro. Works for Qwen3.5 (qwen3_5_moe, </think>) and Gemma-4 (gemma4, <|think|> channel)."""
import argparse, json, os, re, time

def build_mmlu_pro(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for r in ds:
        items.append((r["question"], r["options"], int(r["answer_index"])))
    return items

def post_think(s):
    # answer is after the final </think> (Qwen) ; Gemma emits no </think> -> whole text
    if "</think>" in s:
        return s.rsplit("</think>", 1)[-1]
    return s

def parse_letter(s, k):
    """Tail-focused, last-match letter parse. Robust to long CoT (full of stray A-J letters) and to either
    model's thinking format: the final answer is at the END (we instruct "end with 'Answer: <letter>'"), so
    look only at the tail and take the LAST answer-marker match. Avoids the bias where un-stripped reasoning
    letters get grabbed by a naive search (under-counted Gemma's 10-option MMLU-Pro)."""
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:k]   # widened: HLE multiple-choice has up to ~22 options (golds to letter V)
    tail = s[-600:]
    for p in [r"(?:final\s+answer|answer|option|correct)\s*(?:is|:)?\s*\*{0,2}\(?([%s])\)?\*{0,2}" % L,
              r"\\boxed\{\s*\(?([%s])\)?\s*\}" % L]:
        cand = re.findall(p, tail, re.I)
        if cand:
            return L.index(cand[-1].upper())
    # fallback: last standalone letter on the last non-empty line
    for line in reversed([x for x in tail.splitlines() if x.strip()]):
        m = re.findall(r"\b([%s])\b" % L, line)
        if m:
            return L.index(m[-1].upper())
    return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-tokens", type=int, default=32768)        # thinking budget
    ap.add_argument("--max-tokens-direct", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--quantization", default=None, help="vLLM quantization (e.g. bitsandbytes) for footprint-compliant run")
    ap.add_argument("--benchmarks", default="mmlu_pro,gpqa,gsm8k,humaneval")
    ap.add_argument("--mmlu-n", type=int, default=300)
    ap.add_argument("--gpqa-n", type=int, default=198)
    ap.add_argument("--gsm-n", type=int, default=200)
    ap.add_argument("--he-n", type=int, default=164)
    args = ap.parse_args()
    os.makedirs("results/vllm_eval", exist_ok=True)
    from vllm import LLM, SamplingParams
    import moe_eval as EV
    t0 = time.time()
    llm_kw = dict(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
                  max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_mem, trust_remote_code=True,
                  enforce_eager=False, disable_custom_all_reduce=True)  # lossless NCCL all-reduce; avoid IPC race
    if args.quantization:
        llm_kw["quantization"] = args.quantization
    llm = LLM(**llm_kw)
    tok = llm.get_tokenizer()
    print(f"[vllm] {args.tag} loaded in {time.time()-t0:.0f}s think={args.think}", flush=True)
    budget = args.max_tokens if args.think else args.max_tokens_direct

    def gen(prompts, max_tokens, math=False):
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens)
        texts = []
        for p in prompts:
            m = [{"role": "user", "content": p}]
            try:
                texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=args.think))
            except TypeError:
                texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
        outs = llm.generate(texts, sp)
        txt = [o.outputs[0].text for o in outs]
        trunc = sum(o.outputs[0].finish_reason == "length" for o in outs)
        return txt, trunc

    rec = {"tag": args.tag, "model": args.model, "think": args.think, "budget": budget}
    bset = args.benchmarks.split(",")
    dbg = {}

    if "mmlu_pro" in bset:
        items = build_mmlu_pro(args.mmlu_n)
        prompts = [q + "\n" + "\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i, c in enumerate(opts)) +
                   "\n\nReason step by step, then end with a line 'Answer: <letter>'." for q, opts, _ in items]
        out, trunc = gen(prompts, budget)
        corr = sum(parse_letter(post_think(o), len(items[i][1])) == items[i][2] for i, o in enumerate(out))
        dbg["mmlu_pro"] = [{"gold": "ABCDEFGHIJ"[items[i][2]], "parsed": ("ABCDEFGHIJ"[p] if (p:=parse_letter(post_think(o), len(items[i][1]))) >= 0 else "?"), "tail": post_think(o)[-200:]} for i, o in enumerate(out[:20])]
        rec["mmlu_pro_acc"] = 100.0 * corr / len(items); rec["mmlu_pro_n"] = len(items); rec["mmlu_pro_trunc"] = trunc
        print(f"[vllm] MMLU-Pro {rec['mmlu_pro_acc']:.1f} (n={len(items)} trunc={trunc})", flush=True)

    if "gpqa" in bset:
        gpqa, _ = EV.build_gpqa(args.gpqa_n)
        if gpqa:
            prompts = [q + "\n\nReason step by step, then end with a line 'Answer: <letter>'." for q, _ in gpqa]
            out, trunc = gen(prompts, budget)
            corr = sum(parse_letter(post_think(o), 4) == gpqa[i][1] for i, o in enumerate(out))
            rec["gpqa_acc"] = 100.0 * corr / len(gpqa); rec["gpqa_n"] = len(gpqa); rec["gpqa_trunc"] = trunc
            print(f"[vllm] GPQA-D {rec['gpqa_acc']:.1f} (n={len(gpqa)} trunc={trunc})", flush=True)

    if "gsm8k" in bset:
        items = EV.build_gsm8k(args.gsm_n)
        prompts = [q + "\n\nSolve step by step and end with 'The answer is <number>'." for q, _ in items]
        out, trunc = gen(prompts, budget)
        corr = 0
        for (q, gold), o in zip(items, out):
            p = EV._last_num(post_think(o))
            try:
                if p is not None and abs(float(p) - float(gold)) < 1e-3: corr += 1
            except ValueError: pass
        rec["gsm8k_acc"] = 100.0 * corr / len(items); rec["gsm8k_n"] = len(items); rec["gsm8k_trunc"] = trunc
        print(f"[vllm] GSM8K {rec['gsm8k_acc']:.1f} (n={len(items)} trunc={trunc})", flush=True)

    if "humaneval" in bset:
        items = EV.build_humaneval(args.he_n)
        prompts = [f"Complete this Python function. Reason if needed, then give the FULL function in a "
                   f"```python code block.\n\n```python\n{p}```" for p, _, _ in items]
        out, trunc = gen(prompts, budget)
        passed = 0
        for (p, test, ep), o in zip(items, out):
            code = EV._he_extract(p, post_think(o))
            if EV._he_run(code, test, ep): passed += 1
        rec["humaneval_pass@1"] = 100.0 * passed / len(items); rec["he_n"] = len(items); rec["he_trunc"] = trunc
        print(f"[vllm] HumanEval {rec['humaneval_pass@1']:.1f} (n={len(items)} trunc={trunc})", flush=True)

    rec["seconds"] = time.time() - t0
    json.dump(rec, open(f"results/vllm_eval/{args.tag}.json", "w"), indent=2)
    if dbg:
        json.dump(dbg, open(f"results/vllm_eval/{args.tag}_dbg.json", "w"), indent=2)
    print(f"[vllm] DONE {args.tag}: " + " ".join(f"{k}={v}" for k, v in rec.items()
          if k.endswith("acc") or k.endswith("pass@1")), flush=True)

if __name__ == "__main__":
    main()
