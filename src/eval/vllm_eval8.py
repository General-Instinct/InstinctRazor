#!/usr/bin/env python3
"""Comprehensive 8-benchmark eval (our 47GB model vs Gemma-4-31B-it), thinking mode, vLLM TP=4.
Benchmarks (6 from Gemma-4 card + 2 standard code): MMLU-Pro, GPQA-Diamond, MMMLU, BBEH, AIME-2025,
MATH-500, HumanEval, MBPP. Reuses vllm_eval (gen/parse/post_think/build_mmlu_pro), moe_eval (gpqa/humaneval),
bench8_loaders (the 5 new ones). Reports per-bench accuracy, accuracy-AMONG-FINISHED, and truncation rate.

among-finished = correct restricted to finish_reason != 'length' (separates the token-inefficiency/truncation
deficit from a genuine capability gap). Computed round-exact from the per-sample finished mask, not int()-reconstructed."""
import argparse, json, os, re, time, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--benchmarks", default="mmlu_pro,gpqa,mmmlu,bbeh,aime,math500,humaneval,mbpp")
    ap.add_argument("--mmlu-n", type=int, default=150)
    ap.add_argument("--gpqa-n", type=int, default=198)
    ap.add_argument("--mmmlu-n", type=int, default=150)
    ap.add_argument("--bbeh-n", type=int, default=120)
    ap.add_argument("--aime-n", type=int, default=30)
    ap.add_argument("--math-n", type=int, default=120)
    ap.add_argument("--he-n", type=int, default=164)
    ap.add_argument("--mbpp-n", type=int, default=150)
    ap.add_argument("--lcb-n", type=int, default=200)
    ap.add_argument("--hmmt-n", type=int, default=30)
    ap.add_argument("--hle-n", type=int, default=500)
    ap.add_argument("--enforce-eager", action="store_true")  # disable CUDA graphs (robust at long/64k context)
    args = ap.parse_args()
    os.makedirs("results/vllm_eval", exist_ok=True)
    from vllm import LLM, SamplingParams
    import moe_eval as EV, vllm_eval as VE, bench8_loaders as B
    t0 = time.time()
    # disable_custom_all_reduce: NCCL all-reduce (lossless), avoids the custom_all_reduce IPC race; max_num_seqs<=508 (Mamba cache)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_mem, trust_remote_code=True,
              disable_custom_all_reduce=True, enforce_eager=args.enforce_eager)
    tok = llm.get_tokenizer()
    print(f"[bench8] {args.tag} loaded in {time.time()-t0:.0f}s", flush=True)

    def gen(prompts, max_tokens):
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens)
        texts = []
        for p in prompts:
            m = [{"role": "user", "content": p}]
            try: texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True))
            except TypeError: texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
        outs = llm.generate(texts, sp)
        fin = [o.outputs[0].finish_reason != "length" for o in outs]      # True = closed naturally (not truncated)
        return [o.outputs[0].text for o in outs], sum(1 for f in fin if not f), fin

    def free_answer(s):
        """extract a short free-form final answer (after 'answer is/:' or last non-empty line)."""
        t = VE.post_think(s)[-600:]
        m = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*(.+)", t, re.I)
        cand = m[-1] if m else next((ln for ln in reversed(t.splitlines()) if ln.strip()), "")
        return cand.strip()

    def norm_free(x):
        x = str(x).strip().lower()
        x = re.sub(r"^[\(\[\{]?\s*([a-e])\s*[\)\]\}]?$", r"\1", x)  # "(A)" -> "a"
        return re.sub(r"[^\w]", "", x)

    def score(correct, fin):
        """overall acc + accuracy-among-finished (finish_reason!='length'); round-exact, not int()-reconstructed."""
        n = len(correct); nf = sum(1 for f in fin if f)
        acc = 100.0 * sum(1 for c in correct if c) / max(n, 1)
        accf = 100.0 * sum(1 for c, f in zip(correct, fin) if c and f) / max(nf, 1)
        return acc, accf, nf

    rec = {"tag": args.tag, "model": args.model, "think": True}
    bset = args.benchmarks.split(",")
    g = lambda ps: gen(ps, args.max_tokens)

    def emit(name, key, correct, fin, tr, n):
        acc, accf, nf = score(correct, fin)
        rec.update({f"{key}_acc": acc, f"{key}_acc_finished": accf, f"{key}_n": n,
                    f"{key}_n_finished": nf, f"{key}_trunc": tr})
        print(f"[bench8] {name} {acc:.1f} (finished {accf:.1f} | n={n} nf={nf} tr={tr})", flush=True)
        return acc, accf

    if "mmlu_pro" in bset:
        it = VE.build_mmlu_pro(args.mmlu_n)
        ps = [q + "\n" + "\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i,c in enumerate(o)) + "\n\nReason, then end with 'Answer: <letter>'." for q,o,_ in it]
        out,tr,fin = g(ps); cor = [VE.parse_letter(VE.post_think(o),len(it[i][1]))==it[i][2] for i,o in enumerate(out)]
        emit("MMLU-Pro", "mmlu_pro", cor, fin, tr, len(it))

    if "gpqa" in bset:
        gp,_ = EV.build_gpqa(args.gpqa_n)
        ps = [q + "\n\nReason, then end with 'Answer: <letter>'." for q,_ in gp]
        out,tr,fin = g(ps); cor = [VE.parse_letter(VE.post_think(o),4)==gp[i][1] for i,o in enumerate(out)]
        emit("GPQA-D", "gpqa", cor, fin, tr, len(gp))

    if "mmmlu" in bset:
        it = B.build_mmmlu(args.mmmlu_n)
        ps = [q + "\n" + "\n".join(f"{'ABCD'[i]}. {c}" for i,c in enumerate(o)) + "\n\nReason, then end with 'Answer: <letter>'." for q,o,_ in it]
        out,tr,fin = g(ps); cor = [VE.parse_letter(VE.post_think(o),4)==it[i][2] for i,o in enumerate(out)]
        emit("MMMLU", "mmmlu", cor, fin, tr, len(it))

    if "bbeh" in bset:
        it = B.build_bbeh(args.bbeh_n)
        ps = [q + "\n\nReason step by step, then end with 'Answer: <answer>'." for q,_ in it]
        out,tr,fin = g(ps); cor = [norm_free(free_answer(o))==norm_free(it[i][1]) for i,o in enumerate(out)]
        emit("BBEH", "bbeh", cor, fin, tr, len(it))

    if "aime" in bset:
        it = B.build_aime(args.aime_n)
        ps = [q + "\n\nReason step by step, then put the final integer answer in \\boxed{}." for q,_ in it]
        out,tr,fin = g(ps); cor = [B.math_eq(B.extract_boxed(VE.post_think(o)), it[i][1]) for i,o in enumerate(out)]
        emit("AIME-2025", "aime", cor, fin, tr, len(it))

    if "aime2026" in bset:
        it = B.build_matharena("MathArena/aime_2026", args.aime_n)
        ps = [q + "\n\nReason step by step, then put the final answer in \\boxed{}." for q,_ in it]
        out,tr,fin = g(ps); cor = [B.math_eq(B.extract_boxed(VE.post_think(o)), it[i][1]) for i,o in enumerate(out)]
        emit("AIME-2026", "aime2026", cor, fin, tr, len(it))

    if "hmmt" in bset:
        it = B.build_matharena("MathArena/hmmt_feb_2025", args.hmmt_n)
        ps = [q + "\n\nReason step by step, then put the final answer in \\boxed{}." for q,_ in it]
        out,tr,fin = g(ps); cor = [B.math_eq(B.extract_boxed(VE.post_think(o)), it[i][1]) for i,o in enumerate(out)]
        emit("HMMT-Feb25", "hmmt", cor, fin, tr, len(it))

    if "math500" in bset:
        it = B.build_math500(args.math_n)
        ps = [q + "\n\nReason step by step, then put the final answer in \\boxed{}." for q,_ in it]
        out,tr,fin = g(ps); cor = [B.math_eq(B.extract_boxed(VE.post_think(o)), it[i][1]) for i,o in enumerate(out)]
        emit("MATH-500", "math500", cor, fin, tr, len(it))

    if "humaneval" in bset:
        it = EV.build_humaneval(args.he_n)
        ps = [f"Complete this Python function. Reason if needed, then give the FULL function in a ```python code block.\n\n```python\n{p}```" for p,_,_ in it]
        out,tr,fin = g(ps); cor = [bool(EV._he_run(EV._he_extract(it[i][0], VE.post_think(o)), it[i][1], it[i][2])) for i,o in enumerate(out)]
        acc, accf, nf = score(cor, fin)
        rec.update(humaneval_pass1=acc, humaneval_pass1_finished=accf, he_n=len(it), he_n_finished=nf, he_trunc=tr)
        print(f"[bench8] HumanEval {acc:.1f} (finished {accf:.1f} | n={len(it)} nf={nf} tr={tr})", flush=True)

    if "mbpp" in bset:
        it = B.build_mbpp(args.mbpp_n)
        ps = [f"{p}\nYour function must pass this test:\n{tl[0]}\n\nWrite the Python function (use the exact name/signature the test expects). Give the FULL function in a ```python code block."
              for p,_,tl,_ in it]
        out,tr,fin = g(ps)
        def code_of(o):
            m = re.findall(r"```(?:python)?\s*(.+?)```", VE.post_think(o), re.S)
            return m[-1] if m else VE.post_think(o)
        cor = [bool(B.mbpp_run(code_of(o), it[i][2], it[i][3])) for i,o in enumerate(out)]
        acc, accf, nf = score(cor, fin)
        rec.update(mbpp_pass1=acc, mbpp_pass1_finished=accf, mbpp_n=len(it), mbpp_n_finished=nf, mbpp_trunc=tr)
        print(f"[bench8] MBPP {acc:.1f} (finished {accf:.1f} | n={len(it)} nf={nf} tr={tr})", flush=True)

    if "lcb" in bset:
        it = B.build_lcb(args.lcb_n)
        ps = [p for p, _ in it]
        out,tr,fin = g(ps)
        def code_of_lcb(o):
            # Robust extraction (LCB gate fix): prefer a fenced block in the post-think ANSWER, then anywhere
            # in the FULL output (code is sometimes emitted only inside <think>). If the final fence is unclosed
            # (length-truncated generation), salvage from the last opener to EOF instead of dumping prose into
            # the runner (the prose->SyntaxError->false-FAIL population that inflated acc-vs-finished spread).
            pt = VE.post_think(o)
            for src in (pt, o):
                m = re.findall(r"```(?:python)?\s*(.+?)```", src, re.S)
                if m: return m[-1]
            for src in (pt, o):
                i = src.rfind("```")
                if i != -1:
                    tail = re.sub(r"^[ \t]*(?:python|py)?[ \t]*\n", "", src[i+3:], count=1)
                    if tail.strip(): return tail
            return ""
        cor = [bool(B.lcb_run(code_of_lcb(o), it[i][1])) for i,o in enumerate(out)]
        acc, accf, nf = score(cor, fin)
        rec.update(lcb_acc=acc, lcb_acc_finished=accf, lcb_n=len(it), lcb_n_finished=nf, lcb_trunc=tr)
        print(f"[bench8] LCB-v6 {acc:.1f} (finished {accf:.1f} | n={len(it)} nf={nf} tr={tr})", flush=True)

    if "hle" in bset:
        it = B.build_hle(args.hle_n)   # (question, answer, answer_type); text-only filtered inside
        ps = [q + "\n\nReason step by step, then end your response with 'Answer: <final answer>' "
                  "(for multiple-choice, give the letter)." for q,_,_ in it]
        out,tr,fin = g(ps)
        def hle_ok(o, gold, at):       # Tier-1 deterministic grader (lower-bound vs official LLM-judge)
            t = VE.post_think(o); gg = str(gold)
            if at == "multipleChoice":
                gi = VE.parse_letter(t, 26)
                return gi >= 0 and "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[gi] == gg.strip().upper()[:1]
            cand = free_answer(t); bx = B.extract_boxed(t)
            if B.math_eq(bx, gg) or B.math_eq(cand, gg): return True
            return norm_free(cand) == norm_free(gg) or (bx is not None and norm_free(bx) == norm_free(gg))
        cor = [hle_ok(o, it[i][1], it[i][2]) for i,o in enumerate(out)]
        emit("HLE", "hle", cor, fin, tr, len(it))

    rec["seconds"] = time.time()-t0
    json.dump(rec, open(f"results/vllm_eval/{args.tag}.json","w"), indent=2)
    print(f"[bench8] DONE {args.tag}: " + " ".join(f"{k}={v:.1f}" for k,v in rec.items()
                                                    if k.endswith(('_acc','_pass1'))), flush=True)

if __name__ == "__main__":
    main()
