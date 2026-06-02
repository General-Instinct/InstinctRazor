#!/usr/bin/env python3
"""Diagnose + fix truncation on our quantized 122B. (1) Dump truncated BBEH generations to see if the model
LOOPS/repeats (->penalties) or genuinely over-reasons (->budget-forcing). (2) Compare sampling configs for
trunc rate + accuracy in ONE model load. Goal: drive trunc to 0 for a clean quality comparison."""
import argparse, json, re, time, sys
from collections import Counter

def repetition_score(s):
    """fraction of the last 2000 chars covered by the single most common 60-char shingle (loop detector)."""
    t = s[-4000:]
    sh = [t[i:i+60] for i in range(0, max(0, len(t)-60), 20)]
    if not sh: return 0.0
    return Counter(sh).most_common(1)[0][1] * 20 / max(1, len(t))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/q122_ptq3b_clip")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--budget", type=int, default=24576)
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    import bench8_loaders as B, vllm_eval as VE
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.budget+2048,
              max_num_seqs=64, gpu_memory_utilization=0.92, trust_remote_code=True)
    tok = llm.get_tokenizer()
    items = B.build_bbeh(args.n)
    prompts = [q + "\n\nReason step by step, then end with 'Answer: <answer>'." for q,_ in items]
    texts = [tok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in prompts]

    def free_ans(s):
        t = VE.post_think(s)[-600:]
        m = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*(.+)", t, re.I)
        c = m[-1] if m else next((ln for ln in reversed(t.splitlines()) if ln.strip()), "")
        return re.sub(r"[^\w]", "", re.sub(r"^[\(\[]?([a-e])[\)\]]?$", r"\1", c.strip().lower()))

    configs = {
        "base":        dict(temperature=0.6, top_p=0.95, top_k=20),
        "pp0.3":       dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.3),
        "pp0.5_fp0.3": dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.5, frequency_penalty=0.3),
        "rep1.1":      dict(temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.1),
    }
    for name, kw in configs.items():
        sp = SamplingParams(max_tokens=args.budget, **kw)
        outs = llm.generate(texts, sp)
        trunc = corr = 0; reps = []; toks = []
        for i, o in enumerate(outs):
            txt = o.outputs[0].text; nt = len(o.outputs[0].token_ids); toks.append(nt)
            ist = (o.outputs[0].finish_reason == "length"); trunc += ist
            reps.append(repetition_score(txt))
            if free_ans(txt) == re.sub(r"[^\w]", "", re.sub(r"^[\(\[]?([a-e])[\)\]]?$", r"\1", str(items[i][1]).strip().lower()))[:60] or free_ans(txt)==re.sub(r"[^\w]","",str(items[i][1]).strip().lower()):
                corr += 1
        hi_rep = sum(r > 0.3 for r in reps)
        print(f"[{name}] trunc={trunc}/{len(outs)} acc={100*corr/len(outs):.1f} meanTok={sum(toks)/len(toks):.0f} "
              f"loopy(rep>0.3)={hi_rep}", flush=True)
        if name == "base":
            for i, o in enumerate(outs):
                if o.outputs[0].finish_reason == "length":
                    print(f"   TRUNC#{i} rep={reps[i]:.2f} tail={o.outputs[0].text[-220:]!r}", flush=True)
                    if sum(1 for j,oo in enumerate(outs) if oo.outputs[0].finish_reason=='length' and j<=i) >= 3: break
    print("DIAG_DONE", flush=True)

if __name__ == "__main__":
    main()
