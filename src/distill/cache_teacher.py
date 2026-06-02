#!/usr/bin/env python3
"""Stage 2 foundation: cache the BF16 teacher's OWN verified CoT on math+code TRAINING prompts (not test).
Teacher-consistency (Lightning-OPD): the same BF16 122B that we PTQ'd is the distillation teacher. Keep only
generations the teacher gets RIGHT (math_eq / code exec) -> high-quality concise-CoT SFT targets for the
quantized student. Output: jsonl of {domain, prompt, target} (target = teacher's full <think>..</think>+answer)."""
import argparse, json, re, time, random

def load_math(n):
    from datasets import load_dataset
    last=None
    for name,conf,split,qk,ak in [
        ("EleutherAI/hendrycks_math","algebra","train","problem","solution"),
        ("open-r1/OpenR1-Math-220k",None,"train","problem","answer"),
        ("nlile/hendrycks-MATH-benchmark",None,"train","problem","answer")]:
        try:
            if name=="EleutherAI/hendrycks_math":
                from datasets import get_dataset_config_names, concatenate_datasets
                parts=[load_dataset(name,c,split="train") for c in get_dataset_config_names(name)]
                ds=concatenate_datasets(parts)
            else:
                ds=load_dataset(name,conf,split=split) if conf else load_dataset(name,split=split)
            cols=ds.column_names
            qk= "problem" if "problem" in cols else ("question" if "question" in cols else cols[0])
            ak= "answer" if "answer" in cols else ("solution" if "solution" in cols else None)
            ds=ds.shuffle(seed=0).select(range(min(n,len(ds))))
            import bench8_loaders as B
            out=[]
            for r in ds:
                gold=str(r.get(ak) if ak else None)
                # gold may be a full worked solution; extract the final \boxed{} answer if present
                if "\\boxed" in gold:
                    bx=B.extract_boxed(gold)
                    if bx is not None: gold=bx
                out.append((r[qk], gold))
            print(f"[cache] math={name} n={len(out)} q={qk} a={ak}",flush=True); return out
        except Exception as e: last=f"{name}:{str(e)[:70]}"
    print(f"[cache] math load FAILED: {last}",flush=True); return []

def load_code(n):
    from datasets import load_dataset
    try:
        ds=load_dataset("google-research-datasets/mbpp","full",split="train")
        out=[(r["text"], r["test_list"], r.get("test_setup_code","") or "") for r in ds.select(range(min(n,len(ds))))]
        print(f"[cache] code=mbpp-train n={len(out)}",flush=True); return out
    except Exception as e:
        print(f"[cache] code load FAILED: {str(e)[:80]}",flush=True); return []

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--out",default="results/stage2/teacher_cot.jsonl")
    ap.add_argument("--math-n",type=int,default=1500); ap.add_argument("--code-n",type=int,default=400)
    ap.add_argument("--budget",type=int,default=16384); ap.add_argument("--tp",type=int,default=4)
    args=ap.parse_args()
    import os; os.makedirs(os.path.dirname(args.out),exist_ok=True)
    from vllm import LLM, SamplingParams
    import bench8_loaders as B, vllm_eval as VE
    llm=LLM(model=args.model,tensor_parallel_size=args.tp,max_model_len=args.budget+2048,
            max_num_seqs=64,gpu_memory_utilization=0.92,trust_remote_code=True)
    tok=llm.get_tokenizer()
    sp=SamplingParams(temperature=0.6,top_p=0.95,top_k=20,max_tokens=args.budget)
    def tmpl(p): return tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True,enable_thinking=True)

    kept=[]; t0=time.time()
    # math
    math=load_math(args.math_n)
    if math:
        prompts=[q+"\n\nReason step by step, then put the final answer in \\boxed{}." for q,_ in math]
        outs=llm.generate([tmpl(p) for p in prompts],sp)
        for (q,gold),pr,o in zip(math,prompts,outs):
            txt=o.outputs[0].text
            if o.outputs[0].finish_reason=="length": continue
            pred=B.extract_boxed(VE.post_think(txt)); gboxed=B.extract_boxed(str(gold)) or str(gold)
            if B.math_eq(pred,gboxed):
                kept.append({"domain":"math","prompt":pr,"target":txt})
        nm=sum(k["domain"]=="math" for k in kept); print(f"[cache] math kept {nm}/{len(math)} ({time.time()-t0:.0f}s)",flush=True)
    # code
    code=load_code(args.code_n)
    if code:
        prompts=[f"{t}\nYour function must pass:\n{tl[0]}\n\nReason if needed, then give the FULL function in a ```python block." for t,tl,_ in code]
        outs=llm.generate([tmpl(p) for p in prompts],sp)
        for (text,tl,setup),o,pr in zip(code,outs,prompts):
            txt=o.outputs[0].text
            if o.outputs[0].finish_reason=="length": continue
            m=re.findall(r"```(?:python)?\s*(.+?)```",VE.post_think(txt),re.S)
            codeblk=m[-1] if m else ""
            if codeblk and B.mbpp_run(codeblk,tl,[setup] if setup else []):
                kept.append({"domain":"code","prompt":pr,"target":txt})
        nc=sum(k["domain"]=="code" for k in kept); print(f"[cache] code kept {nc} ({time.time()-t0:.0f}s)",flush=True)
    with open(args.out,"w") as f:
        for k in kept: f.write(json.dumps(k)+"\n")
    print(f"[cache] DONE kept={len(kept)} -> {args.out}",flush=True)

if __name__=="__main__": main()
