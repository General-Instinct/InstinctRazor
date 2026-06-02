#!/usr/bin/env python3
"""LCB gate diagnostic: why does the fixed eval read ~12pt under official? Generate teacher on n LCB problems,
then per problem dump: code-block present in post_think vs full output, pass with each extraction, + truncation.
Tests whether the solution lives in/around <think> and post_think-only extraction misses it. (32k for stability;
the extraction issue is budget-independent.)"""
import argparse, json, re


def codeblock(s):
    m = re.findall(r"```(?:python)?\s*(.+?)```", s, re.S)
    return m[-1] if m else None


def main():
    from vllm import LLM, SamplingParams
    import bench8_loaders as B, vllm_eval as VE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="./logs/lcb_diag_dump.json")
    args = ap.parse_args()

    llm = LLM(model=args.model, tensor_parallel_size=4, max_model_len=40960, gpu_memory_utilization=0.90,
              max_num_seqs=64, trust_remote_code=True, disable_custom_all_reduce=True)
    tok = llm.get_tokenizer()
    it = B.build_lcb(args.n)
    texts = []
    for q, _ in it:
        m = [{"role": "user", "content": q}]
        try: texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True))
        except TypeError: texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
    outs = llm.generate(texts, sp)

    rows = []; n_pt = n_full = n_recov = 0
    for i, o in enumerate(outs):
        full = o.outputs[0].text; fr = o.outputs[0].finish_reason
        pt = VE.post_think(full)
        code_pt = codeblock(pt); code_full = codeblock(full)
        rec = it[i][1]
        pass_pt = bool(B.lcb_run(code_pt, rec)) if code_pt else False
        pass_full = bool(B.lcb_run(code_full, rec)) if code_full else False
        n_pt += pass_pt; n_full += pass_full
        if pass_full and not pass_pt: n_recov += 1
        rows.append({"i": i, "type": rec["type"], "trunc": fr == "length", "has_think": "</think>" in full,
                     "code_in_pt": code_pt is not None, "code_in_full": code_full is not None,
                     "pass_pt": pass_pt, "pass_full": pass_full})
    summary = {"n": len(it), "pass_postthink": n_pt, "pass_fulloutput": n_full, "recovered_by_full": n_recov,
               "trunc": sum(r["trunc"] for r in rows),
               "no_code_in_pt": sum(1 for r in rows if not r["code_in_pt"]),
               "no_code_in_full": sum(1 for r in rows if not r["code_in_full"])}
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print("SUMMARY:", json.dumps(summary))


if __name__ == "__main__":
    main()
