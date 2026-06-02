#!/usr/bin/env python3
"""GGUF accuracy eval for qwen3_5_moe via llama.cpp (vLLM cannot load this arch's GGUF: transformers'
GGUF arch allowlist stops at qwen3_moe, no expert-remap, hybrid SSM unsupported). This is a thin adapter:
it reuses our EXACT prompt-build (vllm_eval.build_mmlu_pro / moe_eval.build_gpqa) and grading
(post_think + parse_letter + score), templates the chat ourselves with the HF tokenizer (enable_thinking),
and drives a running llama-server /completion endpoint. Output JSON keys match vllm_eval8.py so the numbers
are SAME-HARNESS comparable to results/vllm_eval/*.json (the fake-quant clip baselines).

Run llama-server first (see pipelines/eval_gguf.sh), then:
  python llamacpp_eval8.py --tokenizer Qwen/Qwen3.5-122B-A10B --tag gguf_iq3 --benchmarks mmlu_pro,gpqa \
      --port 8099 --budget 32768 --mmlu-n 150 --gpqa-n 198 --parallel 8
"""
import argparse, json, os, urllib.request, concurrent.futures as cf
import vllm_eval as VE, moe_eval as EV

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-122B-A10B")  # HF tokenizer for templating (NOT the GGUF's)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--benchmarks", default="mmlu_pro,gpqa")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--mmlu-n", type=int, default=150); ap.add_argument("--gpqa-n", type=int, default=198)
    ap.add_argument("--parallel", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    url = f"http://{args.host}:{args.port}/completion"

    def template(prompt):
        m = [{"role": "user", "content": prompt}]
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    def gen_one(prompt):
        body = json.dumps({"prompt": template(prompt), "n_predict": args.budget, "temperature": 0.6,
                           "top_p": 0.95, "top_k": 20, "seed": args.seed, "cache_prompt": False}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read())
        # stopped_limit==True => hit n_predict => truncated => NOT finished (matches finish_reason=="length")
        return d.get("content", ""), (not d.get("stopped_limit", False))

    def g(prompts):
        outs = [None] * len(prompts); fins = [False] * len(prompts)
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(gen_one, p): i for i, p in enumerate(prompts)}
            done = 0
            for f in cf.as_completed(futs):
                i = futs[f]
                try: outs[i], fins[i] = f.result()
                except Exception as e: outs[i], fins[i] = f"[ERR {type(e).__name__}: {str(e)[:80]}]", True
                done += 1
                if done % 20 == 0: print(f"  ... {done}/{len(prompts)}", flush=True)
        tr = sum(1 for x in fins if not x)
        return outs, tr, fins

    def score(correct, fin):
        n = len(correct); nf = sum(1 for f in fin if f)
        acc = 100.0 * sum(correct) / n if n else 0.0
        accf = 100.0 * sum(c for c, f in zip(correct, fin) if f) / nf if nf else 0.0
        return acc, accf, nf

    rec = {"tag": args.tag, "tokenizer": args.tokenizer, "backend": "llama.cpp", "budget": args.budget}
    bset = args.benchmarks.split(",")
    if "mmlu_pro" in bset:
        it = VE.build_mmlu_pro(args.mmlu_n)
        ps = [q + "\n" + "\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i, c in enumerate(o)) + "\n\nReason, then end with 'Answer: <letter>'." for q, o, _ in it]
        out, tr, fin = g(ps); cor = [VE.parse_letter(VE.post_think(o), len(it[i][1])) == it[i][2] for i, o in enumerate(out)]
        acc, accf, nf = score(cor, fin)
        rec.update(mmlu_pro_acc=acc, mmlu_pro_acc_finished=accf, mmlu_pro_n=len(it), mmlu_pro_n_finished=nf, mmlu_pro_trunc=tr)
        print(f"[gguf] MMLU-Pro {acc:.1f} (finished {accf:.1f} | n={len(it)} nf={nf} tr={tr})", flush=True)
    if "gpqa" in bset:
        gp, _ = EV.build_gpqa(args.gpqa_n)
        ps = [q + "\n\nReason, then end with 'Answer: <letter>'." for q, _ in gp]
        out, tr, fin = g(ps); cor = [VE.parse_letter(VE.post_think(o), 4) == gp[i][1] for i, o in enumerate(out)]
        acc, accf, nf = score(cor, fin)
        rec.update(gpqa_acc=acc, gpqa_acc_finished=accf, gpqa_n=len(gp), gpqa_n_finished=nf, gpqa_trunc=tr)
        print(f"[gguf] GPQA-D {acc:.1f} (finished {accf:.1f} | n={len(gp)} nf={nf} tr={tr})", flush=True)

    os.makedirs("results/vllm_eval", exist_ok=True)
    json.dump(rec, open(f"results/vllm_eval/{args.tag}.json", "w"), indent=2)
    print(f"[gguf] DONE {args.tag} -> results/vllm_eval/{args.tag}.json", flush=True)

if __name__ == "__main__":
    main()
