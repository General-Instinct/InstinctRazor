#!/usr/bin/env python3
"""Faithful multimodal eval on Gemma-4's reported vision benchmarks (MMMU, MMMU-Pro, MATH-Vision, MedXpertQA-MM)
via vLLM. Images passed as base64 data-URIs in llm.chat messages (portable). Validate by reproducing Gemma's
official numbers, then compare our 47GB model. MC -> letter parse; MATH-Vision -> boxed/letter."""
import argparse, json, re, io, base64, time

def img_uri(im):
    im = im.convert("RGB")
    if max(im.size) > 1024:
        r = 1024/max(im.size); im = im.resize((int(im.size[0]*r), int(im.size[1]*r)))
    b = io.BytesIO(); im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--benches", default="mmmu,mmmu_pro,mathvision,medxpert")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--think", action="store_true", default=True)
    ap.add_argument("--enforce-eager", action="store_true")  # A4B/Gemma4 long-context stability
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    import bench_mm as M, vllm_eval as VE, bench8_loaders as B
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.budget+4096,
              max_num_seqs=32, gpu_memory_utilization=0.92, trust_remote_code=True,
              limit_mm_per_prompt={"image": 8}, disable_custom_all_reduce=True, enforce_eager=args.enforce_eager)
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.budget)

    def run(items, kind):  # items: (q, [imgs], opts, gold)
        msgs = []
        for q, imgs, opts, gold in items:
            content = [{"type": "image_url", "image_url": {"url": img_uri(im)}} for im in imgs[:8]]
            text = q
            if opts:
                text += "\n" + "\n".join(f"{'ABCDEFGHIJ'[i]}. {c}" for i, c in enumerate(opts))
                text += "\n\nReason step by step, then end with 'Answer: <letter>'."
            else:
                text += "\n\nReason step by step, then put the final answer in \\boxed{}."
            content.append({"type": "text", "text": text})
            msgs.append([{"role": "user", "content": content}])
        outs = llm.chat(msgs, sp)
        corr = trunc = 0
        for i, o in enumerate(outs):
            txt = o.outputs[0].text; t = VE.post_think(txt)
            trunc += (o.outputs[0].finish_reason == "length")
            q, imgs, opts, gold = items[i]
            if opts:
                k = len(opts); pred = VE.parse_letter(t, k)
                gi = gold if isinstance(gold, int) else ("ABCDEFGHIJ".index(str(gold).strip().upper()) if str(gold).strip().upper() in "ABCDEFGHIJ" else -1)
                corr += (pred == gi)
            else:
                corr += B.math_eq(B.extract_boxed(t), str(gold))
        return 100.0*corr/len(items), trunc, len(items)

    rec = {"tag": args.tag, "model": args.model}
    for b in args.benches.split(","):
        try:
            if b == "mmmu": items = M.build_mmmu(args.n)
            elif b == "mmmu_pro": items = M.build_mmmu(args.n, pro=True)
            elif b == "mathvision": items = M.build_mathvision(args.n)
            elif b == "medxpert": items = M.build_medxpert_mm(args.n)
            else: continue
            acc, tr, n = run(items, b)
            rec[f"{b}_acc"] = acc; rec[f"{b}_n"] = n; rec[f"{b}_trunc"] = tr
            print(f"[mm] {b}: acc={acc:.1f} n={n} trunc={tr}", flush=True)
        except Exception as e:
            print(f"[mm] {b} FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
    import os; os.makedirs("results/mm", exist_ok=True)
    json.dump(rec, open(f"results/mm/{args.tag}.json", "w"), indent=2)
    print(f"[mm] DONE {args.tag}", flush=True)

if __name__ == "__main__":
    main()
