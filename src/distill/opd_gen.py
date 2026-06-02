#!/usr/bin/env python3
"""Lightning-OPD Phase A: the STUDENT generates on-policy rollouts on math prompts (vLLM), each verified for
correctness (math_eq on boxed answer). Records prompt_ids + gen_token_ids + reward -> rollouts.jsonl, which
Phase B (teacher scoring) and Phase C (KL training) consume. Sampling matches cache_teacher (T=0.6/top_p0.95/
top_k20)."""
import argparse, json, time
from vllm import LLM, SamplingParams
import bench8_loaders as B, vllm_eval as VE, cache_teacher as CT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)          # the (SFT-seeded / prev-iter) merged student ckpt
    ap.add_argument("--out", required=True)            # rollouts.jsonl
    ap.add_argument("--n", type=int, default=256)      # prompts
    ap.add_argument("--k", type=int, default=4)        # rollouts per prompt
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_tokens + 2048,
              max_num_seqs=64, gpu_memory_utilization=0.92, trust_remote_code=True,
              disable_custom_all_reduce=True)  # NCCL all-reduce (lossless); avoids custom_all_reduce IPC race
    tok = llm.get_tokenizer()
    math = CT.load_math(args.n)
    prompts = [q + "\n\nReason step by step, then put the final answer in \\boxed{}." for q, _ in math]
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for p in prompts]
    sp = SamplingParams(n=args.k, temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens)
    outs = llm.generate(texts, sp)
    ncorr = ntrunc = nrow = 0
    with open(args.out, "w") as f:
        for pid, (o, (q, gold)) in enumerate(zip(outs, math)):
            pids = list(o.prompt_token_ids)
            for k, out in enumerate(o.outputs):
                gen_ids = list(out.token_ids)
                correct = bool(B.math_eq(B.extract_boxed(VE.post_think(out.text)), str(gold)))
                trunc = (out.finish_reason == "length")
                ncorr += correct; ntrunc += trunc; nrow += 1
                f.write(json.dumps({"pid": pid, "k": k, "prompt_ids": pids, "gen_token_ids": gen_ids,
                                    "correct": correct, "finish_reason": out.finish_reason}) + "\n")
    print(f"[opd_gen] DONE {nrow} rollouts ({100*ncorr/max(1,nrow):.1f}% correct, {ntrunc} truncated) "
          f"-> {args.out} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
