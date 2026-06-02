"""Lightning-OPD Phase B: the BF16 TEACHER scores the student's cached rollouts via vLLM prompt_logprobs
(prefill-only), caching top-k teacher logprobs/ids per completion token -> teacher_lp.npz (the KL targets).
Teacher and student share the tokenizer/vocab (same base), so token ids transfer verbatim."""
import argparse, json, time
import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--rollouts", required=True)       # rollouts.jsonl from Phase A
    ap.add_argument("--out", required=True)            # teacher_lp.npz
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=18432)
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    rolls = [json.loads(l) for l in open(args.rollouts)]
    rolls = [r for r in rolls if len(r["prompt_ids"]) + len(r["gen_token_ids"]) <= args.max_len
             and len(r["gen_token_ids"]) >= 1]
    # prompt_logprobs over the 248K vocab spikes ~ max_num_batched_tokens * vocab * 4B; bound the chunk to 2048
    # (~2GB) and leave transient headroom (gpu_mem 0.85) so the logprobs alloc never OOMs against the KV cache.
    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, dtype="bfloat16", max_model_len=args.max_len,
              max_num_seqs=16, gpu_memory_utilization=0.85, max_num_batched_tokens=2048,
              enable_prefix_caching=True, trust_remote_code=True,
              disable_custom_all_reduce=True)  # NCCL all-reduce (lossless); avoids custom_all_reduce IPC race
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.k, detokenize=False)
    prompts = [TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["gen_token_ids"]) for r in rolls]
    outs = llm.generate(prompts, sp)
    records = []
    for r, o in zip(rolls, outs):
        plen = len(r["prompt_ids"]); full = r["prompt_ids"] + r["gen_token_ids"]
        plp = o.prompt_logprobs                         # len(full); [0] is None
        t_idx, t_lp = [], []
        for pos in range(plen, len(full)):
            dist = plp[pos]
            if not dist:
                continue
            top = sorted(dist.items(), key=lambda kv: kv[1].logprob, reverse=True)[:args.k]
            ids = [tid for tid, _ in top]; lps = [lg.logprob for _, lg in top]
            while len(ids) < args.k:                    # pad short rows
                ids.append(ids[-1] if ids else 0); lps.append(-30.0)
            t_idx.append(ids); t_lp.append(lps)
        if not t_idx:
            continue
        records.append({"pid": int(r["pid"]), "k": int(r["k"]),
                        "t_idx": np.asarray(t_idx, dtype=np.int32),
                        "t_lp": np.asarray(t_lp, dtype=np.float16), "plen": plen})
    np.savez(args.out, records=np.asarray(records, dtype=object))
    print(f"[opd_score] DONE {len(records)} scored -> {args.out} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
