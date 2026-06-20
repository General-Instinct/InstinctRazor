#!/usr/bin/env python3
"""Probe per-(layer,expert) routing statistics for MoE bit-allocation, in ONE calibration pass.

Hooks every Qwen3_5MoeTopKRouter (`...mlp.gate`) and accumulates, per layer x expert (E=256):
  - freq:  # tokens that selected the expert (top-k membership count)        [expert-frequency-aware]
  - wmass: sum of router softmax weight assigned to the expert               [router-load-aware]
  - asal:  sum over routed tokens of ||hidden||_2 (input activation norm)    [activation-salience-aware]
Also reports load-balance skew (entropy / max-share) and throughput. Writes results/moe_probe_<tag>.json.

This is the foundation for moe_alloc.py: cold/rarely-routed experts can be quantized hard for ~free.
"""
import argparse, json, time, os
import torch
import torch.nn.functional as F

CALIB_INLINE = [
    "The integral of x^2 from 0 to 1 is 1/3. To see this, recall the power rule for integration.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "The French Revolution began in 1789 and fundamentally transformed the political landscape of Europe.",
    "Solve for x: 3x + 7 = 22. Subtract 7 from both sides to get 3x = 15, then divide by 3 to find x = 5.",
]

def build_calib(n_seq, seqlen, tok):
    texts = []
    try:
        from datasets import load_dataset
        try:
            m = load_dataset("HuggingFaceH4/MATH-500", split="test")
            texts += [m[i]["problem"] + "\n" + m[i]["solution"] for i in range(min(n_seq // 3, len(m)))]
        except Exception as e: print("calib math skip:", e, flush=True)
        try:
            h = load_dataset("openai/openai_humaneval", split="test")
            texts += [h[i]["prompt"] + h[i]["canonical_solution"] for i in range(min(n_seq // 3, len(h)))]
        except Exception as e: print("calib code skip:", e, flush=True)
        try:
            a = load_dataset("yahma/alpaca-cleaned", split=f"train[:{n_seq}]")
            texts += [(r["instruction"] + "\n" + (r["output"] or "")) for r in a]
        except Exception as e: print("calib alpaca skip:", e, flush=True)
    except Exception as e:
        print("datasets unavailable:", e, flush=True)
    if len(texts) < n_seq:
        texts += CALIB_INLINE * ((n_seq - len(texts)) // len(CALIB_INLINE) + 1)
    texts = texts[:n_seq]
    # no padding: tokenize each sequence separately so every counted position is a real token
    seqs = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seqlen)["input_ids"]
        if ids.shape[1] >= 8:
            seqs.append(ids)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--tag", default="122b")
    ap.add_argument("--n-seq", type=int, default=48)
    ap.add_argument("--seqlen", type=int, default=384)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--max-mem-gib", type=int, default=74)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/moe_probe_{args.tag}.json"
    os.makedirs("results", exist_ok=True)

    import model_adapters as MA
    from transformers import AutoProcessor
    n = torch.cuda.device_count()
    print(f"[probe] loading {args.model} across {n} GPUs ...", flush=True)
    t0 = time.time()
    model, adapter = MA.load_model(args.model, max_mem_gib=args.max_mem_gib)
    print(f"[probe] loaded ({type(adapter).__name__}) in {time.time()-t0:.0f}s", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = adapter.text_config(model.config)
    E = cfg.num_experts
    nL = cfg.num_hidden_layers
    freq = torch.zeros(nL, E, dtype=torch.float64)
    wmass = torch.zeros(nL, E, dtype=torch.float64)
    asal = torch.zeros(nL, E, dtype=torch.float64)

    # locate routers and map module -> layer idx
    import re
    hooks = []
    def make_hook(L):
        def hook(mod, inp, out):
            hs = inp[0]                              # (tokens, hidden)
            router_scores, router_indices = adapter.read_router_output(out)
            idx = router_indices.detach().reshape(-1).to("cpu")            # (tokens*topk)
            sc = router_scores.detach().float().reshape(-1).to("cpu")      # (tokens*topk)
            tn = hs.detach().float().norm(dim=-1)                          # (tokens,)
            topk = router_indices.shape[-1]
            tn_rep = tn.unsqueeze(-1).expand(-1, topk).reshape(-1).to("cpu")
            freq[L].index_add_(0, idx, torch.ones_like(sc, dtype=torch.float64))
            wmass[L].index_add_(0, idx, sc.double())
            asal[L].index_add_(0, idx, tn_rep.double())
        return hook
    for name, mod in model.named_modules():
        if adapter.is_router(mod):
            L = int(re.search(r"layers\.(\d+)\.", name).group(1))
            hooks.append(mod.register_forward_hook(make_hook(L)))
    print(f"[probe] hooked {len(hooks)} routers (E={E}, layers={nL})", flush=True)

    seqs = build_calib(args.n_seq, args.seqlen, tok)
    dev0 = next(model.parameters()).device
    n_tok = int(sum(s.numel() for s in seqs))
    print(f"[probe] calib: {len(seqs)} seqs (no pad), {n_tok} real tokens", flush=True)

    t1 = time.time()
    with torch.no_grad():
        for i, b_ids in enumerate(seqs):
            model(input_ids=b_ids.to(dev0), use_cache=False, logits_to_keep=1)
            if (i + 1) % 8 == 0:
                print(f"  fwd {i+1}/{len(seqs)}", flush=True)
    dt = time.time() - t1
    for h in hooks: h.remove()

    tot_routed = freq.sum().item()
    # per-layer skew metrics
    p = (freq + 1e-9); p = p / p.sum(dim=1, keepdim=True)
    ent = (-(p * p.log()).sum(dim=1) / torch.log(torch.tensor(float(E)))).tolist()  # normalized entropy in [0,1]
    maxshare = (freq.max(dim=1).values / freq.sum(dim=1).clamp(min=1)).tolist()
    cold_frac = ((freq < (freq.mean(dim=1, keepdim=True) * 0.1)).float().mean(dim=1)).tolist()

    res = {"model": args.model, "num_experts": E, "num_layers": nL,
           "n_seq": args.n_seq, "seqlen": args.seqlen, "n_real_tokens": n_tok,
           "throughput_tok_s": n_tok / dt, "fwd_seconds": dt,
           "freq": freq.tolist(), "wmass": wmass.tolist(), "asal": asal.tolist(),
           "layer_norm_entropy": ent, "layer_maxshare": maxshare, "layer_cold_frac": cold_frac}
    json.dump(res, open(out, "w"))
    print(f"[probe] wrote {out}", flush=True)
    print(f"[probe] throughput {res['throughput_tok_s']:.0f} tok/s; total routed assignments {tot_routed:.0f}", flush=True)
    print(f"[probe] mean normalized routing entropy {sum(ent)/len(ent):.3f} (1=uniform); "
          f"mean max-expert share {sum(maxshare)/len(maxshare):.3f}; "
          f"mean cold-expert frac {sum(cold_frac)/len(cold_frac):.3f}", flush=True)
    print(f"[probe] peak GPU mem (GiB): "
          + ", ".join(f"{torch.cuda.max_memory_allocated(i)/2**30:.1f}" for i in range(n)), flush=True)


if __name__ == "__main__":
    main()
