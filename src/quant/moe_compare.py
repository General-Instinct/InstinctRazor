#!/usr/bin/env python3
"""Apples-to-apples expert-quant METHOD comparison: how good is OUR quant vs AWQ / GPTQ / RTN?

Fixes everything (model, experts sampled, bit-width, group, protection) and varies ONLY the algorithm
used to quantize the fused per-expert weight slices. Metric = per-expert END-TO-END output NMSE on REAL
calibration activations (the dispatched tokens each expert actually sees), i.e.
    o_fp = (act(Xe @ Wgu^T).gate * .up) @ Wdn^T          # full-precision expert output
    o_m  = (act(Xe @ Wgu_q^T).gate * .up) @ Wdn_q^T       # method-m quantized expert output
    NMSE = ||o_m - o_fp||^2 / ||o_fp||^2
This is the kernel-level "reconstruction quality" of each method; capability is measured separately by the
eval harness (and prior findings: at ~3b the methods tend to TIE on capability even when NMSE differs).

Methods:
  ours_clip   = the SHIPPED recipe: symmetric int + MSE clip-search (moe_quant.fakequant, clip on)
  ours_absmax = symmetric int, plain absmax (clip off) -> isolates the clip-search gain
  rtn_asym    = asymmetric uint round-to-nearest (moe_ptq) -> no-calibration baseline
  awq         = activation-aware per-in-channel scaling + RTN (moe_ptq, needs calib)
  gptq        = Hessian error-compensated (moe_ptq, needs calib)

Calibration capture (model-general, via the ModelAdapter): for each sampled layer we hook the router
(adapter.read_router_output -> top-k indices) + the MoE block input (hidden states); per expert e we
gather the rows routed to e (capped). Works for any FUSED-expert MoE (Qwen3.5/3.6, OLMoE, ...).

  python src/quant/moe_compare.py --model Qwen/Qwen3.6-35B-A3B --bits 3 --n-layers 4 --experts-per-layer 32
"""
import argparse, json, os, re, time
import numpy as np
import torch
import torch.nn.functional as F
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import transformers; transformers.logging.set_verbosity_error()
import moe_quant as MQ
import model_adapters as MA
import moe_ptq as PTQ
import moe_probe as PB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--clip-steps", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=4, help="# layers sampled, spread across depth")
    ap.add_argument("--experts-per-layer", type=int, default=32, help="experts sampled per layer (0=all)")
    ap.add_argument("--calib-n", type=int, default=32)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--max-tok-per-expert", type=int, default=256)
    ap.add_argument("--max-mem-gib", type=int, default=76)
    ap.add_argument("--compute-device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/quant_compare_{re.sub(r'[^A-Za-z0-9]+','_',args.model)}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cdev = args.compute_device

    t0 = time.time()
    model, adapter = MA.load_model(args.model, max_mem_gib=args.max_mem_gib)
    print(f"[compare] loaded {type(adapter).__name__} in {time.time()-t0:.0f}s", flush=True)
    cfg = adapter.text_config(model.config)
    nL = cfg.num_hidden_layers
    from transformers import AutoProcessor, AutoTokenizer
    try:
        proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # sampled layers spread across depth
    sampled = sorted(set(int(round(i * (nL - 1) / max(1, args.n_layers - 1))) for i in range(args.n_layers)))
    print(f"[compare] sampling layers {sampled} of {nL}", flush=True)

    # locate, per sampled layer, the router module + its parent MoE block + the experts module
    routers, mlp_blocks, experts_mods = {}, {}, {}
    name_of = {}
    for name, mod in model.named_modules():
        if adapter.is_router(mod):
            L = adapter.layer_idx_of(name)
            if L in sampled:
                routers[L] = mod
                parent = name.rsplit(".", 1)[0]            # ...mlp.gate -> ...mlp
                mlp_blocks[L] = model.get_submodule(parent)
                experts_mods[L] = model.get_submodule(parent + ".experts")
                name_of[L] = parent
    assert routers, "no routers found via adapter.is_router"

    # capture buffers
    # Capture the experts module's INPUTS (model-general): scan its forward args for the hidden-states
    # tensor (float, last dim == hidden_size) and the routing-index tensor (integer). Avoids depending on
    # any router-output arity. Per expert e -> the rows whose top-k index includes e.
    H = cfg.hidden_size
    cap_in = {}     # L -> hidden states [T,H]
    cap_idx = {}    # L -> top-k indices [T,k]
    hooks = []
    def mk_capture(L):
        def pre(mod, a, kw):
            hs = idx = None
            for v in list(a) + list(kw.values()):
                if not torch.is_tensor(v):
                    continue
                if hs is None and v.is_floating_point() and v.shape[-1] == H:
                    hs = v.detach().reshape(-1, H)
                elif idx is None and not v.is_floating_point() and v.dim() >= 1:
                    idx = v.detach()
            if hs is not None and idx is not None and idx.numel() % hs.shape[0] == 0:
                cap_in[L] = hs
                cap_idx[L] = idx.reshape(hs.shape[0], -1)
        return pre
    for L in sampled:
        hooks.append(experts_mods[L].register_forward_pre_hook(mk_capture(L), with_kwargs=True))

    # per-(L,e) accumulated calib inputs (on CPU, capped)
    E = cfg.num_experts
    acc = {L: [[] for _ in range(E)] for L in sampled}
    filled = {L: 0 for L in sampled}
    seqs = PB.build_calib(args.calib_n, args.calib_len, tok)
    dev0 = next(model.parameters()).device
    print(f"[compare] calib: {len(seqs)} seqs; capturing dispatched tokens (cap {args.max_tok_per_expert}/expert)", flush=True)
    with torch.no_grad():
        for si, ids in enumerate(seqs):
            cap_in.clear(); cap_idx.clear()
            model(input_ids=ids.to(dev0), use_cache=False, logits_to_keep=1)
            for L in sampled:
                hs, idx = cap_in.get(L), cap_idx.get(L)
                if hs is None or idx is None:
                    continue
                T = min(hs.shape[0], idx.shape[0])
                hs = hs[:T].float().cpu(); idx = idx[:T].cpu()
                for e in range(E):
                    cur = sum(x.shape[0] for x in acc[L][e])
                    if cur >= args.max_tok_per_expert:
                        continue
                    rows = (idx == e).any(dim=-1).nonzero(as_tuple=True)[0]
                    if rows.numel():
                        acc[L][e].append(hs[rows[: args.max_tok_per_expert - cur]])
            if (si + 1) % 8 == 0:
                print(f"  calib fwd {si+1}/{len(seqs)}", flush=True)
    for h in hooks:
        h.remove()

    # choose experts to compare (those with enough calib tokens), sample per layer
    methods = ["ours_clip", "ours_absmax", "rtn_asym", "awq", "gptq"]
    nmse = {m: [] for m in methods}
    per_layer = {}
    for L in sampled:
        em = experts_mods[L]
        act_fn = getattr(em, "act_fn", F.silu)
        gu_all = em.gate_up_proj    # [E, 2I, H]
        dn_all = em.down_proj       # [E, H, I]
        # experts with >=64 calib tokens
        cand = [e for e in range(E) if acc[L][e] and sum(x.shape[0] for x in acc[L][e]) >= 64]
        if args.experts_per_layer and len(cand) > args.experts_per_layer:
            step = len(cand) / args.experts_per_layer
            cand = [cand[int(i * step)] for i in range(args.experts_per_layer)]
        layer_nmse = {m: [] for m in methods}
        for e in cand:
            Xall = torch.cat(acc[L][e], 0)[: args.max_tok_per_expert].to(cdev)       # [n,H]
            ncal = Xall.shape[0] // 2                                                 # held-out split (fair to calib methods)
            Xcal, Xev = Xall[:ncal], Xall[ncal:]
            Wgu = gu_all[e].detach().to(cdev).float()                                 # [2I,H]
            Wdn = dn_all[e].detach().to(cdev).float()                                 # [H,I]
            def expert_out(X, wgu, wdn):
                o = X @ wgu.t(); g, u = o.chunk(2, dim=-1); h = act_fn(g) * u
                return h @ wdn.t(), h
            o_fp, _ = expert_out(Xev, Wgu, Wdn)                                        # FP output on HELD-OUT tokens
            h_cal = expert_out(Xcal, Wgu, Wdn)[1]                                      # FP intermediate for down-proj calib
            denom = o_fp.pow(2).mean().clamp(min=1e-12)
            for m in methods:
                if m == "ours_clip":
                    MQ.set_clip_search(args.clip_steps)
                    wgu_q = MQ.fakequant(Wgu, args.bits, args.group); wdn_q = MQ.fakequant(Wdn, args.bits, args.group)
                    MQ.set_clip_search(0)
                elif m == "ours_absmax":
                    MQ.set_clip_search(0)
                    wgu_q = MQ.fakequant(Wgu, args.bits, args.group); wdn_q = MQ.fakequant(Wdn, args.bits, args.group)
                elif m == "rtn_asym":
                    wgu_q = PTQ.rtn_asym(Wgu, args.bits, args.group); wdn_q = PTQ.rtn_asym(Wdn, args.bits, args.group)
                elif m == "awq":            # calibrate on Xcal/h_cal, evaluate on Xev
                    wgu_q = PTQ.awq_quant(Wgu, Xcal, args.bits, args.group); wdn_q = PTQ.awq_quant(Wdn, h_cal, args.bits, args.group)
                elif m == "gptq":
                    wgu_q = PTQ.gptq_quant(Wgu, Xcal, args.bits, args.group); wdn_q = PTQ.gptq_quant(Wdn, h_cal, args.bits, args.group)
                o_m, _ = expert_out(Xev, wgu_q.float(), wdn_q.float())
                val = ((o_m - o_fp).pow(2).mean() / denom).item()
                nmse[m].append(val); layer_nmse[m].append(val)
        per_layer[L] = {m: float(np.mean(v)) if v else None for m, v in layer_nmse.items()}
        print(f"[compare] layer {L}: {len(cand)} experts | " +
              " ".join(f"{m}={np.mean(layer_nmse[m]):.4f}" for m in methods if layer_nmse[m]), flush=True)

    summary = {m: {"mean_nmse": float(np.mean(nmse[m])), "median_nmse": float(np.median(nmse[m])),
                   "n_experts": len(nmse[m])} for m in methods if nmse[m]}
    print("\n=== EXPERT OUTPUT NMSE @ int%d, group %d (lower = better reconstruction) ===" % (args.bits, args.group))
    base = summary["ours_clip"]["mean_nmse"] if "ours_clip" in summary else None
    for m in methods:
        if m in summary:
            rel = f"  ({summary[m]['mean_nmse']/base:.2f}x ours_clip)" if base else ""
            print(f"  {m:12s} mean={summary[m]['mean_nmse']:.4f} median={summary[m]['median_nmse']:.4f} "
                  f"(n={summary[m]['n_experts']}){rel}", flush=True)
    res = {"model": args.model, "bits": args.bits, "group": args.group, "clip_steps": args.clip_steps,
           "sampled_layers": sampled, "summary": summary, "per_layer": per_layer}
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n[compare] wrote {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
