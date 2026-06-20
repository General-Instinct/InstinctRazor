#!/usr/bin/env python3
"""Build a FULL dequant-bf16 checkpoint where the routed experts are quantized by a CHOSEN method
(awq / gptq / rtn), at matched bit-width + group, with the same int4 backbone + protection as the
shipped clip recipe. Lets us run a controlled CAPABILITY A/B (our clip int3 vs AWQ int3) on the same
eval harness — isolating the expert-quant ALGORITHM at a matched bit budget (no bit-width confound).

Pipeline: load (fused-expert MoE) -> capture per-expert calibration activations (hook experts inputs)
-> per-expert AWQ/GPTQ bake of gate_up + down -> int4 backbone via apply_ptq (experts left untouched)
-> save_pretrained. Reuses moe_ptq.{awq_quant,gptq_quant,rtn_asym} and moe_quant.apply_ptq.

  python src/quant/moe_quant_method.py --model Qwen/Qwen3.6-35B-A3B --method awq --bits 3 \
      --out models/q36_awq3b
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import transformers; transformers.logging.set_verbosity_error()
import moe_quant as MQ
import model_adapters as MA
import moe_ptq as PTQ
import moe_probe as PB


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["awq", "gptq", "rtn"], default="awq")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--linear-bits", type=float, default=4.0)
    ap.add_argument("--calib-n", type=int, default=24)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--max-tok-per-expert", type=int, default=64)
    ap.add_argument("--max-mem-gib", type=int, default=76)
    ap.add_argument("--compute-device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cdev = args.compute_device
    t0 = time.time()
    model, adapter = MA.load_model(args.model, max_mem_gib=args.max_mem_gib)
    cfg = adapter.text_config(model.config)
    E, nL, H = cfg.num_experts, cfg.num_hidden_layers, cfg.hidden_size
    print(f"[qmethod] loaded {type(adapter).__name__} (E={E} nL={nL} H={H}) in {time.time()-t0:.0f}s", flush=True)
    from transformers import AutoProcessor, AutoTokenizer
    try:
        proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # locate experts modules per layer (fused) + map layer idx
    experts_by_L = {}
    for name, mod in model.named_modules():
        if name.endswith(".experts") and hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj"):
            experts_by_L[adapter.layer_idx_of(name)] = mod
    assert experts_by_L, "no fused experts modules found (this builder is fused-expert only)"
    print(f"[qmethod] hooking {len(experts_by_L)} expert modules to capture calib inputs", flush=True)

    # capture per-(L,e) expert input hidden states (the dispatched tokens), capped
    acc = {L: [[] for _ in range(E)] for L in experts_by_L}
    cap = {}
    hooks = []
    def mk(L):
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
                cap[L] = (hs, idx.reshape(hs.shape[0], -1))
        return pre
    for L, m in experts_by_L.items():
        hooks.append(m.register_forward_pre_hook(mk(L), with_kwargs=True))

    seqs = PB.build_calib(args.calib_n, args.calib_len, tok)
    dev0 = next(model.parameters()).device
    for si, ids in enumerate(seqs):
        cap.clear()
        model(input_ids=ids.to(dev0), use_cache=False, logits_to_keep=1)
        for L, (hs, idx) in cap.items():
            hs = hs.float().cpu(); idx = idx.cpu()
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

    # per-expert bake with the chosen method (calib-free RTN fallback for experts with no tokens)
    MNAME = args.method
    t1 = time.time(); nbaked = 0
    for L, em in experts_by_L.items():
        gu = em.gate_up_proj; dn = em.down_proj                  # [E,2I,H], [E,H,I]
        for e in range(E):
            xs = acc[L][e]
            Wgu = gu[e].detach().to(cdev).float()
            Wdn = dn[e].detach().to(cdev).float()
            if xs and MNAME in ("awq", "gptq"):
                Xe = torch.cat(xs, 0).to(cdev)
                h = F.silu((Xe @ Wgu.t()).chunk(2, -1)[0]) * (Xe @ Wgu.t()).chunk(2, -1)[1]
                fn = PTQ.awq_quant if MNAME == "awq" else PTQ.gptq_quant
                Wgu_q = fn(Wgu, Xe, args.bits, args.group); Wdn_q = fn(Wdn, h, args.bits, args.group)
            else:
                Wgu_q = PTQ.rtn_asym(Wgu, args.bits, args.group); Wdn_q = PTQ.rtn_asym(Wdn, args.bits, args.group)
            gu[e] = Wgu_q.to(gu.dtype).to(gu.device)
            dn[e] = Wdn_q.to(dn.dtype).to(dn.device)
        nbaked += 1
        if nbaked % 8 == 0:
            print(f"  baked {MNAME} experts: layer {L} ({nbaked}/{len(experts_by_L)}, {time.time()-t1:.0f}s)", flush=True)

    # int4 backbone via apply_ptq with experts UNTOUCHED (default_expert_bits=None -> skip experts)
    spec = MQ.AllocSpec(default_expert_bits=None, default_linear_bits=args.linear_bits, group=args.group)
    MQ.apply_ptq(model, spec, verbose=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    try:
        transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    except Exception as e:
        print(f"[qmethod] tokenizer save warn: {e}", flush=True)
    try:
        transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    except Exception as e:
        print(f"[qmethod] processor save warn (ok if text-only): {str(e)[:80]}", flush=True)
    json.dump({"method": MNAME, "expert_bits": args.bits, "linear_bits": args.linear_bits, "group": args.group,
               "calib_n": args.calib_n, "max_tok_per_expert": args.max_tok_per_expert},
              open(os.path.join(args.out, "_ptq_meta.json"), "w"), indent=2)
    print(f"[qmethod] DONE {MNAME}-{args.bits}b -> {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
