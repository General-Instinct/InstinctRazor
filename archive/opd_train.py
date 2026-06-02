#!/usr/bin/env python3
"""Lightning-OPD Phase C: on-policy reverse-KL distillation of the BF16 teacher into the quantized student via
per-expert LoRA+STE (footprint-preserving). Trains on the STUDENT's own rollouts (cached) using the teacher's
cached top-k logprobs as targets: loss = reverse-KL(student||teacher), per-token, length-normalized, with a
'rest' bucket for off-support mass; logits chunked over time to avoid the 248K-vocab OOM. Reuses moe_lora
unchanged. Inputs: rollouts_t.jsonl (student gens+reward) + teacher_lp_t.npz (teacher top-k). Outputs: persistent
adapter.pt + merged deployable ckpt (Phase-A fuel for next iter)."""
import argparse, json, os, time, random
import numpy as np, torch, torch.nn.functional as F
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import transformers; transformers.logging.set_verbosity_error()
import moe_lora as ML, moe_quant as MQ

V = 248320; VCHUNK = 128   # 128: bound the 248K-vocab logit/logprob scratch (~127MB fp32) to cut the forward peak

def experts_mods(model):
    return [m for m in model.modules() if m.__class__.__name__ == "Qwen3_5MoeExperts" and hasattr(m, "_lora")]

def text_backbone(model):
    """Return the module whose forward yields last_hidden_state (text decoder) + the lm_head."""
    lm = model.get_output_embeddings()
    base = model.model
    if hasattr(base, "language_model"):   # multimodal wrapper -> text submodule
        base = base.language_model
    return base, lm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--teacher-lp", required=True)
    ap.add_argument("--adapter-in", default="")
    ap.add_argument("--adapter-out", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=float, default=3.0); ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=1.0)   # KD temperature: KL between temp-1 dists (sampling temp only selects data)
    ap.add_argument("--alpha-ce", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--max-mem-gib", type=int, default=62)
    ap.add_argument("--correct-only", type=int, default=0); ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--reverse", type=int, default=1)   # 1=reverse-KL(student||teacher); 0=forward-KL (A/B)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    nd = torch.cuda.device_count(); t0 = time.time()
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory={i: f"{args.max_mem_gib}GiB" for i in range(nd)}, trust_remote_code=True)
    model.config.use_cache = False; model.gradient_checkpointing_enable()
    MQ.set_clip_search(0)
    trainable = ML.attach_expert_lora(model, bits=args.bits, group=args.group, rank=args.rank, scale=args.scale)
    mods = experts_mods(model)
    if args.adapter_in:
        sd = torch.load(os.path.join(args.adapter_in, "adapter.pt"), map_location="cpu")
        for m, d in zip(mods, sd):
            for k in ("Agu", "Bgu", "Adn", "Bdn"):
                getattr(m._lora, k).data.copy_(d[k].to(getattr(m._lora, k).device, getattr(m._lora, k).dtype))
        print(f"[opd] loaded adapter_in ({len(sd)} modules)", flush=True)
    backbone, lmhead = text_backbone(model)
    emb_dev = model.get_input_embeddings().weight.device

    # ---- load cached rollouts + teacher logprobs, aligned by (pid,k) ----
    npz = np.load(args.teacher_lp, allow_pickle=True)
    ro = {(r["pid"], r["k"]): r for r in (json.loads(l) for l in open(args.rollouts))}
    samples = []
    for entry in npz["records"]:
        key = (int(entry["pid"]), int(entry["k"]))
        if key not in ro: continue
        r = ro[key]
        if args.correct_only and not r.get("correct", False): continue
        ids = list(r["prompt_ids"]) + list(r["gen_token_ids"]); plen = len(r["prompt_ids"])
        if len(ids) > args.max_len or len(ids) <= plen + 1: continue
        ti = torch.tensor(np.asarray(entry["t_idx"]), dtype=torch.long, device=emb_dev)
        tl = torch.tensor(np.asarray(entry["t_lp"]), dtype=torch.float32, device=emb_dev)
        samples.append((ids, plen, ti, tl))
    print(f"[opd] {len(samples)} training rollouts (correct_only={args.correct_only})", flush=True)
    assert samples, "no samples"

    def opd_loss(ids, plen, t_idx, t_lp):
        x = torch.tensor([ids], device=emb_dev)
        hidden = backbone(input_ids=x).last_hidden_state[0]          # [L,H]
        H = hidden[plen:len(ids) - 1]                                # predict completion tokens
        t_idx = t_idx.to(H.device); t_lp = t_lp.to(H.device)         # device_map: KL targets must sit on the lm_head stage (not emb stage)
        k = t_idx.size(-1)
        # teacher logprobs were stored over completion span; align lengths defensively
        T = min(H.size(0), t_idx.size(0)); H = H[:T]; t_idx = t_idx[:T]; t_lp = t_lp[:T]
        p_t = F.softmax(t_lp, dim=-1)
        r = (1.0 - p_t.sum(-1)).clamp_min(0.0)
        log_rest = torch.log((r / (V - k)).clamp_min(1e-12))
        kl_sum = H.new_zeros(()).float()
        for c0 in range(0, T, VCHUNK):
            c = slice(c0, min(c0 + VCHUNK, T))
            z = lmhead(H[c]) / args.tau                              # [Tc,V] bf16
            lse = torch.logsumexp(z.float(), dim=-1)                 # [Tc]
            s_lp = z.gather(-1, t_idx[c]).float() - lse[:, None]     # [Tc,k] student logprob @ teacher ids
            q = s_lp.exp(); q_rest = (1.0 - q.sum(-1)).clamp_min(1e-9)
            if args.reverse:   # reverse-KL(student||teacher)
                kl = (q * (s_lp - torch.log(p_t[c].clamp_min(1e-12)))).sum(-1) \
                     + q_rest * (torch.log(q_rest) - log_rest[c])
            else:              # forward-KL(teacher||student)
                kl = (p_t[c] * (torch.log(p_t[c].clamp_min(1e-12)) - s_lp)).sum(-1) \
                     + r[c] * (log_rest[c] - torch.log(q_rest))
            kl_sum = kl_sum + kl.sum(); del z
        loss = kl_sum / max(T, 1)
        return loss

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
        print("[opd] optimizer = bnb AdamW8bit", flush=True)
    except Exception:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    model.train()
    random.Random(0).shuffle(samples)
    nsteps = args.smoke or max(1, args.epochs * len(samples) // args.accum)
    di = 0; run = 0.0
    for step in range(nsteps):
        opt.zero_grad(set_to_none=True); acc = 0.0
        for _ in range(args.accum):
            ids, plen, ti, tl = samples[di % len(samples)]; di += 1
            loss = opd_loss(ids, plen, ti, tl)
            (loss / args.accum).backward(); acc += loss.item() / args.accum
        if step == 0:
            gsum = sum(p.grad.abs().sum().item() for p in trainable if p.grad is not None)
            assert gsum > 0, "zero grad to LoRA — STE/checkpointing broken"
            print(f"[opd] step-0 grad OK {gsum:.3e}", flush=True)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        run = 0.9 * run + 0.1 * acc if step else acc
        if step % 5 == 0 or step == nsteps - 1:
            print(f"[opd] step {step}/{nsteps} kl={acc:.4f} ema={run:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if step and step % 25 == 0:   # insurance: a late failure still leaves a trainable adapter to merge/eval
            os.makedirs(args.adapter_out, exist_ok=True)
            torch.save([{k: getattr(m._lora, k).detach().cpu() for k in ("Agu", "Bgu", "Adn", "Bdn")} for m in mods],
                       os.path.join(args.adapter_out, "adapter.pt"))
            print(f"[opd] checkpoint adapter @ step {step}", flush=True)
    if args.smoke:
        print("[opd] SMOKE OK", flush=True); return
    os.makedirs(args.adapter_out, exist_ok=True)
    state = [{k: getattr(m._lora, k).detach().cpu() for k in ("Agu", "Bgu", "Adn", "Bdn")} for m in mods]
    torch.save(state, os.path.join(args.adapter_out, "adapter.pt"))
    MQ.set_clip_search(24); ML.merge_and_requantize(model, bits=args.bits, group=args.group)
    os.makedirs(args.out, exist_ok=True); model.save_pretrained(args.out, safe_serialization=True)
    transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    try: transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    except Exception: pass
    print(f"[opd] DONE -> {args.out} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
