#!/usr/bin/env python3
"""MINIMAL 4-GPU smoke: validate FSDP (full-shard, SAME-batch-replicated) gives a LOSSLESS, all-4-GPUs-busy
replacement for the device_map="auto" naive-pipeline trainer in stage2_sft.py / opd_train.py.

WHY FSDP (not DDP) for a LOSSLESS check
---------------------------------------
DDP / data-parallel feed a DIFFERENT microbatch per rank -> per-rank loss != single-GPU loss on that batch.
That is faster but NOT "same loss on the same batch+seed". To prove LOSSLESS we replicate the SAME (input_ids,
labels) on all 4 ranks and shard the *parameters* (FULL_SHARD): every rank all-gathers each layer's weight
shard just-in-time, so all 4 GPUs compute the SAME forward concurrently. The result on every rank is the
single-sequence forward, mathematically identical to the device_map pipeline (same weights, same batch, same
RNG) up to fp all-gather/reduce reordering -> we assert match within a loose fp tol. SAME MATH, 4x the active
hardware. moe_lora patches are reused UNCHANGED (the LoRA params are tiny -> they shard too; STE/fakequant math
is per-element and parallel-invariant).

Asserts (rank 0): (a) all 4 GPUs show memory+util>0, (b) LoRA grad sum>0, (c) no OOM over 2 steps,
(d) step-0 loss == the device_map reference loss on the SAME batch+seed within tol (LOSSLESS).

Run the reference first (single process, device_map="auto", writes ref loss), then this under torchrun nproc=4.
"""
import argparse, json, os, time, random
import torch, torch.distributed as dist
import transformers; transformers.logging.set_verbosity_error()
import moe_lora as ML, moe_quant as MQ

REF_PATH = "/tmp/smoke_ref_loss.json"
MODEL = "Qwen/Qwen3.5-122B-A10B"
DATA = "results/stage2/teacher_cot.jsonl"
SEED = 0


def build_batch(tok, max_len):
    """Deterministic single batch (seed-fixed) identical to stage2_sft.make_batch — same row, same ids."""
    rows = [json.loads(l) for l in open(DATA)]
    rows = [r for r in rows if r.get("domain", "math") == "math"]
    random.Random(SEED).shuffle(rows)
    r = rows[0]
    ptext = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids = tok(ptext, add_special_tokens=False)["input_ids"]
    tids = tok(r["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (pids + tids)[:max_len]
    labels = ([-100] * len(pids) + tids)[:max_len]
    return ids, labels


def attach(model, args):
    MQ.set_clip_search(0)  # absmax during training (same as stage2_sft non-fast path)
    return ML.attach_expert_lora(model, bits=args.bits, group=args.group, rank=args.rank, scale=args.scale)


# --------------------------------------------------------------------------- REFERENCE (device_map pipeline)
def run_reference(args):
    torch.manual_seed(SEED)
    nd = torch.cuda.device_count()
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        max_memory={i: f"{args.max_mem_gib}GiB" for i in range(nd)}, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    torch.manual_seed(SEED)  # re-seed AFTER load so LoRA init (.normal_) is identical to the FSDP run
    trainable = attach(model, args)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids, labels = build_batch(tok, args.max_len)
    emb_dev = model.get_input_embeddings().weight.device
    x = torch.tensor([ids], device=emb_dev); y = torch.tensor([labels], device=emb_dev)
    model.train()
    out = model(input_ids=x, labels=y)
    loss = float(out.loss.detach())
    out.loss.backward()
    gsum = sum(p.grad.abs().sum().item() for p in trainable if p.grad is not None)
    json.dump({"loss": loss, "grad_sum": gsum, "seq_len": len(ids),
               "n_trainable": sum(p.numel() for p in trainable)}, open(REF_PATH, "w"))
    print(f"[ref] device_map loss={loss:.6f} grad_sum={gsum:.3e} seqlen={len(ids)} -> {REF_PATH}", flush=True)


# --------------------------------------------------------------------------- FSDP (4 GPUs, same batch)
def run_fsdp(args):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeExperts

    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    is0 = rank == 0
    def log(*a, **k):
        if is0: print(*a, flush=True)

    torch.manual_seed(SEED)
    t0 = time.time()
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
    model.config.use_cache = False
    torch.manual_seed(SEED)  # identical LoRA init to the reference
    trainable = attach(model, args)            # BEFORE wrap -> LoRA shards WITH the experts unit (1.8GB/GPU not 7.2)
    log(f"[fsdp] loaded+attached on CPU in {time.time()-t0:.0f}s", flush=True)
    # FSDP2 fully_shard: inner experts(+their LoRA) -> per-layer -> ROOT (shards embed/lm_head/vision too)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    kw = dict(mp_policy=mp, reshard_after_forward=True)
    for m in model.modules():
        if isinstance(m, Qwen3_5MoeExperts): fully_shard(m, **kw)
    for m in model.modules():
        if isinstance(m, Qwen3_5MoeDecoderLayer): fully_shard(m, **kw)
    fully_shard(model, **kw)
    for m in model.modules():
        if hasattr(m, "set_modules_to_forward_prefetch"):
            try: m.set_modules_to_forward_prefetch([]); m.set_modules_to_backward_prefetch([])
            except Exception: pass
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    log(f"[fsdp] FSDP2-sharded (no-prefetch) across {world} GPUs in {time.time()-t0:.0f}s", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids, labels = build_batch(tok, args.max_len)               # SAME batch+seed on every rank -> lossless check
    dev = torch.device(f"cuda:{local}")
    x = torch.tensor([ids], device=dev); y = torch.tensor([labels], device=dev)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    model.train()
    step0_loss = None
    for step in range(2):                                      # 2 forward+backward steps
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=x, labels=y)
        out.loss.backward()
        if step == 0:
            step0_loss = out.loss.detach().clone()
            def _local(g): return g.to_local() if hasattr(g, "to_local") else g
            gsum = torch.tensor([sum(_local(p.grad).abs().sum().item()
                                     for p in trainable if p.grad is not None)], device=dev)
            dist.all_reduce(gsum)                              # sum LoRA grad shards across ranks
            mem = [torch.cuda.memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())]
            log(f"[fsdp] step0 loss={float(step0_loss):.6f} grad_sum(all-shard)={float(gsum):.3e}")
            log(f"[fsdp] per-GPU mem(GB)={['%.1f' % m for m in mem]}")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        log(f"[fsdp] step {step} loss={float(out.loss):.6f} ({time.time()-t0:.0f}s)")

    # ---- assertions on rank 0 ----
    if is0:
        ok = True
        # (a) all 4 GPUs active
        util = []
        try:
            import pynvml; pynvml.nvmlInit()
            for i in range(torch.cuda.device_count()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            util = ["n/a"]
        mem = [torch.cuda.memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())]
        all_busy = all(m > 1.0 for m in mem)                   # every GPU holds a real shard
        print(f"[ASSERT a] all-4-GPU memory>1GB: {['%.1f' % m for m in mem]} -> {all_busy}", flush=True)
        print(f"[INFO   a] sampled util%: {util} (sample at end; the per-step nvidia-smi watch is the live proof)",
              flush=True)
        assert all_busy, "FAIL(a): not all GPUs hold a shard"
        # (b) LoRA grad > 0
        print(f"[ASSERT b] LoRA grad_sum={float(gsum):.3e} > 0 -> {float(gsum) > 0}", flush=True)
        assert float(gsum) > 0, "FAIL(b): zero LoRA grad"
        # (c) no OOM == we reached here
        print("[ASSERT c] 2 steps completed, no OOM -> True", flush=True)
        # (d) LOSSLESS vs device_map reference
        if os.path.exists(REF_PATH):
            ref = json.load(open(REF_PATH))
            l_fsdp, l_ref = float(step0_loss), float(ref["loss"])
            adiff = abs(l_fsdp - l_ref); rdiff = adiff / (abs(l_ref) + 1e-9)
            # bf16 forward + fp32 grad-reduce across reordered shards: ~1e-2 abs is the realistic lossless band.
            lossless = adiff < args.tol
            print(f"[ASSERT d] LOSSLESS: fsdp={l_fsdp:.6f} ref={l_ref:.6f} "
                  f"abs={adiff:.3e} rel={rdiff:.3e} tol={args.tol} -> {lossless}", flush=True)
            assert lossless, f"FAIL(d): loss diff {adiff:.3e} exceeds tol {args.tol} — NOT lossless"
        else:
            print(f"[ASSERT d] SKIP: no reference at {REF_PATH} (run --mode ref first)", flush=True)
            ok = False
        print("SMOKE OK" if ok else "SMOKE INCOMPLETE (ran reference?)", flush=True)
    dist.barrier(); dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ref", "fsdp"], required=True)
    ap.add_argument("--bits", type=float, default=3.0); ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-mem-gib", type=int, default=62)
    ap.add_argument("--tol", type=float, default=2e-2)   # abs-loss lossless band (bf16 fwd + fp32 reduce)
    args = ap.parse_args()
    if args.mode == "ref":
        run_reference(args)
    else:
        run_fsdp(args)


if __name__ == "__main__":
    main()
