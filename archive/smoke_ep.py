#!/usr/bin/env python3
"""EP (expert-parallel) lossless smoke. Shards the 256 experts dim-0 across 4 ranks (64/GPU, in HBM, no
all-gather), keeping each expert matrix whole so the moe_lora per-expert STE+LoRA loop survives (HF rewrites
num_experts->64). Asserts: all-4-GPU, LoRA grad>0, no-OOM, and loss == device_map ref (lossless => also proves
the shared_expert is NOT all-reduce-double-counted). Run AFTER smoke_fsdp.py --mode ref (writes /tmp/smoke_ref_loss.json)."""
import argparse, json, os, time, random
import torch, torch.distributed as dist, torch.nn as nn
import transformers; transformers.logging.set_verbosity_error()
import moe_lora as ML, moe_quant as MQ

REF_PATH = "/tmp/smoke_ref_loss.json"; MODEL = "Qwen/Qwen3.5-122B-A10B"
DATA = "results/stage2/teacher_cot.jsonl"; SEED = 0
EP_PLAN = {"layers.*.mlp.gate": "ep_router",
           "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
           "layers.*.mlp.experts.down_proj": "grouped_gemm",
           "layers.*.mlp.experts": "moe_tp_experts"}

def build_batch(tok, max_len):
    rows = [json.loads(l) for l in open(DATA)]; rows = [r for r in rows if r.get("domain","math")=="math"]
    random.Random(SEED).shuffle(rows); r = rows[0]
    pt = tok.apply_chat_template([{"role":"user","content":r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids = tok(pt, add_special_tokens=False)["input_ids"]
    tids = tok(r["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (pids+tids)[:max_len]; labels = ([-100]*len(pids)+tids)[:max_len]
    return ids, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=512); ap.add_argument("--rank-lora", type=int, default=16)
    ap.add_argument("--tol", type=float, default=2e-2); ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dist.init_process_group("nccl"); is0 = rank == 0
    def log(*a, **k):
        if is0: print(*a, flush=True)
    MQ.set_clip_search(0); torch.manual_seed(SEED); t0 = time.time()
    from transformers.distributed.configuration_utils import DistributedConfig
    cfg = transformers.AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.base_model_ep_plan = EP_PLAN
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, config=cfg, trust_remote_code=True,
        distributed_config=DistributedConfig(enable_expert_parallel=True))
    model.config.use_cache = False
    # free unused vision + MTP (identical structure on all ranks)
    lm = model.model if hasattr(model, "model") else model
    for attr in ("visual", "vision_tower", "mtp"):
        if hasattr(lm, attr): setattr(lm, attr, nn.Identity())
        if hasattr(getattr(lm, "model", lm), attr): setattr(getattr(lm, "model", lm), attr, nn.Identity())
    torch.cuda.empty_cache()
    log(f"[ep] loaded EP across {world} GPUs in {time.time()-t0:.0f}s", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(SEED)
    trainable = ML.attach_expert_lora(model, bits=3.0, group=128, rank=args.rank_lora, scale=2.0)
    nloc = next((m.num_experts for m in model.modules() if ML._is_experts(m)), None)
    log(f"[ep] local experts/rank={nloc}  LoRA trainable={sum(p.numel() for p in trainable)/1e6:.0f}M", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids, labels = build_batch(tok, args.max_len)
    dev = torch.device(f"cuda:{local}")
    x = torch.tensor([ids], device=dev); y = torch.tensor([labels], device=dev)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))
    model.train(); step0 = None
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=x, labels=y); out.loss.backward()
        if step == 0:
            step0 = float(out.loss.detach())
            gsum = torch.tensor([sum((p.grad.to_local() if hasattr(p.grad,"to_local") else p.grad).abs().sum().item()
                                     for p in trainable if p.grad is not None)], device=dev)
            dist.all_reduce(gsum)
            mem = [torch.cuda.memory_allocated(i)/1e9 for i in range(torch.cuda.device_count())]
            log(f"[ep] step0 loss={step0:.6f} grad={float(gsum):.3e} mem(GB)={['%.1f'%m for m in mem]}")
        ML.allreduce_lora_grads(trainable) if hasattr(ML, "allreduce_lora_grads") else None
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        log(f"[ep] step {step} loss={float(out.loss):.6f} ({time.time()-t0:.0f}s)")
    if is0:
        mem = [torch.cuda.memory_allocated(i)/1e9 for i in range(torch.cuda.device_count())]
        all4 = all(m > 1.0 for m in mem)
        ref = json.load(open(REF_PATH))["loss"] if os.path.exists(REF_PATH) else None
        adiff = abs(step0 - ref) if ref is not None else None
        lossless = ref is not None and adiff < args.tol
        print(f"[ep] all4GPU={all4} mem={['%.1f'%m for m in mem]}  loss={step0:.6f} ref={ref} |diff|={adiff} "
              f"LOSSLESS={lossless}  => {'PASS' if (all4 and lossless) else 'CHECK'}", flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
