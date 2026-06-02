#!/usr/bin/env python3
"""Bake an FSDP-trained per-expert LoRA adapter (adapter.pt) into the frozen base and re-quantize -> deployable
ckpt in the SAME BF16-with-3bit-values format as q122_ptq3b_clip (vLLM-loadable; ~239GB on disk, ~3.05 effective
bits). Single-process device_map, no-grad. Mirrors opd_train.py's tail. The adapter module order matches
attach_expert_lora's iteration (ML._is_experts), so adapter[i] <-> experts-module[i]."""
import argparse, os, time
import torch
import transformers; transformers.logging.set_verbosity_error()
import moe_lora as ML, moe_quant as MQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--adapter", required=True)            # dir containing adapter.pt
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=float, default=3.0); ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--clip-steps", type=int, default=24)  # quality-preserving clip-search requant (matches clip ckpt)
    ap.add_argument("--max-mem-gib", type=int, default=72)
    args = ap.parse_args()
    nd = torch.cuda.device_count(); t0 = time.time()
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory={i: f"{args.max_mem_gib}GiB" for i in range(nd)}, trust_remote_code=True)
    MQ.set_clip_search(0)
    ML.attach_expert_lora(model, bits=args.bits, group=args.group, rank=args.rank, scale=args.scale)
    mods = [m for m in model.modules() if ML._is_experts(m) and hasattr(m, "_lora")]
    sd = torch.load(os.path.join(args.adapter, "adapter.pt"), map_location="cpu")
    assert len(sd) == len(mods), f"adapter modules {len(sd)} != model experts {len(mods)}"
    for m, d in zip(mods, sd):
        for k in ("Agu", "Bgu", "Adn", "Bdn"):
            tgt = getattr(m._lora, k)
            tgt.data.copy_(d[k].to(tgt.device, tgt.dtype))
    print(f"[merge] loaded adapter ({len(sd)} modules) ({time.time()-t0:.0f}s)", flush=True)
    MQ.set_clip_search(args.clip_steps)
    ML.merge_and_requantize(model, bits=args.bits, group=args.group)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    try:
        transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    except Exception:
        pass
    print(f"[merge] DONE -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
