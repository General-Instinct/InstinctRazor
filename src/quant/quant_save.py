#!/usr/bin/env python3
"""Capstone: bake our expert-PTQ recipe (3b routed experts + 4b backbone/embeds ≈ 3.05 effective bits ≈ ~47GB,
UNDER Gemma's 62.6GB) into the 122B and SAVE the dequantized checkpoint, so vLLM can eval its capability at
full fidelity. Reports effective_bits = the real deployment footprint of the packed recipe."""
import argparse, time, json, os, shutil
import torch
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import transformers; transformers.logging.set_verbosity_error()
import moe_quant as MQ
import model_adapters as MA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--expert-bits", type=float, default=3.0)
    ap.add_argument("--linear-bits", type=float, default=4.0)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--clip-steps", type=int, default=0, help="MSE-optimal clip search steps (0=absmax)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mem-gib", type=int, default=72)
    args = ap.parse_args()
    t0 = time.time()
    # Load via the model adapter: it picks AutoModelForImageTextToText vs CausalLM by config.model_type
    # and registers the active adapter so the quant iterators walk this model correctly. For Qwen3.5/3.6
    # the FULL multimodal model is loaded so the saved checkpoint keeps the multimodal config + vision
    # weights vLLM requires (experts live under model.language_model.* either way).
    model, adapter = MA.load_model(args.model, max_mem_gib=args.max_mem_gib)
    print(f"[quant_save] loaded ({type(adapter).__name__}) in {time.time()-t0:.0f}s", flush=True)
    if args.clip_steps:
        MQ.set_clip_search(args.clip_steps)
        print(f"[quant_save] MSE clip search ENABLED ({args.clip_steps} steps)", flush=True)
    # Flat uniform recipe: experts @ expert_bits, backbone @ linear_bits. The adapter routes
    # separate-expert (non-fused) models' experts through the linear loop at expert_bits.
    spec = adapter.flat_spec(model, args.expert_bits, args.linear_bits, args.group)
    eb = MQ.effective_bits(model, spec)
    avg = float(eb["avg_bits_all"])
    total_b = sum(p.numel() for p in model.parameters()) / 1e9
    gb = total_b * avg / 8
    print(f"[quant_save] effective_bits={avg:.3f} (experts {eb['avg_bits_experts']:.2f}, "
          f"expert_frac {eb['expert_param_frac']:.3f}) over {total_b:.1f}B params -> footprint ~{gb:.1f} GB", flush=True)
    MQ.apply_ptq(model, spec, verbose=True)
    print(f"[quant_save] PTQ baked; saving dequantized checkpoint to {args.out} ...", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    # copy tokenizer + processor + any chat template / config aux files
    try:
        transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    except Exception as e:
        print(f"[quant_save] tokenizer save warn: {e}", flush=True)
    try:
        transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=True).save_pretrained(args.out)
    except Exception as e:
        print(f"[quant_save] processor save warn (ok if text-only): {str(e)[:80]}", flush=True)
    json.dump({"effective_bits": avg, "avg_bits_experts": float(eb["avg_bits_experts"]),
               "footprint_gb": gb, "expert_bits": args.expert_bits,
               "linear_bits": args.linear_bits, "group": args.group},
              open(os.path.join(args.out, "_ptq_meta.json"), "w"), indent=2)
    print(f"[quant_save] DONE in {time.time()-t0:.0f}s -> {args.out} (eff_bits={avg:.3f}, ~{gb:.1f}GB)", flush=True)

if __name__ == "__main__":
    main()
