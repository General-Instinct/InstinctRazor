#!/usr/bin/env python3
"""Capstone: bake our expert-PTQ recipe (3b routed experts + 4b backbone/embeds ≈ 3.05 effective bits ≈ ~47GB,
UNDER Gemma's 62.6GB) into the 122B and SAVE the dequantized checkpoint, so vLLM can eval its capability at
full fidelity. Reports effective_bits = the real deployment footprint of the packed recipe."""
import argparse, time, json, os, shutil
import torch
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import transformers; transformers.logging.set_verbosity_error()
import moe_quant as MQ

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
    nd = torch.cuda.device_count()
    t0 = time.time()
    # Load the FULL multimodal model (Qwen3_5MoeForConditionalGeneration) so the saved checkpoint has the
    # multimodal config + vision weights that vLLM requires (AutoModelForCausalLM saved a text-only config
    # that vLLM rejects). Experts live under model.language_model.* either way, so apply_ptq is unchanged.
    mm = {i: f"{args.max_mem_gib}GiB" for i in range(nd)}
    try:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", max_memory=mm, trust_remote_code=True).eval()
        print(f"[quant_save] loaded MULTIMODAL in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"[quant_save] multimodal load failed ({str(e)[:80]}); falling back to CausalLM", flush=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", max_memory=mm, trust_remote_code=True).eval()
        print(f"[quant_save] loaded in {time.time()-t0:.0f}s", flush=True)
    if args.clip_steps:
        MQ.set_clip_search(args.clip_steps)
        print(f"[quant_save] MSE clip search ENABLED ({args.clip_steps} steps)", flush=True)
    spec = MQ.AllocSpec(default_expert_bits=args.expert_bits, default_linear_bits=args.linear_bits, group=args.group)
    eb = MQ.effective_bits(model, spec)
    avg = float(eb["avg_bits_all"])
    gb = 122.6 * avg / 8
    print(f"[quant_save] effective_bits={avg:.3f} (experts {eb['avg_bits_experts']:.2f}, "
          f"expert_frac {eb['expert_param_frac']:.3f}) -> footprint ~{gb:.1f} GB (Gemma BF16 = 62.6 GB)", flush=True)
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
