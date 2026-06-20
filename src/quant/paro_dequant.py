#!/usr/bin/env python3
"""Reconstruct z-lab ParoQuant's EFFECTIVE bf16 weights so its published Qwen3.6-35B-A3B checkpoint can be
benchmarked on CUDA — ParoQuant's CUDA backends (vllm + transformers) don't wire up the fused MoE experts
(only MLX does), so the model otherwise can't run on NVIDIA.

Math: ParoQuant forward is  y = AWQ_GEMM(rotate(x)) = (x @ M) @ W_dq, where rotate(.) is a FIXED linear map
(pairs/theta/channel_scales are constant buffers), so M = rotate(I). The effective weight is therefore
  W_eff = M @ W_dq = RotateQuantizedLinear.forward(I)
i.e. a plain Linear with W_eff reproduces ParoQuant's output exactly (validated ~8e-4 rel err, fp16).
We fold each quantized module to bf16 and write a standard Qwen3.6 checkpoint = ParoQuant's capability
ceiling (same dequant-bf16 methodology used for our clip/awq), evaluable in normal vLLM.

  python src/quant/paro_dequant.py --paro z-lab/Qwen3.6-35B-A3B-PARO --base Qwen/Qwen3.6-35B-A3B \
      --out models/q36_paro_bf16
"""
import argparse, glob, os, re, time
import torch
import paroquant.kernels.cuda            # noqa: registers torch.ops.rotation.rotate (needs venv activated)
from paroquant.inference.backends.transformers.modules import RotateQuantizedLinear
from safetensors import safe_open
import transformers; transformers.logging.set_verbosity_error()
import model_adapters as MA


@torch.no_grad()
def reconstruct(bufs, dev):
    """bufs: dict with qweight/qzeros/scales (+ optional theta/pairs/channel_scales). Returns W_eff [in,out]."""
    qw = bufs["qweight"]; sc = bufs["scales"]
    inf = qw.shape[0]; outf = sc.shape[1]; gs = inf // sc.shape[0]
    krot = bufs["theta"].shape[0] if "theta" in bufs else 8
    m = RotateQuantizedLinear(inf, outf, bias=False, group_size=gs, bits=4, krot=krot).to(dev).half()
    m.qweight.copy_(qw.to(dev)); m.qzeros.copy_(bufs["qzeros"].to(dev)); m.scales.copy_(sc.to(dev))
    if "theta" in bufs:                  # else leave identity rotation (theta=0, channel_scales=1)
        m.theta.copy_(bufs["theta"].to(dev)); m.pairs.copy_(bufs["pairs"].to(dev))
        m.channel_scales.copy_(bufs["channel_scales"].to(dev))
    return m(torch.eye(inf, dtype=torch.float16, device=dev))      # [in,out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paro", default="z-lab/Qwen3.6-35B-A3B-PARO")
    ap.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mem-gib", type=int, default=76)
    ap.add_argument("--compute-device", default="cuda:0")
    args = ap.parse_args()
    cdev = args.compute_device
    t0 = time.time()

    # base bf16 model (PARO leaves non-quantized modules untouched -> base copy is faithful for them)
    model, adapter = MA.load_model(args.base, max_mem_gib=args.max_mem_gib)
    cfg = adapter.text_config(model.config)
    H = cfg.hidden_size; I = cfg.moe_intermediate_size
    print(f"[paro] loaded base {type(adapter).__name__} H={H} I={I} in {time.time()-t0:.0f}s", flush=True)

    snap = args.paro if os.path.isdir(args.paro) else glob.glob(
        f"/home/ubuntu/.cache/huggingface/hub/models--{args.paro.replace('/','--')}/snapshots/*/model.safetensors")[0]
    st = safe_open(snap, framework="pt", device="cpu")
    keys = set(st.keys())
    qmods = sorted({k[:-len(".qweight")] for k in keys if k.endswith(".qweight")})
    print(f"[paro] {len(qmods)} quantized modules in PARO checkpoint", flush=True)

    def bufs_for(mod, rot_prefix=None):
        b = {s: st.get_tensor(f"{mod}.{s}") for s in ("qweight", "qzeros", "scales")}
        rp = rot_prefix if rot_prefix is not None else mod
        if f"{rp}.theta" in keys or f"{rp}_theta" in keys:
            sep = "." if f"{rp}.theta" in keys else "_"
            for s in ("theta", "pairs", "channel_scales"):
                b[s] = st.get_tensor(f"{rp}{sep}{s}")
        return b

    n_bb = n_exp = 0
    exp_re = re.compile(r"^(model\.language_model\.layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
    for mod in qmods:
        m = exp_re.match(mod)
        if not m:                                   # backbone Linear (self_attn / linear_attn): per-module rotation
            W = reconstruct(bufs_for(mod), cdev)     # [in,out]
            tgt = model.get_submodule(mod)
            tgt.weight.data.copy_(W.t().to(tgt.weight.dtype).to(tgt.weight.device))
            n_bb += 1
        else:                                        # fused-MoE expert: shared per-layer rotation
            layer, e, proj = m.group(1), int(m.group(2)), m.group(3)
            rp = f"{layer}.mlp.experts." + ("gate_up_weight" if proj in ("gate_proj", "up_proj") else "down_weight")
            W = reconstruct(bufs_for(mod, rot_prefix=rp), cdev)             # [in,out]
            em = model.get_submodule(f"{layer}.mlp.experts")
            if proj == "gate_proj":
                em.gate_up_proj.data[e, :I, :].copy_(W.t().to(em.gate_up_proj.dtype).to(em.gate_up_proj.device))
            elif proj == "up_proj":
                em.gate_up_proj.data[e, I:, :].copy_(W.t().to(em.gate_up_proj.dtype).to(em.gate_up_proj.device))
            else:
                em.down_proj.data[e].copy_(W.t().to(em.down_proj.dtype).to(em.down_proj.device))
            n_exp += 1
        if (n_bb + n_exp) % 2000 == 0:
            print(f"  reconstructed {n_bb} backbone + {n_exp} expert linears ({time.time()-t0:.0f}s)", flush=True)
    print(f"[paro] DONE reconstruct: {n_bb} backbone + {n_exp} expert linears", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    for cls in (transformers.AutoTokenizer, transformers.AutoProcessor):
        try:
            cls.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
        except Exception as e:
            print(f"[paro] {cls.__name__} save warn: {str(e)[:80]}", flush=True)
    print(f"[paro] saved bf16 PARO-effective checkpoint -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
