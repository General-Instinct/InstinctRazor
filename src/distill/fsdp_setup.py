#!/usr/bin/env python3
"""FSDP2 setup for LOSSLESS 4-GPU training of the 122B frozen-base + per-expert-LoRA + STE custom MoE forward.
Key mechanic (verified): fully_shard registers a prepend pre-forward hook that UNSHARDS a module's params to
full local shape BEFORE the module's (monkeypatched) forward runs. So wrapping each Qwen3_5MoeExperts as its
own unit makes self.gate_up_proj[e] index the FULL [2048,3072] tensor -> bit-identical math to device_map.
LoRA kept REPLICATED (attach AFTER wrap); grads all-reduced -> DDP-equivalent (lossless-in-expectation)."""
import os, torch, torch.distributed as dist
import transformers
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeExperts

def dist_init():
    dist.init_process_group("nccl")
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    return lr, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

def load_model_fsdp(model_name):
    """Rank-0 real load to CPU (245GB); other ranks meta-init; FSDP broadcasts on first all-gather."""
    rank = int(os.environ["RANK"])
    cfg = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if rank == 0:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    else:
        with torch.device("meta"):
            model = transformers.AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16)
    model.config.use_cache = False
    return model

def wrap_fsdp(model):
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    kw = dict(mp_policy=mp, reshard_after_forward=True)
    for m in model.modules():
        if isinstance(m, Qwen3_5MoeExperts):
            fully_shard(m, **kw)            # INNER (mandatory): unshard experts before _patched_forward
    for m in model.modules():
        if isinstance(m, Qwen3_5MoeDecoderLayer):
            fully_shard(m, **kw)            # per-layer reshard
    fully_shard(model, **kw)                # root
    # disable prefetch: keep only ONE unit's params resident at a time (saves ~one unit-gather of headroom)
    for m in model.modules():
        if hasattr(m, "set_modules_to_forward_prefetch"):
            try:
                m.set_modules_to_forward_prefetch([]); m.set_modules_to_backward_prefetch([])
            except Exception:
                pass
    return model

def allreduce_lora_grads(trainable):
    for p in trainable:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
