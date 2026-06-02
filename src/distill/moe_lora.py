#!/usr/bin/env python3
"""Stage 2 keystone: STE-fakequant + per-expert LoRA on the fused MoE experts (footprint-preserving QAD).
Patches Qwen3_5MoeExperts.forward so the base expert weights are FROZEN and quantized via STE (the student
sees 3b experts), while small trainable per-expert LoRA adapters (A,B) learn to recover the teacher's
reasoning policy. After training, merge B@A into the base and re-quantize at the same bits -> 47GB preserved.

Base forward (per expert e, from modeling_qwen3_5_moe.py):
  gate,up = linear(x, gate_up_proj[e]).chunk(2); h = act(gate)*up; out = linear(h, down_proj[e])
Patched:
  Wgu = ste_quant(gate_up_proj[e],bits); o = linear(x,Wgu) + s*linear(linear(x,Agu[e]),Bgu[e]); gate,up=o.chunk(2)
  Wdn = ste_quant(down_proj[e],bits);    out = linear(h,Wdn) + s*linear(linear(h,Adn[e]),Bdn[e])
"""
import torch, torch.nn as nn, torch.nn.functional as F
import moe_quant as MQ

def _is_experts(m):
    """Detect Qwen3_5MoeExperts by duck-typing (robust to FSDP2 fully_shard __class__ swap)."""
    return hasattr(m, "gate_up_proj") and hasattr(m, "down_proj") and hasattr(m, "num_experts")

class ExpertLoRA(nn.Module):
    """Per-expert low-rank adapters for one Qwen3_5MoeExperts module."""
    def __init__(self, E, hidden, inter, rank, dtype, device):
        super().__init__()
        k = rank
        # gate_up: x[.,H] -> [.,2I];  A:[E,k,H]  B:[E,2I,k]
        self.Agu = nn.Parameter(torch.empty(E, k, hidden, dtype=dtype, device=device).normal_(0, 0.01))
        self.Bgu = nn.Parameter(torch.zeros(E, 2 * inter, k, dtype=dtype, device=device))
        # down: h[.,I] -> [.,H];      A:[E,k,I]  B:[E,H,k]
        self.Adn = nn.Parameter(torch.empty(E, k, inter, dtype=dtype, device=device).normal_(0, 0.01))
        self.Bdn = nn.Parameter(torch.zeros(E, hidden, k, dtype=dtype, device=device))

def _patched_forward(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    bits, g, s = self._q_bits, self._q_group, self._lora_scale
    lora = self._lora
    with torch.no_grad():
        em = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
    # _static_loop=True (set for FSDP training): iterate ALL experts so the saved-tensor COUNT is CONSTANT
    # regardless of routing -> compatible with gradient checkpointing (the dynamic `hit` loop's data-dependent
    # length is exactly what makes non-reentrant recompute save a different #tensors -> CheckpointError, and
    # reentrant gives zero grad). Math identical (empty experts -> empty index_add no-op). ~32x expert compute,
    # acceptable for ONE short OPD round; the dynamic path stays default for inference/merge. See opd_train_fsdp.py.
    if getattr(self, "_static_loop", False):
        _experts = list(range(self.num_experts))
    else:
        _experts = [int(ei[0]) for ei in hit if int(ei[0]) != self.num_experts]
    for e in _experts:
        pos, tok = torch.where(em[e])
        x = hidden_states[tok]
        # merge-consistent: quantize (W.detach()+s*BA) so train == deploy (merge bakes the same).
        # base detached -> STE grad flows only to the LoRA adapter.
        Wgu = MQ.ste_quant(self.gate_up_proj[e].detach() + s * (lora.Bgu[e] @ lora.Agu[e]), bits, g)
        o = F.linear(x, Wgu)
        gate, up = o.chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        Wdn = MQ.ste_quant(self.down_proj[e].detach() + s * (lora.Bdn[e] @ lora.Adn[e]), bits, g)
        ch = F.linear(h, Wdn)
        ch = ch * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, ch.to(final.dtype))
    return final

def attach_expert_lora(model, bits=3.0, group=128, rank=8, scale=2.0, static_loop=False):
    """Freeze base; attach per-expert LoRA + STE-quant to every Qwen3_5MoeExperts; return trainable params."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    n = 0
    for mod in model.modules():
        if _is_experts(mod):
            E = mod.num_experts; H = mod.hidden_dim; I = mod.intermediate_dim
            dev = mod.gate_up_proj.device; dt = mod.gate_up_proj.dtype
            lora = ExpertLoRA(E, H, I, rank, dt, dev)
            mod._lora = lora; mod._q_bits = bits; mod._q_group = group; mod._lora_scale = scale
            mod._static_loop = static_loop   # all-experts loop -> constant saved-tensor count for grad-checkpointing
            mod.forward = _patched_forward.__get__(mod, mod.__class__)
            for p in lora.parameters():
                p.requires_grad_(True); trainable.append(p)
            n += 1
    print(f"[moe_lora] attached LoRA(rank={rank}) + STE-{bits}b to {n} expert modules; "
          f"trainable params = {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)
    return trainable

# ---- FAST path: bake quantized experts ONCE (no per-step fakequant) + output-space QLoRA ----
def _patched_forward_fast(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    s = self._lora_scale; lora = self._lora
    with torch.no_grad():
        em = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
    for ei in hit:
        e = ei[0]
        if e == self.num_experts:
            continue
        pos, tok = torch.where(em[e])
        x = hidden_states[tok]
        # base is ALREADY quantized in-place (frozen) -> no per-step fakequant; LoRA in output space
        o = F.linear(x, self.gate_up_proj[e]) + s * F.linear(F.linear(x, lora.Agu[e]), lora.Bgu[e])
        gate, up = o.chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        ch = F.linear(h, self.down_proj[e]) + s * F.linear(F.linear(h, lora.Adn[e]), lora.Bdn[e])
        ch = ch * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, ch.to(final.dtype))
    return final

@torch.no_grad()
def attach_expert_lora_fast(model, bits=3.0, group=128, rank=16, scale=2.0, clip_steps=24):
    """FAST QLoRA: bake fakequant(experts) in-place ONCE (with clip-search for quality), freeze, then train
    output-space per-expert LoRA. No per-step quantization -> 2-4x faster, memory-neutral, standard QLoRA.
    Tiny train/deploy gap (re-quantize at merge); verify after."""
    for p in model.parameters():
        p.requires_grad_(False)
    MQ.set_clip_search(clip_steps)
    CH = 8
    trainable = []; n = 0
    for mod in model.modules():
        if _is_experts(mod):
            for W in (mod.gate_up_proj, mod.down_proj):     # bake quantized base in-place (chunked)
                for c0 in range(0, W.shape[0], CH):
                    W.data[c0:c0+CH] = MQ.fakequant(W.data[c0:c0+CH], bits, group)
            E = mod.num_experts; H = mod.hidden_dim; I = mod.intermediate_dim
            lora = ExpertLoRA(E, H, I, rank, mod.gate_up_proj.dtype, mod.gate_up_proj.device)
            mod._lora = lora; mod._lora_scale = scale
            mod.forward = _patched_forward_fast.__get__(mod, mod.__class__)
            for p in lora.parameters():
                p.requires_grad_(True); trainable.append(p)
            n += 1
    MQ.set_clip_search(0)
    print(f"[moe_lora] FAST: baked {bits}b experts + output-LoRA(rank={rank}) on {n} modules; "
          f"trainable = {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)
    return trainable

@torch.no_grad()
def merge_fast(model, bits=3.0, group=128):
    """Merge output-space LoRA into the (already-baked-quantized) base and re-quantize -> deployable ckpt."""
    MQ.set_clip_search(24)
    for mod in model.modules():
        if _is_experts(mod) and hasattr(mod, "_lora"):
            s = mod._lora_scale; L = mod._lora
            for e in range(mod.num_experts):
                mod.gate_up_proj[e] = MQ.fakequant(mod.gate_up_proj[e] + s * (L.Bgu[e] @ L.Agu[e]), bits, group)
                mod.down_proj[e] = MQ.fakequant(mod.down_proj[e] + s * (L.Bdn[e] @ L.Adn[e]), bits, group)
            del mod._lora
    print("[moe_lora] FAST merge: output-LoRA merged + re-quantized", flush=True)

@torch.no_grad()
def merge_and_requantize(model, bits=3.0, group=128):
    """Merge scale*B@A into base expert weights and re-quantize (fakequant) -> footprint-preserving final."""
    for mod in model.modules():
        if _is_experts(mod) and hasattr(mod, "_lora"):
            s = mod._lora_scale; L = mod._lora
            for e in range(mod.num_experts):
                dgu = s * (L.Bgu[e] @ L.Agu[e])        # [2I,H]
                ddn = s * (L.Bdn[e] @ L.Adn[e])        # [H,I]
                mod.gate_up_proj[e] = MQ.fakequant(mod.gate_up_proj[e] + dgu, bits, group)
                mod.down_proj[e] = MQ.fakequant(mod.down_proj[e] + ddn, bits, group)
            del mod._lora
    print("[moe_lora] merged LoRA + re-quantized experts (footprint preserved)", flush=True)
