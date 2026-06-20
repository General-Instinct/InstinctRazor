#!/usr/bin/env python3
"""Self-contained MoE-aware fake-quantization for Qwen3.5-122B-A10B (and any qwen3_5(_moe)).

Why this exists: 94.6% of the 122B's params live in FUSED 3D expert tensors
(`...mlp.experts.gate_up_proj` [E,2I,H], `...mlp.experts.down_proj` [E,H,I]) which are
nn.Parameter tensors, NOT nn.Linear modules — so EdgeRazor's nn.Linear targeting silently
skips them. This module quantizes those fused tensors PER-EXPERT (a bit-width per expert
index e in [0,E)), plus the ordinary nn.Linear weights (attn / shared-expert / embed / lm_head),
under a single allocation spec. Supports PTQ (bake dequantized weights) and STE (for QAT/QAD).

Quant grid (symmetric, per-block along the input/contraction axis):
  tag 1.58 -> ternary (TWN: per-block threshold 0.75*mean|w|, magnitude = mean|w| over kept)
  tag 3    -> int3  (qmax=3, 7 levels)
  tag 4    -> int4  (qmax=7, 15 levels)
  tag 8    -> int8  (qmax=127)
Effective bits reported = the tag value (1.58/3/4/8); scale-overhead is reported separately.
"""
import math, re
import torch
import torch.nn as nn

VISUAL_RE = re.compile(r"(^|\.)visual\.")

# ----------------------------------------------------------------------------- core quant
def _ternary_perblock(w, group):
    """TWN-style ternary per block of size `group` along last axis. w: [..., in]."""
    shp = w.shape
    in_dim = shp[-1]
    g = group if (group and in_dim % group == 0) else in_dim
    wb = w.reshape(-1, g).float()
    thr = 0.75 * wb.abs().mean(dim=-1, keepdim=True)
    mask = (wb.abs() > thr)
    # per-block magnitude = mean |w| over kept entries (fallback to absmax if none kept)
    kept = (wb.abs() * mask).sum(dim=-1, keepdim=True)
    cnt = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    alpha = torch.where(mask.any(dim=-1, keepdim=True), kept / cnt, wb.abs().amax(dim=-1, keepdim=True))
    q = torch.sign(wb) * mask.float() * alpha
    return q.reshape(shp).to(w.dtype)

# Our improvement over plain absmax scales: MSE-optimal per-block CLIP SEARCH (OmniQuant/AWQ-clip family).
# absmax wastes the few quant levels on outliers; searching the clip ratio per block minimizes reconstruction
# error and recovers low-bit (2-3b) quality with ZERO inference overhead (pure offline scale choice, no
# rotation/invariance risk). _CLIP_GRID=None -> legacy absmax; set via set_clip_search() to enable.
_CLIP_GRID = None
def set_clip_search(n_steps, lo=0.65, hi=1.0):
    """Enable per-block MSE-optimal clip search with n_steps candidate ratios in [lo,hi] (0/None -> absmax)."""
    global _CLIP_GRID
    _CLIP_GRID = None if (not n_steps) else torch.linspace(hi, lo, int(n_steps))

def _int_perblock(w, bits, group):
    """Symmetric per-block integer quant, qmax = 2^(bits-1)-1, along last axis.
    If _CLIP_GRID is set, choose the per-block clip ratio that minimizes reconstruction MSE."""
    qmax = (1 << (bits - 1)) - 1
    shp = w.shape
    in_dim = shp[-1]
    g = group if (group and in_dim % group == 0) else in_dim
    wb = w.reshape(-1, g).float()
    amax = wb.abs().amax(dim=-1, keepdim=True)
    if _CLIP_GRID is None:
        scale = amax / qmax
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.clamp(torch.round(wb / scale), -qmax, qmax) * scale
        return q.reshape(shp).to(w.dtype)
    best_q = None; best_err = None
    for a in _CLIP_GRID.tolist():
        scale = (a * amax) / qmax
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.clamp(torch.round(wb / scale), -qmax, qmax) * scale
        err = (q - wb).pow(2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_q = err, q
        else:
            better = err < best_err
            best_q = torch.where(better, q, best_q)
            best_err = torch.where(better, err, best_err)
    return best_q.reshape(shp).to(w.dtype)

def fakequant(w, tag, group=128):
    """Dequantized fake-quant of w at the given bit `tag` (float: 1.58/3/4/8/16). 16 = no-op."""
    tag = float(tag)
    if tag >= 16:
        return w
    if abs(tag - 1.58) < 0.3:        # ternary
        return _ternary_perblock(w, group)
    return _int_perblock(w, int(round(tag)), group)

class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, tag, group):
        return fakequant(w, tag, group)
    @staticmethod
    def backward(ctx, g):
        return g, None, None

def ste_quant(w, tag, group=128):
    """Straight-through fake-quant for QAT/QAD (grad flows to w)."""
    return _STE.apply(w, tag, group)

# ----------------------------------------------------------------------------- model walking
# Model-specific walking (which params are experts vs backbone linears vs routers, multimodal vs
# text-only) is delegated to a ModelAdapter (see model_adapters.py). The module-level functions below
# are thin shims over the *active* adapter, so existing imports (`from moe_quant import
# iter_expert_tensors`) keep working unchanged. The active adapter lazily defaults to the Qwen3.5/3.6
# (fused, multimodal) behavior, so any path that does not call model_adapters.load_model() behaves
# exactly as before.
_ACTIVE_ADAPTER = None

def set_active_adapter(adapter):
    """Set the model adapter used by the module-level iterator shims (called by load_model)."""
    global _ACTIVE_ADAPTER
    _ACTIVE_ADAPTER = adapter

def active_adapter():
    """The active adapter, lazily defaulting to the Qwen3.5/3.6 (fused, multimodal) adapter."""
    global _ACTIVE_ADAPTER
    if _ACTIVE_ADAPTER is None:
        import model_adapters
        _ACTIVE_ADAPTER = model_adapters.default_adapter()
    return _ACTIVE_ADAPTER

def iter_expert_tensors(model):
    """Yield (name, param) for fused expert weight tensors [E, *, *] (empty for separate-expert models)."""
    yield from active_adapter().iter_expert_tensors(model)

def iter_quant_linears(model, include_embed=True):
    """Yield (name, module) for nn.Linear/nn.Embedding weights we quantize, per the active adapter.
    For separate-expert models this also yields the per-expert Linears (tagged 'expert' by alloc)."""
    yield from active_adapter().iter_quant_linears(model, include_embed=include_embed)

def layer_idx_of(name):
    mobj = re.search(r"layers\.(\d+)\.", name)
    return int(mobj.group(1)) if mobj else -1

# ----------------------------------------------------------------------------- allocation spec
class AllocSpec:
    """Holds the bit-width assignment.
       expert_bits: dict name -> tensor[E] (per-expert tag) for each fused expert tensor.
       linear_bits: dict name -> tag for specific non-expert linears/embeddings.
       default_linear_bits: tag applied to any quantizable linear/embed not in linear_bits
                            (None = leave at 16/bf16). default_expert_bits likewise for experts.
       group: per-block group size for quantization."""
    def __init__(self, expert_bits=None, linear_bits=None, group=128,
                 default_linear_bits=4, default_expert_bits=None, meta=None):
        self.expert_bits = expert_bits or {}
        self.linear_bits = linear_bits or {}
        self.default_linear_bits = default_linear_bits
        self.default_expert_bits = default_expert_bits
        self.group = group
        self.meta = meta or {}

    def linear_tag(self, name):
        return self.linear_bits.get(name, self.default_linear_bits)

# ----------------------------------------------------------------------------- apply (PTQ bake)
@torch.no_grad()
def apply_ptq(model, spec: AllocSpec, verbose=True):
    """Bake fake-quantized weights into the model in place (PTQ). Returns stats dict."""
    g = spec.group
    nE = 0
    CH = 8  # experts per chunk — bounds fp32 fakequant scratch (~8*25MB) so we never clone the 6.4GB tensor
    # experts (in-place, chunked — memory-safe on a device_map-packed GPU)
    for n, p in iter_expert_tensors(model):
        if n in spec.expert_bits:
            bits = spec.expert_bits[n].to(p.device)
        elif spec.default_expert_bits is not None:
            bits = torch.full((p.shape[0],), float(spec.default_expert_bits), device=p.device)
        else:
            continue
        for b in torch.unique(bits).tolist():
            if float(b) >= 16:
                continue
            idx = (bits == b).nonzero(as_tuple=True)[0]
            for c0 in range(0, idx.numel(), CH):
                sub = idx[c0:c0 + CH]
                p.data[sub] = fakequant(p.data[sub], float(b), g)
        nE += 1
    # linears / embeddings
    nL = 0
    for n, m in iter_quant_linears(model):
        tag = spec.linear_tag(n)
        if tag is not None and float(tag) < 16:
            m.weight.data.copy_(fakequant(m.weight.data, tag, g))
            nL += 1
    if verbose:
        print(f"[apply_ptq] quantized {nE} expert tensors, {nL} linears/embeds (group={g})", flush=True)
    return {"n_expert_tensors": nE, "n_linears": nL}

# ----------------------------------------------------------------------------- bit accounting
def effective_bits(model, spec: AllocSpec):
    """Param-weighted average effective bits over all quantized + non-quantized (16b) weights.
    Non-quantized non-expert weights (e.g. router, norms) count at 16 (bf16)."""
    num = 0.0   # sum(params * bits)
    den = 0.0   # sum(params)
    expert_num = 0.0; expert_den = 0.0
    for n, p in iter_expert_tensors(model):
        E = p.shape[0]
        per = p.numel() // E
        if n in spec.expert_bits:
            bits = spec.expert_bits[n].float()
            num += (bits * per).sum().item(); den += p.numel()
            expert_num += (bits * per).sum().item(); expert_den += p.numel()
        else:
            # match apply_ptq: experts absent from expert_bits use default_expert_bits
            # (else stay at 16b/bf16). Previously hardcoded 16 here, which over-counted the
            # flat quant_save recipe's footprint (~15b instead of ~3b) — see clip_122b.env.
            b = float(spec.default_expert_bits) if spec.default_expert_bits is not None else 16.0
            num += b * p.numel(); den += p.numel()
            expert_num += b * p.numel(); expert_den += p.numel()
    for n, m in iter_quant_linears(model):
        w = m.weight
        tag = spec.linear_tag(n)
        tag = 16 if tag is None else float(tag)
        num += tag * w.numel(); den += w.numel()
    return {"avg_bits_all": num / den, "avg_bits_experts": (expert_num / expert_den) if expert_den else 0.0,
            "expert_param_frac": expert_den / den}

# ----------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== quant error vs bits (random gaussian [256,512]) ===")
    W = torch.randn(256, 512)
    for tag in [1.58, 3, 4, 8, 16]:
        Wq = fakequant(W, tag, group=128)
        err = (Wq - W).pow(2).mean().sqrt().item()
        rel = err / W.std().item()
        print(f"  tag={tag:<5} rmse={err:.4f} rel={rel:.4f}")
    # monotonicity check
    errs = [(fakequant(W, t, 128) - W).pow(2).mean().item() for t in [1.58, 3, 4, 8]]
    assert errs[0] > errs[1] > errs[2] > errs[3], f"error not monotone: {errs}"
    print("  monotonic error in bits: OK")
    # STE grad check
    Wp = torch.randn(64, 128, requires_grad=True)
    y = ste_quant(Wp, 4, 64).sum()
    y.backward()
    assert torch.allclose(Wp.grad, torch.ones_like(Wp)), "STE grad should be ones"
    print("  STE straight-through grad: OK")
    # expert-tensor per-expert alloc
    E = 8
    W3 = torch.randn(E, 64, 128)
    bits = torch.tensor([4, 4, 3, 3, 1.58, 1.58, 16, 16])
    out = W3.clone()
    for b in torch.unique(bits).tolist():
        idx = (bits == b).nonzero(as_tuple=True)[0]
        out[idx] = fakequant(W3[idx], float(b), 64)
    e16 = (out[6] - W3[6]).abs().max().item()
    assert e16 == 0.0, "16-bit expert should be untouched"
    print(f"  per-expert alloc: 16b expert untouched (err={e16}); OK")
    print("ALL SELF-TESTS PASSED")
