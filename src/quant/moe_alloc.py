#!/usr/bin/env python3
"""MoE bit-allocation strategies -> AllocSpec, grounded in the 2025-26 survey (SURVEY_MOE.md).

Signals per (layer L, expert e), at block granularity (gate_up, down):
  freq[L,e]   : token-activation count                       (expert-frequency-aware)
  wmass[L,e]  : sum router softmax weight                     (router-load-aware)
  asal[L,e]   : sum ||hidden||_2 over routed tokens           (activation-salience-aware)
  wfro[L,e,b] : ||W_{L,e,b}||_F  (data-free Hutchinson Tr(H) = E_v||W v||^2 = ||W||_F^2; MoPEQ 2509.02512)
  outdist     : ||O(W)-O(Q(W))||_2 block output distortion    (MxMoE/GEMQ; measured, optional 2nd pass)

composite = rank-fusion of {wmass, asal, wfro} (+freq) -> rank experts; assign high bits to top, low to tail,
to hit an expert-average bit budget. Protected carve-outs (shared expert / attn / ssm / router / embed) are
the ~6% non-expert params kept at a high floor (nearly free). See SURVEY_MOE.md sections 2.2-2.5.
"""
import re, json
import torch
import numpy as np
from moe_quant import AllocSpec, iter_expert_tensors, iter_quant_linears, VISUAL_RE, active_adapter

EXPERT_TAG_SETS = {
    "t4":   [1.58, 4.0],          # ternary / int4 (2-level, matches prior dense study)
    "t34":  [1.58, 3.0, 4.0],     # ternary / int3 / int4 (3-level, smoother)
    "234":  [1.58, 3.0, 4.0],     # alias
}

# ----------------------------------------------------------------- weight-based stats (from model)
@torch.no_grad()
def compute_weight_stats(model):
    """Per-(layer,expert,block) Frobenius norm of expert weights (data-free salience).
    Returns dict layer -> {'gate_up': tensor[E], 'down': tensor[E]} and router-row norms."""
    ad = active_adapter()
    wfro = {}
    rnorm = {}
    for n, p in iter_expert_tensors(model):
        L = int(re.search(r"layers\.(\d+)\.", n).group(1))
        blk = "gate_up" if n.endswith("gate_up_proj") else "down"
        # p: [E, out, in]; per-expert Frobenius norm — computed per-expert slice to avoid a
        # full-tensor fp32 upcast (6.4GB->12.8GB) on an already-near-full GPU.
        E = p.shape[0]
        fro = torch.empty(E)
        for e in range(E):
            fro[e] = p[e].detach().float().norm()
        wfro.setdefault(L, {})[blk] = fro.cpu()
    for n, m in model.named_modules():
        if ad.is_router(m) and hasattr(m, "weight"):
            L = int(re.search(r"layers\.(\d+)\.", n).group(1))
            rnorm[L] = m.weight.detach().float().norm(dim=1).cpu()  # [E] row norms
    return {"wfro": wfro, "rnorm": rnorm}

# ----------------------------------------------------------------- ranking
def _rank01(x):
    """Return ranks in [0,1] (0=smallest, 1=largest), ties broken arbitrarily."""
    x = np.asarray(x, dtype=np.float64)
    order = x.argsort()
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / max(1, len(x) - 1)

def expert_importance(stats, probe, strategy, nL, E, seed=0):
    """Return importance[L] = np.array[E] (higher = keep more bits). Block-agnostic at expert level;
    block split handled in build_spec via per-block wfro for 'composite_block'."""
    freq = np.asarray(probe["freq"])    # [nL,E]
    wmass = np.asarray(probe["wmass"])
    asal = np.asarray(probe["asal"])
    wfro_gu = np.stack([stats["wfro"][L]["gate_up"].numpy() for L in range(nL)])  # [nL,E]
    wfro_dn = np.stack([stats["wfro"][L]["down"].numpy() for L in range(nL)])
    wfro = wfro_gu + wfro_dn
    rnorm = np.stack([stats["rnorm"][L].numpy() for L in range(nL)]) if stats.get("rnorm") else None
    rng = np.random.default_rng(seed)
    imp = np.zeros((nL, E))
    for L in range(nL):
        if strategy == "freq":
            imp[L] = freq[L]
        elif strategy == "wmass":
            imp[L] = wmass[L]
        elif strategy == "asal":
            imp[L] = asal[L]
        elif strategy == "wfro":
            imp[L] = wfro[L]
        elif strategy == "actxw":            # activation x weight output-error proxy
            imp[L] = _rank01(asal[L]) + _rank01(wfro[L])
        elif strategy in ("composite", "composite_block"):
            imp[L] = _rank01(wmass[L]) + _rank01(asal[L]) + _rank01(wfro[L])
        elif strategy == "rnorm_asc":        # small router norm -> more bits (2604.06515)
            imp[L] = -_rank01(rnorm[L]) if rnorm is not None else freq[L]
        elif strategy == "blind" or strategy == "uniform":
            imp[L] = np.arange(E)            # index order (importance-blind)
        elif strategy == "random":
            imp[L] = rng.random(E)
        elif strategy == "inverse":          # anti-control: protect LEAST important
            imp[L] = -( _rank01(wmass[L]) + _rank01(asal[L]) + _rank01(wfro[L]) )
        else:
            raise ValueError(f"unknown strategy {strategy}")
    return imp, {"wfro_gu": wfro_gu, "wfro_dn": wfro_dn}

# ----------------------------------------------------------------- budget -> per-expert tags
def _assign_two_tag(imp_flat, frac_hi, hi, lo):
    """Top frac_hi (by importance) -> hi tag, rest -> lo tag. Returns tag array."""
    n = len(imp_flat)
    n_hi = int(round(frac_hi * n))
    order = np.argsort(-imp_flat)            # desc importance
    tags = np.full(n, lo, dtype=np.float64)
    tags[order[:n_hi]] = hi
    return tags

def build_spec(model, probe, strategy="composite", expert_bits=3.0, tagset="t4",
               protect=True, shared_bits=8.0, attn_bits=4.0, ssm_bits=4.0, router_bits=8.0,
               embed_bits=4.0, group=128, global_rank=True, seed=0, precomputed_stats=None):
    """Build an AllocSpec. expert_bits = target average bits over routed experts.
       protect=True keeps non-expert path at high floors (the ~6% carve-out)."""
    ad = active_adapter()
    cfg = ad.text_config(model.config)
    E, nL = cfg.num_experts, cfg.num_hidden_layers
    stats = precomputed_stats or compute_weight_stats(model)
    imp, extra = expert_importance(stats, probe, strategy, nL, E, seed=seed)

    tags = EXPERT_TAG_SETS[tagset]
    lo, hi = tags[0], tags[-1]
    # fraction at hi to hit expert_bits with 2-tag {lo,hi}: bits = lo + f(hi-lo)
    frac_hi = float(np.clip((expert_bits - lo) / (hi - lo), 0.0, 1.0))

    expert_bits_map = {}
    if global_rank:
        # rank experts across ALL layers jointly (lets some layers stay hotter)
        flat = imp.reshape(-1)
        tagflat = _assign_two_tag(flat, frac_hi, hi, lo).reshape(nL, E)
    else:
        tagflat = np.stack([_assign_two_tag(imp[L], frac_hi, hi, lo) for L in range(nL)])

    for L in range(nL):
        t = torch.tensor(tagflat[L], dtype=torch.float32)
        gu, dn = ad.expert_tensor_names(L)
        expert_bits_map[gu] = t.clone()
        expert_bits_map[dn] = t.clone()

    # non-expert linears, classified by the adapter (ssm_out/in get a floor; tiny gate projs kept high)
    cat_bits = {"shared_expert": shared_bits, "attn": attn_bits, "ssm": ssm_bits,
                "embed": embed_bits, "expert": expert_bits, "backbone": 4.0}
    linear_bits = {}
    for n, m in iter_quant_linears(model):
        if not protect:
            linear_bits[n] = 4.0
            continue
        linear_bits[n] = cat_bits.get(ad.classify_linear(n), 4.0)
    spec = AllocSpec(expert_bits=expert_bits_map, linear_bits=linear_bits, group=group,
                     default_linear_bits=4.0,
                     meta={"strategy": strategy, "expert_bits_target": expert_bits, "tagset": tagset,
                           "frac_hi": frac_hi, "protect": protect, "global_rank": global_rank, "seed": seed})
    return spec, stats
