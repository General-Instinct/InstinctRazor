#!/usr/bin/env python3
"""Regression net for the ModelAdapter refactor (no GPU / no model download).

Two guarantees:
  1. INVARIANCE — the default Qwen3.5/3.6 adapter walks a model EXACTLY like the original hardcoded
     iter_expert_tensors / iter_quant_linears / classification did (verbatim reference inlined below).
  2. GENERALIZATION — the separate-expert adapter (OLMoE-style) now INCLUDES per-expert nn.Linear
     experts (which the old code skipped entirely -> zero expert coverage) and tags them at expert_bits.
  3. effective_bits no longer over-counts flat-recipe experts at 16b.

Run:  python tests/test_adapter_invariance.py     (or: pytest tests/test_adapter_invariance.py)
"""
import os, re, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "quant"))
import moe_quant as MQ
import model_adapters as MA


# --------------------------------------------------------------------- reference (ORIGINAL) logic
_VIS = re.compile(r"(^|\.)visual\.")

def _ref_expert_tensors(model):
    for n, p in model.named_parameters():
        if _VIS.search(n):
            continue
        if n.endswith("experts.gate_up_proj") or n.endswith("experts.down_proj"):
            yield n, p

def _ref_quant_linears(model, include_embed=True):
    for n, m in model.named_modules():
        if _VIS.search(n):
            continue
        if isinstance(m, nn.Linear) and ".experts." not in n:
            yield n, m
        elif include_embed and isinstance(m, nn.Embedding) and "language_model" in n:
            yield n, m

def _ref_linear_bits(name, shared=8.0, attn=4.0, ssm=4.0, embed=4.0):
    if ".shared_expert" in name:
        return shared
    elif ".self_attn." in name:
        return attn
    elif ".linear_attn." in name:
        return ssm
    elif name.endswith("embed_tokens") or name.endswith("lm_head"):
        return embed
    return 4.0


# --------------------------------------------------------------------- stub models
class Qwen3_5MoeTopKRouter(nn.Module):   # name matters: adapter.is_router matches by class name
    def __init__(self, E, H):
        super().__init__(); self.weight = nn.Parameter(torch.randn(E, H))

class OlmoeTopKRouter(nn.Module):        # OLMoE's router class (transformers 5.x)
    def __init__(self, E, H):
        super().__init__(); self.weight = nn.Parameter(torch.randn(E, H))

class _FusedExperts(nn.Module):
    def __init__(self, E, I, H):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(E, H, I) * 0.02)
        self.num_experts = E

def _module(**kw):
    m = nn.Module()
    for k, v in kw.items():
        setattr(m, k, v)
    return m

def build_fused(nL=2, E=4, I=8, H=16, vocab=100):
    """Mimics Qwen3.5/3.6: fused experts, multimodal language_model.* prefix, TopKRouter, vision tower."""
    layers = nn.ModuleList()
    for _ in range(nL):
        mlp = _module(
            experts=_FusedExperts(E, I, H),
            gate=Qwen3_5MoeTopKRouter(E, H),
            shared_expert=_module(gate_proj=nn.Linear(H, I, bias=False),
                                  up_proj=nn.Linear(H, I, bias=False),
                                  down_proj=nn.Linear(I, H, bias=False)),
            shared_expert_gate=nn.Linear(H, 1, bias=False))
        layer = _module(
            self_attn=_module(q_proj=nn.Linear(H, H, bias=False), o_proj=nn.Linear(H, H, bias=False)),
            linear_attn=_module(in_proj=nn.Linear(H, H, bias=False)),
            mlp=mlp)
        layers.append(layer)
    lang = _module(layers=layers, embed_tokens=nn.Embedding(vocab, H))
    model = _module(model=_module(language_model=lang), lm_head=nn.Linear(H, vocab, bias=False),
                    visual=_module(patch=nn.Linear(H, H, bias=False)))
    return model

def build_olmoe_fused(nL=2, E=4, I=8, H=16, vocab=100):
    """Mimics OLMoE under transformers 5.x: FUSED batched experts (gate_up_proj/down_proj 3D tensors),
    OlmoeTopKRouter, text-only, NO language_model.* prefix."""
    layers = nn.ModuleList()
    for _ in range(nL):
        mlp = _module(experts=_FusedExperts(E, I, H), gate=OlmoeTopKRouter(E, H))
        layer = _module(self_attn=_module(q_proj=nn.Linear(H, H, bias=False)), mlp=mlp)
        layers.append(layer)
    model = _module(model=_module(layers=layers, embed_tokens=nn.Embedding(vocab, H)),
                    lm_head=nn.Linear(H, vocab, bias=False))
    return model

def build_separate(nL=2, E=4, I=8, H=16, vocab=100):
    """Mimics a LEGACY per-expert-nn.Linear MoE (pre-batching transformers / Mixtral-style). Handled by
    GenericMoEAdapter's include_expert_linears path."""
    layers = nn.ModuleList()
    for _ in range(nL):
        experts = nn.ModuleList([_module(gate_proj=nn.Linear(H, I, bias=False),
                                         up_proj=nn.Linear(H, I, bias=False),
                                         down_proj=nn.Linear(I, H, bias=False)) for _ in range(E)])
        mlp = _module(experts=experts, gate=nn.Linear(H, E, bias=False))
        layer = _module(self_attn=_module(q_proj=nn.Linear(H, H, bias=False)), mlp=mlp)
        layers.append(layer)
    model = _module(model=_module(layers=layers, embed_tokens=nn.Embedding(vocab, H)),
                    lm_head=nn.Linear(H, vocab, bias=False))
    return model


# --------------------------------------------------------------------- tests
def test_qwen35_iterators_byte_identical():
    model = build_fused()
    ad = MA.Qwen35MoeAdapter()
    assert [n for n, _ in ad.iter_expert_tensors(model)] == [n for n, _ in _ref_expert_tensors(model)]
    assert [n for n, _ in ad.iter_quant_linears(model)] == [n for n, _ in _ref_quant_linears(model)]
    # expert tensors: exactly the 2 fused params per layer, none from the vision tower
    exp = [n for n, _ in ad.iter_expert_tensors(model)]
    assert len(exp) == 2 * 2 and all("visual" not in n for n in exp)

def test_qwen35_classification_identical():
    model = build_fused()
    ad = MA.Qwen35MoeAdapter()
    cat_bits = {"shared_expert": 8.0, "attn": 4.0, "ssm": 4.0, "embed": 4.0, "expert": 3.0, "backbone": 4.0}
    for n, _ in ad.iter_quant_linears(model):
        assert cat_bits.get(ad.classify_linear(n), 4.0) == _ref_linear_bits(n), n

def test_qwen35_router_and_expert_names():
    model = build_fused()
    ad = MA.Qwen35MoeAdapter()
    routers = [n for n, m in model.named_modules() if ad.is_router(m)]
    assert len(routers) == 2 and all(n.endswith("mlp.gate") for n in routers)
    gu, dn = ad.expert_tensor_names(0)
    assert gu == "model.language_model.layers.0.mlp.experts.gate_up_proj"
    assert dn == "model.language_model.layers.0.mlp.experts.down_proj"

def test_effective_bits_flat_recipe_not_overcounted():
    model = build_fused()
    MQ.set_active_adapter(MA.Qwen35MoeAdapter())
    spec = MQ.AllocSpec(default_expert_bits=3.0, default_linear_bits=4.0, group=128)
    eb = MQ.effective_bits(model, spec)
    # experts must count at 3b (the bits apply_ptq actually applies), NOT the old 16b artifact
    assert abs(eb["avg_bits_experts"] - 3.0) < 1e-6, eb
    assert eb["avg_bits_all"] < 5.0, eb            # dominated by 3b experts, not ~15b

def test_olmoe_fused_experts_caught():
    # OLMoE under transformers 5.x is FUSED -> experts ride iter_expert_tensors, not the linear loop
    model = build_olmoe_fused()
    ad = MA.get_adapter("olmoe")                                 # -> FlatMoEAdapter (standard fused MoE)
    et = [n for n, _ in ad.iter_expert_tensors(model)]
    assert len(et) == 2 * 2, et                              # nL x {gate_up_proj, down_proj}
    assert all(n.endswith(("experts.gate_up_proj", "experts.down_proj")) for n in et)
    ql = [n for n, _ in ad.iter_quant_linears(model)]
    assert not any(".experts." in n for n in ql)             # fused experts NOT double-counted as linears
    assert ad.is_router(model.model.layers[0].mlp.gate)       # OlmoeTopKRouter detected -> protected
    assert not ad.supports_opd                                # OLMoE is not OPD-supported (Qwen-only)

def test_legacy_separate_experts_now_quantized():
    # GenericMoEAdapter handles the legacy per-expert-nn.Linear layout (old code skipped these entirely)
    model = build_separate()
    ad = MA.GenericMoEAdapter()
    names = [n for n, _ in ad.iter_quant_linears(model)]
    expert_linears = [n for n in names if re.search(r"\.experts\.\d+\.", n)]
    assert len(expert_linears) == 2 * 4 * 3, names           # nL*E*3 — the experts the old code SKIPPED
    assert all(ad.classify_linear(n) == "expert" for n in expert_linears)
    assert not any(n.endswith("mlp.gate") for n in names)     # nn.Linear gate protected via router_linear_re
    old = [n for n, _ in MA.Qwen35MoeAdapter().iter_quant_linears(model)]
    assert not any(re.search(r"\.experts\.\d+\.", n) for n in old)

def test_generic_flat_spec_tags_experts_at_expert_bits():
    model = build_separate()
    ad = MA.GenericMoEAdapter()
    spec = ad.flat_spec(model, expert_bits=3.0, linear_bits=4.0, group=128)
    for n in spec.linear_bits:
        want = 3.0 if re.search(r"\.experts\.\d+\.", n) else 4.0
        assert spec.linear_bits[n] == want, (n, spec.linear_bits[n])
    assert any(re.search(r"\.experts\.\d+\.", n) for n in spec.linear_bits)


def test_adapter_registry_resolution():
    class C:  # fake config objects
        pass
    fused = C(); fused.model_type = "qwen3_5_moe"
    assert isinstance(MA.get_adapter(fused), MA.Qwen35MoeAdapter)
    olmoe = C(); olmoe.model_type = "olmoe"
    assert isinstance(MA.get_adapter(olmoe), MA.FlatMoEAdapter)        # standard fused MoE family
    dense = C(); dense.model_type = "llama"
    assert isinstance(MA.get_adapter(dense), MA.DenseAdapter)          # dense model
    unknown = C(); unknown.model_type = "some_new_moe"
    assert isinstance(MA.get_adapter(unknown), MA.GenericMoEAdapter)   # graceful fallback


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  PASS  {fn.__name__}")
    print(f"ALL {len(fns)} ADAPTER TESTS PASSED")
