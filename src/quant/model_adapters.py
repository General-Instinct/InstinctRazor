#!/usr/bin/env python3
"""Model adapters: decouple MoE quant/probe/alloc from model-specific naming.

The quant MATH in moe_quant.py is already model-agnostic; the only model-specific logic is
*model walking* — how to find experts, backbone linears, and routers by name, which AutoModel
class to load, and whether the config is nested (multimodal `text_config`) or flat. A
`ModelAdapter` encapsulates exactly that.

Adapters are keyed on `config.model_type` (NOT the model-id string), so Qwen3.6-35B-A3B — which
reports `model_type: qwen3_5_moe` just like Qwen3.5-122B-A10B — resolves to the same adapter for
free. The default `Qwen35MoeAdapter` reproduces the original hardcoded behavior byte-for-byte
(see tests/test_adapter_invariance.py). A genuinely new MoE family is a one-line `@register`.

Two expert layouts are supported:
  - FUSED (Qwen3.5/3.6): experts are 3D nn.Parameter tensors `...experts.gate_up_proj` [E,2I,H] /
    `...experts.down_proj` [E,H,I]. Quantized by moe_quant.apply_ptq's expert loop.
  - SEPARATE (Mixtral/Qwen3-MoE/OLMoE): experts are per-expert nn.Linear modules
    (`...experts.{e}.gate_proj` ...). These flow through the EXISTING iter_quant_linears loop
    tagged 'expert' — the per-block quant math (group along the contraction axis) is identical
    for a 2D [out,in] expert Linear and a 3D [E,out,in] fused tensor, so no new math is needed.
"""
import re
import torch
import torch.nn as nn
import moe_quant as MQ

_REGISTRY = {}   # model_type -> ModelAdapter subclass


def register(*model_types):
    def deco(cls):
        for mt in model_types:
            _REGISTRY[mt] = cls
        return cls
    return deco


def get_adapter(config_or_model_type):
    """Resolve an adapter from a HF config (preferred) or a model_type string.
    Tries the top-level model_type, then a nested text_config.model_type, then the generic
    separate-expert fallback (never raises on unknown — degrades gracefully)."""
    if isinstance(config_or_model_type, str):
        return _REGISTRY.get(config_or_model_type, GenericMoEAdapter)()
    cfg = config_or_model_type
    for mt in _candidate_model_types(cfg):
        if mt in _REGISTRY:
            return _REGISTRY[mt]()
    return GenericMoEAdapter()


def _candidate_model_types(cfg):
    out = []
    mt = getattr(cfg, "model_type", None)
    if mt:
        out.append(mt)
    tc = getattr(cfg, "text_config", None)
    if tc is not None and getattr(tc, "model_type", None):
        out.append(tc.model_type)
    return out


def default_adapter():
    """The adapter assumed when none has been set — preserves the original Qwen3.5 behavior."""
    return Qwen35MoeAdapter()


# --------------------------------------------------------------------------- base adapter
class ModelAdapter:
    # ---- config / loading ----
    nested_text_config = True                # model.config.text_config (multimodal) vs flat config
    auto_model_class = "image_text_to_text"  # "image_text_to_text" | "causal_lm"

    # ---- expert layout ----
    # Modern transformers (5.x) BATCHES MoE experts into FUSED 3D tensors (`experts.gate_up_proj`
    # [E,2I,H] / `experts.down_proj` [E,H,I]) even when the checkpoint stores per-expert weights —
    # verified true for Qwen3.5/3.6 AND OLMoE. iter_expert_tensors handles that (the common) case.
    # Set include_expert_linears=True only for a model that still exposes per-expert nn.Linear modules.
    include_expert_linears = False           # True -> iter_quant_linears also yields `.experts.{e}.` Linears
    expert_linear_re = None                  # regex identifying a per-expert Linear (when include_expert_linears)
    fused_expert_suffixes = ("experts.gate_up_proj", "experts.down_proj")
    supports_opd = False                     # OPD/expert-LoRA monkeypatches Qwen3_5MoeExperts -> Qwen3.5/3.6 only

    @property
    def separate_experts(self):
        """Back-compat alias: True iff experts are exposed as separate per-expert nn.Linear modules."""
        return self.include_expert_linears

    # ---- naming ----
    visual_re = MQ.VISUAL_RE                  # weights to never quantize (vision tower)
    router_class_names = ("Qwen3_5MoeTopKRouter",)
    # Protect the MoE gate/router (keep it BF16 — it is tiny and sensitive). Matches the gate/router MODULE
    # across families: `...mlp.gate`, `...mlp.router`, `...ffn.router.layer`, `...block_sparse_moe.router.layer`,
    # `...feed_forward.router`, etc. Does NOT match `gate_proj` / `gate_up_proj` (the "_proj" suffix). This is
    # the default for ALL adapters: in transformers 5.x several families expose the gate as an nn.Linear
    # (phimoe, deepseek_v2, jamba) which would otherwise be silently quantized.
    router_linear_re = re.compile(r"\.(gate|router)(\.|$)")
    embed_name_substr = "language_model"      # only quantize nn.Embedding whose name contains this; None=all

    # ---- config shape ----
    def text_config(self, config):
        if self.nested_text_config:
            return getattr(config, "text_config", config)
        return config

    def layer_idx_of(self, name):
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else -1

    # ---- fused expert tensor iteration (the common case; empty when a model has no fused tensors) ----
    def iter_expert_tensors(self, model):
        for n, p in model.named_parameters():
            if self.visual_re.search(n):
                continue
            if any(n.endswith(s) for s in self.fused_expert_suffixes):
                yield n, p

    # ---- backbone (and, for separate-expert models, expert) linear/embed iteration ----
    def iter_quant_linears(self, model, include_embed=True):
        for n, m in model.named_modules():
            if self.visual_re.search(n):
                continue
            if isinstance(m, nn.Linear):
                if not self.include_expert_linears and ".experts." in n:
                    continue
                if self.router_linear_re is not None and self.router_linear_re.search(n):
                    continue
                yield n, m
            elif include_embed and isinstance(m, nn.Embedding) and self._include_embedding(n):
                yield n, m

    def _include_embedding(self, name):
        return self.embed_name_substr is None or self.embed_name_substr in name

    # ---- linear classification -> bit category ----
    def classify_linear(self, name):
        if self.include_expert_linears and self.expert_linear_re is not None and self.expert_linear_re.search(name):
            return "expert"
        if ".shared_expert" in name:
            return "shared_expert"
        if ".self_attn." in name:
            return "attn"
        if ".linear_attn." in name:
            return "ssm"
        if name.endswith("embed_tokens") or name.endswith("lm_head"):
            return "embed"
        return "backbone"

    # ---- router detection / output ----
    def is_router(self, module):
        """A module is a MoE router/gate if it's a known class or its class name ends Router/Routing/Gating —
        covers MixtralTopKRouter, Qwen*MoeTopKRouter, DeepseekV3TopkRouter, GraniteMoeTopKGating, etc.
        (Plain-nn.Linear gates have no router class; they are still PROTECTED by name via router_linear_re.)"""
        n = type(module).__name__
        return n in self.router_class_names or n.endswith(("Router", "Routing", "Gating"))

    def read_router_output(self, out):
        """Normalize the per-family router forward output to (scores, indices)."""
        logits, scores, indices = out
        return scores, indices

    # ---- fused-expert param names for layer L (used by alloc to BUILD the spec dict) ----
    def expert_tensor_names(self, L):
        base = f"model.language_model.layers.{L}.mlp.experts"
        return (f"{base}.gate_up_proj", f"{base}.down_proj")

    # ---- flat (uniform) spec for quant_save ----
    def flat_spec(self, model, expert_bits, linear_bits, group):
        """The shipped uniform recipe: experts @ expert_bits, backbone @ linear_bits.
        Fused path is byte-identical to the original quant_save AllocSpec."""
        if not self.include_expert_linears:
            # fused experts ride iter_expert_tensors via default_expert_bits (byte-identical to original)
            return MQ.AllocSpec(default_expert_bits=expert_bits, default_linear_bits=linear_bits, group=group)
        # per-expert Linears ride the linear loop -> tag them; default_expert_bits still covers any fused tensors
        lb = {}
        for n, m in self.iter_quant_linears(model):
            lb[n] = expert_bits if self.classify_linear(n) == "expert" else linear_bits
        return MQ.AllocSpec(linear_bits=lb, default_linear_bits=linear_bits,
                            default_expert_bits=expert_bits, group=group)


# --------------------------------------------------------------------------- registered adapters
# Quant coverage is broad because transformers 5.x batches almost every MoE family into FUSED 3D expert
# tensors named `...experts.gate_up_proj` / `...experts.down_proj` — matched by the base fused_expert_suffixes.
# So most families need only registration + config flags (flat-vs-nested, causal-vs-VLM); experts ride the
# fused path, gate/router is protected by the base router_linear_re, and shared experts ride the backbone path.
# Atypical expert namings (DBRX, GraniteMoE) override fused_expert_suffixes.

@register("qwen3_5_moe")
class Qwen35MoeAdapter(ModelAdapter):
    """Qwen3.5-122B-A10B AND Qwen3.6-35B-A3B (both report model_type=qwen3_5_moe). The base defaults already
    encode the original hardcoded behavior verbatim (multimodal loader, nested text_config, `language_model.*`
    names). OPD / expert-LoRA recovery is wired here only."""
    supports_opd = True


@register("deepseek_v2", "phimoe", "gpt_oss", "minimax", "jamba")
class FlatMoEAdapter(ModelAdapter):
    """Fused-expert MoE that loads as a flat text CausalLM — the common case in transformers 5.x. Experts at
    `...experts.gate_up_proj`/`down_proj` (default suffixes); gate/router (`mlp.gate`, `mlp.router`,
    `feed_forward.router`, …) protected by the base regex; shared experts (`.shared_expert(s)`) ride the
    backbone path. Quantization only (supports_opd stays False): these families either differ from the OPD
    expert-forward reference (gpt_oss: bias + clamp + interleaved GLU) or are unverified for OPD."""
    nested_text_config = False
    auto_model_class = "causal_lm"
    embed_name_substr = None

    def expert_tensor_names(self, L):
        base = f"model.layers.{L}.mlp.experts"
        return (f"{base}.gate_up_proj", f"{base}.down_proj")


@register("mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "qwen3_next", "deepseek_v3")
class OpdMoEAdapter(FlatMoEAdapter):
    """Group-A fused MoE: the experts module exposes the SAME forward as the Qwen reference — params
    `gate_up_proj`/`down_proj`, `gate,up = (x@gate_up_proj[e]).chunk(2); h = act_fn(gate)*up; out = h@down_proj[e]`,
    routing computed in the parent and passed as `(hidden_states, top_k_index, top_k_weights)`. So the OPD
    expert-LoRA STE hook (moe_lora) works as-is and these are OPD-capable. (deepseek_v3's experts class is
    `DeepseekV3NaiveMoe`; OPD requires eager experts impl — the default. The hybrid jamba/gpt_oss/etc. stay on
    the quant-only FlatMoEAdapter.)"""
    supports_opd = True


@register("dbrx")
class DbrxAdapter(FlatMoEAdapter):
    """DBRX: experts are `ffn.experts.mlp.{w1,v1,w2}` (not gate_up/down_proj); attention is `attn`; gate is
    `ffn.router.layer` (protected by the base router regex). EXPERIMENTAL — verify expert shapes quantize."""
    fused_expert_suffixes = ("experts.mlp.w1", "experts.mlp.v1", "experts.mlp.w2")


@register("granitemoe")
class GraniteMoeAdapter(FlatMoEAdapter):
    """GraniteMoE: 3D parallel-expert tensors `block_sparse_moe.{input_linear,output_linear}` (no `.experts.`
    segment). EXPERIMENTAL — verify expert shapes quantize."""
    fused_expert_suffixes = ("block_sparse_moe.input_linear.weight", "block_sparse_moe.output_linear.weight",
                             "block_sparse_moe.input_linear", "block_sparse_moe.output_linear")


@register("llama", "mistral", "qwen2", "qwen3", "gemma2", "gemma3_text", "phi3")
class DenseAdapter(ModelAdapter):
    """Dense models (no MoE): quantize all attention + MLP linears, protect norms/embeds. The quant math is
    shape-agnostic, so this is just the backbone path with zero expert tensors."""
    nested_text_config = False
    auto_model_class = "causal_lm"
    embed_name_substr = None
    fused_expert_suffixes = ()                # no experts


@register("gemma3")
class Gemma3Adapter(DenseAdapter):
    """Gemma-3 multimodal (`model_type=gemma3` -> ForConditionalGeneration): text under `model.language_model.*`,
    vision tower protected. (The text-only `model_type=gemma3_text` uses the flat DenseAdapter.)"""
    nested_text_config = True
    auto_model_class = "image_text_to_text"
    embed_name_substr = "language_model"


class GenericMoEAdapter(ModelAdapter):
    """Fallback for an unrecognized model_type. Handles fused experts via the default suffixes AND legacy
    per-expert nn.Linear modules (older transformers) via include_expert_linears, so experts are never
    silently skipped; gate/router protected by the base regex; works for dense models too (no experts found).
    Text-only, flat config."""
    nested_text_config = False
    auto_model_class = "causal_lm"
    include_expert_linears = True
    expert_linear_re = re.compile(r"\.experts\.\d+\.")
    embed_name_substr = None


# --------------------------------------------------------------------------- centralized loader
def load_model(model_id, max_mem_gib=None, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True):
    """Single HF loader: peek config -> pick adapter -> pick AutoModel class -> load.
    Returns (model, adapter) and sets moe_quant's active adapter so the iterators delegate correctly.
    Preserves quant_save's ImageTextToText -> CausalLM fallback for multimodal families."""
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    adapter = get_adapter(cfg)
    n = torch.cuda.device_count()
    mm = {i: f"{max_mem_gib}GiB" for i in range(n)} if (max_mem_gib and n) else None
    kw = dict(dtype=dtype, device_map=device_map, max_memory=mm, trust_remote_code=trust_remote_code)
    if adapter.auto_model_class == "image_text_to_text":
        try:
            model = transformers.AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        except Exception as e:
            print(f"[load_model] ImageTextToText failed ({str(e)[:80]}); CausalLM fallback", flush=True)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    MQ.set_active_adapter(adapter)
    return model, adapter
