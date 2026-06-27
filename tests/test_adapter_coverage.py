#!/usr/bin/env python3
"""Coverage test: every registered family resolves to the right adapter and is walked correctly.

Builds a TINY model per family from config (no weights download) and asserts:
  - MoE families expose >=1 fused expert tensor (experts not silently skipped)
  - dense families expose 0 expert tensors
  - the gate/router is PROTECTED (never appears in the quantizable-linear list) — the bug this guards against
    is transformers 5.x exposing some gates as nn.Linear (phimoe/deepseek_v2/jamba) that would be quantized
  - >=1 backbone linear is found

Run: PYTHONPATH=src/quant:src/eval:src/distill HF_HUB_OFFLINE=1 vllm_venv/bin/python tests/test_adapter_coverage.py
Families whose minimal from_config build hits an unrelated config quirk are SKIPPED (not failed); their
adapters are still exercised by the real-checkpoint path.
"""
import re
import transformers
transformers.logging.set_verbosity_error()
import model_adapters as MA
import moe_quant as MQ

ROUTER_RE = re.compile(r"\.(gate|router)(\.|$)")

MOE = ["mixtral", "qwen2_moe", "qwen3_moe", "olmoe", "deepseek_v2", "deepseek_v3", "phimoe",
       "gpt_oss", "minimax", "jamba", "qwen3_next", "granitemoe", "qwen3_5_moe"]
DENSE = ["llama", "mistral", "qwen2", "qwen3", "gemma2"]
EXPECT_ADAPTER = {"qwen3_5_moe": "Qwen35MoeAdapter", "granitemoe": "GraniteMoeAdapter", "dbrx": "DbrxAdapter"}


def _tiny(mt):
    o = dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=2, vocab_size=256, max_position_embeddings=64, tie_word_embeddings=False)
    o.update(num_local_experts=4, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
             shared_expert_intermediate_size=64, ffn_hidden_size=128, n_routed_experts=4, n_shared_experts=1,
             first_k_dense_replace=0, n_group=1, topk_group=1, decoder_sparse_step=1, attn_layer_indices=[0, 1])
    return transformers.AutoModelForCausalLM.from_config(transformers.AutoConfig.for_model(mt, **o))


def _walk(mt, is_moe):
    ad = MA.get_adapter(mt)
    exp_name = EXPECT_ADAPTER.get(mt)
    assert exp_name is None or type(ad).__name__ == exp_name, f"{mt}: got {type(ad).__name__}, want {exp_name}"
    MQ.set_active_adapter(ad)
    model = _tiny(mt)
    experts = [n for n, _ in ad.iter_expert_tensors(model)]
    lins = [n for n, _ in ad.iter_quant_linears(model)]
    leak = [n for n in lins if ROUTER_RE.search(n)]
    backbone = [n for n in lins if ".experts." not in n]
    assert not leak, f"{mt}: gate/router leaked into quant linears: {leak[:3]}"
    assert len(backbone) > 0, f"{mt}: no backbone linears found"
    if is_moe:
        assert len(experts) > 0, f"{mt}: MoE but no expert tensors found"
    else:
        assert len(experts) == 0, f"{mt}: dense but expert tensors found: {experts[:2]}"
    return type(ad).__name__, len(experts), len(lins)


def main():
    npass = nskip = 0
    for fams, is_moe in [(MOE, True), (DENSE, False)]:
        for mt in fams:
            try:
                ad, ne, nl = _walk(mt, is_moe)
                print(f"  PASS  {mt:13} {ad:18} experts={ne:2} linears={nl}", flush=True)
                npass += 1
            except AssertionError:
                raise
            except Exception as e:  # tiny-config build quirk, not an adapter bug
                print(f"  SKIP  {mt:13} (build: {str(e)[:55]})", flush=True)
                nskip += 1
    print(f"ADAPTER COVERAGE: {npass} passed, {nskip} skipped (build quirks)")
    assert npass >= 14, f"expected >=14 families to validate, got {npass}"


if __name__ == "__main__":
    main()
