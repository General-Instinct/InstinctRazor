#!/usr/bin/env python3
"""Master driver for the 122B MoE allocation study — loads the model ONCE, then:
  1) probe routing stats (freq/wmass/asal) + report frequency-skew (is routing load-balanced?)
  2) compute weight stats (Frobenius norms, router norms)
  3) snapshot FP weights to CPU
  4) BF16 baseline evals (calib ppl/KL + MMLU + GSM8K + HumanEval)
  5) loop allocation specs: restore FP -> build AllocSpec -> apply_ptq -> eval -> save incrementally

Amortizes the ~226s 250GB load across the whole sweep. Results stream to results/study/<tag>.json.
"""
import argparse, json, os, time, re, gc
import numpy as np
import torch
import torch.nn.functional as F
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import transformers
transformers.logging.set_verbosity_error()
try:
    transformers.utils.logging.disable_progress_bar()
except Exception:
    pass
import moe_eval as EV
import moe_quant as Q
import moe_alloc as A
import moe_probe as P

OUTDIR = "results/study"
_MAXMEM = 66

def load_model(model_id, max_mem):
    from transformers import AutoModelForImageTextToText
    n = torch.cuda.device_count()
    m = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        max_memory={i: f"{max_mem}GiB" for i in range(n)}, trust_remote_code=True).eval()
    return m

def _free():
    gc.collect(); torch.cuda.empty_cache()

def spec_list():
    """The allocation sweep. gen=True specs also get GSM8K+HumanEval (slow)."""
    specs = [{"tag": "bf16", "bf16": True, "gen": True}]
    B = 3.0  # main expert-avg-bit budget
    # strategy comparison at the main budget (calib+mmlu screen; gen on a few)
    for strat in ["composite", "wmass", "asal", "wfro", "freq", "blind", "random", "inverse"]:
        specs.append({"tag": f"s_{strat}_b30", "strategy": strat, "expert_bits": B,
                      "tagset": "t4", "protect": True, "global_rank": True,
                      "gen": strat in ("composite", "blind", "inverse")})
    # protection ablation (E1) at composite b3.0
    specs.append({"tag": "s_composite_b30_noprot", "strategy": "composite", "expert_bits": B,
                  "tagset": "t4", "protect": False, "global_rank": True, "gen": False})
    # per-layer vs global rank
    specs.append({"tag": "s_composite_b30_perlayer", "strategy": "composite", "expert_bits": B,
                  "tagset": "t4", "protect": True, "global_rank": False, "gen": False})
    # budget curve (composite) — incl low-bit floor search for minimum effective precision
    for bb in [2.0, 2.25, 2.5, 3.5, 4.0]:
        specs.append({"tag": f"s_composite_b{str(bb).replace('.', '')}", "strategy": "composite",
                      "expert_bits": bb, "tagset": "t4", "protect": True, "global_rank": True,
                      "gen": bb in (2.0, 2.25, 2.5, 3.5, 4.0)})
    # uniform int3 reference (all experts exactly 3.0)
    specs.append({"tag": "s_uniform_int3", "strategy": "blind", "expert_bits": 3.0,
                  "tagset": "t34_pure3", "protect": True, "global_rank": True, "gen": True})
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--max-mem-gib", type=int, default=74)
    ap.add_argument("--calib-n", type=int, default=16)
    ap.add_argument("--calib-len", type=int, default=384)
    ap.add_argument("--probe-n", type=int, default=48)
    ap.add_argument("--mmlu-n", type=int, default=200)
    ap.add_argument("--gsm-n", type=int, default=80)
    ap.add_argument("--he-n", type=int, default=40)
    ap.add_argument("--gpqa-n", type=int, default=100)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--only", default=None, help="comma list of spec tags to run (default all)")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    from transformers import AutoProcessor
    mm = args.max_mem_gib
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # ---- one-time setup load: probe + weight stats + teacher (all spec-independent) ----
    print(f"[study] setup-load {args.model} ...", flush=True); t0 = time.time()
    model = load_model(args.model, mm)
    print(f"[study] loaded in {time.time()-t0:.0f}s", flush=True)
    cfg = model.config.text_config
    E, nL = cfg.num_experts, cfg.num_hidden_layers
    print("[study] probing routing stats ...", flush=True)
    probe = run_probe(model, tok, args.probe_n, args.calib_len, nL, E)
    json.dump(probe, open("results/moe_probe_122b.json", "w"))
    report_skew(probe, nL, E)
    print("[study] computing weight stats (Frobenius norms) ...", flush=True)
    stats = A.compute_weight_stats(model)
    calib = P.build_calib(args.calib_n, args.calib_len, tok)
    print(f"[study] calib {len(calib)} seqs; caching BF16 teacher ...", flush=True)
    teacher = EV.cache_teacher(model, calib, topk=256)
    print(f"[study] teacher ppl={float(np.exp(teacher['teacher_nll'])):.2f}", flush=True)
    del model; _free()

    mmlu = EV.build_mmlu(args.mmlu_n); gsm = EV.build_gsm8k(args.gsm_n)
    he = EV.build_humaneval(args.he_n); gpqa, gpqa_cfg = EV.build_gpqa(args.gpqa_n)
    print(f"[study] datasets: mmlu={len(mmlu)} gsm={len(gsm)} he={len(he)} gpqa={len(gpqa) if gpqa else 'N/A'}", flush=True)

    only = set(args.only.split(",")) if args.only else None
    specs = [s for s in spec_list() if (only is None or s["tag"] in only)]
    # skip bf16 if already saved (it is expensive and spec-independent)
    specs = [s for s in specs if not (s.get("bf16") and os.path.exists(f"{OUTDIR}/bf16.json"))]
    # run fast screen-only specs first so the allocation ranking lands quickly; gen specs last
    specs.sort(key=lambda s: bool(s.get("gen", False)))
    print(f"[study] {len(specs)} specs to run (reload-per-spec; screen-only first)", flush=True)

    for s in specs:
      tag = s["tag"]; outp = f"{OUTDIR}/{tag}.json"
      try:
        print(f"\n[study] === {tag} ===", flush=True); tspec = time.time()
        model = load_model(args.model, mm)               # fresh FP model (RAM-safe; ~40s from page cache)
        if s.get("bf16"):
            ebits = {"avg_bits_all": 16.0, "avg_bits_experts": 16.0, "expert_param_frac": 0.946}
            rec = {"tag": tag, "meta": {"bf16": True}}
        else:
            spec, _ = build_spec_from(s, model, probe, stats)
            Q.apply_ptq(model, spec, verbose=False)
            ebits = Q.effective_bits(model, spec)
            rec = {"tag": tag, "meta": spec.meta}
        rec["eff_bits"] = ebits
        # screen
        rec.update(EV.eval_calib(model, calib, teacher))
        rec.update(EV.eval_mmlu(model, tok, mmlu, batch=8))
        print(f"  avg_bits={ebits['avg_bits_all']:.2f} exp={ebits['avg_bits_experts']:.2f} "
              f"ppl={rec['ppl']:.2f} kl={rec['topk_kl']:.3f} mmlu={rec['mmlu_acc']:.1f}", flush=True)
        if s.get("gen"):
            if gpqa:
                g = EV.eval_gpqa(model, tok, gpqa, batch=8); rec["gpqa_acc"] = g["gpqa_acc"]; rec["gpqa_cfg"] = gpqa_cfg
                print(f"  gpqa={rec['gpqa_acc']:.1f}", flush=True)
            tg = time.time()
            rec.update(EV.eval_gsm8k(model, tok, gsm, batch=args.gen_batch))
            rec.update(EV.eval_humaneval(model, tok, he, batch=args.gen_batch))
            print(f"  gsm8k={rec['gsm8k_acc']:.1f} humaneval={rec['humaneval_pass@1']:.1f} (gen {time.time()-tg:.0f}s)", flush=True)
        rec["seconds"] = time.time() - tspec
        json.dump(rec, open(outp, "w"), indent=2)
        print(f"  wrote {outp} ({rec['seconds']:.0f}s)", flush=True)
      except Exception as e:
        import traceback
        print(f"  [ERROR] spec {tag} failed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
      finally:
        try: del model
        except Exception: pass
        _free()
    print("\n[study] DONE", flush=True)


def build_spec_from(s, model, probe, stats):
    tagset = s["tagset"]
    if tagset == "t34_pure3":  # all experts exactly int3
        return _uniform_int3_spec(model), None
    return A.build_spec(model, probe, strategy=s["strategy"], expert_bits=s["expert_bits"],
                        tagset=tagset, protect=s["protect"], global_rank=s["global_rank"],
                        seed=s.get("seed", 0), precomputed_stats=stats)

def _uniform_int3_spec(model):
    cfg = model.config.text_config; E, nL = cfg.num_experts, cfg.num_hidden_layers
    eb = {}
    for L in range(nL):
        t = torch.full((E,), 3.0)
        eb[f"model.language_model.layers.{L}.mlp.experts.gate_up_proj"] = t.clone()
        eb[f"model.language_model.layers.{L}.mlp.experts.down_proj"] = t.clone()
    lb = {}
    for n, m in Q.iter_quant_linears(model):
        if ".shared_expert" in n: lb[n] = 8.0
        else: lb[n] = 4.0
    return Q.AllocSpec(expert_bits=eb, linear_bits=lb, default_linear_bits=4.0,
                       meta={"strategy": "uniform_int3", "expert_bits_target": 3.0, "protect": True})


@torch.no_grad()
def run_probe(model, tok, n_seq, seqlen, nL, E):
    freq = torch.zeros(nL, E, dtype=torch.float64); wmass = torch.zeros(nL, E, dtype=torch.float64)
    asal = torch.zeros(nL, E, dtype=torch.float64)
    hooks = []
    def mk(L):
        def hook(mod, inp, out):
            hs = inp[0]; _, sc, idx = out
            idx = idx.detach().reshape(-1).cpu(); sc = sc.detach().float().reshape(-1).cpu()
            tn = hs.detach().float().norm(dim=-1); k = out[2].shape[-1]
            tnr = tn.unsqueeze(-1).expand(-1, k).reshape(-1).cpu()
            freq[L].index_add_(0, idx, torch.ones_like(sc, dtype=torch.float64))
            wmass[L].index_add_(0, idx, sc.double()); asal[L].index_add_(0, idx, tnr.double())
        return hook
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Qwen3_5MoeTopKRouter":
            L = int(re.search(r"layers\.(\d+)\.", name).group(1)); hooks.append(mod.register_forward_hook(mk(L)))
    seqs = P.build_calib(n_seq, seqlen, tok); dev = next(model.parameters()).device
    for i, ids in enumerate(seqs):
        model(input_ids=ids.to(dev), use_cache=False, logits_to_keep=1)
    for h in hooks: h.remove()
    pf = freq + 1e-9; pf = pf / pf.sum(1, keepdim=True)
    ent = (-(pf * pf.log()).sum(1) / np.log(E)).tolist()
    return {"freq": freq.tolist(), "wmass": wmass.tolist(), "asal": asal.tolist(),
            "num_experts": E, "num_layers": nL, "n_seq": len(seqs),
            "layer_norm_entropy": ent}

def report_skew(probe, nL, E):
    freq = np.array(probe["freq"]); g = freq.sum(0)
    ent = np.array(probe["layer_norm_entropy"])
    print(f"[skew] global per-expert freq: max/mean={g.max()/g.mean():.2f} cold(<10%mean)={(g<0.1*g.mean()).sum()}/{E} "
          f"zero={(g==0).sum()}/{E}", flush=True)
    print(f"[skew] norm. routing entropy mean={ent.mean():.3f} (1=uniform) min={ent.min():.3f} "
          f"early={ent[:nL//3].mean():.3f} mid={ent[nL//3:2*nL//3].mean():.3f} late={ent[2*nL//3:].mean():.3f}", flush=True)
    for topn in [16, 32, 64]:
        fr = [np.sort(freq[L])[::-1][:topn].sum() / max(1, freq[L].sum()) for L in range(nL)]
        print(f"[skew] top-{topn} experts capture {100*np.mean(fr):.1f}% (uniform={100*topn/E:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
