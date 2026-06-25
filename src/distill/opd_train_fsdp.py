#!/usr/bin/env python3
"""Lightning-OPD Phase C on FSDP2 (4-GPU DATA-PARALLEL) — replaces the 1x device_map pipeline (~4x faster, even
weight memory). Frozen BF16 base sharded via fully_shard; tiny rank-16 per-expert LoRA shards WITH the experts
unit (grad/optimizer memory negligible). Forward peak controlled: chunked 248K-vocab reverse-KL (VCHUNK=128),
vision/MTP freed, embed+norm+lm_head wrapped as their own units so opd_loss can call backbone()/lmhead()
separately (chunked logits) without the full-vocab CE materialization that OOMs the SFT smoke.

Each rank trains a DISJOINT slice of the rollouts (DP); FSDP reduce-scatters the (sharded) LoRA grads -> correct
data-parallel averaging. Saves the gathered LoRA adapter on rank 0 (.full_tensor()); merge_adapter.py bakes it
into the 47GB ckpt. Run:
  torchrun --nproc_per_node=4 opd_train_fsdp.py --rollouts ... --teacher-lp ... --adapter-out ... [--smoke 2]
"""
import argparse, json, os, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as tcp
import torch.distributed as dist
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import transformers; transformers.logging.set_verbosity_error()
import moe_lora as ML, moe_quant as MQ, model_adapters as MA
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeExperts
except Exception:   # OPD is fused-Qwen-only; placeholders so this module still imports without the arch
    class Qwen3_5MoeDecoderLayer: pass
    class Qwen3_5MoeExperts: pass

V = 248320; VCHUNK = 128   # shared Qwen3.5/3.6 vocab; set from config at load (bound the vocab scratch)


def free_unused(model):
    """Drop the unused vision tower + MTP head so FSDP never shards/holds them (saves GBs)."""
    roots = [model]
    lm = getattr(model, "model", None)
    if lm is not None:
        roots.append(lm)
        inner = getattr(lm, "model", None)
        if inner is not None: roots.append(inner)
    for r in roots:
        for attr in ("visual", "vision_tower", "mtp"):
            if hasattr(r, attr): setattr(r, attr, nn.Identity())


def text_backbone(model):
    lm = model.get_output_embeddings()
    base = model.model
    if hasattr(base, "language_model"):
        base = base.language_model
    return base, lm


def wrap_fsdp_opd(model, backbone, lmhead, lora_params=(), cpu_offload=True):
    """fully_shard inner->outer. CRUCIAL: backbone (embed+norm) and lmhead are their OWN units so calling them
    separately in opd_loss unshards them via their hooks. lmhead reshard_after_forward=False -> gathered once,
    reused across the VCHUNK loop (no per-chunk all-gather).
    cpu_offload=True adds CPUOffloadPolicy: the frozen base shards live in CPU RAM (~885GB) and are fetched to
    GPU per unit-use, freeing ~61GB/GPU of VRAM for ACTIVATIONS. That lets us DISABLE gradient checkpointing
    (see main) -> no recompute -> the nondeterministic-recompute CheckpointError (moe_lora's dynamic per-expert
    loop) cannot fire. The patched per-expert STE forward is unchanged (it still sees the full all-gathered
    tensor on GPU). Trades param-fetch speed for correctness — fine for a short OPD round.
    lora_params: the per-expert LoRA params are passed as FSDP `ignored_params` to EVERY wrap that encloses an
    experts module (experts/decoder/root) so they stay UN-sharded (replicated full tensors). Rationale: the
    patched forward indexes them per-expert (lora.Bgu[e]); if FSDP shards them along dim-0 (E), the gather
    happens for forward but grad does NOT flow back to the sharded DTensor on backward -> zero LoRA grad. Kept
    replicated, lora.Bgu[e] is a plain local index and grad accumulates normally; DP correctness is restored by
    a manual all-reduce(AVG) of the LoRA grads in the train loop (see main). LoRA is ~0.9GB at rank 16."""
    ign = set(lora_params)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    kw = dict(mp_policy=mp, reshard_after_forward=True, ignored_params=ign)
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffloadPolicy
        kw["offload_policy"] = CPUOffloadPolicy()
    for m in model.modules():
        if isinstance(m, Qwen3_5MoeExperts): fully_shard(m, **kw)
    for m in model.modules():
        if isinstance(m, Qwen3_5MoeDecoderLayer): fully_shard(m, **kw)
    fully_shard(backbone, **kw)                                   # text decoder unit: embed + final norm
    lmkw = dict(mp_policy=mp, reshard_after_forward=False)        # output head: keep gathered across chunks
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffloadPolicy
        lmkw["offload_policy"] = CPUOffloadPolicy()
    fully_shard(lmhead, **lmkw)
    fully_shard(model, **kw)                                     # root (≈empty flat param after inner wraps)
    for m in model.modules():                                    # no prefetch: one unit resident at a time
        if hasattr(m, "set_modules_to_forward_prefetch"):
            try: m.set_modules_to_forward_prefetch([]); m.set_modules_to_backward_prefetch([])
            except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--teacher-lp", required=True)
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--bits", type=float, default=3.0); ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--reverse", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=0)   # save adapter every N steps (0=only at end)
    ap.add_argument("--cpu-offload", type=int, default=0)  # CPU-offload frozen base (slow per-step fetch + 1.65x-mem pathology for frozen params). Default OFF: base stays GPU-resident (~17.5GB/GPU) -> compute-bound, full GPU util
    ap.add_argument("--ckpt-experts", type=int, default=1) # ELEGANT path: checkpoint the expert body (recompute reconstructed Wgu/Wdn on backward -> frees the dominant ~64GB; routing is a saved input -> deterministic, no CheckpointError). Unlocks long seq + GPU-resident base
    ap.add_argument("--checkpoint", type=int, default=0)   # LEGACY whole-layer gradient checkpointing (recomputes the router -> CheckpointError on dynamic MoE). Kept for comparison; default OFF in favor of --ckpt-experts
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    local = int(os.environ["LOCAL_RANK"]); rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local); dist.init_process_group("nccl")
    # Deterministic algorithms so gradient-checkpointing RECOMPUTE is bit-identical to the forward -> the MoE
    # router's top-k (and thus expert dispatch shapes) match on recompute. Without this, bf16 GEMM nondeterminism
    # flips routing in recompute -> IndexAddBackward shape mismatch. warn_only=True: ops lacking a deterministic
    # kernel (some fla/FSDP) warn instead of erroring; the key router GEMM is made deterministic via cuBLAS
    # (CUBLAS_WORKSPACE_CONFIG set in the launcher). Needs --cpu-offload + static expert loop (both above).
    torch.use_deterministic_algorithms(True, warn_only=True)
    is0 = rank == 0
    def log(*a):
        if is0: print(*a, flush=True)
    dev = torch.device(f"cuda:{local}"); t0 = time.time()

    # all-rank CPU load (safetensors mmap shares page cache) -> free vision/MTP -> attach LoRA (CPU) -> shard
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
    model.config.use_cache = False
    # set the active adapter (so attach_expert_lora's fused-only guard sees the right model) and read
    # the real vocab size from config (Qwen3.5 and Qwen3.6 both report 248320; never hardcode).
    global V
    adapter = MA.get_adapter(model.config); MQ.set_active_adapter(adapter)
    V = int(getattr(adapter.text_config(model.config), "vocab_size", V))
    free_unused(model)
    torch.manual_seed(0)                          # identical LoRA init on every rank (then sharded)
    MQ.set_clip_search(0)
    # ELEGANT memory strategy (default): checkpoint the EXPERT BODY (--ckpt-experts). The dominant training
    # memory is the reconstructed per-expert weights Wgu/Wdn held for the F.linear backward (~64GB across 40
    # un-checkpointed layers, all 256 experts hit). Checkpointing the expert body recomputes them on backward
    # -> frees ~64GB; because the routing is a SAVED INPUT (the router ran upstream) the recompute is
    # deterministic -> no CheckpointError (unlike legacy --checkpoint, which checkpoints the whole layer incl.
    # the router). Freed term is SEQUENCE-INDEPENDENT, so long seq fits; and the base can stay GPU-resident
    # (no --cpu-offload) -> compute-bound, full GPU utilization. static_loop only for the legacy layer-ckpt path.
    trainable = ML.attach_expert_lora(model, bits=args.bits, group=args.group, rank=args.rank, scale=args.scale,
                                      static_loop=bool(args.checkpoint), ckpt=bool(args.ckpt_experts))
    model.enable_input_require_grads()   # frozen base + reentrant ckpt: make embed output require grad so grad reaches LoRA
    backbone, lmhead = text_backbone(model)
    log(f"[fsdp-opd] loaded+attached on CPU ({time.time()-t0:.0f}s)")
    # LoRA params are FSDP-IGNORED (replicated, not sharded) so per-expert indexing keeps the autograd path.
    wrap_fsdp_opd(model, backbone, lmhead, lora_params=trainable, cpu_offload=bool(args.cpu_offload))
    # FSDP leaves ignored params on their original device (CPU); the patched forward runs on GPU -> move them.
    # Identical across ranks (same torch.manual_seed(0) at attach), so the replicas start bitwise-equal; the
    # per-step grad all-reduce(AVG) in the train loop keeps them in lockstep.
    for p in trainable:
        p.data = p.data.to(dev)
        if p.grad is not None: p.grad = p.grad.to(dev)
    log(f"[fsdp-opd] LoRA kept replicated (FSDP-ignored) on {dev}: "
        f"{sum(p.numel() for p in trainable)/1e6:.1f}M params x{world} ranks")
    # ================= KNOWN BLOCKER: P4 OPD round NOT run (see RESULTS.md "P4") =================
    # The monkeypatched dynamic per-expert MoE forward + gradient-checkpointing RECOMPUTE are fundamentally
    # incompatible, root-caused over 10 --smoke 2 runs (base loads+shards fine via CPU-offload at 3.6GB/GPU):
    #   reentrant ckpt                       -> ZERO LoRA grad (reentrant severs the STE->LoRA path)
    #   non-reentrant ckpt                   -> CheckpointError "diff #tensors saved" (data-dependent hit-loop length)
    #   + set_checkpoint_early_stop(False) / SAC(MUST_SAVE matmuls) -> still CheckpointError (count is the LOOP, not ops)
    #   STATIC expert loop (static_loop=True, below) -> fixes count+grad+OOM, BUT the recompute RE-ROUTES
    #                                           (router top-k differs) -> IndexAddBackward shape mismatch (e.g. 29->7 tokens)
    #   + use_deterministic_algorithms       -> SAME mismatch (not GEMM noise; full determinism unreachable here)
    #   no checkpointing + offload           -> OOM ~77GB (backward holds all 48 layers' gathered params)
    # => gradient-checkpointing recompute x data-dependent MoE indexing cannot be reconciled in FSDP+monkeypatch.
    # Real fix = Expert/Tensor parallelism keeping each expert weight WHOLE (transformers grouped_gemm/ep_plan),
    # a larger rewrite (see deep-research). The OPD round is NOT run (running it would mean faking grad/results);
    # the code gap is characterized + recoverable (P2: clip among-finished 72.7 > A4B 69.2; deficit=30.5% trunc).
    # The config below (static loop + non-reentrant ckpt + cpu-offload + det-algos) is the furthest-progressing
    # attempt; it still fails at --smoke 2 with the IndexAddBackward routing-recompute mismatch.
    # SAC that MUST_SAVEs the ROUTING ops (topk/softmax/sort/argmax/sigmoid + matmuls) so the recompute REUSES
    # the saved router top-k instead of re-routing -> the expert dispatch (and IndexAddBackward) matches the
    # forward. (Plain non-reentrant ckpt re-routes -> IndexAddBackward shape mismatch.) Static expert loop keeps
    # the per-layer op COUNT constant; cpu-offload frees the base. This targets the routing-recompute wall.
    from torch.utils.checkpoint import checkpoint as _ckpt, create_selective_checkpoint_contexts as _sacctx, CheckpointPolicy as _CP
    _A = torch.ops.aten
    _SAVE = {_A.mm.default, _A.addmm.default, _A.bmm.default, _A.topk.default, _A._softmax.default,
             _A.sort.default, _A.argmax.default, _A.sigmoid.default, _A.one_hot.default}
    def _pol(ctx, func, *a, **k):
        return _CP.MUST_SAVE if func in _SAVE else _CP.PREFER_RECOMPUTE
    def _sacck(fn, *a, **k):
        k.pop("use_reentrant", None); k.pop("determinism_check", None)
        return _ckpt(fn, *a, use_reentrant=False, context_fn=lambda: _sacctx(_pol), **k)
    if args.checkpoint:
        log(f"[fsdp-opd] SAC ckpt (MUST_SAVE routing topk/softmax/matmuls) + static loop; cpu_offload={args.cpu_offload}")
        model.gradient_checkpointing_enable()
        model._gradient_checkpointing_func = _sacck
    else:
        log(f"[fsdp-opd] legacy whole-layer checkpointing OFF. ckpt_experts={args.ckpt_experts} "
            f"(checkpoint expert body -> recompute reconstructed weights on backward, deterministic via saved "
            f"routing -> frees ~64GB, no CheckpointError); cpu_offload={args.cpu_offload} "
            f"(0 = base GPU-resident, compute-bound)")
    torch.cuda.synchronize()
    log(f"[fsdp-opd] FSDP2-sharded across {world} GPUs ({time.time()-t0:.0f}s); "
        f"rank{rank} base-resident={torch.cuda.memory_allocated(local)/1e9:.1f}GB "
        f"reserved={torch.cuda.memory_reserved(local)/1e9:.1f}GB")

    # ---- load rollouts + teacher logprobs aligned by (pid,k); DP-shard across ranks ----
    npz = np.load(args.teacher_lp, allow_pickle=True)
    ro = {(r["pid"], r["k"]): r for r in (json.loads(l) for l in open(args.rollouts))}
    samples = []
    for entry in npz["records"]:
        key = (int(entry["pid"]), int(entry["k"]))
        if key not in ro: continue
        r = ro[key]
        ids = list(r["prompt_ids"]) + list(r["gen_token_ids"]); plen = len(r["prompt_ids"])
        if len(ids) > args.max_len or len(ids) <= plen + 1: continue
        samples.append((ids, plen, np.asarray(entry["t_idx"]), np.asarray(entry["t_lp"])))
    random.Random(0).shuffle(samples)
    mine = samples[rank::world]                   # disjoint DP slice
    log(f"[fsdp-opd] {len(samples)} total rollouts -> {len(mine)}/rank (DP x{world})")
    assert mine, "no samples on this rank"

    def opd_loss(ids, plen, t_idx_np, t_lp_np):
        x = torch.tensor([ids], device=dev)
        hidden = backbone(input_ids=x).last_hidden_state[0]        # [L,H] on local GPU (FSDP gathers units)
        H = hidden[plen:len(ids) - 1]
        t_idx = torch.tensor(t_idx_np, dtype=torch.long, device=dev)
        t_lp = torch.tensor(t_lp_np, dtype=torch.float32, device=dev)
        T = min(H.size(0), t_idx.size(0)); H = H[:T]; t_idx = t_idx[:T]; t_lp = t_lp[:T]
        k = t_idx.size(-1)
        p_t = F.softmax(t_lp, dim=-1)
        rest = (1.0 - p_t.sum(-1)).clamp_min(0.0)
        log_rest = torch.log((rest / (V - k)).clamp_min(1e-12))
        kl_sum = H.new_zeros(()).float()
        for c0 in range(0, T, VCHUNK):
            c = slice(c0, min(c0 + VCHUNK, T))
            z = lmhead(H[c]) / args.tau                            # [Tc,V]
            lse = torch.logsumexp(z.float(), dim=-1)
            s_lp = z.gather(-1, t_idx[c]).float() - lse[:, None]
            q = s_lp.exp(); q_rest = (1.0 - q.sum(-1)).clamp_min(1e-9)
            if args.reverse:
                kl = (q * (s_lp - torch.log(p_t[c].clamp_min(1e-12)))).sum(-1) + q_rest * (torch.log(q_rest) - log_rest[c])
            else:
                kl = (p_t[c] * (torch.log(p_t[c].clamp_min(1e-12)) - s_lp)).sum(-1) + rest[c] * (log_rest[c] - torch.log(q_rest))
            kl_sum = kl_sum + kl.sum(); del z
        return kl_sum / max(T, 1)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))  # plain (DTensor-safe)
    model.train()
    nsteps = args.smoke or max(1, args.epochs * len(mine) // args.accum)
    di = 0; run = 0.0

    def save_adapter():
        mods = [m for m in model.modules() if ML._is_experts(m) and hasattr(m, "_lora")]
        state = []
        for m in mods:                                            # .full_tensor() is collective -> all ranks call
            d = {kk: (getattr(m._lora, kk).full_tensor() if hasattr(getattr(m._lora, kk), "full_tensor")
                      else getattr(m._lora, kk)).detach().cpu() for kk in ("Agu", "Bgu", "Adn", "Bdn")}
            state.append(d)
        if is0:
            os.makedirs(args.adapter_out, exist_ok=True)
            torch.save(state, os.path.join(args.adapter_out, "adapter.pt"))
        return len(state)

    for step in range(nsteps):
        opt.zero_grad(set_to_none=True); acc = 0.0
        for _ in range(args.accum):
            ids, plen, ti, tl = mine[di % len(mine)]; di += 1
            loss = opd_loss(ids, plen, ti, tl)
            (loss / args.accum).backward()   # SAC (above) controls save-vs-recompute deterministically
            acc += loss.item() / args.accum
        # LoRA is FSDP-IGNORED (replicated), so FSDP does NOT reduce its grads. Each rank trained a disjoint
        # rollout slice -> manually all-reduce(AVG) the LoRA grads so every replica steps on the global grad
        # and stays in lockstep (the DP correctness FSDP would otherwise provide via reduce-scatter).
        # CRITICAL: every rank must all-reduce EVERY param in the SAME order. With the dynamic expert loop
        # different ranks route to different experts -> different LoRA params get a grad -> conditioning on
        # `p.grad is not None` would make the ranks issue a DIFFERENT set of collectives -> NCCL desync/hang
        # (DistBackendError). So materialize zero grads for the un-hit params and reduce ALL of them: a rank
        # that didn't touch expert e contributes 0, and AVG gives the correct global-batch mean.
        for p in trainable:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if step == 0:
            # per-rank local grad magnitude BEFORE the AVG above already mutated nothing destructive (AVG keeps
            # magnitude scale); confirms grad actually reached the (replicated) LoRA — the old failure was 0.
            gsum = torch.tensor([sum(p.grad.abs().sum().item() for p in trainable if p.grad is not None)], device=dev)
            mem = [torch.cuda.memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())]
            log(f"[fsdp-opd] step0 kl={acc:.4f} grad(LoRA,avg)={float(gsum):.3e} mem(GB)={['%.1f' % m for m in mem]}")
            assert float(gsum) > 0, "zero LoRA grad — STE/LoRA path severed under FSDP"
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        run = 0.9 * run + 0.1 * acc if step else acc
        if step % 5 == 0 or step == nsteps - 1:
            log(f"[fsdp-opd] step {step}/{nsteps} kl={acc:.4f} ema={run:.4f} ({time.time()-t0:.0f}s)")
        if args.ckpt_every and step and step % args.ckpt_every == 0:
            n = save_adapter(); log(f"[fsdp-opd] checkpoint adapter ({n} mods) @ step {step}")

    if args.smoke:
        log("[fsdp-opd] SMOKE OK"); dist.barrier(); dist.destroy_process_group(); return
    n = save_adapter()
    log(f"[fsdp-opd] DONE saved adapter ({n} mods) -> {args.adapter_out}/adapter.pt ({time.time()-t0:.0f}s)")
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
