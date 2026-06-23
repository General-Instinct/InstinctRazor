#!/usr/bin/env python3
"""razor — InstinctRazor unified CLI.

One command takes a Hugging Face model -> quantizes it -> emits a deployable .gguf -> (optionally)
evaluates it on benchmarks. Wraps the framework's pieces (model_adapters, quant_save / moe_quant_method,
llama.cpp convert+quantize, the eval harnesses) behind one developer-facing entrypoint.

Examples
--------
  # HF model -> 3-bit GGUF (MoE-aware "InstinctRazor" protected recipe)
  ./razor --model Qwen/Qwen3.6-35B-A3B --quant instinct-iq3 --out runs/q36

  # HF model -> standard llama.cpp 4-bit GGUF, then eval on 2 benchmarks
  ./razor --model meta-llama/Llama-3.1-8B-Instruct --quant Q4_K_M --eval mmlu_pro,gpqa --budget 32k

  # Research path: our fake-quant capability-ceiling checkpoint (no GGUF), eval in vLLM
  ./razor --model Qwen/Qwen3.6-35B-A3B --recipe awq --expert-bits 3 --no-gguf --eval mmlu_pro,gpqa,math500

Stages (each can be skipped):
  1. resolve   — download the HF model (or use a local path)
  2. quantize  — EITHER --quant <gguf-type|instinct-*> (deployment GGUF) OR --recipe <clip|awq|gptq|rtn>
                 (our PTQ -> dequant-bf16 capability-ceiling checkpoint, for research/eval)
  3. gguf      — convert_hf_to_gguf + llama-quantize (with the protected tensor-type recipe for instinct-*)
  4. eval      — --eval <benchmarks>: GGUF -> llama.cpp server harness ; bf16 ckpt -> vLLM harness
"""
import argparse, os, shutil, subprocess, sys, time, signal, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_Q, SRC_E = HERE / "src/quant", HERE / "src/eval"
for p in (SRC_Q, SRC_E, HERE / "src/distill"):
    sys.path.insert(0, str(p))

# llama.cpp quantize types we pass straight through; "instinct-*" map to protected tensor-type recipes.
GGUF_TYPES = {"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K",
              "Q8_0", "IQ2_XXS", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL", "BF16", "F16"}
INSTINCT_RECIPES = {  # name -> (tensor-type file, base llama.cpp type, needs_imatrix)
    "instinct-q3":  ("configs/gguf_tensor_types.txt",     "Q3_K",    False),
    "instinct-iq3": ("configs/gguf_tensor_types_iq3.txt", "IQ3_XXS", True),
}
PTQ_RECIPES = {"clip", "awq", "gptq", "rtn"}


def log(msg): print(f"\033[1m[razor]\033[0m {msg}", flush=True)
def die(msg, code=2):
    print(f"\033[31m[razor] ERROR:\033[0m {msg}", file=sys.stderr); sys.exit(code)


def run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def find_llama_cpp(arg):
    for cand in (arg, os.environ.get("LLAMA_CPP"), HERE / "llama.cpp", "./llama.cpp"):
        if cand and Path(cand).expanduser().exists():
            d = Path(cand).expanduser().resolve()
            if (d / "convert_hf_to_gguf.py").exists():
                return d
    return None


def resolve_model(model):
    if Path(model).exists():
        return str(Path(model).resolve())
    log(f"resolving HF model {model} (snapshot_download) ...")
    from huggingface_hub import snapshot_download
    return snapshot_download(model)


def stage_gguf(args, model_dir, py):
    """convert -> quantize -> path to the .gguf."""
    lcpp = find_llama_cpp(args.llama_cpp)
    if not lcpp:
        die("llama.cpp not found. Build it (CPU is enough to quantize):\n"
            "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp\n"
            "  cmake -B build -DGGML_CUDA=OFF && cmake --build build -j --target llama-quantize\n"
            "then set LLAMA_CPP=/path/to/llama.cpp (must support the model's arch).")
    out = Path(args.out); gguf_dir = out / "gguf"; gguf_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.model_id).name                  # clean name (not the HF snapshot hash)
    bf16 = gguf_dir / f"{name}-bf16.gguf"
    env = {**os.environ, "PYTHONPATH": f"{lcpp/'gguf-py'}:{os.environ.get('PYTHONPATH','')}"}
    # --no-mtp is only valid for Qwen3.5/3.6 (qwen3_5_moe); auto-detect from config.
    mt = ""
    cfgp = Path(model_dir) / "config.json"
    if cfgp.exists():
        try: mt = json.loads(cfgp.read_text()).get("model_type", "")
        except Exception: pass
    use_no_mtp = args.no_mtp or mt == "qwen3_5_moe"

    final = gguf_dir / f"{name}-{args.quant}.gguf"
    if final.exists() and not args.dry_run:
        log(f"GGUF already exists, skipping convert+quantize: {final}")
        return str(final)

    # 1. convert HF -> bf16 GGUF
    if not bf16.exists():
        cmd = [py, lcpp / "convert_hf_to_gguf.py", model_dir, "--outfile", bf16, "--outtype", "bf16"]
        if use_no_mtp: cmd.append("--no-mtp")
        if args.dry_run: log("DRY: " + " ".join(map(str, cmd)))
        else: run(cmd, env=env)
    else:
        log(f"bf16 gguf exists: {bf16}")

    # 2. quantize
    q = args.quant
    qbin = lcpp / "build/bin/llama-quantize"
    final = gguf_dir / f"{name}-{q}.gguf"
    cmd = [qbin]
    if q in INSTINCT_RECIPES:
        ttf, base, needs_im = INSTINCT_RECIPES[q]
        clean = gguf_dir / ".tt_clean.txt"
        if not args.dry_run:
            with open(HERE / ttf) as f, open(clean, "w") as o:
                for line in f:
                    if line.strip() and not line.lstrip().startswith("#"): o.write(line)
        cmd += ["--tensor-type-file", clean, "--output-tensor-type", "q8_0", "--token-embedding-type", "q8_0"]
        if args.imatrix: cmd += ["--imatrix", args.imatrix]
        elif needs_im: log("WARN: instinct-iq3 benefits from an imatrix; pass --imatrix <file> for best quality.")
        cmd += [bf16, final, base]
    else:
        if q not in GGUF_TYPES: die(f"unknown --quant '{q}'. GGUF types: {sorted(GGUF_TYPES)}; or {sorted(INSTINCT_RECIPES)}")
        if args.imatrix: cmd += ["--imatrix", args.imatrix]
        cmd += [bf16, final, q]
    if args.dry_run: log("DRY: " + " ".join(map(str, cmd)))
    else: run(cmd)
    if not args.keep_bf16 and not args.dry_run and bf16.exists() and final.exists():
        bf16.unlink(); log(f"removed intermediate {bf16.name} (use --keep-bf16 to retain)")
    log(f"GGUF ready: {final}")
    return str(final)


def stage_recipe(args, model_dir, py):
    """Our PTQ -> dequant-bf16 capability-ceiling checkpoint (research/eval path)."""
    out = Path(args.out); ckpt = out / f"ckpt-{args.recipe}{args.expert_bits:g}b"
    if args.recipe == "clip":
        cmd = [py, SRC_Q / "quant_save.py", "--model", model_dir, "--expert-bits", args.expert_bits,
               "--linear-bits", args.linear_bits, "--group", args.group, "--clip-steps", args.clip_steps,
               "--max-mem-gib", args.max_mem_gib, "--out", ckpt]
    else:
        cmd = [py, SRC_Q / "moe_quant_method.py", "--model", model_dir, "--method", args.recipe,
               "--bits", int(args.expert_bits), "--linear-bits", args.linear_bits, "--group", args.group,
               "--max-mem-gib", args.max_mem_gib, "--out", ckpt]
    if args.dry_run: log("DRY: " + " ".join(map(str, cmd))); return str(ckpt)
    run(cmd); log(f"checkpoint ready: {ckpt}"); return str(ckpt)


def stage_eval_gguf(args, gguf, py):
    lcpp = find_llama_cpp(args.llama_cpp)
    ctx = 66560 if args.budget == "64k" else 40960
    budget = args.eval_budget or (65536 if args.budget == "64k" else 32768)
    nflags = ["--mmlu-n", args.eval_n, "--gpqa-n", args.eval_n] if args.eval_n else []
    server = lcpp / "build/bin/llama-server"
    cmd = [server, "-m", gguf, "--port", args.port, "-c", ctx, "--parallel", args.parallel,
           "-ngl", args.ngl, "--host", "127.0.0.1"]
    if args.dry_run:
        log("DRY: launch " + " ".join(map(str, cmd)))
        log(f"DRY: {py} {SRC_E/'llamacpp_eval8.py'} --tokenizer {args.model} --tag {args.tag} "
            f"--benchmarks {args.eval} --port {args.port} --budget {budget}")
        return
    log("launching llama-server ...")
    srv = subprocess.Popen([str(c) for c in cmd], stdout=open(Path(args.out)/"llama_server.log", "w"), stderr=subprocess.STDOUT)
    try:
        import urllib.request
        for _ in range(180):  # wait up to ~6 min for model load
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2); break
            except Exception: time.sleep(2)
        ecmd = [py, SRC_E / "llamacpp_eval8.py", "--tokenizer", args.model, "--tag", args.tag,
                "--benchmarks", args.eval, "--port", args.port, "--budget", budget, *nflags]
        run(ecmd, cwd=HERE)
    finally:
        srv.send_signal(signal.SIGINT); time.sleep(3); srv.kill()
    log(f"eval done -> results/vllm_eval/{args.tag}.json")


def stage_eval_ckpt(args, ckpt, py):
    budget = "64k" if args.budget == "64k" else "32k"
    cmd = ["bash", HERE / "pipelines/eval.sh", "--model", ckpt, "--tag", args.tag,
           "--benchmarks", args.eval, "--budget", budget]
    if args.dry_run: log("DRY: " + " ".join(map(str, cmd))); return
    run(cmd, cwd=HERE); log(f"eval done -> results/vllm_eval/{args.tag}.json")


def main():
    ap = argparse.ArgumentParser(prog="razor", description="InstinctRazor: HF model -> quantize -> .gguf -> eval")
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--quant", help=f"GGUF type ({sorted(GGUF_TYPES)}) or InstinctRazor recipe ({sorted(INSTINCT_RECIPES)})")
    ap.add_argument("--recipe", choices=sorted(PTQ_RECIPES), help="research PTQ -> dequant-bf16 capability-ceiling ckpt")
    ap.add_argument("--no-gguf", action="store_true", help="skip GGUF (with --recipe, just produce+eval the bf16 ckpt)")
    ap.add_argument("--out", default=None, help="output dir (default: runs/<model_name>)")
    ap.add_argument("--eval", default=None, help="comma benchmarks, e.g. mmlu_pro,gpqa,math500 (omit = no eval)")
    ap.add_argument("--budget", default="32k", choices=["32k", "64k"])
    ap.add_argument("--tag", default=None, help="eval results tag (default: derived)")
    # quant/recipe knobs
    ap.add_argument("--expert-bits", type=float, default=3.0); ap.add_argument("--linear-bits", type=float, default=4.0)
    ap.add_argument("--group", type=int, default=128); ap.add_argument("--clip-steps", type=int, default=16)
    ap.add_argument("--max-mem-gib", type=int, default=76)
    # gguf/eval infra
    ap.add_argument("--llama-cpp", default=None, help="path to llama.cpp (default: $LLAMA_CPP or ./llama.cpp)")
    ap.add_argument("--imatrix", default=None, help="imatrix file for i-quants (IQ3_*)")
    ap.add_argument("--no-mtp", action="store_true", help="pass --no-mtp to convert (Qwen3.5/3.6)")
    ap.add_argument("--keep-bf16", action="store_true")
    ap.add_argument("--eval-n", type=int, default=None, help="cap samples per benchmark (quick eval)")
    ap.add_argument("--eval-budget", type=int, default=None, help="override n_predict (else 32k/64k preset)")
    ap.add_argument("--port", type=int, default=8099); ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--ngl", type=int, default=999, help="GPU layers for llama-server eval (0 = CPU)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    if not args.quant and not args.recipe:
        die("specify --quant <gguf-type|instinct-*> (deployment GGUF) and/or --recipe <clip|awq|gptq|rtn> (research ckpt).")
    args.out = args.out or str(HERE / "runs" / Path(args.model).name)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    py = sys.executable
    args.model_id = args.model

    log(f"model={args.model}  quant={args.quant}  recipe={args.recipe}  eval={args.eval or 'no'}  out={args.out}")
    model_dir = args.model if args.dry_run else resolve_model(args.model)

    target = None        # what eval runs on
    if args.recipe:                                  # research bf16-ceiling path
        ckpt = stage_recipe(args, model_dir, py)
        target = ("ckpt", ckpt)
    if args.quant and not args.no_gguf:              # deployment GGUF path
        gguf = stage_gguf(args, model_dir, py)
        target = ("gguf", gguf)

    if args.eval:
        args.tag = args.tag or (Path(args.out).name + "_" + (args.quant or args.recipe)).replace(".", "")
        kind, path = target
        log(f"=== EVAL ({kind}) benchmarks={args.eval} budget={args.budget} ===")
        (stage_eval_gguf if kind == "gguf" else stage_eval_ckpt)(args, path, py)

    log("DONE.")


if __name__ == "__main__":
    main()
