#!/usr/bin/env python3
"""Model-agnostic eval functions for the MoE quant study + CPU-snapshot restore for multi-spec runs.

Screening (cheap, no generation): calib NLL (perplexity) + top-K KL vs a cached BF16 teacher.
Capability axes (the real signal): MMLU (knowledge, MC loglik), GSM8K (math, 4-shot CoT gen),
HumanEval (code, gen+exec), GPQA (hard reasoning, MC loglik; skipped if dataset gated).
"""
import json, re, subprocess, tempfile, os, sys
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------- snapshot / restore (multi-spec)
@torch.no_grad()
def snapshot_fp(model):
    """Save a CPU FP copy of all quantizable weights (experts + non-expert linears/embeds)."""
    from moe_quant import iter_expert_tensors, iter_quant_linears
    snap = {}
    for n, p in iter_expert_tensors(model):
        snap[("E", n)] = p.detach().to("cpu", copy=True)
    for n, m in iter_quant_linears(model):
        snap[("L", n)] = m.weight.detach().to("cpu", copy=True)
    return snap

@torch.no_grad()
def restore_fp(model, snap):
    from moe_quant import iter_expert_tensors, iter_quant_linears
    for n, p in iter_expert_tensors(model):
        if ("E", n) in snap:
            p.data.copy_(snap[("E", n)].to(p.device))
    for n, m in iter_quant_linears(model):
        if ("L", n) in snap:
            m.weight.data.copy_(snap[("L", n)].to(m.weight.device))

# ----------------------------------------------------------------- calibration KL / NLL
@torch.no_grad()
def cache_teacher(model, seqs, topk=512):
    """Run the (BF16) model and cache, per position, the true-token NLL + top-K logprobs/indices."""
    dev = next(model.parameters()).device
    cache = []
    nll_sum, ntok = 0.0, 0
    for ids in seqs:
        ids = ids.to(dev)
        logits = model(input_ids=ids, use_cache=False).logits[0].float()  # (T, V)
        lp = F.log_softmax(logits, dim=-1)
        tgt = ids[0, 1:]
        tok_nll = -lp[:-1].gather(1, tgt[:, None]).squeeze(1)              # (T-1,)
        nll_sum += tok_nll.sum().item(); ntok += tgt.numel()
        tk = lp[:-1].topk(topk, dim=-1)
        cache.append({"idx": tk.indices.to("cpu"), "lp": tk.values.to("cpu"), "tgt": tgt.to("cpu")})
    return {"per_seq": cache, "teacher_nll": nll_sum / ntok, "ntok": ntok, "topk": topk}

@torch.no_grad()
def eval_calib(model, seqs, teacher):
    """Student NLL (perplexity) + approx top-K KL(teacher||student) vs cached teacher."""
    dev = next(model.parameters()).device
    nll_sum, kl_sum, ntok = 0.0, 0.0, 0
    for ids, tc in zip(seqs, teacher["per_seq"]):
        ids = ids.to(dev)
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
        lp = F.log_softmax(logits, dim=-1)[:-1]                           # (T-1, V)
        tgt = tc["tgt"].to(dev)
        nll_sum += (-lp.gather(1, tgt[:, None]).squeeze(1)).sum().item(); ntok += tgt.numel()
        t_idx = tc["idx"].to(dev); t_lp = tc["lp"].to(dev)                # (T-1, K)
        s_lp = lp.gather(1, t_idx)                                        # student logprob at teacher topK
        pt = t_lp.exp()
        kl_sum += (pt * (t_lp - s_lp)).sum().item()                       # sum over K, sum over positions
    return {"student_nll": nll_sum / ntok, "ppl": float(torch.tensor(nll_sum / ntok).exp()),
            "topk_kl": kl_sum / ntok}

# ----------------------------------------------------------------- MMLU (MC loglik, 0-shot)
def build_mmlu(n=200, seed=0):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    import random; random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        r = ds[i]
        items.append((r["question"], r["choices"], r["answer"]))
    return items

@torch.no_grad()
def eval_mmlu(model, tok, items, batch=8):
    dev = next(model.parameters()).device
    letters = ["A", "B", "C", "D"]
    lids = [tok(f" {L}", add_special_tokens=False)["input_ids"][-1] for L in letters]
    correct = 0
    prompts = []
    for q, ch, _ in items:
        s = q + "\n" + "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(ch)) + "\nAnswer:"
        prompts.append(s)
    tok.padding_side = "left"
    for b in range(0, len(items), batch):
        ps = prompts[b:b+batch]
        enc = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()   # (B,V)
        pick = logits[:, lids].argmax(dim=-1).tolist()
        for j, (_, _, ans) in enumerate(items[b:b+batch]):
            if pick[j] == ans:
                correct += 1
    return {"mmlu_acc": 100.0 * correct / len(items), "mmlu_n": len(items)}

# ----------------------------------------------------------------- GPQA (MC loglik, 0-shot)
def build_gpqa(n=198, seed=0):
    """fingertap/GPQA-Diamond (ungated): question already contains options a)-d); answer is a letter."""
    from datasets import load_dataset
    try:
        ds = load_dataset("fingertap/GPQA-Diamond", split="test")
        items = []
        for r in ds:
            ans = r["answer"].strip().upper()
            if ans in "ABCD":
                items.append((r["question"], "ABCD".index(ans)))
        return items[:n], "fingertap/diamond"
    except Exception as e:
        print(f"[gpqa] unavailable: {e}", flush=True)
        return None, None

@torch.no_grad()
def eval_gpqa(model, tok, items, batch=8):
    """MC loglik on GPQA-Diamond — options are embedded in the question; compare A/B/C/D letter logits."""
    dev = next(model.parameters()).device
    lids = [tok(f" {L}", add_special_tokens=False)["input_ids"][-1] for L in ["A", "B", "C", "D"]]
    tok.padding_side = "left"
    correct = 0
    for b in range(0, len(items), batch):
        chunk = items[b:b+batch]
        ps = [q + "\n\nAnswer with the letter of the correct option (A, B, C, or D).\nAnswer:" for q, _ in chunk]
        enc = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(dev)
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        pick = logits[:, lids].argmax(dim=-1).tolist()
        for j, (_, ans) in enumerate(chunk):
            if pick[j] == ans:
                correct += 1
    return {"gpqa_acc": 100.0 * correct / len(items), "gpqa_n": len(items)}

# ----------------------------------------------------------------- GSM8K (4-shot CoT gen)
GSM_FEWSHOT = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?",
     "In April she sold 48 clips. In May she sold half as many, so 48/2 = 24 clips. Altogether 48 + 24 = 72. The answer is 72."),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "Per minute she earns 12/60 = $0.2. For 50 minutes she earned 50 * 0.2 = $10. The answer is 10."),
    ("Betty is saving money for a new wallet which costs $100. She has half of the money she needs. Her parents give her $15 and her grandparents twice as much as her parents. How much more money does Betty need?",
     "Betty has 100/2 = $50. Grandparents give 2*15 = $30. Total now 50 + 15 + 30 = $95. She needs 100 - 95 = $5 more. The answer is 5."),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "Each time he writes 3*2 = 6 pages. Twice a week that's 6*2 = 12 pages/week. In a year 12*52 = 624 pages. The answer is 624."),
]

def _last_num(s):
    s = s.split("The answer is")[-1] if "The answer is" in s else s
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", s)
    if not nums: return None
    return nums[0].replace("$", "").replace(",", "").rstrip(".")

def build_gsm8k(n=80, seed=0):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = [(ds[i]["question"], ds[i]["answer"].split("####")[-1].strip().replace(",", "")) for i in range(min(n, len(ds)))]
    return items

@torch.no_grad()
def eval_gsm8k(model, tok, items, batch=8, gen_tok=256):
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    fs = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in GSM_FEWSHOT)
    correct = 0
    for b in range(0, len(items), batch):
        chunk = items[b:b+batch]
        ps = [fs + f"Question: {q}\nAnswer:" for q, _ in chunk]
        enc = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=1280).to(dev)
        out = model.generate(**enc, max_new_tokens=gen_tok, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        comp = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (q, gold), c in zip(chunk, comp):
            c = c.split("Question:")[0]
            pred = _last_num(c)
            try:
                if pred is not None and abs(float(pred) - float(gold)) < 1e-3:
                    correct += 1
            except ValueError:
                pass
    return {"gsm8k_acc": 100.0 * correct / len(items), "gsm_n": len(items)}

# ----------------------------------------------------------------- HumanEval (gen + exec)
def _he_extract(prompt, completion):
    body = completion
    # strip markdown code fences (instruct models wrap code in ```python ... ```)
    if "```" in body:
        seg = body.split("```")
        # take the largest fenced block; drop a leading 'python' language tag
        cand = max((s for s in seg[1::2]), key=len, default=body)
        cand = re.sub(r"^[ \t]*python[ \t]*\n", "", cand)
        # if the fenced block redefines the function, use it as the full body sans prompt
        body = cand
        if body.lstrip().startswith(("def ", "from ", "import ")):
            return body
    for stop in ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@", "\n```"]:
        i = body.find(stop)
        if i != -1: body = body[:i]
    return prompt + body

def _he_run(full_code, test_code, entry_point, timeout=10):
    src = full_code + "\n" + test_code + f"\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        os.unlink(path)

def build_humaneval(n=40):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [(ds[i]["prompt"], ds[i]["test"], ds[i]["entry_point"]) for i in range(min(n, len(ds)))]

@torch.no_grad()
def eval_humaneval(model, tok, items, batch=8, gen_tok=320):
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    passed = 0
    for b in range(0, len(items), batch):
        chunk = items[b:b+batch]
        ps = [p for p, _, _ in chunk]
        enc = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
        out = model.generate(**enc, max_new_tokens=gen_tok, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        comps = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (p, test, ep), comp in zip(chunk, comps):
            if _he_run(_he_extract(p, comp), test, ep):
                passed += 1
    return {"humaneval_pass@1": 100.0 * passed / len(items), "he_n": len(items)}
