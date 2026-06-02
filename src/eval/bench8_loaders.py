#!/usr/bin/env python3
"""Loaders + scorers for the 8-benchmark comprehensive comparison (our 47GB model vs Gemma-4-31B-it).
New benchmarks (MMMLU, HLE, AIME-2025, MATH-500, MBPP) beyond the existing MMLU-Pro/GPQA-D/HumanEval.
Defensive: probes dataset schemas, tolerant of column-name variants. Run `python bench8_loaders.py probe`
to validate dataset access + schemas WITHOUT loading any model."""
import re, sys

# ---------------- math answer extraction / comparison ----------------
def extract_boxed(s):
    # last \boxed{...} with balanced braces
    i = s.rfind("\\boxed")
    if i < 0:
        # fallback: 'answer is X' / final number
        m = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*\$?\\?\(?([-\d.,/]+)\)?\$?", s, re.I)
        return m[-1].strip(" .") if m else None
    j = s.find("{", i)
    if j < 0: return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{": depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j+1:k].strip()
    return None

def norm_math(x):
    if x is None: return None
    x = str(x).strip()
    x = x.replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace(" ", "")
    x = x.replace("\\left", "").replace("\\right", "").replace("$", "")
    x = x.replace("\\text{", "").replace("\\mathrm{", "").rstrip("}")
    x = x.replace("dollars", "").replace("\\%", "").replace("%", "")
    x = x.replace(",", "")  # thousands
    if x.startswith("\\frac"): pass
    return x.strip(".")

def math_eq(pred, gold):
    if pred is None: return False
    p, g = norm_math(pred), norm_math(gold)
    if p is None or g is None: return False
    if p == g: return True
    try:
        return abs(float(p) - float(g)) < 1e-3
    except ValueError:
        return False

# ---------------- dataset builders ----------------
def build_math500(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.select(range(min(n, len(ds))))
    return [(r["problem"], str(r["answer"])) for r in ds]   # (question, gold_answer)

def build_aime(n):
    from datasets import load_dataset
    last = None
    for name, conf, split in [("yentinglin/aime_2025", None, "train"),
                              ("opencompass/AIME2025", None, "test"),
                              ("Maxwell-Jia/AIME_2024", None, "train"),
                              ("HuggingFaceH4/aime_2024", None, "train")]:
        try:
            ds = load_dataset(name, conf, split=split) if conf else load_dataset(name, split=split)
            cols = ds.column_names
            qk = "problem" if "problem" in cols else ("question" if "question" in cols else cols[0])
            ak = "answer" if "answer" in cols else ("solution" if "solution" in cols else cols[-1])
            out = [(r[qk], str(r[ak])) for r in ds.select(range(min(n, len(ds))))]
            print(f"[aime] using {name} ({len(ds)} items, q={qk} a={ak})", flush=True)
            return out
        except Exception as e:
            last = f"{name}: {str(e)[:80]}"
    raise RuntimeError(f"no AIME dataset loaded: {last}")

def build_mmmlu(n, seed=0):
    from datasets import load_dataset
    # openai/MMMLU: columns Question, A, B, C, D, Answer (letter), Subject, Locale
    ds = load_dataset("openai/MMMLU", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for r in ds:
        opts = [r["A"], r["B"], r["C"], r["D"]]
        ans = "ABCD".index(r["Answer"].strip().upper())
        items.append((r["Question"], opts, ans))
    return items

def build_hle(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("cais/hle", split="test")
    # text-only subset (no image)
    def has_img(r):
        v = r.get("image", "")
        return bool(v) and v not in ("", None)
    ds = ds.filter(lambda r: not has_img(r))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for r in ds:
        items.append((r["question"], str(r["answer"]), r.get("answer_type", "exactMatch")))
    return items

def build_mbpp(n):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    ds = ds.select(range(min(n, len(ds))))
    out = []
    for r in ds:
        out.append((r["prompt"], r["code"], r["test_list"], r.get("test_imports", []) or []))
    return out

def build_bbeh(n, seed=0):
    """BigBench-Extra-Hard (Gemma reports it). Free-form exact-match after normalization."""
    from datasets import load_dataset
    last = None
    for name in ["google/bbeh", "BBEH/bbeh", "lukaemon/bbh"]:
        try:
            ds = load_dataset(name, split="train") if name != "lukaemon/bbh" else None
            if ds is None:
                # lukaemon/bbh needs a config (task); use a representative subset via 'all'? fallback skip
                raise RuntimeError("bbh needs per-task config")
            cols = ds.column_names
            qk = "input" if "input" in cols else ("question" if "question" in cols else cols[0])
            ak = "target" if "target" in cols else ("answer" if "answer" in cols else cols[-1])
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
            print(f"[bbeh] using {name} (q={qk} a={ak})", flush=True)
            return [(r[qk], str(r[ak])) for r in ds]
        except Exception as e:
            last = f"{name}: {str(e)[:80]}"
    raise RuntimeError(f"no BBEH dataset: {last}")

def mbpp_run(code, test_list, test_imports, timeout=8):
    import subprocess, tempfile, os
    src = "\n".join(test_imports) + "\n" + code + "\n" + "\n".join(test_list) + "\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src); path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        os.unlink(path)
        return r.returncode == 0
    except Exception:
        return False

# ---------------- LiveCodeBench v6 (release_v6 = test.jsonl..test6.jsonl cumulative, 1055 problems) ----------------
_LCB_CACHE = None
def build_lcb(n, seed=0):
    """LCB v6. Returns (prompt, record) with record={'type':stdin|functional,'fn':func_name,'tests':[(in,out)]}.
    Same seed-0 sample across models. (release_v6 spans 2023-05..2025-04; we sample the full set for breadth —
    relative same-harness deltas are the trustworthy quantity, not the vendor-exact absolute.)"""
    global _LCB_CACHE
    import json, base64, zlib, random, pickle
    from huggingface_hub import hf_hub_download
    def _decode_priv(s):  # LCB private_test_cases = base64(zlib(pickle(json_str))) OR base64(zlib(json_str))
        raw = zlib.decompress(base64.b64decode(s))
        try: obj = json.loads(raw)
        except Exception: obj = pickle.loads(raw)
        return json.loads(obj) if isinstance(obj, str) else obj
    if _LCB_CACHE is None:
        rows = []
        for f in ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]:
            p = hf_hub_download("livecodebench/code_generation_lite", f, repo_type="dataset")
            for line in open(p):
                line = line.strip()
                if line: rows.append(json.loads(line))
        _LCB_CACHE = rows
    rows = list(_LCB_CACHE); random.Random(seed).shuffle(rows); rows = rows[:n]
    out = []
    for r in rows:
        try: pub = json.loads(r["public_test_cases"])
        except Exception: pub = []
        try: priv = _decode_priv(r["private_test_cases"])
        except Exception: priv = []
        tests = [(t["input"], t["output"]) for t in (pub + priv)]
        ttype = pub[0]["testtype"] if pub else "stdin"
        try: fn = json.loads(r.get("metadata") or "{}").get("func_name")
        except Exception: fn = None
        starter = r.get("starter_code", "") or ""
        if ttype == "functional" and starter.strip():
            prompt = (r["question_content"] + "\n\nComplete the solution using EXACTLY this class/method signature:\n```python\n"
                      + starter + "\n```\nReason if needed, then give the COMPLETE solution in a ```python code block.")
        else:
            prompt = (r["question_content"] + "\n\nRead from stdin, write to stdout. Reason if needed, then give the "
                      "COMPLETE program in a ```python code block.")
        out.append((prompt, {"type": ttype, "fn": fn, "tests": tests}))
    return out

# Standard LCB exec preamble: solutions reproduce the starter signature (e.g. `details: List[str]`) and use
# collections/math/etc WITHOUT importing them (the official harness provides these). Missing them -> NameError
# -> false failure. Also raise recursion limit (DFS solutions hit the default 1000). This was THE cause of the
# uniform ~6pt under-shoot vs official (teacher 72.6 vs 78.9).
_LCB_PREAMBLE = (
    "import sys, math, collections, heapq, bisect, itertools, functools, operator, re, string, random, copy\n"
    "from typing import *\nfrom collections import *\nfrom math import *\nfrom functools import *\n"
    "from itertools import *\nfrom heapq import *\nfrom bisect import *\n"
    "sys.setrecursionlimit(1000000)\n")

def lcb_run(code, rec, timeout=10, max_tests=300):
    """pass@1: run model code against LCB tests. functional -> all tests in ONE driver subprocess with a
    PER-TEST SIGALRM timeout (a slow-but-correct solution is no longer aggregate-TLE'd), call
    Solution().fn(*args), TOLERANT compare (recursive list<->tuple coercion + float abs-tol 1e-6, matching
    official LCB semantics); stdin -> per-test subprocess (feed stdin, compare stdout). Args parsed as a single
    whole-input JSON first, then per-line (handles multi-line single-arg inputs). Prepends _LCB_PREAMBLE.
    max_tests=300 covers the full LCB v6 test set (max observed 103)."""
    import subprocess, sys as _sys, json
    tests = rec.get("tests", [])[:max_tests]
    if not tests: return False
    if rec.get("type") == "functional" and rec.get("fn"):
        driver = (_LCB_PREAMBLE + code + "\nimport json as _J, signal as _sig\n_T=" + json.dumps(tests) + f"""
def _eq(a,b):
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(_eq(x,y) for x,y in zip(a,b))
    if isinstance(a,bool) or isinstance(b,bool): return a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        try: return abs(float(a)-float(b))<=1e-6
        except Exception: return a==b
    return a==b
def _args(_inp):
    try: return [_J.loads(_inp)]
    except Exception: pass
    try: return [_J.loads(_l) for _l in _inp.split(chr(10)) if _l.strip()!='']
    except Exception: return [_inp]
def _to(*_a): raise TimeoutError()
_sig.signal(_sig.SIGALRM, _to)
_ok=True
for _inp,_exp in _T:
    try:
        _sig.alarm({int(timeout)})
        _r=Solution().{rec['fn']}(*_args(_inp))
        _sig.alarm(0)
        if not _eq(_r, _J.loads(_exp)): _ok=False; break
    except Exception:
        _sig.alarm(0); _ok=False; break
print('PASS' if _ok else 'FAIL')
""")
        try:
            outer = min(int(timeout) * len(tests) + 30, 600)
            p = subprocess.run([_sys.executable, "-c", driver], capture_output=True, timeout=outer, text=True)
            return p.stdout.strip().endswith("PASS")
        except Exception:
            return False
    else:
        for inp, exp in tests:
            try:
                p = subprocess.run([_sys.executable, "-c", _LCB_PREAMBLE + code], input=inp, capture_output=True, timeout=timeout, text=True)
                if p.returncode != 0: return False
                if "\n".join(l.rstrip() for l in p.stdout.rstrip().split("\n")) != \
                   "\n".join(l.rstrip() for l in exp.rstrip().split("\n")): return False
            except Exception:
                return False
        return True

# ---------------- MathArena competition math (AIME 2026, HMMT Feb 2025) ----------------
def build_matharena(dataset_id, n=30, seed=0):
    """MathArena set (AIME/HMMT): cols {problem, answer(int)}. Returns [(problem, str(answer))]. Full set is 30."""
    from datasets import load_dataset
    try: ds = load_dataset(dataset_id, split="train")
    except Exception: ds = load_dataset(dataset_id, split="test")
    rows = list(ds)
    if n < len(rows):
        import random
        random.Random(seed).shuffle(rows); rows = rows[:n]
    return [(r["problem"], str(r["answer"])) for r in rows]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        for name, fn in [("MATH-500", lambda: build_math500(3)), ("AIME", lambda: build_aime(3)),
                         ("MMMLU", lambda: build_mmmlu(3)), ("HLE", lambda: build_hle(3)),
                         ("MBPP", lambda: build_mbpp(3))]:
            try:
                items = fn()
                print(f"OK {name}: n_probe={len(items)} sample={str(items[0])[:160]}")
            except Exception as e:
                print(f"FAIL {name}: {type(e).__name__}: {str(e)[:160]}")
