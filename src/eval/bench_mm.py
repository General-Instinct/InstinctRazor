#!/usr/bin/env python3
"""Faithful multimodal loaders for Gemma-4's reported vision benchmarks: MMMU, MMMU-Pro, MATH-Vision,
MedXpertQA-MM. Returns (question_text, [PIL images], options_or_None, gold). vLLM consumes images via the
chat template's image placeholders. Run `python bench_mm.py probe` to validate schemas WITHOUT a model."""
import sys, re

def _imgs_from(row):
    """collect PIL images from common columns: image, decoded_image, image_1..image_7, and images (list)."""
    out = []
    for k in ["image", "decoded_image"] + [f"image_{i}" for i in range(1, 8)]:
        v = row.get(k)
        if v is not None and hasattr(v, "size"):  # PIL image
            out.append(v)
    v = row.get("images")
    if isinstance(v, list):
        for im in v:
            if hasattr(im, "size"):
                out.append(im)
            elif isinstance(im, dict) and im.get("bytes"):  # HF Image stored as {bytes,path}
                import io; from PIL import Image
                try: out.append(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
                except Exception: pass
    return out

def build_mmmu(n, seed=0, pro=False):
    from datasets import load_dataset, get_dataset_config_names
    repo = "MMMU/MMMU_Pro" if pro else "MMMU/MMMU"
    if pro:
        ds = load_dataset(repo, "standard (4 options)", split="test")
    else:
        confs = get_dataset_config_names(repo)
        from datasets import concatenate_datasets
        parts = []
        for c in confs:
            try: parts.append(load_dataset(repo, c, split="validation"))
            except Exception: pass
        ds = concatenate_datasets(parts)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for r in ds:
        opts = r.get("options")
        if isinstance(opts, str):
            try: opts = eval(opts)
            except Exception: opts = None
        ans = r.get("answer")
        gold = "ABCDEFGHIJ".index(ans.strip().upper()) if (isinstance(ans, str) and ans.strip().upper() in "ABCDEFGHIJ" and opts) else ans
        items.append((r["question"], _imgs_from(r), opts, gold))
    return items

def build_mathvision(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("MathLLMs/MathVision", split="test").shuffle(seed=seed).select(range(min(n, 3040)))
    items = []
    for r in ds:
        opts = r.get("options") or None
        if isinstance(opts, list) and len(opts) == 0: opts = None
        items.append((r["question"], _imgs_from(r), opts, str(r["answer"])))
    return items

def build_medxpert_mm(n, seed=0):
    from datasets import load_dataset
    last = None
    for name, conf in [("TsinghuaC3I/MedXpertQA", "MM"), ("TsinghuaC3I/MedXpertQA", None)]:
        try:
            ds = load_dataset(name, conf, split="test") if conf else load_dataset(name, split="test")
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
            items = []
            for r in ds:
                q = r.get("question") or r.get("Question")
                opts = r.get("options") or r.get("choices")
                if isinstance(opts, dict): opts = [opts[k] for k in sorted(opts)]
                ans = r.get("label") or r.get("answer")
                gold = "ABCDEFGHIJ".index(ans.strip().upper()) if (isinstance(ans, str) and len(ans.strip())==1) else ans
                items.append((q, _imgs_from(r), opts, gold))
            print(f"[medxpert] using {name}/{conf}", flush=True)
            return items
        except Exception as e:
            last = f"{name}/{conf}: {str(e)[:80]}"
    raise RuntimeError(f"no MedXpertQA: {last}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        for nm, fn in [("MMMU", lambda: build_mmmu(3)), ("MMMU-Pro", lambda: build_mmmu(3, pro=True)),
                       ("MATH-Vision", lambda: build_mathvision(3)), ("MedXpertQA-MM", lambda: build_medxpert_mm(3))]:
            try:
                it = fn(); q, imgs, opts, gold = it[0]
                print(f"OK {nm}: n={len(it)} imgs={len(imgs)} opts={('list[%d]'%len(opts)) if opts else None} gold={gold} q={q[:70]!r}")
            except Exception as e:
                print(f"FAIL {nm}: {type(e).__name__}: {str(e)[:150]}")
