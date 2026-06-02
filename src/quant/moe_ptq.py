#!/usr/bin/env python3
"""PTQ baseline methods applied PER-EXPERT to the fused MoE expert tensors (off-the-shelf libs target
nn.Linear and can't touch these). Clean apples-to-apples: same protection+allocation, only the quant
algorithm varies. Methods: rtn_asym (baseline), hqq (lib, half-quadratic), gptq (Hessian error-comp),
awq (activation-aware scaling + RTN). All asymmetric uint b-bit, per-block group along the INPUT axis.

A weight slice W is [out, in]; F.linear computes x@W.T (contract last axis=in). We group along `in`.
GPTQ/AWQ take per-expert calibration inputs Xe [n_tok, in].
"""
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------- asymmetric uint per-block (RTN)
def _qparams(w, nbits, group):
    """Per-block (group along last axis) asymmetric uint: returns scale, zero (per block)."""
    shp = w.shape; in_dim = shp[-1]
    g = group if (group and in_dim % group == 0) else in_dim
    wb = w.reshape(-1, g).float()
    qmax = (1 << nbits) - 1
    mn = wb.amin(-1, keepdim=True); mx = wb.amax(-1, keepdim=True)
    s = (mx - mn).clamp(min=1e-8) / qmax
    z = torch.round(-mn / s)
    return wb, s, z, qmax, shp, g

def rtn_asym(w, nbits, group):
    wb, s, z, qmax, shp, g = _qparams(w, nbits, group)
    q = torch.clamp(torch.round(wb / s) + z, 0, qmax)
    return ((q - z) * s).reshape(shp).to(w.dtype)

# ----------------------------------------------------------------- HQQ (authoritative lib)
def hqq_quant(w, nbits, group, axis=1):
    from hqq.core.quantize import Quantizer
    inp = w.detach().contiguous()                      # HQQ kernels need contiguous input
    wq, meta = Quantizer.quantize(inp, nbits=nbits, group_size=group, axis=axis,
                                  optimize=True, round_zero=(nbits == 4), bitpack=False,
                                  compute_dtype=torch.float32)
    dq = Quantizer.dequantize(wq, meta)
    return dq.reshape(w.shape).to(w.dtype)

# ----------------------------------------------------------------- AWQ (activation-aware scaling + RTN)
@torch.no_grad()
def awq_quant(w, Xe, nbits, group, grid=20):
    """Search a per-input-channel scale s (from activation magnitude^alpha) minimizing output MSE, then RTN.
    w:[out,in], Xe:[n,in]. Scales columns of w by s, divides activations by s (folded into next op approx
    via dividing w columns) — here we apply the standard AWQ: w' = w * s (per-in-channel), quant, w_q / s."""
    if Xe is None or Xe.shape[0] == 0:
        return rtn_asym(w, nbits, group)
    w = w.float(); Xe = Xe.float()
    x_scale = Xe.abs().mean(0).clamp(min=1e-6)            # per-in-channel activation magnitude [in]
    out_ref = Xe @ w.t()
    best = None; best_err = float("inf")
    for i in range(grid):
        alpha = i / (grid - 1)
        s = x_scale.pow(alpha); s = (s / s.mean()).clamp(min=1e-4)       # [in]
        wq = rtn_asym(w * s.view(1, -1), nbits, group) / s.view(1, -1)
        err = (Xe @ wq.t() - out_ref).pow(2).mean().item()
        if err < best_err:
            best_err = err; best = wq
    return best.to(torch.bfloat16) if best is not None else rtn_asym(w, nbits, group)

# ----------------------------------------------------------------- GPTQ (Hessian error compensation)
@torch.no_grad()
def gptq_quant(w, Xe, nbits, group, percdamp=0.01, blocksize=128):
    """GPTQ: greedy per-column quant with Hessian (H=XᵀX) error compensation. w:[out,in], Xe:[n,in]."""
    if Xe is None or Xe.shape[0] < 8:
        return rtn_asym(w, nbits, group)
    dev = w.device; W = w.float().clone(); out, cin = W.shape
    X = Xe.float()
    H = (X.t() @ X)                                              # [in,in]
    del X
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0
    damp = percdamp * torch.diag(H).mean()
    H[range(cin), range(cin)] += damp
    # Hinv via Cholesky of inverse (GPTQ trick)
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)          # upper-tri factor
    except Exception:
        return rtn_asym(w, nbits, group)
    qmax = (1 << nbits) - 1
    Q = torch.zeros_like(W)
    for i0 in range(0, cin, blocksize):
        i1 = min(i0 + blocksize, cin); count = i1 - i0
        W1 = W[:, i0:i1].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i0:i1, i0:i1]
        for j in range(count):
            col = i0 + j
            # per-(out,group) asym params for this column's group
            gstart = (col // group) * group; gend = min(gstart + group, cin)
            blk = W[:, gstart:gend]
            mn = blk.amin(1, keepdim=True); mx = blk.amax(1, keepdim=True)
            s = (mx - mn).clamp(min=1e-8) / qmax; z = torch.round(-mn / s)
            w_col = W1[:, j:j+1]
            q = torch.clamp(torch.round(w_col / s) + z, 0, qmax)
            qd = (q - z) * s
            Q1[:, j:j+1] = qd
            err = (w_col - qd) / Hinv1[j, j]
            W1[:, j:j+1] = w_col  # keep
            if j + 1 < count:
                W1[:, j+1:] -= err @ Hinv1[j:j+1, j+1:]
            Err1[:, j:j+1] = err
        Q[:, i0:i1] = Q1
        if i1 < cin:
            W[:, i1:] -= Err1 @ Hinv[i0:i1, i1:]
    return Q.reshape(w.shape).to(w.dtype)

METHODS = {"rtn_asym": rtn_asym, "hqq": hqq_quant, "awq": awq_quant, "gptq": gptq_quant}
NEEDS_CALIB = {"awq", "gptq"}

# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    W = torch.randn(256, 512)
    X = torch.randn(400, 512)
    yref = X @ W.t()
    print("method   3-bit_outMSE  2-bit_outMSE")
    for name, fn in METHODS.items():
        for nb in (3, 2):
            wq = fn(W, X, nb, 128) if name in NEEDS_CALIB else fn(W, nb, 128)
            mse = (X @ wq.float().t() - yref).pow(2).mean().item() / yref.pow(2).mean().item()
            print(f"  {name:9s} {nb}b nmse={mse:.4f}", end="  ")
        print()
