# vsbound.py: packed verify vs the stock per-position path when a verify step's rows straddle a KV page boundary
# (page size 960, as served). For each sequence length L (the 6 or 8 verify rows sit at positions L-q_len .. L-1) around
# 1*960, 2*960, 4*960 and 120*960, B=1 (32 splits, the runtime rule) and B=4 (8 splits), random block tables.
# Prints max |packed - stock| per case; FAIL when it is > 50x the median of the non-straddling cases.
import math, os, sys, torch
sys.path.insert(0, os.environ.get("B70_RUNTIME", "."))
os.environ["B70_SPEC_PACKED"] = "1"; os.environ["B70_SPEC_PACKED_QLENS"] = "any"
import spec_packed  # noqa
D, HQ, HKV, BS, MAXLEN = 256, 24, 4, 960, 262144
SCALE = 1 / math.sqrt(D); dev = "xpu"
ks = torch.full((1,), 0.0371, device=dev); vs = torch.full((1,), 0.0213, device=dev)
mode = os.environ.get("VB_SPLITS", "rule")   # rule = 4*cores/(B*Hkv) as in the runtime; or a number
res = []
for B in (1, 4):
    sp = (32 if B == 1 else 8) if mode == "rule" else int(mode)
    spec_packed._graph_splits = lambda *a, _s=sp, **k: _s
    for base in (960, 1920, 3840, 115200):
        for L in range(base - 3, base + 10):
            torch.manual_seed(L * 7 + B)
            nb = (L + BS - 1) // BS
            k = (torch.randn((B * nb + 1, BS, HKV, D), device=dev) * 8).to(torch.float8_e4m3fn)
            v = (torch.randn((B * nb + 1, BS, HKV, D), device=dev) * 8).to(torch.float8_e4m3fn)
            bt = torch.randperm(B * nb + 1, device=dev)[: B * nb].view(B, nb).to(torch.int32)
            seq = torch.full((B,), L, dtype=torch.int32, device=dev)
            for ql in (6, 8):
                q = torch.randn(B * ql, HQ, D, device=dev, dtype=torch.float16)
                cu = torch.arange(0, B * ql + 1, ql, dtype=torch.int32, device=dev)
                for msk in (MAXLEN, L):
                    args = (q, k, v, None, cu, seq, bt, msk, ks, vs, SCALE, None, (-1, -1), 0.0)
                    ref = spec_packed._orig(*args).float(); got = spec_packed._packed(*args).float()
                    torch.xpu.synchronize()
                    d = (ref - got).abs()
                    rowmax = d.view(B, ql, HQ, D).amax(dim=(0, 2, 3)).tolist()
                    strad = (L - ql) // BS != (L - 1) // BS
                    res.append((B, sp, L, ql, msk, strad, d.max().item(), rowmax))
            del k, v
import statistics
base_med = statistics.median([r[6] for r in res if not r[5]]) or 1e-6
bad = 0
for B, sp, L, ql, msk, strad, mx, rowmax in res:
    flag = mx > 50 * base_med
    bad += flag
    if flag or strad or L % BS in (0, 1):
        print(f"B={B} splits={sp:3d} L={L:6d} q_len={ql} max_seqlen_k={msk:6d} straddle={int(strad)}: max {mx:.5f} "
              f"per row {' '.join(f'{x:.4f}' for x in rowmax)}{'  FAIL' if flag else ''}", flush=True)
print(f"median non-straddling max diff {base_med:.6f}; cases {len(res)}; BOUND", "PASS" if not bad else f"FAIL ({bad})")
