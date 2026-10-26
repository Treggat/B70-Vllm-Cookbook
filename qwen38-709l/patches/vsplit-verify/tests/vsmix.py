# vsmix.py: packed verify vs the stock per-position path for MIXED batches: sequence 0 ends right at a 960-token page
# edge (its verify rows straddle it), the others sit at unrelated depths. Splits = the runtime rule 4*32/(B*Hkv)
# (B=2: 16, B=3: 10, B=4: 8). Also every batch order (the straddling sequence first, last, middle).
import math, os, sys, statistics, torch
sys.path.insert(0, os.environ.get("B70_RUNTIME", "."))
os.environ["B70_SPEC_PACKED"] = "1"; os.environ["B70_SPEC_PACKED_QLENS"] = "any"
import spec_packed  # noqa
D, HQ, HKV, BS, MAXLEN = 256, 24, 4, 960, 262144
SCALE = 1 / math.sqrt(D); dev = "xpu"
ks = torch.full((1,), 0.0371, device=dev); vs = torch.full((1,), 0.0213, device=dev)
OTHERS = [1500, 23000, 61000, 3000, 100000]
res = []
for B in (2, 3, 4):
    sp = max(1, (4 * 32) // (B * HKV))
    spec_packed._graph_splits = lambda *a, _s=sp, **k: _s
    for edge in (3840, 4800, 1920):
        for L0 in range(edge - 2, edge + 7, 2):
            for pos in range(B):
                lens = OTHERS[:B - 1]; lens = lens[:pos] + [L0] + lens[pos:]
                torch.manual_seed(L0 * 13 + B * 7 + pos)
                nbs = [(L + BS - 1) // BS for L in lens]; nbmax = max(nbs); tot = sum(nbs)
                k = (torch.randn((tot + 1, BS, HKV, D), device=dev) * 8).to(torch.float8_e4m3fn)
                v = (torch.randn((tot + 1, BS, HKV, D), device=dev) * 8).to(torch.float8_e4m3fn)
                perm = torch.randperm(tot + 1, device=dev)[:tot].to(torch.int32)
                bt = torch.zeros((B, nbmax), dtype=torch.int32, device=dev); o = 0
                for b, nb in enumerate(nbs):
                    bt[b, :nb] = perm[o:o + nb]; o += nb
                seq = torch.tensor(lens, dtype=torch.int32, device=dev)
                for ql in (6,):
                    q = torch.randn(B * ql, HQ, D, device=dev, dtype=torch.float16)
                    cu = torch.arange(0, B * ql + 1, ql, dtype=torch.int32, device=dev)
                    args = (q, k, v, None, cu, seq, bt, MAXLEN, ks, vs, SCALE, None, (-1, -1), 0.0)
                    ref = spec_packed._orig(*args).float(); got = spec_packed._packed(*args).float()
                    torch.xpu.synchronize()
                    per_seq = (ref - got).abs().view(B, ql, HQ, D).amax(dim=(1, 2, 3)).tolist()
                    res.append((B, sp, lens, pos, max(per_seq), per_seq))
                del k, v
med = statistics.median(r[4] for r in res)
bad = 0
for B, sp, lens, pos, mx, per_seq in res:
    flag = mx > max(50 * med, 1e-3)
    bad += flag
    if flag or lens[pos] % BS in (1, 2, 3):
        print(f"B={B} splits={sp:2d} lens={lens} straddling #{pos}: max {mx:.5f} per seq {' '.join(f'{x:.5f}' for x in per_seq)}{'  FAIL' if flag else ''}", flush=True)
print(f"median max diff {med:.6f}; cases {len(res)}; MIX", "PASS" if not bad else f"FAIL ({bad})")
