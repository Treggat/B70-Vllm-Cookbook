# vsrepeat.py: determinism stress for the packed verify path. Fixed inputs, the same call N times, every output must
# be bit-identical to the first (a race in the kernel's SLM exchange would show up as a rare differing call).
# Same call and shapes as kt_depth.py (709-L full-attention layer, fp8 KV, B=1, q_len 6, graph-like max_seqlen_k).
import math, os, sys, torch
sys.path.insert(0, os.environ.get("B70_RUNTIME", "."))
import spec_packed  # noqa
dev = "xpu"
D, HQ, HKV, BS, MAXLEN = 256, 24, 4, 960, 262144
SCALE = 1 / math.sqrt(D)
N = int(os.environ.get("VR_N", "500"))
ks = torch.full((1,), 0.037, device=dev); vs = torch.full((1,), 0.021, device=dev)
torch.manual_seed(0)
bad_total = 0
for ctx in [int(x) for x in os.environ.get("VR_CTX", "200,1500,4000,23000,100000").split(",")]:
    for ql in (6, 8):
        nb = (ctx + BS - 1) // BS
        k = (torch.randn((nb + 1, BS, HKV, D), device=dev) * 8).to(torch.float8_e4m3fn)
        v = (torch.randn((nb + 1, BS, HKV, D), device=dev) * 8).to(torch.float8_e4m3fn)
        bt = torch.arange(nb, device=dev, dtype=torch.int32)[None]
        seq = torch.full((1,), ctx, dtype=torch.int32, device=dev)
        q = torch.randn(ql, HQ, D, device=dev, dtype=torch.float16)
        cu = torch.tensor([0, ql], dtype=torch.int32, device=dev)
        call = lambda: spec_packed._packed(q, k, v, None, cu, seq, bt, MAXLEN, ks, vs, SCALE, None, (-1, -1), 0.0)
        ref = call().clone(); torch.xpu.synchronize()
        bad = 0; worst = 0.0
        for i in range(N):
            o = call()
            if not torch.equal(o, ref):
                bad += 1; worst = max(worst, (o.float() - ref.float()).abs().max().item())
        torch.xpu.synchronize()
        finite = bool(torch.isfinite(ref).all())
        print(f"ctx={ctx:6d} q_len={ql}: {N} calls, {bad} differ from the first (max |diff| {worst:.6f}), finite={finite}", flush=True)
        bad_total += bad + (not finite)
        del k, v
print("REPEAT PASS" if bad_total == 0 else f"REPEAT FAIL ({bad_total})")
