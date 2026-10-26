# kt_depth.py: one 709-L full-attention layer at decode depth through riceharvest's packed-verify runtime
# (rh-runtime/spec_packed.py, patched by ../patch_spec_packed_vsplit_032.py) and the installed attention library:
# q_len 6 (target verify, packed) and q_len 1 (drafter decode, stock path), fp8 KV, B=1, max_seqlen_k = max_model_len
# (what a replayed graph passes). Prints us/layer and effective KV GB/s.
# The runtime's split rule only acts during graph capture, so pass the count the captured graph uses:
#   KT_SPLITS=32 with B70_SPEC_VSPLIT=1 (V-split tile), KT_SPLITS=12 with it unset (8-row tile, riceharvest's rule).
# Run from a shell where vllm_xpu_kernels and spec_packed import (B70_RUNTIME = the rh-runtime dir), e.g.
#   B70_RUNTIME=$VENV/rh-runtime B70_SPEC_PACKED=1 B70_SPEC_PACKED_QLENS=any B70_SPEC_VSPLIT=1 KT_SPLITS=32 python kt_depth.py
import math, os, sys, torch
sys.path.insert(0, os.environ.get("B70_RUNTIME", "."))
import spec_packed  # noqa
dev = "xpu"
D, HQ, HKV, BS, MAXLEN = 256, 24, 4, 960, 262144
SCALE = 1 / math.sqrt(D)
ks = torch.full((1,), 0.037, device=dev); vs = torch.full((1,), 0.021, device=dev)
graph_like = os.environ.get("KT_GRAPHLIKE", "1") == "1"
SPLITS = [int(x) for x in os.environ.get("KT_SPLITS", "0").split(",")]   # 0 = the live heuristic
_heur = spec_packed._graph_splits


def timeit(fn, it=20):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        fn()
    e1.record(); torch.xpu.synchronize()
    return e0.elapsed_time(e1) * 1000 / it


for ctx in [int(x) for x in os.environ.get("KT_CTX", "23000,61000,100000,150000,200000,250000").split(",")]:
    nb = (ctx + BS - 1) // BS
    KVD = torch.float16 if os.environ.get("KT_KV") == "fp16" else torch.float8_e4m3fn
    k = (torch.randn((nb + 1, BS, HKV, D), device=dev) * (0.1 if KVD == torch.float16 else 8)).to(KVD)
    v = (torch.randn((nb + 1, BS, HKV, D), device=dev) * (0.1 if KVD == torch.float16 else 8)).to(KVD)
    bt = torch.arange(nb, device=dev, dtype=torch.int32)[None]
    seq = torch.full((1,), ctx, dtype=torch.int32, device=dev)
    for ql, sp in [(6, s) for s in SPLITS] + [(1, 0)]:
        spec_packed._graph_splits = _heur if sp == 0 else (lambda *a, _s=sp, **kw: _s)
        q = torch.randn(ql, HQ, D, device=dev, dtype=torch.float16)
        cu = torch.tensor([0, ql], dtype=torch.int32, device=dev)
        msk = MAXLEN if graph_like else ctx
        kd, vd = (None, None) if KVD == torch.float16 else (ks, vs)
        us = timeit(lambda: spec_packed._packed(q, k, v, None, cu, seq, bt, msk, kd, vd, SCALE, None, (-1, -1), 0.0))
        gbs = ctx * HKV * D * 2 * k.element_size() / (us * 1e-6) / 1e9
        print(f"ctx={ctx:6d} q_len={ql} splits={sp or 'heur':>4}: {us:7.1f} us/layer  x16 = {16 * us / 1000:5.2f} ms  KV {gbs:5.0f} GB/s", flush=True)
    del k, v
