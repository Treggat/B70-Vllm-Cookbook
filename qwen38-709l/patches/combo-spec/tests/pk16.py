"""pk16.py: is riceharvest's packed verify right (and faster) at q_len 11 and 16, not only 6?
Compares spec_packed._packed (one decode query per KV head, per-row causal mask in the b70 kernel) with the stock
per-position path (_orig) on the 709-L shapes: Hq 24, Hkv 4, head 256, fp8 e4m3 paged KV, page 896, B=1 and B=2.
q_len 6 is the control (known good in prod). Run on a free card with the live venv:
  VLLM032_ROOT=<venv dir> ZE_AFFINITY_MASK=0 B70_SPEC_PACKED=1 B70_SPEC_PACKED_QLENS=any python pk16.py
Pass = packed vs stock max abs diff at q_len 11/16 within ~2x of the q_len-6 control at every depth."""
import math, os, sys, torch
sys.path.insert(0, os.path.join(os.environ.get("VLLM032_ROOT") or sys.exit("set VLLM032_ROOT (the venv dir)"), "rh-runtime"))
os.environ["B70_SPEC_PACKED"] = "1"
os.environ["B70_SPEC_PACKED_QLENS"] = "any"
import vllm_xpu_kernels.flash_attn_interface as fai
import spec_packed  # noqa: E402  (installs _packed; keeps _orig)

D, HQ, HKV, BS = 256, 24, 4, 896
SCALE = 1 / math.sqrt(D)
dev = "xpu"
ks = torch.full((1,), 0.0371, device=dev, dtype=torch.float32)
vs = torch.full((1,), 0.0213, device=dev, dtype=torch.float32)


def kv(B, L):
    nb = (L + BS - 1) // BS
    k = (torch.randn((B * nb + 1, BS, HKV, D), device=dev) * 8).to(torch.float8_e4m3fn)
    v = (torch.randn((B * nb + 1, BS, HKV, D), device=dev) * 8).to(torch.float8_e4m3fn)
    bt = torch.randperm(B * nb + 1, device=dev)[: B * nb].view(B, nb).to(torch.int32)
    return k, v, bt


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    st, en = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    st.record()
    for _ in range(n):
        fn()
    en.record(); torch.xpu.synchronize()
    return st.elapsed_time(en) / n * 1000


bad = 0
ctrl = {}
for B in (1, 2):
    for ctx in (2048, 45000, 115000):
        torch.manual_seed(ctx + B)
        k, v, bt = kv(B, ctx)
        seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
        for ql in (6, 11, 16):
            q = torch.randn(B * ql, HQ, D, device=dev, dtype=torch.float16)
            cu = torch.arange(0, B * ql + 1, ql, dtype=torch.int32, device=dev)
            args = (q, k, v, None, cu, seq, bt, ctx, ks, vs, SCALE, None, (-1, -1), 0.0)
            ref = spec_packed._orig(*args).float()
            got = spec_packed._packed(*args).float()
            torch.xpu.synchronize()
            diff = (ref - got).abs().max().item()
            t_orig = timeit(lambda: spec_packed._orig(*args))
            t_pack = timeit(lambda: spec_packed._packed(*args))
            if ql == 6:
                ctrl[(B, ctx)] = max(diff, 1e-4)
            ok = diff <= 2.5 * ctrl[(B, ctx)] + 1e-3
            bad += not ok
            print(f"B={B} ctx={ctx:6d} q_len={ql:2d}: max|packed-stock| {diff:.5f} ({'ok' if ok else 'FAIL'})  "
                  f"stock {t_orig:8.1f} us  packed {t_pack:8.1f} us  ({t_orig / t_pack:4.2f}x)", flush=True)
        del k, v
print("PK16", "PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
