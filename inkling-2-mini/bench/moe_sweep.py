"""Standalone sweep of the NVFP4 emulation MoE kernel at Inkling 2 Mini expert shapes on a B70 (no model load).
E=128 experts, top-8, w13 [E,1152,2304], w2 [E,2304,576], bf16 activations, random weights/routing.
Reports ms per call for the w13 and w2 GEMMs at each token count; est step = 44 layer passes x (w13 + w2).
  python moe_sweep.py <phase> <out.jsonl>      phase = A (N,K,warps grid) | B (extras around a base config, see BASE)
Run with the recipe venv and the serving environment (VLLM_TARGET_DEVICE=xpu, the card's ZE_AFFINITY_MASK)."""
import itertools, json, os, sys, time
import torch
from vllm.triton_utils import tl
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import invoke_fused_moe_nvfp4_emulation_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

dev = torch.device("xpu")
E, TOPK, H, I = 128, 8, 2304, 576
SHAPES = {"w13": (2 * I, H), "w2": (H, I)}
TOKENS = [int(x) for x in os.environ.get("SWEEP_TOKENS", "8,16,128").split(",")]
g = torch.Generator().manual_seed(0)
W = {}
for name, (n, k) in SHAPES.items():
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, generator=g).to(dev)
    s = torch.randint(0x30, 0x48, (E, n, k // 16), dtype=torch.uint8, generator=g).to(dev)  # sane positive e4m3
    W[name] = (w, s, torch.ones(E, dtype=torch.float32, device=dev))
IDS = {t: torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(t)]).to(torch.int32).to(dev) for t in TOKENS}
X = {t: torch.randn(t, H, generator=g).to(torch.bfloat16).to(dev) for t in TOKENS}
ACT = {t: torch.randn(t * TOPK, I, generator=g).to(torch.bfloat16).to(dev) for t in TOKENS}


def gemm(name, a, top_k, align, cfg, split, opts):
    w, s, gs = W[name]
    out = torch.empty((a.shape[0], top_k, w.shape[1]), dtype=torch.bfloat16, device=dev)
    invoke_fused_moe_nvfp4_emulation_kernel(a, w, out, s, None, gs, None, align[0], align[1], align[2], False, top_k,
                                            cfg, compute_type=tl.bfloat16, launch_options=opts, split_dots=split)
    return out


def bench(cfg, split, opts, reps=20):
    res = {}
    for t in TOKENS:
        align = moe_align_block_size(IDS[t], cfg["BLOCK_SIZE_M"], E, ignore_invalid_experts=True)[:3]
        for name, a, tk in (("w13", X[t], TOPK), ("w2", ACT[t], 1)):
            for _ in range(3):
                gemm(name, a, tk, align, cfg, split, opts)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                gemm(name, a, tk, align, cfg, split, opts)
            torch.xpu.synchronize()
            res[f"{name}@{t}"] = round((time.perf_counter() - t0) / reps * 1000, 3)
        res[f"step@{t}"] = round(44 * (res[f"w13@{t}"] + res[f"w2@{t}"]), 1)
    return res


def cases(phase):
    if phase == "A":
        yield {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1}, True, {"num_warps": 8, "num_stages": 2}
        for n, k, w in itertools.product((64, 128, 256), (64, 128, 256), (4, 8, 16)):
            if (n, k, w) == (128, 128, 8):
                continue
            yield {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k, "GROUP_SIZE_M": 1}, True, {"num_warps": w, "num_stages": 2}
    elif phase == "C":  # prefill-sized batches (set SWEEP_TOKENS=512,2048): vLLM's current choice, then a grid
        from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
        for t in TOKENS:
            cur = try_get_optimal_moe_config(W["w13"][0].size(), W["w2"][0].size(), TOPK, None, t)
            yield ({k: cur[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")}, False,
                   {k: cur[k] for k in ("num_warps", "num_stages") if k in cur})
        for split, m, n, k, w in itertools.product((False, True), (32, 64, 128), (64, 128, 256), (64, 128), (8, 16)):
            yield {"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k, "GROUP_SIZE_M": 1}, split, {"num_warps": w, "num_stages": 2}
    else:
        n, k, w = (int(x) for x in os.environ["BASE"].split(","))
        base = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k, "GROUP_SIZE_M": 1}
        yield dict(base), True, {"num_warps": w, "num_stages": 2}
        yield dict(base, BLOCK_SIZE_M=32), True, {"num_warps": w, "num_stages": 2}
        yield dict(base, BLOCK_SIZE_M=64), True, {"num_warps": w, "num_stages": 2}
        yield dict(base), True, {"num_warps": w, "num_stages": 3}
        yield dict(base), True, {"num_warps": w, "num_stages": 1}
        yield dict(base), True, {"num_warps": w, "num_stages": 2, "grf_mode": "large"}
        yield dict(base, BLOCK_SIZE_K=512), True, {"num_warps": w, "num_stages": 2}
        yield dict(base, BLOCK_SIZE_N=512), True, {"num_warps": w, "num_stages": 2}
        yield dict(base, BLOCK_SIZE_M=32), False, {"num_warps": w, "num_stages": 2}
        yield dict(base, BLOCK_SIZE_M=64), False, {"num_warps": w, "num_stages": 2}


out = open(sys.argv[2], "a")
for cfg, split, opts in cases(sys.argv[1]):
    rec = {"cfg": cfg, "split": split, "opts": opts}
    t0 = time.time()
    try:
        rec.update(bench(cfg, split, opts))
    except Exception as e:  # a config Triton or the driver rejects
        rec["error"] = repr(e)[:300]
    rec["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
print("SWEEP_DONE", flush=True)
