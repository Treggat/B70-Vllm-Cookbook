"""B70 target lm_head fp8 helper. riceharvest's helper (MIT, LICENSE-riceharvest beside this file), adapted 2026-11-26 for
Inkling 2: logits() returns head_dtype (float32 for Inkling) so the final softcap keeps its precision. The int4 draft
head branch of the original is dropped (no MTP/DSpark on XPU). Active only with B70_TARGET_LMHEAD_FP8=1."""
import os
import torch


@torch.no_grad()
def convert(model) -> None:
    if os.environ.get("B70_TARGET_LMHEAD_FP8") != "1":
        return
    head = getattr(model, "lm_head", None)
    if head is None or getattr(head, "_b70_fp8", None) is not None:
        return
    W = getattr(head, "weight", None)
    if W is None or W.dim() != 2 or W.numel() == 0 or W.dtype not in (torch.float16, torch.bfloat16):
        return
    if getattr(head, "tp_size", 1) != 1:
        print("[B70] target fp8 head: TP>1, skipped", flush=True); return
    W = W.data
    V, K = W.shape
    Wq = torch.empty((V, K), dtype=torch.float8_e4m3fn, device=W.device)
    sc = torch.empty(V, dtype=torch.float32, device=W.device)
    for i in range(0, V, 8192):
        w = W[i:i + 8192].float()
        s = w.abs().amax(dim=1).clamp(min=1e-8) / 448.0
        sc[i:i + 8192] = s
        Wq[i:i + 8192] = (w / s[:, None]).to(torch.float8_e4m3fn)
    head._b70_fp8 = (Wq.t(), sc)
    head.weight = torch.nn.Parameter(torch.empty((0, K), dtype=W.dtype, device=W.device), requires_grad=False)
    del W
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    print(f"[B70] target lm_head fp8: {V}x{K}, bf16 freed", flush=True)


def logits(head, x: torch.Tensor, out_dtype=None) -> torch.Tensor:
    Wt, sc = head._b70_fp8
    flat = x.reshape(-1, x.shape[-1])
    if flat.dtype not in (torch.float16, torch.bfloat16):
        flat = flat.to(torch.bfloat16)
    out = torch.ops._xpu_C.fp8_gemm_w8a16(flat, Wt, sc, None)
    if out_dtype is not None and out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out.reshape(*x.shape[:-1], Wt.shape[1])
