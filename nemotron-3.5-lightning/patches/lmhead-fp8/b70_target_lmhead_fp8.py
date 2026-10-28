"""B70 target lm_head fp8 helper (written by patch_target_lmhead_fp8.py)."""
import json, os
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
    if os.environ.get("B70_DRAFT_LMHEAD_INT4") == "1":
        from vllm.model_executor.models.b70_draft_lmhead_int4 import quantize_lmhead_to_int4
        ids, w = None, W
        vf = os.environ.get("B70_DRAFT_VOCAB")
        if vf:
            ids = torch.tensor(sorted(json.load(open(vf))), dtype=torch.long, device=W.device)
            w = W.index_select(0, ids).contiguous()
        head._b70_draft_int4 = (*quantize_lmhead_to_int4(w), ids, V)
        del w
        print(f"[B70] draft int4 head prebuilt ({V if ids is None else ids.numel()} rows)", flush=True)
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
    print(f"[B70] target lm_head fp8: {V}x{K}, fp16 freed", flush=True)


def logits(head, x: torch.Tensor) -> torch.Tensor:
    Wt, sc = head._b70_fp8
    flat = x.reshape(-1, x.shape[-1])
    out = torch.ops._xpu_C.fp8_gemm_w8a16(flat, Wt, sc, None)
    return out.reshape(*x.shape[:-1], Wt.shape[1])
