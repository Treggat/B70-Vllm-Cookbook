"""B70 shortlist draft head for the 709L MTP drafter (patches/shortlist, installed by patch_shortlist_032.py).

Greedy drafts only need the argmax of the draft head. Instead of the full 98,816-row int4 head (440 us of the
~1.09 ms draft pass), score every row with a covariance-weighted rank-r factorization, keep the top N rows and
rescore only those with the real int4 head:
    z = h @ Q                       Q   [5120, r] fp16
    s = int4_gemm_w4a16(z, P)       P   [r, 98816] int4 g128 (seat layout)
    cand = topk(s, N)                (or, with B70_DRAFT_SHORTLIST_BLOCK=b: the rows of the top N/b b-row blocks)
    draft = ids[cand[argmax(int4_gemm_w4a16(h, head[cand]))]]   the candidates rescored by the head's own GEMM;
                                                                 ties go to the lowest row (= the stock argmax)
The target verifies every draft, so a shortlist miss costs acceptance, never output.

Probabilistic drafts (draft_sample_method "probabilistic", 2026-10-22): logits_row() scatters the N
rescored candidate logits into a -inf row of the full vocab width; the speculator samples from that row with the stock
gumbel_sample, which also caches it as the draft distribution q the rejection sampler checks against, so drafts and q
always agree (lossless). q = softmax over the N candidates only; the target's top_k 20 lies inside them almost always
(offline recall 0.996-0.998 for the top-1 at N = 128). B70_DRAFT_SHORTLIST_PROB=0 sends probabilistic drafts back to
the stock full-head path.

Gate: B70_DRAFT_SHORTLIST=<artifact .pt from build_factors.py>; B70_DRAFT_SHORTLIST_N=<N> (default 128);
B70_DRAFT_SHORTLIST_BLOCK=<b> (default 0 = plain topk; offline 2026-10-22 only b = 4 with N = 256 held up, -0.24%).
Fail-closed: any mismatch (no int4 draft head, other vocab rows, other head bytes, missing op on XPU) disables the
shortlist with one log line and the stock path runs.
"""
import os

import torch

_STATE = {}          # id(model) -> dict, or None when disabled for that model
_DEQ = {}            # CPU test fallback only: dequantized int4 weights


def _log(msg):
    print(f"[B70] draft shortlist: {msg}", flush=True)


def _int4_mm(x, qweight, scales, qzeros, group_size, cache=True):
    """x [n, K] @ int4 [K, N] -> [n, N]; the seat's int4_gemm_w4a16, or a dequant fallback off XPU (tests).
    cache=False for per-call gathered weights (their storage is reused, so a data_ptr key would go stale)."""
    if x.device.type == "xpu":
        return torch.ops._xpu_C.int4_gemm_w4a16(x, qweight, None, scales, qzeros, group_size, None)
    if not cache:
        return x.float() @ _dequant_nt(qweight, scales, group_size).t()
    key = (qweight.data_ptr(), tuple(qweight.shape))
    if key not in _DEQ:                     # test-only path: dequantize once per weight
        _DEQ[key] = _dequant_nt(qweight, scales, group_size).t().to(x.device)
    return x.float() @ _DEQ[key]


def _dequant_nt(qweight, scales, group_size):
    """qweight [K//8, N] (NT view of [N, K//8]), scales [g, N] -> fp32 [N, K]"""
    qc = qweight.t()
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qc.device)
    q = ((qc[:, :, None] >> shifts) & 0xF).reshape(qc.shape[0], -1).float() - 8.0
    return q * scales.t().float().repeat_interleave(group_size, dim=1)


def _same_head(rows, scales, fp_rows, fp_scales):
    """Same head as the artifact's? The int4 scales (maxabs / 7) are bit-identical on every device; the nibbles can
    differ by 1 where a value sits on a rounding tie (0.145% of nibbles CPU vs CUDA, 2026-10-22). So: scales equal,
    >= 99.5% of nibbles equal, none off by more than 1. A different checkpoint fails on the scales."""
    if fp_scales is None or rows.shape != fp_rows.shape or scales.shape != fp_scales.shape:
        return False, "fingerprint shape/scales missing"
    if not torch.equal(scales.half(), fp_scales.half()):
        return False, "scales differ"
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    d = (((rows[:, :, None] >> sh) & 0xF) - ((fp_rows[:, :, None] >> sh) & 0xF)).abs()
    eq, mx = float((d == 0).float().mean()), int(d.max())
    return eq >= 0.995 and mx <= 1, f"nibbles equal {eq:.5f}, max diff {mx}"


def _init(model, device):
    path = os.environ.get("B70_DRAFT_SHORTLIST", "")
    n = int(os.environ.get("B70_DRAFT_SHORTLIST_N", "128") or 128)
    blk = int(os.environ.get("B70_DRAFT_SHORTLIST_BLOCK", "0") or 0)
    if getattr(model, "_b70_lmhead_int4", None) is None and os.environ.get("B70_DRAFT_LMHEAD_INT4") == "1":
        try:
            from vllm.model_executor.models.b70_draft_lmhead_int4 import build_draft_lmhead_int4
            build_draft_lmhead_int4(model)
        except Exception as e:  # noqa: BLE001
            _log(f"OFF: building the int4 draft head failed: {e!r}")
            return None
    head = getattr(model, "_b70_lmhead_int4", None)
    if head is None:
        _log("OFF: no int4 draft head on this model (B70_DRAFT_LMHEAD_INT4 off or another drafter)")
        return None
    if device.type == "xpu" and not hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
        _log("OFF: torch.ops._xpu_C.int4_gemm_w4a16 missing")
        return None
    try:
        art = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        _log(f"OFF: cannot load {path!r}: {e!r}")
        return None
    qweight, scales, qzeros, group_size = head
    ids = getattr(model, "_b70_draft_vocab_ids", None)
    rows = qweight.shape[1]
    if art.get("version") != 1 or art["rows"] != rows or art["group_size"] != group_size:
        _log(f"OFF: artifact rows/group {art.get('rows')}/{art.get('group_size')} != head {rows}/{group_size}")
        return None
    if ids is None or ids.numel() != rows or not torch.equal(ids.cpu(), art["ids"]):
        _log("OFF: draft vocab ids differ from the artifact's (B70_DRAFT_VOCAB changed?)")
        return None
    ok, same = _same_head(qweight.t()[::997].cpu(), scales.t()[::997].cpu(),
                          art["head_fp_rows"], art.get("head_fp_scales"))
    if not ok:
        _log(f"OFF: int4 head differs from the artifact's ({same})")
        return None
    n = max(1, min(n, rows))
    if blk and (rows % blk or n % blk):
        _log(f"OFF: block {blk} must divide the rows ({rows}) and N ({n})")
        return None
    st = {
        "n": n,
        "Q": art["Q"].to(device=device, dtype=torch.float16).contiguous(),
        "Pq": art["P_qweight"].to(device),
        "Ps": art["P_scales"].to(device=device, dtype=torch.float16),
        "Pz": art["P_qzeros"].to(device),
        "Pg": int(art["group_size"]),
        "rows_q": qweight.t(),                                    # [rows, 640] contiguous view of the head
        "rows_s": scales.t().contiguous(),                        # [rows, 40]
        "qz": qzeros,
        "g": int(group_size),
        "ids": ids.to(device),
        "rows": rows,
        "blk": blk,
        "blk_ar": torch.arange(max(blk, 1), device=device),
    }
    _log(f"ON: rank {art['rank']}, N {n}, block {blk or 'off'}, {rows} rows, head fingerprint {art['head_fingerprint']} "
         f"({same}), built {art.get('built')}")
    return st


def active(model, device) -> bool:
    key = id(model)
    if key not in _STATE:
        _STATE[key] = _init(model, device)
    return _STATE[key] is not None


def prob_on() -> bool:
    return os.environ.get("B70_DRAFT_SHORTLIST_PROB", "1") != "0"


def _candidates(st, h):
    """shortlist rows cand [n, N] and their exact rescored logits [n, N] (NaN -> -inf) for fp16 h [n, 5120]"""
    n_rows, N = h.shape[0], st["n"]
    z = h @ st["Q"]                                                               # [n, r]
    s = _int4_mm(z, st["Pq"], st["Ps"], st["Pz"], st["Pg"])                       # [n, rows]
    b = st["blk"]
    if b:                                                                         # rows of the top N/b blocks
        top = torch.topk(s.view(n_rows, -1, b).amax(-1), N // b, dim=-1, sorted=False).indices
        cand = (top[:, :, None] * b + st["blk_ar"]).reshape(n_rows, N)
    else:
        cand = torch.topk(s, N, dim=-1, sorted=False).indices                     # [n, N]
    flat = cand.reshape(-1)
    # rescoring = the head's own int4 GEMM on the gathered candidate rows: rows_q[flat] is [n*N, 640] contiguous, so
    # its .t() is the head's NT layout [640, n*N]; every request's h meets every gathered column, keep its own N
    qT = st["rows_q"].index_select(0, flat).t()
    sT = st["rows_s"].index_select(0, flat).t().contiguous()                      # [40, n*N]
    lg = _int4_mm(h, qT, sT, st["qz"], st["g"], cache=False).view(n_rows, n_rows, N)
    exact = torch.diagonal(lg, dim1=0, dim2=1).t()                                # [n, N]
    # NaN -> -inf: the drafter's warm-up feeds garbage hidden states (seat-B load crash 2026-10-22 9:26 PM)
    return cand, torch.nan_to_num(exact, nan=float("-inf"))


@torch.no_grad()
def logits_row(model, hidden_states: torch.Tensor, vocab: int, dtype: torch.dtype) -> torch.Tensor:
    """draft logits [n, vocab] in `dtype`: the N rescored candidates at their token ids, -inf elsewhere"""
    st = _STATE[id(model)]
    h = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.float16)
    cand, exact = _candidates(st, h)
    # a row with no finite candidate (warm-up garbage) gets 0 on its first candidate: q stays a distribution
    dead = ~torch.isfinite(exact).any(dim=1, keepdim=True)
    exact = torch.where(dead & (torch.arange(exact.shape[1], device=exact.device) == 0), 0.0, exact)
    # +inf (overflow) would make softmax NaN: clamp to the dtype's range, -inf stays -inf
    exact = exact.clamp(max=torch.finfo(dtype).max)
    row = torch.full((h.shape[0], vocab), float("-inf"), dtype=dtype, device=h.device)
    row.scatter_(1, st["ids"][cand], exact.to(dtype))
    return row


@torch.no_grad()
def top_tokens(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """greedy draft token ids [n] (full-vocab ids) for hidden_states [n, 5120]"""
    st = _STATE[id(model)]
    h = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.float16)
    n_rows, N = h.shape[0], st["n"]
    z = h @ st["Q"]                                                               # [n, r]
    s = _int4_mm(z, st["Pq"], st["Ps"], st["Pz"], st["Pg"])                       # [n, rows]
    b = st["blk"]
    if b:                                                                         # rows of the top N/b blocks
        top = torch.topk(s.view(n_rows, -1, b).amax(-1), N // b, dim=-1, sorted=False).indices
        cand = (top[:, :, None] * b + st["blk_ar"]).reshape(n_rows, N)
    else:
        cand = torch.topk(s, N, dim=-1, sorted=False).indices                     # [n, N]
    flat = cand.reshape(-1)
    # rescoring = the head's own int4 GEMM on the gathered candidate rows: rows_q[flat] is [n*N, 640] contiguous, so
    # its .t() is the head's NT layout [640, n*N]; every request's h meets every gathered column, keep its own N
    qT = st["rows_q"].index_select(0, flat).t()
    sT = st["rows_s"].index_select(0, flat).t().contiguous()                      # [40, n*N]
    lg = _int4_mm(h, qT, sT, st["qz"], st["g"], cache=False).view(n_rows, n_rows, N)
    exact = torch.diagonal(lg, dim1=0, dim2=1).t()                                # [n, N]
    # tie-break on the lowest row among the maxima, like the stock argmax over the full head: topk(sorted=False)
    # gives the candidates in no fixed order, so argmax-first-in-list could pick a different tied row per run.
    # NaN -> -inf first: the drafter's warm-up feeds garbage hidden states, an all-NaN row has no maximum, and the
    # sentinel row would index past the head (seat-B load crash 2026-10-22 9:26 PM); the clamp is the backstop.
    exact = torch.nan_to_num(exact, nan=float("-inf"))
    top = exact == exact.max(dim=1, keepdim=True).values
    best = torch.where(top, cand, torch.full_like(cand, st["rows"])).min(dim=1).values.clamp(max=st["rows"] - 1)
    return st["ids"][best].view(hidden_states.shape[:-1])
