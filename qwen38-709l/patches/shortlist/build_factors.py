# Build the shortlist draft head artifact for the 709-L seat (patches/shortlist; runs on any CUDA or CPU box, no B70).
#
# Rebuilds the seat's exact draft head: lm_head.weight (bf16 in the checkpoint) cast to fp16 as vLLM loads it, the rows
# listed in the draft-vocab JSON (scripts/make_draft_vocab.py, 98,816 ids), int4-quantized with the seat's own
# quantize_lmhead_to_int4 (copied below verbatim, minus the device plumbing). The covariance C = E[h h^T] of the drafter's
# head inputs comes from --inputs (a .pt tensor [n, 5120] of hidden states as they reach the drafter's sample_draft,
# fp16/bf16/fp32) or from a saved --cov (.pt with {"C": [5120, 5120] float64, "rows": n}). Ours came from 287,000 rows
# of our own agent traffic (6 sessions) replayed through an offline copy of the MTP drafter. Use traffic like yours:
# the subspace fits what it was fit on (README.md). Factors:
#   M = C^(1/2),  V = top-r eigenvectors of M W^T W M,  P = W M V  [rows, r],  Q = M^-1 V  [5120, r]
#   W h ~ P (Q^T h); P is int4-quantized with the same function (groups of 128 along r) for int4_gemm_w4a16.
# Writes <out>.pt (the artifact the seat loads: P qweight/scales/qzeros, Q fp16, ids, head fingerprint, meta) and
# <out>.json (the meta alone). The head comes from --model-dir (safetensors, key lm_head.weight) or --head-bin (the
# same tensor as raw bf16 bytes, [248320, 5120]).
# usage: python build_factors.py --model-dir <AWQ model dir> --vocab ../../serve/draft_vocab_98304.json #            (--inputs head_inputs.pt | --cov cov.pt) --rank 512 --out shortlist_r512
import argparse, hashlib, json, os, time
import numpy as np
import torch

V_FULL, D = 248320, 5120


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def quantize_lmhead_to_int4(weight: torch.Tensor, group_size: int = 128):
    """b70_draft_lmhead_int4.quantize_lmhead_to_int4 (vllm032-709l tree, 2026-10-22), same arithmetic."""
    device = weight.device
    N, K = weight.shape
    num_groups = K // group_size
    chunk = 4096
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device)
    parts, scale_parts = [], []
    for i in range(0, N, chunk):
        wc = weight[i:i + chunk].float()
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)
        scale = maxabs / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(wc.shape[0], K // 8)
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight_contig = torch.cat(parts, dim=0)
    scales_contig = torch.cat(scale_parts, dim=0)
    return qweight_contig.t(), scales_contig.t().contiguous(), torch.tensor([8], dtype=torch.int8), group_size


def dequant_int4(qweight, scales, group_size):
    """inverse of the above: qweight [K//8, N] (NT view), scales [g, N] -> fp32 [N, K]"""
    qc = qweight.t().contiguous()                       # [N, K//8]
    N = qc.shape[0]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qc.device)
    q = ((qc[:, :, None] >> shifts) & 0xF).reshape(N, -1).float() - 8.0
    return q * scales.t().float().repeat_interleave(group_size, dim=1)


def fingerprint(qweight, scales):
    """every 997th head row: packed nibbles + the int4 scales (b70_shortlist_head._same_head compares them)"""
    qc = qweight.t().contiguous()
    sr = scales.t()[::997].contiguous()
    sha = hashlib.sha256(sr.cpu().numpy().tobytes()).hexdigest()[:16]   # scales are bit-identical on any device
    return sha, qc[::997].clone(), sr.clone()


@torch.no_grad()
def covariance(inputs_path, dev):
    """C = E[h h^T] over a tensor of drafter head inputs [n, 5120] (fp16 / bf16 / fp32), accumulated in float64."""
    H = torch.load(inputs_path, weights_only=False)
    if isinstance(H, dict):
        H = H.get("h", H.get("inputs"))
    H = H.reshape(-1, D)
    C = torch.zeros(D, D, dtype=torch.float64, device=dev)
    for i in range(0, H.shape[0], 8192):
        h = H[i:i + 8192].to(dev).float()
        C += (h.t() @ h).double()
    n = int(H.shape[0])
    log(f'covariance from {os.path.basename(inputs_path)}: rows {n}')
    return (C / n).cpu(), n


def load_head(model_dir, head_bin):
    """lm_head.weight as bf16 [248320, 5120]: from the model's safetensors, or from a raw bf16 dump."""
    if head_bin:
        raw = np.fromfile(head_bin, dtype=np.uint16)
        assert raw.size == V_FULL * D, raw.size
        return torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16).view(V_FULL, D)
    from safetensors import safe_open
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        shard = json.load(open(idx))["weight_map"]["lm_head.weight"]
    else:
        shard = "model.safetensors"
    with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
        W = f.get_tensor("lm_head.weight")
    assert tuple(W.shape) == (V_FULL, D), tuple(W.shape)
    return W.to(torch.bfloat16)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='', help='AutoAWQ model dir (safetensors with lm_head.weight)')
    ap.add_argument('--head-bin', default='', help='or: lm_head.weight as raw bf16 bytes [248320, 5120]')
    ap.add_argument('--vocab', required=True, help='draft-vocab JSON id list (scripts/make_draft_vocab.py)')
    ap.add_argument('--inputs', default='', help='.pt tensor [n, 5120] of drafter head inputs for the covariance')
    ap.add_argument('--cov', default='', help='or: a saved covariance .pt ({"C", "rows"}) from an earlier run')
    ap.add_argument('--save-cov', default='', help='write the covariance here for reuse')
    ap.add_argument('--rank', type=int, default=512)
    ap.add_argument('--out', default='shortlist_r512')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if not (a.model_dir or a.head_bin):
        ap.error('give --model-dir or --head-bin')
    if not (a.inputs or a.cov):
        ap.error('give --inputs or --cov')
    dev = torch.device(a.device)
    W_full = load_head(a.model_dir, a.head_bin)
    ids = torch.tensor(sorted(json.load(open(a.vocab))), dtype=torch.long)
    vocab_sha = hashlib.sha256(open(a.vocab, 'rb').read()).hexdigest()[:16]
    w16 = W_full.to(torch.float16).index_select(0, ids).contiguous()       # vLLM: bf16 -> fp16 at load, then rows
    log(f'seat head rows {tuple(w16.shape)} from {os.path.basename(a.vocab)} (sha {vocab_sha})')
    qw, sc, qz, gs = quantize_lmhead_to_int4(w16.to(dev))
    fp_sha, fp_rows, fp_scales = fingerprint(qw, sc)
    W = dequant_int4(qw, sc, gs)                                            # fp32 [98816, 5120] on dev
    log(f'int4 head built, fingerprint {fp_sha}')
    if a.cov:
        cv = torch.load(a.cov, weights_only=False); C, nrows = cv['C'], cv['rows']; cfiles = cv.get('files', [os.path.basename(a.cov)])
        log(f'covariance loaded from {a.cov} ({nrows} rows)')
    else:
        C, nrows = covariance(a.inputs, dev)
        cfiles = [os.path.basename(a.inputs)]
        if a.save_cov:
            torch.save({'C': C, 'rows': nrows, 'files': cfiles}, a.save_cov)
            log(f'wrote {a.save_cov}')
    ev, evec = torch.linalg.eigh(C.to(dev))
    ev = ev.clamp(min=ev.max() * 1e-8)
    Mh = ((evec * ev.sqrt()) @ evec.t()).float()
    Mi = ((evec * ev.rsqrt()) @ evec.t()).float()
    WM = W @ Mh
    G = (WM.t() @ WM).double()
    lam, Vv = torch.linalg.eigh(G)
    lam, Vv = lam.flip(0), Vv.flip(1)
    r = a.rank
    Vr = Vv[:, :r].float()
    P = WM @ Vr                                                             # [98816, r]
    Q = Mi @ Vr                                                             # [5120, r]
    kept = float(lam[:r].sum() / lam.sum())
    log(f'rank {r}: weighted energy kept {kept:.5f}')
    del WM, G
    Pq, Ps, Pz, Pg = quantize_lmhead_to_int4(P.to(torch.float16))
    # self-check: dequantized factors vs the full head on random C-shaped inputs
    z = torch.randn(256, D, device=dev) @ Mh
    full = z @ W.t()
    approx = (z @ Q.half().float()) @ dequant_int4(Pq, Ps, Pg).t()
    rel = float((full - approx).norm() / full.norm())
    log(f'self-check on C-shaped inputs: relative logit error {rel:.4f}')
    art = {'version': 1, 'rank': r, 'group_size': Pg, 'rows': int(ids.numel()), 'vocab_full': V_FULL,
           'P_qweight': Pq.cpu(), 'P_scales': Ps.cpu(), 'P_qzeros': Pz.cpu(), 'Q': Q.half().cpu().contiguous(),
           'ids': ids, 'vocab_sha': vocab_sha, 'head_fingerprint': fp_sha, 'head_fp_rows': fp_rows.cpu(), 'head_fp_scales': fp_scales.cpu(),
           'cov_files': cfiles, 'cov_rows': nrows, 'energy_kept': kept, 'self_check_rel_err': rel,
           'built': time.strftime('%Y-%m-%d %H:%M'), 'by': 'patches/shortlist/build_factors.py'}
    torch.save(art, a.out + '.pt')
    sz = os.path.getsize(a.out + '.pt') / 1e6
    with open(a.out + '.json', 'w') as f:
        json.dump({k: v for k, v in art.items() if not isinstance(v, torch.Tensor)}, f, indent=1)
    log(f'wrote {a.out}.pt ({sz:.1f} MB) + {a.out}.json')


if __name__ == '__main__':
    main()
