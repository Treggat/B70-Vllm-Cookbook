# CPU test of b70_shortlist_head.py against a plain reference (patches/shortlist; no B70 needed).
#   SHORTLIST_DATA=<dir> python test_shortlist_cpu.py [--artifact shortlist_r512.pt]   (exit 1 on any failure)
# <dir> holds lm_head_bf16.bin (the model's lm_head.weight as raw bf16, see build_factors.py), the draft-vocab JSON
# (scripts/make_draft_vocab.py) and hm_held_sample.pt, a [n, 5120] tensor of real drafter head inputs.
# --make-sample is how we captured ours (an offline replica of the MTP drafter over recorded traffic; needs that replica).
# Checks: (1) the fail-closed gates of _init (wrong ids, other head bytes, other row count, missing file), (2) the
# module's draft ids equal a straightforward fp32 reference of the same algorithm (topk of the dequantized factors,
# exact rescoring with the dequantized seat head), (3) agreement with the full-head argmax on real held-out head
# inputs at N = 128 and 64 (sanity bound; the per-session numbers are in ../README.md), (4) shapes.
import argparse, glob, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
DATA = os.environ.get('SHORTLIST_DATA', HERE)
VOCAB = os.environ.get('SHORTLIST_VOCAB', os.path.join(DATA, 'draft_vocab_98304.json'))
fails = []


def check(cond, what):
    print(('PASS ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        fails.append(what)


def make_sample(a):
    import mtplib as M
    dev = torch.device('cuda')
    cache = os.path.join(DATA, 'cache')
    cfg = type('C', (), {})()
    for k, v in json.load(open(os.path.join(cache, 'cfg.json'))).items():
        setattr(cfg, k, v)
    emb = torch.load(os.path.join(cache, 'emb.pt')); embf = lambda ids: emb[ids.cpu()].to(dev)
    w = torch.load(os.path.join(cache, 'mtp_hf.pt'))
    blk = M.MTPBlock(cfg, w, dtype=torch.bfloat16).to(dev).eval(); del w
    one = torch.zeros(1, 5120, dtype=torch.bfloat16, device=dev)
    out = []
    for fp in sorted(glob.glob(os.path.join(DATA, 'held', '*.mtpd')))[-2:]:   # 000006 (code) + 000007 (rp)
        s = M.read_mtpd(fp)
        tok = torch.from_numpy(s.tokens.astype(np.int64)).to(dev); h = s.h().to(dev)
        lp = torch.from_numpy(s.loss_pos.astype(np.int64)).to(dev)
        rows = []
        grab = lambda hm: (rows.append(hm.float().cpu()), hm @ one.t())[1]
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            teacher = M.teacher_rows(blk, embf, h, tok)
            st = torch.arange(0, min(lp.shape[0], 1024), device=dev)
            M.chain_forward(blk, grab, embf, h, tok, lp, 5, chunk=128, starts=st, teacher=teacher)
        out.append(torch.cat(rows))
        print(os.path.basename(fp), s.id, sum(r.shape[0] for r in rows), 'rows')
    hm = torch.cat(out)
    torch.save(hm, os.path.join(DATA, 'hm_held_sample.pt'))
    print('saved', hm.shape)


class FakeModel:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-sample', action='store_true')
    ap.add_argument('--artifact', default=os.path.join(DATA, 'shortlist_r512.pt'))
    ap.add_argument('--rows', type=int, default=3000, help='held-out head inputs to test')
    a = ap.parse_args()
    if a.make_sample:
        return make_sample(a)
    torch.manual_seed(0)
    import build_factors as BF
    import b70_shortlist_head as SL
    dev = torch.device('cpu')
    # the seat's int4 head, rebuilt exactly as build_factors.py does
    raw = np.fromfile(os.path.join(DATA, 'lm_head_bf16.bin'), dtype=np.uint16)
    Wf = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16).view(248320, 5120)
    ids = torch.tensor(sorted(json.load(open(VOCAB))), dtype=torch.long)
    t0 = time.time()
    qw, sc, qz, gs = BF.quantize_lmhead_to_int4(Wf.to(torch.float16).index_select(0, ids).contiguous())
    del Wf, raw
    W = BF.dequant_int4(qw, sc, gs)                                       # fp32 [98816, 5120]
    print(f'seat head rebuilt on CPU in {time.time() - t0:.0f}s', flush=True)
    art = torch.load(a.artifact, weights_only=True)

    def model_with(q=qw, s=sc, i=ids):
        m = FakeModel(); m._b70_lmhead_int4 = (q, s, qz, gs); m._b70_draft_vocab_ids = i; return m

    # (1) fail-closed gates
    os.environ['B70_DRAFT_SHORTLIST'] = a.artifact
    os.environ['B70_DRAFT_SHORTLIST_N'] = '128'
    check(SL._init(model_with(), dev) is not None, 'gate: matching head + ids -> ON')
    bad_ids = ids.clone(); bad_ids[5] = 200000
    check(SL._init(model_with(i=bad_ids), dev) is None, 'gate: other vocab ids -> OFF')
    q2 = qw.t().contiguous().clone(); q2[::997] = torch.randint_like(q2[::997], -2**31, 2**31 - 1); q2 = q2.t()
    check(SL._init(model_with(q=q2), dev) is None, 'gate: scrambled head nibbles -> OFF')
    check(SL._init(model_with(s=(sc.float() * 1.01).half()), dev) is None, 'gate: other scales (checkpoint) -> OFF')
    check(SL._init(model_with(), dev) is not None, 'gate: CPU-quantized head (rounding ties vs the CUDA build) -> ON')
    check(SL._init(model_with(q=qw[:, :-8], s=sc[:, :-8], i=ids[:-8]), dev) is None, 'gate: other row count -> OFF')
    m0 = FakeModel(); m0._b70_lmhead_int4 = None
    os.environ.pop('B70_DRAFT_LMHEAD_INT4', None)
    check(SL._init(m0, dev) is None, 'gate: no int4 head -> OFF')
    os.environ['B70_DRAFT_SHORTLIST'] = a.artifact + '.missing'
    check(SL._init(model_with(), dev) is None, 'gate: missing artifact -> OFF')
    os.environ['B70_DRAFT_SHORTLIST'] = a.artifact

    # (2)+(3) on real held-out head inputs
    hm = torch.load(os.path.join(DATA, 'hm_held_sample.pt'))
    sel = torch.randperm(hm.shape[0])[:a.rows]
    H = hm[sel]
    full = ids[(H.half().float() @ W.t()).argmax(1)]                     # the stock greedy draft
    P = BF.dequant_int4(art['P_qweight'], art['P_scales'], art['group_size'])   # [rows, r]
    Q = art['Q'].float()
    for N, BLK in ((128, 0), (64, 0), (256, 4)):
        os.environ['B70_DRAFT_SHORTLIST_N'] = str(N); os.environ['B70_DRAFT_SHORTLIST_BLOCK'] = str(BLK)
        m = model_with(); SL._STATE.clear()
        check(SL.active(m, dev), f'N={N} block={BLK}: active')
        got = torch.cat([SL.top_tokens(m, H[i:i + 4]) for i in range(0, H.shape[0], 4)])
        h16 = H.half().float()
        s_ref = ((h16 @ Q.half().float()).half().float()) @ P.t()
        if BLK:
            tb = torch.topk(s_ref.view(s_ref.shape[0], -1, BLK).amax(-1), N // BLK, dim=1).indices
            cand = (tb[:, :, None] * BLK + torch.arange(BLK)).reshape(s_ref.shape[0], N)
        else:
            cand = torch.topk(s_ref, N, dim=1).indices
        exact = torch.cat([torch.einsum('nkd,nd->nk', W[cand[i:i + 200]], h16[i:i + 200])
                           for i in range(0, cand.shape[0], 200)])
        ref = ids[cand.gather(1, exact.argmax(1, keepdim=True)).squeeze(1)]
        agree_ref = float((got == ref).float().mean())
        agree_full = float((got == full).float().mean())
        ref_full = float((ref == full).float().mean())
        print(f'  N={N} block={BLK}: module == reference {agree_ref:.5f}, module == full head {agree_full:.5f}, '
              f'reference == full head {ref_full:.5f} ({H.shape[0]} rows)')
        check(agree_ref >= 0.999, f'N={N} block={BLK}: module matches the fp32 reference (>= 0.999)')
        # informational: this sample (first 1024 chains of held files 000006 code + 000007 rp) is harder than the
        # eval average; on 2026-10-22 it read 0.989 / 0.980 (N 128 / 64). The bound only catches a broken artifact.
        check(agree_full >= 0.97, f'N={N} block={BLK}: agreement with the full-head argmax {agree_full:.4f} (sanity >= 0.97)')
    os.environ['B70_DRAFT_SHORTLIST_N'] = '100'; os.environ['B70_DRAFT_SHORTLIST_BLOCK'] = '8'
    check(SL._init(model_with(), dev) is None, 'gate: block 8 does not divide N 100 -> OFF')
    # tie-break: reversing topk's candidate order must not change a single draft
    os.environ['B70_DRAFT_SHORTLIST_N'] = '128'; os.environ['B70_DRAFT_SHORTLIST_BLOCK'] = '0'
    m = model_with(); SL._STATE.clear(); SL.active(m, dev)
    _topk = torch.topk

    class _Rev:
        def __init__(self, r):
            self.values, self.indices = r.values.flip(-1), r.indices.flip(-1)
    Hs = H[:400]
    fwd = torch.cat([SL.top_tokens(m, Hs[i:i + 4]) for i in range(0, Hs.shape[0], 4)])
    torch.topk = lambda *a, **k: _Rev(_topk(*a, **k))
    try:
        rev = torch.cat([SL.top_tokens(m, Hs[i:i + 4]) for i in range(0, Hs.shape[0], 4)])
    finally:
        torch.topk = _topk
    check(bool((fwd == rev).all()), f'tie-break: reversed candidate order -> identical drafts ({Hs.shape[0]} rows)')
    # garbage warm-up inputs (seat-B load crash 2026-10-22 9:26 PM): NaN / inf rows must give valid ids, not raise
    for name, bad in (('all NaN', torch.full((4, 5120), float('nan'))), ('inf', torch.full((4, 5120), float('inf'))),
                      ('zeros', torch.zeros(4, 5120))):
        try:
            out = SL.top_tokens(m, bad)
            check(bool(torch.isin(out, ids).all()), f'{name} hidden states -> valid draft ids')
        except Exception as e:  # noqa: BLE001
            check(False, f'{name} hidden states raised {e!r}')
    # (4) shapes
    os.environ['B70_DRAFT_SHORTLIST_N'] = '128'; os.environ['B70_DRAFT_SHORTLIST_BLOCK'] = '0'
    m = model_with(); SL._STATE.clear(); SL.active(m, dev)
    for n in (1, 2, 4):
        out = SL.top_tokens(m, H[:n])
        check(tuple(out.shape) == (n,) and out.dtype == torch.long, f'shape: {n} rows -> {tuple(out.shape)}')
    out = SL.top_tokens(m, H[:4].view(2, 2, 5120))
    check(tuple(out.shape) == (2, 2), f'shape: [2, 2, D] -> {tuple(out.shape)}')
    print('ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
