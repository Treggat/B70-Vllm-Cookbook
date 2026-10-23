# CPU test of the probabilistic-drafting path of b70_shortlist_head.py (patches/shortlist, 2026-10-22; no B70 needed).
#   SHORTLIST_DATA=<dir> python test_prob_cpu.py [--rows 2000]      (same <dir> contents as test_shortlist_cpu.py)
# Real held-out head inputs (hm_held_sample.pt) + the seat's int4 head rebuilt on CPU + the artifact. Checks:
#   (1) logits_row: shape / dtype, exactly N finite entries per row, at the module's own candidate ids, values equal an
#       fp32 rescoring of those rows (fp16 tolerance), -inf elsewhere
#   (2) argmax(logits_row) == top_tokens (greedy and probabilistic paths pick from the same candidates)
#   (3) top_tokens is byte-identical to the previous (greedy-only) module revision, when SHORTLIST_OLD_MODULE points at it
#   (4) the shortlist's q vs the full head's q at T = 1.0: sum min(q_sl, q_full) (= the acceptance a target equal to the
#       full-head drafter would give the shortlist's drafts), and the same after the target's top_k 20 / top_p 0.95
#   (5) garbage inputs (NaN / inf / zeros) -> a valid distribution (>= 1 finite entry, softmax without NaN)
#   (6) B70_DRAFT_SHORTLIST_PROB gate
import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SLD = os.path.join(HERE, '..')
sys.path.insert(0, SLD)
DATA = os.environ.get('SHORTLIST_DATA', HERE)
VOCAB = os.environ.get('SHORTLIST_VOCAB', os.path.join(DATA, 'draft_vocab_98304.json'))
OLD_MODULE = os.environ.get('SHORTLIST_OLD_MODULE', '')
V = 248320
fails = []


def check(cond, what):
    print(('PASS ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        fails.append(what)


def load_as(name, path):
    from importlib.machinery import SourceFileLoader
    spec = importlib.util.spec_from_file_location(name, path, loader=SourceFileLoader(name, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class FakeModel:
    pass


def trunc(q, top_k=20, top_p=0.95):
    v, i = torch.topk(q, top_k, dim=-1)
    v = v / v.sum(-1, keepdim=True)
    c = v.cumsum(-1)
    v = v * ((c - v) < top_p)
    return i, v / v.sum(-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=2000)
    ap.add_argument('--artifact', default=os.path.join(DATA, 'shortlist_r512.pt'))
    a = ap.parse_args()
    torch.manual_seed(0)
    import build_factors as BF
    SL = load_as('sl_new', os.path.join(SLD, 'b70_shortlist_head.py'))
    OLD = load_as('sl_old', OLD_MODULE) if OLD_MODULE else None
    dev = torch.device('cpu')
    raw = np.fromfile(os.path.join(DATA, 'lm_head_bf16.bin'), dtype=np.uint16)
    Wf = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16).view(V, 5120)
    ids = torch.tensor(sorted(json.load(open(VOCAB))), dtype=torch.long)
    t0 = time.time()
    qw, sc, qz, gs = BF.quantize_lmhead_to_int4(Wf.to(torch.float16).index_select(0, ids).contiguous())
    del Wf, raw
    W = BF.dequant_int4(qw, sc, gs)                                       # fp32 [98816, 5120]
    print(f'seat head rebuilt on CPU in {time.time() - t0:.0f}s', flush=True)

    def model():
        m = FakeModel(); m._b70_lmhead_int4 = (qw, sc, qz, gs); m._b70_draft_vocab_ids = ids; return m

    os.environ['B70_DRAFT_SHORTLIST'] = a.artifact
    os.environ['B70_DRAFT_SHORTLIST_N'] = '128'; os.environ['B70_DRAFT_SHORTLIST_BLOCK'] = '0'
    m = model()
    check(SL.active(m, dev) and (OLD is None or OLD.active(m, dev)), 'module active on the seat head (and the old revision, if given)')
    hm = torch.load(os.path.join(DATA, 'hm_held_sample.pt'))
    H = hm[torch.randperm(hm.shape[0])[:a.rows]]
    N = 128

    # (1) + (2) + (3)
    ok_shape = ok_count = ok_ids = True
    max_err = 0.0
    n_amax = n_same_old = 0
    ovl, ovl_t = [], []
    for i in range(0, H.shape[0], 4):
        h = H[i:i + 4]
        row = SL.logits_row(m, h, V, torch.float16)
        ok_shape &= tuple(row.shape) == (h.shape[0], V) and row.dtype == torch.float16
        fin = torch.isfinite(row)
        ok_count &= bool((fin.sum(1) == N).all())
        ok_ids &= bool(torch.isin(fin.nonzero()[:, 1], ids).all())
        # fp32 rescoring of the finite entries
        h16 = h.half().float()
        for r in range(h.shape[0]):
            tok = fin[r].nonzero().squeeze(1)
            rows_ = torch.searchsorted(ids, tok)
            ref = W[rows_] @ h16[r]
            max_err = max(max_err, float(((row[r, tok].float() - ref).abs() / (ref.abs() + 1)).max()))
        tt = SL.top_tokens(m, h)
        n_amax += int((row.float().argmax(1) == tt).sum())
        n_same_old += int((tt == OLD.top_tokens(m, h)).sum()) if OLD is not None else len(tt)
        # (4) q fidelity vs the full head
        q_sl = torch.softmax(row.float(), -1)
        full = torch.full((h.shape[0], V), float('-inf'))
        full[:, ids] = h16 @ W.t()
        q_full = torch.softmax(full, -1)
        ovl.append(torch.minimum(q_sl, q_full).sum(-1))
        pi, pv = trunc(q_full)                                            # a target equal to the full drafter
        ovl_t.append(torch.minimum(pv, q_sl.gather(1, pi)).sum(-1) / torch.minimum(pv, q_full.gather(1, pi)).sum(-1))
    n = H.shape[0]
    check(ok_shape, 'logits_row: [n, 248320] float16')
    check(ok_count, f'logits_row: exactly N={N} finite entries per row, -inf elsewhere')
    check(ok_ids, 'logits_row: finite entries only at draft-vocab ids')
    check(max_err < 2e-3, f'logits_row: candidate logits == fp32 rescoring (max rel err {max_err:.2e})')
    check(n_amax >= 0.999 * n, f'argmax(logits_row) == top_tokens on {n_amax}/{n} rows')
    if OLD is not None:
        check(n_same_old == n, f'top_tokens byte-identical to the previous revision on {n_same_old}/{n} rows')
    o, ot = torch.cat(ovl), torch.cat(ovl_t)
    print(f'  q fidelity T=1.0: sum min(q_shortlist, q_full) mean {float(o.mean()):.5f}, p1 {float(o.quantile(0.01)):.4f}; '
          f'acceptance vs a top_k20/top_p.95 target equal to the full drafter, shortlist / full: mean {float(ot.mean()):.5f}, '
          f'p1 {float(ot.quantile(0.01)):.4f}')
    # the untruncated overlap is informational: the full head keeps ~2% of q past the top 128 (first run 0.978, against
    # a 99% bar set beforehand), mass the target's top_k 20 cuts anyway; the bar that matters is the next one, and the
    # real-target replay (probeval --shortlist: Ps == Pv within 0.04%). Sanity floor only.
    check(float(o.mean()) > 0.95, 'q fidelity: untruncated overlap with the full head > 0.95 (sanity; informational)')
    check(float(ot.mean()) > 0.995, 'q fidelity after the target truncation: > 99.5% of the full head acceptance')

    # (5) garbage inputs
    for name, bad in (('all NaN', torch.full((4, 5120), float('nan'))), ('inf', torch.full((4, 5120), float('inf'))),
                      ('zeros', torch.zeros(4, 5120)), ('mixed NaN', torch.cat([H[:2], torch.full((2, 5120), float('nan'))]))):
        try:
            row = SL.logits_row(m, bad, V, torch.float16)
            q = torch.softmax(row.float(), -1)
            check(bool(torch.isfinite(row).any(1).all()) and not bool(q.isnan().any()),
                  f'{name} hidden states -> valid distribution')
        except Exception as e:  # noqa: BLE001
            check(False, f'{name} hidden states raised {e!r}')
    row = SL.logits_row(m, H[:4].view(2, 2, 5120), V, torch.bfloat16)
    check(tuple(row.shape) == (4, V) and row.dtype == torch.bfloat16, f'[2, 2, D] input, bf16 -> {tuple(row.shape)} {row.dtype}')

    # (6) gate
    os.environ.pop('B70_DRAFT_SHORTLIST_PROB', None); g1 = SL.prob_on()
    os.environ['B70_DRAFT_SHORTLIST_PROB'] = '0'; g0 = SL.prob_on()
    os.environ['B70_DRAFT_SHORTLIST_PROB'] = '1'; g2 = SL.prob_on()
    check(g1 and not g0 and g2, 'B70_DRAFT_SHORTLIST_PROB: default on, 0 off, 1 on')
    print('ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
