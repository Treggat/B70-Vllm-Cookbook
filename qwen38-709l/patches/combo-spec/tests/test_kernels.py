"""Check _ngram_scan_kernel + _combo_finalize_kernel against a Python reference.
Run with TRITON_INTERPRET=1 on CPU tensors (no GPU): the kernels are pulled out of the source files with ast so
no vllm import is needed. argv: <ngram/speculator.py> <b70_combo_speculator.py>"""
import ast, random, sys
import torch
import triton
import triton.language as tl


def load_kernel(path, name):
    src = open(path).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            code = ast.get_source_segment(src, node)
            dec = "".join("@" + ast.get_source_segment(src, d) + "\n" for d in node.decorator_list)
            ns = {"triton": triton, "tl": tl}
            exec(dec + code, ns)
            return ns[name]
    raise SystemExit(f"{name} not in {path}")


scan = load_kernel(sys.argv[1], "_ngram_scan_kernel")
fin = load_kernel(sys.argv[2], "_combo_finalize_kernel")


def ref(seq, MIN_N, MAX_N, scan_len, K, mtp, KM, USE_MIN_N, MIN_AVAIL, AGREE, num_sampled):
    L = len(seq)
    best = None
    for n in range(MIN_N, MAX_N + 1):
        if L < n + 1:
            continue
        suf = seq[L - n:]
        for pos in range(0, min(L - n - 1, scan_len - 1) + 1):
            if seq[pos:pos + n] == suf:
                if best is None or (n, pos) > best:
                    best = (n, pos)
    mtp_out = [mtp[k] if k < KM else mtp[KM - 1] for k in range(K)]
    if best is None or num_sampled == 0 or L < MIN_N:
        return mtp_out, 0
    n, pos = best
    start = pos + n
    avail = L - start
    gathered = [seq[start + k] for k in range(min(avail, K))]
    use = n >= USE_MIN_N and avail >= MIN_AVAIL and (not AGREE or gathered[0] == mtp[0])
    if not use:
        return mtp_out, 0
    last = gathered[-1]
    return gathered + [last] * (K - len(gathered)), 1


random.seed(0)
MAXLEN, BLOCK, K, KM = 4096, 256, 15, 5
fails = 0
cases = 0
copied = 0
for trial in range(60):
    B = random.randint(1, 4)
    toks = torch.zeros((4, MAXLEN), dtype=torch.int32)
    lens, seqs = [], []
    for b in range(B):
        L = random.randint(8, 3000)
        V = random.choice([5, 50, 5000])  # small vocab = many matches
        s = [random.randrange(V) for _ in range(L)]
        if random.random() < 0.5 and L > 200:  # make the tail copy an earlier stretch
            a = random.randrange(0, L - 150)
            ln = random.randint(5, 40)
            s = s[:L - ln] + s[a:a + ln]
        toks[b, :L] = torch.tensor(s, dtype=torch.int32)
        lens.append(L); seqs.append(s)
    idx = torch.arange(B, dtype=torch.int64)
    total = torch.zeros(4, dtype=torch.int32); total[:B] = torch.tensor(lens, dtype=torch.int32)
    num_sampled = torch.tensor([random.choice([1, 1, 1, 0]) for _ in range(B)], dtype=torch.int32)
    scan_ctx = random.choice([1024, 2048, 4096])
    nb = triton.cdiv(scan_ctx, BLOCK)
    scratch = torch.zeros((4, nb), dtype=torch.int64)
    mtp = torch.randint(0, 50, (4, K), dtype=torch.int64)
    for b in range(B):  # often agree with the copy so the copy path is exercised
        if random.random() < 0.6:
            r0, _ = ref(seqs[b], 2, 5, scan_ctx, K, [None] * K, 1, 1, 1, False, 1)
            if r0[0] is not None:
                mtp[b, 0] = r0[0]
    out = torch.zeros((4, K), dtype=torch.int64)
    use = torch.zeros(4, dtype=torch.int32)
    AGREE = random.random() < 0.7
    scan[(B, nb)](toks, toks.stride(0), idx, total, num_sampled, scratch, scratch.stride(0), MAXLEN,
                  2, 5, 8, BLOCK)
    fin[(B,)](toks, toks.stride(0), idx, total, num_sampled, scratch, scratch.stride(0), mtp, mtp.stride(0),
              out, use, MAXLEN, nb, K, 16, max(1, triton.next_power_of_2(nb)), KM, 3, 4, AGREE)
    for b in range(B):
        cases += 1
        e_out, e_use = ref(seqs[b], 2, 5, nb * BLOCK, K, mtp[b].tolist(), KM, 3, 4, AGREE, int(num_sampled[b]))
        g_out, g_use = out[b].tolist(), int(use[b])
        copied += g_use
        if e_use != g_use or (e_out != g_out):
            fails += 1
            if fails <= 5:
                print("MISMATCH trial", trial, "b", b, "L", lens[b], "use", e_use, g_use, "\n exp", e_out, "\n got", g_out)
print(f"{cases} rows checked ({copied} took the copy path), {fails} mismatches")
sys.exit(1 if fails else 0)
