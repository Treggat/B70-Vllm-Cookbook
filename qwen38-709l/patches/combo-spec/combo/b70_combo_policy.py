# 2026-10-21 B70_COMBO_SPEC: scheduler-side policy for the 709L combined copy + MTP speculator.
# Installed as vllm/v1/core/sched/b70_combo_policy.py by patch_combo_spec_032.py; imported by scheduler.py only
# when B70_COMBO_SPEC=1.
#
# Each step the scheduler picks the draft count K for the WHOLE batch (as Dynamic SD does): K_COPY (the configured
# num_speculative_tokens, 15) when every scheduled request is copying from its own context, else K_MTP (5).
# "Copying" = the last N tokens (prompt + output so far) occur earlier in the request at a 4-byte-aligned spot with
# >= MIN_AVAIL tokens after it, and the request is under MAX_CTX tokens. B70_COMBO_SUSTAIN=1 (default) also wants
# proof that a copy is under way: a match only becomes the copy source after the tokens generated since the last
# check equal the source tokens that follow it (exact compare, so repeated 4-grams in code cannot reset it). Each
# check first tries to continue the current source; only when that fails does it rfind a new candidate. Searched with bytes.rfind over a per-request
# int32 buffer that is extended incrementally (C speed, ~0.1 ms at 32k tokens). Under async scheduling the tokens
# seen here lag the worker by ~2 steps; copy stretches are hundreds of tokens, so that only delays entry.
# Back-off: K_COPY steps that yield < MIN_YIELD tokens (accepted + bonus) twice in a row put the request on a
# COOLDOWN-step cooldown (false triggers on boilerplate code cost ~1 step each, then stop).
# The worker (B70ComboSpeculator) still decides per request inside a K_COPY step: copied drafts only when its own
# n-gram match agrees with MTP's first draft, else MTP drafts.
# Depth tiers (2026-10-21 evening): B70_COMBO_MTP_TIERS / B70_COMBO_COPY_TIERS = "ctx:K,ctx:K,..." pick the MTP and the
# copy draft count by the longest scheduled request (largest ctx <= its length wins); unset = K_MTP / the configured
# num_speculative_tokens at every depth. Every K used must have graphs (B70_COMBO_CAPTURE_KS) and its q_len in
# B70_GDN_VARLEN_SPEC. A step is a copy step when K >= COPY_MIN_K (B70_COMBO_COPY_MIN_K, default the configured K);
# the worker uses the same rule.
import os
from array import array

_E = os.environ.get
N = int(_E("B70_COMBO_SCHED_N", "4"))
MIN_AVAIL = int(_E("B70_COMBO_MIN_AVAIL", "10"))
MAX_CTX = int(_E("B70_COMBO_MAX_CTX", "32768"))
MAX_BS = int(_E("B70_COMBO_MAX_BS", "4"))
K_MTP = int(_E("B70_COMBO_K_MTP", "5"))
MIN_YIELD = int(_E("B70_COMBO_MIN_YIELD", "6"))
COOLDOWN = int(_E("B70_COMBO_COOLDOWN", "16"))
DEBUG = _E("B70_COMBO_DEBUG") == "1"
SUSTAIN = _E("B70_COMBO_SUSTAIN", "1") == "1"
# B70_COMBO_CTL (tests only): a file of key=value lines (max_ctx, off, mtp_tiers, copy_tiers) re-read every 8 steps,
# so one load can A/B.
CTL = _E("B70_COMBO_CTL")
_ctl = {}


def _tiers(spec, default_k):
    """"0:5,48000:3" -> [(0, 5), (48000, 3)] sorted; empty/None -> [(0, default_k)]."""
    out = []
    for part in (spec or "").split(","):
        if ":" in part:
            c, k = part.split(":", 1)
            out.append((int(c), int(k)))
    return sorted(out) or [(0, default_k)]


def _pick(tiers, T):
    k = tiers[0][1]
    for c, kk in tiers:
        if T >= c:
            k = kk
    return k


MTP_TIERS = _tiers(_E("B70_COMBO_MTP_TIERS"), K_MTP)
COPY_TIERS = None  # resolved on first use (needs the configured K)
COPY_MIN_K = int(_E("B70_COMBO_COPY_MIN_K", "0"))  # 0 = the configured num_speculative_tokens

_state: dict[str, list] = {}  # req_id -> [buf, n_tokens_in_buf, bad_steps, cooldown, src_ptr, T_at_last_check]
_calls = 0
_stats = {"copy_steps": 0, "mtp_steps": 0}


def _st(req_id: str) -> list:
    s = _state.get(req_id)
    if s is None:
        s = _state[req_id] = [bytearray(), 0, 0, 0, -1, 0]
    return s


def _copying(request) -> bool:
    toks = request._all_token_ids
    T = len(toks)
    s = _st(request.request_id)
    if s[3] > 0:
        s[3] -= 1
        return False
    if T >= MAX_CTX or T < N + MIN_AVAIL + 1:
        if T >= MAX_CTX and s[0]:
            s[0] = bytearray(); s[1] = 0
        return False
    if s[1] > T:  # preempted / rolled back: rebuild
        s[0] = bytearray(); s[1] = 0
    if s[1] < T:
        s[0] += array("i", toks[s[1]:T]).tobytes()
        s[1] = T
    buf = s[0]
    src, t0 = s[4], s[5]
    s[5] = T
    if SUSTAIN and src >= 0 and t0 < T and src + (T - t0) <= t0:
        d = T - t0  # the source continued in step with the output -> copying
        if buf[4 * src:4 * (src + d)] == buf[4 * t0:4 * T]:
            s[4] = src + d
            return T - s[4] >= MIN_AVAIL
    s[4] = -1
    pat = bytes(buf[-4 * N:])
    end = len(buf) - 4  # the match must end at least one token before the current end
    for _ in range(8):
        pos = buf.rfind(pat, 0, end)
        if pos < 0:
            return False
        if pos % 4 == 0:
            s[4] = pos // 4 + N  # candidate source: the token after the match
            return (not SUSTAIN) and T - s[4] >= MIN_AVAIL
        end = pos + len(pat) - 1
    return False


def choose_k(sched, num_scheduled_tokens: dict, default_k: int) -> int:
    """Draft count for the next step (scheduler.schedule, where Dynamic SD picks it)."""
    global _calls
    _calls += 1
    if _calls % 512 == 0:
        for rid in [r for r in _state if r not in sched.requests]:
            del _state[rid]
    global MAX_CTX, MTP_TIERS, COPY_TIERS, COPY_MIN_K
    k_max = sched.num_spec_tokens
    if COPY_TIERS is None:
        COPY_TIERS = _tiers(_E("B70_COMBO_COPY_TIERS"), k_max)
        COPY_MIN_K = COPY_MIN_K or k_max
    if CTL and _calls % 8 == 1:
        try:
            new = dict(l.strip().split("=", 1) for l in open(CTL) if "=" in l)
        except OSError:
            new = {}
        if new != _ctl:
            _ctl.clear(); _ctl.update(new)
            MAX_CTX = int(_ctl.get("max_ctx", _E("B70_COMBO_MAX_CTX", "32768")))
            MTP_TIERS = _tiers(_ctl.get("mtp_tiers", _E("B70_COMBO_MTP_TIERS")), K_MTP)
            COPY_TIERS = _tiers(_ctl.get("copy_tiers", _E("B70_COMBO_COPY_TIERS")), k_max)
            print(f"[B70 combo] ctl {_ctl} -> max_ctx {MAX_CTX} mtp {MTP_TIERS} copy {COPY_TIERS}", flush=True)
    T = 0
    copying = _ctl.get("off") != "1" and len(num_scheduled_tokens) <= MAX_BS
    for req_id in num_scheduled_tokens:
        req = sched.requests.get(req_id)
        if req is None:
            copying = False
            continue
        T = max(T, len(req._all_token_ids))
        # every row must be a decode with output so far, and every row must be copying
        if copying and (req.is_prefill_chunk or req.num_output_tokens == 0 or not _copying(req)):
            copying = False
    k = _pick(COPY_TIERS, T) if copying else _pick(MTP_TIERS, T)
    if copying:
        _stats["copy_steps"] += 1
    else:
        _stats["mtp_steps"] += 1
    if DEBUG and _calls % 256 == 0:
        print(f"[B70 combo] steps copy={_stats['copy_steps']} mtp={_stats['mtp_steps']}", flush=True)
    return k


def observe(request, num_draft_tokens: int, num_accepted: int, k_copy: int) -> None:
    """Per-request verify result (scheduler.update_from_output): back off after weak copy steps."""
    if num_draft_tokens < (COPY_MIN_K or k_copy):
        return
    s = _st(request.request_id)
    if num_accepted + 1 < MIN_YIELD:
        s[2] += 1
        if s[2] >= 2:
            s[2] = 0
            s[3] = COOLDOWN
    else:
        s[2] = 0
