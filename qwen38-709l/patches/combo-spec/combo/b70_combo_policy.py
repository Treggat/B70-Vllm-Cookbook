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
# B70_COMBO_CTL (tests only): a file of key=value lines (max_ctx, off) re-read every 64 steps, so one load can A/B.
CTL = _E("B70_COMBO_CTL")
_ctl = {}

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
    global MAX_CTX
    if CTL and _calls % 64 == 1:
        try:
            new = dict(l.strip().split("=", 1) for l in open(CTL) if "=" in l)
        except OSError:
            new = {}
        if new != _ctl:
            _ctl.clear(); _ctl.update(new)
            MAX_CTX = int(_ctl.get("max_ctx", _E("B70_COMBO_MAX_CTX", "32768")))
            print(f"[B70 combo] ctl {_ctl} -> max_ctx {MAX_CTX}", flush=True)
    k_copy = sched.num_spec_tokens
    k = k_copy
    if _ctl.get("off") == "1" or len(num_scheduled_tokens) > MAX_BS:
        k = K_MTP
    else:
        for req_id in num_scheduled_tokens:
            req = sched.requests.get(req_id)
            # every row must be a decode with output so far, and every row must be copying
            if req is None or req.is_prefill_chunk or req.num_output_tokens == 0 or not _copying(req):
                k = K_MTP
                break
    if k != K_MTP:
        _stats["copy_steps"] += 1
    else:
        _stats["mtp_steps"] += 1
    if DEBUG and _calls % 256 == 0:
        print(f"[B70 combo] steps copy={_stats['copy_steps']} mtp={_stats['mtp_steps']}", flush=True)
    return k


def observe(request, num_draft_tokens: int, num_accepted: int, k_copy: int) -> None:
    """Per-request verify result (scheduler.update_from_output): back off after weak copy steps."""
    if num_draft_tokens != k_copy:
        return
    s = _st(request.request_id)
    if num_accepted + 1 < MIN_YIELD:
        s[2] += 1
        if s[2] >= 2:
            s[2] = 0
            s[3] = COOLDOWN
    else:
        s[2] = 0
