#!/usr/bin/env python3
"""2026-10-21 B70 packed-verify q_len guard for the vLLM 0.32 709L seat (rh-runtime/spec_packed.py).

spec_packed._packed() packs every uniform verify batch with q_len >= 2 into one decode query per KV head. Only
q_len 2, 3 and 6 (1, 2 or 5 drafts) have ever been checked lossless on this .so (8d1bf4d1ea74); every other q_len
(MTP-6/7 = 7/8, ngram K=15 = 16) has only ever run with B70_SPEC_PACKED off. This guard sends any other q_len back to
the stock _spec_decode_varlen_fwd (exact, one decode row per position), so turning on more drafts can never verify
through an unchecked packed shape.

Switch: B70_SPEC_PACKED_QLENS = comma list of q_len values that may pack (default "2,3,6"); "any" = the old behaviour.
Widen it only after a kcheck of that q_len, packed vs unpacked.

apply.sh copies rh-runtime/ fresh from the 031rh tree, so this must run AFTER that copy (it does: listed after
b70/patch_spec_packed.py). Idempotent (marker), fail-closed (exact anchors), leaves a .bak of the unguarded file.
Target: $VLLM032_ROOT/rh-runtime/spec_packed.py (default /appdata/vllm032-709l)."""
import os, pathlib, py_compile, shutil, sys

ROOT = os.environ.get("VLLM032_ROOT", "/appdata/vllm032-709l")
p = pathlib.Path(ROOT) / "rh-runtime" / "spec_packed.py"
MARK = "B70_SPEC_PACKED_QLENS"

A1 = "_DBG = os.environ.get('B70_SPEC_DEBUG') == '1'; _seen = set()\n"
N1 = A1 + (
    "# B70_SPEC_PACKED_QLENS (2026-10-21 guard): pack only the q_len values checked lossless on this .so; any other\n"
    "# q_len goes to the stock per-position path. Default 2,3,6 = 1, 2 or 5 drafts; \"any\" = pack every q_len >= 2.\n"
    "_QL_ENV = os.environ.get('B70_SPEC_PACKED_QLENS', '2,3,6').strip()\n"
    "_PACK_QLENS = None if _QL_ENV == 'any' else frozenset(int(x) for x in _QL_ENV.split(',') if x.strip())\n"
)
A2 = "    if s_aux is not None or softcap != 0.0 or tuple(window_size) != (-1, -1) or ql < 2:\n"
N2 = ("    if (s_aux is not None or softcap != 0.0 or tuple(window_size) != (-1, -1) or ql < 2\n"
      "            or (_PACK_QLENS is not None and ql not in _PACK_QLENS)):\n")

if not p.is_file():
    sys.exit(f"refusing: {p} not found")
s = p.read_text()
if MARK in s:
    print(f"already guarded: {p}")
    sys.exit(0)
for a in (A1, A2):
    if s.count(a) != 1:
        sys.exit(f"refusing: anchor found {s.count(a)} times (want 1): {a.strip()!r}")
bak = p.with_name(p.name + ".bak-2026-10-21-pre-qlen-guard")
if not bak.exists():
    shutil.copy2(p, bak)
t = s.replace(A1, N1).replace(A2, N2)
p.write_text(t, newline="\n")
try:
    py_compile.compile(str(p), doraise=True)
except py_compile.PyCompileError as e:
    p.write_text(s, newline="\n")
    sys.exit(f"compile failed, original restored: {e}")
print(f"guarded {p} (backup {bak.name})")
