#!/usr/bin/env python3
"""2026-10-25 KV split rule for the V-split verify kernel, vLLM 0.32 709L seat (rh-runtime/spec_packed.py).

The V-split verify kernel (README.md beside this file) runs up to 48 packed verify rows as ONE work-group of 8
subgroups per (sequence, KV head, split). It is fastest at exactly 4 work-groups per Xe core, one full wave (128 on
the B70): 32 splits for one sequence, 8 for four. The split count riceharvest's _graph_splits picks is sized for the
8-row tile (5 work-groups per KV head) and roughly doubles the V-split kernel's time, so with B70_SPEC_VSPLIT=1 this
rule takes over for those calls. Like the rest of _graph_splits it only applies while a graph is being captured.

Switch: B70_SPEC_VSPLIT=1 (the kernel reads the same variable). Unset, the file behaves exactly as before.

Run AFTER patches/combo-spec/patch_spec_packed_guard_032.py (apply_stack.sh copies rh-runtime/ fresh, then the guard
patch edits it). Idempotent (marker), fail-closed (exact anchors), leaves a .bak of the unpatched file.
Target: $VLLM032_ROOT/rh-runtime/spec_packed.py (VLLM032_ROOT = the venv dir given to scripts/apply_stack.sh)."""
import os, pathlib, py_compile, shutil, sys

ROOT = os.environ.get("VLLM032_ROOT") or sys.exit("set VLLM032_ROOT to the venv dir given to scripts/apply_stack.sh")
p = pathlib.Path(ROOT) / "rh-runtime" / "spec_packed.py"
MARK = "_VSPLIT"

A1 = "_XE_CORES = {}\n"
N1 = A1 + '_VSPLIT = os.environ.get("B70_SPEC_VSPLIT") == "1"   # the kernel reads the same switch (paged_decode_xe2.cpp)\n'
A2 = "    tile = 8 if (rows <= 8 or D > 128) else 16\n"
N2 = (
    "    # B70 V-split verify tile (B70_SPEC_VSPLIT=1, 2026-10-25, patches/vsplit-verify/): up to 48 packed rows run as ONE\n"
    "    # 8-subgroup work-group per (sequence, KV head, split); measured fastest at exactly 4 work-groups per Xe core (one\n"
    "    # full wave, 128 on the B70: B=1 32 splits, B=4 8); 36 splits at B=1 (144 work-groups) already cost ~50%.\n"
    "    if _VSPLIT and D == 256 and rows <= 48 and cores > 0:\n"
    "        return max(1, min(128, (4 * cores) // (B * Hkv)))\n"
) + A2

if not p.is_file():
    sys.exit(f"refusing: {p} not found")
s = p.read_text()
if MARK in s:
    print(f"already patched: {p}")
    sys.exit(0)
if "_PACK_QLENS" not in s:
    sys.exit("refusing: run patches/combo-spec/patch_spec_packed_guard_032.py first")
for a in (A1, A2):
    if s.count(a) != 1:
        sys.exit(f"refusing: anchor found {s.count(a)} times (want 1): {a.strip()!r}")
bak = p.with_name(p.name + ".bak-2026-10-25-pre-vsplit")
if not bak.exists():
    shutil.copy2(p, bak)
t = s.replace(A1, N1).replace(A2, N2)
p.write_text(t, newline="\n")
try:
    py_compile.compile(str(p), doraise=True)
except py_compile.PyCompileError as e:
    p.write_text(s, newline="\n")
    sys.exit(f"compile failed, original restored: {e}")
print(f"patched {p} (backup {bak.name})")
