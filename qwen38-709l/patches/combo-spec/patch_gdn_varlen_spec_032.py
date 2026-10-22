#!/usr/bin/env python3
"""2026-10-21 B70_GDN_VARLEN_SPEC: let the GDN metadata builder take a pure spec-decode step SHORTER than
num_speculative_tokens + 1 (vLLM 0.32, 709L combined copy + MTP speculator).

Why: with K max 15 (copy steps verify 16 tokens) and MTP steps verifying 6, the builder's pure-spec branch
(gdn_attn.py) expects num_spec_decodes * 16 tokens. A 6-token step then looks like a truncated final group,
B70_MTP_PARTIAL_FINAL_GROUP reclassifies it as a stateful PREFILL (which cannot roll back rejected drafts), and
spec_token_indx would be sized for 16 tokens per request. The XPU GDN op already supports ragged spec steps
(vllm-xpu-kernels 0.1.15.4: gated_delta_rule_spec_kernel reads the initial state from column num_accepted-1 and
writes columns 0..t-1; causal_conv1d_spec shifts by the actual token count; gdn_attention checks
spec_token <= num_spec_decodes * (num_spec + 1)), so only the expected size changes. The state-index width stays
num_spec + 1: the initial state can sit in any of the 16 columns after a copy step.

Switch: B70_GDN_VARLEN_SPEC = comma list of per-request verify lengths (q_len) accepted as COMPLETE groups,
e.g. "6,16". Unset = stock behaviour. A step is treated as complete only when every spec row has the same length
and that length is in the list; anything else (a real max-length truncation) still takes the
B70_MTP_PARTIAL_FINAL_GROUP path. Requires patch_mtp_boundary_032.py first (its marker anchors this one).
Target: $VLLM_ROOT (vllm/ package dir) or argv[1]. Idempotent, fail-closed, .bak, py_compile.
"""
import os, pathlib, py_compile, shutil, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VLLM_ROOT", ""))
p = root / "v1" / "attention" / "backends" / "gdn_attn.py"
MARK = "B70_GDN_VARLEN_SPEC"

A_IMPORT = "import torch\n"
N_IMPORT = ("import torch\n"
            "\n"
            "# B70_GDN_VARLEN_SPEC: verify lengths accepted as complete pure-spec groups (unset = stock).\n"
            "_B70_GDN_VARLEN_QLENS = frozenset(\n"
            "    int(_x) for _x in __import__('os').environ.get('B70_GDN_VARLEN_SPEC', '').split(',') if _x.strip()\n"
            ") or None\n")
A_EXP = ("                expected_spec_token_size = num_spec_decodes * (self.num_spec + 1)\n"
         "                actual_spec_token_size = query_start_loc_cpu[-1].item()\n")
N_EXP = A_EXP + (
    "                # B70_GDN_VARLEN_SPEC: a uniform step of an allowed shorter length is complete.\n"
    "                if (\n"
    "                    _B70_GDN_VARLEN_QLENS is not None\n"
    "                    and uniform_spec_sequence_length in _B70_GDN_VARLEN_QLENS\n"
    "                    and actual_spec_token_size\n"
    "                    == num_spec_decodes * uniform_spec_sequence_length\n"
    "                ):\n"
    "                    expected_spec_token_size = actual_spec_token_size\n")

if not p.is_file():
    sys.exit(f"refusing: {p} not found (set VLLM_ROOT to the vllm/ package dir)")
s = p.read_text()
if MARK in s:
    print(f"already patched: {p}")
    sys.exit(0)
if "B70_MTP_PARTIAL_FINAL_GROUP" not in s:
    sys.exit("refusing: patch_mtp_boundary_032.py not applied yet")
for a in (A_EXP,):
    if s.count(a) != 1:
        sys.exit(f"refusing: anchor found {s.count(a)} times (want 1): {a.strip()[:80]!r}")
# first top-level 'import torch' line only
lines = s.split("\n")
try:
    i = lines.index("import torch")
except ValueError:
    sys.exit("refusing: no top-level 'import torch' line")
lines[i] = N_IMPORT.rstrip("\n")
t = "\n".join(lines).replace(A_EXP, N_EXP)
bak = p.with_name(p.name + ".bak-2026-10-21-pre-gdn-varlen")
if not bak.exists():
    shutil.copy2(p, bak)
p.write_text(t, newline="\n")
try:
    py_compile.compile(str(p), doraise=True)
except py_compile.PyCompileError as e:
    p.write_text(s, newline="\n")
    sys.exit(f"compile failed, original restored: {e}")
print(f"patched {p} (backup {bak.name})")
