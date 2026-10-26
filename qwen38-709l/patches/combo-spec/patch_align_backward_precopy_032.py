#!/usr/bin/env python3
"""2026-10-25 B70_ALIGN_BACKWARD_PRECOPY: fix a stale GDN state after a backward align column move (vLLM 0.32, 709L
combined copy + MTP speculator).

vLLM 0.32 with prefix caching (mamba_cache_mode=align, V2 model runner) keeps a request's running GDN state in block
column (num_computed + query_len - 1) // block_size. With a variable step length (the combined speculator's
16-token copy steps followed by 6-token MTP steps, or num_speculative_tokens_per_batch_size) that column can move
BACKWARDS: a 16-token step from block 3 moves it to 4, few drafts are accepted, the next 6-token step ends in block 3
again. preprocess_mamba_align_fused_kernel then resets num_accepted to 1, but the backward-copy guard that
SergiioB's patch_fix_backward_copy.py adds to precopy_mamba_align_fused_kernel (vllm#53505) returns before copying,
so the GDN layers read their SSM and conv state from the old column: stale by the tokens accepted in the copy step.
On our seat that made about 1 sampled bench/crossing.py reply in 20 go wrong near a 960-token page edge.

With B70_ALIGN_BACKWARD_PRECOPY=1 the precopy also copies backwards: state[bt[src_col + num_accepted - 1]] ->
state[bt[dst_col]], the same copy as a forward move. The destination is always the block holding position
num_computed, never a completed, cached block. Unset, the kernel is compiled with ALLOW_BACKWARD=False, which is
the guarded code unchanged.

Run after scripts/apply_stack.sh (it needs SergiioB's guard in place; stock vLLM has no guard and copies backwards
already). Target: $VLLM_ROOT (the vllm/ package dir) or argv[1]. Idempotent, fail-closed, .bak, py_compile."""
import os, pathlib, py_compile, shutil, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VLLM_ROOT", ""))
p = root / "v1" / "worker" / "mamba_utils.py"
MARK = "B70_ALIGN_BACKWARD_PRECOPY"
R = [
    ("    HAS_IDX_MAPPING: tl.constexpr = True,\n"
     "    # TEMPORAL_TILES: see postprocess_mamba_fused_kernel. Default 1 preserves\n"
     "    # the 2D-grid contract; > 1 requires a 3D grid.\n"
     "    TEMPORAL_TILES: tl.constexpr = 1,\n"
     "):\n"
     '    """Pre-copy mamba "align" state across block boundaries.\n',
     "    HAS_IDX_MAPPING: tl.constexpr = True,\n"
     "    # TEMPORAL_TILES: see postprocess_mamba_fused_kernel. Default 1 preserves\n"
     "    # the 2D-grid contract; > 1 requires a 3D grid.\n"
     "    TEMPORAL_TILES: tl.constexpr = 1,\n"
     "    # B70_ALIGN_BACKWARD_PRECOPY: also migrate the running state when the column moves back (variable q_len).\n"
     "    ALLOW_BACKWARD: tl.constexpr = False,\n"
     "):\n"
     '    """Pre-copy mamba "align" state across block boundaries.\n'),
    ("    if src_col > dst_col:\n        return\n\n    if src_col < 0 or src_col == dst_col:\n",
     "    if not ALLOW_BACKWARD:\n        if src_col > dst_col:\n            return\n\n    if src_col < 0 or src_col == dst_col:\n"),
    ("            HAS_IDX_MAPPING=idx_mapping is not None,\n            TEMPORAL_TILES=_TEMPORAL_TILES,\n        )\n",
     "            HAS_IDX_MAPPING=idx_mapping is not None,\n            TEMPORAL_TILES=_TEMPORAL_TILES,\n"
     "            ALLOW_BACKWARD=_B70_ALIGN_BACKWARD_PRECOPY,\n        )\n"),
    ("@triton.jit(do_not_specialize=[\"num_reqs\"])\ndef precopy_mamba_align_fused_kernel(",
     "_B70_ALIGN_BACKWARD_PRECOPY = __import__(\"os\").environ.get(\"B70_ALIGN_BACKWARD_PRECOPY\") == \"1\"\n\n\n"
     "@triton.jit(do_not_specialize=[\"num_reqs\"])\ndef precopy_mamba_align_fused_kernel("),
]
if not p.is_file():
    sys.exit(f"refusing: {p} not found (set VLLM_ROOT to the vllm/ package dir)")
s = p.read_text()
if MARK in s:
    print(f"already patched: {p}"); sys.exit(0)
if "B70_BACKWARD_COPY_GUARD_PRECOPY" not in s:
    sys.exit("refusing: no backward-copy guard in the precopy kernel (apply scripts/apply_stack.sh first; stock vLLM does not need this)")
for a, _ in R:
    if s.count(a) != 1:
        sys.exit(f"refusing: anchor found {s.count(a)} times: {a[:70]!r}")
t = s
for a, b in R:
    t = t.replace(a, b)
bak = p.with_name(p.name + ".bak-2026-10-25-pre-backward-precopy")
if not bak.exists():
    shutil.copy2(p, bak)
p.write_text(t, newline="\n")
try:
    py_compile.compile(str(p), doraise=True)
except py_compile.PyCompileError as e:
    p.write_text(s, newline="\n"); sys.exit(f"compile failed, restored: {e}")
print(f"patched {p} (backup {bak.name})")
