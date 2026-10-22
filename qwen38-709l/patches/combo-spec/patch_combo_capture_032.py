#!/usr/bin/env python3
"""2026-10-21 B70_COMBO_CAPTURE_KS: capture XPU graphs for extra draft counts (vLLM 0.32, 709L combined speculator).

Dynamic SD captures one decode graph set per K in its batch-size schedule, so with max_num_seqs 4 at most 4 K values.
The depth tiers (b70_combo_policy B70_COMBO_MTP_TIERS / B70_COMBO_COPY_TIERS) use more: e.g. MTP 2-5 and copy 10/15.
Without graphs those steps would run eagerly. B70_COMBO_CAPTURE_KS="2,3,4,5,10,15" adds these K values to:
  v1/worker/gpu/cudagraph_utils.py                               target + MTP-prefill decode graphs (q_len K+1)
  v1/worker/gpu/spec_decode/target_dependent_ar/cudagraph_utils.py  MTP multi-step decode specializations
Unset = stock. Requires a Dynamic SD schedule in the SPEC (that is what turns on the per-K capture path).
Target: $VLLM_ROOT (vllm/ package dir) or argv[1]. Idempotent, fail-closed, .bak, py_compile.
"""
import os, pathlib, py_compile, shutil, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VLLM_ROOT", ""))
MARK = "B70_COMBO_CAPTURE_KS"
BAK = ".bak-2026-10-21-pre-capture-ks"
IMP = "from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup\n"
DEF = (IMP + "\n# B70_COMBO_CAPTURE_KS: extra draft counts to capture graphs for (unset = stock).\n"
       "_B70_CAPTURE_KS = [\n"
       "    int(_k) for _k in __import__(\"os\").environ.get(\"B70_COMBO_CAPTURE_KS\", \"\").split(\",\") if _k.strip()\n"
       "]\n")
EDITS = {
    "v1/worker/gpu/cudagraph_utils.py": [
        (IMP, DEF),
        ("                    for num_spec in dense_schedule[1:]\n",
         "                    for num_spec in [*dense_schedule[1:], *_B70_CAPTURE_KS]\n"),
    ],
    "v1/worker/gpu/spec_decode/target_dependent_ar/cudagraph_utils.py": [
        (IMP, DEF),
        ("                    for k in sorted(set(dense_schedule) - {0, 1}, reverse=True)\n",
         "                    for k in sorted(\n"
         "                        (set(dense_schedule) | set(_B70_CAPTURE_KS)) - {0, 1}, reverse=True\n"
         "                    )\n"),
    ],
}
if not (root / "__init__.py").is_file():
    sys.exit(f"refusing: {root} is not a vllm/ package dir (set VLLM_ROOT)")
plan = []
for rel, pairs in EDITS.items():
    p = root / rel
    s = p.read_text()
    if MARK in s:
        print(f"already patched: {p}")
        continue
    t = s
    for a, n in pairs:
        if t.count(a) != 1:
            sys.exit(f"refusing: {rel}: anchor found {t.count(a)} times (want 1): {a.strip()[:80]!r}")
        t = t.replace(a, n)
    plan.append((p, s, t))
for p, s, t in plan:  # all anchors checked first: all files or none
    bak = p.with_name(p.name + BAK)
    if not bak.exists():
        shutil.copy2(p, bak)
    p.write_text(t, newline="\n")
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as e:
        for q, s0, _ in plan:
            q.write_text(s0, newline="\n")
        sys.exit(f"compile failed in {p}, originals restored: {e}")
    print(f"patched {p}")
print("CAPTURE_PATCH_DONE")
