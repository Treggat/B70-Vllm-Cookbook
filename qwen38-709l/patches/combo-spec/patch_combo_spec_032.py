#!/usr/bin/env python3
"""2026-10-21 B70_COMBO_SPEC: the 709L combined copy + MTP speculator for vLLM 0.32 (design: README.md in this folder).

Installs two new modules and hooks four files. Everything is gated on B70_COMBO_SPEC=1 at runtime: unset, the
hooks are no-ops and the seat runs stock MTP.
  combo/b70_combo_policy.py      -> vllm/v1/core/sched/b70_combo_policy.py       (scheduler: K per step)
  combo/b70_combo_speculator.py  -> vllm/v1/worker/gpu/spec_decode/b70_combo/speculator.py
  scheduler.py                   import the policy; let it pick num_spec_tokens_to_schedule (after Dynamic SD);
                                 feed it each request's verify result
  spec_decode/__init__.py        build B70ComboSpeculator for method "mtp"
  worker/gpu/model_runner.py     keep all_token_ids dense on the GPU (the n-gram scan reads it; UVA = PCIe reads)
Needs at runtime (test.env): SPEC with num_speculative_tokens 15 and a Dynamic SD schedule holding 5 and 15 (graphs
for q_len 6 and 16), B70_GDN_VARLEN_SPEC=6,16 (patch_gdn_varlen_spec_032.py), and the spec_packed q_len guard.
Target: $VLLM_ROOT (vllm/ package dir) or argv[1]. Idempotent (markers; module files re-copied only when they
differ), fail-closed (exact anchors), .bak per edited file, py_compile of every touched file.
"""
import os, pathlib, py_compile, shutil, sys

HERE = pathlib.Path(__file__).resolve().parent
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VLLM_ROOT", ""))
if not (root / "__init__.py").is_file() or not (root / "v1").is_dir():
    sys.exit(f"refusing: {root} is not a vllm/ package dir (set VLLM_ROOT)")
MARK = "B70_COMBO_SPEC"
BAK = ".bak-2026-10-21-pre-combo"
touched = []


def install(src, dst):
    s = (HERE / "combo" / src).read_text()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.read_text() == s:
        print(f"same: {dst}")
    else:
        dst.write_text(s, newline="\n")
        print(f"installed: {dst}")
    touched.append(dst)


def edit(rel, pairs):
    p = root / rel
    if not p.is_file():
        sys.exit(f"refusing: {p} not found")
    s = p.read_text()
    if MARK in s:
        print(f"already patched: {p}")
        touched.append(p)
        return
    t = s
    for a, n in pairs:
        if t.count(a) != 1:
            sys.exit(f"refusing: {rel}: anchor found {t.count(a)} times (want 1): {a.strip()[:90]!r}")
        t = t.replace(a, n)
    bak = p.with_name(p.name + BAK)
    if not bak.exists():
        shutil.copy2(p, bak)
    p.write_text(t, newline="\n")
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as e:
        p.write_text(s, newline="\n")
        sys.exit(f"compile failed in {rel}, original restored: {e}")
    print(f"patched {p}")
    touched.append(p)


install("b70_combo_policy.py", root / "v1/core/sched/b70_combo_policy.py")
sd = root / "v1/worker/gpu/spec_decode/b70_combo"
install("b70_combo_speculator.py", sd / "speculator.py")
if not (sd / "__init__.py").exists():
    (sd / "__init__.py").write_text("", newline="\n")

IMP = "from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup\n"
DSD = ("            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
       "                len(num_scheduled_tokens)\n"
       "            ]\n")
REJ = "                num_rejected = num_draft_tokens - num_accepted\n"
edit("v1/core/sched/scheduler.py", [
    (IMP, IMP + "\n"
          "# B70_COMBO_SPEC: per-step copy / MTP draft-count policy (unset = stock).\n"
          "_B70_COMBO = None\n"
          "if __import__(\"os\").environ.get(\"B70_COMBO_SPEC\") == \"1\":\n"
          "    from vllm.v1.core.sched import b70_combo_policy as _B70_COMBO\n"),
    (DSD, DSD +
          "        if _B70_COMBO is not None and len(num_scheduled_tokens) > 0:\n"
          "            num_spec_tokens_to_schedule = _B70_COMBO.choose_k(\n"
          "                self, num_scheduled_tokens, num_spec_tokens_to_schedule\n"
          "            )\n"),
    (REJ, REJ +
          "                if _B70_COMBO is not None:\n"
          "                    _B70_COMBO.observe(\n"
          "                        request, num_draft_tokens, num_accepted, self.num_spec_tokens\n"
          "                    )\n"),
])

GEM = "    elif speculative_config.use_gemma4_mtp():\n"
edit("v1/worker/gpu/spec_decode/__init__.py", [
    (GEM,
     "    elif (\n"
     "        __import__(\"os\").environ.get(\"B70_COMBO_SPEC\") == \"1\"\n"
     "        and speculative_config.method == \"mtp\"\n"
     "        and not speculative_config.use_multi_module_mtp()\n"
     "    ):\n"
     "        from vllm.v1.worker.gpu.spec_decode.b70_combo.speculator import (\n"
     "            B70ComboSpeculator,\n"
     "        )\n"
     "\n"
     "        return B70ComboSpeculator(vllm_config, device, req_states)\n" + GEM),
])

DENSE = "            self.speculative_config is not None and self.speculative_config.use_ngram()\n"
edit("v1/worker/gpu/model_runner.py", [
    (DENSE,
     "            self.speculative_config is not None and self.speculative_config.use_ngram()\n"
     "        ) or (  # B70_COMBO_SPEC: the combined speculator's n-gram scan reads this every copy step\n"
     "            __import__(\"os\").environ.get(\"B70_COMBO_SPEC\") == \"1\"\n"),
])

for f in touched:
    py_compile.compile(str(f), doraise=True)
print("COMBO_PATCH_DONE")
