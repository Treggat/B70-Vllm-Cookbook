#!/usr/bin/env python3
"""Install / remove the B70 shortlist draft head on a vLLM 0.32 709-L tree (patches/shortlist).

- copies b70_shortlist_head.py (next to this file) to vllm/model_executor/models/
- patches vllm/v1/worker/gpu/spec_decode/speculator.py: DraftModelSpeculator.sample_draft returns the shortlist
  head's greedy draft when B70_DRAFT_SHORTLIST is set at start-up, drafts are greedy (draft_logits is None) and no
  acceptance estimator reads the logits; with probabilistic drafts (draft_logits set, revision 2 of 2026-10-22) it
  samples with the stock sampler from the shortlist's scattered logits row, which the sampler also caches as q
  (B70_DRAFT_SHORTLIST_PROB=0: stock full-head path); DraftModelSpeculator.load_model builds the shortlist right
  after the drafter is loaded (before warm-up and graph capture). Unset = the import never happens and the stock
  code runs. A tree with the revision-1 hook (greedy only, our live seat on 2026-10-22) is upgraded in place.
Target: VLLM_ROOT (the vllm/ package dir) or argv[1]. Fail-closed (exact anchors), idempotent (marker), backups
<file>.bak-2026-10-22-pre-shortlist, py_compile. --restore puts the backups back and removes the module.
"""
import os, py_compile, shutil, sys
from pathlib import Path

MARK = "B70_DRAFT_SHORTLIST"
TAG = ".bak-2026-10-22-pre-shortlist"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
root = Path(args[0] if args else os.environ.get("VLLM_ROOT", ""))
spec = root / "v1/worker/gpu/spec_decode/speculator.py"
mod_dst = root / "model_executor/models/b70_shortlist_head.py"
mod_src = Path(__file__).resolve().parent / "b70_shortlist_head.py"

if not spec.is_file():
    sys.exit(f"{MARK}: {spec} not found (VLLM_ROOT?)")

if "--restore" in sys.argv:
    bak = Path(str(spec) + TAG)
    if bak.is_file():
        shutil.copy2(bak, spec); bak.unlink(); print("restored", spec)
    else:
        print("no backup, speculator.py left as is:", "patched" if MARK in spec.read_text() else "stock")
    if mod_dst.is_file():
        mod_dst.unlink(); print("removed", mod_dst)
    sys.exit(0)

A1 = "class DraftModelSpeculator(BaseSpeculator):\n"
ADD1 = '''# B70_DRAFT_SHORTLIST (patches/shortlist): read once at import; unset = stock sample_draft
import os as _b70_os
_B70_SHORTLIST = bool(_b70_os.environ.get("B70_DRAFT_SHORTLIST"))


'''
A2 = ("        if draft_logits is None and self.use_local_argmax_reduction:\n"
      "            return self.get_draft_top_tokens(hidden_states, spec_step_idx)\n")
ADD2_V1 = '''        if _B70_SHORTLIST and draft_logits is None and self.acceptance_estimator is None:
            from vllm.model_executor.models import b70_shortlist_head as _b70_sl
            if _b70_sl.active(self.model, hidden_states.device):
                return _b70_sl.top_tokens(self.model, hidden_states)
'''
MARK2 = "_b70_sl.logits_row("
ADD2 = '''        if _B70_SHORTLIST and self.acceptance_estimator is None:
            from vllm.model_executor.models import b70_shortlist_head as _b70_sl
            if _b70_sl.active(self.model, hidden_states.device):
                if draft_logits is None:
                    return _b70_sl.top_tokens(self.model, hidden_states)
                if _b70_sl.prob_on():
                    _b70_lg = _b70_sl.logits_row(
                        self.model, hidden_states, draft_logits.shape[-1], draft_logits.dtype
                    )
                    _b70_sampler = (
                        gumbel_sample
                        if self.draft_watermarker is None
                        else self.draft_watermarker.sample
                    )
                    return _b70_sampler(
                        _b70_lg,
                        idx_mapping,
                        temperature,
                        seeds,
                        sample_src_positions,
                        apply_temperature=True,
                        is_drafting=True,
                        logits_cache=draft_logits,
                        logits_cache_col=draft_step,
                        use_fp64=self.use_fp64_gumbel,
                    )
'''

# eager init right after the drafter is loaded (the target has already prebuilt the int4 draft head on the shared
# lm_head by then), so the artifact's host->device copies never happen inside a graph capture
A3 = "        self.model = self.load_draft_model(target_model, target_attn_layer_names)\n"
ADD3 = '''        if _B70_SHORTLIST:
            from vllm.model_executor.models import b70_shortlist_head as _b70_sl
            _b70_sl.active(self.model, self.device)
'''

s = spec.read_text()
if MARK in s and MARK2 in s:
    print("already patched", spec)
elif MARK in s:
    if s.count(ADD2_V1) != 1:
        sys.exit(f"{MARK}: revision-1 hook not found exactly once in {spec}; not upgraded")
    bak = Path(str(spec) + TAG)
    if not bak.exists():
        sys.exit(f"{MARK}: patched tree without {bak.name}; not upgraded (restore needs the stock file)")
    shutil.copy2(spec, Path(str(spec) + ".bak-2026-10-22-pre-shortlist-prob"))
    spec.write_text(s.replace(ADD2_V1, ADD2, 1))
    try:
        py_compile.compile(str(spec), doraise=True)
    except py_compile.PyCompileError as e:
        spec.write_text(s)
        sys.exit(f"{MARK}: py_compile failed, revision 1 put back: {e}")
    print("upgraded revision-1 hook", spec)
else:
    for a in (A1, A2, A3):
        if s.count(a) != 1:
            sys.exit(f"{MARK}: anchor count {s.count(a)} != 1 in {spec}; not patched:\n{a}")
    new = s.replace(A1, ADD1 + A1, 1).replace(A2, ADD2 + A2, 1).replace(A3, A3 + ADD3, 1)
    bak = Path(str(spec) + TAG)
    if not bak.exists():
        shutil.copy2(spec, bak)
    spec.write_text(new)
    try:
        py_compile.compile(str(spec), doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copy2(bak, spec)
        sys.exit(f"{MARK}: py_compile failed, restored: {e}")
    print("patched", spec)

if not mod_src.is_file():
    sys.exit(f"{MARK}: module {mod_src} missing")
shutil.copy2(mod_src, mod_dst)
py_compile.compile(str(mod_dst), doraise=True)
print("installed", mod_dst)
