#!/usr/bin/env python3
"""Optional fp8 lm_head for Nemotron (NemotronH) on vLLM 0.32 XPU, switched on with B70_TARGET_LMHEAD_FP8=1.

Installs riceharvest's helper (b70_target_lmhead_fp8.py beside this file, MIT, LICENSE-riceharvest) as
vllm/model_executor/models/b70_target_lmhead_fp8.py and hooks two files:
  model_executor/layers/logits_processor.py   _apply_head: use the fp8 head when one was built
  model_executor/models/nemotron_h.py         NemotronHForCausalLM.load_weights: build it after loading
At load the BF16 head (131,072 x 2,688) is converted to per-row fp8 e4m3 and freed; logits then run through the XPU
fp8_gemm_w8a16 op. Measured 2026-10-27 with MTP-3: +11 % decode at 1 stream, +5 % at 16, more KV pool.
Unset, the hooks do nothing. Target: $VLLM_ROOT (the vllm/ package dir) or argv[1].
Idempotent (marker), fail-closed (exact anchors), .bak per edited file, py_compile of every touched file.
"""
import os, pathlib, py_compile, shutil, sys

HERE = pathlib.Path(__file__).resolve().parent
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VLLM_ROOT", ""))
if not (root / "__init__.py").is_file() or not (root / "model_executor").is_dir():
    sys.exit(f"refusing: {root} is not a vllm/ package dir (set VLLM_ROOT)")
MARK = "_b70_fp8"
BAK = ".bak-pre-lmhead-fp8"

helper = root / "model_executor" / "models" / "b70_target_lmhead_fp8.py"
shutil.copy2(HERE / "b70_target_lmhead_fp8.py", helper)
print("installed", helper)

EDITS = {
    "model_executor/layers/logits_processor.py": (
        '        """Project hidden states through the lm_head, honoring head_dtype."""\n',
        '        """Project hidden states through the lm_head, honoring head_dtype."""\n'
        '        if getattr(lm_head, "_b70_fp8", None) is not None:\n'
        '            from vllm.model_executor.models.b70_target_lmhead_fp8 import logits as _b70_lg\n'
        '            return _b70_lg(lm_head, hidden_states)\n'),
    "model_executor/models/nemotron_h.py": (
        "        loader = AutoWeightsLoader(self)\n"
        "        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n",
        "        loader = AutoWeightsLoader(self)\n"
        "        _b70_loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
        "        # optional fp8 lm_head (B70_TARGET_LMHEAD_FP8=1), riceharvest helper\n"
        "        from vllm.model_executor.models.b70_target_lmhead_fp8 import convert as _b70_cv\n"
        "        _b70_cv(self)\n"
        "        return _b70_loaded\n"),
}
for rel, (anchor, new) in EDITS.items():
    p = root / rel
    s = p.read_text()
    if ("_b70_cv" if rel.endswith("nemotron_h.py") else MARK) in s:
        print("already patched:", p)
        continue
    if s.count(anchor) != 1:
        sys.exit(f"refusing: {rel}: anchor found {s.count(anchor)} times (want 1)")
    bak = p.with_name(p.name + BAK)
    if not bak.exists():
        shutil.copy2(p, bak)
    p.write_text(s.replace(anchor, new), newline="\n")
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as e:
        p.write_text(s, newline="\n")
        sys.exit(f"compile failed in {rel}, original restored: {e}")
    print("patched", p)
py_compile.compile(str(helper), doraise=True)
print("LMHEAD_FP8_PATCH_DONE")
