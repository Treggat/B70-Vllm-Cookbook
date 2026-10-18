"""vLLM 0.30/0.31: Qwen3_5MultiTokenPredictor builds `fc` BEFORE the block where B70_MTP_BF16_DRAFT nulls the quant config,
so fc still comes out quantized and a checkpoint whose mtp.* is BF16 (e.g. the 709-L AWQ) fails with 'no parameter fc.weight'.
Extend the same switch to fc. Idempotent; aborts if the anchor moved."""
import sys
import os
# Target tree: the vllm/ package dir of the install you are building, e.g. <venv>/src/vllm (editable install).
# Deliberately explicit: patch scripts that "find every vLLM on the box" end up editing trees you did not mean to.
ROOT = os.environ.get("VLLM_ROOT") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not ROOT or not os.path.isfile(os.path.join(ROOT, "__init__.py")):
    sys.exit("usage: VLLM_ROOT=/path/to/vllm python " + os.path.basename(__file__) + "  (the vllm package dir)")
P = os.path.join(ROOT, "model_executor/models/qwen3_5_mtp.py")
MARK = "B70_MTP_BF16_FC"
t = open(P).read()
if MARK in t:
    print("fc bf16 draft patch already applied"); sys.exit(0)
old = ("        fc_quant = (\n"
       "            None\n"
       "            if (quant_config and quant_config.get_name() == \"modelopt_fp4\")\n"
       "            else quant_config\n"
       "        )\n")
new = ("        fc_quant = (\n"
       "            None\n"
       "            if (quant_config and quant_config.get_name() == \"modelopt_fp4\")\n"
       "            or os.environ.get(\"B70_MTP_BF16_DRAFT\") == \"1\"  # " + MARK + ": fc is BF16 in the checkpoint too\n"
       "            else quant_config\n"
       "        )\n")
if t.count(old) != 1:
    sys.exit("fc_quant anchor not found in qwen3_5_mtp.py; not patched")
t = t.replace(old, new, 1)
if "\nimport os\n" not in t:
    sys.exit("qwen3_5_mtp.py has no 'import os' (patch_mtp_nightly should have added it); not patched")
compile(t, P, "exec")
open(P, "w").write(t)
print("fc bf16 draft patch applied to", P)