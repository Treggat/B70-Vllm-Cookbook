"""vLLM 0.30/0.31 XPU: getMemoryInfo from vllm-xpu-kernels reports 0 GiB free on some Level Zero driver/container combinations (torch's
own query says 31.7 GiB), so the startup check refuses to run. Fall back to torch.xpu.mem_get_info when the op answers
zero. Idempotent; aborts if the anchor moved."""
import sys
import os
# Target tree: the vllm/ package dir of the install you are building, e.g. <venv>/src/vllm (editable install).
# Deliberately explicit: patch scripts that "find every vLLM on the box" end up editing trees you did not mean to.
ROOT = os.environ.get("VLLM_ROOT") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not ROOT or not os.path.isfile(os.path.join(ROOT, "__init__.py")):
    sys.exit("usage: VLLM_ROOT=/path/to/vllm python " + os.path.basename(__file__) + "  (the vllm package dir)")
P = os.path.join(ROOT, "platforms/xpu.py")
MARK = "B70_MEMINFO_FALLBACK"
t = open(P).read()
if MARK in t:
    print("meminfo fallback already applied"); sys.exit(0)
old = "    free, total = torch.ops._C_cache_ops.getMemoryInfo(device)\n\n    return free, total\n"
new = ("    free, total = torch.ops._C_cache_ops.getMemoryInfo(device)\n"
       "    if free <= 0:  # " + MARK + ": kernels op reads 0 free under this L0 driver; torch's query is right\n"
       "        free, total = torch.xpu.mem_get_info(device)\n"
       "\n    return free, total\n")
if t.count(old) != 1:
    sys.exit("meminfo anchor not found in xpu.py; not patched")
t = t.replace(old, new, 1)
compile(t, P, "exec")
open(P, "w").write(t)
print("meminfo fallback applied to", P)