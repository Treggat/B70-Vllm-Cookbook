"""Does a reply that crosses a KV/state page boundary get corrupted? Short prompt, long fully predictable reply
(func_0..func_399), so the first wrong line is easy to find. Run it after ANY change to MTP, prefix caching or the
mamba/GDN cache flags: this is the bug that silently truncated long replies on older XPU builds (see docs/PITFALLS.md).
Usage: python crossing.py [max_tokens=4000]      (endpoint/model from env, see common.py)"""
import re, sys
from common import complete

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
seed = "\n".join(f"def func_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(6))
prompt = ("Continue this Python file with the same pattern up to and including func_399 "
          "(func_i returns x * i + (i % 7)). Output the whole file, code only, no commentary, no code fences.\n\n" + seed)


def check(text):
    ok = 0
    for m in re.finditer(r"def func_(\d+)\(x\):\n    return (.*)", text):
        i = int(m.group(1))
        if i != ok or m.group(2).strip() != f"x * {i} + {i % 7}":
            return ok, m.group(0)
        ok += 1
    return ok, None


fails = 0
for temp in (0.0, 1.0, 1.0):
    c, u = complete(prompt, N, temp)
    t = c["message"].get("content") or ""
    ok, bad = check(t)
    fails += bad is not None
    print(f"temp {temp}: prompt {u['prompt_tokens']} gen {u['completion_tokens']} finish={c['finish_reason']} | "
          f"correct funcs in order: {ok} | first bad: {bad!r}")
print("PASS" if not fails else f"FAIL ({fails}/3)")
