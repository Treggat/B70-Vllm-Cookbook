"""crossing_repeat.py: bench/crossing.py's prompt and check, sampled (temperature 1.0) only, R times, printing where each
wrong reply goes wrong. A reply cut by max_tokens mid-line is not counted as wrong.
usage (from this folder): python crossing_repeat.py ../../../bench <R> <tag>      (endpoint/model from env, see bench/common.py)"""
import re, sys, json, os
sys.path.insert(0, sys.argv[1])
from common import complete
N, R, tag = 4000, int(sys.argv[2]), sys.argv[3]
seed = "\n".join(f"def func_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(6))
prompt = ("Continue this Python file with the same pattern up to and including func_399 "
          "(func_i returns x * i + (i % 7)). Output the whole file, code only, no commentary, no code fences.\n\n" + seed)
def check(text):
    ok = 0
    for m in re.finditer(r"def func_(\d+)\(x\):\n    return (.*)\n", text):
        i = int(m.group(1))
        if i != ok or m.group(2).strip() != f"x * {i} + {i % 7}":
            return ok, m.start()
        ok += 1
    return ok, None
for r in range(R):
    c, u = complete(prompt, N, 1.0)
    t = c["message"].get("content") or ""
    ok, pos = check(t)
    print(f"{tag} run {r}: gen {u['completion_tokens']} finish={c['finish_reason']} correct {ok} {'FAIL' if pos is not None else 'ok'}", flush=True)
    if pos is not None:
        print("   context:", repr(t[max(0,pos-80):pos+60]), flush=True)
