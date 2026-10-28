"""The 709-L recipe's crossing.py, adapted for Nemotron: a long, fully predictable reply (func_6..func_399) that
crosses many KV / Mamba-state block edges, so the first wrong line is easy to find. Nemotron continues after the seed
(func_6) instead of repeating it, so the order check starts at the first index found.

Note: Nemotron itself slips at func_95 ("+ 3" instead of "+ 4", 95 mod 7 = 4) with or without speculation, so compare
a change against a run without it instead of reading FAIL as corruption (see ../docs/PITFALLS.md).
Usage: python crossing_nemo.py [max_tokens=12000] [runs=6]   (BASE_URL / MODEL from env, see
../../qwen38-709l/bench/common.py)"""
import os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "qwen38-709l", "bench"))
from common import complete

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
seed = "\n".join(f"def func_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(6))
prompt = ("Continue this Python file with the same pattern up to and including func_399 "
          "(func_i returns x * i + (i % 7)). Output the whole file, code only, no commentary, no code fences.\n\n" + seed)


def check(text):
    first = re.search(r"def func_(\d+)\(x\):", text)
    start = ok = int(first.group(1)) if first else 0
    for m in re.finditer(r"def func_(\d+)\(x\):\n    return (.*)", text):
        i = int(m.group(1))
        if i != ok or m.group(2).strip() != f"x * {i} + {i % 7}":
            return start, ok, m.group(0)
        ok += 1
    return start, ok, None


fails = 0
for k in range(RUNS):
    temp = 0.0 if k == 0 else 1.0
    c, u = complete(prompt, N, temp)
    t = c["message"].get("content") or ""
    start, ok, bad = check(t)
    done = ok == 400
    fails += bad is not None or not done
    print(f"temp {temp}: gen {u['completion_tokens']} finish={c['finish_reason']} | funcs {start}..{ok - 1} in order"
          f"{'' if done else ' (did not reach func_399)'} | first bad: {bad!r}", flush=True)
print(f"{'OK' if not fails else 'FAIL'} ({fails}/{RUNS})")
