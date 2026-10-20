"""How badly does one long cold prefill starve another chat's decode, per chunked-prefill setting?
A = a streaming code generation (2,500 tokens). Once A has decoded for ~6 s, B = a cold ~83k-token prompt arrives
(unique first line, max 32 tokens). Measures A's decode rate before B, while B prefills, and after; B's prefill time; and
A's longest gap between two deltas. Run once per serve config (test.env/test.extra + use-test, see serve/); writes
starve_<tag>.json. Results in docs/RESULTS.md.
Usage: python starve.py <tag>      (endpoint/model from env, see common.py)"""
import json, os, sys, threading, time, urllib.request
from common import BASE, HDR, MODEL, OFF, stdlib_mix

TAG = sys.argv[1]
URL = BASE + "/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = stdlib_mix(["typing.py", "argparse.py", "inspect.py", "subprocess.py", "dataclasses.py", "functools.py", "logging/__init__.py",
                  "shutil.py", "tarfile.py", "zipfile/__init__.py", "email/_header_value_parser.py", "asyncio/base_events.py",
                  "collections/__init__.py", "enum.py", "json/decoder.py", "http/client.py", "urllib/request.py", "unittest/mock.py"])
BIG = SRC[: int(100_000 * 3.6)]


def stream(content, n, marks):
    """POST a streaming chat request; append (time, chars_so_far) per content delta to marks. Returns usage."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0.7, "seed": 5,
            "stream": True, "stream_options": {"include_usage": True}, **OFF}
    chars = 0; usage = {}
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), HDR), timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                t = c.get("delta", {}).get("content")
                if t:
                    chars += len(t); marks.append((time.time(), chars))
    return usage


for _i in range(60):  # loaded + warm: retry while a supervisor is still starting the server
    try:
        stream("hi", 4, []); break
    except Exception as e:
        print("warm retry:", repr(e)[:80], flush=True); time.sleep(20)
else:
    sys.exit("seat never came up")
A = []; B = []; res = {}
ta = threading.Thread(target=lambda: res.__setitem__("A", stream(
    "Write a complete, long Python module implementing an in-memory SQL-like query engine with a tokenizer, parser, "
    "planner and executor, plus pytest tests. Code only.", 2500, A)))
ta.start()
while not A or time.time() - A[0][0] < 6:
    time.sleep(0.2)
tb0 = time.time()
tb = threading.Thread(target=lambda: res.__setitem__("B", stream(f"# {time.time_ns()}\n```python\n{BIG}\n```\nName one function above.", 32, B)))
tb.start(); tb.join(); ta.join()
tb1 = B[0][0] if B else time.time()
cpt = A[-1][1] / max(1, res["A"].get("completion_tokens", 1))   # chars per token of A's own output


def rate(t0, t1):
    pts = [(t, c) for t, c in A if t0 <= t <= t1]
    if len(pts) < 2:
        return 0.0
    return round((pts[-1][1] - pts[0][1]) / cpt / (pts[-1][0] - pts[0][0]), 1)


out = dict(tag=TAG, b_prompt=res["B"].get("prompt_tokens"), b_prefill_s=round(tb1 - tb0, 1),
           b_prefill_tps=round(res["B"].get("prompt_tokens", 0) / max(0.1, tb1 - tb0)),
           a_before=rate(A[0][0], tb0), a_during=rate(tb0, tb1), a_after=rate(tb1, A[-1][0]),
           a_tokens=res["A"].get("completion_tokens"),
           a_longest_gap_s=round(max((A[i + 1][0] - A[i][0]) for i in range(len(A) - 1) if tb0 <= A[i][0] <= tb1) if any(tb0 <= t <= tb1 for t, _ in A[:-1]) else 0, 1))
json.dump(out, open(os.path.join(HERE, f"starve_{TAG}.json"), "w"), indent=1)
print(out)
