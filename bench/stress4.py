"""Four long chats at once: does the KV pool hold them without preemption or OOM?
Each request is a distinct ~TOK-token slice of stdlib source (different offsets + unique first line, so no prefix
sharing) plus a short question. Samples /metrics every 5 s: KV usage, running/waiting, preemptions.
Note: vLLM prefills one long prompt at a time, so the others show as "waiting" during prefill; that is not a KV problem.
Usage: python stress4.py <tag> [tokens_per_chat=95000]"""
import json, sys, threading, time
from common import complete, gauge, stdlib_mix

TAG = sys.argv[1]
TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 95000
SRC = stdlib_mix(["typing.py", "argparse.py", "inspect.py", "subprocess.py", "dataclasses.py", "functools.py", "logging/__init__.py",
                  "shutil.py", "tarfile.py", "zipfile/__init__.py", "email/_header_value_parser.py", "asyncio/base_events.py",
                  "collections/__init__.py", "enum.py", "json/decoder.py", "http/client.py", "urllib/request.py", "unittest/mock.py",
                  "pydoc.py", "configparser.py", "datetime.py", "_pydecimal.py", "pickle.py", "ast.py", "doctest.py", "difflib.py",
                  "smtplib.py", "imaplib.py", "ftplib.py", "turtle.py", "statistics.py", "fractions.py"])
N = int(TOK * 4.25)


def ask(i, res):
    off = (i * len(SRC) // 5) % max(1, len(SRC) - N)
    q = (f"# chat {i} {time.time_ns()}\n```python\n{SRC[off: off + N]}\n```\n\n"
         "Name three functions defined in the code above and say in one sentence what each does.")
    t0 = time.time()
    try:
        c, u = complete(q, 250)
        res[i] = dict(ok=True, sec=round(time.time() - t0, 1), prompt=u["prompt_tokens"], gen=u["completion_tokens"],
                      text=c["message"]["content"])
    except Exception as e:
        res[i] = dict(ok=False, sec=round(time.time() - t0, 1), err=repr(e)[:300])


def sample():
    return dict(kv=gauge("kv_cache_usage_perc"), running=gauge("num_requests_running"), waiting=gauge("num_requests_waiting"),
                preempt=gauge("num_preemptions_total"))


complete("hi", 4)  # warm
m0 = sample(); res = [None] * 4; samples = []
th = [threading.Thread(target=ask, args=(i, res)) for i in range(4)]
t0 = time.time(); [t.start() for t in th]
while any(t.is_alive() for t in th):
    s = sample(); s["t"] = round(time.time() - t0); samples.append(s)
    time.sleep(5)
[t.join() for t in th]
m1 = sample()
out = dict(tag=TAG, tokens_per_chat=TOK, wall=round(time.time() - t0, 1), results=res, kv_peak=max(s["kv"] for s in samples),
           max_running=max(s["running"] for s in samples), max_waiting=max(s["waiting"] for s in samples),
           preemptions=m1["preempt"] - m0["preempt"], samples=samples)
json.dump(out, open(f"stress4_{TAG}.json", "w"), indent=1)
for i, r in enumerate(res):
    print(i, {k: v for k, v in r.items() if k != "text"})
print("wall", out["wall"], "kv_peak", out["kv_peak"], "preemptions", out["preemptions"])
