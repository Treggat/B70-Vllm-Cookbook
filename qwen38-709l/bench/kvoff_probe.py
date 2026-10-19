"""Does vLLM's CPU KV offload tier (--kv-offloading-size N) ever hit on this hybrid (GDN + attention) model?
Prompt A (~60k tokens, unique first line) cold, then 7 more distinct ~60k prompts (more than the GPU pool holds, so A is
evicted from the GPU), then A again. If the tier works, the replay reports cached_tokens close to prompt_tokens and
answers in seconds; if not, cached_tokens is 0 and the prompt is prefilled again in full. Needs --enable-prompt-tokens-details
on the server (it is on the serve line). Result on vLLM 0.32 with the 709-L stack: never hits, so serve-709l.sh leaves the
flag out; Vcvzgbzz's cookbook saw the same on 0.31.
Usage: python kvoff_probe.py [tokens_per_prompt=60000]   (endpoint/model from env, see common.py)"""
import json, os, sys, time, urllib.request
from common import BASE, HDR, MODEL, OFF, stdlib_mix

TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
SRC = stdlib_mix(["typing.py", "argparse.py", "inspect.py", "subprocess.py", "dataclasses.py", "functools.py", "logging/__init__.py",
                  "shutil.py", "tarfile.py", "zipfile/__init__.py", "email/_header_value_parser.py", "asyncio/base_events.py",
                  "collections/__init__.py", "enum.py", "json/decoder.py", "http/client.py", "urllib/request.py", "unittest/mock.py",
                  "pydoc.py", "configparser.py", "datetime.py", "_pydecimal.py", "pickle.py", "ast.py", "doctest.py", "difflib.py",
                  "smtplib.py", "imaplib.py", "ftplib.py", "turtle.py", "statistics.py", "fractions.py", "threading.py", "socket.py",
                  "ssl.py", "os.py", "tokenize.py", "textwrap.py", "random.py", "heapq.py"])
N = int(TOK * 4.0)
NONCE = time.time_ns()


def ask(i):
    off = (i * N) % max(1, len(SRC) - N)
    body = {"model": MODEL, "max_tokens": 4, "temperature": 0, **OFF,
            "messages": [{"role": "user", "content": f"# probe {NONCE} slice {i}\n```python\n{SRC[off:off + N]}\n```\nSay ok."}]}
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), HDR), timeout=1800))
    u = d.get("usage", {})
    r = {"slice": i, "secs": round(time.time() - t, 1), "prompt_tokens": u.get("prompt_tokens"),
         "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
    print(r, flush=True)
    return r


out = {"A_cold": ask(0), "fill": [ask(i) for i in range(1, 8)], "A_replay": None}
out["A_replay"] = ask(0)
c = out["A_replay"]["cached_tokens"]
out["verdict"] = ("offload tier HITS (keep --kv-offloading-size)" if c and c > 0.5 * out["A_replay"]["prompt_tokens"]
                  else "no hit after eviction: the offload tier never finds the blocks (drop the flag)")
print(out["verdict"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kvoff_probe.json"), "w"), indent=1)
