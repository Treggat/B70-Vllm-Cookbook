"""Speed + sanity bench for one serve config. Run it once per config, same day, same box, then compare the JSONs.
Usage: python bench.py <tag> [tests]      tests = comma list of code,four,depth,greedy (default all)
  code   = 3 x short code generation (700 tokens), decode t/s + MTP acceptance
  four   = 4 different writers at once (800 tokens each): per-stream t/s, summed, total-gen / wall
  depth  = one cold read at 24k/32k/65k/74k/100k tokens of stdlib source (unique first line, so no prefix-cache hit):
           prefill t/s, decode t/s, and a retrieval check (how many argparse helper names the answer cites)
  greedy = temperature-0 replies to fixed prompts; compare their sha1 across configs (lossless check)
Every reply also goes through a corruption canary ('!!!!' runs, replacement chars, loops). Writes bench_<tag>.json."""
import hashlib, json, sys, threading, time
from common import accept, canary, metrics, stdlib_mix, stream

TAG = sys.argv[1]
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"code", "four", "depth", "greedy"}
out = {"tag": TAG, "when": time.strftime("%Y-%m-%d %H:%M")}
stream("Say hi.", 16)  # warm-up

if "code" in ONLY:
    CODE = "Write a complete Python module implementing an LRU cache class with get, put, delete and a pytest section. Code only, no commentary."
    m0 = metrics()
    rs = [stream(CODE, 700, seed=7 + i) for i in range(3)]
    out["short_code"] = dict(tps=[round(r["tps"], 1) for r in rs], avg=round(sum(r["tps"] for r in rs) / 3, 1),
                             canary=[canary(r["text"]) for r in rs], mtp=accept(m0, metrics()))
    print("short code:", out["short_code"], flush=True)

if "four" in ONLY:
    W = ["Write a detailed essay on the history of the printing press. No headings.",
         "Write a complete Python module for a thread-safe token-bucket rate limiter with tests. Code only.",
         "Write a short story about a lighthouse keeper who finds a message in a bottle.",
         "Write a complete JavaScript module implementing a debounce and throttle utility with JSDoc. Code only."]
    m0 = metrics(); res = [None] * 4
    th = [threading.Thread(target=lambda i=i: res.__setitem__(i, stream(W[i], 800, seed=11 + i))) for i in range(4)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    out["four_writers"] = dict(each=[round(r["tps"], 1) for r in res], summed=round(sum(r["tps"] for r in res), 1),
                               gen_over_wall=round(sum(r["gen"] for r in res) / wall, 1), wall=round(wall, 1),
                               canary=[canary(r["text"]) for r in res], mtp=accept(m0, metrics()))
    print("4 writers:", out["four_writers"], flush=True)

if "depth" in ONLY:
    # argparse.py goes FIRST so every depth actually contains it. (Our first version put it after typing.py, which pushed it
    # past the 24k/32k slices: at those depths the model answered from memory and the "cites" score was noise.)
    SRC = stdlib_mix(["argparse.py", "typing.py", "inspect.py", "subprocess.py", "dataclasses.py", "functools.py",
                      "logging/__init__.py", "shutil.py", "tarfile.py", "zipfile/__init__.py", "email/_header_value_parser.py",
                      "asyncio/base_events.py", "collections/__init__.py", "enum.py", "json/decoder.py", "http/client.py",
                      "urllib/request.py", "unittest/mock.py", "pydoc.py", "configparser.py", "datetime.py", "_pydecimal.py"])
    NAMES = ["_parse_known_args", "consume_optional", "consume_positionals", "_match_argument"]
    ask = ("\n\nAbove are several Python stdlib files. Write a 300-word technical overview of how argparse.ArgumentParser.parse_known_args "
           "processes optionals and positionals, citing the helper method names it calls.")
    out["depth"] = {}
    for k in (24, 32, 65, 74, 100):
        body = SRC[: int(k * 1000 * 4.25)]   # ~4.25 chars/token on this mix with the Qwen3.8 tokenizer
        body = body[: body.rfind("\n")]
        m0 = metrics()
        r = stream(f"# run {time.time_ns()}\n```python\n" + body + "\n```" + ask, 400, temp=0.3)
        out["depth"][f"{k}k"] = dict(prompt=r["prompt"], cold_ttft=round(r["ttft"], 1), prefill_tps=round(r["prompt"] / r["ttft"]),
                                     decode=round(r["tps"], 1), gen=r["gen"], canary=canary(r["text"]),
                                     cites=sum(x in r["text"] for x in NAMES), names_in_prompt=sum(x in body for x in NAMES),
                                     mtp=accept(m0, metrics()))
        print(f"depth {k}k:", out["depth"][f"{k}k"], flush=True)

if "greedy" in ONLY:
    G = ["Write a Python function that merges overlapping intervals, with a docstring and three doctests. Code only.",
         "Explain in about 200 words why the sky is blue, for a curious ten-year-old.",
         "List the planets of the solar system with one fact each, as a numbered list."]
    out["greedy"] = []
    for g in G:
        r = stream(g, 400, temp=0.0)
        out["greedy"].append(dict(sha=hashlib.sha1(r["text"].encode()).hexdigest()[:12], gen=r["gen"], canary=canary(r["text"])))
    print("greedy:", out["greedy"], flush=True)

json.dump(out, open(f"bench_{TAG}.json", "w"), indent=1)
