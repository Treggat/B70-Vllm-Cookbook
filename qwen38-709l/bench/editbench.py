# editbench.py <tag>: whole-file EDIT/REWRITE decode t/s through $BASE_URL (hearth), thinking off.
# E1 = Vcvzgbzz eval/bench.py's edit prompt (dd083d5): llama.cpp tools/server/tests/unit/test_stream.py, rename `server` -> `srv`.
# E2 = gguf-py/gguf/utility.py (~3.5k tok): add a `timeout` parameter to SafetensorRemote.get_data_by_range.
# E3 = scripts/server-bench.py (~3.5k tok) with one injected bug: fix it.
# Each task twice: run g = greedy (temperature 0), run s = server-default sampling. Correctness = the returned code block
# vs the expected file (exact match, difflib ratio). Files from G:\AI\Projects\llama.cpp-b70-patches (14964cee3).
import difflib, json, os, re, sys, time, urllib.request
URL = os.environ["BASE_URL"].rstrip("/") + "/chat/completions"
MODEL = os.environ.get("MODEL", "LLM:Fable-709L")
HDRS = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["API_KEY"]}
L = os.environ.get("LLAMACPP_TREE", r"G:\AI\Projects\llama.cpp-b70-patches")  # a llama.cpp checkout at 14964cee3
rd = lambda p: open(os.path.join(L, p), encoding="utf-8").read()

f1 = rd(r"tools\server\tests\unit\test_stream.py")
f2 = rd(r"gguf-py\gguf\utility.py")
f3 = rd(r"scripts\server-bench.py")
TAIL = " Return the COMPLETE updated file in one code block and nothing else."
A2 = "    def get_data_by_range(cls, url: str, start: int, size: int = -1) -> bytes:"
B2 = "        response = requests.get(url, allow_redirects=True, headers=headers)\n        response.raise_for_status()\n\n        # Get raw byte data"
GOOD3 = "        ret.append(random.randint(prompt_length_min, prompt_length_max))"
BAD3 = "        ret.append(random.randint(prompt_length_min, prompt_length_min))"
assert f2.count(A2) == 1 and f2.count(B2) == 1 and f3.count(GOOD3) == 1
TASKS = [
    ("E1 rename", "Here is a Python test file:\n\n```python\n" + f1 + "\n```\n\nRename the module-level variable `server` to `srv` "
     "everywhere in this file." + TAIL, re.sub(r"\bserver\b", "srv", f1), 3000),
    ("E2 param", "Here is a Python module:\n\n```python\n" + f2 + "\n```\n\nAdd a keyword parameter `timeout: float = 30.0` to "
     "`SafetensorRemote.get_data_by_range` (after `size`) and pass `timeout=timeout` to its `requests.get` call. Change nothing else."
     + TAIL, f2.replace(A2, A2.replace("size: int = -1)", "size: int = -1, timeout: float = 30.0)")).replace(
         B2, B2.replace("headers=headers)", "headers=headers, timeout=timeout)")), 6000),
    ("E3 bugfix", "Here is a Python script:\n\n```python\n" + f3.replace(GOOD3, BAD3) + "\n```\n\nThere is a bug: "
     "`get_prompt_lengths_rng` always returns `prompt_length_min` instead of a random length in the range. Fix the bug and change "
     "nothing else." + TAIL.replace("updated", "fixed"), f3, 6000),
]


def run(prompt, mt, greedy):
    body = {"model": MODEL, "stream": True, "max_tokens": mt, "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none", "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    if greedy:
        body["temperature"] = 0
    req = urllib.request.Request(URL, json.dumps(body).encode(), HDRS)
    t0 = time.time(); first = None; usage = None; out = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            for c in d.get("choices") or []:
                t = (c.get("delta") or {}).get("content")
                if t:
                    first = first or time.time()
                    out.append(t)
            if d.get("usage"):
                usage = d["usage"]
    end = time.time()
    n = usage["completion_tokens"]
    return usage["prompt_tokens"], n, first - t0, (n - 1) / (end - first), "".join(out)


def code_of(text):
    m = re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)```", text, re.S)
    return max(m, key=len) if m else text


tag = sys.argv[1]
res = []
for name, prompt, want, mt in TASKS:
    for mode in ("g", "s"):
        p, n, ttft, tps, txt = run(prompt, mt, mode == "g")
        got = code_of(txt)
        exact = got.strip() == want.strip()
        ratio = difflib.SequenceMatcher(None, got, want, autojunk=False).ratio()
        res.append({"task": name, "mode": mode, "prompt": p, "out": n, "ttft": round(ttft, 2), "tps": round(tps, 1),
                    "exact": exact, "ratio": round(ratio, 4)})
        print(f"{tag} {name:9s} {mode}: {tps:6.1f} t/s  ({n} out, prompt {p}, ttft {ttft:.1f}s)  exact={exact} ratio={ratio:.4f}",
              flush=True)
        open(f"edit_{tag}_{name.split()[0]}{mode}.txt", "w", encoding="utf-8").write(txt)
avg = lambda xs: sum(xs) / len(xs)
print(f"{tag} EDIT avg: greedy {avg([r['tps'] for r in res if r['mode']=='g']):.1f} t/s, sampled "
      f"{avg([r['tps'] for r in res if r['mode']=='s']):.1f} t/s, all {avg([r['tps'] for r in res]):.1f} t/s; "
      f"exact {sum(r['exact'] for r in res)}/{len(res)}", flush=True)
json.dump(res, open(f"edit_{tag}.json", "w"), indent=1)
