"""KV capacity and speed bench for a gpt-oss vLLM server. The same script for every variant, so numbers compare.

  python bench_kv.py --base http://localhost:8000 --model gpt-oss --tag bf16 [--only pool,greedy,speed,long,fill16]
  python bench_kv.py --base http://localhost:8000 --model gpt-oss --tag fp8 --ref bf16

Tests:
  pool    KV pool tokens from the server's /metrics (vllm:cache_config_info kv_cache_size_tokens)
  greedy  20 short prompts at temperature 0, saved; with --ref, an exact-match count against that earlier tag
  speed   short prompt, decode tok/s at 1, 4, 16 streams (streaming, per-stream rate after the first token)
  long    one ~50k-token read: first token, decode tok/s, needle retrieval
  fill16  16 unique ~69k-token reads primed one after another, then all 16 sent again AT ONCE (how 16 long agent
          sessions come back for their next turn): prefix-cache hit, first-token spread, preemptions, needle hits
Results: runs/<tag>.json next to this script. Source text = the running Python's stdlib .py files, with a unique
header per request so prefix caching can't share the long prompts. --metrics-url if /metrics is behind a proxy path
(llama-swap: <base>/upstream/<model>/metrics).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import sysconfig
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "http://localhost:8000"  # set from --base
MODEL = "gpt-oss"  # set from --model
METRICS = ""  # set from --metrics-url
REF = ""  # set from --ref
HERE = Path(__file__).parent
OUT = HERE / "runs"
STDLIB = Path(sysconfig.get_paths()["stdlib"])
CHARS_PER_TOKEN = 3.6  # rough for Python source with o200k; real counts come back in usage


def post(body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def stream(body: dict, timeout: int = 1800) -> dict:
    """Streaming chat call; returns ttft, decode rate after the first token, text, usage."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = t0; parts = []; usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    parts.append(delta.get("content") or "")
    n = usage.get("completion_tokens", 0)
    dec = (n - 1) / (last - first) if first and last > first and n > 1 else 0.0
    return {"ttft": round((first or last) - t0, 2), "decode_tps": round(dec, 1), "wall": round(time.time() - t0, 2),
            "completion_tokens": n, "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "text": "".join(parts)}


def metrics() -> str:
    return urllib.request.urlopen(METRICS or BASE + "/metrics", timeout=30).read().decode()


def metric(text: str, name: str) -> float:
    tot = 0.0; hit = False
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith(name + "_created"):
            m = re.match(r"^" + re.escape(name) + r"(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
            if m:
                tot += float(m.group(2)); hit = True
    return tot if hit else float("nan")


def pool() -> dict:
    t = metrics()
    m = re.search(r'vllm:cache_config_info\{([^}]*)\}', t)
    info = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1))) if m else {}
    return {"pool_tokens": int(info.get("kv_cache_size_tokens", 0) or 0),
            "max_concurrency_at_max_len": float(info.get("kv_cache_max_concurrency", 0) or 0),
            "num_gpu_blocks": info.get("num_gpu_blocks"), "cache_dtype": info.get("cache_dtype"),
            "skip_layers": info.get("kv_cache_dtype_skip_layers"), "offload_size": info.get("kv_offloading_size"),
            "info": info}


def source_text(n_chars: int, seed: int) -> str:
    files = sorted(p for p in STDLIB.glob("*.py") if p.stat().st_size > 20000)
    random.Random(seed).shuffle(files)
    out = []; size = 0
    for p in files:
        s = p.read_text(encoding="utf-8", errors="replace")
        out.append(f"\n# ===== file: {p.name} =====\n{s}"); size += len(s)
        if size >= n_chars:
            break
    return "".join(out)[:n_chars]


def needle_prompt(tokens: int, seed: int) -> tuple[str, str]:
    code = f"{random.Random(seed).randrange(10**7, 10**8)}"
    body = source_text(int(tokens * CHARS_PER_TOKEN), seed)
    cut = len(body) // 3 + random.Random(seed + 1).randrange(len(body) // 3)  # needle somewhere in the middle third
    body = body[:cut] + f"\n# NOTE: the vault access code is {code}.\n" + body[cut:]
    q = (f"[request {seed}-{time.time_ns()}] Below is a large mix of Python source files.\n{body}\n\n"
         "Question: what is the vault access code mentioned in a comment above? Reply with the number only.")
    return q, code


def run_speed() -> dict:
    prompt = "Write a Python function that parses an ISO-8601 date string without using datetime. Include tests."
    res = {}
    for n in (1, 4, 16):
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 700,
                "temperature": 0.7, "reasoning_effort": "low"}
        stream(body)  # warm this batch size's graph
        with ThreadPoolExecutor(n) as ex:
            rs = list(ex.map(lambda _: stream(body), range(n)))
        per = [r["decode_tps"] for r in rs]
        res[f"x{n}"] = {"per_stream": per, "summed": round(sum(per), 1), "mean": round(sum(per) / n, 1)}
        print(f"  speed x{n}: summed {res[f'x{n}']['summed']} tok/s, per stream {per}", flush=True)
    return res


def run_long(tokens: int = 60000) -> dict:
    q, code = needle_prompt(tokens, seed=60)
    r = stream({"model": MODEL, "messages": [{"role": "user", "content": q}], "max_tokens": 400,
                "temperature": 0, "reasoning_effort": "low"})
    r["needle_ok"] = code in r["text"]; r["text"] = r["text"][:200]
    print(f"  long {r['prompt_tokens']} tok: ttft {r['ttft']} s, decode {r['decode_tps']} tok/s, needle {r['needle_ok']}",
          flush=True)
    return r


def run_fill16(tokens: int = 70000, n: int = 16) -> dict:
    pre0 = metric(metrics(), "vllm:num_preemptions_total")
    peak = {"usage": 0.0, "running": 0.0, "waiting": 0.0}
    done = False

    def watch():
        while not done:
            try:
                t = metrics()
                peak["usage"] = max(peak["usage"], metric(t, "vllm:kv_cache_usage_perc"))
                peak["running"] = max(peak["running"], metric(t, "vllm:num_requests_running"))
                peak["waiting"] = max(peak["waiting"], metric(t, "vllm:num_requests_waiting"))
            except Exception:
                pass
            time.sleep(2)

    # Two phases, like 16 long agent sessions: phase 1 primes each context into the prefix cache (serially, 1 token);
    # phase 2 sends all 16 again AT ONCE with a long answer. What the pool holds shows up as cached_tokens in
    # phase 2, as how many run together, and as preemptions.
    prompts = [needle_prompt(tokens, seed=1000 + i) for i in range(n)]
    t0 = time.time()
    for q, _ in prompts:
        post({"model": MODEL, "messages": [{"role": "user", "content": q}], "max_tokens": 1,
              "temperature": 0, "reasoning_effort": "low"}, timeout=3600)
    prime = round(time.time() - t0, 1)
    pre0 = metric(metrics(), "vllm:num_preemptions_total")

    def one(i):
        q, code = prompts[i]
        r = stream({"model": MODEL, "messages": [{"role": "user", "content": q}], "max_tokens": 1500,
                    "temperature": 0, "reasoning_effort": "low"}, timeout=3600)
        r["needle_ok"] = code in r["text"]; r["text"] = r["text"][:120]
        return r

    t1 = time.time()
    with ThreadPoolExecutor(n + 1) as ex:
        w = ex.submit(watch)
        rs = list(ex.map(one, range(n)))
        done = True; w.result()
    pre = metric(metrics(), "vllm:num_preemptions_total") - pre0
    ptot = sum(r["prompt_tokens"] or 0 for r in rs); ctot = sum(r["cached_tokens"] or 0 for r in rs)
    out = {"streams": n, "target_tokens": tokens, "prime_wall": prime, "wall": round(time.time() - t1, 1),
           "prompt_tokens_total": ptot, "cached_tokens_total": ctot, "cache_hit": round(ctot / max(ptot, 1), 3),
           "needle_hits": sum(r["needle_ok"] for r in rs), "preemptions": pre,
           "ttft": sorted(r["ttft"] for r in rs), "decode_tps": [r["decode_tps"] for r in rs],
           "completion_tokens": [r["completion_tokens"] for r in rs],
           "peak_kv_usage": peak["usage"], "peak_running": peak["running"], "peak_waiting": peak["waiting"]}
    print(f"  fill16: {ptot} prompt tok, cache hit {out['cache_hit']:.1%} in phase 2, wall {out['wall']} s "
          f"(prime {prime} s), preemptions {pre}, peak running {peak['running']} waiting {peak['waiting']} "
          f"kv {peak['usage']:.2f}, needles {out['needle_hits']}/{n}, ttft {out['ttft'][0]}-{out['ttft'][-1]} s, "
          f"decode {min(out['decode_tps'])}-{max(out['decode_tps'])} tok/s", flush=True)
    return out


GREEDY = ["What is the capital of Australia? One sentence.", "Explain what a mutex is in two sentences.",
          "Write a haiku about rain on a tin roof.", "List the first ten prime numbers.",
          "Translate 'the library opens at nine' into French.", "What does HTTP status 418 mean?",
          "Write a Python one-liner that reverses a string.", "Name three moons of Jupiter.",
          "Summarise the plot of Hamlet in three sentences.", "What is 17 * 23? Show the working.",
          "Give two differences between TCP and UDP.", "Write a SQL query that counts rows per country in a table users.",
          "What causes the seasons on Earth?", "Suggest a name for a grey cat. One word.",
          "Convert 100 Fahrenheit to Celsius.", "What is a closure in JavaScript? Short answer.",
          "Write a regex that matches a US ZIP code.", "Who wrote 'Pride and Prejudice'?",
          "Explain big-O of binary search in one line.", "Write a bash command that counts lines in all .py files."]


def run_greedy(tag: str) -> dict:
    def one(p):
        d = post({"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 400,
                  "temperature": 0, "reasoning_effort": "low", "seed": 0})
        m = d["choices"][0]["message"]
        return {"prompt": p, "content": m.get("content"), "reasoning": m.get("reasoning_content") or m.get("reasoning")}
    rs = [one(p) for p in GREEDY]  # serial: batch shape must not differ between runs
    (OUT / f"greedy-{tag}.json").write_text(json.dumps(rs, indent=1, ensure_ascii=False), encoding="utf-8")
    ref = OUT / f"greedy-{REF}.json"
    cmp = None
    if REF and ref.exists():
        base = json.loads(ref.read_text(encoding="utf-8"))
        same = sum(a["content"] == b["content"] for a, b in zip(base, rs))
        cmp = {"identical_content": same, "of": len(rs)}
        print(f"  greedy vs {REF}: {same}/{len(rs)} identical", flush=True)
    return {"n": len(rs), "vs_base": cmp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="gpt-oss")
    ap.add_argument("--metrics-url", default="")
    ap.add_argument("--ref", default="", help="earlier tag to diff greedy output against")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only", default="pool,greedy,speed,long,fill16")
    ap.add_argument("--fill-tokens", type=int, default=84000, help="target size of each fill16 read; 84000 gives about 69k real tokens")
    a = ap.parse_args()
    global BASE, MODEL, METRICS, REF
    BASE, MODEL, METRICS, REF = a.base.rstrip("/"), a.model, a.metrics_url, a.ref
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{a.tag}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["tag"] = a.tag; res.setdefault("runs", []).append(time.strftime("%Y-%m-%d %I:%M %p"))
    print(f"[{a.tag}] warm-up", flush=True)
    print("  ", post({"model": MODEL, "messages": [{"role": "user", "content": "Capital of France? One word."}],
                     "max_tokens": 60, "reasoning_effort": "low"}, timeout=1200)["choices"][0]["message"]["content"])
    for t in a.only.split(","):
        print(f"[{a.tag}] {t}", flush=True)
        if t == "pool":
            res["pool"] = pool(); print("  ", {k: v for k, v in res["pool"].items() if k != "info"}, flush=True)
        elif t == "speed":
            res["speed"] = run_speed()
        elif t == "long":
            res["long"] = run_long()
        elif t == "fill16":
            res["fill16"] = run_fill16(a.fill_tokens)
        elif t == "greedy":
            res["greedy"] = run_greedy(a.tag)
        path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[{a.tag}] done -> {path}")


if __name__ == "__main__":
    sys.exit(main())
