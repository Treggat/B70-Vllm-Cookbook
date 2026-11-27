"""Inkling 2 Mini: decode speed against an OpenAI-compatible vLLM server.

  python bench/speed.py [--model inkling-2-mini] [--base http://localhost:8000] [--streams 1,4,16] [--tokens 512]

Every request streams a fixed number of output tokens (ignore_eos + min_tokens), so streams are comparable.
Per stream: decode tok/s = (tokens - 1) / (last token time - first token time). Aggregate = all tokens / wall time
of the slowest stream's decode window. One warm-up request first (Triton JIT/autotune). Writes results/<tag>.json.
"""
import argparse, json, threading, time, urllib.request
from pathlib import Path

TEMP = 0.7
PROMPT = ("Write a long, detailed technical essay about how a mixture-of-experts transformer routes tokens to experts, "
          "covering gating, load balancing, capacity, and inference costs. Use plain paragraphs.")


def one(base, model, tokens, out, idx):
    body = {"model": model, "stream": True, "max_tokens": tokens, "min_tokens": tokens, "ignore_eos": True,
            "temperature": TEMP, "messages": [{"role": "user", "content": f"Task {idx}. " + PROMPT}],
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.time(); n += 1
                    first = first or now; last = now
    toks = (usage or {}).get("completion_tokens", n)
    out[idx] = {"ttft": (first or t0) - t0, "tokens": toks, "chunks": n,
                "decode_tps": (toks - 1) / (last - first) if last and first and last > first else None,
                "t_first": first, "t_last": last}


def run(base, model, streams, tokens):
    out = [None] * streams
    th = [threading.Thread(target=one, args=(base, model, tokens, out, i)) for i in range(streams)]
    [t.start() for t in th]; [t.join() for t in th]
    firsts = [o["t_first"] for o in out]; lasts = [o["t_last"] for o in out]
    agg = sum(o["tokens"] for o in out) / (max(lasts) - min(firsts))
    per = [o["decode_tps"] for o in out if o["decode_tps"]]
    return {"streams": streams, "aggregate_tps": round(agg, 1), "per_stream_tps_mean": round(sum(per) / len(per), 1),
            "ttft_max_s": round(max(o["ttft"] for o in out), 2), "tokens_each": out[0]["tokens"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--streams", default="1,4,16")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--code", action="store_true", help="code-writing prompt instead of the prose essay")
    a = ap.parse_args()
    global TEMP, PROMPT
    TEMP = a.temp
    if a.code:
        PROMPT = ("Write a complete, well-structured Python module implementing an in-memory key-value store with TTL "
                  "expiry, an LRU eviction policy, thread safety, type hints, and a pytest test suite. Code only.")
    if not a.no_warmup:
        t = time.time(); run(a.base, a.model, 1, 64); print(f"warm-up 1x64 {time.time() - t:.1f}s", flush=True)
        t = time.time(); run(a.base, a.model, 16, 32); print(f"warm-up 16x32 {time.time() - t:.1f}s", flush=True)
    res = []
    for s in [int(x) for x in a.streams.split(",")]:
        r = run(a.base, a.model, s, a.tokens); res.append(r); print(json.dumps(r), flush=True)
    p = Path(__file__).with_name("results"); p.mkdir(exist_ok=True)
    (p / f"speed-{a.tag}.json").write_text(json.dumps({"model": a.model, "tokens": a.tokens, "runs": res}, indent=1))


if __name__ == "__main__":
    main()
