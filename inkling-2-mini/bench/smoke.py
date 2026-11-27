"""Inkling 2 Mini: functional smoke test against an OpenAI-compatible vLLM server.

  python bench/smoke.py [--model inkling-2-mini] [--base http://localhost:8000] [--long-tokens 30000]

1 tool call: one tool declared, the model must call it with sane JSON args (inkling tool parser).
2 tool round trip: feed the tool result back, the model must answer from it.
3 long context: a needle in ~N tokens of Python stdlib source, cold read; reports TTFT and the answer.
4 reasoning effort: same question with reasoning_effort low/high, reasoning token counts.
"""
import argparse, json, random, time, urllib.request
from pathlib import Path

STDLIB = Path(r"C:\Python314\Lib")
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                         "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]


def post(base, body, timeout=1800):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--long-tokens", type=int, default=30000)
    a = ap.parse_args(); res = {}

    msgs = [{"role": "user", "content": "What's the weather in Denver right now, in fahrenheit?"}]
    d, dt = post(a.base, {"model": a.model, "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
                          "max_tokens": 2048, "temperature": 0})
    m = d["choices"][0]["message"]; calls = m.get("tool_calls") or []
    ok = bool(calls) and calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"]) if ok else None
    res["tool_call"] = {"ok": ok and "denver" in str(args).lower(), "args": args, "content": m.get("content"),
                        "finish": d["choices"][0]["finish_reason"], "s": round(dt, 1)}
    print("tool_call", json.dumps(res["tool_call"]), flush=True)

    if ok:
        msgs += [{"role": "assistant", "content": m.get("content") or "", "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"temp": 41, "unit": "f", "sky": "snow"})}]
        d, dt = post(a.base, {"model": a.model, "messages": msgs, "tools": TOOLS, "max_tokens": 2048, "temperature": 0})
        c = d["choices"][0]["message"].get("content") or ""
        res["tool_round_trip"] = {"ok": "41" in c, "content": c[:300], "s": round(dt, 1)}
        print("tool_round_trip", json.dumps(res["tool_round_trip"]), flush=True)

    files = sorted(p for p in STDLIB.glob("*.py") if p.stat().st_size > 20000); random.Random(7).shuffle(files)
    text = ""; target = int(a.long_tokens * 3.6)
    for f in files:
        text += f"\n# ==== {f.name} ====\n" + f.read_text(encoding="utf-8", errors="replace")
        if len(text) > target: break
    text = text[:target]; cut = len(text) * 2 // 3
    needle = "\n# The vault access code for project HEARTHSTONE is 7741-QX.\n"
    text = text[:cut] + needle + text[cut:]
    q = text + "\n\nQuestion: what is the vault access code for project HEARTHSTONE? Reply with the code only."
    d, dt = post(a.base, {"model": a.model, "messages": [{"role": "user", "content": q}], "max_tokens": 1024,
                          "temperature": 0})
    c = d["choices"][0]["message"].get("content") or ""; u = d.get("usage", {})
    res["long"] = {"ok": "7741-QX" in c, "prompt_tokens": u.get("prompt_tokens"), "answer": c[:120], "s": round(dt, 1)}
    print("long", json.dumps(res["long"]), flush=True)

    res["effort"] = {}
    for eff in ("low", "high"):
        d, dt = post(a.base, {"model": a.model, "reasoning_effort": eff, "max_tokens": 8192, "temperature": 0,
                              "messages": [{"role": "user", "content": "How many primes are there below 200? Give the number."}]})
        u = d.get("usage", {}); c = d["choices"][0]["message"].get("content") or ""
        res["effort"][eff] = {"answer": c[:80], "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                              "completion_tokens": u.get("completion_tokens"), "s": round(dt, 1), "ok": "46" in c}
        print("effort", eff, json.dumps(res["effort"][eff]), flush=True)

    p = Path(__file__).with_name("results"); p.mkdir(exist_ok=True)
    (p / f"smoke-{time.strftime('%Y%m%d-%H%M')}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
