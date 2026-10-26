"""crossing_trace.py: bench/crossing.py's prompt, sampled (temperature 1.0), seeded per run, with logprobs. On a wrong
reply print the tokens around the first fault with the probability the TARGET gave each sampled token and its top-3.
On this text a healthy target puts ~1.0 on the next token. If its own top-3 is spread out at the fault (and a few
tokens before it), the target's state was wrong at that step; if the target still put ~1.0 on another token, the
sampler or verifier emitted a token the target ruled out. A leading code fence or a reply cut by max_tokens is not
counted as a fault. usage (endpoint from BASE_URL, MODEL): python crossing_trace.py <first_seed> <runs> [stop_after_fails=3]"""
import json, math, os, re, sys, urllib.request
BASE = os.environ["BASE_URL"]; MODEL = os.environ["MODEL"]
s0, R = int(sys.argv[1]), int(sys.argv[2]); STOP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
seed_txt = "\n".join(f"def func_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(6))
prompt = ("Continue this Python file with the same pattern up to and including func_399 "
          "(func_i returns x * i + (i % 7)). Output the whole file, code only, no commentary, no code fences.\n\n" + seed_txt)
expect = "\n".join(f"def func_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(400))
fails = 0
for r in range(R):
    seed = s0 + r
    req = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000, "temperature": 1.0,
           "seed": seed, "logprobs": True, "top_logprobs": 3, "chat_template_kwargs": {"enable_thinking": False}}
    j = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=900))
    c = j["choices"][0]; toks = c["logprobs"]["content"]; text = "".join(t["token"] for t in toks)
    ptok = j["usage"]["prompt_tokens"]
    # first char where the reply leaves the expected text (ignore a cut at the very end)
    off = text.find("def func_0(x):")
    body = text[off:] if off >= 0 else text
    n = min(len(body), len(expect)); k = next((i for i in range(n) if body[i] != expect[i]), None)
    if k is not None:
        k += max(off, 0)
        # a reply cut by max_tokens ends exactly at len(text): only count it when the mismatch is not just the cut
        if k >= len(text) - 1:
            k = None
    early = c["finish_reason"] == "stop" and len(body) < len(expect) - 50
    if k is None and not early:
        print(f"seed {seed}: ok ({len(toks)} tok, finish={c['finish_reason']})", flush=True); continue
    fails += 1
    if k is None:
        k = len(text)
    # token index containing char k
    acc = 0; ti = len(toks) - 1
    for i, t in enumerate(toks):
        if acc + len(t["token"]) > k:
            ti = i; break
        acc += len(t["token"])
    print(f"seed {seed}: FAIL at char {k}, token {ti} = absolute position {ptok + ti} (prompt {ptok}, page edge 3840), "
          f"finish={c['finish_reason']}, gen {len(toks)}", flush=True)
    for i in range(max(0, ti - 8), min(len(toks), ti + 4)):
        t = toks[i]
        tops = "  ".join(f"{x['token']!r} {math.exp(x['logprob']):.4f}" for x in t["top_logprobs"])
        print(f"   {'>>' if i == ti else '  '} pos {ptok + i:5d} {t['token']!r:10} p={math.exp(t['logprob']):.6f} | top: {tops}", flush=True)
    if fails >= STOP:
        break
print(f"done: {fails} fail(s) in {r + 1} runs")
