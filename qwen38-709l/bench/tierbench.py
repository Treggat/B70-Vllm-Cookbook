# tierbench.py: per-depth decode and copy-edit numbers for the 709L combined seat, with spec counters per run.
#   python tierbench.py deep <tag> 23000 58000 115000      explain-the-last-file decode (deep_decode.py's prompt)
#   python tierbench.py edit <tag> 45000 90000             E3 bugfix (3.6k-token whole-file answer) behind N tokens of
#                                                          filler, exact-match check: copy at depth
# Each depth: one fill call (warms the prefix cache), then the measured call between two /metrics snapshots.
# Prints t/s, tokens per verify step, drafts per step and ms per step; appends JSON lines to tier_<tag>.jsonl.
import glob, json, os, re, sys, sysconfig, time, urllib.request
URL = os.environ["BASE_URL"].rstrip("/") + "/chat/completions"
MODEL = os.environ.get("MODEL", "LLM:Fable-709L")
HDRS = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["API_KEY"]}
MET = "http://192.168.1.218:8080/upstream/VLLM-Qwen3.8-Fable-709L/metrics"
KEYS = {"drafts": "vllm:spec_decode_num_drafts_total", "dtok": "vllm:spec_decode_num_draft_tokens_total",
        "acc": "vllm:spec_decode_num_accepted_tokens_total"}
L = os.environ.get("LLAMACPP_TREE", r"G:\AI\Projects\llama.cpp-b70-patches")

src = ""
for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**/*.py"), recursive=True)):
    src += open(f, errors="ignore").read()
    if len(src) > 2_000_000:
        break
f3 = open(os.path.join(L, r"scripts\server-bench.py"), encoding="utf-8").read()
GOOD3 = "        ret.append(random.randint(prompt_length_min, prompt_length_max))"
BAD3 = "        ret.append(random.randint(prompt_length_min, prompt_length_min))"
E3 = ("Here is a Python script:\n\n```python\n" + f3.replace(GOOD3, BAD3) + "\n```\n\nThere is a bug: "
      "`get_prompt_lengths_rng` always returns `prompt_length_min` instead of a random length in the range. Fix the bug and "
      "change nothing else. Return the COMPLETE fixed file in one code block and nothing else.")


def snap():
    t = urllib.request.urlopen(MET, timeout=20).read().decode()
    return {k: float(re.search(r"^" + re.escape(n) + r"\{[^}]*\} ([0-9.e+]+)$", t, re.M).group(1)) for k, n in KEYS.items()}


def call(prompt, mt):
    body = {"model": MODEL, "stream": True, "max_tokens": mt, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none", "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time(); first = None; usage = None; out = []
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), HDRS), timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            for c in d.get("choices") or []:
                t = (c.get("delta") or {}).get("content")
                if t:
                    first = first or time.time(); out.append(t)
            if d.get("usage"):
                usage = d["usage"]
    end = time.time()
    n = usage["completion_tokens"]
    return usage["prompt_tokens"], n, (n - 1) / (end - first), end - first, "".join(out)


def code_of(text):
    m = re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)```", text, re.S)
    return max(m, key=len) if m else text


mode, tag = sys.argv[1], sys.argv[2]
for n in map(int, sys.argv[3:]):
    if mode == "deep":
        prompt, mt = src[: n * 4] + "\n\nExplain in detail what the last file above does, function by function.", 500
    else:
        prompt, mt = src[: max(0, n - 4000) * 4] + "\n\n" + E3, 6000
    call(prompt, 1 if mode == "edit" else mt)  # fill the prefix cache (edit: prompt only)
    a = snap()
    p, out, tps, secs, txt = call(prompt, mt)
    b = snap()
    d = {k: b[k] - a[k] for k in KEYS}
    steps = d["drafts"] or 1
    row = {"mode": mode, "tag": tag, "prompt": p, "out": out, "tps": round(tps, 1), "tok_step": round((d["acc"] + steps) / steps, 2),
           "drafts_step": round(d["dtok"] / steps, 2), "ms_step": round(1000 * secs / steps, 1)}
    if mode == "edit":
        row["exact"] = code_of(txt).strip() == f3.strip()
    print(f"{tag} {mode} {p:6d} tok: {tps:6.1f} t/s, {row['tok_step']:5.2f} tok/step, {row['drafts_step']:5.2f} drafts/step, "
          f"{row['ms_step']:5.1f} ms/step" + (f", exact={row['exact']}" if mode == "edit" else ""), flush=True)
    open(f"tier_{tag}.jsonl", "a").write(json.dumps(row) + "\n")
