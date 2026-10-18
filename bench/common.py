"""Shared bits for the bench scripts. Point them at any OpenAI-compatible endpoint (vLLM directly, or a router in front):
  BASE_URL     default http://localhost:8000/v1
  MODEL        default qwen38-709l (the --served-model-name in serve/serve-709l.sh)
  API_KEY      optional bearer token
  METRICS_URL  optional vLLM /metrics URL (default <BASE_URL without /v1>/metrics); used for MTP acceptance + preemptions.
All tests run with thinking off (chat_template_kwargs.enable_thinking=false)."""
import json, os, re, sysconfig, time, urllib.request

BASE = os.environ.get("BASE_URL", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("MODEL", "qwen38-709l")
KEY = os.environ.get("API_KEY", "")
METRICS = os.environ.get("METRICS_URL", BASE[:-3] + "/metrics" if BASE.endswith("/v1") else BASE + "/metrics")
HDR = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})}
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
SPEC = ["spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
        "num_preemptions_total", "generation_tokens_total"]
STDLIB = sysconfig.get_paths()["stdlib"]   # numbers in docs/RESULTS.md were taken with the Python 3.14 stdlib


def metrics():
    try:
        txt = urllib.request.urlopen(METRICS, timeout=30).read().decode()
    except Exception:
        return {}
    return {k: sum(float(v) for v in re.findall(rf"^vllm:{k}\{{[^}}]*\}} ([0-9.e+]+)", txt, re.M)) for k in SPEC}


def gauge(name):
    txt = urllib.request.urlopen(METRICS, timeout=30).read().decode()
    return sum(float(v) for v in re.findall(rf"^vllm:{name}\{{[^}}]*\}} ([0-9.e+]+)", txt, re.M))


def accept(a, b):
    if not a or not b:
        return None
    d = {k: b[k] - a[k] for k in SPEC}
    dr, dt, ac = d["spec_decode_num_drafts_total"], d["spec_decode_num_draft_tokens_total"], d["spec_decode_num_accepted_tokens_total"]
    return dict(per_token=round(ac / dt, 3) if dt else None, tok_per_step=round(1 + ac / dr, 2) if dr else None,
                preempt=d["num_preemptions_total"])


def canary(text):
    """None = looks fine; otherwise why the reply looks corrupted."""
    if not text.strip():
        return "empty"
    if "!!!!" in text:
        return "bangs"
    if "�" in text:
        return "replacement-char"
    pr = sum(c.isprintable() or c in "\n\t" for c in text) / len(text)
    if pr < 0.97:
        return f"printable {pr:.2f}"
    if re.search(r"(.{8,40}?)\1{6,}", text, re.S):
        return "loop"
    return None


def stream(content, n, temp=1.0, seed=7):
    """Streamed chat completion. Returns prompt/gen tokens, time to first token, decode t/s (after the first token)."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": temp, "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}, **OFF}
    t0 = time.time(); tf = None; usage = {}; text = []
    with urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), HDR), timeout=2400) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                    tf = tf or time.time()
                    if dl.get("content"):
                        text.append(dl["content"])
    t1 = time.time(); gen = usage.get("completion_tokens", 0)
    return dict(prompt=usage.get("prompt_tokens", 0), gen=gen, ttft=(tf - t0) if tf else None,
                tps=gen / (t1 - tf) if tf and gen and t1 > tf else 0, text="".join(text))


def complete(content, n, temp=0.2):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": temp, **OFF}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), HDR), timeout=2400))
    return r["choices"][0], r["usage"]


def stdlib_mix(files):
    return "".join(f"\n\n# ===== {f} =====\n" + open(os.path.join(STDLIB, f), encoding="utf-8").read() for f in files)
