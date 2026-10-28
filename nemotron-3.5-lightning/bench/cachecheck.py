"""Prefix-cache correctness with --prefix-match-unit: P (~16k tokens) + 8 questions, greedy, thinking off.
Cached = plain request after P is warm (hits land on 64-token boundaries inside a 4224-token block);
cold = same tokens with a unique cache_salt (no cache sharing). Answers must be byte-identical."""
import json, sys, uuid, urllib.request, glob, os
# usage: python cachecheck.py [base_url] [model]   (default http://localhost:8000 nemotron)
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
URL = BASE + "/v1/chat/completions"; MODEL = sys.argv[2] if len(sys.argv) > 2 else "nemotron"
src = "".join(open(f, encoding="utf-8", errors="ignore").read() for f in sorted(glob.glob(os.path.join(os.path.dirname(json.__file__), "*.py"))))
P = "Reference source code follows.\n\n" + (src * 3)[:60000]
QS = ["Which function in this code decodes a JSON document? Name it only.", "What exception class is raised on a decode error?",
      "List three module names that appear in import statements.", "What does the 'indent' parameter control?",
      "Name one class defined in this code.", "What is the default value of 'ensure_ascii'?",
      "Summarize in one sentence what this code does.", "Which function would you call to serialize to a file?"]
def ask(q, salt=None, mid=None):
    b = {"model": MODEL, "temperature": 0, "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": P + "\n\nQuestion: " + q + (mid or "")}]}
    if salt: b["cache_salt"] = salt
    d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900))
    return d["choices"][0]["message"]["content"], (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"), d["usage"]["prompt_tokens"]
ask("warm-up")
same = 0
for q in QS:
    a, c, p = ask(q); b, c2, _ = ask(q, salt=uuid.uuid4().hex)
    same += a == b
    print(f"prompt {p} cached {c} (cold run cached {c2})  identical={a == b}  {'' if a == b else repr(a[:80]) + ' | ' + repr(b[:80])}", flush=True)
print(f"IDENTICAL {same}/{len(QS)}")
