"""No-tool chat check, the first gate for any fine-tune that touches the harmony channel choice. Round 3 variants C and
C2 failed because in plain chat (no tools) the channel token after the analysis became `comment`, giving empty answers
or commentary text in the answer. On 50 of the held-out loop prompts (measure_loops sample, seed 7, same template), at
temperature 1.0 (vLLM's default when a client sends none) and medium effort, count per answer:
  COMMENTARY  the model opened a commentary message anywhere (r2b: never in plain chat)
  EMPTY       content empty
  LEAK        content starts with reasoning ("We need ...", "Let's ...", ...)
Pass = all three at the baseline model's level (0, or 1 at most). Run the same on the baseline first.
Token ids are gpt-oss's (o200k_harmony): 200005 = <|channel|>, 12606 = "comment".

  python chat_check.py --model gpt-oss --out runs/chat-check.jsonl          # BASE_URL / OPENAI_API_KEY as measure_loops.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from measure_loops import BASE, KEY, TEMPLATE, sample_prompts  # noqa: E402

ID_CHANNEL, ID_COMMENT = 200005, 12606
LEAK = ("we need", "we should", "let's", "we must", "the user", "respond", "now ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--model", default="gpt-oss")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    key = KEY or None
    prompts = sample_prompts(a.n, 7)
    url = a.base.rstrip("/") + "/v1/chat/completions"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(a.out, "w", encoding="utf-8")
    st = collections.Counter()

    def one(p):
        body = {"model": a.model, "messages": [{"role": "user", "content": TEMPLATE.format(prompt=p["prompt"])}], "max_tokens": 4000,
                "temperature": 1.0, "top_p": 1.0, "reasoning_effort": a.effort, "return_token_ids": True}
        h = {"Content-Type": "application/json"}
        if key:
            h["Authorization"] = "Bearer " + key
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers=h), timeout=900))
        except Exception as e:  # noqa: BLE001
            return {"id": p["id"], "error": repr(e)[:200]}
        c = d["choices"][0]
        ids = c.get("token_ids") or []
        content = c["message"].get("content") or ""
        commentary = any(ids[j - 1] == ID_CHANNEL and ids[j] == ID_COMMENT for j in range(1, len(ids)))
        return {"id": p["id"], "source": p["source"], "finish": c.get("finish_reason"), "commentary": commentary,
                "empty": not content.strip(), "leak": content[:40].lower().startswith(LEAK), "content": content[:200]}

    with ThreadPoolExecutor(a.threads) as ex:
        for r in ex.map(one, prompts):
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            if "error" in r:
                st["error"] += 1
                continue
            st["n"] += 1
            for k in ("commentary", "empty", "leak"):
                st[k] += r[k]
            st["capped"] += r["finish"] == "length"
    print(f"{a.model}: {dict(st)}")
    ok = st["commentary"] <= 1 and st["empty"] - st["capped"] <= 1 and st["leak"] <= 1
    print("CHAT CHECK", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
