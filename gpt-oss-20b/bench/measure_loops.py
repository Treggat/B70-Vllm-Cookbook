"""Measure the doom-loop rate of an OpenAI-compatible chat endpoint on the Antidoom prompt mix.

Uses Antidoom's own detector (antidoom/repetition.py from Liquid AI's repo) and its
default generation settings (temperature 0.1, top_p 1.0, 4000 new tokens,
same prompt template).

  BASE_URL       server root, default http://localhost:8000 (a trailing /v1 is accepted)
  OPENAI_API_KEY optional bearer key
  ANTIDOOM_SRC   Antidoom's src dir, default ../antidoom/upstream/src (git clone of Liquid4All/antidoom)
  ANTIDOOM_MIX   the prompt mix as JSONL, default ../antidoom/data/antidoom-mix-v1.0.jsonl

  python measure_loops.py --model gpt-oss --n 200 --out runs/loops-medium.jsonl --effort medium
Re-running with the same --out resumes; finished prompt ids are skipped.
The 200-prompt sample (seed 7) is drawn from the mix file in file order, so the same file gives the same prompts.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, os.environ.get("ANTIDOOM_SRC", str(HERE.parent / "antidoom" / "upstream" / "src")))
from antidoom.repetition import find_inner_repetition  # noqa: E402  (pure python)

# grade_eval.py and chat_check.py import call(), BASE and KEY from here.
BASE = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/").removesuffix("/v1")
ENDPOINT = BASE + "/v1/chat/completions"
KEY = os.environ.get("OPENAI_API_KEY", "")
DATA = Path(os.environ.get("ANTIDOOM_MIX", str(HERE.parent / "antidoom" / "data" / "antidoom-mix-v1.0.jsonl")))
TEMPLATE = '{prompt}\n\nThink through the problem step by step, then respond with your final answer as "Answer: <your answer>".'
REP = dict(min_repeats=4, max_period=1024, min_period=1, min_total_repeated=60, sample_len=16, sample_interval=128)


def first_human(conv: str) -> str | None:
    try:
        turns = ast.literal_eval(conv) if isinstance(conv, str) else conv
    except Exception:
        return None
    for t in turns:
        if t.get("from") in ("human", "user"):
            return t.get("value")
    return None


def sample_prompts(n: int, seed: int) -> list[dict]:
    by_src: dict[str, list[dict]] = collections.defaultdict(list)
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_src[r["source"]].append(r)
    rng = random.Random(seed)
    for rows in by_src.values():
        rng.shuffle(rows)
    # interleave sources evenly, like Antidoom's default
    out, i, srcs = [], 0, sorted(by_src)
    while len(out) < n:
        for s in srcs:
            if i < len(by_src[s]) and len(out) < n:
                r = by_src[s][i]
                p = first_human(r["conversations"])
                if p and len(p) < 8000:
                    out.append({"id": r["id"], "source": s, "prompt": p})
        i += 1
    return out


def call(model: str, prompt: str, max_tokens: int, timeout: int, effort: str | None = None) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": TEMPLATE.format(prompt=prompt)}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 1.0,
    }
    if effort:
        body["reasoning_effort"] = effort
    headers = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers=headers)
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    c = d["choices"][0]
    m = c["message"]
    return {
        "served_by": d.get("model"),
        "secs": round(time.time() - t, 1),
        "finish": c.get("finish_reason"),
        "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
        "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
        "content": m.get("content") or "",
    }


def judge(rec: dict) -> dict:
    hits = {}
    for part in ("reasoning", "content"):
        found, hit = find_inner_repetition(rec[part], **REP)
        if found:
            hits[part] = {"period": hit.period, "repeats": hit.repeats, "snippet": hit.snippet}
    rec["loop"] = bool(hits)
    rec["loop_hits"] = hits
    return rec


def summarize(path: Path) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ok = [r for r in rows if "error" not in r]
    errs = len(rows) - len(ok)
    if not ok:
        print("no completed rows", "errors:", errs)
        return
    loops = sum(r["loop"] for r in ok)
    capped = sum(r["finish"] == "length" for r in ok)
    both = sum(r["loop"] and r["finish"] == "length" for r in ok)
    print(f"{path.name}: {len(ok)} done, {errs} errors")
    print(f"  loop rate      {loops}/{len(ok)} = {100*loops/len(ok):.1f}%")
    print(f"  hit max_tokens {capped}/{len(ok)} = {100*capped/len(ok):.1f}%  (loop AND capped: {both})")
    toks = sorted(r["completion_tokens"] or 0 for r in ok)
    print(f"  median tokens  {toks[len(toks)//2]}")
    per = collections.defaultdict(lambda: [0, 0])
    for r in ok:
        per[r["source"]][0] += 1
        per[r["source"]][1] += r["loop"]
    for s, (n, l) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"    {s:55s} {l:3d}/{n:<3d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--effort", default=None, help="reasoning_effort to send (default: none, the server's default)")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.summary_only:
        summarize(out)
        return
    done = set()
    if out.exists():
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        done = {r["id"] for r in rows if "error" not in r}
        # drop old error rows so the summary counts each prompt once
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                if "error" not in r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    todo = [p for p in sample_prompts(a.n, a.seed) if p["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to run on {a.model}", flush=True)
    lock = threading.Lock()
    count = [0]

    def work(p: dict) -> None:
        try:
            rec = judge({**p, **call(a.model, p["prompt"], a.max_tokens, a.timeout, effort=a.effort)})
        except Exception as e:  # keep going; errors are retried on resume
            rec = {**p, "error": repr(e)[:300]}
        rec.pop("prompt", None)
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count[0] += 1
            tag = "ERR" if "error" in rec else ("LOOP" if rec["loop"] else "ok")
            print(f"[{count[0]}/{len(todo)}] {tag:4s} {rec.get('finish','')} {rec.get('completion_tokens','')} tok "
                  f"{rec.get('secs','')}s {p['source']}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    summarize(out)


if __name__ == "__main__":
    main()
