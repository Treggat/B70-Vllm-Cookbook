"""Graded accuracy of an OpenAI-compatible chat endpoint on public test sets with known answers.

GSM8K test, MATH-500, MMLU test and CommonsenseQA validation, a fixed random
sample of each (same items for every model). Same prompt template and sampling
as measure_loops.py, so loop and accuracy numbers describe the same behaviour.
None of these sets were used to mine the Antidoom pairs. Endpoint and key: BASE_URL / OPENAI_API_KEY (measure_loops.py).

  python grade_eval.py --model gpt-oss --effort medium --out runs/grade-medium.jsonl
  python grade_eval.py --summary runs/grade-stock.jsonl runs/grade-antidoom.jsonl
Re-running with the same --out resumes.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from measure_loops import REP, TEMPLATE, call  # noqa: E402
from measure_loops import find_inner_repetition  # noqa: E402

PER_TASK = {"gsm8k": 200, "math500": 150, "mmlu": 200, "csqa": 150}
LETTERS = "ABCDE"


def items(seed: int) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    out: list[dict] = []

    def take(name, rows):
        rows = list(rows)
        rng.shuffle(rows)
        out.extend(dict(r, task=name, id=f"{name}:{r['id']}") for r in rows[: PER_TASK[name]])

    g = load_dataset("openai/gsm8k", "main", split="test")
    take("gsm8k", ({"id": i, "prompt": f"Question: {r['question']}", "gold": r["answer"].split("####")[-1].strip()}
                   for i, r in enumerate(g)))
    m = load_dataset("HuggingFaceH4/MATH-500", split="test")
    take("math500", ({"id": r["unique_id"], "prompt": f"Question: {r['problem']}", "gold": r["answer"]} for r in m))
    u = load_dataset("cais/mmlu", "all", split="test")
    take("mmlu", ({"id": i, "prompt": f"Question: {r['question']}\n" + "\n".join(
        f"{LETTERS[j]}. {c}" for j, c in enumerate(r["choices"])), "gold": LETTERS[r["answer"]]} for i, r in enumerate(u)))
    c = load_dataset("tau/commonsense_qa", split="validation")
    take("csqa", ({"id": r["id"], "prompt": f"Question: {r['question']}\n" + "\n".join(
        f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"])), "gold": r["answerKey"]} for r in c))
    return out


def final_answer(text: str) -> str | None:
    hits = re.findall(r"Answer:\s*(.+)", text)
    if hits:
        return hits[-1].strip()
    boxed = re.findall(r"\\boxed\{(.+)\}", text)
    return boxed[-1].strip() if boxed else None


def norm_math(s: str) -> str:
    s = s.strip().rstrip(".")
    m = re.search(r"\\boxed\{(.*)\}", s)
    if m:
        s = m.group(1)
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def number(s: str) -> float | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def grade(task: str, text: str, gold: str) -> bool:
    ans = final_answer(text)
    if ans is None:
        return False
    if task == "gsm8k":
        a, g = number(ans), number(gold)
        return a is not None and g is not None and abs(a - g) < 1e-6
    if task in ("mmlu", "csqa"):
        m = re.match(r"\(?([A-E])\b", ans.strip().lstrip("*").strip())
        return bool(m) and m.group(1) == gold
    a, g = norm_math(ans), norm_math(gold)
    if a == g:
        return True
    na, ng = number(a), number(g)
    return na is not None and ng is not None and re.fullmatch(r"-?[\d.,]+", g or "") is not None and abs(na - ng) < 1e-6


def run(a) -> None:
    out = Path(a.out)
    done = set()
    if out.exists():
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        done = {r["id"] for r in rows if "error" not in r}
        with open(out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if "error" not in r)
    todo = [it for it in items(a.seed) if it["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to run on {a.model}", flush=True)
    lock, n = threading.Lock(), [0]

    def work(it):
        try:
            r = call(a.model, it["prompt"], a.max_tokens, a.timeout, effort=a.effort)
            r["correct"] = grade(it["task"], r["content"], it["gold"])
            r["loop"] = any(find_inner_repetition(r[p], **REP)[0] for p in ("reasoning", "content"))
            rec = {"id": it["id"], "task": it["task"], "gold": it["gold"], "answer": final_answer(r["content"]), **r}
        except Exception as e:
            rec = {"id": it["id"], "task": it["task"], "error": repr(e)[:300]}
        with lock:
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[0] += 1
            tag = "ERR" if "error" in rec else ("ok" if rec["correct"] else "--")
            print(f"[{n[0]}/{len(todo)}] {tag} {it['task']}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))


def summary(paths: list[str]) -> None:
    import collections

    for p in paths:
        rows = [r for r in (json.loads(l) for l in open(p, encoding="utf-8")) if "error" not in r]
        per = collections.defaultdict(lambda: [0, 0, 0, 0])
        for r in rows:
            s = per[r["task"]]
            s[0] += 1; s[1] += r["correct"]; s[2] += r["finish"] == "length"; s[3] += r["loop"]
        tot = [sum(v[i] for v in per.values()) for i in range(4)]
        print(f"{Path(p).name}: {tot[1]}/{tot[0]} = {100*tot[1]/max(tot[0],1):.1f}% correct, "
              f"{tot[2]} hit max_tokens, {tot[3]} loops")
        for t in PER_TASK:
            if t in per:
                n, c, cap, lp = per[t]
                print(f"    {t:8s} {c:3d}/{n:<3d} = {100*c/n:5.1f}%   capped {cap}   loops {lp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--effort", default=None, help="reasoning_effort to send (default: none, the server's default)")
    ap.add_argument("--summary", nargs="+")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    else:
        run(a)


if __name__ == "__main__":
    main()
