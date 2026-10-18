"""Build the input_jsonl for Antidoom's generator from the Antidoom mix.

Keeps only the first human turn (as measure_loops.py does), limited to
the sources where gpt-oss actually looped in the baseline, and leaves out
every prompt used in a baseline run (HELDOUT_RUNS) so those stay a held-out eval set.

  python build_prompts.py --out data/prompts-gptoss.jsonl
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data" / "antidoom-mix-v1.0.jsonl"
HELDOUT_RUNS = ["runs/gptoss-200.jsonl"]  # measure_loops.py outputs whose prompt ids must stay out of training

# sources with loops in the gpt-oss baseline (runs/gptoss-200.jsonl)
LOOPY = [
    "Post-training-Data-Flywheel/AutoIF-instruct-61k",
    "mlabonne/lmsys-arena-human-preference-55k-sharegpt",
    "mlabonne/ultrachat_200k_sft",
    "codeparrot/apps",
    "LiquidAI/ifstruct_generated",
    "allenai/math_qa",
    "theblackcat102/evol-codealpaca-v1",
]


def first_human(conv) -> str | None:
    try:
        turns = ast.literal_eval(conv) if isinstance(conv, str) else conv
    except Exception:
        return None
    for t in turns:
        if t.get("from") in ("human", "user"):
            return t.get("value")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    heldout: set[str] = set()
    for run in HELDOUT_RUNS:
        p = HERE / run
        if p.exists():
            heldout |= {json.loads(l)["id"] for l in open(p, encoding="utf-8")}

    by_src: dict[str, list[dict]] = collections.defaultdict(list)
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] in LOOPY and r["id"] not in heldout:
                by_src[r["source"]].append(r)

    rng = random.Random(a.seed)
    picked: dict[str, list[dict]] = {}
    for src, rows in by_src.items():
        rng.shuffle(rows)
        keep = []
        for r in rows:
            p = first_human(r["conversations"])
            if p and len(p) < 8000:
                keep.append({"id": r["id"], "source": src, "conversations": [{"from": "human", "value": p}]})
            if len(keep) >= a.per_source:
                break
        picked[src] = keep

    # interleave so any prefix of the file is a balanced mix
    out = []
    for i in range(max(len(v) for v in picked.values())):
        for src in LOOPY:
            if i < len(picked.get(src, [])):
                out.append(picked[src][i])
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} prompts to {a.out} ({len(heldout)} held-out ids excluded)")
    for src in LOOPY:
        print(f"  {src:55s} {len(picked.get(src, []))}")


if __name__ == "__main__":
    main()
