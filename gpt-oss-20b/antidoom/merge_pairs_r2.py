"""Assemble the round 2 training set: round 1 clean pairs + round 2 pairs (chat at high, tools at high), filtered.

Applies filter_pairs.py's rule to the round 2 files (drop "loops" whose snippet has <= 2 distinct characters and
whose repetition span is under 100 chars: markdown dividers, *** rules, Python docstring underlines), keeps the
round 1 file as it is (already filtered), and writes one JSONL plus a report: pairs per source_task, where the
loop sat (reasoning vs tool call), and the longest context_with_chat_template in TOKENS, which sets
max_seq_length in configs/xpu_train_r2b_noheads.yaml.

  python merge_pairs_r2.py --r2 runs/mine2-chat/iter_0_ftpo_pairs.jsonl runs/mine2-tools/iter_0_ftpo_pairs.jsonl \
      --round1 runs/mine1/pairs-clean.jsonl --out runs/mine2/pairs-r2.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

MAX_SPAN = 100  # same rule as filter_pairs.py


def divider_fp(row: dict) -> bool:
    rep = row["repetition"]
    snip = rep["snippet"].strip()
    # A run of "!" of any length is a NaN signature on the server (token id 0), never a loop the model chose: drop it.
    if snip and set(snip) == {"!"} or row.get("rejected_decoded", "").strip("! ") == "" and "!" in row.get("rejected_decoded", ""):
        return True
    return len(set(snip)) <= 2 and rep["end_char"] - rep["start_char"] < MAX_SPAN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2", nargs="+", required=True, help="round 2 pair files (unfiltered)")
    ap.add_argument("--round1", default="runs/mine1/pairs-clean.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-round1", action="store_true", help="round 2 pairs only")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    rows: list[dict] = []
    dropped = collections.Counter()
    for f in a.r2:
        n = 0
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if divider_fp(r):
                dropped[f] += 1
                continue
            r["_origin"] = f
            rows.append(r)
            n += 1
        print(f"{f}: kept {n}, dropped {dropped[f]} divider / bang false positives")
    if not a.no_round1:
        n = 0
        for line in open(a.round1, encoding="utf-8"):
            r = json.loads(line)
            r["_origin"] = a.round1
            rows.append(r)
            n += 1
        print(f"{a.round1}: {n} (already filtered)")

    by_task = collections.Counter(r.get("source_task") for r in rows)
    by_origin = collections.Counter(r["_origin"] for r in rows)
    in_call = sum(1 for r in rows if "<|channel|>commentary to=functions." in r["generation_raw"])
    high = sum(1 for r in rows if "Reasoning: high" in r["context_with_chat_template"])
    lengths = sorted(len(tok(r["context_with_chat_template"])["input_ids"]) for r in rows)
    print(f"\nTOTAL {len(rows)} pairs | effort high {high}, medium {len(rows) - high} | loop inside a tool call {in_call}")
    print("by origin:", dict(by_origin))
    print("by source_task:")
    for k, v in by_task.most_common():
        print(f"  {v:5d}  {k}")
    print(f"context tokens: median {lengths[len(lengths)//2]}, p95 {lengths[int(len(lengths)*0.95)]}, max {lengths[-1]}"
          f"  -> max_seq_length must be >= {lengths[-1]}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            r.pop("_origin", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
