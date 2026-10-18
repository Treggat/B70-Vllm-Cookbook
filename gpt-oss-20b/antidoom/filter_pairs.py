"""Drop FTPO pairs whose "loop" is a short run of one or two characters.

The detector's floor (60 repeated chars) is crossed by ordinary markdown
table dividers, *** rules and the like. Those are dropped; a long run of the
same characters (a real runaway, hundreds of repeats) is kept.

  python filter_pairs.py runs/mine1/iter_0_ftpo_pairs.jsonl runs/mine1/pairs-clean.jsonl
"""
import collections
import json
import sys

MAX_SPAN = 100  # chars; real loops in this run span thousands

src, dst = sys.argv[1], sys.argv[2]
kept, dropped = [], collections.Counter()
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    rep = r["repetition"]
    snip = rep["snippet"].strip()
    if len(set(snip)) <= 2 and rep["end_char"] - rep["start_char"] < MAX_SPAN:
        dropped[repr(snip[:6])] += 1
        continue
    kept.append(line)
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(kept)
print(f"kept {len(kept)}, dropped {sum(dropped.values())} -> {dst}")
for k, n in dropped.most_common():
    print(f"  {n:4d} {k}")
