# prose_compare.py <tag_a> <tag_b>: compare two prose_ab.py runs (tag_a = the baseline).
# Per mode: tokens/step and decode t/s for each arm, the change, and a noise yardstick:
#   G (greedy): paired per prompt (same prompt, mostly the same text): mean and spread of the per-prompt tok/step change,
#               plus how many of the 10 texts are identical across arms
#   S (sampled, 2 seeds): the base arm's own seed-1 vs seed-2 difference as the noise yardstick
#   T (thinking, 1 seed): aggregate only
import json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
load = lambda t: [json.loads(l) for l in open(os.path.join(HERE, f"prose_{t}.jsonl"), encoding="utf-8")]
A, B = load(sys.argv[1]), load(sys.argv[2])


def agg(rows):
    dr = sum(r["drafts"] for r in rows); acc = sum(r["accepted"] for r in rows)
    tok = sum(r["tokens"] for r in rows); dec = sum(r["decode_s"] for r in rows)
    return 1 + acc / max(dr, 1), tok / max(dec, 1e-6), len(rows), tok


print(f"{'mode':4s} {'base tok/step':>14s} {'sl tok/step':>12s} {'change':>8s} | {'base t/s':>9s} {'sl t/s':>8s} {'change':>8s} | n")
for mode in ("G", "S", "T"):
    a = [r for r in A if r["mode"] == mode]; b = [r for r in B if r["mode"] == mode]
    if not a or not b:
        continue
    (ta, va, na, _), (tb, vb, nb, _) = agg(a), agg(b)
    print(f"{mode:4s} {ta:14.3f} {tb:12.3f} {100 * (tb / ta - 1):+7.2f}% | {va:9.1f} {vb:8.1f} {100 * (vb / va - 1):+7.2f}% | {na}/{nb}")
    if mode == "G":
        ka = {r["prompt"]: r for r in a}; kb = {r["prompt"]: r for r in b}
        tps = lambda r: 1 + r["accepted"] / max(r["drafts"], 1)   # same zero guard as agg()
        ch = [100 * (tps(kb[p]) / tps(ka[p]) - 1) for p in ka if p in kb]
        same = sum(ka[p]["text_sha"] == kb[p]["text_sha"] for p in ka if p in kb)
        if ch:
            print(f"     G paired per prompt: mean {st.mean(ch):+.2f}%, median {st.median(ch):+.2f}%, min {min(ch):+.2f}%, "
                  f"max {max(ch):+.2f}%; identical texts {same}/{len(ch)}")
    if mode == "S":
        s1 = agg([r for r in a if r["seed"] == 1])[0]; s2 = agg([r for r in a if r["seed"] == 2])[0]
        print(f"     S noise yardstick (base seed 1 vs seed 2): {100 * (s2 / s1 - 1):+.2f}%")
