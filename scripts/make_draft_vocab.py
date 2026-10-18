"""Write the draft-vocab JSON for riceharvest's patch_draft_vocab (B70_DRAFT_VOCAB=<path>), reproducing exactly the rows
Vcvzgbzz's integer trim keeps (B70_DRAFT_VOCAB=98304 in that cookbook): the first KEEP ids, padded so the total row count
is a multiple of 256, plus the added/special block from SPECIALS_START to the end of the vocab.

Why: both cookbooks read the SAME env var with different meanings (an int vs a JSON path), and riceharvest's fp8 target
lm_head patch reads it as a path too. A JSON list lets you run riceharvest's patches with Vcvzgbzz's proven trim.

Qwen3.8-27B defaults: vocab 248,320, specials from 248,044. KEEP 98,304 -> ids 0..98,539 + 248,044..248,319 = 98,816 rows.
Usage: python make_draft_vocab.py out.json [keep=98304] [vocab=248320] [specials_start=248044]"""
import json, sys

out = sys.argv[1]
keep = int(sys.argv[2]) if len(sys.argv) > 2 else 98304
org = int(sys.argv[3]) if len(sys.argv) > 3 else 248320
hi = min(int(sys.argv[4]) if len(sys.argv) > 4 else 248044, org)
lo = keep + (-(keep + org - hi)) % 256          # same padding rule as the integer trim
ids = list(range(lo)) + list(range(hi, org))
json.dump(ids, open(out, "w"))
print(f"{out}: {len(ids)} ids (0..{lo - 1} + {hi}..{org - 1})")
