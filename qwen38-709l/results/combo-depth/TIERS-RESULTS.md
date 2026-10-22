# 709L depth tiers, cutoff, packed-16, block verification: card-A window 2026-10-21 9:06-10:00 PM
Through hearth unless noted. Runner tw.sh (steps), t_l1.sh / t_l2.sh, tierbench.py, snap.py (DSH turns). Raw: t_l1.log,
t_l2.log, tier_*.jsonl, ed_l1base/ed_l2, rh_l1_mtp*, rh_l2, ml_l2, pk16.out. Prod restored 10:00 PM.

## Packed verify at q_len 11/16 (pk16.py, card A, B=1/2, ctx 2k/45k/115k)
Identical to the stock per-position path (max diff 0.00000-0.00002, same as the q_len 6 control) -> PASS. 1.2-1.3x
faster at depth (115k, B=1, one layer: q16 2.91 -> 2.40 ms), 1.4-2.0x at 2k.

## MTP draft count by depth (copy off; single stream, greedy explain-the-file; t/s, ms/step)
| K | 23k | 61k | 120k | 4 writers (sampled t=1.0), summed / tok/step |
|---|---|---|---|---|
| **5 (packed q6)** | **84.8** (39.5 ms) | **77.8** (45.5) | **67.4** (54.8) | 263.3 / 2.85 |
| 4 (unpacked q5) | 79.0 (38.7) | 74.0 (44.6) | 63.7 (53.6) | 254.1 / 2.73 |
| 3 (unpacked q4) | 77.6 (36.3) | 72.9 (41.2) | 64.2 (48.6) | 264.6 / 2.64 |
| 2 (packed q3) | 74.5 (33.1) | 70.5 (36.5) | 61.3 (41.7) | 253.3 / 2.25 |
-> MTP-5 wins at every depth. No MTP depth tiers.

## Copy at depth (E3 bugfix, 3.6k-token whole-file answer behind filler, greedy, exact)
| | 47k | 94k |
|---|---|---|
| MTP-5 (copy off) | 130.2 | 110.7 |
| copy 10, unpacked | 168.3 | (invalid: scan limit) |
| copy 15, unpacked (L1) | 204.4 | (invalid: scan limit 64k) |
| **copy 15, packed q16, full scan (L2)** | **216.6** (11.4 tok/step, 52.6 ms) | **171.0** (11.2 tok/step, 65.6 ms) |
All exact. -> copy 15 beats copy 10 and MTP at every depth; no copy-K tiers. The worker's n-gram scan must cover the
whole window (B70_COMBO_SCAN_CTX=262144; the default 64k found no source at 94k).

## Cutoff (deep explain-the-file, greedy): copy allowed everywhere costs nothing
MTP-5 84.8 / 77.8 / 67.4; cutoff lifted L1 82.8 / 77.9 / 65.5; L2 (+packed16, block) 94.5 / 78.4 / 67.4.

## DSH turns (tests workspace, LLM:Fable-709L, effort medium; decode t/s from the seat's counters, copy-step share)
| turn | A: prod (cutoff 32k) | B: cutoff lifted | C: B + packed16 + block |
|---|---|---|---|
| fresh write (~40k ctx) | 95.4, 4.07 tok/step, 0% | 107.5, 4.73, 8% | 97.8, 4.20, 3% |
| whole-file rewrite (~45k) | 121.6, 5.30, 0% | 151.8, 7.40, 30% | **166.6, 8.01, 38%** |
| natural edit | 89.6, 3.99, 0% | 116.1, 5.60, 17% | - |
| prose explanation (~55k, sampled) | 60.1, 2.70, 34.0% acc | 56.6, 2.60, 31.9% | 63.3, 2.83, 36.5% |
Turns are not identical across chats (different generated code), so single turns carry noise; the copy-step share and
tok/step are the robust part.

## L2 gate (packed q11/16 + block + cutoff lifted + full scan)
edit bench 235.8 6/6 exact; crossing 3/3; short code 131.1; rhbench short 130.0 / 4 writers 263.8 (2.82 tok/step);
262k max-len PASS (262,144, finish length, clean); resets 4 -> 4 through 2 loads + the kernel check.

## Verdict per depth
| context | best method |
|---|---|
| any, fresh code / prose | MTP-5 packed (K 2-4 lose at every depth) |
| any, output copies context (rewrite, refactor, re-emitted file/tool args) | copy 15, packed q16 |
| block verification | lossless; prose +5% / 4-writers 0% (single runs): neutral-to-small gain, no cost seen |
