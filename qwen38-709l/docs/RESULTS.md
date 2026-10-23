# Results

All numbers: one Arc Pro B70, the 709-L AWQ, fp8 KV, 4 streams max, vision on, requests through an OpenAI-compatible
endpoint with thinking off, MTP-5 unless a section says otherwise. Raw files are in `results/`, named next to each
table. Every run is a single pass, so treat ±5% as noise (compare `plain031` with `base` in the riceharvest table, two
configs with no riceharvest switch on).

Newest first within each topic. Older sections stay as they were measured, with a note where something replaced them.

- [Current seat](#current-seat-2026-10-23)
- [Speculative decoding](#speculative-decoding)
  - [Probabilistic MTP drafting](#probabilistic-mtp-drafting) (live)
  - [Shortlist draft head](#shortlist-draft-head) (live)
  - [Combined speculator at every depth](#combined-speculator-at-every-depth) (live)
  - [Combined copy + MTP speculator, first version](#combined-copy--mtp-speculator-first-version)
  - [DFlash2 drafter on the 709-L](#dflash2-drafter-on-the-709-l)
- [Engine versions](#engine-versions)
  - [0.31 → 0.32](#031--riceharvest-allkv--032-same-stack-2026-10-19)
  - [0.30 → 0.31 → 0.31 + riceharvest](#030--031--031--riceharvest-allkv)
  - [riceharvest layer, one switch at a time](#riceharvest-layer-one-switch-at-a-time)
- [Context and KV cache](#context-and-kv-cache)
  - [262k window](#262k-window), [max-model-len boundary](#max-model-len-boundary),
    [four long chats at once](#four-long-chats-at-once), [CPU KV offload tier](#cpu-kv-offload-tier)
- [Sharing the card: chunked-prefill sweep](#sharing-the-card-chunked-prefill-sweep)
- [Against Vcvzgbzz's own tests](#against-vcvzgbzzs-own-tests)
- [Quality checks](#quality-checks)

## Current seat (2026-10-23)

What our live seat runs, and the checks it passed after its last reload (2026-10-23 1:32 AM). Each line links the
section with the full numbers and raw files.

- **Engine:** vLLM 0.32.0 with the stack in [RECIPE.md](RECIPE.md), the 709-L AWQ W4A16, fp8 KV in a 14.5 GiB pool,
  `--max-model-len 262144` (375,067 tokens of KV with the combined speculator on), 4 streams,
  `--long-prefill-token-threshold 512`.
- **Speculative decoding:** the [combined copy + MTP speculator](../patches/combo-spec/README.md) at every depth
  (MTP-5 by default, 15 copied drafts when the output copies its context, block verification), with the
  [shortlist draft head](../patches/shortlist/README.md) (N = 128) and **probabilistic drafts**
  (`draft_sample_method: "probabilistic"`).

| check | result | section |
|---|---|---|
| short code (avg of 9) | 135.0 t/s | [probabilistic drafting](#probabilistic-mtp-drafting) |
| edit bench (3 whole-file edits, greedy + sampled) | 238.8 t/s, 6/6 exact | same |
| prose, one stream, the model's default sampling (20 requests) | 63.6 t/s, 2.19 tokens per step | same |
| prose, thinking on at effort medium (10) | 74.9 t/s, 2.59 | same |
| prose, greedy (10) | 66.4 t/s, 2.24 | same |
| 4 writers sampled, summed (x2) | 292.4, 284.5 t/s | same |
| decode behind 61k / 120k, sampled, 3 seeds | 83.3 / 65.1 t/s (3.77 / 3.52 tokens per step) | same |
| decode behind 23k / 61k, greedy, ms per step (x3) | 38.3-39.1 / 44.0-44.2 | same |
| page-boundary test (`bench/crossing.py`) | 3/3 | same |
| image and video | correct | same |
| greedy code x5 | one transcript, the one the seat already produced before | same |
| whole-file edit behind 94k of context | 171.1 t/s, exact (2026-10-21, not re-measured) | [combined speculator at every depth](#combined-speculator-at-every-depth) |
| 262k max-model-len test (`bench/maxlen.py`) | pass (2026-10-21) | same |
| a chat decoding while an 83k prompt prefills | 10.7 t/s, longest stall 0.6 s (2.2 t/s, 2.9 s without the threshold) | [sharing the card](#sharing-the-card-chunked-prefill-sweep) |

Not re-measured on this exact config:

- **Decode after a cold 100k prompt, greedy** (56–60 t/s) comes from plain MTP-5 runs on 0.32; the combined
  speculator matched them wherever we checked ([at every depth](#combined-speculator-at-every-depth)), and the
  shortlist cut every measured step time by 3.4% ([shortlist](#shortlist-draft-head)).
- **First token at 100k** was about 75 s before the prefill threshold went on. The threshold makes one big cold prefill
  about 22% slower (1,326 → 1,097 t/s at 83k, [sweep](#sharing-the-card-chunked-prefill-sweep)), so expect about
  90 s now.

## Speculative decoding

### Probabilistic MTP drafting

0.32, 2026-10-23, live since 1:32 AM. Raw in `results/prob-draft/` (`offline/` the replay, `seat-ab/` the A/B on the
second card, `promotion/` the live seat before and after). This is the live config.

vLLM's default `draft_sample_method` is `greedy`: each MTP draft is the argmax of the drafter's distribution q, and a
request that samples accepts it with probability p(draft). With `probabilistic`, the draft is sampled from q and
accepted with probability sum min(p, q), which recovers nearly all the acceptance that sampling costs. Greedy requests
skip draft sampling and do not change. Two things had to be fixed first ([combo-spec](../patches/combo-spec/README.md),
[shortlist](../patches/shortlist/README.md#probabilistic-drafts)): the copied drafts needed one-hot draft-logit rows,
and the shortlist head needed to feed the sampler a logits row instead of stepping aside.

**What the server actually samples with:** our clients send no sampling fields, and with no `--generation-config`
flag vLLM fills them from the model's `generation_config.json`: temperature 1.0, top-k 20, top-p 0.95. So every
ordinary chat and agent turn on this seat is a sampled request; only our greedy benches are not.

**Offline replay** (`offline/held_K5.log`: 86,010 teacher-forced K = 5 chain starts from 8 held-out sessions of our
own traffic, the seat's int4 draft head, the target's stored top-20 at temperature 1.0 / top-k 20 / top-p 0.95):
chain tokens per step greedy -> probabilistic: all 4.693 -> 4.796 (+2.2%), code +1.8%, prose +4.0%; the gain grows as
acceptance falls (+10% on the one session at 3.0 tokens per step). At temperature 0.6 the gains halve.

**Seat A/B** on the second card (`seat-ab/base.out`, `prob.out`, `compare-*.out`, `prose_pd-*.jsonl`; base = the live
config of the day, greedy drafts):

| | greedy drafts | probabilistic | change |
|---|---|---|---|
| prose, default sampling, 20 requests: tokens per step / decode t/s | 1.891 / 54.8 | **2.240 / 64.4** | **+18.5% / +17.7%** (seed-to-seed noise 4.2%) |
| prose, thinking on (medium), 10 | 2.278 / 65.7 | **2.578 / 74.0** | **+13.2% / +12.5%** |
| prose, greedy, 10 | 2.256 / 66.5 | 2.268 / 66.6 | +0.5% / +0.1% |
| 4 writers sampled, summed (x2) | 260.2, 260.6 | **279.2, 275.6** | **+7.0%** |
| greedy decode behind 23k / 61k / 120k, ms per step | 38.7 / 44.8 / 54.3 | 38.7 / 44.7 / 54.5 | no step-time cost |
| short code, greedy (avg of 9) | 133.3 | 133.8 | +0.4% |
| edits, greedy and sampled (6 exact) | 238.7 (6/6) | 233.9 (6/6) | -2.0% (one-hot rows on copy steps) |
| crossing, greedy code x5 | 3/3, 5/5 identical | 3/3, 5/5 identical to base | |

The served gain on prose is about 4x the offline estimate because served sampled prose starts at 1.9 tokens per
step, far below the replay's 4.2, and the gain rises as acceptance falls. Sampled prose now matches greedy prose.

**Live seat, before and after** (`promotion/live-pre.out`, `live-post.out`, `live-compare.out`, `live-verify.out`,
`prose_live-*.jsonl`, `tier_live-*.jsonl`; all through our router; prod was down 5 min 50 s for the reload):

| | before (shortlist, greedy drafts) | **after (live)** | change |
|---|---|---|---|
| prose, default sampling (20): tokens per step / t/s | 1.909 / 55.8 | **2.186 / 63.6** | **+14.5% / +14.0%** (noise 3.3%) |
| prose, thinking on (10) | 2.338 / 67.9 | **2.590 / 74.9** | **+10.8% / +10.2%** |
| prose, greedy (10) | 2.252 / 67.0 | 2.237 / 66.4 | -0.7% / -0.9% |
| decode behind 61k, sampled, 3 seeds | 3.54 / 2.97 / 3.14 (mean 3.22), 70.9 t/s | **3.77 / 3.80 / 3.73 (3.77), 83.3 t/s** | **+17%**, every seed up |
| decode behind 120k, sampled, 3 seeds | 2.90 / 3.06 / 3.33 (3.10), 57.0 t/s | **3.37 / 3.49 / 3.70 (3.52), 65.1 t/s** | **+14%**, every seed up |
| 4 writers sampled, summed (x2) | 265.0, 270.2 | **292.4, 284.5** | **+7.8%** |
| short code, greedy (avg of 9) | 135.0 | 135.0 | same |
| greedy decode behind 23k / 61k, ms per step (x3) | 38.2-38.6 / 44.1-44.2 | 38.3-39.1 / 44.0-44.2 | no step-time cost |
| edits, greedy and sampled | 240.7, 6/6 exact | 238.8, **6/6 exact** | -0.8% |
| crossing / vision / greedy code x5 | 3/3 / correct / 4x one transcript + 1x the other | 3/3 / correct / 5x one transcript | both transcripts are ones the seat produced before |

Seven agent turns through our harness before and after (served tokens per step from the seat's counters,
`promotion/live-harness-turns.log`; every reply read in full, no breakage): the four prose turns went from 2.37 to
2.70 tokens per step (+14%), the three code turns 4.11 / 6.57 / 5.84 -> 4.34 / 6.72 / 6.66. One caveat on that
measurement: our harness has cross-chat memory, which surfaced the "before" chat's replies to the "after" chat on two
of the prose turns, so those two are not independent samples (turns 4 and 7 were unaffected and gained 11%).

The sampled-depth runs are single seeded texts per depth and showed copy steps on one of them, so the A/B's depth
rows were inconclusive; the three-seed runs on the live seat settle it. Edits lose about 1-2% on the sampled copy
path (the one-hot rows) and stay exact.

### Shortlist draft head

0.32, 2026-10-22, live since 10:16 PM that day. Raw in `results/shortlist/` (`offline/` the replay and the artifact
gate, `microbench/` the head step on the card, `seat-ab/` the A/Bs on the second card, `promotion/` the live seat
before and after). Design and build: [patches/shortlist/README.md](../patches/shortlist/README.md).

The MTP drafter reads its whole int4 draft head (98,816 x 5,120, 261 MB) once per draft pass to take an argmax. The
shortlist scores every row with a covariance-weighted rank-512 factorization, keeps 128 candidates and rescores only
those with the real head. Offline (`offline/artifact_r512_eval.log`): -0.11% chain tokens per step. Plain SVD and
frequency-trimmed heads fail (`offline/sweep1_int4.log`, [README](../patches/shortlist/README.md#why-weighted-and-why-not-a-trimmed-head)).

**Head step on the card** (`microbench/kbench-int4-rescore.out`, captured graph, microseconds per call):

| streams | stock int4 head GEMM + scatter + argmax | shortlist N = 128 | saved |
|---|---|---|---|
| 1 | 490 | **162** | 328 |
| 2 | 495 | 164 | 331 |
| 4 | 502 | 169 | 333 |

0.33 ms per draft pass = about 1.65 ms per K = 5 step. The first version rescored the candidates with an fp32
dequantize (205 / 237 / 297 us, `microbench/kbench-fp32-rescore.out`); rescoring with the head's own int4 GEMM on the
gathered rows is 17 us and uses the stock head's numerics.

**Seat A/B** on the second card, same day (`seat-ab/base.out`, `sl.out`, `tier_sl-*.jsonl`; a determinism rerun with
3 depth runs and 5 greedy code runs per arm in `det-base.out`, `det-sl.out`):

| | stock head | shortlist | change |
|---|---|---|---|
| short code (avg of 9) | 129.9 t/s | **134.6** | **+3.6%** (rerun 130.6 -> 135.0) |
| decode behind 23k / 61k / 120k, ms per step | 40.1 / 46.2 / 56.4 | **38.7 / 44.5 / 54.3** | **-3.5 / -3.7 / -3.7%** (rerun x3: -3.3 / -3.3% at 23k / 61k) |
| tokens per step at 23k / 61k (rerun, x3 each) | 3.25 / 3.54 | 3.22 / 3.53 | -0.7 / -0.3% |
| 4 writers sampled, summed (x2) | 258.4 avg | 260.3 avg | noise |
| edits, greedy and sampled | 233.8, 6/6 exact | **239.0, 6/6 exact** | +2.2% |
| crossing | 3/3 | 3/3 | |
| acceptance over the bench | 66.5% | 67.3% | |
| greedy code, distinct transcripts of 5 | 2 | 2, **the same two** | the shortlist adds no greedy behaviour the seat did not have |

The greedy code prompt flips between two transcripts on the stock head too (2 of 5 runs), at a token where the
target's top two are 0.016 nats apart: speculative decoding is lossless in distribution, but how tokens group into
verify steps changes the numerics, so an exact near-tie can go either way under greedy. Byte-identical greedy output
is not a usable gate once drafts change ([PITFALLS](PITFALLS.md#speculative-decoding)).

**Prose A/B** on the second card, pre-shortlist vs shortlist (`seat-ab/prose-base.out`, `prose-sl.out`,
`prose_base.jsonl`, `prose_sl.jsonl`; `bench/prose_ab.py`, 10 fixed prompts):

| mode | tokens per step | decode t/s |
|---|---|---|
| greedy, thinking off (600 tokens) | 2.337 -> 2.276 (**-2.6%**; -2.3% on the 6 of 10 identical texts) | 66.1 -> 67.1 (+1.5%) |
| the model's default sampling, 2 seeds (500) | 1.943 -> 1.941 (-0.1%) | 53.9 -> 56.2 (+4.2%) |
| thinking on, effort medium (1,000) | 2.350 -> 2.278 (**-3.1%**, per-prompt sd 11.6) | 65.0 -> 65.7 (+1.1%) |

So the shortlist does cost 2-3% acceptance on short-context greedy and thinking prose, where the offline replay
(code-heavy traffic) said -0.1%, and nothing on sampled prose. No mode is slower.

**Live seat, before and after** (`promotion/live-pre.out`, `live-post.out`, `live-verify.out`, `tier_live-*.jsonl`;
through our router; prod down 3 min 16 s):

| | before | **after (live)** | change |
|---|---|---|---|
| short code (avg of 9) | 129.6 | **134.9** | **+4.1%** |
| decode behind 23k / 61k, ms per step (x3) | 39.8 / 45.7 | **38.5 / 44.1** | **-3.4 / -3.4%** |
| decode behind 23k / 61k, t/s | 83.6 / 81.8 | 93.0 / 84.2 | +11 / +3% (23k carries tokens-per-step noise) |
| 4 writers sampled, summed (x2) | 263.8, 265.3 | 264.9, 270.6 | +1.2% |
| edits, greedy and sampled | 237.1, 6/6 exact | **242.6, 6/6 exact** | +2.3% |
| crossing / vision / greedy code x5 | 3/3 / correct / 5x one transcript | 3/3 / correct / 5x the same transcript | |
| acceptance over the bench | 68.0% | 66.9% | -1.1 points |

Agent turns through our harness, one each before and after: write 3.87 -> 4.19 tokens per step, whole-file rewrite
6.91 -> 6.63, small edit 5.19 -> 4.37, prose 2.89 -> 2.62. Single turns of different generated text cannot separate
acceptance from content; the fixed-prompt prose A/B above is the clean measurement.

### Combined speculator at every depth

0.32, 2026-10-21 evening: MTP draft count, copy draft count, cutoff, packed 16, block verification. Raw in
`results/combo-depth/`, lab notes in [TIERS-RESULTS.md](../results/combo-depth/TIERS-RESULTS.md). This is the live
config.

The first combined seat copied only below 32k tokens. In an agent harness that never happens: our own harness's system
prompt and tools alone are ~40k tokens, so every agent request ran plain MTP-5. One window on the first card, two
loads, with the knobs switched at runtime through `B70_COMBO_CTL` (`bench/tierbench.py`, `bench/snap.py`, `patches/combo-spec/tests/pk16.py`):

**MTP draft count by depth** (copy off; greedy "explain the last file" at 23k / 61k / 120k, t/s and ms per step; 4
writers sampled at temperature 1.0):

| MTP K | 23k | 61k | 120k | 4 writers summed (tok/step) |
|---|---|---|---|---|
| **5 (packed verify)** | **84.8** (39.5 ms) | **77.8** (45.5) | **67.4** (54.8) | 263.3 (2.85) |
| 4 | 79.0 (38.7) | 74.0 (44.6) | 63.7 (53.6) | 254.1 (2.73) |
| 3 | 77.6 (36.3) | 72.9 (41.2) | 64.2 (48.6) | 264.6 (2.64) |
| 2 (packed) | 74.5 (33.1) | 70.5 (36.5) | 61.3 (41.7) | 253.3 (2.25) |

Fewer drafts save less time per step than they lose in acceptance, at every depth. K 4 and 3 also lose packed verify
(their verify lengths 5 and 4 do not pack). No MTP tiers.

**Copy at depth** (the 3.6k-token E3 bugfix answer behind filler, greedy, every run exact):

| | 47k | 94k |
|---|---|---|
| MTP-5 | 130.2 | 110.7 |
| copy 10 | 168.3 | – |
| copy 15 | 204.4 | (the scan stopped at 64k: no source found) |
| **copy 15, packed q16, scan the whole window** | **216.6** (11.4 tok/step, 52.6 ms) | **171.0** (11.2, 65.6 ms) |

Copy 15 wins at every depth, so there are no copy tiers either; the cutoff only has to go. Packed verify at q_len 16 is
exact (`pk16.py`: identical to the stock path at 2k-115k) and saves ~3 ms per copy step. Lifting the cutoff cost nothing
on non-copy work: deep 82.8-94.5 / 77.9-78.4 / 65.5-67.4 vs 84.8 / 77.8 / 67.4.

**Agent turns through our harness** (effort medium, decode t/s from the seat's counters; generated code differs from
chat to chat, so read the copy-step share and tokens per step more than single t/s):

| turn | cutoff 32k (old prod) | cutoff lifted | + packed 16 + block (new prod) |
|---|---|---|---|
| write a new module (~40k context) | 95.4 (4.07 tok/step, 0% copy) | 107.5 (4.73, 8%) | 97.8 (4.20, 3%) |
| rewrite the whole file | 121.6 (5.30, 0%) | 151.8 (7.40, 30%) | **166.6** (8.01, 38%) |
| natural edit | 89.6 (3.99, 0%) | 116.1 (5.60, 17%) | – |
| prose explanation, sampled (~55k) | 60.1 (2.70, 34% accepted) | 56.6 (2.60, 32%) | 63.3 (2.83, 37%) |

Block verification (`rejection_sample_method: "block"`) kept greedy output exact (edit bench 6/6) and did nothing for
the four writers (263.8 vs 263.3, 2.82 vs 2.85 tok/step); the sampled agent prose turn came out 5-12% faster, a single
run. We kept it: no cost seen anywhere.

New prod (2026-10-21 10:04 PM): `B70_COMBO_MAX_CTX=262144 B70_COMBO_SCAN_CTX=262144`, packed verify for q_len 2/3/6/16,
block verification. Checked after the reload: edit bench 235.7 (6/6), short code 130.9, crossing 3/3, the 94k copy edit
171.1 (exact), 262k max-len pass (load 2), and copy steps now show up in agent turns.

### Combined copy + MTP speculator, first version

0.32, 2026-10-21, `patches/combo-spec/`, raw in `results/combo-spec/`. Copy only below 32k tokens; [replaced the same evening](#combined-speculator-at-every-depth).

One engine that verifies 15 copied drafts when the output copies its context and runs MTP-5 otherwise (design in
[patches/combo-spec/README.md](../patches/combo-spec/README.md)). All through our router, greedy unless noted. The edit
bench is `bench/editbench.py`: E1 renames a variable in a 1.5k-token test file, E2 adds a parameter to a 3.3k-token
module, E3 fixes an injected bug in a 3.6k-token script. Each task runs greedy and sampled, and the returned file must
match exactly.

| config | edits t/s (exact) | short code | deep 23k / 58k / 115k | rhbench short / 4 writers | crossing | 262k max-len |
|---|---|---|---|---|---|---|
| prod MTP-5 (packed verify) | 155 (6/6) | 131.7 | 84.2 / 82.1 / 66.5 | 132.2 / 272.3 | 3/3 | pass |
| n-gram K=15 alone (2026-10-21 probe) | 267 (6/6) | 33.1 | 22.7 / 20.5 / 15.8 | 43.0 / 70.2 | 3/3 | – |
| combined, first policy (match only; debug host sync on) | 227 (6/6) | 124.2 | 77.4 / 84.1 / 65.1 | 124.6 / 271.3 | 3/3 | pass |
| combined, policy checks the match offset holds | 205 (6/6) | 130.5 | 76.2 / 84.0 / 64.1 | 132.5 / 279.0 | 3/3 | pass |
| **combined, policy checks the source continues (live)** | **233 (6/6)** | **131.5** | 78.8 / 84.1 / 65.1 | 128.3 / 265.4 | 3/3 | pass |
| live seat after promotion | **235 (6/6)**; E1 185, E2 242, E3 279 | 131.0 | – | 128.4 + 129.7 / 272.4 + 266.7 | 3/3 | – |

Reading:
- **Edits +50%, everything else at MTP-5.** On E3, where most of the file comes back unchanged, the live seat runs
  279 t/s. On E1 (a rename every few lines) it runs 185, because each edit point costs a few MTP steps before the copy
  resumes.
- **Policy, version 1:** "the last 4 tokens occur earlier" fires on fresh code that repeats its own patterns. Short
  code lost 5%: each false copy step pays a 16-row verify for an MTP-sized gain.
- **Policy, version 2:** requiring the match offset to hold across two checks fixed fresh code. It also broke real
  copies, because code repeats 4-grams and the rightmost match keeps jumping to a nearer repeat (edits dropped to 205).
- **Policy, live version:** checking that the newly generated tokens equal what follows the source fixes both.
- **23k single runs are noisy:** prod gave 84.2 on its own and 93.5 straight after the crossing test, because greedy
  output shifts with prefix-cache state. The fair comparison is an A/B inside one load with the policy switched off by
  a control file: 83.1 off, 82.3 on (−1%).
- **Depth:** above 32k the policy never copies (a 16-row verify at 115k costs ~80 ms), and 58k/115k match prod.
- **4 writers:** nearly identical to prod. Four requests rarely all copy at once, and the batch only copies when they
  all do.
- **Full MTP-5 on copy steps** (`B70_COMBO_MTP_IN_COPY=1`) instead of only its first pass: no measurable difference
  on edits under the same (offset) policy, 207 with it vs 205 without. We kept first-pass-only; it has not been
  measured on the live policy. (The first-policy row cannot be compared with it: that run also had a debug host sync
  per copy step left on.)

### DFlash2 drafter on the 709-L

0.32, `results/engine/vllm032-dflash2-stock-drafter.json`. Training a drafter for the merge: [DRAFTER-TRAINING.md](DRAFTER-TRAINING.md).

[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), trained on base Qwen3.8, through the
re-based patches in `patches/dflash2/` (BF16 drafter under the FP16 target, int4 drafter body, int4 draft head,
sliding-window prefix-cache fix), 7 drafts, stock attention kernel, same bench and 262k window as MTP-5:

| | MTP-5 | DFlash2-7 |
|---|---|---|
| short code | 130.8 t/s, 72.8% per draft token, 4.64 tok/step | 37.7 t/s, 5.1%, 1.35 |
| 4 writers, summed | 259.2 t/s, 2.94 tok/step | 96.2 t/s, 1.27 |
| decode 24k / 65k / 100k | 63.0 / 67.1 / 58.1 t/s | 27.9 / 22.1 / 18.1 t/s |
| first token 100k, cold | 75.1 s | 97.0 s (stock kernel) |

Per draft position on a code reply: 23%, 8.6%, 2.8%, 0.7%, then 0. Lossless: page-boundary test 3/3, image and video
correct. A control with the stock head paths (no fp8 target head, no int4 draft head, BF16 drafter body) gave 2.9%
per draft token, so the B70 changes are not the cause: the drafter does not recognise the merge's hidden states.
The step costs the same as MTP-5's on short code (35.8 vs 35.4 ms), so the whole gap is acceptance.

## Engine versions

### 0.31 + riceharvest ("allkv") → 0.32, same stack (2026-10-19)

Same bench, same box, same day, prod config (14.5 GiB KV), single runs (`results/engine/vllm032.json`, a second run
`results/engine/vllm032-run2.json` after the KV-offload flag was dropped, vs `results/per-patch/allkv.json`). The 0.32
stack drops the four riceharvest patches 0.32 made obsolete (RECIPE.md) and carries SergiioB's boundary fix re-based.

| | 0.31 allkv | 0.32, run 1 | **0.32, run 2** |
|---|---|---|---|
| short code, t/s ×3 | 126.7 / 136.3 / 131.7 (131.1) | 127.4 / 135.7 / 132.1 (131.7) | 119.8 / 134.8 / 132.3 (129.0) |
| 4 writers, summed t/s | 272.7 | 267.2 | 274.7 |
| prefill 24k / 32k / 65k / 74k / 100k, t/s | 1,784 / 1,712 / 1,482 / 1,425 / 1,291 | 1,633 / 1,675 / 1,450 / 1,423 / 1,289 | 1,781 / 1,731 / 1,497 / 1,439 / 1,303 |
| decode 24k / 32k / 65k / 74k / 100k, t/s | 66.0 / 61.6 / 65.2 / 61.0 / 58.4 | 66.2 / 57.0 / 61.9 / 55.0 / 56.1 | 63.5 / 68.0 / 69.5 / 58.8 / 60.3 |
| first token 100k, cold | 75.8 s | 75.9 s | 75.1 s |
| cites at 65k / 74k / 100k | 4 / 4 / 4 | 4 / 4 / 4 | 4 / 4 / 4 |
| image (2,251 prompt tokens) | 1.5 s | 3.2 s (first image after the load) | 3.3 s |
| video (4 s clip) | 0.2 s | 0.6 s | 0.6 s |
| greedy hashes vs allkv | — | code and list identical; prose differs at char 86 (engine change, same as plain-0.31 vs base) | same |
| KV pool | 389,315 tokens | 389,315 tokens | 389,315 tokens |
| MTP acceptance, code | 73% | 73.5% (4.67 tok/step) | 71.8% (4.59) |

Parity within the ±5% single-run noise on every row; run 1's 74k decode (−10%) did not repeat. Cold load with a warm
compile cache: 241 s. 0 GPU resets across the day's loads once the drafter's embedding table stayed on the GPU.

### 0.30 → 0.31 → 0.31 + riceharvest ("allkv")

Same script for all three (`results/engine/`):

| | 0.30 | 0.31 | allkv |
|---|---|---|---|
| short code, t/s ×3 | 115.7 / 107.0 / 119.1 | 114.2 / 121.5 / 122.6 | 126.7 / 136.3 / 131.7 |
| 4 writers, summed t/s | 221.3 | 236.6 | 265.0 |
| 4 writers, total gen / wall | 152.8 | 136.0 | 168.3 |
| 75k cold read, first token | 66.9 s | 66.8 s | 53.8 s |
| 75k prefill | 1,133 t/s | 1,134 | 1,408 |
| 75k decode ×2 | 50.6 / 44.7 | 49.1 / 46.5 | 59.8 / 63.0 |
| 75k cached replay, first token | 0.57 s | 0.55 s | 0.42 s |
| image (2,251 prompt tokens) | 1.6 s | 3.4 s | 1.5 s |
| video (4 s clip) | 0.6 s | 0.6 s | 0.2 s |

The 3.4 s image time on 0.31 is a one-off: the first image after a load pays about 2–3 s of kernel JIT, and that run
happened to send its image cold.

"Summed" adds the four streams' individual speeds, while "total gen / wall" divides all tokens generated by elapsed
time. They differ when reply lengths differ between runs.

### riceharvest layer, one switch at a time

`results/per-patch/summary.csv`. Each config is one load with **only** that switch on. `base` = the riceharvest
build with every switch off, stock kernel. `kern` = riceharvest's v2 attention `.so`. `packed` = kern + packed MTP
verify. `allkv` = kern + packed + fp8 head + kv-splits + embed UVA, with the KV pool raised to 13 GiB.

| config | short code | 4 summed | prefill 24k | prefill 65k | prefill 100k | decode 24k | decode 65k | decode 100k | first token 100k |
|---|---|---|---|---|---|---|---|---|---|
| plain 0.31 | 120.9 | 245.7 | 1,620 | 1,220 | 1,003 | 50.6 | 52.1 | 40.6 | 97.6 s |
| base | 122.3 | 255.2 | 1,609 | 1,217 | 1,003 | 52.8 | 55.4 | 43.3 | 97.5 s |
| kern | 125.6 | 249.4 | 1,792 | 1,488 | 1,295 | 58.1 | 61.1 | 52.9 | 75.6 s |
| packed | 127.5 | 246.7 | 1,792 | 1,487 | 1,296 | 57.0 | 66.7 | 56.6 | 75.5 s |
| fp8head | 133.2 | 271.6 | 1,611 | 1,219 | 1,004 | 57.9 | 54.5 | 48.0 | 97.5 s |
| kvsplits | 124.3 | 252.5 | 1,611 | 1,218 | 1,004 | 53.8 | 52.9 | 44.9 | 97.5 s |
| uva (embed + vision) | 124.8 | 245.2 | 1,598 | 1,212 | 1,000 | 55.3 | 56.7 | 44.7 | 97.9 s |
| uva (embed only) | 125.4 | 258.1 | 1,603 | 1,214 | 1,001 | 52.3 | 56.7 | 46.0 | 97.8 s |
| **allkv** | **131.1** | **272.7** | **1,784** | **1,482** | **1,291** | **66.0** | **65.2** | **58.4** | **75.8 s** |

What each switch does:

- **kernel:** all of the prefill gain (+11% at 24k up to +29% at 100k) and most of the decode-at-depth gain.
- **packed MTP verify:** more decode at depth on top of the kernel (+9% at 65k, +7% at 100k). Requires the v2 kernel.
- **fp8 target lm_head:** the only switch that helps short work (+9% code, +6% four streams).
- **kv-splits, embed UVA:** no speed change. UVA frees VRAM instead, which the KV pool can use.
- **UVA with the vision tower too** (`--cpu-offload-params embed_tokens visual`): text speed is the same, but an image
  request took **20.3 s instead of about 3 s**. Offload the embeddings only.
- **GDN sync patches** (`B70_GDN_ZERO_DRAFT`, `B70_GDN_MIXED_NO_SYNC`): the card hung twice (GPU copy-engine reset). Not measured, not used.

MTP acceptance doesn't change with any switch: 71–76% per draft token on code. The corruption canary was clean in
every run, with 0 preemptions throughout.

## Context and KV cache

### 262k window

0.32, `--max-model-len 262144`, `results/engine/vllm032-ctx262k.json`.

Same bench on the same seat with the window raised from 131,072 to 262,144 (KV pool 416,591 tokens, same 14.5 GiB):
short code 130.8 t/s, 4 writers 259.2 summed, prefill 24k / 32k / 65k / 74k / 100k = 1,797 / 1,737 / 1,502 / 1,441 /
1,303 t/s, decode 63.0 / 67.9 / 67.1 / 53.9 / 58.1 t/s, cites 4 / 4 / 4 at 65k+. Parity with the 131k runs above.
A 139k-token prompt is admitted and answered in 124 s.

### Max-model-len boundary

0.32, `bench/maxlen.py`.

`results/context/maxlen_032.json`: a 128,977-token prompt with `max_tokens: 8000` on the 131,072 window. The reply stopped at
exactly 131,072 total tokens (2,095 generated, `finish_reason: length`), no `!!!!` runs, no replacement characters, and
the server answered the follow-up. This is the case `patch_mtp_boundary_032.py` exists for: the last MTP step is a
partial speculative group right at the window. Decode at that depth, prefix cached: 2,095 tokens in 21.8 s (96 t/s).

### Four long chats at once

`bench/stress4.py`: 4 distinct prompts of 93k–102k tokens sent at once, 14.5 GiB pool (389,315 tokens). Raw in
`results/context/`.

| engine | raw | answers | preemptions | KV peak | wall |
|---|---|---|---|---|---|
| 0.32 (2026-10-19) | `stress4_032.json` | 4/4 correct | 0 | 50% | 321 s |
| 0.31 + riceharvest | `stress4_kv145.json` | 4/4 correct, no OOM | 0 | 49% | 316 s |

Wall time is nearly all prefill. vLLM prefills one long prompt at a time, so the others wait and the first one
finishes before the last one starts. The KV pool, not `--max-num-seqs 4`, is what limits concurrent long chats.

### CPU KV offload tier

0.32, `bench/kvoff_probe.py`.

`results/context/kvoff_probe_032.json`: a 55,564-token prompt, then 7 distinct 55k–64k prompts (405k tokens, more than the
389k pool), then the first prompt again. The replay reported `cached_tokens: 0` and took the full 36.4 s again, the
same as cold. The same probe with a 20k prompt replayed immediately shows 19,776 cached tokens in 0.2 s, so the field
works; the CPU tier just never returns a block. `--kv-offloading-size 8` is off the serve line.

## Sharing the card: chunked-prefill sweep

One big cold prompt vs everyone else. 0.32, `bench/starve.py`.

vLLM prefills a long prompt in chunks of `--max-num-batched-tokens` (3,584 here, about 2.7 s per chunk at depth) and
decode requests get one step per chunk. So while a 100k prompt prefills, every other chat on the seat decodes at
about 3 t/s. `--long-prefill-token-threshold N` caps how many tokens of a long prompt go into one step, leaving the
rest of the step budget for decodes. Four configs, each loaded through the test switch with the live prod config plus
the flags shown; raw JSON in `results/prefill-sweep/`.

**A chat is decoding when an 83k-token cold prompt arrives** (`starve.py`: A = 2,500-token code generation, B = the
cold prompt 6 s in):

| config | A before B | **A while B prefills** | A's longest stall | B's prefill | prefill t/s |
|---|---|---|---|---|---|
| live: batched 3584, no threshold | 91.5 t/s | **2.2 t/s** | 2.9 s | 62.5 s | 1,326 |
| 3584 + threshold 512 | 75.4 | **10.2** | 0.6 s | 76.4 s | 1,085 |
| batched 2048 + threshold 512 | 78.0 | **11.1** | 0.6 s | 76.2 s | 1,087 |
| batched 1024 + threshold 256 | 87.1 | **18.4** | 0.5 s | 92.9 s | 893 |

**One 112k-token chat plus three 47k chats sent together** (a four-chat script we have not published, built on the prompts of our
262k context test; the three small ones arrive 3 s after the big one):

| config | small chats' first token | small chats' decode while sharing | big chat's first token | all 4 running at once |
|---|---|---|---|---|
| live | 122 / 154 / 178 s | ~3 t/s | 93 s | no (max 2) |
| threshold 512 | 125 / 125 / 125 s | 7.8 | 184 s | yes |
| 2048 + 512 | 124 / 124 / 124 s | 7.9 | 186 s | yes |
| 1024 + 256 | 123 / 123 / 123 s | 11.1 | 189 s | yes |

All runs: every answer correct (6/6 current values, 0 stale), 0 preemptions, KV peak 40–62%.

Reading: the threshold is a real lever for the "agent chat mid-answer while a teammate's big prompt lands" case: a
512-token cap turns a 2.2 t/s crawl with 3 s stalls into 10 t/s with sub-second stalls, for 22% more cold prefill time
on the big prompt (and 2× on the big chat's first token when four arrive together, since the chunks interleave). The
1024/256 config buys another 1.8× decode during prefill for 1.5× slower cold prefill. It does not get the small chats
their first token sooner when all four arrive at once: FCFS still prefills in arrival order, the chunks just interleave.
We first left the live seat on the default (no threshold), because single-stream first-token time is what we optimise
first. On 2026-10-21 we turned on `--long-prefill-token-threshold 512`, because the seat serves several agent chats at
once. Checked live: a chat decoding while an 83k prompt prefills went from 2.2 to 10.7 t/s, its longest stall from 2.9
to 0.6 s; short code unchanged (132.0); the big prompt's own prefill 1,326 → 1,097 t/s.

## Against Vcvzgbzz's own tests

Our MTP seat on his `tests/` scripts, raw in `results/vcvzgbzz-tests/`.

To check whether the speeds in [Vcvzgbzz's cookbook](https://github.com/Vcvzgbzz/b70-cookbook) are within reach,
his `tests/` scripts (dd083d5, unchanged apart from an auth header) ran against our live MTP-5 seat at the 262k
window. His model is base Qwen3.8-27B in AutoRound int4; ours is the 709-L merge in AWQ.

| his test | his MTP-5 | our MTP-5 | his DFlash2 |
|---|---|---|---|
| `short_code.py`, greedy, 9 runs | 124.6 t/s | **130.8** | 160.3 |
| `deep_decode.py`, decode after a cached 23k / 55k / 110k prompt | 87 / 86 / 75 | **84.1 / 82.0 / 64.5** (23k / 58k / 115k) | 86 / 81 / 87 |
| `prefill_ab.py`, cold first token at ~30k / ~120k | 17.8 s / 101.9 s | **17.7 s / 105.8 s** (31k / 125k; same speed per token) | 19.8 / 111.4 s |
| `arrivals.py`, total with 3 requests arriving mid-decode | 69-70 | **56.9** | 74-75 |

On the same tests the MTP seats are level: short code 5% faster, prefill the same per token, deep decode within
3-5% up to 55k. Two real gaps: decode at ~110k (-14%) and the arrivals test (-18%, one arriving request decodes at
6 t/s while another's 80k prompt prefills; `--long-prefill-token-threshold 512` targets exactly this, see the sweep
above). His headline 160 / 216 t/s is the DFlash2 column, which needs a drafter trained for the model it serves
([DRAFTER-TRAINING.md](DRAFTER-TRAINING.md)).

Our own bench reads lower at depth (58-67 t/s) because it is a harder test: cold prompts, temperature 0.3 and a
prose answer, against his cached prompt, greedy decoding and "explain the last file function by function".

## Quality checks

- **Greedy hashes:** the code and list prompts give byte-identical output in every config. The prose prompt differs
  between some configs (it already differed between plain 0.31 and `base`, which have no switch on), so treat it as
  numerics noise, not a regression.
- **Retrieval at depth:** at 65k, 74k and 100k every config cited 4/4 of the argparse helpers.
- **The 24k/32k "cites" numbers in the raw JSON mean nothing.** In the first version of the bench, `argparse.py`
  started 134,661 characters into the mix, so it wasn't in the 24k or 32k slices, and the model was answering from
  memory. A 6× rerun at 32k gave allkv `[1,3,2,3,2,3]` and plain 0.31 `[3,3,3,2,2,2]`, which is noise
  (`results/quality/rerun32_*.json`). `bench/bench.py` now puts argparse first and records `names_in_prompt`.
- **Page boundary** (`bench/crossing.py`): 3/3 clean on 0.30, 0.31 and allkv.
