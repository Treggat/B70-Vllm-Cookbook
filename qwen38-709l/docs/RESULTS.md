# Results

All numbers: one Arc Pro B70, the 709-L AWQ, fp8 KV, 4 streams max, vision on, requests through an OpenAI-compatible
endpoint with thinking off, MTP-5 unless a section says otherwise. Raw files are in `results/`, named next to each
table. Every run is a single pass, so treat ±5% as noise (compare `plain031` with `base` in the riceharvest table, two
configs with no riceharvest switch on).

Newest first within each topic. Older sections stay as they were measured, with a note where something replaced them.

- [Current seat](#current-seat-2026-10-25)
- [Speculative decoding](#speculative-decoding)
  - [GDN state after a backward block move](#gdn-state-after-a-backward-block-move) (live)
  - [V-split verify kernel](#v-split-verify-kernel) (live)
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
- [What we tried and dropped](#what-we-tried-and-dropped): where a decode step's time goes, and five settled experiments

## Current seat (2026-10-25)

What our live seat runs, and the checks it passed. The last reload was the GDN state fix (2026-10-25 10:45 PM);
rows dated 2026-10-23 were measured on the same config before the V-split kernel, which only changes verify step time
at depth. Each line links the section with the full numbers and raw files.

- **Engine:** vLLM 0.32.0 with the stack in [RECIPE.md](RECIPE.md), the 709-L AWQ W4A16, fp8 KV in a 14.5 GiB pool,
  `--max-model-len 262144` (375,067 tokens of KV with the combined speculator on), 4 streams,
  `--long-prefill-token-threshold 512`.
- **Speculative decoding:** the [combined copy + MTP speculator](../patches/combo-spec/README.md) at every depth
  (MTP-5 by default, 15 copied drafts when the output copies its context, block verification), with the
  [shortlist draft head](../patches/shortlist/README.md) (N = 128) and **probabilistic drafts**
  (`draft_sample_method: "probabilistic"`).
- **Verify kernel:** the [V-split verify kernel](../patches/vsplit-verify/README.md) for MTP-5 steps
  (`B70_SPEC_VSPLIT=1`); the 16-row copy steps keep riceharvest's 8-row tile.
- **GDN state fix:** [`patch_align_backward_precopy_032.py`](../patches/combo-spec/patch_align_backward_precopy_032.py)
  (`B70_ALIGN_BACKWARD_PRECOPY=1`), since 2026-10-25 10:45 PM.

| check | result | section |
|---|---|---|
| decode behind 64k / 105k / 210k, greedy | 91.2 / 92.1 / 63.6-64.8 t/s (41.8 / 49.7 / 55.8-56.0 ms per step) | [V-split](#v-split-verify-kernel) |
| short code (avg of 9), 2026-10-25 | 134.4-134.7 t/s | same |
| `bench/bench.py`: short code / 4 writers / cold 100k read | 142.4 / 286.8 t/s / 71.0 t/s, first token 77.0 s | same |
| greedy code probe x2 | byte-identical to the seat before the V-split | same |
| sampled page-boundary test, 60 seeded runs (`crossing_trace.py`), 2026-10-25 | 0 of 60 wrong (3 of 60 before the GDN state fix) | [GDN state](#gdn-state-after-a-backward-block-move) |
| short code (avg of 9), 2026-10-23 | 135.0 t/s | [probabilistic drafting](#probabilistic-mtp-drafting) |
| edit bench (3 whole-file edits, greedy + sampled) | 238.8 t/s, 6/6 exact | same |
| prose, one stream, the model's default sampling (20 requests) | 63.6 t/s, 2.19 tokens per step | same |
| prose, thinking on at effort medium (10) | 74.9 t/s, 2.59 | same |
| prose, greedy (10) | 66.4 t/s, 2.24 | same |
| 4 writers sampled, summed (x2) | 292.4, 284.5 t/s | same |
| decode behind 61k / 120k, sampled, 3 seeds | 83.3 / 65.1 t/s (3.77 / 3.52 tokens per step) | same |
| decode behind 23k / 61k, greedy, ms per step (x3) | 38.3-39.1 / 44.0-44.2 | same |
| page-boundary test (`bench/crossing.py`), 2026-10-23 | 3/3 (sampled runs with fixed seeds) | same |
| image and video | correct | same |
| greedy code x5 | one transcript, the one the seat already produced before | same |
| whole-file edit behind 94k of context | 171.1 t/s, exact (2026-10-21, not re-measured) | [combined speculator at every depth](#combined-speculator-at-every-depth) |
| 262k max-model-len test (`bench/maxlen.py`) | pass (2026-10-21) | same |
| a chat decoding while an 83k prompt prefills | 10.7 t/s, longest stall 0.6 s (2.2 t/s, 2.9 s without the threshold) | [sharing the card](#sharing-the-card-chunked-prefill-sweep) |

Not re-measured on this exact config:

- **Decode behind 61k / 120k, sampled** (83 / 65 t/s) is from 2026-10-23, before the V-split kernel, which cut greedy
  step time 3.2 ms at 64k and 4.4 ms at 105k ([V-split](#v-split-verify-kernel)).
- **First token at 100k** is measured again: 77.0 s for a 97,825-token cold prompt with `bench/bench.py`
  ([V-split](#v-split-verify-kernel)), against the 75 s before the prefill threshold and the about 90 s we had
  estimated for it.

## Speculative decoding

### GDN state after a backward block move

0.32, 2026-10-25, live since 10:45 PM. Raw in `results/align-precopy/`. Fix:
[`patches/combo-spec/patch_align_backward_precopy_032.py`](../patches/combo-spec/patch_align_backward_precopy_032.py)
(`B70_ALIGN_BACKWARD_PRECOPY=1`). This is the live config.

This is the fault the [V-split section](#v-split-verify-kernel) found: sampled `bench/crossing.py` replies that go
wrong near a 960-token page boundary. It was not the attention kernel.

**What the logprobs show.** `patches/combo-spec/tests/crossing_trace.py` reruns the crossing prompt with a seed per run
and asks for the target's logprobs. On a wrong reply the target's own distribution is off, starting several tokens
before the page edge: at position 3,833 (edge at 3,840) it gave the right `
` 0.63 and end-of-text 0.25, where a
healthy state gives about 1.0 (`crossing-trace-first.out`; that file's two fails at position 160 are a leading code
fence, which a later version of the script no longer counts). A wrong token the target itself was unsure about, with
the damage starting before the edge, means the state the step started from was wrong, not the sampler and not a key
at the edge.

**The mechanism.** With prefix caching, vLLM 0.32 keeps each request's GDN state in per-block columns: the running
column is `(num_computed + query_len - 1) // 960` (`preprocess_mamba_align_fused_kernel`). A combined-speculator copy
step verifies 16 tokens. If it starts a few tokens before a block edge, it moves the column to the next block, and
the state is copied there. If only a few of its drafts are accepted, the next MTP step (6 tokens) ends inside the old
block again, and the column moves **back**. vLLM then resets `num_accepted` to 1 so the kernels read the state at the
start of the column. But the backward-copy guard from SergiioB's `patch_fix_backward_copy.py` (vllm#53505), which
`apply_stack.sh` applies, returns before copying when the column moves back. So the GDN layers start from the old
column: a state that has not seen the tokens accepted in the copy step. With a constant step length the column never
moves back; the combined speculator's 16-then-6 steps can, and so can `num_speculative_tokens_per_batch_size` (15
drafts at one request, 5 at two).

**The test.** Live seat, then two test loads, the same 60 seeds in each (2000-2059), temperature 1.0, 4,000 tokens:

| config | wrong replies | where |
|---|---|---|
| live seat | 3 of 60 | all at position 2,877, the 3rd page edge, the same text and distribution on all three seeds |
| combined speculator off (plain MTP-5, constant q_len 6) | 0 of 60 | |
| live seat + the fix (`B70_ALIGN_BACKWARD_PRECOPY=1`) | 0 of 60 | |

(`crossing-prod.out`, `crossing-combo-off.out`, `crossing-fix.out`; the bench files label the two test loads `fixA` and
`fixB`.) Turning the combined speculator off removes the
fault and confirms the mechanism, but it costs the copy speed: edits behind 47k / 94k at 141.9 / 126.4 t/s
(`bench-combo-off.out`). The fix copies the running state back in the same way it copies it forward
(`state[col + num_accepted - 1] -> state[new col]`). The column it writes is always the block that holds the
request's current position, never a completed block in the prefix cache.

**Speed with the fix** (`bench-fix.out`, `bench/tierbench.py`, greedy, one stream), against the live seat the same
evening ([V-split](#v-split-verify-kernel)):

| bench | live seat | with the fix |
|---|---|---|
| decode behind 23k, ms per step | 37.2 | 37.2 |
| decode behind 64k / 105k / 210k, t/s | 91.2 / 92.1 / 63.6-64.8 | 91.2 / 94.6 / 66.1 |
| whole-file edit behind 47k / 94k, exact | 216.7 / 171.6 | 216.5 / 170.2 |
| 4 writers, summed | 271.8 | 292.8 |
| short code, avg of 9 | 134.7 | 134.1 |

Parity (single runs, so we read the 4-writer gap as noise, not a gain): a backward move needs a copy step to
straddle a block edge, and the copy it adds is one state per GDN layer.
Promoted with a backup of both changed files, then rolled back and promoted again to check the rollback.

### V-split verify kernel

0.32, 2026-10-25, live since 6:50 PM. Raw in `results/vsplit/`. Design, build and install:
[patches/vsplit-verify/](../patches/vsplit-verify/README.md). This is the live config.

At depth a decode step is mostly the target's verify pass over the KV cache. The 8-row tile runs MTP-5's 36 packed
verify rows as 5 work-groups per KV head, each streaming the same KV. The V-split kernel runs all of them in one
8-subgroup work-group per KV head and split (Q·K and P·V both split across subgroups, P exchanged through shared
local memory), so each KV tile is read once per step.

**One attention layer** (`kernel-exact-and-timing.out`, run on the seat's own card beside the loaded seat, q_len 6,
fp8 KV, one request, each tile with the split count the seat's captured graph uses: 12 for the 8-row tile, 32 for the
V-split, microseconds):

| context | 8-row tile | V-split (two passes) | change |
|---|---|---|---|
| 23k | 370.6 | 245.0 / 224.7 | -34% / -39% |
| 61k | 619.2 | 380.6 / 378.5 | -39% |
| 100k | 975.1 | 565.7 / 566.0 | -42% |
| 150k | 1,429.6 | 818.1 / 820.4 | -43% |
| 200k | 1,898.8 | 1,066.6 / 1,066.6 | -44% |
| 250k | 2,352.5 | 1,285.5 / 1,297.5 | -45% |

The split count matters as much as the kernel (`kernel-split-sweep.out`): 32 splits = 128 work-groups = exactly 4 per
Xe core is the fastest at every depth; 28 cost 8-13%, 36 52-104%, and riceharvest's 12 take 1.8-2.6x as long. So the
runtime patch gives the V-split tile its own rule, `4 x Xe cores / (requests x KV heads)`.

**On the seat.** Two windows on our live seat, the live config first in each, then a test load of the same config with
the new library and runtime plus `B70_SPEC_VSPLIT=1`, then the live config restored. `bench/tierbench.py`, greedy,
one stream, cold prompts of the size shown; tokens per step are the same in both arms (same output), so the change is
all step time. Window 1 ran the first build, window 2 the final one (the same kernel plus a lower single-split
threshold, which fixed the short-code dip); the final build matched the first at 23k and 210k.

| bench | before | V-split | change | raw |
|---|---|---|---|---|
| decode behind 23k, t/s (ms per step) | 80.5 (38.3) / 80.9 (38.3) | 96.7-99.5 (37.7) / 87.0 (37.2) | -0.6 to -1.1 ms per step | both windows |
| decode behind 64k | 84.7 (45.0) | 91.2 (41.8) | +7.7%, -3.2 ms | window 1 |
| decode behind 105k | 84.7 (54.1) | 92.1 (49.7) | +8.7%, -4.4 ms | window 1 |
| decode behind 210k | 52.5 (67.9) / 51.1 (67.8) | 64.8 (55.8) / 63.6 (56.0) | +23-24%, -11.8 to -12.1 ms | both windows |
| whole-file edit behind 47k / 94k (exact) | 222.7 / 169.1 | 216.7 / 171.6 | noise (copy steps keep the 8-row tile) | window 1 |
| 4 writers, summed | 271.7 | 271.8 | flat | window 1 |
| short code, avg of 9 | 134.9 / 134.3, 135.0 | 132.9 (first build) / 134.7, 134.7 | parity with the final build | both windows |

Raw: window 1 `seat-window1-before.out`, `seat-window1-vsplit-build1.out`; window 2 `seat-window2-before.out`,
`seat-window2-vsplit.out` (the files label the arms by our internal build names: `prodvs*` = before, `vs8` = first
build, `vs9` = final build). After promotion the live seat read 134.4 / 134.6 short code, 86.6 t/s (37.2 ms) at 23k and
63.8 t/s (55.8 ms) at 210k (`seat-after-promotion.out`). The 23k "before" row has a 107.6 t/s run that hit copy steps;
we compare ms per step there.

The kernel predicted -13.3 ms per step at 200k over 16 layers; the seat measured -12 ms at 210k.

**Exactness.** Against the stock per-position path: max difference 2e-5 in 416 cases with the verify rows straddling
a 960-token page boundary (one request and four, 6 and 8 rows) and in 135 mixed-length batches (2-4 requests, the
straddling one in every position) (`kernel-exact-and-timing.out`). The same call 500 times gives 500 bit-identical
outputs at 200 to 100k tokens (`kernel-determinism.out`). On the seat the greedy code probe is byte-identical before,
in the test load and after promotion (`exact-probes.txt`).

**Sanity on the promoted seat** (`bench.out`, `bench.json`, `bench/bench.py`, 2026-10-25, nothing else running at
its start): short code 142.4 t/s, 4 writers 286.8 summed, cold reads at 24k / 32k / 65k / 74k / 100k decode 80.2 / 72.6
/ 73.9 / 69.7 / 71.0 t/s, first token at 97,825 tokens 77.0 s, no canary hits, 0 preemptions. bench.py has no run on
the 2026-10-23 seat to compare with; the tierbench windows above are the before/after.

**The page-boundary test failed, and it is not this kernel.** `bench/crossing.py` printed FAIL (1/3) on the promoted
seat (`crossing.out`): one sampled reply skipped `func_185`. Repeated sampled runs (`crossing-repeat-1.out`, an
earlier version of `tests/crossing_repeat.py` that also counted a reply cut mid-line by `max_tokens` as wrong) failed
once more at the same place: `func_185` returning `x * 186`. That is about 3,840 tokens in, the 4th 960-token page
boundary, and the model gives the right digit probability 1.0000 there (`crossing-func185-logprobs.out`). Then an A/B
on the seat, 20 sampled runs each (`crossing-ab-vsplit-on.out`, `crossing-ab-vsplit-off.out`):

| arm | runs | wrong at func_185 | ended early at func_185 |
|---|---|---|---|
| V-split on (the promoted seat) | 20 | 0 | 0 |
| V-split off (test load: the same config without `B70_SPEC_VSPLIT`) | 20 | 1 | 1 |

Both arms fail at the same spot, so the fault was there before the V-split kernel. It needs the reply itself to be
decoded up to that point: 90 short sampled continuations from a prefilled assistant turn across the same boundary gave
0 wrong (`crossing-func185-prefilled.out`). Our earlier crossing checks sampled with fixed seeds, so they only ever
tested two sampled paths. Found and fixed the same evening: [GDN state after a backward block move](#gdn-state-after-a-backward-block-move).

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

## What we tried and dropped

2026-10-21 to 10-23, all on 0.32 with the live stack of the day. Each one is settled with numbers, so nobody has to
try it again without a new mechanism. Raw files in `results/settled/`.

- [Where a decode step's time goes](#where-a-decode-steps-time-goes) (the profile the rest rests on)
- [A sliding window for the MTP head](#a-sliding-window-for-the-mtp-head)
- [A 64-row verify tile](#a-64-row-verify-tile)
- [Sibling-tree speculation](#sibling-tree-speculation)
- [A per-step draft count from the drafter's confidence](#a-per-step-draft-count-from-the-drafters-confidence)
- [A multi-row GDN kernel](#a-multi-row-gdn-kernel) (works, not in the live stack)

### Where a decode step's time goes

2026-10-22, second card, torch profiler, `results/settled/profile/`. The XPU profiler sees nothing inside a graph
replay (a K = 5 step showed 35 launches and 0.2 ms of device time against a 40 ms step), so the kernel split comes
from a second load with graphs off (`--compilation-config '{"cudagraph_mode":"NONE"}'`; the dotted
`-O.cudagraph_mode` form is silently ignored on 0.32). Eager steps are launch-bound (93 ms), but the kernel durations
are the production kernels, and the graphed step time says how much replay already hides.

Device time per decode step (`pt_ea.out`, `pt_eb.out`, `pt_ec.out`, `pt_ed.out`; `slices.out` has the step times):

| rows (requests x q_len) | context | device busy | int4 GEMM | attention | GDN gated-delta | GDN conv1d | rest |
|---|---|---|---|---|---|---|---|
| 1 x 3 (K = 2) | 23k | 28.8 ms | 23.8 | 3.3 | 0.94 | 0.33 | 0.4 |
| **1 x 6 (K = 5, the live step)** | 23k | **32.3 ms** | **24.7** | **4.9** | **1.83** | **0.51** | 0.4 |
| 1 x 6 | 63k | 37.6 ms | 24.8 | 10.0 | 1.83 | 0.51 | 0.4 |
| 1 x 6 | ~0.3k | 27.6 ms | 24.4 | 0.44 | 1.85 | 0.52 | 0.4 |
| 4 x 6 = 24 (4 writers) | ~0.3k | 40.8 ms | 31.3 | 0.45 | 7.79 | 0.82 | 0.4 |

- **The GEMMs are the floor: 24 ms of every step, 76% at 23k.** 304 `gemm_kernel` launches stream the whole 27B int4
  weight set once per step: 14-15 GB in 24 ms is about 600 GB/s, the card's memory bandwidth. It moves +0.3 ms per
  extra verify row and not at all with context. Only fewer bytes per step could shrink it (TP2 and lighter quants,
  both settled no-gos on this model).
- **Attention is the context cost:** 4.9 ms at 23k, 10.0 at 63k (so about 19 at 120k), 0.44 at short context; about
  0.5 ms per verify row at 23k and 1 ms at 63k, all in the two fmha kernels (split-KV decode + reduce).
- **GDN is exactly linear in rows: 0.31 ms per row, 48 launches regardless.** The gated-delta kernel costs 0.94 /
  1.83 / 3.75 / 7.79 ms at 3 / 6 / 12 / 24 rows: a serial row loop inside the kernel, not launch overhead. 6% of a
  single-stream step, 21% of a 4-writer step.
- **The graphed step is fully kernel-bound.** The 39.9 ms live step is these 32.3 ms plus the target LM head
  (2.17 ms, fp8 GEMM over 248,320 x 5,120), the five MTP draft passes (about 5.6 ms) and rejection sampling, which
  launch after the step's profiler annotation closes. 32.3 + 7.8 = 40.2 ms of device work for a 39.9 ms step: there
  is no host-side slack to win, and graph replay already removes the 35-44 ms of launch gaps the eager step shows.
- **One MTP draft pass at 23k** (passes 2 to 5): fc 51 us, qkv 69, o_proj 34, gate_up 161, down 80, **draft head
  440** (int4, 98,816 rows, 261 MB), GEMMs 835 us at 575 GB/s, attention 172 us. The head is 54% of the bytes a pass
  moves, which is why the [shortlist](#shortlist-draft-head) was the one lever left. The per-draft slope of a step
  is one pass (about 0.96 ms) plus one extra target verify row (about 0.7 ms: GEMM 0.3, GDN 0.31, rest 0.1).

So on this hybrid a verify row costs 0.5 to 3 ms depending on context, not the attention kernel's 0.06 ms per row
per layer. That number decides the next three entries.

### A sliding window for the MTP head

2026-10-21, `results/settled/mtp-window/`. A patch gave the MTP layer's attention a sliding window (8,192 or 2,048
tokens; the layer's KV spec, prefix caching and the target untouched, so only acceptance could change). Short code
did not move (130.8 / 131.2 / 131.6 t/s for off / 2,048 / 8,192, `sc_*.out`). Decode behind 23k / 58k / 115k
(`dd_*.out`, tokens per step from the counters):

| window | 23k | 58k | 115k | tokens per step over the three |
|---|---|---|---|---|
| off (live) | 84.5 t/s | 83.2 | 64.1 | 3.55 |
| 8,192 | 72.9 | 77.5 | 56.4 (-12%) | 3.10 (-13%) |
| 2,048 | 75.3 | 65.1 | 60.3 (-6%) | 2.97 (-16%) |

The drafter needs the whole context to draft well at depth; the window saves nothing on the attention side that
acceptance does not take back. Crossing 3/3 in every arm (`cr_*.out`). Dropped.

### A 64-row verify tile

2026-10-22, `results/settled/verify-tile/`. riceharvest's attention kernel carries a 64-row packed-verify tile next
to the 8-row one the live seat uses (his notes call it "slower in serving"). We rebuilt the kernel from source on a
PC (DPC++ 2026.0, the same vllm-xpu-kernels 0.1.15.4 tree) and it reproduced his `.so` **bit-identically** on every
MTP shape (`kcheck-tile1-REPORT.txt`), so the kernel is reproducible. Then the tile question (`ktile.out`, one
attention layer, one request, packed verify, microseconds; exactness passed for every tile, split count, batch,
context and q_len):

| context | q_len (rows) | 8-row tile, heuristic splits | 64-row tile, 32 splits |
|---|---|---|---|
| 45k | 6 (36) | 467 | 497 |
| 45k | 10 (60) | 669 | 530 (-21%) |
| 115k | 6 (36) | 1,121 | 1,074 |
| 115k | 10 (60) | 1,623 | 1,151 (-29%) |

The wide tile streams each KV tile once for all rows, so its cost barely grows with rows (36 -> 60 rows: +7% against
+45% on the 8-row path at 115k), but one work-group then walks a whole KV split, so it wants few, long splits: about
4 work-groups per Xe core (32 splits at one request, 16 at two); 64 or 128 splits cost 10-30%, 8 splits 50%. A
32-row tile was 25-34% slower than the 8-row path at every split count. In the seat at the live q_len 6 (K = 5) the
two tiles are at parity (`seat-8row.out`, `seat-64row.out`: 39.9 / 46.0 / 55.7 vs 40.2 / 46.1 / 55.9 ms per step at
23k / 61k / 120k, 4 writers 260.1 vs 259.8, short code 131.2 vs 129.6). Nothing to promote on its own; its value was
for longer verifies, which the next entry closed.

### Sibling-tree speculation

2026-10-22, `results/settled/tree/`. One sibling path beside the MTP-5 chain: when the chain's first draft is
rejected, a second row carries the drafter's second choice, so that step can still emit a token instead of falling
back to the target's sample (the drafter's rank-2 matched 44-51% of first-draft rejections in a probe). q_len 8, a
per-row ancestor mask added to the decode kernel (bit-exact against a reference on every tree shape), slot and GDN
state fix-ups on the vLLM side.

The rows alone, with no wins taken (`window1-tree-on.out` vs `window1-tree-off.out`, same load):

| ms per step (tokens per step) | 23k | 61k | 120k | 4 writers | short code |
|---|---|---|---|---|---|
| chain, K = 5 (live) | 39.8 (3.29) | 45.1 (3.51) | 54.2 (3.63) | 255.7 | 129.2 |
| + 2 tree rows | 44.1 (3.25) | 51.9 (3.54) | 65.0 (3.62) | - | - |
| + 2 tree rows, single-sequence GDN, no pre-copies (`window2-b-*.out`) | 42.7 | 50.0 | 62.6 | - | 124.3 |

Two extra rows cost +11% / +15% / +20% of the step, of which 1.6-2.3 ms is the two-sequence GDN path plus 96 per-layer
state pre-copies and the rest is the price of the rows themselves (the profile above). With the wins on
(`window2-g-wins-on.out`): rank-2 matched 38% of first-draft rejections and a win fired on 15% of tree steps at the
same step time, but tokens per step **fell** (2.82 / 3.46 / 3.11 / 3.04 against 3.25 / 3.54 / 3.64 / 3.74 at 23k /
61k / 120k / 210k) and short code dropped to 90.5, because the win path's K/V, GDN-slot and hidden-state copies emit
wrong tokens (the greedy code transcript leaves the chain's about 150 tokens in; a correct win must reproduce it
exactly). Not pursued further: even a correct win path has a ceiling of about +5% tokens per step (15% of steps x 1
token) against +11-20% step time, because every extra row pays the serial GDN cost in all 48 layers. On this model,
longer or wider speculation only pays where the rows are nearly free, which is nowhere.

### A per-step draft count from the drafter's confidence

2026-10-22, `results/settled/conf-k/`. The idea: verify K = 2 instead of K = 5 on steps where the drafter's
position-0 confidence is low (deep prose), saving about 4 ms per step. vLLM 0.32's own version
(`enable_adaptive_verification`) refuses to start on this model: the base attention class gates it on
`not is_ssm()` and the GDN backend is an SSM backend, so the runner raises before any knob is read. Our own policy in
the combined speculator's scheduler hook ran one observe-only arm on the live card (`observe-only.out`: short code
130.0, 4 writers 253.3 / 262.2, decode 39.7 / 45.7 / 55.0 / 69.6 ms per step at 23k / 61k / 120k / 210k, edits 6/6
exact, all at parity with the day's live numbers) and recorded, for every K = 5 verify, the confidence the scheduler
would have had and how many drafts were accepted (`stats-*.json`). Under async scheduling the scheduler fixes K before
the confidence of those drafts exists; the newest it has is 2-3 draft rounds old, and that lagged signal carries
almost no information: on deep prose the mean accepted count is 2.0-2.3 in the low confidence buckets against 2.6 in
the top one. Every cut from 0.3 to 0.9 predicts a loss (-0.3% to -4.9% at 23k, less at depth). An exact per-step
choice needs the decision on the device inside the verify graph, which is the same GDN-builder work adaptive
verification lacks. Dropped as a policy.

### A multi-row GDN kernel

2026-10-22, `results/settled/gdn-rows/`. The stock `gated_delta_rule_spec` kernel's cost is linear in rows because of
its state stores (4 scalar half stores per column, 270-400 GB/s). Rebuilt with the state I/O vectorized to one 8-byte
access per lane, plus the rows of a request staged in local memory:

| rows (requests x per request) | installed, cold us per layer | new | speedup |
|---|---|---|---|
| 1 x 6 | 30.8 | 19.2 | 1.60x |
| 4 x 6 (4 writers) | 125.3 | 73.3 | 1.71x |
| 1 x 16 (a copy step) | 80.1 | 45.4 | 1.78x |
| 4 x 16 | 384.7 | 200.3 | 1.97x |

Output and the whole state pool **bit-identical** to the installed kernel on 240 of 240 cases (`kernel-test.out`);
the new kernel moves state at 600 GB/s, the memory floor. In one fused op with the conv1d (`fused-op-test.out`)
1.24-1.86x. In the seat on the second card, as a side library hooked into the GDN op: short code 130.5 / 129.8 ->
135.0 / 134.1 t/s (+3.5%, two loads each; `seat-*.out`), greedy code identical to the live seat's text; the 4-writer
batch step dropped about 1 ms (43.2 -> 42.2 ms), a third of what the per-layer numbers project, with the summed t/s
inside the noise. The 4-writer lines in those seat files are **not** the kernel's: the bench that produced them
ignored its endpoint override and measured the live seat on the first card instead (fixed since; the short-code and
greedy lines did go to the test seat). Status: exact, safe, +3.5% on single-stream copy-heavy work, not promoted.
Promoting it means shipping the library with the tree or upstreaming the vectorized stores into vllm-xpu-kernels.

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
