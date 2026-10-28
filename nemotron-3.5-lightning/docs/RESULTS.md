# Nemotron 3.5 Lightning: results

[← recipe page](../README.md)

Every measurement, newest first. All on vLLM 0.32.0 XPU, one B70, the W4A16 checkpoint, fp8 KV cache,
`--max-model-len 131072 --max-num-seqs 16`. Single runs; our noise on single runs is about ±5%.

The speed and gate numbers come from our bake-off harness, a model-neutral copy of
[`../../gpt-oss-20b/bench/bench_kv.py`](../../gpt-oss-20b/bench/bench_kv.py) plus that recipe's `grade_eval.py`,
`measure_loops.py`, `tool_flow.py` and `ptc_probe.py`, the 709-L recipe's
[`editbench.py`](../../qwen38-709l/bench/editbench.py) and `bench.py`, and three honesty tests described under
2026-10-26. The harness itself is not published yet; the raw outputs are in `results/`.

- **2026-10-27** (`results/2026-10-27/`): tests went straight to vLLM on the card. The live checks after each change
  also went through our router.
- **2026-10-26** (`results/speed/`, `results/bakeoff/`): through our router, so compare those numbers with each other.

# 2026-10-27: MTP, fp8 lm_head, cache granularity, thinking budgets

## Where it ended up (current config)

| | 2026-10-26 config | **current config** |
|---|---|---|
| decode 1 / 4 / 16 streams | 121.5 / 320.0 / 686.4 t/s | **220.9 / 550.8 / 1,114.4 t/s** |
| decode after a cold 45k read | 150.0 t/s | 316.6 t/s |
| whole-file edits (`editbench.py`, avg of 6) | 121.6 t/s | 272.6 t/s |
| KV pool (per stream at 16) | 2,630,376 (164k) | 1,827,858 (114k) |
| 16 × 69k sent again at once | 10.1 s (96.3% hit) | 4.1 s (99.8% hit) |
| graded, no thinking budget, 589 items no run was cut off on | 503 | 499 |
| graded with the effort plugin, all 700 items | 545 (no budget) | 557-567 (budgets 2k-8k) |
| loops / 200 | 0 | 0 |
| honesty: made up / misreported / invented | 5/80, 0/48, 0/40 | 3/80, 0/48, 0/40 |

Files: `speed/base.json`, `speed/live-after-promotion.json`, `speed/final-util093.json` (long read),
`speed/final-util093-fill16.json`, `edits/`, `graded-paired.csv`, `loops-paired.csv`, `gates/`. The first column is
the 2026-10-26 config measured again on 2026-10-27 (`base`).

## Step by step (`speed/`, `edits/`)

Decode: summed rate of N streams on a short prompt, 700 tokens each. Edits: `editbench.py`, three whole-file edit
tasks, greedy and sampled, thinking off. Code / 4 writers: `bench.py`. Acceptance: MTP drafts accepted per draft
token / tokens per verify step, on short code.

| step | decode 1 / 4 / 16 t/s | KV pool (per stream at 16) | edits t/s | code / 4 writers | acceptance |
|---|---|---|---|---|---|
| 2026-10-26 config (`base`) | 121.5 / 320.0 / 686.4 | 2,630,376 (164k) | 121.6 | 122.4 / 313.6 | - |
| + `--prefix-match-unit 64` (`pmu64`) | 122.1 / 320.1 / 690.8 | 2,630,376 (164k) | - | - | - |
| + MTP K=3, BF16 head (`mtp3-bf16head`) | 191.0 / 441.3 / 958.6 | 1,329,785 (83k) | 221.2 | 201.6 / 371.2 | 0.85 / 3.55 |
| + fp8 lm_head (`mtp3-bf16head-fp8lm`) | 212.5 / 465.0 / 1,005.7 | 1,372,681 (85k) | 253.4 | 223.7 / 402.6 | 0.80 / 3.41 |
| ...K=2 instead (`mtp2-bf16head-fp8lm`) | 201.0 / 463.6 / 908.3 | 1,474,560 (92k) | - | - | - |
| ...K=5 instead (`mtp5-bf16head-fp8lm`) | 225.9 / 441.2 / 962.8 | 1,200,453 (75k) | - | - | - |
| int4 MTP head instead of BF16 (`mtp3-int4head-fp8lm`) | 227.7 / 523.1 / 1,118.2 | 1,668,189 (104k) | 272.6 | 254.0 / 434.9 | 0.86 / 3.59 |
| + `--gpu-memory-utilization 0.93` (`final-util093`) | 232.1 / 535.9 / 1,124.3 | 1,827,858 (114k) | - | - | - |
| current config, live, after the swap (`live-after-promotion`) | 220.9 / 550.8 / 1,114.4 | 1,827,858 (114k) | - | - | - |

- **MTP** is the big step: +57% at 1 stream, +38-39% at 4 and 16. The pool shrinks because vLLM keeps Mamba state per
  draft position and the BF16 head takes 2.5 GiB. K=3 is the best balance.
- **The int4 MTP head** (2.49 → 0.68 GiB, `../scripts/quant_mtp_rtn.py`) is faster than the BF16 one (smaller draft
  reads), gives back 19k of context per stream, and accepts as well (0.86 vs 0.80-0.85).
- **fp8 lm_head**: +11% at 1 stream, +5% at 16. The BF16 head was reading at the card's bandwidth limit (below).
- **`--prefix-match-unit 64`** changes nothing in decode; its effect is on re-sent context (next section).
- **Utilization 0.93**: 16 × 69k at once (`final-util093-fill16.json`): 0 preemptions, 16/16 needles, healthy after.
- The live number (`live-after-promotion`) is the same config as `final-util093` on the serving seat, measured the
  same way; the difference (232.1 vs 220.9 at 1 stream) is within our single-run noise.

## Prefix-cache granularity (`speed/pmu64-fill.json`, `speed/mamba-block-1056.json`, `cache-check.out`)

| config | 16 × 69k: prime / sent again | hit |
|---|---|---|
| 2026-10-26 config | 171.0 s / 10.1 s | 96.3% |
| + `--prefix-match-unit 64` | 170.9 s / **2.0 s** | **99.9%** |
| + `--mamba-block-size 1056` (instead) | 170.4 s / 10.2 s | 96.3% |
| current config (MTP + 0.93) | 186.9 s / 4.1 s | 99.8% |

- With a 64-token match unit, a hit can land inside a 4,224-token block, so only the new tail is computed.
  `--mamba-block-size` had no effect (the block stayed 4,224).
- Correctness (`../bench/cachecheck.py`): 15k-token prefix + 8 questions, greedy; cached (15,168 tokens from cache,
  mid-block) vs cold (unique `cache_salt`): 7/8 identical. The one difference is the "summarize in one sentence"
  question, and two cold runs on the same server differ on that same question (15/16 identical), so it is greedy noise,
  not the cache.

## Where the time goes at 1, 4 and 16 streams (`profile/`)

Eager torch profiles (graphs-on profiles can't split kernels: the replayed graph shows as one blob), prefix caching
on, no speculation, ~36 decode steps each (`parse_prof.py`):

| kernel time per step | 1 stream | 4 streams | 16 streams |
|---|---|---|---|
| total | 8.46 ms | 12.51 ms | 21.32 ms |
| int4 MoE grouped GEMM | 2.11 | 5.46 | 11.53 |
| `gemm_kernel` (dense int4 + BF16 lm_head + router, 128 calls) | 3.29 | 3.51 | 4.32 |
| Mamba state update (tuned) | 0.15 | 0.46 | 2.17 |
| `grouped_topk` | 0.43 | 0.43 | 0.45 |

At 1 stream (`eager-x1-gemm-by-shape.txt`): the BF16 lm_head (131,072 × 2,688) is 1.15 ms per step, 704 MB at about
613 GB/s, the bandwidth limit. That is why fp8 helps at every batch size. Dense int4 runs at 60-70% of the limit, the
MoE at about 57% at 1 stream. The graphs-on step at 1 stream (8.23 ms) is about the eager kernel sum, so the card is
busy, not waiting on launches.

## Thinking budgets (`gates/effort-plugin/`, `graded-paired.csv`)

`../plugin/` maps `reasoning_effort` to `thinking_token_budget`. Graded 700 with `grade_eval.py --effort <level>`, the
server's budgets low 2,048 / medium 4,096 / high 8,192, `max_tokens` = budget + about 3,000:

| thinking budget | correct /700 | cut off | mean time / answer | slowest 10% | MATH-500 |
|---|---|---|---|---|---|
| none, 4,000-token limit (`final`, `mtp3-bf16head-fp8lm` runs) | 540 / 543 | 58 / 56 | 18.6 / 21.1 s | 54 / 57 s | 97 / 105 |
| low 2k | **567** | 18 | **17.4 s** | **38 s** | 111 |
| medium 4k | 557 | 13 | 19.4 s | 51 s | 106 |
| high 8k | 566 | **4** | 22.8 s | 50 s | **116** |

- Every budget beats no budget: far fewer answers run out of room mid-thought. Between budgets the difference is noise.
- The budget is enforced: on a deliberately hard question the thinking stopped at about 2,170 / 3,870 / 8,650 tokens
  for low / medium / no effort sent (server default 8,192), and 0 for `none`. On that question the model then kept
  working in the visible answer until `max_tokens`, so a budget bounds thinking, not total length.
- tool_flow with the plugin: 0 faults in 172 calls (`gates/effort-plugin/quality.json`).

## Quality and honesty gates (`gates/`, `graded-paired.csv`, `loops-paired.csv`, `honesty-rejudge.out`)

Same scripts and settings as the 2026-10-26 bake-off, no thinking budget (the request sends no effort). "Graded" here
counts the 589 items none of the seven unbounded runs was cut off on.

| run | graded (589) | all 700 / cut off | loops /200 | tool_flow faults | made up /80 | misreported /48 (hand) | invented /40 |
|---|---|---|---|---|---|---|---|
| 2026-10-26 config (bake-off) | 503 | 545 / 57 | 0 | 0 / 174 | 5 | 0 | 0 |
| 2026-10-26 config, again (`no-mtp-control`) | 503 | 536 / 67 | 0 | 1 / 150 | - | - | - |
| MTP-3 BF16 head (`mtp3-bf16head`) | 495 | 537 / 54 | 1 | 1 / 174 | - | - | - |
| + fp8 lm_head (`mtp3-bf16head-fp8lm`) | 502 | 543 / 56 | 0 | 0 / 153 | 4 | 0 | 2 |
| **current config** (`final`) | 499 | 540 / 58 | 0 | 1 / 167 | 3 | 0 | 0 |

- Two runs without MTP agree (503 / 503). The three MTP runs land 1-8 items lower (495-502). No single pair is
  significant (McNemar p about 0.2); a small cost of about 1 point on unbudgeted graded answers is possible.
- With speculation on, greedy output is not byte-identical to greedy without it (mostly a blank line). On the
  709-L, MTP kept greedy output identical; on this Mamba hybrid it does not. The gates say it doesn't matter.
- "Misreported" is after a hand check of every flag. `honesty-rejudge.out`: the judge prompt used to show only a
  one-line summary of the tool output, so reports that quoted the real output were flagged; with every tool call and
  its exact output in the prompt, the judge agrees with every hand check (gpt-oss-20b 6/48, Nemotron 0/48).
- Crossing test (`../bench/crossing_nemo.py`, `crossing.out`): every run stops at `func_95` ("+ 3" instead of + 4),
  with MTP and without it: the model's own arithmetic slip, not corruption. Use it with a no-change control.

## Tried and dropped

### Stock n-gram drafting (`speed/ngram-k*.json`, `edits/ngram-k*`)

| | K=5 | K=15 |
|---|---|---|
| edits | 312.9 t/s | 507.8 t/s |
| decode 1 / 4 / 16 | 86.4 / 193.6 / 387.0 | 67.5 / 110.0 / 310.3 |
| decode after a 45k read | 8.0 t/s | 32.7 t/s |
| KV pool (per stream at 16) | 1,812,122 (113k) | 1,065,446 (66k) |

Edits up to 4.2× faster, everything else slower (CPU n-gram scan over the whole context every step), and less pool.

### Copy + MTP combo speculator (`speed/combo-*.json`, `edits/combo-*`, `gates/combo-copy8*`)

The 709-L recipe's [combined copy + MTP speculator](../../qwen38-709l/patches/combo-spec/README.md) runs here after one
extra fix: vLLM's Mamba metadata builder asserts that every speculative decode step is exactly
`num_speculative_tokens + 1` long while capturing graphs, so the shorter MTP steps (4 tokens) could not be captured.
Allowing the listed lengths in that check was enough; the decode path already handles uneven steps.

| | current config | copy 15 | copy 8 | copy 8, prompt only |
|---|---|---|---|---|
| edits | 272.6 t/s | 473.4 | 382.3 | 396.5 |
| decode 1 / 4 / 16 | 220.9 / 550.8 / 1,114.4 | 219.4 / 550.3 / 1,305.1 | 231.0 / 539.7 / 1,138.8 | 234.4 / 535.1 / 1,127.3 |
| KV pool (per stream at 16) | 1,827,858 (114k) | 943,458 (58k) | 1,324,889 (82k) | 1,324,889 (82k) |
| loops /200 | 0 | - | **3** | **2** |
| E2 edit keeps `response.raise_for_status()` | yes | **no** | yes | **no** |

- Copy 8 gates (`gates/combo-copy8/`): graded 500 of 589, tool_flow 0 / 195, honesty 1/80, 0/48, 0/40, but 3 real
  thinking loops (`? 1+` × 151, a 142-character paragraph × 53) on prompts the current config finishes.
- "Prompt only" (match the prompt, never the model's own output) did not fix the loops (2/200).
- The E2 edit dropped a line in 3 of 5 runs with copy drafting (stock n-gram K=15, copy 15, copy 8 prompt only) and in
  0 of 5 runs without it (`edits/*.edit.json`, E2 `out` 3,285 instead of 3,292 tokens).
- Not kept: +40-45% on edits does not pay for a quarter less context, a line that can silently go missing, and loops.

### fp16 Mamba state (`speed/fp16-state.json`, `gates/fp16-state/`, `ssm-tuning-fp16.txt`)

`--mamba-ssm-cache-dtype float16` (vLLM uses fp32 for NemotronH by default), with the state update re-tuned for fp16
(9/9 validated; 32.4 µs vs 85.8 µs for the tuned fp32 state at 16 streams):

| | current config | + fp16 state |
|---|---|---|
| decode 1 / 4 / 16 | 220.9 / 550.8 / 1,114.4 | 210.0 / 557.9 / 1,091.2 |
| KV pool (per stream at 16) | 1,827,858 (114k) | 2,218,748 (138k) |
| graded (589) / loops / tool_flow | 499 / 0 / 1 of 167 | 498 / 0 / 0 of 184 |
| made up / misreported / invented | 3/80, 0/48, 0/40 | **8/80, 1/48**, 1/40 |
| E2 edit keeps `response.raise_for_status()` | yes | **no** (greedy and sampled) |

More context, no speed (the state update is hidden behind MTP), worse honesty, and the same dropped line. Not kept.

# 2026-10-26: bring-up and bake-off against gpt-oss-20b

## Speed

### Step by step (`results/speed/`)

Replaced on 2026-10-27 by the MTP config above; kept as the record of the bring-up.

Decode is the summed rate of N streams on a short prompt, 700 tokens each, streaming, after the first token.
"Long read" is one cold ~45k-token prompt with a needle in it. "Fill 16" primes 16 unique ~69k-token reads one after
another, then sends all 16 again at once, the way 16 agent sessions come back for their next turn.

| config | file | decode 1 / 4 / 16 t/s | KV pool (per stream at 16) | long read, first token | fill 16: prime / again at once (hit) |
|---|---|---|---|---|---|
| **+ tuned Mamba SSU config (current on 2026-10-26)** | `ssm.json` | **121.5 / 317.7 / 691.4** | 2,630,376 (164,398) | 7.42 s | not re-run |
| graphs + prefix caching | `graphs-cache.json` | 119.8 / 306.9 / 648.6 | 2,630,376 (164,398) | 7.40 s | 170.7 s / 10.1 s (96.3%) |
| `FULL_DECODE_ONLY` graphs | `graphs.json` | 121.3 / 308.5 / 649.6 | 3,214,904 (200,931) | 7.06 s | 162.2 s / 163.7 s (0%) |
| eager, no prefix caching | `eager.json` | 24.6 / 97.4 / 385.4 | 3,083,832 (192,739) | 7.10 s | not run |
| gpt-oss-20b, its tuned config, same harness | `gpt-oss.json` | 116.7 / 304.0 / 833.7 | 1,287,170 (80,448) | 8.06 s (43,532 tokens) | 192.1 s / 4.1 s (99.9%) |

- Eager decode was flat at about 24 t/s per stream: per-step overhead, not bandwidth. Graphs fixed it.
- Without prefix caching, the 16 sent again were all prefilled from scratch and only 2 ran at a time. With it, 9 ran
  at once, 0 preemptions, 16 of 16 needles found.
- The long read found the needle in every config.
- Cache correctness: 8 questions on one 16.5k-token prefix at temperature 0, cold then cached (12,672 tokens = 3
  blocks from cache): 8 of 8 answers byte-identical, no `!!!!` runs.
- Sampling: the decode bench sends no top_p. With NVIDIA's top_p 0.95 at 16 streams we read 686 and 690 t/s against
  706 and 705 at top_p 1.0 on the same load, so like-for-like with gpt-oss's bench the 16-stream number is about
  706 vs 834 (−15%).

### Where the 16-stream time goes

A torch profile (CPU + XPU, `results/profile-16-streams.txt`) of 61 decode steps at 16 streams, eager, before the SSU tuning: about 27 ms of GPU time
per step.

| kernel | ms per step | share |
|---|---|---|
| int4 MoE grouped GEMM (`MoE::GemmCuteName` / `cutlass_grouped_gemm`) | 13.2 | ~50% |
| `gemm_kernel` (dense int4 path) | 4.3 | 16% |
| Mamba state update (`_selective_scan_update_kernel`) | 3.7 | 14% |
| `int4_gemm_w4a16` (dense int4 linears) | 2.7 | 10% |
| `aten::mm` (`lm_head` in BF16, router) | 1.4 | 5% |
| top-p sampler | 0.8 | 3% |

The MoE is at the card's bandwidth limit. 16 tokens × top-6 of 128 experts touch about 68 distinct experts per layer
(128 × (1 − (1 − 6/128)^16)), × 23 MoE layers × 5.3 MB per int4 expert = about 8.3 GB read per step. At the B70's
608 GB/s that is 13.7 ms; the profile shows 13.2 ms. No flag changes that, and the XPU int4 MoE kernel has no tuning
knobs. `--max-num-batched-tokens` and graph capture sizes don't apply to pure decode at 16 (16 is already captured),
so we left them alone.

### Mamba state update tuning (`results/ssm-tuning.txt`)

vLLM's `benchmark_selective_state_update.py` (head dim 64, state 128, 8 groups, 64 heads, BF16, fp32 state), 9 of 9
batch sizes validated. At an effective batch of 1,024 (16 streams × 64 heads): 162.94 → 85.78 µs (1.90×). Smaller
batches gain 2.35× to 5.09×. End to end, decode at 16 streams went 648.6 → 691.4 t/s (+6.6%).

fp16 state (`mamba_ssm_cache_dtype float16`) was tried on 2026-10-27 and not kept; see "Tried and dropped" above.

## Bake-off against gpt-oss-20b (`results/bakeoff/`)

Both models served on the same card the same evening, through the same harness and router. gpt-oss-20b: its page's
current config (fp8 KV, rebuilt kernels, the gptoss031 plugin with effort capped at medium). Nemotron: the current
config above, thinking on with no budget, NVIDIA's sampling (temperature 1.0, top_p 0.95) where the test doesn't fix
its own.

### Honesty (`honesty.json` in each model's folder)

| test | Nemotron | gpt-oss-20b |
|---|---|---|
| abstain: 40 questions about things that don't exist, asked twice; made up an answer | **5/80** (6%) | 43/80 (54%) |
| abstain: 15 real controls, asked twice; right | 30/30 | 30/30 |
| faithful: 6 scripted tool results × 8; final report misstates them, after hand check | **0/48** | 6/48 |
| faithful: final report empty after "Stop using tools now" | 10/48 | 13/48 |
| persona: 4 fields nobody can measure from one file × 10 runs; invented a number | **0/40** | 11/40 |

- **abstain** asks about made-up papers, people, prizes and APIs mixed with real ones. A separate model (our Qwen3.8
  709-L on the first card, temperature 0) judges each answer; empty answers are scored empty and never judged, since
  the judge once invented a quote for one. We hand-checked a sample of verdicts for both models.
- **faithful** gives the model a task and scripted tool results it must report (tests fail, a file has 312 lines, a
  file is missing, an HTTP 404, a grep with no hits, a write that came out short). Every verdict that counted against
  a model was checked by hand. The judge flagged one Nemotron report; it was a wrong guess about a file's contents,
  not a success claim, and we un-flagged it by the same standard we used for gpt-oss. All 6 gpt-oss misstatements
  claim a 500-line file when its own check showed 312.
- **persona** asks for a code review with fields for test coverage, tech-debt hours, measured p95 latency and
  throughput. Any number in those fields counts as invented (regex). gpt-oss invented tech-debt hours 8 times of 10.
- In 5 real jobs in our agent harness (parents on another model, sub-agents on the candidate), neither model wrote a
  dishonest sub-agent report. One Nemotron sub-agent miscounted the files in a folder.

### Quality (`quality.json`, `grade-summary.log`, `graded-paired.csv`, `loops-paired.csv`)

| test | Nemotron | gpt-oss-20b |
|---|---|---|
| graded, 700 items, 4,000-token limit (as run) | 77.9% (545/700) | 83.9% (587/700) |
| ...answers that hit the 4,000-token limit | 57 (all scored wrong) | 4 |
| **graded, the 643 items neither model was cut off on** | **545 (84.8%)** | **549 (85.4%)** |
| ...per set, as run: GSM8K / MATH-500 / MMLU / CSQA | 90.0 / 68.7 / 75.5 / 74.0% | 96.0 / 77.3 / 79.0 / 80.7% |
| loops, 200 held-out Antidoom prompts, Antidoom's detector | 0/200 (40 hit the limit) | 0/200 (6 hit the limit) |
| ...the 160 neither was cut off on | 0 | 0 |
| tool_flow, 2 × 16 sessions × 8 turns: format faults | 0 in 174 calls | 0 in 154 calls |
| PTC probe, 72 `run_code` calls | 16/72 | 52/72 |

- The first-run gap is Nemotron thinking past the limit, not wrong answers: gpt-oss's server caps its effort at
  medium, Nemotron had no budget. A real agent run has no 4,000-token limit, so we compare on the items neither was
  cut off on (`graded-paired.csv` has both models' finish reason, token count and verdict per item).
- 32k re-run of Nemotron's cut-off items (`nemotron/graded-32k-rerun.csv`, `nemotron/loops-32k-rerun.csv`): stopped
  early after 23 of 57 graded and 6 of 40 loop items. Graded 18/23 right, none cut off, up to 21,845 tokens; loops 0/6.
  gpt-oss's 4 + 6 cut-off items were not re-run.
- PTC failures (`nemotron/ptc.log`): calls `tools.write` / `tools.edit` / `tools.grep` as tool names, writes code that
  never calls a tool, or runs out of tokens while thinking. With ordinary function calling it had no format faults.
- tool_flow's harmony-specific checks don't apply to Nemotron; the generic ones (bad name, bad JSON, HTTP errors, `!`
  runs, markup and control-token leaks) all read 0.

## First checks on load (eager)

- 17 × 23 = 391 with thinking off (4 tokens) and with a 1,024-token budget; "name three moons" right; the bat-and-ball
  puzzle right with thinking on.
- A tool call (`read {"filePath": "/work/app.py"}`) parsed into `tool_calls` with empty content.
- Log: `Using XPUExpertsWNA16`, `Model loading took 16.61 GiB`, attention block 4,224 tokens, Mamba state fp32.
