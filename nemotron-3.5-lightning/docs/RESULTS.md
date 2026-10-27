# Nemotron 3.5 Lightning: results

[← recipe page](../README.md)

Every measurement, newest first within each section. All on vLLM 0.32.0 XPU, one B70, the W4A16 checkpoint, fp8 KV
cache, `--max-model-len 131072 --max-num-seqs 16`, 2026-10-26. Single runs; our noise on single runs is about ±5%.

The speed and bake-off numbers come from our bake-off harness, a model-neutral copy of
[`../../gpt-oss-20b/bench/bench_kv.py`](../../gpt-oss-20b/bench/bench_kv.py) plus that recipe's `grade_eval.py`,
`measure_loops.py`, `tool_flow.py` and `ptc_probe.py`, and three honesty tests described below. The harness talks to
both models through our router in front of vLLM, so its numbers compare with each other, not exactly with the gpt-oss
page's direct-to-vLLM numbers. The harness itself is not published yet; the raw outputs are in `results/`.

## Speed

### Step by step (`results/speed/`)

Decode is the summed rate of N streams on a short prompt, 700 tokens each, streaming, after the first token.
"Long read" is one cold ~45k-token prompt with a needle in it. "Fill 16" primes 16 unique ~69k-token reads one after
another, then sends all 16 again at once, the way 16 agent sessions come back for their next turn.

| config | file | decode 1 / 4 / 16 t/s | KV pool (per stream at 16) | long read, first token | fill 16: prime / again at once (hit) |
|---|---|---|---|---|---|
| **+ tuned Mamba SSU config (current)** | `ssm.json` | **121.5 / 317.7 / 691.4** | 2,630,376 (164,398) | 7.42 s | not re-run |
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

Not tried: `mamba_ssm_cache_dtype float16` would halve the state traffic (about 2 ms per step) but changes the
numerics NVIDIA ships (vLLM uses fp32 for NemotronH), so it would need the quality gates re-run.

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
