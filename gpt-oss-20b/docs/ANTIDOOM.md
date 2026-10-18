# Antidoom on two B70s: training gpt-oss-20b out of its loops

## The problem

gpt-oss-20b sometimes repeats itself until it runs out of tokens: a phrase, a list item, a line of a table, again
and again, almost always inside its reasoning. The client sees a long wait and then no answer, or a cut-off one.

We measured it with `bench/measure_loops.py`: 200 prompts from Liquid AI's
[antidoom-mix-v1.0](https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0) (seed 7, interleaved across its 14
sources), temperature 0.1, 4,000 new tokens, Antidoom's prompt template and its own loop detector. These 200 prompts
were held out of every mining round.

| stock gpt-oss-20b | loops | hit the 4,000-token cap | median tokens |
|---|---|---|---|
| effort medium | 17/200 (8.5%) | 26 | 418 |
| effort high | 79/200 (39.5%) | 91 | 2,348 |

About 2 of the 17 at medium are false positives (markdown table rules). Loops cluster in open-ended chat and
instruction following (AutoIF, lmsys-arena, ultrachat, evol-codealpaca, apps); math and short QA almost never loop.
At high, 11 of 15 AutoIF prompts and 11 of 14 ultrachat prompts looped. For comparison, a Qwen3.8-27B fine-tune on the
same 200 prompts looped once, so this is a gpt-oss problem, not a prompt-set artefact.

## The method (Liquid AI's Antidoom)

[Antidoom][antidoom] targets exactly this failure:

1. Sample completions from the model itself, with logprobs and the top 20 alternatives for every token.
2. Run a repetition detector over each completion. Where a loop is found, find the token where the loop starts.
3. Make a preference pair at that one position: rejected = the loop-start token, chosen = up to 20 coherent
   alternatives from the model's own top logprobs.
4. Train a LoRA with **FTPO** (final token preference optimisation, from
   [Antislop](https://arxiv.org/abs/2510.15061)): a single-token preference loss on the last position of each pair
   context, plus a penalty that keeps the other logits close to the reference model (`lambda_mse`, `mse_elem` in the
   logs). `chosen_win` is the share of pairs where the chosen token now beats the rejected one; training stops early
   when it passes a threshold.

Everything is the model's own output, so there's no teacher model. We left upstream's code untouched. The scripts in
`antidoom/` wrap it.

## Porting it to Intel XPU

Upstream targets CUDA (and ROCm). What we changed, all in `antidoom/xpu_train.py`, none of it in the loss:

- **bf16 across both cards.** gpt-oss-20b in bf16 is about 42 GB. The MXFP4 experts are dequantised to bf16 at load
  and the layers are split over two 32 GB B70s with `device_map="auto"` and `max_memory` 24 GiB per card. The rest of
  each card holds activations, the LoRA, its gradients and the optimiser state. Both cards are busy for the whole
  run, so nothing else can be served during training.
- **No QLoRA.** transformers' MXFP4 expert path calls a Triton matmul with no backward, so training on the 4-bit
  experts would need a custom backward. bf16 on two cards was simpler.
- **Dequantise on the CPU.** transformers unpacks MXFP4 on the target device in chunks of up to 32M rows. On the
  B70 that gather hung the GPU (xe engine reset) and every later allocation failed with
  `UR_RESULT_ERROR_OUT_OF_RESOURCES`. Unpacking on the CPU and then moving each tensor to its card is the same
  arithmetic. Load then takes about 90 s.
- **Eager attention.** gpt-oss's attention sinks have no SDPA/flash path on XPU.
- **Chunked attention** (added in round 3, on by default, `ANTIDOOM_ATTN_CHUNK=512`). Eager attention builds a
  heads × seq × seq score tensor per layer. At 4–5K-token pair contexts that's 6–10 GB per layer, the cards spilled
  into host RAM through the xe driver's TTM eviction, and the host OOM killer ended the run at step 0. The patch does
  the same math over query chunks, each under its own checkpoint. Exact on a CPU check (identical logits, gradients
  within 6e-8). Round 2 variant B was trained before this existed; its contexts were up to 5,070 tokens and fit.
- **Last-position logits only.** FTPO reads `logits[:, -1, :]`, so the forward is asked for `logits_to_keep=1` and
  a 201k-vocab × seq-len logit tensor is never built.
- **Pad to the batch, not to `max_seq_length`.** Rows stay left-padded with position ids from the mask, so the forward
  sees the same tokens at the same positions.
- **Gradient checkpointing, `adamw_torch`** (no bitsandbytes on XPU), batch 1 × 16 accumulation instead of
  upstream's 4 × 4 (same effective batch).

`antidoom/dryrun_tiny.py` runs the same code path on the CPU with a 2-layer random gpt-oss (no card needed). It
checks that `logits_to_keep=1` and batch padding don't change the forward, trains a few steps, and checks the merge
against PEFT's own `merge_and_unload()`. On ours all three matched (logit differences 0 and 3e-8, merge difference 0).

### Merging back into MXFP4

gpt-oss stores attention projections and `lm_head` in bf16 next to its MXFP4 experts. The experts are fused
parameters, not `Linear` layers, so LoRA targets like `gate_proj`/`up_proj`/`down_proj` never match: only attention
(and `lm_head`, if targeted) get adapters. `antidoom/merge_into_mxfp4.py` therefore writes `W + (alpha/r)·B·A` into
those bf16 tensors of the **original** checkpoint and copies every other tensor byte for byte. The output has the
same format and size as the input (13.8 GB) and serves on one card exactly like stock.

PEFT saves a frozen copy of `lm_head` (`base_layer`) in the adapter when `lm_head` is a target; the merge skips it.
PEFT's `tie_word_embeddings=True` warning at load is spurious for gpt-oss (its config says false).

### Software we trained with

The trainer ran in a separate Python environment from the 0.31 serving venv: torch 2.12.0+xpu, transformers 5.8.0,
accelerate 1.14.0, plus peft 0.19.1 and trl 1.2.0 (upstream Antidoom's lock-file versions), on a host with 62 GB of
RAM. Antidoom at commit `bd6a1264` (2026-07-07).

## Round 1: loops at medium

- **Mining.** Antidoom's `generate` with its `chat` backend against our running vLLM server (vLLM's chat endpoint
  returns logprobs for the reasoning tokens too, and Antidoom rebuilds the text from them). 21,000 prompts from the 7
  sources that looped (`antidoom/build_prompts.py`, the 200 held-out ids excluded), 10 then 16 streams, effort
  medium (the server default). About 15% of attempts produced a pair. 2,000 pairs.
- **Filtering.** 237 of the 2,000 were markdown dividers or `***` rules (short runs of one or two characters) that
  cross the detector's 60-character floor. `antidoom/filter_pairs.py` drops pairs whose snippet has ≤ 2 distinct
  characters and spans < 100 characters; real runaways span thousands. 1,763 pairs left.
- **Training.** Upstream's settings: LoRA r 256, alpha 128, lr 1.5e-5, early stop at `chosen_win` 0.4, targets
  q/k/v/o **and lm_head**, 179.6M trainable parameters. About 51 s per step; early stop at step 20 of 106, after 14
  minutes. Loss 5.13 → 2.83, `chosen_win` 0 → 0.46.
- **Result.** Loops 1/200 at medium and 2/200 at high. Graded accuracy 83.3% vs 78.4%. Reasoning about half as long.
- **But tools got worse.** PTC probe 26/36 vs stock 32/36 in the same run (Python syntax and escaping errors inside
  the `run_code` string, and code that never used the `tools.` helpers), and one HTTP 500 from the grammar in
  `tool_flow`. Not a drop-in replacement for agent work.

## Round 2: high effort, tool contexts, no lm_head

Three changes:

1. **Mine at high effort**, where stock loops 39.5% of the time. This needs a server with the effort cap lifted
   (`GPTOSS_EFFORT_CAP=high`).
2. **Mine inside tool-calling prompts** (`antidoom/build_prompts_r2.py`): coding tasks from evol-codealpaca and apps,
   wrapped half as a `run_code` prompt (as `bench/ptc_probe.py`) and half as an OpenCode-style agent with
   read/write/edit/bash/grep (as `bench/tool_flow.py`). The probes' own tasks are excluded, so they stay clean tests.
3. **Leave `lm_head` alone** (variant B). `lm_head` shifts every token's prior at once; the round 1 tool regressions
   looked like that kind of global drift.

Upstream's client sends neither `reasoning_effort` nor `tools`, and renders the pair context without them, so a pair
mined at high effort in a tool prompt would be trained on a context the server never saw. `antidoom/mine_r2.py`
patches both sides in memory. We checked that the Hugging Face gpt-oss chat template with `tools=` and
`reasoning_effort=` reproduces vLLM's `prompt_token_ids` token for token (169/169).

**Mining** at high, 16 streams: chat 1,404 attempts → 800 pairs (57% yield); tool prompts 940 attempts → 400 pairs
(43%), 2 requests failed with the grammar 500 described in [PITFALLS.md](PITFALLS.md). Of the 400 tool-prompt loops,
399 sat in the reasoning before the call and 1 inside a call. `run_code` prompts looped more than agent prompts (293
vs 105 pairs from an even split).

**Training set** (`antidoom/merge_pairs_r2.py`): round 1's 1,763 + 798 chat + 398 tool pairs = 2,959, after 4 more
divider false positives. It also drops any pair whose "loop" is a run of `!` (a NaN signature, not a model choice).
Context length: median 1,143 tokens, p95 3,514, max 5,070, so `max_seq_length` 5120.

Two variants in one training window:

| | lr | early stop | targets | outcome |
|---|---|---|---|---|
| A "gentle" | 5e-6 | `chosen_win` 0.25 | q/k/v/o + lm_head | killed by the host OOM killer at step 25 when another workload on the host spiked; `chosen_win` was only 0.04 then. Not rerun. |
| **B "no heads"** | 1.5e-5 | 0.4 | q/k/v/o | early stop at step 25 (`chosen_win` 0.42), ~79 s/step, 33 min. Loss 4.72 → 2.96, `mse_elem` 0.33. 96 tensors merged (24 layers × q/k/v/o). |

Config: `antidoom/configs/xpu_train_r2b_noheads.yaml`.

### Variant B results

Same benchmarks as stock, same server build and flags. Stock at medium was measured with no `reasoning_effort` in the
request (the server default is medium); B with the effort sent explicitly.

| | stock | round 1 | **B** |
|---|---|---|---|
| loops /200, medium | 17 | 1 | **0** |
| loops /200, high | 79 | 2 | **2** |
| hit 4,000 tokens /200, medium / high | 26 / 91 | 6 / 44 | 7 / 22 |
| median completion tokens, medium / high | 418 / 2,348 | 282 / 1,133 | 305 / 774 |
| graded 700, medium | 78.4% | 83.3% | **83.0%** |
| · GSM8K (200) | 95.0 | 95.5 | 95.0 |
| · MATH-500 (150) | 69.3 | 78.0 | 74.7 |
| · MMLU (200) | 74.0 | 76.0 | 78.0 |
| · CommonsenseQA (150) | 71.3 | 82.0 | 82.0 |
| · capped / loops in the graded run | 33 / 31 | 3 / 0 | 5 / 2 |
| PTC probe, 6 rounds, medium, same session | 57/72 | (3 rounds: 26/36 vs stock 32/36) | **57/72** |
| · median seconds per call / avg reasoning chars | 5.2 / 1,444 | – | 2.8 / 508 |
| tool_flow 2 × 16, medium: faults / sessions finished in 8 turns | 0 / 19 of 32 | 1 HTTP 500 / 25 of 32 | 0 / 23 of 32 |
| tool_flow 2 × 16, high: faults / finished | 0 / 15 of 32 | – | 0 / 16 of 32 |
| PTC 3 rounds, high | 6/36 | – | 16/36 |

Graded eval (`bench/grade_eval.py`): GSM8K test 200, MATH-500 150, MMLU test 200, CommonsenseQA validation 150, seed
13, same template and sampling as the loop test. Paired against stock, B fixes 47 items and breaks 15 (McNemar χ²
15.5). Against round 1 it fixes 18 and breaks 20: the same. None of these sets were used for mining.

On the agent harness (28 coding tasks per model, graded on disk): B 20 pass vs stock 17. See
[AGENT-HARNESS.md](AGENT-HARNESS.md). Then a 4-hour 16-stream soak: 1 proxy 502 in 5,180 answered requests, no NaN
output, about 0.1% loops (3 full, 2 recovered). We have served B since.

## Round 3: the narrated tool call (what failed and why)

In the agent-harness test, both models sometimes "narrated" instead of acting: the first tool call, then a final
answer that writes the remaining commands as text and reports results that never happened (B 3 of 28, stock 1 of 28).
In harmony terms this is one token: after `<|start|>assistant<|channel|>` the model picks `final` (id 17196)
instead of `comment` (12606, the start of `commentary`, the channel tool calls go on). FTPO can target that token like
a loop start.

We rebuilt three fabricated sessions' exact requests (about 31K tokens each) and replayed each 10 times at temperature
1.0. Not a tool call at a step where one was due: **stock 24/30, B 27/30.** Three passing sessions replayed the same
way: 0/30 for every model. The fabrication only showed at temperature 1.0, which is what vLLM uses when a client sends
none.

"Commitment pairs" were mined from agent sessions run against a no-network sandbox, on B, at temperature 1.0: miss
pairs (chose `final` where a call was due → chosen `comment`), anchors (correctly chose `final` after the work ran)
and near-misses. Each variant was trained from B, attention only, and gated before anything else:

| | C | C2 | C3 | B (baseline) |
|---|---|---|---|---|
| pairs | 82: 39 miss, 34 anchor, 9 near | 84: 39 miss + 45 leak tethers | 262: C2's 84 + 158 tool-less chat anchors + 20 finished-task anchors | – |
| training | early stop 0.90 at step 25 (epoch 4.2) | 2 epochs, `lambda_mse` 1.0, stop at step 10 | 2 epochs, `lambda_mse` 1.0, `lambda_mse_target` 0.2 | – |
| drift `mse_elem` | 0.75 | 0.50 | 0.23 | 0.33 |
| fabrication replay /30 | 11 | **1** | 9 | 27 |
| loops /200 medium / high | 1 / 1 | 0 / 1 | 0 / 0 | 0 / 2 |
| graded 700 | 81.7% (GSM8K 90.5) | **77.3%** (GSM8K 82.5) | 83.6% | 83.0% |
| reasoning leaked into the answer | 17.5% / 35.5% / 8.7% (loops med / high / graded) | 33 of 700 graded answers empty | 0 | 0 |
| PTC 6 rounds | 54/72 | 57/72 | **53/72 and 48/72** | 57/72 (56 in C3's session) |
| tool_flow faults | 0 | 0 | 0 format, 1 HTTP 500 | 0 |
| verdict | fails: leak, graded, PTC | fails: chat broken | fails: PTC | – |

What went wrong:

- **C.** The anchors pushed `final` up at the channel position and the misses pushed `comment`; nothing held the first
  tokens of the final message, and four epochs on 82 pairs overran the KL-like tether (`mse_elem` 0.75). The model
  opened `final` and kept reasoning in it ("We should double-check: ... Answer: 5").
- **C2.** Leak tethers fixed C's leak shape but the real cause showed in the tokens: in plain chat with no tools, the
  channel choice after the analysis became `comment` with probability 0.78 (B: `final` ≈ 1.0). Every miss pair came
  from a tool context, so FTPO learned "commentary over final" with no condition on tools. The model then wrote
  `commentary to=final ... Answer: 21` (empty content) or a commentary message vLLM put into `content`.
- **C3.** 158 tool-less chat anchors (B's own `final` over `comment`, already won) made the shift conditional: chat
  was clean (`bench/chat_check.py`: 0 commentary, 0 leaks, 0 empty in 50), graded 83.6%. But code inside tool calls
  got worse (PTC 53 and 48 of 72 vs B 56 in the same session), which fails "tools at least as good as stock".

Lessons:

- A one-token pair at the channel decision moves behaviour a lot (27/30 → 1/30), and it moves it everywhere the token
  appears. Anchor the condition you don't want to change (no-tool chat) in the same training set.
- Put leak pairs inside the leaked message (the first token after `<|message|>`), not on the channel label, and take
  the chosen token from what the baseline model actually does there.
- Mine and gate at the temperature the client really uses.
- Gate in order of cost: no-tool chat check, then the replay set, then the full table.
- High-effort agent sessions are hard to mine for this: the step-1 plan alone is about 1,700 tokens, so step-2
  contexts pass the 5,120-token training limit (2 usable pairs from 65 sessions).

What fixed it instead was changing the harness, not the weights: [AGENT-HARNESS.md](AGENT-HARNESS.md).

## Reproduce

From `antidoom/`. Every script that talks to a server reads `BASE_URL` (default `http://localhost:8000`) and an
optional `OPENAI_API_KEY`.

```sh
git clone https://github.com/Liquid4All/antidoom upstream && git -C upstream checkout bd6a1264
# mining is a plain HTTP client: no GPU and no torch needed. upstream's full dependency list pulls CUDA vLLM, so:
pip install --no-deps -e upstream && pip install datasets numpy pyyaml requests tqdm transformers
# training needs the XPU environment described above (torch-xpu, transformers, accelerate, peft, trl)
mkdir -p data && python -c "from datasets import load_dataset; \
  load_dataset('LiquidAI/antidoom-mix-v1.0', split='train').to_json('data/antidoom-mix-v1.0.jsonl')"

# held-out baseline first (its ids are excluded from mining)
python ../bench/measure_loops.py --model gpt-oss --n 200 --effort medium --out runs/gptoss-200.jsonl

# round 1 pairs (server at its default cap, medium)
python build_prompts.py --out data/prompts-gptoss.jsonl
python mine_r2.py --effort medium --model gpt-oss -c configs/r2_chat.yaml --input-jsonl data/prompts-gptoss.jsonl \
    -r runs/mine1 --threads 16 --target-pairs 2000
python filter_pairs.py runs/mine1/iter_0_ftpo_pairs.jsonl runs/mine1/pairs-clean.jsonl

# round 2 pairs: restart the server with GPTOSS_EFFORT_CAP=high first
python build_prompts_r2.py
python mine_r2.py --effort high --model gpt-oss -c configs/r2_chat.yaml  -r runs/mine2-chat  --threads 16 --target-pairs 800
python mine_r2.py --effort high --model gpt-oss -c configs/r2_tools.yaml -r runs/mine2-tools --threads 16 --target-pairs 400
python merge_pairs_r2.py --r2 runs/mine2-chat/iter_0_ftpo_pairs.jsonl runs/mine2-tools/iter_0_ftpo_pairs.jsonl \
    --round1 runs/mine1/pairs-clean.jsonl --out runs/mine2/pairs-r2.jsonl

# train + merge (stop every server on both cards first)
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 BASE=/models/gpt-oss-20b OUT=/models/gpt-oss-20b-antidoom-r2b sh run_train.sh
```

Then serve `/models/gpt-oss-20b-antidoom-r2b` with `serve/serve-gpt-oss.sh` (cap back to medium) and rerun the
`bench/` set.

Notes:

- Round 1 used upstream's `antidoom generate` directly with no effort in the request; `mine_r2.py --effort medium`
  renders the same context (medium is the chat template's default).
- `measure_loops.py` draws its 200 prompts from the mix file in file order. Export the dataset as above and you should
  get the same 200; a different export order gives a different sample.
- `merge_pairs_r2.py` loads the tokenizer `openai/gpt-oss-20b` from the Hub to count context tokens.
- Mining takes hours: round 2's 1,200 pairs at high effort took about 4 hours at 16 streams.

[antidoom]: https://github.com/Liquid4All/antidoom
