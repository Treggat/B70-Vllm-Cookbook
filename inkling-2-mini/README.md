# Inkling 2 Mini on one Intel Arc Pro B70 (vLLM XPU, tested and not adopted)

[← B70 vLLM Cookbook](../README.md)

A record of bringing up Thinking Machines' **Inkling 2 Mini** (26B MoE, 128 experts, top-8, about 3.4B active, 24
layers of which the middle 12 run three times) on **one Arc Pro B70 (32 GB)** at 16 concurrent streams, with its
DSpark drafter working on XPU.

**We do not serve this model.** We tested it on 2026-11-26 as a replacement for
[Nemotron 3.5 Lightning](../nemotron-3.5-lightning/README.md) behind our coding agents. It runs correctly and it
is much better at one thing (code inside a single tool call), but at 16 streams it is about four times slower than
Nemotron, holds about a fifth of the context, and scores slightly lower on our graded questions. The page is here
so the next person with a B70 can skip the bring-up and knows what to expect.

What is in it:

- **The DSpark drafter on XPU.** The engine rejects it on Intel cards; three small edits make it run.
- **fp8 KV cache on XPU**, also rejected by default.
- **Two tile settings for the 4-bit expert kernel**, found with a standalone sweep, worth +35% at 16 streams.
- **A bake-off against Nemotron** on honesty, tool use, graded questions and loops.

## Results

Engine: Intel's Inkling 2 XPU branch of vLLM (see [docs/RECIPE.md](docs/RECIPE.md)), one B70, 16 streams, 65k context limit,
2026-11-26. The official checkpoint (experts NVFP4 in layers 0-19 and MXFP8 in layers 20-23, the rest BF16).
Speeds are 512 forced output tokens per stream, summed over streams, from `bench/speed.py`; prose is sampled at
temperature 0.7, code is greedy. Single runs; read single-digit differences as noise. Raw files are in
[`results/2026-11-26/speed/`](results/2026-11-26/speed/).

### Speed, step by step

| config | prose, 1 / 4 / 16 streams | code, 1 / 4 / 16 streams | cold 33k read | raw file |
|---|---|---|---|---|
| eager, no prefix cache | 10.1 / 39.1 / 148.1 t/s | - | - | `speed-eager-base.json` |
| + XPU graphs, prefix cache | 26.8 / 63.1 / 150.9 t/s | - | 99.5 s | `speed-graphs-prefix16.json`, `smoke-20261126-0533.json` |
| + DSpark, greedy drafts | 25.5 / 55.0 / 121.4 t/s | 37.2 / 91.0 / 219.3 t/s | - | `speed-dspark-graphs.json`, `speed-dspark-graphs-code-t0.json` |
| + DSpark, probabilistic drafts | 30.3 / 65.0 / 134.7 t/s | 37.5 / 87.1 / 203.1 t/s | 82.4 s | `speed-dspark-prob-*.json`, `smoke-20261126-0707.json` |
| + expert tile settings | **41.7 / 88.7 / 181.6 t/s** | **63.7 / 133.8 / 273.0 t/s** | 55.5 s | `speed-tiles-prob-*.json`, `smoke-20261126-0811.json` |
| + fp8 KV cache (the config we tested) | not measured | 51.0 / 131.0 / 252.2 t/s | 74.5 s | `speed-fp8kv-code-t0.json`, `smoke-20261126-0854.json` |

For scale, Nemotron 3.5 Lightning on the same card does 220.9 / 550.8 / 1,114.4 t/s (its own page, a different
script and day).

Read it like this:

- **The drafter only pays with probabilistic drafts.** With the default greedy drafts it made sampled prose slower
  than no drafter at all. Set `"draft_sample_method":"probabilistic"`.
- **The 4-bit expert kernel is the bottleneck.** Inkling's experts are NVFP4, and on XPU they run through vLLM's
  NVFP4 emulation kernel (Triton). A profile of the drafter config put that kernel at 64% of a decode step at 1
  stream and 76% at 16 ([`results/2026-11-26/profile/`](results/2026-11-26/profile/)). The MXFP8 layers use a
  native kernel and cost a fraction of that per call.
- **Tile settings help, and they are free.** See [the sweep](#the-expert-tile-sweep).
- **fp8 KV costs speed here and buys context.** The KV pool went to 338,386 tokens (read from the startup log),
  about 21k per stream at 16. Nemotron gets 114k per stream on the same card.
- **A native 4-bit expert kernel is the missing piece.** We did not write one. Our estimate from the profile is
  about twice the speed at best, which still leaves it at half of Nemotron.

### Bake-off against Nemotron

The config in the last row above, against Nemotron's 2026-10-26 bake-off
([`../nemotron-3.5-lightning/results/bakeoff/`](../nemotron-3.5-lightning/results/bakeoff/)), same harness. The
judge for the honesty tests is our larger local model. Raw summaries:
[`results/2026-11-26/bakeoff/`](results/2026-11-26/bakeoff/).

| test | Inkling 2 Mini | Nemotron 3.5 Lightning |
|---|---|---|
| graded, 700 items, 4,000-token cap, temperature 1.0 | 528 (75.4%), 35 cut off, 0 loops | not run |
| graded, same, temperature 0.1 | 534 (76.3%), 47 cut off, **9 loops** | 545 (77.9%), 57 cut off, 0 loops |
| loops / 200 held-out prompts, temperature 1.0 | 0, 26 cut off | not run |
| loops / 200, temperature 0.1 | **7**, 40 cut off | 0, 40 cut off |
| made-up answers about things that don't exist | 5/80 (17 empty answers) | 5/80 |
| tool results misreported in the final report | 0/48 | 0/48 |
| invented numbers nobody measured | 2/40 | 0/40 |
| streaming tool calls with a format fault | 0 of 259 | 0 of 174 |
| scripted tool sessions finished within 8 turns | 0 of 32 | 18 of 32 |
| code inside one `run_code` tool call | **53/72** | 16/72 |

Read it like this:

- **Run it at temperature 1.0.** Our graded and loop tests send 0.1 to every model. At 0.1 Inkling's reasoning
  fell into repeats (the same fragment until the token cap) on 16 of 900 prompts. We re-ran those 16 without the
  drafter and with BF16 KV: 6 still looped at 0.1 and none at 1.0 (`loop-control-plain.jsonl`). So it is the
  temperature, not the drafter. At 1.0 the full tests had no loops.
- **Graded accuracy is a little under Nemotron** at either temperature. The CommonsenseQA part dropped from 90 to
  81 of 150 at temperature 1.0; we did not check those answers by hand.
- **It keeps calling tools.** No scripted session finished on its own in 8 turns, and all 48 report tasks needed
  to be told to stop before they reported. The reports were then accurate. We did not read the transcripts to find
  out why.
- **Code in a single tool call is its clear win**, and Nemotron's weak spot.
- Judge verdicts on this run were not checked by hand.

### The expert tile sweep

`bench/moe_sweep.py` times the NVFP4 emulation kernel alone at Inkling's expert shapes (random weights and
routing, no model load) and estimates a decode step as 44 expert passes. Raw:
[`results/2026-11-26/moe-sweep/`](results/2026-11-26/moe-sweep/).

| path | tiles | estimated step |
|---|---|---|
| decode (small batches), engine default | M16 N128 K128, 8 warps | 91.0 / 85.1 / 108.6 ms at 8 / 16 / 128 tokens |
| decode, ours | M16 N128 **K64, 16 warps** | **39.2 / 60.9 / 61.5 ms** |
| prefill at 2,048 tokens, engine's choice | M128 N128 K64, 8 warps | 523.4 ms |
| prefill at 2,048 tokens, ours | M128 N128 K64, **16 warps** | **243.9 ms** |

Triton on XPU rejected `grf_mode: large`, and N256/K256 or K512 tiles ran out of shared memory.

## Quick start

Run these from this folder (`inkling-2-mini/`) on a Linux box with a B70 and the Level Zero runtime.

```sh
# 1. the engine: Intel's Inkling 2 XPU branch plus one commit of vLLM PR 69444 (see docs/RECIPE.md)
mkdir -p /opt/b70/vllm-ink2 && cd /opt/b70/vllm-ink2
git clone https://github.com/afierka-intel/vllm src && cd src
git checkout 19dc25c            # branch afierka/ink2-v2-9-autotune-row-kernels-r11
git fetch https://github.com/vllm-project/vllm pull/69444/head && git cherry-pick 0811005   # one conflict, see docs/RECIPE.md
cd .. && python3 -m venv venv
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/xpu https://wheels.vllm.ai/xpu/"
venv/bin/python -m pip install -U pip wheel "setuptools>=78.1.1,<81" "setuptools-scm>=9.2.0" "setuptools-rust>=1.9.0"
(cd src && VLLM_TARGET_DEVICE=xpu ../venv/bin/python -m pip install -r requirements/xpu.txt)
(cd src && VLLM_TARGET_DEVICE=xpu VLLM_USE_PRECOMPILED_RUST=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT=65ada0c40e18d1100deee9c3deb351326a309e82 \
  ../venv/bin/python -m pip install --no-build-isolation -e .)
```

Then, back in this folder:

```sh
# 2. the checkpoint (about 19 GB; the drafter mtp.safetensors ships inside it)
hf download thinkingmachines/Inkling-2-Mini --local-dir /models/inkling-2-mini

# 3. serve: the launcher applies patches/patch_ink2_b70.py on every start (idempotent)
cp serve/inkling.env.example serve/inkling.env
cp serve/inkling.extra.example serve/inkling.extra
VENV=/opt/b70/vllm-ink2 MODEL=/models/inkling-2-mini CARD=0 sh serve/serve-inkling.sh 8000

# 4. check it: one tool call, a tool round trip, a cold 33k read, reasoning effort low and high; then speed
python bench/smoke.py --base http://localhost:8000
python bench/speed.py --base http://localhost:8000 --code --temp 0
```

The first load of the fp8 KV config is slow twice: about 7 minutes of silent kernel compilation, then about 38
minutes of Triton autotuning during the first requests (our first reply took 1,483 s). Both are cached per config,
so it happens once per change to `inkling.env` or `inkling.extra`. Send temperature 1.0 from your client.

## Engine and pitfalls

[docs/RECIPE.md](docs/RECIPE.md) has the engine source and pins, what each patch switch does and the first-load
times. [docs/PITFALLS.md](docs/PITFALLS.md) has what broke.

## Layout

| path | what |
|---|---|
| `serve/serve-inkling.sh` | launcher: engine folder, card, compile cache per config, applies the patch, serves 16 streams |
| `serve/inkling.env.example`, `serve/inkling.extra.example` | the switches and flags of the config we tested (copy without `.example`) |
| `patches/patch_ink2_b70.py` | the engine edits listed above, idempotent, exits 1 if an anchor moved |
| `patches/b70_target_lmhead_fp8.py`, `patches/LICENSE-riceharvest` | riceharvest's fp8 lm_head helper adapted to Inkling, with his license |
| `bench/speed.py` | decode speed at 1 / 4 / 16 streams, forced output length |
| `bench/smoke.py` | tool call, tool round trip, cold long read, reasoning effort |
| `bench/moe_sweep.py` | standalone tile sweep of the NVFP4 emulation expert kernel |
| `docs/RECIPE.md`, `docs/PITFALLS.md` | engine source and pins, patch switches, first-load times; what broke |
| `results/2026-11-26/` | `speed/` (speed and smoke runs), `profile/` (1 and 16 streams), `moe-sweep/`, `bakeoff/` (summaries; generated text removed) |

## Credits

- **[Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** for Inkling 2 Mini and its DSpark
  drafter.
- **[afierka-intel](https://github.com/afierka-intel/vllm)** (Intel) for the Inkling 2 XPU branch of vLLM, and the
  author of vLLM PR 69444.
- **[vLLM](https://github.com/vllm-project/vllm)** (Apache-2.0) for the Inkling 2 model code, the DSpark
  speculator and the NVFP4 emulation MoE kernel.
- **[riceharvest/b70-qwen38-cookbook](https://github.com/riceharvest/b70-qwen38-cookbook)** (MIT) for the fp8 target
  lm_head helper.
- Eval sets: as on the [Nemotron page](../nemotron-3.5-lightning/README.md#credits).
- **Claude** (Anthropic; Claude Fable 5.2, through Claude Code): wrote the patches and scripts, ran every
  measurement and wrote this page.

## License

Our code and docs: MIT ([LICENSE](../LICENSE)). `patches/b70_target_lmhead_fp8.py` derives from riceharvest's helper
(MIT, `patches/LICENSE-riceharvest`). The model keeps its own license; read the model card before you use it.
