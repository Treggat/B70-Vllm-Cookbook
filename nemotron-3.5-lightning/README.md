# Nemotron 3.5 Lightning 30B-A3B on one Intel Arc Pro B70 (vLLM 0.32 XPU)

[← B70 vLLM Cookbook](../README.md)

A recipe for serving NVIDIA's **Nemotron 3.5 Lightning 30B-A3B** (a hybrid Mamba2 / attention / MoE model) on **one
Arc Pro B70 (32 GB)** with vLLM 0.32.0 XPU, at 16 concurrent streams with 131k of context each. It runs on a stock
vLLM build: no patches to the model code, no custom kernels. The only tuning is XPU graphs, prefix caching and a
tuned Mamba state-update config for this card.

We brought it up on 2026-10-26 as a candidate to replace [gpt-oss-20b](../gpt-oss-20b/README.md) on our second card,
ran both through the same bake-off the same day, and now serve Nemotron in that slot.

This page covers:

- serving: the checkpoint, the flags, thinking on/off and thinking budgets ([docs/RECIPE.md](docs/RECIPE.md))
- every measurement, from the first eager load to the tuned config, and the bake-off against gpt-oss-20b
  ([docs/RESULTS.md](docs/RESULTS.md))
- what to watch for ([docs/PITFALLS.md](docs/PITFALLS.md))

## Results

vLLM 0.32.0, one B70, fp8 KV cache, XPU graphs (`FULL_DECODE_ONLY`), prefix caching on, 2026-10-26. gpt-oss-20b is
our tuned gpt-oss setup from its own page (fp8 KV, rebuilt kernels, effort capped at medium), measured with the same
harness the same day. Single runs; read single-digit differences as noise.

| | **Nemotron 3.5 Lightning** | gpt-oss-20b |
|---|---|---|
| decode, 1 / 4 / 16 streams, summed | **121.5 / 317.7** / 691.4 t/s | 116.7 / 304.0 / **833.7** t/s |
| KV pool (fp8) / context per stream at 16 | **2,630,376 tokens / 164k** | 1,287,170 tokens / 80k |
| cold read, first token | **7.42 s** (45,433 tokens) | 8.06 s (43,532 tokens) |
| 16 × 69k reads: first pass / all 16 sent again at once | **170.7 s** / 10.1 s | 192.1 s / **4.1 s** |
| made-up answers about things that don't exist, 80 asks | **5/80** | 43/80 |
| tool results misreported in the final report, 48 runs | **0/48** | 6/48 |
| invented numbers nobody measured, 40 fields | **0/40** | 11/40 |
| graded accuracy, 643 items neither model was cut off on | 545 (84.8%) | 549 (85.4%) |
| repetition loops, 160 items neither was cut off on | 0 | 0 |
| tool_flow, 2 × 16 streaming multi-turn sessions: format faults | 0 (174 calls) | 0 (154 calls) |
| PTC probe (code inside one `run_code` call), 72 calls | 16/72 | **52/72** |

Read it like this:

- **Honesty is the clear win.** Nemotron says "I don't know" about things that don't exist (6% made up vs 54%),
  reports tool results as they were, and leaves fields it can't measure empty.
- **Speed is split.** Nemotron is faster at 1–4 streams and on prefill, and holds twice the context per stream.
  At 16 streams Nemotron is 17% slower than gpt-oss (about 15% with the same sampling). The 16-stream gap is the MoE reading its
  int4 expert weights at the card's bandwidth limit ([docs/RESULTS.md](docs/RESULTS.md#where-the-16-stream-time-goes)).
- **Accuracy is a tie** once answers cut off by the 4,000-token limit are left out. Nemotron thinks longer: 57 of its
  700 graded answers hit the limit (gpt-oss 4). Re-run with a 32k limit, 18 of the first 23 came back right.
- **Code-in-a-tool-call is the one loss.** Nemotron calls `tools.write` as if it were a tool name, or writes code that
  never calls a tool. Use it with plain function calling, not a single `run_code` tool.

Not run yet: a multi-hour soak on Nemotron.

## Quick start

Run these from this folder (`nemotron-3.5-lightning/`) on a Linux box with a B70 and the Level Zero runtime.

```sh
# 1. vLLM 0.32.0 XPU venv, the same build as the other recipes (stock attention kernels)
sh ../qwen38-709l/scripts/build_venv.sh /opt/b70/vllm032

# 2. the checkpoint (17.8 GB), then check it against the repo's own checksums
hf download useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16 --local-dir /models/nemotron-3.5-lightning-w4a16
(cd /models/nemotron-3.5-lightning-w4a16 && sha256sum -c SHA256SUMS)

# 3. serve (XPU graphs and prefix caching on via the .extra; tuned Mamba config from ../ssm-configs)
cp serve/nemotron.extra.example serve/nemotron.extra
VENV=/opt/b70/vllm032 MODEL=/models/nemotron-3.5-lightning-w4a16 CARD=0 sh serve/serve-nemotron.sh 8000

# 4. check it before trusting it: streaming multi-turn tool calls, 0 format faults expected
python ../gpt-oss-20b/bench/tool_flow.py http://localhost:8000 nemotron --sessions 16 --turns 8
```

The first load takes about 4 minutes longer than later ones (Triton warm-up of the Mamba kernels). Thinking is on by
default; [docs/RECIPE.md](docs/RECIPE.md#thinking-on-off-and-budgets) shows how to turn it off or cap it per request.

## Layout

| path | what |
|---|---|
| `serve/serve-nemotron.sh` | launcher: venv, card, tuned SSM config and compile cache as variables |
| `serve/nemotron.extra.example` | the flags that turn XPU graphs and prefix caching on (copy to `nemotron.extra`) |
| `ssm-configs/` | tuned `selective_state_update` config for the B70 (head dim 64, state 128, fp32 state) |
| `results/speed/` | KV pool, decode at 1/4/16 streams, long read and 16 × 69k fill, one JSON per config step, plus gpt-oss |
| `results/profile-16-streams.txt` | torch profiler summary of 61 decode steps at 16 streams (eager, before the SSU tuning) |
| `results/ssm-tuning.txt` | vLLM's SSM tuning benchmark output: heuristic vs tuned, per batch size |
| `results/bakeoff/` | honesty and quality summaries for both models, per-item paired CSVs (no prompts, no generations), logs |

## Credits

- **[NVIDIA](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)** for Nemotron 3.5 Lightning.
- **[useful-quants](https://huggingface.co/useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16)** for the
  symmetric int4 (group 64) GPTQ checkpoint, made with
  [llm-compressor](https://github.com/vllm-project/llm-compressor).
- **[vLLM](https://github.com/vllm-project/vllm)** (Apache-2.0) for the NemotronH model, the XPU int4 MoE path, the
  `nemotron_v3` reasoning parser and the SSM tuning benchmark.
- **[Vcvzgbzz/b70-cookbook](https://github.com/Vcvzgbzz/b70-cookbook)** (MIT): the vLLM XPU build line our venv
  follows. More B70 credits are on the [Qwen page](../qwen38-709l/README.md#credits).
- Eval sets: [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500),
  [MMLU](https://huggingface.co/datasets/cais/mmlu), [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa),
  and Liquid AI's [antidoom-mix-v1.0](https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0) for the loop test.
- **Claude** (Anthropic; Claude Opus 5.5, through Claude Code): brought the model up, tuned it, ran every
  measurement and wrote this page.

## License

Our code and docs: MIT ([LICENSE](../LICENSE)). The model and the quantized checkpoint keep NVIDIA's license; read
the model card before you use it.
