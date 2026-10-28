# Nemotron 3.5 Lightning 30B-A3B on one Intel Arc Pro B70 (vLLM 0.32 XPU)

[← B70 vLLM Cookbook](../README.md)

A recipe for serving NVIDIA's **Nemotron 3.5 Lightning 30B-A3B** (a hybrid Mamba2 / attention / MoE model) on **one
Arc Pro B70 (32 GB)** with vLLM 0.32.0 XPU, at 16 concurrent streams with 114k of context each.

- **MTP speculative decoding** from NVIDIA's own draft layer. The W4A16 checkpoint drops that layer; we take it from
  NVIDIA's BF16 repo (one 2.7 GB shard) and make it int4.
- **An fp8 lm_head** (riceharvest's helper, ported to Nemotron).
- **Finer prefix-cache hits**, so long chats sent again come back fast.
- **Thinking budgets** from `reasoning_effort`, through a small vLLM plugin.

We serve it as the model behind our coding agents on the second card of a two-B70 box. It replaced
[gpt-oss-20b](../gpt-oss-20b/README.md) there after a same-day bake-off on 2026-10-26.

This page covers:

- serving: the checkpoint, the MTP draft head, the engine patch, the flags, thinking budgets
  ([docs/RECIPE.md](docs/RECIPE.md))
- every measurement, from the first eager load to the current config, the bake-off against gpt-oss-20b, and what we
  tried and dropped ([docs/RESULTS.md](docs/RESULTS.md))
- what to watch for ([docs/PITFALLS.md](docs/PITFALLS.md))

## Results

vLLM 0.32.0, one B70, fp8 KV cache. "First config" is the 2026-10-26 bring-up (XPU graphs, prefix caching, tuned
Mamba state update); "current" adds MTP-3 with the int4 head, the fp8 lm_head, `--prefix-match-unit 64` and
`--gpu-memory-utilization 0.93` (2026-10-27). gpt-oss-20b is our tuned setup from its own page, measured with the same
harness on 2026-10-26. Single runs; read single-digit differences as noise.

| | first config | **current** | gpt-oss-20b |
|---|---|---|---|
| decode, 1 / 4 / 16 streams, summed | 121.5 / 320.0 / 686.4 t/s | **220.9 / 550.8 / 1,114.4 t/s** | 116.7 / 304.0 / 833.7 t/s |
| decode after a cold 45k read | 150.0 t/s | 316.6 t/s | 115 t/s |
| whole-file edits | 121.6 t/s | 272.6 t/s | - |
| KV pool (fp8) / context per stream at 16 | 2,630,376 / 164k | 1,827,858 / **114k** | 1,287,170 / 80k |
| 16 × 69k reads sent again at once | 10.1 s | 4.1 s | 4.1 s |
| graded, 700 items, thinking budget 2k / 4k / 8k | - | **567 / 557 / 566** | - |
| graded, no budget, items no run was cut off on | 503 of 589 | 499 of 589 | 549 of 643 (10-26 pairing) |
| loops / 200 held-out prompts | 0 | 0 | 0 |
| made-up answers about things that don't exist | 5/80 | **3/80** | 43/80 |
| tool results misreported in the final report | 0/48 | **0/48** | 6/48 |
| invented numbers nobody measured | 0/40 | **0/40** | 11/40 |

Read it like this:

- **Speed:** MTP is the big step: +82% at 1 stream and +62% at 16 over the first config, with the fp8 head and the
  int4 draft head on top. At 16 streams it is now 34% faster than gpt-oss-20b.
- **Context:** 114k per stream at 16, against 164k without MTP, because vLLM keeps Mamba state for each draft
  position. Still more than gpt-oss-20b's 80k.
- **Honesty:** the clear win over gpt-oss-20b, and unchanged by the speed work.
- **Accuracy:** with a thinking budget it improves (fewer answers cut off mid-thought). Without one, the MTP runs sit
  about 1 point below the no-MTP runs, within noise.
- **One weak spot:** code inside a single `run_code` tool call (16/72 vs gpt-oss-20b 52/72). Use it with plain
  function calling.

Tried and not kept: stock n-gram drafting, the 709-L recipe's copy + MTP combo speculator (+40-45% on edits, but less
context, more loops, and a line that went missing from an edit), fp16 Mamba state, and a smaller Mamba block size.
[docs/RESULTS.md](docs/RESULTS.md#tried-and-dropped) has every number.

## Quick start

Run these from this folder (`nemotron-3.5-lightning/`) on a Linux box with a B70 and the Level Zero runtime.

```sh
# 1. vLLM 0.32.0 XPU venv (stock attention kernels) + the fp8 lm_head patch
sh ../qwen38-709l/scripts/build_venv.sh /opt/b70/vllm032
VLLM_ROOT=/opt/b70/vllm032/src/vllm /opt/b70/vllm032/venv/bin/python patches/lmhead-fp8/patch_nemotron_lmhead_fp8_032.py

# 2. the checkpoint (17.8 GB), checked against the repo's own checksums
hf download useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16 --local-dir /models/nemotron-3.5-lightning-w4a16
(cd /models/nemotron-3.5-lightning-w4a16 && sha256sum -c SHA256SUMS)

# 3. the MTP draft head: one shard of NVIDIA's BF16 repo, extracted, then int4 (sha256 check in docs/RECIPE.md)
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
  config.json model.safetensors.index.json model-00014-of-00014.safetensors --local-dir /models/nemotron-bf16-mtp-src
/opt/b70/vllm032/venv/bin/python scripts/extract_mtp_head.py /models/nemotron-bf16-mtp-src /models/nemotron-3.5-lightning-w4a16 /models/nemotron-3.5-lightning-mtp-bf16
/opt/b70/vllm032/venv/bin/python scripts/quant_mtp_rtn.py /models/nemotron-3.5-lightning-mtp-bf16 /models/nemotron-3.5-lightning-mtp-w4a16

# 4. serve (graphs, prefix caching, MTP via the .extra; fp8 head, tuned Mamba config and the effort plugin via the launcher)
cp serve/nemotron.extra.example serve/nemotron.extra
VENV=/opt/b70/vllm032 MODEL=/models/nemotron-3.5-lightning-w4a16 CARD=0 sh serve/serve-nemotron.sh 8000

# 5. check it before trusting it: streaming multi-turn tool calls (0-1 format faults expected) and the cache check
python ../gpt-oss-20b/bench/tool_flow.py http://localhost:8000 nemotron --sessions 16 --turns 8
python bench/cachecheck.py http://localhost:8000 nemotron
```

The first load takes about 4 minutes (Triton warm-up of the Mamba kernels, graph capture). If your MTP draft folder
is not at `/models/nemotron-3.5-lightning-mtp-w4a16`, change the path in `serve/nemotron.extra`. Thinking is on by
default; `reasoning_effort` none / low / medium / high gives off / 2k / 4k / 8k thinking tokens
([docs/RECIPE.md](docs/RECIPE.md#thinking-on-off-and-budgets)).

## Layout

| path | what |
|---|---|
| `serve/serve-nemotron.sh` | launcher: venv, card, tuned SSM config, fp8 head switch, effort plugin and compile cache as variables |
| `serve/nemotron.extra.example` | the flags for XPU graphs, prefix caching, `--prefix-match-unit 64`, utilization 0.93 and MTP-3 (copy to `nemotron.extra`) |
| `scripts/extract_mtp_head.py` | builds the BF16 MTP draft folder from the one shard of NVIDIA's BF16 repo that holds it |
| `scripts/quant_mtp_rtn.py` | makes that draft folder symmetric int4 g64 (2.49 → 0.68 GiB) |
| `patches/lmhead-fp8/` | riceharvest's fp8 lm_head helper (MIT, license beside it) and our installer that hooks it into `nemotron_h.py` |
| `plugin/` | `nemoeffort` vLLM plugin: `reasoning_effort` → `thinking_token_budget`, with its offline `selftest.py` |
| `bench/cachecheck.py` | cached vs cold answers on a long shared prefix (prefix-cache correctness) |
| `bench/crossing_nemo.py` | long predictable reply across many block edges; the 709-L crossing test adapted to Nemotron |
| `ssm-configs/` | tuned `selective_state_update` config for the B70 (head dim 64, state 128, fp32 state) |
| `results/2026-10-27/` | the MTP / fp8 / cache / budget work and what was dropped: speed, edits, gates, profiles, paired per-item CSVs |
| `results/speed/`, `results/bakeoff/` | the 2026-10-26 bring-up steps and the bake-off against gpt-oss-20b |
| `results/profile-16-streams.txt`, `results/ssm-tuning.txt` | 2026-10-26 profile at 16 streams and the fp32 SSM tuning run |

## Credits

- **[NVIDIA](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)** for Nemotron 3.5 Lightning
  and its MTP layer.
- **[useful-quants](https://huggingface.co/useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16)** for the
  symmetric int4 (group 64) GPTQ checkpoint, made with
  [llm-compressor](https://github.com/vllm-project/llm-compressor).
- **[riceharvest/b70-qwen38-cookbook](https://github.com/riceharvest/b70-qwen38-cookbook)** (MIT) for the fp8 target
  lm_head helper in `patches/lmhead-fp8/`, included unchanged under his license.
- **[vLLM](https://github.com/vllm-project/vllm)** (Apache-2.0) for the NemotronH model and MTP drafter, the XPU int4
  MoE path, the `nemotron_v3` reasoning parser, thinking budgets and the SSM tuning benchmark.
- **[Vcvzgbzz/b70-cookbook](https://github.com/Vcvzgbzz/b70-cookbook)** (MIT): the vLLM XPU build line our venv
  follows. More B70 credits are on the [Qwen page](../qwen38-709l/README.md#credits).
- Eval sets: [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500),
  [MMLU](https://huggingface.co/datasets/cais/mmlu), [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa),
  and Liquid AI's [antidoom-mix-v1.0](https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0) for the loop test.
- **Claude** (Anthropic; Claude Opus 5.5, through Claude Code): brought the model up, tuned it, wrote the scripts, the
  patch installer and the plugin, ran every measurement and wrote this page.

## License

Our code and docs: MIT ([LICENSE](../LICENSE)). `patches/lmhead-fp8/b70_target_lmhead_fp8.py` is riceharvest's (MIT,
`patches/lmhead-fp8/LICENSE-riceharvest`). The model, the quantized checkpoint and the MTP layer keep NVIDIA's license;
read the model card before you use them.
