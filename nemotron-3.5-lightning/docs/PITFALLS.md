# Nemotron 3.5 Lightning: pitfalls

[← recipe page](../README.md)

## Engine and kernels

- **The model won't load on a venv with gpt-oss's rebuilt attention kernels.** Those only hold head-64 variants;
  Nemotron's attention is head 128. Serve it from a venv with the stock `vllm-xpu-kernels` library
  ([RECIPE](RECIPE.md#the-engine)).
- **AWQ checkpoints with zero points get no XPU MoE backend.** The XPU int4 MoE path is symmetric only. Use a
  symmetric GPTQ / AutoRound / llm-compressor W4A16 checkpoint.
- **NVIDIA's FP8 checkpoint (30.1 GB) leaves no room for KV** on a 32 GB card, and NVFP4 is NVIDIA-only.

## Loading

- **The checkpoint's `ignore` list names every expert**, yet the experts are packed int4 in the file. On vLLM 0.32
  this is harmless: they load as int4 (`XPUExpertsWNA16`, 16.61 GiB). If a later vLLM treats them as unquantized,
  the load asks for about 55 GB; remove `mixer.experts.*` from `ignore` in a local copy of `config.json` (metadata
  only, the tensors are already int4).
- **The first load is slow**: about 4 minutes of Mamba2 Triton warm-up. With eager mode the first request took
  7 min 42 s including the load; with graphs and a warm compile cache, 231–234 s. Don't kill it as hung.

## Mamba state update config

- **vLLM's bundled B70 tuning files never match on our driver**, which reports the card as
  `Intel(R)_Graphics_[0xe223]`. The log then says "Using default Mamba SSU config". Point `VLLM_TUNED_CONFIG_FOLDER`
  at [`../ssm-configs/`](../ssm-configs/) (the launcher does), or run the tuning benchmark yourself
  ([RECIPE](RECIPE.md#tuning-the-mamba-state-update-for-your-card)) and keep the file it writes.
- **The tuning benchmark sees the wrong card** unless you `unset ONEAPI_DEVICE_SELECTOR` and set
  `ZE_AFFINITY_MASK` to the card, the same way the launcher does.

## Prefix caching

- **The cache works in 4,224-token blocks** on this model (the attention block is raised to the Mamba page size), so
  each request computes its last partial block again. 16 × 69k sent again at once took 10.1 s at 96.3% hit; gpt-oss,
  with 64-token blocks, does it in 4.1 s at 99.9%.
- **Prefix caching shrinks the KV pool** (3,214,904 → 2,630,376 tokens), because align mode keeps Mamba state per
  cached block. Still 164k per stream at 16.

## Requests

- **`reasoning_effort` does nothing.** The template only has `enable_thinking`; cap thinking with
  `thinking_token_budget` ([RECIPE](RECIPE.md#thinking-on-off-and-budgets)).
- **Unbudgeted thinking gets long**: 57 of 700 graded answers hit a 4,000-token limit while still thinking, and the
  longest re-run answer used 21,845 tokens. Set `max_tokens` generously or send a budget.
- **Don't give it a single `run_code` tool that calls other tools from inside Python.** It scored 16/72 on that probe
  (gpt-oss 52/72): it calls `tools.write` as a tool name, or writes code that never calls a tool. Ordinary function
  calling had 0 format faults in 174 calls.
