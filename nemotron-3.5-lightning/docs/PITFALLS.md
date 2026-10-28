# Nemotron 3.5 Lightning: pitfalls

[← recipe page](../README.md)

## Engine and kernels

- **The model won't load on a venv with gpt-oss's rebuilt attention kernels.** Those only hold head-64 variants;
  Nemotron's attention is head 128. Serve it from a venv with the stock `vllm-xpu-kernels` library
  ([RECIPE](RECIPE.md#the-engine)).
- **AWQ checkpoints with zero points get no XPU MoE backend.** The XPU int4 MoE path is symmetric only. Use a
  symmetric GPTQ / AutoRound / llm-compressor W4A16 checkpoint.
- **NVIDIA's FP8 checkpoint (30.1 GB) leaves no room for KV** on a 32 GB card, and NVFP4 is NVIDIA-only.
- **`B70_TARGET_LMHEAD_FP8=1` does nothing on an unpatched venv.** Apply `patches/lmhead-fp8/` first; the log line
  `[B70] target lm_head fp8` tells you it is on.

## Loading

- **The checkpoint's `ignore` list names every expert**, yet the experts are packed int4 in the file. On vLLM 0.32
  this is harmless: they load as int4 (`XPUExpertsWNA16`, 16.61 GiB). If a later vLLM treats them as unquantized,
  the load asks for about 55 GB; remove `mixer.experts.*` from `ignore` in a local copy of `config.json` (metadata
  only, the tensors are already int4).
- **The W4A16 checkpoint has no MTP layer.** Pointing `--speculative-config {"method":"mtp"}` at it without a
  `"model"` finds nothing to draft with. Build the separate draft folder ([RECIPE](RECIPE.md#the-mtp-draft-head)).
- **The first load is slow**: about 4 minutes of Mamba2 Triton warm-up. With eager mode the first request took
  7 min 42 s including the load; with graphs, MTP and a warm compile cache, about 4 minutes. Don't kill it as hung.

## Mamba state update config

- **vLLM's bundled B70 tuning files never match on our driver**, which reports the card as
  `Intel(R)_Graphics_[0xe223]`. The log then says "Using default Mamba SSU config". Point `VLLM_TUNED_CONFIG_FOLDER`
  at [`../ssm-configs/`](../ssm-configs/) (the launcher does), or run the tuning benchmark yourself
  ([RECIPE](RECIPE.md#tuning-the-mamba-state-update-for-your-card)) and keep the file it writes.
- **The tuning benchmark needs the source tree on `PYTHONPATH`** (it imports `tests.kernels.mamba.utils`) and sees
  the wrong card unless you `unset ONEAPI_DEVICE_SELECTOR` and set `ZE_AFFINITY_MASK`, as the launcher does.
- **fp16 Mamba state is not worth it** here: `--mamba-ssm-cache-dtype float16` gives 24k more context per stream but
  no speed with MTP on, and it made honesty worse and dropped a line in an edit
  ([RESULTS](RESULTS.md#fp16-mamba-state-speedfp16-statejson-gatesfp16-state-ssm-tuning-fp16txt)).

## Prefix caching

- **Without `--prefix-match-unit` the cache works in whole 4,224-token blocks** (the attention block is raised to the
  Mamba page), so each re-sent request computes its last partial block again: 10.1 s for 16 × 69k at 96.3% hit.
  `--prefix-match-unit 64` lets a hit end inside a block: 2.0 s at 99.9%. `--mamba-block-size` had no effect.
- **Prefix caching and MTP both shrink the KV pool** (align mode keeps Mamba state per cached block; speculation keeps
  it per draft position). The current config holds 114k per stream at 16.

## Speculative decoding

- **Greedy output is not byte-identical with speculation on**, unlike MTP on the 709-L's GDN model. Mostly a blank
  line; the quality gates show no real change. Don't use byte-identity as a correctness test for this model.
- **Copy (n-gram) drafting drops lines from edits.** In 3 of 5 copy-drafting runs the same whole-file edit lost
  `response.raise_for_status()`; it never did without copy drafting. Stock n-gram and the 709-L combo speculator
  both did it, and the combo also added thinking loops. We don't use copy drafting on this model.
- **vLLM 0.32's Mamba metadata builder assumes every speculative step has the full length** while capturing graphs
  (`mamba_attn.py`, `build_for_cudagraph_capture`), so any per-batch-size schedule with shorter steps fails at load
  with an `AssertionError`. Plain MTP with one K is fine.
- **The crossing test can't judge this model on its own.** Nemotron writes `x * 95 + 3` at `func_95` (95 mod 7 = 4)
  with or without speculation. Compare a change against a run without it.
- **Profile without graphs.** A torch profile with XPU graphs on shows the replayed graph as one kernel; profile eager
  to see the split.

## Requests

- **`reasoning_effort` does nothing without the plugin.** The template only has `enable_thinking`; vLLM maps `none`
  to thinking off, nothing else. Load `../plugin/` (the launcher does) or send `thinking_token_budget` yourself.
- **A thinking budget bounds thinking, not the whole answer.** On a very hard question the model kept working in the
  visible answer after its budget and ran to `max_tokens` (16k). Keep a sensible `max_tokens`.
- **Don't send `thinking_token_budget` to a gpt-oss server** behind the same router: it answers HTTP 400. Send
  `reasoning_effort` and let the plugin map it on the Nemotron side.
- **Don't give it a single `run_code` tool that calls other tools from inside Python.** It scored 16/72 on that probe
  (gpt-oss 52/72): it calls `tools.write` as a tool name, or writes code that never calls a tool. Ordinary function
  calling had 0-1 format faults in about 170 calls per run.

## Measuring

- **An LLM judge must see the real tool output.** Our honesty judge first saw only a one-line summary of what each
  tool returned, and flagged reports that quoted the real output as invented. With every call and its exact output in
  the prompt (and a separate "no report" verdict), it agreed with every hand check (`results/2026-10-27/honesty-rejudge.out`).
