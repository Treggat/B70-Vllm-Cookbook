# Pitfalls (things that cost us time)

## Serving

- **vLLM 0.31's own harmony grammar 500s on `<|call|>`.** With `--tool-strict-level function`, 1 of 75 tool calls
  ended in HTTP 500 ("grammar rejected tokens [200012]"). With no grammar, 7 of 167 calls had leaked or invalid tool
  names and 4 streams died on a `HarmonyError`. Use the plugin's grammar with `--tool-strict-level auto`
  ([RECIPE.md](RECIPE.md#2-the-plugin-plugin)).
- **`<|end|><|return|>` leaks as text.** If a tool-call grammar lets a call end on `<|end|>`, the only legal next token
  is the EOS `<|return|>`, and vLLM's harmony parser streams it as message text. End calls on the stop token only.
- **`reasoning_effort` other than low/medium/high is a 400** on 0.31's harmony path. Clients that send `none`,
  `minimal`, `xhigh` or `max` need mapping; the plugin does it.
- **High effort runs out of tokens.** At high, stock gpt-oss-20b looped on 39.5% of our loop set and made a valid
  `run_code` call on 6–8 of 36 tasks at 3,000 tokens. Cap agent traffic at medium.
- **`--kv-cache-dtype-skip-layers sliding_window` doesn't start on vllm-xpu-kernels 0.1.15.4.** With fp8 on the full
  layers only, the bf16 window layers get 32-token pages, and there is no head-64 local+sink paged-decode kernel for
  that page size ("Paged decode kernel not compiled for this configuration", then a crash in the PyTorch fallback
  during graph capture). Use `--kv-cache-dtype fp8` on all layers ([RECIPE.md](RECIPE.md#kv-cache-fp8)).
- **A "kernel not compiled" fallback is not a crash, it's a slowdown.** vllm-xpu-kernels falls back to PyTorch
  reference attention for configs its build doesn't include and prints the config line to add. Grep the server log
  for "not compiled" after any change to KV dtype, block size or the kernel library.
- **Model-specific kernel builds don't carry over.** A custom `libattn_kernels_xe_2.so` built only for another model's
  head sizes (for example a Qwen build with head 256/96/128 only) has no head-64 kernels for gpt-oss.
- **Eager mode is 2.7× slower** (about 43 vs 106–117 tok/s on one stream). Without `serve/gpt-oss.extra` the launcher
  starts eager on purpose; copy the example.
- **`oss-harmony` and `openai-harmony` both install `openai_harmony`.** vLLM 0.31 wants oss-harmony ≥ 0.0.11; an
  `openai-harmony` 0.0.8 next to it shadows it (vLLM #55128). Check after every build.
- **A folder named `vllm` in the working directory** gets imported instead of the install. Start vLLM from the venv
  dir.
- **`!!!!` output (NaN).** On the older vLLM 0.26 image, the server once started answering only `!` (token id 0)
  about an hour into a 16-stream run; a canary request caught it and an unload/reload cleared it. We found no named
  fix in 0.31.0; the closest open upstream issues are vLLM #60560 / #60767 (stale sliding-window block-table entries).
  Two 4-hour 16-stream soaks on 0.31 had none. If you see it, `VLLM_SERVER_DEV_MODE=1` gives you
  `POST /reset_prefix_cache` to test prefix-cache poisoning before a reload. `--disable-hybrid-kv-cache-manager` is a
  candidate fix we have not tested.
- **Rare grammar 500s on long agent contexts** remain, on stock and fine-tuned weights alike (about 0.5–1% of tool
  requests in 30K-token agent contexts at temperature 1.0). See [AGENT-HARNESS.md](AGENT-HARNESS.md#grammar-500s-on-long-agent-contexts).
- **Noise, not faults:** an `EngineDeadError` traceback on every clean unload (after SIGTERM); torch 2.14 deprecation
  warnings at start; "Auto-initialization of reasoning token IDs failed".
- **Read streams to EOF.** A client that closes at `data: [DONE]` before the proxy finishes makes some proxies log the
  request as aborted. The probes here read to the end.
- **A non-streaming request to a streaming-only proxy path** may log a "no valid JSON data found in stream" warning
  for a request that succeeded. Harmless, but confusing in logs.

## Training on XPU

- **MXFP4 dequant on the B70 hung the GPU** (xe engine reset, then `UR_RESULT_ERROR_OUT_OF_RESOURCES` on every
  allocation). Dequantise on the CPU (`xpu_train._dequantize_on_cpu`).
- **No 4-bit training of gpt-oss's experts:** the MXFP4 expert matmul has no backward. Train in bf16 on two cards.
- **LoRA targets `gate_proj`/`up_proj`/`down_proj` silently match nothing** on gpt-oss: the experts are fused
  parameters. Only attention and `lm_head` train, which is also what lets the adapter merge into the MXFP4 checkpoint
  in place.
- **Eager attention at 4–5K tokens overflows into host RAM.** The xe driver evicts to system memory (TTM) instead of
  failing, the host runs out, and the OOM killer ends the trainer at step 0. Use the chunked attention in
  `xpu_train.py` (default on).
- **Keep the host quiet during training.** A round 2 run was killed by the host OOM killer when an unrelated workload
  on the same machine spiked. The trainer holds about 7 GB of host RAM; the cards' eviction can add more.
- **Stop every server on both cards first.** Training uses both B70s fully.
- **PEFT's `tie_word_embeddings=True` warning** is spurious for gpt-oss, and PEFT stores a frozen `lm_head` copy in
  the adapter when `lm_head` is a target. `merge_into_mxfp4.py` skips that copy.

## Mining and measuring

- **The loop detector flags markdown.** About 10% of raw pairs were table dividers or `***` rules (237 of 2,000).
  Filter short one- or two-character runs (`filter_pairs.py`) before training. In a 4-hour soak, 89 of 97 flagged
  "loops" were dividers.
- **A run of `!` is a NaN, not a loop.** Drop those pairs (`merge_pairs_r2.py` does).
- **Not every flagged repetition is a loop.** One soak prompt asked the model to count a 765-asterisk string, and
  copying it was correct. Check long hits by hand before reusing soak output as training data.
- **Mine at the effort the server really runs.** If the server caps effort, a request for high is served at medium,
  and pairs labelled high are medium pairs. Lift the cap on the mining server.
- **Render the pair context exactly as the server did** (tools and `reasoning_effort` included). Check it against
  vLLM's `return_token_ids` before mining.
- **Use vLLM's chat endpoint for mining** if a proxy sits in front: it returns logprobs for the reasoning tokens too,
  and some proxies don't route `/v1/completions`.
- **A held-out set must stay held out.** The prompt builders exclude the ids of the 200-prompt loop set; keep the
  probes' own tasks out of tool prompts too.
- **Gate on the condition you didn't train.** Round 3's pairs were all tool contexts, and plain chat broke. Run
  `bench/chat_check.py` before anything slower.
- **Test at the temperature the client uses.** The narrated-tool-call fabrication only showed at temperature 1.0.
