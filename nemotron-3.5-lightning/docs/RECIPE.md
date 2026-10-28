# Nemotron 3.5 Lightning: recipe

[← recipe page](../README.md)

What we run now on one B70, step by step. Every step was run on vLLM 0.32.0 XPU; the current config since
2026-10-27.

## The model and the checkpoint

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (`NemotronHForCausalLM`) has 52 layers: 23 Mamba2 mixers, 6 attention
layers (32 query heads, 2 KV heads, head dim 128) and 23 MoE layers (128 experts, top-6, sigmoid router, one shared
expert, relu² activation). About 3B parameters are active per token. Native context is 262k. NVIDIA also ships one
multi-token-prediction (MTP) layer for speculative decoding.

The BF16 weights (about 60 GB) don't fit on a 32 GB card, and NVIDIA's own FP8 (30.1 GB) leaves no room for KV.
NVFP4 runs on NVIDIA hardware only. We use
[useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16](https://huggingface.co/useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16):

- compressed-tensors GPTQ, **symmetric int4, group 64**, one 17.8 GB safetensors file (17,826,201,400 bytes)
- all routed and shared experts, the Mamba projections and attention q/k/v/o are int4; norms, Mamba A/D/dt/conv,
  the router, embeddings and `lm_head` stay BF16
- **the MTP layer is removed** (`num_nextn_predict_layers: 0`); we take it from NVIDIA's BF16 checkpoint below
- the chat template is NVIDIA's, unchanged

Symmetric int4 matters on XPU: vLLM's XPU int4 MoE path (`XPUExpertsWNA16`) accepts symmetric weights only. An AWQ
checkpoint with zero points gets no XPU MoE backend.

Download it and check it, from any folder with the Hugging Face CLI:

```sh
hf download useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16 --local-dir /models/nemotron-3.5-lightning-w4a16
cd /models/nemotron-3.5-lightning-w4a16 && sha256sum -c SHA256SUMS
```

All 25 files must say OK.

## The MTP draft head

NVIDIA's BF16 repo keeps all 270 `mtp.*` tensors in its last shard (2,670,685,240 bytes), so you don't need the other
60 GB. We pin the revision we used. From any folder with the Hugging Face CLI:

```sh
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
  config.json model.safetensors.index.json model-00014-of-00014.safetensors --local-dir /models/nemotron-bf16-mtp-src
cd /models/nemotron-bf16-mtp-src && echo "64577b275ca4e7e5266eae0903674f7f46ec2a8cbf4f4f1a3207f80d503cd1d0  model-00014-of-00014.safetensors" | sha256sum -c
```

Then, from this recipe's folder with the venv below, build a separate draft folder and make it int4:

```sh
/opt/b70/vllm032/venv/bin/python scripts/extract_mtp_head.py /models/nemotron-bf16-mtp-src /models/nemotron-3.5-lightning-w4a16 /models/nemotron-3.5-lightning-mtp-bf16
/opt/b70/vllm032/venv/bin/python scripts/quant_mtp_rtn.py /models/nemotron-3.5-lightning-mtp-bf16 /models/nemotron-3.5-lightning-mtp-w4a16
```

- `extract_mtp_head.py` writes `model.safetensors` with only the `mtp.*` tensors (2.49 GiB, BF16), NVIDIA's
  `config.json` (`num_nextn_predict_layers: 1`) and the tokenizer.
- `quant_mtp_rtn.py` makes the routed + shared experts and attention q/k/v/o symmetric int4 g64 (the target's format),
  keeps `eh_proj`, the router and the norms BF16: 0.68 GiB, round-trip error 0.10-0.11.
- The draft folder is passed to vLLM as `"model"` in `--speculative-config` (`method: "mtp"`), and vLLM loads it as
  `NemotronHMTPModel`. The W4A16 target and its quantization config stay untouched.

The int4 head beat the BF16 one on every count: faster (smaller draft reads), 19k more context per stream, and the
same acceptance ([RESULTS](RESULTS.md#step-by-step-speed-edits)).

## The engine

The stock vLLM 0.32.0 XPU build from [`../../qwen38-709l/scripts/build_venv.sh`](../../qwen38-709l/scripts/build_venv.sh),
plus one optional patch for the fp8 lm_head. Run from this recipe's folder:

```sh
sh ../qwen38-709l/scripts/build_venv.sh /opt/b70/vllm032
VLLM_ROOT=/opt/b70/vllm032/src/vllm /opt/b70/vllm032/venv/bin/python patches/lmhead-fp8/patch_nemotron_lmhead_fp8_032.py
```

- Everything Nemotron needs is in 0.32: the NemotronH model and its MTP drafter, the Triton Mamba2 mixer, the XPU int4
  MoE experts with relu², the sigmoid-with-bias router, head-128 attention in the stock attention library, the
  `nemotron_v3` reasoning parser and the `qwen3_coder` tool parser.
- The patch installs riceharvest's fp8 lm_head helper (`patches/lmhead-fp8/b70_target_lmhead_fp8.py`, MIT, license
  beside it) and hooks `logits_processor.py` and `nemotron_h.py`. With `B70_TARGET_LMHEAD_FP8=1` (the launcher's
  default) the BF16 head is converted to per-row fp8 at load and freed. Without the patch the switch does nothing.
  The patch is idempotent and leaves a `.bak` of each file.
- If you installed gpt-oss's optional rebuilt kernels ([../../gpt-oss-20b/kernels/](../../gpt-oss-20b/kernels/README.md))
  into this venv, Nemotron will not load: that library only has gpt-oss's head-64 variants. Use a second venv with the
  stock library.

## Serving

From this recipe's folder, with the venv above:

```sh
cp serve/nemotron.extra.example serve/nemotron.extra
VENV=/opt/b70/vllm032 MODEL=/models/nemotron-3.5-lightning-w4a16 CARD=0 sh serve/serve-nemotron.sh 8000
```

The launcher's base line is the safe eager config; `nemotron.extra` is appended last and turns on what we run:

| flag / setting | why |
|---|---|
| `--max-model-len 131072 --max-num-seqs 16` | 16 agent sessions of up to 131k; the pool holds 114k per stream at 16 |
| `--kv-cache-dtype fp8` | only 6 of 52 layers keep KV: about 3 KB per token in fp8 |
| `--tool-call-parser qwen3_coder --enable-auto-tool-choice` | Nemotron writes tool calls in the Qwen3-Coder XML form |
| `--reasoning-parser nemotron_v3` | splits thinking into `reasoning_content`; also needed for thinking budgets |
| `--no-enforce-eager` + `FULL_DECODE_ONLY` graphs (.extra) | decode at 1 stream 24.6 → 121.3 t/s |
| `--enable-prefix-caching --prefix-match-unit 64` (.extra) | cache hits inside a block: 16 long chats sent again 10.1 → 2.0 s (4.1 s with MTP on) |
| `--speculative-config {"method":"mtp","model":<int4 draft folder>,"num_speculative_tokens":3}` (.extra) | decode +57% at 1 stream, +39% at 16 |
| `--gpu-memory-utilization 0.93` (.extra) | 1,827,858-token pool instead of 1,668,189; held 16 × 69k at once with 0 preemptions |
| `B70_TARGET_LMHEAD_FP8=1` (launcher) | +11% at 1 stream, +5% at 16 (needs the patch) |
| `VLLM_TUNED_CONFIG_FOLDER=../ssm-configs` (launcher) | Mamba state update 163 → 86 µs per call at 16 streams |
| `../plugin` on `PYTHONPATH` (launcher) | `reasoning_effort` → thinking budget (below) |

About `--prefix-match-unit 64`: with prefix caching on, vLLM uses `mamba_cache_mode=align` and raises the attention
block to the Mamba state page (4,224-4,416 tokens here). Without a finer match unit a cache hit can only end on a block
edge, so each re-sent chat computes its last partial block again. With 64 it can end inside a block. Cached and cold
answers matched within greedy noise ([RESULTS](RESULTS.md#prefix-cache-granularity-speedpmu64-filljson-speedmamba-block-1056json-cache-checkout)).

Check the log on the first load for these lines:

- `Using 'XPU' WNA16 MoE backend` and `Using XPUExpertsWNA16`: the experts are int4.
- `Model loading took 16.61 GiB`
- `Using SSM config from ...` pointing at `ssm-configs/`. "Using default Mamba SSU config" means the tuned file was
  not found (see [PITFALLS](PITFALLS.md#mamba-state-update-config)).
- `[B70] target lm_head fp8: 131072x2688, fp16 freed` (the fp8 head is on).
- `nemoeffort: budgets {...}, default 8192` (the plugin is loaded).

The first load spends about 4 minutes warming up the Mamba2 Triton kernels; with graphs and MTP the first request
took about 4 minutes including the load. Later loads with the same flags reuse the compile cache.

## Thinking: on, off and budgets

Thinking is on by default. Nemotron's template only knows `enable_thinking`; vLLM 0.32 turns `reasoning_effort: none`
into `enable_thinking: false` by itself. Every other effort level was ignored, so thinking was unbounded (up to about
22k tokens on hard items). The plugin in `../plugin/` (`nemoeffort`) sets `thinking_token_budget` from the effort level
when a request did not set one itself:

| request sends | thinking |
|---|---|
| `"reasoning_effort": "none"` (or `"chat_template_kwargs": {"enable_thinking": false}`) | off |
| `"reasoning_effort": "minimal"` / `"low"` / `"medium"` / `"high"` | 512 / 2,048 / 4,096 / 8,192 tokens |
| `"reasoning_effort": "xhigh"` or `"max"` | unbounded |
| no `reasoning_effort` | `NEMO_DEFAULT_BUDGET` (launcher default 8,192) |
| its own `"thinking_token_budget": N` | N (always wins) |

On 700 graded items every budget beat no budget (540-543 → 557-567 correct), mostly because far fewer answers ran out
of room mid-thought ([RESULTS](RESULTS.md#thinking-budgets-gateseffort-plugin-graded-pairedcsv)). Set
`NEMO_EFFORT_BUDGETS` / `NEMO_DEFAULT_BUDGET` to change them, or `PLUGIN_DIR=` (empty) to turn the plugin off. The
budget bounds thinking, not total length: on a very hard question the model keeps working in the visible answer, so
still send a sensible `max_tokens`.

Check the plugin offline (no GPU) from this recipe's folder with the venv; it must end "20/20 passed":

```sh
PYTHONPATH=$PWD/plugin /opt/b70/vllm032/venv/bin/python plugin/selftest.py
```

A plain request, from any shell:

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nemotron", "messages": [{"role": "user", "content": "What is 17*23?"}], "reasoning_effort": "none"}'
```

NVIDIA's recommended sampling is temperature 1.0, top_p 0.95. If a router in front of vLLM sends the same request to
gpt-oss as well, don't add `thinking_token_budget` there: our gpt-oss server answers HTTP 400 for it. Send
`reasoning_effort` and let the plugin map it.

## Checks after a change

From this recipe's folder, with the server above on port 8000:

```sh
python ../gpt-oss-20b/bench/tool_flow.py http://localhost:8000 nemotron --sessions 16 --turns 8
python bench/cachecheck.py http://localhost:8000 nemotron
BASE_URL=http://localhost:8000/v1 MODEL=nemotron python bench/crossing_nemo.py 12000 6
```

- tool_flow: 0 format faults expected (we saw 0-1 in about 170 calls per run).
- cachecheck: cached vs cold answers identical apart from greedy noise; run it twice if one question differs.
- crossing: Nemotron slips at `func_95` with or without any change, so compare against a run without the change.
- For edits and speed we used the 709-L recipe's `editbench.py` and `bench.py` with `MODEL=nemotron`.

## Tuning the Mamba state update for your card

We ship our tuned config in [`../ssm-configs/`](../ssm-configs/). It is keyed by the device name your driver reports.
If yours differs, run vLLM's own benchmark from the venv's source tree with the card pinned:

```sh
cd /opt/b70/vllm032/src
unset ONEAPI_DEVICE_SELECTOR; export ZE_AFFINITY_MASK=0 PYTHONPATH=$PWD
../venv/bin/python benchmarks/kernels/benchmark_selective_state_update.py --headdim 64 --dstate 128 --ngroups 8 \
  --nheads 64 --batch-sizes 1 2 4 8 12 16 20 24 32 --save-configs --compare --validate
```

Our run is in [`../results/ssm-tuning.txt`](../results/ssm-tuning.txt): 9 of 9 sizes validated, 1.2× to 5.1× faster
than the default heuristic.
