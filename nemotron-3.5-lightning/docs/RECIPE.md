# Nemotron 3.5 Lightning: recipe

[← recipe page](../README.md)

What we run now on one B70, step by step. Every step was run on vLLM 0.32.0 XPU on 2026-10-26.

## The model and the checkpoint

`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (`NemotronHForCausalLM`) has 52 layers: 23 Mamba2 mixers, 6 attention
layers (32 query heads, 2 KV heads, head dim 128) and 23 MoE layers (128 experts, top-6, sigmoid router, one shared
expert, relu² activation). About 3B parameters are active per token. Native context is 262k.

The BF16 weights (about 60 GB) don't fit on a 32 GB card, and NVIDIA's own FP8 (30.1 GB) leaves no room for KV.
NVFP4 runs on NVIDIA hardware only. We use
[useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16](https://huggingface.co/useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16):

- compressed-tensors GPTQ, **symmetric int4, group 64**, one 17.8 GB safetensors file (17,826,201,400 bytes)
- all routed and shared experts, the Mamba projections and attention q/k/v/o are int4; norms, Mamba A/D/dt/conv,
  the router, embeddings and `lm_head` stay BF16
- the multi-token-prediction head is removed (`num_nextn_predict_layers: 0`), so there is no built-in speculative
  decoding
- the chat template is NVIDIA's, unchanged

Symmetric int4 matters on XPU: vLLM's XPU int4 MoE path (`XPUExpertsWNA16`) accepts symmetric weights only. An AWQ
checkpoint with zero points gets no XPU MoE backend.

Download it and check it, from any folder with the Hugging Face CLI:

```sh
hf download useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16 --local-dir /models/nemotron-3.5-lightning-w4a16
cd /models/nemotron-3.5-lightning-w4a16 && sha256sum -c SHA256SUMS
```

All 25 files must say OK.

## The engine

The stock vLLM 0.32.0 XPU build from [`../../qwen38-709l/scripts/build_venv.sh`](../../qwen38-709l/scripts/build_venv.sh)
serves it as is. Run from this recipe's folder:

```sh
sh ../qwen38-709l/scripts/build_venv.sh /opt/b70/vllm032
```

Everything Nemotron needs is in 0.32: the NemotronH model, the Triton Mamba2 mixer (no platform guard), the XPU int4
MoE experts with relu², the sigmoid-with-bias router (`topk_sigmoid` is in the XPU MoE library), head-128 attention
in the stock attention library, the `nemotron_v3` reasoning parser and the `qwen3_coder` tool parser.

If you installed gpt-oss's optional rebuilt kernels ([../../gpt-oss-20b/kernels/](../../gpt-oss-20b/kernels/README.md))
into this venv, Nemotron will not load: that library only has gpt-oss's head-64 variants. Use a second venv with the
stock library. We made ours with `cp -al` of the first venv and an unlinked copy of the stock `.so`, so the two share
everything else.

## Serving

From this recipe's folder, with the venv above:

```sh
cp serve/nemotron.extra.example serve/nemotron.extra
VENV=/opt/b70/vllm032 MODEL=/models/nemotron-3.5-lightning-w4a16 CARD=0 sh serve/serve-nemotron.sh 8000
```

The launcher's base line is the safe eager config; `nemotron.extra` is appended last and turns on what we run:

| flag | why |
|---|---|
| `--max-model-len 131072 --max-num-seqs 16` | 16 agent sessions of up to 131k each; the pool holds 164k per stream at 16 |
| `--kv-cache-dtype fp8` | only 6 of 52 layers keep KV: about 3 KB per token in fp8 |
| `--gpu-memory-utilization 0.9` | weights load as 16.61 GiB; the rest goes to KV and Mamba state |
| `--tool-call-parser qwen3_coder --enable-auto-tool-choice` | Nemotron writes tool calls in the Qwen3-Coder XML form |
| `--reasoning-parser nemotron_v3` | splits thinking into `reasoning_content`; also needed for `thinking_token_budget` |
| `--no-enforce-eager` + `FULL_DECODE_ONLY` graphs (in the .extra) | decode at 1 stream 24.6 → 121.3 t/s |
| `--enable-prefix-caching` (in the .extra) | 16 long chats sent again come back in 10.1 s instead of 163.7 s |
| `VLLM_TUNED_CONFIG_FOLDER=../ssm-configs` (in the launcher) | Mamba state update 163 → 86 µs per call at 16 streams, decode +6.6% at 16 |

With prefix caching on, vLLM picks `mamba_cache_mode=align` by itself and raises the attention block size to 4,224
tokens to match the Mamba state page. That costs some pool (3,214,904 → 2,630,376 tokens) and means the last partial
block of each request is computed again (96.3% hit on the 16 × 69k test). Cached and uncached answers at temperature
0 were byte-identical on 8 of 8 questions.

Check the log on the first load for these lines:

- `Using 'XPU' WNA16 MoE backend` and `Using XPUExpertsWNA16`: the experts are int4. If they were treated as
  unquantized, the load would try to allocate about 55 GB of BF16 experts.
- `Model loading took 16.61 GiB`
- `Using SSM config from ...` pointing at `ssm-configs/`. "Using default Mamba SSU config" means the tuned file was
  not found (see [PITFALLS](PITFALLS.md#mamba-state-update-config)).

The first load spends about 4 minutes warming up the Mamba2 Triton kernels; with graphs on, the first request took
231–234 s including the load. Later loads with the same flags reuse the compile cache.

## Thinking: on, off and budgets

Thinking is on by default and unbounded. Nemotron ignores `reasoning_effort`: its template only knows
`enable_thinking`. Two request fields control it:

| you want | send |
|---|---|
| no thinking | `"chat_template_kwargs": {"enable_thinking": false}` |
| thinking capped at N tokens | `"thinking_token_budget": N` (needs `--reasoning-parser`, which the launcher sets) |

A plain example, from any shell:

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nemotron", "messages": [{"role": "user", "content": "What is 17*23?"}],
  "chat_template_kwargs": {"enable_thinking": false}}'
```

That answer takes 4 tokens. Unbudgeted thinking ran up to about 22k tokens on hard graded items, so give agents a
budget if latency matters. NVIDIA's recommended sampling is temperature 1.0, top_p 0.95; top_p costs about 2.5% of
decode speed at 16 streams.

If a router in front of vLLM sends the same request to gpt-oss as well, don't add `thinking_token_budget` there:
our gpt-oss server answers HTTP 400 for it because it runs without that reasoning config.

## Tuning the Mamba state update for your card

We ship our tuned config in [`../ssm-configs/`](../ssm-configs/). It is keyed by the device name your driver reports.
If yours differs, run vLLM's own benchmark from the venv's source tree with the card pinned:

```sh
cd /opt/b70/vllm032/src
unset ONEAPI_DEVICE_SELECTOR; export ZE_AFFINITY_MASK=0
../venv/bin/python benchmarks/kernels/benchmark_selective_state_update.py --headdim 64 --dstate 128 --ngroups 8 \
  --nheads 64 --batch-sizes 1 2 4 8 12 16 20 24 32 --save-configs --compare --validate
```

Our run is in [`../results/ssm-tuning.txt`](../results/ssm-tuning.txt): 9 of 9 sizes validated, 1.2× to 5.1× faster
than the default heuristic.
