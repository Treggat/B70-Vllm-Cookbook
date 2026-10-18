# Qwen3.8-27B 709-L on one Intel Arc Pro B70 (vLLM 0.31 XPU)

A reproducible recipe for serving a **merged Qwen3.8-27B fine-tune** ([DavidAU](https://huggingface.co/DavidAU)'s [*TWIN-TURBO Fable Cold-Fusion 709-L*](https://huggingface.co/DavidAU/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-NM-DAU-NEO-MTP-GGUF))
as an **AWQ W4A16** checkpoint on **one Arc Pro B70 (32 GB)**, with MTP speculative decoding, vision, 131k context and
4 concurrent streams. vLLM 0.31.0 (XPU, built from source), the [SergiioB][sergiiob] and [Vcvzgbzz][vcv] cookbook
patches, and [riceharvest][rice]'s b70 layer on top, plus three small patches of our own.

Most B70 write-ups cover base Qwen3.8 in GPTQ or EXL3. This one covers:

- a merged model shipped as compressed-tensors W4A16, re-laid into the AutoAWQ layout that vLLM-XPU can load ([docs/AWQ.md](docs/AWQ.md))
- stacking three community patch sets on one tree, including the one clash between them ([docs/RECIPE.md](docs/RECIPE.md))
- per-patch measurements of riceharvest's layer, each patch loaded on its own ([docs/RESULTS.md](docs/RESULTS.md))
- what broke along the way ([docs/PITFALLS.md](docs/PITFALLS.md))

Before vLLM we ran the same model on llama.cpp SYCL with our own kernel patches. That work is in
[b70-llamacpp-cookbook](https://github.com/Treggat/b70-llamacpp-cookbook).

## Results

Same benchmark script, same box, same day, through an OpenAI-compatible endpoint, thinking off. Single runs: treat
about ±5% as noise.

| | vLLM 0.30 + cookbook | vLLM 0.31 + cookbook | **0.31 + cookbook + riceharvest layer** |
|---|---|---|---|
| short code, t/s (avg of 3) | 113.9 | 119.4 | **131.6** (+16%) |
| 4 streams, summed t/s | 221.3 | 236.6 | **265.0** (+20%) |
| 75k-token read, first token (cold) | 66.9 s | 66.8 s | **53.8 s** (−20%) |
| 75k prefill | 1,133 t/s | 1,134 | **1,408** (+24%) |
| 75k decode | 47.7 t/s | 47.8 | **61.4** (+29%) |
| KV pool | 268k tokens | 268k | **389k** (+45%, 4 × ~97k)¹ |
| MTP acceptance (code) | 72% | 75% | 73% |

¹ The speed numbers in that column were measured with a 13 GiB pool (349k tokens). We raised it to 14.5 GiB afterwards and stress-tested 4 × 95k chats at once (0 preemptions).

At 100k context (0.31 → 0.31 + riceharvest): first token 97.6 s → 75.8 s, decode 40.6 → 58.4 t/s.
All three configs pass the vision, video and page-boundary tests (`bench/crossing.py`). Every run, including the per-patch ones, passes the corruption canary with 0 preemptions.

## Quick start

```sh
sh scripts/build_venv.sh /opt/b70/vllm031                         # vLLM 0.31.0 XPU from source, own venv
# riceharvest's v2 attention kernel (zero-sum guard): their GitHub release v3, or their kernels/build.sh.
# The one we ran: sha256 8d1bf4d1ea74...; apply_stack.sh refuses the v1 kernel (aa241d52a601...)
sh scripts/apply_stack.sh /opt/b70/vllm031 ./libattn_kernels_xe_2.b70.so
python scripts/make_draft_vocab.py serve/draft_vocab_98304.json
python scripts/ct2awq.py <downloaded W4A16 dir> <out dir>          # one-time, see docs/AWQ.md
cp serve/prod.env.example serve/prod.env; cp serve/prod.extra.example serve/prod.extra
VENV=/opt/b70/vllm031 MODEL=<out dir> sh serve/serve-709l.sh 8000
```

Then check it before trusting it: `cd bench && python crossing.py && python bench.py mine`.

## Layout

| path | what |
|---|---|
| `patches/` | our three vLLM patches (memory-query fallback, BF16 MTP `fc`, video sent as `image_url`) |
| `scripts/build_venv.sh` | vLLM 0.31.0 XPU source build in its own venv (indexes, pins) |
| `scripts/apply_stack.sh` | clones the upstream cookbooks at pinned commits, points their patches at one tree, applies the whole stack fail-closed |
| `scripts/ct2awq.py` | compressed-tensors pack-quantized W4A16 → AutoAWQ gemm, bit-identical |
| `scripts/make_draft_vocab.py` | Vcvzgbzz's 98,304-row draft trim as a JSON id list for riceharvest's patch |
| `serve/` | launcher + the measured config (`prod.env.example`, `prod.extra.example`) |
| `bench/` | speed/sanity bench, page-boundary test, 4 × long-chat KV stress |
| `results/` | raw JSON for every run in the tables (generated text stripped) |

## Credits

This stands on other people's work. Please go star them:

- **[SergiioB/intel-arc-pro-b70-inference-cookbook][sergiiob]** (MIT): the base patch set (MTP nightly/boundary, int4 draft lm_head + MTP linears, prefix-cache fixes) and most of what is known about vLLM on the B70.
- **[Vcvzgbzz/b70-cookbook][vcv]** (MIT): the draft-vocab trim and the recipe our first vLLM seat was built from.
- **[riceharvest/b70-qwen38-cookbook][rice]** (MIT): packed MTP verify, the b70 attention kernel, fp8 lm_head, embed UVA, kv-splits.
- **Claude** (Anthropic; Claude Opus 5.5, through Claude Code): ported and tested the patch stack, wrote our three
  patches and the scripts, ran the measurements and wrote this cookbook.
- Model: **[DavidAU](https://huggingface.co/DavidAU)** for the [709-L merge](https://huggingface.co/DavidAU/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-NM-DAU-NEO-MTP-GGUF), and **TheUnderscore** for the [W4A16 quant](https://huggingface.co/TheUnderscore/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-W4A16-AWQ). See [docs/AWQ.md](docs/AWQ.md).

We link to the upstream patches and don't copy them in. `apply_stack.sh` fetches them at the commits we measured.

## License

Our code and docs: MIT ([LICENSE](LICENSE)). Upstream patches and kernels keep their own licenses.

[sergiiob]: https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
[vcv]: https://github.com/Vcvzgbzz/b70-cookbook
[rice]: https://github.com/riceharvest/b70-qwen38-cookbook
