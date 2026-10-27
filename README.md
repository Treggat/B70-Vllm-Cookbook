# B70 vLLM Cookbook

Recipes for serving LLMs on the **Intel Arc Pro B70 (32 GB)** with **vLLM 0.32 (XPU)**. Every recipe was run on
the same box, two B70s with one model per card, and every number in it is ours. Each recipe is one folder with its
page, patches, launcher, benchmarks and raw results.

## Recipes

| recipe | model | format | what's in it | speed on one B70 |
|---|---|---|---|---|
| [**Qwen3.8-27B 709-L**](qwen38-709l/README.md) | DavidAU's merged Qwen3.8-27B fine-tune | AWQ W4A16 | MTP speculative decoding that switches to drafts copied from the context when the output copies it (whole-file edits +50%), a shortlist draft head (-3.4% step time), probabilistic drafts (sampled prose +14-18%) and a V-split verify kernel (decode behind 210k of context +23%), vision and video, the full 262k context, three community patch sets stacked on one tree with per-patch measurements, and an attempt to train a DFlash2 drafter for the merge | 135 t/s one stream on code, 64 t/s on sampled prose, 239 t/s on whole-file edits, about 290 t/s across 4 |
| [**gpt-oss-20b**](gpt-oss-20b/README.md) | OpenAI gpt-oss-20b, plus our anti-loop fine-tune ([weights](https://huggingface.co/Taggert2222/gpt-oss-20b-antidoom)) | MXFP4 | harmony tool calling with a strict tool-call grammar, a reasoning-effort cap, an [Antidoom](https://github.com/Liquid4All/antidoom) fine-tune trained on two B70s that removes doom loops, and what an agent harness does to make gpt-oss fake its tool calls | 117 t/s one stream, 826 t/s across 16 (fp8 KV) |
| [**Nemotron 3.5 Lightning**](nemotron-3.5-lightning/README.md) | NVIDIA Nemotron 3.5 Lightning 30B-A3B (Mamba2 / attention / MoE hybrid) | W4A16 (int4 g64) | stock vLLM with XPU graphs, prefix caching and a tuned Mamba state-update config, 164k of context per stream at 16, thinking on/off and budgets, and a same-day bake-off against gpt-oss-20b on speed, honesty and accuracy | 121 t/s one stream, 691 t/s across 16 (fp8 KV) |

Speeds are single-run measurements through an OpenAI-compatible endpoint. Each recipe page gives the conditions and
the noise.

## What the recipes share

- **The engine.** All three run vLLM 0.32.0 built from source for XPU in its own venv, with torch 2.14.0+xpu, triton
  3.8.0+xpu and vllm-xpu-kernels 0.1.15.4 (the same pins as 0.31), since 2026-10-19.
  [`qwen38-709l/scripts/build_venv.sh`](qwen38-709l/scripts/build_venv.sh) builds it, and its git history holds the
  0.31 version. Each page marks which of its numbers were taken on 0.31; the gpt-oss page adds one extra check for
  its harmony library.
- **One card per model.** On a two-card box, pin each server to its card with `ZE_AFFINITY_MASK` (0 or 1). With
  torch 2.13 and later, the card is picked the Level Zero way, so don't also set `ONEAPI_DEVICE_SELECTOR`.
- **Check before you trust.** Every recipe ships a sanity test next to its speed bench. Run it after any change to
  flags, patches or weights.

## Adding a recipe

One folder per model, named after it, holding `README.md` (the page), `docs/`, and whichever of `patches/`,
`scripts/`, `serve/`, `bench/` and `results/` it needs. Paths in a recipe are relative to its own folder. Add one row
to the table above. [AGENTS.md](AGENTS.md) has the full house rules (for people and AI agents alike), and
`python tools/check_docs.py` checks links, layout tables and hygiene before a commit.

## Related

Before vLLM we ran Qwen3.8 on llama.cpp SYCL with our own kernel patches:
[b70-llamacpp-cookbook](https://github.com/Treggat/b70-llamacpp-cookbook).

## Credits

Each recipe credits the people and projects it builds on. Please go star them. The recipes were ported, tested,
measured and written with **Claude** (Anthropic), through Claude Code.

## License

Our code and docs: MIT ([LICENSE](LICENSE)). Upstream patches, kernels, models and training code keep their own
licenses, and each recipe says which.
