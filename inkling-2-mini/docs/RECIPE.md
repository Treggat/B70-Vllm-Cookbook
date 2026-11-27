# Inkling 2 Mini: recipe

[← Inkling 2 Mini](../README.md)

The commands are in the [quick start](../README.md#quick-start). This page says what the engine is and what the
patch changes. Paths are relative to the recipe folder (`inkling-2-mini/`).

## Engine

- **Source:** [afierka-intel/vllm](https://github.com/afierka-intel/vllm), branch
  `afierka/ink2-v2-9-autotune-row-kernels-r11` at `19dc25c` (vLLM main `65ada0c40e` plus Intel's Inkling 2 XPU
  work), with the first commit of vLLM PR 69444 (`0811005`, readable conversation history in the KV cache)
  cherry-picked. The cherry-pick has one conflict, in `vllm/v1/kv_cache_interface.py`; keep both sides. The PR's
  second commit targets a newer core and is not applied.
- **Pins** (from the branch's `requirements/xpu.txt`): torch 2.15.0+xpu, triton 3.9.0+xpu, vllm_xpu_kernels 0.1.17.
- **Stock limits on XPU:** text only, BF16 KV, no DSpark drafter.

`patches/patch_ink2_b70.py` edits the checkout in place. Each change does nothing unless its switch is set:

| switch | what it does | measured here |
|---|---|---|
| (always on) | falls back to torch's free-memory query when the kernels' query reads 0 | needed to load |
| `B70_INK2_DSPARK=1` | lets the DSpark drafter load on XPU: a portable RMSNorm in place of the CUDA one, no multi-GPU all-reduce, and a value select in place of a pointer select in one Triton kernel | yes |
| `B70_INK2_MOE_TILE`, `B70_INK2_MOE_PREFILL` | tile sizes and warps for the NVFP4 emulation expert kernel | yes |
| `B70_INK2_FP8KV=1` | allows `--kv-cache-dtype fp8` on XPU; the drafter's own KV stays BF16 | yes |
| `B70_TARGET_LMHEAD_FP8=1` | fp8 output head; cannot be combined with the drafter, which reads the BF16 head | no numbers on this page |
| `B70_INK2_VISION=1` | image input through vLLM's XPU attention (drop `--language-model-only` to use it) | no numbers on this page |

## First load

The first load of the fp8 KV config is slow twice: about 7 minutes of silent kernel compilation, then about 38
minutes of Triton autotuning during the first requests (our first reply took 1,483 s). Both are cached per config
(`serve/serve-inkling.sh` keys the cache on `inkling.env` and `inkling.extra`).

## Sampling

Send temperature 1.0. At 0.1 the reasoning looped on 16 of 900 test prompts
([results](../README.md#bake-off-against-nemotron)).
