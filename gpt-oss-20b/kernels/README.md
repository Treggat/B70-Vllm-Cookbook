# Optional: faster fp8 prefill kernels for gpt-oss

[← gpt-oss-20b](../README.md) · [RECIPE.md](../docs/RECIPE.md#kv-cache-fp8)

With an fp8 KV cache on the stock vllm-xpu-kernels 0.1.15.4, gpt-oss-20b's cold prefill is about 25% slower than with
bf16, because converting fp8 (e4m3) back to bf16 takes 11 instructions per 64 values on Xe2. This folder rebuilds the
attention library (`libattn_kernels_xe_2.so`) with a 5-instruction conversion and only the kernel variants gpt-oss
uses. Decode speed doesn't change, and the output is byte-identical to the stock kernels.

You don't need this. The stock kernels work fine with fp8 KV. It only makes the first read of a long prompt faster,
and you have to rebuild it whenever vllm-xpu-kernels or vLLM changes.

## What it bought us

One B70, vLLM 0.31.0, variant B weights, fp8 KV on all layers, measured 2026-10-18:

| | bf16 KV | fp8 KV, stock kernels | **fp8 KV, this build** |
|---|---|---|---|
| 51k-token cold read: first token | 9.0 s | 11.1 s | **10.0 s** |
| 16 × 69k cold fill (928k prompt tokens, same run for both)¹ | – | 217 s | **195 s** |
| decode, 1 / 4 / 16 streams (summed tok/s) | 118 / 308 / 820 | 117 / 309 / 818 | 117 / 308 / 821 |
| greedy output, 20 prompts | – | – | 20/20 byte-identical to stock fp8, reasoning included |

¹ The 217 s was measured on a build with riceharvest's patches only. They don't touch the bf16 conversion gpt-oss
uses (its 51k first token stayed at 11.16 s), so it stands in for stock here.

So it wins back about half of fp8's prefill cost. Gates on this build (variant B, fp8 all layers): loops 1/200 at
medium and 1/200 at high, graded 83.3%, tool_flow 0 format faults. PTC came in at 50–51/72, but the stock kernels on
the same seat in the same hour scored 48 and 55, so we read that as the probe's noise on that evening, not the kernel.

It also makes `--kv-cache-dtype-skip-layers sliding_window` load (see [PITFALLS.md](../docs/PITFALLS.md)). We don't use
it: no faster, a 3% smaller pool (1,248,218 tokens), and its gates came in one item under our bars.

## What's in it

| file | what | from |
|---|---|---|
| `patches/tla-fast3-b70.patch` | sycl-tla: fast e4m3 → fp16 conversion | [riceharvest/b70-qwen38-cookbook](https://github.com/riceharvest/b70-qwen38-cookbook) (MIT, see `patches/LICENSE-riceharvest`); upstream as [intel/sycl-tla#859](https://github.com/intel/sycl-tla/pull/859) |
| `patches/bf16fast5.py` | sycl-tla: the same trick for e4m3 → bf16 (what gpt-oss needs), run after tla-fast3 | ours |
| `patches/xk-fold-b70.patch` | vllm-xpu-kernels: folds the conversion's scale into softmax/V, split-K epilogue fix (its packed spec-verify path stays off: gpt-oss has no MTP) | riceharvest (MIT) |
| `patches/zero-sum-guard.patch` | vllm-xpu-kernels: a fully masked split writes 0 instead of NaN | riceharvest (MIT) |
| `gptoss_decode.conf`, `gptoss_prefill.conf` | the kernel variants gpt-oss uses: head 64, with and without sliding window and sinks, page sizes 16/32/64, bf16 and fp8 | ours |
| `build.sh` | install the compiler, fetch the sources at the pinned commits, patch, build | ours |

Pinned sources: vllm-xpu-kernels `release/0.1.15.4` (`ddf336d`), sycl-tla `87f6850680a580654b9ea2c80dbc01aeb36ad231`.
That's what vLLM 0.31.0 and 0.32.0 pin. For any other vllm-xpu-kernels version, rebuild against that version's
sources, or go back to stock.

The configs include the bf16 variants, so this build also serves a bf16 KV cache (dropping fp8 later won't hit a
missing kernel). They don't include anything for other models: a venv shared with another model needs the stock
library.

## Build

```sh
WORK=/scratch/xk-gptoss VENV=/opt/b70/vllm031 ./build.sh
```

It needs Docker, an 18 GB memory cap for the build container (two compiler jobs; ours peaked around 10 GB), and about
40 minutes. It installs Intel's oneAPI DPC++ 2026.0 compiler into `$WORK/oneapi`. The result is about 42 MB (stock is
307 MB, because it covers every model's head sizes).

We built with `VENV` set, so CMake used our vLLM 0.31 venv's torch, inside a plain Ubuntu 24.04 image that only
supplied cmake and ninja. Without `VENV`, the script uses the `vllm/vllm-openai-xpu:v0.31.0` image's own Python, the
way riceharvest builds his. We haven't run that route ourselves.

## Install

The running server has the library memory-mapped, so never copy over it while it's loaded.

```sh
K=$VENV/venv/lib/python3.12/site-packages/vllm_xpu_kernels
# 1. stop or unload the server
cp -p $K/libattn_kernels_xe_2.so $K/libattn_kernels_xe_2.so.stock        # once
cp -p libattn_kernels_xe_2.so $K/.libattn_kernels_xe_2.so.new
mv -f $K/.libattn_kernels_xe_2.so.new $K/libattn_kernels_xe_2.so           # rename, not overwrite
# 2. start the server, then:
grep -c "not compiled" <server log>   # must be 0: a missing variant falls back to slow PyTorch attention
```

Then check the output didn't change: `bench/bench_kv.py --only greedy` on stock and on this build, and compare. Ours
matched 20/20. Rollback is the same rename with the `.stock` file.

## Credits

The fp16 conversion, the scale fold and the zero-sum guard are riceharvest's work for Qwen3.8 on the B70
([b70-qwen38-cookbook](https://github.com/riceharvest/b70-qwen38-cookbook), MIT). vllm-xpu-kernels is Apache-2.0 and
sycl-tla is BSD-3-Clause. A library built from them is a derivative work under those licenses.
