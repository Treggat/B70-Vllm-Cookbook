# One attention kernel for MTP and the DFlash2 drafter

riceharvest's b70 attention library (`libattn_kernels_xe_2.b70.so`) only contains the shapes the MTP seat uses. The
DFlash2 drafter (head 128, paged, non-causal, sliding window 2048) is not among them, so a drafter test dies in warmup
with "Chunk prefill kernel tuple not compiled for this configuration" and has to fall back to the slow stock library.

These two lists build one library with both: riceharvest's live lists plus the drafter's head-128 lines from
[Vcvzgbzz's](https://github.com/Vcvzgbzz/b70-cookbook) `cp_b70.conf`. Same sources and patches as riceharvest's
[`kernels/build.sh`](https://github.com/riceharvest/b70-qwen38-cookbook) (vllm-xpu-kernels release/0.1.15.4 + sycl-tla
87f68506, with `tla-fast3`, `xk-fold` and `zero-sum-guard`); only the lists change. Point that script's
`VLLM_CHUNK_PREFILL_CONFIG` / `VLLM_PAGED_DECODE_CONFIG` at these files. About 70 minutes at one compile job, peak
~8.4 GiB per job.

**Exactness, checked against stock with Vcvzgbzz's `tests/kcheck.py`:**

| | cases | result |
|---|---|---|
| MTP shapes, packed verify on, against riceharvest's live library | 16 | bit-identical (it is the same kernel) |
| MTP shapes, packed verify off, against stock | 16 | bit-identical |
| drafter shapes (head 128, 8-token blocks, window on and off, fp8 and fp16 KV, 1.5k-65k) against stock | 72 | bit-identical, no fallback |

One known gap: with packed verify on, a *causal* head-128 drafter without a window needs decode
`16,128,64,false,false,false`, which is not in the list. The DFlash2 drafter here runs non-causal with a window, so it
never asks for it. Add the line and rebuild (ninja only compiles the new kernel) if yours does.

Measured effect in [docs/RESULTS.md](../../docs/RESULTS.md): the MTP seat on this library is at parity with
riceharvest's; the drafter's decode at 65k-100k context is 27-45% faster than on the stock library.
