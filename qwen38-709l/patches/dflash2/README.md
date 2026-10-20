# DFlash2 drafter on vLLM 0.32 (opt-in, not used by this recipe)

Re-bases of [Vcvzgbzz/b70-cookbook][vcv] dd083d5's DFlash2 patches (written against 0.31) onto vLLM 0.32. They are
here so the measurement in [docs/RESULTS.md](../../docs/RESULTS.md) can be reproduced and so anyone with a drafter
trained on their own target can try it. **On the 709-L merge the published drafter is 3× slower than MTP-5**, so
`apply_stack.sh` does not apply these.

| file | what it does |
|---|---|
| `patch_dflash2_drafter_032.py` | runs the BF16-trained drafter under the FP16 target (`B70_DRAFT_DTYPE=bfloat16`, casts at the three places data crosses), int4 drafter linears (`B70_DFLASH_DRAFT_INT4=1`, needs `b70_dflash_draft_int4.py` copied into `vllm/model_executor/models/`), and DFlash2 candidates from the int4 draft head (`B70_DRAFT_LMHEAD_INT4=1`, mapped through riceharvest's JSON draft-vocab ids). Every change is a no-op with the gates unset. |
| `patch_swa_prefix_cache_032.py` | the cookbook's sliding-window prefix-cache fix (`B70_SWA_FINE_GRAINED=1`): without it the drafter's sliding-window layers make the hybrid target reuse nothing from the prefix cache |
| `b70_dflash_draft_int4.py` | the int4 helper, unchanged from the cookbook |

Both scripts take `VLLM_ROOT` (the `vllm/` package dir) or a path argument, are idempotent and fail-closed (every
anchor exactly once, all files or none) and `py_compile` what they touch.

Serve with `--speculative-config '{"method":"dflash","model":"<drafter dir>","num_speculative_tokens":7,"kv_cache_dtype":"auto"}'`
and the gates above. **Use the stock `libattn_kernels_xe_2.so`, not riceharvest's**: the b70 build has no
chunk-prefill kernel for the drafter's shape (head 128, paged, non-causal, sliding window) and the first load dies
in warmup with "Chunk prefill kernel tuple not compiled for this configuration". Vcvzgbzz's `patches/kernels/build.sh`
with `cp_b70.conf` builds the fast kernels including that shape (needs oneAPI 2026.x).

[vcv]: https://github.com/Vcvzgbzz/b70-cookbook
