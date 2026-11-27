# Inkling 2 Mini: pitfalls

[← Inkling 2 Mini](../README.md)

- **Any change to the flags or switches pays the compile and autotune cost again.** The cache is keyed on both
  files.
- **The profiler can be started once per load.** A second `POST /start_profile` in the same load hung the engine
  for us; reload and profile the case you need first.
- **Do not judge the drafter on greedy drafts**, and do not judge the model at temperature 0.1.
- **`grf_mode: large` is rejected by Triton on XPU**, and N256/K256 or K512 expert tiles run out of shared memory.
- **The fp8 lm_head switch and the drafter do not combine.** The drafter reads the target's BF16 head; the patch
  exits if both are set.
