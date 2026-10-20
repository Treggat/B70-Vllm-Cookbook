# Pitfalls (things that cost us time)

## Correctness

- **Replies corrupted at KV/state page boundaries (MTP + prefix caching).** On an older vLLM-XPU image (0.26.x), a
  reply that crossed a page boundary was sometimes garbled and then cut short, at different points on identical greedy
  runs. A speed-only benchmark won't notice: the replies just come out shorter. In our builds the fix is the mamba
  shared-prefix checkpoint (`--enable-mamba-shared-prefix-checkpoint` on 0.31) together with SergiioB's
  boundary/backward-copy patches. **Run `bench/crossing.py` after every change to MTP, prefix caching or the
  mamba/GDN cache flags.**
- **riceharvest v1 kernel + packed verify gives `!!!!` output** at some KV lengths. Use the v2 `.so` (zero-sum guard).
  `apply_stack.sh` refuses the v1 hash.
- **The GDN sync patches hung the card twice** (GPU copy-engine reset). The card recovered after unload both times. We
  leave them patched in but switched off. Other people run them, so your driver or firmware may behave differently;
  test them on their own if you try them.
- **A benchmark that doesn't contain its own answer.** Our depth test asked about argparse, but at 24k/32k argparse
  wasn't in the prompt. Make sure your retrieval prompt actually includes the text it asks about at every depth.

## Build

- **0.31 renamed a flag without listing it in the release summary:** `--enable-mamba-fine-grained-prefix-cache` →
  `--enable-mamba-shared-prefix-checkpoint`. The first load fails with "unrecognized arguments". Check every flag
  against `vllm serve --help=all` after an upgrade.
- **`VLLM_XPU_ENABLE_XPU_GRAPH` is gone in 0.31.** XPU graphs are on by default. On 0.30 you still had to set it,
  or short-context decode lost about 25%.
- **0.31 rejects per-request multimodal kwargs** (`mm_processor_kwargs`, `media_io_kwargs`) unless
  `--trust-request-mm-kwargs` is set. Check that your clients don't send them.
- **`triton==3.8.0+xpu` only exists on vLLM's XPU index, and since 0.32 that index is versioned:**
  `https://wheels.vllm.ai/0.32.0/xpu/`. The bare `/xpu/` path stopped tracking the release the day 0.32 came out.
- **Patch scripts that patch every vLLM they find** (see RECIPE). Pin them to one tree.
- **"dubious ownership":** don't `chown` the vLLM git checkout away from the user who runs pip.
- **Compile cache keyed on flags, not env:** give each switch combination its own `VLLM_CACHE_ROOT`.
- **`--max-num-batched-tokens 16384`** made cold prefill 2–3× slower on this XPU build. We use 3584.
- **Memory query reads 0 GiB free** under some Level Zero driver/container combinations. Fixed by `patch_meminfo_fallback`.
- **0.32 rewrote the GDN speculative-decode metadata path** (`v1/attention/backends/gdn_attn.py`): SergiioB's
  `patch_mtp_boundary` and riceharvest's three `gdn_*` patches all refuse on it. The sync-free slicing they added is
  upstream now; the boundary fix is not, so we re-based it (`patches/patch_mtp_boundary_032.py`) and test it with
  `bench/maxlen.py`. Dry-run every patch script against a clean clone of the new tag before you build.

## Serving

- **"4 streams" is an admission cap, not a promise.** What actually limits concurrent long chats is the KV pool.
  With 268k tokens, two 90k chats filled it and the 3rd and 4th waited. 389k fits 4 × ~95k. During cold prefill
  vLLM still runs one long prompt at a time.
- **GPU copy-engine reset while loading, with embed UVA on.** The drafter builds its own `embed_tokens`, and
  riceharvest's `patch_embed_uva` offloads that one too, so the draft's weight load writes 2.37 GiB into a second pinned
  host copy over the copy engine. Twice that write ended in an `Engine reset: engine_class=bcs` in `dmesg`, the engine
  core died and the server hung in "starting". vLLM throws the drafter's table away right after the load anyway
  (`_maybe_share_embeddings`), so `apply_stack.sh` now leaves it on the GPU (`B70_EMBED_UVA_DRAFT=1` restores the old
  behaviour). 5 back-to-back cold loads after the change: no resets, same speed, byte-identical greedy output. Symptom
  to look for: the log's "Total CPU offloaded parameters" jumping from 2.37 to 4.74 right before the hang.
- **Don't redirect the server's output inside its launch config** (`exec >> file` in an env file the launcher
  sources). The server runs fine, but a supervisor such as llama-swap then shows an empty log for the model.
- **llama-swap users:** scraping `/upstream/<model>/metrics` for a model that isn't loaded **loads it**, swapping out
  whatever is running. Only scrape the seat that's already up.
- **Do the VRAM math before raising KV.** CUDA-graph-style capture memory isn't counted by
  `gpu_memory_utilization`. Set `--kv-cache-memory-bytes` explicitly and leave about 1.5 GiB of headroom.
- **`--kv-offloading-size` does nothing useful for this hybrid model** (0.31 and 0.32): blocks go to the CPU tier and
  are never found again, so an evicted prompt re-prefills in full. It does pin N GiB of `/dev/shm`. Leave it out.
- **The MTP drafter's weight load is where a host-RAM-starved box resets the GPU.** Both copy-engine resets we had
  came during that step while another container was compiling with ~20 GiB of RAM free. Keep the box's RAM quiet
  during loads, and don't give the drafter its own UVA copy (above).

## Two cards (not used here)

TP2 across two B70s worked for us but was slower than one card. We had to set `CCL_TOPO_P2P_ACCESS=0` to get past an
IPC error with torch's bundled oneCCL, and that probably turned off direct card-to-card transfers.
[zebgop-ops](https://github.com/zebgop-ops/qwen38-27b-xpu-tp) reports that XPU graphs under TP need oneCCL ≥ 2022.1.2:
older versions give garbage output and 0% draft acceptance. Read that before trying TP2.
