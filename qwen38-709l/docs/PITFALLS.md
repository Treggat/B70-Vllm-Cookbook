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

## Speculative decoding

- **Variable draft counts on the hybrid (GDN) target.** vLLM 0.32's GDN metadata builder sizes a pure spec step as
  `num_speculative_tokens + 1` tokens per request. With Dynamic SD, or anything else that verifies fewer drafts than
  the configured maximum, the shorter step looks like a truncated final group: it goes through the stateful prefill
  path, which cannot roll back rejected drafts. See `patches/combo-spec/patch_gdn_varlen_spec_032.py`.
- **Packed verify only for checked shapes.** riceharvest's `spec_packed.py` packs any uniform q_len; only q_len 2, 3
  and 6 are measured on the b70 `.so`. Turning on 6, 7 or 15 drafts with `B70_SPEC_PACKED=1` silently packs an
  unchecked shape. `patches/combo-spec/patch_spec_packed_guard_032.py` sends other q_len values to the stock path.
- **A copy detector that trusts the rightmost match.** In code, 4-grams repeat, so `rfind` of the last few tokens
  keeps landing on a nearer repeat instead of the real source. Confirm a copy by comparing the newly generated
  tokens with what follows the source; do not track the match offset.
- **A copy cutoff below your agent's system prompt.** We capped copy drafting at 32k tokens to keep a 16-token verify
  cheap. Our agent harness's system prompt and tools alone are ~40k, so the agent never got a single copy step while every direct
  benchmark looked great. Measure through the real harness. The same goes for the worker's scan range
  (`B70_COMBO_SCAN_CTX`): a source past it is invisible.
- **A drafter trained on the base model does not transfer to a merge.** The published DFlash2 drafter accepts 2-5% of
  its tokens on the 709-L (45%+ on base Qwen3.8), because it reads inner layers the merge rewrote. MTP is fine: it
  reads only the final hidden state. Measure acceptance before building anything around a drafter.
- **riceharvest's attention `.so` only has the MTP shapes.** A DFlash2 drafter (head 128, non-causal, sliding window)
  dies in warmup with "Chunk prefill kernel tuple not compiled for this configuration". Use the stock kernel, or
  build one that includes that shape.
- **Greedy output is not run-to-run deterministic once drafts change.** The target's verify numerics depend on how
  tokens group into steps (packed q_len paths, copy steps), so at an exact near-tie the greedy pick flips. We measured
  one: the top two tokens 0.016 nats apart, and the unchanged seat produced both transcripts (2 of 5 runs). Do not
  gate a drafter change on byte-identical greedy output. Gate on "no transcript the baseline never produces" over
  several runs, exact edits (`bench/editbench.py`) and the crossing test.
- **The drafter's warm-up feeds garbage hidden states.** vLLM warms the drafter with dummy inputs before graph
  capture. The shortlist's rescored logits came out NaN there, a tie-break that compared against the max was false
  everywhere, and its sentinel row indexed past the head: engine init died with an index-out-of-bounds. Anything on
  the draft path has to return valid ids for NaN, inf and zero inputs; the shortlist tests cover it.
- **Gate a drafter change on prose, not only on the traffic you fit it on.** The shortlist's offline replay (our own,
  code-heavy traffic) said -0.1% acceptance; a fixed-prompt prose A/B said -2 to -3% on greedy and thinking prose
  (none on sampled). Prose is where MTP is weakest on this model (about 2 tokens per step against 4.6 on code), so a
  drafter change shows there first. `bench/prose_ab.py` is the bench; keep it next to the code benches.
- **Know what your clients actually sample with.** With no `--generation-config`, vLLM fills missing sampling fields
  from the model's `generation_config.json` (709-L: temperature 1.0, top-k 20, top-p 0.95), and our router and agent
  harness send none. So every agent turn is a sampled request, probabilistic drafting applies to all of it, and a
  greedy-only bench would never have shown the +14-18% on prose. Check the engine's argv and a real request before
  judging a change that depends on sampling.
- **Every verify row costs 0.5 to 3 ms on this hybrid model, not microseconds.** The 48 GDN layers process rows
  serially (0.31 ms per row) and attention adds 0.5-1 ms per row at depth, so a wider or longer speculation pays that
  in every step whether or not it wins. Sibling-tree speculation (+2 rows) cost +11-20% step time for a +5% ceiling;
  MTP K 2-4 lose to K 5 at every depth. Profile the step before building anything that adds rows
  ([RESULTS](RESULTS.md#where-a-decode-steps-time-goes)).
- **vLLM's `enable_adaptive_verification` does not start on GDN models.** The attention base class gates it on
  `not is_ssm()`; the runner raises before any knob is read. A per-step draft count from the drafter's confidence
  also does not work as a scheduler policy here: under async scheduling the confidence the scheduler sees is 2-3
  draft rounds old and nearly flat against acceptance ([RESULTS](RESULTS.md#a-per-step-draft-count-from-the-drafters-confidence)).
- **An agent harness with cross-chat memory breaks A/B turns.** Ours surfaced the "before" chat's replies to the
  "after" chat (same example numbers, same story elements), which inflates the second arm's acceptance. Use fresh
  topics per arm or turn that memory off for A/Bs.

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
- **`python -I` ignores `PYTHONPATH`.** A kernel test launched that way skipped its side package, found no B70 kernel
  in the stock library and fell back to PyTorch reference attention at 45k tokens. On the xe driver, device
  allocations past the card's 32 GB spill into host RAM, so the 62 GB host went to a global OOM that killed the live
  server's process and two unrelated containers. Treat a "falling back to PyTorch reference attention" line at long
  context as an OOM in progress and kill it; run a 2k case first and check the printed library path.
- **The XPU torch profiler sees nothing inside a graph replay.** A graphed step shows 35 launches and 0.2 ms of
  device time. Profile a graphs-off load (`--compilation-config '{"cudagraph_mode":"NONE"}'`; the dotted
  `-O.cudagraph_mode` form is silently ignored on 0.32), read the kernel durations from there, and take the step
  time from the graphed load. Ops launched after the engine step's annotation closes (the target LM head, the MTP
  draft loop, rejection sampling) are easy to miss: count them separately.
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
- **Check which server a bench actually hit.** One of our bench scripts ignored its endpoint override and measured the
  live seat for a whole A/B window on the test seat; the numbers looked plausible. Read `model_name` in the
  `/metrics` lines the bench records, and keep the two servers' served names different (`bench/common.py` reads one
  set of settings for every script here).
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
