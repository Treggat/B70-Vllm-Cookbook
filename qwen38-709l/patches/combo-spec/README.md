# Combined copy + MTP speculator on vLLM 0.32 (opt-in, live on our seat since 2026-10-21)

vLLM 0.32 runs one speculative method per engine. On the 709-L, MTP-5 is the best general drafter (131 t/s short
code). But when the output copies its own context (an edit that returns the whole file, a rename, a refactor), MTP is
capped at 6 tokens per step even at 94% acceptance. Prompt-copy n-gram drafting with 15 drafts reaches 267 t/s on
those edits, and is 3-4× slower than MTP on everything else. These patches put both in one engine and choose per
step:

- **Scheduler** (`combo/b70_combo_policy.py`, installed as `vllm/v1/core/sched/b70_combo_policy.py`): each step it
  picks the draft count for the whole batch, the way Dynamic SD does.
  - K = 15 only when every scheduled request is a decode that is *copying* and under `B70_COMBO_MAX_CTX` tokens
    (the live seat sets it to the whole 262k window: an agent's system prompt alone can be 40k).
  - A request counts as copying when its last 4 tokens match earlier in the request, at least 10 tokens remain after
    the match, and the tokens generated since the last check equal the source tokens that follow the match. That
    last test is an exact compare, which makes repeated 4-grams in code harmless.
  - After two weak copy steps (< 6 tokens), the request cools down for 16 steps.
  - Cost: about 0.2 ms of CPU per request per step at 32k tokens, hidden behind the GPU step.
- **Worker** (`combo/b70_combo_speculator.py`, a `MTPSpeculator` subclass in `spec_decode/b70_combo/`):
  - On K = 5 steps it is plain MTP-5.
  - On K = 15 steps MTP runs only its first pass. That pass covers every token the target just verified, so the MTP
    layer's KV stays complete after a 16-token step.
  - Then it runs the n-gram Triton scan from vLLM's `NgramGPUSpeculator`. A request uses the 15 copied drafts when the
    match is at least 3 tokens long and the first copied token equals MTP's first draft. Otherwise it gets MTP's
    draft padded to 15.
  - Drafts are greedy (one-hot) on both paths, so mixing them in one verify is lossless. This relies on vLLM's
    default `draft_sample_method: "greedy"`. With `"probabilistic"` the rejection sampler would check the copied tokens
    against MTP's cached draft logits, so do not combine the two.
- **GDN** (`patch_gdn_varlen_spec_032.py`, `B70_GDN_VARLEN_SPEC=6,16`): the metadata builder assumes every spec decode
  has `num_speculative_tokens + 1` tokens.
  - With K max 15, a 6-token MTP step looked like a truncated final group, so the boundary patch sent it to the stateful
    prefill path, which cannot roll back rejected drafts.
  - The XPU GDN op in vllm-xpu-kernels 0.1.15.4 already takes ragged spec steps (initial state from column
    `num_accepted - 1`; conv1d shifts by the actual count), so only the expected size changes.
  - Needs `patch_mtp_boundary_032.py` first.
- **Packed verify guard** (`patch_spec_packed_guard_032.py`, `B70_SPEC_PACKED_QLENS`, default `2,3,6`): riceharvest's
  `spec_packed.py` packs any uniform q_len, but only 2, 3 and 6 have been checked on the b70 `.so`. Other q_len values go
  to the stock path, so copy steps (q_len 16) verify unpacked and MTP steps (q_len 6) stay packed.
  - Worth applying even without the combined speculator.
  - The kernel's row mask is right for any q_len (it derives the group from the packed tensor). `tests/pk16.py` checks
    q_len 11 and 16 against the stock path at 2k-115k context: identical output, 1.2-1.3x faster at depth. The live
    seat packs 2,3,6,16.
- `patch_combo_capture_032.py` (`B70_COMBO_CAPTURE_KS`, optional): Dynamic SD captures graphs for at most one K per
  batch size (4 with `max_num_seqs` 4). This adds more K values to the capture set, for the depth-tier experiments.
- `patch_combo_spec_032.py` installs both modules, hooks `scheduler.py` (`choose_k` after Dynamic SD, `observe` after
  each verify) and `spec_decode/__init__.py`, and keeps `all_token_ids` dense on the GPU in `model_runner.py`.
  Otherwise the scan would read UVA host memory over PCIe every copy step.

The patch scripts take `VLLM_ROOT` (the `vllm/` package dir, `<venv dir>/src/vllm`) or a path argument; the guard
takes `VLLM032_ROOT` (the venv dir, which holds `rh-runtime/`). They are idempotent and fail-closed, leave a `.bak`,
and `py_compile` what they touch. Every hook is a no-op with its switch unset. Apply them after
`scripts/apply_stack.sh`, which does not apply them; the commands are in the [recipe's quick start](../../README.md#quick-start)
(`patch_combo_capture_032.py` is only for the depth-tier experiments).

## Serving

```sh
export B70_SPEC_PACKED=1 B70_SPEC_PACKED_QLENS=2,3,6,16 B70_GDN_VARLEN_SPEC=6,16 B70_COMBO_SPEC=1
export B70_COMBO_MAX_CTX=262144 B70_COMBO_SCAN_CTX=262144
SPEC='{"method":"mtp","num_speculative_tokens":15,"num_speculative_tokens_per_batch_size":[[1,1,15],[2,4,5]],"rejection_sample_method":"block"}'
```

The Dynamic SD schedule is there only so that vLLM captures graphs for q_len 6 and 16 (target, MTP prefill and MTP
decode). The policy overrides its choice. KV drops about 10% (416,591 → 375,067 tokens in our 14.5 GiB pool): the GDN
layers keep a state slot per draft position. `B70_COMBO_SCAN_CTX` must cover every context you copy from: the worker's
n-gram scan stops there (default 65536), so at 94k with the default it found no source and paid 16-token verifies for
MTP-sized gains. `rejection_sample_method: "block"` (vLLM's block verification) keeps greedy output exact; on sampled
prose it was neutral to slightly positive. Other knobs (env, defaults): `B70_COMBO_MAX_CTX` 32768,
`B70_COMBO_SCHED_N` 4, `B70_COMBO_MIN_AVAIL` 10, `B70_COMBO_MIN_YIELD` 6, `B70_COMBO_COOLDOWN` 16, `B70_COMBO_MAX_BS` 4,
`B70_COMBO_MTP_IN_COPY` 0 (1 = full MTP-5 on copy steps, so a request that falls back gets 5 real MTP drafts instead
of 1; no measurable difference on edits under the offset policy, 207 vs 205),
`B70_COMBO_AGREE` 1, `B70_COMBO_USE_MIN_N` 3. Depth-tier knobs (inert unless set): `B70_COMBO_MTP_TIERS` and
`B70_COMBO_COPY_TIERS` (`"ctx:K,ctx:K"`, the K for the longest scheduled request), `B70_COMBO_COPY_MIN_K` (a step with
K at or above this is a copy step; default the configured K), `B70_COMBO_CTL` (tests only: a key=value file re-read every
8 steps, so one load can A/B `max_ctx`, `off`, `mtp_tiers`, `copy_tiers`). Measured: neither MTP nor copy tiers pay off
on this seat ([docs/RESULTS.md](../../docs/RESULTS.md)), so the live seat sets none of them.

## Tests

- `tests/test_policy.py`: an offline simulation of the scheduler policy. On a copy with an edit every 400 tokens, 95%
  of steps are copy steps; on a code-like repetitive source, 93%. Fresh tokens and boilerplate give none, and so does
  anything above the context cap.
- `tests/test_kernels.py`: the scan + finalize kernels against a Python reference. It runs under `TRITON_INTERPRET=1`
  on CPU tensors, no card needed: 145 rows, 41 through the copy path, 0 mismatches.
- `tests/pk16.py`: packed vs stock verify at q_len 6/11/16 on the card (no server).
- On the card: `bench/editbench.py` (6/6 exact), `bench/tierbench.py` (decode and copy-edit at depth with per-run spec
  counters), `bench/snap.py` (spec counters around agent turns), `bench/crossing.py`, `bench/maxlen.py` at 262k, short code, depth and
  4 writers. The numbers are in [docs/RESULTS.md](../../docs/RESULTS.md).
