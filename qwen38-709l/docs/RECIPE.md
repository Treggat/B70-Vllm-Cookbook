# Recipe

## 0. Box

- One Intel Arc Pro B70 (BMG-G31, 32 GB). The card runs one model. Our machine has two B70s; we pinned this server to
  one of them with `ZE_AFFINITY_MASK` (see `serve/prod.env.example`).
- Linux with the `xe` driver. We ran vLLM inside an existing container that already had the Level Zero / compute
  runtime, and installed vLLM into a venv on a mounted volume instead of building a new image. That way you can keep
  several engines side by side and switch between them by changing one file.
- About 40 GB of host RAM is plenty: the embeddings in host RAM take about 4 GiB, and the venv's loads are the rest.
  (The 0.31 recipe also pinned 8 GiB for vLLM's CPU KV offload tier; see section 4 for why that is gone.)

## 1. Build vLLM 0.32.0

`scripts/build_venv.sh <dir>` builds an editable source install of v0.32.0 for XPU. You end up with torch 2.14.0+xpu,
triton 3.8.0+xpu and vllm-xpu-kernels 0.1.15.4, the same pins as 0.31 (so riceharvest's attention `.so` still fits).
The 0.31 version of the script is one commit back in this folder's history.

- `triton==3.8.0+xpu` only exists on **vLLM's own XPU index**, and since 0.32 that index is **versioned**:
  `https://wheels.vllm.ai/0.32.0/xpu/`. The old `/xpu/` path no longer tracks the release. With only PyTorch's index,
  the install fails.
- 0.32 dropped Python 3.10; 3.12 is what we run. `transformers>=5.16.1,<5.20`.
- The script ends by checking every flag on the serve line against `vllm serve --help=all`, because 0.31 renamed one
  without telling us (PITFALLS).
- Keep `src/` owned by the user who runs pip. A clone that was `chown`ed to someone else makes pip's git calls fail
  with "dubious ownership".

## 2. Apply the patch stack

`scripts/apply_stack.sh <dir> <kernel .so>` applies the stack in this order. It stops at the first patch whose anchor
is missing.

| order | patch | from | what it does |
|---|---|---|---|
| 1 | `patch_mtp_nightly` | SergiioB | MTP on Qwen3.8 hybrid (GDN) models |
| 2 | `patch_mtp_boundary_032` | **ours** (SergiioB's fix re-based) | a partial final speculative group at exactly max-model-len runs as a non-spec step; 0.32 rewrote the GDN metadata path his script anchored on. `bench/maxlen.py` tests it |
| 3 | `patch_draft_lmhead_int4` | SergiioB | the **drafter** gets an int4 copy of the 248k-vocab lm_head; target untouched |
| 4 | `patch_draft_mtp_int4` | SergiioB | int4 copies of the 5 MTP linears for the drafter |
| 5 | `patch_fix_backward_copy` | SergiioB | prefix caching + MTP state copy fix |
| 6 | `patch_meminfo_fallback` | **ours** | kernels' `getMemoryInfo` reads 0 GiB free under some L0 drivers → fall back to torch's query |
| 7 | `patch_mtp_fc_bf16` | **ours** | `B70_MTP_BF16_DRAFT` didn't cover `fc`, so a checkpoint with BF16 `mtp.*` failed ("no parameter fc.weight") |
| 8 | `patch_image_url_video` | **ours** | clients that send video as an `image_url` with a `data:video/...` URL (the llama.cpp convention) got a 400 |
| 9 | riceharvest `b70/*` | riceharvest | embed UVA, draft vocab (JSON), fp8 target lm_head, greedy warmup, eager token pad, packed MTP verify |
| 10 | `libattn_kernels_xe_2.so` | riceharvest | v2 attention kernel (fast fp8-KV prefill, zero-sum guard for packed verify); the stock one is kept as `.stock` |

Not applied: riceharvest's `patch_asym_gptq` (GPTQ checkpoints only), `patch_moe_gemv` and `patch_linear_fp8` (MoE and
bf16-linear checkpoints). **Dropped when we moved to 0.32:** his three GDN sync patches (`gdn_spec_sync`, `gdn_zero_draft`,
`gdn_mixed_no_sync`: 0.32 rewrote that path and now slices the padded rows upstream, so the scripts refuse) and
`xpu_graph_kv_splits` (anchor changed, and it measured no gain on 0.31). Their `B70_GRAPH_KV_SPLITS` / `B70_GDN_*`
switches are gone from `prod.env.example`.

### Vcvzgbzz's 2026-10-19 update (dd083d5)

The day we moved to 0.32, [Vcvzgbzz/b70-cookbook][vcv] published its kernel patches and DFlash2 work. We diffed it against
this stack: its MTP-side pieces are the mechanisms we already run from riceharvest's layer, under the same `B70_*` names:
the 3-instruction fp8 to fp16 convert with the x256 folded into the softmax and V scales (`B70_E4M3_FAST=3`), the per-row
causal mask for packed MTP verify inside the decode kernel (`B70_SPEC_ROW_MASK`) with the 8-row tile for head 256 and
the zero-sum guard, `packed_verify.py` (= riceharvest's `spec_packed.py`), the fp8 target head that releases its fp16 copy
once the int4 draft head is built, embed UVA, and the draft-vocab trim. His `gdn-spec-no-sync.patch` is what 0.32 ships
upstream. **Nothing to add for MTP.** What is new is the **DFlash2 drafter**
([incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), 7 drafts per step; +29% short code,
+37% edits on base Qwen3.8 in his numbers). We re-based his patches onto 0.32 (`patches/dflash2/`) and measured it on
the 709-L: **2-5% acceptance per draft token, about 3× slower than MTP-5** (RESULTS.md). The drafter reads five inner
layers of the target (5, 19, 33, 47, 61), and the merge changed every layer's weights by 19-31% against base
Qwen3.8. The MTP head survives because it reads only the final hidden state, which the merge kept consistent with
the LM head. A published drafter only helps the model it was trained on; for a merge, that means training one on
the merge's own hidden states. `apply_stack.sh` does not apply the DFlash2 patches.

[vcv]: https://github.com/Vcvzgbzz/b70-cookbook

### The one clash: `B70_DRAFT_VOCAB`

Vcvzgbzz's trim and riceharvest's `patch_draft_vocab` both edit `b70_draft_lmhead_int4.py`, and both read the env var
`B70_DRAFT_VOCAB`, but with different meanings:

- Vcvzgbzz expects **an integer** (`98304`, plus `B70_DRAFT_SPECIALS_START`).
- riceharvest expects **a path to a JSON list of token ids**. Its `patch_target_lmhead_fp8` reads it the same way.

We use riceharvest's patch with a JSON file that keeps exactly the rows Vcvzgbzz's trim keeps:
`scripts/make_draft_vocab.py` → ids 0..98,539 + 248,044..248,319 = 98,816 rows. The draft is unchanged and the fp8
head works.

### Patch scripts that patch everything they can find

Several upstream patch scripts look for vLLM in a list of well-known paths (`/opt/venv/...`, `/workspace/vllm`) as
well as the one being imported, and patch **all** of them. Building a venv inside a container that ships its own vLLM
image therefore also edits the image's copy. That happened to us. `apply_stack.sh` copies every upstream script and
rewrites those paths to the tree being built. It also sets `B70_VLLM_ROOTS`, which SergiioB's backward-copy fix
honors. Our own patches refuse to run without an explicit `VLLM_ROOT`.

## 3. Model

See [AWQ.md](AWQ.md). Short version: TheUnderscore's W4A16 quant of the 709-L comes in compressed-tensors
"pack-quantized" layout, which vLLM-XPU can't load. `scripts/ct2awq.py` re-lays the nibbles into AutoAWQ gemm.

## 4. Serve

`serve/serve-709l.sh` plus `prod.env` and `prod.extra`. The flags that matter:

- `--kv-cache-dtype fp8` and `--kv-cache-memory-bytes 15569256448` (14.5 GiB): 389,315 tokens, 4 streams × ~97k.
  Card budget: 31.5 GiB free at start − 14.47 GiB model − 0.9 GiB graphs − 14.5 GiB KV ≈ 1.6 GiB headroom for
  activations and the vision encoder. We stress-tested it with 4 × 95k chats at once.
- `--cpu-offload-gb 4 --cpu-offload-params embed_tokens` + `B70_EMBED_UVA=1`: keeps the embedding table in pinned host
  RAM. That alone didn't speed anything up, but the VRAM it frees is part of what made the 14.5 GiB pool possible.
  Offload only `embed_tokens`. Adding `visual` (the vision tower) to `--cpu-offload-params` made an image request take
  20.3 s instead of about 3 s. `apply_stack.sh` also keeps the MTP drafter's own embedding table off the offloader
  (see PITFALLS): the target's offloaded table is the one both end up using.
- `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`.
- **No `--kv-offloading-size`.** The 0.31 line had `--kv-offloading-size 8`. vLLM's CPU KV tier stores blocks for
  this hybrid model but never finds them again: `bench/kvoff_probe.py` evicts a 55k-token prompt with 7 x 58k others
  and replays it, and the replay reports 0 cached tokens and a full re-prefill (36 s), on 0.32 exactly as Vcvzgbzz
  found on 0.31. The flag only pinned an 8.58 GB `/dev/shm` mmap. Prefix caching on the GPU is unaffected (a 20k
  replay shows 19,776 cached tokens, 0.2 s).
- `--enable-prefix-caching --prefix-match-unit 64 --enable-mamba-shared-prefix-checkpoint --mamba-ssm-cache-dtype float16`.
  0.31 **renamed** `--enable-mamba-fine-grained-prefix-cache` to `--enable-mamba-shared-prefix-checkpoint`. Without
  it, replies crossing a state page get corrupted (see PITFALLS).
- `--max-model-len 262144` (in `prod.extra`, over the launcher's 131072). The 709-L is native 262k (plain rope). With the
  14.5 GiB pool the KV cache reads 416,591 tokens (1.59 × 262k), so one 262k request or 4 × ~100k fit. Speed is
  unchanged against the 131k window (RESULTS.md), including under XPU graphs, where `max_seqlen_k` is the window at
  capture time. Whatever sits in front of the server (a router's admission check, a harness's compaction threshold)
  has to learn the new number too: we raise the router's declared window and the harness's `contextWindow` to
  262144 − 8192 and restart both, since each reads it once at start.
- `--max-num-batched-tokens 3584`. Don't raise it on this build: 16384 made cold prefill 2–3× slower for us.
  Don't lower it for single-stream use either. If the seat serves several chats and one long cold prompt must not stall
  the others, add `--long-prefill-token-threshold 512` (decode during another chat's prefill 2.2 → 10 t/s, that
  prefill 22% slower; the sweep is in RESULTS.md).
- `--max-num-seqs 4`.

**Prod vs test config:** the launcher reads `prod.env`/`prod.extra`, unless a file named `use-test` exists, in which
case it reads `test.env`/`test.extra`. Experiments never touch the live config, and `rm use-test` restores it. Give each
switch combination its own `VLLM_CACHE_ROOT`: the compile cache is keyed on flags, not env, so a stale graph can
otherwise be reused.

First load compiles graphs (about 3.5–7 min). Later loads reuse the cache.

**0.32 behaviour change to know about:** `max_tokens` is now an upper bound, not reserved from the context window
(vLLM #57035). A prompt of 128,977 tokens with `max_tokens: 8000` on a 131,072 window runs to exactly 131,072 and
stops with `finish_reason: length` (`bench/maxlen.py`); on 0.31 the same request was refused. A router that pre-checks
`prompt + max_tokens` against the window will over-count.

**Chat template:** `serve-709l.sh` uses the template bundled with the model. Some agent harnesses send system messages
after user turns, which Qwen templates reject or mishandle. Our own deployment adds a small template change that moves
all system messages to the front and merges them. If your client does this, pass `--chat-template` with a template
that does the same.

## 5. Verify

```sh
cd bench
python crossing.py          # page-boundary corruption test: must print PASS
python bench.py mine        # speed + canary + greedy hashes
python stress4.py mine      # 4 x 95k chats at once: 0 preemptions, no OOM
python maxlen.py            # after an engine change: a 129k prompt generates into the 131k edge and must end cleanly (direct to vLLM)
```
