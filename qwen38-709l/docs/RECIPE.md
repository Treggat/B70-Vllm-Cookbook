# Recipe

## 0. Box

- One Intel Arc Pro B70 (BMG-G31, 32 GB). The card runs one model. Our machine has two B70s; we pinned this server to
  one of them with `ZE_AFFINITY_MASK` (see `serve/prod.env.example`).
- Linux with the `xe` driver. We ran vLLM inside an existing container that already had the Level Zero / compute
  runtime, and installed vLLM into a venv on a mounted volume instead of building a new image. That way you can keep
  several engines side by side and switch between them by changing one file.
- About 45 GB of host RAM is plenty: KV offload uses 8 GiB, and the embeddings in host RAM take about 4 GiB.

## 1. Build vLLM 0.31.0

`scripts/build_venv.sh <dir>` builds an editable source install of v0.31.0 for XPU. You end up with torch 2.14.0+xpu,
triton 3.8.0+xpu and vllm-xpu-kernels 0.1.15.4.

- `triton==3.8.0+xpu` only exists on **vLLM's own XPU index** (`https://wheels.vllm.ai/xpu/`). With only PyTorch's
  index, the install fails.
- Keep `src/` owned by the user who runs pip. A clone that was `chown`ed to someone else makes pip's git calls fail
  with "dubious ownership".

## 2. Apply the patch stack

`scripts/apply_stack.sh <dir> <kernel .so>` applies the stack in this order. It stops at the first patch whose anchor
is missing.

| order | patch | from | what it does |
|---|---|---|---|
| 1 | `patch_mtp_nightly` | SergiioB | MTP on Qwen3.8 hybrid (GDN) models |
| 2 | `patch_mtp_boundary` | SergiioB | MTP state at page boundaries |
| 3 | `patch_draft_lmhead_int4` | SergiioB | the **drafter** gets an int4 copy of the 248k-vocab lm_head; target untouched |
| 4 | `patch_draft_mtp_int4` | SergiioB | int4 copies of the 5 MTP linears for the drafter |
| 5 | `patch_fix_backward_copy` | SergiioB | prefix caching + MTP state copy fix |
| 6 | `patch_meminfo_fallback` | **ours** | kernels' `getMemoryInfo` reads 0 GiB free under some L0 drivers → fall back to torch's query |
| 7 | `patch_mtp_fc_bf16` | **ours** | `B70_MTP_BF16_DRAFT` didn't cover `fc`, so a checkpoint with BF16 `mtp.*` failed ("no parameter fc.weight") |
| 8 | `patch_image_url_video` | **ours** | clients that send video as an `image_url` with a `data:video/...` URL (the llama.cpp convention) got a 400 |
| 9 | riceharvest `b70/*` | riceharvest | embed UVA, draft vocab (JSON), fp8 target lm_head, GDN sync (patched in, switched **off**), kv-splits, greedy warmup, eager token pad, packed MTP verify |
| 10 | `libattn_kernels_xe_2.so` | riceharvest | v2 attention kernel (fast fp8-KV prefill, zero-sum guard for packed verify); the stock one is kept as `.stock` |

Not applied: riceharvest's `patch_asym_gptq` (GPTQ checkpoints only), `patch_moe_gemv` and `patch_linear_fp8` (MoE and
bf16-linear checkpoints).

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
- `--enable-prefix-caching --prefix-match-unit 64 --enable-mamba-shared-prefix-checkpoint --mamba-ssm-cache-dtype float16`.
  0.31 **renamed** `--enable-mamba-fine-grained-prefix-cache` to `--enable-mamba-shared-prefix-checkpoint`. Without
  it, replies crossing a state page get corrupted (see PITFALLS).
- `--max-num-batched-tokens 3584`. Don't raise it on this build: 16384 made cold prefill 2–3× slower for us.
- `--max-num-seqs 4`.

**Prod vs test config:** the launcher reads `prod.env`/`prod.extra`, unless a file named `use-test` exists, in which
case it reads `test.env`/`test.extra`. Experiments never touch the live config, and `rm use-test` restores it. Give each
switch combination its own `VLLM_CACHE_ROOT`: the compile cache is keyed on flags, not env, so a stale graph can
otherwise be reused.

First load compiles graphs (about 3.5–7 min). Later loads reuse the cache.

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
```
