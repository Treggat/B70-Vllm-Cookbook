# V-split verify kernel for packed MTP verify at depth (opt-in, live on our seat since 2026-10-25)

At long context, a decode step on the 709-L is mostly the target's verify pass over the KV cache: 16 full-attention
layers, each reading the whole context once per step. riceharvest's packed verify already folds the q_len x GQA rows of
one step (6 x 6 = 36 rows at MTP-5) into one decode query per KV head. But it runs them on the 8-row tile, five
work-groups per KV head, and each of them streams the same KV. This kernel runs all 36 (up to 48) rows in **one
work-group per KV head and split**, so each KV tile is read once per step instead of five times.

One attention layer, q_len 6, fp8 KV, one request, the split counts the seat's captured graphs use
(`results/vsplit/kernel-exact-and-timing.out`, 2026-10-25, microseconds):

| context | 8-row tile, 12 splits | V-split, 32 splits (two passes) | change |
|---|---|---|---|
| 23k | 370.6 | 245.0 / 224.7 | -34% / -39% |
| 61k | 619.2 | 380.6 / 378.5 | -39% |
| 100k | 975.1 | 565.7 / 566.0 | -42% |
| 150k | 1,429.6 | 818.1 / 820.4 | -43% |
| 200k | 1,898.8 | 1,066.6 / 1,066.6 | -44% |
| 250k | 2,352.5 | 1,285.5 / 1,297.5 | -45% |

From 61k on, the KV read rate goes from about 210 GB/s to 330-400 GB/s. Over 16 layers that predicts 13 ms less per step at 200k.
The seat measured 11.8-12.1 ms less: decode behind 210k of context goes from 51-53 to 64-65 t/s (+23%), behind 105k
from 84.7 to 92.1 t/s, with short code, edits and 4 writers at parity and greedy output byte-identical. Numbers and
raw files: [docs/RESULTS.md](../../docs/RESULTS.md#v-split-verify-kernel).

## How it works

The stock decode mainloop (vllm-xpu-kernels' `DecodeFwdMainloop`) gives each subgroup its own query rows and all 256
columns of V. riceharvest's wide 32- and 64-row tiles stack more subgroups along the rows that way, and were at parity
with the 8-row tile in serving ([A 64-row verify tile](../../docs/RESULTS.md#a-64-row-verify-tile)). This mainloop
splits the work two ways inside one work-group of 8 subgroups instead:

- **Q·K** on a 2 x 4 grid (row groups x 16-key column groups) over a 48 x 64 tile (48 rows, 64 keys), in 16-wide
  chunks of the head dimension.
- **S and the online softmax** go through shared local memory (SLM): each subgroup stores its S block, a barrier, then
  one pass per row updates the running max and sum and writes P (fp16) to SLM, **key-major** (`pT[key * 48 + row]`).
- **P·V** on a 2 x 4 grid (row groups x 64-column slices of V), in 16-key chunks. Each subgroup fills its P fragment
  from SLM with vector loads: in the DPAS A fragment, lane = key and value i = row, so a lane's 24 rows are contiguous
  in the key-major layout. Each subgroup keeps 24 rows x 64 columns of O (96 registers).
- **O rescale** is skipped for a row group unless one of its row maxima moved in this KV tile (an SLM flag per row
  group). At depth most tiles move no maximum.
- The split-K epilogue and `ReduceSplitK` are the stock ones. The epilogue writes the log-sum-exp for every row a
  subgroup holds (the stock code assumed at most 16).

Two rules matter for speed:

- **Exactly 4 work-groups per Xe core.** With 8 subgroups at 256 registers each, the card holds 4 of these work-groups
  per Xe core, 128 on the B70. 32 splits for one request (4 KV heads x 32 = 128) is a sharp optimum at every depth
  (`results/vsplit/kernel-split-sweep.out`): 28 splits cost 8-13%, 36 splits (144 work-groups, a second partial wave)
  52-104%, 48 splits 30-78%. `patch_spec_packed_vsplit_032.py` sets `splits = 4 x Xe cores / (requests x KV heads)`
  (32 at one request, 8 at four) for this tile. riceharvest's rule sizes the split count for the 8-row tile's 5
  work-groups per KV head (12 splits at one request), which takes this kernel 1.8-2.6x as long.
- **A lower single-split threshold.** The stock kernel only splits a sequence once it has 32 KV tiles (2k tokens);
  below that, this tile would run one work-group per KV head, 4 on the whole card. The mainloop sets
  `kMinBlocksForSplit = 2` through a trait the kernel and `ReduceSplitK` both read (the default is unchanged for every
  other mainloop). Without it short code was 1.5% slower; with it, at parity.

It is used for a packed verify call when all of these hold (`paged_decode_xe2.cpp`, `b70_vs_ok`): `B70_SPEC_VSPLIT=1`,
q_len > 1, head size 256, page size a multiple of 64, fp16 Q, fp8 e4m3 KV, no sliding window, no sinks, and at most 48
packed rows. Anything else takes the stock path. In our config, MTP-5 steps (36 rows) use it, and the combined
speculator's 16-row copy steps (96 rows) keep the 8-row tile.

## Files

| file | what |
|---|---|
| `b70-verify-kernels.patch` | vllm-xpu-kernels patch, on top of riceharvest's `xk-fold-b70` and `zero-sum-guard`: the V-split mainloop (`collective/decode_vsplit_mainloop.hpp`), its dispatch and policy, the split threshold trait, the epilogue fix, plus two opt-in experiments that stay off (below) |
| `patch_spec_packed_vsplit_032.py` | installer for riceharvest's runtime (`rh-runtime/spec_packed.py`): the split rule for this tile behind `B70_SPEC_VSPLIT=1`; fail-closed, idempotent, keeps a `.bak` |
| `tests/vsbound.py`, `tests/vsmix.py` | exactness against the stock per-position path with the verify rows straddling a 960-token page boundary: one request and four, then batches of different lengths in every order |
| `tests/vsrepeat.py` | determinism: the same call 500 times at 200 to 100k tokens, every output bit-identical to the first |
| `tests/kt_depth.py` | one attention layer at 23k-250k, microseconds and KV GB/s (the table above) |
| `tests/crossing_repeat.py`, `tests/crossing_func185_*.py` | the page-boundary bug hunt in [PITFALLS.md](../../docs/PITFALLS.md) (not caused by this kernel) |
| `LICENSE-vllm-xpu-kernels`, `LICENSE-riceharvest` | licenses of the code the patch modifies and sits on |

The patch carries two more opt-in switches, both off unless set. They are settled experiments, kept because the V-split
work grew out of them:

- `B70_SPEC_TREE=<hex masks>`: per-row ancestor masks for tree verify
  ([sibling-tree speculation](../../docs/RESULTS.md#sibling-tree-speculation), dropped).
- `B70_SPEC_QTILE=32|64|128`: riceharvest's wide verify tiles, plus a 128-row variant
  ([a 64-row verify tile](../../docs/RESULTS.md#a-64-row-verify-tile), parity).

`B70_SPEC_VSPLIT=2..5` select timing variants that give **wrong results**: 2 skips the P fill, 3 the O rescale, 4 the
S store and softmax, 5 prints the fragment layouts. They are how we found where the time went. Never serve with them.

## Build

The library is built like riceharvest's (vllm-xpu-kernels `release/0.1.15.4` = `ddf336d`, sycl-tla `87f68506` with
`tla-fast3`, oneAPI DPC++ 2026.0, AOT for BMG-G31), with our patch on top and the
[kernels-union](../kernels-union/README.md) lists. [`gpt-oss-20b/kernels/build.sh`](../../../gpt-oss-20b/kernels/build.sh)
does every step for gpt-oss; for the 709-L change three things in its step 2:

1. skip `bf16fast5.py` (gpt-oss needs it, the 709-L doesn't);
2. after `zero-sum-guard.patch`, apply ours: `git -C "$W/xk" apply b70-verify-kernels.patch`;
3. copy `../kernels-union/b70_709l_prefill.conf` and `b70_709l_decode.conf` instead of the gptoss lists, and pass them
   as `VLLM_CHUNK_PREFILL_CONFIG` / `VLLM_PAGED_DECODE_CONFIG`.

We checked that `ddf336d` + `xk-fold-b70` + `zero-sum-guard` + this patch reproduces our source tree file for file.
Our build (3 compile jobs in Docker on a PC) gave a 45,433,280-byte library, sha256
`2420cf7921a54c63abd6e968f7780fdb01aa97e32ea9135a3f3997412d31c617`. The compiler reports no register spill for the
V-split kernel; of its timing variants, only variant 4 spills.

## Install

On the serving box, from the recipe folder (`qwen38-709l/`) with the built library copied there, and with the server
stopped or unloaded (the library is memory-mapped while it runs). `VENV` is the directory given to
`scripts/apply_stack.sh`; apply the combo-spec guard patch first ([RECIPE.md](../../docs/RECIPE.md), section 2).

```sh
K=$VENV/venv/lib/python3.12/site-packages/vllm_xpu_kernels
cp -p $K/libattn_kernels_xe_2.so $K/libattn_kernels_xe_2.so.pre-vsplit             # once
cp -p libattn_kernels_xe_2.so $K/.libattn_kernels_xe_2.so.new && mv -f $K/.libattn_kernels_xe_2.so.new $K/libattn_kernels_xe_2.so
VLLM032_ROOT=$VENV $VENV/venv/bin/python patches/vsplit-verify/patch_spec_packed_vsplit_032.py
```

Then add `export B70_SPEC_VSPLIT=1` to `serve/prod.env` (the commented block in `prod.env.example`) and start the server.
The kernel and the runtime both read that one variable; unset, the new library behaves like riceharvest's. Rollback:
remove the line, or restore the `.pre-vsplit` library and the runtime's `.bak-2026-10-25-pre-vsplit`.

`apply_stack.sh` reinstalls riceharvest's library and runtime. After running it, install both again.

## Exactness

- Against the stock per-position path (`results/vsplit/kernel-exact-and-timing.out`): max difference 2e-5 in all 416
  page-boundary cases (one request and four; 6 and 8 rows; both `max_seqlen_k` modes) and all 135 mixed-batch
  cases.
- Determinism (`kernel-determinism.out`): 0 of 500 repeated calls differ, at 200 / 1,500 / 4,000 / 23k / 100k tokens.
- On the seat (`exact-probes.txt`): the greedy code probe is byte-identical before, in the test load and after
  promotion, two runs each.

## Credits and license

Built on [riceharvest's](https://github.com/riceharvest/b70-qwen38-cookbook) packed verify, b70 kernel and runtime
(MIT, `LICENSE-riceharvest`), and on [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) and
[sycl-tla](https://github.com/intel/sycl-tla). The patch modifies vllm-xpu-kernels sources, which are Apache-2.0
(`LICENSE-vllm-xpu-kernels`), and is offered under the same license so the patched files keep one. The installer and
tests are MIT like the rest of this repo.
