# Shortlist draft head for the 709-L MTP drafter (opt-in, live on our seat since 2026-10-22)

On the 709-L, each MTP draft pass reads the drafter's whole int4 draft head: 98,816 rows x 5,120, 261 MB, 440 of the
~1,090 us a pass costs at 23k context, and the head is 54% of the bytes the pass moves. The drafter only needs the
**argmax** of that head (greedy drafts), or a distribution over a few likely rows (probabilistic drafts). This patch
scores every row with a small covariance-weighted low-rank factorization, keeps the top N rows, and rescores only
those with the real head:

```
z     = h @ Q                               Q  [5120, 512]  fp16, 5 MB
s     = int4_gemm_w4a16(z, P)               P  [512, 98816] int4 g128, 26 MB (the seat's own layout and GEMM)
cand  = topk(s, N)                          N = 128
draft = ids[cand[argmax(int4_gemm_w4a16(h, head[cand]))]]     the candidates rescored by the head's own GEMM
```

The target verifies every draft, so a shortlist miss costs acceptance, never output. Measured on our seat: **0.33 ms
less per draft pass at 1, 2 and 4 streams, -3.4% step time at every depth from 23k to 120k, short code +4%, edits
+2%**, no measurable acceptance cost on code and sampled prose, about 2-3% acceptance cost on greedy and thinking
short-context prose (prose still decodes faster). Numbers and raw files: [docs/RESULTS.md](../../docs/RESULTS.md#shortlist-draft-head).

## Why weighted, and why not a trimmed head

Plain SVD of the head fails: rank 2,560 with 1,024 candidates still loses 8% of chain tokens per step, rank 512 with
64 candidates loses 40%. The head's logits on real hidden states live in a small subspace that plain SVD, which treats
every input direction alike, does not find. Weighting the factorization by the covariance C = E[h h^T] of real head
inputs (M = C^1/2, SVD of W M) is near-exact from rank 512 on:

| rank r | N = 32 | N = 64 | N = 128 | N = 256 |
|---|---|---|---|---|
| 256 | -4.3% | -2.3% | -1.1% | -0.49% |
| 384 | -1.6% | -0.78% | -0.31% | -0.09% |
| **512** | -0.60% | -0.22% | **-0.05%** | 0.00% |
| 768 | -0.12% | -0.03% | 0.00% | +0.02% |

(chain tokens per step vs the full int4 head, teacher-forced K = 5 replay of 86,010 chain starts from 8 held-out
sessions of our own agent traffic; covariance from 6 other sessions; `results/shortlist/offline/sweep2_int4_trainC.log`,
plain SVD in `sweep1_int4.log`.) A static trimmed head (the top 12k to 49k ids by how often the model generated them)
loses 1.2 to 5.6%: only 22,706 distinct ids appeared in 1.42M generated tokens, but the rare ones matter to the chain.
Grouping rows into blocks and shortlisting block maxima (`B70_DRAFT_SHORTLIST_BLOCK`) was slower on the card than a
plain top-k and lost more acceptance; it stays in the code, off.

With the artifact built from the seat's exact head (the 98,816 rows including the 276 special ids, int4-quantized with
the seat's own function): N = 128 costs -0.11% (code -0.10%, roleplay -0.17%, worst session -1.15%), N = 256 -0.05%
(`results/shortlist/offline/artifact_r512_eval.log`, `artifact_r512_perfile.log`).

## Files

| file | what |
|---|---|
| `b70_shortlist_head.py` | the module vLLM imports: lazy, fail-closed init; `top_tokens()` (greedy draft ids); `logits_row()` (the N rescored logits scattered into a -inf row of the full vocab width, for probabilistic drafts) |
| `patch_shortlist_032.py` | installer for a vLLM 0.32 tree: copies the module to `model_executor/models/`, hooks `DraftModelSpeculator.sample_draft` and `load_model` in `v1/worker/gpu/spec_decode/speculator.py`; `--restore` |
| `build_factors.py` | builds the artifact from the model's `lm_head.weight`, the draft-vocab JSON and a covariance (a saved one, or a tensor of drafter head inputs) |
| `tests/test_shortlist_cpu.py` | CPU test: the fail-closed gates, module == fp32 reference of the same algorithm, agreement with the full head on real head inputs, tie-break, NaN/inf inputs, shapes |
| `tests/test_prob_cpu.py` | CPU test of the probabilistic path: `logits_row` shape, exactly N finite entries at draft-vocab ids, values == fp32 rescoring, argmax == `top_tokens`, q fidelity vs the full head, garbage inputs, the `B70_DRAFT_SHORTLIST_PROB` gate |

## The artifact

`shortlist_r512.pt` (32.4 MB): P as int4 qweight/scales/qzeros in the seat's layout, Q fp16, the 98,816 ids, a
fingerprint of the head it was built for (the int4 scales of every 997th row plus their nibbles), the covariance's
row count, the weighted energy kept (0.919) and a self-check. It is a derivative of the model's `lm_head` and carries
the model's license. It is not in this repo (no binaries). The one we serve is published at
[Taggert2222/qwen38-709l-shortlist-head](https://huggingface.co/Taggert2222/qwen38-709l-shortlist-head) on Hugging Face,
sha256 `3a54099c7fdc3dc80b548c39d7ceb444aa84b696987e52e48d498c009fdaef43`; its metadata is
`results/shortlist/offline/shortlist_r512.json`.

To build your own, from this folder with any torch (CUDA or CPU; ours took a few minutes on a desktop GPU):

```sh
python build_factors.py --model-dir <AutoAWQ model dir from scripts/ct2awq.py> --vocab ../../serve/draft_vocab_98304.json \
    --inputs head_inputs.pt --rank 512 --out shortlist_r512
```

`head_inputs.pt` is a tensor `[n, 5120]` of hidden states as they reach the drafter's `sample_draft` (the MTP block's
output before the head), from traffic like the traffic you serve. We used 287,000 rows from 6 sessions of our own
agent traffic, replayed through an offline copy of the MTP drafter. **The subspace fits what it was fit on:** our
covariance was mostly code and long context, and the one place the shortlist measurably costs acceptance is short
greedy prose (below). `--save-cov` keeps the covariance for later ranks.

The module checks the artifact against the head it finds at start-up (`[B70] draft shortlist: ON: rank 512, N 128,
... head fingerprint ..., nibbles equal 1.00000`) and turns itself off with one `OFF` line otherwise: a different
checkpoint, a different draft vocab, a different row count, a missing file or a missing XPU op all give the stock
path. The int4 scales are bit-identical on every device; the nibbles can differ by 1 where a value sits on a rounding
tie (0.145% of nibbles between a CPU and a CUDA build), which the check tolerates.

## Apply and serve

After `scripts/apply_stack.sh` (which does not apply this), from the recipe folder with the venv's Python:

```sh
VLLM_ROOT=/opt/b70/vllm032/src/vllm /opt/b70/vllm032/venv/bin/python patches/shortlist/patch_shortlist_032.py
```

The installer is idempotent, fail-closed (exact anchors), leaves `.bak-2026-10-22-pre-shortlist` copies and
`py_compile`s what it touched. It also upgrades a tree carrying the first (greedy-only) revision of the hook in place.
Then in `prod.env` (the commented lines in `serve/prod.env.example`):

```sh
export B70_DRAFT_SHORTLIST=/opt/b70/vllm032/shortlist/shortlist_r512.pt B70_DRAFT_SHORTLIST_N=128
```

Unset = the module is never imported and the stock code runs. The init runs right after the drafter's weights load,
before warm-up and graph capture; check that the `ON` line comes before the first `Capturing` line in the server log.
With the combined speculator the MTP drafts go through the shortlist and the copied drafts never do. Knobs:
`B70_DRAFT_SHORTLIST_N` (128; 64 saves 2 us more per pass and costs 0.14 points more acceptance; 256 halves the
offline miss and costs 10-20 us), `B70_DRAFT_SHORTLIST_BLOCK` (0), `B70_DRAFT_SHORTLIST_PROB` (1: probabilistic
drafts sample from the shortlist's row; 0: they take the stock full-head path).

## Probabilistic drafts

With `draft_sample_method: "probabilistic"` in the speculative config, the stock speculator samples each draft from
the drafter's distribution and caches that distribution as the q the rejection sampler checks against. The first
revision of this hook stepped aside in that case (correct, but no speedup). Now `logits_row()` scatters the N rescored
logits into a -inf row of the full vocab width and the hook samples from it with the stock `gumbel_sample(...,
logits_cache=draft_logits)`, so the cached q is by construction the distribution the draft came from: lossless. q is a
softmax over the N candidates only; the target's top-k 20 lies inside them almost always (offline: probabilistic
shortlist vs probabilistic full head -0.01% chain tokens per step, `results/prob-draft/offline/held_K5_shortlist.log`).
The combined speculator's copied drafts get one-hot rows from its own kernel ([combo-spec](../combo-spec/README.md)).

## What it cost on prose

A fixed-prompt prose A/B on the same seat, pre-shortlist vs shortlist (`bench/prose_ab.py`, 10 prompts;
`results/shortlist/seat-ab/prose-base.out`, `prose-sl.out`):

| mode | tokens per step | decode t/s |
|---|---|---|
| greedy, thinking off | 2.337 -> 2.276 (**-2.6%**; on the 6 of 10 identical texts -2.3%) | 66.1 -> 67.1 (+1.5%) |
| the model's default sampling (temperature 1.0, top-k 20, top-p 0.95) | 1.943 -> 1.941 (-0.1%) | 53.9 -> 56.2 (**+4.2%**) |
| thinking on, effort medium | 2.350 -> 2.278 (**-3.1%**, per-prompt spread ±12%) | 65.0 -> 65.7 (+1.1%) |

The offline replay (code-heavy traffic) had predicted -0.1%. Short greedy prose is where the drafter is weakest (1.9
to 2.4 tokens per step against 4.6 on code) and where a subspace fit on code misses most. The -3.4% step time still
wins in every mode, so the shortlist stays on; a prose-heavy covariance or rank 768 would close the gap.

## Tests

- `tests/test_shortlist_cpu.py` and `tests/test_prob_cpu.py` run on a CPU with the artifact, the model's `lm_head`
  (`lm_head_bf16.bin`, raw bf16 as `build_factors.py --head-bin` reads it) and a sample of real head inputs
  (`hm_held_sample.pt`) in `SHORTLIST_DATA`. Ours: ALL PASS (`results/shortlist/offline/test_shortlist_cpu.out`,
  `results/prob-draft/offline/test_prob_cpu.out`).
- On the card: `bench/bench.py`, `bench/editbench.py` (6/6 exact), `bench/crossing.py`, `bench/tierbench.py`,
  `bench/prose_ab.py`. The numbers are in [docs/RESULTS.md](../../docs/RESULTS.md#shortlist-draft-head).

## Credits

The int4 draft head this shortlists is [SergiioB][sergiiob]'s (`patch_draft_lmhead_int4`), with the row trim from
[Vcvzgbzz][vcv] via [riceharvest][rice]'s `patch_draft_vocab`. The int4 GEMM is vLLM-XPU's own `int4_gemm_w4a16`.

[sergiiob]: https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
[vcv]: https://github.com/Vcvzgbzz/b70-cookbook
[rice]: https://github.com/riceharvest/b70-qwen38-cookbook
