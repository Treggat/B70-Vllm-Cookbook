# Qwen3.8-27B 709-L on one Intel Arc Pro B70 (vLLM 0.32 XPU)

[← B70 vLLM Cookbook](../README.md)

A reproducible recipe for serving a **merged Qwen3.8-27B fine-tune** ([DavidAU](https://huggingface.co/DavidAU)'s [*TWIN-TURBO Fable Cold-Fusion 709-L*](https://huggingface.co/DavidAU/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-NM-DAU-NEO-MTP-GGUF))
as an **AWQ W4A16** checkpoint on **one Arc Pro B70 (32 GB)**, with MTP speculative decoding, vision, the model's full
262k context and 4 concurrent streams. vLLM 0.32.0 (XPU, built from source; the 0.31 recipe is in this folder's git history), the
[SergiioB][sergiiob] and [Vcvzgbzz][vcv] cookbook patches, and [riceharvest][rice]'s b70 layer on top, plus four small
patches of our own (one of them SergiioB's MTP boundary fix re-based onto 0.32).

Most B70 write-ups cover base Qwen3.8 in GPTQ or EXL3. This one covers:

- a merged model shipped as compressed-tensors W4A16, re-laid into the AutoAWQ layout that vLLM-XPU can load ([docs/AWQ.md](docs/AWQ.md))
- stacking three community patch sets on one tree, including the one clash between them ([docs/RECIPE.md](docs/RECIPE.md))
- per-patch measurements of riceharvest's layer, each patch loaded on its own, and the 0.31 → 0.32 move with the same
  stack ([docs/RESULTS.md](docs/RESULTS.md))
- a max-model-len boundary test, a check that vLLM's CPU KV offload tier never hits on this hybrid model, and a
  chunked-prefill sweep for the "one big chat starves the small ones" case ([docs/RESULTS.md](docs/RESULTS.md))
- what broke along the way ([docs/PITFALLS.md](docs/PITFALLS.md))
- why the published DFlash2 drafter fails on a merge, and a staged attempt to train one for it on the B70s
  ([docs/DRAFTER-TRAINING.md](docs/DRAFTER-TRAINING.md); Stage 2 missed its gate, so MTP-5 stays), and a kernel library that serves both
  ([patches/kernels-union/](patches/kernels-union/README.md))
- the seat on [Vcvzgbzz's](https://github.com/Vcvzgbzz/b70-cookbook) own test scripts, side by side with his numbers ([docs/RESULTS.md](docs/RESULTS.md))
- one engine that switches each step between MTP-5 and 15 copied drafts, at every context depth: edits that return the
  whole file go from 155 to 235 t/s, exact, with everything else at MTP-5 speed ([patches/combo-spec/](patches/combo-spec/README.md))
- a shortlist draft head: the MTP drafter scores 128 candidates from a rank-512 factorization instead of reading its
  whole int4 head every pass, -3.4% step time at every depth ([patches/shortlist/](patches/shortlist/README.md))
- a V-split verify kernel: MTP-5's 36 verify rows read the KV once per step instead of five times, one attention
  layer 34-45% faster from 23k to 250k, decode behind 210k of context +23% ([patches/vsplit-verify/](patches/vsplit-verify/README.md))
- probabilistic MTP drafts: sampled prose +14-18% tokens per step, 4 writers +8%, greedy unchanged, and why that
  applies to every ordinary request on the seat ([docs/RESULTS.md](docs/RESULTS.md#probabilistic-mtp-drafting))
- a fixed-prompt prose bench next to the code benches, because prose is where MTP is weakest (`bench/prose_ab.py`)
- where a decode step's time goes on this model, and five experiments settled with numbers: a sliding window for
  the MTP head, a 64-row verify tile, sibling-tree speculation, a per-step draft count from the drafter's
  confidence, and a 1.7x GDN kernel ([docs/RESULTS.md](docs/RESULTS.md#what-we-tried-and-dropped))

Before vLLM we ran the same model on llama.cpp SYCL with our own kernel patches. That work is in
[b70-llamacpp-cookbook](https://github.com/Treggat/b70-llamacpp-cookbook).

## Results

Our live seat on 2026-10-25: vLLM 0.32, the full stack below plus the combined speculator, the shortlist draft head,
probabilistic drafts and the V-split verify kernel, 262k window, fp8 KV in a 14.5 GiB pool. Through an
OpenAI-compatible endpoint, one stream unless said, single runs (treat about ±5% as noise). "Sampled" is the model's default sampling (temperature 1.0,
top-k 20, top-p 0.95), which is what every request without its own sampling fields gets. Conditions, older configs
and raw files: [docs/RESULTS.md](docs/RESULTS.md).

| | live seat | MTP-5, greedy drafts, full draft head |
|---|---|---|
| short code, greedy | **135.0 t/s**⁶ | 131.7 |
| whole-file edits (3 tasks, greedy and sampled, all 6 exact) | **238.8 t/s** | 155 |
| whole-file edit behind 94k tokens of context | **171.1 t/s**¹ | 110.7 |
| prose, sampled | **63.6 t/s** (2.19 tokens per step) | 53.9 (1.94)² |
| prose, thinking on at effort medium | **74.9 t/s** (2.59) | 65.0 (2.35)² |
| prose, greedy | 66.4 t/s (2.24) | 66.1 (2.34)² |
| 4 writers, sampled, summed | **284–292 t/s** | 265–270³ |
| decode behind 61k / 120k of context, sampled | **83 / 65 t/s**⁶ | 71 / 57³ |
| decode behind 64k / 105k / 210k of context, greedy | **91.2 / 92.1 / 63.6–64.8 t/s** | 84.7 / 84.7 / 51.1–52.5⁷ |
| decode after a cold 100k prompt, greedy | 56–60 t/s⁴ | 56–60 |
| first token, cold 98k prompt | 77.0 s⁵ | 75 s |
| KV pool | 375,067 tokens: one 262k chat, or 4 of ~93k | 416,591 |

¹ Measured 2026-10-21 with the combined speculator and greedy drafts; not re-measured since.

² The same seat on 2026-10-22 before the shortlist, in the fixed-prompt prose bench (`bench/prose_ab.py`): the
combined speculator never copies on prose, so this is MTP-5 with the full draft head and greedy drafts.

³ The live seat on 2026-10-23 just before probabilistic drafting went on (shortlist, greedy drafts).

⁴ Measured with MTP-5 alone. The combined speculator matched MTP-5 at depth wherever we checked (23k–120k), and the
shortlist cut every measured step time by 3.4%.

⁵ `bench/bench.py` on 2026-10-25 (97,825 tokens). The 75 s was measured before we turned on
`--long-prefill-token-threshold 512`, which keeps other chats decoding during a big prefill (2.2 → 10.7 t/s).

⁶ Measured 2026-10-23, before the V-split kernel. It left short code at parity (134.7 t/s, 2026-10-25) and cut greedy
step time 3.2 ms at 64k and 4.4 ms at 105k; the sampled depth rows were not re-measured.

⁷ The live seat the same evening before the V-split kernel, in the same test window.

How we got here, each step measured on the same box ([docs/RESULTS.md](docs/RESULTS.md)):

- **riceharvest's b70 layer on vLLM 0.31:** short code +16% (119 → 132 t/s), 4 streams +20%, first token at 75k −20%,
  decode at 75k +29%, KV pool +45%. Per-patch numbers show where each gain comes from.
- **vLLM 0.31 → 0.32, same stack** (2026-10-19), and **131k → 262k window:** parity within noise on every row.
- **Combined copy + MTP speculator** (2026-10-21): whole-file edits +50%, first below 32k tokens, then at every depth.
- **Shortlist draft head** (2026-10-22): 0.33 ms less per draft pass, -3.4% step time from 23k to 120k, short code
  +4%, edits +2%; costs 2-3% acceptance on short greedy prose, none on sampled prose, no mode slower.
- **Probabilistic drafts** (2026-10-23): sampled prose +14.5% tokens per step, thinking prose +11%, 4 writers +8%,
  sampled decode behind 61k / 120k +17% / +14%; greedy requests unchanged.
- **V-split verify kernel** (2026-10-25): one attention layer -34 to -45% from 23k to 250k; decode behind 64k / 105k /
  210k +7.7% / +8.7% / +23%, short code, edits and 4 writers at parity, greedy output byte-identical.
- **DFlash2:** the published drafter is about 3× slower than MTP-5 on this merge; the drafter we trained for it missed
  its gate, so MTP-5 stays.
- **Dropped, with numbers** ([what we tried](docs/RESULTS.md#what-we-tried-and-dropped)): a sliding window for the MTP
  head (-13 to -16% acceptance at depth), sibling-tree speculation (+11-20% step time for a +5% ceiling: every verify
  row costs 0.5-3 ms on this hybrid), a per-step draft count from a confidence the scheduler only sees 2-3 rounds
  late, and a 64-row verify tile that is at parity at K = 5. A bit-exact 1.7x GDN kernel works but is not in the
  live stack.

Every config passes the vision, video and page-boundary tests (`bench/crossing.py`), and every run passes the
corruption canary with 0 preemptions.

## Quick start

Run these on the Linux box with the B70, from this folder (`qwen38-709l/`).

```sh
sh scripts/build_venv.sh /opt/b70/vllm032                         # vLLM 0.32.0 XPU from source, own venv
# riceharvest's v2 attention kernel (zero-sum guard): their GitHub release v3, or their kernels/build.sh.
# The one we ran: sha256 8d1bf4d1ea74...; apply_stack.sh refuses the v1 kernel (aa241d52a601...)
sh scripts/apply_stack.sh /opt/b70/vllm032 ./libattn_kernels_xe_2.b70.so
python scripts/make_draft_vocab.py serve/draft_vocab_98304.json
python scripts/ct2awq.py <downloaded W4A16 dir> <out dir>          # one-time, see docs/AWQ.md
cp serve/prod.env.example serve/prod.env; cp serve/prod.extra.example serve/prod.extra
VENV=/opt/b70/vllm032 MODEL=<out dir> sh serve/serve-709l.sh 8000
```

That serves plain MTP-5. For the live config (the combined speculator, the shortlist draft head and probabilistic
drafts), apply their patches to the same venv, put the shortlist artifact in place, uncomment both blocks in
`serve/prod.env` and restart:

```sh
P=/opt/b70/vllm032/venv/bin/python
VLLM_ROOT=/opt/b70/vllm032/src/vllm $P patches/combo-spec/patch_gdn_varlen_spec_032.py
VLLM_ROOT=/opt/b70/vllm032/src/vllm $P patches/combo-spec/patch_combo_spec_032.py
VLLM032_ROOT=/opt/b70/vllm032 $P patches/combo-spec/patch_spec_packed_guard_032.py
VLLM_ROOT=/opt/b70/vllm032/src/vllm $P patches/combo-spec/patch_align_backward_precopy_032.py
VLLM_ROOT=/opt/b70/vllm032/src/vllm $P patches/shortlist/patch_shortlist_032.py
mkdir -p /opt/b70/vllm032/shortlist   # the artifact: build it with patches/shortlist/build_factors.py or download ours (its README)
```

The V-split verify kernel is a rebuilt attention library: build it, install it and patch the runtime as in
[patches/vsplit-verify/](patches/vsplit-verify/README.md), then uncomment its `B70_SPEC_VSPLIT=1` line in
`serve/prod.env`.

Then check it before trusting it, from `bench/` with `BASE_URL` pointing at the server (see `bench/common.py`):
`python crossing.py && python bench.py mine`, `python editbench.py mine` for the speculator, `python prose_ab.py mine`
for prose, and once per engine change `python maxlen.py` (sent to the server directly, see the script).

## Layout

| path | what |
|---|---|
| `patches/*.py` | our four vLLM patches (memory-query fallback, BF16 MTP `fc`, video sent as `image_url`, SergiioB's MTP boundary fix re-based onto 0.32); `apply_stack.sh` applies them |
| `patches/combo-spec/` | the combined copy + MTP speculator: scheduler policy, speculator (with one-hot draft-logit rows for probabilistic drafts), GDN, GDN-state and packed-verify fixes, tests ([README](patches/combo-spec/README.md)) |
| `patches/shortlist/` | the shortlist draft head: module, installer, artifact builder, CPU tests ([README](patches/shortlist/README.md)) |
| `patches/dflash2/` | Vcvzgbzz's DFlash2 drafter patches re-based onto 0.32, for the drafter measurements (not in the live stack) |
| `patches/kernels-union/` | a kernel library config that serves both MTP and DFlash2 shapes ([README](patches/kernels-union/README.md)) |
| `patches/vsplit-verify/` | the V-split verify kernel: vllm-xpu-kernels patch (Apache-2.0), runtime installer, exactness, determinism and timing tests ([README](patches/vsplit-verify/README.md)) |
| `scripts/build_venv.sh` | vLLM 0.32.0 XPU source build in its own venv (indexes, pins, a flag check of the serve line) |
| `scripts/apply_stack.sh` | clones the upstream cookbooks at pinned commits, points their patches at one tree, applies the whole stack fail-closed |
| `scripts/ct2awq.py` | compressed-tensors pack-quantized W4A16 → AutoAWQ gemm, bit-identical |
| `scripts/make_draft_vocab.py` | Vcvzgbzz's 98,304-row draft trim as a JSON id list for riceharvest's patch |
| `serve/` | launcher + the measured config (`prod.env.example`, `prod.extra.example`) |
| `bench/bench.py`, `bench/crossing.py` | speed and sanity bench (short code, 4 writers, depth, greedy hashes); page-boundary corruption test |
| `bench/maxlen.py`, `bench/stress4.py`, `bench/kvoff_probe.py` | max-model-len boundary test; 4 long chats at once; does the CPU KV offload tier ever hit |
| `bench/starve.py` | decode of one chat while another's long cold prompt prefills (the chunked-prefill sweep) |
| `bench/editbench.py`, `bench/tierbench.py`, `bench/snap.py` | whole-file edit speed with an exact-match check; decode and edits at depth (`TIER_TEMP` for sampled runs); spec-decode counters around agent turns |
| `bench/prose_ab.py`, `bench/prose_compare.py` | fixed-prompt prose bench: greedy, the model's default sampling and thinking on, tokens per step and t/s per mode; and the A/B comparison of two runs |
| `bench/common.py` | endpoint settings shared by the bench scripts (`BASE_URL`, `MODEL`, `API_KEY`, `METRICS_URL`) |
| `results/` | raw output for every number in the docs, one folder per topic (generated text stripped) |

## Credits

This stands on other people's work. Please go star them:

- **[SergiioB/intel-arc-pro-b70-inference-cookbook][sergiiob]** (MIT): the base patch set (MTP nightly/boundary, int4 draft lm_head + MTP linears, prefix-cache fixes) and most of what is known about vLLM on the B70.
- **[Vcvzgbzz/b70-cookbook][vcv]** (MIT): the draft-vocab trim and the recipe our first vLLM seat was built from. His
  2026-10-19 update (dd083d5) publishes the fast fp8 convert, in-kernel packed verify, fp8 head and embed UVA with the
  same `B70_*` switches as riceharvest's layer, plus a DFlash2 drafter; we checked it against this stack (RECIPE.md).
- **[riceharvest/b70-qwen38-cookbook][rice]** (MIT): packed MTP verify, the b70 attention kernel, fp8 lm_head, embed UVA, kv-splits.
- **Claude** (Anthropic; Claude Opus 5.5 and Claude Fable 5.1, through Claude Code): ported and tested the patch stack,
  wrote our patches and the scripts, ran the measurements and wrote this cookbook.
- Model: **[DavidAU](https://huggingface.co/DavidAU)** for the [709-L merge](https://huggingface.co/DavidAU/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-NM-DAU-NEO-MTP-GGUF), and **TheUnderscore** for the [W4A16 quant](https://huggingface.co/TheUnderscore/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-W4A16-AWQ). See [docs/AWQ.md](docs/AWQ.md).

We link to the upstream patches and don't copy them in. `apply_stack.sh` fetches them at the commits we measured.

## License

Our code and docs: MIT ([LICENSE](../LICENSE)). Upstream patches and kernels keep their own licenses. The exception is
`patches/vsplit-verify/b70-verify-kernels.patch`: it modifies vllm-xpu-kernels sources, which are Apache-2.0, and is
offered under Apache-2.0 too (its license file sits beside it).

[sergiiob]: https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook
[vcv]: https://github.com/Vcvzgbzz/b70-cookbook
[rice]: https://github.com/riceharvest/b70-qwen38-cookbook
