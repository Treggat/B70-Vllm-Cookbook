# Training a DFlash2 drafter for the 709-L (stopped: MTP-5 stays)

The published DFlash2 drafter accepts 2-5% of its tokens on the 709-L (RESULTS.md), because it reads inner layers the
merge rewrote. A drafter for a merge has to learn that merge's hidden states. This page records the staged attempt on
our two B70s. Stage 0 and Stage 2 are done and served-tested. Stage 2 failed its gate, so the seat keeps MTP-5
(and, since 2026-10-21, the combined copy + MTP speculator in [patches/combo-spec/](../patches/combo-spec/README.md)).

## The plan in three stages

| stage | what is trained | data | decides |
|---|---|---|---|
| 0 | only the input projection, the convolutions, the selector and the norms (325M of 1,924M parameters); the 5-layer body stays frozen | ~400 samples | does the published drafter's training transfer once its inputs are re-mapped to the merge? |
| 1 | (skipped: serving the Stage 0 drafter answered its question, below) | | |
| 2 | Stage 0's parameters plus the last two drafter layers, warm-started from Stage 0 | ~15k samples: the 709-L's real replies plus its own regenerations | served tokens per step against MTP-5: **failed** (below) |

Tooling: [SpecForge](https://github.com/sgl-project/specforge) in offline mode (`training.strategy: dflash`, not
`dspark`: the `dflash` strategy exports the layout stock vLLM samples, so the off-by-one fix that DSpark-trained
drafters need does not apply). It has no published XPU port, so ours runs it eagerly on XPU with a freeze hook for
Stage 0. The warm start is the published drafter. The training samples are the 709-L's own replies from our agent
harness (on-policy data matters: the SpecForge recipe notes that acceptance on off-policy answers plateaus far below).
The test split is later sessions than the training split, so it is scored on chats it never saw.

## Stage 0 result

One B70, about 1 h 45 min including capturing the target's hidden states for 524 samples (1.28M tokens, 62 GB).

| test split, teacher-forced per-position accuracy | |
|---|---|
| published drafter, untrained (served: 2.9-5.1% per draft token) | 5.5% |
| step 50 | 35.1% |
| step 100 | 43.0% |
| step 150 | 47.0% |
| step 200 | **48.2%** (still rising, +1.2 points over the last 50 steps) |

The untrained number matches what the server measured, so the offline metric can be trusted. 48% is roughly what the
published drafter scores on the base model it was trained for. This is teacher-forced accuracy, not a served rate:
served tokens per step will land between the independent-position bound (about 1.9) and the training-batch figure
(4.5-5.2, optimistic because those samples had been seen). The served test below says where it really landed.

## Stage 0, served (`results/drafter/`)

The Stage 0 drafter loaded on the 709-L seat, same bench as everything else in this recipe, 262k window, through an
OpenAI-compatible endpoint. Two runs: on the stock attention library, and on a library that has both the MTP and the
drafter's kernels ([patches/kernels-union/](../patches/kernels-union/README.md)) with packed verify on.

| | MTP-5 (prod) | Stage 0, stock library | Stage 0, union library |
|---|---|---|---|
| short code, t/s (3 runs) | 130.8 (127.6 / 134.5 / 130.4) | 73.1 (80.8 / 77.0 / 61.4) | 68.2 (74.2 / 68.9 / 61.5) |
| tokens per step / per draft token | 4.64 / 72.8% | 2.64 / 23.4% | 2.41 / 20.2% |
| time per step, short context | 35.5 ms | 36.1 ms | 35.3 ms |
| 4 streams, summed | 259.2 | 157.9 | 160.7 |
| decode 24k / 32k / 65k / 74k / 100k | 63.0 / 67.9 / 67.1 / 53.9 / 58.1 | 50.6 / 46.2 / 40.2 / 37.0 / 31.8 | 53.0 / 52.7 / 51.1 / 47.4 / 46.2 |
| tokens per step at 24k / 65k / 100k | 2.59 / 3.15 / 3.02 | 2.29 / 2.31 / 2.16 | 2.16 / 2.40 / 2.43 |
| first token 100k, cold | 75.1 s | 97.1 s | 75.0 s |
| KV pool | 416,591 tokens | 288,738 | 288,738 |

Acceptance per draft position (union run, whole bench): 67.5, 50.7, 42.2, 37.6, 33.9, 30.2, 26.8%. On the
repetitive page-boundary test: 96, 91, 87, 84, 80, 74, 67%, i.e. 6.8 tokens per step. Image, video and the boundary
test pass on both.

What it says:

- **The drafter now reads the merge.** Twice the stock drafter's acceptance (1.35 to 2.6 tokens per step on code),
  and the first draft position is right two times in three.
- **Offline accuracy overstates served acceptance by about 2x** here: 48% teacher-forced became ~20-23% per draft
  token served.
- **With the union library, the drafter's step costs the same as MTP-5's** (35.3 vs 35.5 ms at short context,
  ~52.6 vs 52.0 ms at 100k). Acceptance is the whole gap. It beats MTP-5 at depth once it passes ~3.0 tokens per step
  (~29% per draft token) and on short code once it passes ~4.65 (~52%).
- **The library only helps at depth:** +27-45% decode at 65k-100k against the stock library, prefill back to the
  MTP seat's speed, nothing at short context, where attention is a small share of the step. The ~26 ms step
  Vcvzgbzz reports for DFlash2 at short context is not explained by the kernels alone.
- **The drafter costs KV room:** its own KV cache takes part of the pool, 416k to 289k tokens. One 262k request still
  fits.

## Stage 2 result: the gate fails (`results/drafter/stage2-*`)

Stage 0's trained parameters plus the last two drafter layers (965M of 1,924M parameters), warm-started from Stage 0,
one epoch over about 14,800 samples (the 709-L's real replies plus its own regenerations, contexts up to 6,144 tokens),
global batch 8, lr 5e-5 constant. It trained on both cards as two independent trainers with local SGD (below), in 9
rounds of ~100 steps per card, about 9 hours including captures.

| test split, teacher-forced accuracy | acc | loss |
|---|---|---|
| Stage 0 drafter | 50.3% | 2.872 |
| after chunk 1 (~5k samples) | 55.7% | 2.533 |
| after chunk 2 | 57.1% | |
| **after chunk 3, final** | **58.0%** | **2.376** |

Served, same bench and union library with packed verify:

| | MTP-5 (prod) | Stage 0 | **Stage 2** | gate |
|---|---|---|---|---|
| short code, t/s | 130.8 | 68.2 | **77.7** | |
| short code, tokens per step | 4.64 | 2.41 | **2.76** | > 4.65 |
| 4 streams, summed t/s | 259.2 | 160.7 | **167.2** | |
| decode 24k / 32k / 65k / 74k / 100k | 63.0 / 67.9 / 67.1 / 53.9 / 58.1 | 53.0 / 52.7 / 51.1 / 47.4 / 46.2 | **54.7 / 52.7 / 50.6 / 48.6 / 45.3** | |
| tokens per step at 65k / 100k | 3.15 / 3.02 | 2.40 / 2.43 | **2.38 / 2.39** | > 3.0 |

Acceptance per draft position over the whole bench: 72.4, 54.2, 44.3, 39.2, 36.1, 33.4, 31.4% (Stage 0: 67.5 ... 26.8%).
Position 0 is still the highest, so the drafter is aligned with vLLM's readout. The page-boundary test passes at
7.07 tokens per step; image and video pass.

What it says:

- **More data moved short context only.** 37x Stage 0's samples and two more trainable layers gave +7.7 points offline
  and +15% served on short code. Depth did not move at all.
- **Likely cause: the drafter never saw long context.** Training ran at `max_length: 4096` on contexts up to 6,144
  tokens; the depth bench runs at 24k-100k. More short samples will not close that gap.
- **The next lever would be long-context training**: prefill each sample at its real depth (32k-200k) and keep the
  hidden states only for the last few thousand tokens. A full capture costs ~50 KB per token, about 10 GB for one 200k
  sample, so tail-only capture is what makes it affordable. Even then it targets depth; short code still has to go
  from 2.76 to 4.65 tokens per step. We stopped here.

## What tripped us (fix before running it yourself)

- **oneCCL wants `/dev/dri/by-path`.** `--device /dev/dri` does not create it in a container. Link only the training
  card (`pci-<addr>-card`, `-render`): Level Zero then enumerates only that card, and the serving card stays invisible.
- **The training process needs the same environment as the server:** the venv's `lib/` on `LD_LIBRARY_PATH` (oneCCL
  plugins), `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `VLLM_TARGET_DEVICE=xpu`.
- **A capture script that starts vLLM needs a `__main__` guard**: vLLM spawns workers that re-import it.
- **SpecForge saves a full checkpoint (22 GB here) at the end, at every save interval and at every better eval.** Ours
  briefly pushed a share below its free-space floor. Save only the trained parameters for a partial run (0.65 GB).
- **Hold the training card for the whole window.** Anything that routes to it (a model swapper reloading its seat)
  collides with the training job. We stopped the clients that use that card for the window.
- **Put hidden-state captures on bulk storage.** They take about 50 KB per token for this model (five 5,120-wide
  layers in bf16).
- **Two-card training over xccl does not work on this stack.** A plain `torchrun` with two ranks (one per card,
  `CCL_TOPO_P2P_ACCESS=0`) died on its first cross-card `all_reduce` (oneCCL `atl_ofi` "Bad address") and took a
  copy-engine reset on one card with it. vLLM's own tensor-parallel workers do run across both cards; plain torchrun
  xccl does not. What worked: two independent single-card trainers, each on its half of the data, averaged on the CPU
  after every round (local SGD). A 30-step validation round: each card alone 51.7% / 51.8%, the average 52.3%.
- **One card runs out of memory at Stage 2's size** (965M trainable): it spilled ~27-30 GB into system RAM and hit
  `DEVICE_LOST` at step ~140. Gradient checkpointing on the drafter layers plus SpecForge's
  `optimizer_cpu_offload: true` (fp32 Adam state on the host) kept it at 25-26 GB on the card, 0.5 GB spill, and made
  it faster (13.7 s/step).
- **Slow array disks starve the trainer.** One card's captures sat on a drive that read 64-90 MB/s through Unraid's
  md layer with mmap. Reading each capture with O_DIRECT in 16 MB blocks helped, but that card still ran ~23 s per
  step against 15.4 on the other. Put captures on your fastest disk.
