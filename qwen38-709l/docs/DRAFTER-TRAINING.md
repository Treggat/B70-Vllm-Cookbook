# Training a DFlash2 drafter for the 709-L (in progress)

The published DFlash2 drafter accepts 2-5% of its tokens on the 709-L (RESULTS.md), because it reads inner layers the
merge rewrote. A drafter for a merge has to learn that merge's hidden states. This page records the staged attempt on
our two B70s. Stage 0 is done; Stage 1, the served test, has not run.

## The plan in three stages

| stage | what is trained | data | decides |
|---|---|---|---|
| 0 | only the input projection, the convolutions, the selector and the norms (325M of 1,924M parameters); the 5-layer body stays frozen | ~400 samples | does the published drafter's training transfer once its inputs are re-mapped to the merge? |
| 1 | the whole drafter, 400 steps, warm-started from Stage 0 | ~1,600 samples | served speed on the real bench against MTP-5 |
| 2 | the whole drafter at recipe scale | ~15k samples | only if Stage 1 beats MTP-5 |

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
(4.5-5.2, optimistic because those samples had been seen). Only Stage 1's served bench says whether it beats MTP-5.

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
