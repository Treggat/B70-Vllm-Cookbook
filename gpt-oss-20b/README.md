# gpt-oss-20b on one Intel Arc Pro B70 (vLLM 0.31 XPU), plus an anti-loop fine-tune

[← B70 vLLM Cookbook](../README.md)

A recipe for serving OpenAI's **[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)** (MXFP4) on **one Arc Pro
B70 (32 GB)** with vLLM 0.31.0 XPU at 16 concurrent streams, with reliable tool calls. It also covers how we trained
gpt-oss-20b out of its repetition loops on **two** B70s with Liquid AI's [Antidoom][antidoom], and what we learned
running it as the model behind a coding agent.

This page covers:

- serving: a small vLLM plugin that supplies the harmony tool-call grammar, maps and caps reasoning effort, and keeps a
  parser error from killing a stream ([docs/RECIPE.md](docs/RECIPE.md))
- the anti-loop fine-tune: Antidoom's FTPO LoRA ported to Intel XPU, merged back into the MXFP4 checkpoint so it
  still serves on one card, three rounds with every number ([docs/ANTIDOOM.md](docs/ANTIDOOM.md))
- agent harness lessons: two harness messages that made gpt-oss write a report instead of calling a tool, and the
  fixes ([docs/AGENT-HARNESS.md](docs/AGENT-HARNESS.md))
- what broke along the way ([docs/PITFALLS.md](docs/PITFALLS.md))

**Weights: [Taggert2222/gpt-oss-20b-antidoom](https://huggingface.co/Taggert2222/gpt-oss-20b-antidoom)** on Hugging
Face. That's variant B, merged into the MXFP4 checkpoint and ready to serve, with the unmerged LoRA in its `adapter/`
folder. To serve it, set `MODEL=Taggert2222/gpt-oss-20b-antidoom` (or a local download) in the quick start below. The
scripts here also rebuild it from the public base model and Liquid AI's public prompt mix.

## Results

Our own evals on one box (two B70s; serving on one of them), October 2026. Stock = `openai/gpt-oss-20b`. Variant B =
stock plus our Antidoom round 2 LoRA (attention only) merged in. Both served by the same vLLM 0.31 build, plugin and
flags. Sample sizes are in the table. They are small, so read single-digit differences as noise unless a test is
given.

| | stock | **variant B** |
|---|---|---|
| repetition loops, 200 held-out prompts, effort medium | 17/200 (8.5%) | **0/200** |
| repetition loops, same 200, effort high | 79/200 (39.5%) | **2/200** (1.0%) |
| graded accuracy, 700 items (GSM8K, MATH-500, MMLU, CommonsenseQA), medium | 78.4% (549) | **83.0%** (581)¹ |
| PTC probe (code inside a tool call), 72 calls, medium, same session | 57/72 | 57/72 |
| tool_flow, 2 × 16 streaming multi-turn sessions, medium: format faults | 0 | 0 |
| coding-agent tasks in our agent harness, 28 per model, verified on disk | 17 pass | **20 pass**² |

¹ Paired on the same 700 items: B fixes 47, breaks 15 (McNemar χ² 15.5, p < 0.001). For the round 1 model, 24 of its
49 fixes were items where stock looped or hit the 4,000-token cap.
² Stock: 17 pass, 8 partial, 1 fabricated report, 1 honest fail, 1 API error, and 2 sessions that ran `rm -rf` on
shared folders. B: 20 pass, 5 partial, 3 fabricated reports, 0 of the rest. The fabrication difference (3 vs 1) is
not significant (Fisher p ≈ 0.6). See [docs/AGENT-HARNESS.md](docs/AGENT-HARNESS.md).

**4-hour soak of B**, 16 streams, effort medium, chat prompts: 4,950 generations, 5,180 requests answered, one 502
from the proxy in front of vLLM (the server logged nothing), no NaN/canary events, no server errors. 3 generations
looped to the token cap and 2 had a short degenerate stretch that recovered: about 0.1%. (10 more requests show as
aborted: the soak's own timeout killed them in flight at the end.)

**Engine speed**, measured 2026-10-14 on stock weights, vLLM 0.31.0, one B70, XPU graphs on, short prompts:
1 stream 106–117 tok/s, 4 streams 257 tok/s summed, 16 streams 698 tok/s summed. Eager mode (no graphs) did about
43 tok/s on one stream. B has the same tensors and shapes, so we did not re-measure speed. It writes shorter replies:
median 305 completion tokens on the loop set at medium, stock 418.

**fp8 KV cache** (`serve/gpt-oss.extra.example` since 2026-10-18): the KV pool doubles from 643,585 to 1,287,170
tokens at the same decode speed. 16 agent-sized contexts of 69k each all stay cached (bf16: 0.1%), so their next turn
starts in 0.3–1.5 s instead of 15–230 s. A cold long read is about 25% slower on the stock kernels, about 11% with
our optional rebuilt kernels ([kernels/](kernels/README.md); output byte-identical). Variant B's gates hold: loops 1/200 at
medium and high, graded 83.9%, tool_flow 0 format faults, PTC 56/72. A 69-minute 16-stream soak on fp8 (339 chat
replies, 465 long reads, 625 tool calls) had 0 errors and 0 NaN replies. Details in
[docs/RECIPE.md](docs/RECIPE.md#kv-cache-fp8).

**What did not work:** round 3 tried to also train out a "narrated tool call" habit (the model writes the call or a
made-up result as text). Three variants cut it on our replay set (27/30 → 11, 1 and 9 of 30) but each broke
something else: reasoning leaked into answers, plain-chat answers came back empty, or code-in-tool-call scores
dropped. Changing two harness messages instead took B from 27/30 to 2/30 with no weight change. Details in
[docs/ANTIDOOM.md](docs/ANTIDOOM.md#round-3-the-narrated-tool-call-what-failed-and-why) and
[docs/AGENT-HARNESS.md](docs/AGENT-HARNESS.md).

## Quick start

Run these from this folder (`gpt-oss-20b/`). Serving needs one B70; training needs two.

```sh
# 1. vLLM 0.31.0 XPU venv: the same build as the Qwen page (check oss-harmony afterwards, see docs/RECIPE.md)
sh ../qwen38-709l/scripts/build_venv.sh /opt/b70/vllm031

# 2. check the plugin against that venv (offline, no GPU): must end "31/31 passed"
PYTHONPATH=$PWD/plugin GPTOSS_MODEL_DIR=/models/gpt-oss-20b /opt/b70/vllm031/venv/bin/python plugin/selftest.py

# 3. serve (XPU graphs and fp8 KV cache on via the .extra)
cp serve/gpt-oss.extra.example serve/gpt-oss.extra
VENV=/opt/b70/vllm031 MODEL=/models/gpt-oss-20b CARD=0 sh serve/serve-gpt-oss.sh 8000

# 4. check it before trusting it
python bench/tool_flow.py http://localhost:8000 gpt-oss --sessions 16 --turns 8 --extra '{"reasoning_effort":"medium"}'
python bench/ptc_probe.py http://localhost:8000 gpt-oss --rounds 3 --extra '{"reasoning_effort":"medium"}'
```

To rebuild variant B, see [docs/ANTIDOOM.md](docs/ANTIDOOM.md#reproduce): clone Antidoom into `antidoom/upstream`,
mine pairs against the server above, train on both cards with `antidoom/run_train.sh`, then serve the merged folder
with `MODEL=` pointing at it.

## Layout

| path | what |
|---|---|
| `plugin/` | `gptoss031` vLLM plugin (tool-call grammar, effort map + cap, HarmonyError guard) and its offline `selftest.py` |
| `serve/serve-gpt-oss.sh` | launcher: venv, card, plugin, effort cap and compile cache as variables |
| `serve/gpt-oss.extra.example` | the flags that turn XPU graphs and the fp8 KV cache on (copy to `gpt-oss.extra`) |
| `kernels/` | optional rebuild of the XPU attention library: fp8 → bf16 conversion in 5 instructions, gpt-oss kernel variants only |
| `bench/bench_kv.py` | KV pool, decode speed at 1/4/16 streams, a long read, and the 16 × 69k "all come back at once" test |
| `bench/measure_loops.py` | loop rate on 200 held-out Antidoom prompts, with Antidoom's own detector |
| `bench/grade_eval.py` | graded accuracy on 700 public test items |
| `bench/tool_flow.py` | streaming multi-turn tool-call sessions with simulated tool results; counts format faults |
| `bench/ptc_probe.py` | one `run_code` call per task, with awkward strings inside the Python; checks JSON and syntax |
| `bench/chat_check.py` | no-tool chat check for empty answers and commentary-channel leaks (round 3 gate) |
| `antidoom/` | XPU trainer wrapper, MXFP4 merge, miner wrapper, prompt builders, pair filters, variant B config |
| `results/` | the headline numbers as CSV/JSON (no prompts, no generations) |

## Credits

- **[Liquid AI][antidoom]** for Antidoom (Apache-2.0): the loop detector, pair mining, the FTPO trainer and the
  [antidoom-mix-v1.0](https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0) prompt set. We wrap it and don't
  copy it in. Antidoom builds on [Antislop](https://arxiv.org/abs/2510.15061) by Sam Paech et al.
  ([auto-antislop](https://github.com/sam-paech/auto-antislop)).
- **[OpenAI](https://huggingface.co/openai/gpt-oss-20b)** for gpt-oss-20b (Apache-2.0) and the
  [harmony](https://github.com/openai/harmony) response format.
- **[vLLM](https://github.com/vllm-project/vllm)** (Apache-2.0). `plugin/gptoss031/tags.py` is adapted from vLLM's
  harmony structural tag (`vllm/parser/harmony.py`, PRs #45560 and #52222).
- **[riceharvest/b70-qwen38-cookbook](https://github.com/riceharvest/b70-qwen38-cookbook)** (MIT): the fast fp8
  conversion, scale fold and zero-sum guard that `kernels/` builds on (his patches are included unchanged under his
  license).
- **[Vcvzgbzz/b70-cookbook](https://github.com/Vcvzgbzz/b70-cookbook)** (MIT): the vLLM 0.31 XPU build line our venv
  follows. More B70 credits are on the [Qwen page](../qwen38-709l/README.md#credits).
- Eval sets: [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500),
  [MMLU](https://huggingface.co/datasets/cais/mmlu), [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa).
- **Claude** (Anthropic; Claude Opus 5.5 and Claude Fable 5.1, through Claude Code): ported the trainer and the
  plugin, ran the mining, training and every measurement, and wrote this page.

## License

Our code and docs: MIT ([LICENSE](../LICENSE)). `plugin/gptoss031/tags.py` is adapted from vLLM (Apache-2.0).
Antidoom is not included: `antidoom/` expects your own clone of it, which keeps its Apache-2.0 license. gpt-oss-20b
and any merge of it fall under its Apache-2.0 license.

[antidoom]: https://github.com/Liquid4All/antidoom
