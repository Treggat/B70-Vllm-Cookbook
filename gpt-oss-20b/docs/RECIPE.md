# Recipe: serving gpt-oss-20b on one B70

## 0. Box

- One Intel Arc Pro B70 (BMG-G31, 32 GB) serves the model. gpt-oss-20b in its native MXFP4 checkpoint is about
  13.8 GB on disk. With `--gpu-memory-utilization 0.9` vLLM gave us a KV pool of 643,585 tokens, shared by up to 16
  streams (about five full 131k contexts at once).
- Our machine has two B70s. We pinned the server to one with `ZE_AFFINITY_MASK` (`CARD=` in the launcher). Training
  ([ANTIDOOM.md](ANTIDOOM.md)) needs both.
- Linux with the `xe` driver. As on the [Qwen page](../../qwen38-709l/docs/RECIPE.md), we ran vLLM from a venv on a
  mounted volume inside a container that already had the Level Zero / compute runtime.

## 1. Build vLLM 0.31.0

Use [`../qwen38-709l/scripts/build_venv.sh`](../../qwen38-709l/scripts/build_venv.sh). It builds an editable source
install of v0.31.0 for XPU with torch 2.14.0+xpu, triton 3.8.0+xpu and vllm-xpu-kernels 0.1.15.4, which is the stack
we served gpt-oss on. None of the Qwen patch stack is needed here.

Our own gpt-oss venv was built from the same tag and pins by a slightly different script: it ran
`pip install -r requirements/xpu.txt` and then `pip install --no-build-isolation -e .`. Whichever way you build it,
check two things afterwards, because the tool-call path depends on them:

```sh
$VENV/venv/bin/python -m pip list | grep -i -E '^(vllm|torch|triton|xgrammar|oss-harmony|openai-harmony) '
# want: oss-harmony >= 0.0.11 (v0.31.0's requirements/xpu.txt pins it), xgrammar (we had 0.2.7),
#       and NO openai-harmony: both install a module called openai_harmony (vLLM #55128)
$VENV/venv/bin/python -c "import openai_harmony, xgrammar, vllm; print(vllm.__version__)"
```

The launcher also runs our
[`patch_meminfo_fallback.py`](../../qwen38-709l/patches/patch_meminfo_fallback.py) on the tree (idempotent). Under
some Level Zero drivers the kernels' memory query reads 0 GiB free and vLLM refuses to start.

## 2. The plugin (`plugin/`)

`gptoss031` is a vLLM "general plugin": a Python package plus a `dist-info` folder whose `entry_points.txt` registers
it under `vllm.general_plugins`. vLLM loads it in the API server and the engine core when the folder is on
`PYTHONPATH`, and only then. Nothing in site-packages changes, so other models served from the same venv are not
affected. It patches three things in memory.

### Tool-call grammar (`GPTOSS_TOOL_STRICT`, default `function`)

gpt-oss writes tool calls in harmony format (`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}<|call|>`).
Without a grammar, a small sampling slip produces a broken call. With vLLM 0.31's own grammar, a different problem
appears. Measured on 0.31 with `bench/tool_flow.py`, 16 streaming multi-turn sessions:

| grammar | tool calls | problems |
|---|---|---|
| none | 167 | 4 streams killed by `HarmonyError`, 7 tool names leaked or invalid |
| vLLM 0.31 upstream (`--tool-strict-level function`) | 75 | 1 HTTP 500: xgrammar "grammar rejected tokens [200012]" (`<|call|>`) |
| **ours** (`gptoss031`, upstream's off with `--tool-strict-level auto`) | 168 | **0** |

The difference: upstream ends a tool call on `["<|end|>", ""]`, ours on `""` (the stop token) only. With `<|end|>`
allowed there, the grammar is complete after `<|end|>` and only the EOS `<|return|>` is legal next, so the model
writes `<|end|><|return|>` and vLLM's harmony parser streams the stray `<|return|>` as message text. In an earlier
measurement 7 of 24 calls ended that way and all 7 leaked; 0 of 17 that ended on the bare stop token leaked.

The grammar (`tags.py`) also accepts what gpt-oss actually emits: the recipient in the role header or in the channel
header, ` json` and ` <|constrain|>json`, and calls on the commentary, analysis or final channel. Levels:

- `function` (default): pins the envelope and the tool name; the arguments must be a JSON object. A tool the client
  marks `strict: true` gets its full schema anyway.
- `parameter`: every tool's arguments must match its schema.
- `off`: the plugin adds no grammar.

The grammar is added only to chat requests that carry tools and a `tool_choice` other than `none`. `auto` still lets
the model answer in plain text; `required` or a named tool forces a call.

### Effort mapping and cap (`GPTOSS_EFFORT_CAP`, launcher default `medium`)

- Always: `none`/`minimal` → `low`, `xhigh`/`max` → `high`. vLLM 0.31's harmony path answers HTTP 400 for anything
  but low/medium/high, and some clients send the other names.
- Cap: a request for more than the cap is served at the cap. The plugin's own default is `high` (no cap); the
  launcher sets `medium`.

Why medium. At high, gpt-oss-20b reasons until it runs out of tokens:

- stock, PTC probe at 3,000 tokens: 8/36 at high (14/36 even at 12,000 tokens) vs 28/36 capped to medium
  (2026-10-14). A rerun on 2026-10-17 gave 6/36 at high, median 41 s per call.
- stock, loop set: 79/200 loops at high vs 17/200 at medium.
- variant B fixes the loops at high (2/200) but still reasons long: PTC 16/36 at high vs 27/36 at medium, and on
  graded accuracy the round 1 model scored 78.9% at high with 32,000 tokens vs 83.3% at medium, 4× slower.

So the cap stays at medium for agent work. For chat-only use, B at high is usable (1% loops, 11% hit the 4,000-token
cap, vs 39.5% and 45.5% for stock).

### HarmonyError guard (`GPTOSS_HARMONY_GUARD`, default on)

When the streaming harmony parser meets a token it can't place (for example "unexpected tokens remaining in message
header"), it raises and the client gets an HTTP 500 in the middle of the stream. The guard skips that token instead.
vLLM main has a similar guard in `process_chunk`; 0.31.0 does not. Without a grammar, 4 streams died this way in the
test above; with the grammar on, the guard is a safety net for what the grammar doesn't cover.

### Self-test

`plugin/selftest.py` runs 31 offline tests with the venv's vllm, xgrammar and openai_harmony and the gpt-oss tokenizer
(no GPU, no model load): the grammar token by token as vLLM's xgrammar backend builds it, the hook points, effort
mapping, the guard, and that the entry point is found. Run it after every vLLM upgrade.

## 3. Serve (`serve/`)

`serve/serve-gpt-oss.sh <port>` with `VENV` and `MODEL` set. The flags and why:

- `--tool-call-parser openai --reasoning-parser openai_gptoss --enable-auto-tool-choice`: harmony tool calls and
  reasoning come back as OpenAI `tool_calls` and `reasoning_content`.
- `--tool-strict-level auto`: keeps vLLM's own harmony grammar off for clients that send `strict: false`, so the
  plugin's grammar is the one used.
- `--structured-outputs-config '{"backend":"xgrammar"}'`: the plugin's structural tag is an xgrammar format.
- `--max-num-seqs 16 --max-model-len 131072 --gpu-memory-utilization 0.9`.
- `--enable-prefix-caching`: agent sessions resend long prefixes.
- `--enforce-eager` in the base line, then `serve/gpt-oss.extra` overrides it with
  `--no-enforce-eager --compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",...}`. With that file, one
  stream does 106–117 tok/s; eager does about 43 (2.7× slower). The launcher keeps the eager line so a missing or
  empty `.extra` still starts. Graph capture took about 1 s.
- `ZE_AFFINITY_MASK=$CARD` and `unset ONEAPI_DEVICE_SELECTOR`: with torch 2.13+ the card is chosen the Level Zero way.
- `VLLM_CACHE_ROOT` per `.extra` content (md5 of the file): the compile cache is keyed on flags, so a changed flag
  must not reuse a stale graph.
- The launcher `cd`s into the venv dir. If the current directory holds a folder called `vllm` (some launchers start
  in one), `python -m vllm...` imports that instead of the install.
- `exec` at the end, so a launcher's SIGTERM reaches vLLM.

Speed, 2026-10-14, stock weights, short prompts, graphs on: 1 stream 106–117 tok/s, 4 streams 257 summed, 16 streams
698 summed. The same model on the older vLLM 0.26 image did 109–119 / 261 / 702, so 0.31 is on par. First load takes
about 2.5–3 minutes on our box (a cold reload measured 144–169 s); later requests are warm.

If a launcher or proxy sits in front of the server, keep its concurrency limit at or above `--max-num-seqs` so it
doesn't queue requests vLLM could take.

## 4. Verify

```sh
python bench/tool_flow.py http://localhost:8000 gpt-oss --sessions 16 --turns 8 --extra '{"reasoning_effort":"medium"}'
#   want: bad_name 0, ctrl_leak 0, bad_json 0, http_err 0, bang 0
python bench/ptc_probe.py http://localhost:8000 gpt-oss --rounds 3 --extra '{"reasoning_effort":"medium"}'
#   our 3-round runs at medium: stock 28-32/36, B 27-29/36; failures are mostly Python syntax inside the code string
```

`bang` counts replies with more than 50 `!`. A reply of only `!!!!` is what a NaN looks like from the outside (token
id 0). We saw it once on the older 0.26 image, about an hour into a 16-stream run (see
[PITFALLS.md](PITFALLS.md#serving)). Two 4-hour 16-stream soaks on 0.31 (stock, then B) had none.
