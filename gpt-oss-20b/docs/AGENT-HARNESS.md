# gpt-oss-20b behind a coding agent: harness lessons

We run gpt-oss-20b as the worker model of our agent harness (DSH). The lessons below are about how gpt-oss reads the
messages a harness puts around its tool calls. They should apply to any harness that drives gpt-oss through an
OpenAI-compatible chat endpoint.

## The test

14 small coding tasks (write a module and its tests, fix a config file, a SQL string, a heredoc, a shell script...),
each run as its own agent session, 14 at once, effort medium, twice per model: 28 sessions per model. Each prompt was
about 30K tokens (the harness's system prompt and a large toolset). We graded on disk: files checked and test commands
re-run by us, not taken from the agent's report.

| 28 sessions each | stock | variant B |
|---|---|---|
| verified pass | 17 | **20** |
| partial (wrong folder, doubled backslashes, a placeholder test) | 8 | 5 |
| fabricated report (says it ran things it didn't) | 1 | 3 |
| honest fail | 1 | 0 |
| API error | 1 (a 58K-token request aborted) | 0 |
| `rm -rf` on folders shared with other sessions | **2** | 0 |
| requests to the server, first batch of 14 | 165 | 84 |

B did the same work in about half the turns. The fabrication difference (3 vs 1) is not significant (Fisher
p ≈ 0.6); all three of B's were in the first batch.

Two things in the table are about the harness, not the model:

- **Both models** wrote files into a doubled path (`<cwd>/workspace/tests/...` when the cwd already was
  `/workspace/tests`) and doubled backslashes in YAML/Markdown strings. That's a cwd and quoting quirk of the task
  setup.
- **Stock** read "work in a fresh folder" as "wipe the folder": one session ran `rm -rf` on the shared test tree,
  deleting every other session's work, and another wiped the doubled-path tree after copying its own files out. The
  one stock fabrication was in that batch: it reported "2 passed" with no successful test run in its transcript, and
  its tests fail on disk. A harness should deny or confirm `rm -rf` outside the task folder, whatever the model.

## Finding the cause of the fabrications

A fabricated session looked like this: one real tool call (a `mkdir`, or the harness's "publish artifact" tool), then
a final answer that writes the remaining steps as JSON or heredoc blocks and reports invented output, once tagged
"[verified: ran]". The folders were empty.

We rebuilt the exact request each fabricated session sent at the failing step (system prompt, tools, every message,
reasoning replayed as `reasoning_content`) and replayed it 10 times at temperature 1.0, counting answers that were not
a tool call. Variant B: 27/30 on the three fabricated contexts, stock 24/30; 0/30 for both on three passing contexts.
Then we removed parts of the request one at a time. Two triggers:

1. **A mid-turn user message.** After the first tool result, the harness inserted a "Context meter" status line as a
   **user** message (how much of the context window was in use). Dropping just that message turned the fabrication
   into a real tool call. A user message is gpt-oss's turn boundary: the OpenAI reasoning guide passes reasoning back
   "since the last user message". Our reading: the model takes the new user message as a new turn and answers it,
   summarising the job as if it were done. It only happened at temperature 1.0, the vLLM default when a client sends none, which our
   harness does.
2. **A tool result that sounds like completion.** In the third context the model published a "Result Summary"
   artifact before doing the work, and the tool replied `Published artifact "Result Summary" (markdown).` That reads
   like the end of the task, and the model wrote the final answer next.

## The fixes (harness side, no weight change)

1. **Meter text goes into the tool result.** The mid-turn status line is appended to the end of the preceding tool
   result instead of being sent as a user message. (At the start of a turn it stays a user message; there it is a turn
   boundary anyway.)
2. **Artifact tool result made non-final.** The reply now ends with: "Publishing does not create files or run
   commands: the task is not complete until the files exist and the command has run; continue with the next step."
   The harness's system prompt also gained: "Never publish a result summary artifact before the work it summarizes has
   actually run."

Replay on variant B, the same saved requests rewritten into the new wire form, 10 samples each, temperature 1.0,
medium:

| not a tool call | original | meter in tool result | + artifact reply reworded |
|---|---|---|---|
| context 1 (meter trigger) | 8/10 | 0/10 | 0/10 |
| context 2 (artifact trigger) | 10/10 | 10/10 | 0/10 |
| context 3 (meter trigger) | 9/10 | 2/10 | 2/10 |
| **total** | **27/30** | **12/30** | **2/30** |
| 3 passing contexts | 0/30 | 0/30 | 0/30 |

No answer or reasoning mentioned the meter text or echoed the new wording. After deploying both, a live session that
published a "Plan" artifact first got the new reply and then did every step.

For comparison, the best weight-only attempt (round 3, variant C2) got the old-form contexts to 1/30 but broke plain
chat. See [ANTIDOOM.md](ANTIDOOM.md#round-3-the-narrated-tool-call-what-failed-and-why).

## Grammar 500s on long agent contexts

With the plugin's grammar on, a small share of tool requests still end in HTTP 500: xgrammar's "grammar rejected
tokens [200012]" (`<|call|>`), "Failed to advance FSM", and vLLM terminates that request (the engine stays up). On our
mining traffic in agent-shaped contexts at temperature 1.0 it was 7–9 per 1,000 tool requests; a 4-hour chat-only soak
had none. Replaying the 12 logged failing requests 5 times each: stock 3/60, B 5/60, on different contexts. So it comes
from the traffic shape, not from the fine-tune. A harness should treat a 500 on a tool step as retryable.

## What others found

From a short survey of public reports on gpt-oss tool use (October 2026). Where a point is our inference, it says so.

- **Tool calls on the wrong channel or as text.** At temperature 0 without the standard harmony system block,
  gpt-oss-20b put 17/200 tool calls on `final` (`<|constrain|>functions.NAME`); with the block, 0–1/200
  ([tinker-cookbook #947](https://github.com/thinking-machines-lab/tinker-cookbook/issues/947)). Under a harmony
  structural tag the model can imitate the commentary frame inside `final` content
  ([vLLM #42031](https://github.com/vllm-project/vllm/pull/42031), open; proposes masking `final` when a call is
  required). Calls nested in preambles or on `analysis`: [vLLM #35907](https://github.com/vllm-project/vllm/pull/35907).
- **Where the tool definitions go.** Moving them from the developer to the system message raised call rates 3.4–18×
  in [one study](https://arxiv.org/html/2604.00362v1) (Table 5); their harness also retries on "no tool call and no
  final".
- **Replay the reasoning between tool calls.** OpenAI's
  [raw CoT guide](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot) keeps analysis while the last
  assistant message was a tool call. llama.cpp saw 1–4% malformed headers at temperature 1.0 until `reasoning_content`
  was sent back on tool-call messages ([#27720](https://github.com/ggml-org/llama.cpp/issues/27720)); vLLM used to drop
  it silently ([#42664](https://github.com/vllm-project/vllm/pull/42664), merged); SGLang had the same bug class
  ([#44931](https://github.com/sgl-project/sglang/issues/44931)).
- **An assistant message with content and no `tool_calls` renders as `final`** in vLLM's harmony path, which drops
  earlier analysis and marks the turn ended (read from vLLM's `harmony_utils.py`; the consequence is inference). Don't
  replay mid-turn assistant text that way.
- **Grammar and stop tokens.** "Unexpected token 200012" came from `<|call|>` missing from the stop ids
  ([vLLM #35076](https://github.com/vllm-project/vllm/pull/35076), merged); a stop token that is also a grammar
  terminal can never be accepted ([xgrammar #701](https://github.com/mlc-ai/xgrammar/issues/701)). A grammar that only
  accepts recipient-first headers scored 0% on BFCL multi-turn vs about 55% lenient
  ([vLLM #52222](https://github.com/vllm-project/vllm/pull/52222)). Our plugin already ends calls on the stop token and
  accepts both header orders.
- **Temperature.** gpt-oss recommends 1.0 / top_p 1.0. Lowering temperature did not consistently reduce tool failures
  in [one study that includes gpt-oss-20b](https://arxiv.org/html/2608.26199v1), and the tinker failures above were at
  0. We kept 1.0.
- **No public report** we found covers a mid-turn user message or a completion-sounding tool result with gpt-oss. The
  closest are OpenAI's guidance on preambles being treated as final answers and the GPT-4.1 guide's persistence
  reminders, both for OpenAI API models.
