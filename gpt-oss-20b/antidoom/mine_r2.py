"""Round 2 miner: Antidoom `generate` against an OpenAI-compatible endpoint, with reasoning_effort and per-prompt
tools (no upstream edits).

Why a wrapper: upstream's VllmClient sends neither reasoning_effort nor tools, and prompts.render_chat_context
renders the pair context without them, so a pair mined at high effort or inside a tool-call prompt would be
trained on a context the server never saw. Checked: the HF gpt-oss chat template called with tools=... and
reasoning_effort=... reproduces vLLM's prompt_token_ids token for token (169/169 on a tools request with
return_token_ids), so patching both sides keeps the two identical.

What it patches (all in memory, at import):
  prompts._record_from_row   - the record's messages become a TaggedMessages list carrying the row's "tools"
  generate.render_chat_context - apply_chat_template(messages, tools=..., reasoning_effort=EFFORT, ...)
  client.VllmClient.__init__ - session.post adds reasoning_effort (+ tools, tool_choice auto) to the chat body

Input rows (OpenAI-style "conversations": [{"role","content"}] or ShareGPT) may carry "tools": [...] (OpenAI tool
specs) and "source" (-> source_task in the pairs). The effort MUST be what the server really runs: the gptoss031
plugin caps effort (GPTOSS_EFFORT_CAP), so mine at high only on a server started with GPTOSS_EFFORT_CAP=high.

  BASE_URL       server root, default http://localhost:8000 (passed to antidoom as --api-base-url <BASE_URL>/v1)
  OPENAI_API_KEY optional bearer key (antidoom gets "EMPTY" when unset)

Usage (extra args go to `antidoom generate`):
  python mine_r2.py --effort high --model gpt-oss -c configs/r2_chat.yaml \
      -r runs/mine2-chat --threads 16 --target-pairs 800
  python mine_r2.py --effort high --model gpt-oss -c configs/r2_tools.yaml \
      -r runs/mine2-tools --threads 16 --target-pairs 400
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "upstream" / "src"))  # git clone of Liquid4All/antidoom, unless it is pip-installed
EFFORTS = ("low", "medium", "high")


class TaggedMessages(list):
    """A plain list of messages (JSON-serialises as one) that also remembers the prompt's tools."""

    tools: list | None = None


def install(effort: str) -> None:
    import antidoom.client as client
    import antidoom.generate as generate
    import antidoom.prompts as prompts

    orig_record = prompts._record_from_row

    @functools.wraps(orig_record)
    def record_from_row(prompt_id, row, cfg):
        rec = orig_record(prompt_id, row, cfg)
        msgs = TaggedMessages(rec.messages)
        tools = row.get("tools")
        msgs.tools = list(tools) if isinstance(tools, list) and tools else None
        return dataclasses.replace(rec, messages=msgs)

    prompts._record_from_row = record_from_row

    def render_chat_context(messages, tokenizer):
        tools = getattr(messages, "tools", None)
        return tokenizer.apply_chat_template(
            list(messages), tools=tools, reasoning_effort=effort, tokenize=False, add_generation_prompt=True
        )

    # generate.py imported the name, so both bindings are replaced.
    generate.render_chat_context = render_chat_context
    prompts.render_chat_context = render_chat_context

    orig_init = client.VllmClient.__init__

    @functools.wraps(orig_init)
    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        orig_post = self.session.post

        def post(url, *pargs, json=None, **pkwargs):
            if isinstance(json, dict) and "messages" in json:
                json["reasoning_effort"] = effort
                tools = getattr(json["messages"], "tools", None)
                if tools:
                    json["tools"] = tools
                    json["tool_choice"] = "auto"
            return orig_post(url, *pargs, json=json, **pkwargs)

        self.session.post = post

    client.VllmClient.__init__ = init


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--effort", required=True, choices=EFFORTS, help="reasoning_effort sent AND rendered")
    ap.add_argument("--model", required=True, help="served model name, e.g. gpt-oss")
    ap.add_argument("-c", "--config", required=True, help="antidoom generation config yaml")
    a, rest = ap.parse_known_args()

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    base = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/").removesuffix("/v1") + "/v1"
    key = os.environ.get("OPENAI_API_KEY") or "EMPTY"

    install(a.effort)
    from antidoom.cli import main as cli_main

    sys.argv = ["antidoom", "-c", a.config, "generate", "--model-name", a.model, "--api-base-url", base,
                "--api-key", key, *rest]
    print(f"mine_r2: effort={a.effort} model={a.model} base={base} config={a.config} args={rest}", flush=True)
    cli_main()


if __name__ == "__main__":
    main()
