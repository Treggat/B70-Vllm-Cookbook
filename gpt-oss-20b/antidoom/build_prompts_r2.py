"""Build the round 2 input files for mine_r2.py from the Antidoom mix.

Two files, both FRESH: every id used by a held-out eval run (../bench measure_loops outputs listed in HELDOUT_RUNS)
or by round 1 mining (data/prompts-gptoss.jsonl) is excluded. Edit HELDOUT_RUNS to point at your own eval outputs.

  data/prompts-chat-r2.jsonl   first human turn of the 7 loopy sources (as round 1), OpenAI-style messages.
  data/prompts-tools-r2.jsonl  coding tasks (evol-codealpaca-v1, apps) wrapped as a tool-calling agent prompt:
      half "ptc"   = programmatic tool calling: system prompt + run_code tool, as in ../bench/ptc_probe.py
      half "agent" = OpenCode-style tools: system prompt + read/write/edit/bash/grep, as in ../bench/tool_flow.py
    The probes own tasks are NOT used, so ptc_probe.py and tool_flow.py stay clean tests.

  python build_prompts_r2.py            # writes both files
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data" / "antidoom-mix-v1.0.jsonl"
HELDOUT_RUNS = ["runs/gptoss-200.jsonl"]  # measure_loops.py outputs whose prompt ids must stay out of training
ROUND1_PROMPTS = ["data/prompts-gptoss.jsonl"]

LOOPY = [
    "Post-training-Data-Flywheel/AutoIF-instruct-61k",
    "mlabonne/lmsys-arena-human-preference-55k-sharegpt",
    "mlabonne/ultrachat_200k_sft",
    "codeparrot/apps",
    "LiquidAI/ifstruct_generated",
    "allenai/math_qa",
    "theblackcat102/evol-codealpaca-v1",
]
CODING = ["theblackcat102/evol-codealpaca-v1", "codeparrot/apps"]

# --- toolsets, verbatim from ../bench/ptc_probe.py and ../bench/tool_flow.py ---
PTC_SYSTEM = """You are a coding agent. `run_code` is the ONLY tool you can call. Its `code` argument is the BODY of an
async PYTHON function (top-level await and return work). Inside it, call tools as `await tools.name({...})` with a
plain dict. Available: tools.bash({"command": str}), tools.read({"file_path": str}), tools.write({"file_path": str,
"content": str}), tools.edit({"file_path": str, "old_string": str, "new_string": str}), tools.grep({"pattern": str,
"path": str}). Print what you want to see with print(). Do exactly what the user asks in ONE run_code call. Do not
explain, do not ask questions."""
PTC_TOOLS = [{"type": "function", "function": {"name": "run_code", "description": "Run the body of an async Python function with `tools` bound.",
              "parameters": {"type": "object", "additionalProperties": False, "required": ["code", "description"],
                             "properties": {"code": {"type": "string", "description": "Body of an async Python function."},
                                            "description": {"type": "string", "description": "Short summary."}}}}}]


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "strict": False,
            "parameters": {"type": "object", "properties": props, "required": req}}}


_S = {"type": "string"}
AGENT_TOOLS = [_fn("read", "Read a file.", {"filePath": _S}, ["filePath"]),
               _fn("write", "Write a whole file.", {"filePath": _S, "content": _S}, ["filePath", "content"]),
               _fn("edit", "Replace exact text in a file.", {"filePath": _S, "oldString": _S, "newString": _S}, ["filePath", "oldString", "newString"]),
               _fn("bash", "Run a shell command.", {"command": _S, "description": _S}, ["command", "description"]),
               _fn("grep", "Search files with a regex.", {"pattern": _S, "path": _S}, ["pattern"])]
AGENT_SYSTEM = ("You are a coding agent working in /work. Use the tools to do the task step by step: look first, then write, "
                "then run the tests with bash, then fix anything that fails. When the task is done, reply with one short "
                "sentence and no tool call.")
# Dataset coding prompts read "write a function that ..."; the suffix turns them into the file-and-run task an agent gets.
AGENT_SUFFIX = "\n\nPut the solution in a file under /work with a short test next to it, then run the test."
PTC_SUFFIX = "\n\nWrite the solution to a file under /tmp and print the file back."


def first_human(conv) -> str | None:
    try:
        turns = ast.literal_eval(conv) if isinstance(conv, str) else conv
    except Exception:
        return None
    for t in turns:
        if t.get("from") in ("human", "user"):
            return t.get("value")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-per-source", type=int, default=2500)
    ap.add_argument("--tools-total", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()

    used: set[str] = set()
    for run in HELDOUT_RUNS:
        p = HERE / run
        if p.exists():
            used |= {json.loads(l)["id"] for l in open(p, encoding="utf-8")}
    n_heldout = len(used)
    for pf in ROUND1_PROMPTS:
        p = HERE / pf
        if p.exists():
            used |= {json.loads(l)["id"] for l in open(p, encoding="utf-8")}
    print(f"excluding {n_heldout} held-out ids and {len(used) - n_heldout} round 1 mining ids")

    by_src: dict[str, list[dict]] = collections.defaultdict(list)
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] in LOOPY and r["id"] not in used:
                by_src[r["source"]].append(r)

    rng = random.Random(a.seed)
    for rows in by_src.values():
        rng.shuffle(rows)

    # ---- tool prompts first (they take from the coding sources), then chat from what is left ----
    tools_rows: list[dict] = []
    per_coding = a.tools_total // len(CODING)
    taken: set[str] = set()
    for src in CODING:
        n = 0
        for r in by_src[src]:
            p = first_human(r["conversations"])
            if not p or len(p) > 3000:
                continue
            kind = "ptc" if n % 2 == 0 else "agent"
            if kind == "ptc":
                msgs = [{"role": "system", "content": PTC_SYSTEM}, {"role": "user", "content": p.rstrip() + PTC_SUFFIX}]
                tools = PTC_TOOLS
            else:
                msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": p.rstrip() + AGENT_SUFFIX}]
                tools = AGENT_TOOLS
            # source_task is read first by antidoom's _source_metadata, so the pairs say tools/<kind>;
            # source_dataset keeps the coding dataset the task came from.
            tools_rows.append({"id": r["id"], "source": f"tools/{kind}", "source_task": f"tools/{kind}",
                               "source_dataset": src, "conversations": msgs, "tools": tools})
            taken.add(r["id"])
            n += 1
            if n >= per_coding:
                break
    ptc = [r for r in tools_rows if r["source"] == "tools/ptc"]
    agent = [r for r in tools_rows if r["source"] == "tools/agent"]
    tools_out = []
    for i in range(max(len(ptc), len(agent))):
        if i < len(ptc):
            tools_out.append(ptc[i])
        if i < len(agent):
            tools_out.append(agent[i])

    picked: dict[str, list[dict]] = {}
    for src in LOOPY:
        keep = []
        for r in by_src.get(src, []):
            if r["id"] in taken:
                continue
            p = first_human(r["conversations"])
            if p and len(p) < 8000:
                keep.append({"id": r["id"], "source": src,
                             "conversations": [{"role": "user", "content": p}]})
            if len(keep) >= a.chat_per_source:
                break
        picked[src] = keep
    chat_out = []
    for i in range(max(len(v) for v in picked.values())):
        for src in LOOPY:
            if i < len(picked.get(src, [])):
                chat_out.append(picked[src][i])

    for name, rows in (("prompts-chat-r2.jsonl", chat_out), ("prompts-tools-r2.jsonl", tools_out)):
        with open(HERE / "data" / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows to data/{name}")
    for src in LOOPY:
        print(f"  chat  {src:55s} {len(picked.get(src, []))}")
    print(f"  tools ptc {len(ptc)}  agent {len(agent)}")


if __name__ == "__main__":
    main()
