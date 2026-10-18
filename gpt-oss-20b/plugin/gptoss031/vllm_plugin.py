"""vLLM 0.31 general plugin for gpt-oss: harmony tool-call grammar, effort mapping and cap, HarmonyError guard.

Loaded only by a server whose launcher puts this folder on PYTHONPATH (the entry point is in
gptoss031-0.1.0.dist-info); the patches live in memory and nothing in site-packages changes. Run vLLM with
--tool-strict-level auto so upstream's own harmony grammar stays out of the way (clients that send strict:false get
no upstream tag under auto); this plugin supplies the grammar instead.

Why not upstream's grammar: measured 2026-10-14 on 0.31 (16 streaming multi-turn sessions, bench/tool_flow.py):
upstream --tool-strict-level function = 1 HTTP 500 in 75 calls (xgrammar "grammar rejected tokens [200012]" =
<|call|>); no grammar = 4 HarmonyError stream deaths + 7 leaked/invalid tool names in 167 calls; this grammar =
0 problems in 168 calls. Difference: upstream ends a tool call on ["<|end|>", ""], ours on the stop token only
(tags.py).

GPTOSS_TOOL_STRICT = off | function | parameter   (default function)
GPTOSS_EFFORT_CAP  = low | medium | high          (default high = no cap)
  Always: none/minimal -> low, xhigh/max -> high (0.31 harmony answers 400 for anything but low/medium/high).
  Cap: measured on 0.31 the same as 0.26: at high gpt-oss-20b reasons until max_tokens and never calls the tool
  (PTC probe 8/36 at 3K tokens, 14/36 at 12K) vs 28/36 capped to medium.
GPTOSS_HARMONY_GUARD = 1 | 0   (default 1)
  A token the streaming harmony parser rejects (HarmonyError, e.g. "unexpected tokens remaining in message
  header") is skipped instead of killing the stream with HTTP 500. Same behaviour as vLLM main's process_chunk
  guard (not in 0.31.0).
"""
from __future__ import annotations

import functools
import json
import os
from dataclasses import replace
from typing import Any

from gptoss031.tags import LEVELS, build_final_tag, build_tool_call_tag

LEVEL_ENV, CAP_ENV, GUARD_ENV = "GPTOSS_TOOL_STRICT", "GPTOSS_EFFORT_CAP", "GPTOSS_HARMONY_GUARD"
EFFORTS = ("low", "medium", "high")
_EFFORT_MAP = {"none": "low", "minimal": "low", "xhigh": "high", "max": "high"}

_installed: dict[str, Any] = {}


def strict_level() -> str:
    level = os.environ.get(LEVEL_ENV, "function").strip().lower()
    if level not in LEVELS:
        raise ValueError(f"{LEVEL_ENV}={level!r} is not one of {', '.join(LEVELS)}")
    return level


def configured_effort_cap() -> str:
    cap = os.environ.get(CAP_ENV, "high").strip().lower()
    if cap not in EFFORTS:
        raise ValueError(f"{CAP_ENV}={cap!r} is not one of {', '.join(EFFORTS)}")
    return cap


def guard_enabled() -> bool:
    return os.environ.get(GUARD_ENV, "1").strip() != "0"


def map_effort(effort: str | None, cap: str = "high") -> str | None:
    effort = _EFFORT_MAP.get(effort, effort)
    if effort in EFFORTS and EFFORTS.index(effort) > EFFORTS.index(cap):
        return cap
    return effort


def _final_content(params: Any) -> Any:
    """What a constrained final answer must look like (port of v0.30 _params_to_final_content)."""
    from xgrammar import StructuralTag
    from xgrammar.structural_tag import (
        ConstStringFormat, GrammarFormat, JSONSchemaFormat, OrFormat, RegexFormat, TagFormat, TriggeredTagsFormat,
    )

    if params.json_object:
        return JSONSchemaFormat(json_schema={"type": "object"})
    if params.json is not None:
        schema = json.loads(params.json) if isinstance(params.json, str) else params.json
        return JSONSchemaFormat(json_schema=schema)
    if params.regex is not None:
        return RegexFormat(pattern=params.regex)
    if params.choice is not None:
        return OrFormat(elements=[ConstStringFormat(value=choice) for choice in params.choice])
    if params.grammar is not None:
        return GrammarFormat(grammar=params.grammar)
    if params.structural_tag is not None:
        s_tag = json.loads(params.structural_tag)
        if "structures" in s_tag:
            return TriggeredTagsFormat(
                triggers=s_tag["triggers"],
                tags=[TagFormat(begin=s["begin"], content=JSONSchemaFormat(json_schema=s["schema"]), end=s["end"])
                      for s in s_tag["structures"]],
            )
        return StructuralTag.model_validate(s_tag).format
    return None


def constrain_request(request: Any, level: str) -> Any:
    """Give a chat request the whole-output harmony grammar for its tools. Mutates and returns it.
    Without tools (or tool_choice none) the request is left as upstream adjusted it."""
    from vllm.sampling_params import StructuredOutputsParams

    if level == "off":
        return request
    choice = request.tool_choice
    if not request.tools or choice in (None, "none"):
        return request
    named = getattr(getattr(choice, "function", None), "name", None)
    offered = [
        {"name": t.function.name, "parameters": t.function.parameters, "strict": t.function.strict}
        for t in request.tools
        if getattr(t, "function", None) is not None and (named is None or t.function.name == named)
    ]
    if not offered:
        return request
    tag = build_tool_call_tag(offered, tool_choice="auto" if choice == "auto" else "required", level=level)
    request.structured_outputs = StructuredOutputsParams(structural_tag=json.dumps(tag.model_dump()))
    request.response_format = None
    return request


def install(level: str | None = None, effort_cap: str | None = None, guard: bool | None = None,
            renderer: Any = None, parser_cls: Any = None) -> list[str]:
    """Patch vLLM in this process. Safe to call again. `renderer` / `parser_cls` are for the self-test."""
    level = strict_level() if level is None else level
    cap = configured_effort_cap() if effort_cap is None else effort_cap
    guard = guard_enabled() if guard is None else guard
    if level not in LEVELS:
        raise ValueError(f"level {level!r} is not one of {', '.join(LEVELS)}")
    if cap not in EFFORTS:
        raise ValueError(f"effort cap {cap!r} is not one of {', '.join(EFFORTS)}")
    if renderer is None:
        from vllm.renderers.online_renderer import OnlineRenderer as renderer
    if parser_cls is None and guard:
        from openai_harmony import StreamableParser as parser_cls
    settings = (level, cap, guard, renderer, parser_cls)
    if _installed.get("settings") == settings:
        return _installed["names"]
    uninstall()

    names, originals = [], []

    adjust = renderer.adjust_harmony_request

    @functools.wraps(adjust)
    def adjust_constrained(self, request, *args, **kwargs):
        result = adjust(self, request, *args, **kwargs)
        if hasattr(request, "tool_choice") and hasattr(request, "messages"):  # chat completions only
            constrain_request(request, level)
        return result

    renderer.adjust_harmony_request = adjust_constrained
    names.append(f"{renderer.__name__}.adjust_harmony_request")
    originals.append((renderer, "adjust_harmony_request", adjust))

    render = renderer._make_request_with_harmony

    @functools.wraps(render)
    def render_mapped(self, request, *args, **kwargs):
        request.reasoning_effort = map_effort(request.reasoning_effort, cap)
        return render(self, request, *args, **kwargs)

    renderer._make_request_with_harmony = render_mapped
    names.append(f"{renderer.__name__}._make_request_with_harmony")
    originals.append((renderer, "_make_request_with_harmony", render))

    if guard:
        from openai_harmony import HarmonyError

        process = parser_cls.process

        @functools.wraps(process)
        def process_guarded(self, token):
            try:
                return process(self, token)
            except HarmonyError:
                _installed["guarded"] = _installed.get("guarded", 0) + 1
                return self

        parser_cls.process = process_guarded
        names.append(f"{parser_cls.__name__}.process (HarmonyError guard)")
        originals.append((parser_cls, "process", process))

    _installed.update(settings=settings, names=names, originals=originals)
    return names


def uninstall() -> None:
    for owner, name, original in _installed.get("originals", []):
        setattr(owner, name, original)
    _installed.clear()


def register() -> None:
    """Entry point (vllm.general_plugins). Runs in the API server and in the engine core."""
    from vllm.logger import init_logger

    logger = init_logger("vllm.gptoss031")
    level, cap, guard = strict_level(), configured_effort_cap(), guard_enabled()
    names = install(level, cap, guard)
    logger.info("gptoss031: %s=%s, %s=%s, %s=%s, pid %s, patched: %s", LEVEL_ENV, level, CAP_ENV, cap,
                GUARD_ENV, int(guard), os.getpid(), ", ".join(names))
