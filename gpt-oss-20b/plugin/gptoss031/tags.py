"""Harmony structural tags for gpt-oss: a whole-output grammar covering reasoning, preamble, tool call
and final answer. Port of vLLM v0.30.0 vllm/parser/harmony.py:390-505 (upstream #45560 + #52222), adapted for vLLM 0.31:
0.31 upstream still ends tool calls on ["<|end|>", ""]; this grammar ends them on "" (the stop token) only.

Depends on xgrammar only, so it can be tested without vLLM.

SPDX-License-Identifier: Apache-2.0
Derived from vLLM (https://github.com/vllm-project/vllm), Copyright the vLLM project contributors, licensed under
the Apache License 2.0. Modified for this cookbook: the tool-call end tag is the stop token only, and the grammar is
packaged as a standalone vLLM plugin.
"""
from __future__ import annotations

from typing import Any

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    SequenceFormat,
    TagFormat,
)

LEVELS = ("off", "function", "parameter")

# The model writes " <|constrain|>json"; openai-harmony 0.0.8 re-renders history as " json". Both occur.
_JSON_CONSTRAINS = [" json", " <|constrain|>json"]
# "" stands for harmony's stop tokens (<|return|>, <|call|>): xgrammar does not let a grammar name them.
_END_TAG = ["<|end|>", ""]
# A tool call ends on the stop token only. With _END_TAG here, after <|end|> the grammar was complete and
# xgrammar allowed only the EOS <|return|>, so the model wrote "<|end|><|return|>" and vLLM's harmony parser
# streamed the stray <|return|> as message text (a "<|return|>" leak seen in agent sub-sessions).
# Measured: 7 of 24 calls ended <|end|><|return|> and all 7 leaked; 0 of 17 bare-<|return|> endings leaked. The final answer keeps _END_TAG: xgrammar needs a non-empty end for unbounded text.
_CALL_END = [""]
_FINAL_BEGIN = "<|channel|>final{constrain}<|message|>"
_TOOL_CALL_CHANNELS = ["<|channel|>commentary", "<|channel|>analysis", "<|channel|>final"]
# Recipient in the role header, or in the channel header. gpt-oss emits both.
_FUNCTION_CALL_BEGINS = [
    " to=functions.{name}{channel}{constrain}<|message|>",
    "{channel} to=functions.{name}{constrain}<|message|>",
]
_ANY_OBJECT: dict[str, Any] = {"type": "object"}


def _body_schema(tool: dict[str, Any], level: str) -> dict[str, Any]:
    """`function` pins only the envelope and name; `parameter` pins the tool's schema too.
    A tool the client marked strict always gets its schema."""
    if level == "parameter" or tool.get("strict") is True:
        return tool.get("parameters") or _ANY_OBJECT
    return _ANY_OBJECT


def _optional_message(channel: str) -> OptionalFormat:
    return OptionalFormat(
        content=SequenceFormat(
            elements=[
                TagFormat(begin=f"<|channel|>{channel}<|message|>", content=AnyTextFormat(), end="<|end|>"),
                ConstStringFormat(value="<|start|>assistant"),
            ]
        )
    )


def _assemble(allow_commentary: bool, content: Any) -> StructuralTag:
    elements: list[Any] = [_optional_message("analysis")]
    if allow_commentary:
        elements.append(_optional_message("commentary"))
    elements.append(content)
    return StructuralTag(format=SequenceFormat(elements=elements))


def build_tool_call_tag(tools: list[dict[str, Any]], tool_choice: str = "auto", level: str = "function") -> StructuralTag:
    """tools: [{"name", "parameters", "strict"}]. tool_choice: "auto" also allows a final answer;
    anything else ("required", "forced") demands a call."""
    if level not in ("function", "parameter"):
        raise ValueError(f"level must be function or parameter, got {level!r}")
    if not tools:
        raise ValueError("no tools to build a tag for")

    tags = [
        TagFormat(
            begin=pattern.format(name=tool["name"], channel=channel, constrain=constrain),
            content=JSONSchemaFormat(json_schema=_body_schema(tool, level)),
            end=_CALL_END,
        )
        for tool in tools
        for pattern in _FUNCTION_CALL_BEGINS
        for constrain in _JSON_CONSTRAINS
        for channel in _TOOL_CALL_CHANNELS
    ]
    if tool_choice == "auto":
        tags.extend(
            TagFormat(begin=_FINAL_BEGIN.format(constrain=constrain), content=AnyTextFormat(), end=_END_TAG)
            for constrain in _JSON_CONSTRAINS + [""]
        )
    return _assemble(allow_commentary=True, content=OrFormat(elements=tags))


def build_final_tag(content: Any) -> StructuralTag:
    """A constrained final answer (response_format and friends) with reasoning still allowed before it."""
    constrains = _JSON_CONSTRAINS + [""] if isinstance(content, JSONSchemaFormat) else [""]
    finals = [
        TagFormat(begin=_FINAL_BEGIN.format(constrain=constrain), content=content, end=_END_TAG)
        for constrain in constrains
    ]
    return _assemble(allow_commentary=False, content=OrFormat(elements=finals) if len(finals) > 1 else finals[0])
