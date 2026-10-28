"""vLLM 0.32 general plugin for Nemotron 3.5 Lightning: turn `reasoning_effort` into a thinking budget.

Nemotron's chat template only knows `enable_thinking`, and vLLM 0.32 already maps `reasoning_effort: none` to
`enable_thinking: false`. Every other effort level was ignored, so thinking was unbounded (up to ~22k tokens seen
on 2026-10-26). This plugin sets `thinking_token_budget` from the effort level when the request did not set one
itself. Needs the server's `--reasoning-parser` (the budget is enforced by it).

NEMO_EFFORT_BUDGETS   level=tokens list (default "minimal=512,low=1024,medium=4096,high=16384"); a level that is
                      not listed (xhigh, max) stays unbounded
NEMO_DEFAULT_BUDGET   budget for requests that send no reasoning_effort (default unset = unbounded)
An explicit `thinking_token_budget` in the request always wins. Loaded only by a server whose launcher puts this
folder on PYTHONPATH (entry point in nemoeffort-0.1.0.dist-info); nothing in site-packages changes.
"""
from __future__ import annotations

import functools
import os
from typing import Any

BUDGETS_ENV, DEFAULT_ENV = "NEMO_EFFORT_BUDGETS", "NEMO_DEFAULT_BUDGET"
_DEFAULT_BUDGETS = "minimal=512,low=1024,medium=4096,high=16384"
_installed: dict[str, Any] = {}


def parse_budgets(text: str) -> dict[str, int]:
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        level, _, n = part.partition("=")
        if not n.strip().isdigit() or int(n) < 1:
            raise ValueError(f"{BUDGETS_ENV}: bad entry {part!r} (want level=positive int)")
        out[level.strip().lower()] = int(n)
    return out


def configured() -> tuple[dict[str, int], int | None]:
    budgets = parse_budgets(os.environ.get(BUDGETS_ENV, _DEFAULT_BUDGETS))
    d = os.environ.get(DEFAULT_ENV, "").strip()
    if d and (not d.isdigit() or int(d) < 1):
        raise ValueError(f"{DEFAULT_ENV}={d!r} is not a positive int")
    return budgets, (int(d) if d else None)


def budget_for(effort: str | None, budgets: dict[str, int], default: int | None) -> int | None:
    if effort is None:
        return default
    if effort == "none":  # vLLM turns thinking off itself (enable_thinking false); no budget needed
        return None
    return budgets.get(effort)


def install(budgets: dict[str, int] | None = None, default: int | None = None, request_cls: Any = None) -> str:
    """Patch ChatCompletionRequest.to_sampling_params in this process. Safe to call again."""
    if budgets is None:
        budgets, default = configured()
    if request_cls is None:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest as request_cls
    settings = (tuple(sorted(budgets.items())), default, request_cls)
    if _installed.get("settings") == settings:
        return _installed["name"]
    uninstall()
    orig = request_cls.to_sampling_params

    @functools.wraps(orig)
    def to_sampling_params(self, *args, **kwargs):
        if getattr(self, "thinking_token_budget", None) is None:
            b = budget_for(getattr(self, "reasoning_effort", None), budgets, default)
            if b is not None:
                self.thinking_token_budget = b
        return orig(self, *args, **kwargs)

    request_cls.to_sampling_params = to_sampling_params
    name = f"{request_cls.__name__}.to_sampling_params"
    _installed.update(settings=settings, name=name, original=(request_cls, orig))
    return name


def uninstall() -> None:
    if "original" in _installed:
        cls, orig = _installed["original"]
        cls.to_sampling_params = orig
    _installed.clear()


def register() -> None:
    """Entry point (vllm.general_plugins). Runs in the API server and in the engine core."""
    from vllm.logger import init_logger

    budgets, default = configured()
    name = install(budgets, default)
    init_logger("vllm.nemoeffort").info("nemoeffort: budgets %s, default %s, pid %s, patched: %s", budgets,
                                        default, os.getpid(), name)
