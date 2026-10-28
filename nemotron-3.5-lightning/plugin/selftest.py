"""Offline self-test (no GPU): the patched request gets the right thinking_token_budget.
Run with the server's venv:  PYTHONPATH=<this folder> python selftest.py"""
import os, sys
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from nemoeffort.vllm_plugin import budget_for, install, parse_budgets, uninstall

ok = fail = 0
def check(name, got, want):
    global ok, fail
    if got == want: ok += 1
    else: fail += 1; print(f"FAIL {name}: got {got!r}, want {want!r}")

B = parse_budgets("minimal=512,low=1024,medium=4096,high=16384")
check("parse", B, {"minimal": 512, "low": 1024, "medium": 4096, "high": 16384})
for bad in ("low=x", "low=0", "low"):
    try: parse_budgets(bad); check(f"parse rejects {bad}", "accepted", "ValueError")
    except ValueError: ok += 1
check("none", budget_for("none", B, 777), None)
check("no effort -> default", budget_for(None, B, 777), 777)
check("no effort, no default", budget_for(None, B, None), None)
check("xhigh unbounded", budget_for("xhigh", B, 777), None)
check("max unbounded", budget_for("max", B, None), None)

def sp(effort=None, budget=None):
    kw = {"model": "nemotron", "messages": [{"role": "user", "content": "hi"}]}
    if effort is not None: kw["reasoning_effort"] = effort
    if budget is not None: kw["thinking_token_budget"] = budget
    return ChatCompletionRequest(**kw).to_sampling_params(100, {}).thinking_token_budget

check("unpatched low", sp("low"), None)
install(B, 2048)
for e, want in (("low", 1024), ("medium", 4096), ("high", 16384), ("minimal", 512), ("none", None),
                ("xhigh", None), (None, 2048)):
    check(f"patched {e}", sp(e), want)
check("explicit budget wins", sp("low", 9999), 9999)
check("install twice is a no-op", install(B, 2048), "ChatCompletionRequest.to_sampling_params")
uninstall()
check("uninstalled low", sp("low"), None)
print(f"{ok}/{ok + fail} passed")
sys.exit(1 if fail else 0)
