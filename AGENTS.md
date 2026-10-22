# AGENTS.md

Rules for any AI agent (Claude, Codex, DSH, Copilot, ...) that edits this repo. Humans are welcome to follow them too.
This is a **public** cookbook: people copy its commands onto their own B70s and quote its numbers. Keep it correct
first, then tidy.

Run `python tools/check_docs.py` before every commit. It must print `OK`.

## What this repo is

Recipes for serving LLMs on the Intel Arc Pro B70 (32 GB) with vLLM XPU. One folder per model. Every number was
measured by us on our own box, and every number can be traced to a raw file in that recipe's `results/`.

```
README.md             front page: one row per recipe, what the recipes share
AGENTS.md             this file (CLAUDE.md just points here)
tools/check_docs.py   link, layout and hygiene checks
<recipe>/             one model, e.g. qwen38-709l/, gpt-oss-20b/
  README.md           the recipe page: what it is, current results, quick start, layout, credits
  docs/RECIPE.md      how to build, patch and serve it, step by step
  docs/RESULTS.md     every measurement, with its conditions and raw file (once they outgrow the page)
  docs/PITFALLS.md    what broke and how to avoid it
  docs/<TOPIC>.md     longer write-ups (AWQ.md, ANTIDOOM.md, ...)
  patches/ scripts/ serve/ bench/ results/ plugin/ kernels/ ...   whatever the recipe needs
```

## Where things go

| you have | put it in | and also |
|---|---|---|
| a new measurement | a section in `docs/RESULTS.md` (or the topic doc it belongs to) + the raw file in `results/<topic>/` | update the recipe README's results table only if it changes the current recommendation |
| a new current config | `serve/*.example` and `docs/RECIPE.md` | the README quick start, if a step changes |
| something that broke | one bullet in `docs/PITFALLS.md`, under the right heading | |
| a new patch, script or bench | its folder | a row in the recipe README's **Layout** table |
| a new patch set with its own design | `patches/<name>/README.md` beside the code | link it from the recipe README |
| a new model | a new `<recipe>/` folder following the tree above | one row in the front-page table |
| something the recipes share (engine pins, card pinning) | the front page, "What the recipes share" | |

Paths in a recipe are relative to that recipe's folder. Link between files with relative Markdown links, never with
absolute GitHub URLs to this repo.

## Current vs history

- The **recipe README** and **RECIPE.md** describe what we run *now*. When a config is replaced, update them in
  place; don't stack "update:" paragraphs.
- **RESULTS.md** (and the topic docs) keep history. A superseded measurement stays, in its own dated section, with a line saying what
  replaced it. Never delete or edit a measured number to make a table look tidier. If a number was wrong, say so next
  to it and say why (see the 24k/32k "cites" note in the 709-L RESULTS for the pattern).
- Old versions of scripts live in git history. Say so ("the 0.31 version is one commit back") rather than keeping a
  second copy.
- Put the newest result where a reader looks first: the top of its topic section, not the bottom of the file.

## Numbers

- Every number in a doc comes from a file in `results/`, or from a command anyone can rerun. Name the file next to the
  table.
- Give the conditions: vLLM version, card count, quant, context, streams, thinking on or off, greedy or sampled,
  date (YYYY-MM-DD). Say when a run is a single pass and what the noise is (±5% for our single runs).
- Don't round a measured number into a nicer one. Don't compare numbers from different scripts or different days
  without saying so.
- Units: `t/s` for tokens per second, `s` for seconds, `GiB` for memory sizes vLLM reports in GiB. Thousands get a
  comma (`1,408 t/s`, `389,315 tokens`), `k` for context sizes (`75k`, `262k`).
- Raw results: strip generated text and prompts before committing. JSON/CSV/plain `.out` logs are all fine.

## Writing

- Plain, short, factual. Say what happened and what to do. No hype words ("blazing", "massive", "game-changing"), no
  emoji, no "we're excited".
- Write for a stranger with a B70 who has never seen our setup. Don't use names they can't look up: no private hosts,
  IP addresses, home paths, container or seat names, internal tool names ("hearth", "DSH", "card A") or "our scratch
  dir". Say "our router", "our agent harness", "the first card", or link the public project once and explain it.
- Every command block says where it runs (which folder, which venv) in the sentence before it.
- Headings in sentence case. One topic per section. Tables for comparisons, bullets for lists of independent facts,
  prose for reasoning.
- Lead with the result, then the reason. "Packed verify adds 9% at 65k. It needs the v2 kernel." Not the reverse.
- Keep the voice: "we" for what we did, imperative for what the reader should do.

## Credits and licenses

- Credit every person and project a recipe builds on, with a link, in that recipe's **Credits** section.
- Upstream patches are **fetched at a pinned commit**, not copied in (see `qwen38-709l/scripts/apply_stack.sh`). If a
  file must be copied (kernel patches in `gpt-oss-20b/kernels/patches/`), keep its license file beside it.
- Our code and docs are MIT. Don't add code under another license without saying so in the recipe's License section.

## Hygiene

- No secrets, tokens, API keys or private hostnames anywhere, including results and example configs.
- Local configs (`serve/prod.env`, `serve/*.extra`, `use-test`, bench outputs) are gitignored. Ship `*.example`
  files instead.
- No large binaries: no weights, no `.so` files, no datasets. Link to where they come from and give a sha256.
- Every file in a recipe is either listed in its README's Layout table or sits inside a folder that is.
- Don't leave TODOs, commented-out experiments or `.bak` files in the tree.

## Changing the repo

- One topic per branch and per PR. Branch names: `<recipe>/<topic>` (`qwen38-709l/combo-depth`) or
  `docs/<topic>` for repo-wide edits.
- Commit subjects start with the recipe folder or the file touched: `qwen38-709l: ...`, `gpt-oss-20b: ...`,
  `README: ...`. Imperative, under 72 characters. The body says what changed and why, with the key numbers.
- Commits are authored by **Treggat <taggert2222@gmail.com>**. Each AI that worked on the change adds a trailer, e.g.
  `Co-Authored-By: Claude <noreply@anthropic.com>`. PR descriptions end "Opened by Treggat, assisted by ...".
- Never push to `main`. Never merge a PR yourself unless the owner asked you to in this conversation.
- A change to serving flags, patches or weights isn't done until its recipe's sanity tests pass on real hardware
  (`bench/crossing.py` and `bench/bench.py` for the 709-L, `plugin/selftest.py`, `bench/tool_flow.py` and
  `bench/ptc_probe.py` for gpt-oss). Put the results in the PR.
- Docs-only changes: run `tools/check_docs.py`, and re-read every page you touched as rendered Markdown.

## Before you commit

1. `python tools/check_docs.py` prints `OK`.
2. Every new file is in a Layout table; every removed file is out of it.
3. The front-page row, the recipe README and RECIPE.md agree on the current config and headline speeds.
4. New numbers name their raw file and conditions.
5. `git status` shows no local config, outputs or `.bak` files staged.
