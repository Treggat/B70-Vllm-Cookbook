#!/bin/sh
# Serve gpt-oss-20b (MXFP4, stock or an Antidoom merge) on one Arc Pro B70 with the vLLM 0.31 XPU venv
# (../../qwen38-709l/scripts/build_venv.sh) and the gptoss031 plugin (../plugin).
# Usage: VENV=/opt/b70/vllm031 MODEL=/models/gpt-oss-20b sh serve-gpt-oss.sh <port>
#
# Variables (all optional except VENV and MODEL):
#   VENV         venv dir from build_venv.sh (holds venv/ and src/)
#   MODEL        model dir: openai/gpt-oss-20b as downloaded, or a merge from ../antidoom/merge_into_mxfp4.py
#   SERVED_NAME  model id clients send (default gpt-oss)
#   CARD         Level Zero device index for ZE_AFFINITY_MASK (default 0; on our 2-card box 0 = first, 1 = second)
#   PLUGIN_DIR   folder that holds gptoss031/ and gptoss031-0.1.0.dist-info/ (default ../plugin)
#   GPTOSS_TOOL_STRICT  off | function | parameter (default function): the plugin's tool-call grammar
#   GPTOSS_EFFORT_CAP   low | medium | high (default medium): requests asking for more are served at this effort
#   EXTRA_FILE   plain flags appended LAST, the last flag wins (default ./gpt-oss.extra; copy gpt-oss.extra.example)
#
# Measured on vLLM 0.31.0, stock weights, 2026-10-14, with gpt-oss.extra.example (see docs/RECIPE.md):
#   1 stream 106-117 tok/s, 4 streams 257 summed, 16 streams 698 summed. Without the .extra (eager) about 43 tok/s.
#   bench/tool_flow.py 2 x 16 streaming multi-turn sessions: 168 calls, 0 problems. bench/ptc_probe.py at effort high
#   capped to medium: 28/36 (uncapped: 8/36).
HERE=$(cd "$(dirname "$0")" && pwd)
V=${VENV:?set VENV to the venv dir from build_venv.sh}
MODEL=${MODEL:?set MODEL to the gpt-oss-20b model dir}
PORT=${1:?usage: serve-gpt-oss.sh <port>}
PY="$V/venv/bin/python"
PLUGIN_DIR=${PLUGIN_DIR:-$HERE/../plugin}
EXTRA_FILE=${EXTRA_FILE:-$HERE/gpt-oss.extra}
unset PYTHONPATH VIRTUAL_ENV
export PYTHONDONTWRITEBYTECODE=1
# The plugin is active only for this server: its entry point is found because PLUGIN_DIR is on PYTHONPATH.
export PYTHONPATH="$PLUGIN_DIR"
export GPTOSS_TOOL_STRICT="${GPTOSS_TOOL_STRICT:-function}"
export GPTOSS_EFFORT_CAP="${GPTOSS_EFFORT_CAP:-medium}"
export PATH="$V/venv/bin:$PATH"
export LD_LIBRARY_PATH="$V/venv/lib:$LD_LIBRARY_PATH"
export VLLM_TARGET_DEVICE=xpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE PYTORCH_ALLOC_CONF=expandable_segments:True
# torch 2.13+ picks the card the Level Zero way; ONEAPI_DEVICE_SELECTOR would fight ZE_AFFINITY_MASK.
unset ONEAPI_DEVICE_SELECTOR
export ZE_AFFINITY_MASK="${CARD:-0}"
# Optional diagnostics we kept on while chasing a rare "!!!!" (NaN) output: dev mode adds POST /reset_prefix_cache,
# so a poisoned prefix cache can be tested before a reload.
# export VLLM_SERVER_DEV_MODE=1
# One compile cache per flag combination: a changed flag must not reuse a stale cache. The folder name comes from the
# .extra contents (no .extra = "eager"), so editing the .extra switches cache by itself.
if [ -s "$EXTRA_FILE" ]; then
  TAG=$(tr -d '\r' < "$EXTRA_FILE" | md5sum | cut -c1-8)
else
  TAG=eager
fi
export VLLM_CACHE_ROOT="$V/cache/gptoss-$TAG"
mkdir -p "$VLLM_CACHE_ROOT"
# Run from the venv dir: a folder named vllm in the current directory would be imported instead of the install.
cd "$V" || exit 1
# getMemoryInfo reads 0 GiB free under some Level Zero drivers; the fallback patch is idempotent.
MEMPATCH="$HERE/../../qwen38-709l/patches/patch_meminfo_fallback.py"
if [ -f "$MEMPATCH" ]; then
  "$PY" "$MEMPATCH" "$V/src/vllm" || exit 1
fi
# $EXTRA is word-split on purpose; keep any JSON in the .extra free of spaces. set -f stops [ ] globbing.
set -f
EXTRA=$(cat "$EXTRA_FILE" 2>/dev/null)
exec "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-gpt-oss}" \
  --port "$PORT" \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --tool-strict-level auto \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --structured-outputs-config '{"backend":"xgrammar"}' $EXTRA
