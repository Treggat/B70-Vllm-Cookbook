#!/bin/sh
# Serve NVIDIA Nemotron 3.5 Lightning 30B-A3B (W4A16) on one Arc Pro B70 with the vLLM 0.32 XPU venv
# (../../qwen38-709l/scripts/build_venv.sh). The venv must have the STOCK attention library: gpt-oss's optional rebuilt
# kernels (../../gpt-oss-20b/kernels/) only hold head-64 variants, and Nemotron's attention is head 128.
# Usage: VENV=/opt/b70/vllm032 MODEL=/models/nemotron-3.5-lightning-w4a16 sh serve-nemotron.sh <port>
#
# Variables (all optional except VENV and MODEL):
#   VENV         venv dir from build_venv.sh (holds venv/ and src/)
#   MODEL        model dir: useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16 as downloaded
#   SERVED_NAME  model id clients send (default nemotron)
#   CARD         Level Zero device index for ZE_AFFINITY_MASK (default 0; on our 2-card box 0 = first, 1 = second)
#   SSM_CONFIGS  folder with the tuned Mamba state-update config (default ../ssm-configs); empty = vLLM's default
#   EXTRA_FILE   plain flags appended LAST, the last flag wins (default ./nemotron.extra; copy nemotron.extra.example)
#
# Measured on vLLM 0.32.0, 2026-10-26, with nemotron.extra.example and the tuned SSM config (docs/RESULTS.md):
#   121.5 / 317.7 / 691.4 tok/s at 1 / 4 / 16 streams, KV pool 2,630,376 tokens (164k per stream at 16).
#   Without the .extra (eager, no prefix caching): 24.6 / 97.4 / 385.4 tok/s.
HERE=$(cd "$(dirname "$0")" && pwd)
V=${VENV:?set VENV to the venv dir from build_venv.sh}
MODEL=${MODEL:?set MODEL to the Nemotron W4A16 model dir}
PORT=${1:?usage: serve-nemotron.sh <port>}
PY="$V/venv/bin/python"
EXTRA_FILE=${EXTRA_FILE:-$HERE/nemotron.extra}
unset PYTHONPATH VIRTUAL_ENV
export PYTHONDONTWRITEBYTECODE=1
export PATH="$V/venv/bin:$PATH"
export LD_LIBRARY_PATH="$V/venv/lib:$LD_LIBRARY_PATH"
export VLLM_TARGET_DEVICE=xpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE PYTORCH_ALLOC_CONF=expandable_segments:True
# torch 2.13+ picks the card the Level Zero way; ONEAPI_DEVICE_SELECTOR would fight ZE_AFFINITY_MASK.
unset ONEAPI_DEVICE_SELECTOR
export ZE_AFFINITY_MASK="${CARD:-0}"
# Tuned selective_state_update config for this card (163 -> 86 us per call at 16 streams, +6.6% decode). vLLM ships
# B70 configs too, but our driver names the card "Intel(R)_Graphics_[0xe223]", so the bundled ones never match.
SSM_CONFIGS=${SSM_CONFIGS-$HERE/../ssm-configs}
[ -n "$SSM_CONFIGS" ] && export VLLM_TUNED_CONFIG_FOLDER="$SSM_CONFIGS"
# One compile cache per flag combination: a changed flag must not reuse a stale cache.
if [ -s "$EXTRA_FILE" ]; then
  TAG=$(tr -d '\r' < "$EXTRA_FILE" | md5sum | cut -c1-8)
else
  TAG=eager
fi
export VLLM_CACHE_ROOT="$V/cache/nemotron-$TAG"
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
  --served-model-name "${SERVED_NAME:-nemotron}" \
  --port "$PORT" \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enable-prompt-tokens-details $EXTRA
