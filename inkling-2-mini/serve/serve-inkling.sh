#!/bin/sh
# Serve Inkling 2 Mini on one Arc Pro B70. $1 = port.
#   VENV=/opt/b70/vllm-ink2 MODEL=/models/inkling-2-mini CARD=0 sh serve/serve-inkling.sh 8000
# VENV holds venv/ and src/ (the engine checkout, see the README). CARD is the Level Zero index of the B70.
# Switches come from serve/inkling.env, flags from serve/inkling.extra (appended last, so they win).
HERE=$(cd "$(dirname "$0")" && pwd)
V=${VENV:?set VENV to the engine folder}
MODEL=${MODEL:?set MODEL to the checkpoint folder}
[ -f "$HERE/inkling.env" ] && . "$HERE/inkling.env"
PY="$V/venv/bin/python"
unset PYTHONPATH VIRTUAL_ENV ONEAPI_DEVICE_SELECTOR
export PYTHONDONTWRITEBYTECODE=1 PATH="$V/venv/bin:$PATH" LD_LIBRARY_PATH="$V/venv/lib:$LD_LIBRARY_PATH"
export VLLM_TARGET_DEVICE=xpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE PYTORCH_ALLOC_CONF=expandable_segments:True
export ZE_AFFINITY_MASK=${CARD:-0}
# One compile cache per config: compiled kernels and the MoE autotune results are keyed on it.
TAG=$(cat "$HERE/inkling.extra" "$HERE/inkling.env" 2>/dev/null | tr -d '\r' | md5sum | cut -c1-8)
export VLLM_CACHE_ROOT="$V/cache/inkling-$TAG"
mkdir -p "$VLLM_CACHE_ROOT"
cd "$V" || exit 1
VLLM_ROOT="$V/src/vllm" "$PY" "$HERE/../patches/patch_ink2_b70.py" || exit 1
set -f
EXTRA=$(cat "$HERE/inkling.extra" 2>/dev/null | tr -d '\r')
exec "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name inkling-2-mini \
  --port "$1" \
  --language-model-only \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser inkling \
  --reasoning-parser inkling \
  --enable-prompt-tokens-details $EXTRA
