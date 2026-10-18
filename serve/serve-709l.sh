#!/bin/sh
# Serve the 709-L AWQ on one Arc Pro B70 with the patched vLLM 0.31 venv (scripts/build_venv.sh + scripts/apply_stack.sh).
# Config lives in two small files next to this script (copy the *.example ones):
#   prod.env   = sh lines sourced here: runtime switches, cache dir, log
#   prod.extra = plain flags appended LAST (the last flag wins), e.g. the KV pool size
# For experiments, create test.env / test.extra and `touch use-test`: the live config is never edited for a test,
# and `rm use-test` puts the next load back on prod.
# Usage: sh serve-709l.sh <port>
HERE=$(cd "$(dirname "$0")" && pwd)
V=${VENV:?set VENV to the venv dir from build_venv.sh}
MODEL=${MODEL:?set MODEL to the AutoAWQ-gemm model dir (scripts/ct2awq.py output)}
unset PYTHONPATH VIRTUAL_ENV
export PATH="$V/venv/bin:$PATH"
export LD_LIBRARY_PATH="$V/venv/lib:$LD_LIBRARY_PATH"
export VLLM_TARGET_DEVICE=xpu ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export PYTORCH_ALLOC_CONF=expandable_segments:True
# SergiioB draft overlay: the MTP drafter uses int4 copies of the lm_head and MTP linears (target untouched).
# The 709-L AWQ keeps mtp.* in BF16, hence B70_MTP_BF16_DRAFT=1 (+ our patch_mtp_fc_bf16).
export B70_MTP_BF16_DRAFT=1 B70_DRAFT_LMHEAD_INT4=1 B70_DRAFT_MTP_INT4=1
export B70_DRAFT_VOCAB=${B70_DRAFT_VOCAB:-$HERE/draft_vocab_98304.json}   # scripts/make_draft_vocab.py
export VLLM_CACHE_ROOT=$V/cache/default
SPEC='{"method":"mtp","num_speculative_tokens":5}'
CONF=prod; [ -f "$HERE/use-test" ] && CONF=test
[ -s "$HERE/$CONF.env" ] && . "$HERE/$CONF.env"
EXTRA=$(cat "$HERE/$CONF.extra" 2>/dev/null)
mkdir -p "$VLLM_CACHE_ROOT"
PORT=${1:?usage: serve-709l.sh <port>}
exec "$V/venv/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name qwen38-709l \
  --port "$PORT" \
  --dtype float16 --max-model-len 131072 \
  --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --kv-offloading-size 8 \
  --max-num-seqs 4 --max-num-batched-tokens 3584 \
  --enable-prefix-caching --prefix-match-unit 64 \
  --enable-mamba-shared-prefix-checkpoint --mamba-ssm-cache-dtype float16 \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config "$SPEC" \
  --limit-mm-per-prompt '{"image":4,"video":0}' $EXTRA
