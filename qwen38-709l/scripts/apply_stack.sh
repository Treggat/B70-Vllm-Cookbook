#!/bin/sh
# Apply the full 709-L patch stack to ONE vLLM 0.32.0 source tree, fail-closed (any missing anchor stops the run).
#   1. SergiioB cookbook patches (MTP nightly, draft int4 lm_head + MTP linears, backward-copy fix) + OUR 0.32 re-base of
#      his MTP boundary fix (patches/patch_mtp_boundary_032.py: 0.32 rewrote the GDN spec-decode metadata path)
#   2. our other patches (patches/ in this repo)
#   3. riceharvest's b70 layer (draft vocab as a JSON id list, fp8 target lm_head, packed MTP verify, ...)
#   4. riceharvest's v2 attention kernel .so swapped in over the stock one (stock kept as .stock)
# Dropped from the 0.31 stack: riceharvest's gdn_spec_sync / gdn_zero_draft / gdn_mixed_no_sync (0.32 has the sync-free
# slicing upstream; the three scripts refuse the new layout) and xpu_graph_kv_splits (anchor changed, no gain measured).
# Checked against Vcvzgbzz/b70-cookbook dd083d5 (2026-10-19): its MTP-side patches are the same mechanisms as this layer
# (3-instruction fp8 convert + per-row verify mask in the kernel, packed verify, fp8 head, embed UVA, draft-vocab trim);
# nothing to add for MTP. Its DFlash2 drafter is not ported (see docs/RECIPE.md).
# Upstream repos are cloned at the commits this stack was measured with (pins below). Their scripts hard-code
# container paths (/opt/venv/..., /workspace/vllm), so we copy them to $V/pp and rewrite those paths to OUR tree first:
# otherwise some of them patch every vLLM install they can find.
# Usage: sh apply_stack.sh <venv dir from build_venv.sh> <path to libattn_kernels_xe_2.b70.so (riceharvest v2)>
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
V=${1:?usage: apply_stack.sh <venv dir> <b70 kernel .so>}
SO=${2:?usage: apply_stack.sh <venv dir> <b70 kernel .so>}
PY=$V/venv/bin/python
SP=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])') || exit 1
VL=$V/src/vllm
fail() { echo "APPLY_DONE rc=1 ($*)"; exit 1; }
[ -f "$VL/__init__.py" ] || fail "no vllm source tree at $VL"
grep -q '^__version__ = "0.32' "$VL/_version.py" 2>/dev/null || $PY -c 'import vllm,sys; sys.exit(0 if vllm.__version__.startswith("0.32") else 1)' \
  || fail "this stack is for vLLM 0.32 (use the previous commit of this file for 0.31)"

SERGIIOB=c4d33b9053f9    # SergiioB/intel-arc-pro-b70-inference-cookbook, 2026-10-17
RICEHARVEST=54f63e1a370b # riceharvest/b70-qwen38-cookbook, 2026-10-16
U=$V/upstream; mkdir -p "$U"
fetch() {  # repo dir commit
  [ -d "$U/$2/.git" ] || git clone -q "https://github.com/$1" "$U/$2" || fail "clone $1"
  git -C "$U/$2" checkout -q "$3" || fail "checkout $1@$3"
}
fetch SergiioB/intel-arc-pro-b70-inference-cookbook sergiiob $SERGIIOB
fetch riceharvest/b70-qwen38-cookbook riceharvest $RICEHARVEST

P=$V/pp; rm -rf "$P"; mkdir -p "$P/sergiiob" "$P/b70" "$P/ours"
pin() {  # src dst: copy one patch script with every hard-coded vLLM path pointed at OUR tree
  sed -e "s#/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels#$SP/vllm_xpu_kernels#g" \
      -e "s#/opt/venv/lib/python3.12/site-packages/vllm#$VL#g" \
      -e "s#/opt/vllm/lib/python3.12/site-packages/vllm#$VL#g" \
      -e "s#/workspace/vllm/vllm#$VL#g" \
      -e "s#'/opt/b70'#'$V/rh-runtime'#g" "$1" > "$2"
}
for f in "$U"/sergiiob/patches/*.py; do pin "$f" "$P/sergiiob/$(basename "$f")"; done
for f in "$U"/riceharvest/patches/b70/*.py; do pin "$f" "$P/b70/$(basename "$f")"; done
mkdir -p "$V/rh-runtime"; cp -r "$U"/riceharvest/runtime/. "$V/rh-runtime/" 2>/dev/null || true
export B70_VLLM_ROOTS="$VL"   # SergiioB's backward-copy fix honours this: patch only our tree
export VLLM_ROOT="$VL"        # our patches require it

run() { echo "--- $1"; $PY "$P/$1" || fail "$1"; }
ours() { echo "--- ours/$1"; $PY "$HERE/patches/$1" || fail "$1"; }
run sergiiob/patch_mtp_nightly.py
ours patch_mtp_boundary_032.py            # SergiioB's boundary fix re-based onto 0.32's rewritten pure-spec block
run sergiiob/patch_draft_lmhead_int4.py
run sergiiob/patch_draft_mtp_int4.py
run sergiiob/patch_fix_backward_copy.py
ours patch_meminfo_fallback.py
ours patch_mtp_fc_bf16.py
ours patch_image_url_video.py             # chat_utils.py moved to vllm/renderers/ in 0.32; the script tries both paths
# riceharvest b70 layer. Switches are runtime env (see serve/prod.env.example).
# NOT applied: patch_asym_gptq (GPTQ-only), patch_moe_gemv + patch_linear_fp8 (MoE / bf16-linear checkpoints),
# patch_gdn_* and patch_xpu_graph_kv_splits (see the header).
for p in patch_embed_uva patch_draft_vocab patch_target_lmhead_fp8 patch_spec_greedy_warmup patch_eager_token_pad patch_spec_packed; do
  run b70/$p.py
done
# patch_embed_uva also wraps the DRAFTER's embed_tokens (qwen3_5_mtp.py). vLLM discards that table right after the load
# (_maybe_share_embeddings, still so in 0.32) and shares the target's, so the wrap only adds a second pinned copy and a
# copy-engine write during the draft's weight load; that write reset our card twice. Gate it behind B70_EMBED_UVA_DRAFT (default off).
MTP=$VL/model_executor/models/qwen3_5_mtp.py
grep -q "B70_EMBED_UVA_DRAFT" "$MTP" || sed -i "s|environ.get('B70_EMBED_UVA', '1') == '1'|environ.get('B70_EMBED_UVA_DRAFT', '0') == '1'|" "$MTP"
grep -q "B70_EMBED_UVA_DRAFT" "$MTP" || fail "draft embed gate not applied"

K=$SP/vllm_xpu_kernels/libattn_kernels_xe_2.so
[ -f "$K.stock" ] || cp -p "$K" "$K.stock"
case "$(sha256sum "$SO" | cut -c1-12)" in
  aa241d52a601) fail "that is riceharvest's v1 kernel: it corrupts output with packed MTP verify ('!!!!'); use the v2 .so";;
esac
cp "$SO" "$K"
sha256sum "$K.stock" "$K"

echo "--- compile check of every patched file"
( cd "$V/src" && git status --short | sed 's/^...//' | grep '\.py$' | xargs "$PY" -m py_compile ) || fail "py_compile"
echo "APPLY_DONE rc=0"
