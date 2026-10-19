#!/usr/bin/env bash
# Build libattn_kernels_xe_2.so for gpt-oss-20b on the B70: vllm-xpu-kernels release/0.1.15.4 + sycl-tla 87f68506
# + riceharvest's tla-fast3 / xk-fold / zero-sum-guard + our bf16fast5, gpt-oss kernel variants only. See README.md.
#
# usage: WORK=/scratch/xk-gptoss ./build.sh
#   WORK    scratch dir (needs ~8 GB: oneAPI ~3.5 GB, sources, build tree)
#   IMG     image that supplies cmake/ninja and a python with torch XPU (default vllm/vllm-openai-xpu:v0.31.0)
#   VENV    optional: your vLLM 0.31 venv dir (holds venv/); mounted read-only and its python used instead of the
#           image's /opt/venv/bin/python. This is how we built ours (with a local Ubuntu 24.04 image that only
#           supplied cmake and ninja); the default IMG route follows riceharvest's build and we have not run it.
#   JOBS    parallel compile jobs (default 2: each icpx job peaks around 8 GB; ours peaked at ~10 GB total)
#   MEM     memory cap for the build container (default 18g)
# Takes about 40 minutes. Output: $WORK/build/libattn_kernels_xe_2.so (ours: 41,912,952 bytes).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=${WORK:?set WORK to a scratch dir}
IMG=${IMG:-vllm/vllm-openai-xpu:v0.31.0}
JOBS=${JOBS:-2}
MEM=${MEM:-18g}
XK_REF=release/0.1.15.4          # ddf336d86e3c8602888572a3502f951abd51df12 when we built
TLA_REF=87f6850680a580654b9ea2c80dbc01aeb36ad231
mkdir -p "$W/oneapi"
cd "$W"

# 1. Intel oneAPI DPC++/C++ 2026.0 into $W/oneapi (a throwaway Ubuntu container; ~3.5 GB). Ours reported
#    "Intel(R) oneAPI DPC++/C++ Compiler 2026.0.0 (2026.0.0.20260331)".
if [ ! -x "$W/oneapi/compiler/2026.0/bin/icpx" ]; then
  docker run --rm -v "$W/oneapi:/opt/intel/oneapi" ubuntu:24.04 bash -c '
    set -e; apt-get update -qq; apt-get install -y -qq gpg wget ca-certificates >/dev/null
    wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq intel-oneapi-compiler-dpcpp-cpp-2026.0 >/dev/null
    /opt/intel/oneapi/compiler/2026.0/bin/icpx --version'
fi

# 2. Sources at the pinned commits, patched. riceharvest's original .patch files have CRLF line endings; the copies in
#    patches/ are LF, and we strip CRs again in case your checkout converted them.
if [ ! -d "$W/xk" ]; then
  git clone -q -b "$XK_REF" https://github.com/vllm-project/vllm-xpu-kernels.git "$W/xk"
  git clone -q https://github.com/intel/sycl-tla.git "$W/tla"
  git -C "$W/tla" checkout -q "$TLA_REF"
  mkdir -p "$W/patches"
  for p in xk-fold-b70 zero-sum-guard tla-fast3-b70; do sed 's/\r$//' "$HERE/patches/$p.patch" > "$W/patches/$p.patch"; done
  cp "$HERE/patches/bf16fast5.py" "$W/patches/"
  git -C "$W/xk" apply "$W/patches/xk-fold-b70.patch"
  git -C "$W/xk" apply "$W/patches/zero-sum-guard.patch"
  git -C "$W/tla" apply "$W/patches/tla-fast3-b70.patch"
  # bf16fast5 goes on top of tla-fast3 (it uses the image's python3, so the host needs none)
  docker run --rm -v "$W:/w" --entrypoint python3 "$IMG" /w/patches/bf16fast5.py /w/tla/include/cute/arch/reorder_xe.hpp
  cp "$HERE/gptoss_prefill.conf" "$HERE/gptoss_decode.conf" "$W/xk/csrc/xpu/attn/kernel_configs/"
fi

# 3. Build (AOT for BMG-G31 only, attention library only).
PYMOUNT=(); PY=/opt/venv/bin/python
if [ -n "${VENV:-}" ]; then PYMOUNT=(-v "$VENV:/venv-src:ro"); PY=/venv-src/venv/bin/python; fi
docker run --rm --name xk-gptoss-build --memory="$MEM" --memory-swap="$MEM" \
  -v "$W:/w" -v "$W/oneapi:/opt/intel/oneapi:ro" ${PYMOUNT[@]+"${PYMOUNT[@]}"} \
  -e VLLM_XPU_AOT_DEVICES=bmg-g31-a0 -e VLLM_XPU_XE2_AOT_DEVICES=bmg-g31-a0 \
  --entrypoint bash "$IMG" -c "
set -eo pipefail
export CMPLR_ROOT=/opt/intel/oneapi/compiler/2026.0
export PATH=/opt/intel/oneapi/compiler/2026.0/bin:/opt/venv/bin:\$PATH
git config --global --add safe.directory '*'
cd /w/xk
[ -f /w/build/build.ninja ] || cmake -S . -B /w/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=xpu \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.cmake -DVLLM_PYTHON_EXECUTABLE=$PY \
  -DVLLM_XPU_ENABLE_XE3P=OFF -DVLLM_XPU_ENABLE_XE_DEFAULT=OFF -DBASIC_KERNELS_ENABLED=OFF -DMOE_KERNELS_ENABLED=OFF \
  -DGDN_KERNELS_ENABLED=OFF -DMQA_LOGITS_KERNELS_ENABLED=OFF -DMHC_KERNELS_ENABLED=OFF -DXPU_SPECIFIC_KERNELS_ENABLED=OFF \
  -DXPUMEM_ALLOCATOR_ENABLED=OFF -DVLLM_XPU_ENABLE_ONEDNN=OFF -DFA2_KERNELS_ENABLED=ON \
  -DVLLM_CHUNK_PREFILL_CONFIG=gptoss_prefill.conf -DVLLM_PAGED_DECODE_CONFIG=gptoss_decode.conf -DVLLM_CUTLASS_SRC_DIR=/w/tla
ninja -C /w/build -j $JOBS attn_kernels_xe_2"

ls -l "$W/build/libattn_kernels_xe_2.so"
echo "built $W/build/libattn_kernels_xe_2.so - install it as described in README.md (stop the server first)"
