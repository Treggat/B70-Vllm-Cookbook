#!/bin/sh
# Build a vLLM 0.31.0 XPU venv from source (editable install) in $1. Nothing outside $1 is touched.
# Tested on: Intel Arc Pro B70 (BMG-G31, 32 GB), Linux xe driver, Python 3.12, inside a container that already has the
# Level Zero / compute runtime. Result: torch 2.14.0+xpu, triton 3.8.0+xpu, vllm-xpu-kernels 0.1.15.4.
# Usage: sh build_venv.sh /opt/b70/vllm031      (progress on stdout; last line BUILD_DONE rc=<n>)
V=${1:?usage: build_venv.sh <target dir>}
mkdir -p "$V" && cd "$V" || exit 1
unset PYTHONPATH VIRTUAL_ENV
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TMPDIR="$V/tmp"
mkdir -p "$V/tmp" "$V/cache"
step() { echo "=== $(date '+%H:%M:%S') $*"; }
fail() { echo "BUILD_DONE rc=1 ($*)"; exit 1; }
# torch/xpu wheels come from PyTorch's index; triton==3.8.0+xpu only exists on vLLM's own XPU index.
IDX="--extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/"

step "venv"
[ -x venv/bin/python ] || python3 -m venv venv || fail venv
venv/bin/python -m pip install -U pip wheel setuptools || fail pip

step "source v0.31.0"
# keep src root-owned / owned by the user that runs pip: a chown'd clone makes pip's git calls fail ("dubious ownership")
[ -d src/.git ] || git clone --depth 1 --branch v0.31.0 https://github.com/vllm-project/vllm.git src || fail clone
git -C src log -1 --format='%H %s'

step "install (VLLM_TARGET_DEVICE=xpu, editable)"
( cd src && VLLM_TARGET_DEVICE=xpu ../venv/bin/python -m pip install -e . $IDX ) || fail "pip install vllm"

step "kernels pin"
venv/bin/python -m pip install "vllm-xpu-kernels==0.1.15.4" $IDX || echo "WARN: kernels pin failed, keeping what the install pulled"

step "versions"
venv/bin/python -m pip list 2>/dev/null | grep -i -E '^(vllm|torch|triton|transformers)'
export LD_LIBRARY_PATH="$V/venv/lib:$LD_LIBRARY_PATH"
venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, 'xpu', torch.xpu.is_available(), torch.xpu.device_count()); print('vllm', vllm.__version__)" || fail "import check"
echo "BUILD_DONE rc=0"
