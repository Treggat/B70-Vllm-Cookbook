"""B70 patches for the Inkling 2 XPU engine (2026-11-26). Run with the recipe venv before every serve:
  VLLM_ROOT=/opt/b70/vllm-ink2/src/vllm python patches/patch_ink2_b70.py
Idempotent, fail-closed (exit 1 if an anchor
moved). Every behaviour change is a no-op unless its switch is set:
 1. meminfo fallback (always on): vllm-xpu-kernels getMemoryInfo reads 0 free under this L0 driver.
 2. fp8 lm_head (B70_TARGET_LMHEAD_FP8=1): riceharvest's helper, adapted (b70_target_lmhead_fp8.py beside this file), hooked into
    LogitsProcessor._apply_head and run after Inkling2ForConditionalGeneration.load_weights. Inkling's head is
    201024x2304 BF16 with head_dtype float32; on XPU the stock path casts the whole head to fp32 EVERY step
    (logits_processor.py "falls back to the cast path"), ~1.85 GB of traffic per step. The fp8 head returns fp32
    logits so the 30.0 final softcap still runs in fp32.
 3. vision on XPU (B70_INK2_VISION=1): Intel's tree rejects image inputs on XPU only because the ViT calls CUDA
    flash_attn_varlen_func and one PTX bf16-round helper; route the ViT attention through vLLM's XPU
    flash_attn_varlen_func (fa_utils, the path Qwen-VL / the 709-L recipe's vision uses) and give _bf16 a portable branch.
 4. fp8 KV on XPU (B70_INK2_FP8KV=1): the Inkling2 attention reads/writes unit-scale E4M3 KV through generic Triton
    loads/stores; upstream only allows it on SM12x. K/V are norm-clipped (qk_l2_clip_threshold, value norm), which is
    why unit scale fits. Gate opened on XPU when the switch is set.
 6. DSpark drafter on XPU (B70_INK2_DSPARK=1; add --speculative-config {"method":"dspark","num_speculative_tokens":7}
    to .extra): the draft (mtp.safetensors, 3 dense layers, block 8) is rejected on XPU only because it lives in
    nvidia/. Its attention already runs on the portable _rel_verify_kernel, its linears fall back to
    torch.nn.functional.linear, and its own kernels are plain Triton on libdevice calls that triton-xpu has. What is
    CUDA-only: the flashinfer RMSNorm (replaced on XPU by common/ops/norm.rms_norm, same fp32 math, sum normalized
    before rounding), the FlashInfer fused all-reduce lookup (skipped on XPU: one card), and one int->pointer select
    in the q/k norm kernel (rewritten as a value select, identical numerics, all platforms). The draft KV must stay
    BF16 when the target KV is fp8 (the target's unit-scale E4M3 does not fit the draft's unnormalized K/V): the
    speculator config forces that on XPU too when B70_INK2_FP8KV=1.
    Not compatible with B70_TARGET_LMHEAD_FP8=1: the draft folds and reads the target's BF16 lm_head weight, which
    the fp8 head frees. Do not set both."""
import os, sys, shutil
SRC = os.environ.get("VLLM_ROOT") or sys.exit("set VLLM_ROOT to the vllm package folder of the engine checkout")
HERE = os.path.dirname(os.path.abspath(__file__))

def edit(path, mark, old, new):
    t = open(path).read()
    if mark in t:
        print("already:", mark, path); return
    if t.count(old) != 1:
        sys.exit(f"anchor not found ({mark}) in {path}; not patched")
    t = t.replace(old, new, 1); compile(t, path, "exec"); open(path, "w").write(t)
    print("patched:", mark, path)

# 1 meminfo
edit(f"{SRC}/platforms/xpu.py", "B70_MEMINFO_FALLBACK",
     "    free, total = torch.ops._C_cache_ops.getMemoryInfo(device)\n\n    return free, total\n",
     "    free, total = torch.ops._C_cache_ops.getMemoryInfo(device)\n"
     "    if free <= 0:  # B70_MEMINFO_FALLBACK: kernels op reads 0 free under this L0 driver; torch's query is right\n"
     "        free, total = torch.xpu.mem_get_info(device)\n\n    return free, total\n")

# 2 fp8 lm_head
shutil.copyfile(f"{HERE}/b70_target_lmhead_fp8.py", f"{SRC}/model_executor/models/b70_target_lmhead_fp8.py")
print("copied: b70_target_lmhead_fp8.py")
edit(f"{SRC}/model_executor/layers/logits_processor.py", "B70_TARGET_LMHEAD_FP8_HOOK",
     '        """Project hidden states through the lm_head, honoring head_dtype."""\n',
     '        """Project hidden states through the lm_head, honoring head_dtype."""\n'
     '        if getattr(lm_head, "_b70_fp8", None) is not None:  # B70_TARGET_LMHEAD_FP8_HOOK\n'
     '            from vllm.model_executor.models.b70_target_lmhead_fp8 import logits as _b70_lg\n'
     '            return _b70_lg(lm_head, hidden_states, self.head_dtype)\n')
edit(f"{SRC}/models/inkling2/common/model.py", "B70_TARGET_LMHEAD_FP8_CONVERT",
     '        if missing := params.keys() - loaded:\n'
     '            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")\n'
     '        return loaded\n',
     '        if missing := params.keys() - loaded:\n'
     '            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")\n'
     '        # B70_TARGET_LMHEAD_FP8_CONVERT: optional fp8 lm_head (B70_TARGET_LMHEAD_FP8=1)\n'
     '        from vllm.model_executor.models.b70_target_lmhead_fp8 import convert as _b70_cv\n'
     '        _b70_cv(self)\n'
     '        return loaded\n')
# 3 vision on XPU
I2 = f"{SRC}/models/inkling2"
edit(f"{I2}/nvidia/ops/vision.py", "B70_INK2_VISION_BF16",
     'def _bf16(x):\n    # Round to bf16 and back; inline asm keeps Triton from folding the casts.\n'
     '    rounded = tl.inline_asm_elementwise(\n        "cvt.rn.bf16.f32 $0, $1;",\n        "=h,f",\n        [x],\n'
     '        dtype=tl.bfloat16,\n        is_pure=True,\n        pack=1,\n    )\n    return rounded.to(tl.float32)\n',
     'def _bf16(x):\n    # Round to bf16 and back; inline asm keeps Triton from folding the casts.\n'
     '    if _B70_IS_CUDA:  # B70_INK2_VISION_BF16: PTX on CUDA only, a plain round-trip cast elsewhere (XPU)\n'
     '        rounded = tl.inline_asm_elementwise(\n            "cvt.rn.bf16.f32 $0, $1;",\n            "=h,f",\n'
     '            [x],\n            dtype=tl.bfloat16,\n            is_pure=True,\n            pack=1,\n        )\n'
     '    else:\n        rounded = x.to(tl.bfloat16)\n    return rounded.to(tl.float32)\n')
edit(f"{I2}/nvidia/ops/vision.py", "_B70_IS_CUDA = tl.constexpr",
     "from vllm.triton_utils import tl, tldevice, triton\n",
     "from vllm.triton_utils import tl, tldevice, triton\n"
     "from vllm.platforms import current_platform as _b70_platform\n\n"
     "_B70_IS_CUDA = tl.constexpr(_b70_platform.is_cuda())\n")
edit(f"{I2}/nvidia/vision.py", "B70_INK2_VISION_ATTN",
     "        else:\n            from vllm.vllm_flash_attn import flash_attn_varlen_func\n\n"
     "            version = 3 if current_platform.is_device_capability_family(90) else 2\n"
     "            out = flash_attn_varlen_func(q, k, v, fa_version=version, **kwargs)\n",
     "        elif current_platform.is_xpu():  # B70_INK2_VISION_ATTN: vLLM's XPU varlen attention (vllm-xpu-kernels)\n"
     "            from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func as _xpu_fa\n\n"
     "            out = _xpu_fa(q.contiguous(), k.contiguous(), v.contiguous(), **kwargs)\n"
     "        else:\n            from vllm.vllm_flash_attn import flash_attn_varlen_func\n\n"
     "            version = 3 if current_platform.is_device_capability_family(90) else 2\n"
     "            out = flash_attn_varlen_func(q, k, v, fa_version=version, **kwargs)\n")
edit(f"{I2}/common/model.py", "B70_INK2_VISION_GATE",
     '    unsupported = []\n    if vision:\n',
     '    unsupported = []\n'
     '    if vision and __import__("os").environ.get("B70_INK2_VISION") != "1":  # B70_INK2_VISION_GATE\n')
# 4 fp8 KV on XPU
edit(f"{I2}/common/attention.py", "B70_INK2_FP8KV",
     '    if current_platform.is_device_capability_family(120):\n        supported += ("float16", "fp8", "fp8_e4m3")\n',
     '    if current_platform.is_device_capability_family(120):\n        supported += ("float16", "fp8", "fp8_e4m3")\n'
     '    elif current_platform.is_xpu() and __import__("os").environ.get("B70_INK2_FP8KV") == "1":  # B70_INK2_FP8KV\n'
     '        supported += ("fp8", "fp8_e4m3")\n')
# 5 loop_share_cache on a non-varloop checkpoint (B70_INK2_LOOP_SHARE=1), EXPERIMENT: upstream only allows
# --hf-overrides {"text_config":{"loop_share_cache":true}} on is_varloop checkpoints, because Inkling 2 Mini was
# trained with one KV cache per loop pass. Sharing cuts the full-attention KV from 8 passes to 4 per token; expect a
# quality change, measure it.
edit(f"{I2}/configs.py", "B70_INK2_LOOP_SHARE",
     '        if (self.loop_num_passes is not None or self.loop_share_cache) and not varloop:\n',
     '        _b70_ls = __import__("os").environ.get("B70_INK2_LOOP_SHARE") == "1"  # B70_INK2_LOOP_SHARE\n'
     '        if (self.loop_num_passes is not None or self.loop_share_cache) and not varloop and not _b70_ls:\n')
# 6 DSpark drafter on XPU (B70_INK2_DSPARK=1)
if os.environ.get("B70_INK2_DSPARK") == "1" and os.environ.get("B70_TARGET_LMHEAD_FP8") == "1":
    sys.exit("B70_INK2_DSPARK=1 needs the target's BF16 lm_head; unset B70_TARGET_LMHEAD_FP8")
edit(f"{I2}/common/model.py", "B70_INK2_DSPARK_GATE",
     '    if spec is not None and spec.use_dspark():\n        unsupported.append("DSpark speculative decoding")\n',
     '    if (\n        spec is not None\n        and spec.use_dspark()\n'
     '        and __import__("os").environ.get("B70_INK2_DSPARK") != "1"  # B70_INK2_DSPARK_GATE\n    ):\n'
     '        unsupported.append("DSpark speculative decoding")\n')
edit(f"{I2}/nvidia/dspark.py", "B70_INK2_DSPARK_IMPORT",
     "from vllm.forward_context import get_forward_context\n",
     "from vllm.forward_context import get_forward_context\n"
     "from vllm.platforms import current_platform as _b70_platform  # B70_INK2_DSPARK_IMPORT\n")
edit(f"{I2}/nvidia/dspark.py", "B70_INK2_DSPARK_NORM",
     "        from flashinfer.norm import fused_add_rmsnorm, rmsnorm\n\n        if residual is None:\n",
     "        if _b70_platform.is_xpu():  # B70_INK2_DSPARK_NORM: flashinfer is CUDA-only\n"
     "            from vllm.models.inkling2.common.ops.norm import rms_norm as _b70_rms\n\n"
     "            if residual is None:\n"
     "                return _b70_rms(x.contiguous(), self.weight, self.eps)\n"
     "            # residual += x; x = norm(residual), normalized from the unrounded fp32 sum.\n"
     "            total = residual.float() + x.float()\n"
     "            residual.copy_(total)\n"
     "            x.copy_(_b70_rms(total, self.weight, self.eps, x.dtype))\n"
     "            return x\n"
     "        from flashinfer.norm import fused_add_rmsnorm, rmsnorm\n\n        if residual is None:\n")
edit(f"{I2}/nvidia/dspark.py", "B70_INK2_DSPARK_AR",
     '    fi_ar = getattr(get_tp_group().device_communicator, "fi_ar_comm", None)\n',
     '    if _b70_platform.is_xpu():  # B70_INK2_DSPARK_AR: one card, nothing to all-reduce\n'
     '        return norm(x, residual)\n'
     '    fi_ar = getattr(get_tp_group().device_communicator, "fi_ar_comm", None)\n')
edit(f"{I2}/nvidia/ops/dspark.py", "B70_INK2_DSPARK_WSEL",
     "        w_ptr = tl.where(is_q, Q_WEIGHT.to(tl.int64), K_WEIGHT.to(tl.int64)).to(\n"
     "            tl.pointer_type(tl.bfloat16)\n        )\n"
     "        x = tl.load(row + d_ar).to(tl.float32)\n"
     "        ms = tl.sum(x * x, 0) / D\n"
     "        inv = 1.0 / tl.sqrt(ms + EPS)\n"
     "        w1 = tl.load(w_ptr + half_ar).to(tl.float32)\n"
     "        w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)\n",
     "        # B70_INK2_DSPARK_WSEL: select the q or k norm weight by value, not by an int->pointer cast\n"
     "        x = tl.load(row + d_ar).to(tl.float32)\n"
     "        ms = tl.sum(x * x, 0) / D\n"
     "        inv = 1.0 / tl.sqrt(ms + EPS)\n"
     "        w1 = tl.where(is_q, tl.load(Q_WEIGHT + half_ar), tl.load(K_WEIGHT + half_ar)).to(tl.float32)\n"
     "        w2 = tl.where(\n"
     "            is_q, tl.load(Q_WEIGHT + HALF + half_ar), tl.load(K_WEIGHT + HALF + half_ar)\n"
     "        ).to(tl.float32)\n")
edit(f"{I2}/speculator_config.py", "B70_INK2_DSPARK_KV",
     "        if current_platform.is_device_capability_family(120):\n",
     "        if current_platform.is_device_capability_family(120) or (  # B70_INK2_DSPARK_KV\n"
     '            current_platform.is_xpu() and __import__("os").environ.get("B70_INK2_FP8KV") == "1"\n'
     "        ):\n")
# 7 NVFP4 emulation MoE launch configs from our own sweep on this card (bench/moe_sweep.py, 2026-11-26).
# B70_INK2_MOE_TILE="N,K,warps" overrides the split-dots (decode / verify) tile; Intel's 128,128,8 was swept for 1-64
# tokens only, a 16-stream DSpark verify is 128. B70_INK2_MOE_PREFILL="M,N,K,warps" overrides the default-path
# (prefill-sized batch) tile, 0 = keep the default's value. Unset = Intel's values. BLOCK_SIZE_K changes the fp32 accumulation order only.
edit(f"{I2}/xpu/experts.py", "B70_INK2_MOE_TILE",
     '        "BLOCK_SIZE_N": 128,\n        "BLOCK_SIZE_K": 128,\n        "GROUP_SIZE_M": 1,\n'
     '        "num_warps": 8,\n        "num_stages": 2,\n    }\n',
     '        "BLOCK_SIZE_N": _B70_TILE[0],\n        "BLOCK_SIZE_K": _B70_TILE[1],\n        "GROUP_SIZE_M": 1,\n'
     '        "num_warps": _B70_TILE[2],\n        "num_stages": 2,\n    }\n\n\n'
     '# B70_INK2_MOE_TILE / B70_INK2_MOE_PREFILL: launch tiles from our sweep (see patches/patch_ink2_b70.py)\n'
     '_B70_TILE = tuple(\n'
     '    int(v) for v in (__import__("os").environ.get("B70_INK2_MOE_TILE") or "128,128,8").split(",")\n)\n'
     '_B70_PREFILL = tuple(\n'
     '    int(v) for v in (__import__("os").environ.get("B70_INK2_MOE_PREFILL") or "").split(",") if v\n)\n')
edit(f"{I2}/xpu/experts.py", "B70_INK2_MOE_PREFILL_USE",
     '                w13.size(), w2.size(), topk, None, config_tokens\n            )\n',
     '                w13.size(), w2.size(), topk, None, config_tokens\n            )\n'
     '            if _B70_PREFILL:  # B70_INK2_MOE_PREFILL_USE\n'
     '                config = dict(\n'
     '                    config,\n'
     '                    BLOCK_SIZE_M=_B70_PREFILL[0] or config["BLOCK_SIZE_M"],\n'
     '                    BLOCK_SIZE_N=_B70_PREFILL[1] or config["BLOCK_SIZE_N"],\n'
     '                    BLOCK_SIZE_K=_B70_PREFILL[2] or config["BLOCK_SIZE_K"],\n'
     '                    num_warps=_B70_PREFILL[3] or config["num_warps"],\n'
     '                )\n')
print("PATCHES_OK")
