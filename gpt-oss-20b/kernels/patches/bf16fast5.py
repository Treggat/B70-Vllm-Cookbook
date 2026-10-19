#!/usr/bin/env python3
"""bf16 "fast5" fp8 KV conversion for gpt-oss on Xe2: e4m3 -> bf16 in 5 instructions instead of 11.

Apply to sycl-tla's include/cute/arch/reorder_xe.hpp AFTER riceharvest's tla-fast3-b70.patch:

  python bf16fast5.py <sycl-tla>/include/cute/arch/reorder_xe.hpp

Why: tla-fast3 speeds up e4m3 -> fp16 (riceharvest's Qwen build runs in fp16). gpt-oss runs in bf16, so it takes the
e4m3 -> bf16 path, which tla-fast3 doesn't touch, and a stock fp8 KV cache made its cold prefill about 25% slower.
This does the same trick for bf16: a left shift of the SIGNED byte by 4 sign-extends into bits 15..11, the 0x87FF mask
keeps sign + 4-bit exponent + 3-bit mantissa (= value * 2^-120 as bf16), and the stock x2^120 multiply restores it.
Only the NaN compare and fix-up are dropped: vLLM's fp8 KV cache saturates on write, so e4m3 NaN codes never occur.

Outputs on gpt-oss-20b were byte-identical to the stock kernels (20 greedy prompts, reasoning included).
"""
import sys

p = sys.argv[1]
with open(p, newline="") as f:  # keep the file's own line endings
    s = f.read()
start = s.index("#define CUTE_XE_REORDER_E4M3_BF16_SEQ")
end = s.index("\n\n", start)
new_seq = r'''#define CUTE_XE_REORDER_E4M3_BF16_SEQ \
    ".decl OUT_UD v_type=G type=UD num_elts=32 alias=<%0,0>\n" \
    ".decl OUT_BF v_type=G type=BF num_elts=64 alias=<%0,0>\n" \
    /* gpt-oss bf16 fast5: input is the signed byte shifted left 4 (sign-extended); */ \
    /* stock mask and x2^120 rescale, NaN fix-up dropped (vLLM fp8 KV saturates). */ \
    "and (M1_NM, 32) OUT_UD(0,0)<1> OUT_UD(0,0)<1;1,0> 0x87FF87FF:ud\n" \
    "mul (M1_NM, 32) OUT_BF(0,0)<1> OUT_BF(0,0)<1;1,0> 0x7B800000:f\n" \
    "mul (M1_NM, 32) OUT_BF(1,0)<1> OUT_BF(1,0)<1;1,0> 0x7B800000:f\n"'''
s = s[:start] + new_seq + s[end:]
for kind in ("UU", "VV"):
    a = s.index("struct Xe_Reorder<ReorderKind::%s, float_e4m3_t, bfloat16_t>" % kind)
    b = s.index("CUTE_XE_REORDER_E4M3_BF16_SEQ", a)
    seg = s[a:b]
    i = seg.index("asm (     /*")
    tail = (seg[i:]
            .replace(".decl IN_UB v_type=G type=UB", ".decl IN_UB v_type=G type=B")
            .replace(".decl OUT_UW v_type=G type=UW", ".decl OUT_UW v_type=G type=W")
            .replace("> 8:uw\\n", "> 4:w\\n"))
    assert tail.count("> 4:w\\n") == 2, kind
    s = s[:a] + seg[:i] + tail + s[b:]
with open(p, "w", newline="") as f:
    f.write(s)
print("patched UU+VV and BF16_SEQ")
