# The model and the AWQ conversion

- **Source merge:** [DavidAU](https://huggingface.co/DavidAU), [*Qwen3.8-27B TWIN-TURBO Fable Cold-Fusion 709-L*](https://huggingface.co/DavidAU/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-NM-DAU-NEO-MTP-GGUF) (BF16 source about 55.6 GB; the linked page is DavidAU's GGUF release of the merge).
- **Quant we serve:** [`TheUnderscore/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-W4A16-AWQ`](https://huggingface.co/TheUnderscore/Qwen3.8-27B-TWIN-TURBO-Fable-Cold-Fusion-709-L-Uncensored-W4A16-AWQ)
  (about 19.6 GB; llmcompressor, int4 asymmetric, group 128; vision tower, MTP head and lm_head kept in BF16).

## Why convert

That quant ships in compressed-tensors **pack-quantized** layout. vLLM's `wNa16` scheme for that layout only has a
CUDA (Marlin) kernel, so vLLM-XPU won't load it. vLLM-XPU's int4 path is the AWQ one (`vllm_xpu_kernels`).

## What `scripts/ct2awq.py` does

It only re-orders nibbles. The int4 values, zero points and scales stay the same, so the dequantized weights are
**bit-identical**, and the script checks a round trip for every layer.

| compressed-tensors | AutoAWQ gemm |
|---|---|
| `weight_packed` int32 `[out, in/8]` | `qweight` int32 `[in, out/8]`, interleave `[0,2,4,6,1,3,5,7]` |
| `weight_zero_point` int32 `[out/8, G]` (+8 offset) | `qzeros` int32 `[G, out/8]`, same packing |
| `weight_scale` bf16 `[out, G]` | `scales` bf16 `[G, out]` |

Dequant is `(q − zp)·s` either way, because the +8 offsets cancel.

`config.json` gets `quantization_config = {awq, 4 bit, group 128, zero_point true, version gemm}`, and
`modules_to_not_convert` = the compressed-tensors ignore list without the bare `linear_attn` entries, plus the
`mtp.*` linears.

```sh
pip install torch safetensors   # CPU-only torch is fine
python scripts/ct2awq.py <downloaded W4A16 dir> <out dir>
```

It runs on the CPU, one shard at a time, and writes about the same size as the input (about 19 GB).

Because `mtp.*` stays BF16, serve with `B70_MTP_BF16_DRAFT=1`, and apply `patch_mtp_fc_bf16` so `fc` stays BF16 too.

## Draft acceptance on a merge

The MTP head comes from the base model, so acceptance on a merge is lower than on base Qwen3.8: about 72–75% per draft
token on code, 30–40% on prose and agent text at depth. Base Qwen3.8 drafters get more. Drafters trained on base Qwen3.8
(DFlash/DSpark) only reached 3–12% acceptance on 709-L for us. That's why the draft-side speedups (int4 draft head,
vocab trim, packed verify) matter more here than they would on the base model.
