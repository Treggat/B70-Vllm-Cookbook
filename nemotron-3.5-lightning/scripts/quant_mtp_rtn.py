"""RTN int4 (sym, group 64, compressed-tensors pack-quantized) of Nemotron 3.5 Lightning's MTP head.
Same format as the useful-quants W4A16 target (symmetric int4, group 64, scale = absmax / 7.5 in BF16), so vLLM
uses the same XPU int4 MoE path for the draft. Quantized: routed experts, shared experts, attention q/k/v/o.
Kept bf16: eh_proj, mixer.gate (router + bias), norms.  usage: python quant_mtp_rtn.py BF16_DRAFT_DIR (from extract_mtp_head.py) OUT_DIR
Measured 2026-10-27: 262 tensors quantized, 8 kept, round-trip error 0.10-0.11, 2.49 -> 0.68 GiB, draft acceptance unchanged."""
import json, os, re, shutil, sys
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32
from safetensors import safe_open
from safetensors.torch import save_file

G = 64
QUANT = re.compile(r"^mtp\.layers\.\d+\.mixer\.(q_proj|k_proj|v_proj|o_proj|experts\.\d+\.(up|down)_proj|"
                   r"shared_experts\.(up|down)_proj)\.weight$")


def quantize(w):
    out, inp = w.shape
    g = w.float().view(out, inp // G, G)
    scale = (g.abs().amax(dim=-1) / 7.5).to(torch.bfloat16)
    s = scale.float().clamp_min(torch.finfo(torch.bfloat16).tiny).unsqueeze(-1)
    q = torch.round(g / s).clamp_(-8, 7).to(torch.int8).view(out, inp)
    return pack_to_int32(q, 4), scale


src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
out = {}; nq = nk = 0; errs = []
with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        if QUANT.match(k):
            packed, scale = quantize(t)
            if len(errs) < 3:
                q = unpack_from_int32(packed, 4, torch.Size(t.shape))
                deq = (q.float().view(t.shape[0], -1, G) * scale.float().unsqueeze(-1)).view(t.shape)
                errs.append(round(((deq - t.float()).norm() / t.float().norm()).item(), 4))
            b = k[:-len(".weight")]
            out[b + ".weight_packed"] = packed; out[b + ".weight_scale"] = scale
            out[b + ".weight_shape"] = torch.tensor(list(t.shape), dtype=torch.int64)
            nq += 1
        else:
            out[k] = t.contiguous(); nk += 1
assert all(e < 0.2 for e in errs), errs
save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
for n in ("generation_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
    shutil.copy2(os.path.join(src, n), os.path.join(dst, n))
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["quantization_config"] = {
    "config_groups": {"group_0": {"format": "pack-quantized", "input_activations": None, "output_activations": None,
                                  "targets": ["Linear"],
                                  "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                                              "group_size": G, "num_bits": 4, "observer": "memoryless_minmax",
                                              "observer_kwargs": {}, "scale_dtype": None, "strategy": "group",
                                              "symmetric": True, "type": "int", "zp_dtype": None}}},
    "format": "pack-quantized", "global_compression_ratio": None,
    "ignore": ["lm_head", "re:.*eh_proj$", r"re:.*mixer\.gate$"],
    "kv_cache_scheme": None, "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "sparsity_config": {}, "transform_config": {}, "version": "0.17.0"}
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
print(f"MTP_QUANT_DONE quantized={nq} kept={nk} selftest rel err {errs} "
      f"size {os.path.getsize(os.path.join(dst, 'model.safetensors')) / 2**30:.3f} GiB")
