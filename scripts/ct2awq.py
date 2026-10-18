"""Re-layout a compressed-tensors pack-quantized W4A16 (asym, group) checkpoint into the AutoAWQ "gemm" layout
that vLLM-XPU's AWQ kernel loads. Pure nibble re-ordering: the int4 values, zero points and scales are unchanged,
so the dequantized weights are bit-identical.

compressed-tensors (llmcompressor):
  X.weight_packed     int32 [out, in/8]   nibble i of column c = value (q+8) of input 8c+i
  X.weight_zero_point int32 [out/8, G]    nibble i of row r    = value (zp+8) of output 8r+i
  X.weight_scale      bf16  [out, G]
  dequant: (q - zp) * s == ((q+8) - (zp+8)) * s
AutoAWQ gemm:
  X.qweight int32 [in, out/8]   8 outputs per int32, interleaved [0,2,4,6,1,3,5,7], nibble i <- element interleave[i]
  X.qzeros  int32 [G, out/8]    same packing of the unsigned zero points
  X.scales  bf16  [G, out]
  dequant: (u - z) * s  with unsigned nibbles -> u = q+8, z = zp+8: the nibbles carry over unchanged.
"""
import json, os, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = sys.argv[1]
DST = sys.argv[2]
IL = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
RIL = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])   # inverse of IL
SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)


def unpack_cols(p):            # int32 [R, C/8] -> uint8 [R, C], nibble i = element 8c+i
    v = p.to(torch.int32)
    out = (v[:, :, None] >> SHIFTS[None, None, :]) & 0xF
    return out.reshape(v.shape[0], -1).to(torch.uint8)


def unpack_rows(p):            # int32 [R/8, C] -> uint8 [R, C], nibble i = element 8r+i
    v = p.to(torch.int32)
    out = (v[:, None, :] >> SHIFTS[None, :, None]) & 0xF
    return out.reshape(-1, v.shape[1]).to(torch.uint8)


def awq_pack_cols(u):          # uint8 [K, N] -> int32 [K, N/8] with the AWQ interleave
    K, N = u.shape
    g = u.reshape(K, N // 8, 8)[:, :, IL].to(torch.int32)
    res = torch.zeros((K, N // 8), dtype=torch.int32)
    for i in range(8):
        res |= g[:, :, i] << (4 * i)
    return res


def awq_unpack_cols(p):        # inverse of awq_pack_cols, for the self-check
    v = p.to(torch.int32)
    g = (v[:, :, None] >> SHIFTS[None, None, :]) & 0xF          # [K, N/8, 8] in packed order
    g = g[:, :, RIL]                                            # back to natural order
    return g.reshape(v.shape[0], -1).to(torch.uint8)


def main():
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json"), encoding="utf-8"))
    wmap = idx["weight_map"]
    files = sorted(set(wmap.values()))
    new_map = {}
    n_conv = 0
    # a layer's packed weight and its scale / zero point / shape can sit in different shards: look them up by index
    handles = {f: safe_open(os.path.join(SRC, f), framework="pt") for f in files}
    def get(name):
        return handles[wmap[name]].get_tensor(name)
    for fn in files:
        src_path = os.path.join(SRC, fn)
        if fn == "model-nonquant.safetensors":
            shutil.copyfile(src_path, os.path.join(DST, fn))
            with safe_open(src_path, framework="pt") as f:
                for k in f.keys():
                    new_map[k] = fn
            print(f"copied {fn}", flush=True)
            continue
        out = {}
        with safe_open(src_path, framework="pt") as f:
            keys = list(f.keys())
            packed = [k for k in keys if k.endswith(".weight_packed")]
            for k in keys:
                if k.endswith((".weight_packed", ".weight_zero_point", ".weight_scale", ".weight_shape")):
                    continue
                out[k] = f.get_tensor(k).contiguous()
            for k in packed:
                base = k[: -len(".weight_packed")]
                wp = f.get_tensor(k)
                zp = get(base + ".weight_zero_point")
                sc = get(base + ".weight_scale")
                shape = get(base + ".weight_shape").tolist()
                O, I = int(shape[0]), int(shape[1])
                U = unpack_cols(wp)                     # [out, in]
                assert U.shape == (O, I), (k, U.shape, shape)
                Z = unpack_rows(zp)                     # [out, G]
                G = sc.shape[1]
                assert Z.shape == (O, G) and sc.shape == (O, G), (k, Z.shape, sc.shape)
                UT = U.t().contiguous()                 # [in, out]
                ZT = Z.t().contiguous()                 # [G, out]
                qweight = awq_pack_cols(UT)             # [in, out/8]
                qzeros = awq_pack_cols(ZT)              # [G, out/8]
                scales = sc.t().contiguous()            # [G, out]
                # self-check: the packing round-trips exactly
                assert torch.equal(awq_unpack_cols(qweight), UT), ("qweight roundtrip", k)
                assert torch.equal(awq_unpack_cols(qzeros), ZT), ("qzeros roundtrip", k)
                out[base + ".qweight"] = qweight
                out[base + ".qzeros"] = qzeros
                out[base + ".scales"] = scales
                n_conv += 1
        for k in out:
            new_map[k] = fn
        save_file(out, os.path.join(DST, fn), metadata={"format": "pt"})
        print(f"{fn}: {len(packed)} linears converted, {len(out)} tensors written, {time.time()-t0:.0f}s", flush=True)

    json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_map},
              open(os.path.join(DST, "model.safetensors.index.json"), "w", encoding="utf-8"), indent=2)

    # config: swap the quantization block for an AutoAWQ one
    cfg = json.load(open(os.path.join(SRC, "config.json"), encoding="utf-8"))
    ct = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config")
    ignore = ct["ignore"]
    skip = [x for x in ignore if not x.endswith("linear_attn") and not x.endswith("linear_attn.norm")]
    skip += ["mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj",
             "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.o_proj", "mtp.layers.0.self_attn.q_proj",
             "mtp.layers.0.self_attn.v_proj"]
    awq = {"quant_method": "awq", "bits": 4, "group_size": int(ct["config_groups"]["group_0"]["weights"]["group_size"]),
           "zero_point": True, "version": "gemm", "modules_to_not_convert": skip}
    cfg["quantization_config"] = awq
    if "quantization_config" in cfg.get("text_config", {}):
        cfg["text_config"]["quantization_config"] = awq
    json.dump(cfg, open(os.path.join(DST, "config.json"), "w", encoding="utf-8"), indent=2)
    json.dump(awq, open(os.path.join(DST, "quantization_config.json"), "w", encoding="utf-8"), indent=2)

    for fn in os.listdir(SRC):
        if fn.endswith((".json", ".jinja", ".md", ".txt")) and fn not in ("config.json", "model.safetensors.index.json"):
            shutil.copyfile(os.path.join(SRC, fn), os.path.join(DST, fn))
    print(f"DONE: {n_conv} linears, {len(new_map)} tensors, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
