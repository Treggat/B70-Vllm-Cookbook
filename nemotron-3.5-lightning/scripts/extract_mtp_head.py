#!/usr/bin/env python3
"""Build a separate MTP draft folder from NVIDIA's BF16 checkpoint, without downloading the whole 60 GB.

The useful-quants W4A16 checkpoint drops NVIDIA's multi-token-prediction head (num_nextn_predict_layers 0). NVIDIA's
BF16 repo keeps it: all 270 `mtp.*` tensors (one attention + one 128-expert MoE layer, 2.49 GiB) sit in the last
shard. vLLM 0.32 loads such a folder as the draft model of `--speculative-config {"method":"mtp","model":<folder>}`
(NemotronHMTPModel), so the W4A16 target and its quantization config stay untouched.

usage: python extract_mtp_head.py BF16_SNAPSHOT_DIR W4A16_DIR OUT_DIR
  BF16_SNAPSHOT_DIR  holds config.json, model.safetensors.index.json and the shard(s) with mtp.* tensors, e.g.
                     hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
                       config.json model.safetensors.index.json model-00014-of-00014.safetensors --local-dir <dir>
  W4A16_DIR          the target checkpoint (tokenizer files are copied from it)
  OUT_DIR            new draft folder: model.safetensors (mtp.* only, BF16), NVIDIA's config.json, tokenizer
Then make it int4 with quant_mtp_rtn.py (0.68 GiB, faster and more KV pool).
"""
import json, os, shutil, sys

from safetensors import safe_open
from safetensors.torch import save_file

src, target, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
wmap = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
shards = sorted({f for k, f in wmap.items() if k.startswith("mtp.")})
tensors = {}
for fn in shards:
    with safe_open(os.path.join(src, fn), "pt") as f:
        for k in f.keys():
            if k.startswith("mtp."):
                tensors[k] = f.get_tensor(k)
assert tensors, "no mtp.* tensors found"
cfg = json.load(open(os.path.join(src, "config.json")))
assert cfg.get("num_nextn_predict_layers", 0) >= 1, "config has no MTP layer"
save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
shutil.copy2(os.path.join(src, "config.json"), os.path.join(out, "config.json"))
for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    if os.path.exists(os.path.join(target, n)):
        shutil.copy2(os.path.join(target, n), os.path.join(out, n))
size = sum(v.numel() * v.element_size() for v in tensors.values())
print(f"MTP_EXTRACT_DONE {len(tensors)} tensors from {shards}, {size / 2**30:.3f} GiB -> {out}")
