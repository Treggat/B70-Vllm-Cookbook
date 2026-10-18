"""Merge an FTPO LoRA adapter into the original gpt-oss MXFP4 checkpoint.

The adapter only touches attention projections and lm_head, which gpt-oss
stores in bf16 next to its MXFP4 experts. Each of those tensors becomes
W + (alpha / r) * B @ A (computed in fp32, stored bf16); every other tensor,
the experts included, is copied byte for byte. The output is the same format
and size as the input, so it serves on one card exactly like the original.

  python merge_into_mxfp4.py --base /models/gpt-oss-20b \
      --adapter runs/ftpo_lora_r2b --out /models/gpt-oss-20b-antidoom-r2b
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")

    acfg = json.loads((a.adapter / "adapter_config.json").read_text())
    if acfg.get("use_rslora") or acfg.get("use_dora"):
        raise SystemExit("rslora/dora adapters are not handled")
    scale = acfg["lora_alpha"] / acfg["r"]
    lora = load_file(a.adapter / "adapter_model.safetensors")

    # base_model.model.<module>.lora_A.weight -> <module>.weight
    deltas: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k in lora:
        if k.endswith(".lora_A.weight"):
            mod = k[: -len(".lora_A.weight")].removeprefix("base_model.model.")
            deltas[mod + ".weight"] = (lora[k], lora[k.replace("lora_A", "lora_B")])
    # PEFT also saves a frozen copy of lm_head (base_layer) when lm_head is a
    # target; it is the unchanged base weight, so it is not merged
    other = [k for k in lora if "lora_A" not in k and "lora_B" not in k and not k.endswith(".base_layer.weight")]
    if other:
        raise SystemExit(f"adapter has non-LoRA tensors this script does not merge: {other[:5]}")

    index = json.loads((a.base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = sorted(set(deltas) - set(index))
    if missing:
        raise SystemExit(f"adapter targets not in checkpoint: {missing[:5]}")

    a.out.mkdir(parents=True)
    for f in a.base.iterdir():
        if not f.name.endswith(".safetensors"):
            shutil.copy2(f, a.out / f.name)

    for shard in sorted({index[k] for k in index}):
        src = a.base / shard
        touched = [k for k in deltas if index[k] == shard]
        if not touched:
            shutil.copy2(src, a.out / shard)
            continue
        with safe_open(src, "pt") as fh:
            meta = fh.metadata()
            tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        for k in touched:
            w = tensors[k]
            A, B = deltas[k]
            if (B.shape[0], A.shape[1]) != tuple(w.shape):
                raise SystemExit(f"shape mismatch for {k}: weight {tuple(w.shape)}, B@A {(B.shape[0], A.shape[1])}")
            tensors[k] = (w.float() + scale * (B.float() @ A.float())).to(w.dtype)
        save_file(tensors, a.out / shard, metadata=meta)
        print(f"{shard}: merged {len(touched)} tensors")
    print(f"done: {len(deltas)} tensors merged (scale {scale}) -> {a.out}")


if __name__ == "__main__":
    main()
