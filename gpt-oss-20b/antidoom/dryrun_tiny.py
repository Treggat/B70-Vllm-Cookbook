"""CPU dry run of the XPU trainer port on a tiny random gpt-oss. No card is used.

1. builds a 2-layer gpt-oss with the real tokenizer and vocab (random weights)
2. checks the two input changes against the unpatched forward, in fp32:
   logits_to_keep=1 vs full logits, and batch-max padding vs max_seq_length padding
3. trains a few steps through xpu_train.py --cpu-dry-run (upstream FTPOTrainer)
4. merges the adapter with merge_into_mxfp4.py and compares every merged
   tensor with PEFT's own merge_and_unload()

  ONEAPI_DEVICE_SELECTOR=opencl:cpu python dryrun_tiny.py --base /models/gpt-oss-20b --work dry \
      --pairs runs/mine2/pairs-r2.jsonl
(--base is only read for its config and tokenizer files.)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "upstream" / "src"))


def build_tiny(base: Path, out: Path) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(base)
    cfg.num_hidden_layers = 2
    cfg.layer_types = cfg.layer_types[:2]
    cfg.hidden_size = 64
    cfg.intermediate_size = 64
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.num_local_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.tie_word_embeddings = False  # as in gpt-oss-20b
    cfg.quantization_config = None
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    model.save_pretrained(out, max_shard_size="20MB")
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja", "generation_config.json"):
        if (base / f).exists():
            shutil.copy2(base / f, out / f)
    print(f"tiny model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params -> {out}")


def check_equivalence(tiny: Path, pairs: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny)
    model = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.float32, attn_implementation="eager").eval()
    rows = [json.loads(l) for l in open(pairs, encoding="utf-8")][:4]
    seqs = [tok(r["context_with_chat_template"], add_special_tokens=False).input_ids[-200:] for r in rows]

    def batch(pad_to: int):
        ids = torch.full((len(seqs), pad_to), tok.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids[i, -len(s):] = torch.tensor(s)
            mask[i, -len(s):] = True
        L = ids.shape[1]
        pos = (torch.arange(L).unsqueeze(0) - (L - mask.sum(1)).unsqueeze(1)).clamp(min=0).masked_fill(~mask, 0)
        return ids, mask, pos

    with torch.no_grad():
        ids, mask, pos = batch(512)  # stands in for max_seq_length padding
        full = model(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits[:, -1, :]
        keep1 = model(ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        ids2, mask2, pos2 = batch(max(len(s) for s in seqs))  # batch-max padding
        trimmed = model(ids2, attention_mask=mask2, position_ids=pos2, use_cache=False, logits_to_keep=1).logits[:, -1, :]
    d1 = (full - keep1).abs().max().item()
    d2 = (full - trimmed).abs().max().item()
    print(f"logits_to_keep=1 vs full: max abs diff {d1:.2e}")
    print(f"batch-max padding vs fixed padding: max abs diff {d2:.2e}")
    assert d1 < 1e-5 and d2 < 1e-4, "input changes alter the forward"


def write_config(real_cfg: Path, tiny: Path, pairs: Path, work: Path) -> Path:
    import yaml

    data = yaml.safe_load(real_cfg.read_text())
    data["model_name"] = str(tiny)
    t = data["train"]
    sub = work / "pairs-48.jsonl"
    sub.write_text("".join(open(pairs, encoding="utf-8").readlines()[:48]), encoding="utf-8")
    t.update(dataset_jsonl=str(sub), output_dir=str(work / "lora"), merged_output_dir=str(work / "unused"),
             max_train_examples=48, gradient_accumulation_steps=4, early_stopping_chosen_win=None,
             learning_rate=1e-3)
    out = work / "dry.yaml"
    out.write_text(yaml.safe_dump(data))
    return out


def check_merge(tiny: Path, work: Path) -> None:
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    merged_dir = work / "merged"
    subprocess.run([sys.executable, str(HERE / "merge_into_mxfp4.py"), "--base", str(tiny),
                    "--adapter", str(work / "lora"), "--out", str(merged_dir)], check=True)
    ref = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.bfloat16),
                                    work / "lora").merge_and_unload().state_dict()
    mine = {}
    for f in merged_dir.glob("*.safetensors"):
        mine.update(load_file(f))
    base = {}
    for f in tiny.glob("*.safetensors"):
        base.update(load_file(f))
    changed = [k for k in mine if not torch.equal(mine[k], base[k])]
    worst = max((mine[k].float() - ref[k].float()).abs().max().item() for k in mine)
    print(f"merge: {len(changed)} tensors changed, worst diff vs PEFT merge_and_unload {worst:.2e}")
    assert changed, "merge changed nothing"
    assert worst < 2e-2, "merged weights disagree with PEFT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--pairs", type=Path, required=True, help="any FTPO pairs file (the first 48 rows are used)")
    ap.add_argument("--config", type=Path, default=HERE / "configs/xpu_train_r2b_noheads.yaml")
    a = ap.parse_args()
    if torch.xpu.is_available():
        raise SystemExit("an XPU is visible; run with ONEAPI_DEVICE_SELECTOR=opencl:cpu")
    if a.work.exists():
        shutil.rmtree(a.work)
    a.work.mkdir(parents=True)
    tiny = a.work / "tiny-gptoss"
    build_tiny(a.base, tiny)
    check_equivalence(tiny, a.pairs)
    cfg = write_config(a.config, tiny, a.pairs, a.work)
    subprocess.run([sys.executable, str(HERE / "xpu_train.py"), "-c", str(cfg), "--cpu-dry-run"], check=True)
    check_merge(tiny, a.work)
    print("DRY RUN OK")


if __name__ == "__main__":
    main()
