"""Run Antidoom's FTPO trainer on Intel XPU (two Arc B70s) for gpt-oss-20b.

Upstream is left untouched; this wraps antidoom.ftpo_train.run_train and
changes only how the model is placed and fed, never the loss math:

- gpt-oss ships MXFP4 experts; they are dequantized to bf16 (~42 GB) and the
  layers are split across both cards with device_map="auto" + max_memory.
- eager attention (gpt-oss attention sinks have no SDPA/flash path on XPU).
- the forward only returns logits for the last position (logits_to_keep=1),
  moved to the input device. FTPO reads logits[:, -1, :] only, so the loss is
  identical, but a 201k-vocab x seq_len logit tensor is never built.
- the collator pads to the longest row in the batch instead of max_seq_length.
  Rows stay left-padded and position ids come from the mask, so the forward
  sees the same tokens at the same positions.
- gradient checkpointing; adamw_torch (no bitsandbytes on XPU).
- merge_lora is forced off: the adapter is merged into the original MXFP4
  checkpoint by merge_into_mxfp4.py, so the result still fits on one card.

Antidoom itself is imported from ./upstream/src (git clone https://github.com/Liquid4All/antidoom upstream).

  ONEAPI_DEVICE_SELECTOR=level_zero:0,1 python xpu_train.py -c configs/xpu_train_r2b_noheads.yaml
  python xpu_train.py -c configs/xpu_train_r2b_noheads.yaml --check   # data + config only, no model
  python xpu_train.py -c <tiny-model config> --cpu-dry-run              # same code path on CPU, no card (dryrun_tiny.py)
  ANTIDOOM_ATTN_CHUNK=0 disables the chunked attention below (round 2 variant B was trained before it existed)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "upstream" / "src"))

logger = logging.getLogger("xpu_train")

# per-card budget for weights; the rest of each 32 GB card is left for
# activations, the LoRA, its gradients and the optimizer state
DEFAULT_MAX_MEMORY_GIB = 24


def install_chunked_attention(chunk: int = 512) -> None:
    """Added in round 3: gpt-oss eager attention materialises (heads x seq x seq) scores per layer; at ~4-5K-token
    contexts that is 6-10 GB per layer and the cards overflowed into host RAM through the xe driver's TTM eviction
    until the host OOM killer ended the run. This computes the SAME math (row-wise max shift, sink column, softmax,
    drop sink) over query chunks, each under its own non-reentrant checkpoint, so the transient is one chunk wide in
    forward and in backward. Rows are independent, so the result is exact (checked on CPU against the original:
    identical logits, gradients within 6e-8)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    import transformers.models.gpt_oss.modeling_gpt_oss as m

    repeat_kv = m.repeat_kv

    def rows(query, key_states, value_states, mask, sinks_w, scaling):
        w = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if mask is not None:
            w = w + mask
        sinks = sinks_w.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        comb = torch.cat([w, sinks], dim=-1)
        comb = comb - comb.max(dim=-1, keepdim=True).values
        probs = F.softmax(comb, dim=-1, dtype=comb.dtype)
        return torch.matmul(probs[..., :-1].to(value_states.dtype), value_states)

    def chunked_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        q_len = query.shape[-2]
        outs = []
        for s in range(0, q_len, chunk):
            e = min(q_len, s + chunk)
            mask = attention_mask[:, :, s:e, :] if attention_mask is not None else None
            if torch.is_grad_enabled():
                o = checkpoint(rows, query[:, :, s:e], key_states, value_states, mask, module.sinks, scaling, use_reentrant=False)
            else:
                o = rows(query[:, :, s:e], key_states, value_states, mask, module.sinks, scaling)
            outs.append(o)
        attn_output = torch.cat(outs, dim=2).transpose(1, 2).contiguous()
        return attn_output, None

    m.eager_attention_forward = chunked_attention_forward
    logger.info("chunked eager attention installed (query chunk %d)", chunk)


def _patch(max_memory_gib: int, cpu: bool = False) -> None:
    import torch
    from antidoom import ftpo_train
    from antidoom.reference_ftpo_trainer import FTPOTrainer
    from transformers import Mxfp4Config

    n = torch.xpu.device_count()
    if cpu:
        if n:
            raise RuntimeError(f"--cpu-dry-run must not see a card, but {n} XPU(s) are visible")
    elif n < 1:
        raise RuntimeError("no XPU visible; set ONEAPI_DEVICE_SELECTOR=level_zero:0,1")
    else:
        logger.info("XPUs visible: %d (%s)", n, ", ".join(torch.xpu.get_device_name(i) for i in range(n)))

    if not cpu:
        _dequantize_on_cpu()
    chunk = int(os.environ.get("ANTIDOOM_ATTN_CHUNK", "512"))
    if chunk > 0:
        install_chunked_attention(chunk)

    real_from_pretrained = ftpo_train.AutoModelForCausalLM.from_pretrained

    def from_pretrained(name, **kw):
        kw.pop("device_map", None)
        kw.pop("torch_dtype", None)
        kw.update(dtype=torch.bfloat16, attn_implementation="eager")
        if cpu:
            kw.update(device_map={"": "cpu"})
        else:
            kw.update(
                device_map="auto",
                max_memory={i: f"{max_memory_gib}GiB" for i in range(n)},
                quantization_config=Mxfp4Config(dequantize=True),
            )
        model = real_from_pretrained(name, **kw)
        logger.info("device map: %s", sorted(set(getattr(model, "hf_device_map", {}).values()), key=str))
        return model

    ftpo_train.AutoModelForCausalLM.from_pretrained = from_pretrained

    real_get_peft_model = ftpo_train.get_peft_model

    def get_peft_model(model, cfg):
        model.enable_input_require_grads()
        peft_model = real_get_peft_model(model, cfg)
        # on the PeftModel: its inner tuner calls the base model's .forward()
        # directly, which would skip hooks registered on the base model
        _last_logits_only(peft_model)
        return peft_model

    ftpo_train.get_peft_model = get_peft_model

    real_make_dpo_config = ftpo_train._make_dpo_config

    def make_dpo_config(cfg):
        args = real_make_dpo_config(cfg)
        args.gradient_checkpointing = True
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        if cpu:
            args.use_cpu = True
        return args

    ftpo_train._make_dpo_config = make_dpo_config

    class XPUFTPOTrainer(FTPOTrainer):
        def ftpo_collator(self, features):
            batch = super().ftpo_collator(features)
            first = int(batch["attention_mask"].any(0).nonzero()[0])
            batch["prompt_ids"] = batch["prompt_ids"][:, first:]
            batch["attention_mask"] = batch["attention_mask"][:, first:]
            return batch

    ftpo_train.FTPOTrainer = XPUFTPOTrainer


def _dequantize_on_cpu() -> None:
    """Unpack the MXFP4 experts on the CPU, then move each tensor to its card.

    transformers unpacks on the target device in chunks of up to 32M rows; on
    the B70 that gather kernel hung the GPU (xe engine reset) and every later
    allocation failed with UR_RESULT_ERROR_OUT_OF_RESOURCES. The arithmetic is
    the same on either device.
    """
    from transformers.integrations import mxfp4

    real = mxfp4._convert_moe_packed_tensors

    def convert(blocks, scales, **kw):
        device = blocks.device
        return real(blocks.cpu(), scales.cpu(), **kw).to(device)

    mxfp4._convert_moe_packed_tensors = convert


def _last_logits_only(model) -> None:
    """Make every forward return [B, 1, V] logits on the device the ids came from."""

    def pre(_mod, args, kwargs):
        kwargs["logits_to_keep"] = 1
        return args, kwargs

    def post(_mod, args, kwargs, out):
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is not None and out.logits.device != ids.device:
            out.logits = out.logits.to(ids.device)
        return out

    model.register_forward_pre_hook(pre, with_kwargs=True)
    model.register_forward_hook(post, with_kwargs=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, type=Path)
    ap.add_argument("--max-memory-gib", type=int, default=DEFAULT_MAX_MEMORY_GIB)
    ap.add_argument("--check", action="store_true", help="validate config and data, load no model")
    ap.add_argument("--cpu-dry-run", action="store_true", help="run the same path on CPU (tiny model)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from antidoom.config import load_config

    cfg = load_config(a.config)
    if cfg.model_name and not cfg.train.model_name:
        cfg.train.model_name = cfg.model_name
    if cfg.train.merge_lora:
        logger.info("merge_lora forced off; merge with merge_into_mxfp4.py")
        cfg.train.merge_lora = False
    cfg.train.auto_merged_output_name = False

    if a.cpu_dry_run:
        cfg.train.bf16 = False  # HF Trainer refuses bf16 autocast without a card

    if a.check:
        _check(cfg.train)
        return
    _patch(a.max_memory_gib, cpu=a.cpu_dry_run)
    from antidoom.ftpo_train import run_train

    run_train(cfg.train)


def _check(tc) -> None:
    import json
    from collections import Counter

    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(tc.dataset_jsonl, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(tc.model_name)
    over = 0
    for r in rows:
        n = len(tok(r["context_with_chat_template"], add_special_tokens=False).input_ids)
        over += n + 1 > tc.max_seq_length
    print(f"rows {len(rows)}, over max_seq_length={tc.max_seq_length}: {over}")
    print("sources:", Counter(r["source_task"].split("/")[-1] for r in rows).most_common())
    print("optim", tc.optim, "| lr", tc.learning_rate, "| r", tc.lora_r, "alpha", tc.lora_alpha,
          "| batch", tc.batch_size, "x", tc.gradient_accumulation_steps, "| targets", tc.target_modules)


if __name__ == "__main__":
    main()
