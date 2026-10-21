#!/usr/bin/env python3
"""vLLM 0.32 re-base of Vcvzgbzz/b70-cookbook dd083d5 patches/vllm/dflash2-drafter.patch (MARKER B70_DFLASH2_032).

Runs the DFlash2 drafter (incoai/Qwen3.8-27B-DFlash2, BF16-trained, overflows FP16) under the FP16 target:
  B70_DRAFT_DTYPE=bfloat16   the drafter alone is built in BF16 (config, params, buffers, KV); values are cast at the
                             three crossings: target aux states in (fc), the shared target embedding table in, the
                             drafter's final hidden state out to the shared LM head.
  B70_DFLASH_DRAFT_INT4=1    drafter linears to int4 g128 after load (b70_dflash_draft_int4.py, copied alongside).
  B70_DRAFT_LMHEAD_INT4=1    DFlash2's candidate top-k comes from the int4 draft head (SergiioB helper, riceharvest
                             B70_DRAFT_VOCAB trim honoured) instead of the full target head.
Every change is a no-op with the gates unset, so the MTP seat (prod) is unaffected.

Differences from the 0.31 cookbook patch, beyond moved anchors:
  * our draft-vocab trim is riceharvest's (model._b70_draft_vocab_ids, a JSON id list), not the cookbook's
    _b70_lmhead_idx (first-N + specials): the int4 candidate path maps top-k ids through _b70_draft_vocab_ids.
  * speculator.py's buffer dtype only follows the drafter when B70_DRAFT_DTYPE is set (belt and braces for MTP).

Target: $VLLM_ROOT (the vllm/ package dir) or argv[1]. Idempotent (marker), fail-closed (every anchor exactly once,
all files patched or none), py_compile of every touched file.
"""
from __future__ import annotations

import os
import py_compile
import sys
from pathlib import Path

MARKER = "B70_DFLASH2_032"

root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VLLM_ROOT", ""))
if not (root / "__init__.py").is_file():
    sys.exit(f"{MARKER}: VLLM_ROOT does not point at a vllm package dir: {root}")

EDITS: dict[str, list[tuple[str, str]]] = {}

# 1. config/speculative.py: the drafter's ModelConfig dtype (B70_DRAFT_DTYPE, else the target's as before)
EDITS["config/speculative.py"] = [(
    """                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
""",
    """                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=(  # B70_DFLASH2_032: B70_DRAFT_DTYPE builds the drafter in its own dtype
                        __import__("os").environ.get("B70_DRAFT_DTYPE")
                        or self.target_model_config.dtype
                    ),
                    seed=self.target_model_config.seed,
""",
)]

# 2. model_executor/models/qwen3_dflash.py: the three dtype crossings
EDITS["model_executor/models/qwen3_dflash.py"] = [
    (
        """def dflash_has_any_non_causal(config: Qwen3Config) -> bool:
""",
        """def _b70_draft_dtype(
    hidden_states: torch.Tensor, lm_head: nn.Module, processor: nn.Module
) -> torch.Tensor:
    \"\"\"B70_DFLASH2_032: cast drafter hidden states to a shared target lm_head's dtype (B70_DRAFT_DTYPE).\"\"\"
    w = getattr(lm_head, "weight", None)
    if isinstance(w, torch.Tensor) and w.dtype in (torch.float16, torch.bfloat16):
        if w.dtype != hidden_states.dtype:
            processor.head_dtype = w.dtype  # the head runs in the target dtype
        return hidden_states.to(w.dtype)
    return hidden_states


def dflash_has_any_non_causal(config: Qwen3Config) -> bool:
""",
    ),
    (
        """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
""",
        """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids).to(self.mask_embedding.dtype)  # B70_DFLASH2_032: shared table may be fp16
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
""",
    ),
    (
        """        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits
""",
        """        logits = self.logits_processor(
            self.lm_head, _b70_draft_dtype(hidden_states, self.lm_head, self.logits_processor)
        )  # B70_DFLASH2_032
        if self.draft_id_to_target_id is None:
            return logits
""",
    ),
    (
        """        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        expected = self.model.fc.input_size
""",
        """        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        hidden_states = hidden_states.to(self.model.mask_embedding.dtype)  # B70_DFLASH2_032: target aux states may be fp16
        expected = self.model.fc.input_size
""",
    ),
]

# 3. model_executor/models/qwen3_dflash2.py: int4 drafter body after load, int4 draft head for candidates
EDITS["model_executor/models/qwen3_dflash2.py"] = [
    (
        """import torch
import torch.nn.functional as F
from torch import nn
""",
        """import os  # B70_DFLASH2_032

import torch
import torch.nn.functional as F
from torch import nn
""",
    ),
    (
        """        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
""",
        """        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config  # B70_DFLASH2_032: build_draft_lmhead_int4 reads the TP size here
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
""",
    ),
    (
        """    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )
""",
        """    def load_weights(self, weights):  # B70_DFLASH2_032
        loaded = super().load_weights(weights)
        from vllm.model_executor.models.b70_dflash_draft_int4 import quantize_dflash_drafter_int4

        quantize_dflash_drafter_int4(self)  # no-op unless B70_DFLASH_DRAFT_INT4=1
        return loaded

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.models.qwen3_dflash import _b70_draft_dtype

        if os.environ.get("B70_DRAFT_LMHEAD_INT4") == "1" and not getattr(
            self, "_b70_lmhead_int4_tp_blocked", False
        ):  # B70_DFLASH2_032: candidates from the int4 draft head
            from vllm.model_executor.models.b70_draft_lmhead_int4 import build_draft_lmhead_int4

            if getattr(self, "_b70_lmhead_int4", None) is None:
                build_draft_lmhead_int4(self)
            if getattr(self, "_b70_lmhead_int4", None) is not None:
                return self._b70_int4_candidates(hidden_states)
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head,
            _b70_draft_dtype(hidden_states, self.lm_head, self.candidate_logits_processor),
            self.model.candidate_selector.top_k,
        )

    def _b70_int4_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        \"\"\"B70_DFLASH2_032: get_top_k_tokens over the int4 draft head (riceharvest B70_DRAFT_VOCAB trim honoured).\"\"\"
        from vllm.model_executor.models.b70_draft_lmhead_int4 import int4_lmhead_logits

        proc = self.candidate_logits_processor
        qweight, scales, qzeros, group_size = self._b70_lmhead_int4
        logits = int4_lmhead_logits(
            hidden_states.to(scales.dtype), qweight, scales, qzeros, group_size
        )
        kept = getattr(self, "_b70_draft_vocab_ids", None)  # rows of the trimmed head -> vocab ids
        if kept is None:
            logits = logits[..., : proc.org_vocab_size]
        values, ids = torch.topk(logits, self.model.candidate_selector.top_k, dim=-1)
        ids = kept[ids] if kept is not None else ids.to(torch.int64)
        values = values.float()
        if proc.scale != 1.0:
            values = values * proc.scale
        if proc.soft_cap is not None:
            values = torch.tanh(values / proc.soft_cap) * proc.soft_cap
        return ids, values
""",
    ),
]

# 4. v1/worker/gpu/spec_decode/dflash/utils.py: build the drafter under a model_config of its own dtype
EDITS["v1/worker/gpu/spec_decode/dflash/utils.py"] = [
    (
        """import torch.nn as nn

from vllm.config import VllmConfig, replace
""",
        """import copy  # B70_DFLASH2_032

import torch.nn as nn

from vllm.config import VllmConfig, replace
""",
    ),
    (
        """    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
""",
        """    # B70_DFLASH2_032 (B70_DRAFT_DTYPE): the drafter builds its params, buffers and KV in its own dtype
    model_config = vllm_config.model_config
    if draft_model_config.dtype != model_config.dtype:
        model_config = copy.copy(model_config)
        object.__setattr__(model_config, "dtype", draft_model_config.dtype)
    draft_vllm_config = replace(
        vllm_config,
        model_config=model_config,
        attention_config=replace(
""",
    ),
]

# 5. v1/worker/gpu/spec_decode/speculator.py: drafter-side buffers in the drafter's dtype
EDITS["v1/worker/gpu/spec_decode/speculator.py"] = [(
    """        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype
""",
    """        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = (  # B70_DFLASH2_032 (B70_DRAFT_DTYPE): drafter-side buffers in the drafter's dtype
            self.draft_model_config.dtype
            if __import__("os").environ.get("B70_DRAFT_DTYPE")
            else vllm_config.model_config.dtype
        )
""",
)]

# --- apply: check everything first, then write everything (all or nothing)
helper_src = Path(__file__).resolve().parent / "b70_dflash_draft_int4.py"
helper_dst = root / "model_executor/models/b70_dflash_draft_int4.py"
if not helper_src.is_file():
    sys.exit(f"{MARKER}: helper missing next to this script: {helper_src}")
for dep in ("b70_draft_lmhead_int4.py", "b70_draft_mtp_int4.py"):
    if not (root / "model_executor/models" / dep).is_file():
        sys.exit(f"{MARKER}: {dep} missing in the tree (run the SergiioB line first)")

texts: dict[str, str] = {}
pending: list[str] = []
for rel, edits in EDITS.items():
    p = root / rel
    if not p.is_file():
        sys.exit(f"{MARKER}: missing {p}")
    s = p.read_text()
    if MARKER in s:
        # already patched: every replacement must be present, and no anchor that the replacement consumed may remain
        for old, new in edits:
            if s.count(new) != 1 or (old not in new and old in s):
                sys.exit(f"{MARKER}: {rel} carries the marker but is only partly patched; restore it from git")
        texts[rel] = s
        continue
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit(f"{MARKER}: anchor not found exactly once in {rel}:\n{old}")
        s = s.replace(old, new)
    texts[rel] = s
    pending.append(rel)

if not pending and helper_dst.is_file() and helper_dst.read_text() == helper_src.read_text():
    print(f"{MARKER}: already applied ({root})")
    sys.exit(0)

for rel in pending:
    (root / rel).write_text(texts[rel])
helper_dst.write_text(helper_src.read_text())
for rel in list(EDITS) + ["model_executor/models/b70_dflash_draft_int4.py"]:
    py_compile.compile(str(root / rel), doraise=True)
print(f"{MARKER}: patched {len(pending)} files + helper in {root}: {', '.join(pending) or '(helper only)'}")
