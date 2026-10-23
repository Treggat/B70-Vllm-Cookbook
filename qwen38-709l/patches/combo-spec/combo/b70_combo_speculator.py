# 2026-10-21 B70_COMBO_SPEC: combined copy + MTP speculator for the 709L seat (vLLM 0.32 V2 runner).
# Installed as vllm/v1/worker/gpu/spec_decode/b70_combo/speculator.py by patch_combo_spec_032.py and built by
# init_speculator only when B70_COMBO_SPEC=1 and method == "mtp".
#
# The scheduler (v1/core/sched/b70_combo_policy.py) picks the draft count K per step for the whole batch:
#   K < max (MTP step, K = 5):  plain MTPSpeculator, unchanged.
#   K == max (copy step, 15):   MTP runs step 0 only (B70_COMBO_MTP_IN_COPY=1: all K_MTP steps). Step 0 is the draft
#                               "prefill" over every token the target just verified, so the MTP layer's KV/hidden
#                               state stays complete after a 16-token step (the catch-up). Then the n-gram scan of
#                               NgramGPUSpeculator runs, and per request:
#                                 copy  = longest match >= USE_MIN_N tokens, >= WORKER_MIN_AVAIL tokens after it, and
#                                         (B70_COMBO_AGREE) its first copied token == MTP's first draft
#                                 then  drafts = the 15 copied tokens
#                                 else  drafts = MTP's drafts, the rest padded with its last draft (a real token id;
#                                         the rejection sampler simply rejects them)
# Greedy drafts (prod) are one-hot on both paths, so mixing them in one verify is lossless.
# draft_sample_method "probabilistic" (2026-10-22, prob-draft/): the rejection sampler reads the cached draft logits
# self.draft_logits[req, col] for every draft. On a copy step the MTP pass filled only cols < km (and those hold MTP's
# logits even for rows that copy); every other col still holds an older step's logits. So after the finalize every
# draft that is not MTP's own (copy rows: all cols; MTP rows: cols >= km) gets a one-hot row (0 at the draft token,
# -inf elsewhere): q = 1 there, exactly like a greedy draft, so the mix stays lossless. No-op when draft_logits is None.
# Verify q_len is 16 on copy steps (stock XPU spec fast path; spec_packed's B70_SPEC_PACKED_QLENS guard keeps it
# unpacked) and 6 on MTP steps (packed).
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import _ngram_scan_kernel

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)
_E = os.environ.get


@triton.jit
def _combo_finalize_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int64  [B]
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    scratch_ptr,  # *int64  [B, scratch_stride]
    scratch_stride,
    mtp_ptr,  # *int64  [>=B, mtp_stride]  MTP drafts, batch order
    mtp_stride,
    out_ptr,  # *int64  [B, K]
    use_ptr,  # *int32  [B]  1 = copied drafts
    L,
    N_BLOCKS,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    N_BLOCKS_PO2: tl.constexpr,
    KM: tl.constexpr,
    USE_MIN_N: tl.constexpr,
    MIN_AVAIL: tl.constexpr,
    AGREE: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    Lp1 = tl.cast(L, tl.int64) + 1
    NB = tl.cast(N_BLOCKS, tl.int64)
    req_state_idx = tl.load(idx_mapping_ptr + b).to(tl.int64)

    nb_iota = tl.arange(0, N_BLOCKS_PO2).to(tl.int64)
    block_scores = tl.load(
        scratch_ptr + b * scratch_stride + nb_iota, mask=nb_iota < NB, other=0
    )
    score = tl.max(block_scores, axis=0)
    seq_len = tl.load(total_len_ptr + req_state_idx).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)

    has_match = score > 0
    s1 = score - 1
    best_n = tl.where(has_match, s1 // Lp1, tl.zeros_like(s1))
    best_pos = tl.where(has_match, s1 - best_n * Lp1, tl.zeros_like(s1))
    draft_start = tl.where(has_match, best_pos + best_n, tl.zeros_like(s1))
    tokens_avail = tl.maximum(seq_len - draft_start, 0)

    k_iota = tl.arange(0, K_PO2).to(tl.int64)
    k_in = k_iota < K
    row_off = req_state_idx * token_ids_stride
    gather_idx = tl.minimum(draft_start + k_iota, tl.cast(L, tl.int64) - 1)
    slot_valid = (k_iota < tokens_avail) & has_match & k_in
    gathered = tl.load(token_ids_ptr + row_off + gather_idx, mask=slot_valid, other=0).to(
        tl.int64
    )

    mtp_row = tl.load(mtp_ptr + b * mtp_stride + k_iota, mask=k_iota < KM, other=0).to(
        tl.int64
    )
    mtp0 = tl.load(mtp_ptr + b * mtp_stride).to(tl.int64)
    mtp_last = tl.load(mtp_ptr + b * mtp_stride + (KM - 1)).to(tl.int64)
    ng0 = tl.sum(tl.where(k_iota == 0, gathered, tl.zeros_like(gathered)), axis=0)

    use_copy = (
        (num_sampled > 0) & has_match & (best_n >= USE_MIN_N) & (tokens_avail >= MIN_AVAIL)
    )
    if AGREE:
        use_copy = use_copy & (ng0 == mtp0)
    # Copied slots past the end of the copy source repeat the last copied token.
    last_valid = tl.sum(
        tl.where(k_iota == tl.minimum(tokens_avail, K) - 1, gathered, tl.zeros_like(gathered)),
        axis=0,
    )
    copied = tl.where(slot_valid, gathered, last_valid)
    mtp_out = tl.where(k_iota < KM, mtp_row, mtp_last)
    out = tl.where(use_copy, copied, mtp_out)
    tl.store(out_ptr + b * K + k_iota, out, mask=k_in)
    tl.store(use_ptr + b, use_copy.to(tl.int32))


@triton.jit
def _combo_onehot_kernel(
    draft_logits_ptr,  # *head_dtype [max_num_reqs, num_cols, V]
    dl_stride_0,
    dl_stride_1,
    idx_mapping_ptr,  # *int64  [B]
    out_ptr,  # *int64  [B, K]  the drafts the verify will see
    use_ptr,  # *int32  [B]  1 = copied drafts
    V,
    K,
    KM,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64)
    use = tl.load(use_ptr + b)
    if use == 0 and col < KM:
        return  # MTP's own draft: its cached logits are the ones it was sampled from
    req_state_idx = tl.load(idx_mapping_ptr + b).to(tl.int64)
    tok = tl.load(out_ptr + b * K + col).to(tl.int64)
    offs = tl.program_id(2).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    val = tl.where(offs == tok, 0.0, float("-inf"))
    ptr = draft_logits_ptr + req_state_idx * dl_stride_0 + col * dl_stride_1 + offs
    tl.store(ptr, val.to(draft_logits_ptr.dtype.element_ty), mask=offs < V)


def write_onehot_draft_logits(
    draft_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    drafts: torch.Tensor,
    use_copy: torch.Tensor,
    num_reqs: int,
    km: int,
) -> None:
    """One-hot draft-logit rows for every non-MTP draft of a copy step (see the header). drafts [>=B, K] int64."""
    K = drafts.shape[1]
    V = draft_logits.shape[-1]
    assert drafts.stride(1) == 1 and drafts.stride(0) == K and K <= draft_logits.shape[1]
    BLOCK = 8192
    _combo_onehot_kernel[(num_reqs, K, triton.cdiv(V, BLOCK))](
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        idx_mapping,
        drafts,
        use_copy,
        V,
        K,
        km,
        BLOCK=BLOCK,
        num_warps=4,
    )


class B70ComboSpeculator(MTPSpeculator):
    """MTP-5 by default; on steps the scheduler gives the full K, copied drafts where they check out."""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, req_states: RequestState
    ):
        super().__init__(vllm_config, device)
        self.req_states = req_states
        self.k_mtp = int(_E("B70_COMBO_K_MTP", "5"))
        self.mtp_in_copy = _E("B70_COMBO_MTP_IN_COPY", "0") == "1"
        self.ng_min = int(_E("B70_COMBO_NG_MIN", "2"))
        self.ng_max = int(_E("B70_COMBO_NG_MAX", "5"))
        self.use_min_n = int(_E("B70_COMBO_USE_MIN_N", "3"))
        self.worker_min_avail = int(_E("B70_COMBO_WORKER_MIN_AVAIL", "4"))
        self.agree = _E("B70_COMBO_AGREE", "1") == "1"
        # A step is a copy step when the scheduler's K >= this (default: the configured K); smaller K = plain MTP-K.
        self.copy_min_k = int(_E("B70_COMBO_COPY_MIN_K", "0")) or self.num_speculative_steps
        self.debug = _E("B70_COMBO_DEBUG_WORKER") == "1"  # host sync per copy step: diagnosis only
        assert 1 <= self.ng_min <= self.use_min_n <= self.ng_max
        assert 1 <= self.k_mtp < self.num_speculative_steps, (
            "B70_COMBO_SPEC needs num_speculative_tokens (copy K) > B70_COMBO_K_MTP"
        )
        self.max_model_len_i = vllm_config.model_config.max_model_len
        # Copy steps only happen under the policy's MAX_CTX, so scan at most this far into a request.
        scan_ctx = min(self.max_model_len_i, int(_E("B70_COMBO_SCAN_CTX", "65536")))
        self.block_l = 256
        self.n_blocks = triton.cdiv(scan_ctx, self.block_l)
        self.scratch = torch.zeros(
            (self.max_num_reqs, self.n_blocks), dtype=torch.int64, device=device
        )
        self.combo_out = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=device,
        )
        self.combo_use = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._dbg_steps = 0
        self._dbg_used = 0
        self._dbg_rows = 0
        logger.info(
            "B70 combo speculator: copy K=%d / MTP K=%d, ngram %d-%d (use >= %d, avail >= %d), "
            "agree=%s, mtp_in_copy=%s, scan %d blocks, copy when K >= %d",
            self.num_speculative_steps, self.k_mtp, self.ng_min, self.ng_max,
            self.use_min_n, self.worker_min_avail, self.agree, self.mtp_in_copy,
            self.n_blocks, self.copy_min_k,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        k = self.num_speculative_steps if num_speculative_tokens is None else num_speculative_tokens
        copy_step = k >= self.copy_min_k and not dummy_run and not is_profile
        mtp = super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=(
                (self.k_mtp if self.mtp_in_copy else 1) if copy_step else k
            ),
        )
        if not copy_step:
            return mtp

        num_reqs = input_batch.num_reqs
        token_ids = self.req_states.all_token_ids.gpu
        idx_mapping = input_batch.idx_mapping
        total_len = self.req_states.total_len.gpu
        _ngram_scan_kernel[(num_reqs, self.n_blocks)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            total_len,
            num_sampled,
            self.scratch,
            self.scratch.stride(0),
            self.max_model_len_i,
            self.ng_min,
            self.ng_max,
            max(1, triton.next_power_of_2(self.ng_max)),
            self.block_l,
            num_warps=4,
            num_stages=2,
        )
        km = self.k_mtp if self.mtp_in_copy else 1
        _combo_finalize_kernel[(num_reqs,)](
            token_ids,
            token_ids.stride(0),
            idx_mapping,
            total_len,
            num_sampled,
            self.scratch,
            self.scratch.stride(0),
            mtp,
            mtp.stride(0),
            self.combo_out,
            self.combo_use,
            self.max_model_len_i,
            self.n_blocks,
            self.num_speculative_steps,
            triton.next_power_of_2(self.num_speculative_steps),
            max(1, triton.next_power_of_2(self.n_blocks)),
            km,
            self.use_min_n,
            self.worker_min_avail,
            self.agree,
            num_warps=2,
            num_stages=1,
        )
        if self.draft_logits is not None:  # probabilistic drafts only
            write_onehot_draft_logits(
                self.draft_logits, idx_mapping, self.combo_out, self.combo_use, num_reqs, km
            )
        if self.debug:
            self._dbg_steps += 1
            self._dbg_rows += num_reqs
            self._dbg_used += int(self.combo_use[:num_reqs].sum())  # host sync: debug only
            if self._dbg_steps % 64 == 0:
                logger.info(
                    "B70 combo: %d copy steps, %d/%d rows used copied drafts",
                    self._dbg_steps, self._dbg_used, self._dbg_rows,
                )
        return self.combo_out[:num_reqs]
