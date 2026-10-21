#!/usr/bin/env python3
"""vLLM 0.32 re-base of Vcvzgbzz/b70-cookbook dd083d5 patches/vllm/swa-prefix-cache.patch (MARKER B70_SWA_FINE_GRAINED).

The DFlash2 drafter's layers are all sliding-window (2048). vLLM's SlidingWindowManager answers prefix-cache lookups only
at whole-block granularity (supports_fine_grained_hash_lookup = False), and one such group turns fine-grained hits off
for the whole hybrid model (kv_cache_utils: "Disabling fine-grained prefix-cache hits because these KV cache managers
require block-aligned lookups"), which also loses the Gated DeltaNet prompt-tail checkpoints: a repeated long prompt
reuses nothing. With B70_SWA_FINE_GRAINED=1 the sliding-window group finds hits at hash-block (--prefix-match-unit)
granularity: the cached prompt-tail partial block plus the window's full blocks before it, and caches the partial tail
itself (FullAttentionManager._cache_partial_tail_block, unchanged in 0.32).

0.32 vs the 0.31 patch: the ClassVar, the fine-grained branch inside find_longest_cache_hit (after resolve_block_hashes;
0.32 added dcp/pcp kwargs around it) and the cache_blocks override (0.32 signature with replay_boundaries) are the same
mechanism. Extra guard here: the fine path also requires alignment_tokens == block_pool.hash_block_size (the raw hashes
are indexed at that unit); anything else falls through to the stock block-aligned path.

Gate off (default): the ClassVar stays False and nothing else is reachable, so the MTP seat is unaffected.
Target: $VLLM_ROOT (the vllm/ package dir) or argv[1]. Idempotent, fail-closed, py_compile.
"""
from __future__ import annotations

import os
import py_compile
import sys
from pathlib import Path

MARKER = "B70_SWA_FINE_GRAINED"
root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VLLM_ROOT", ""))
if not (root / "__init__.py").is_file():
    sys.exit(f"{MARKER}: VLLM_ROOT does not point at a vllm package dir: {root}")
p = root / "v1/core/single_type_kv_cache_manager.py"
if not p.is_file():
    sys.exit(f"{MARKER}: missing {p}")

EDITS = [
    (
        """class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
""",
        """class SlidingWindowManager(SingleTypeKVCacheManager):
    # B70_SWA_FINE_GRAINED: hash-block-granular hits (prompt-tail partial block + the window's full blocks).
    supports_fine_grained_hash_lookup: ClassVar[bool] = (
        __import__("os").environ.get("B70_SWA_FINE_GRAINED") == "1"
    )

    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
""",
    ),
    (
        """            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )

        # The number of contiguous blocks needed for a prefix cache hit.
        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
""",
        """            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
            alignment_tokens=alignment_tokens,
        )

        block_size = kv_cache_spec.block_size
        if (  # B70_SWA_FINE_GRAINED
            cls.supports_fine_grained_hash_lookup
            and alignment_tokens < block_size
            and block_size % alignment_tokens == 0
            and alignment_tokens == block_pool.hash_block_size
            and not isinstance(block_hashes, BlockHashListWithBlockSize)
        ):
            return cls._find_fine_grained_hit(
                block_hashes, max_length, kv_cache_group_ids, block_pool,
                kv_cache_spec, drop_eagle_block, alignment_tokens,
            )

        # The number of contiguous blocks needed for a prefix cache hit.
        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
""",
    ),
    (
        """        hit_length = len(computed_blocks[0]) * block_size
        return computed_blocks, hit_length

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
        dcp_world_size: int = 1,
        final_segment_end_block: int | None = None,
    ) -> list[bool] | None:
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
""",
        """        hit_length = len(computed_blocks[0]) * block_size
        return computed_blocks, hit_length

    @classmethod
    def _find_fine_grained_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: SlidingWindowSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        \"\"\"B70_SWA_FINE_GRAINED: longest hit at hash-block granularity: the block holding the hit end (a cached
        partial prompt tail or a cached full block, whose prefix covers it) plus every full block in the attention
        window before it; earlier blocks are null. With EAGLE the hit drops one hash unit and the window is checked
        for the reduced length.\"\"\"
        assert isinstance(block_hashes, Sequence)
        block_size = kv_cache_spec.block_size
        window = kv_cache_spec.sliding_window
        full_hashes = BlockHashListWithBlockSize(block_hashes, alignment_tokens, block_size)
        drop = alignment_tokens if drop_eagle_block else 0
        top = min(max_length // alignment_tokens, len(block_hashes))
        for fine_idx in range(top - 1, -1, -1):
            end = (fine_idx + 1) * alignment_tokens
            hit = end - drop
            if hit <= 0:
                break
            num_full_end, partial = divmod(end, block_size)
            tail = None
            if partial:
                # Any block cached at this or a later boundary of the same block covers our prefix through `end`.
                last_in_block = min((num_full_end + 1) * (block_size // alignment_tokens), len(block_hashes)) - 1
                for k in range(fine_idx, last_in_block + 1):
                    tail = block_pool.get_cached_block(block_hashes[k], kv_cache_group_ids)
                    if tail:
                        break
                if not tail:
                    continue
            first = max(0, hit - window + 1) // block_size
            window_blocks = []
            for j in range(first, num_full_end):
                cached = block_pool.get_cached_block(full_hashes[j], kv_cache_group_ids)
                if not cached:
                    break
                window_blocks.append(cached)
            if len(window_blocks) != num_full_end - first:
                continue
            computed: tuple[list[KVCacheBlock], ...] = tuple(
                [block_pool.null_block] * first for _ in range(len(kv_cache_group_ids))
            )
            for cached in window_blocks + ([tail] if tail else []):
                for out, blk in zip(computed, cached):
                    out.append(blk)
            num_blocks = cdiv(hit, block_size)
            for out in computed:
                del out[num_blocks:]
            return computed, hit
        return tuple([] for _ in range(len(kv_cache_group_ids))), 0

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        super().cache_blocks(
            request,
            num_tokens,
            retention_interval=retention_interval,
            replay_boundaries=replay_boundaries,
        )
        if self.supports_fine_grained_hash_lookup and self.block_size != self.block_pool.hash_block_size:
            # B70_SWA_FINE_GRAINED: register the prompt-tail partial block, as the full-attention group does
            FullAttentionManager._cache_partial_tail_block(self, request, num_tokens)

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
        dcp_world_size: int = 1,
        final_segment_end_block: int | None = None,
    ) -> list[bool] | None:
        assert isinstance(kv_cache_spec, SlidingWindowSpec)
""",
    ),
]

s = p.read_text()
if MARKER in s:
    for old, new in EDITS:
        if s.count(new) != 1 or (old not in new and old in s):
            sys.exit(f"{MARKER}: {p} carries the marker but is only partly patched; restore it from git")
    print(f"{MARKER}: already applied ({p})")
    sys.exit(0)
for old, new in EDITS:
    if s.count(old) != 1:
        sys.exit(f"{MARKER}: anchor not found exactly once in {p}:\n{old}")
    s = s.replace(old, new)
# names the new code relies on must already be imported in this module
for name in ("BlockHashListWithBlockSize", "BlockHashList", "cdiv", "Sequence", "ClassVar", "FullAttentionManager"):
    if name not in s:
        sys.exit(f"{MARKER}: {name} not available in {p}")
p.write_text(s)
py_compile.compile(str(p), doraise=True)
print(f"{MARKER}: patched {p}")
