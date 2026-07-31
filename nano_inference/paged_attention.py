"""
nano_inference/paged_attention.py

Custom PagedAttention implementation with Triton GPU kernel acceleration
and PyTorch reference fallback.
"""

import torch

# Graceful import check for Triton (Windows/Linux cross-compatibility)
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _paged_attention_decode_kernel(
        exp_sums_ptr,
        max_scores_ptr,
        out_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        scale,
        num_seqs,
        num_heads,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        max_num_blocks_per_seq: tl.constexpr,
        stride_q_seq,
        stride_q_head,
        stride_k_block,
        stride_k_head,
        stride_k_tok,
        stride_v_block,
        stride_v_head,
        stride_v_tok,
        stride_out_seq,
        stride_out_head,
        stride_bt_seq,
    ):
        """Triton kernel for single-query token decode attention using Paged KV Cache."""
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        if seq_idx >= num_seqs or head_idx >= num_heads:
            return

        context_len = tl.load(context_lens_ptr + seq_idx)
        num_blocks = (context_len + block_size - 1) // block_size

        dim_offsets = tl.arange(0, head_dim)
        q_ptr = query_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head + dim_offsets
        q = tl.load(q_ptr)

        max_score = -float("inf")
        acc = tl.zeros([head_dim], dtype=tl.float32)
        exp_sum = 0.0

        for block_idx in range(num_blocks):
            physical_block_id = tl.load(
                block_tables_ptr + seq_idx * stride_bt_seq + block_idx
            )

            for slot in range(block_size):
                token_pos = block_idx * block_size + slot
                if token_pos < context_len:
                    k_ptr = (
                        key_cache_ptr
                        + physical_block_id * stride_k_block
                        + head_idx * stride_k_head
                        + slot * stride_k_tok
                        + dim_offsets
                    )
                    k = tl.load(k_ptr)

                    score = tl.sum(q * k) * scale

                    if score > max_score:
                        alpha = tl.exp(max_score - score)
                        acc = acc * alpha
                        exp_sum = exp_sum * alpha
                        max_score = score

                    p = tl.exp(score - max_score)
                    exp_sum += p

                    v_ptr = (
                        value_cache_ptr
                        + physical_block_id * stride_v_block
                        + head_idx * stride_v_head
                        + slot * stride_v_tok
                        + dim_offsets
                    )
                    v = tl.load(v_ptr)
                    acc += p * v

        if exp_sum > 0:
            out = acc / exp_sum
            out_ptr_final = (
                out_ptr + seq_idx * stride_out_seq + head_idx * stride_out_head + dim_offsets
            )
            tl.store(out_ptr_final, out.to(out_ptr.dtype.element_ty))


def pytorch_paged_attention_fallback(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int = 16,
) -> torch.Tensor:
    """PyTorch reference fallback implementation for PagedAttention gathering."""
    num_seqs, num_heads, head_dim = query.shape
    out = torch.zeros_like(query)

    for i in range(num_seqs):
        c_len = context_lens[i].item()
        num_b = (c_len + block_size - 1) // block_size
        p_blocks = block_tables[i, :num_b]

        # Gather keys and values across physical block table
        k_blocks = key_cache[p_blocks]  # [num_b, num_heads, block_size, head_dim]
        v_blocks = value_cache[p_blocks]

        # Reshape to continuous sequence
        keys = k_blocks.transpose(0, 1).reshape(num_heads, num_b * block_size, head_dim)[:, :c_len, :]
        values = v_blocks.transpose(0, 1).reshape(num_heads, num_b * block_size, head_dim)[:, :c_len, :]

        # Scaled dot product attention
        q_i = query[i].unsqueeze(1)  # [num_heads, 1, head_dim]
        attn_scores = torch.matmul(q_i, keys.transpose(-2, -1)) * scale  # [num_heads, 1, c_len]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, values).squeeze(1)  # [num_heads, head_dim]

        out[i] = context

    return out


def paged_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int = 16,
) -> torch.Tensor:
    """Dispatches to Triton GPU kernel if available, otherwise falls back to PyTorch gather."""
    if TRITON_AVAILABLE and query.is_cuda:
        try:
            num_seqs, num_heads, head_dim = query.shape
            max_num_blocks_per_seq = block_tables.shape[1]

            out = torch.empty_like(query)
            exp_sums = torch.empty((num_seqs, num_heads), dtype=torch.float32, device=query.device)
            max_scores = torch.empty((num_seqs, num_heads), dtype=torch.float32, device=query.device)

            grid = (num_seqs, num_heads)

            _paged_attention_decode_kernel[grid](
                exp_sums,
                max_scores,
                out,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                scale,
                num_seqs,
                num_heads,
                head_dim=head_dim,
                block_size=block_size,
                max_num_blocks_per_seq=max_num_blocks_per_seq,
                stride_q_seq=query.stride(0),
                stride_q_head=query.stride(1),
                stride_k_block=key_cache.stride(0),
                stride_k_head=key_cache.stride(1),
                stride_k_tok=key_cache.stride(2),
                stride_v_block=value_cache.stride(0),
                stride_v_head=value_cache.stride(1),
                stride_v_tok=value_cache.stride(2),
                stride_out_seq=out.stride(0),
                stride_out_head=out.stride(1),
                stride_bt_seq=block_tables.stride(0),
            )
            return out
        except Exception:
            # Fall back to PyTorch gather on Triton runtime mismatch
            pass

    return pytorch_paged_attention_fallback(
        query, key_cache, value_cache, block_tables, context_lens, scale, block_size
    )