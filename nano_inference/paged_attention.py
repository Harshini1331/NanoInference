"""
nano_inference/paged_attention.py

Production-grade PagedAttention decode kernel using Triton GPU acceleration
with online Softmax and vectorized memory loading, with PyTorch reference fallback.
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
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 16}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_SIZE': 16}, num_warps=8, num_stages=4),
        ],
        key=['head_dim']
    )
    @triton.jit
    def _paged_attention_decode_kernel(
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
        BLOCK_SIZE: tl.constexpr,  # Autotuned symbol
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
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        if seq_idx >= num_seqs or head_idx >= num_heads:
            return

        context_len = tl.load(context_lens_ptr + seq_idx)
        num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        dim_offsets = tl.arange(0, head_dim)
        
        # Load Query vector [1, head_dim]
        q_ptr = query_ptr + seq_idx * stride_q_seq + head_idx * stride_q_head + dim_offsets
        q = tl.load(q_ptr)

        # Accumulators for Online Softmax
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([head_dim], dtype=tl.float32)

        slot_offsets = tl.arange(0, BLOCK_SIZE)

        # Loop over physical KV blocks
        for block_idx in range(num_blocks):
            physical_block_id = tl.load(
                block_tables_ptr + seq_idx * stride_bt_seq + block_idx
            )

            # Mask out invalid padding tokens in the final block
            token_positions = block_idx * BLOCK_SIZE + slot_offsets
            mask = token_positions < context_len

            # Vectorized Key Load: [BLOCK_SIZE, head_dim]
            k_ptrs = (
                key_cache_ptr
                + physical_block_id * stride_k_block
                + head_idx * stride_k_head
                + slot_offsets[:, None] * stride_k_tok
                + dim_offsets[None, :]
            )
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)

            # Compute Attention Scores for the block: q * K^T -> [BLOCK_SIZE]
            qk = tl.sum(q[None, :] * k, axis=1) * scale
            qk = tl.where(mask, qk, -float("inf"))

            # Online Softmax Update
            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            p = tl.exp(qk - m_ij)
            l_ij = tl.sum(p, 0)

            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            # Vectorized Value Load: [BLOCK_SIZE, head_dim]
            v_ptrs = (
                value_cache_ptr
                + physical_block_id * stride_v_block
                + head_idx * stride_v_head
                + slot_offsets[:, None] * stride_v_tok
                + dim_offsets[None, :]
            )
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)

            # Accumulate Attention-Weighted Values
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_ij

        # Final Normalization
        if l_i > 0:
            acc = acc / l_i
            out_ptr_final = (
                out_ptr + seq_idx * stride_out_seq + head_idx * stride_out_head + dim_offsets
            )
            tl.store(out_ptr_final, acc.to(out_ptr.dtype.element_ty))


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

            out = torch.empty_like(query)
            grid = (num_seqs, num_heads)

            _paged_attention_decode_kernel[grid](
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
                BLOCK_SIZE=block_size,
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
            # Fall back to PyTorch gather on Triton runtime mismatch or execution error
            pass

    return pytorch_paged_attention_fallback(
        query, key_cache, value_cache, block_tables, context_lens, scale, block_size
    )