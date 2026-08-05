"""
nano_inference/rmsnorm.py

Fused Triton RMSNorm (Root Mean Square Normalization) with PyTorch reference fallback.
Optimizes memory bandwidth by computing normalization and weight-scaling in a single pass.
"""

import torch
import torch.nn as nn

# Graceful import check for Triton
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _rmsnorm_kernel(
        X_ptr,          # Pointer to input tensor X [M, N]
        Y_ptr,          # Pointer to output tensor Y [M, N]
        W_ptr,          # Pointer to scale weight tensor W [N]
        stride_x_row,   # Stride of rows in X
        stride_y_row,   # Stride of rows in Y
        N_cols,         # Number of columns in X (hidden dimension)
        eps,            # Epsilon float value
        BLOCK_SIZE: tl.constexpr,  # Next power of 2 >= N_cols
    ):
        # The program ID (1D) points to the row index
        row_idx = tl.program_id(0)

        # Generate column offsets
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N_cols

        # Load the input row of X [BLOCK_SIZE]
        x_ptrs = X_ptr + row_idx * stride_x_row + col_offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Compute root-mean-square: sqrt(1/N * sum(x^2) + eps)
        x_sq = x * x
        variance = tl.sum(x_sq, axis=0) / N_cols
        rsqrt = 1.0 / tl.sqrt(variance + eps)

        # Load weights W [BLOCK_SIZE]
        w_ptrs = W_ptr + col_offsets
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Normalize and apply scale weights
        y = x * rsqrt * w

        # Store normalized result to Y [BLOCK_SIZE]
        y_ptrs = Y_ptr + row_idx * stride_y_row + col_offsets
        tl.store(y_ptrs, y, mask=mask)


class TritonRMSNorm(nn.Module):
    """Fused Triton RMSNorm layer."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input to 2D: [batch_size * seq_len, hidden_size]
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        M, N = x_flat.shape

        if not TRITON_AVAILABLE or not x.is_cuda:
            # Fallback to PyTorch reference
            input_dtype = x_flat.dtype
            x_flat_fp32 = x_flat.to(torch.float32)
            variance = x_flat_fp32.pow(2).mean(-1, keepdim=True)
            y = x_flat_fp32 * torch.rsqrt(variance + self.eps) * self.weight.to(torch.float32)
            return y.to(input_dtype).view(orig_shape)

        y_flat = torch.empty_like(x_flat)
        grid = (M,)
        
        # BLOCK_SIZE must be a power of 2 greater than or equal to N
        block_size = triton.next_power_of_2(N)

        _rmsnorm_kernel[grid](
            x_flat,
            y_flat,
            self.weight,
            x_flat.stride(0),
            y_flat.stride(0),
            N,
            self.eps,
            BLOCK_SIZE=block_size,
        )
        return y_flat.view(orig_shape)


class PyTorchRMSNorm(nn.Module):
    """Pure PyTorch reference RMSNorm implementation (for correctness validation)."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        y = x_fp32 * torch.rsqrt(variance + self.eps) * self.weight.to(torch.float32)
        return y.to(input_dtype)
