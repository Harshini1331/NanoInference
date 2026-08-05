"""
tests/test_correctness_gate.py

Correctness Gate evaluating numerical divergence between Triton kernels and PyTorch.
Compares absolute, relative, and Root-Mean-Square differences across precision datatypes.
"""

import torch
from nano_inference.rmsnorm import TritonRMSNorm, PyTorchRMSNorm
from nano_inference.paged_attention import paged_attention_decode, pytorch_paged_attention_fallback


def calculate_divergence_metrics(triton_out: torch.Tensor, pytorch_out: torch.Tensor):
    """Computes L1, L2, and L_inf divergence metrics."""
    diff = (triton_out.float() - pytorch_out.float()).abs()
    
    l_inf = diff.max().item()                    # Max absolute difference
    l_1 = diff.mean().item()                     # Mean absolute error
    l_2 = torch.sqrt((diff ** 2).mean()).item()  # Root Mean Square error
    
    return l_1, l_2, l_inf


def run_correctness_gate():
    print("=" * 75)
    print("[GATE] NANOINFERENCE NUMERICAL CORRECTNESS EVALUATION GATE")
    print("=" * 75)
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA is required to evaluate Triton kernels.")
        return
        
    device = "cuda"
    dtypes = [torch.float16, torch.bfloat16]
    
    # ----------------------------------------------------
    # TEST 1: Triton RMSNorm vs PyTorch Eager RMSNorm
    # ----------------------------------------------------
    print("\n[EVAL] Evaluating Triton RMSNorm Kernel correctness...")
    batch, seq_len, hidden_size = 16, 512, 896
    
    for dtype in dtypes:
        x = torch.randn((batch, seq_len, hidden_size), dtype=dtype, device=device) * 5.0
        
        triton_rmsnorm = TritonRMSNorm(hidden_size).to(device)
        pytorch_rmsnorm = PyTorchRMSNorm(hidden_size).to(device)
        
        # Clone weights to align them
        pytorch_rmsnorm.weight.data.copy_(triton_rmsnorm.weight.data)
        
        with torch.no_grad():
            t_out = triton_rmsnorm(x)
            p_out = pytorch_rmsnorm(x)
            
        l1, l2, linf = calculate_divergence_metrics(t_out, p_out)
        
        # Target tolerances
        tol = 4e-3 if dtype == torch.float16 else 4e-2
        status = "PASS" if linf < tol else "FAIL"
        
        print(f"[{dtype}] L1: {l1:8.6f} | L2: {l2:8.6f} | L_inf: {linf:8.6f} | Tolerance Limit: {tol} -> [{status}]")
        assert linf < tol, f"Triton RMSNorm numerical divergence too high in {dtype}"

    # ----------------------------------------------------
    # TEST 2: Triton PagedAttention vs PyTorch Fallback
    # ----------------------------------------------------
    print("\n[EVAL] Evaluating Triton PagedAttention Kernel correctness...")
    num_seqs = 8
    num_heads = 16
    head_dim = 64
    block_size = 16
    context_len = 128
    scale = 1.0 / (head_dim ** 0.5)
    
    num_blocks = (context_len + block_size - 1) // block_size
    
    for dtype in dtypes:
        query = torch.randn((num_seqs, num_heads, head_dim), dtype=dtype, device=device)
        key_cache = torch.randn((num_seqs * num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)
        value_cache = torch.randn((num_seqs * num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)
        
        block_tables = torch.zeros((num_seqs, num_blocks), dtype=torch.int32, device=device)
        for i in range(num_seqs):
            block_tables[i] = torch.arange(i * num_blocks, (i + 1) * num_blocks, dtype=torch.int32)
            
        context_lens = torch.full((num_seqs,), context_len, dtype=torch.int32, device=device)
        
        with torch.no_grad():
            t_out = paged_attention_decode(query, key_cache, value_cache, block_tables, context_lens, scale, block_size)
            p_out = pytorch_paged_attention_fallback(query, key_cache, value_cache, block_tables, context_lens, scale, block_size)
            
        l1, l2, linf = calculate_divergence_metrics(t_out, p_out)
        
        tol = 4e-3 if dtype == torch.float16 else 4e-2
        status = "PASS" if linf < tol else "FAIL"
        
        print(f"[{dtype}] L1: {l1:8.6f} | L2: {l2:8.6f} | L_inf: {linf:8.6f} | Tolerance Limit: {tol} -> [{status}]")
        assert linf < tol, f"Triton PagedAttention divergence too high in {dtype}"

    print("\n[SUCCESS] ALL CORRECTNESS EVALUATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_correctness_gate()
