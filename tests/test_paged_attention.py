import torch
from nano_inference.paged_attention import paged_attention_decode, TRITON_AVAILABLE

def test_paged_attention():
    # Enforce GPU requirement for PagedAttention GPU tests
    assert torch.cuda.is_available(), "❌ CUDA is NOT available! Check PyTorch GPU installation."
    
    device = "cuda"
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Triton Available: {TRITON_AVAILABLE}")

    num_seqs = 2
    num_heads = 4
    head_dim = 64
    block_size = 16

    # Create tensors directly on GPU VRAM
    query = torch.randn((num_seqs, num_heads, head_dim), dtype=torch.float16, device=device)
    key_cache = torch.randn((16, num_heads, block_size, head_dim), dtype=torch.float16, device=device)
    value_cache = torch.randn((16, num_heads, block_size, head_dim), dtype=torch.float16, device=device)

    block_tables = torch.tensor([[0, 1, 0, 0], [2, 3, 0, 0]], dtype=torch.int32, device=device)
    context_lens = torch.tensor([32, 28], dtype=torch.int32, device=device)

    out = paged_attention_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        scale=1.0 / (head_dim ** 0.5),
        block_size=block_size,
    )

    print(f"Output Tensor Device: {out.device}")
    print("PagedAttention Execution Successful on GPU! Output shape:", out.shape)

if __name__ == "__main__":
    test_paged_attention()