"""
tests/test_model_runner.py
"""

import torch
from nano_inference.block_manager import BlockAllocator
from nano_inference.scheduler import Request, Scheduler
from nano_inference.model_runner import ModelRunner, PhysicalKVCachePool


def test_model_runner_execution():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running ModelRunner test on device: {device}")

    allocator = BlockAllocator(total_num_blocks=20, block_size=16)
    scheduler = Scheduler(allocator=allocator, max_num_batched_tokens=1024, max_num_seqs=4)

    # Load a lightweight model for fast local verification
    runner = ModelRunner(model_name="Qwen/Qwen2.5-0.5B-Instruct", device=device)

    # Initialize physical KV cache pool
    kv_pool = PhysicalKVCachePool(
        num_blocks=20,
        block_size=16,
        num_layers=runner.num_layers,
        num_kv_heads=runner.num_kv_heads,
        head_dim=runner.head_dim,
        device=device,
    )

    # Create user request
    prompt_text = "What is the capital of France?"
    prompt_tokens = runner.tokenizer.encode(prompt_text)
    req = Request(request_id="req-1", prompt_token_ids=prompt_tokens, max_tokens=10)
    scheduler.add_request(req)

    # Step 1: Prefill pass
    outputs = scheduler.schedule()
    prefill_tokens = runner.prefill_step(outputs.prefill_requests, kv_pool)
    print(f"Generated first token after Prefill: {runner.tokenizer.decode(prefill_tokens)}")

    # Step 2: Decode pass
    outputs_decode = scheduler.schedule()
    decode_tokens = runner.decode_step(outputs_decode.decode_requests, kv_pool)
    print(f"Generated second token after Decode step: {runner.tokenizer.decode(decode_tokens)}")

    scheduler.free_finished_request(req)
    print("✅ ModelRunner Execution Test Passed!")


if __name__ == "__main__":
    test_model_runner_execution()