"""
tests/test_scheduler.py
"""

from nano_inference.block_manager import BlockAllocator
from nano_inference.scheduler import Request, Scheduler, RequestStatus


def test_continuous_batching_scheduler():
    allocator = BlockAllocator(total_num_blocks=10, block_size=16)
    # Set max_num_seqs=1 so we can clearly see req1 decode while req2 waits
    scheduler = Scheduler(allocator=allocator, max_num_batched_tokens=32, max_num_seqs=1)

    # Request 1: 20 prompt tokens (requires 2 physical blocks)
    req1 = Request(request_id="req-1", prompt_token_ids=list(range(20)), max_tokens=5)
    # Request 2: 10 prompt tokens
    req2 = Request(request_id="req-2", prompt_token_ids=list(range(10)), max_tokens=5)

    scheduler.add_request(req1)
    scheduler.add_request(req2)

    # Iteration 1: req1 gets scheduled for prefill (max_num_seqs limit = 1)
    output1 = scheduler.schedule()
    assert len(output1.prefill_requests) == 1
    assert output1.prefill_requests[0][0].request_id == "req-1"
    assert req1.status == RequestStatus.RUNNING
    print(f"Iteration 1 Batched Tokens: {output1.num_batched_tokens}")

    # Simulate token generation step for req1
    req1.output_token_ids.append(101)

    # Iteration 2: req1 executes DECODE step
    output2 = scheduler.schedule()
    assert len(output2.decode_requests) == 1
    assert output2.decode_requests[0].request_id == "req-1"
    print(f"Iteration 2 Scheduled Decodes: {[r.request_id for r in output2.decode_requests]}")

    # Finish req1 and free its memory
    scheduler.free_finished_request(req1)

    # Iteration 3: Now req2 is scheduled for prefill from waiting queue
    output3 = scheduler.schedule()
    assert len(output3.prefill_requests) == 1
    assert output3.prefill_requests[0][0].request_id == "req-2"
    print(f"Iteration 3 Prefilled Next Req: {output3.prefill_requests[0][0].request_id}")

    scheduler.free_finished_request(req2)
    assert allocator.num_free_blocks == 10
    print("✅ Continuous Batching Scheduler Unit Test Passed!")


if __name__ == "__main__":
    test_continuous_batching_scheduler()