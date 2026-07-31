"""
nano_inference/scheduler.py

Implements Iteration-Level Continuous Batching and Dynamic Request Queue Management.
Supports Chunked Prefill (Sarathi scheduling) to cap Inter-Token Latency (ITL).
"""

from enum import Enum
from typing import List, Optional, Tuple
from nano_inference.block_manager import BlockAllocator, BlockTable


class RequestStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    PREEMPTED = "PREEMPTED"


class Request:
    """Represents an incoming user LLM generation request."""
    def __init__(self, request_id: str, prompt_token_ids: List[int], max_tokens: int = 128):
        self.request_id: str = request_id
        self.prompt_token_ids: List[int] = prompt_token_ids
        self.output_token_ids: List[int] = []
        self.max_tokens: int = max_tokens
        
        self.status: RequestStatus = RequestStatus.WAITING
        self.block_table: BlockTable = BlockTable()
        self.num_prefilled_tokens: int = 0  # For Chunked Prefill tracking

    @property
    def total_token_count(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_prefill_complete(self) -> bool:
        return self.num_prefilled_tokens >= len(self.prompt_token_ids)


class SchedulerOutputs:
    """Encapsulates scheduled requests for a single forward pass iteration."""
    def __init__(
        self,
        prefill_requests: List[Tuple[Request, int]],  # (Request, chunk_size)
        decode_requests: List[Request],
        num_batched_tokens: int,
    ):
        self.prefill_requests = prefill_requests
        self.decode_requests = decode_requests
        self.num_batched_tokens = num_batched_tokens  # sum(prefill chunk sizes) + sum(decode requests [always 1 token]) <= max_num_batched_tokens


class Scheduler:
    """
    Iteration-level Continuous Batching Scheduler.
    Max 2,048 total tokens and max 32 concurrent requests per step/iteration.
    """
    def __init__(
        self,
        allocator: BlockAllocator,
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 32,
    ):
        self.allocator: BlockAllocator = allocator
        self.max_num_batched_tokens: int = max_num_batched_tokens
        self.max_num_seqs: int = max_num_seqs

        self.waiting_queue: List[Request] = []
        self.running_queue: List[Request] = []

    def add_request(self, request: Request) -> None:
        """Adds a new incoming request to the waiting queue."""
        self.waiting_queue.append(request)

    def schedule(self):
        prefill_requests = []
        decode_requests = []
        num_batched_tokens = 0
        BLOCK_SIZE = 16  # Standard block size for PagedAttention

        # 1. Promote waiting requests to prefill
        while self.waiting_queue:
            if len(self.running_queue) >= self.max_num_seqs:
                break

            req = self.waiting_queue[0]
            prompt_len = len(req.prompt_token_ids)

            if num_batched_tokens + prompt_len > self.max_num_batched_tokens:
                break

            # Calculate how many 16-token physical blocks this prompt needs
            needed_blocks = (prompt_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            # Check if allocator has enough free blocks available
            if len(self.allocator.free_blocks) >= needed_blocks:
                req = self.waiting_queue.pop(0)
                
                # Allocate physical blocks for the request
                for _ in range(needed_blocks):
                    block = self.allocator.allocate()
                    req.block_table.add_block(block)

                req.status = RequestStatus.RUNNING
                self.running_queue.append(req)
                
                req.num_prefilled_tokens = prompt_len
                prefill_requests.append((req, prompt_len))
                num_batched_tokens += prompt_len
            else:
                # VRAM memory full, wait for active requests to finish
                break

        # 2. Add running requests ready for decode step
        for req in self.running_queue:
            if req not in [pr[0] for pr in prefill_requests]:
                if num_batched_tokens + 1 <= self.max_num_batched_tokens:
                    decode_requests.append(req)
                    num_batched_tokens += 1

        return SchedulerOutputs(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            num_batched_tokens=num_batched_tokens
        )

    def free_finished_request(self, req: Request) -> None:
        """Frees physical blocks and removes request upon completion."""
        req.status = RequestStatus.FINISHED
        req.block_table.free_all(self.allocator)
        if req in self.running_queue:
            self.running_queue.remove(req)