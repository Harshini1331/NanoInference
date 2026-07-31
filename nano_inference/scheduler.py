"""
nano_inference/scheduler.py

Implements Iteration-Level Continuous Batching, Chunked Prefill (Sarathi Scheduling),
and Automatic Prefix Caching (APC).
"""

from enum import Enum
from typing import List, Optional, Tuple
from nano_inference.block_manager import BlockAllocator, BlockTable
from nano_inference.metrics import PREFIX_CACHE_HITS, PREFIX_CACHE_MISSES


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
        self.num_prefilled_tokens: int = 0  # Tracks processed prompt tokens for Chunked Prefill

    @property
    def total_token_count(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_prefill_complete(self) -> bool:
        return self.num_prefilled_tokens >= len(self.prompt_token_ids)

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(0, len(self.prompt_token_ids) - self.num_prefilled_tokens)


class SchedulerOutputs:
    """Encapsulates scheduled requests for a single forward pass iteration."""
    def __init__(
        self,
        prefill_requests: List[Tuple[Request, int]],  # List of (Request, chunk_size)
        decode_requests: List[Request],
        num_batched_tokens: int,
    ):
        self.prefill_requests = prefill_requests
        self.decode_requests = decode_requests
        self.num_batched_tokens = num_batched_tokens


class Scheduler:
    """
    Iteration-level Continuous Batching Scheduler with Chunked Prefill (Sarathi)
    and Automatic Prefix Caching (APC).
    """
    def __init__(
        self,
        allocator: BlockAllocator,
        max_num_batched_tokens: int = 2048,
        max_num_seqs: int = 32,
        chunk_size: int = 256,  # Max tokens per prefill chunk pass
    ):
        self.allocator: BlockAllocator = allocator
        self.max_num_batched_tokens: int = max_num_batched_tokens
        self.max_num_seqs: int = max_num_seqs
        self.chunk_size: int = chunk_size

        self.waiting_queue: List[Request] = []
        self.running_queue: List[Request] = []

    def add_request(self, request: Request) -> None:
        """Adds a new incoming request to the waiting queue."""
        self.waiting_queue.append(request)

    def schedule(self) -> SchedulerOutputs:
        prefill_requests = []
        decode_requests = []
        num_batched_tokens = 0
        BLOCK_SIZE = 16  # Standard block size for PagedAttention

        # 1. Schedule Chunked Prefill requests
        while self.waiting_queue:
            if len(self.running_queue) >= self.max_num_seqs:
                break

            req = self.waiting_queue[0]
            remaining = req.remaining_prefill_tokens

            # Cap chunk size to remaining prefill tokens and max_num_batched_tokens limit
            current_chunk = min(remaining, self.chunk_size)
            if num_batched_tokens + current_chunk > self.max_num_batched_tokens:
                break

            # Calculate physical 16-token blocks needed for this chunk
            start_token_idx = req.num_prefilled_tokens
            end_token_idx = start_token_idx + current_chunk
            needed_blocks = (end_token_idx + BLOCK_SIZE - 1) // BLOCK_SIZE - len(req.block_table.physical_blocks)

            # Allocate blocks if memory available
            if len(self.allocator.free_blocks) >= needed_blocks:
                req = self.waiting_queue.pop(0)

                # Allocate/reuse physical blocks using content hashing & APC for this chunk
                for idx in range(start_token_idx, end_token_idx, BLOCK_SIZE):
                    block, is_hit = req.block_table.allocate_slot_for_token(
                        token_index=idx,
                        tokens=req.prompt_token_ids,
                        allocator=self.allocator,
                    )
                    if block is not None:
                        if is_hit:
                            PREFIX_CACHE_HITS.inc()
                        else:
                            PREFIX_CACHE_MISSES.inc()

                req.num_prefilled_tokens += current_chunk
                prefill_requests.append((req, current_chunk))
                num_batched_tokens += current_chunk

                # If prefill is incomplete, keep in waiting_queue at front for next iteration;
                # otherwise promote to running_queue for decode
                if not req.is_prefill_complete:
                    self.waiting_queue.insert(0, req)
                else:
                    req.status = RequestStatus.RUNNING
                    self.running_queue.append(req)
            else:
                # VRAM full, wait for active requests to finish
                break

        # 2. Schedule running decode requests
        for req in self.running_queue:
            if req not in [pr[0] for pr in prefill_requests]:
                if num_batched_tokens + 1 <= self.max_num_batched_tokens:
                    # Allocate block boundary expansion for next decode token if needed
                    current_tokens = req.prompt_token_ids + req.output_token_ids
                    next_token_idx = len(current_tokens)
                    
                    block, is_hit = req.block_table.allocate_slot_for_token(
                        token_index=next_token_idx,
                        tokens=current_tokens,
                        allocator=self.allocator,
                    )
                    if block is not None:
                        if is_hit:
                            PREFIX_CACHE_HITS.inc()
                        else:
                            PREFIX_CACHE_MISSES.inc()

                    decode_requests.append(req)
                    num_batched_tokens += 1

        return SchedulerOutputs(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            num_batched_tokens=num_batched_tokens,
        )

    def free_finished_request(self, req: Request) -> None:
        """Frees physical blocks and removes request upon completion."""
        req.status = RequestStatus.FINISHED
        req.block_table.free_all(self.allocator)
        if req in self.running_queue:
            self.running_queue.remove(req)
        if req in self.waiting_queue:
            self.waiting_queue.remove(req)