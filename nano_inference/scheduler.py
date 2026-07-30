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

    def schedule(self) -> SchedulerOutputs:
        """
        Main scheduling step invoked on every forward pass.
        Selects requests for prefill/decode based on VRAM capacity & token budget.
        """
        scheduled_prefills: List[Tuple[Request, int]] = []
        scheduled_decodes: List[Request] = []
        num_batched_tokens = 0

        # 1. Schedule active DECODE requests first to maintain continuous output stream
        for req in list(self.running_queue):
            if req.status == RequestStatus.RUNNING:
                # Determine index for token to be appended in this decode step
                next_token_index = req.total_token_count
                
                # Pre-allocate physical VRAM block if hitting a block boundary
                if next_token_index % req.block_table.block_size == 0:
                    if self.allocator.num_free_blocks < 1:
                        # VRAM full: Preempt or stop scheduling decodes
                        break
                    req.block_table.allocate_slot_for_token(next_token_index, self.allocator)

                scheduled_decodes.append(req)
                num_batched_tokens += 1

        # 2. Schedule WAITING/PREFILL requests into remaining token/slot budget
        while self.waiting_queue and len(scheduled_decodes) + len(scheduled_prefills) < self.max_num_seqs:
            req = self.waiting_queue[0]
            remaining_prompt = len(req.prompt_token_ids) - req.num_prefilled_tokens

            # Calculate chunk size within remaining batched token budget
            available_token_budget = self.max_num_batched_tokens - num_batched_tokens
            if available_token_budget <= 0:
                break

            chunk_size = min(remaining_prompt, available_token_budget)
            
            # Allocate blocks for this prompt chunk
            start_idx = req.num_prefilled_tokens
            for idx in range(start_idx, start_idx + chunk_size):
                req.block_table.allocate_slot_for_token(idx, self.allocator)

            req.num_prefilled_tokens += chunk_size
            num_batched_tokens += chunk_size
            scheduled_prefills.append((req, chunk_size))

            # Move request from waiting to running if prefill is fully chunked
            if req.is_prefill_complete:
                req.status = RequestStatus.RUNNING
                self.running_queue.append(self.waiting_queue.pop(0))
            else:
                break  # Partial chunk allocated; continue next step

        return SchedulerOutputs(
            prefill_requests=scheduled_prefills,
            decode_requests=scheduled_decodes,
            num_batched_tokens=num_batched_tokens,
        )

    def free_finished_request(self, req: Request) -> None:
        """Frees physical blocks and removes request upon completion."""
        req.status = RequestStatus.FINISHED
        req.block_table.free_all(self.allocator)
        if req in self.running_queue:
            self.running_queue.remove(req)