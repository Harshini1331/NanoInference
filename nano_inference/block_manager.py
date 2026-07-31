"""
Implements virtual memory block management for KV-Cache tensors.
Decouples logical sequence token positions from physical contiguous VRAM slots.
"""

from typing import Dict, List, Optional


class PhysicalTokenBlock:
    """Represents a fixed-size chunk of memory in physical GPU VRAM."""
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id: int = block_id
        self.block_size: int = block_size
        self.ref_count: int = 0  # Number of sequences referencing this block (e.g., prefix caching)

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockAllocator:
    """Manages the pool of available physical GPU VRAM blocks."""
    def __init__(self, total_num_blocks: int, block_size: int = 16):
        self.block_size: int = block_size
        self.total_num_blocks: int = total_num_blocks
        
        # Initialize free block pool
        self.free_blocks: List[PhysicalTokenBlock] = [
            PhysicalTokenBlock(block_id=i, block_size=block_size)
            for i in range(total_num_blocks)
        ]
        self.allocated_blocks: Dict[int, PhysicalTokenBlock] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self) -> PhysicalTokenBlock:
        """Allocates a free physical block from the VRAM pool."""
        if not self.free_blocks:
            raise MemoryError("Out of VRAM Physical Blocks! High concurrency limit reached.")
        
        block = self.free_blocks.pop(0)
        block.ref_count = 1
        self.allocated_blocks[block.block_id] = block
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        """Frees a physical block or decrements its reference count."""
        block.ref_count -= 1
        if block.ref_count <= 0:
            if block.block_id in self.allocated_blocks:
                del self.allocated_blocks[block.block_id]
            self.free_blocks.append(block)


class BlockTable:
    """
    Logical page table mapping a specific request's sequence of tokens
    to physical non-contiguous VRAM blocks.
    """
    def __init__(self, block_size: int = 16):
        self.block_size: int = block_size
        self.physical_blocks: List[PhysicalTokenBlock] = []

    def add_block(self, block: PhysicalTokenBlock) -> None:
        """Appends an allocated physical VRAM block to this request's page table."""
        self.physical_blocks.append(block)

    def allocate_slot_for_token(self, token_index: int, allocator: BlockAllocator) -> Optional[PhysicalTokenBlock]:
        """
        Allocates a new physical block when the token index reaches a block boundary.
        Example (block_size=16): Token 0, Token 16, Token 32 trigger a new block allocation.
        """
        if token_index % self.block_size == 0:
            new_block = allocator.allocate()
            self.add_block(new_block)
            return new_block
        return None

    def get_block_ids(self) -> List[int]:
        """Returns the physical block IDs mapped to this sequence."""
        return [b.block_id for b in self.physical_blocks]

    def free_all(self, allocator: BlockAllocator) -> None:
        """Frees all physical VRAM blocks held by this request."""
        for block in self.physical_blocks:
            allocator.free(block)
        self.physical_blocks.clear()