"""
Implements virtual memory block management for KV-Cache tensors
with Automatic Prefix Caching (APC) and LRU block recycling.
Decouples logical sequence token positions from physical contiguous VRAM slots.
"""

from typing import Dict, List, Optional, Tuple


class PhysicalTokenBlock:
    """Represents a fixed-size chunk of memory in physical GPU VRAM."""
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id: int = block_id
        self.block_size: int = block_size
        self.ref_count: int = 0  # Number of active sequences referencing this block
        self.hash: Optional[int] = None  # Content hash for prefix matching

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockAllocator:
    """Manages physical VRAM blocks with Automatic Prefix Caching (APC) and LRU eviction."""
    def __init__(self, total_num_blocks: int, block_size: int = 16):
        self.block_size: int = block_size
        self.total_num_blocks: int = total_num_blocks
        
        # Initialize free physical block pool
        self.free_blocks: List[PhysicalTokenBlock] = [
            PhysicalTokenBlock(block_id=i, block_size=block_size)
            for i in range(total_num_blocks)
        ]
        self.allocated_blocks: Dict[int, PhysicalTokenBlock] = {}
        
        # Hash table mapping prefix content hash -> PhysicalTokenBlock (APC)
        self.cached_blocks: Dict[int, PhysicalTokenBlock] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self, block_hash: Optional[int] = None) -> Tuple[PhysicalTokenBlock, bool]:
        """
        Allocates a physical block. Reuses cached prefix block if a hash match occurs.
        Returns: (PhysicalTokenBlock, is_cache_hit)
        """
        # 1. Prefix Cache Hit
        if block_hash is not None and block_hash in self.cached_blocks:
            cached_block = self.cached_blocks[block_hash]
            cached_block.ref_count += 1
            
            # If it was in the free queue awaiting LRU eviction, remove it
            if cached_block in self.free_blocks:
                self.free_blocks.remove(cached_block)
                self.allocated_blocks[cached_block.block_id] = cached_block
                
            return cached_block, True  # Cache Hit!

        # 2. Cache Miss - Allocate new block from free pool
        if not self.free_blocks:
            # If free pool is empty, attempt to evict oldest unreferenced cache entry
            self.evict_oldest_cache()
            if not self.free_blocks:
                raise MemoryError("Out of VRAM Physical Blocks! High concurrency limit reached.")
        
        block = self.free_blocks.pop(0)
        block.ref_count = 1
        block.hash = block_hash
        
        self.allocated_blocks[block.block_id] = block
        
        # Index in hash cache if valid hash provided
        if block_hash is not None:
            self.cached_blocks[block_hash] = block
            
        return block, False  # Cache Miss

    def free(self, block: PhysicalTokenBlock) -> None:
        """Frees a physical block or decrements its reference count (LRU caching)."""
        block.ref_count -= 1
        
        # When reference count drops to 0, move to LRU queue while preserving hash in cached_blocks
        if block.ref_count <= 0:
            block.ref_count = 0
            if block.block_id in self.allocated_blocks:
                del self.allocated_blocks[block.block_id]
            
            # Append to end of free list (LRU order: older blocks evicted first from front)
            if block not in self.free_blocks:
                self.free_blocks.append(block)

    def evict_oldest_cache(self) -> None:
        """Evicts the oldest cached block from the hash lookup table when VRAM is tight."""
        if self.free_blocks:
            oldest_block = self.free_blocks[0]
            if oldest_block.hash in self.cached_blocks:
                del self.cached_blocks[oldest_block.hash]
                oldest_block.hash = None


class BlockTable:
    """
    Logical page table mapping a specific request's sequence of tokens
    to physical non-contiguous VRAM blocks with prefix caching support.
    """
    def __init__(self, block_size: int = 16):
        self.block_size: int = block_size
        self.physical_blocks: List[PhysicalTokenBlock] = []

    def add_block(self, block: PhysicalTokenBlock) -> None:
        """Appends an allocated physical VRAM block to this request's page table."""
        self.physical_blocks.append(block)

    @staticmethod
    def compute_block_hash(token_chunk: List[int], parent_hash: Optional[int] = None) -> int:
        """Computes a deterministic content hash for a 16-token block given its prefix history."""
        return hash((parent_hash, tuple(token_chunk)))

    def allocate_slot_for_token(
        self, 
        token_index: int, 
        tokens: List[int], 
        allocator: BlockAllocator
    ) -> Tuple[Optional[PhysicalTokenBlock], bool]:
        """
        Allocates a physical block on block boundaries (0, 16, 32...).
        Computes block prefix hash and attempts cache hit in BlockAllocator.
        Returns: (allocated_block, is_cache_hit)
        """
        if token_index % self.block_size == 0:
            chunk = tokens[token_index : token_index + self.block_size]
            
            # Compute cumulative hash linked to previous block's hash
            parent_hash = self.physical_blocks[-1].hash if self.physical_blocks else None
            block_hash = self.compute_block_hash(chunk, parent_hash) if len(chunk) == self.block_size else None
            
            new_block, is_hit = allocator.allocate(block_hash=block_hash)
            self.add_block(new_block)
            return new_block, is_hit
            
        return None, False

    def get_block_ids(self) -> List[int]:
        """Returns the physical block IDs mapped to this sequence."""
        return [b.block_id for b in self.physical_blocks]

    def free_all(self, allocator: BlockAllocator) -> None:
        """Frees all physical VRAM blocks held by this request while preserving prefix hashes in LRU pool."""
        for block in self.physical_blocks:
            allocator.free(block)
        self.physical_blocks.clear()