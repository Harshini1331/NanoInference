from nano_inference.block_manager import BlockAllocator, BlockTable


def test_block_allocator_and_table():
    # Setup allocator with 4 physical blocks of size 16 tokens
    allocator = BlockAllocator(total_num_blocks=4, block_size=16)
    assert allocator.num_free_blocks == 4

    block_table = BlockTable(block_size=16)

    # Token 0: Triggers 1st physical block allocation
    b1 = block_table.allocate_slot_for_token(0, allocator)
    assert b1 is not None
    assert allocator.num_free_blocks == 3
    assert block_table.get_block_ids() == [0]

    # Tokens 1 to 15: No new block allocation required
    for i in range(1, 16):
        assert block_table.allocate_slot_for_token(i, allocator) is None
    assert allocator.num_free_blocks == 3

    # Token 16: Triggers 2nd physical block allocation
    b2 = block_table.allocate_slot_for_token(16, allocator)
    assert b2 is not None
    assert allocator.num_free_blocks == 2
    assert block_table.get_block_ids() == [0, 1]

    # Clean up request memory
    block_table.free_all(allocator)
    assert allocator.num_free_blocks == 4
    print("✅ Block Manager Unit Test Passed!")


if __name__ == "__main__":
    test_block_allocator_and_table()