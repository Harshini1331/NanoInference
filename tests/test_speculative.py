import torch
from nano_inference.speculative import SpeculativeEngine

def test_speculative_verification():
    engine = SpeculativeEngine(k_speculative_tokens=3)
    
    draft_tokens = [101, 202, 303]
    
    # Mock target logits where index 0 matches 101, index 1 matches 202, but index 2 predicts 999
    vocab_size = 1000
    target_logits = torch.zeros((1, 4, vocab_size))
    target_logits[0, 0, 101] = 10.0  # Accepts 101
    target_logits[0, 1, 202] = 10.0  # Accepts 202
    target_logits[0, 2, 999] = 10.0  # Rejects 303 -> Replaces with 999

    accepted = engine.verify_and_accept(draft_tokens, target_logits)
    print("Draft Tokens:", draft_tokens)
    print("Accepted/Corrected Tokens:", accepted)
    
    assert accepted == [101, 202, 999], "Speculative verification failed!"
    print("✅ Speculative Decoding Verification Test Passed!")

if __name__ == "__main__":
    test_speculative_verification()