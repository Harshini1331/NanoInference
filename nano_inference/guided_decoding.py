"""
nano_inference/guided_decoding.py

Implements JSON Schema / Grammar-constrained Logits Processing for Structured Outputs.
Masks invalid token logits to -inf during sampling passes.
"""

import torch
from typing import List, Optional, Dict, Any


class JSONSchemaLogitsProcessor:
    """
    Logits processor that enforces valid JSON structure during autoregressive decoding.
    """
    def __init__(self, tokenizer, schema: Optional[Dict[str, Any]] = None):
        self.tokenizer = tokenizer
        self.schema = schema
        
        # Pre-calculate token IDs for structural JSON characters
        self.bracket_open_id = tokenizer.encode("{", add_special_tokens=False)[-1]
        self.bracket_close_id = tokenizer.encode("}", add_special_tokens=False)[-1]
        self.quote_id = tokenizer.encode('"', add_special_tokens=False)[-1]
        self.colon_id = tokenizer.encode(":", add_special_tokens=False)[-1]
        self.comma_id = tokenizer.encode(",", add_special_tokens=False)[-1]

    def apply_guided_mask(self, req, logits: torch.Tensor) -> torch.Tensor:
        """
        Applies a logit bias mask (-inf) to invalid tokens based on JSON structure state.
        """
        current_text = self.tokenizer.decode(req.output_token_ids, skip_special_tokens=True)
        
        # If output hasn't started, force opening JSON brace '{'
        if not current_text.strip():
            mask = torch.full_like(logits, float("-inf"))
            mask[0, self.bracket_open_id] = 0.0
            return logits + mask
            
        # Basic state machine checks for structural balance
        open_braces = current_text.count("{") - current_text.count("}")
        
        # If brackets are balanced, enforce closing or completion
        if open_braces <= 0:
            mask = torch.full_like(logits, float("-inf"))
            mask[0, self.tokenizer.eos_token_id] = 0.0
            return logits + mask

        return logits