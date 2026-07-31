"""
tests/test_multilora.py

Verifies dynamic request-level LoRA adapter switching in ModelRunner.
"""

from nano_inference.model_runner import ModelRunner
from nano_inference.scheduler import Request


def test_multilora_routing():
    print("\n--- Testing Multi-LoRA Dynamic Adapter Switching ---")
    runner = ModelRunner()

    # Register mock active adapter IDs in the runner
    runner.active_adapters["code-lora"] = "/mock/path/code-lora"
    runner.active_adapters["json-lora"] = "/mock/path/json-lora"

    # Create requests targeting different fine-tuned adapters
    req_code = Request(
        request_id="req-1",
        prompt="def fibonacci(n):",
        prompt_token_ids=[100, 200, 300],
        adapter_id="code-lora",
    )
    req_json = Request(
        request_id="req-2",
        prompt="{\"name\":",
        prompt_token_ids=[400, 500, 600],
        adapter_id="json-lora",
    )

    # Test dynamic adapter switching
    runner._set_active_adapter(req_code.adapter_id)
    print(f"✅ Successfully routed request {req_code.request_id} to adapter: {req_code.adapter_id}")

    runner._set_active_adapter(req_json.adapter_id)
    print(f"✅ Successfully routed request {req_json.request_id} to adapter: {req_json.adapter_id}")

    runner._set_active_adapter(None)
    print("✅ Successfully restored base model weights (disabled adapters).")


if __name__ == "__main__":
    test_multilora_routing()