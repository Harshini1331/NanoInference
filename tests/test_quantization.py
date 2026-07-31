import torch
from nano_inference.model_runner import ModelRunner

def test_fp8_quantization():
    print("\n--- Testing Native FP8 Quantization ---")
    runner = ModelRunner(quantization="fp8")
    param = next(runner.model.parameters())
    print("FP8 Model Parameter Dtype:", param.dtype)
    assert param.dtype == torch.float8_e4m3fn, "FP8 quantization failed!"
    print("✅ FP8 Model Loading Test Passed!")

def test_int4_quantization():
    print("\n--- Testing INT4 BitsAndBytes Quantization ---")
    runner = ModelRunner(quantization="int4")
    print("✅ INT4 Model Loading Test Passed!")

if __name__ == "__main__":
    test_fp8_quantization()
    test_int4_quantization()