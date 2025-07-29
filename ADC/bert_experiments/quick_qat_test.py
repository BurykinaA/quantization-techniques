import torch
import torch.nn as nn
from qat_layers import QATLinear, LearnableQuantizer

def test_qat_linear():
    """Test QAT Linear layer"""
    print("=== Testing QAT Linear Layer ===")
    
    # Create QAT linear layer
    qat_linear = QATLinear(512, 256, weight_bits=8, activation_bits=8)
    
    # Create dummy input
    x = torch.randn(32, 512)  # batch_size=32, input_features=512
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    
    # Forward pass
    qat_linear.train()  # Enable training mode for quantization
    output = qat_linear(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Check gradients
    loss = output.sum()
    loss.backward()
    
    # Check if gradients exist
    weight_grad_norm = qat_linear.weight.grad.norm().item() if qat_linear.weight.grad is not None else 0
    print(f"Weight gradient norm: {weight_grad_norm:.6f}")
    
    # Check quantizer gradients
    weight_scale_grad = qat_linear.weight_quantizer.scale.grad
    activation_scale_grad = qat_linear.activation_quantizer.scale.grad
    
    print(f"Weight quantizer scale gradient: {weight_scale_grad}")
    print(f"Activation quantizer scale gradient: {activation_scale_grad}")
    
    return qat_linear

def test_quantizer():
    """Test individual quantizer"""
    print("\n=== Testing Quantizer ===")
    
    quantizer = LearnableQuantizer(num_bits=8, symmetric=True)
    x = torch.randn(10, 20) * 2  # Random data
    
    print(f"Input mean: {x.mean().item():.3f}, std: {x.std().item():.3f}")
    
    # Forward pass
    quantizer.train()
    x_quant = quantizer(x)
    
    print(f"Quantized mean: {x_quant.mean().item():.3f}, std: {x_quant.std().item():.3f}")
    print(f"Scale: {quantizer.scale.item():.6f}")
    
    # Test gradients
    loss = x_quant.sum()
    loss.backward()
    
    print(f"Input gradient norm: {x.grad.norm().item() if x.grad is not None else 'No gradient'}")
    print(f"Scale gradient: {quantizer.scale.grad.item() if quantizer.scale.grad is not None else 'No gradient'}")

def compare_with_fp32():
    """Compare QAT with FP32 baseline"""
    print("\n=== Comparing QAT vs FP32 ===")
    
    # Create layers
    fp32_linear = nn.Linear(256, 128)
    qat_linear = QATLinear(256, 128, weight_bits=8, activation_bits=8)
    
    # Copy weights to make fair comparison
    with torch.no_grad():
        qat_linear.weight.copy_(fp32_linear.weight)
        qat_linear.bias.copy_(fp32_linear.bias)
    
    # Test input
    x = torch.randn(16, 256)
    
    # Forward pass
    fp32_linear.train()
    qat_linear.train()
    
    fp32_output = fp32_linear(x)
    qat_output = qat_linear(x)
    
    # Compare outputs
    mse = nn.MSELoss()(fp32_output, qat_output)
    print(f"MSE between FP32 and QAT outputs: {mse.item():.6f}")
    
    # Compare gradients
    fp32_loss = fp32_output.sum()
    qat_loss = qat_output.sum()
    
    fp32_loss.backward()
    qat_loss.backward()
    
    fp32_grad_norm = fp32_linear.weight.grad.norm().item()
    qat_grad_norm = qat_linear.weight.grad.norm().item()
    
    print(f"FP32 gradient norm: {fp32_grad_norm:.6f}")
    print(f"QAT gradient norm: {qat_grad_norm:.6f}")
    print(f"Gradient ratio (QAT/FP32): {qat_grad_norm/fp32_grad_norm:.3f}")

if __name__ == "__main__":
    # Run tests
    test_quantizer()
    test_qat_linear()
    compare_with_fp32()
    
    print("\n=== All tests completed! ===")