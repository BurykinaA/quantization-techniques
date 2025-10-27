"""
Test script for the fixed QAT implementation
"""
import torch
import torch.nn as nn
from ADC.bert_clean.core.qat_layers import QATLinear, LearnableQuantizer, QATTransformerBlock

def test_gradient_flow():
    """Test that gradients flow correctly to scale parameter"""
    print("=" * 60)
    print("Test 1: Gradient Flow")
    print("=" * 60)
    
    quantizer = LearnableQuantizer(num_bits=8, symmetric=True)
    x = torch.randn(32, 128, requires_grad=True)
    
    # Calibrate
    quantizer.calibrate(x)
    initial_scale = quantizer.scale.data.clone()
    print(f"Initial scale: {initial_scale.item():.6f}")
    
    # Forward + backward
    y = quantizer(x)
    loss = y.pow(2).mean()
    loss.backward()
    
    # Check gradient exists and is non-zero
    assert quantizer.scale.grad is not None, "Scale gradient is None!"
    assert quantizer.scale.grad.abs().sum() > 0, "Scale gradient is zero!"
    print(f"Scale gradient: {quantizer.scale.grad.item():.6f} ✓")
    
    # Check zero_point has no gradient (it's a buffer)
    assert not quantizer.zero_point.requires_grad, "Zero-point should not require grad!"
    print(f"Zero-point requires_grad: False ✓")
    print()

def test_qat_linear():
    """Test QATLinear layer"""
    print("=" * 60)
    print("Test 2: QATLinear Layer")
    print("=" * 60)
    
    layer = QATLinear(256, 512, weight_bits=8, activation_bits=8)
    x = torch.randn(16, 256)
    
    # Test calibration
    layer.calibrate(x)
    print(f"Weight scale shape: {layer.weight_quantizer.scale.shape}")
    print(f"Activation scale shape: {layer.activation_quantizer.scale.shape}")
    assert layer.weight_quantizer.calibrated, "Weight quantizer not calibrated!"
    assert layer.activation_quantizer.calibrated, "Activation quantizer not calibrated!"
    print("Calibration: ✓")
    
    # Test forward
    y = layer(x)
    assert y.shape == (16, 512), f"Wrong output shape: {y.shape}"
    print(f"Output shape: {y.shape} ✓")
    
    # Test gradient flow
    loss = y.pow(2).mean()
    loss.backward()
    assert layer.weight.grad is not None, "Weight gradient is None!"
    assert layer.weight_quantizer.scale.grad is not None, "Weight scale gradient is None!"
    assert layer.activation_quantizer.scale.grad is not None, "Activation scale gradient is None!"
    print("Gradient flow: ✓")
    
    # Test enable/disable quantization
    layer.disable_quantization()
    y_fp = layer(x)
    layer.enable_quantization()
    y_quant = layer(x)
    diff = (y_fp - y_quant).abs().mean()
    print(f"FP vs Quant difference: {diff.item():.6f}")
    assert diff > 1e-6, "Quantization has no effect!"
    print("Enable/Disable: ✓")
    print()

def test_auto_calibration():
    """Test auto-calibration on first forward"""
    print("=" * 60)
    print("Test 3: Auto-Calibration")
    print("=" * 60)
    
    layer = QATLinear(128, 256, weight_bits=4, activation_bits=8)
    x = torch.randn(8, 128)
    
    # Check not calibrated initially
    assert not layer.weight_quantizer.calibrated, "Should not be calibrated initially!"
    assert not layer.activation_quantizer.calibrated, "Should not be calibrated initially!"
    print("Initial state: not calibrated ✓")
    
    # First forward should auto-calibrate
    y = layer(x)
    assert layer.weight_quantizer.calibrated, "Weight quantizer should be auto-calibrated!"
    assert layer.activation_quantizer.calibrated, "Activation quantizer should be auto-calibrated!"
    print("After first forward: auto-calibrated ✓")
    print(f"Weight scale: {layer.weight_quantizer.scale.data.mean().item():.6f}")
    print(f"Activation scale: {layer.activation_quantizer.scale.data.item():.6f}")
    print()

def test_per_channel_quantization():
    """Test per-channel weight quantization"""
    print("=" * 60)
    print("Test 4: Per-Channel Quantization")
    print("=" * 60)
    
    # Weight quantizer should be per-channel
    layer = QATLinear(64, 128, weight_bits=8, activation_bits=8)
    x = torch.randn(4, 64)
    
    # Calibrate
    layer.calibrate(x)
    
    # Check weight scale is per-channel (one scale per output channel)
    out_features = layer.out_features
    assert layer.weight_quantizer.scale.shape[0] == out_features, \
        f"Weight scale should have {out_features} elements, got {layer.weight_quantizer.scale.shape}"
    print(f"Weight scale shape: {layer.weight_quantizer.scale.shape} (per-channel) ✓")
    
    # Check activation scale is per-tensor (single value)
    assert layer.activation_quantizer.scale.shape[0] == 1, \
        f"Activation scale should be scalar, got {layer.activation_quantizer.scale.shape}"
    print(f"Activation scale shape: {layer.activation_quantizer.scale.shape} (per-tensor) ✓")
    print()

def test_transformer_block():
    """Test QATTransformerBlock"""
    print("=" * 60)
    print("Test 5: QATTransformerBlock")
    print("=" * 60)
    
    block = QATTransformerBlock(
        d_model=256, 
        num_heads=8, 
        d_ff=1024,
        weight_bits=8,
        activation_bits=8
    )
    
    x = torch.randn(4, 32, 256)  # (batch, seq_len, d_model)
    
    # Test forward
    y = block(x)
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    print(f"Output shape: {y.shape} ✓")
    
    # Test gradient flow
    loss = y.pow(2).mean()
    loss.backward()
    
    # Check all QAT layers have gradients
    qat_layers = [m for m in block.modules() if isinstance(m, QATLinear)]
    print(f"Number of QAT layers: {len(qat_layers)}")
    
    for i, layer in enumerate(qat_layers):
        assert layer.weight.grad is not None, f"Layer {i} weight gradient is None!"
        assert layer.weight_quantizer.scale.grad is not None, f"Layer {i} weight scale gradient is None!"
    print("All layers have gradients ✓")
    
    # Test enable/disable
    block.disable_quantization()
    y_fp = block(x)
    block.enable_quantization()
    y_quant = block(x)
    diff = (y_fp - y_quant).abs().mean()
    print(f"FP vs Quant difference: {diff.item():.6f}")
    print()

def test_asymmetric_quantization():
    """Test asymmetric quantization for activations"""
    print("=" * 60)
    print("Test 6: Asymmetric Quantization")
    print("=" * 60)
    
    quantizer = LearnableQuantizer(num_bits=8, symmetric=False)
    
    # Create data with positive bias (ReLU-like)
    x = torch.randn(100, 128).relu()
    
    quantizer.calibrate(x)
    
    print(f"Data range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    print(f"Scale: {quantizer.scale.item():.6f}")
    print(f"Zero-point: {quantizer.zero_point.item():.3f}")
    print(f"Qmin: {quantizer.qmin}, Qmax: {quantizer.qmax}")
    
    # Check zero_point is used for asymmetric
    assert quantizer.zero_point.item() >= quantizer.qmin, "Zero-point below qmin!"
    assert quantizer.zero_point.item() <= quantizer.qmax, "Zero-point above qmax!"
    print("Zero-point in valid range ✓")
    
    # Check quantization reduces range
    y = quantizer(x)
    # With 8-bit quantization, we should see discretization
    unique_values = torch.unique(y)
    print(f"Unique values after quantization: {len(unique_values)}")
    assert len(unique_values) <= 256, "Too many unique values for 8-bit!"
    print()

def test_low_bit_quantization():
    """Test 4-bit quantization"""
    print("=" * 60)
    print("Test 7: Low-Bit Quantization (4-bit)")
    print("=" * 60)
    
    quantizer = LearnableQuantizer(num_bits=4, symmetric=True)
    x = torch.randn(50, 64)
    
    quantizer.calibrate(x)
    y = quantizer(x)
    
    print(f"Qmin: {quantizer.qmin}, Qmax: {quantizer.qmax}")
    print(f"Expected levels: {quantizer.qmax - quantizer.qmin + 1}")
    
    unique_values = torch.unique(y)
    print(f"Actual unique values: {len(unique_values)}")
    
    # For 4-bit symmetric: range is [-8, 7] = 16 levels
    assert quantizer.qmin == -8, f"Wrong qmin: {quantizer.qmin}"
    assert quantizer.qmax == 7, f"Wrong qmax: {quantizer.qmax}"
    print("Quantization levels: ✓")
    print()

def main():
    print("\n" + "=" * 60)
    print("Testing Fixed QAT Implementation")
    print("=" * 60 + "\n")
    
    torch.manual_seed(42)
    
    try:
        test_gradient_flow()
        test_qat_linear()
        test_auto_calibration()
        test_per_channel_quantization()
        test_transformer_block()
        test_asymmetric_quantization()
        test_low_bit_quantization()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()

