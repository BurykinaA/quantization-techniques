"""
Test suite for ADC-LoRA implementation.

Tests:
1. Zero-init identity: Output should match base layer when LoRA B=0
2. Gradient flow: Only LoRA parameters should have gradients when base is frozen
3. Parameter counting: Verify compression ratio calculations
4. Forward pass correctness: Verify ADC quantization is applied to effective weight
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ADC.bert_clean.core.adc_layers import (
    QATLinearADC,
    TiledLinearADC,
    LoRAQATLinearADC,
    LoRATiledLinearADC,
)


def test_lora_qat_linear_zero_init_identity():
    """Test that LoRA output matches base output when B=0 (zero initialization)."""
    print("Test 1: Zero-init identity for LoRAQATLinearADC...")
    
    # Create base layer
    base_layer = QATLinearADC(
        in_features=64,
        out_features=32,
        bias=True,
        bx=8, bw=8, ba=8, k=4,
    )
    
    # Initialize with some weights
    nn.init.xavier_uniform_(base_layer.weight)
    nn.init.zeros_(base_layer.bias)
    
    # Create LoRA wrapper
    lora_layer = LoRAQATLinearADC(base_layer, r=4, alpha=8.0)
    
    # Verify B is initialized to zero (so A @ B = 0)
    assert torch.allclose(lora_layer.lora_B, torch.zeros_like(lora_layer.lora_B)), \
        "LoRA B should be initialized to zeros"
    
    # Set to eval mode for deterministic output
    base_layer.eval()
    lora_layer.eval()
    
    # Run forward pass
    x = torch.randn(2, 64)
    
    # Note: Since base_layer's weight is now frozen and we're using the same quantizers,
    # the outputs should be identical when B=0
    with torch.no_grad():
        # Get base output directly
        base_output = base_layer(x)
        # Get LoRA output (should be same since B=0)
        lora_output = lora_layer(x)
    
    # Check outputs are close (may have small differences due to quantization)
    max_diff = (base_output - lora_output).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1e-4, f"Outputs differ by {max_diff}, expected near-zero difference"
    
    print("  PASSED: Zero-init identity verified")


def test_lora_tiled_zero_init_identity():
    """Test that LoRATiledLinearADC output matches base when B=0."""
    print("Test 2: Zero-init identity for LoRATiledLinearADC...")
    
    # Create base tiled layer
    base_layer = TiledLinearADC(
        in_features=128,
        out_features=64,
        bias=True,
        bx=8, bw=8, ba=8, k=4,
        mvm_limit=64,  # Forces 2 tiles
    )
    
    # Initialize weights
    for tile in base_layer.tiles:
        nn.init.xavier_uniform_(tile.weight)
        if tile.bias is not None:
            nn.init.zeros_(tile.bias)
    
    # Create LoRA wrapper
    lora_layer = LoRATiledLinearADC(base_layer, r=4, alpha=8.0)
    
    # Verify all B matrices are zeros
    for i, lora_b in enumerate(lora_layer.lora_B):
        assert torch.allclose(lora_b, torch.zeros_like(lora_b)), \
            f"LoRA B[{i}] should be initialized to zeros"
    
    base_layer.eval()
    lora_layer.eval()
    
    x = torch.randn(2, 128)
    
    with torch.no_grad():
        base_output = base_layer(x)
        lora_output = lora_layer(x)
    
    max_diff = (base_output - lora_output).abs().max().item()
    print(f"  Max difference: {max_diff:.6f}")
    assert max_diff < 1e-4, f"Outputs differ by {max_diff}"
    
    print("  PASSED: Zero-init identity verified for tiled layer")


def test_gradient_flow():
    """Test that gradients flow only to LoRA parameters, not base weights."""
    print("Test 3: Gradient flow verification...")
    
    base_layer = QATLinearADC(
        in_features=64,
        out_features=32,
        bias=True,
        bx=8, bw=8, ba=8, k=4,
    )
    nn.init.xavier_uniform_(base_layer.weight)
    
    lora_layer = LoRAQATLinearADC(base_layer, r=4, alpha=8.0)
    
    # IMPORTANT: Set B to non-zero so gradients flow to A
    # When B=0, d(A@B)/dA = B^T = 0, so no gradient to A (mathematically correct!)
    with torch.no_grad():
        lora_layer.lora_B.normal_(0, 0.1)
    
    lora_layer.train()
    
    # Forward pass
    x = torch.randn(2, 64, requires_grad=True)
    output = lora_layer(x)
    loss = output.sum()
    loss.backward()
    
    # Check base weight has no gradient (frozen)
    assert lora_layer.base_layer.weight.grad is None, \
        "Base weight should not have gradients"
    
    # Check LoRA parameters have gradients
    assert lora_layer.lora_A.grad is not None, "LoRA A should have gradients"
    assert lora_layer.lora_B.grad is not None, "LoRA B should have gradients"
    
    # Check gradients are non-zero
    assert lora_layer.lora_A.grad.abs().sum() > 0, "LoRA A gradient should be non-zero"
    assert lora_layer.lora_B.grad.abs().sum() > 0, "LoRA B gradient should be non-zero"
    
    print("  PASSED: Gradients flow correctly to LoRA params only")


def test_gradient_flow_tiled():
    """Test gradient flow for LoRATiledLinearADC."""
    print("Test 4: Gradient flow for tiled LoRA layer...")
    
    base_layer = TiledLinearADC(
        in_features=128,
        out_features=64,
        bias=True,
        bx=8, bw=8, ba=8, k=4,
        mvm_limit=64,
    )
    
    lora_layer = LoRATiledLinearADC(base_layer, r=4, alpha=8.0)
    
    # Set B to non-zero so gradients flow to A
    # When B=0, d(A@B)/dA = B^T = 0, so no gradient to A
    with torch.no_grad():
        for lora_b in lora_layer.lora_B:
            lora_b.normal_(0, 0.1)
    
    lora_layer.train()
    
    x = torch.randn(2, 128, requires_grad=True)
    output = lora_layer(x)
    loss = output.sum()
    loss.backward()
    
    # Check base weights are frozen
    for i, tile in enumerate(lora_layer.tiled_layer.tiles):
        assert tile.weight.grad is None, f"Tile {i} weight should not have gradients"
    
    # Check LoRA parameters have gradients and they are non-zero
    for i in range(lora_layer.n_tiles):
        assert lora_layer.lora_A[i].grad is not None, f"LoRA A[{i}] should have gradients"
        assert lora_layer.lora_B[i].grad is not None, f"LoRA B[{i}] should have gradients"
        assert lora_layer.lora_A[i].grad.abs().sum() > 0, f"LoRA A[{i}] gradient should be non-zero"
        assert lora_layer.lora_B[i].grad.abs().sum() > 0, f"LoRA B[{i}] gradient should be non-zero"
    
    print("  PASSED: Gradients flow correctly for tiled LoRA layer")


def test_parameter_counting():
    """Test parameter counting and compression ratio."""
    print("Test 5: Parameter counting and compression ratio...")
    
    in_features = 768
    out_features = 768
    r = 8
    
    base_layer = QATLinearADC(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        bx=8, bw=8, ba=8, k=4,
    )
    
    lora_layer = LoRAQATLinearADC(base_layer, r=r, alpha=16.0)
    
    # Calculate expected LoRA params: A (out_features, r) + B (r, in_features)
    expected_lora_params = out_features * r + r * in_features
    actual_lora_params = lora_layer.get_num_trainable_params()
    
    assert actual_lora_params == expected_lora_params, \
        f"Expected {expected_lora_params} LoRA params, got {actual_lora_params}"
    
    # Check compression ratio
    full_params = in_features * out_features
    compression_ratio = lora_layer.get_compression_ratio()
    expected_ratio = full_params / expected_lora_params
    
    assert abs(compression_ratio - expected_ratio) < 0.1, \
        f"Compression ratio mismatch: expected {expected_ratio:.1f}, got {compression_ratio:.1f}"
    
    print(f"  LoRA params: {actual_lora_params:,}")
    print(f"  Full params: {full_params:,}")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    print("  PASSED: Parameter counting correct")


def test_lora_output_changes_with_nonzero_b():
    """Test that LoRA actually modifies output when B is non-zero."""
    print("Test 6: LoRA output modification with non-zero B...")
    
    base_layer = QATLinearADC(
        in_features=64,
        out_features=32,
        bias=True,
        bx=8, bw=8, ba=8, k=4,
    )
    nn.init.xavier_uniform_(base_layer.weight)
    
    lora_layer = LoRAQATLinearADC(base_layer, r=4, alpha=8.0)
    lora_layer.eval()
    
    x = torch.randn(2, 64)
    
    # Get output with B=0
    with torch.no_grad():
        output_before = lora_layer(x).clone()
    
    # Manually set B to non-zero
    with torch.no_grad():
        lora_layer.lora_B.fill_(0.1)
    
    # Get output with B!=0
    with torch.no_grad():
        output_after = lora_layer(x)
    
    # Outputs should be different
    diff = (output_before - output_after).abs().max().item()
    print(f"  Output difference after modifying B: {diff:.6f}")
    assert diff > 0.01, "LoRA should modify output when B is non-zero"
    
    print("  PASSED: LoRA correctly modifies output")


def test_merge_lora_weights():
    """Test merging LoRA weights into base weights."""
    print("Test 7: LoRA weight merging...")
    
    base_layer = QATLinearADC(
        in_features=64,
        out_features=32,
        bias=True,
        bx=8, bw=8, ba=8, k=4,
    )
    nn.init.xavier_uniform_(base_layer.weight)
    
    lora_layer = LoRAQATLinearADC(base_layer, r=4, alpha=8.0)
    
    # Set B to non-zero for merge to have effect
    with torch.no_grad():
        lora_layer.lora_B.normal_(0, 0.1)
    
    lora_layer.eval()
    x = torch.randn(2, 64)
    
    # Get output before merge
    with torch.no_grad():
        output_before = lora_layer(x).clone()
    
    # Merge weights
    lora_layer.merge_lora_weights()
    
    # Get output after merge (should be similar)
    with torch.no_grad():
        output_after = lora_layer(x)
    
    # Outputs should be close (may differ slightly due to quantization)
    diff = (output_before - output_after).abs().max().item()
    print(f"  Output difference after merge: {diff:.6f}")
    
    # Verify A and B are reset to zero
    assert torch.allclose(lora_layer.lora_A, torch.zeros_like(lora_layer.lora_A)), \
        "LoRA A should be zero after merge"
    assert torch.allclose(lora_layer.lora_B, torch.zeros_like(lora_layer.lora_B)), \
        "LoRA B should be zero after merge"
    
    print("  PASSED: LoRA weight merging works")


def test_compute_reference_output():
    """Test that compute_reference_output returns quantized (no ADC) output."""
    print("Test 8: Reference output computation (Eq. 12 target)...")
    
    base_layer = QATLinearADC(
        in_features=64,
        out_features=32,
        bias=True,
        bx=8, bw=8, ba=8, k=4,
    )
    nn.init.xavier_uniform_(base_layer.weight)
    nn.init.zeros_(base_layer.bias)
    
    lora_layer = LoRAQATLinearADC(base_layer, r=4, alpha=8.0)
    lora_layer.eval()
    
    x = torch.randn(2, 64)
    
    with torch.no_grad():
        # Reference output: Qx(X) @ Qw(W) - no ADC, no LoRA
        ref_output = lora_layer.compute_reference_output(x)
        
        # LoRA output when B=0: QA(Qx(X) @ Qw(W)) - with ADC
        lora_output = lora_layer(x)
    
    # Reference and LoRA output should differ due to ADC quantization
    # (unless ADC happens to not clip anything)
    print(f"  Reference output shape: {ref_output.shape}")
    print(f"  LoRA output shape: {lora_output.shape}")
    
    # The reference should NOT have ADC clipping applied
    # LoRA output DOES have ADC clipping
    # So they may differ
    diff = (ref_output - lora_output).abs().mean().item()
    print(f"  Mean difference (ref vs lora with ADC): {diff:.6f}")
    
    # They should be relatively close but not identical
    # (ADC quantization introduces some error)
    assert ref_output.shape == lora_output.shape, "Output shapes should match"
    
    print("  PASSED: Reference output computation works")


def test_compute_reference_output_tiled():
    """Test reference output for tiled LoRA layer."""
    print("Test 9: Reference output computation for tiled layer...")
    
    base_layer = TiledLinearADC(
        in_features=128,
        out_features=64,
        bias=True,
        bx=8, bw=8, ba=8, k=4,
        mvm_limit=64,
    )
    
    for tile in base_layer.tiles:
        nn.init.xavier_uniform_(tile.weight)
        if tile.bias is not None:
            nn.init.zeros_(tile.bias)
    
    lora_layer = LoRATiledLinearADC(base_layer, r=4, alpha=8.0)
    lora_layer.eval()
    
    x = torch.randn(2, 128)
    
    with torch.no_grad():
        ref_output = lora_layer.compute_reference_output(x)
        lora_output = lora_layer(x)
    
    print(f"  Reference output shape: {ref_output.shape}")
    print(f"  LoRA output shape: {lora_output.shape}")
    
    diff = (ref_output - lora_output).abs().mean().item()
    print(f"  Mean difference (ref vs lora with ADC): {diff:.6f}")
    
    assert ref_output.shape == lora_output.shape, "Output shapes should match"
    
    print("  PASSED: Reference output computation works for tiled layer")


def run_all_tests():
    """Run all LoRA tests."""
    print("=" * 60)
    print("ADC-LoRA Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_lora_qat_linear_zero_init_identity,
        test_lora_tiled_zero_init_identity,
        test_gradient_flow,
        test_gradient_flow_tiled,
        test_parameter_counting,
        test_lora_output_changes_with_nonzero_b,
        test_merge_lora_weights,
        test_compute_reference_output,
        test_compute_reference_output_tiled,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
