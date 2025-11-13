#!/usr/bin/env python3
"""
Diagnostic script to check QAT quantizer behavior
Run this BEFORE training to see what's happening
"""

import torch
from transformers import AutoTokenizer, BertForQuestionAnswering
from datasets import load_dataset
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ADC.bert_clean.core.qat_layers import QATLinear


def diagnose_quantizer_behavior():
    """Check if quantizers are properly initialized or updating during training"""
    
    print("="*80)
    print("DIAGNOSTIC: QAT Quantizer Behavior")
    print("="*80)
    
    # Create a simple QATLinear layer
    layer = QATLinear(768, 768, bias=True, weight_bits=8, activation_bits=8)
    
    # Initialize with random weights
    torch.nn.init.normal_(layer.weight, mean=0, std=0.02)
    
    # Create fake input
    x = torch.randn(8, 128, 768)  # batch_size=8, seq_len=128, hidden=768
    
    print("\n1. Initial state (before any forward pass):")
    print(f"   Act quantizer scale: {layer.activation_quantizer.scale.item():.6f}")
    print(f"   Act quantizer scale.requires_grad: {layer.activation_quantizer.scale.requires_grad}")
    print(f"   Weight quantizer scale[0]: {layer.weight_quantizer.scale[0].item():.6f}")
    print(f"   Weight quantizer scale.requires_grad: {layer.weight_quantizer.scale.requires_grad}")
    
    print("\n2. After first forward pass (train mode):")
    layer.train()
    _ = layer(x)
    print(f"   Act quantizer scale: {layer.activation_quantizer.scale.item():.6f}")
    print(f"   Weight quantizer scale[0]: {layer.weight_quantizer.scale[0].item():.6f}")
    
    print("\n3. After second forward pass (train mode):")
    _ = layer(x)
    print(f"   Act quantizer scale: {layer.activation_quantizer.scale.item():.6f}")
    print(f"   Weight quantizer scale[0]: {layer.weight_quantizer.scale[0].item():.6f}")
    
    print("\n4. After third forward pass (train mode):")
    _ = layer(x)
    print(f"   Act quantizer scale: {layer.activation_quantizer.scale.item():.6f}")
    print(f"   Weight quantizer scale[0]: {layer.weight_quantizer.scale[0].item():.6f}")
    
    print("\n5. Switch to eval mode:")
    layer.eval()
    _ = layer(x)
    print(f"   Act quantizer scale: {layer.activation_quantizer.scale.item():.6f}")
    print(f"   Weight quantizer scale[0]: {layer.weight_quantizer.scale[0].item():.6f}")
    
    print("\n" + "="*80)
    print("ISSUE IDENTIFIED:")
    print("="*80)
    print("\n⚠️  If scales are CHANGING across forward passes in train mode,")
    print("   this means update_params() is being called repeatedly.")
    print("   This is WRONG for QAT - scales should be:")
    print("   - Initialized ONCE during calibration")
    print("   - FIXED (no update_params) during training")
    print("   - Only updated via gradients if requires_grad=True")
    print("\n⚠️  Your friend's code has 3 states:")
    print("   - State 0: Initialize (use observers, calculate scales)")
    print("   - State 1: Learn (scales have requires_grad=True, learn via backprop)")
    print("   - State 2: Fix (scales frozen)")
    print("\n⚠️  Your code is missing this state management!")
    print("   - update_params() is called in EVERY forward pass during training")
    print("   - This continuously updates scales with EMA, destabilizing training")
    print("="*80)


if __name__ == "__main__":
    diagnose_quantizer_behavior()

