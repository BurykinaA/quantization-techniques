#!/usr/bin/env python3
"""
Simple script to debug ADC pipeline on your model
"""

import sys
import torch
from transformers import AutoTokenizer, BertForQuestionAnswering, AutoConfig
from adc_pipeline_visualizer import debug_model
from bert_adc_integration import BertADCConverter
import argparse


def main():
    parser = argparse.ArgumentParser(description="Debug ADC pipeline visualization")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to ADC model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./adc_debug_output",
                       help="Where to save debug visualizations")
    parser.add_argument("--layers", type=str, nargs="+", 
                       default=["layer.0.attention.output.dense", 
                               "layer.5.intermediate.dense",
                               "layer.11.output.dense"],
                       help="Layer patterns to debug (e.g., 'layer.0.attention')")
    parser.add_argument("--text", type=str, 
                       default="What is the capital of France? Paris is the capital of France.",
                       help="Text to use for forward pass")
    args = parser.parse_args()
    
    print("="*80)
    print("ADC PIPELINE DEBUGGER")
    print("="*80)
    print(f"\nLoading model from: {args.model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model with ADC layers
    config = AutoConfig.from_pretrained(args.model_path)
    base_model = BertForQuestionAnswering(config)
    
    # Convert to ADC
    model = BertADCConverter.replace_linear_with_adc_qat(
        base_model,
        bx=8, bw=8, ba=8, k=4,
        ashift=False,  # Set to True if you used ashift
        signed_activations=False,
        exclude_patterns=["embeddings", "pooler", "qa_outputs"],
        mvm_limit=256,
        use_dynamic_delta=False,  # Use fixed delta for debugging
        use_delta_anneal=False,
        delta_loss_weight=0.0
    )
    
    # Load weights
    try:
        state_dict = torch.load(f"{args.model_path}/pytorch_model.bin", map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded model weights")
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")
        print("Using randomly initialized weights for debugging")
    
    # Prepare input
    print(f"\nInput text: {args.text[:100]}...")
    inputs = tokenizer(args.text, args.text, 
                      return_tensors="pt", 
                      padding="max_length", 
                      max_length=128,
                      truncation=True)
    
    print(f"\nDebugging layers matching patterns: {args.layers}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Run debugger
    debugger = debug_model(
        model,
        inputs['input_ids'],
        inputs['attention_mask'],
        layer_patterns=args.layers,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE!")
    print("="*80)
    print(f"\n📁 Check visualizations in: {args.output_dir}/")
    print("\nEach plot shows:")
    print("  1. Raw activation X")
    print("  2. Quantized X codes")
    print("  3. After A-shift")
    print("  4. Raw weights W")
    print("  5. Quantized W codes")
    print("  6. Full precision Y = X @ W")
    print("  7. Integer MM: code_X @ code_W")
    print("  8. ADC quantized codes")
    print("  9. ADC output")
    print("  10. Final dequantized output")
    print("  11. FP vs Quantized comparison")
    print("  12. Quantization error distribution")
    print()


if __name__ == "__main__":
    main()

