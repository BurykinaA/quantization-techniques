#!/usr/bin/env python3
"""
Visualize PTQ-calibrated ADC model pipeline
"""

import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoConfig
from adc_pipeline_visualizer import debug_model


def main():
    parser = argparse.ArgumentParser(description="Visualize PTQ ADC model pipeline")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to PTQ-calibrated ADC model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./ptq_debug_output",
                       help="Where to save debug visualizations")
    parser.add_argument("--layers", type=str, nargs="+", 
                       default=["layer.0.attention.output.dense", 
                               "layer.5.intermediate.dense",
                               "layer.11.output.dense"],
                       help="Layer patterns to debug")
    parser.add_argument("--text", type=str, 
                       default="What is the capital of France? Paris is the capital of France.",
                       help="Text to use for forward pass")
    args = parser.parse_args()
    
    print("="*80)
    print("PTQ ADC PIPELINE VISUALIZER")
    print("="*80)
    print(f"\nLoading PTQ model from: {args.model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load PTQ-calibrated model directly
    # The model is already converted with proper ADC layers and calibrated scales
    from transformers import BertForQuestionAnswering
    
    try:
        # Try loading with from_pretrained (if config.json exists)
        model = BertForQuestionAnswering.from_pretrained(args.model_path)
        print("✓ Loaded PTQ model with from_pretrained")
    except Exception as e:
        print(f"Warning: Could not load with from_pretrained: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: Load config and weights separately
        config = AutoConfig.from_pretrained(args.model_path)
        from bert_adc_integration import BertADCConverter
        
        base_model = BertForQuestionAnswering(config)
        
        # Convert to ADC (structure only, weights will be loaded)
        model = BertADCConverter.replace_linear_with_adc_qat(
            base_model,
            bx=8, bw=8, ba=8, k=4,
            ashift=False,
            signed_activations=False,
            exclude_patterns=["embeddings", "pooler", "qa_outputs"],
            mvm_limit=256,
            use_dynamic_delta=False,
            use_delta_anneal=False,
            delta_loss_weight=0.0
        )
        
        # Load PTQ weights
        state_dict = torch.load(f"{args.model_path}/pytorch_model.bin", map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded PTQ model weights")
    
    # Prepare input
    print(f"\nInput text: {args.text[:100]}...")
    inputs = tokenizer(args.text, args.text, 
                      return_tensors="pt", 
                      padding="max_length", 
                      max_length=128,
                      truncation=True)
    
    print(f"\nVisualizing layers matching patterns: {args.layers}")
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
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\n📁 Check visualizations in: {args.output_dir}/")
    print("\n🔍 What to look for:")
    print("  - Weight codes should span [-128, 127] after PTQ")
    print("  - Activation codes should use full range")
    print("  - Check for excessive clipping in ADC quantization")
    print("  - Compare full precision vs quantized outputs")
    print()


if __name__ == "__main__":
    main()

