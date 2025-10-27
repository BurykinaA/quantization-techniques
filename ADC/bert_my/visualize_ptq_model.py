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
    parser.add_argument("--bx", type=int, default=8, help="Activation bits")
    parser.add_argument("--bw", type=int, default=8, help="Weight bits")
    parser.add_argument("--ba", type=int, default=8, help="ADC bits")
    parser.add_argument("--k", type=int, default=4, help="Hardware design parameter")
    parser.add_argument("--ashift", action="store_true", help="Use A-shift")
    parser.add_argument("--signed_activations", action="store_true", help="Use signed activations")
    args = parser.parse_args()
    
    print("="*80)
    print("PTQ ADC PIPELINE VISUALIZER")
    print("="*80)
    print(f"\nLoading PTQ model from: {args.model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load PTQ-calibrated model
    # IMPORTANT: from_pretrained() doesn't work for custom ADC layers!
    # We must manually reconstruct the ADC architecture and load weights
    from transformers import BertForQuestionAnswering
    from bert_adc_integration import BertADCConverter
    
    print("Loading config and reconstructing ADC model...")
    config = AutoConfig.from_pretrained(args.model_path)
    base_model = BertForQuestionAnswering(config)
    
    # Convert to ADC (structure must match PTQ model)
    print(f"Config: bx={args.bx}, bw={args.bw}, ba={args.ba}, k={args.k}, ashift={args.ashift}, signed={args.signed_activations}")
    model = BertADCConverter.replace_linear_with_adc_qat(
        base_model,
        bx=args.bx, 
        bw=args.bw, 
        ba=args.ba, 
        k=args.k,
        ashift=args.ashift,
        signed_activations=args.signed_activations,
        exclude_patterns=["embeddings", "pooler", "qa_outputs"],
        mvm_limit=256,
        use_dynamic_delta=False,
        use_delta_anneal=False,
        delta_loss_weight=0.0
    )
    
    # Load PTQ-calibrated weights (includes calibrated scales!)
    # Prefer safetensors if present
    import os
    state_dict = None
    pt_path = os.path.join(args.model_path, "pytorch_model.bin")
    st_path = os.path.join(args.model_path, "model.safetensors")
    if os.path.exists(pt_path):
        state_dict = torch.load(pt_path, map_location='cpu')
    elif os.path.exists(st_path):
        try:
            from safetensors.torch import load_file as safe_load
            state_dict = safe_load(st_path)
        except Exception as e:
            raise RuntimeError(f"Found safetensors but failed to load: {e}")
    else:
        raise FileNotFoundError(f"No weights found in {args.model_path} (expected pytorch_model.bin or model.safetensors)")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"⚠️  Missing keys: {len(missing)} (this is OK if they're optimizer states)")
    if unexpected:
        print(f"⚠️  Unexpected keys: {len(unexpected)}")
    
    print("✓ Loaded PTQ model with calibrated scales")
    
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

