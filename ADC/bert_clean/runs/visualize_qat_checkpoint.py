#!/usr/bin/env python3
"""
Visualize QAT checkpoint and optionally upload to existing WandB run.

Usage:
    python visualize_qat_checkpoint.py \
        --checkpoint_dir ./checkpoints/outputs_adc_qat_k16_conservative/checkpoint-2000 \
        --wandb_run_id hvuslkoh \
        --wandb_project bert-adc-qat
"""

import argparse
import os
import logging
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

from ADC.bert_clean.core.adc_layers import TiledLinearADC, QATLinearADC
from ADC.bert_clean.runs.bert_adc_integration import (
    load_qa_model_robust,
    find_last_checkpoint_dir,
    prepare_validation_features,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_adc_visualizations(model, sample_input, layer_patterns, title_prefix="", output_subdir="./viz_qat"):
    """
    Generate visualizations for ADC layers showing before/after ADC quantization.
    Same as PTQ visualization but for QAT checkpoint.
    """
    os.makedirs(output_subdir, exist_ok=True)
    logger.info(f"Generating ADC visualizations: {title_prefix}")
    
    # Find ADC layers to visualize
    layers_to_viz = []
    for name, module in model.named_modules():
        if isinstance(module, QATLinearADC):
            if any(pattern in name for pattern in layer_patterns):
                layers_to_viz.append((name, module))
        elif isinstance(module, TiledLinearADC) and len(module.tiles) > 0:
            if any(pattern in name for pattern in layer_patterns):
                # Visualize first tile as representative
                layers_to_viz.append((name + ".tiles.0", module.tiles[0]))
    
    if not layers_to_viz:
        logger.warning(f"No ADC layers found matching patterns: {layer_patterns}")
        return {}
    
    logger.info(f"Found {len(layers_to_viz)} layers to visualize")
    
    # Capture data for each layer
    captured_data = {}
    
    def make_hook(layer_name):
        def hook(module, input, output):
            with torch.no_grad():
                x = input[0]
                
                # Get quantizers
                act_q = module.activation_quantizer
                w_q = module.weight_quantizer
                
                # Build codes
                s_x = act_q.scale
                if act_q.symmetric:
                    code_x = torch.clamp(torch.round(x / s_x), act_q.qmin, act_q.qmax)
                else:
                    zp_x = act_q.zero_point
                    code_x_temp = torch.clamp(torch.round(x / s_x + zp_x), 0, act_q.qmax)
                    
                    if hasattr(module, 'ashift') and module.ashift:
                        code_x = code_x_temp - module.C
                    else:
                        code_x = code_x_temp - zp_x
                
                # Weight codes
                s_w_vec = w_q.scale
                s_w_b = s_w_vec.view(-1, 1)
                code_w = torch.clamp(torch.round(module.weight / s_w_b), w_q.qmin, w_q.qmax)
                
                # Integer MM (before ADC)
                y_int = F.linear(code_x, code_w, bias=None)
                
                # ADC quantization
                delta = module.adc_quantizer._delta
                na = module.adc_quantizer.na
                pa = module.adc_quantizer.pa
                y_adc_codes = torch.clamp(torch.floor(y_int / delta), na, pa)
                y_after_adc = y_adc_codes * delta
                
                captured_data[layer_name] = {
                    'x_raw': x.detach().cpu().numpy(),
                    'code_x': code_x.detach().cpu().numpy(),
                    'w_raw': module.weight.detach().cpu().numpy(),
                    'code_w': code_w.detach().cpu().numpy(),
                    'y_int_before_adc': y_int.detach().cpu().numpy(),
                    'y_after_adc': y_after_adc.detach().cpu().numpy(),
                    'y_adc_codes': y_adc_codes.detach().cpu().numpy(),
                    's_x': s_x.detach().cpu().item() if s_x.numel() == 1 else s_x.detach().cpu().mean().item(),
                    's_w': s_w_vec.detach().cpu().numpy(),
                    'delta': delta.detach().cpu().item(),
                    'na': na,
                    'pa': pa,
                }
        return hook
    
    # Attach hooks
    hooks = []
    for name, module in layers_to_viz:
        hook = module.register_forward_hook(make_hook(name))
        hooks.append(hook)
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        _ = model(**sample_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Generate plots
    result_paths = {}
    for name, _ in layers_to_viz:
        if name not in captured_data:
            continue
        
        try:
            clean_name = name.replace(".", "_").replace("/", "_")
            filename = f"{clean_name}_{title_prefix.replace(' ', '_')}.png"
            filepath = os.path.join(output_subdir, filename)
            
            plot_adc_pipeline(captured_data[name], name, title_prefix, filepath)
            
            result_paths[clean_name] = filepath
            logger.info(f"  Created: {name}")
        except Exception as e:
            logger.error(f"  Failed: {name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Generated {len(result_paths)} visualizations in {output_subdir}")
    return result_paths


def plot_adc_pipeline(data: Dict, layer_name: str, title_prefix: str, filepath: str):
    """Plot ADC pipeline showing before/after ADC quantization"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{title_prefix}: {layer_name}", fontsize=14, fontweight='bold')
    
    # Take first sample for visualization
    x_raw = data['x_raw'][0].flatten()[:1000]
    code_x = data['code_x'][0].flatten()[:1000]
    w_raw = data['w_raw'].flatten()[:1000]
    code_w = data['code_w'].flatten()[:1000]
    y_before = data['y_int_before_adc'][0].flatten()[:1000]
    y_after = data['y_after_adc'][0].flatten()[:1000]
    y_codes = data['y_adc_codes'][0].flatten()[:1000]
    
    # Row 1: Activations and Weights
    axes[0, 0].hist(x_raw, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title(f'X_raw\nrange: [{x_raw.min():.3f}, {x_raw.max():.3f}]')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(code_x, bins=50, alpha=0.7, color='cyan', edgecolor='black')
    axes[0, 1].set_title(f'X_codes\nscale: {data["s_x"]:.6f}')
    axes[0, 1].set_xlabel('Code')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].hist(w_raw, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 2].set_title(f'W_raw\nrange: [{w_raw.min():.3f}, {w_raw.max():.3f}]')
    axes[0, 2].set_xlabel('Value')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[0, 3].hist(code_w, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 3].set_title(f'W_codes\nscale: [{data["s_w"].min():.6f}, {data["s_w"].max():.6f}]')
    axes[0, 3].set_xlabel('Code')
    axes[0, 3].set_ylabel('Count')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: ADC Input/Output
    na, pa = data['na'], data['pa']
    delta = data['delta']
    
    axes[1, 0].hist(y_before, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(na * delta, color='red', linestyle='--', linewidth=2, label=f'ADC min={na*delta:.1f}')
    axes[1, 0].axvline(pa * delta, color='red', linestyle='--', linewidth=2, label=f'ADC max={pa*delta:.1f}')
    axes[1, 0].set_title(f'BEFORE ADC (Y_int)\nrange: [{y_before.min():.1f}, {y_before.max():.1f}]')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(y_codes, bins=min(50, int(pa - na + 1)), alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].axvline(na, color='darkred', linestyle='--', linewidth=2, label=f'na={na}')
    axes[1, 1].axvline(pa, color='darkred', linestyle='--', linewidth=2, label=f'pa={pa}')
    axes[1, 1].set_title(f'ADC codes\ndelta={delta:.3f}')
    axes[1, 1].set_xlabel('ADC Code')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(y_after, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].set_title(f'AFTER ADC\nrange: [{y_after.min():.1f}, {y_after.max():.1f}]')
    axes[1, 2].set_xlabel('Value')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Comparison: before vs after ADC
    axes[1, 3].hist(y_before, bins=50, alpha=0.5, color='orange', label='Before ADC', edgecolor='black')
    axes[1, 3].hist(y_after, bins=50, alpha=0.5, color='purple', label='After ADC', edgecolor='black')
    axes[1, 3].axvline(na * delta, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[1, 3].axvline(pa * delta, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[1, 3].set_title('Before vs After ADC')
    axes[1, 3].set_xlabel('Value')
    axes[1, 3].set_ylabel('Count')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    
    # Calculate clipping statistics
    clipped_low = (y_codes == na).sum()
    clipped_high = (y_codes == pa).sum()
    total = y_codes.size
    clip_pct = 100.0 * (clipped_low + clipped_high) / total
    
    # Add text with statistics
    stats_text = f"Clipping: {clip_pct:.2f}% ({clipped_low} low, {clipped_high} high)"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Visualize QAT checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Path to QAT checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="./viz_qat",
                       help="Where to save visualizations")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                       help="WandB run ID to resume and upload to (e.g., 'hvuslkoh')")
    parser.add_argument("--wandb_project", type=str, default="bert-adc-qat",
                       help="WandB project name")
    parser.add_argument("--visualize_layers", type=str, nargs="+",
                       default=["layer.0.attention.output.dense", 
                               "layer.5.intermediate.dense", 
                               "layer.11.output.dense"],
                       help="Layer patterns to visualize")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Find checkpoint
    checkpoint_dir = find_last_checkpoint_dir(args.checkpoint_dir)
    logger.info(f"Loading checkpoint from: {checkpoint_dir}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = load_qa_model_robust(checkpoint_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on {device}")
    
    # Prepare sample input
    logger.info("Loading sample data...")
    raw = load_dataset("squad")
    sample_examples = raw["validation"].select(range(1))
    sample_dataset = sample_examples.map(
        lambda x: prepare_validation_features(x, tokenizer, 384, 128),
        batched=True,
        remove_columns=sample_examples.column_names,
    )
    
    sample_input = {
        'input_ids': torch.tensor([sample_dataset[0]['input_ids']]).to(device),
        'attention_mask': torch.tensor([sample_dataset[0]['attention_mask']]).to(device),
    }
    if 'token_type_ids' in sample_dataset[0]:
        sample_input['token_type_ids'] = torch.tensor([sample_dataset[0]['token_type_ids']]).to(device)
    
    logger.info(f"Sample input prepared: {sample_input['input_ids'].shape}")
    
    # Generate visualizations
    viz_paths = generate_adc_visualizations(
        model, 
        sample_input, 
        args.visualize_layers,
        title_prefix="QAT Checkpoint",
        output_subdir=args.output_dir
    )
    
    # Upload to WandB if requested
    if args.wandb_run_id and WANDB_AVAILABLE:
        logger.info(f"Resuming WandB run: {args.wandb_run_id}")
        wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="must"
        )
        
        for name, img_path in viz_paths.items():
            wandb.log({f"viz_qat/{name}": wandb.Image(img_path)})
            logger.info(f"  Uploaded: {name}")
        
        wandb.finish()
        logger.info("WandB upload complete!")
    
    logger.info("=" * 60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Generated {len(viz_paths)} visualizations in: {args.output_dir}")
    if args.wandb_run_id:
        logger.info(f"Uploaded to WandB run: {args.wandb_run_id}")


if __name__ == "__main__":
    main()


# python ADC/bert_clean/runs/visualize_qat_checkpoint.py \
#     --checkpoint_dir ./ADC/bert_clean/checkpoints/outputs_adc_qat_k16_conservative/checkpoint-2000 \
#     --wandb_run_id hvuslkoh \
#     --wandb_project bert-adc-qat \
#     --output_dir ./viz_qat