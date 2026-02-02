#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) for ADC-based LLaMA models
Calibrates ADC quantizers using a calibration dataset and evaluates perplexity

Supports:
- meta-llama/Llama-3.1-8B
- meta-llama/Llama-3.2-3B
- meta-llama/Llama-3.2-1B
"""

import argparse
import os
import logging
import math
from typing import Optional, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from torch.utils.data import DataLoader
from datetime import datetime

import sys
from pathlib import Path

# Import ADC layers from llama module
from ADC.llama.core.adc_layers import TiledLinearADC, QATLinearADC
# WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be disabled")

# No complex monitoring needed for PTQ - we'll use simple visualization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaADCConverter:
    """Convert LLaMA model to use ADC QAT layers for causal LM."""

    @staticmethod
    def is_after_silu(name: str) -> bool:
        """
        Detect if this linear layer follows a SiLU activation.
        In LLaMA MLP: output = down_proj(silu(gate_proj(x)) * up_proj(x))
        So down_proj receives SiLU output.
        """
        return "down_proj" in name

    @staticmethod
    def replace_linear_with_adc(
        model: nn.Module,
        bx: int = 8,
        bw: int = 8,
        ba: int = 8,
        k: int = 4,
        ashift: bool = False,
        exclude_patterns: Optional[list[str]] = None,
        mvm_limit: int = 256,
        use_kurtosis_loss: bool = False,
        kurtosis_weight: float = 0.0,
        target_kurtosis: float = 1.8,
    ) -> nn.Module:
        """
        Replace all nn.Linear layers in the LLaMA model with TiledLinearADC.

        Args:
            model: LLaMA model to convert.
            bx: Bits for activation quantization.
            bw: Bits for weight quantization.
            ba: Bits for ADC quantization.
            k: Hardware design parameter for ADC.
            ashift: Enable A-shift for layers after SiLU. When True, down_proj
                    uses asymmetric quantization + A-shift, all others use symmetric.
            exclude_patterns: List of substrings of module names to exclude.
            mvm_limit: Memory vector multiplication limit for tiling.
        """
        if exclude_patterns is None:
            # Default: don't quantize embeddings and lm_head
            exclude_patterns = ["embed_tokens", "lm_head"]

        def should_exclude(name: str) -> bool:
            return any(pat in name for pat in exclude_patterns)

        def replace_recursive(module: nn.Module, name: str = ""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                if isinstance(child_module, nn.Linear) and not should_exclude(full_name):
                    # Apply A-shift ONLY to layers that receive SiLU outputs (down_proj)
                    layer_ashift = ashift and LlamaADCConverter.is_after_silu(full_name)
                    # A-shift requires asymmetric quantization, others use symmetric
                    layer_signed_activations = not layer_ashift

                    adc_layer = TiledLinearADC(
                        in_features=child_module.in_features,
                        out_features=child_module.out_features,
                        bias=(child_module.bias is not None),
                        bx=bx,
                        bw=bw,
                        ba=ba,
                        k=k,
                        ashift=layer_ashift,
                        signed_activations=layer_signed_activations,
                        mvm_limit=mvm_limit,
                        use_kurtosis_loss=use_kurtosis_loss,
                        kurtosis_weight=kurtosis_weight,
                        target_kurtosis=target_kurtosis,
                    )
                    # Load weights from original layer
                    adc_layer.load_weights(child_module)

                    quant_type = "A-shift (asymmetric)" if layer_ashift else "symmetric"
                    logger.info(f"Replaced {full_name} with TiledLinearADC ({quant_type}, bx={bx}, bw={bw}, ba={ba}, k={k})")

                    setattr(module, child_name, adc_layer)
                else:
                    replace_recursive(child_module, full_name)

        replace_recursive(model)
        return model

    @staticmethod
    def count_adc_layers(model: nn.Module) -> dict:
        """Count ADC and regular linear layers in the model."""
        counts = {"adc_linear": 0, "regular_linear": 0, "total_params": 0}
        for _, module in model.named_modules():
            if isinstance(module, TiledLinearADC):
                counts["adc_linear"] += 1
            elif isinstance(module, nn.Linear):
                counts["regular_linear"] += 1
            if hasattr(module, "parameters"):
                counts["total_params"] += sum(p.numel() for p in module.parameters())
        return counts


def append_current_date_to_path(path_base: str) -> str:
    """Append current date to a path (for output directories)"""
    current_date = datetime.now().strftime("%Y%m%d")
    path_with_date = f"{path_base}_{current_date}"
    logger.info(f"Output directory with current date: {path_with_date}")
    return path_with_date


def show_model_with_adc_hooks(model, visualize_patterns):
    """
    Print the full module tree and highlight:
      • every TiledLinearADC (parent layer) that will be calibrated  →  📊
      • layers that will also be visualised                          →  ⭐
    
    Note: Individual tiles are NOT shown, only parent TiledLinearADC layers
    
    Returns:
        str: The formatted model structure text
    """
    def will_visualise(name):
        return any(pat in name for pat in visualize_patterns)
    
    def is_tile_child(name):
        """Check if this is an individual tile (e.g., 'dense.tiles.0')"""
        return '.tiles.' in name

    def format_line(level, name, module):
        # Skip individual tiles - we only show parent TiledLinearADC
        if is_tile_child(name):
            return None
            
        bullet = "└─ " if level > 0 else ""
        indent = "   " * max(level - 1, 0) + bullet
        module_type = module.__class__.__name__
        tag = ""
        
        # Mark TiledLinearADC (parent layer with tiles)
        if isinstance(module, TiledLinearADC):
            tag = " 📊"  # hooked by calibrator (each tile will be hooked)
            if will_visualise(name):
                tag += "⭐"  # also plotted
        
        # Only show leaves or modules with tags
        if tag or not list(module.children()):
            return f"{indent}{name or 'model'} ({module_type}){tag}"
        return None

    # Build the output as a list of lines
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("MODEL STRUCTURE WITH ADC HOOKS")
    output_lines.append("="*80)
    output_lines.append("Legend: 📊 = calibration hook,  ⭐ = visualization")
    output_lines.append("-"*80)
    
    for name, module in model.named_modules():
        level = len(name.split(".")) if name else 0
        line = format_line(level, name, module)
        if line:
            output_lines.append(line)
    
    output_lines.append("="*80)
    
    # Join and log
    output_text = "\n".join(output_lines)
    logger.info("\n" + output_text)
    
    return output_text


class ADCCalibrator:
    """Calibrates ADC quantizers using activation statistics"""
    
    def __init__(self, model: nn.Module, method: str = "minmax", bx: int = 8, bw: int = 8):
        self.model = model
        self.method = method
        self.bx = bx
        self.bw = bw
        self.stats = {}
        # Note: signed_activations is now detected per-layer from each quantizer
        
    def register_hooks(self):
        """Register forward hooks to collect activation statistics"""
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if name not in self.stats:
                    self.stats[name] = {
                        'act_min': [],
                        'act_max': [],
                        'act_absmax': [],
                        'w_min': [],
                        'w_max': [],
                        'w_absmax': [],
                        'y_int_min': [],
                        'y_int_max': [],
                        'y_int_absmax': [],
                        'module': module,  # Store module reference to check quantizer settings
                    }
                
                # Collect activation stats
                x = input[0].detach()
                self.stats[name]['act_min'].append(x.min().item())
                self.stats[name]['act_max'].append(x.max().item())
                self.stats[name]['act_absmax'].append(x.abs().max().item())
                
                # Collect weight stats
                w = module.weight.detach()
                self.stats[name]['w_min'].append(w.min().item())
                self.stats[name]['w_max'].append(w.max().item())
                self.stats[name]['w_absmax'].append(w.abs().max().item())
                
                # Collect integer MM output (before ADC)
                # Simulate quantization to get y_int
                with torch.no_grad():
                    act_q = module.activation_quantizer
                    s_x = act_q.scale
                    if act_q.symmetric:
                        code_x = torch.clamp(torch.round(x / s_x), act_q.qmin, act_q.qmax)
                    else:
                        # Unsigned path: quantize to [0, 2^bx - 1] using zero_point offset
                        zp_x = act_q.zero_point
                        code_x_temp = torch.clamp(torch.round(x / s_x + zp_x), 0, act_q.qmax)
                        
                        if hasattr(module, 'ashift') and module.ashift:
                            # A-shift: subtract fixed C instead of learned zp_x
                            code_x = code_x_temp - module.C
                        else:
                            # Standard asymmetric: subtract learned zero_point to center
                            code_x = code_x_temp - zp_x
                    
                    w_q = module.weight_quantizer
                    s_w_vec = w_q.scale
                    s_w_b = s_w_vec.view(-1, 1)
                    code_w = torch.clamp(torch.round(w / s_w_b), w_q.qmin, w_q.qmax)
                    
                    y_int = torch.nn.functional.linear(code_x, code_w, bias=None)
                    
                    self.stats[name]['y_int_min'].append(y_int.min().item())
                    self.stats[name]['y_int_max'].append(y_int.max().item())
                    self.stats[name]['y_int_absmax'].append(y_int.abs().max().item())
            
            return hook
        
        # Register hooks on QATLinearADC layers (including tiles inside TiledLinearADC)
        # First, let's see what modules exist (debug)
        logger.info("Scanning model for ADC layers...")
        logger.info(f"QATLinearADC class: {QATLinearADC}")
        logger.info(f"TiledLinearADC class: {TiledLinearADC}")
        
        adc_count = 0
        tiled_count = 0
        sample_module_type = None
        checked_sample = False
        
        for name, module in self.model.named_modules():
            if isinstance(module, QATLinearADC):
                adc_count += 1
                if adc_count <= 3:  # Show first 3
                    logger.info(f"  Found QATLinearADC: {name}")
            elif isinstance(module, TiledLinearADC):
                tiled_count += 1
                if tiled_count <= 3:  # Show first 3
                    logger.info(f"  Found TiledLinearADC: {name} with {len(module.tiles)} tiles")
            
            # Check for a sample module (use full path)
            if not checked_sample and 'bert.encoder.layer.0.attention.self.query' == name:
                checked_sample = True
                sample_module_type = type(module).__name__
                logger.info(f"  *** Sample module 'bert.encoder.layer.0.attention.self.query':")
                logger.info(f"      Type: {type(module)}")
                logger.info(f"      Class name: {type(module).__name__}")
                logger.info(f"      Module: {type(module).__module__}")
                logger.info(f"      Is QATLinearADC? {isinstance(module, QATLinearADC)}")
                logger.info(f"      Is TiledLinearADC? {isinstance(module, TiledLinearADC)}")
                if hasattr(module, 'tiles'):
                    logger.info(f"      Has 'tiles' attribute with {len(module.tiles)} tiles")
                    if len(module.tiles) > 0:
                        first_tile = module.tiles[0]
                        logger.info(f"      First tile type: {type(first_tile)}")
                        logger.info(f"      First tile is QATLinearADC? {isinstance(first_tile, QATLinearADC)}")
                        logger.info(f"      QATLinearADC from adc_layers: {QATLinearADC}")
                        logger.info(f"      First tile's class: {type(first_tile)}")
                        logger.info(f"      Are they same class? {type(first_tile) is QATLinearADC}")
        
        logger.info(f"Total: {adc_count} QATLinearADC, {tiled_count} TiledLinearADC")
        logger.info("Registering hooks...")
        
        # Now register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, QATLinearADC):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
                logger.info(f"Registered hook on QATLinearADC: {name}")
            elif isinstance(module, TiledLinearADC):
                # Register hooks on individual tiles
                for tile_idx, tile in enumerate(module.tiles):
                    tile_name = f"{name}.tiles.{tile_idx}"
                    hook = tile.register_forward_hook(make_hook(tile_name))
                    hooks.append(hook)
                    logger.info(f"Registered hook on tile: {tile_name}")
        
        return hooks
    
    def calibrate(self, dataloader, num_batches: int = 100):
        """Run calibration on dataloader"""
        logger.info(f"Running calibration on {num_batches} batches...")
        
        self.model.eval()
        hooks = self.register_hooks()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Calibrating")):
                if i >= num_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                try:
                    _ = self.model(**batch)
                except Exception as e:
                    logger.warning(f"Error in batch {i}: {e}")
                    continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        logger.info(f"Collected stats for {len(self.stats)} layers")
    
    def compute_optimal_params(self, log_to_wandb: bool = False) -> Dict[str, Dict]:
        """
        Compute optimal quantization SCALES (not delta!) from collected statistics.
        
        Key insight: Delta is a hardware constant and cannot be changed.
        We calibrate activation/weight scales to optimally use the fixed ADC range.
        
        Goal: Make y_int values fit well within the ADC dynamic range
        given fixed delta and ADC range [-na, pa].
        """
        optimal_params = {}
        
        # Aggregate statistics for wandb logging
        all_act_scales = []
        all_w_scales = []
        all_y_int_targets = []
        
        for name, stats in self.stats.items():
            if not stats['y_int_absmax']:  # No data collected
                continue
            
            # Collect statistics
            act_absmax_arr = np.array(stats['act_absmax'])
            w_absmax_arr = np.array(stats['w_absmax'])
            y_int_absmax_arr = np.array(stats['y_int_absmax'])
            
            # Use percentile to be robust against outliers
            if self.method == "minmax":
                act_absmax = act_absmax_arr.max()
                w_absmax = w_absmax_arr.max()
                y_int_target = y_int_absmax_arr.max()
            elif self.method == "percentile":
                act_absmax = np.percentile(act_absmax_arr, 99.9)
                w_absmax = np.percentile(w_absmax_arr, 99.9)
                y_int_target = np.percentile(y_int_absmax_arr, 99.9)
            elif self.method == "mse":
                act_absmax = self._find_mse_optimal_threshold(act_absmax_arr)
                w_absmax = self._find_mse_optimal_threshold(w_absmax_arr)
                y_int_target = self._find_mse_optimal_threshold(y_int_absmax_arr)
            else:
                act_absmax = act_absmax_arr.max()
                w_absmax = w_absmax_arr.max()
                y_int_target = y_int_absmax_arr.max()
            
            # Compute optimal scales
            # Check if this layer uses symmetric (signed) or asymmetric (unsigned) quantization
            # by inspecting the actual quantizer settings
            module = stats.get('module')
            if module and hasattr(module, 'activation_quantizer'):
                is_symmetric = module.activation_quantizer.symmetric
            else:
                # Fallback: assume symmetric if can't determine
                is_symmetric = True
            
            # For symmetric quantization: scale = absmax / (2^(n-1) - 1)
            # For asymmetric: scale = absmax / (2^n - 1)
            if is_symmetric:
                act_levels = 2 ** (self.bx - 1) - 1  # e.g., 127 for 8-bit signed
            else:
                act_levels = 2 ** self.bx - 1  # e.g., 255 for 8-bit unsigned
            
            w_levels = 2 ** (self.bw - 1) - 1  # Weights always symmetric, e.g., 127 for 8-bit
            
            optimal_act_scale = act_absmax / float(act_levels)
            optimal_w_scale = w_absmax / float(w_levels)
            
            optimal_params[name] = {
                'act_scale': optimal_act_scale,
                'w_scale': optimal_w_scale,
                'y_int_target': y_int_target,  # For monitoring
            }
            
            all_act_scales.append(optimal_act_scale)
            all_w_scales.append(optimal_w_scale)
            all_y_int_targets.append(y_int_target)
        
        # Log calibration statistics to wandb
        if log_to_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "calibration/num_layers": len(optimal_params),
                "calibration/act_scale_mean": np.mean(all_act_scales),
                "calibration/act_scale_std": np.std(all_act_scales),
                "calibration/act_scale_min": np.min(all_act_scales),
                "calibration/act_scale_max": np.max(all_act_scales),
                "calibration/w_scale_mean": np.mean(all_w_scales),
                "calibration/w_scale_std": np.std(all_w_scales),
                "calibration/w_scale_min": np.min(all_w_scales),
                "calibration/w_scale_max": np.max(all_w_scales),
                "calibration/y_int_target_mean": np.mean(all_y_int_targets),
                "calibration/y_int_target_std": np.std(all_y_int_targets),
                "calibration/y_int_target_max": np.max(all_y_int_targets),
            })
        
        return optimal_params
    
    def _find_mse_optimal_threshold(self, values: np.ndarray) -> float:
        """Find threshold that minimizes MSE"""
        candidates = np.percentile(values, [90, 95, 99, 99.5, 99.9, 100])
        best_mse = float('inf')
        best_threshold = candidates[-1]
        
        for threshold in candidates:
            clipped = np.clip(values, -threshold, threshold)
            mse = np.mean((values - clipped) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_threshold = threshold
        
        return best_threshold
    
    def apply_calibration(self, optimal_params: Dict[str, Dict]):
        """
        Apply calibrated SCALES to the model.
        
        IMPORTANT: We do NOT change delta - it's a hardware constant!
        We only calibrate activation and weight scales to optimally use
        the fixed ADC range.
        """
        logger.info("Applying calibrated scales to model...")
        
        updated_act = 0
        updated_w = 0
        
        for name, module in self.model.named_modules():
            # Handle QATLinearADC (standalone or as tiles)
            if isinstance(module, QATLinearADC) and name in optimal_params:
                params = optimal_params[name]
                
                with torch.no_grad():
                    # Update activation quantizer scale
                    if hasattr(module, 'activation_quantizer'):
                        old_scale = module.activation_quantizer.scale.item()
                        module.activation_quantizer.scale.copy_(
                            torch.tensor(params['act_scale'], dtype=torch.float32)
                        )
                        module.activation_quantizer._scale_initialized = True
                        
                        logger.info(f"{name} [ACT]: scale {old_scale:.6f} -> {params['act_scale']:.6f}")
                        updated_act += 1
                    
                    # Update weight quantizer scale (per-channel)
                    if hasattr(module, 'weight_quantizer'):
                        w_q = module.weight_quantizer
                        old_scale_mean = w_q.scale.mean().item() if w_q.scale.numel() > 0 else 0.01
                        
                        # For per-channel, compute scales from actual weights
                        if w_q.per_channel:
                            # Compute per-channel scales directly from weights
                            weight = module.weight.detach()  # [out_features, in_features]
                            
                            if w_q.channel_dim == 0:
                                # For each output channel, find absmax across input features
                                per_channel_absmax = weight.abs().max(dim=1)[0]  # [out_features]
                            else:
                                # For other channel dims
                                weight_transposed = weight.transpose(w_q.channel_dim, 0)
                                per_channel_absmax = weight_transposed.contiguous().view(weight_transposed.shape[0], -1).abs().max(dim=1)[0]
                            
                            # Avoid division by zero
                            per_channel_absmax = torch.clamp(per_channel_absmax, min=1e-6)
                            
                            # Compute per-channel scales for symmetric quantization
                            # scale = absmax / (2^(n-1) - 1) = absmax / 127
                            new_scales = per_channel_absmax / 127.0
                            
                            # Update scales - resize if necessary
                            if w_q.scale.numel() != new_scales.numel():
                                # Resize the scale parameter to match per-channel size
                                w_q.scale.data = w_q.scale.data.new_zeros(new_scales.shape)
                            w_q.scale.data.copy_(new_scales)
                            w_q._scale_initialized = True
                        else:
                            # Per-tensor: use scalar scale
                            w_q.scale.copy_(
                                torch.tensor(params['w_scale'], dtype=torch.float32)
                            )
                            w_q._scale_initialized = True
                        
                        new_scale_mean = w_q.scale.mean().item()
                        
                        logger.info(f"{name} [W]: scale {old_scale_mean:.6f} -> {new_scale_mean:.6f}")
                        updated_w += 1
                    
                    # NOTE: ADC delta stays as analytical value - it's a hardware constant!
                    # We just ensure that with calibrated scales, y_int fits well in ADC range
        
        logger.info(f"Updated {updated_act} activation quantizers and {updated_w} weight quantizers")
        logger.info("NOTE: ADC delta values remain as hardware-defined constants")


def main():
    parser = argparse.ArgumentParser(description="PTQ for ADC-based LLaMA models")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                       help="HuggingFace model name (e.g., meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.1-8B)")
    parser.add_argument("--output_dir", type=str, default="./outputs_llama_adc_ptq",
                       help="Where to save calibrated model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Model dtype for loading")
    
    # ADC settings
    parser.add_argument("--bx", type=int, default=8, help="Activation bits")
    parser.add_argument("--bw", type=int, default=8, help="Weight bits")
    parser.add_argument("--ba", type=int, default=8, help="ADC bits")
    parser.add_argument("--k", type=int, default=4, help="Hardware parameter")
    parser.add_argument("--ashift", action="store_true",
                       help="Enable A-shift quantization strategy: "
                            "asymmetric (unsigned) quantization + A-shift for SiLU outputs (down_proj). "
                            "If False, uses symmetric (signed) quantization for all activations.")
    parser.add_argument("--mvm_limit", type=int, default=256)
    
    # Calibration settings
    parser.add_argument("--calibration_method", type=str, default="percentile",
                       choices=["minmax", "percentile", "mse"],
                       help="Calibration method for quantizer scales")
    parser.add_argument("--num_calibration_batches", type=int, default=100,
                       help="Number of batches for calibration")
    parser.add_argument("--calibration_batch_size", type=int, default=4)
    
    # Evaluation settings
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length for tokenization")
    parser.add_argument("--max_eval_batches", type=int, default=100,
                       help="Maximum batches for perplexity evaluation")
    
    # WandB settings
    parser.add_argument("--wandb_project", type=str, default="llama-adc-ptq",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--disable_wandb", action="store_true",
                       help="Disable WandB logging")
    
    # Visualization settings
    parser.add_argument("--disable_visualizations", action="store_true",
                       help="Disable ADC visualizations")
    parser.add_argument("--visualize_layers", type=str, nargs="+",
                       default=["layers.0.self_attn.q_proj", "layers.0.mlp.down_proj", "layers.15.mlp.gate_proj"],
                       help="Layer patterns to visualize")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Log quantization strategy
    logger.info(f"Quantization strategy: ashift={args.ashift}")
    if args.ashift:
        logger.info("  → Asymmetric (unsigned) + A-shift for layers AFTER SiLU (down_proj)")
        logger.info("  → Symmetric (signed) for all OTHER activations")
    else:
        logger.info("  → Symmetric (signed) quantization for ALL activations")
    
    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and not args.disable_wandb
    model_short_name = args.model_name.split("/")[-1]
    if use_wandb:
        run_name = args.wandb_run_name or f"ptq_{model_short_name}_bx{args.bx}_bw{args.bw}_ba{args.ba}_k{args.k}_{args.calibration_method}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": args.model_name,
                "bx": args.bx,
                "bw": args.bw,
                "ba": args.ba,
                "k": args.k,
                "ashift": args.ashift,
                "ashift_mode": "per_layer_silu" if args.ashift else "none",
                "quantization_note": "A-shift on down_proj only" if args.ashift else "Symmetric for all",
                "mvm_limit": args.mvm_limit,
                "calibration_method": args.calibration_method,
                "num_calibration_batches": args.num_calibration_batches,
                "calibration_batch_size": args.calibration_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "max_length": args.max_length,
                "torch_dtype": args.torch_dtype,
                "seed": args.seed,
            }
        )
        logger.info(f"WandB initialized: project={args.wandb_project}, run={run_name}")
    else:
        logger.info("WandB logging disabled")
    
    # Load LLaMA model
    logger.info(f"Loading LLaMA model: {args.model_name}")
    
    # Add current date to output directory
    args.output_dir = append_current_date_to_path(args.output_dir)
    
    # Determine dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.torch_dtype, torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # LLaMA uses left padding for generation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model - use device_map for large models
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Get the actual device the model is on
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'hf_device_map'):
        # For device_map="auto", get the first device
        device = next(iter(model.hf_device_map.values())) if model.hf_device_map else device
    
    logger.info(f"Model loaded with dtype={torch_dtype}, device={device}")
    
    # Convert to ADC layers
    logger.info("Converting to ADC layers...")
    model = LlamaADCConverter.replace_linear_with_adc(
        model,
        bx=args.bx,
        bw=args.bw,
        ba=args.ba,
        k=args.k,
        ashift=args.ashift,
        exclude_patterns=["embed_tokens", "lm_head"],
        mvm_limit=args.mvm_limit,
        # PTQ: no kurtosis loss during calibration
        use_kurtosis_loss=False,
        kurtosis_weight=0.0,
        target_kurtosis=1.8,
    )
    
    # Move model to device if not already there
    if not hasattr(model, 'hf_device_map'):
        model = model.to(device)
    
    stats = LlamaADCConverter.count_adc_layers(model)
    logger.info(f"Model: {stats['adc_linear']} ADC layers, {stats['regular_linear']} regular Linear, {stats['total_params']:,} params")
    
    # Show model structure with ADC hooks
    model_structure_text = show_model_with_adc_hooks(model, args.visualize_layers)
    
    # Log model structure to wandb
    if use_wandb:
        wandb.run.summary["model_structure"] = model_structure_text
        logger.info("Logged model structure to WandB")
    
    # Prepare sample input for visualization (if enabled)
    sample_input = None
    if not args.disable_visualizations:
        logger.info("Preparing sample input for visualizations...")
    
    # Load calibration data - WikiText-2 for language modeling
    logger.info("Loading WikiText-2 dataset for calibration...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Tokenize function for causal LM
    def tokenize_function(examples):
        # Filter out empty texts
        texts = [t for t in examples["text"] if t.strip()]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors=None,
        )
        return tokenized
    
    # Use train split for calibration
    calibration_dataset = raw["train"].filter(lambda x: len(x["text"].strip()) > 50)
    calibration_dataset = calibration_dataset.select(range(min(2000, len(calibration_dataset))))
    calibration_dataset = calibration_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing calibration data",
    )
    # Filter out empty examples
    calibration_dataset = calibration_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    # Get sample for visualization
    if not args.disable_visualizations and len(calibration_dataset) > 0:
        sample_input = {
            'input_ids': torch.tensor([calibration_dataset[0]['input_ids']]).to(device),
            'attention_mask': torch.tensor([calibration_dataset[0]['attention_mask']]).to(device),
        }
        logger.info(f"Sample input prepared for visualization (shape: {sample_input['input_ids'].shape})")
    
    # Custom collator for causal LM
    def calibration_collator(features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
        }
        return batch
    
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.calibration_batch_size,
        shuffle=False,
        collate_fn=calibration_collator,
    )
    
    # Run calibration
    logger.info("="*80)
    logger.info("STEP 1: CALIBRATION")
    logger.info("="*80)
    
    # Visualize BEFORE calibration
    viz_before = None
    if sample_input is not None and not args.disable_visualizations:
        viz_before = _generate_adc_visualizations(
            model, sample_input, args.visualize_layers,
            title_prefix="BEFORE Calibration",
            output_subdir=os.path.join(args.output_dir, "viz_before")
        )
    
    calibrator = ADCCalibrator(
        model, 
        method=args.calibration_method,
        bx=args.bx,
        bw=args.bw
    )
    calibrator.calibrate(calibration_loader, num_batches=args.num_calibration_batches)
    
    # Compute optimal parameters
    logger.info("Computing optimal quantization scales...")
    optimal_params = calibrator.compute_optimal_params(log_to_wandb=use_wandb)
    
    # Apply calibration
    calibrator.apply_calibration(optimal_params)

    # Set quantizers to 'fixed' mode to prevent accidental updates
    for name, module in model.named_modules():
        if isinstance(module, (QATLinearADC, TiledLinearADC)):
            if hasattr(module, 'set_quantizer_mode'):
                module.set_quantizer_mode('fixed')
    logger.info("✓ Quantizers set to 'fixed' mode after calibration")
    
    # Visualize AFTER calibration
    viz_after = None
    if sample_input is not None and not args.disable_visualizations:
        viz_after = _generate_adc_visualizations(
            model, sample_input, args.visualize_layers,
            title_prefix="AFTER Calibration",
            output_subdir=os.path.join(args.output_dir, "viz_after")
        )
    
    # Log per-layer calibration stats to wandb
    if use_wandb and len(optimal_params) > 0:
        # Create histograms of calibrated parameters
        layer_names = list(optimal_params.keys())
        act_scales = [p['act_scale'] for p in optimal_params.values()]
        w_scales = [p['w_scale'] for p in optimal_params.values()]
        y_int_targets = [p['y_int_target'] for p in optimal_params.values()]
        
        wandb.log({
            "calibration/act_scale_histogram": wandb.Histogram(act_scales),
            "calibration/w_scale_histogram": wandb.Histogram(w_scales),
            "calibration/y_int_target_histogram": wandb.Histogram(y_int_targets),
        })
        
        logger.info(f"✅ Logged calibration histograms for {len(optimal_params)} layers to WandB")
        
        # Log before/after visualizations
        if viz_before:
            for name, img_path in viz_before.items():
                wandb.log({f"viz_before/{name}": wandb.Image(img_path)})
            logger.info(f"✅ Uploaded {len(viz_before)} BEFORE visualizations to WandB")
        if viz_after:
            for name, img_path in viz_after.items():
                wandb.log({f"viz_after/{name}": wandb.Image(img_path)})
            logger.info(f"✅ Uploaded {len(viz_after)} AFTER visualizations to WandB")
    
    # Evaluate perplexity
    logger.info("="*80)
    logger.info("STEP 2: PERPLEXITY EVALUATION")
    logger.info("="*80)
    
    # Prepare validation dataset
    eval_dataset = raw["validation"].filter(lambda x: len(x["text"].strip()) > 50)
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation data",
    )
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    # Custom collator for evaluation
    def eval_collator(features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
        }
        return batch
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_collator,
    )
    
    logger.info(f"Running perplexity evaluation on {len(eval_dataset)} samples...")
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", total=min(args.max_eval_batches, len(eval_loader))):
            if num_batches >= args.max_eval_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # For causal LM, labels = input_ids (shifted internally by the model)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            # Count non-padding tokens
            num_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    eval_metrics = {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_batches": num_batches,
    }
    
    logger.info("="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Perplexity:    {perplexity:.2f}")
    logger.info(f"Avg Loss:      {avg_loss:.4f}")
    logger.info(f"Total Tokens:  {total_tokens:,}")
    
    # Log all metrics to wandb
    if use_wandb:
        wandb.log({
            "eval/perplexity": perplexity,
            "eval/avg_loss": avg_loss,
            "eval/total_tokens": total_tokens,
        })
        
        # Summary statistics
        wandb.run.summary["perplexity"] = perplexity
        wandb.run.summary["avg_loss"] = avg_loss
        wandb.run.summary["num_calibrated_layers"] = len(optimal_params)
    
    # Save calibrated model
    logger.info(f"Saving calibrated model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save calibration info
    with open(os.path.join(args.output_dir, "calibration_info.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write("LLaMA ADC POST-TRAINING QUANTIZATION (PTQ) RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Calibration method: {args.calibration_method}\n")
        f.write(f"Calibration batches: {args.num_calibration_batches}\n")
        f.write(f"ADC hardware config: bx={args.bx}, bw={args.bw}, ba={args.ba}, k={args.k}\n")
        f.write(f"A-shift: {args.ashift}\n")
        if args.ashift:
            f.write(f"Quantization strategy: Per-layer (A-shift for SiLU outputs only)\n")
            f.write(f"  - Layers after SiLU (down_proj): Asymmetric + A-shift\n")
            f.write(f"  - All other layers: Symmetric (signed)\n")
        else:
            f.write(f"Quantization strategy: Symmetric (signed) for all activations\n")
        f.write(f"\n")
        f.write("NOTE: ADC delta is a HARDWARE CONSTANT and cannot be changed!\n")
        f.write("      We calibrate activation/weight SCALES to optimally use the fixed ADC range.\n")
        f.write(f"\n")
        f.write(f"Results:\n")
        f.write(f"  Perplexity:  {eval_metrics['perplexity']:.2f}\n")
        f.write(f"  Avg Loss:    {eval_metrics['avg_loss']:.4f}\n")
        f.write(f"\n")
        f.write(f"Calibrated layers: {len(optimal_params)}\n")
        f.write(f"\n")
        f.write("Per-layer calibrated scales:\n")
        f.write("-" * 80 + "\n")
        for name, params in sorted(optimal_params.items()):
            f.write(f"\n{name}:\n")
            f.write(f"  Activation scale: {params['act_scale']:.6f}\n")
            f.write(f"  Weight scale:     {params['w_scale']:.6f}\n")
            f.write(f"  Y_int target:     {params['y_int_target']:.2f}\n")
    
    with open(os.path.join(args.output_dir, "eval_metrics.txt"), "w") as f:
        for k, v in sorted(eval_metrics.items()):
            f.write(f"{k}: {v}\n")
    
    logger.info("="*80)
    logger.info("PTQ COMPLETE!")
    logger.info("="*80)
    logger.info(f"Calibrated model saved to: {args.output_dir}")
    logger.info(f"Perplexity: {eval_metrics['perplexity']:.2f}")
    
    if viz_before or viz_after:
        logger.info(f"📊 Visualizations:")
        if viz_before:
            logger.info(f"  Before calibration: {os.path.join(args.output_dir, 'viz_before')}")
        if viz_after:
            logger.info(f"  After calibration:  {os.path.join(args.output_dir, 'viz_after')}")
    
    # Final WandB logging
    if use_wandb:
        # Log paths info to summary (not as separate metrics)
        wandb.run.summary["output_dir"] = args.output_dir
        wandb.run.summary["model_path"] = os.path.join(args.output_dir, "pytorch_model.bin")
        
        if viz_before:
            wandb.run.summary["viz_before_count"] = len(viz_before)
        if viz_after:
            wandb.run.summary["viz_after_count"] = len(viz_after)
        
        logger.info("✅ Logged all metrics to WandB")
        
        # Finish wandb run
        wandb.finish()
        logger.info("WandB run finished")


def _generate_adc_visualizations(model, sample_input, layer_patterns, title_prefix="", output_subdir="./viz"):
    """
    Generate visualizations for ADC layers showing before/after ADC quantization
    
    Returns:
        Dict[str, str]: Mapping of layer_name -> image_path
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_subdir, exist_ok=True)
    logger.info(f"Generating ADC visualizations: {title_prefix}")
    
    # Find ADC layers to visualize
    layers_to_viz = []
    for name, module in model.named_modules():
        if isinstance(module, QATLinearADC):
            # Check against patterns
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
                    # Unsigned path: quantize to [0, 2^bx - 1] using zero_point offset
                    zp_x = act_q.zero_point
                    code_x_temp = torch.clamp(torch.round(x / s_x + zp_x), 0, act_q.qmax)
                    
                    if hasattr(module, 'ashift') and module.ashift:
                        # A-shift: subtract fixed C instead of learned zp_x
                        code_x = code_x_temp - module.C
                    else:
                        # Standard asymmetric: subtract learned zero_point to center
                        code_x = code_x_temp - zp_x
                
                # Weight codes
                s_w_vec = w_q.scale
                s_w_b = s_w_vec.view(-1, 1)
                code_w = torch.clamp(torch.round(module.weight / s_w_b), w_q.qmin, w_q.qmax)
                
                # Integer MM (before ADC)
                y_int = F.linear(code_x, code_w, bias=None)
                
                # ADC quantization
                delta = module.delta
                na = module.na
                pa = module.pa
                # Use floor to match actual ADC implementation (Paper Equation 2)
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
                    's_x': s_x.detach().cpu().item(),
                    's_w': s_w_vec.detach().cpu().numpy(),
                    'delta': delta,
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
            
            _plot_adc_pipeline(captured_data[name], name, title_prefix, filepath)
            
            result_paths[clean_name] = filepath
            logger.info(f"  ✓ {name}")
        except Exception as e:
            logger.error(f"  ✗ {name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Generated {len(result_paths)} visualizations in {output_subdir}")
    return result_paths


def _plot_adc_pipeline(data: Dict, layer_name: str, title_prefix: str, filepath: str):
    """Plot ADC pipeline showing before/after ADC quantization"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{title_prefix}: {layer_name}", fontsize=14, fontweight='bold')
    
    # Take first sample for visualization
    x_raw = data['x_raw'][0].flatten()[:1000]  # First 1000 elements
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
    
    axes[1, 1].hist(y_codes, bins=min(50, pa - na + 1), alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].axvline(na, color='darkred', linestyle='--', linewidth=2, label=f'na={na}')
    axes[1, 1].axvline(pa, color='darkred', linestyle='--', linewidth=2, label=f'pa={pa}')
    axes[1, 1].set_title(f'ADC codes\nΔ={delta:.3f}')
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


if __name__ == "__main__":
    main()

