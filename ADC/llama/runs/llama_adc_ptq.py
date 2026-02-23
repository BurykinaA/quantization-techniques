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

from ADC.llama.core.adc_layers import TiledLinearADC, QATLinearADC

import wandb

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
        signed_activations: bool = None,
        exclude_patterns: list[str] | None = None,
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
            signed_activations: If True, use symmetric (signed) activation quantization
                    for all layers. If False, use asymmetric (unsigned) for all layers.
                    If None (default), behaviour depends on ashift: A-shift layers get
                    asymmetric, non-A-shift layers get symmetric.
            exclude_patterns: List of substrings of module names to exclude.
            mvm_limit: Memory vector multiplication limit for tiling.
        """
        if exclude_patterns is None:
            exclude_patterns = ["embed_tokens", "lm_head"]

        def should_exclude(name: str) -> bool:
            return any(pat in name for pat in exclude_patterns)

        def replace_recursive(module: nn.Module, name: str = ""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                if isinstance(child_module, nn.Linear) and not should_exclude(full_name):
                    # Apply A-shift ONLY to layers that receive SiLU outputs (down_proj)
                    layer_ashift = ashift and LlamaADCConverter.is_after_silu(full_name)

                    # Determine activation quantization mode
                    if signed_activations is not None:
                        # Explicit override from user
                        layer_signed_activations = signed_activations
                    else:
                        # Legacy: A-shift layers use asymmetric, others symmetric
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


def load_dataset_by_name(dataset_name: str, split: str = "train"):
    """
    Load a dataset by name.
    
    Args:
        dataset_name: "wikitext2" or "c4"
        split: "train", "validation", or "test"
    
    Returns:
        HuggingFace dataset
    """
    if dataset_name == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")
        return raw[split]
    elif dataset_name == "c4":
        # C4 is large, use streaming for calibration or load a subset
        if split == "train":
            # For calibration, load a subset of C4
            raw = load_dataset("allenai/c4", "en", split="train", streaming=True)
            # Take first N samples for calibration
            return raw
        else:
            # For validation, use the validation split
            raw = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            return raw
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: wikitext2, c4")


def load_and_tokenize_for_sliding_window(
    dataset_name: str, 
    split: str, 
    tokenizer, 
    max_samples: int = 1000
):
    """
    Load and tokenize a dataset into one long sequence for sliding window evaluation.
    This is the standard approach used in papers like GPTQ, AWQ, FlatQuant.
    
    Args:
        dataset_name: "wikitext2" or "c4"
        split: "train", "validation", or "test"
        tokenizer: Tokenizer to use
        max_samples: Maximum samples for C4 (ignored for WikiText-2)
    
    Returns:
        dict with 'input_ids' tensor of shape [1, total_tokens]
    """
    logger.info(f"Loading {dataset_name} ({split} split) for sliding window evaluation...")
    
    if dataset_name == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        # Concatenate all text
        text = "\n\n".join([t for t in raw["text"] if t.strip()])
    elif dataset_name == "c4":
        # C4 is huge, use streaming and limit samples
        raw = load_dataset("allenai/c4", "en", split=split, streaming=True)
        texts = []
        for i, example in enumerate(raw):
            if max_samples and i >= max_samples:
                break
            if example["text"].strip():
                texts.append(example["text"])
        text = "\n\n".join(texts)
        logger.info(f"Loaded {len(texts)} samples from C4")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Total text length: {len(text):,} characters")
    
    # Tokenize the entire text as one sequence
    encodings = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,  # Don't add BOS/EOS between concatenated texts
    )
    
    logger.info(f"Total tokens: {encodings['input_ids'].size(1):,}")
    
    return encodings


def compute_perplexity_sliding_window(
    model, 
    encodings, 
    device,
    max_length: int = 2048,
    stride: int = None,
    desc: str = "Evaluating"
):
    """
    Compute perplexity using sliding window approach (standard for papers).
    
    This is the proper way to evaluate perplexity on language models:
    1. Concatenate all text into one long sequence
    2. Use sliding window with stride
    3. Only count loss on the "new" tokens (stride portion) to avoid double counting
    
    Args:
        model: The language model
        encodings: Tokenized text (dict with 'input_ids' tensor of shape [1, seq_len])
        device: Device to run on
        max_length: Context window size (should match model's context length)
        stride: How many tokens to advance each step (default: max_length // 2)
        desc: Description for progress bar
    
    Returns:
        dict with perplexity, avg_loss, total_tokens, num_windows
    """
    if stride is None:
        stride = max_length // 2  # 50% overlap is common
    
    model.eval()
    
    input_ids = encodings["input_ids"]
    seq_len = input_ids.size(1)
    
    logger.info(f"  Total tokens in corpus: {seq_len:,}")
    logger.info(f"  Context window: {max_length}, Stride: {stride}")
    
    nlls = []  # Negative log likelihoods
    total_tokens = 0
    
    # Calculate number of windows
    num_windows = max(1, (seq_len - max_length) // stride + 1)
    
    prev_end_loc = 0
    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, stride), desc=desc, total=num_windows):
            end_loc = min(begin_loc + max_length, seq_len)
            
            # Get the window
            input_ids_window = input_ids[:, begin_loc:end_loc].to(device)
            
            # Target length: only count loss on the "new" tokens
            # This avoids double-counting when using overlapping windows
            target_len = end_loc - prev_end_loc
            
            # Create labels: -100 for tokens we've already counted
            labels = input_ids_window.clone()
            labels[:, :-target_len] = -100  # Mask already-counted tokens
            
            outputs = model(
                input_ids=input_ids_window,
                labels=labels,
            )
            
            # Accumulate the loss weighted by number of target tokens
            neg_log_likelihood = outputs.loss * target_len
            nlls.append(neg_log_likelihood.item())
            total_tokens += target_len
            
            prev_end_loc = end_loc
            
            # Stop if we've processed the whole sequence
            if end_loc >= seq_len:
                break
    
    # Compute average loss and perplexity
    avg_loss = sum(nlls) / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_windows": len(nlls),
        "max_length": max_length,
        "stride": stride,
    }


def prepare_dataset_for_lm(dataset, tokenizer, max_length: int, max_samples: int = None, 
                           min_text_length: int = 50, streaming: bool = False):
    """
    Prepare a dataset for language modeling evaluation.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        max_samples: Maximum number of samples to use (None = all)
        min_text_length: Minimum text length to keep
        streaming: Whether the dataset is streaming
    
    Returns:
        Processed dataset ready for DataLoader
    """
    def tokenize_function(examples):
        texts = [t for t in examples["text"] if t.strip() and len(t.strip()) > min_text_length]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        return tokenized
    
    if streaming:
        # For streaming datasets (C4), collect samples first
        samples = []
        for i, example in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            if len(example["text"].strip()) > min_text_length:
                samples.append(example)
        
        # Convert to regular dataset
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
    else:
        # Filter short texts
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > min_text_length)
        
        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
    
    # Tokenize
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    # Filter empty examples
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    return dataset


class ADCCalibrator:
    """Calibrates ADC quantizers using activation statistics"""
    
    def __init__(self, model: nn.Module, method: str = "minmax", bx: int = 8, bw: int = 8):
        self.model = model
        self.method = method
        self.bx = bx
        self.bw = bw
        self.stats = {}
        # Attention mask for the current batch, set by the calibrate() method
        # so that hooks can filter out padding positions
        self._current_attention_mask = None
        
    def register_hooks(self):
        """Register forward hooks to collect activation statistics.
        
        IMPORTANT: Hooks use self._current_attention_mask to exclude padding
        positions from activation statistics.  The mask is set before each
        forward pass in calibrate().
        """
        hooks = []
        calibrator = self  # capture reference for inner function
        
        def make_hook(name):
            def hook(module, input, output):
                if name not in calibrator.stats:
                    calibrator.stats[name] = {
                        'act_min': [],
                        'act_max': [],
                        'act_absmax': [],
                        'w_min': [],
                        'w_max': [],
                        'w_absmax': [],
                        'y_int_min': [],
                        'y_int_max': [],
                        'y_int_absmax': [],
                        'module': module,
                    }
                
                # Collect activation stats -- filter out padding positions
                x = input[0].detach()
                
                mask = calibrator._current_attention_mask
                if mask is not None:
                    # mask shape: [batch, seq_len], x shape: [batch, seq_len, dim]
                    # or after tiling x might be [batch*seq_len, tile_dim]
                    if x.ndim == 3 and mask.ndim == 2 and mask.shape[0] == x.shape[0] and mask.shape[1] == x.shape[1]:
                        # Standard case: x is [batch, seq_len, dim]
                        # Select only non-padding positions
                        bool_mask = mask.bool()  # [batch, seq_len]
                        x_valid = x[bool_mask]   # [N_valid, dim]
                    elif x.ndim == 2 and mask.ndim == 2:
                        # Tiled case: TiledLinearADC reshapes to [batch*seq_len, tile_dim]
                        # Flatten mask to match
                        flat_mask = mask.reshape(-1).bool()  # [batch*seq_len]
                        if flat_mask.shape[0] == x.shape[0]:
                            x_valid = x[flat_mask]
                        else:
                            # Shape mismatch -- fall back to using all values
                            x_valid = x
                    else:
                        x_valid = x
                else:
                    x_valid = x
                
                if x_valid.numel() == 0:
                    # Degenerate case: all positions masked -- skip this batch
                    return
                
                calibrator.stats[name]['act_min'].append(x_valid.min().item())
                calibrator.stats[name]['act_max'].append(x_valid.max().item())
                calibrator.stats[name]['act_absmax'].append(x_valid.abs().max().item())
                
                # Collect weight stats (weights are not affected by attention mask)
                w = module.weight.detach()
                calibrator.stats[name]['w_min'].append(w.min().item())
                calibrator.stats[name]['w_max'].append(w.max().item())
                calibrator.stats[name]['w_absmax'].append(w.abs().max().item())
                
                # Collect integer MM output (before ADC)
                # Use only valid (non-padding) activations for y_int statistics too
                with torch.no_grad():
                    act_q = module.activation_quantizer
                    s_x = act_q.scale.to(x_valid.device)
                    if act_q.symmetric:
                        code_x = torch.clamp(torch.round(x_valid / s_x), act_q.qmin, act_q.qmax)
                    else:
                        zp_x = act_q.zero_point.to(x_valid.device)
                        code_x_temp = torch.clamp(torch.round(x_valid / s_x + zp_x), 0, act_q.qmax)
                        
                        if hasattr(module, 'ashift') and module.ashift:
                            code_x = code_x_temp - module.C
                        else:
                            code_x = code_x_temp - zp_x
                    
                    w_q = module.weight_quantizer
                    s_w_vec = w_q.scale.to(w.device)
                    s_w_b = s_w_vec.view(-1, 1)
                    code_w = torch.clamp(torch.round(w / s_w_b), w_q.qmin, w_q.qmax)
                    
                    y_int = torch.nn.functional.linear(code_x, code_w, bias=None)
                    
                    calibrator.stats[name]['y_int_min'].append(y_int.min().item())
                    calibrator.stats[name]['y_int_max'].append(y_int.max().item())
                    calibrator.stats[name]['y_int_absmax'].append(y_int.abs().max().item())
            
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, QATLinearADC):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
            elif isinstance(module, TiledLinearADC):
                for tile_idx, tile in enumerate(module.tiles):
                    tile_name = f"{name}.tiles.{tile_idx}"
                    hook = tile.register_forward_hook(make_hook(tile_name))
                    hooks.append(hook)
        
        logger.info(f"Registered {len(hooks)} calibration hooks")
        
        return hooks
    
    def calibrate(self, dataloader, num_batches: int = 100):
        """Run calibration on dataloader.
        
        The attention_mask from each batch is stored in
        self._current_attention_mask so that hooks can filter out
        padding positions when collecting activation statistics.
        """
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
                
                # Store attention mask for hooks to use
                self._current_attention_mask = batch.get("attention_mask", None)
                
                # Forward pass
                try:
                    _ = self.model(**batch)
                except Exception as e:
                    logger.warning(f"Error in batch {i}: {e}")
                    continue
                finally:
                    self._current_attention_mask = None
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        logger.info(f"Collected stats for {len(self.stats)} layers")
    
    def compute_optimal_params(self, log_to_wandb: bool = False) -> dict[str, dict]:
        """
        Compute optimal quantization SCALES (and zero-points for asymmetric
        activations) from collected statistics.
        
        Key insight: Delta is a hardware constant and cannot be changed.
        We calibrate activation/weight scales to optimally use the fixed ADC range.
        
        For symmetric quantization:
            scale = absmax / (2^(n-1) - 1)
            zero_point = 0
        
        For asymmetric quantization:
            scale = (x_max - x_min) / (2^n - 1)
            zero_point = clamp(round(-x_min / scale), 0, 2^n - 1)
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
            act_min_arr = np.array(stats['act_min'])
            act_max_arr = np.array(stats['act_max'])
            w_absmax_arr = np.array(stats['w_absmax'])
            y_int_absmax_arr = np.array(stats['y_int_absmax'])
            
            # Check if this layer uses symmetric or asymmetric activation quantization
            module = stats.get('module')
            if module and hasattr(module, 'activation_quantizer'):
                is_symmetric = module.activation_quantizer.symmetric
            else:
                is_symmetric = True
            
            # Robust range estimation
            if self.method == "minmax":
                act_absmax = act_absmax_arr.max()
                act_min_val = act_min_arr.min()
                act_max_val = act_max_arr.max()
                w_absmax = w_absmax_arr.max()
                y_int_target = y_int_absmax_arr.max()
            elif self.method == "percentile":
                act_absmax = np.percentile(act_absmax_arr, 99.9)
                act_min_val = np.percentile(act_min_arr, 0.1)
                act_max_val = np.percentile(act_max_arr, 99.9)
                w_absmax = np.percentile(w_absmax_arr, 99.9)
                y_int_target = np.percentile(y_int_absmax_arr, 99.9)
            elif self.method == "mse":
                act_absmax = self._find_mse_optimal_threshold(act_absmax_arr)
                act_min_val = act_min_arr.min()
                act_max_val = act_max_arr.max()
                w_absmax = self._find_mse_optimal_threshold(w_absmax_arr)
                y_int_target = self._find_mse_optimal_threshold(y_int_absmax_arr)
            else:
                act_absmax = act_absmax_arr.max()
                act_min_val = act_min_arr.min()
                act_max_val = act_max_arr.max()
                w_absmax = w_absmax_arr.max()
                y_int_target = y_int_absmax_arr.max()
            
            # Weights always use symmetric quantization
            w_levels = 2 ** (self.bw - 1) - 1  # e.g., 127 for 8-bit
            optimal_w_scale = max(w_absmax, 1e-8) / float(w_levels)
            
            # Activations: symmetric or asymmetric
            if is_symmetric:
                act_levels = 2 ** (self.bx - 1) - 1  # e.g., 127 for 8-bit signed
                optimal_act_scale = max(act_absmax, 1e-8) / float(act_levels)
                optimal_act_zp = 0.0
            else:
                # Asymmetric: scale = (max - min) / (2^n - 1)
                act_range = max(act_max_val - act_min_val, 1e-8)
                act_qmax = 2 ** self.bx - 1  # e.g., 255 for 8-bit unsigned
                optimal_act_scale = act_range / float(act_qmax)
                # zero_point = clamp(round(-min / scale), 0, qmax)
                optimal_act_zp = float(np.clip(
                    np.round(-act_min_val / optimal_act_scale), 0, act_qmax
                ))
            
            optimal_params[name] = {
                'act_scale': optimal_act_scale,
                'act_zero_point': optimal_act_zp,
                'act_symmetric': is_symmetric,
                'w_scale': optimal_w_scale,
                'y_int_target': y_int_target,
            }
            
            all_act_scales.append(optimal_act_scale)
            all_w_scales.append(optimal_w_scale)
            all_y_int_targets.append(y_int_target)
        
        # Log calibration statistics to wandb
        if log_to_wandb and wandb.run is not None:
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
    
    def apply_calibration(self, optimal_params: dict[str, dict]):
        """
        Apply calibrated SCALES (and zero-points for asymmetric activations)
        to the model.
        
        IMPORTANT: We do NOT change delta -- it's a hardware constant!
        We only calibrate activation and weight scales (and zero-points)
        to optimally use the fixed ADC range.
        """
        logger.info("Applying calibrated scales to model...")
        
        updated_act = 0
        updated_w = 0
        
        for name, module in self.model.named_modules():
            # Handle QATLinearADC (standalone or as tiles)
            if isinstance(module, QATLinearADC) and name in optimal_params:
                params = optimal_params[name]
                
                with torch.no_grad():
                    # Update activation quantizer scale + zero_point
                    if hasattr(module, 'activation_quantizer'):
                        act_q = module.activation_quantizer
                        old_scale = act_q.scale.item()
                        act_q.scale.copy_(
                            torch.tensor(params['act_scale'], dtype=torch.float32)
                        )
                        act_q._scale_initialized = True
                        
                        # Update zero_point for asymmetric quantization
                        if not act_q.symmetric:
                            old_zp = act_q.zero_point.item()
                            act_q.zero_point.copy_(
                                torch.tensor(params['act_zero_point'], dtype=torch.float32)
                            )
                            act_q._zp_initialized = True
                            logger.info(
                                f"{name} [ACT asym]: scale {old_scale:.6f} -> {params['act_scale']:.6f}, "
                                f"zp {old_zp:.2f} -> {params['act_zero_point']:.2f}"
                            )
                        else:
                            logger.info(f"{name} [ACT sym]: scale {old_scale:.6f} -> {params['act_scale']:.6f}")
                        updated_act += 1
                    
                    # Update weight quantizer scale (per-channel, always symmetric)
                    if hasattr(module, 'weight_quantizer'):
                        w_q = module.weight_quantizer
                        old_scale_mean = w_q.scale.mean().item() if w_q.scale.numel() > 0 else 0.01
                        
                        # For per-channel, compute scales from actual weights
                        if w_q.per_channel:
                            weight = module.weight.detach()  # [out_features, in_features]
                            
                            if w_q.channel_dim == 0:
                                per_channel_absmax = weight.abs().max(dim=1)[0]  # [out_features]
                            else:
                                weight_transposed = weight.transpose(w_q.channel_dim, 0)
                                per_channel_absmax = weight_transposed.contiguous().view(weight_transposed.shape[0], -1).abs().max(dim=1)[0]
                            
                            per_channel_absmax = torch.clamp(per_channel_absmax, min=1e-6)
                            
                            w_levels = 2 ** (self.bw - 1) - 1  # e.g. 127 for 8-bit
                            new_scales = per_channel_absmax / float(w_levels)
                            
                            if w_q.scale.numel() != new_scales.numel():
                                w_q.scale.data = w_q.scale.data.new_zeros(new_scales.shape)
                            w_q.scale.data.copy_(new_scales)
                            w_q._scale_initialized = True
                        else:
                            w_q.scale.copy_(
                                torch.tensor(params['w_scale'], dtype=torch.float32)
                            )
                            w_q._scale_initialized = True
                        
                        new_scale_mean = w_q.scale.mean().item()
                        logger.info(f"{name} [W sym]: scale {old_scale_mean:.6f} -> {new_scale_mean:.6f}")
                        updated_w += 1
                    
                    # NOTE: ADC delta stays as analytical value -- it's a hardware constant!
        
        logger.info(f"Updated {updated_act} activation quantizers and {updated_w} weight quantizers")
        logger.info("NOTE: ADC delta values remain as hardware-defined constants")


def diagnose_quantized_model(model, tokenizer, device, num_layers_to_print: int = 5):
    """
    Run diagnostics on the quantized model to verify calibration health.
    
    Checks:
    1. Whether calibrated scales are reasonable (not default, not NaN/Inf)
    2. Whether the model produces valid logits (no NaN/Inf, reasonable magnitude)
    3. Prints a summary of per-layer scale statistics
    """
    logger.info("=" * 80)
    logger.info("DIAGNOSTICS: Checking quantized model health")
    logger.info("=" * 80)

    # --- 1. Check quantizer scale statistics ---
    act_scales = []
    w_scales_mean = []
    n_default_act = 0
    n_default_w = 0
    layer_info = []

    for name, module in model.named_modules():
        if isinstance(module, QATLinearADC):
            aq = module.activation_quantizer
            wq = module.weight_quantizer
            a_s = aq.scale.detach().float()
            w_s = wq.scale.detach().float()

            a_val = a_s.item() if a_s.numel() == 1 else a_s.mean().item()
            w_val = w_s.mean().item()
            act_scales.append(a_val)
            w_scales_mean.append(w_val)

            if abs(a_val - 0.01) < 1e-6 or abs(a_val - 0.02) < 1e-6:
                n_default_act += 1
            if w_s.numel() == 1 and abs(w_val - 0.01) < 1e-6:
                n_default_w += 1

            layer_info.append((name, a_val, w_val, w_s.shape, module.delta,
                               aq.symmetric, wq.per_channel))

    total = len(layer_info)
    logger.info(f"Total QATLinearADC layers: {total}")
    logger.info(f"Layers with DEFAULT act scale (likely uncalibrated): {n_default_act}/{total}")
    logger.info(f"Layers with DEFAULT w scale   (likely uncalibrated): {n_default_w}/{total}")

    if act_scales:
        logger.info(f"Activation scale range: [{min(act_scales):.6f}, {max(act_scales):.6f}]")
    if w_scales_mean:
        logger.info(f"Weight scale mean range: [{min(w_scales_mean):.6f}, {max(w_scales_mean):.6f}]")

    # Print a few layers for inspection
    for name, a_s, w_s, w_shape, delta, sym, pc in layer_info[:num_layers_to_print]:
        logger.info(
            f"  {name}: act_s={a_s:.6f} ({'sym' if sym else 'asym'}), "
            f"w_s_mean={w_s:.6f} (shape={list(w_shape)}, pc={pc}), delta={delta:.2f}"
        )

    if n_default_act > 0 or n_default_w > 0:
        logger.warning(
            ">>> SOME LAYERS STILL HAVE DEFAULT SCALES! "
            "Calibration probably did not reach these layers."
        )

    # --- 2. Run a quick forward pass and check logits ---
    logger.info("Running diagnostic forward pass...")
    text = "The quick brown fox jumps over the lazy dog. " * 5
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    has_nan = logits.isnan().any().item()
    has_inf = logits.isinf().any().item()
    logit_min = logits.min().item()
    logit_max = logits.max().item()
    logit_std = logits.float().std().item()

    logger.info(f"Logits shape: {list(logits.shape)}, dtype: {logits.dtype}")
    logger.info(f"Logits range: [{logit_min:.2f}, {logit_max:.2f}], std: {logit_std:.2f}")
    logger.info(f"Contains NaN: {has_nan}, Contains Inf: {has_inf}")

    if has_nan or has_inf:
        logger.error(">>> CRITICAL: Model produces NaN/Inf! Quantization is numerically broken.")
    elif logit_std < 0.01:
        logger.warning(">>> WARNING: Logit std is near zero -- model output is collapsed.")
    elif logit_std > 1000:
        logger.warning(">>> WARNING: Logit std is huge -- possible scale miscalibration.")
    else:
        logger.info("Logits look healthy (no NaN/Inf, reasonable std).")

    # --- 3. Quick greedy-decode sanity check ---
    pred_ids = logits[0, -1, :].argmax().item()
    pred_token = tokenizer.decode([pred_ids])
    logger.info(f"Next-token prediction for test sentence: '{pred_token}' (id={pred_ids})")

    logger.info("=" * 80)
    return {
        "n_layers": total,
        "n_default_act": n_default_act,
        "n_default_w": n_default_w,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "logit_range": (logit_min, logit_max),
        "logit_std": logit_std,
    }


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
    parser.add_argument("--activation_quant", type=str, default=None,
                       choices=["symmetric", "asymmetric"],
                       help="Activation quantization mode. "
                            "'symmetric' = signed, zero_point=0. "
                            "'asymmetric' = unsigned, with learned zero_point. "
                            "If not set, defaults to per-layer logic based on ashift.")
    parser.add_argument("--mvm_limit", type=int, default=256)
    
    # Calibration settings
    parser.add_argument("--calibration_method", type=str, default="percentile",
                       choices=["minmax", "percentile", "mse"],
                       help="Calibration method for quantizer scales")
    parser.add_argument("--num_calibration_batches", type=int, default=100,
                       help="Number of batches for calibration")
    parser.add_argument("--calibration_batch_size", type=int, default=4)
    
    # Dataset settings
    parser.add_argument("--calibration_dataset", type=str, default="wikitext2",
                       choices=["wikitext2", "c4"],
                       help="Dataset to use for calibration (wikitext2 or c4)")
    parser.add_argument("--eval_datasets", type=str, nargs="+", default=["wikitext2"],
                       choices=["wikitext2", "c4"],
                       help="Datasets to evaluate on (can specify multiple: wikitext2 c4)")
    
    # Evaluation settings (Sliding Window - standard for papers)
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Context window size for perplexity evaluation (2048 is standard for papers)")
    parser.add_argument("--stride", type=int, default=None,
                       help="Stride for sliding window evaluation (default: max_length // 2)")
    parser.add_argument("--eval_split", type=str, default="test",
                       choices=["train", "validation", "test"],
                       help="Split to use for evaluation (test is standard for papers)")
    parser.add_argument("--max_eval_samples", type=int, default=1000,
                       help="Maximum samples for C4 evaluation (ignored for WikiText-2)")
    # Keep calibration batch size for calibration phase
    parser.add_argument("--calibration_max_length", type=int, default=512,
                       help="Max sequence length for calibration (can be shorter than eval)")
    
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
    parser.add_argument("--check_baseline", action="store_true",
                       help="Run FP16 baseline perplexity BEFORE ADC conversion (for debugging)")
    parser.add_argument("--visualize_layers", type=str, nargs="+",
                       default=[], #["layers.0.self_attn.q_proj", "layers.0.mlp.down_proj", "layers.15.mlp.gate_proj"]
                       help="Layer patterns to visualize")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Derive signed_activations flag from --activation_quant
    if args.activation_quant == "symmetric":
        signed_activations = True
    elif args.activation_quant == "asymmetric":
        signed_activations = False
    else:
        signed_activations = None  # per-layer logic based on ashift
    
    # Log quantization strategy
    logger.info(f"Quantization strategy: ashift={args.ashift}, activation_quant={args.activation_quant}")
    if signed_activations is True:
        logger.info("  → Symmetric (signed) activation quantization for ALL layers")
    elif signed_activations is False:
        logger.info("  → Asymmetric (unsigned) activation quantization for ALL layers")
    elif args.ashift:
        logger.info("  → Asymmetric (unsigned) + A-shift for layers AFTER SiLU (down_proj)")
        logger.info("  → Symmetric (signed) for all OTHER activations")
    else:
        logger.info("  → Symmetric (signed) quantization for ALL activations (default)")
    logger.info("  → Weights: always symmetric (signed) per-channel")
    
    # Initialize WandB
    use_wandb = not args.disable_wandb
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
                "activation_quant": args.activation_quant,
                "signed_activations": signed_activations,
                "ashift_mode": "per_layer_silu" if args.ashift else "none",
                "quantization_note": "A-shift on down_proj only" if args.ashift else "Symmetric for all",
                "mvm_limit": args.mvm_limit,
                "calibration_method": args.calibration_method,
                "calibration_dataset": args.calibration_dataset,
                "eval_datasets": args.eval_datasets,
                "num_calibration_batches": args.num_calibration_batches,
                "calibration_batch_size": args.calibration_batch_size,
                "calibration_max_length": args.calibration_max_length,
                "eval_max_length": args.max_length,
                "eval_stride": args.stride,
                "eval_split": args.eval_split,
                "perplexity_method": "sliding_window",
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
    
    # =========================================================================
    # Optional: FP16 baseline perplexity check (before any quantization)
    # =========================================================================
    if args.check_baseline:
        logger.info("=" * 80)
        logger.info("BASELINE CHECK: FP16 perplexity (no quantization)")
        logger.info("=" * 80)
        baseline_enc = load_and_tokenize_for_sliding_window(
            args.eval_datasets[0], args.eval_split, tokenizer,
            max_samples=args.max_eval_samples if args.eval_datasets[0] == "c4" else None,
        )
        baseline_metrics = compute_perplexity_sliding_window(
            model, baseline_enc, device,
            max_length=args.max_length, stride=args.stride,
            desc="Baseline FP16",
        )
        logger.info(f"FP16 BASELINE Perplexity: {baseline_metrics['perplexity']:.4f}  "
                     f"(Loss: {baseline_metrics['avg_loss']:.4f})")
        if use_wandb:
            wandb.log({"baseline/perplexity": baseline_metrics["perplexity"],
                        "baseline/avg_loss": baseline_metrics["avg_loss"]})
        logger.info("=" * 80)
    
    # Convert to ADC layers
    logger.info("Converting to ADC layers...")
    model = LlamaADCConverter.replace_linear_with_adc(
        model,
        bx=args.bx,
        bw=args.bw,
        ba=args.ba,
        k=args.k,
        ashift=args.ashift,
        signed_activations=signed_activations,
        exclude_patterns=["embed_tokens", "lm_head"],
        mvm_limit=args.mvm_limit,
        # PTQ: no kurtosis loss during calibration
        use_kurtosis_loss=False,
        kurtosis_weight=0.0,
        target_kurtosis=1.8,
    )
    
    # Move entire model to device to ensure newly created ADC quantizers are on GPU
    # (The conversion creates new LearnableQuantizer parameters that default to CPU)
    model = model.to(device)
    logger.info(f"Model moved to {device}")
    
    stats = LlamaADCConverter.count_adc_layers(model)
    logger.info(f"Model: {stats['adc_linear']} ADC layers, {stats['regular_linear']} regular Linear, {stats['total_params']:,} params")
    
    # Show model structure with ADC hooks
    viz_patterns = args.visualize_layers if not args.disable_visualizations else []
    model_structure_text = show_model_with_adc_hooks(model, viz_patterns)
    
    # Log model structure to wandb
    if use_wandb:
        wandb.run.summary["model_structure"] = model_structure_text
        logger.info("Logged model structure to WandB")
    
    # Prepare sample input for visualization (if enabled)
    sample_input = None
    if not args.disable_visualizations:
        logger.info("Preparing sample input for visualizations...")
    
    # =========================================================================
    # Load calibration dataset
    # =========================================================================
    logger.info(f"Loading {args.calibration_dataset.upper()} dataset for calibration...")
    
    is_streaming = (args.calibration_dataset == "c4")
    calib_raw = load_dataset_by_name(args.calibration_dataset, split="train")
    
    calibration_dataset = prepare_dataset_for_lm(
        calib_raw, 
        tokenizer, 
        max_length=args.calibration_max_length,  # Use shorter sequences for calibration
        max_samples=2000,
        min_text_length=50,
        streaming=is_streaming
    )
    
    logger.info(f"Calibration dataset: {len(calibration_dataset)} samples from {args.calibration_dataset}")
    
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
    
    # ---------------------------------------------------------------
    # CRITICAL: Bypass ALL quantization during calibration so that
    # hooks capture the true FP16 activations — not activations
    # corrupted by quantization with default (uncalibrated) scales.
    # ---------------------------------------------------------------
    logger.info("Enabling bypass_all mode for calibration (FP16 forward passes)...")
    for _, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            m.set_bypass_all(True)
    
    calibrator = ADCCalibrator(
        model, 
        method=args.calibration_method,
        bx=args.bx,
        bw=args.bw
    )
    calibrator.calibrate(calibration_loader, num_batches=args.num_calibration_batches)
    
    # Disable bypass — re-enable quantization for inference
    for _, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            m.set_bypass_all(False)
    logger.info("bypass_all disabled — quantization re-enabled.")
    
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
    
    # Run diagnostics on the calibrated model
    diag = diagnose_quantized_model(model, tokenizer, device)
    if diag["has_nan"] or diag["has_inf"]:
        logger.error("ABORTING: Model produces NaN/Inf after calibration!")
        if use_wandb:
            wandb.finish(exit_code=1)
        return
    if diag["n_default_act"] > 0:
        logger.warning(f"{diag['n_default_act']} layers have uncalibrated activation scales!")
    
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
    
    # Get model's max context length for safety check
    model_max_length = getattr(model.config, 'max_position_embeddings', 4096)
    if args.max_length > model_max_length:
        logger.warning(f"max_length ({args.max_length}) > model's max ({model_max_length}), using {model_max_length}")
        args.max_length = model_max_length
    
    # =========================================================================
    # STEP 1.5: Diagnostic — quantized tiling WITHOUT ADC
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1.5: DIAGNOSTIC — Tiling + Quantization WITHOUT ADC")
    logger.info("=" * 80)
    logger.info("Bypassing ADC (floor/clamp) to isolate whether ADC causes the issue...")
    
    # Enable ADC bypass on every TiledLinearADC
    for _, module in model.named_modules():
        if isinstance(module, TiledLinearADC):
            module.set_bypass_adc(True)
    
    model.eval()
    
    # Quick perplexity on first eval dataset
    _diag_ds = args.eval_datasets[0]
    _diag_enc = load_and_tokenize_for_sliding_window(
        _diag_ds, args.eval_split, tokenizer,
        max_samples=args.max_eval_samples if _diag_ds == "c4" else None,
    )
    _diag_metrics = compute_perplexity_sliding_window(
        model, _diag_enc, device,
        max_length=args.max_length, stride=args.stride,
        desc="NoADC eval",
    )
    logger.info(
        f"WITHOUT ADC → {_diag_ds.upper()} Perplexity: {_diag_metrics['perplexity']:.4f}  "
        f"(Loss: {_diag_metrics['avg_loss']:.4f})"
    )
    if use_wandb:
        wandb.log({
            "diagnostic/no_adc_perplexity": _diag_metrics["perplexity"],
            "diagnostic/no_adc_avg_loss": _diag_metrics["avg_loss"],
        })
    
    # Restore ADC
    for _, module in model.named_modules():
        if isinstance(module, TiledLinearADC):
            module.set_bypass_adc(False)
    logger.info("ADC re-enabled for full evaluation.")
    logger.info("=" * 80)
    
    # =========================================================================
    # STEP 2: Evaluate perplexity on specified datasets (Sliding Window)
    # =========================================================================
    logger.info("="*80)
    logger.info("STEP 2: PERPLEXITY EVALUATION (Sliding Window) — WITH ADC")
    logger.info("="*80)
    logger.info(f"Evaluating on datasets: {args.eval_datasets}")
    logger.info(f"Method: Sliding window (standard for papers like GPTQ, AWQ, FlatQuant)")
    logger.info(f"Context window: {args.max_length}, Stride: {args.stride or args.max_length // 2}")
    logger.info(f"Evaluation split: {args.eval_split}")
    
    # Store results for all datasets
    all_eval_metrics = {}
    model.eval()
    
    for eval_dataset_name in args.eval_datasets:
        logger.info("-"*40)
        logger.info(f"Evaluating on {eval_dataset_name.upper()} ({args.eval_split} split)...")
        
        # Load and tokenize as one concatenated sequence (standard for papers)
        encodings = load_and_tokenize_for_sliding_window(
            eval_dataset_name,
            args.eval_split,
            tokenizer,
            max_samples=args.max_eval_samples if eval_dataset_name == "c4" else None
        )
        
        # Compute perplexity using sliding window
        metrics = compute_perplexity_sliding_window(
            model,
            encodings,
            device,
            max_length=args.max_length,
            stride=args.stride,
            desc=f"Eval {eval_dataset_name}"
        )
        
        all_eval_metrics[eval_dataset_name] = metrics
        
        logger.info(f"  {eval_dataset_name.upper()} Perplexity: {metrics['perplexity']:.4f}")
    
    # For backward compatibility, use first dataset's metrics as primary
    eval_metrics = all_eval_metrics[args.eval_datasets[0]]
    perplexity = eval_metrics["perplexity"]
    avg_loss = eval_metrics["avg_loss"]
    
    # Print summary
    logger.info("="*80)
    logger.info("RESULTS SUMMARY (Sliding Window Perplexity)")
    logger.info("="*80)
    logger.info(f"Context window: {args.max_length}, Stride: {args.stride or args.max_length // 2}")
    for ds_name, metrics in all_eval_metrics.items():
        logger.info(f"{ds_name.upper():12s} Perplexity: {metrics['perplexity']:.4f}  (Loss: {metrics['avg_loss']:.4f}, Tokens: {metrics['total_tokens']:,}, Windows: {metrics['num_windows']})")
    
    # Log all metrics to wandb
    if use_wandb:
        for ds_name, metrics in all_eval_metrics.items():
            wandb.log({
                f"eval/{ds_name}/perplexity": metrics["perplexity"],
                f"eval/{ds_name}/avg_loss": metrics["avg_loss"],
                f"eval/{ds_name}/total_tokens": metrics["total_tokens"],
                f"eval/{ds_name}/num_windows": metrics["num_windows"],
            })
            wandb.run.summary[f"{ds_name}_perplexity"] = metrics["perplexity"]
            wandb.run.summary[f"{ds_name}_avg_loss"] = metrics["avg_loss"]
        
        wandb.run.summary["num_calibrated_layers"] = len(optimal_params)
        wandb.run.summary["calibration_dataset"] = args.calibration_dataset
        wandb.run.summary["eval_datasets"] = ",".join(args.eval_datasets)
        wandb.run.summary["eval_max_length"] = args.max_length
        wandb.run.summary["eval_stride"] = args.stride or args.max_length // 2
    
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
        f.write(f"activation_quant: {args.activation_quant}\n")
        if signed_activations is True:
            f.write(f"Quantization strategy: Symmetric (signed) activations for all layers\n")
        elif signed_activations is False:
            f.write(f"Quantization strategy: Asymmetric (unsigned) activations for all layers\n")
        elif args.ashift:
            f.write(f"Quantization strategy: Per-layer (A-shift for SiLU outputs only)\n")
            f.write(f"  - Layers after SiLU (down_proj): Asymmetric + A-shift\n")
            f.write(f"  - All other layers: Symmetric (signed)\n")
        else:
            f.write(f"Quantization strategy: Symmetric (signed) for all activations (default)\n")
        f.write(f"Weights: always symmetric (signed) per-channel\n")
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
                
                # Build codes - ensure scales are on the same device as inputs
                s_x = act_q.scale.to(x.device)
                if act_q.symmetric:
                    code_x = torch.clamp(torch.round(x / s_x), act_q.qmin, act_q.qmax)
                else:
                    # Unsigned path: quantize to [0, 2^bx - 1] using zero_point offset
                    zp_x = act_q.zero_point.to(x.device)
                    code_x_temp = torch.clamp(torch.round(x / s_x + zp_x), 0, act_q.qmax)
                    
                    if hasattr(module, 'ashift') and module.ashift:
                        # A-shift: subtract fixed C instead of learned zp_x
                        code_x = code_x_temp - module.C
                    else:
                        # Standard asymmetric: subtract learned zero_point to center
                        code_x = code_x_temp - zp_x
                
                # Weight codes - ensure scales are on the same device as weights
                s_w_vec = w_q.scale.to(module.weight.device)
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


def _plot_adc_pipeline(data: dict, layer_name: str, title_prefix: str, filepath: str):
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

