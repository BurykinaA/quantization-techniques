#!/usr/bin/env python3
"""
Measure perplexity for LLaMA models on WikiText-2 / C4 datasets.

Uses the STANDARD sliding window approach for proper perplexity evaluation,
matching methodology used in papers like GPTQ, AWQ, FlatQuant, etc.

Key features:
- Concatenates all text into one long sequence (no per-sample truncation)
- Uses sliding window with configurable stride
- No padding - pure continuous text evaluation
- Supports both WikiText-2 and C4
- Optional activation/weight distribution visualization (for outlier analysis)

Usage:
    # Basic usage (WikiText-2)
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B"
    
    # With longer context (recommended for accuracy)
    python measure_perplexity.py --model_name "meta-llama/Llama-3.1-8B" --max_length 2048
    
    # Evaluate on C4
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B" --dataset c4
    
    # With visualization (outlier analysis)
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B" --visualize
    
    # With WandB logging
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B" --wandb_project "llama-ppl"
"""

import argparse
import math
import logging
import os
import re
from math import ceil
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Visualization Functions (for outlier analysis)
# =============================================================================

@torch.no_grad()
def visualize_activations_and_weights(
    model,
    input_ids: torch.Tensor,
    layer_idxs: list[int] = None,
    save_path: str = None,
    device: str = "cuda",
    collect_outputs_for: list[str] = None,
):
    """
    Collect and plot activation/weight distributions for specified layers.
    
    This is useful for:
    - Finding outlier channels (like in SmoothQuant, AWQ papers)
    - Understanding activation ranges before quantization
    - Debugging quantization issues
    
    Args:
        model: LLaMA model
        input_ids: Input token IDs [batch, seq_len]
        layer_idxs: Which decoder layers to visualize (default: [0, 1, 5, 10, 15, 20, 25, 30, 31])
        save_path: Directory to save plots
        device: Device to run on
        collect_outputs_for: Layer names to collect outputs (default: ["k", "v"])
    
    Returns:
        dict: Collected statistics
    """
    import matplotlib.pyplot as plt
    
    # Determine number of layers in model
    num_layers = len(model.model.layers)
    
    # Default layer indices (adapt to model size)
    if layer_idxs is None:
        if num_layers <= 16:
            layer_idxs = [0, 1, 2, 4, 8, 12, num_layers - 2, num_layers - 1]
        elif num_layers <= 32:
            layer_idxs = [0, 1, 5, 10, 15, 20, 25, num_layers - 2, num_layers - 1]
        else:
            layer_idxs = [0, 1, 5, 10, 15, 20, 25, 30, 31]
    
    # Filter layer_idxs to valid range
    layer_idxs = [i for i in layer_idxs if i < num_layers]
    
    if collect_outputs_for is None:
        collect_outputs_for = ["k", "v"]
    
    if save_path is None:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = getattr(model.config, '_name_or_path', 'model').split('/')[-1]
        save_path = f"viz_{model_name}_{now}"
    
    logger.info(f"Visualizing layers: {layer_idxs}")
    logger.info(f"Saving to: {save_path}")
    
    # Move inputs to device
    input_ids = input_ids.to(device)
    
    # Storage for collected data
    results = {}
    
    def generate_hook(name, collect_input=True, collect_output=False, collect_weight=True):
        def hook(module, inp, out):
            # Handle different input formats
            if len(inp) > 0:
                x = inp[0]
            else:
                # Some modules receive inputs as kwargs, skip input collection
                x = None
            
            if collect_input and x is not None:
                # Handle case where input might be a tuple
                if isinstance(x, tuple):
                    x = x[0]
                if hasattr(x, 'detach'):
                    results[name + "_input"] = x.detach().cpu()
            
            if collect_output:
                # Handle tuple outputs (like from attention)
                out_tensor = out
                if isinstance(out, tuple):
                    out_tensor = out[0]
                if hasattr(out_tensor, 'detach'):
                    results[name + "_output"] = out_tensor.detach().cpu()
            
            if collect_weight and hasattr(module, 'weight') and module.weight is not None:
                results[name + "_weight"] = module.weight.detach().cpu()
        return hook
    
    # Register hooks
    layers = model.model.layers
    hooks = []
    
    for i in layer_idxs:
        layer = layers[i]
        self_attn = layer.self_attn
        ffn = layer.mlp
        
        # Hook for MHSA input via input_layernorm (input to entire self-attention block)
        # In LLaMA: hidden_states = input_layernorm(hidden_states) -> self_attn(hidden_states)
        if hasattr(layer, 'input_layernorm'):
            hook = layer.input_layernorm.register_forward_hook(
                generate_hook(f"layer{i}_MHSA", collect_input=True, collect_output=False, collect_weight=False)
            )
            hooks.append(hook)
        
        # Hook for FFN input via post_attention_layernorm (input to MLP block)
        # In LLaMA: hidden_states = post_attention_layernorm(hidden_states) -> mlp(hidden_states)
        if hasattr(layer, 'post_attention_layernorm'):
            hook = layer.post_attention_layernorm.register_forward_hook(
                generate_hook(f"layer{i}_FFN", collect_input=True, collect_output=False, collect_weight=False)
            )
            hooks.append(hook)
        
        # Get projection layers
        projections = {
            "q": self_attn.q_proj,
            "k": self_attn.k_proj,
            "v": self_attn.v_proj,
            "o": self_attn.o_proj,
            "gate": ffn.gate_proj,
            "up": ffn.up_proj,
            "down": ffn.down_proj,
        }
        
        for name, module in projections.items():
            hook_name = f"layer{i}_{name}"
            collect_output = name in collect_outputs_for
            hook = module.register_forward_hook(
                generate_hook(hook_name, collect_input=True, collect_output=collect_output, collect_weight=True)
            )
            hooks.append(hook)
    
    # Run forward pass
    logger.info("Running forward pass to collect activations...")
    model.eval()
    with torch.no_grad():
        model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Organize results by layer
    results_by_layer = defaultdict(dict)
    for k, v in results.items():
        match = re.search(r"layer(\d+)", k)
        if match:
            layer_idx = int(match.group(1))
            results_by_layer[layer_idx][k] = v
    
    # Plot distributions
    logger.info("Generating plots...")
    os.makedirs(save_path, exist_ok=True)
    
    for layer_idx in tqdm(sorted(results_by_layer.keys()), desc="Plotting layers"):
        layer_data = results_by_layer[layer_idx]
        
        num_plots = len(layer_data)
        ncols = 1
        nrows = ceil(num_plots / ncols)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 4))
        if nrows == 1:
            axes = [axes]
        else:
            axes = axes.reshape(-1)
        
        for ax, (name, tensor) in zip(axes, layer_data.items()):
            # Prepare data
            if "weight" in name:
                # Weights: [out_features, in_features] -> transpose for per-channel analysis
                value = tensor.T.float()
            else:
                # Activations: [batch, seq, hidden] -> flatten batch and seq
                value = tensor.flatten(0, -2).float()
            
            # Move to GPU for faster quantile computation
            value = value.cuda()
            
            # Compute percentiles along the token/sample dimension (dim=0)
            pmax = torch.amax(value, dim=0).cpu().numpy()
            p9999 = torch.quantile(value, 0.9999, dim=0).cpu().numpy()
            p99 = torch.quantile(value, 0.99, dim=0).cpu().numpy()
            p75 = torch.quantile(value, 0.75, dim=0).cpu().numpy()
            p25 = torch.quantile(value, 0.25, dim=0).cpu().numpy()
            p01 = torch.quantile(value, 0.01, dim=0).cpu().numpy()
            p0001 = torch.quantile(value, 0.0001, dim=0).cpu().numpy()
            pmin = torch.amin(value, dim=0).cpu().numpy()
            
            # Plot
            x_axis = range(len(pmin))
            
            ax.plot(x_axis, pmin, color='blue', label='Min/Max', linewidth=0.3)
            ax.plot(x_axis, pmax, color='blue', linewidth=0.3)
            ax.plot(x_axis, p0001, color='red', label='0.01%/99.99%', linewidth=0.3)
            ax.plot(x_axis, p9999, color='red', linewidth=0.3)
            ax.plot(x_axis, p01, color='purple', label='1%/99%', linewidth=0.3)
            ax.plot(x_axis, p99, color='purple', linewidth=0.3)
            ax.plot(x_axis, p25, color='orange', label='25%/75%', linewidth=0.3)
            ax.plot(x_axis, p75, color='orange', linewidth=0.3)
            
            ax.set_title(name)
            ax.set_xlabel("Hidden dimension index")
            ax.set_ylabel("Value")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for ax in axes[len(layer_data):]:
            ax.set_visible(False)
        
        fig.suptitle(f"Layer {layer_idx} - Activation/Weight Distributions", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        fig_path = os.path.join(save_path, f"layer_{layer_idx}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    logger.info(f"Saved {len(results_by_layer)} layer plots to {save_path}")
    
    # =========================================================================
    # Generate Figure 4 style plots (2D KDE + per-channel variance bar charts)
    # =========================================================================
    logger.info("Generating Figure 4 style plots (2D KDE + variance bar charts)...")
    
    try:
        import seaborn as sns
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        logger.warning("seaborn not available, skipping Figure 4 plots")
        return {
            "save_path": save_path,
            "num_layers_visualized": len(results_by_layer),
            "layer_indices": list(results_by_layer.keys()),
        }
    
    # Collect per-channel statistics for 2D distribution (Mean vs Std like FlatQuant Figure 4)
    # For weights: each point = (mean, std) of one output channel
    # For activations: each point = (mean, std) of one hidden dimension
    all_weight_channel_means = []
    all_weight_channel_stds = []
    all_activation_channel_means = []
    all_activation_channel_stds = []
    
    # Collect per-channel variances for bar chart (per layer separately)
    weight_vars_by_layer = {}  # layer_idx -> list of variances
    activation_vars_by_layer = {}
    
    for layer_idx, layer_data in results_by_layer.items():
        if layer_idx not in weight_vars_by_layer:
            weight_vars_by_layer[layer_idx] = []
            activation_vars_by_layer[layer_idx] = []
        
        for name, tensor in layer_data.items():
            if "weight" in name:
                # Weights: [out_features, in_features]
                value = tensor.float().numpy()
                
                # Per-channel stats (along input features for each output channel)
                channel_means = np.mean(value, axis=1)
                channel_stds = np.std(value, axis=1)
                channel_vars = np.var(value, axis=1)
                
                all_weight_channel_means.extend(channel_means)
                all_weight_channel_stds.extend(channel_stds)
                weight_vars_by_layer[layer_idx].extend(channel_vars)
                
            elif "input" in name:
                # Activations: [batch, seq, hidden] -> flatten to [tokens, hidden]
                value = tensor.flatten(0, -2).float().numpy()
                
                # Per-channel stats (along tokens for each hidden dim)
                channel_means = np.mean(value, axis=0)
                channel_stds = np.std(value, axis=0)
                channel_vars = np.var(value, axis=0)
                
                all_activation_channel_means.extend(channel_means)
                all_activation_channel_stds.extend(channel_stds)
                activation_vars_by_layer[layer_idx].extend(channel_vars)
    
    all_weight_channel_means = np.array(all_weight_channel_means)
    all_weight_channel_stds = np.array(all_weight_channel_stds)
    all_activation_channel_means = np.array(all_activation_channel_means)
    all_activation_channel_stds = np.array(all_activation_channel_stds)
    
    if len(all_weight_channel_means) > 0 and len(all_activation_channel_means) > 0:
        
        # =====================================================================
        # Figure 4(a): 2D KDE Distribution (Mean vs Std per channel)
        # Like FlatQuant Figure 4: each point = one channel's (mean, std)
        # =====================================================================
        def plot_2d_kde_distribution(ax, means, stds, cmap_name, title, mean_color='red'):
            """Plot 2D KDE of per-channel statistics (Mean vs Std)"""
            # Subsample if too many points
            max_points = 5000
            if len(means) > max_points:
                idx = np.random.choice(len(means), max_points, replace=False)
                means = means[idx]
                stds = stds[idx]
            
            # 2D KDE Plot: X = channel mean, Y = channel std
            try:
                sns.kdeplot(x=means, y=stds, ax=ax, 
                           fill=True, cmap=cmap_name, thresh=0.05, levels=15)
            except Exception:
                # Fallback to scatter if KDE fails
                ax.scatter(means, stds, alpha=0.3, s=5, c='blue')
            
            # Mean of all channels (red dot)
            global_mean = np.mean(means)
            global_std_mean = np.mean(stds)
            ax.scatter([global_mean], [global_std_mean], color=mean_color, s=80, 
                      edgecolors='black', linewidths=2, zorder=10, label='Global Mean')
            
            # Formatting
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Channel Mean', fontsize=10)
            ax.set_ylabel('Channel Std', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize=8)
            
            # Inset 1D distribution of means in bottom-left corner
            try:
                inset_ax = ax.inset_axes([0.05, 0.65, 0.35, 0.25])
                sns.kdeplot(means, ax=inset_ax, color=plt.get_cmap(cmap_name)(0.6), fill=True)
                inset_ax.axvline(global_mean, color=mean_color, linestyle='--', linewidth=1.5)
                inset_ax.set_xlabel('')
                inset_ax.set_ylabel('')
                inset_ax.tick_params(labelsize=6)
                inset_ax.set_title('Mean Distribution', fontsize=7)
            except Exception:
                pass
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        plot_2d_kde_distribution(ax1, all_weight_channel_means, all_weight_channel_stds, 
                                 'Blues', 'Distribution of Weight (per channel)')
        plot_2d_kde_distribution(ax2, all_activation_channel_means, all_activation_channel_stds, 
                                 'Greens', 'Distribution of Activation (per channel)')
        
        fig.suptitle('(a) Per-Channel Statistics: Mean vs Std', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(save_path, 'fig4a_2d_kde_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # =====================================================================
        # Figure 4(b): Per-channel Variance Bar Charts (per layer)
        # Show a few representative layers instead of concatenating all
        # =====================================================================
        def plot_variance_bar_chart(ax, channel_vars, title, cmap_name='coolwarm'):
            """Plot per-channel variance with color gradient based on variance value"""
            n_channels = len(channel_vars)
            channels = np.arange(n_channels)
            
            # Create color gradient based on variance VALUE (not index)
            # This highlights high-variance channels
            var_min, var_max = np.min(channel_vars), np.max(channel_vars)
            norm = Normalize(vmin=var_min, vmax=var_max)
            cmap = plt.get_cmap(cmap_name)
            colors = [cmap(norm(v)) for v in channel_vars]
            
            # Plot bars
            ax.bar(channels, channel_vars, color=colors, width=1.0, edgecolor='none')
            
            # Formatting
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Channel Index', fontsize=10)
            ax.set_ylabel('Variance', fontsize=10)
            ax.set_xlim(0, n_channels)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            
            # Add colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Variance', fontsize=8)
        
        # Select representative layers (first, middle, last from visualized set)
        sorted_layers = sorted(weight_vars_by_layer.keys())
        if len(sorted_layers) >= 3:
            representative_layers = [sorted_layers[0], sorted_layers[len(sorted_layers)//2], sorted_layers[-1]]
        else:
            representative_layers = sorted_layers
        
        # Create multi-panel figure: 2 rows (weights, activations) x N layers
        n_layers_to_show = len(representative_layers)
        fig, axes = plt.subplots(2, n_layers_to_show, figsize=(5 * n_layers_to_show, 8))
        
        if n_layers_to_show == 1:
            axes = axes.reshape(2, 1)
        
        for col_idx, layer_idx in enumerate(representative_layers):
            # Weight variance for this layer
            w_vars = np.array(weight_vars_by_layer.get(layer_idx, []))
            if len(w_vars) > 0:
                # Take first 4096 channels (one projection layer)
                w_vars_plot = w_vars[:min(4096, len(w_vars))]
                plot_variance_bar_chart(axes[0, col_idx], w_vars_plot, 
                                       f'Weight Variance - Layer {layer_idx}')
            
            # Activation variance for this layer
            a_vars = np.array(activation_vars_by_layer.get(layer_idx, []))
            if len(a_vars) > 0:
                a_vars_plot = a_vars[:min(4096, len(a_vars))]
                plot_variance_bar_chart(axes[1, col_idx], a_vars_plot, 
                                       f'Activation Variance - Layer {layer_idx}')
        
        fig.suptitle('(b) Inter-channel Variance Disparities (by Layer)', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(save_path, 'fig4b_variance_bar_charts.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved Figure 4 style plots to {save_path}")
    
    return {
        "save_path": save_path,
        "num_layers_visualized": len(results_by_layer),
        "layer_indices": list(results_by_layer.keys()),
    }


# =============================================================================
# Perplexity Computation
# =============================================================================

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
    
    logger.info(f"Total tokens in corpus: {seq_len:,}")
    logger.info(f"Context window: {max_length}, Stride: {stride}")
    
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


# =============================================================================
# Dataset Loading
# =============================================================================

def load_and_tokenize_dataset(dataset_name: str, split: str, tokenizer, max_samples: int = None):
    """
    Load and tokenize a dataset, concatenating all text.
    
    Args:
        dataset_name: "wikitext2" or "c4"
        split: "train", "validation", or "test"
        tokenizer: Tokenizer to use
        max_samples: Maximum samples to use (for C4 which is huge)
    
    Returns:
        dict with 'input_ids' tensor of shape [1, total_tokens]
    """
    logger.info(f"Loading {dataset_name} ({split} split)...")
    
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure LLaMA perplexity using standard sliding window approach"
    )
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                       help="HuggingFace model name or path")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Model dtype")
    parser.add_argument("--seed", type=int, default=42)
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="wikitext2",
                       choices=["wikitext2", "c4"],
                       help="Dataset to evaluate on")
    parser.add_argument("--dataset_split", type=str, default="test",
                       choices=["train", "validation", "test"],
                       help="Which split to evaluate on")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Max samples for C4 (ignored for WikiText-2)")
    
    # Evaluation settings
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Context window size (should match model's context length)")
    parser.add_argument("--stride", type=int, default=None,
                       help="Stride for sliding window (default: max_length // 2)")
    
    # Visualization settings
    parser.add_argument("--visualize", action="store_true",
                       help="Enable activation/weight distribution visualization")
    parser.add_argument("--viz_save_path", type=str, default=None,
                       help="Directory to save visualization plots (auto-generated if not provided)")
    parser.add_argument("--viz_layers", type=int, nargs="+", default=None,
                       help="Layer indices to visualize (default: auto-select based on model size)")
    parser.add_argument("--viz_num_samples", type=int, default=10,
                       help="Number of samples to use for visualization")
    parser.add_argument("--viz_seq_length", type=int, default=2048,
                       help="Sequence length for visualization samples")
    
    # WandB settings
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="WandB project name (None = disable WandB)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                       help="Tags for WandB run")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        model_short_name = args.model_name.split("/")[-1]
        run_name = args.wandb_run_name or f"ppl_{model_short_name}_{args.dataset}_{args.torch_dtype}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=args.tags,
            config={
                "model_name": args.model_name,
                "torch_dtype": args.torch_dtype,
                "dataset": args.dataset,
                "dataset_split": args.dataset_split,
                "max_length": args.max_length,
                "stride": args.stride,
                "visualize": args.visualize,
                "seed": args.seed,
            }
        )
        logger.info(f"WandB initialized: project={args.wandb_project}, run={run_name}")
    
    # =========================================================================
    # Load Model
    # =========================================================================
    logger.info("="*80)
    logger.info(f"Loading model: {args.model_name}")
    logger.info("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Get the actual device (for multi-GPU, get the first one)
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'hf_device_map'):
        devices = list(model.hf_device_map.values())
        device = devices[0] if devices else device
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters, dtype={torch_dtype}")
    
    # Get model's max context length
    model_max_length = getattr(model.config, 'max_position_embeddings', 4096)
    if args.max_length > model_max_length:
        logger.warning(f"max_length ({args.max_length}) > model's max ({model_max_length}), using {model_max_length}")
        args.max_length = model_max_length
    
    # =========================================================================
    # Load and Tokenize Dataset
    # =========================================================================
    logger.info("="*80)
    logger.info(f"Loading {args.dataset.upper()} dataset")
    logger.info("="*80)
    
    encodings = load_and_tokenize_dataset(
        args.dataset,
        args.dataset_split,
        tokenizer,
        max_samples=args.max_samples if args.dataset == "c4" else None
    )
    
    # =========================================================================
    # Visualization (if enabled)
    # =========================================================================
    viz_info = None
    if args.visualize:
        logger.info("="*80)
        logger.info("VISUALIZATION: Activation/Weight Distributions")
        logger.info("="*80)
        
        # Prepare input samples for visualization
        input_ids = encodings["input_ids"]
        total_tokens = input_ids.size(1)
        
        # Extract samples for visualization
        num_samples = min(args.viz_num_samples, total_tokens // args.viz_seq_length)
        if num_samples < 1:
            num_samples = 1
            viz_seq_len = total_tokens
        else:
            viz_seq_len = args.viz_seq_length
        
        # Take evenly spaced samples
        viz_input_ids = input_ids[0, :num_samples * viz_seq_len].reshape(num_samples, viz_seq_len)
        
        logger.info(f"Using {num_samples} samples of length {viz_seq_len} for visualization")
        
        viz_info = visualize_activations_and_weights(
            model,
            viz_input_ids,
            layer_idxs=args.viz_layers,
            save_path=args.viz_save_path,
            device=device,
        )
        
        # Upload to WandB if enabled
        if use_wandb and viz_info:
            import glob
            viz_path = viz_info["save_path"]
            png_files = sorted(glob.glob(os.path.join(viz_path, "*.png")))
            for png_file in png_files:
                layer_name = os.path.basename(png_file).replace(".png", "")
                wandb.log({f"viz/{layer_name}": wandb.Image(png_file)})
            logger.info(f"Uploaded {len(png_files)} visualization plots to WandB")
    
    # =========================================================================
    # Compute Perplexity
    # =========================================================================
    logger.info("="*80)
    logger.info("Computing Perplexity (Sliding Window)")
    logger.info("="*80)
    
    metrics = compute_perplexity_sliding_window(
        model, 
        encodings, 
        device,
        max_length=args.max_length,
        stride=args.stride,
        desc=f"Perplexity ({args.model_name.split('/')[-1]})"
    )
    
    # =========================================================================
    # Results
    # =========================================================================
    logger.info("="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Model:           {args.model_name}")
    logger.info(f"Dtype:           {args.torch_dtype}")
    logger.info(f"Dataset:         {args.dataset.upper()} ({args.dataset_split})")
    logger.info(f"Context window:  {metrics['max_length']}")
    logger.info(f"Stride:          {metrics['stride']}")
    logger.info(f"")
    logger.info(f"Perplexity:      {metrics['perplexity']:.4f}")
    logger.info(f"Avg Loss:        {metrics['avg_loss']:.4f}")
    logger.info(f"Total Tokens:    {metrics['total_tokens']:,}")
    logger.info(f"Num Windows:     {metrics['num_windows']}")
    if viz_info:
        logger.info(f"")
        logger.info(f"Visualizations:  {viz_info['save_path']}")
    logger.info("="*80)
    
    # Log to WandB
    if use_wandb:
        wandb.log({
            "perplexity": metrics['perplexity'],
            "avg_loss": metrics['avg_loss'],
            "total_tokens": metrics['total_tokens'],
            "num_windows": metrics['num_windows'],
        })
        
        # Summary
        wandb.run.summary["perplexity"] = metrics['perplexity']
        wandb.run.summary["avg_loss"] = metrics['avg_loss']
        wandb.run.summary["total_params"] = total_params
        wandb.run.summary["max_length"] = metrics['max_length']
        wandb.run.summary["stride"] = metrics['stride']
        if viz_info:
            wandb.run.summary["viz_path"] = viz_info['save_path']
            wandb.run.summary["viz_num_layers"] = viz_info['num_layers_visualized']
        
        wandb.finish()
        logger.info("Results logged to WandB")
    
    # Return metrics for programmatic use
    return metrics


if __name__ == "__main__":
    main()
