import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import os
from collections import defaultdict


def setup_plotting_style():
    """Set up a consistent plotting style"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })


def plot_weight_distributions(model: torch.nn.Module, save_path: Optional[str] = None, 
                            layer_types: Optional[List[str]] = None):
    """
    Plot weight distributions for different layer types in the model
    
    Args:
        model: PyTorch model
        save_path: Path to save the plot
        layer_types: List of layer type names to include (e.g., ['Linear', 'Conv2d'])
    """
    setup_plotting_style()
    
    # Collect weights by layer type
    weights_by_type = defaultdict(list)
    layer_names = defaultdict(list)
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if layer_types and module_type not in layer_types:
            continue
            
        if hasattr(module, 'weight') and module.weight is not None:
            weights = module.weight.detach().cpu().numpy().flatten()
            weights_by_type[module_type].append(weights)
            layer_names[module_type].append(name)
    
    if not weights_by_type:
        print("No weights found to plot")
        return
    
    # Create subplots
    n_types = len(weights_by_type)
    fig, axes = plt.subplots(2, (n_types + 1) // 2, figsize=(15, 10))
    if n_types == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle('Weight Distributions by Layer Type', fontsize=18)
    
    for idx, (layer_type, weight_lists) in enumerate(weights_by_type.items()):
        ax = axes[idx]
        
        # Combine all weights for this layer type
        all_weights = np.concatenate(weight_lists)
        
        # Plot histogram
        ax.hist(all_weights, bins=50, alpha=0.7, density=True, edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(all_weights)
        std_val = np.std(all_weights)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_val:.4f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
        
        ax.set_title(f'{layer_type} Weights ({len(weight_lists)} layers)')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Weight distribution plot saved to: {save_path}")
    else:
        plt.show()


def plot_gradient_norms(grad_stats: Dict[str, List[float]], save_path: Optional[str] = None,
                       top_k: int = 20):
    """
    Plot gradient norms for different layers
    
    Args:
        grad_stats: Dictionary mapping layer names to lists of gradient norms
        save_path: Path to save the plot
        top_k: Number of layers with highest average gradient norms to show
    """
    setup_plotting_style()
    
    # Calculate average gradient norms
    avg_norms = {}
    for layer_name, norms in grad_stats.items():
        if norms:
            avg_norms[layer_name] = np.mean(norms)
    
    # Sort by average norm and take top_k
    sorted_layers = sorted(avg_norms.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Bar chart of average gradient norms
    layer_names = [item[0] for item in sorted_layers]
    avg_values = [item[1] for item in sorted_layers]
    
    bars = ax1.bar(range(len(layer_names)), avg_values, alpha=0.7)
    ax1.set_title(f'Top {top_k} Layers by Average Gradient Norm')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average Gradient Norm')
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Color bars by magnitude
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Plot 2: Time series of gradient norms for top layers
    ax2.set_title('Gradient Norm Evolution (Top 5 Layers)')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Gradient Norm')
    
    for i, (layer_name, _) in enumerate(sorted_layers[:5]):
        if layer_name in grad_stats and grad_stats[layer_name]:
            steps = range(len(grad_stats[layer_name]))
            ax2.plot(steps, grad_stats[layer_name], label=layer_name, linewidth=2, alpha=0.8)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gradient norms plot saved to: {save_path}")
    else:
        plt.show()


def plot_adc_delta_evolution(adc_stats: Dict[str, Dict[str, List[float]]], 
                           save_path: Optional[str] = None):
    """
    Plot ADC delta evolution over training
    
    Args:
        adc_stats: Dictionary with ADC statistics per layer
        save_path: Path to save the plot
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ADC Delta Analysis', fontsize=18)
    
    # Plot 1: Analytical vs Dynamic Delta
    ax1 = axes[0, 0]
    for layer_name, stats in adc_stats.items():
        if 'analytical_delta' in stats and 'running_absmax' in stats:
            steps = range(len(stats['analytical_delta']))
            ax1.plot(steps, stats['analytical_delta'], '--', 
                    label=f'{layer_name} (analytical)', alpha=0.7, linewidth=2)
            if len(stats['running_absmax']) == len(steps):
                ax1.plot(steps, stats['running_absmax'], '-', 
                        label=f'{layer_name} (dynamic)', alpha=0.7, linewidth=2)
    
    ax1.set_title('Delta Values Over Time')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Delta Value (log scale)')
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Delta Ratios
    ax2 = axes[0, 1]
    for layer_name, stats in adc_stats.items():
        if 'analytical_delta' in stats and 'running_absmax' in stats:
            if len(stats['analytical_delta']) == len(stats['running_absmax']):
                analytical = np.array(stats['analytical_delta'])
                dynamic = np.array(stats['running_absmax'])
                ratios = dynamic / (analytical + 1e-8)  # Avoid division by zero
                steps = range(len(ratios))
                ax2.plot(steps, ratios, label=layer_name, alpha=0.7, linewidth=2)
    
    ax2.set_title('Dynamic/Analytical Delta Ratio')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Ratio')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Delta Distribution (latest values)
    ax3 = axes[1, 0]
    analytical_deltas = []
    dynamic_deltas = []
    
    for layer_name, stats in adc_stats.items():
        if 'analytical_delta' in stats and stats['analytical_delta']:
            analytical_deltas.append(stats['analytical_delta'][-1])
        if 'running_absmax' in stats and stats['running_absmax']:
            dynamic_deltas.append(stats['running_absmax'][-1])
    
    if analytical_deltas:
        ax3.hist(analytical_deltas, bins=15, alpha=0.7, label='Analytical', density=True)
    if dynamic_deltas:
        ax3.hist(dynamic_deltas, bins=15, alpha=0.7, label='Dynamic', density=True)
    
    ax3.set_title('Latest Delta Value Distribution')
    ax3.set_xlabel('Delta Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Epoch Progression
    ax4 = axes[1, 1]
    for layer_name, stats in adc_stats.items():
        if 'current_epoch' in stats and stats['current_epoch']:
            steps = range(len(stats['current_epoch']))
            ax4.plot(steps, stats['current_epoch'], label=layer_name, alpha=0.7, linewidth=2)
    
    ax4.set_title('Training Epoch Progression')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Current Epoch')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ADC delta analysis plot saved to: {save_path}")
    else:
        plt.show()


def plot_quantization_impact(original_stats: Dict, quantized_stats: Dict, 
                           metric_names: List[str], save_path: Optional[str] = None):
    """
    Compare statistics before and after quantization
    
    Args:
        original_stats: Statistics from original model
        quantized_stats: Statistics from quantized model  
        metric_names: List of metric names to compare
        save_path: Path to save the plot
    """
    setup_plotting_style()
    
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    fig.suptitle('Quantization Impact Analysis', fontsize=18)
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        
        # Collect data for this metric
        original_values = []
        quantized_values = []
        layer_names = []
        
        for layer_name in original_stats.keys():
            if layer_name in quantized_stats and metric in original_stats[layer_name] and metric in quantized_stats[layer_name]:
                original_values.append(original_stats[layer_name][metric])
                quantized_values.append(quantized_stats[layer_name][metric])
                layer_names.append(layer_name)
        
        if not original_values:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Scatter plot
        ax.scatter(original_values, quantized_values, alpha=0.7, s=50)
        
        # Add diagonal line (y=x)
        min_val = min(min(original_values), min(quantized_values))
        max_val = max(max(original_values), max(quantized_values))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        
        ax.set_xlabel(f'Original {metric}')
        ax.set_ylabel(f'Quantized {metric}')
        ax.set_title(f'{metric} Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(original_values, quantized_values)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Quantization impact plot saved to: {save_path}")
    else:
        plt.show()


def create_summary_dashboard(stats_logger, save_path: Optional[str] = None):
    """
    Create a comprehensive dashboard with multiple subplots
    
    Args:
        stats_logger: LayerwiseStatsLogger instance
        save_path: Path to save the dashboard
    """
    setup_plotting_style()
    
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Training Statistics Dashboard', fontsize=20)
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Gradient norms heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    grad_data = []
    layer_names = []
    
    for layer_name, stats in stats_logger.stats.items():
        if 'grad_norm' in stats and stats['grad_norm']:
            grad_data.append(stats['grad_norm'])
            layer_names.append(layer_name)
    
    if grad_data:
        # Normalize lengths and create heatmap
        max_len = max(len(data) for data in grad_data)
        grad_matrix = np.full((len(grad_data), max_len), np.nan)
        
        for i, data in enumerate(grad_data):
            grad_matrix[i, :len(data)] = data
        
        im = ax1.imshow(grad_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_title('Gradient Norms Heatmap')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Layer')
        ax1.set_yticks(range(len(layer_names)))
        ax1.set_yticklabels([name.split('.')[-1] for name in layer_names])  # Show only last part
        plt.colorbar(im, ax=ax1)
    
    # 2. Loss evolution
    ax2 = fig.add_subplot(gs[0, 2:])
    for layer_name, stats in stats_logger.stats.items():
        if 'loss' in stats and stats['loss']:
            steps = range(len(stats['loss']))
            ax2.plot(steps, stats['loss'], label=layer_name, alpha=0.7)
    
    ax2.set_title('Loss Evolution')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Weight statistics
    ax3 = fig.add_subplot(gs[1, 0])
    weight_means = []
    weight_stds = []
    
    for layer_name, tensor_stats in stats_logger.distribution_stats.items():
        if 'weight' in tensor_stats and 'mean' in tensor_stats['weight']:
            weight_means.extend(tensor_stats['weight']['mean'])
            weight_stds.extend(tensor_stats['weight']['std'])
    
    if weight_means and weight_stds:
        ax3.scatter(weight_means, weight_stds, alpha=0.6)
        ax3.set_xlabel('Weight Mean')
        ax3.set_ylabel('Weight Std')
        ax3.set_title('Weight Statistics')
        ax3.grid(True, alpha=0.3)
    
    # 4. ADC delta statistics
    ax4 = fig.add_subplot(gs[1, 1])
    adc_layers = {k: v for k, v in stats_logger.stats.items() if '_adc' in k}
    
    if adc_layers:
        deltas = []
        for layer_name, stats in adc_layers.items():
            if 'analytical_delta' in stats and stats['analytical_delta']:
                deltas.extend(stats['analytical_delta'])
        
        if deltas:
            ax4.hist(deltas, bins=20, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Delta Value')
            ax4.set_ylabel('Frequency')
            ax4.set_title('ADC Delta Distribution')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
    
    # 5. Sparsity evolution
    ax5 = fig.add_subplot(gs[1, 2:])
    for layer_name, tensor_stats in stats_logger.distribution_stats.items():
        if 'weight' in tensor_stats and 'sparsity' in tensor_stats['weight']:
            steps = range(len(tensor_stats['weight']['sparsity']))
            ax5.plot(steps, tensor_stats['weight']['sparsity'], label=layer_name, alpha=0.7)
    
    ax5.set_title('Weight Sparsity Evolution')
    ax5.set_xlabel('Training Step')
    ax5.set_ylabel('Sparsity')
    ax5.grid(True, alpha=0.3)
    
    # 6. Gradient distribution
    ax6 = fig.add_subplot(gs[2, :2])
    all_grad_norms = []
    
    for layer_name, stats in stats_logger.stats.items():
        if 'grad_norm' in stats and stats['grad_norm']:
            all_grad_norms.extend(stats['grad_norm'])
    
    if all_grad_norms:
        ax6.hist(all_grad_norms, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax6.set_xlabel('Gradient Norm')
        ax6.set_ylabel('Density')
        ax6.set_title('Gradient Norm Distribution')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
    
    # 7. Training progress summary
    ax7 = fig.add_subplot(gs[2, 2:])
    
    # Create a text summary
    summary_text = f"Training Statistics Summary\n"
    summary_text += f"Total Steps: {stats_logger.step_count}\n"
    summary_text += f"Tracked Layers: {len(stats_logger.stats)}\n"
    summary_text += f"Distribution Stats: {len(stats_logger.distribution_stats)}\n"
    
    if all_grad_norms:
        summary_text += f"Avg Gradient Norm: {np.mean(all_grad_norms):.4f}\n"
        summary_text += f"Max Gradient Norm: {np.max(all_grad_norms):.4f}\n"
    
    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.axis('off')
    ax7.set_title('Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Plotting utilities loaded. Use the functions to create various plots for your training statistics.")
