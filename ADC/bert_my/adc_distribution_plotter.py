import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import os
from datetime import datetime


class ADCDistributionPlotter:
    """
    Specialized plotter for visualizing data distributions before and after ADC quantization
    """
    
    def __init__(self, save_dir: str = "./adc_distributions"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Storage for batch data
        self.batch_data = {}
        self.batch_count = 0
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams.update({
            'figure.figsize': (15, 10),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
        })
    
    def log_batch_distributions(self, layer_name: str, 
                              before_adc: torch.Tensor, 
                              after_adc: torch.Tensor,
                              step: Optional[int] = None):
        """
        Log tensors before and after ADC quantization for a specific layer
        
        Args:
            layer_name: Name of the layer
            before_adc: Tensor before ADC quantization
            after_adc: Tensor after ADC quantization  
            step: Training step (optional)
        """
        if step is None:
            step = self.batch_count
            
        if layer_name not in self.batch_data:
            self.batch_data[layer_name] = {
                'before': [],
                'after': [],
                'steps': []
            }
        
        # Convert to numpy and flatten
        before_np = before_adc.detach().cpu().numpy().flatten()
        after_np = after_adc.detach().cpu().numpy().flatten()
        
        # Debug print to check actual values
        print(f"ADC Debug {layer_name}: Before range=[{before_np.min():.6f}, {before_np.max():.6f}], "
              f"After range=[{after_np.min():.6f}, {after_np.max():.6f}], "
              f"Max diff={np.abs(after_np - before_np[:len(after_np)]).max():.6f}")
        
        self.batch_data[layer_name]['before'].append(before_np)
        self.batch_data[layer_name]['after'].append(after_np)
        self.batch_data[layer_name]['steps'].append(step)
    
    def plot_current_batch_distribution(self, layer_name: str, 
                                      before_adc: torch.Tensor,
                                      after_adc: torch.Tensor,
                                      save_plot: bool = True,
                                      step: Optional[int] = None) -> str:
        """
        Plot distribution comparison for the current batch
        
        Args:
            layer_name: Name of the layer
            before_adc: Tensor before ADC quantization
            after_adc: Tensor after ADC quantization
            save_plot: Whether to save the plot
            step: Training step
            
        Returns:
            Path to saved plot file
        """
        if step is None:
            step = self.batch_count
            
        # Convert to numpy
        before_np = before_adc.detach().cpu().numpy().flatten()
        after_np = after_adc.detach().cpu().numpy().flatten()
        
        # Create the plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'ADC Distribution Analysis - {layer_name} (Step {step})', fontsize=18)
        
        # Plot 1: Histograms comparison
        ax1 = axes[0, 0]
        
        # Use more bins and better range for small differences
        value_range = (min(before_np.min(), after_np.min()), max(before_np.max(), after_np.max()))
        bins = np.linspace(value_range[0], value_range[1], 100)
        
        ax1.hist(before_np, bins=bins, alpha=0.6, label='Before ADC', density=True, color='blue', edgecolor='blue', linewidth=0.5)
        ax1.hist(after_np, bins=bins, alpha=0.6, label='After ADC', density=True, color='red', edgecolor='red', linewidth=0.5)
        ax1.set_title(f'Distribution Comparison\nRange: [{value_range[0]:.4f}, {value_range[1]:.4f}]')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot
        ax2 = axes[0, 1]
        # Sort both arrays for Q-Q plot
        before_sorted = np.sort(before_np)
        after_sorted = np.sort(after_np)
        
        # Ensure same length for comparison
        min_len = min(len(before_sorted), len(after_sorted))
        before_qq = before_sorted[:min_len]
        after_qq = after_sorted[:min_len]
        
        ax2.scatter(before_qq, after_qq, alpha=0.6, s=1)
        
        # Add diagonal line
        min_val = min(before_qq.min(), after_qq.min())
        max_val = max(before_qq.max(), after_qq.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
        
        ax2.set_title('Q-Q Plot (Before vs After)')
        ax2.set_xlabel('Before ADC Quantiles')
        ax2.set_ylabel('After ADC Quantiles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Statistics comparison
        ax3 = axes[0, 2]
        
        stats_before = {
            'Mean': np.mean(before_np),
            'Std': np.std(before_np),
            'Min': np.min(before_np),
            'Max': np.max(before_np),
            'Median': np.median(before_np),
        }
        
        stats_after = {
            'Mean': np.mean(after_np),
            'Std': np.std(after_np),
            'Min': np.min(after_np),
            'Max': np.max(after_np),
            'Median': np.median(after_np),
        }
        
        stat_names = list(stats_before.keys())
        before_vals = list(stats_before.values())
        after_vals = list(stats_after.values())
        
        x = np.arange(len(stat_names))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, before_vals, width, label='Before ADC', alpha=0.7)
        bars2 = ax3.bar(x + width/2, after_vals, width, label='After ADC', alpha=0.7)
        
        ax3.set_title('Statistics Comparison')
        ax3.set_xlabel('Statistic')
        ax3.set_ylabel('Value')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stat_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative distribution
        ax4 = axes[1, 0]
        
        # Calculate CDFs
        before_sorted = np.sort(before_np)
        after_sorted = np.sort(after_np)
        
        before_cdf = np.arange(1, len(before_sorted) + 1) / len(before_sorted)
        after_cdf = np.arange(1, len(after_sorted) + 1) / len(after_sorted)
        
        ax4.plot(before_sorted, before_cdf, label='Before ADC', linewidth=2)
        ax4.plot(after_sorted, after_cdf, label='After ADC', linewidth=2)
        
        ax4.set_title('Cumulative Distribution Function')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Difference histogram
        ax5 = axes[1, 1]
        
        # Calculate difference (after - before) for same positions
        min_len = min(len(before_np), len(after_np))
        diff = after_np[:min_len] - before_np[:min_len]
        
        ax5.hist(diff, bins=50, alpha=0.7, color='green', density=True)
        ax5.axvline(0, color='red', linestyle='--', alpha=0.7, label='No change')
        ax5.set_title('Difference Distribution (After - Before)')
        ax5.set_xlabel('Difference')
        ax5.set_ylabel('Density')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Quantization error analysis
        ax6 = axes[1, 2]
        
        # Calculate quantization error metrics
        mse = np.mean((after_np[:min_len] - before_np[:min_len]) ** 2)
        mae = np.mean(np.abs(after_np[:min_len] - before_np[:min_len]))
        snr = 10 * np.log10(np.var(before_np[:min_len]) / (mse + 1e-10))
        
        error_metrics = ['MSE', 'MAE', 'SNR (dB)']
        error_values = [mse, mae, snr]
        
        bars = ax6.bar(error_metrics, error_values, alpha=0.7, color=['red', 'orange', 'blue'])
        ax6.set_title('Quantization Error Metrics')
        ax6.set_ylabel('Value')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, error_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_layer_name = layer_name.replace('/', '_').replace('.', '_')
            filename = f"adc_distribution_{safe_layer_name}_step_{step}_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"ADC distribution plot saved: {filepath}")
            return filepath
        else:
            plt.show()
            return ""
    
    def plot_evolution_over_batches(self, layer_name: str, max_batches: int = 10):
        """
        Plot how the ADC effect evolves over multiple batches
        
        Args:
            layer_name: Name of the layer to analyze
            max_batches: Maximum number of batches to include
        """
        if layer_name not in self.batch_data:
            print(f"No data found for layer: {layer_name}")
            return
        
        data = self.batch_data[layer_name]
        n_batches = min(len(data['before']), max_batches)
        
        if n_batches == 0:
            print(f"No batch data for layer: {layer_name}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'ADC Effect Evolution - {layer_name} ({n_batches} batches)', fontsize=18)
        
        # Calculate metrics for each batch
        mse_values = []
        snr_values = []
        mean_before = []
        mean_after = []
        std_before = []
        std_after = []
        
        for i in range(n_batches):
            before = data['before'][i]
            after = data['after'][i]
            
            min_len = min(len(before), len(after))
            before_trimmed = before[:min_len]
            after_trimmed = after[:min_len]
            
            mse = np.mean((after_trimmed - before_trimmed) ** 2)
            snr = 10 * np.log10(np.var(before_trimmed) / (mse + 1e-10))
            
            mse_values.append(mse)
            snr_values.append(snr)
            mean_before.append(np.mean(before))
            mean_after.append(np.mean(after))
            std_before.append(np.std(before))
            std_after.append(np.std(after))
        
        steps = data['steps'][:n_batches]
        
        # Plot 1: MSE and SNR evolution
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(steps, mse_values, 'b-o', label='MSE', linewidth=2)
        line2 = ax1_twin.plot(steps, snr_values, 'r-s', label='SNR', linewidth=2)
        
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('MSE', color='b')
        ax1_twin.set_ylabel('SNR (dB)', color='r')
        ax1.set_title('Quantization Error Evolution')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean values evolution
        ax2 = axes[0, 1]
        ax2.plot(steps, mean_before, 'b-o', label='Before ADC', linewidth=2)
        ax2.plot(steps, mean_after, 'r-s', label='After ADC', linewidth=2)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Mean Value')
        ax2.set_title('Mean Values Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Standard deviation evolution
        ax3 = axes[1, 0]
        ax3.plot(steps, std_before, 'b-o', label='Before ADC', linewidth=2)
        ax3.plot(steps, std_after, 'r-s', label='After ADC', linewidth=2)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Standard Deviation')
        ax3.set_title('Std Deviation Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribution overlap visualization
        ax4 = axes[1, 1]
        
        # Show distributions for first and last batch
        if n_batches > 1:
            first_before = data['before'][0]
            first_after = data['after'][0]
            last_before = data['before'][n_batches-1]
            last_after = data['after'][n_batches-1]
            
            ax4.hist(first_before, bins=30, alpha=0.5, label=f'Before (Step {steps[0]})', 
                    density=True, color='lightblue')
            ax4.hist(first_after, bins=30, alpha=0.5, label=f'After (Step {steps[0]})', 
                    density=True, color='lightcoral')
            ax4.hist(last_before, bins=30, alpha=0.5, label=f'Before (Step {steps[-1]})', 
                    density=True, color='blue')
            ax4.hist(last_after, bins=30, alpha=0.5, label=f'After (Step {steps[-1]})', 
                    density=True, color='red')
        else:
            before = data['before'][0]
            after = data['after'][0]
            ax4.hist(before, bins=30, alpha=0.7, label='Before ADC', density=True)
            ax4.hist(after, bins=30, alpha=0.7, label='After ADC', density=True)
        
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the evolution plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_layer_name = layer_name.replace('/', '_').replace('.', '_')
        filename = f"adc_evolution_{safe_layer_name}_{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ADC evolution plot saved: {filepath}")
        
        return filepath
    
    def step(self):
        """Increment batch counter"""
        self.batch_count += 1
    
    def clear_batch_data(self, layer_name: Optional[str] = None):
        """Clear stored batch data"""
        if layer_name:
            if layer_name in self.batch_data:
                del self.batch_data[layer_name]
        else:
            self.batch_data.clear()


# Integration functions for easy use with existing ADC layers
def add_adc_distribution_hooks(model, plotter: ADCDistributionPlotter, 
                              layer_patterns: List[str] = None):
    """
    Add hooks to automatically capture before/after ADC distributions
    
    Args:
        model: PyTorch model with ADC layers
        plotter: ADCDistributionPlotter instance
        layer_patterns: List of layer name patterns to monitor (e.g., ['attention', 'ffn'])
    """
    def should_monitor(name: str) -> bool:
        if layer_patterns is None:
            return True
        return any(pattern in name for pattern in layer_patterns)
    
    for name, module in model.named_modules():
        if hasattr(module, '_adc_quantize_with_loss') and should_monitor(name):
            
            # Patch the _adc_quantize_with_loss method to capture data
            original_method = module._adc_quantize_with_loss
            
            def make_wrapper(layer_name, orig_method):
                def wrapper(y_int):
                    # Capture before ADC
                    before_adc = y_int.clone()
                    
                    # Call original method
                    result, delta_loss = orig_method(y_int)
                    
                    # Capture after ADC and log
                    plotter.log_batch_distributions(layer_name, before_adc, result)
                    
                    return result, delta_loss
                return wrapper
            
            module._adc_quantize_with_loss = make_wrapper(name, original_method)
            print(f"Added ADC distribution monitoring to: {name}")


if __name__ == "__main__":
    # Example usage
    print("ADC Distribution Plotter - Example Usage")
    
    # Create plotter
    plotter = ADCDistributionPlotter(save_dir="./adc_dist_test")
    
    # Simulate some data
    torch.manual_seed(42)
    
    for step in range(5):
        # Simulate matrix multiplication output (before ADC)
        before_adc = torch.randn(32, 768) * 10.0 + step * 2
        
        # Simulate ADC quantization effect
        # Quantize to discrete levels and add some noise
        quantized = torch.round(before_adc / 2.0) * 2.0
        after_adc = quantized + torch.randn_like(quantized) * 0.1
        
        # Log the distributions
        plotter.log_batch_distributions("test_layer", before_adc, after_adc, step)
        
        # Plot current batch distribution
        if step % 2 == 0:  # Plot every 2 steps
            plotter.plot_current_batch_distribution("test_layer", before_adc, after_adc, step=step)
        
        plotter.step()
    
    # Plot evolution over batches
    plotter.plot_evolution_over_batches("test_layer")
    
    print("Example complete! Check ./adc_dist_test/ for generated plots.")
