import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import os
from datetime import datetime
import json


class LayerwiseStatsLogger:
    """Logger for tracking statistics across layers during training"""
    
    def __init__(self, save_dir: str = "./stats", plot_every_n_steps: int = 100):
        # Structure: {layer_name: {stat_name: [values]}}
        self.stats = defaultdict(lambda: defaultdict(list))
        self.enabled = True
        self.save_dir = save_dir
        self.plot_every_n_steps = plot_every_n_steps
        self.step_count = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Distribution stats tracking
        self.distribution_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
    def log_scalar(self, layer_name: str, stat_name: str, value: float):
        """
        Log a single scalar statistic for a specific layer.
        
        Args:
            layer_name (str): Identifier for the layer/module.
            stat_name (str): Name of the statistic (e.g., "grad_norm").
            value (float): Value of the statistic for this batch.
        """
        if not self.enabled:
            return
            
        self.stats[layer_name][stat_name].append(value)
    
    def log_tensor_distribution(self, layer_name: str, tensor_name: str, tensor: torch.Tensor):
        """
        Log distribution statistics for a tensor (weights, gradients, activations, etc.)
        
        Args:
            layer_name (str): Identifier for the layer/module
            tensor_name (str): Name of the tensor (e.g., "weight", "grad", "activation")
            tensor (torch.Tensor): The tensor to analyze
        """
        if not self.enabled or tensor is None:
            return
            
        # Convert to numpy for analysis
        if tensor.requires_grad:
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = tensor.cpu().numpy()
        
        # Flatten for distribution analysis
        flat_tensor = tensor_np.flatten()
        
        # Calculate distribution statistics
        stats = {
            'mean': float(np.mean(flat_tensor)),
            'std': float(np.std(flat_tensor)),
            'min': float(np.min(flat_tensor)),
            'max': float(np.max(flat_tensor)),
            'median': float(np.median(flat_tensor)),
            'q25': float(np.percentile(flat_tensor, 25)),
            'q75': float(np.percentile(flat_tensor, 75)),
            'norm': float(np.linalg.norm(flat_tensor)),
            'sparsity': float(np.mean(np.abs(flat_tensor) < 1e-6)),  # Fraction of near-zero values
        }
        
        # Store distribution stats
        dist_stats = self.distribution_stats[layer_name][tensor_name]
        for stat_name, value in stats.items():
            dist_stats[stat_name].append(value)
    
    def log_adc_quantizer_stats(self, layer_name: str, adc_quantizer):
        """Log ADC quantizer specific statistics"""
        if not self.enabled or adc_quantizer is None:
            return
            
        # Log delta values
        if hasattr(adc_quantizer, '_delta'):
            self.log_scalar(f"{layer_name}_adc", "analytical_delta", float(adc_quantizer._delta.item()))
        
        if hasattr(adc_quantizer, '_running_absmax'):
            self.log_scalar(f"{layer_name}_adc", "running_absmax", float(adc_quantizer._running_absmax.item()))
        
        if hasattr(adc_quantizer, '_current_epoch'):
            self.log_scalar(f"{layer_name}_adc", "current_epoch", float(adc_quantizer._current_epoch.item()))
    
    def step(self):
        """Call this at the end of each training step"""
        self.step_count += 1
        
        if self.step_count % self.plot_every_n_steps == 0:
            self.plot_distributions()
            self.plot_scalar_trends()
    
    def plot_distributions(self, save_plots: bool = True):
        """Plot distribution statistics for all logged tensors"""
        if not self.distribution_stats:
            return
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create distribution plots for each layer and tensor type
        for layer_name, tensor_dict in self.distribution_stats.items():
            for tensor_name, stats_dict in tensor_dict.items():
                if not stats_dict or not stats_dict['mean']:
                    continue
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'{layer_name} - {tensor_name} Distribution Stats', fontsize=16)
                
                steps = list(range(len(stats_dict['mean'])))
                
                # Plot 1: Mean and Std over time
                ax1 = axes[0, 0]
                ax1.plot(steps, stats_dict['mean'], label='Mean', linewidth=2)
                ax1.fill_between(steps, 
                                np.array(stats_dict['mean']) - np.array(stats_dict['std']),
                                np.array(stats_dict['mean']) + np.array(stats_dict['std']),
                                alpha=0.3, label='±1 Std')
                ax1.set_title('Mean ± Std')
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Value')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Min/Max range
                ax2 = axes[0, 1]
                ax2.plot(steps, stats_dict['min'], label='Min', linewidth=2)
                ax2.plot(steps, stats_dict['max'], label='Max', linewidth=2)
                ax2.fill_between(steps, stats_dict['min'], stats_dict['max'], alpha=0.2)
                ax2.set_title('Min/Max Range')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Value')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Quantiles
                ax3 = axes[1, 0]
                ax3.plot(steps, stats_dict['q25'], label='Q25', linewidth=2)
                ax3.plot(steps, stats_dict['median'], label='Median', linewidth=2)
                ax3.plot(steps, stats_dict['q75'], label='Q75', linewidth=2)
                ax3.fill_between(steps, stats_dict['q25'], stats_dict['q75'], alpha=0.2)
                ax3.set_title('Quantiles')
                ax3.set_xlabel('Step')
                ax3.set_ylabel('Value')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Norm and Sparsity
                ax4 = axes[1, 1]
                ax4_twin = ax4.twinx()
                
                line1 = ax4.plot(steps, stats_dict['norm'], 'b-', label='Norm', linewidth=2)
                line2 = ax4_twin.plot(steps, stats_dict['sparsity'], 'r-', label='Sparsity', linewidth=2)
                
                ax4.set_xlabel('Step')
                ax4.set_ylabel('Norm', color='b')
                ax4_twin.set_ylabel('Sparsity', color='r')
                ax4.set_title('Norm and Sparsity')
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax4.legend(lines, labels, loc='upper left')
                
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_plots:
                    safe_layer_name = layer_name.replace('/', '_').replace('.', '_')
                    safe_tensor_name = tensor_name.replace('/', '_').replace('.', '_')
                    filename = f"{safe_layer_name}_{safe_tensor_name}_dist_step_{self.step_count}.png"
                    plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
    
    def plot_scalar_trends(self, save_plots: bool = True):
        """Plot scalar statistics trends"""
        if not self.stats:
            return
        
        # Group stats by type
        stat_groups = defaultdict(list)
        for layer_name, stat_dict in self.stats.items():
            for stat_name, values in stat_dict.items():
                if values:
                    stat_groups[stat_name].append((layer_name, values))
        
        # Plot each stat type
        for stat_name, layer_data in stat_groups.items():
            if not layer_data:
                continue
                
            plt.figure(figsize=(12, 8))
            
            for layer_name, values in layer_data:
                steps = list(range(len(values)))
                plt.plot(steps, values, label=layer_name, linewidth=2, alpha=0.8)
            
            plt.title(f'{stat_name} Trends Across Layers', fontsize=16)
            plt.xlabel('Step')
            plt.ylabel(stat_name)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            if save_plots:
                safe_stat_name = stat_name.replace('/', '_').replace('.', '_')
                filename = f"{safe_stat_name}_trends_step_{self.step_count}.png"
                plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def plot_adc_delta_analysis(self, save_plots: bool = True):
        """Create specialized plots for ADC delta analysis"""
        adc_layers = {k: v for k, v in self.stats.items() if '_adc' in k}
        
        if not adc_layers:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ADC Delta Analysis', fontsize=16)
        
        # Plot 1: Analytical delta vs running absmax
        ax1 = axes[0, 0]
        for layer_name, stats in adc_layers.items():
            if 'analytical_delta' in stats and 'running_absmax' in stats:
                steps = range(len(stats['analytical_delta']))
                ax1.plot(steps, stats['analytical_delta'], '--', label=f'{layer_name} (analytical)', alpha=0.7)
                if len(stats['running_absmax']) == len(steps):
                    ax1.plot(steps, stats['running_absmax'], '-', label=f'{layer_name} (running)', alpha=0.7)
        
        ax1.set_title('Delta Values Over Time')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Delta Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Delta ratios
        ax2 = axes[0, 1]
        for layer_name, stats in adc_layers.items():
            if 'analytical_delta' in stats and 'running_absmax' in stats:
                if len(stats['analytical_delta']) == len(stats['running_absmax']):
                    ratios = np.array(stats['running_absmax']) / np.array(stats['analytical_delta'])
                    steps = range(len(ratios))
                    ax2.plot(steps, ratios, label=layer_name, alpha=0.7)
        
        ax2.set_title('Running/Analytical Delta Ratio')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Epoch progression
        ax3 = axes[1, 0]
        for layer_name, stats in adc_layers.items():
            if 'current_epoch' in stats:
                steps = range(len(stats['current_epoch']))
                ax3.plot(steps, stats['current_epoch'], label=layer_name, alpha=0.7)
        
        ax3.set_title('Epoch Progression')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Current Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Delta distribution histogram (latest values)
        ax4 = axes[1, 1]
        latest_deltas = []
        for layer_name, stats in adc_layers.items():
            if 'analytical_delta' in stats and stats['analytical_delta']:
                latest_deltas.append(stats['analytical_delta'][-1])
        
        if latest_deltas:
            ax4.hist(latest_deltas, bins=20, alpha=0.7, edgecolor='black')
            ax4.set_title('Latest Analytical Delta Distribution')
            ax4.set_xlabel('Delta Value')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"adc_delta_analysis_step_{self.step_count}.png"
            plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_stats_json(self, filename: Optional[str] = None):
        """Save all statistics to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_stats_{timestamp}.json"
        
        # Convert all data to JSON-serializable format
        json_data = {
            'scalar_stats': dict(self.stats),
            'distribution_stats': dict(self.distribution_stats),
            'step_count': self.step_count,
            'save_dir': self.save_dir
        }
        
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Statistics saved to: {filepath}")
    
    def get_epoch_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Computes the mean of all recorded statistics per layer.

        Returns:
            Dict[str, Dict[str, float]]: Nested dict with mean stats per layer.
        """
        summary = {}
        for layer, stat_dict in self.stats.items():
            summary[layer] = {
                stat: sum(values) / len(values)
                for stat, values in stat_dict.items() if values
            }
        return summary

    def reset(self):
        """Clears all stored statistics (e.g., after each epoch)."""
        self.stats.clear()
        self.distribution_stats.clear()

    def disable(self):
        """Disable logging"""
        self.enabled = False
    
    def enable(self):
        """Enable logging"""
        self.enabled = True

    def __repr__(self):
        return f"LayerwiseStatsLogger(stats={len(self.stats)} layers, step={self.step_count})"


# Usage example and helper functions
def add_stats_hooks(model: torch.nn.Module, logger: LayerwiseStatsLogger):
    """Add hooks to a model to automatically log statistics"""
    
    def make_forward_hook(name):
        def hook(module, input, output):
            # Log output activation statistics
            if isinstance(output, torch.Tensor):
                logger.log_tensor_distribution(name, "output", output)
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                logger.log_tensor_distribution(name, "output", output[0])
        return hook
    
    def make_backward_hook(name):
        def hook(module, grad_input, grad_output):
            # Log gradient statistics
            if grad_output and grad_output[0] is not None:
                logger.log_tensor_distribution(name, "grad_output", grad_output[0])
        return hook
    
    # Add hooks to all modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            module.register_forward_hook(make_forward_hook(name))
            module.register_backward_hook(make_backward_hook(name))
    
    # Add parameter gradient hooks
    for name, param in model.named_parameters():
        if param.requires_grad:
            def make_param_hook(param_name):
                def hook(grad):
                    if grad is not None:
                        logger.log_tensor_distribution(param_name, "param_grad", grad)
                        logger.log_scalar(param_name, "grad_norm", float(grad.norm().item()))
                return hook
            
            param.register_hook(make_param_hook(name))


if __name__ == "__main__":
    # Example usage
    logger = LayerwiseStatsLogger(save_dir="./test_stats")
    
    # Simulate some data
    for step in range(50):
        # Simulate layer statistics
        for layer_idx in range(3):
            layer_name = f"layer_{layer_idx}"
            
            # Simulate weight tensor
            weights = torch.randn(100, 50) * (0.1 + step * 0.01)
            logger.log_tensor_distribution(layer_name, "weight", weights)
            
            # Simulate gradient tensor
            grads = torch.randn(100, 50) * (0.5 - step * 0.008)
            logger.log_tensor_distribution(layer_name, "grad", grads)
            
            # Simulate scalar stats
            logger.log_scalar(layer_name, "loss", float(np.random.rand() * (10 - step * 0.1)))
            logger.log_scalar(layer_name, "lr", float(0.001 * (1 - step * 0.01)))
        
        logger.step()
    
    # Generate final plots manually
    logger.plot_distributions(save_plots=True)
    logger.plot_scalar_trends(save_plots=True)
    logger.plot_adc_delta_analysis(save_plots=True)
    
    # Save final statistics
    logger.save_stats_json()
    
    print("Example complete! Check the ./test_stats directory for plots and data.")
    print(f"Generated plots should be in: {os.path.abspath(logger.save_dir)}")
