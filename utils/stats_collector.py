import torch
import wandb
import numpy as np

class StatsCollector:
    def __init__(self, model, num_batches=100):
        self.model = model
        self.num_batches = num_batches
        self.batch_count = 0
        self.activation_stats = {}
        self.weight_stats = {}
        self.hooks = []
        self.setup_hooks()
    
    def setup_hooks(self):
        def get_activation_hook(name):
            def hook(module, input, output):
                if self.batch_count >= self.num_batches:
                    return
                
                if isinstance(output, torch.Tensor):
                    act = output.detach().cpu().numpy()
                    if name not in self.activation_stats:
                        self.activation_stats[name] = {
                            'min': [], 'max': [], 'mean': [], 'std': []
                        }
                    self.activation_stats[name]['min'].append(float(np.min(act)))
                    self.activation_stats[name]['max'].append(float(np.max(act)))
                    self.activation_stats[name]['mean'].append(float(np.mean(act)))
                    self.activation_stats[name]['std'].append(float(np.std(act)))
            return hook

        # Register hooks for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU)):
                self.hooks.append(
                    module.register_forward_hook(get_activation_hook(name))
                )
                
                # Collect weight statistics if the layer has weights
                if hasattr(module, 'weight'):
                    weight = module.weight.detach().cpu().numpy()
                    self.weight_stats[name] = {
                        'min': float(np.min(weight)),
                        'max': float(np.max(weight)),
                        'mean': float(np.mean(weight)),
                        'std': float(np.std(weight))
                    }
    
    def increment_batch(self):
        self.batch_count += 1
    
    def log_stats(self, step, prefix=''):
        # Log weight statistics
        weight_data = {}
        for layer_name, stats in self.weight_stats.items():
            for stat_name, value in stats.items():
                key = f"{prefix}weights/{layer_name}/{stat_name}"
                weight_data[key] = value
        wandb.log(weight_data, step=step)
        
        # Log activation statistics
        if self.batch_count > 0:
            act_data = {}
            for layer_name, stats in self.activation_stats.items():
                for stat_name, values in stats.items():
                    if values:  # Check if we have collected any values
                        key = f"{prefix}activations/{layer_name}/{stat_name}"
                        act_data[key] = np.mean(values)
            wandb.log(act_data, step=step)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = [] 