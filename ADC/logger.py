from collections import defaultdict
from typing import Dict, Any, List

class LayerwiseStatsLogger:
    def __init__(self):
        # Structure: {layer_name: {stat_name: [values]}}
        self.stats = defaultdict(lambda: defaultdict(list))
        self.enabled = True

    def log_string(self, layer_name: str, stat_name: str, value: float):
        """
        Log a single scalar statistic for a specific layer.
        
        Args:
            layer_name (str): Identifier for the layer/module.
            stat_name (str): Name of the statistic (e.g., "grad_norm").
            value (float): Value of the statistic for this batch.
        """
        self.stats[layer_name][stat_name].append(value)
    
    def log_data(self, layer, tensors, names):
        for i in range(len(tensors)):
            if (names[i] not in self.stats[layer.name]):
                self.stats[layer.name][names[i]] = []
            self.stats[layer.name][names[i]].append(tensors[i].cpu())
        


    def _finalize_stat(self, stat, values):
        if (stat[:3] == "min"):
            return min(values)
        if (stat[:3] == "max"):
            return max(values)
        return sum(values) / len(values)

    def get_stats(self):
        return self.stats

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

    def __repr__(self):
        return f"LayerwiseStatsLogger(stats={dict(self.stats)})"
