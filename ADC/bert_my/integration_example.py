#!/usr/bin/env python3
"""
Example of how to integrate the stats logger with your BERT ADC training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stats_logger import LayerwiseStatsLogger
from plotting_utils import create_summary_dashboard, plot_weight_distributions
import torch


def integrate_stats_with_training():
    """
    Example integration with your BERT training script
    """
    
    # 1. Create the logger
    logger = LayerwiseStatsLogger(
        save_dir="./bert_training_stats", 
        plot_every_n_steps=50  # Generate plots every 50 steps
    )
    
    # 2. Add to your training loop (pseudo-code)
    print("=== Integration Example ===")
    print("Add this to your bert_adc_integration.py:")
    print()
    
    integration_code = '''
# At the top of bert_adc_integration.py, add:
from stats_logger import LayerwiseStatsLogger

# In main(), after model creation:
stats_logger = LayerwiseStatsLogger(
    save_dir=os.path.join(args.output_dir, "training_stats"),
    plot_every_n_steps=100
)

# Add this function to log ADC stats:
def log_model_stats(model, stats_logger, step_loss=None, learning_rate=None):
    """Log statistics for the current training step"""
    
    # Log training metrics
    if step_loss is not None:
        stats_logger.log_scalar("training", "loss", float(step_loss))
    if learning_rate is not None:
        stats_logger.log_scalar("training", "learning_rate", float(learning_rate))
    
    # Log ADC quantizer statistics
    for name, module in model.named_modules():
        if hasattr(module, 'adc_quantizer'):
            stats_logger.log_adc_quantizer_stats(name, module.adc_quantizer)
        
        # Log weight distributions for ADC layers
        if hasattr(module, 'weight') and 'adc' in name.lower():
            stats_logger.log_tensor_distribution(name, "weight", module.weight)
        
        # Log gradient norms if available
        if hasattr(module, 'weight') and module.weight.grad is not None:
            grad_norm = module.weight.grad.norm().item()
            stats_logger.log_scalar(name, "grad_norm", grad_norm)
    
    # Call step to potentially generate plots
    stats_logger.step()

# In your ADCLossTrainer.compute_loss method, add logging:
class ADCLossTrainer(Trainer):
    def __init__(self, *args, stats_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats_logger = stats_logger
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # ... your existing compute_loss code ...
        
        # Log the loss and ADC statistics
        if self.stats_logger and hasattr(self, 'state'):
            log_model_stats(
                model, 
                self.stats_logger, 
                step_loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
                learning_rate=self.get_lr()[0] if hasattr(self, 'get_lr') else None
            )
        
        return (loss, outputs) if return_outputs else loss

# Create trainer with stats logger:
trainer = ADCLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=metrics_computer.compute_metrics,
    callbacks=[EpochCallback()],
    stats_logger=stats_logger  # Add this line
)

# At the end of training, generate final plots:
stats_logger.save_stats_json("final_training_stats.json")
create_summary_dashboard(stats_logger, os.path.join(args.output_dir, "training_dashboard.png"))
plot_weight_distributions(model, os.path.join(args.output_dir, "weight_distributions.png"))
'''
    
    print(integration_code)
    print("\n=== Quick Test Commands ===")
    print("1. Test the logger standalone:")
    print("   python ADC/bert_my/stats_logger.py")
    print()
    print("2. Run your BERT training with logging:")
    print("   python ADC/bert_my/bert_adc_integration.py --adc_resume_dir /path/to/checkpoint ...")
    print()
    print("3. View generated plots:")
    print("   ls -la ./bert_training_stats/")
    print("   # Open .png files to view plots")


def show_available_plots():
    """Show what types of plots are available"""
    print("\n=== Available Plot Types ===")
    
    plot_types = {
        "Distribution Analysis": [
            "Weight distributions over time",
            "Gradient distributions", 
            "Activation statistics",
            "Quantile evolution (min, max, median, Q25, Q75)",
            "Sparsity tracking"
        ],
        "ADC-Specific Plots": [
            "Analytical vs dynamic delta evolution",
            "Delta annealing progression", 
            "Running absmax tracking",
            "Delta ratio analysis",
            "Epoch progression"
        ],
        "Training Monitoring": [
            "Loss curves per layer",
            "Gradient norm trends",
            "Learning rate schedules",
            "Training step progression"
        ],
        "Summary Dashboards": [
            "Multi-panel overview",
            "Gradient norms heatmap",
            "Weight statistics scatter plots",
            "Comprehensive training summary"
        ]
    }
    
    for category, plots in plot_types.items():
        print(f"\n{category}:")
        for plot in plots:
            print(f"  • {plot}")
    
    print(f"\nAll plots are saved as PNG files with timestamps.")
    print(f"JSON files contain raw data for custom analysis.")


if __name__ == "__main__":
    integrate_stats_with_training()
    show_available_plots()
