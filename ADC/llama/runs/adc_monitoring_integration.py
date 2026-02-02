"""
Easy integration for monitoring ADC distributions in your BERT training
"""

import torch
from ADC.llama.visualisation.adc_distribution_plotter import ADCDistributionPlotter


def add_adc_monitoring_to_layer(layer, layer_name: str, plotter: ADCDistributionPlotter, monitor_full_pipeline: bool = True):
    """
    Add monitoring to a single QATLinearADC layer
    
    Args:
        layer: QATLinearADC instance
        layer_name: Name for saving plots
        plotter: ADCDistributionPlotter instance
        monitor_full_pipeline: If True, monitor the complete pipeline, else just ADC
    """
    if not hasattr(layer, '_adc_quantize_with_loss'):
        print(f"Warning: Layer {layer_name} doesn't have _adc_quantize_with_loss method")
        return
    
    # Store layer name for internal use
    layer._layer_name = layer_name
    
    if monitor_full_pipeline:
        # Set up full pipeline monitoring
        def pipeline_monitor(layer_name, x_raw, x_quantized, w_raw, w_quantized, before_adc, after_adc):
            plotter.log_full_quantization_pipeline(
                layer_name, x_raw, x_quantized, w_raw, w_quantized, before_adc, after_adc
            )
        
        layer._pipeline_monitor = pipeline_monitor
        print(f"Added full pipeline monitoring to layer: {layer_name}")
    else:
        # Original ADC-only monitoring
        original_method = layer._adc_quantize_with_loss
        
        def monitored_adc_quantize(y_int):
            """Wrapper that captures before/after ADC data"""
            # Capture input (before ADC)
            before_adc = y_int.clone().detach()
            
            # Call original ADC quantization
            adc_output, delta_loss = original_method(y_int)
            
            # Capture output (after ADC)
            after_adc = adc_output.clone().detach()
            
            # Log to plotter
            plotter.log_batch_distributions(layer_name, before_adc, after_adc)
            
            return adc_output, delta_loss
        
        # Replace the method
        layer._adc_quantize_with_loss = monitored_adc_quantize
        print(f"Added ADC-only monitoring to layer: {layer_name}")


def add_adc_monitoring_to_model(model, plotter: ADCDistributionPlotter, 
                               layer_patterns: list = None,
                               max_layers: int = 5,
                               monitor_full_pipeline: bool = True):
    """
    Add ADC monitoring to multiple layers in the model
    
    Args:
        model: Your BERT model with ADC layers
        plotter: ADCDistributionPlotter instance
        layer_patterns: List of patterns to match layer names (e.g., ['attention', 'dense'])
        max_layers: Maximum number of layers to monitor (to avoid too many plots)
        monitor_full_pipeline: If True, monitor complete pipeline, else just ADC
    """
    monitored_count = 0
    
    for name, module in model.named_modules():
        # Check if this is an ADC layer
        if hasattr(module, '_adc_quantize_with_loss'):
            
            # Check if we should monitor this layer
            should_monitor = True
            if layer_patterns:
                should_monitor = any(pattern in name for pattern in layer_patterns)
            
            if should_monitor and monitored_count < max_layers:
                add_adc_monitoring_to_layer(module, name, plotter, monitor_full_pipeline)
                monitored_count += 1
    
    print(f"Added {'full pipeline' if monitor_full_pipeline else 'ADC-only'} monitoring to {monitored_count} layers")
    return monitored_count


# Simple training integration
def create_adc_training_monitor(output_dir: str = "./adc_monitoring",
                              plot_every_n_batches: int = 50,
                              layer_patterns: list = None,
                              max_layers: int = 3):
    """
    Create a simple ADC monitoring setup for training
    
    Args:
        output_dir: Directory to save plots
        plot_every_n_batches: How often to generate detailed plots
        layer_patterns: Which layers to monitor
        max_layers: Maximum layers to monitor
        
    Returns:
        Tuple of (plotter, step_function)
    """
    plotter = ADCDistributionPlotter(save_dir=output_dir)
    step_count = 0
    
    def step_monitor():
        """Call this after each training batch"""
        nonlocal step_count
        step_count += 1
        
        # Generate detailed plots periodically
        if step_count % plot_every_n_batches == 0:
            print(f"\nGenerating ADC distribution plots at step {step_count}...")
            
            # Plot current distributions for all monitored layers
            for layer_name in plotter.batch_data.keys():
                data = plotter.batch_data[layer_name]
                
                # Check if we have full pipeline data
                if 'x_raw' in data and data['x_raw']:
                    # Full pipeline plot
                    try:
                        # Get latest data
                        x_raw = torch.from_numpy(data['x_raw'][-1])
                        x_quantized = torch.from_numpy(data['x_quantized'][-1])
                        w_raw = torch.from_numpy(data['w_raw'][-1])
                        w_quantized = torch.from_numpy(data['w_quantized'][-1])
                        before_adc = torch.from_numpy(data['before_adc'][-1])
                        after_adc = torch.from_numpy(data['after_adc'][-1])
                        
                        plotter.plot_full_quantization_pipeline(
                            layer_name, x_raw, x_quantized, w_raw, w_quantized,
                            before_adc, after_adc, step=step_count
                        )
                    except Exception as e:
                        print(f"Failed to plot full pipeline for {layer_name}: {e}")
                        
                elif 'before_adc' in data and data['before_adc']:
                    # ADC-only plot
                    try:
                        # Get latest data
                        latest_before = data['before_adc'][-1]
                        latest_after = data['after_adc'][-1]
                        
                        # Convert back to tensors for plotting
                        before_tensor = torch.from_numpy(latest_before)
                        after_tensor = torch.from_numpy(latest_after)
                        
                        plotter.plot_current_batch_distribution(
                            layer_name, before_tensor, after_tensor, step=step_count
                        )
                    except Exception as e:
                        print(f"Failed to plot ADC for {layer_name}: {e}")
                        
                elif 'before' in data and data['before']:
                    # Legacy format
                    try:
                        # Get latest data
                        latest_before = data['before'][-1]
                        latest_after = data['after'][-1]
                        
                        # Convert back to tensors for plotting
                        before_tensor = torch.from_numpy(latest_before)
                        after_tensor = torch.from_numpy(latest_after)
                        
                        plotter.plot_current_batch_distribution(
                            layer_name, before_tensor, after_tensor, step=step_count
                        )
                    except Exception as e:
                        print(f"Failed to plot legacy format for {layer_name}: {e}")
            
            print(f"ADC plots saved to: {output_dir}")
        
        plotter.step()
    
    return plotter, step_monitor


# Example integration with your training script
def integrate_with_bert_training():
    """
    Example of how to integrate with your bert_adc_integration.py
    """
    
    integration_code = '''
# Add this to your bert_adc_integration.py

from adc_monitoring_integration import create_adc_training_monitor, add_adc_monitoring_to_model

# In main(), after model creation and conversion to ADC:
print("Setting up ADC distribution monitoring...")

# Create monitor
adc_plotter, adc_step_monitor = create_adc_training_monitor(
    output_dir=os.path.join(args.output_dir, "adc_distributions"),
    plot_every_n_batches=100,  # Generate plots every 100 batches
    layer_patterns=["attention", "intermediate", "output"],  # Monitor these layer types
    max_layers=3  # Don't monitor too many layers
)

# Add monitoring to model
add_adc_monitoring_to_model(model, adc_plotter, 
                           layer_patterns=["attention.output.dense", "intermediate.dense", "output.dense"],
                           max_layers=3)

# Modify your ADCLossTrainer to call the monitor:
class ADCLossTrainer(Trainer):
    def __init__(self, *args, adc_step_monitor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.adc_step_monitor = adc_step_monitor
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # ... your existing compute_loss code ...
        
        # Call ADC monitor after each batch
        if self.adc_step_monitor:
            self.adc_step_monitor()
        
        return (loss, outputs) if return_outputs else loss

# Create trainer with ADC monitor:
trainer = ADCLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=metrics_computer.compute_metrics,
    callbacks=[EpochCallback()],
    adc_step_monitor=adc_step_monitor  # Add this line
)

# At the end of training, generate evolution plots:
print("Generating final ADC evolution plots...")
for layer_name in adc_plotter.batch_data.keys():
    adc_plotter.plot_evolution_over_batches(layer_name, max_batches=20)
'''
    
    return integration_code


if __name__ == "__main__":
    print("=== ADC Distribution Monitoring Integration ===")
    print("\nThis module provides easy integration for monitoring ADC quantization effects.")
    print("\nMain functions:")
    print("1. add_adc_monitoring_to_model() - Automatically add monitoring to ADC layers")
    print("2. create_adc_training_monitor() - Create a complete monitoring setup")
    print("3. ADCDistributionPlotter - Detailed before/after ADC visualization")
    
    print("\n" + "="*50)
    print("INTEGRATION CODE:")
    print("="*50)
    print(integrate_with_bert_training())
    
    print("\n" + "="*50)
    print("QUICK TEST:")
    print("="*50)
    print("Run: python ADC/bert_my/adc_distribution_plotter.py")
    print("This will generate example plots in ./adc_dist_test/")
