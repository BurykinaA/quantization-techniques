import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from ADC.ste import ste_round, ste_floor # Import from your ste.py

# --- STE Functions --- (These will be removed)
# def ste_round(x):
#     """Straight-Through Estimator for round."""
#     return (x.round() - x).detach() + x
# 
# def ste_floor(x):
#     """Straight-Through Estimator for floor."""
#     return (x.floor() - x).detach() + x

# --- Quantization Function ---
def quantize_uniform(x, n_bits, ste_function=None, data_min_val=None, data_max_val=None):
    """
    Performs uniform quantization on the input tensor x over a pre-defined range.

    Args:
        x (torch.Tensor): Input tensor.
        n_bits (int): Number of bits for quantization (must be >= 1).
        ste_function (callable, optional): STE function (e.g., ste_round, ste_floor). 
                                           If None, standard torch.round is used.
        data_min_val (float): Pre-defined minimum value of the input range.
        data_max_val (float): Pre-defined maximum value of the input range.

    Returns:
        torch.Tensor: Quantized and dequantized tensor.
    """
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1 for this quantization function.")
    if data_min_val is None or data_max_val is None:
        raise ValueError("data_min_val and data_max_val must be provided.")

    data_min = torch.tensor(data_min_val, device=x.device, dtype=x.dtype)
    data_max = torch.tensor(data_max_val, device=x.device, dtype=x.dtype)
    
    num_levels = 2**n_bits
    
    if (data_max - data_min).abs() < 1e-9: # Input range is effectively zero
        # All values quantize to data_min (or the single level available if range is zero)
        return torch.full_like(x, data_min)

    # Scale: map [data_min, data_max] to [0, num_levels - 1]
    scale = (data_max - data_min) / (num_levels - 1)
    
    # Transform x to the range [0, num_levels - 1] before rounding
    # Clamp input x to the [data_min, data_max] range first
    x_clamped_input = torch.clamp(x, data_min, data_max)
    x_transformed = (x_clamped_input - data_min) / scale
    
    # Apply rounding (standard or STE)
    if ste_function:
        quantized_levels_float = ste_function(x_transformed)
    else:
        quantized_levels_float = torch.round(x_transformed) # Default: standard rounding
    
    # Clamp to the [0, num_levels - 1] integer level range
    quantized_levels_int = torch.clamp(quantized_levels_float, 0, num_levels - 1)
    
    # Dequantize back to original data scale (approximately)
    x_dequant = quantized_levels_int * scale + data_min
    
    return x_dequant

# --- Main Experiment ---
def run_quantization_analysis():
    n_bits = 3 
    num_samples = 5000

    # Generate input data (e.g., Normal distribution)
    input_data_mean = 0.0
    input_data_std = 1.0
    input_data = torch.randn(num_samples) * input_data_std + input_data_mean
    
    # Define a fixed range for quantization. 
    # This ensures all methods use the same scale and zero-point for fair comparison.
    # Using a range slightly wider than 3-sigma for normal distribution or based on data.
    # fixed_min_val = input_data_mean - 3 * input_data_std
    # fixed_max_val = input_data_mean + 3 * input_data_std
    # Or, more adaptively, from the generated data itself (but fixed for all methods)
    fixed_min_val = input_data.min().item()
    fixed_max_val = input_data.max().item()
    
    # Ensure fixed_max_val is greater than fixed_min_val
    if fixed_max_val <= fixed_min_val:
        fixed_max_val = fixed_min_val + 1.0 # Arbitrary small range if data is constant

    print(f"--- Quantization Analysis (n_bits = {n_bits}, Samples = {num_samples}) ---")
    print(f"Input data generated with mean ~{input_data.mean():.2f}, std ~{input_data.std():.2f}")
    print(f"Actual input data range: [{input_data.min():.4f}, {input_data.max():.4f}]")
    print(f"Using fixed quantization range for all methods: [{fixed_min_val:.4f}, {fixed_max_val:.4f}]")

    # 1. Standard Numeric Quantization (using torch.round)
    quant_standard = quantize_uniform(input_data, n_bits, ste_function=None, 
                                      data_min_val=fixed_min_val, data_max_val=fixed_max_val)
    
    # 2. ADC-style Quantization with STE-Round
    quant_adc_ste_round = quantize_uniform(input_data, n_bits, ste_function=ste_round,
                                           data_min_val=fixed_min_val, data_max_val=fixed_max_val)

    # 3. ADC-style Quantization with STE-Floor
    quant_adc_ste_floor = quantize_uniform(input_data, n_bits, ste_function=ste_floor,
                                           data_min_val=fixed_min_val, data_max_val=fixed_max_val)

    results = {
        "Input Data (Original)": input_data,
        "Standard Quant (torch.round)": quant_standard,
        "ADC STE-Round Quant": quant_adc_ste_round,
        "ADC STE-Floor Quant": quant_adc_ste_floor,
    }

    # --- Analysis and Visualization ---
    num_plots = len(results)
    plt.figure(figsize=(12, num_plots * 4)) # Adjusted figure size
    
    plot_idx = 1
    for name, data_to_plot in results.items():
        plt.subplot(num_plots, 1, plot_idx)
        
        if name == "Input Data (Original)":
            plt.hist(data_to_plot.cpu().numpy(), bins=100, color='gray', alpha=0.7)
            plt.title(f"Distribution: {name}")
        else:
            unique_values = torch.unique(data_to_plot)
            num_unique_bins = len(unique_values)
            
            print(f"\nResults for: {name}")
            print(f"  Number of unique quantized values (bins): {num_unique_bins} (Expected up to {2**n_bits})")
            # print(f"  Unique values: {np.round(unique_values.cpu().numpy(), 5)}") # Can be long

            counts = torch.stack([(data_to_plot == v).sum() for v in unique_values])
            
            # Determine bar width
            bar_width_val = (fixed_max_val - fixed_min_val) / (2**n_bits * 1.5) # Heuristic
            if num_unique_bins > 1:
                sorted_unique_vals = torch.sort(unique_values).values
                min_diff = (sorted_unique_vals[1:] - sorted_unique_vals[:-1]).min().item()
                if min_diff > 1e-6 : # Ensure min_diff is substantial
                     bar_width_val = min_diff * 0.8
            
            plt.bar(unique_values.cpu().numpy(), counts.cpu().numpy(), width=bar_width_val, alpha=0.7, label=f'Unique Bins: {num_unique_bins}')
            plt.title(f"Distribution: {name} (Quantized)")
            plt.legend(loc='upper right')

        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plot_idx += 1

    plt.tight_layout()
    output_dir = os.path.join("ADC", "analysis_results") # Place inside ADC folder
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"quant_distributions_nbits{n_bits}_samples{num_samples}.png")
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    # plt.show() # Uncomment to display plots directly

    print("\n--- Discussion ---")
    print("This analysis visualizes the forward pass of different quantization schemes.")
    print("1. 'Standard Quant (torch.round)' and 'ADC STE-Round Quant' should produce identical distributions")
    print("   because the forward pass of ste_round (from ADC.ste) should be torch.round(x).")
    print("   The Straight-Through Estimator (STE) aspect primarily affects backpropagation by providing a")
    print("   non-zero gradient (typically 1) for the rounding operation, aiding model training.")
    print("2. 'ADC STE-Floor Quant' will likely show a different distribution compared to the round-based methods")
    print("   due to the inherent difference between floor and round operations.")
    print("3. The number of unique bins observed depends on the input data distribution and the quantization")
    print("   range. It can be up to 2^n_bits levels. If the input data is concentrated or does not span")
    print("   all quantization thresholds, fewer bins might be populated.")
    print("4. This simulation does not show any 'smoothing' or 'bin merging' in the forward pass due to STE itself.")
    print("   Such effects, if desired, would typically require different quantization strategies (e.g., soft quantization).")

if __name__ == '__main__':
    run_quantization_analysis() 