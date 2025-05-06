import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from ADC.ste import ste_round, ste_floor # Import from your ste.py
from scipy.stats import laplace, t # For Laplace and Student's t

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
    Returns dequantized values.
    """
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1 for this quantization function.")
    if data_min_val is None or data_max_val is None:
        raise ValueError("data_min_val and data_max_val must be provided.")

    data_min = torch.tensor(data_min_val, device=x.device, dtype=x.dtype)
    data_max = torch.tensor(data_max_val, device=x.device, dtype=x.dtype)
    
    num_levels = 2**n_bits
    
    if num_levels <= 1 or (data_max - data_min).abs() < 1e-9:
        # If only one level or zero range, all values quantize to data_min
        # This also handles data_max == data_min gracefully for num_levels > 1
        return torch.full_like(x, data_min)

    scale = (data_max - data_min) / (num_levels - 1)
    
    x_clamped_input = torch.clamp(x, data_min, data_max)
    # Transform to [0, num_levels - 1] range for quantization
    x_transformed = (x_clamped_input - data_min) / scale
    
    if ste_function:
        quantized_levels_float = ste_function(x_transformed)
    else:
        quantized_levels_float = torch.round(x_transformed) 
    
    quantized_levels_int = torch.clamp(quantized_levels_float, 0, num_levels - 1)
    
    x_dequant = quantized_levels_int * scale + data_min
    
    return x_dequant

# --- Data Generation Functions ---
def generate_normal_data(num_samples, mean=0.0, std=0.1): # Typical for weights
    return torch.randn(num_samples) * std + mean, f"Normal Weights (mean={mean:.2f}, std={std:.2f})"

def generate_exponential_data(num_samples, rate=1.0): # Typical for ReLU activations
    # Exponential Pytorch takes rate = 1/scale. Scale is mean.
    return torch.distributions.Exponential(rate).sample((num_samples,)), f"Exponential Activations (rate={rate:.1f})"

def generate_uniform_data(num_samples, low=-2.0, high=2.0):
    return torch.rand(num_samples) * (high - low) + low, f"Uniform (low={low}, high={high})"

def generate_student_t_data(num_samples, df=3, loc=0.0, scale=1.0):
    return torch.tensor(t.rvs(df=df, loc=loc, scale=scale, size=num_samples), dtype=torch.float32), \
           f"Student-t (df={df}, loc={loc}, scale={scale})"

def generate_laplace_data(num_samples, loc=0.0, scale=1.0): 
    return torch.tensor(laplace.rvs(loc=loc, scale=scale, size=num_samples), dtype=torch.float32), \
           f"Laplace (loc={loc}, scale={scale})"

def generate_bimodal_data(num_samples, mean1=-2.0, std1=0.5, mean2=2.0, std2=0.5):
    n1 = num_samples // 2
    n2 = num_samples - n1
    data1 = torch.randn(n1) * std1 + mean1
    data2 = torch.randn(n2) * std2 + mean2
    return torch.cat((data1, data2)), f"Bimodal (m1={mean1},s1={std1}; m2={mean2},s2={std2})"

# --- New Main Experiment for Neural Network Element Quantization ---
def run_neural_element_quantization_analysis(n_bits=4, num_samples=10000):
    print(f"\n--- Neural Element Quantization Analysis (Weights & Activations) ---")
    print(f"    n_bits = {n_bits}, Samples = {num_samples}")

    weights_data, weights_dist_name = generate_normal_data(num_samples, std=0.05) # Smaller std for weights
    activations_data, activations_dist_name = generate_exponential_data(num_samples, rate=0.5) # rate makes mean 1/rate=2

    # Determine ranges
    # Weights: Symmetric
    w_abs_max = weights_data.abs().max().item()
    w_min_val_sym, w_max_val_sym = -w_abs_max, w_abs_max
    if w_max_val_sym <= w_min_val_sym + 1e-9 : # Avoid zero range for constant data
        w_max_val_sym = w_min_val_sym + 1.0
        w_min_val_sym = -w_max_val_sym


    # Activations: Affine
    x_min_val_aff, x_max_val_aff = activations_data.min().item(), activations_data.max().item()
    if x_max_val_aff <= x_min_val_aff + 1e-9 : # Avoid zero range
         x_max_val_aff = x_min_val_aff + 1.0

    print(f"    {weights_dist_name}: Symmetric Range [{w_min_val_sym:.4f}, {w_max_val_sym:.4f}]")
    print(f"    {activations_dist_name}: Affine Range [{x_min_val_aff:.4f}, {x_max_val_aff:.4f}]")

    # Quantize Weights
    w_q_std_round = quantize_uniform(weights_data, n_bits, ste_function=None, data_min_val=w_min_val_sym, data_max_val=w_max_val_sym)
    w_q_adc_ste_floor = quantize_uniform(weights_data, n_bits, ste_function=ste_floor, data_min_val=w_min_val_sym, data_max_val=w_max_val_sym)
    w_q_adc_ste_round = quantize_uniform(weights_data, n_bits, ste_function=ste_round, data_min_val=w_min_val_sym, data_max_val=w_max_val_sym)

    # Quantize Activations
    x_q_std_round = quantize_uniform(activations_data, n_bits, ste_function=None, data_min_val=x_min_val_aff, data_max_val=x_max_val_aff)
    x_q_adc_ste_floor = quantize_uniform(activations_data, n_bits, ste_function=ste_floor, data_min_val=x_min_val_aff, data_max_val=x_max_val_aff)
    x_q_adc_ste_round = quantize_uniform(activations_data, n_bits, ste_function=ste_round, data_min_val=x_min_val_aff, data_max_val=x_max_val_aff)

    # Products of dequantized values
    y_std_round = w_q_std_round * x_q_std_round
    y_adc_ste_floor = w_q_adc_ste_floor * x_q_adc_ste_floor
    y_adc_ste_round = w_q_adc_ste_round * x_q_adc_ste_round
    
    quantized_data_collections = {
        "Weights": [
            ("Orig.", weights_data, weights_dist_name),
            ("Std Round", w_q_std_round, f"W Quant (Std Round, {n_bits}-bit)"),
            ("ADC STE-Floor", w_q_adc_ste_floor, f"W Quant (ADC STE-Floor, {n_bits}-bit)"),
            ("ADC STE-Round", w_q_adc_ste_round, f"W Quant (ADC STE-Round, {n_bits}-bit)"),
        ],
        "Activations": [
            ("Orig.", activations_data, activations_dist_name),
            ("Std Round", x_q_std_round, f"X Quant (Std Round, {n_bits}-bit)"),
            ("ADC STE-Floor", x_q_adc_ste_floor, f"X Quant (ADC STE-Floor, {n_bits}-bit)"),
            ("ADC STE-Round", x_q_adc_ste_round, f"X Quant (ADC STE-Round, {n_bits}-bit)"),
        ],
        "Products (W_dequant * X_dequant)": [
            ("Std Round", y_std_round, f"Product (Std Round, {n_bits}-bit W&X)"),
            ("ADC STE-Floor", y_adc_ste_floor, f"Product (ADC STE-Floor, {n_bits}-bit W&X)"),
            ("ADC STE-Round", y_adc_ste_round, f"Product (ADC STE-Round, {n_bits}-bit W&X)"),
        ]
    }

    # --- Bin Analysis (Printed) ---
    for data_type, collections in quantized_data_collections.items():
        print(f"\n  --- {data_type} Bin Analysis ---")
        for name_suffix, data, title in collections:
            if "Orig." in name_suffix: # Skip original for bin count
                 print(f"    {title}: Original data, mean={data.mean():.3f}, std={data.std():.3f}")
                 continue
            unique_bins = torch.unique(data)
            num_unique = len(unique_bins)
            print(f"    {title}: Unique Bins = {num_unique} (Max expected for W/X: {2**n_bits})")
            # print(f"      Bins: {np.round(unique_bins.cpu().numpy(), 4)}") # Optional: print bins

    # --- Visualization ---
    fig, axes = plt.subplots(4, 3, figsize=(18, 16)) # 4 rows for Orig W/X, Quant W, Quant X, Product
    fig.suptitle(f"Neural Element Quantization Analysis ({n_bits}-bit, {num_samples} samples)", fontsize=16)

    # Row 0: Original Data
    axes[0, 0].hist(weights_data.cpu().numpy(), bins=100, color='gray', alpha=0.8, density=True)
    axes[0, 0].set_title(weights_dist_name)
    axes[0, 0].grid(True)
    axes[0, 1].hist(activations_data.cpu().numpy(), bins=100, color='gray', alpha=0.8, density=True)
    axes[0, 1].set_title(activations_dist_name)
    axes[0, 1].grid(True)
    axes[0, 2].axis('off') # Empty subplot

    plot_configs = [
        (w_q_std_round, "W Quant (Std Round)"),
        (w_q_adc_ste_floor, "W Quant (ADC STE-Floor)"),
        (w_q_adc_ste_round, "W Quant (ADC STE-Round)"),
        (x_q_std_round, "X Quant (Std Round)"),
        (x_q_adc_ste_floor, "X Quant (ADC STE-Floor)"),
        (x_q_adc_ste_round, "X Quant (ADC STE-Round)"),
        (y_std_round, "Product (Std Round W*X)"),
        (y_adc_ste_floor, "Product (ADC STE-Floor W*X)"),
        (y_adc_ste_round, "Product (ADC STE-Round W*X)"),
    ]
    
    row_map = [1, 1, 1, 2, 2, 2, 3, 3, 3] # Maps plot_configs index to row in subplot
    col_map = [0, 1, 2, 0, 1, 2, 0, 1, 2] # Maps plot_configs index to col in subplot

    for i, (data, title) in enumerate(plot_configs):
        ax = axes[row_map[i], col_map[i]]
        unique_values = torch.unique(data)
        num_unique_bins = len(unique_values)
        
        # For product plots, use histogram; for quantized W/X, use bar if few bins.
        if "Product" in title or num_unique_bins > 2**n_bits + 5 : # Heuristic for histogram vs bar
            ax.hist(data.cpu().numpy(), bins=50 if "Product" in title else num_unique_bins, alpha=0.7, density=True)
        else:
            counts = torch.stack([(data == v).sum() for v in unique_values])
            # Determine bar width for quantized W/X
            data_range = data.max() - data.min()
            bar_width_val = data_range / (max(1,num_unique_bins) * 2.0) # Heuristic
            if num_unique_bins > 1:
                sorted_unique_vals = torch.sort(unique_values).values
                min_diff = (sorted_unique_vals[1:] - sorted_unique_vals[:-1]).min().item()
                if min_diff > 1e-6 : 
                     bar_width_val = min_diff * 0.7
            
            ax.bar(unique_values.cpu().numpy(), counts.cpu().numpy() / num_samples, width=bar_width_val, alpha=0.7, label=f'Bins: {num_unique_bins}')

        ax.set_title(f"{title}\nUnique Vals: {num_unique_bins}")
        ax.set_ylabel("Density / Norm. Freq.")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_dir = os.path.join("ADC", "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"neural_elem_quant_nbits{n_bits}.png")
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    plt.close()

# Commenting out the previous main analysis loop
# def run_quantization_analysis_for_distribution(input_data_generator, n_bits, num_samples):
#    ... (previous code) ...
# def run_all_distribution_analyses():
#    ... (previous code) ...

if __name__ == '__main__':
    run_neural_element_quantization_analysis(n_bits=4)
    # run_neural_element_quantization_analysis(n_bits=3)
    # run_all_distribution_analyses() # Keep if you want the old analysis too 