import torch
import os
import re
import matplotlib.pyplot as plt
from ADC.models import MLP, MLPADC, MLPQuant, MLPADCAshift

# Determine device (primarily for loading, plotting is on CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for model loading (if applicable): {device}")

RESULTS_DIR = './results'
PLOTS_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'weight_distributions')
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)

def get_model_instance_and_params(filename_key):
    """
    Parses the filename key to determine model type and parameters,
    then instantiates the model.
    """
    params = {}
    model_type_str = None
    
    # The filename_key is the part of the filename like "MLPADC_bx8_bw8_ba8_k4"
    # Order of regex matching is important: more specific patterns first.
    # We use re.fullmatch to ensure the entire key string matches the pattern.

    # MLPADCAshift+W-Reshape
    # Example: MLPADCAshiftplusW-Reshape_ashiftTrue_bx8bw8ba8k4lk0.001
    match = re.fullmatch(r"MLPADCAshiftplusW-Reshape_ashift(True|False)_bx(\d+)bw(\d+)ba(\d+)k(\d+)lk([\d.]+)", filename_key)
    if match:
        model_type_str = "MLPADCAshift"
        params['ashift_enabled'] = match.group(1) == "True"
        params['bx'] = int(match.group(2))
        params['bw'] = int(match.group(3))
        params['ba'] = int(match.group(4))
        params['k'] = int(match.group(5))
        # lk (lambda_kurtosis) is a training param, not model architecture
        
    # MLPADCAshift (no W-Reshape)
    # Example: MLPADCAshift_ashiftTrue_bx8bw8ba8k4
    if not model_type_str:
        match = re.fullmatch(r"MLPADCAshift_ashift(True|False)_bx(\d+)bw(\d+)ba(\d+)k(\d+)", filename_key)
        if match:
            model_type_str = "MLPADCAshift"
            params['ashift_enabled'] = match.group(1) == "True"
            params['bx'] = int(match.group(2))
            params['bw'] = int(match.group(3))
            params['ba'] = int(match.group(4))
            params['k'] = int(match.group(5))

    # MLPADC+W-Reshape
    # Example: MLPADCplusW-Reshape_bx8bw8ba8k4lk0.001
    if not model_type_str:
        match = re.fullmatch(r"MLPADCplusW-Reshape_bx(\d+)bw(\d+)ba(\d+)k(\d+)lk([\d.]+)", filename_key)
        if match:
            model_type_str = "MLPADC"
            params['bx'] = int(match.group(1))
            params['bw'] = int(match.group(2))
            params['ba'] = int(match.group(3))
            params['k'] = int(match.group(4))

    # MLPADC (no W-Reshape)
    # Example: MLPADC_bx8bw8ba8k4
    if not model_type_str:
        match = re.fullmatch(r"MLPADC_bx(\d+)bw(\d+)ba(\d+)k(\d+)", filename_key)
        if match:
            model_type_str = "MLPADC"
            params['bx'] = int(match.group(1))
            params['bw'] = int(match.group(2))
            params['ba'] = int(match.group(3))
            params['k'] = int(match.group(4))
            
    # MLPQuant
    # Example: MLPQuant_bx8_bw8
    if not model_type_str:
        match = re.fullmatch(r"MLPQuant_bx(\d+)_bw(\d+)", filename_key)
        if match:
            model_type_str = "MLPQuant"
            params['bx'] = int(match.group(1))
            params['bw'] = int(match.group(2))

    # MLP_Baseline
    # Example: MLP_Baseline
    if not model_type_str:
        match = re.fullmatch(r"MLP_Baseline", filename_key)
        if match:
            model_type_str = "MLP"

    if not model_type_str:
        print(f"Warning: Could not parse model type or params for key '{filename_key}'. Skipping.")
        return None, None

    # Instantiate model
    model = None
    if model_type_str == "MLP":
        model = MLP()
    elif model_type_str == "MLPADC":
        model = MLPADC(**params)
    elif model_type_str == "MLPQuant":
        model = MLPQuant(**params)
    elif model_type_str == "MLPADCAshift":
        model = MLPADCAshift(**params)
    
    if model is None:
        print(f"Warning: Failed to instantiate model for key '{filename_key}' with parsed type '{model_type_str}'. Skipping.")
        return None, None
        
    # Return the instantiated model and the original key for display purposes
    return model, filename_key


def plot_all_weight_distributions():
    weight_files = [f for f in os.listdir(RESULTS_DIR) if re.match(r"model_.*_weights_.*\.pth", f)]

    if not weight_files:
        print(f"No model weight files found in '{RESULTS_DIR}' matching the pattern 'model_*_weights_*.pth'.")
        return

    print(f"Found {len(weight_files)} model weight file(s) to process.")

    for fname in weight_files:
        print(f"\nProcessing file: {fname}")
        filepath = os.path.join(RESULTS_DIR, fname)
        
        # Extract model key and timestamp from filename
        # Example: model_MLPADC_bx8_bw8_ba8_k4_weights_20231120_103045.pth
        # filename_key will be "MLPADC_bx8_bw8_ba8_k4"
        # file_timestamp will be "20231120_103045"
        name_part_match = re.match(r"model_(.*?)_weights_(\d{8}_\d{6})\.pth", fname)
        if not name_part_match:
            print(f"Could not parse model key or timestamp from filename: '{fname}'. Skipping.")
            continue
        
        filename_key = name_part_match.group(1)
        file_timestamp = name_part_match.group(2)

        model, model_display_name = get_model_instance_and_params(filename_key)

        if not model or not model_display_name:
            continue # Error message already printed by get_model_instance_and_params

        try:
            # Load weights to CPU map_location by default for broader compatibility
            model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
            model.to(device) # Then move to the designated device if needed for model operations (eval is fine on CPU)
            model.eval()
            print(f"Successfully loaded model: {model_display_name} from {fname}")
        except Exception as e:
            print(f"Error loading state_dict for {fname}: {e}. Skipping.")
            continue

        layer_weights_data = []
        layer_names = []
        for name, param in model.named_parameters():
            # We are interested in 'weight' parameters, typically from Linear layers or their custom equivalents.
            # Filter for parameters that are likely main weights of layers (e.g., 'layers.0.weight', 'fc.weight')
            # and are at least 2D (to avoid biases if they were named '...weight').
            if param.requires_grad and "weight" in name and param.dim() >= 2:
                layer_weights_data.append(param.data.cpu().numpy().flatten())
                layer_names.append(name)
        
        if not layer_weights_data:
            print(f"No suitable weight layers found for model {model_display_name} from file {fname}.")
            continue

        num_layers_with_weights = len(layer_weights_data)
        
        fig_height = max(5, 3 * num_layers_with_weights) 
        fig_width = 8 

        # Use squeeze=False to ensure axes is always a 2D array for consistent indexing
        fig, axes = plt.subplots(num_layers_with_weights, 1, figsize=(fig_width, fig_height), squeeze=False)
        
        fig.suptitle(f"Weight Distributions for: {model_display_name}\n(Source File Timestamp: {file_timestamp})", fontsize=14)

        for i in range(num_layers_with_weights):
            ax = axes[i, 0] # Access subplot using [row, 0] due to squeeze=False and single column
            weights_flat = layer_weights_data[i]
            layer_name = layer_names[i]
            
            min_val, max_val = weights_flat.min(), weights_flat.max()
            ax.hist(weights_flat, bins=50, alpha=0.75, color='cornflowerblue', edgecolor='black')
            ax.set_title(f"Layer: {layer_name} (Range: [{min_val:.2e}, {max_val:.2e}])", fontsize=10)
            ax.set_xlabel("Weight Value", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle
        
        # Construct a unique plot filename using the original filename key and its timestamp
        plot_savename = f"weights_dist_{filename_key}_{file_timestamp}.png"
        plot_save_path = os.path.join(PLOTS_OUTPUT_DIR, plot_savename)
        
        try:
            plt.savefig(plot_save_path)
            print(f"Saved weight distribution plot to: {plot_save_path}")
        except Exception as e:
            print(f"Error saving plot {plot_save_path}: {e}")
        plt.close(fig) # Close the figure to free memory

if __name__ == '__main__':
    plot_all_weight_distributions() 