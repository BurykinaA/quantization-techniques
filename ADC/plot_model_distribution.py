import torch
import os
import re
import matplotlib.pyplot as plt
import numpy as np # Added for processing activations
from torchvision import datasets, transforms # Added for validation data
from torch.utils.data import DataLoader # Added for validation data

# Attempt to import QuantAct, users may need to adjust this
# This is for identifying custom activation quantization layers.
try:
    from ADC.models import QuantAct
    ADC_QUANT_ACT_TYPE = QuantAct
except ImportError:
    print("Warning: ADC.models.QuantAct could not be imported. Activation hooking might be limited to nn.ReLU.")
    ADC_QUANT_ACT_TYPE = None # Fallback

from ADC.models import MLP, MLPADC, MLPQuant, MLPADCAshift

# Determine device (primarily for loading, plotting is on CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for model loading and forward pass: {device}")

RESULTS_DIR = './results'
PLOTS_WEIGHTS_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'weight_distributions')
PLOTS_ACTIVATIONS_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'activation_distributions')
FASHION_MNIST_DATA_DIR = './data' # Directory to store FashionMNIST data
NUM_ACTIVATION_BATCHES = 1 # Number of validation batches to get activations from

os.makedirs(PLOTS_WEIGHTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_ACTIVATIONS_OUTPUT_DIR, exist_ok=True)
os.makedirs(FASHION_MNIST_DATA_DIR, exist_ok=True)

def get_fashion_mnist_val_loader(batch_size=64):
    """
    Creates a DataLoader for the FashionMNIST validation set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)) # FashionMNIST mean/std
    ])
    val_dataset = datasets.FashionMNIST(FASHION_MNIST_DATA_DIR, train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader

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
    input_dim = 784 # Assuming FashionMNIST (28x28)
    num_classes = 10 # Assuming FashionMNIST (10 classes)
    
    # Adjust model instantiation if they require input_dim, num_classes, etc.
    # The current script doesn't show these params passed to constructors,
    # but MLP typically needs them. Assuming defaults or they are handled inside.
    if model_type_str == "MLP":
        model = MLP() # Might need MLP(input_dim=input_dim, num_classes=num_classes)
    elif model_type_str == "MLPADC":
        # Add input_dim, num_classes if MLPADC constructor expects them
        model = MLPADC(**params) 
    elif model_type_str == "MLPQuant":
        model = MLPQuant(**params)
    elif model_type_str == "MLPADCAshift":
        model = MLPADCAshift(**params)
    
    if model is None:
        print(f"Warning: Failed to instantiate model for key '{filename_key}' with parsed type '{model_type_str}'. Skipping.")
        return None, None
        
    return model, filename_key


def plot_distributions():
    val_loader = get_fashion_mnist_val_loader()
    print(f"Using FashionMNIST validation data for activation plotting. Found {len(val_loader.dataset)} samples.")

    weight_files = [f for f in os.listdir(RESULTS_DIR) if re.match(r"model_.*_weights_.*\.pth", f)]

    if not weight_files:
        print(f"No model weight files found in '{RESULTS_DIR}' matching the pattern 'model_*_weights_*.pth'.")
        return

    print(f"Found {len(weight_files)} model weight file(s) to process.")

    for fname in weight_files:
        print(f"\nProcessing file: {fname}")
        filepath = os.path.join(RESULTS_DIR, fname)
        
        name_part_match = re.match(r"model_(.*?)_weights_(\d{8}_\d{6})\.pth", fname)
        if not name_part_match:
            print(f"Could not parse model key or timestamp from filename: '{fname}'. Skipping.")
            continue
        
        filename_key = name_part_match.group(1)
        file_timestamp = name_part_match.group(2)

        model, model_display_name = get_model_instance_and_params(filename_key)

        if not model or not model_display_name:
            continue

        try:
            model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
            model.to(device)
            model.eval()
            print(f"Successfully loaded model: {model_display_name} from {fname}")
        except Exception as e:
            print(f"Error loading state_dict for {fname}: {e}. Skipping.")
            continue

        # --- Plot Weight Distributions ---
        layer_weights_data = []
        layer_names_weights = []
        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name and param.dim() >= 2:
                layer_weights_data.append(param.data.cpu().numpy().flatten())
                layer_names_weights.append(name)
        
        if not layer_weights_data:
            print(f"No suitable weight layers found for model {model_display_name} from file {fname} for weight plotting.")
        else:
            num_layers_with_weights = len(layer_weights_data)
            fig_height = max(5, 3 * num_layers_with_weights) 
            fig_width = 8 
            fig, axes = plt.subplots(num_layers_with_weights, 1, figsize=(fig_width, fig_height), squeeze=False)
            fig.suptitle(f"Weight Distributions for: {model_display_name}\n(Source File Timestamp: {file_timestamp})", fontsize=14)

            for i in range(num_layers_with_weights):
                ax = axes[i, 0]
                weights_flat = layer_weights_data[i]
                layer_name = layer_names_weights[i]
                min_val, max_val = weights_flat.min(), weights_flat.max()
                ax.hist(weights_flat, bins=50, alpha=0.75, color='cornflowerblue', edgecolor='black')
                ax.set_title(f"Layer: {layer_name} (Range: [{min_val:.2e}, {max_val:.2e}])", fontsize=10)
                ax.set_xlabel("Weight Value", fontsize=9)
                ax.set_ylabel("Frequency", fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_savename_weights = f"weights_dist_{filename_key}_{file_timestamp}.png"
            plot_save_path_weights = os.path.join(PLOTS_WEIGHTS_OUTPUT_DIR, plot_savename_weights)
            try:
                plt.savefig(plot_save_path_weights)
                print(f"Saved weight distribution plot to: {plot_save_path_weights}")
            except Exception as e:
                print(f"Error saving weight plot {plot_save_path_weights}: {e}")
            plt.close(fig)

        # --- Plot Activation Distributions ---
        activations_capture_dict = {}
        def capture_activation_hook_fn(layer_name):
            def hook(module, input, output):
                if layer_name not in activations_capture_dict:
                    activations_capture_dict[layer_name] = []
                # For outputs that might be tuples (e.g. LSTM), take the first element
                act_output = output[0] if isinstance(output, tuple) else output
                activations_capture_dict[layer_name].append(act_output.detach().cpu().numpy().flatten())
            return hook

        hooks = []
        target_layer_types_for_activations = [torch.nn.ReLU]
        if ADC_QUANT_ACT_TYPE: # If QuantAct was successfully imported
             target_layer_types_for_activations.append(ADC_QUANT_ACT_TYPE)
        
        print(f"Identifying layers for activation hooking (types: {target_layer_types_for_activations})...")
        for name, module in model.named_modules():
            if any(isinstance(module, t) for t in target_layer_types_for_activations):
                print(f"  Hooking layer for activations: {name} (type: {type(module).__name__})")
                hooks.append(module.register_forward_hook(capture_activation_hook_fn(name)))
        
        if not hooks:
            print(f"No suitable layers found for activation hooking in model {model_display_name}.")
        else:
            print(f"Running model with validation data to capture activations ({NUM_ACTIVATION_BATCHES} batch(es))...")
            model.eval() # Ensure model is in eval mode
            with torch.no_grad(): # No need to track gradients
                for batch_idx, (data, target) in enumerate(val_loader):
                    if batch_idx >= NUM_ACTIVATION_BATCHES:
                        break
                    data = data.to(device)
                    # For MLP, data might need flattening if not already done by model's forward
                    # Assuming model's forward handles input shape (e.g. (batch, 1, 28, 28) -> (batch, 784))
                    # This is correct for FashionMNIST as well.
                    if isinstance(model, MLP) and data.dim() > 2 : # Basic check for MLP with image-like input
                         data = data.view(data.size(0), -1)
                    model(data) 
            
            for hook in hooks: # Remove hooks
                hook.remove()

            # Process captured activations (concatenate if multiple batches)
            layer_activations_data = {}
            layer_names_activations = []
            for name, act_list in activations_capture_dict.items():
                if act_list: # if any activations were captured for this layer
                    layer_activations_data[name] = np.concatenate(act_list)
                    layer_names_activations.append(name)
            
            if not layer_activations_data:
                print(f"No activations captured for model {model_display_name}.")
            else:
                num_layers_with_activations = len(layer_activations_data)
                fig_height_act = max(5, 3 * num_layers_with_activations)
                fig_width_act = 8
                fig_act, axes_act = plt.subplots(num_layers_with_activations, 1, figsize=(fig_width_act, fig_height_act), squeeze=False)
                fig_act.suptitle(f"Activation Distributions for: {model_display_name}\n(Source File Timestamp: {file_timestamp}, {NUM_ACTIVATION_BATCHES} val batch(es))", fontsize=14)

                for i, layer_name in enumerate(layer_names_activations):
                    ax = axes_act[i, 0]
                    activations_flat = layer_activations_data[layer_name]
                    min_val, max_val = activations_flat.min(), activations_flat.max()
                    ax.hist(activations_flat, bins=50, alpha=0.75, color='lightcoral', edgecolor='black')
                    ax.set_title(f"Layer: {layer_name} (Range: [{min_val:.2e}, {max_val:.2e}])", fontsize=10)
                    ax.set_xlabel("Activation Value", fontsize=9)
                    ax.set_ylabel("Frequency", fontsize=9)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_savename_activations = f"activations_dist_{filename_key}_{file_timestamp}.png"
                plot_save_path_activations = os.path.join(PLOTS_ACTIVATIONS_OUTPUT_DIR, plot_savename_activations)
                try:
                    plt.savefig(plot_save_path_activations)
                    print(f"Saved activation distribution plot to: {plot_save_path_activations}")
                except Exception as e:
                    print(f"Error saving activation plot {plot_save_path_activations}: {e}")
                plt.close(fig_act)


if __name__ == '__main__':
    plot_distributions()
    