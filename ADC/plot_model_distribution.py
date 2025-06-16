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

# Import your specific quantizer types
try:
    from ADC.quantizers import AffineQuantizerPerTensor, SymmetricQuantizerPerTensor
    # Tuple of custom quantizer types that need their 'enabled' flag set to False
    CUSTOM_QUANTIZER_TYPES_TO_DISABLE_OBSERVER = (AffineQuantizerPerTensor, SymmetricQuantizerPerTensor)
except ImportError:
    print("Warning: Could not import AffineQuantizerPerTensor or SymmetricQuantizerPerTensor from ADC.quantizers.")
    CUSTOM_QUANTIZER_TYPES_TO_DISABLE_OBSERVER = tuple()

# Determine device (primarily for loading, plotting is on CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for model loading and forward pass: {device}")

RESULTS_DIR = './results_4_bit_right'
PLOTS_WEIGHTS_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'weight_distributions')
PLOTS_ACTIVATIONS_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'activation_distributions')
FASHION_MNIST_DATA_DIR = './data' # Directory to store FashionMNIST data
NUM_ACTIVATION_BATCHES = 5 # Number of validation batches to get activations from

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
    # Example: MLPADCAshift_ashiftTrue_bx8_bw8_ba8_k4
    if not model_type_str:
        match = re.fullmatch(r"MLPADCAshift_ashift(True|False)_bx(\d+)_bw(\d+)_ba(\d+)_k(\d+)", filename_key)
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
    # Example: MLPADC_bx8_bw8_ba8_k4
    if not model_type_str:
        match = re.fullmatch(r"MLPADC_bx(\d+)_bw(\d+)_ba(\d+)_k(\d+)", filename_key)
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

    print(f"Found {len(weight_files)} model weight file(s) to process in '{RESULTS_DIR}'.")

    for fname in weight_files:
        print(f"\nProcessing file: {fname}")
        filepath = os.path.join(RESULTS_DIR, fname)
        
        # Initialize figure variables to None for robust error handling in except block
        fig = None
        fig_act = None
        model_display_name_for_error = "UnknownModel" # Fallback for error message

        try:
            name_part_match = re.match(r"model_(.*?)_weights_(\d{8}_\d{6})\.pth", fname)
            if not name_part_match:
                print(f"Could not parse model key or timestamp from filename: '{fname}'. Skipping.")
                continue
            
            filename_key = name_part_match.group(1)
            file_timestamp = name_part_match.group(2)
            model_display_name_for_error = filename_key # Use filename_key if model instantiation fails

            model, model_display_name = get_model_instance_and_params(filename_key)

            if not model or not model_display_name:
                # get_model_instance_and_params already prints a warning and returns None
                continue
            
            model_display_name_for_error = model_display_name # Update with parsed display name

            # Load state_dict
            model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
            model.to(device)
            model.eval() 
            print(f"Successfully loaded model: {model_display_name} from {fname}")
            
            # --- Attempt to set quantizers to inference mode ---
            print(f"Attempting to set quantizers' 'enabled' flag to False for {model_display_name}...")
            quantizer_modules_adjusted = 0
            
            # First, enable all quantizers for calibration if needed
            needs_calibration = False
            for module_name, sub_module in model.named_modules():
                if CUSTOM_QUANTIZER_TYPES_TO_DISABLE_OBSERVER and isinstance(sub_module, CUSTOM_QUANTIZER_TYPES_TO_DISABLE_OBSERVER):
                    if hasattr(sub_module, 'enabled'):
                        if not sub_module.params_calculated:
                            needs_calibration = True
                            sub_module.enabled = True
                            print(f"  Enabled {module_name} for calibration (type: {type(sub_module).__name__})")
                        else:
                            sub_module.enabled = False
                            print(f"  Set sub_module.enabled = False on {module_name} (type: {type(sub_module).__name__})")
                        quantizer_modules_adjusted += 1
                    else:
                        print(f"  Warning: Module {module_name} (type: {type(sub_module).__name__}) is a custom quantizer but lacks 'enabled' attribute.")

            # Calibrate if needed
            if needs_calibration:
                print("Calibrating quantizers with validation data...")
                model.train()  # Set to training mode for calibration
                with torch.no_grad():
                    for batch_idx, (data_val, target_val) in enumerate(val_loader):
                        if batch_idx >= NUM_ACTIVATION_BATCHES:
                            break
                        data_val = data_val.to(device)
                        if isinstance(model, MLP) and data_val.dim() > 2:
                            data_val = data_val.view(data_val.size(0), -1)
                        model(data_val)
                
                # Now disable all quantizers for inference
                for module_name, sub_module in model.named_modules():
                    if CUSTOM_QUANTIZER_TYPES_TO_DISABLE_OBSERVER and isinstance(sub_module, CUSTOM_QUANTIZER_TYPES_TO_DISABLE_OBSERVER):
                        if hasattr(sub_module, 'enabled'):
                            sub_module.enabled = False
                            print(f"  Set sub_module.enabled = False on {module_name} after calibration")

            # --- Plot Weight Distributions ---
            layer_weights_data = []
            layer_names_weights = []
            for name_param, param in model.named_parameters(): # Renamed 'name' to 'name_param' to avoid conflict
                if param.requires_grad and "weight" in name_param and param.dim() >= 2:
                    layer_weights_data.append(param.data.cpu().numpy().flatten())
                    layer_names_weights.append(name_param)
            
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
                plt.savefig(plot_save_path_weights)
                print(f"Saved weight distribution plot to: {plot_save_path_weights}")
                plt.close(fig)
                fig = None # Reset fig variable

            # --- Plot Activation Distributions ---
            activations_capture_dict = {}
            def capture_activation_hook_fn(layer_name_hook): # Renamed 'layer_name' to avoid conflict
                def hook(module, input_hook, output_hook): # Renamed 'input', 'output'
                    if layer_name_hook not in activations_capture_dict:
                        activations_capture_dict[layer_name_hook] = []
                    act_output = output_hook[0] if isinstance(output_hook, tuple) else output_hook
                    activations_capture_dict[layer_name_hook].append(act_output.detach().cpu().numpy().flatten())
                return hook

            hooks = []
            target_layer_types_for_activations = [torch.nn.ReLU]
            if ADC_QUANT_ACT_TYPE:
                 target_layer_types_for_activations.append(ADC_QUANT_ACT_TYPE)
            
            print(f"Identifying layers for activation hooking (types: {target_layer_types_for_activations})...")
            found_hookable_layer = False
            for name_module, module_obj in model.named_modules(): # Renamed 'name', 'module'
                if any(isinstance(module_obj, t) for t in target_layer_types_for_activations):
                    print(f"  Hooking layer for activations: {name_module} (type: {type(module_obj).__name__})")
                    hooks.append(module_obj.register_forward_hook(capture_activation_hook_fn(name_module)))
                    found_hookable_layer = True
            
            if not found_hookable_layer: # Check simplified from original code for clarity
                 print(f"No suitable layers (e.g., ReLU, QuantAct) found for activation hooking in model {model_display_name}.")

            if hooks:
                print(f"Running model with validation data to capture activations ({NUM_ACTIVATION_BATCHES} batch(es))...")
                model.eval() 
                with torch.no_grad():
                    for batch_idx, (data_val, target_val) in enumerate(val_loader): # Renamed 'data', 'target'
                        if batch_idx >= NUM_ACTIVATION_BATCHES:
                            break
                        data_val = data_val.to(device)
                        if isinstance(model, MLP) and data_val.dim() > 2 :
                             data_val = data_val.view(data_val.size(0), -1)
                        model(data_val) 
                
                for hook in hooks:
                    hook.remove()

                layer_activations_data = {}
                layer_names_activations = []
                for name_act_data, act_list in activations_capture_dict.items(): # Renamed 'name'
                    if act_list:
                        layer_activations_data[name_act_data] = np.concatenate(act_list)
                        layer_names_activations.append(name_act_data)
                
                if not layer_activations_data:
                    print(f"No activations captured for model {model_display_name}.")
                else:
                    num_layers_with_activations = len(layer_activations_data)
                    fig_height_act = max(5, 3 * num_layers_with_activations)
                    fig_width_act = 8
                    fig_act, axes_act = plt.subplots(num_layers_with_activations, 1, figsize=(fig_width_act, fig_height_act), squeeze=False)
                    fig_act.suptitle(f"Activation Distributions for: {model_display_name}\n(Source File Timestamp: {file_timestamp}, {NUM_ACTIVATION_BATCHES} val batch(es))", fontsize=14)

                    for i, layer_name_act_plot in enumerate(layer_names_activations): # Renamed 'layer_name'
                        ax = axes_act[i, 0]
                        activations_flat = layer_activations_data[layer_name_act_plot]
                        min_val, max_val = activations_flat.min(), activations_flat.max()
                        ax.hist(activations_flat, bins=50, alpha=0.75, color='lightcoral', edgecolor='black')
                        ax.set_title(f"Layer: {layer_name_act_plot} (Range: [{min_val:.2e}, {max_val:.2e}])", fontsize=10)
                        ax.set_xlabel("Activation Value", fontsize=9)
                        ax.set_ylabel("Frequency", fontsize=9)
                        ax.tick_params(axis='both', which='major', labelsize=8)
                        ax.grid(True, linestyle='--', alpha=0.7)
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plot_savename_activations = f"activations_dist_{filename_key}_{file_timestamp}.png"
                    plot_save_path_activations = os.path.join(PLOTS_ACTIVATIONS_OUTPUT_DIR, plot_savename_activations)
                    plt.savefig(plot_save_path_activations)
                    print(f"Saved activation distribution plot to: {plot_save_path_activations}")
                    plt.close(fig_act)
                    fig_act = None # Reset fig_act variable

        except Exception as e:
            print(f"Error processing file {fname} (model key: {model_display_name_for_error}): {e}. Skipping this file.")
            # Ensure any figures created for this problematic file are closed
            if fig is not None and plt.fignum_exists(fig.number):
                plt.close(fig)
            if fig_act is not None and plt.fignum_exists(fig_act.number):
                plt.close(fig_act)
            continue # Explicitly continue to the next file in the loop


if __name__ == '__main__':
    plot_distributions()
    