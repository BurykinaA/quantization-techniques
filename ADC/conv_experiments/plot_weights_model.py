import torch
import os
import re
import matplotlib.pyplot as plt
from ADC.models import resnet18_cifar, resnet18_cifar_adc
from ADC.conv_experiments.conv_experiment_setup import get_config, define_experiments, sanitize_filename

def get_model_instance_from_key(filename_key, config):
    """
    Parses the sanitized filename key to determine model type and parameters,
    then instantiates the model.
    """
    q_params = config['quant_params']
    num_classes = config['num_classes']
    model_args = {'num_classes': num_classes}
    model_class = None

    # --- Regex matching on sanitized filenames ---
    
    # ResNet18 Ashift+W-Reshape
    match = re.fullmatch(r"ResNet18_AshiftplusW-Reshape_ashift(True|False)_bx(\d+)bw(\d+)ba(\d+)k(\d+)lk([\d.]+)", filename_key)
    if match:
        model_class = resnet18_cifar_adc
        # Note: Assuming resnet18_cifar_adc accepts an 'ashift' parameter
        model_args['ashift'] = match.group(1) == "True"
        model_args['bx'] = int(match.group(2))
        model_args['bw'] = int(match.group(3))
        model_args['ba'] = int(match.group(4))
        model_args['k'] = int(match.group(5))

    # ResNet18ADCAshift
    if not model_class:
        match = re.fullmatch(r"ResNet18ADCAshift_ashift(True|False)_bx(\d+)bw(\d+)ba(\d+)k(\d+)", filename_key)
        if match:
            model_class = resnet18_cifar_adc
            model_args['ashift'] = match.group(1) == "True"
            model_args['bx'] = int(match.group(2))
            model_args['bw'] = int(match.group(3))
            model_args['ba'] = int(match.group(4))
            model_args['k'] = int(match.group(5))

    # ResNet18(ADC)+W-Reshape
    if not model_class:
        match = re.fullmatch(r"ResNet18ADCplusW-Reshape_bx(\d+)bw(\d+)ba(\d+)k(\d+)lk([\d.]+)", filename_key)
        if match:
            model_class = resnet18_cifar_adc
            model_args['bx'] = int(match.group(1))
            model_args['bw'] = int(match.group(2))
            model_args['ba'] = int(match.group(3))
            model_args['k'] = int(match.group(4))

    # ResNet18 (ADC)
    if not model_class and filename_key == 'ResNet18_ADC':
        model_class = resnet18_cifar_adc
        model_args.update(q_params)

    # ResNet18 (Baseline)
    if not model_class and filename_key == 'ResNet18_Baseline':
        model_class = resnet18_cifar

    if not model_class:
        print(f"Warning: Could not parse model type for key '{filename_key}'. Skipping.")
        return None, None

    try:
        model = model_class(**model_args)
        return model
    except Exception as e:
        print(f"Error instantiating model for key '{filename_key}': {e}")
        return None, None

def plot_weight_distributions():
    """Finds all model weight files, loads them, and plots their weight distributions."""
    config = get_config()
    device = config['device']
    results_dir = config['results_dir']
    plots_output_dir = os.path.join(results_dir, 'weight_distributions')
    os.makedirs(plots_output_dir, exist_ok=True)

    print(f"Reading model weights from: {results_dir}")
    print(f"Saving plots to: {plots_output_dir}")

    weight_files = [f for f in os.listdir(results_dir) if f.startswith('model_') and f.endswith('.pth') and '_weights_' in f]
    
    if not weight_files:
        print("No model weight files found to process.")
        return

    print(f"Found {len(weight_files)} model weight file(s) to process.")
    experiments = define_experiments(config)

    for fname in weight_files:
        print(f"\nProcessing file: {fname}")
        filepath = os.path.join(results_dir, fname)

        match = re.match(r"model_(.*?)_weights_(\d{8}_\d{6})\.pth", fname)
        if not match:
            print(f"Could not parse filename: '{fname}'. Skipping.")
            continue
        
        filename_key = match.group(1)
        file_timestamp = match.group(2)
        
        original_model_name = next((exp['name'] for exp in experiments if sanitize_filename(exp['name']) == filename_key), filename_key)

        model = get_model_instance_from_key(filename_key, config)

        if not model:
            continue

        try:
            state_dict = torch.load(filepath, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"Successfully loaded model: {original_model_name}")
        except Exception as e:
            print(f"Error loading state_dict for {fname}: {e}. Skipping.")
            continue

        layer_weights_data = []
        layer_names = []
        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name and param.dim() >= 2:
                layer_weights_data.append(param.data.cpu().numpy().flatten())
                layer_names.append(name)
        
        if not layer_weights_data:
            print(f"No suitable weight layers found for model {original_model_name}.")
            continue

        num_layers = len(layer_weights_data)
        cols = 2
        rows = (num_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), squeeze=False)
        axes_flat = axes.flatten()

        fig.suptitle(f"Weight Distributions for: {original_model_name}\n(Timestamp: {file_timestamp})", fontsize=16)

        for i, (weights, layer_name) in enumerate(zip(layer_weights_data, layer_names)):
            ax = axes_flat[i]
            min_val, max_val = weights.min(), weights.max()
            ax.hist(weights, bins=50, alpha=0.75, color='cornflowerblue', edgecolor='black')
            ax.set_title(f"Layer: {layer_name} (Range: [{min_val:.2e}, {max_val:.2e}])")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, linestyle='--', alpha=0.6)

        for i in range(num_layers, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_savename = f"weights_dist_{filename_key}_{file_timestamp}.png"
        plot_save_path = os.path.join(plots_output_dir, plot_savename)
        
        try:
            plt.savefig(plot_save_path)
            print(f"Saved weight distribution plot to: {plot_save_path}")
        except Exception as e:
            print(f"Error saving plot {plot_save_path}: {e}")
        plt.close(fig)

if __name__ == '__main__':
    plot_weight_distributions() 
    