import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

from ADC.models import MLP, MLPADC, MLPQuant # Assuming MLP is not part of this specific experiment comparison
from ADC.train_utils import train_model

def run_experiment():
    # --- Configuration ---
    num_epochs_exp = 20  # Adjust as needed
    batch_size_train = 1024
    batch_size_test = 1024
    learning_rate = 0.001

    # Quantization parameters
    bx_val = 8
    bw_val = 8
    ba_val = 8  # For ADC
    k_val = 4   # For ADC

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    transform_fm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)) # FashionMNIST mean/std
    ])

    train_dataset_fm = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_fm)
    test_dataset_fm = datasets.FashionMNIST('./data', train=False, download=True, transform=transform_fm)

    train_loader = DataLoader(train_dataset_fm, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset_fm, batch_size=batch_size_test, shuffle=False)
    # For calibration, we can use a portion of the training loader or the full one.
    # Using train_loader itself for calibration in train_model if no specific calib_loader is passed.

    criterion = nn.CrossEntropyLoss()

    # --- Experiment with MLP (Baseline) ---
    print("\n--- Experiment with MLP (Baseline) ---")
    model_mlp = MLP()
    optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=learning_rate)

    train_losses_mlp, train_accs_mlp, test_losses_mlp, test_accs_mlp = train_model(
        model_mlp, optimizer_mlp, train_loader, test_loader, criterion, device,
        num_epochs=num_epochs_exp,
        model_name="MLP (Baseline)",
        # No calib_loader needed for baseline MLP
    )

    # --- Experiment with MLPADC ---
    print("\n--- Experiment with MLPADC ---")
    model_adc = MLPADC(bx=bx_val, bw=bw_val, ba=ba_val, k=k_val)
    optimizer_adc = optim.Adam(model_adc.parameters(), lr=learning_rate)
    
    train_losses_adc, train_accs_adc, test_losses_adc, test_accs_adc = train_model(
        model_adc, optimizer_adc, train_loader, test_loader, criterion, device,
        num_epochs=num_epochs_exp, 
        model_name=f"MLPADC (bx={bx_val}, bw={bw_val}, ba={ba_val}, k={k_val})",
        calib_loader=train_loader # Pass train_loader for calibration phase
    )

    # --- Experiment with MLPQuant (Standard Quantization) ---
    print("\n--- Experiment with MLPQuant (Standard Quantization) ---")
    model_quant = MLPQuant(bx=bx_val, bw=bw_val)
    optimizer_quant = optim.Adam(model_quant.parameters(), lr=learning_rate)

    train_losses_quant, train_accs_quant, test_losses_quant, test_accs_quant = train_model(
        model_quant, optimizer_quant, train_loader, test_loader, criterion, device,
        num_epochs=num_epochs_exp, 
        model_name=f"MLPQuant (bx={bx_val}, bw={bw_val})",
        calib_loader=train_loader # Pass train_loader for calibration phase
    )

    # --- Results Logging to CSV ---
    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'./results/experiment_results_{timestamp}.csv'
    plot_filename = f'./results/loss_curves_{timestamp}.png'
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Model Name', 'Epoch', 
                           'Train Loss', 'Train Accuracy', 
                           'Test Loss', 'Test Accuracy', 
                           'Parameters'])
        
        models_results = {
            "MLP (Baseline)": (train_losses_mlp, train_accs_mlp, test_losses_mlp, test_accs_mlp, "N/A"),
            f"MLPADC (bx={bx_val}, bw={bw_val}, ba={ba_val}, k={k_val})": (train_losses_adc, train_accs_adc, test_losses_adc, test_accs_adc, f"bx={bx_val}, bw={bw_val}, ba={ba_val}, k={k_val}"),
            f"MLPQuant (bx={bx_val}, bw={bw_val})": (train_losses_quant, train_accs_quant, test_losses_quant, test_accs_quant, f"bx={bx_val}, bw={bw_val}")
        }

        for model_name, (train_losses, train_accs, test_losses, test_accs, params_str) in models_results.items():
            if not train_losses: # Handle cases where training might not have run
                csvwriter.writerow([model_name, 'N/A', 'No results', 'No results', 'No results', 'No results', params_str if params_str != "N/A" else "Epochs=" + str(num_epochs_exp)])
                continue
            for epoch in range(num_epochs_exp):
                csvwriter.writerow([
                    model_name,
                    epoch + 1,
                    f'{train_losses[epoch]:.4f}' if epoch < len(train_losses) else 'N/A',
                    f'{train_accs[epoch]:.2f}%' if epoch < len(train_accs) else 'N/A',
                    f'{test_losses[epoch]:.4f}' if epoch < len(test_losses) else 'N/A',
                    f'{test_accs[epoch]:.2f}%' if epoch < len(test_accs) else 'N/A',
                    params_str if params_str != "N/A" else "Epochs=" + str(num_epochs_exp)
                ])
    
    print(f"\nExperiment results saved to: {csv_filename}")

    # --- Plotting Section ---
    epochs_range = range(1, num_epochs_exp + 1)
    
    # Define colors for consistency
    colors = {
        "MLP": "blue",
        "MLPADC": "green",
        "MLPQuant": "red"
    }

    # --- Plot 1: Individual Accuracy Plots ---
    model_data_for_plotting = {
        "MLP (Baseline)": (train_accs_mlp, test_accs_mlp, colors["MLP"], "N/A"),
        f"MLPADC (bx={bx_val},bw={bw_val},ba={ba_val},k={k_val})": (train_accs_adc, test_accs_adc, colors["MLPADC"], f"bx={bx_val},bw={bw_val},ba={ba_val},k={k_val}"),
        f"MLPQuant (bx={bx_val},bw={bw_val})": (train_accs_quant, test_accs_quant, colors["MLPQuant"], f"bx={bx_val},bw={bw_val}")
    }

    for model_name_full, (train_accs, test_accs, color, params_str_short) in model_data_for_plotting.items():
        if train_accs and test_accs:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_range, train_accs, label=f'{model_name_full} Train Accuracy', linestyle='-', marker='o', color=color)
            plt.plot(epochs_range, test_accs, label=f'{model_name_full} Test Accuracy', linestyle='--', color=color)
            plt.title(f'Accuracy vs. Epochs - {model_name_full}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            # Sanitize model_name_full for filename
            safe_model_name = model_name_full.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "")
            individual_plot_filename = f'./results/accuracy_{safe_model_name}_{timestamp}.png'
            plt.savefig(individual_plot_filename)
            print(f"Individual accuracy plot saved to: {individual_plot_filename}")
            plt.close() # Close the figure to free memory

    # --- Plot 2: Combined Accuracy Comparison Plot ---
    plt.figure(figsize=(14, 9))
    
    # MLP Accuracy
    if train_accs_mlp and test_accs_mlp:
        plt.plot(epochs_range, train_accs_mlp, label='MLP Train Accuracy', linestyle='-', marker='o', color=colors["MLP"])
        plt.plot(epochs_range, test_accs_mlp, label='MLP Test Accuracy', linestyle='--', color=colors["MLP"])

    # MLPADC Accuracy
    if train_accs_adc and test_accs_adc:
        plt.plot(epochs_range, train_accs_adc, label=f'MLPADC ({bx_val},{bw_val},{ba_val},{k_val}) Train Accuracy', linestyle='-', marker='s', color=colors["MLPADC"])
        plt.plot(epochs_range, test_accs_adc, label=f'MLPADC ({bx_val},{bw_val},{ba_val},{k_val}) Test Accuracy', linestyle='--', color=colors["MLPADC"])

    # MLPQuant Accuracy
    if train_accs_quant and test_accs_quant:
        plt.plot(epochs_range, train_accs_quant, label=f'MLPQuant ({bx_val},{bw_val}) Train Accuracy', linestyle='-', marker='d', color=colors["MLPQuant"])
        plt.plot(epochs_range, test_accs_quant, label=f'MLPQuant ({bx_val},{bw_val}) Test Accuracy', linestyle='--', color=colors["MLPQuant"])
        
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    comparison_plot_filename = f'./results/accuracy_comparison_all_models_{timestamp}.png'
    plt.savefig(comparison_plot_filename)
    print(f"Combined accuracy plot saved to: {comparison_plot_filename}")
    plt.close()

    # --- Plot 3: Combined Loss Curves (Kept from previous version) ---
    loss_plot_filename = f'./results/loss_curves_{timestamp}.png' # Ensure this filename is defined earlier or use a new one
    plt.figure(figsize=(12, 8))

    # MLP Losses
    if train_losses_mlp and test_losses_mlp:
        plt.plot(epochs_range, train_losses_mlp, label='MLP Train Loss', linestyle='-', marker='o', color=colors["MLP"])
        plt.plot(epochs_range, test_losses_mlp, label='MLP Test Loss', linestyle='--', marker='x', color=colors["MLP"])

    # MLPADC Losses
    if train_losses_adc and test_losses_adc:
        plt.plot(epochs_range, train_losses_adc, label=f'MLPADC ({bx_val},{bw_val},{ba_val},{k_val}) Train Loss', linestyle='-', marker='s', color=colors["MLPADC"])
        plt.plot(epochs_range, test_losses_adc, label=f'MLPADC ({bx_val},{bw_val},{ba_val},{k_val}) Test Loss', linestyle='--', marker='^', color=colors["MLPADC"])

    # MLPQuant Losses
    if train_losses_quant and test_losses_quant:
        plt.plot(epochs_range, train_losses_quant, label=f'MLPQuant ({bx_val},{bw_val}) Train Loss', linestyle='-', marker='d', color=colors["MLPQuant"])
        plt.plot(epochs_range, test_losses_quant, label=f'MLPQuant ({bx_val},{bw_val}) Test Loss', linestyle='--', marker='+', color=colors["MLPQuant"])
        
    plt.title('Model Loss Curves Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_filename)
    print(f"Loss curves plot saved to: {loss_plot_filename}")
    plt.close() # Close the figure
    # plt.show() # Uncomment to display the plot

if __name__ == '__main__':
    run_experiment()
    