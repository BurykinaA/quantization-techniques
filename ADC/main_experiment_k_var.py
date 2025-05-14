import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import itertools # For cycling through colors/markers if needed

from ADC.models import MLPADC, MLPADCAshift # MLP and MLPQuant are not used in this k-variation experiment
from ADC.train_utils import train_model

RESULTS_DIR = './results_k_experiment' # Define the constant for the results directory
K_VALUES = [2, 3, 4] # Define k values to iterate over

def run_k_variation_experiment():
    # --- Configuration ---
    num_epochs_exp = 20  # Adjust as needed, kept same as original for consistency
    batch_size_train = 1024
    batch_size_test = 1024
    learning_rate = 0.001

    # Quantization parameters (bx, bw, ba will be fixed, k will vary)
    bx_val = 8
    bw_val = 8
    ba_val = 8  # For ADC
    lambda_k_val = 0.001 # Coefficient for Kurtosis penalty
    ashift_mode = True # For MLPADCAshift experiments

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
    
    criterion = nn.CrossEntropyLoss()

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Helper function to sanitize model names for filenames
    def sanitize_filename(name):
        return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "").replace("+", "plus")

    # --- Data Storage for Experiments ---
    # Stores results for MLPADCAshift + W-Reshape, keyed by k
    # Each entry: {'train_losses': [], 'train_accs': [], 'test_losses': [], 'test_accs': [], 'model_name': str}
    results_ashift_wr_k_var = {} 
    # Stores results for MLPADC + W-Reshape, keyed by k
    results_adc_wr_k_var = {}

    # --- Experiment Loop over K_VALUES ---
    for k_current in K_VALUES:
        print(f"\n\n--- Starting Experiments for k = {k_current} ---")

        # --- Experiment with MLPADCAshift + Weight Reshaping ---
        print(f"\n--- Experiment with MLPADCAshift + Weight Reshaping (k={k_current}) ---")
        model_adc_ashift_wr = MLPADCAshift(bx=bx_val, bw=bw_val, ba=ba_val, k=k_current, ashift_enabled=ashift_mode)
        optimizer_adc_ashift_wr = optim.Adam(model_adc_ashift_wr.parameters(), lr=learning_rate)
        ashift_wr_model_name = f"MLPADCAshift+W-Reshape (k={k_current}, ashift={ashift_mode}, lk={lambda_k_val})"
        
        train_l, train_a, test_l, test_a = train_model(
            model_adc_ashift_wr, optimizer_adc_ashift_wr, train_loader, test_loader, criterion, device,
            num_epochs=num_epochs_exp,
            model_name=ashift_wr_model_name,
            calib_loader=train_loader,
            lambda_kurtosis=lambda_k_val 
        )
        results_ashift_wr_k_var[k_current] = {
            'train_losses': train_l, 'train_accs': train_a, 
            'test_losses': test_l, 'test_accs': test_a,
            'model_name': ashift_wr_model_name
        }
        weights_filename = f'{RESULTS_DIR}/model_{sanitize_filename(ashift_wr_model_name)}_weights_{timestamp}.pth'
        torch.save(model_adc_ashift_wr.state_dict(), weights_filename)
        print(f"Saved {ashift_wr_model_name} weights to: {weights_filename}")

        # --- Experiment with MLPADC + Weight Reshaping ---
        print(f"\n--- Experiment with MLPADC + Weight Reshaping (k={k_current}) ---")
        model_adc_wr = MLPADC(bx=bx_val, bw=bw_val, ba=ba_val, k=k_current) 
        optimizer_adc_wr = optim.Adam(model_adc_wr.parameters(), lr=learning_rate)
        adc_wr_model_name = f"MLPADC+W-Reshape (k={k_current}, lk={lambda_k_val})"

        train_l, train_a, test_l, test_a = train_model(
            model_adc_wr, optimizer_adc_wr, train_loader, test_loader, criterion, device,
            num_epochs=num_epochs_exp,
            model_name=adc_wr_model_name, 
            calib_loader=train_loader, 
            lambda_kurtosis=lambda_k_val 
        )
        results_adc_wr_k_var[k_current] = {
            'train_losses': train_l, 'train_accs': train_a,
            'test_losses': test_l, 'test_accs': test_a,
            'model_name': adc_wr_model_name
        }
        weights_filename = f'{RESULTS_DIR}/model_{sanitize_filename(adc_wr_model_name)}_weights_{timestamp}.pth'
        torch.save(model_adc_wr.state_dict(), weights_filename)
        print(f"Saved {adc_wr_model_name} weights to: {weights_filename}")

    # --- Results Logging to CSV ---
    csv_filename = f'{RESULTS_DIR}/experiment_k_variation_results_{timestamp}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Model Base Name', 'k Value', 'Epoch', 
                           'Train Loss', 'Train Accuracy', 
                           'Test Loss', 'Test Accuracy', 
                           'Full Model Name', 'Parameters'])
        
        # Logging for MLPADCAshift + W-Reshape
        for k_val, data in results_ashift_wr_k_var.items():
            model_base_name = "MLPADCAshift+W-Reshape"
            params_str = f"bx={bx_val},bw={bw_val},ba={ba_val},k={k_val},lk={lambda_k_val},ashift={ashift_mode}"
            if not data['train_losses']: continue
            for epoch in range(num_epochs_exp):
                csvwriter.writerow([
                    model_base_name, k_val, epoch + 1,
                    f"{data['train_losses'][epoch]:.4f}" if epoch < len(data['train_losses']) else 'N/A',
                    f"{data['train_accs'][epoch]:.2f}%" if epoch < len(data['train_accs']) else 'N/A',
                    f"{data['test_losses'][epoch]:.4f}" if epoch < len(data['test_losses']) else 'N/A',
                    f"{data['test_accs'][epoch]:.2f}%" if epoch < len(data['test_accs']) else 'N/A',
                    data['model_name'], params_str
                ])
        
        # Logging for MLPADC + W-Reshape
        for k_val, data in results_adc_wr_k_var.items():
            model_base_name = "MLPADC+W-Reshape"
            params_str = f"bx={bx_val},bw={bw_val},ba={ba_val},k={k_val},lk={lambda_k_val}"
            if not data['train_losses']: continue
            for epoch in range(num_epochs_exp):
                csvwriter.writerow([
                    model_base_name, k_val, epoch + 1,
                    f"{data['train_losses'][epoch]:.4f}" if epoch < len(data['train_losses']) else 'N/A',
                    f"{data['train_accs'][epoch]:.2f}%" if epoch < len(data['train_accs']) else 'N/A',
                    f"{data['test_losses'][epoch]:.4f}" if epoch < len(data['test_losses']) else 'N/A',
                    f"{data['test_accs'][epoch]:.2f}%" if epoch < len(data['test_accs']) else 'N/A',
                    data['model_name'], params_str
                ])
    print(f"\nK-variation experiment results saved to: {csv_filename}")

    # --- Plotting Section ---
    epochs_range = range(1, num_epochs_exp + 1)
    # Define some colors and markers for plotting different k values
    plot_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    plot_markers = ['o', 's', 'd', 'x', 'p', 'h']
    color_cycle = itertools.cycle(plot_colors)
    marker_cycle = itertools.cycle(plot_markers)

    # Plot 1: MLPADCAshift + W-Reshape Accuracy vs. Epochs (Varying k)
    plt.figure(figsize=(12, 8))
    current_color_cycle = itertools.cycle(plot_colors)
    current_marker_cycle = itertools.cycle(plot_markers)
    for k_val, data in results_ashift_wr_k_var.items():
        if data['train_accs'] and data['test_accs']:
            color = next(current_color_cycle)
            marker = next(current_marker_cycle)
            plt.plot(epochs_range, data['train_accs'], label=f'k={k_val} Train Acc', linestyle='-', marker=marker, color=color)
            plt.plot(epochs_range, data['test_accs'], label=f'k={k_val} Test Acc', linestyle='--', color=color) # Same color, diff linestyle
    
    plt.title(f'MLPADCAshift+W-Reshape Accuracy (ashift={ashift_mode}, lk={lambda_k_val}) vs. k')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plot_filename_ashift_k = f'{RESULTS_DIR}/accuracy_MLPADCAshift_WReshape_k_variation_{timestamp}.png'
    plt.savefig(plot_filename_ashift_k)
    print(f"Plot saved to: {plot_filename_ashift_k}")
    plt.close()

    # Plot 2: MLPADC + W-Reshape Accuracy vs. Epochs (Varying k)
    plt.figure(figsize=(12, 8))
    current_color_cycle = itertools.cycle(plot_colors) # Reset for new plot
    current_marker_cycle = itertools.cycle(plot_markers)
    for k_val, data in results_adc_wr_k_var.items():
        if data['train_accs'] and data['test_accs']:
            color = next(current_color_cycle)
            marker = next(current_marker_cycle)
            plt.plot(epochs_range, data['train_accs'], label=f'k={k_val} Train Acc', linestyle='-', marker=marker, color=color)
            plt.plot(epochs_range, data['test_accs'], label=f'k={k_val} Test Acc', linestyle='--', color=color) # Same color, diff linestyle

    plt.title(f'MLPADC+W-Reshape Accuracy (lk={lambda_k_val}) vs. k')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plot_filename_adc_k = f'{RESULTS_DIR}/accuracy_MLPADC_WReshape_k_variation_{timestamp}.png'
    plt.savefig(plot_filename_adc_k)
    print(f"Plot saved to: {plot_filename_adc_k}")
    plt.close()
    
    # Optional: Plot loss curves if needed, similar to accuracy plots
    # For brevity, only accuracy plots are generated as per primary request.

if __name__ == '__main__':
    run_k_variation_experiment() 