import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import os
from datetime import datetime

from ADC.models import MLPADC, MLPQuant # Assuming MLP is not part of this specific experiment comparison
from ADC.train_utils import train_model

def run_experiment():
    # --- Configuration ---
    num_epochs_exp = 5  # Adjust as needed
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.001

    # Quantization parameters
    bx_val = 4
    bw_val = 4
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
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Model', 'Parameters', 'Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss'])
        
        # Write MLPADC results
        if test_accs_adc:
            csvwriter.writerow([
                'MLPADC', 
                f'bx={bx_val}, bw={bw_val}, ba={ba_val}, k={k_val}, Epochs={num_epochs_exp}',
                f'{train_accs_adc[-1]:.2f}%',
                f'{test_accs_adc[-1]:.2f}%',
                f'{train_losses_adc[-1]:.4f}',
                f'{test_losses_adc[-1]:.4f}'
            ])
        else:
            csvwriter.writerow(['MLPADC', f'bx={bx_val}, bw={bw_val}, ba={ba_val}, k={k_val}, Epochs={num_epochs_exp}', 'No results', 'No results', 'No results', 'No results'])
        
        # Write MLPQuant results
        if test_accs_quant:
            csvwriter.writerow([
                'MLPQuant',
                f'bx={bx_val}, bw={bw_val}, Epochs={num_epochs_exp}',
                f'{train_accs_quant[-1]:.2f}%',
                f'{test_losses_quant[-1]:.4f}',
                f'{test_accs_quant[-1]:.2f}%',
                f'{train_losses_quant[-1]:.4f}'
            ])
        else:
            csvwriter.writerow(['MLPQuant', f'bx={bx_val}, bw={bw_val}, Epochs={num_epochs_exp}', 'No results', 'No results', 'No results', 'No results'])
    
    print(f"\nExperiment results saved to: {csv_filename}")

if __name__ == '__main__':
    run_experiment()
    