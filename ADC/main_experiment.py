import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

    # --- Results Logging ---
    print("\n\n--- Experiment Results ---")
    print(f"Parameters: bx={bx_val}, bw={bw_val}, ba={ba_val} (ADC), k={k_val} (ADC), Epochs={num_epochs_exp}")

    if test_accs_adc:
        print("\nMLPADC Performance:")
        print(f"  Final Train Accuracy: {train_accs_adc[-1]:.2f}%")
        print(f"  Final Test Accuracy:  {test_accs_adc[-1]:.2f}%")
        print(f"  Final Train Loss:     {train_losses_adc[-1]:.4f}")
        print(f"  Final Test Loss:      {test_losses_adc[-1]:.4f}")
    else:
        print("\nMLPADC Performance: No results recorded.")
        
    if test_accs_quant:
        print("\nMLPQuant (Standard Quantization) Performance:")
        print(f"  Final Train Accuracy: {train_accs_quant[-1]:.2f}%")
        print(f"  Final Test Accuracy:  {test_accs_quant[-1]:.2f}%")
        print(f"  Final Train Loss:     {train_losses_quant[-1]:.4f}")
        print(f"  Final Test Loss:      {test_losses_quant[-1]:.4f}")
    else:
        print("\nMLPQuant Performance: No results recorded.")

if __name__ == '__main__':
    run_experiment()
    