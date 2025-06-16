import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

from ADC.models import resnet18_cifar, resnet18_cifar_adc # Assuming R18 is not part of this specific experiment comparison
from ADC.train_utils import train_model

RESULTS_DIR = './results_resnet18' # Define the constant for the results directory


def run_experiment():
    # --- Configuration ---
    num_epochs_exp = 20  # Adjust as needed
    batch_size_train = 512
    batch_size_test = 512
    learning_rate = 0.0003
    #learning_rate = 0.1
    #momentum = 0.9
    #weight_decay = 5e-4

    # Quantization parameters
    bx_val = 8
    bw_val = 8
    ba_val = 8  # For ADC
    k_val = 4   # For ADC
    lambda_k_val = 0.001 # Coefficient for Kurtosis penalty
    ashift_mode = True # For R18ADCAshift experiments

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])])

    train_dataset_cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    test_dataset_cifar = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

    train_loader = DataLoader(train_dataset_cifar, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset_cifar, batch_size=batch_size_test, shuffle=False)
    # For calibration, we can use a portion of the training loader or the full one.
    # Using train_loader itself for calibration in train_model if no specific calib_loader is passed.

    criterion = nn.CrossEntropyLoss()

    # Create results directory if it doesn't exist (moved earlier for weight saving)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Helper function to sanitize model names for filenames
    def sanitize_filename(name):
        return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "").replace("+", "plus")

    # --- Experiment with ResNet18 (ADC) ---
    print("\n--- Experiment with ResNet18 (ADC) ---")
    model_r18_adc = resnet18_cifar_adc(10, bx = bx_val, bw=bw_val, ba=ba_val, k=k_val)
    optimizer_r18_adc = optim.Adam(model_r18_adc.parameters(), lr=learning_rate)
    #optimizer_r18_adc = torch.optim.SGD(model_r18_adc.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    r18_adc_model_name = "ResNet18 (ADC)"

    train_losses_r18_adc, train_accs_r18_adc, test_losses_r18_adc, test_accs_r18_adc = train_model(
        model_r18_adc, optimizer_r18_adc, train_loader, test_loader, criterion, device,
        num_epochs=num_epochs_exp,
        model_name=r18_adc_model_name,
        # No calib_loader needed for baseline R18
    )
    r18_adc_weights_filename = f'{RESULTS_DIR}/model_{sanitize_filename(r18_adc_model_name)}_weights_{timestamp}.pth'
    torch.save(model_r18_adc.state_dict(), r18_adc_weights_filename)
    print(f"Saved {r18_adc_model_name} weights to: {r18_adc_weights_filename}")

    # --- Experiment with ResNet18 (Baseline) ---
    print("\n--- Experiment with ResNet18 (Baseline) ---")
    model_r18 = resnet18_cifar(10)
    optimizer_r18 = optim.Adam(model_r18.parameters(), lr=learning_rate)
    #optimizer_r18 = torch.optim.SGD(model_r18.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    r18_model_name = "ResNet18 (Baseline)"

    train_losses_r18, train_accs_r18, test_losses_r18, test_accs_r18 = train_model(
        model_r18, optimizer_r18, train_loader, test_loader, criterion, device,
        num_epochs=num_epochs_exp,
        model_name=r18_model_name,
        # No calib_loader needed for baseline R18
    )
    r18_weights_filename = f'{RESULTS_DIR}/model_{sanitize_filename(r18_model_name)}_weights_{timestamp}.pth'
    torch.save(model_r18.state_dict(), r18_weights_filename)
    print(f"Saved {r18_model_name} weights to: {r18_weights_filename}")

    # --- Experiment with R18ADC + Weight Reshaping ---
    print("\n--- Experiment with ResNet18(ADC) + Weight Reshaping ---")
    model_w_reshape = resnet18_cifar_adc(10, bx = bx_val, bw=bw_val, ba=ba_val, k=k_val)
    optimizer_w_reshape = optim.Adam(model_w_reshape.parameters(), lr=learning_rate)
    #optimizer_w_reshape = torch.optim.SGD(model_w_reshape.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    adc_wr_model_name = f"ResNet18(ADC)+W-Reshape (bx={bx_val},bw={bw_val},ba={ba_val},k={k_val},lk={lambda_k_val})"

    train_losses_wr, train_accs_wr, test_losses_wr, test_accs_wr = train_model(
        model_w_reshape, optimizer_w_reshape, train_loader, test_loader, criterion, device,
        num_epochs=num_epochs_exp,
        model_name=adc_wr_model_name, 
        calib_loader=train_loader, 
        lambda_kurtosis=lambda_k_val 
    )
    adc_wr_weights_filename = f'{RESULTS_DIR}/model_{sanitize_filename(adc_wr_model_name)}_weights_{timestamp}.pth'
    torch.save(model_w_reshape.state_dict(), adc_wr_weights_filename)
    print(f"Saved {adc_wr_model_name} weights to: {adc_wr_weights_filename}")


    # # --- Experiment with ResNet18 ADC + Ashift ---
    # print("\n--- Experiment with ResNet18 ADC + Ashift ---")
    # model_adc_ashift = resnet18_cifar_adc(10, bx = bx_val, bw=bw_val, ba=ba_val, k=k_val, ashift=True)
    # optimizer_adc_ashift = optim.Adam(model_adc_ashift.parameters(), lr=learning_rate)
    # ashift_model_name = f"ResNet18ADCAshift (ashift={ashift_mode}, bx={bx_val}, bw={bw_val}, ba={ba_val}, k={k_val})"
    
    # train_losses_ashift, train_accs_ashift, test_losses_ashift, test_accs_ashift = train_model(
    #     model_adc_ashift, optimizer_adc_ashift, train_loader, test_loader, criterion, device,
    #     num_epochs=num_epochs_exp, 
    #     model_name=ashift_model_name,
    #     calib_loader=train_loader 
    # )
    # ashift_weights_filename = f'{RESULTS_DIR}/model_{sanitize_filename(ashift_model_name)}_weights_{timestamp}.pth'
    # torch.save(model_adc_ashift.state_dict(), ashift_weights_filename)
    # print(f"Saved {ashift_model_name} weights to: {ashift_weights_filename}")

    # # --- Experiment with ResNet18 ADCAshift + Weight Reshaping ---
    # print("\n--- Experiment with ResNet18 ADCAshift + Weight Reshaping ---")
    # model_adc_ashift_wr = resnet18_cifar_adc(10, bx = bx_val, bw=bw_val, ba=ba_val, k=k_val, ashift=True)
    # optimizer_adc_ashift_wr = optim.Adam(model_adc_ashift_wr.parameters(), lr=learning_rate)
    # ashift_wr_model_name = f"ResNet18 Ashift+W-Reshape (ashift={ashift_mode}, bx={bx_val},bw={bw_val},ba={ba_val},k={k_val},lk={lambda_k_val})"

    # train_losses_ashift_wr, train_accs_ashift_wr, test_losses_ashift_wr, test_accs_ashift_wr = train_model(
    #     model_adc_ashift_wr, optimizer_adc_ashift_wr, train_loader, test_loader, criterion, device,
    #     num_epochs=num_epochs_exp,
    #     model_name=ashift_wr_model_name,
    #     calib_loader=train_loader,
    #     lambda_kurtosis=lambda_k_val 
    # )
    # ashift_wr_weights_filename = f'{RESULTS_DIR}/model_{sanitize_filename(ashift_wr_model_name)}_weights_{timestamp}.pth'
    # torch.save(model_adc_ashift_wr.state_dict(), ashift_wr_weights_filename)
    # print(f"Saved {ashift_wr_model_name} weights to: {ashift_wr_weights_filename}")

    # --- Results Logging to CSV ---
    # Create results directory if it doesn't exist - MOVED EARLIER
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") - MOVED EARLIER
    csv_filename = f'{RESULTS_DIR}/experiment_results_{timestamp}.csv'
    plot_filename = f'{RESULTS_DIR}/loss_curves_{timestamp}.png'
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Model Name', 'Epoch', 
                           'Train Loss', 'Train Accuracy', 
                           'Test Loss', 'Test Accuracy', 
                           'Parameters'])
        
        models_results = {
            r18_model_name: (train_losses_r18, train_accs_r18, test_losses_r18, test_accs_r18, "N/A"),
            r18_adc_model_name: (train_losses_r18_adc, train_accs_r18_adc, test_losses_r18_adc, test_accs_r18_adc, f"bx={bx_val}, bw={bw_val}, ba={ba_val}, k={k_val}"),
            #quant_model_name: (train_losses_quant, train_accs_quant, test_losses_quant, test_accs_quant, f"bx={bx_val}, bw={bw_val}"),
            adc_wr_model_name: (train_losses_wr, train_accs_wr, test_losses_wr, test_accs_wr, f"bx={bx_val},bw={bw_val},ba={ba_val},k={k_val},lk={lambda_k_val}"),
            #ashift_model_name: (train_losses_ashift, train_accs_ashift, test_losses_ashift, test_accs_ashift, f"ashift={ashift_mode}, bx={bx_val}, bw={bw_val}, ba={ba_val}, k={k_val}"),
            #ashift_wr_model_name: (train_losses_ashift_wr, train_accs_ashift_wr, test_losses_ashift_wr, test_accs_ashift_wr, f"ashift={ashift_mode}, bx={bx_val},bw={bw_val},ba={ba_val},k={k_val},lk={lambda_k_val}")
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
        "R18": "blue",
        "R18ADC": "green",
        "R18Quant": "red",
        "R18ADCWReshape": "purple", # Renamed from MLPQuantWReshape
        "R18ADCAshift": "cyan",
        "R18ADCAshiftWReshape": "magenta" 
    }

    # --- Plot 1: Individual Accuracy Plots ---
    model_data_for_plotting = {
        r18_model_name: (train_accs_r18, test_accs_r18, colors["R18"], "N/A"),
        r18_adc_model_name: (train_accs_r18_adc, test_accs_r18_adc, colors["R18ADC"], f"bx={bx_val},bw={bw_val},ba={ba_val},k={k_val}"),
        #quant_model_name: (train_accs_quant, test_accs_quant, colors["R18Quant"], f"bx={bx_val},bw={bw_val}"),
        adc_wr_model_name: (train_accs_wr, test_accs_wr, colors["R18ADCWReshape"], f"bx={bx_val},bw={bw_val},ba={ba_val},k={k_val},lk={lambda_k_val}"),
        #ashift_model_name: (train_accs_ashift, test_accs_ashift, colors["R18ADCAshift"], f"ashift={ashift_mode},bx={bx_val},bw={bw_val},ba={ba_val},k={k_val}"),
        #ashift_wr_model_name: (train_accs_ashift_wr, test_accs_ashift_wr, colors["R18ADCAshiftWReshape"], f"ashift={ashift_mode},bx={bx_val},bw={bw_val},ba={ba_val},k={k_val},lk={lambda_k_val}")
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
            safe_model_name = sanitize_filename(model_name_full) # Use helper
            individual_plot_filename = f'{RESULTS_DIR}/accuracy_{safe_model_name}_{timestamp}.png'
            plt.savefig(individual_plot_filename)
            print(f"Individual accuracy plot saved to: {individual_plot_filename}")
            plt.close() # Close the figure to free memory

    # # --- Plot 2: Combined Accuracy Comparison Plot ---
    # plt.figure(figsize=(14, 9))
    
    # R18 Accuracy
    if train_accs_r18 and test_accs_r18:
        plt.plot(epochs_range, train_accs_r18, label='R18 Train Accuracy', linestyle='-', marker='o', color=colors["R18"])
        plt.plot(epochs_range, test_accs_r18, label='R18 Test Accuracy', linestyle='--', color=colors["R18"])

    # R18ADC Accuracy
    if train_accs_r18_adc and test_accs_r18_adc:
        plt.plot(epochs_range, train_accs_r18_adc, label=f'{r18_adc_model_name} Train Accuracy', linestyle='-', marker='s', color=colors["R18ADC"])
        plt.plot(epochs_range, test_accs_r18_adc, label=f'{r18_adc_model_name} Test Accuracy', linestyle='--', color=colors["R18ADC"])

    # # R18Quant Accuracy
    # if train_accs_quant and test_accs_quant:
    #     plt.plot(epochs_range, train_accs_quant, label=f'{quant_model_name} Train Accuracy', linestyle='-', marker='d', color=colors["R18Quant"])
    #     plt.plot(epochs_range, test_accs_quant, label=f'{quant_model_name} Test Accuracy', linestyle='--', color=colors["R18Quant"])

    # R18ADC + W-Reshape Accuracy
    if train_accs_wr and test_accs_wr:
        plt.plot(epochs_range, train_accs_wr, label=f'{adc_wr_model_name} Train Accuracy', linestyle='-', marker='x', color=colors["R18ADCWReshape"]) 
        plt.plot(epochs_range, test_accs_wr, label=f'{adc_wr_model_name} Test Accuracy', linestyle='--', color=colors["R18ADCWReshape"]) 
        
    # # R18ADCAshift Accuracy
    # if train_accs_ashift and test_accs_ashift:
    #     plt.plot(epochs_range, train_accs_ashift, label=f'{ashift_model_name} Train Acc', linestyle='-', marker='p', color=colors["R18ADCAshift"])
    #     plt.plot(epochs_range, test_accs_ashift, label=f'{ashift_model_name} Test Acc', linestyle='--', color=colors["R18ADCAshift"])

    # # R18ADCAshift + W-Reshape Accuracy
    # if train_accs_ashift_wr and test_accs_ashift_wr:
    #     plt.plot(epochs_range, train_accs_ashift_wr, label=f'{ashift_wr_model_name} Train Acc', linestyle='-', marker='h', color=colors["R18ADCAshiftWReshape"])
    #     plt.plot(epochs_range, test_accs_ashift_wr, label=f'{ashift_wr_model_name} Test Acc', linestyle='--', color=colors["R18ADCAshiftWReshape"])
        
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    comparison_plot_filename = f'{RESULTS_DIR}/accuracy_comparison_all_models_{timestamp}.png'
    plt.savefig(comparison_plot_filename)
    print(f"Combined accuracy plot saved to: {comparison_plot_filename}")
    plt.close()

    # --- Plot 2b: Specific ADC Models Accuracy Comparison Plot ---
    plt.figure(figsize=(14, 9))
    
    # R18ADC Accuracy
    if train_accs_r18_adc and test_accs_r18_adc:
        plt.plot(epochs_range, train_accs_r18_adc, label=f'{r18_adc_model_name} Train Accuracy', linestyle='-', marker='s', color=colors["R18ADC"])
        plt.plot(epochs_range, test_accs_r18_adc, label=f'{r18_adc_model_name} Test Accuracy', linestyle='--', color=colors["R18ADC"])

    # # R18ADCAshift Accuracy
    # if train_accs_ashift and test_accs_ashift:
    #     plt.plot(epochs_range, train_accs_ashift, label=f'{ashift_model_name} Train Acc', linestyle='-', marker='p', color=colors["R18ADCAshift"])
    #     plt.plot(epochs_range, test_accs_ashift, label=f'{ashift_model_name} Test Acc', linestyle='--', color=colors["R18ADCAshift"])

    # # R18ADCAshift + W-Reshape Accuracy
    # if train_accs_ashift_wr and test_accs_ashift_wr:
    #     plt.plot(epochs_range, train_accs_ashift_wr, label=f'{ashift_wr_model_name} Train Acc', linestyle='-', marker='h', color=colors["R18ADCAshiftWReshape"])
    #     plt.plot(epochs_range, test_accs_ashift_wr, label=f'{ashift_wr_model_name} Test Acc', linestyle='--', color=colors["R18ADCAshiftWReshape"])

    # R18ADC + W-Reshape Accuracy
    if train_accs_wr and test_accs_wr: 
        plt.plot(epochs_range, train_accs_wr, label=f'{adc_wr_model_name} Train Accuracy', linestyle='-', marker='x', color=colors["R18ADCWReshape"])
        plt.plot(epochs_range, test_accs_wr, label=f'{adc_wr_model_name} Test Accuracy', linestyle='--', color=colors["R18ADCWReshape"])
        
    plt.title('Specific ADC Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    specific_comparison_plot_filename = f'{RESULTS_DIR}/accuracy_comparison_specific_adc_models_{timestamp}.png'
    plt.savefig(specific_comparison_plot_filename)
    print(f"Specific ADC models accuracy plot saved to: {specific_comparison_plot_filename}")
    plt.close()

    # --- Plot 3: Combined Loss Curves (Kept from previous version) ---
    loss_plot_filename = f'{RESULTS_DIR}/loss_curves_{timestamp}.png' # Ensure this filename is defined earlier or use a new one
    plt.figure(figsize=(12, 8))

    # R18 Losses
    if train_losses_r18 and test_losses_r18:
        plt.plot(epochs_range, train_losses_r18, label='R18 Train Loss', linestyle='-', marker='o', color=colors["R18"])
        plt.plot(epochs_range, test_losses_r18, label='R18 Test Loss', linestyle='--', marker='x', color=colors["R18"])

    # R18ADC Losses
    if train_losses_r18_adc and test_losses_r18_adc:
        plt.plot(epochs_range, train_losses_r18_adc, label=f'{r18_adc_model_name} Train Loss', linestyle='-', marker='s', color=colors["R18ADC"])
        plt.plot(epochs_range, test_losses_r18_adc, label=f'{r18_adc_model_name} Test Loss', linestyle='--', marker='^', color=colors["R18ADC"])

    # # R18Quant Losses
    # if train_losses_quant and test_losses_quant:
    #     plt.plot(epochs_range, train_losses_quant, label=f'{quant_model_name} Train Loss', linestyle='-', marker='d', color=colors["R18Quant"])
    #     plt.plot(epochs_range, test_losses_quant, label=f'{quant_model_name} Test Loss', linestyle='--', marker='+', color=colors["R18Quant"])

    # R18ADC + W-Reshape Losses
    if train_losses_wr and test_losses_wr:
        plt.plot(epochs_range, train_losses_wr, label=f'{adc_wr_model_name} Train Loss', linestyle='-', marker='x', color=colors["R18ADCWReshape"]) 
        plt.plot(epochs_range, test_losses_wr, label=f'{adc_wr_model_name} Test Loss', linestyle='--', marker='1', color=colors["R18ADCWReshape"]) 
        
    # # R18ADCAshift Losses
    # if train_losses_ashift and test_losses_ashift:
    #     plt.plot(epochs_range, train_losses_ashift, label=f'{ashift_model_name} Train Loss', linestyle='-', marker='p', color=colors["R18ADCAshift"])
    #     plt.plot(epochs_range, test_losses_ashift, label=f'{ashift_model_name} Test Loss', linestyle='--', marker='2', color=colors["R18ADCAshift"])

    # # R18ADCAshift + W-Reshape Losses
    # if train_losses_ashift_wr and test_losses_ashift_wr:
    #     plt.plot(epochs_range, train_losses_ashift_wr, label=f'{ashift_wr_model_name} Train Loss', linestyle='-', marker='h', color=colors["R18ADCAshiftWReshape"])
    #     plt.plot(epochs_range, test_losses_ashift_wr, label=f'{ashift_wr_model_name} Test Loss', linestyle='--', marker='3', color=colors["R18ADCAshiftWReshape"])
        
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
    