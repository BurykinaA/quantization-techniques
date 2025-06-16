import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

from ADC.models import MLP, MLPADC, MLPQuant, MLPADCAshift
from ADC.train_utils import train_model

def get_config():
    """Returns a dictionary of experiment configurations."""
    return {
        "results_dir": './results_4_bit_right',
        "num_epochs": 20,
        "batch_size_train": 1024,
        "batch_size_test": 1024,
        "learning_rate": 0.001,
        "quant_params": {
            "bx": 4,
            "bw": 4,
            "ba": 8,
            "k": 4,
        },
        "lambda_k_val": 0.001,
        "ashift_mode": True,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

def setup_dataloaders(batch_size_train, batch_size_test):
    """Sets up and returns the FashionMNIST dataloaders."""
    transform_fm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)) # FashionMNIST mean/std
    ])

    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform_fm)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform_fm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "").replace("+", "plus")

def run_single_experiment(exp_config, loaders, criterion, device, num_epochs, lr, results_dir, timestamp):
    """Runs a single experiment based on the provided configuration."""
    model_name = exp_config['name']
    print(f"\n--- Experiment with {model_name} ---")

    train_loader, test_loader = loaders
    
    model_class = exp_config['model_class']
    model_args = exp_config.get('model_args', {})
    model = model_class(**model_args).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_args = exp_config.get('train_args', {})
    # Use the actual train_loader if calibration is requested
    if train_args.get('calib_loader') is True:
        train_args['calib_loader'] = train_loader

    train_losses, train_accs, test_losses, test_accs = train_model(
        model, optimizer, train_loader, test_loader, criterion, device,
        num_epochs=num_epochs,
        model_name=model_name,
        **train_args
    )
    
    weights_filename = f'{results_dir}/model_{sanitize_filename(model_name)}_weights_{timestamp}.pth'
    torch.save(model.state_dict(), weights_filename)
    print(f"Saved {model_name} weights to: {weights_filename}")

    return {
        "name": model_name,
        "params_str": exp_config['params_str'],
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
    }

def log_results_to_csv(all_results, num_epochs, csv_filename):
    """Logs the results of all experiments to a single CSV file."""
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Model Name', 'Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Parameters'])
        
        for result in all_results:
            model_name = result['name']
            params_str = result['params_str']
            train_losses = result['train_losses']
            train_accs = result['train_accs']
            test_losses = result['test_losses']
            test_accs = result['test_accs']

            if not train_losses:
                csvwriter.writerow([model_name, 'N/A', 'No results', 'No results', 'No results', 'No results', params_str])
                continue
            
            for epoch in range(num_epochs):
                csvwriter.writerow([
                    model_name,
                    epoch + 1,
                    f'{train_losses[epoch]:.4f}' if epoch < len(train_losses) else 'N/A',
                    f'{train_accs[epoch]:.2f}%' if epoch < len(train_accs) else 'N/A',
                    f'{test_losses[epoch]:.4f}' if epoch < len(test_losses) else 'N/A',
                    f'{test_accs[epoch]:.2f}%' if epoch < len(test_accs) else 'N/A',
                    params_str
                ])
    print(f"\nExperiment results saved to: {csv_filename}")

def define_experiments(config):
    """Defines all experiments to be run."""
    q_params = config['quant_params']
    bx, bw, ba, k = q_params['bx'], q_params['bw'], q_params['ba'], q_params['k']
    lk = config['lambda_k_val']
    ashift = config['ashift_mode']
    num_epochs = config['num_epochs']

    experiments = [
        {
            'name': "MLP (Baseline)",
            'model_class': MLP,
            'train_args': {},
            'params_str': f"Epochs={num_epochs}",
        },
        {
            'name': f"MLPADC (bx={bx}, bw={bw}, ba={ba}, k={k})",
            'model_class': MLPADC,
            'model_args': {'bx': bx, 'bw': bw, 'ba': ba, 'k': k},
            'train_args': {'calib_loader': True},
            'params_str': f"bx={bx}, bw={bw}, ba={ba}, k={k}",
        },
        {
            'name': f"MLPQuant (bx={bx}, bw={bw})",
            'model_class': MLPQuant,
            'model_args': {'bx': bx, 'bw': bw},
            'train_args': {'calib_loader': True},
            'params_str': f"bx={bx}, bw={bw}",
        },
        {
            'name': f"MLPADC+W-Reshape (bx={bx},bw={bw},ba={ba},k={k},lk={lk})",
            'model_class': MLPADC,
            'model_args': {'bx': bx, 'bw': bw, 'ba': ba, 'k': k},
            'train_args': {'calib_loader': True, 'lambda_kurtosis': lk},
            'params_str': f"bx={bx},bw={bw},ba={ba},k={k},lk={lk}",
        },
        {
            'name': f"MLPADCAshift (ashift={ashift}, bx={bx}, bw={bw}, ba={ba}, k={k})",
            'model_class': MLPADCAshift,
            'model_args': {'bx': bx, 'bw': bw, 'ba': ba, 'k': k, 'ashift_enabled': ashift},
            'train_args': {'calib_loader': True},
            'params_str': f"ashift={ashift}, bx={bx}, bw={bw}, ba={ba}, k={k}",
        },
        {
            'name': f"MLPADCAshift+W-Reshape (ashift={ashift}, bx={bx},bw={bw},ba={ba},k={k},lk={lk})",
            'model_class': MLPADCAshift,
            'model_args': {'bx': bx, 'bw': bw, 'ba': ba, 'k': k, 'ashift_enabled': ashift},
            'train_args': {'calib_loader': True, 'lambda_kurtosis': lk},
            'params_str': f"ashift={ashift}, bx={bx},bw={bw},ba={ba},k={k},lk={lk}",
        }
    ]
    return experiments

def plot_results(all_results, num_epochs, results_dir, timestamp):
    """Generates and saves plots for the experiment results."""
    epochs_range = range(1, num_epochs + 1)
    
    colors = {
        "MLP (Baseline)": "blue",
        "MLPADC": "green",
        "MLPQuant": "red",
        "MLPADC+W-Reshape": "purple",
        "MLPADCAshift": "cyan",
        "MLPADCAshift+W-Reshape": "magenta" 
    }
    markers = {
        "MLP (Baseline)": 'o',
        "MLPADC": 's',
        "MLPQuant": 'd',
        "MLPADC+W-Reshape": 'x',
        "MLPADCAshift": 'p',
        "MLPADCAshift+W-Reshape": 'h'
    }

    # Helper to find the correct key for colors/markers dict
    def get_dict_key(name):
        for key in colors:
            if key in name:
                return key
        return None

    # --- Plot 1: Individual Accuracy Plots ---
    for result in all_results:
        if result['train_accs'] and result['test_accs']:
            plt.figure(figsize=(10, 6))
            dict_key = get_dict_key(result['name'])
            color = colors.get(dict_key, 'gray')
            plt.plot(epochs_range, result['train_accs'], label=f'{result["name"]} Train Accuracy', linestyle='-', marker='o', color=color)
            plt.plot(epochs_range, result['test_accs'], label=f'{result["name"]} Test Accuracy', linestyle='--', color=color)
            plt.title(f'Accuracy vs. Epochs - {result["name"]}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            individual_plot_filename = f'{results_dir}/accuracy_{sanitize_filename(result["name"])}_{timestamp}.png'
            plt.savefig(individual_plot_filename)
            print(f"Individual accuracy plot saved to: {individual_plot_filename}")
            plt.close()

    # --- Plot 2: Combined Plots (Accuracy, Loss, Specific Accuracy) ---
    plot_configs = [
        {'type': 'Accuracy', 'y_label': 'Accuracy (%)', 'title': 'Model Accuracy Comparison'},
        {'type': 'Loss', 'y_label': 'Loss', 'title': 'Model Loss Curves Comparison'},
        {'type': 'Specific_ADC_Accuracy', 'y_label': 'Accuracy (%)', 'title': 'Specific ADC Model Accuracy Comparison'}
    ]

    for p_config in plot_configs:
        plt.figure(figsize=(14, 9))
        
        for result in all_results:
            is_adc_model = "MLPADC" in result['name'] or "MLPADCAshift" in result['name']
            if p_config['type'] == 'Specific_ADC_Accuracy' and not is_adc_model:
                continue

            dict_key = get_dict_key(result['name'])
            color = colors.get(dict_key, 'gray')
            marker = markers.get(dict_key, '*')

            if p_config['type'].endswith('Accuracy'):
                train_data, test_data = result['train_accs'], result['test_accs']
                train_label, test_label = 'Train Acc', 'Test Acc'
            else: # Loss
                train_data, test_data = result['train_losses'], result['test_losses']
                train_label, test_label = 'Train Loss', 'Test Loss'
            
            if train_data and test_data:
                plt.plot(epochs_range, train_data, label=f'{result["name"]} {train_label}', linestyle='-', marker=marker, color=color)
                plt.plot(epochs_range, test_data, label=f'{result["name"]} {test_label}', linestyle='--', color=color)

        plt.title(p_config['title'])
        plt.xlabel('Epoch')
        plt.ylabel(p_config['y_label'])
        plt.legend(loc='lower right' if 'Specific' in p_config['type'] else 'best')
        plt.grid(True)
        filename = f'{results_dir}/{p_config["title"].lower().replace(" ", "_")}_{timestamp}.png'
        plt.savefig(filename)
        print(f"Plot saved to: {filename}")
        plt.close()


def run_experiment():
    """Main function to run all experiments."""
    config = get_config()
    os.makedirs(config['results_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Using device: {config['device']}")

    train_loader, test_loader = setup_dataloaders(config['batch_size_train'], config['batch_size_test'])
    criterion = nn.CrossEntropyLoss()
    
    experiments = define_experiments(config)
    all_results = []

    for exp_config in experiments:
        result = run_single_experiment(
            exp_config=exp_config,
            loaders=(train_loader, test_loader),
            criterion=criterion,
            device=config['device'],
            num_epochs=config['num_epochs'],
            lr=config['learning_rate'],
            results_dir=config['results_dir'],
            timestamp=timestamp
        )
        all_results.append(result)

    csv_filename = f"{config['results_dir']}/experiment_results_{timestamp}.csv"
    log_results_to_csv(all_results, config['num_epochs'], csv_filename)

    plot_results(all_results, config['num_epochs'], config['results_dir'], timestamp)


if __name__ == '__main__':
    run_experiment()
    