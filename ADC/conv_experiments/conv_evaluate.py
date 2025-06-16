import torch
import csv
import os
import matplotlib.pyplot as plt
import argparse

from ADC.conv_experiments.conv_experiment_setup import get_config, sanitize_filename

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

def plot_results(all_results, num_epochs, results_dir, timestamp):
    """Generates and saves plots for the experiment results."""
    epochs_range = range(1, num_epochs + 1)
    
    colors = {
        "ResNet18 (Baseline)": "blue",
        "ResNet18 (ADC)": "green",
        "ResNet18(ADC)+W-Reshape": "purple",
        "ResNet18ADCAshift": "cyan",
        "ResNet18 Ashift+W-Reshape": "magenta" 
    }
    markers = {
        "ResNet18 (Baseline)": 'o',
        "ResNet18 (ADC)": 's',
        "ResNet18(ADC)+W-Reshape": 'x',
        "ResNet18ADCAshift": 'p',
        "ResNet18 Ashift+W-Reshape": 'h'
    }

    def get_dict_key(name, dictionary):
        for key in dictionary:
            if key in name:
                return key
        return None

    # --- Plot 1: Individual Accuracy Plots ---
    for result in all_results:
        if result['train_accs'] and result['test_accs']:
            plt.figure(figsize=(10, 6))
            dict_key = get_dict_key(result['name'], colors)
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
            is_adc_model = "ADC" in result['name']
            if p_config['type'] == 'Specific_ADC_Accuracy' and not is_adc_model:
                continue
            
            # Skip baseline for specific ADC comparison plot
            if p_config['type'] == 'Specific_ADC_Accuracy' and 'Baseline' in result['name']:
                continue

            dict_key = get_dict_key(result['name'], colors)
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

def main_evaluate(timestamp):
    """Main function to evaluate a completed experiment."""
    config = get_config()
    results_dir = config['results_dir']
    
    results_path = f"{results_dir}/results_{timestamp}.pt"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}. Please run training first.")
    
    data = torch.load(results_path)
    all_results = data['all_results']
    num_epochs = data['num_epochs']
    
    print(f"Evaluating experiment from timestamp: {timestamp}")

    csv_filename = f"{results_dir}/experiment_results_{timestamp}.csv"
    log_results_to_csv(all_results, num_epochs, csv_filename)

    plot_results(all_results, num_epochs, results_dir, timestamp)
    print("\nEvaluation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ResNet18 Quantization Experiments.')
    parser.add_argument('--timestamp', type=str, required=True, help='Timestamp of the experiment to evaluate (e.g., YYYYMMDD_HHMMSS).')
    args = parser.parse_args()
    
    main_evaluate(args.timestamp) 