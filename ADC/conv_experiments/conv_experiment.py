import torch
import torch.optim as optim
from torch import nn
import os
from datetime import datetime

from ADC.train_utils import train_model
from ADC.conv_experiments.conv_experiment_setup import get_config, setup_dataloaders, define_experiments, sanitize_filename


def run_single_experiment(exp_config, loaders, criterion, device, num_epochs, lr, results_dir, timestamp):
    """Runs a single experiment and saves the weights."""
    model_name = exp_config['name']
    print(f"\n--- Training {model_name} ---")

    train_loader, test_loader = loaders
    
    model_class = exp_config['model_class']
    model_args = exp_config.get('model_args', {})
    model = model_class(**model_args).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_args = exp_config.get('train_args', {})
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

def main_train():
    """Main function to run all training experiments."""
    config = get_config()
    os.makedirs(config['results_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Using device: {config['device']}")
    print(f"Results will be saved with timestamp: {timestamp}")

    train_loader, test_loader = setup_dataloaders(config['batch_size_train'], config['batch_size_test'])
    criterion = nn.CrossEntropyLoss()
    
    experiments = define_experiments(config)
    all_results = []

    for exp_config in experiments:
        print(f"Running experiment: {exp_config['name']}")
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

    # Save all results for the evaluation script
    results_path = f"{config['results_dir']}/results_{timestamp}.pt"
    torch.save({
        'all_results': all_results,
        'num_epochs': config['num_epochs'],
        'timestamp': timestamp
    }, results_path)
    print(f"\nTraining complete. All results saved to {results_path}")
    print(f"To evaluate, run: python ADC/conv_evaluate.py --timestamp {timestamp}")


if __name__ == '__main__':
    main_train()
    