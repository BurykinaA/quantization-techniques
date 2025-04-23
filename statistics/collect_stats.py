import torch
import numpy as np
import os
from tqdm import tqdm
from datasets.cifar_dataset import get_cifar_dataloaders
from model.resnet import resnet18

def collect_activation_stats(model, data_loader, device, save_dir='activation_stats'):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Dictionary to store activation hooks
    activation_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if name not in activation_maps:
                    activation_maps[name] = []
                # Convert to numpy and flatten
                out_np = output.detach().cpu().numpy().flatten()
                activation_maps[name].append(out_np)
        return hook
    
    # Register hooks for all layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, 
                             torch.nn.BatchNorm2d, torch.nn.MaxPool2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run model on dataset
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(data_loader)):
            if batch_idx >= 100:  # Limit to 100 batches for memory efficiency
                break
            data = data.to(device)
            model(data)
    
    # Save statistics for each layer
    for name, activations in activation_maps.items():
        # Concatenate all batches
        all_activations = np.concatenate(activations)
        # Save to file
        np.save(os.path.join(save_dir, f'{name.replace(".", "_")}.npy'), all_activations)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

def main():
    # Load model and dataset
    model = resnet18(num_classes=10)
    # Load your trained weights here
    model.load_state_dict(torch.load('/home/alina/git-projects/quantization-techniques/datasets/resnet18.pt'))
    
    train_loader, test_loader = get_cifar_dataloaders(batch_size=128)
    
    # Combine train and test loaders into a single loader
    combined_loader = torch.utils.data.DataLoader(
        train_loader.dataset + test_loader.dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Collect statistics using the combined loader
    collect_activation_stats(model, combined_loader, device, save_dir='activation_stats/combined')

if __name__ == '__main__':
    main()
