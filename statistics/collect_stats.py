import torch
import numpy as np
import os
from tqdm import tqdm
from datasets.cifar_dataset import get_cifar_dataloaders
from model.resnet import resnet18

def collect_activation_stats(model, data_loader, device, save_dir='statistics/activation_stats_small'):
    os.makedirs(save_dir, exist_ok=True)
    activation_maps = {}

    def hook_fn(name, module):
        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return
            out = output.detach().cpu().numpy()
            # Если сверточный слой — сохраняем по каналам
            if isinstance(module, torch.nn.Conv2d):
                # out.shape = (batch, C, H, W)
                trans = out.transpose(1, 0, 2, 3)               # (C, batch, H, W)
                ch_vals = trans.reshape(trans.shape[0], -1)     # (C, batch*H*W)
                activation_maps.setdefault(name, []).append(ch_vals)
            else:
                flat = out.flatten()
                activation_maps.setdefault(name, []).append(flat)
        return hook

    # Навешиваем хуки
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (
            torch.nn.Conv2d,
            torch.nn.Linear,
            torch.nn.ReLU,
            torch.nn.BatchNorm2d,
            torch.nn.MaxPool2d
        )):
            hooks.append(module.register_forward_hook(hook_fn(name, module)))

    # Сбор активаций по первым 69 батчам
    model.to(device).eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(data_loader, desc='Collecting activations')):
            if batch_idx >= 1:
                break
            model(data.to(device))

    # Сохраняем
    for name, chunks in activation_maps.items():
        # Conv2d даёт 2D-фрагменты (C, batch*H*W) → concat по axis=1
        # Остальные — 1D-векторы → concat по axis=0
        axis = 1 if chunks[0].ndim == 2 else 0
        arr = np.concatenate(chunks, axis=axis)
        fname = os.path.join(save_dir, f'{name.replace(".", "_")}.npy')
        np.save(fname, arr)
        print(f'Saved {name}: {arr.shape} → {fname}')

    # Убираем хуки
    for h in hooks:
        h.remove()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load(
        '/home/coder/project/datasets/resnet18.pt',
        map_location=device
    ))

    train_loader, test_loader = get_cifar_dataloaders(batch_size=10)
    # combined = torch.utils.data.ConcatDataset([train_loader.dataset, test_loader.dataset])
    # loader = torch.utils.data.DataLoader(
    #     combined, batch_size=128, shuffle=False, num_workers=2
    # )

    collect_activation_stats(
        model,
        test_loader,
        device,
    )

if __name__ == '__main__':
    main()
