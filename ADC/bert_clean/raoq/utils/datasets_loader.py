import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os


_DATASETS_ROOT_PATH = '/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization'
_dataset_path = {
    'cifar10': os.path.join(_DATASETS_ROOT_PATH, 'CIFAR10'),
    'imagenet': {
        'train': os.path.join(_DATASETS_ROOT_PATH, 'train'),
        'val': os.path.join(_DATASETS_ROOT_PATH, 'val')
    }
}

__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
__cifar_stats = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}

def transform_test(name, input_size, normalize=None):
    if name == 'imagenet':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])

    elif name == 'cifar10':
        return transforms.Compose([
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])


def transform_train(name, input_size, scale_size=None, normalize=None):
    if name == 'imagenet':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])
    elif name == 'cifar10':    
        padding = int((scale_size - input_size) / 2)

        return transforms.Compose([
            transforms.RandomCrop(input_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)
        ])


def get_transform(name='imagenet', train=True):
    if name == 'imagenet':
        if train:
            return transform_train('imagenet', None, normalize=__imagenet_stats)
        else:
            return transform_test('imagenet', None, normalize=__imagenet_stats)
    elif 'cifar' in name:
        if train:
            return transform_train('cifar10', 32, scale_size=40, normalize=__cifar_stats)
        else:
            return transform_test('cifar10', 32, normalize=__cifar_stats)


def get_dataset(name, train, transform=None, target_transform=None, download=True):
    if name == 'cifar10':
        return datasets.CIFAR10(root=_dataset_path['cifar10'],
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'imagenet':
        train = 'train' if train else 'val'
        path = _dataset_path[name][train]
        return datasets.ImageFolder(root=path,
                                    transform=transform,
                                    target_transform=target_transform)


def build_dataset(dataset, batch_size, workers, distributed):
    # load data
    train_data = get_dataset(dataset, True, get_transform(dataset, True))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data,
                    batch_size=batch_size, shuffle=(train_sampler is None),
                    num_workers=workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)
    
    test_data = get_dataset(dataset, False, get_transform(dataset, False))
    test_loader = torch.utils.data.DataLoader(test_data,
                    batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True, persistent_workers=True)

    return train_loader, train_sampler, test_loader


