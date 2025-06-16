import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ADC.models import resnet18_cifar, resnet18_cifar_adc


def get_config():
    """Returns a dictionary of experiment configurations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return {
        "results_dir": './results_resnet18',
        "num_epochs": 3,
        "batch_size_train": 512,
        "batch_size_test": 512,
        "learning_rate": 0.0003,
        "quant_params": {
            "bx": 8,
            "bw": 8,
            "ba": 8,
            "k": 4, #do we even have it like this in our ADC?
        },
        "lambda_k_val": 0.001,
        "ashift_mode": True, #now is commented 
        "num_classes": 10,
        "device": device,
    }

def setup_dataloaders(batch_size_train, batch_size_test):
    """Sets up and returns the CIFAR10 dataloaders."""
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "").replace("+", "plus")

def define_experiments(config):
    """Defines all experiments to be run."""
    q_params = config['quant_params']
    bx, bw, ba, k = q_params['bx'], q_params['bw'], q_params['ba'], q_params['k']
    lk = config['lambda_k_val']
    ashift = config['ashift_mode']
    num_classes = config['num_classes']

    # ADC Experiment
    adc_exp = {
        'name': "ResNet18 (ADC)",
        'model_class': resnet18_cifar_adc,
        'model_args': {'num_classes': num_classes, 'bx': bx, 'bw': bw, 'ba': ba, 'k': k},
        'train_args': {}, # No calibration in original script for this specific model
        'params_str': f"bx={bx}, bw={bw}, ba={ba}, k={k}",
    }

    # Baseline Experiment
    baseline_exp = {
        'name': "ResNet18 (Baseline)",
        'model_class': resnet18_cifar,
        'model_args': {'num_classes': num_classes},
        'train_args': {},
        'params_str': "N/A",
    }

    # ADC with Weight Reshaping Experiment
    adc_w_reshape_exp = {
        'name': f"ResNet18(ADC)+W-Reshape (bx={bx},bw={bw},ba={ba},k={k},lk={lk})",
        'model_class': resnet18_cifar_adc,
        'model_args': {'num_classes': num_classes, 'bx': bx, 'bw': bw, 'ba': ba, 'k': k},
        'train_args': {'calib_loader': True, 'lambda_kurtosis': lk},
        'params_str': f"bx={bx},bw={bw},ba={ba},k={k},lk={lk}",
    }
    
    #ADC with Ashift Experiment (currently does not work)
    adc_ashift_exp = {
        'name': f"ResNet18ADCAshift (ashift={ashift}, bx={bx}, bw={bw}, ba={ba}, k={k})",
        'model_class': resnet18_cifar_adc,
        'model_args': {'num_classes': num_classes, 'bx': bx, 'bw': bw, 'ba': ba, 'k': k, 'ashift': True},
        'train_args': {'calib_loader': True},
        'params_str': f"ashift={ashift}, bx={bx}, bw={bw}, ba={ba}, k={k}",
    }

    #ADC with Ashift and Weight Reshaping Experiment (currently does not work)
    adc_ashift_w_reshape_exp = {
        'name': f"ResNet18 Ashift+W-Reshape (ashift={ashift}, bx={bx},bw={bw},ba={ba},k={k},lk={lk})",
        'model_class': resnet18_cifar_adc,
        'model_args': {'num_classes': num_classes, 'bx': bx, 'bw': bw, 'ba': ba, 'k': k, 'ashift': True},
        'train_args': {'calib_loader': True, 'lambda_kurtosis': lk},
        'params_str': f"ashift={ashift}, bx={bx},bw={bw},ba={ba},k={k},lk={lk}",
    }


    experiments = [
        adc_exp,
        baseline_exp,
        adc_w_reshape_exp,
        # adc_ashift_exp,
        # adc_ashift_w_reshape_exp,
    ]
    return experiments 
