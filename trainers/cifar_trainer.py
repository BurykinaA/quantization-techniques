from trainers.base_trainer import BaseTrainer
from datasets.cifar_dataset import get_cifar_dataloaders
import torch
import torch.nn as nn
from model.resnet import resnet18
from utils.quantization import get_qconfig_for_bitwidth
import os

class CIFARTrainer(BaseTrainer):
    def prepare_data(self):
        self.train_loader, self.test_loader = get_cifar_dataloaders(self.config.batch_size)
    
    def prepare_model(self):
        if self.config.quant_type == 'ptq':
            # Initialize model with 10 classes directly
            self.model = resnet18(pretrained=False, num_classes=10)
            
            # Load pretrained weights from local file
            pretrained_path = os.path.join('datasets', 'resnet18.pt')
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"Pretrained weights not found at {pretrained_path}. "
                    "Please ensure the weights file is in the datasets directory."
                )
            
            # Load the state dict and handle any key mismatches
            state_dict = torch.load(pretrained_path, map_location=self.device)
            
            # If the state dict has 'fc.weight' with different shape, remove fc layer keys
            if 'fc.weight' in state_dict and state_dict['fc.weight'].shape[0] != 10:
                # Remove fc layer from state dict as it's for 1000 classes
                state_dict.pop('fc.weight', None)
                state_dict.pop('fc.bias', None)
            
            # Load the state dict, ignoring missing keys for fc layer
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights: {msg}")
            
            # Fuse modules and prepare for quantization
            self.model.eval()
            modules_to_fuse = self.model.modules_to_fuse()
            self.model = torch.ao.quantization.fuse_modules(self.model, modules_to_fuse)
            self.model.qconfig = get_qconfig_for_bitwidth(self.config.bitwidth)
            torch.ao.quantization.prepare(self.model, inplace=True)
            
            # Calibrate the model
            self.calibrate_model()
        else:
            # For QAT or no quantization, start from scratch
            self.model = resnet18(pretrained=False, num_classes=10)
            
            if self.config.quant_type == 'qat':
                self.model.eval()
                modules_to_fuse = self.model.modules_to_fuse()
                self.model = torch.ao.quantization.fuse_modules(self.model, modules_to_fuse)
                self.model.qconfig = get_qconfig_for_bitwidth(self.config.bitwidth)
                self.model.train()
                torch.ao.quantization.prepare_qat(self.model, inplace=True)
        
        if self.config.quant_type != 'ptq':
            self.model = self.model.to(self.device)
    
    def calibrate_model(self):
        """Calibrate the model with training data for PTQ"""
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_loader):
                if batch_idx > 10:  # Calibrate with ~1000-2000 images
                    break
                self.model(inputs) 