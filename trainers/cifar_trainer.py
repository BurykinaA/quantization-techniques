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
            # PTQ needs to stay on CPU
            self.model = resnet18(pretrained=False, num_classes=10)
            
            # Load pretrained weights from local file
            pretrained_path = os.path.join('datasets', 'resnet18.pt')
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"Pretrained weights not found at {pretrained_path}. "
                    "Please ensure the weights file is in the datasets directory."
                )
            
            state_dict = torch.load(pretrained_path, map_location=self.device)
            
            if 'fc.weight' in state_dict and state_dict['fc.weight'].shape[0] != 10:
                state_dict.pop('fc.weight', None)
                state_dict.pop('fc.bias', None)
            
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights: {msg}")
            
            # PTQ specific preparation
            self.model.eval()
            modules_to_fuse = self.model.modules_to_fuse()
            self.model = torch.ao.quantization.fuse_modules(self.model, modules_to_fuse)
            self.model.qconfig = get_qconfig_for_bitwidth(self.config.bitwidth)
            torch.ao.quantization.prepare(self.model, inplace=True)
            
            self.calibrate_model()
        else:
            # For QAT or no quantization
            self.model = resnet18(pretrained=False, num_classes=10)
            
            # First move to GPU if available
            self.model = self.model.to(self.device)
            
            if self.config.quant_type == 'qat':
                # Prepare for QAT after moving to GPU
                self.model.train()  # Ensure training mode
                modules_to_fuse = self.model.modules_to_fuse()
                self.model = torch.ao.quantization.fuse_modules(self.model, modules_to_fuse)
                self.model.qconfig = get_qconfig_for_bitwidth(self.config.bitwidth)
                torch.ao.quantization.prepare_qat(self.model, inplace=True)
    
    def calibrate_model(self):
        """Calibrate the model with training data for PTQ"""
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_loader):
                if batch_idx > 10:  # Calibrate with ~1000-2000 images
                    break
                self.model(inputs) 