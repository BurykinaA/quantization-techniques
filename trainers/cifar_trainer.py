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
            # PTQ needs to stay on CPU and use pretrained weights
            self.model = resnet18(pretrained=False, num_classes=10)
            
            # Load pretrained weights from local file
            pretrained_path = os.path.join('datasets', 'resnet18.pt')
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(
                    f"Pretrained weights not found at {pretrained_path}. "
                    "Please ensure the weights file is in the datasets directory."
                )
            
            # Load and adjust weights
            state_dict = torch.load(pretrained_path, map_location=self.device)
            if 'conv1.weight' in state_dict:
                self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            # Load weights
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights for PTQ: {msg}")
            
            # PTQ preparation
            self.model.eval()
            modules_to_fuse = self.model.modules_to_fuse()
            self.model = torch.ao.quantization.fuse_modules(self.model, modules_to_fuse)
            self.model.qconfig = get_qconfig_for_bitwidth(self.config.bitwidth)
            torch.ao.quantization.prepare(self.model, inplace=True)
            self.calibrate_model()
        
        elif self.config.quant_type == 'qat':
            # For QAT, start with fresh model
            self.model = resnet18(pretrained=False, num_classes=10)
            self.model = self.model.to(self.device)
            
            # Prepare for QAT
            self.model.eval()  # Temporarily set to eval for fusion
            modules_to_fuse = self.model.modules_to_fuse()
            self.model = torch.ao.quantization.fuse_modules(self.model, modules_to_fuse)
            self.model.qconfig = get_qconfig_for_bitwidth(self.config.bitwidth)
            torch.ao.quantization.prepare_qat(self.model, inplace=True)
            self.model.train()  # Set back to training mode
        
        else:
            # Regular training without quantization
            self.model = resnet18(pretrained=False, num_classes=10)
            self.model = self.model.to(self.device)
    
    def calibrate_model(self):
        """Calibrate the model with training data for PTQ"""
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_loader):
                if batch_idx > 10:  # Calibrate with ~1000-2000 images
                    break
                self.model(inputs) 