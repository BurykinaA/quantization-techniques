from trainers.base_trainer import BaseTrainer
from datasets.mnist_dataset import get_mnist_dataloaders
import torch
import torch.nn as nn
from model.resnet import resnet18
from utils.quantization import get_qconfig_for_bitwidth
import os

class MNISTTrainer(BaseTrainer):
    def prepare_data(self):
        self.train_loader, self.test_loader = get_mnist_dataloaders(self.config.batch_size)
    
    def prepare_model(self):
        if self.config.quant_type == 'ptq':
            # For PTQ, use pretrained model
            self.model = resnet18(pretrained=False, num_classes=10)
            
            print("Warning: Using PTQ with MNIST might give suboptimal results since")
            print("we're using an ImageNet pretrained model on grayscale images.")
            
            # Load pretrained weights
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
            
            # First fuse modules in eval mode
            self.model.eval()
            modules_to_fuse = self.model.modules_to_fuse()
            self.model = torch.ao.quantization.fuse_modules(self.model, modules_to_fuse)
            
            # Set qconfig
            self.model.qconfig = get_qconfig_for_bitwidth(self.config.bitwidth)
            
            # Set to train mode BEFORE prepare_qat
            self.model.train()
            
            # Prepare for QAT
            torch.ao.quantization.prepare_qat(self.model, inplace=True)
        
        else:
            # Regular training without quantization
            self.model = resnet18(pretrained=False, num_classes=10)
            self.model = self.model.to(self.device) 