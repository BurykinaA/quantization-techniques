from trainers.base_trainer import BaseTrainer
from datasets.mnist_dataset import get_mnist_dataloaders
import torch
import torch.nn as nn
from model.resnet import resnet18
from utils.quantization import get_qconfig_for_bitwidth

class MNISTTrainer(BaseTrainer):
    def prepare_data(self):
        self.train_loader, self.test_loader = get_mnist_dataloaders(self.config.batch_size)
    
    def prepare_model(self):
        if self.config.quant_type == 'ptq':
            # For PTQ, always use pretrained model
            self.model = resnet18(pretrained=True, num_classes=1000)
            # Modify the final layer for MNIST
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 10)
            
            print("Warning: Using PTQ with MNIST might give suboptimal results since")
            print("we're using an ImageNet pretrained model on grayscale images.")
            
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
            
            # First move to GPU if available
            self.model = self.model.to(self.device)
            
            if self.config.quant_type == 'qat':
                # Set to eval mode for fusion
                self.model.eval()
                modules_to_fuse = self.model.modules_to_fuse()
                self.model = torch.ao.quantization.fuse_modules(self.model, modules_to_fuse)
                self.model.qconfig = get_qconfig_for_bitwidth(self.config.bitwidth)
                # Prepare for QAT
                torch.ao.quantization.prepare_qat(self.model, inplace=True)
                # Set back to training mode
                self.model.train()
        
        if self.config.quant_type != 'ptq':
            self.model = self.model.to(self.device) 