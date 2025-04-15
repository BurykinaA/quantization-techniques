import torch
import torch.nn as nn
import wandb
from utils.quantization import get_qconfig_for_bitwidth, get_device
from model.resnet import resnet18
from utils.stats_collector import StatsCollector

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self.init_wandb()
        self.stats_collector = None
    
    def init_wandb(self):
        wandb.init(
            project="quantization-experiments",
            config={
                "dataset": self.config.dataset,
                "bitwidth": self.config.bitwidth,
                "quantization_type": self.config.quant_type,
                "num_epochs": self.config.epochs,
                "learning_rate": self.config.lr,
                "batch_size": self.config.batch_size
            }
        )
    
    def prepare_model(self):
        raise NotImplementedError
        
    def prepare_data(self):
        raise NotImplementedError
        
    def calibrate_model(self):
        """Calibrate the model with training data for PTQ"""
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_loader):
                if batch_idx > 10:  # Calibrate with ~1000-2000 images
                    break
                self.model(inputs)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create new stats collector for this epoch
        self.stats_collector = StatsCollector(self.model)
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            if self.config.quant_type != 'ptq':
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            if self.config.quant_type != 'ptq':
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update stats collector
            self.stats_collector.increment_batch()
            
            if batch_idx % 100 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_accuracy": 100. * correct / total
                }, step=step)
                
                # Log activation and weight statistics
                self.stats_collector.log_stats(step, prefix='train/')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc
        })
        
        # Clean up hooks
        if self.stats_collector:
            self.stats_collector.remove_hooks()
        
        return epoch_loss, epoch_acc

    def evaluate(self):
        self.model.eval()
        if self.config.quant_type in ['qat', 'ptq']:
            self.model.apply(torch.ao.quantization.fake_quantize.disable_observer)
            self.model = torch.ao.quantization.convert(self.model)
        
        correct = 0
        total = 0
        
        # Create new stats collector for evaluation
        self.stats_collector = StatsCollector(self.model)
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                if self.config.quant_type not in ['ptq', 'qat']:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update stats collector
                self.stats_collector.increment_batch()
        
        accuracy = 100. * correct / total
        
        # Log test metrics and statistics
        wandb.log({
            "test_accuracy": accuracy,
            "test_total": total,
            "test_correct": correct
        })
        
        # Log final activation and weight statistics
        self.stats_collector.log_stats(0, prefix='test/')
        
        # Clean up hooks
        if self.stats_collector:
            self.stats_collector.remove_hooks()
        
        return accuracy

    def train(self):
        self.prepare_data()
        self.prepare_model()
        
        if self.config.quant_type != 'ptq':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9)
            self.criterion = nn.CrossEntropyLoss()
            
            for epoch in range(self.config.epochs):
                loss, acc = self.train_epoch(epoch)
                print(f"Epoch {epoch+1}/{self.config.epochs} - Loss: {loss:.4f}, Acc: {acc:.2f}%")
        
        test_acc = self.evaluate()
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        wandb.finish()
        return test_acc 