import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging

class QATTrainer:
    """
    Quantization-Aware Training trainer with gradient flow monitoring
    """
    def __init__(self, 
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 gradient_clip_norm: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_gradients(self) -> Dict[str, float]:
        """Check gradient statistics to ensure proper flow"""
        grad_stats = {}
        total_norm = 0.0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_stats[name] = grad_norm
                total_norm += grad_norm ** 2
                param_count += 1
            else:
                grad_stats[name] = 0.0
                self.logger.warning(f"No gradient for parameter: {name}")
        
        grad_stats['total_norm'] = total_norm ** 0.5
        grad_stats['param_count'] = param_count
        
        return grad_stats
    
    def enable_quantization_gradually(self, epoch: int, warmup_epochs: int = 5):
        """Gradually enable quantization to stabilize training"""
        if epoch < warmup_epochs:
            # Disable quantization during warmup
            for module in self.model.modules():
                if hasattr(module, 'disable_quantization'):
                    module.disable_quantization()
        else:
            # Enable quantization after warmup
            for module in self.model.modules():
                if hasattr(module, 'enable_quantization'):
                    module.enable_quantization()
    
    def train_step(self, 
                   batch: Dict[str, torch.Tensor], 
                   criterion: nn.Module) -> Dict[str, float]:
        """Single training step with gradient monitoring"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = criterion(outputs, batch.get('labels', batch.get('targets')))
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_stats = self.check_gradients()
        
        # Clip gradients if needed
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_stats['total_norm'],
            'param_count_with_grad': grad_stats['param_count']
        }
    
    def validate_step(self, 
                     batch: Dict[str, torch.Tensor], 
                     criterion: nn.Module) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = criterion(outputs, batch.get('labels', batch.get('targets')))
        
        return {'loss': loss.item()}
    
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   criterion: nn.Module,
                   epoch: int,
                   warmup_epochs: int = 5) -> Dict[str, float]:
        """Train for one epoch"""
        # Enable quantization gradually
        self.enable_quantization_gradually(epoch, warmup_epochs)
        
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                metrics = self.train_step(batch, criterion)
                total_loss += metrics['loss']
                total_grad_norm += metrics['grad_norm']
                num_batches += 1
                
                # Log progress
                if batch_idx % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}: "
                        f"Loss = {metrics['loss']:.6f}, "
                        f"Grad Norm = {metrics['grad_norm']:.6f}"
                    )
                    
                    # Check for gradient flow issues
                    if metrics['grad_norm'] < 1e-8:
                        self.logger.warning("Very small gradient norm detected!")
                    if metrics['param_count_with_grad'] == 0:
                        self.logger.error("No parameters have gradients!")
                        
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_grad_norm = total_grad_norm / max(num_batches, 1)
        
        return {
            'avg_loss': avg_loss,
            'avg_grad_norm': avg_grad_norm,
            'num_batches': num_batches
        }
    
    def validate_epoch(self, 
                      val_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            try:
                metrics = self.validate_step(batch, criterion)
                total_loss += metrics['loss']
                num_batches += 1
            except Exception as e:
                self.logger.error(f"Error in validation batch: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'avg_loss': avg_loss}

# Example usage function
def create_qat_model_and_trainer(model_config: Dict[str, Any], 
                                device: torch.device) -> tuple:
    """Create QAT model and trainer"""
    from qat_layers import QATTransformerBlock
    
    # Example: Create a simple QAT model
    class SimpleQATModel(nn.Module):
        def __init__(self, d_model=512, num_heads=8, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(model_config.get('vocab_size', 30000), d_model)
            self.layers = nn.ModuleList([
                QATTransformerBlock(d_model, num_heads, d_model * 4)
                for _ in range(num_layers)
            ])
            self.classifier = nn.Linear(d_model, model_config.get('num_classes', 2))
        
        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.classifier(x.mean(dim=1))  # Global average pooling
    
    model = SimpleQATModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    trainer = QATTrainer(model, optimizer, device)
    
    return model, trainer 