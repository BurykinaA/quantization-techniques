import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoConfig, DataCollatorWithPadding, Trainer, TrainingArguments
)
from datasets import load_dataset
from qat_layers import QATLinear, LearnableQuantizer
from qat_training import QATTrainer
import numpy as np
from typing import Dict, Any, Optional
import logging
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertQATConverter:
    """Convert BERT model to use QAT layers"""
    
    @staticmethod
    def replace_linear_with_qat(model: nn.Module, 
                               weight_bits: int = 8, 
                               activation_bits: int = 8,
                               exclude_patterns: list = None):
        """
        Replace all Linear layers in BERT with QAT versions
        
        Args:
            model: BERT model to convert
            weight_bits: Bits for weight quantization
            activation_bits: Bits for activation quantization
            exclude_patterns: List of layer name patterns to exclude from quantization
        """
        if exclude_patterns is None:
            exclude_patterns = ['classifier', 'pooler', 'embeddings']
        
        def should_exclude(name):
            return any(pattern in name for pattern in exclude_patterns)
        
        def replace_recursive(module, name=""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child_module, nn.Linear) and not should_exclude(full_name):
                    # Replace with QAT version
                    qat_layer = QATLinear(
                        child_module.in_features,
                        child_module.out_features,
                        bias=(child_module.bias is not None),
                        weight_bits=weight_bits,
                        activation_bits=activation_bits
                    )
                    
                    # Copy weights and bias
                    with torch.no_grad():
                        qat_layer.weight.copy_(child_module.weight)
                        if child_module.bias is not None:
                            qat_layer.bias.copy_(child_module.bias)
                    
                    # Replace the module
                    setattr(module, child_name, qat_layer)
                    logger.info(f"Replaced {full_name} with QAT version")
                else:
                    # Recursively process child modules
                    replace_recursive(child_module, full_name)
        
        replace_recursive(model)
        return model
    
    @staticmethod
    def count_qat_layers(model: nn.Module) -> Dict[str, int]:
        """Count QAT layers in the model"""
        counts = {'qat_linear': 0, 'regular_linear': 0, 'total_params': 0}
        
        for name, module in model.named_modules():
            if isinstance(module, QATLinear):
                counts['qat_linear'] += 1
            elif isinstance(module, nn.Linear):
                counts['regular_linear'] += 1
            
            if hasattr(module, 'parameters'):
                counts['total_params'] += sum(p.numel() for p in module.parameters())
        
        return counts

class BertQATExperiment:
    """Complete BERT QAT experiment runner"""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 num_labels: int = 2,
                 weight_bits: int = 8,
                 activation_bits: int = 8,
                 device: str = None):
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing BERT QAT experiment with {model_name}")
        logger.info(f"Weight bits: {weight_bits}, Activation bits: {activation_bits}")
        logger.info(f"Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._load_and_convert_model()
        
    def _load_and_convert_model(self):
        """Load BERT model and convert to QAT"""
        logger.info("Loading pre-trained BERT model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        
        logger.info("Converting to QAT model...")
        model = BertQATConverter.replace_linear_with_qat(
            model, 
            weight_bits=self.weight_bits,
            activation_bits=self.activation_bits
        )
        
        # Print conversion stats
        stats = BertQATConverter.count_qat_layers(model)
        logger.info(f"Conversion complete: {stats['qat_linear']} QAT layers, "
                   f"{stats['regular_linear']} regular layers, "
                   f"{stats['total_params']:,} total parameters")
        
        return model.to(self.device)
    
    def prepare_dataset(self, dataset_name: str = "imdb", max_length: int = 512, num_samples: int = None):
        """Prepare dataset for training"""
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "imdb":
            dataset = load_dataset("imdb")
            text_column = "text"
            label_column = "label"
        elif dataset_name == "sst2":
            dataset = load_dataset("glue", "sst2")
            text_column = "sentence"
            label_column = "label"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Limit samples if specified
        if num_samples:
            dataset['train'] = dataset['train'].select(range(min(num_samples, len(dataset['train']))))
            dataset['test'] = dataset['test'].select(range(min(num_samples // 4, len(dataset['test']))))
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            )
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Prepare for PyTorch
        tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        logger.info(f"Dataset ready: {len(tokenized_dataset['train'])} train, {len(tokenized_dataset['test'])} test")
        return tokenized_dataset
    
    def create_data_loaders(self, dataset, batch_size: int = 16):
        """Create data loaders"""
        train_loader = DataLoader(
            dataset['train'], 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=DataCollatorWithPadding(self.tokenizer)
        )
        
        eval_loader = DataLoader(
            dataset['test'], 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=DataCollatorWithPadding(self.tokenizer)
        )
        
        return train_loader, eval_loader
    
    def run_qat_training(self, 
                        dataset,
                        num_epochs: int = 3,
                        learning_rate: float = 2e-5,
                        batch_size: int = 16,
                        warmup_epochs: int = 1,
                        save_model: bool = True):
        """Run QAT training using our custom trainer"""
        logger.info("Starting QAT training...")
        
        # Create data loaders
        train_loader, eval_loader = self.create_data_loaders(dataset, batch_size)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Create QAT trainer
        trainer = QATTrainer(
            model=self.model,
            optimizer=optimizer,
            device=self.device,
            gradient_clip_norm=1.0
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_accuracy = 0.0
        training_history = []
        
        for epoch in range(num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # Train
            train_metrics = trainer.train_epoch(
                train_loader, 
                criterion, 
                epoch, 
                warmup_epochs=warmup_epochs
            )
            
            # Evaluate
            eval_metrics = self.evaluate_model(eval_loader)
            
            # Log results
            logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}, "
                       f"Grad Norm: {train_metrics['avg_grad_norm']:.6f}")
            logger.info(f"Eval Loss: {eval_metrics['loss']:.4f}, "
                       f"Accuracy: {eval_metrics['accuracy']:.4f}")
            
            # Save best model
            if eval_metrics['accuracy'] > best_accuracy:
                best_accuracy = eval_metrics['accuracy']
                if save_model:
                    self.save_model(f"best_bert_qat_{self.weight_bits}bit.pth")
                    logger.info(f"Saved best model with accuracy: {best_accuracy:.4f}")
            
            # Track history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['avg_loss'],
                'eval_loss': eval_metrics['loss'],
                'eval_accuracy': eval_metrics['accuracy'],
                'grad_norm': train_metrics['avg_grad_norm']
            })
        
        logger.info(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")
        return training_history
    
    def evaluate_model(self, eval_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = criterion(outputs.logits, batch['labels'])
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += batch['labels'].size(0)
                total_loss += loss.item()
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(eval_loader)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def save_model(self, path: str):
        """Save the QAT model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'weight_bits': self.weight_bits,
            'activation_bits': self.activation_bits,
            'num_labels': self.num_labels
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved QAT model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
    
    def compare_with_baseline(self, dataset, batch_size: int = 16):
        """Compare QAT model with FP32 baseline"""
        logger.info("Loading FP32 baseline model for comparison...")
        
        # Load baseline model
        baseline_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        ).to(self.device)
        
        # Create data loader
        _, eval_loader = self.create_data_loaders(dataset, batch_size)
        
        # Evaluate QAT model
        self.model.eval()
        qat_metrics = self.evaluate_model(eval_loader)
        
        # Evaluate baseline
        baseline_model.eval()
        baseline_metrics = self._evaluate_baseline(baseline_model, eval_loader)
        
        # Calculate model sizes (approximate)
        qat_size = sum(p.numel() * self.weight_bits / 8 for p in self.model.parameters()) / (1024**2)  # MB
        baseline_size = sum(p.numel() * 32 / 8 for p in baseline_model.parameters()) / (1024**2)  # MB
        
        logger.info("\n=== QAT vs Baseline Comparison ===")
        logger.info(f"QAT Model     - Loss: {qat_metrics['loss']:.4f}, Accuracy: {qat_metrics['accuracy']:.4f}")
        logger.info(f"Baseline (FP32) - Loss: {baseline_metrics['loss']:.4f}, Accuracy: {baseline_metrics['accuracy']:.4f}")
        logger.info(f"Accuracy Drop: {baseline_metrics['accuracy'] - qat_metrics['accuracy']:.4f}")
        logger.info(f"Model Size - QAT: {qat_size:.1f}MB, Baseline: {baseline_size:.1f}MB")
        logger.info(f"Size Reduction: {(1 - qat_size/baseline_size)*100:.1f}%")
        
        return {
            'qat': qat_metrics,
            'baseline': baseline_metrics,
            'size_reduction': (1 - qat_size/baseline_size)*100
        }
    
    def _evaluate_baseline(self, model, eval_loader):
        """Evaluate baseline model"""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = criterion(outputs.logits, batch['labels'])
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += batch['labels'].size(0)
                total_loss += loss.item()
        
        return {
            'loss': total_loss / len(eval_loader),
            'accuracy': correct_predictions / total_predictions
        }

def run_complete_bert_qat_experiment():
    """Run a complete BERT QAT experiment"""
    logger.info("🚀 Starting Complete BERT QAT Experiment")
    
    # Configuration
    config = {
        'model_name': 'bert-base-uncased',
        'dataset': 'sst2',  # or 'imdb'
        'num_labels': 2,
        'weight_bits': 8,
        'activation_bits': 8,
        'num_epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'max_length': 128,
        'num_samples': 5000,  # Limit for faster experimentation
    }
    
    # Initialize experiment
    experiment = BertQATExperiment(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        weight_bits=config['weight_bits'],
        activation_bits=config['activation_bits']
    )
    
    # Prepare dataset
    dataset = experiment.prepare_dataset(
        dataset_name=config['dataset'],
        max_length=config['max_length'],
        num_samples=config['num_samples']
    )
    
    # Run QAT training
    history = experiment.run_qat_training(
        dataset=dataset,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        warmup_epochs=1
    )
    
    # Compare with baseline
    comparison = experiment.compare_with_baseline(dataset, config['batch_size'])
    
    logger.info("🎉 Experiment completed successfully!")
    return experiment, history, comparison

if __name__ == "__main__":
    try:
        experiment, history, comparison = run_complete_bert_qat_experiment()
        print("\n✅ BERT QAT Integration successful!")
        print(f"Final QAT Accuracy: {history[-1]['eval_accuracy']:.4f}")
        print(f"Size Reduction: {comparison['size_reduction']:.1f}%")
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc() 