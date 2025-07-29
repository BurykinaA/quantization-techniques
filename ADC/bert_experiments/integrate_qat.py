import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from qat_layers import QATLinear, QATTransformerBlock
from qat_training import QATTrainer

class SimpleQATBert(nn.Module):
    """
    Simple BERT-like model with QAT layers
    """
    def __init__(self, vocab_size=30522, d_model=768, num_heads=12, num_layers=2, num_classes=2):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers (not quantized for now)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)
        
        # QAT Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            QATTransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_model * 4,
                weight_bits=8,
                activation_bits=8
            ) for _ in range(num_layers)
        ])
        
        # QAT Classification head
        self.classifier = QATLinear(
            d_model, 
            num_classes, 
            weight_bits=8, 
            activation_bits=8
        )
        
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Create embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer blocks
        hidden_states = embeddings
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mask=attention_mask)
        
        # Global average pooling
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        output = {'logits': logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            output['loss'] = loss
        
        return output

def create_dummy_data(num_samples=1000, seq_len=128, vocab_size=30522, num_classes=2):
    """Create dummy data for testing"""
    print("Creating dummy dataset...")
    
    # Random input IDs
    input_ids = torch.randint(1, vocab_size, (num_samples, seq_len))
    
    # Random attention masks (some tokens are padded)
    attention_mask = torch.ones(num_samples, seq_len)
    for i in range(num_samples):
        # Randomly mask some positions as padding
        mask_len = torch.randint(seq_len//2, seq_len, (1,)).item()
        attention_mask[i, mask_len:] = 0
    
    # Random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    return TensorDataset(input_ids, attention_mask, labels)

def run_qat_training():
    """Main function to run QAT training"""
    print("Starting QAT Training Example")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    model_config = {
        'vocab_size': 30522,
        'd_model': 512,  # Smaller for faster training
        'num_heads': 8,
        'num_layers': 2,  # Start with fewer layers
        'num_classes': 2
    }
    
    # Create model
    print("Creating QAT model...")
    model = SimpleQATBert(**model_config).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Create trainer
    trainer = QATTrainer(model, optimizer, device, gradient_clip_norm=1.0)
    
    # Create dummy data
    train_dataset = create_dummy_data(num_samples=800, seq_len=64)  # Smaller for faster training
    val_dataset = create_dummy_data(num_samples=200, seq_len=64)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 3
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, criterion, epoch, warmup_epochs=1)
        print(f"Train - Loss: {train_metrics['avg_loss']:.4f}, "
              f"Grad Norm: {train_metrics['avg_grad_norm']:.6f}")
        
        # Validate
        val_metrics = trainer.validate_epoch(val_loader, criterion)
        print(f"Val - Loss: {val_metrics['avg_loss']:.4f}")
        
        # Save best model
        if val_metrics['avg_loss'] < best_val_loss:
            best_val_loss = val_metrics['avg_loss']
            torch.save(model.state_dict(), 'best_qat_model.pth')
            print("Saved best model!")
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    return model, trainer

def test_model_inference(model, device):
    """Test the trained model inference"""
    print("\n=== Testing Model Inference ===")
    
    model.eval()
    
    # Create a test batch
    test_input_ids = torch.randint(1, 30522, (2, 32)).to(device)
    test_attention_mask = torch.ones(2, 32).to(device)
    
    with torch.no_grad():
        outputs = model(test_input_ids, test_attention_mask)
        logits = outputs['logits']
        predictions = torch.softmax(logits, dim=-1)
    
    print(f"Input shape: {test_input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Predictions: {predictions}")
    print(f"Predicted classes: {torch.argmax(predictions, dim=-1)}")

def check_quantization_status(model):
    """Check which layers have quantization enabled"""
    print("\n=== Quantization Status ===")
    
    for name, module in model.named_modules():
        if hasattr(module, 'quantization_enabled'):
            status = "ENABLED" if module.quantization_enabled else "DISABLED"
            print(f"{name}: {status}")
        elif hasattr(module, 'weight_quantizer'):
            print(f"{name}: Has quantizers")

if __name__ == "__main__":
    try:
        # Run training
        model, trainer = run_qat_training()
        
        # Test inference
        test_model_inference(model, trainer.device)
        
        # Check quantization status
        check_quantization_status(model)
        
        print("\n=== QAT Example completed successfully! ===")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
```

**
</rewritten_file>