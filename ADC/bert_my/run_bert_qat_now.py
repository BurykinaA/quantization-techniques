"""Immediate runnable BERT QAT experiment"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from qat_layers import QATLinear
from qat_training import QATTrainer
import torch.nn as nn

def create_dummy_bert_data(tokenizer, num_samples=500, max_length=64):
    """Create dummy data for BERT"""
    # Generate random sentences
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(1, vocab_size, (num_samples, max_length))
    attention_mask = torch.ones(num_samples, max_length)
    
    # Random mask some tokens (simulate real padding)
    for i in range(num_samples):
        mask_start = torch.randint(max_length//2, max_length, (1,)).item()
        attention_mask[i, mask_start:] = 0
        input_ids[i, mask_start:] = tokenizer.pad_token_id
    
    # Random binary labels
    labels = torch.randint(0, 2, (num_samples,))
    
    return TensorDataset(input_ids, attention_mask, labels)

def convert_bert_to_qat_simple(model):
    """Simple BERT to QAT conversion"""
    count = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and 'attention' in name:
                # Only convert attention layers for this demo
                qat_layer = QATLinear(
                    child.in_features, child.out_features,
                    bias=(child.bias is not None),
                    weight_bits=8, activation_bits=8
                )
                with torch.no_grad():
                    qat_layer.weight.copy_(child.weight)
                    if child.bias is not None:
                        qat_layer.bias.copy_(child.bias)
                setattr(module, child_name, qat_layer)
                count += 1
                print(f"Converted {name}.{child_name} to QAT")
    print(f"Total conversions: {count}")
    return model

def main():
    print("🚀 Running BERT QAT Experiment")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Convert to QAT
    model = convert_bert_to_qat_simple(model)
    model = model.to(device)
    
    # Create data
    train_dataset = create_dummy_bert_data(tokenizer, num_samples=400)
    val_dataset = create_dummy_bert_data(tokenizer, num_samples=100)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    trainer = QATTrainer(model, optimizer, device)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    for epoch in range(2):  # Just 2 epochs for demo
        print(f"\nEpoch {epoch + 1}/2")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, criterion, epoch, warmup_epochs=1)
        print(f"Train Loss: {train_metrics['avg_loss']:.4f}, Grad Norm: {train_metrics['avg_grad_norm']:.6f}")
        
        # Simple validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.4f}")
    
    print("\n✅ BERT QAT training completed!")
    
    # Save model
    torch.save(model.state_dict(), 'bert_qat_demo.pth')
    print("Model saved as 'bert_qat_demo.pth'")

if __name__ == "__main__":
    main()