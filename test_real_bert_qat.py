from integrate_qat import replace_linear_with_qat, convert_bert_to_qat
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def test_with_real_bert():
    # Load a real BERT model
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=2
    )
    
    # Convert to QAT
    replace_linear_with_qat(model, weight_bits=8, activation_bits=8)
    
    # Test with real input
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    text = "This is a test sentence for QAT."
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    model.train()  # Enable quantization
    outputs = model(**inputs)
    
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits: {outputs.logits}")
    
    # Test backward pass
    loss = outputs.logits.sum()
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    
    print(f"Total gradient norm: {total_grad_norm ** 0.5}")

if __name__ == "__main__":
    test_with_real_bert() 