"""Quick test script for BERT QAT integration"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from qat_layers import QATLinear
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_bert_qat_test():
    """Quick test to verify BERT QAT integration works"""
    logger.info("🧪 Quick BERT QAT Test")
    
    # Load small model for testing
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    logger.info(f"Original model loaded: {model_name}")
    
    # Count original linear layers
    original_linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    logger.info(f"Original Linear layers: {original_linear_count}")
    
    # Convert to QAT (simplified version)
    def replace_linear_with_qat(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                # Skip classifier and embeddings for this test
                if 'classifier' not in name and 'embeddings' not in name:
                    qat_layer = QATLinear(
                        child.in_features, 
                        child.out_features,
                        bias=(child.bias is not None),
                        weight_bits=8,
                        activation_bits=8
                    )
                    # Copy weights
                    with torch.no_grad():
                        qat_layer.weight.copy_(child.weight)
                        if child.bias is not None:
                            qat_layer.bias.copy_(child.bias)
                    setattr(module, name, qat_layer)
                    logger.info(f"Replaced {name} with QAT layer")
            else:
                replace_linear_with_qat(child)
    
    # Apply conversion
    replace_linear_with_qat(model)
    
    # Count QAT layers
    qat_count = sum(1 for m in model.modules() if isinstance(m, QATLinear))
    remaining_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    logger.info(f"QAT layers: {qat_count}, Remaining Linear: {remaining_linear}")
    
    # Test forward pass
    test_text = "This is a test sentence for QAT BERT."
    inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
    
    logger.info("Testing forward pass...")
    model.train()  # Enable quantization
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=-1)
    
    logger.info(f"Forward pass successful!")
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Predictions: {predictions}")
    
    # Test backward pass
    logger.info("Testing backward pass...")
    model.zero_grad()
    loss = logits.sum()
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    qat_layers_with_grad = 0
    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            if module.weight.grad is not None:
                total_grad_norm += module.weight.grad.norm().item() ** 2
                qat_layers_with_grad += 1
                # Check quantizer gradients
                weight_scale_grad = module.weight_quantizer.scale.grad
                activation_scale_grad = module.activation_quantizer.scale.grad
                logger.info(f"{name}: weight_scale_grad={weight_scale_grad is not None}, "
                           f"activation_scale_grad={activation_scale_grad is not None}")
    
    total_grad_norm = total_grad_norm ** 0.5
    logger.info(f"Total gradient norm: {total_grad_norm:.6f}")
    logger.info(f"QAT layers with gradients: {qat_layers_with_grad}/{qat_count}")
    
    if total_grad_norm > 0 and qat_layers_with_grad > 0:
        logger.info("✅ BERT QAT integration test PASSED!")
        return True
    else:
        logger.error("❌ BERT QAT integration test FAILED!")
        return False

if __name__ == "__main__":
    success = quick_bert_qat_test()
    if success:
        print("\n🎉 Ready to run full BERT QAT experiment!")
        print("Next steps:")
        print("1. Run: python bert_qat_integration.py")
        print("2. Or modify the config in bert_qat_integration.py for your needs")
    else:
        print("\n❌ Issues detected. Check the logs above.") 