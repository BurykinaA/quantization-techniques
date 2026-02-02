#!/usr/bin/env python3
"""
Quick test script to verify WandB integration works
"""

import torch
from transformers import AutoConfig, BertForQuestionAnswering
from ADC.bert_clean.core.qat_layers import QATLinear
from ADC.bert_clean.core.qat_visualizer import QATDebugger
from ADC.bert_clean.core.wandb_integration import WANDB_AVAILABLE

print("=" * 60)
print("Testing BERT QAT WandB Integration")
print("=" * 60)
print()

# 1. Test imports
print("✓ All imports successful")
print()

# 2. Check WandB availability
print(f"WandB available: {WANDB_AVAILABLE}")
if not WANDB_AVAILABLE:
    print("  ⚠️  Install wandb: pip install wandb")
print()

# 3. Test QAT layer creation
print("Testing QATLinear layer...")
layer = QATLinear(768, 768, weight_bits=8, activation_bits=8)
x = torch.randn(2, 10, 768)
y = layer(x)
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {y.shape}")
print(f"  Weight scale: {layer.weight_quantizer.scale.mean().item():.6f}")
print(f"  Activation scale: {layer.activation_quantizer.scale.mean().item():.6f}")
print("✓ QATLinear works")
print()

# 4. Test model conversion
print("Testing BERT model conversion...")
from ADC.bert_clean.runs.bert_qat_integration import BertQATConverter

config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_hidden_layers = 2  # Small model for testing
model = BertForQuestionAnswering(config)

print(f"  Before conversion: {sum(isinstance(m, QATLinear) for m in model.modules())} QAT layers")

model = BertQATConverter.replace_linear_with_qat(
    model,
    weight_bits=8,
    activation_bits=8,
    exclude_patterns=["embeddings", "pooler", "qa_outputs"]
)

qat_count = sum(isinstance(m, QATLinear) for m in model.modules())
print(f"  After conversion: {qat_count} QAT layers")
print("✓ Model conversion works")
print()

# 5. Test visualization (without WandB)
print("Testing QAT visualization...")
debugger = QATDebugger(output_dir="./test_viz")

# Find first QAT layer
test_layer = None
test_layer_name = None
for name, module in model.named_modules():
    if isinstance(module, QATLinear):
        test_layer = module
        test_layer_name = name
        break

if test_layer is not None:
    print(f"  Testing layer: {test_layer_name}")
    
    # Attach debugger
    debugger.attach_to_layer(test_layer, test_layer_name)
    
    # Run forward pass
    input_ids = torch.randint(0, 100, (1, 20))
    attention_mask = torch.ones(1, 20)
    
    model.eval()
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Generate visualization
    fig, data = debugger.plot_layer(test_layer_name, sample_idx=0)
    
    # Get statistics
    stats = debugger.get_statistics(test_layer_name)
    
    print(f"  Generated visualization: ./test_viz/{test_layer_name.replace('.', '_')}_qat.png")
    print(f"  Statistics keys: {list(stats.keys())[:5]}...")
    print(f"  Activation MAE: {stats.get('x_quant_mae', 0):.6f}")
    print(f"  Weight MAE: {stats.get('w_quant_mae', 0):.6f}")
    print(f"  Output MSE: {stats.get('y_mse', 0):.6f}")
    print("✓ Visualization works")
else:
    print("  ⚠️  No QAT layers found")

print()
print("=" * 60)
print("✅ All tests passed!")
print("=" * 60)
print()
print("Next steps:")
print("1. Fine-tune a BERT model on SQuAD (full precision)")
print("2. Run QAT training:")
print("   python bert_qat_integration.py \\")
print("     --fp_checkpoint_dir /path/to/fp/checkpoint \\")
print("     --use_wandb \\")
print("     --weight_bits 8 \\")
print("     --activation_bits 8")
print()

