#!/usr/bin/env python3
"""
ADC Pipeline Visualizer - Debug tool to understand what's happening in each step
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple
import os


class ADCPipelineDebugger:
    """
    Captures and visualizes every step of the ADC quantization pipeline
    """
    
    def __init__(self, output_dir: str = "./adc_debug"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.captured_data = {}
    
    def attach_to_layer(self, layer, layer_name: str):
        """Attach debugger to a QATLinearADC layer"""
        
        def capture_hook(module, input, output):
            # Get the input
            x_raw = input[0].detach().cpu()
            
            # Capture internal state
            with torch.no_grad():
                # Get quantizers
                act_q = module.activation_quantizer
                w_q = module.weight_quantizer
                
                # 1. Raw input
                x = input[0]
                
                # 2. Quantized activation codes
                s_x = act_q.scale
                if act_q.symmetric:
                    code_x = torch.round(x / s_x)
                    code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
                else:
                    zp_x = act_q.zero_point
                    code_x = torch.round(x / s_x + zp_x)
                    code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
                
                # Apply ashift if enabled
                if module.ashift:
                    code_x_shifted = code_x - module.C
                else:
                    code_x_shifted = code_x
                
                # Dequantized activation
                if act_q.symmetric:
                    x_dequant = code_x * s_x
                else:
                    x_dequant = (code_x - zp_x) * s_x
                
                # 3. Weight quantization
                w_raw = module.weight
                s_w_vec = w_q.scale
                s_w_b = s_w_vec.view(-1, 1)
                code_w = torch.round(w_raw / s_w_b)
                code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)
                w_dequant = code_w * s_w_b
                
                # 4. Matrix multiplications
                # Full precision
                y_fp = F.linear(x, w_raw, bias=None)
                
                # Quantized (in code domain)
                y_int = F.linear(code_x_shifted, code_w, bias=None)
                
                # 5. ADC quantization
                delta = module.adc_quantizer._delta
                na = module.adc_quantizer.na
                pa = module.adc_quantizer.pa
                
                y_adc_codes = torch.round(y_int / delta)
                y_adc_codes = torch.clamp(y_adc_codes, na, pa)
                y_adc = y_adc_codes * delta
                
                # 6. Final dequantization
                y_real = y_adc * s_x * s_w_vec
                
                # Store everything
                self.captured_data[layer_name] = {
                    # Step 1: Raw inputs
                    'x_raw': x.detach().cpu(),
                    'w_raw': w_raw.detach().cpu(),
                    
                    # Step 2: Quantization parameters
                    's_x': s_x.detach().cpu(),
                    's_w': s_w_vec.detach().cpu(),
                    'delta': delta.detach().cpu() if torch.is_tensor(delta) else torch.tensor(delta),
                    'ashift_C': module.C if module.ashift else 0,
                    
                    # Step 3: Quantized codes
                    'code_x': code_x.detach().cpu(),
                    'code_x_shifted': code_x_shifted.detach().cpu(),
                    'code_w': code_w.detach().cpu(),
                    'x_dequant': x_dequant.detach().cpu(),
                    'w_dequant': w_dequant.detach().cpu(),
                    
                    # Step 4: Matrix multiplications
                    'y_fp': y_fp.detach().cpu(),
                    'y_int': y_int.detach().cpu(),
                    
                    # Step 5: ADC quantization
                    'y_adc_codes': y_adc_codes.detach().cpu(),
                    'y_adc': y_adc.detach().cpu(),
                    
                    # Step 6: Final output
                    'y_real': y_real.detach().cpu(),
                    'y_final': output.detach().cpu(),
                    
                    # Clipping info
                    'na': na,
                    'pa': pa,
                    'qmin_x': act_q.qmin,
                    'qmax_x': act_q.qmax,
                    'qmin_w': w_q.qmin,
                    'qmax_w': w_q.qmax,
                }
        
        # Register hook
        layer.register_forward_hook(capture_hook)
        return self
    
    def plot_pipeline(self, layer_name: str, sample_idx: int = 0):
        """Plot the complete ADC pipeline for a specific layer"""
        
        if layer_name not in self.captured_data:
            print(f"No data for layer {layer_name}")
            return
        
        data = self.captured_data[layer_name]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Extract one sample for visualization
        def get_sample(tensor, idx=sample_idx):
            if tensor.dim() > 2:
                return tensor[idx].flatten()[:1000].numpy()  # Take first 1000 elements
            elif tensor.dim() == 2:
                if idx < tensor.shape[0]:
                    return tensor[idx].flatten()[:1000].numpy()
                else:
                    return tensor[0].flatten()[:1000].numpy()
            else:
                return tensor.flatten()[:1000].numpy()
        
        # 1. Raw activation X
        ax1 = plt.subplot(4, 3, 1)
        x_raw = get_sample(data['x_raw'])
        ax1.hist(x_raw, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title(f'1. Raw Activation X\nMean: {x_raw.mean():.4f}, Std: {x_raw.std():.4f}')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # 2. Quantized activation codes
        ax2 = plt.subplot(4, 3, 2)
        code_x = get_sample(data['code_x'])
        ax2.hist(code_x, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_title(f'2. Quantized Activation Codes\nRange: [{data["qmin_x"]}, {data["qmax_x"]}]')
        ax2.axvline(data['qmin_x'], color='r', linestyle='--', label='qmin')
        ax2.axvline(data['qmax_x'], color='r', linestyle='--', label='qmax')
        ax2.set_xlabel('Code Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. A-shift effect
        ax3 = plt.subplot(4, 3, 3)
        code_x_shifted = get_sample(data['code_x_shifted'])
        ax3.hist(code_x_shifted, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.set_title(f'3. After A-Shift (C={data["ashift_C"]})\nMean: {code_x_shifted.mean():.4f}')
        ax3.set_xlabel('Shifted Code')
        ax3.grid(True, alpha=0.3)
        
        # 4. Raw weights W
        ax4 = plt.subplot(4, 3, 4)
        w_raw = data['w_raw'].flatten()[:1000].numpy()
        ax4.hist(w_raw, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_title(f'4. Raw Weights W\nMean: {w_raw.mean():.4f}, Std: {w_raw.std():.4f}')
        ax4.set_xlabel('Value')
        ax4.grid(True, alpha=0.3)
        
        # 5. Quantized weight codes
        ax5 = plt.subplot(4, 3, 5)
        code_w = data['code_w'].flatten()[:1000].numpy()
        ax5.hist(code_w, bins=50, alpha=0.7, color='brown', edgecolor='black')
        ax5.set_title(f'5. Quantized Weight Codes\nRange: [{data["qmin_w"]}, {data["qmax_w"]}]')
        ax5.axvline(data['qmin_w'], color='r', linestyle='--', label='qmin')
        ax5.axvline(data['qmax_w'], color='r', linestyle='--', label='qmax')
        ax5.set_xlabel('Code Value')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Full precision MM output
        ax6 = plt.subplot(4, 3, 6)
        y_fp = get_sample(data['y_fp'])
        ax6.hist(y_fp, bins=50, alpha=0.7, color='cyan', edgecolor='black')
        ax6.set_title(f'6. Full Precision Y = X @ W\nMean: {y_fp.mean():.4f}, Std: {y_fp.std():.4f}')
        ax6.set_xlabel('Value')
        ax6.grid(True, alpha=0.3)
        
        # 7. Integer domain MM output (before ADC)
        ax7 = plt.subplot(4, 3, 7)
        y_int = get_sample(data['y_int'])
        ax7.hist(y_int, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax7.set_title(f'7. Integer MM: code_X @ code_W\nMean: {y_int.mean():.4f}, Std: {y_int.std():.4f}')
        ax7.set_xlabel('Integer Value')
        ax7.grid(True, alpha=0.3)
        
        # 8. ADC quantized codes
        ax8 = plt.subplot(4, 3, 8)
        y_adc_codes = get_sample(data['y_adc_codes'])
        ax8.hist(y_adc_codes, bins=50, alpha=0.7, color='magenta', edgecolor='black')
        ax8.set_title(f'8. ADC Quantized Codes\nRange: [{data["na"]}, {data["pa"]}], Δ={data["delta"].item():.2f}')
        ax8.axvline(data['na'], color='r', linestyle='--', label='na')
        ax8.axvline(data['pa'], color='r', linestyle='--', label='pa')
        ax8.set_xlabel('ADC Code')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. ADC output (codes * delta)
        ax9 = plt.subplot(4, 3, 9)
        y_adc = get_sample(data['y_adc'])
        ax9.hist(y_adc, bins=50, alpha=0.7, color='lime', edgecolor='black')
        ax9.set_title(f'9. ADC Output: codes × Δ\nMean: {y_adc.mean():.4f}')
        ax9.set_xlabel('Value')
        ax9.grid(True, alpha=0.3)
        
        # 10. Final dequantized output
        ax10 = plt.subplot(4, 3, 10)
        y_real = get_sample(data['y_real'])
        ax10.hist(y_real, bins=50, alpha=0.7, color='gold', edgecolor='black')
        ax10.set_title(f'10. Final Output (after dequant)\nMean: {y_real.mean():.4f}')
        ax10.set_xlabel('Value')
        ax10.grid(True, alpha=0.3)
        
        # 11. Comparison: FP vs Quantized
        ax11 = plt.subplot(4, 3, 11)
        ax11.scatter(y_fp[:500], y_real[:500], alpha=0.5, s=1)
        ax11.plot([y_fp.min(), y_fp.max()], [y_fp.min(), y_fp.max()], 'r--', label='y=x')
        ax11.set_title('11. FP vs Quantized Output')
        ax11.set_xlabel('Full Precision')
        ax11.set_ylabel('Quantized')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # 12. Error distribution
        ax12 = plt.subplot(4, 3, 12)
        error = y_fp - y_real
        ax12.hist(error, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax12.set_title(f'12. Quantization Error\nMSE: {(error**2).mean():.4f}')
        ax12.set_xlabel('Error (FP - Quant)')
        ax12.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{layer_name.replace(".", "_")}_pipeline.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved pipeline visualization for {layer_name}")
        
        # Print statistics
        self._print_statistics(layer_name, data)
    
    def _print_statistics(self, layer_name: str, data: Dict):
        """Print detailed statistics"""
        
        print(f"\n{'='*80}")
        print(f"LAYER: {layer_name}")
        print(f"{'='*80}\n")
        
        print("📊 QUANTIZATION PARAMETERS:")
        print(f"  Activation scale (s_x):  {data['s_x'].mean().item():.6f}")
        print(f"  Weight scale (s_w):      {data['s_w'].mean().item():.6f}")
        print(f"  ADC delta (Δ):           {data['delta'].item():.6f}")
        print(f"  A-shift constant (C):    {data['ashift_C']}")
        print()
        
        print("📈 ACTIVATION STATISTICS:")
        x_raw = data['x_raw'].flatten()
        print(f"  Raw X:          mean={x_raw.mean():.4f}, std={x_raw.std():.4f}, min={x_raw.min():.4f}, max={x_raw.max():.4f}")
        code_x = data['code_x'].flatten()
        print(f"  Quantized X:    mean={code_x.mean():.4f}, std={code_x.std():.4f}, min={code_x.min():.4f}, max={code_x.max():.4f}")
        print(f"  Clipping range: [{data['qmin_x']}, {data['qmax_x']}]")
        clipped_x = ((code_x == data['qmin_x']) | (code_x == data['qmax_x'])).sum().item()
        print(f"  Clipped values: {clipped_x} / {code_x.numel()} ({100*clipped_x/code_x.numel():.2f}%)")
        print()
        
        print("⚖️ WEIGHT STATISTICS:")
        w_raw = data['w_raw'].flatten()
        print(f"  Raw W:          mean={w_raw.mean():.4f}, std={w_raw.std():.4f}, min={w_raw.min():.4f}, max={w_raw.max():.4f}")
        code_w = data['code_w'].flatten()
        print(f"  Quantized W:    mean={code_w.mean():.4f}, std={code_w.std():.4f}, min={code_w.min():.4f}, max={code_w.max():.4f}")
        print(f"  Clipping range: [{data['qmin_w']}, {data['qmax_w']}]")
        clipped_w = ((code_w == data['qmin_w']) | (code_w == data['qmax_w'])).sum().item()
        print(f"  Clipped values: {clipped_w} / {code_w.numel()} ({100*clipped_w/code_w.numel():.2f}%)")
        print()
        
        print("🔢 ADC QUANTIZATION:")
        y_int = data['y_int'].flatten()
        print(f"  Before ADC:     mean={y_int.mean():.4f}, std={y_int.std():.4f}, min={y_int.min():.4f}, max={y_int.max():.4f}")
        y_adc_codes = data['y_adc_codes'].flatten()
        print(f"  ADC codes:      mean={y_adc_codes.mean():.4f}, std={y_adc_codes.std():.4f}, min={y_adc_codes.min():.4f}, max={y_adc_codes.max():.4f}")
        print(f"  Clipping range: [{data['na']}, {data['pa']}]")
        clipped_adc = ((y_adc_codes == data['na']) | (y_adc_codes == data['pa'])).sum().item()
        print(f"  Clipped values: {clipped_adc} / {y_adc_codes.numel()} ({100*clipped_adc/y_adc_codes.numel():.2f}%)")
        print()
        
        print("🎯 FINAL OUTPUT:")
        y_fp = data['y_fp'].flatten()
        y_real = data['y_real'].flatten()
        error = (y_fp - y_real).abs()
        print(f"  Full precision: mean={y_fp.mean():.4f}, std={y_fp.std():.4f}")
        print(f"  Quantized:      mean={y_real.mean():.4f}, std={y_real.std():.4f}")
        print(f"  MAE:            {error.mean():.4f}")
        print(f"  MSE:            {(error**2).mean():.4f}")
        print(f"  Max error:      {error.max():.4f}")
        print()


def debug_model(model, input_ids, attention_mask, layer_patterns=None, output_dir="./adc_debug"):
    """
    Debug a model by capturing pipeline for specific layers
    
    Args:
        model: The ADC QAT model
        input_ids: Input token IDs
        attention_mask: Attention mask
        layer_patterns: List of layer name patterns to capture (e.g., ["layer.0.attention", "layer.11.output"])
        output_dir: Where to save visualizations
    """
    
    debugger = ADCPipelineDebugger(output_dir=output_dir)
    
    # Find layers to monitor
    layers_to_monitor = []
    for name, module in model.named_modules():
        if hasattr(module, 'adc_quantizer'):  # It's a QATLinearADC
            if layer_patterns is None:
                layers_to_monitor.append((name, module))
            else:
                if any(pattern in name for pattern in layer_patterns):
                    layers_to_monitor.append((name, module))
    
    print(f"Found {len(layers_to_monitor)} ADC layers to monitor")
    
    # Attach debugger to layers
    for name, module in layers_to_monitor:
        debugger.attach_to_layer(module, name)
        print(f"  ✓ Attached to {name}")
    
    # Run forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    for name, _ in layers_to_monitor:
        debugger.plot_pipeline(name, sample_idx=0)
    
    print(f"\n✅ Done! Visualizations saved to: {output_dir}")
    
    return debugger


if __name__ == "__main__":
    print("ADC Pipeline Visualizer")
    print("Import this module and use debug_model() function")
    print("\nExample usage:")
    print("""
    from adc_pipeline_visualizer import debug_model
    from transformers import AutoTokenizer, BertForQuestionAnswering
    
    # Load your model
    model = BertForQuestionAnswering.from_pretrained("your_checkpoint")
    tokenizer = AutoTokenizer.from_pretrained("your_checkpoint")
    
    # Prepare input
    text = "What is the capital of France?"
    inputs = tokenizer(text, text, return_tensors="pt", padding="max_length", max_length=128)
    
    # Debug specific layers
    debugger = debug_model(
        model, 
        inputs['input_ids'], 
        inputs['attention_mask'],
        layer_patterns=["layer.0.attention.output", "layer.11.output"],
        output_dir="./debug_output"
    )
    """)

