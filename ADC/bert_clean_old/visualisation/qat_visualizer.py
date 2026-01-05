#!/usr/bin/env python3
"""
QAT Visualizer - Visualize quantization effects in QAT layers
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
import os


class QATDebugger:
    """
    Captures and visualizes QAT layer activations, weights, and quantization effects
    """
    
    def __init__(self, output_dir: str = "./qat_debug"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.captured_data = {}
    
    def attach_to_layer(self, layer, layer_name: str):
        """Attach debugger to a QATLinear layer"""
        from .qat_layers import QATLinear
        
        if not isinstance(layer, QATLinear):
            return self
        
        def capture_hook(module, input, output):
            with torch.no_grad():
                # Get input activation
                x_raw = input[0]
                
                # Get quantizers
                act_q = module.activation_quantizer
                w_q = module.weight_quantizer
                
                # 1. Quantize activation
                s_x = act_q.scale
                if act_q.symmetric:
                    code_x = torch.round(x_raw / s_x)
                    code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
                    x_quant = code_x * s_x
                else:
                    zp_x = act_q.zero_point
                    code_x = torch.round(x_raw / s_x + zp_x)
                    code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
                    x_quant = (code_x - zp_x) * s_x
                
                # 2. Quantize weight
                w_raw = module.weight
                s_w_vec = w_q.scale
                
                # Per-channel quantization
                if w_q.per_channel:
                    s_w_b = s_w_vec.view(-1, 1)
                    code_w = torch.round(w_raw / s_w_b)
                    code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)
                    w_quant = code_w * s_w_b
                else:
                    code_w = torch.round(w_raw / s_w_vec)
                    code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)
                    w_quant = code_w * s_w_vec
                
                # 3. Compute outputs
                # Full precision
                y_fp = F.linear(x_raw, w_raw, module.bias)
                
                # Quantized
                y_quant = F.linear(x_quant, w_quant, module.bias)
                
                # Store everything
                self.captured_data[layer_name] = {
                    # Raw values
                    'x_raw': x_raw.detach().cpu(),
                    'w_raw': w_raw.detach().cpu(),
                    
                    # Quantization parameters
                    's_x': s_x.detach().cpu(),
                    's_w': s_w_vec.detach().cpu(),
                    
                    # Quantized codes
                    'code_x': code_x.detach().cpu(),
                    'code_w': code_w.detach().cpu(),
                    
                    # Dequantized values
                    'x_quant': x_quant.detach().cpu(),
                    'w_quant': w_quant.detach().cpu(),
                    
                    # Outputs
                    'y_fp': y_fp.detach().cpu(),
                    'y_quant': y_quant.detach().cpu(),
                    'y_final': output.detach().cpu(),
                    
                    # Range info
                    'qmin_x': act_q.qmin,
                    'qmax_x': act_q.qmax,
                    'qmin_w': w_q.qmin,
                    'qmax_w': w_q.qmax,
                    'num_bits_x': act_q.num_bits,
                    'num_bits_w': w_q.num_bits,
                }
        
        # Register hook
        layer.register_forward_hook(capture_hook)
        return self
    
    def plot_layer(self, layer_name: str, sample_idx: int = 0):
        """Plot quantization effects for a specific layer"""
        
        if layer_name not in self.captured_data:
            print(f"No data for layer {layer_name}")
            return None
        
        data = self.captured_data[layer_name]
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Extract one sample for visualization
        def get_sample(tensor, idx=sample_idx):
            if tensor.dim() > 2:
                return tensor[idx].flatten()[:1000].numpy()
            elif tensor.dim() == 2:
                if idx < tensor.shape[0]:
                    return tensor[idx].flatten()[:1000].numpy()
                else:
                    return tensor[0].flatten()[:1000].numpy()
            else:
                return tensor.flatten()[:1000].numpy()
        
        # 1. Raw activation X
        ax1 = plt.subplot(3, 4, 1)
        x_raw = get_sample(data['x_raw'])
        ax1.hist(x_raw, bins=50, alpha=0.7, edgecolor='black', color='blue')
        ax1.set_title(f'1. Raw Activation X\nMean: {x_raw.mean():.4f}, Std: {x_raw.std():.4f}')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # 2. Quantized activation codes
        ax2 = plt.subplot(3, 4, 2)
        code_x = get_sample(data['code_x'])
        ax2.hist(code_x, bins=min(50, 2**data['num_bits_x']), alpha=0.7, color='orange', edgecolor='black')
        ax2.set_title(f'2. Activation Codes ({data["num_bits_x"]}-bit)\nRange: [{data["qmin_x"]}, {data["qmax_x"]}]')
        ax2.axvline(data['qmin_x'], color='r', linestyle='--', linewidth=2, label='qmin')
        ax2.axvline(data['qmax_x'], color='r', linestyle='--', linewidth=2, label='qmax')
        ax2.set_xlabel('Code Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Dequantized activation
        ax3 = plt.subplot(3, 4, 3)
        x_quant = get_sample(data['x_quant'])
        ax3.hist(x_quant, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.set_title(f'3. Dequantized Activation\nMean: {x_quant.mean():.4f}')
        ax3.set_xlabel('Value')
        ax3.grid(True, alpha=0.3)
        
        # 4. Activation quantization error
        ax4 = plt.subplot(3, 4, 4)
        x_error = get_sample(data['x_raw']) - x_quant
        ax4.hist(x_error, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax4.set_title(f'4. Activation Quant Error\nMAE: {np.abs(x_error).mean():.4f}')
        ax4.set_xlabel('Error')
        ax4.grid(True, alpha=0.3)
        
        # 5. Raw weights W
        ax5 = plt.subplot(3, 4, 5)
        w_raw = data['w_raw'].flatten()[:5000].numpy()
        ax5.hist(w_raw, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax5.set_title(f'5. Raw Weights W\nMean: {w_raw.mean():.4f}, Std: {w_raw.std():.4f}')
        ax5.set_xlabel('Value')
        ax5.grid(True, alpha=0.3)
        
        # 6. Quantized weight codes
        ax6 = plt.subplot(3, 4, 6)
        code_w = data['code_w'].flatten()[:5000].numpy()
        ax6.hist(code_w, bins=min(50, 2**data['num_bits_w']), alpha=0.7, color='brown', edgecolor='black')
        ax6.set_title(f'6. Weight Codes ({data["num_bits_w"]}-bit)\nRange: [{data["qmin_w"]}, {data["qmax_w"]}]')
        ax6.axvline(data['qmin_w'], color='r', linestyle='--', linewidth=2, label='qmin')
        ax6.axvline(data['qmax_w'], color='r', linestyle='--', linewidth=2, label='qmax')
        ax6.set_xlabel('Code Value')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Dequantized weights
        ax7 = plt.subplot(3, 4, 7)
        w_quant = data['w_quant'].flatten()[:5000].numpy()
        ax7.hist(w_quant, bins=50, alpha=0.7, color='cyan', edgecolor='black')
        ax7.set_title(f'7. Dequantized Weights\nMean: {w_quant.mean():.4f}')
        ax7.set_xlabel('Value')
        ax7.grid(True, alpha=0.3)
        
        # 8. Weight quantization error
        ax8 = plt.subplot(3, 4, 8)
        w_error = w_raw - w_quant
        ax8.hist(w_error, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax8.set_title(f'8. Weight Quant Error\nMAE: {np.abs(w_error).mean():.4f}')
        ax8.set_xlabel('Error')
        ax8.grid(True, alpha=0.3)
        
        # 9. Full precision output
        ax9 = plt.subplot(3, 4, 9)
        y_fp = get_sample(data['y_fp'])
        ax9.hist(y_fp, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax9.set_title(f'9. FP Output\nMean: {y_fp.mean():.4f}, Std: {y_fp.std():.4f}')
        ax9.set_xlabel('Value')
        ax9.grid(True, alpha=0.3)
        
        # 10. Quantized output
        ax10 = plt.subplot(3, 4, 10)
        y_quant = get_sample(data['y_quant'])
        ax10.hist(y_quant, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax10.set_title(f'10. Quantized Output\nMean: {y_quant.mean():.4f}')
        ax10.set_xlabel('Value')
        ax10.grid(True, alpha=0.3)
        
        # 11. FP vs Quantized scatter
        ax11 = plt.subplot(3, 4, 11)
        ax11.scatter(y_fp[:500], y_quant[:500], alpha=0.5, s=2)
        ax11.plot([y_fp.min(), y_fp.max()], [y_fp.min(), y_fp.max()], 'r--', linewidth=2, label='y=x')
        ax11.set_title('11. FP vs Quantized')
        ax11.set_xlabel('Full Precision')
        ax11.set_ylabel('Quantized')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # 12. Output quantization error
        ax12 = plt.subplot(3, 4, 12)
        y_error = y_fp - y_quant
        ax12.hist(y_error, bins=50, alpha=0.7, color='red', edgecolor='black')
        mse = (y_error**2).mean()
        ax12.set_title(f'12. Output Error\nMSE: {mse:.6f}')
        ax12.set_xlabel('Error (FP - Quant)')
        ax12.grid(True, alpha=0.3)
        
        plt.suptitle(f'QAT Layer: {layer_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'{layer_name.replace(".", "_")}_qat.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved QAT visualization for {layer_name}")
        
        return fig, data
    
    def get_statistics(self, layer_name: str) -> Dict:
        """Get quantization statistics for a layer"""
        if layer_name not in self.captured_data:
            return {}
        
        data = self.captured_data[layer_name]
        
        x_raw = data['x_raw'].flatten()
        x_quant = data['x_quant'].flatten()
        w_raw = data['w_raw'].flatten()
        w_quant = data['w_quant'].flatten()
        y_fp = data['y_fp'].flatten()
        y_quant = data['y_quant'].flatten()
        code_x = data['code_x'].flatten()
        code_w = data['code_w'].flatten()
        
        # Clipping statistics
        clipped_x = ((code_x == data['qmin_x']) | (code_x == data['qmax_x'])).sum().item()
        clipped_w = ((code_w == data['qmin_w']) | (code_w == data['qmax_w'])).sum().item()
        
        stats = {
            # Activation stats
            'x_mean': float(x_raw.mean()),
            'x_std': float(x_raw.std()),
            'x_min': float(x_raw.min()),
            'x_max': float(x_raw.max()),
            'x_quant_mae': float((x_raw - x_quant).abs().mean()),
            'x_quant_mse': float(((x_raw - x_quant)**2).mean()),
            'x_clipped_pct': 100.0 * clipped_x / code_x.numel(),
            
            # Weight stats
            'w_mean': float(w_raw.mean()),
            'w_std': float(w_raw.std()),
            'w_min': float(w_raw.min()),
            'w_max': float(w_raw.max()),
            'w_quant_mae': float((w_raw - w_quant).abs().mean()),
            'w_quant_mse': float(((w_raw - w_quant)**2).mean()),
            'w_clipped_pct': 100.0 * clipped_w / code_w.numel(),
            
            # Output stats
            'y_fp_mean': float(y_fp.mean()),
            'y_fp_std': float(y_fp.std()),
            'y_quant_mean': float(y_quant.mean()),
            'y_quant_std': float(y_quant.std()),
            'y_mae': float((y_fp - y_quant).abs().mean()),
            'y_mse': float(((y_fp - y_quant)**2).mean()),
            'y_max_error': float((y_fp - y_quant).abs().max()),
            
            # Quantization params
            's_x_mean': float(data['s_x'].mean()),
            's_w_mean': float(data['s_w'].mean()),
            'num_bits_x': data['num_bits_x'],
            'num_bits_w': data['num_bits_w'],
        }
        
        return stats


def debug_qat_model(model, input_ids, attention_mask, layer_patterns=None, output_dir="./qat_debug"):
    """
    Debug a QAT model by capturing pipeline for specific layers
    
    Args:
        model: The QAT model
        input_ids: Input token IDs
        attention_mask: Attention mask
        layer_patterns: List of layer name patterns to capture
        output_dir: Where to save visualizations
    
    Returns:
        debugger: QATDebugger instance with captured data
    """
    from .qat_layers import QATLinear
    
    debugger = QATDebugger(output_dir=output_dir)
    
    # Find layers to monitor
    layers_to_monitor = []
    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            if layer_patterns is None:
                layers_to_monitor.append((name, module))
            else:
                if any(pattern in name for pattern in layer_patterns):
                    layers_to_monitor.append((name, module))
    
    print(f"Found {len(layers_to_monitor)} QAT layers to monitor")
    
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
        debugger.plot_layer(name, sample_idx=0)
    
    print(f"\n✅ Done! Visualizations saved to: {output_dir}")
    
    return debugger

