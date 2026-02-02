import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class StraightThroughQuantize(torch.autograd.Function): #схуя у меня 2 класса 
    """
    Straight-through estimator for quantization with correct gradient flow to scale.
    Note: For asymmetric quantization, zero_point gradient is ~0 (it cancels out mathematically).
    """
    @staticmethod
    def forward(ctx, input, scale, zero_point, qmin, qmax, symmetric, per_channel, channel_dim):
        ctx.save_for_backward(input, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.symmetric = symmetric
        ctx.per_channel = per_channel
        ctx.channel_dim = channel_dim
        
        # Quantize
        if symmetric:
            x_scaled = input / scale
            x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
            output = x_quant * scale
        else:
            x_scaled = input / scale + zero_point
            x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
            output = (x_quant - zero_point) * scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, scale = ctx.saved_tensors
        
        # Straight-through for input gradients
        grad_input = grad_output
        
        # Compute gradient for scale parameter
        # d(output)/d(scale) = quantized_levels - input/scale
        quantized_levels = torch.clamp(torch.round(input / scale), ctx.qmin, ctx.qmax)
        scale_grad_per_element = grad_output * (quantized_levels - input / scale)
        
        # Sum gradients appropriately for per-channel vs per-tensor
        if ctx.per_channel and input.ndim > 1:
            # For per-channel, sum over all dims except channel_dim
            dims_to_sum = list(range(input.ndim))
            dims_to_sum.remove(ctx.channel_dim)
            grad_scale = scale_grad_per_element.sum(dim=dims_to_sum, keepdim=False)
            # Ensure shape matches scale.shape
            if grad_scale.shape != scale.shape:
                grad_scale = grad_scale.view_as(scale)
        else:
            # For per-tensor, sum everything
            grad_scale = scale_grad_per_element.sum().view_as(scale)
        
        # zero_point gradient is mathematically ~0 (cancels out in forward pass)
        # We don't learn it via gradients
        
        return grad_input, grad_scale, None, None, None, None, None, None

class LearnableQuantizer(nn.Module):
    """
    Simplified learnable quantizer with proper gradient flow.
    - Supports per-tensor and per-channel quantization
    - Scale is learned via backprop
    - Zero-point is computed from statistics (not learned, as gradient is ~0)
    """
    def __init__(self, 
                 num_bits: int = 8, 
                 symmetric: bool = True,
                 per_channel: bool = False,
                 channel_dim: int = 0):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.calibrated = False
        
        # Quantization range
        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1
        
        # Scale is learnable (will be initialized during calibration)
        # Always keep in FP32 for mixed precision training
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.1)
        
        # Zero-point is a buffer (computed from stats, not learned)
        if not symmetric:
            self.register_buffer('zero_point', torch.zeros(1, dtype=torch.float32))
        else:
            self.register_buffer('zero_point', torch.zeros(1, dtype=torch.float32))
    
    def calibrate(self, x: torch.Tensor):
        """
        Calibrate quantizer parameters based on input statistics.
        Call this during warmup phase before enabling fake quantization.
        """
        with torch.no_grad():
            # Work in FP32 for calibration
            x = x.float()
            
            if self.per_channel:
                # Per-channel statistics
                if self.channel_dim == 0:
                    x_reshaped = x.view(x.shape[0], -1)
                else:
                    x_transposed = x.transpose(self.channel_dim, 0).contiguous()
                    x_reshaped = x_transposed.view(x_transposed.shape[0], -1)
                
                x_min = x_reshaped.min(dim=1)[0]
                x_max = x_reshaped.max(dim=1)[0]
            else:
                # Per-tensor statistics
                x_min = x.min()
                x_max = x.max()
            
            # Compute scale and zero_point
            if self.symmetric:
                x_absmax = torch.max(x_min.abs(), x_max.abs())
                scale = x_absmax / (2 ** (self.num_bits - 1) - 1)
                scale = torch.clamp(scale, min=1e-8)
                zero_point = torch.zeros_like(scale)
            else:
                scale = (x_max - x_min) / (2 ** self.num_bits - 1)
                scale = torch.clamp(scale, min=1e-8)
                zero_point = -x_min / scale
                zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
            
            # Update parameters
            if self.calibrated:
                # EMA update if already calibrated
                momentum = 0.1
                self.scale.data = (1 - momentum) * self.scale.data + momentum * scale
                self.zero_point.data = (1 - momentum) * self.zero_point.data + momentum * zero_point
            else:
                # First calibration
                if self.per_channel:
                    channel_size = x.shape[self.channel_dim]
                    self.scale.data = scale.view(channel_size)
                    self.zero_point.data = zero_point.view(channel_size)
                else:
                    self.scale.data = scale.view(1)
                    self.zero_point.data = zero_point.view(1)
                self.calibrated = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Auto-calibrate on first forward if not calibrated
        if not self.calibrated:
            self.calibrate(x)
        
        # Ensure scale stays in FP32 for mixed precision training
        # Convert input to FP32 for quantization, then back to original dtype
        input_dtype = x.dtype
        x_fp32 = x.float()
        
        # Broadcast for per-channel quantization
        if self.per_channel and x.ndim > 1:
            shape = [1] * x_fp32.ndim
            shape[self.channel_dim] = -1
            scale = self.scale.view(shape)
            zero_point = self.zero_point.view(shape)
        else:
            scale = self.scale
            zero_point = self.zero_point
        
        # Apply quantization in FP32
        output = StraightThroughQuantize.apply(
            x_fp32, scale, zero_point, self.qmin, self.qmax, self.symmetric,
            self.per_channel, self.channel_dim
        )
        
        # Convert back to original dtype
        return output.to(input_dtype)

class QATLinear(nn.Linear):
    """
    Simplified QAT Linear layer with calibration support.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 weight_bits: int = 8,
                 activation_bits: int = 8,
                 weight_symmetric: bool = True,
                 activation_symmetric: bool = False):
        super().__init__(in_features, out_features, bias)
        
        # Weight quantizer (per-channel, symmetric)
        self.weight_quantizer = LearnableQuantizer(
            num_bits=weight_bits,
            symmetric=weight_symmetric,
            per_channel=True,
            channel_dim=0
        )
        
        # Activation quantizer (per-tensor)
        self.activation_quantizer = LearnableQuantizer(
            num_bits=activation_bits,
            symmetric=activation_symmetric,
            per_channel=False
        )
        
        self.quantization_enabled = True
    
    def calibrate(self, x: torch.Tensor):
        """Calibrate quantizers with a batch of data"""
        with torch.no_grad():
            self.activation_quantizer.calibrate(x.float())
            self.weight_quantizer.calibrate(self.weight.float())
    
    def enable_quantization(self):
        self.quantization_enabled = True
    
    def disable_quantization(self):
        self.quantization_enabled = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.linear(x, self.weight, self.bias)
        
        # Store original dtype
        input_dtype = x.dtype
        
        # Quantize activation (handles FP32 conversion internally)
        x_quant = self.activation_quantizer(x)
        
        # Quantize weights (convert to FP32, quantize, convert back)
        weight_fp32 = self.weight.float()
        weight_quant = self.weight_quantizer(weight_fp32)
        weight_quant = weight_quant.to(input_dtype)
        
        # Ensure x_quant is in the correct dtype
        x_quant = x_quant.to(input_dtype)
        
        return F.linear(x_quant, weight_quant, self.bias) 