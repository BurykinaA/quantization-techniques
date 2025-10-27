import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class StraightThroughQuantize(torch.autograd.Function):
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
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Zero-point is a buffer (computed from stats, not learned)
        if not symmetric:
            self.register_buffer('zero_point', torch.zeros(1))
        else:
            self.register_buffer('zero_point', torch.zeros(1))
    
    def calibrate(self, x: torch.Tensor):
        """
        Calibrate quantizer parameters based on input statistics.
        Call this during warmup phase before enabling fake quantization.
        """
        with torch.no_grad():
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
        
        # Broadcast for per-channel quantization
        if self.per_channel and x.ndim > 1:
            shape = [1] * x.ndim
            shape[self.channel_dim] = -1
            scale = self.scale.view(shape)
            zero_point = self.zero_point.view(shape)
        else:
            scale = self.scale
            zero_point = self.zero_point
        
        # Apply quantization
        return StraightThroughQuantize.apply(
            x, scale, zero_point, self.qmin, self.qmax, self.symmetric,
            self.per_channel, self.channel_dim
        )

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
            self.activation_quantizer.calibrate(x)
            self.weight_quantizer.calibrate(self.weight)
    
    def enable_quantization(self):
        self.quantization_enabled = True
    
    def disable_quantization(self):
        self.quantization_enabled = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.linear(x, self.weight, self.bias)
        
        # Quantize and compute
        x_quant = self.activation_quantizer(x)
        weight_quant = self.weight_quantizer(self.weight)
        return F.linear(x_quant, weight_quant, self.bias)

class QATMultiHeadAttention(nn.Module):
    """
    Simplified QAT Multi-Head Attention
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 dropout: float = 0.1,
                 weight_bits: int = 8,
                 activation_bits: int = 8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / (self.d_k ** 0.5)
        
        # Q, K, V, O projections
        self.w_q = QATLinear(d_model, d_model, bias=False, 
                            weight_bits=weight_bits, activation_bits=activation_bits)
        self.w_k = QATLinear(d_model, d_model, bias=False,
                            weight_bits=weight_bits, activation_bits=activation_bits)
        self.w_v = QATLinear(d_model, d_model, bias=False,
                            weight_bits=weight_bits, activation_bits=activation_bits)
        self.w_o = QATLinear(d_model, d_model, bias=False,
                            weight_bits=weight_bits, activation_bits=activation_bits)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Q, K, V projections and reshape
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention and output projection
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(context)
    
    def enable_quantization(self):
        for layer in [self.w_q, self.w_k, self.w_v, self.w_o]:
            layer.enable_quantization()
    
    def disable_quantization(self):
        for layer in [self.w_q, self.w_k, self.w_v, self.w_o]:
            layer.disable_quantization()

class QATTransformerBlock(nn.Module):
    """
    Simplified QAT Transformer Block
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 weight_bits: int = 8,
                 activation_bits: int = 8):
        super().__init__()
        
        self.attention = QATMultiHeadAttention(
            d_model, num_heads, dropout, weight_bits, activation_bits
        )
        
        self.ff1 = QATLinear(d_model, d_ff, weight_bits=weight_bits, activation_bits=activation_bits)
        self.ff2 = QATLinear(d_ff, d_model, weight_bits=weight_bits, activation_bits=activation_bits)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention + residual
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward + residual
        ff_output = self.ff2(F.gelu(self.ff1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
    
    def enable_quantization(self):
        self.attention.enable_quantization()
        self.ff1.enable_quantization()
        self.ff2.enable_quantization()
    
    def disable_quantization(self):
        self.attention.disable_quantization()
        self.ff1.disable_quantization()
        self.ff2.disable_quantization() 