import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-through estimator for quantization operations.
    Forward: apply quantization
    Backward: pass gradients through unchanged
    """
    @staticmethod
    def forward(ctx, input, quantize_fn):
        return quantize_fn(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient straight through, ignore quantize_fn gradient
        return grad_output, None

def ste_quantize(x, quantize_fn):
    """Apply quantization with straight-through gradients"""
    return StraightThroughEstimator.apply(x, quantize_fn)

class LearnableQuantizer(nn.Module):
    """
    Learnable quantizer with proper gradient flow
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
        
        # Initialize scale parameter with correct shape
        if per_channel:
            # We'll set the correct size during the first forward pass
            self.register_parameter('scale', nn.Parameter(torch.ones(1)))
            self._scale_initialized = False
        else:
            self.register_parameter('scale', nn.Parameter(torch.ones(1)))
            self._scale_initialized = True
        
        if not symmetric:
            # Learnable zero point for asymmetric quantization
            if per_channel:
                self.register_parameter('zero_point', nn.Parameter(torch.zeros(1)))
                self._zp_initialized = False
            else:
                self.register_parameter('zero_point', nn.Parameter(torch.zeros(1)))
                self._zp_initialized = True
        else:
            self.register_buffer('zero_point', torch.zeros(1))
            self._zp_initialized = True
        
        # Quantization levels
        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1
    
    def _initialize_parameters(self, x: torch.Tensor):
        """Initialize parameters with correct shape on first forward pass"""
        if self.per_channel and not self._scale_initialized:
            # Get the channel dimension size
            channel_size = x.shape[self.channel_dim]
            
            # Reinitialize scale with correct shape
            with torch.no_grad():
                self.scale.data = torch.ones(channel_size, device=x.device, dtype=x.dtype)
            self._scale_initialized = True
            
            # Reinitialize zero_point if asymmetric
            if not self.symmetric and not self._zp_initialized:
                with torch.no_grad():
                    self.zero_point.data = torch.zeros(channel_size, device=x.device, dtype=x.dtype)
                self._zp_initialized = True
    
    def update_params(self, x: torch.Tensor):
        """Update quantization parameters based on input statistics"""
        with torch.no_grad():
            if self.per_channel:
                # Per-channel quantization
                # Move channel dim to front and flatten other dims
                x_transposed = x.transpose(self.channel_dim, 0).contiguous()
                original_shape = x_transposed.shape
                x_flat = x_transposed.view(original_shape[0], -1)
                
                x_min = x_flat.min(dim=1)[0]
                x_max = x_flat.max(dim=1)[0]
            else:
                # Per-tensor quantization
                x_min = x.min()
                x_max = x.max()
            
            if self.symmetric:
                # Symmetric quantization
                x_absmax = torch.max(x_min.abs(), x_max.abs())
                scale = x_absmax / (2 ** (self.num_bits - 1) - 1)
                scale = torch.clamp(scale, min=1e-8)  # Prevent division by zero
                
                # Ensure scale has the right shape
                if self.per_channel:
                    if scale.numel() != self.scale.numel():
                        print(f"Warning: Scale shape mismatch. Expected {self.scale.shape}, got {scale.shape}")
                        return
                self.scale.data.copy_(scale)
            else:
                # Asymmetric quantization
                scale = (x_max - x_min) / (2 ** self.num_bits - 1)
                scale = torch.clamp(scale, min=1e-8)
                zero_point = -x_min / scale
                zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
                
                # Ensure shapes match
                if self.per_channel:
                    if scale.numel() != self.scale.numel():
                        print(f"Warning: Scale shape mismatch. Expected {self.scale.shape}, got {scale.shape}")
                        return
                    if zero_point.numel() != self.zero_point.numel():
                        print(f"Warning: Zero point shape mismatch. Expected {self.zero_point.shape}, got {zero_point.shape}")
                        return
                
                self.scale.data.copy_(scale)
                self.zero_point.data.copy_(zero_point)
    
    def quantize_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Quantization function (used in forward pass)"""
        if self.per_channel:
            # For per-channel, we need to reshape scale and zero_point to broadcast correctly
            shape = [1] * x.ndim
            shape[self.channel_dim] = -1
            scale = self.scale.view(shape)
            
            if not self.symmetric:
                zero_point = self.zero_point.view(shape)
        else:
            scale = self.scale
            if not self.symmetric:
                zero_point = self.zero_point
        
        if self.symmetric:
            x_scaled = x / scale
            x_quant = torch.clamp(torch.round(x_scaled), self.qmin, self.qmax)
        else:
            x_scaled = x / scale + zero_point
            x_quant = torch.clamp(torch.round(x_scaled), self.qmin, self.qmax)
            x_quant = x_quant - zero_point
        
        return x_quant * scale
    
    def forward(self, x: torch.Tensor, update_stats: bool = None) -> torch.Tensor:
        if update_stats is None:
            update_stats = self.training
        
        # Initialize parameters on first forward pass
        self._initialize_parameters(x)
        
        if update_stats:
            self.update_params(x)
        
        # Apply quantization with straight-through gradients
        return ste_quantize(x, self.quantize_fn)

class QATLinear(nn.Linear):
    """
    Quantization-Aware Training Linear layer with proper gradient flow
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
        
        # Weight quantizer (per-channel for output channels)
        self.weight_quantizer = LearnableQuantizer(
            num_bits=weight_bits,
            symmetric=weight_symmetric,
            per_channel=True,
            channel_dim=0  # Output channels are dim 0 in weight tensor
        )
        
        # Activation quantizer (per-tensor)
        self.activation_quantizer = LearnableQuantizer(
            num_bits=activation_bits,
            symmetric=activation_symmetric,
            per_channel=False
        )
        
        self.quantization_enabled = True
    
    def enable_quantization(self):
        self.quantization_enabled = True
    
    def disable_quantization(self):
        self.quantization_enabled = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.linear(x, self.weight, self.bias)
        
        # Quantize activations
        x_quant = self.activation_quantizer(x)
        
        # Quantize weights
        weight_quant = self.weight_quantizer(self.weight)
        
        # Standard linear operation with quantized inputs
        return F.linear(x_quant, weight_quant, self.bias)

class QATMultiHeadAttention(nn.Module):
    """
    Quantization-Aware Training Multi-Head Attention
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
        
        # QAT linear layers for Q, K, V projections
        self.w_q = QATLinear(d_model, d_model, bias=False, 
                            weight_bits=weight_bits, activation_bits=activation_bits)
        self.w_k = QATLinear(d_model, d_model, bias=False,
                            weight_bits=weight_bits, activation_bits=activation_bits)
        self.w_v = QATLinear(d_model, d_model, bias=False,
                            weight_bits=weight_bits, activation_bits=activation_bits)
        self.w_o = QATLinear(d_model, d_model, bias=False,
                            weight_bits=weight_bits, activation_bits=activation_bits)
        
        # Attention score quantizer
        self.attention_quantizer = LearnableQuantizer(
            num_bits=activation_bits,
            symmetric=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.d_k ** 0.5)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Generate Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Quantize attention scores
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_quantizer(attention_weights)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(context)
        return output
    
    def enable_quantization(self):
        self.w_q.enable_quantization()
        self.w_k.enable_quantization()
        self.w_v.enable_quantization()
        self.w_o.enable_quantization()
    
    def disable_quantization(self):
        self.w_q.disable_quantization()
        self.w_k.disable_quantization()
        self.w_v.disable_quantization()
        self.w_o.disable_quantization()

class QATTransformerBlock(nn.Module):
    """
    Quantization-Aware Training Transformer Block
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
        
        self.feed_forward = nn.Sequential(
            QATLinear(d_model, d_ff, weight_bits=weight_bits, activation_bits=activation_bits),
            nn.GELU(),
            QATLinear(d_ff, d_model, weight_bits=weight_bits, activation_bits=activation_bits),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x
    
    def enable_quantization(self):
        self.attention.enable_quantization()
        for module in self.feed_forward:
            if hasattr(module, 'enable_quantization'):
                module.enable_quantization()
    
    def disable_quantization(self):
        self.attention.disable_quantization()
        for module in self.feed_forward:
            if hasattr(module, 'disable_quantization'):
                module.disable_quantization() 