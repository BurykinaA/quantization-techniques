import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-through estimator for quantization that allows gradients to flow to scale parameters
    """
    @staticmethod
    def forward(ctx, input, scale, zero_point, qmin, qmax, symmetric, per_channel, channel_dim, original_scale, original_zp):
        ctx.save_for_backward(input, original_scale, original_zp)  # Save original parameters for gradient computation
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.symmetric = symmetric
        ctx.per_channel = per_channel
        ctx.channel_dim = channel_dim
        
        # Perform quantization using the broadcasted scale/zero_point
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
        input, original_scale, original_zp = ctx.saved_tensors
        
        # For gradient computation, we need to use the broadcasted versions
        if ctx.per_channel:
            # Recreate the broadcasted scale and zero_point
            shape = [1] * input.ndim
            shape[ctx.channel_dim] = -1
            scale = original_scale.view(shape)
            
            if not ctx.symmetric:
                zero_point = original_zp.view(shape)
            else:
                zero_point = torch.zeros_like(scale)
        else:
            scale = original_scale
            if not ctx.symmetric:
                zero_point = original_zp
            else:
                zero_point = torch.zeros_like(scale)
        
        # Straight-through for input gradients
        grad_input = grad_output
        
        # Compute gradients for scale parameter
        if ctx.symmetric:
            quantized_levels = torch.clamp(torch.round(input / scale), ctx.qmin, ctx.qmax)
            # Gradient of output w.r.t. scale
            scale_grad_per_element = grad_output * (quantized_levels - input / scale)
        else:
            x_scaled = input / scale + zero_point
            quantized_levels = torch.clamp(torch.round(x_scaled), ctx.qmin, ctx.qmax) - zero_point
            scale_grad_per_element = grad_output * (quantized_levels - input / scale)
        
        # Sum over appropriate dimensions to match original scale shape
        if ctx.per_channel:
            # For per-channel, sum over all dimensions except the channel dimension
            dims_to_sum = list(range(input.ndim))
            dims_to_sum.remove(ctx.channel_dim)
            grad_scale = torch.sum(scale_grad_per_element, dim=dims_to_sum, keepdim=False)
            
            # Make sure the shape matches exactly
            if grad_scale.shape != original_scale.shape:
                grad_scale = grad_scale.view_as(original_scale)
        else:
            # For per-tensor, sum over all dimensions but keep as tensor with same shape as scale
            grad_scale = torch.sum(scale_grad_per_element).view_as(original_scale)
        
        # Compute gradients for zero_point (if not symmetric)
        if not ctx.symmetric:
            if ctx.per_channel:
                # For per-channel zero point
                zp_grad_per_element = grad_output * scale
                dims_to_sum = list(range(input.ndim))
                dims_to_sum.remove(ctx.channel_dim)
                grad_zero_point = torch.sum(zp_grad_per_element, dim=dims_to_sum, keepdim=False)
                
                if grad_zero_point.shape != original_zp.shape:
                    grad_zero_point = grad_zero_point.view_as(original_zp)
            else:
                # For per-tensor zero point
                zp_grad_per_element = grad_output * scale
                grad_zero_point = torch.sum(zp_grad_per_element).view_as(original_zp)
        else:
            grad_zero_point = None
        
        return grad_input, None, None, None, None, None, None, None, grad_scale, grad_zero_point

class ADCQuantizer(nn.Module):
    """
    ADC Quantizer implementing the quantization described in equation (2) and (3)
    """
    def __init__(self, M: int, bx: int, bw: int, ba: int, k: int = 4, signed_activations: bool = False):
        super().__init__()
        self.M = M  # Memory dimension
        self.bx = bx  # Activation bits
        self.bw = bw  # Weight bits
        self.ba = ba  # ADC bits
        self.k = k   # Hardware design parameter
        self.signed_activations = signed_activations
        
        # Calculate quantization step (delta) according to equation (3)
        if signed_activations:
            activation_range = 2**(bx-1) - 1
        else:
            activation_range = 2**bx - 1
            
        weight_range = 2**(bw-1) - 1  # Assuming symmetric quantization for weights
        
        # Delta calculation from equation (3)
        self.delta = (2 * M * activation_range * weight_range) / (2**ba * k)
        self.delta = max(self.delta, 1e-6)
        
        # ADC quantization range
        self.na = -(2**(ba-1))  # Negative clipping value
        self.pa = 2**(ba-1) - 1  # Positive clipping value
        
        self.register_buffer('_delta', torch.tensor(self.delta, dtype=torch.float32))
        self.register_buffer('_zero_point', torch.zeros(1))
        
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply ADC quantization according to equation (2):
        y_q = round(clip(y/delta, na, pa))
        """
        # Use StraightThroughQuantize with fixed delta as scale
        return StraightThroughQuantize.apply(
            y, self._delta, self._zero_point, self.na, self.pa,
            True, False, 0, self._delta, self._zero_point
        )

class LearnableQuantizer(nn.Module):
    """
    Learnable quantizer with proper gradient flow to scale parameters
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
            
            # Calculate initial scale values
            with torch.no_grad():
                if self.channel_dim == 0:
                    x_reshaped = x.view(channel_size, -1)
                    x_absmax = x_reshaped.abs().max(dim=1)[0]
                else:
                    # Handle other channel dimensions
                    x_transposed = x.transpose(self.channel_dim, 0)
                    x_reshaped = x_transposed.contiguous().view(channel_size, -1)
                    x_absmax = x_reshaped.abs().max(dim=1)[0]
                
                init_scale = x_absmax / (2 ** (self.num_bits - 1) - 1)
                # Ensure scale is never too small
                init_scale = torch.clamp(init_scale, min=1e-4)
                
                # Resize the existing parameter instead of creating new one
                self.scale.data = self.scale.data.new_zeros(channel_size)
                self.scale.data.copy_(init_scale)
            
            self._scale_initialized = True
            
            # Reinitialize zero_point if asymmetric
            if not self.symmetric and not self._zp_initialized:
                with torch.no_grad():
                    self.zero_point.data = self.zero_point.data.new_zeros(channel_size)
                self._zp_initialized = True
    
    def update_params(self, x: torch.Tensor):
        """Update quantization parameters based on input statistics (for initialization)"""
        if not self.training:
            return  # Only update during training
            
        with torch.no_grad():
            if self.per_channel:
                # Per-channel quantization
                if self.channel_dim == 0:
                    x_reshaped = x.view(x.shape[0], -1)
                    x_min = x_reshaped.min(dim=1)[0]
                    x_max = x_reshaped.max(dim=1)[0]
                else:
                    # Handle other channel dimensions
                    x_transposed = x.transpose(self.channel_dim, 0).contiguous()
                    x_reshaped = x_transposed.view(x_transposed.shape[0], -1)
                    x_min = x_reshaped.min(dim=1)[0]
                    x_max = x_reshaped.max(dim=1)[0]
            else:
                # Per-tensor quantization
                x_min = x.min()
                x_max = x.max()
            
            # Only update if scale is very different from current (avoid oscillation)
            if self.symmetric:
                x_absmax = torch.max(x_min.abs(), x_max.abs())
                new_scale = x_absmax / (2 ** (self.num_bits - 1) - 1)
                # Ensure scale is never too small
                new_scale = torch.clamp(new_scale, min=1e-4)
                
                # Exponential moving average update
                momentum = 0.1
                self.scale.data = (1 - momentum) * self.scale.data + momentum * new_scale
            else:
                new_scale = (x_max - x_min) / (2 ** self.num_bits - 1)
                # Ensure scale is never too small
                new_scale = torch.clamp(new_scale, min=1e-4)
                new_zero_point = -x_min / new_scale
                new_zero_point = torch.clamp(new_zero_point, self.qmin, self.qmax)
                
                # Exponential moving average update
                momentum = 0.1
                self.scale.data = (1 - momentum) * self.scale.data + momentum * new_scale
                self.zero_point.data = (1 - momentum) * self.zero_point.data + momentum * new_zero_point
    
    def forward(self, x: torch.Tensor, update_stats: bool = None) -> torch.Tensor:
        if update_stats is None:
            update_stats = self.training
        
        # Initialize parameters on first forward pass
        self._initialize_parameters(x)
        
        if update_stats:
            self.update_params(x)
        
        # Prepare scale and zero_point for broadcasting
        if self.per_channel:
            # For per-channel, we need to reshape scale and zero_point to broadcast correctly
            shape = [1] * x.ndim
            shape[self.channel_dim] = -1
            scale_broadcasted = self.scale.view(shape)
            
            if not self.symmetric:
                zero_point_broadcasted = self.zero_point.view(shape)
            else:
                zero_point_broadcasted = torch.zeros_like(scale_broadcasted)
        else:
            scale_broadcasted = self.scale
            if not self.symmetric:
                zero_point_broadcasted = self.zero_point
            else:
                zero_point_broadcasted = torch.zeros_like(scale_broadcasted)
        
        # Apply quantization with learnable parameters
        # Pass both broadcasted versions (for computation) and original versions (for gradients)
        return StraightThroughQuantize.apply(
            x, scale_broadcasted, zero_point_broadcasted, self.qmin, self.qmax, 
            self.symmetric, self.per_channel, self.channel_dim,
            self.scale, self.zero_point  # Original parameters for gradient computation
        )

class QATLinearADC(nn.Linear):
    """
    ADC-based Quantization-Aware Training Linear layer
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 bx: int = 8,  # Activation bits
                 bw: int = 8,  # Weight bits
                 ba: int = 8,  # ADC bits
                 k: int = 4,   # Hardware design parameter
                 ashift: bool = False,
                 signed_activations: bool = False):
        super().__init__(in_features, out_features, bias)
        
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k
        self.ashift = ashift
        self.signed_activations = signed_activations
        
        # Activation quantizer (affine for unsigned, symmetric for signed)
        self.activation_quantizer = LearnableQuantizer(
            num_bits=bx,
            symmetric=signed_activations,
            per_channel=False
        )
        
        # Weight quantizer (per-channel symmetric)
        self.weight_quantizer = LearnableQuantizer(
            num_bits=bw,
            symmetric=True,
            per_channel=True,
            channel_dim=0  # Output channels are dim 0 in weight tensor
        )
        
        # ADC quantizer
        self.adc_quantizer = ADCQuantizer(
            M=in_features, 
            bx=bx, 
            bw=bw, 
            ba=ba, 
            k=k,
            signed_activations=signed_activations
        )
        
        # Ashift constant
        if ashift:
            self.C = 2 ** (bx - 1)
        else:
            self.C = 0
        
        self.quantization_enabled = True
    
    def enable_quantization(self):
        self.quantization_enabled = True
    
    def disable_quantization(self):
        self.quantization_enabled = False
    
    def dequantize(self, yq_adc: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
        """
        Dequantize ADC output back to full precision
        """
        # Get quantization parameters
        x_scale = self.activation_quantizer.scale
        w_scale = self.weight_quantizer.scale
        
        # Ensure scales are not too small
        x_scale = torch.clamp(x_scale, min=1e-6)
        w_scale = torch.clamp(w_scale, min=1e-6)
        
        if not self.signed_activations:
            x_zp = self.activation_quantizer.zero_point
        else:
            x_zp = torch.zeros_like(x_scale)
        
        # Dequantize: y = yq_adc * delta
        y = yq_adc * self.adc_quantizer._delta
        
        # Add ashift correction if enabled
        if self.ashift:
            y = y + self.C * wq.sum(axis=-1)
        
        # Subtract zero-point correction
        if not self.signed_activations:
            # For zero-point correction: y = y - (x_zp / w_scale) * weight_sum
            weight_sum = self.weight.sum(axis=-1)  # Sum over input features, shape: (out_features,)
            correction = (x_zp / w_scale) * weight_sum  # Both should broadcast to (out_features,)
            # y has shape (batch_size, out_features), correction has shape (out_features,)
            y = y - correction
        
        # Scale back to full precision
        # y: (batch_size, out_features)
        # x_scale: scalar or (1,)
        # w_scale: (out_features,)
        y = y * x_scale * w_scale
        
        # Check for NaN and clamp if necessary
        if torch.isnan(y).any():
            print("Warning: NaN detected in dequantize output, clamping...")
            y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.linear(x, self.weight, self.bias)
        
        # Quantize activations
        xq = self.activation_quantizer(x)
        
        # Apply ashift if enabled
        if self.ashift:
            xq = xq - self.C
        
        # Quantize weights
        wq = self.weight_quantizer(self.weight)
        
        # Compute matrix-vector multiplication
        y_for_adc = F.linear(xq, wq, bias=None)  # No bias here, add later
        
        # Apply ADC quantization
        yq_adc = self.adc_quantizer(y_for_adc)
        
        # Dequantize
        out = self.dequantize(yq_adc, wq)
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
            
        return out

class QATMultiHeadAttentionADC(nn.Module):
    """
    ADC-based Quantization-Aware Training Multi-Head Attention
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 dropout: float = 0.1,
                 bx: int = 8,
                 bw: int = 8,
                 ba: int = 8,
                 k: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # ADC-based QAT linear layers for Q, K, V projections
        self.w_q = QATLinearADC(d_model, d_model, bias=False, 
                               bx=bx, bw=bw, ba=ba, k=k)
        self.w_k = QATLinearADC(d_model, d_model, bias=False,
                               bx=bx, bw=bw, ba=ba, k=k)
        self.w_v = QATLinearADC(d_model, d_model, bias=False,
                               bx=bx, bw=bw, ba=ba, k=k)
        self.w_o = QATLinearADC(d_model, d_model, bias=False,
                               bx=bx, bw=bw, ba=ba, k=k)
        
        # Attention score quantizer
        self.attention_quantizer = LearnableQuantizer(
            num_bits=ba,
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

class QATTransformerBlockADC(nn.Module):
    """
    ADC-based Quantization-Aware Training Transformer Block
    """
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 bx: int = 8,
                 bw: int = 8,
                 ba: int = 8,
                 k: int = 4):
        super().__init__()
        
        self.attention = QATMultiHeadAttentionADC(
            d_model, num_heads, dropout, bx, bw, ba, k
        )
        
        self.feed_forward = nn.Sequential(
            QATLinearADC(d_model, d_ff, bx=bx, bw=bw, ba=ba, k=k),
            nn.GELU(),
            QATLinearADC(d_ff, d_model, bx=bx, bw=bw, ba=ba, k=k),
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