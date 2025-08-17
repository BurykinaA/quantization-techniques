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
        
        # Compute pre-quantized values to determine where clamping occurs
        if ctx.symmetric:
            pre_quant = input / scale
        else:
            pre_quant = input / scale + zero_point
        
        # Create mask for values that are NOT clamped
        mask = (pre_quant > ctx.qmin) & (pre_quant < ctx.qmax)
        
        # Apply straight-through only where not clamped
        grad_input = grad_output #* mask.float()
        
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
                # For per-channel zero point - using gradient = -s approach
                zp_grad_per_element = -grad_output * scale
                dims_to_sum = list(range(input.ndim))
                dims_to_sum.remove(ctx.channel_dim)
                grad_zero_point = torch.sum(zp_grad_per_element, dim=dims_to_sum, keepdim=False)
                
                if grad_zero_point.shape != original_zp.shape:
                    grad_zero_point = grad_zero_point.view_as(original_zp)
            else:
                # For per-tensor zero point
                zp_grad_per_element = -grad_output * scale
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
        
        # Delta calculation from equation (3) - corrected formula
        self.delta = (2 * M * activation_range * weight_range) / (2**ba * k)
        
        # Add reasonable bounds to prevent numerical issues
        # self.delta = max(self.delta, 1e-2)  # Minimum bound
        # self.delta = min(self.delta, 1e4)   # Maximum bound to prevent overflow
        
        # ADC quantization range
        self.na = -(2**(ba-1))  # Negative clipping value
        self.pa = 2**(ba-1) - 1  # Positive clipping value
        
        self.register_buffer('_delta', torch.tensor(self.delta, dtype=torch.float32))
        self.register_buffer('_zero_point', torch.zeros(1))
        # self.register_buffer('_tmp_scale_1', torch.ones(1))
        
        print(f"ADC Quantizer: M={M}, delta={self.delta:.6f}, range=[{self.na}, {self.pa}]")
        
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply ADC quantization according to equation (2):
        y_q = round(clip(y/delta, na, pa))
        """
        # Check input for NaN/inf
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"Warning: NaN/inf in ADC input, max={y.max().item()}, min={y.min().item()}")
            raise
            y = torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Use StraightThroughQuantize with fixed delta as scale
        result = StraightThroughQuantize.apply(
            y, self._delta, self._zero_point, self.na, self.pa,
            True, False, 0, self._delta, self._zero_point
        )
        
        # Check output for NaN/inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"Warning: NaN/inf in ADC output, clamping...")
            raise
            result = torch.nan_to_num(result, nan=0.0, posinf=self.pa, neginf=self.na)
        
        return result

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
            self.register_parameter('scale', nn.Parameter(torch.ones(1) * 0.1))  # Better initial value
            self._scale_initialized = False
        else:
            self.register_parameter('scale', nn.Parameter(torch.ones(1) * 0.1))  # Better initial value
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
                #init_scale = torch.clamp(init_scale, min=1e-3, max=10.0)
                
                # Handle case where x_absmax is 0
                init_scale = torch.where(init_scale == 0, torch.ones_like(init_scale) * 0.1, init_scale)
                
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
            # Check input for NaN/inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Warning: NaN/inf in quantizer input, skipping parameter update")
                return
                
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
                # Ensure scale is never too small or too large
                #new_scale = torch.clamp(new_scale, min=1e-3, max=10.0)
                
                # Handle zero case
                new_scale = torch.where(new_scale == 0, torch.ones_like(new_scale) * 0.1, new_scale)
                
                # Exponential moving average update
                momentum = 0.01  # Reduced momentum for stability
                self.scale.data = (1 - momentum) * self.scale.data + momentum * new_scale
                
                # Clamp the final scale
                #self.scale.data = torch.clamp(self.scale.data, min=1e-3, max=10.0)
            else:
                new_scale = (x_max - x_min) / (2 ** self.num_bits - 1)
                # Ensure scale is never too small
                #new_scale = torch.clamp(new_scale, min=1e-3, max=10.0)
                new_zero_point = -x_min / new_scale
                new_zero_point = torch.clamp(new_zero_point, self.qmin, self.qmax)
                
                # Exponential moving average update
                momentum = 0.01  # Reduced momentum
                self.scale.data = (1 - momentum) * self.scale.data + momentum * new_scale
                self.zero_point.data = (1 - momentum) * self.zero_point.data + momentum * new_zero_point
                
                # Clamp final values
                #self.scale.data = torch.clamp(self.scale.data, min=1e-3, max=10.0)
    
    def forward(self, x: torch.Tensor, update_stats: bool = None) -> torch.Tensor:
        if update_stats is None:
            update_stats = self.training
        
        # Check input for NaN/inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/inf in quantizer input, range=[{x.min():.3f}, {x.max():.3f}]")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Initialize parameters on first forward pass
        self._initialize_parameters(x)
        
        if update_stats:
            self.update_params(x)
        
        # Ensure scale is valid before using
        if torch.isnan(self.scale).any() or (self.scale <= 0).any():
            print("Warning: Invalid scale detected, resetting")
            with torch.no_grad():
                if self.per_channel:
                    self.scale.data.fill_(0.1)
                else:
                    self.scale.data.fill_(0.1)
        
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
        result = StraightThroughQuantize.apply(
            x, scale_broadcasted, zero_point_broadcasted, self.qmin, self.qmax, 
            self.symmetric, self.per_channel, self.channel_dim,
            self.scale, self.zero_point  # Original parameters for gradient computation
        )
        
        # Check output for issues
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN/inf in quantizer output")
            result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return result

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
        
        # Ensure scales are not too small or too large
        # x_scale = torch.clamp(x_scale, min=1e-6, max=1e3)
        # w_scale = torch.clamp(w_scale, min=1e-6, max=1e3)
        
        if not self.signed_activations:
            x_zp = self.activation_quantizer.zero_point
        else:
            x_zp = torch.zeros_like(x_scale)
        
        # Check inputs
        if torch.isnan(yq_adc).any():
            print("Warning: NaN in yq_adc input to dequantize")
            yq_adc = torch.nan_to_num(yq_adc, nan=0.0)
        
        # Dequantize: y = yq_adc * delta
        y = yq_adc * self.adc_quantizer._delta
        
        # Check for overflow after multiplication
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"Warning: Overflow after delta multiplication. Delta={self.adc_quantizer._delta}, yq_adc range=[{yq_adc.min():.3f}, {yq_adc.max():.3f}]")
            y = torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Add ashift correction if enabled
        if self.ashift:
            ashift_correction = self.C * wq.sum(axis=-1)
            if torch.isnan(ashift_correction).any():
                print("Warning: NaN in ashift correction")
                ashift_correction = torch.nan_to_num(ashift_correction, nan=0.0)
            y = y + ashift_correction
        
        # Subtract zero-point correction
        if not self.signed_activations:
            # For zero-point correction: y = y - (x_zp / w_scale) * weight_sum
            weight_sum = self.weight.sum(axis=-1)  # Sum over input features, shape: (out_features,)
            
            # Check weight_sum for issues
            if torch.isnan(weight_sum).any():
                print("Warning: NaN in weight_sum")
                weight_sum = torch.nan_to_num(weight_sum, nan=0.0)
            
            correction = (x_zp / w_scale) * weight_sum  # Both should broadcast to (out_features,)
            
            # Check correction for issues
            if torch.isnan(correction).any():
                print("Warning: NaN in zero-point correction")
                correction = torch.nan_to_num(correction, nan=0.0)
            
            y = y - correction
        
        # Scale back to full precision with careful handling
        # y: (batch_size, out_features)
        # x_scale: scalar or (1,)
        # w_scale: (out_features,)
        
        # Check intermediate values
        if torch.isnan(y).any():
            print("Warning: NaN before final scaling")
            y = torch.nan_to_num(y, nan=0.0)
        
        # Apply scaling in stages to prevent overflow
        y = y * x_scale
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: Overflow after x_scale multiplication")
            y = torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        
        y = y * w_scale
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: Overflow after w_scale multiplication")
            y = torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.linear(x, self.weight, self.bias)
        
        #print('===================')
        #print('input ', 'max:', torch.max(x),' min:', torch.min(x), ' M:', self.in_features)
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
        #yq_adc = self.adc_quantizer(y_for_adc)
        
        # Dequantize
        #out = self.dequantize(yq_adc, wq)
        #out = yq_adc

        out = self.adc_quantizer(y_for_adc)
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
            
        #print('output', torch.max(out), torch.min(out))
        #print()
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

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    
    print("="*50)
    print("Testing ADC Layers")
    print("="*50)
    
    # Set up for gradient tracking
    torch.manual_seed(42)
    
    def check_tensor(tensor, name):
        """Helper function to check tensor for issues"""
        if tensor is None:
            print(f"{name}: None")
            return
        
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        print(f"{name}: shape={tensor.shape}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, nan={has_nan}, inf={has_inf}")
        
        if has_nan or has_inf:
            print(f"  WARNING: {name} contains NaN or inf!")
            return False
        return True
    
    # Test 1: LearnableQuantizer
    print("\n1. Testing LearnableQuantizer")
    print("-" * 30)
    
    # Test per-tensor symmetric quantizer (like weight quantizer)
    weight_quantizer = LearnableQuantizer(num_bits=8, symmetric=True, per_channel=False)
    dummy_weight = torch.randn(10, 5, requires_grad=True) * 0.1  # Small weights
    check_tensor(dummy_weight, "Input weights")
    
    print("Weight quantizer forward pass...")
    quantized_weight = weight_quantizer(dummy_weight)
    check_tensor(quantized_weight, "Quantized weights")
    check_tensor(weight_quantizer.scale, "Weight scale")
    
    # Test per-channel symmetric quantizer  
    weight_quantizer_pc = LearnableQuantizer(num_bits=8, symmetric=True, per_channel=True, channel_dim=0)
    print("Per-channel weight quantizer forward pass...")
    quantized_weight_pc = weight_quantizer_pc(dummy_weight)
    check_tensor(quantized_weight_pc, "PC Quantized weights")
    check_tensor(weight_quantizer_pc.scale, "PC Weight scale")
    
    # Test per-tensor asymmetric quantizer (like activation quantizer)
    act_quantizer = LearnableQuantizer(num_bits=8, symmetric=False, per_channel=False)
    dummy_activation = torch.randn(3, 10, requires_grad=True) * 0.5 + 0.5  # Positive activations
    check_tensor(dummy_activation, "Input activations")
    
    print("Activation quantizer forward pass...")
    quantized_activation = act_quantizer(dummy_activation)
    check_tensor(quantized_activation, "Quantized activations")
    check_tensor(act_quantizer.scale, "Activation scale")
    check_tensor(act_quantizer.zero_point, "Activation zero_point")
    
    # Test 2: ADCQuantizer
    print("\n2. Testing ADCQuantizer")
    print("-" * 30)
    
    adc_quantizer = ADCQuantizer(M=5, bx=8, bw=8, ba=8, k=4)
    
    # Simulate matrix multiplication output
    dummy_mm_output = torch.randn(3, 10, requires_grad=True) * 10  # Matrix mult output
    check_tensor(dummy_mm_output, "Matrix mult output")
    
    print("ADC quantizer forward pass...")
    adc_output = adc_quantizer(dummy_mm_output)
    check_tensor(adc_output, "ADC output")
    print(f"ADC delta: {adc_quantizer._delta.item():.6f}")
    
    # Test 3: QATLinearADC
    print("\n3. Testing QATLinearADC")
    print("-" * 30)
    
    linear_adc = QATLinearADC(in_features=5, out_features=10, bias=True, 
                              bx=8, bw=8, ba=8, k=4, ashift=False)
    
    dummy_input = torch.randn(3, 5, requires_grad=True) * 0.5  # Small input
    check_tensor(dummy_input, "Linear input")
    check_tensor(linear_adc.weight, "Linear weight")
    check_tensor(linear_adc.bias, "Linear bias")
    
    print("QATLinearADC forward pass...")
    
    # Step by step forward pass with logging
    print("  Step 1: Quantize activations")
    xq = linear_adc.activation_quantizer(dummy_input)
    check_tensor(xq, "  Quantized activations")
    check_tensor(linear_adc.activation_quantizer.scale, "  Act scale")
    
    print("  Step 2: Quantize weights")
    wq = linear_adc.weight_quantizer(linear_adc.weight)
    check_tensor(wq, "  Quantized weights")
    check_tensor(linear_adc.weight_quantizer.scale, "  Weight scale")
    
    print("  Step 3: Matrix multiplication")
    y_for_adc = F.linear(xq, wq, bias=None)
    check_tensor(y_for_adc, "  MM output")
    
    print("  Step 4: ADC quantization")
    yq_adc = linear_adc.adc_quantizer(y_for_adc)
    check_tensor(yq_adc, "  ADC quantized")
    
    print("  Step 5: Dequantization")
    try:
        dequant_output = linear_adc.dequantize(yq_adc, wq)
        check_tensor(dequant_output, "  Dequantized")
    except Exception as e:
        print(f"  ERROR in dequantization: {e}")
    
    print("  Step 6: Full forward")
    try:
        final_output = linear_adc(dummy_input)
        check_tensor(final_output, "  Final output")
    except Exception as e:
        print(f"  ERROR in forward: {e}")
    
    # Test 4: Gradient flow
    print("\n4. Testing Gradient Flow")
    print("-" * 30)
    
    try:
        # Create fresh layer for gradient test
        test_layer = QATLinearADC(in_features=5, out_features=2, bias=True,
                                  bx=8, bw=8, ba=8, k=4, ashift=False)
        test_input = torch.randn(2, 5, requires_grad=True) * 0.1
        
        print("Forward pass...")
        output = test_layer(test_input)
        check_tensor(output, "Test output")
        
        print("Backward pass...")
        loss = output.sum()
        print(f"Loss: {loss.item():.6f}")
        
        loss.backward()
        
        print("Checking gradients...")
        check_tensor(test_input.grad, "Input grad")
        check_tensor(test_layer.weight.grad, "Weight grad")
        check_tensor(test_layer.bias.grad, "Bias grad")
        check_tensor(test_layer.activation_quantizer.scale.grad, "Act scale grad")
        check_tensor(test_layer.weight_quantizer.scale.grad, "Weight scale grad")
        
        # Check for exploding gradients
        if test_layer.weight.grad is not None:
            grad_norm = test_layer.weight.grad.norm().item()
            print(f"Weight gradient norm: {grad_norm:.6f}")
            if grad_norm > 100:
                print("WARNING: Gradient norm is very large!")
        
    except Exception as e:
        print(f"ERROR in gradient test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("Testing complete")
    print("="*50)
 