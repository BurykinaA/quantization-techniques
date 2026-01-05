import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def floor_ste(x: torch.Tensor) -> torch.Tensor:
    """Floor with Straight-Through Estimator for gradient flow"""
    return x + (torch.floor(x) - x).detach()


class SafeDivideFunction(torch.autograd.Function):
    """Division with gradient clipping by norm for the scale (denominator).
    
    Computes x / scale in forward, but clips the gradient w.r.t. scale
    in backward to prevent explosion while preserving direction.
    """
    MAX_GRAD_NORM = 1000.0  # Maximum gradient norm for scale
    
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x, scale)
        return x / scale
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        
        # Gradient for x: grad_output / scale (standard)
        grad_x = grad_output / scale
        
        # Gradient for scale: -grad_output * x / scale²
        grad_scale = -grad_output * x / (scale ** 2)
        
        # Clip by norm to prevent explosion while preserving direction
        grad_norm = grad_scale.norm()
        if grad_norm > SafeDivideFunction.MAX_GRAD_NORM:
            grad_scale = grad_scale * (SafeDivideFunction.MAX_GRAD_NORM / grad_norm)
        
        return grad_x, grad_scale


def safe_divide(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Divide x by scale with gradient clipping for scale.
    
    Forward: x / scale (unchanged)
    Backward: Clips gradient w.r.t. scale by norm to prevent explosion
    while preserving gradient direction.
    """
    return SafeDivideFunction.apply(x, scale)


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with Straight-Through Estimator for gradient flow."""
    return x + (torch.round(x) - x).detach()


def compute_kurtosis_loss(weight: torch.Tensor, target_kurtosis: float = 1.8) -> torch.Tensor:
    """
    Compute kurtosis penalty for W-reshape (Equation 6 from paper).
    
    The kurtosis penalty encourages flatter weight distributions to maximize
    Var[W] and improve ADC utilization.
    
    Args:
        weight: The weight tensor (can be quantized or not)
        target_kurtosis: Target kurtosis value (paper uses ~1.8 for uniform-like distribution)
        
    Returns:
        Kurtosis loss: (kurtosis - target)^2
    """
    # Equation 6: κ = E[((W - μ_W) / σ_W)^4]
    mean_w = weight.mean()
    std_w = weight.std()
    
    # Avoid division by zero
    std_w = torch.clamp(std_w, min=1e-6)
    
    normalized = (weight - mean_w) / std_w
    kurtosis = (normalized ** 4).mean()
    
    # Loss is squared difference from target
    # Note: Gaussian has kurtosis=3, uniform has kurtosis=1.8
    loss = (kurtosis - target_kurtosis) ** 2
    
    return loss


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
        #mask = (pre_quant > ctx.qmin) & (pre_quant < ctx.qmax)
        
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
            grad_scale = torch.mean(scale_grad_per_element, dim=dims_to_sum, keepdim=False)
            
            # Make sure the shape matches exactly
            if grad_scale.shape != original_scale.shape:
                grad_scale = grad_scale.view_as(original_scale)
        else:
            # For per-tensor, sum over all dimensions but keep as tensor with same shape as scale
            grad_scale = torch.mean(scale_grad_per_element).view_as(original_scale)
        
        # Compute gradients for zero_point (if not symmetric)
        if not ctx.symmetric:
            if ctx.per_channel:
                # For per-channel zero point - using gradient = -s approach
                zp_grad_per_element = -grad_output * scale
                zp_grad_per_element = torch.clamp(zp_grad_per_element, -1.0, 1.0)
                dims_to_sum = list(range(input.ndim))
                dims_to_sum.remove(ctx.channel_dim)
                grad_zero_point = torch.mean(zp_grad_per_element, dim=dims_to_sum, keepdim=False)
                
                if grad_zero_point.shape != original_zp.shape:
                    grad_zero_point = grad_zero_point.view_as(original_zp)
            else:
                # For per-tensor zero point
                zp_grad_per_element = -grad_output * scale
                zp_grad_per_element = torch.clamp(zp_grad_per_element, -1.0, 1.0)
                grad_zero_point = torch.mean(zp_grad_per_element).view_as(original_zp)
        else:
            grad_zero_point = None
        
        return grad_input, None, None, None, None, None, None, None, grad_scale, grad_zero_point

class ADCQuantizer(nn.Module):
    """
    ADC Quantizer implementing the quantization described in equation (2) and (3)
    """
    def __init__(self, M: int, bx: int, bw: int, ba: int, k: int = 4, signed_activations: bool = False, use_dynamic_delta: bool = True, delta_momentum: float = 0.05, use_delta_anneal: bool = True, delta_anneal_epochs: float = 1.0, delta_loss_weight: float = 0.01):
        super().__init__()
        self.M = M  # Memory dimension
        self.bx = bx  # Activation bits
        self.bw = bw  # Weight bits
        self.ba = ba  # ADC bits
        self.k = k   # Hardware design parameter
        self.signed_activations = signed_activations
        self.use_dynamic_delta = use_dynamic_delta
        self.delta_momentum = delta_momentum
        self.use_delta_anneal = use_delta_anneal
        self.delta_anneal_epochs = delta_anneal_epochs  # Number of epochs for annealing
        self.delta_loss_weight = delta_loss_weight  # Weight for delta MSE loss

        # Calculate quantization step (delta) according to equation (3)
        if signed_activations:
            activation_range = 2**(bx-1) - 1
        else:
            activation_range = 2**bx - 1

        weight_range = 2**(bw-1) - 1  # symmetric weights

        # Analytical delta calculation from paper Equation (3)
        # Unsigned: ∆a = 2M(2^bx - 1)(2^(bw-1) - 1) / (2^ba × k)
        # Signed:   ∆a = 2M(2^(bx-1) - 1)(2^(bw-1) - 1) / (2^ba × k)
        if signed_activations:
            activation_level_magnitude = float(2 ** (bx - 1) - 1)  # Fixed: added -1
        else:
            activation_level_magnitude = float(2 ** bx - 1)
        weight_level_max = float(2 ** (bw - 1) - 1) if bw > 1 else 1.0
        denom = float((2 ** ba) * k)
        self.delta = (2.0 * float(M) * activation_level_magnitude * weight_level_max) / denom

        # ADC quantization range
        self.na = -(2**(ba-1))  # Negative clipping value
        self.pa = 2**(ba-1) - 1  # Positive clipping value

        self.register_buffer('_delta', torch.tensor(self.delta, dtype=torch.float32))
        # Running absmax for dynamic delta calibration
        self.register_buffer('_running_absmax', torch.tensor(0.0, dtype=torch.float32))
        # Current epoch for annealing
        self.register_buffer('_current_epoch', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('_zero_point', torch.zeros(1))

    def set_epoch(self, epoch: float):
        """Set the current training epoch for delta annealing"""
        self._current_epoch.copy_(torch.tensor(epoch, dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ADC quantization according to equation (2):
        y_q = round(clip(y/delta, na, pa))

        Returns:
            Tuple of (quantized_output, delta_loss)
        """
        scale_for_quant = self._delta
        delta_loss = torch.tensor(0.0, device=y.device, dtype=y.dtype)

        if self.use_dynamic_delta:
            with torch.no_grad():
                current_absmax = y.detach().abs().max()
                if torch.isfinite(current_absmax):
                    if self._running_absmax.item() == 0.0:
                        self._running_absmax.copy_(current_absmax)
                    else:
                        self._running_absmax.copy_((1 - self.delta_momentum) * self._running_absmax + self.delta_momentum * current_absmax)

            dynamic_delta = torch.clamp(self._running_absmax / max(self.pa, 1), min=1e-6)

            # Add MSE loss between dynamic and analytical delta
            if self.training and self.delta_loss_weight > 0:
                delta_loss = self.delta_loss_weight * F.mse_loss(dynamic_delta, self._delta)

            if self.use_delta_anneal:
                # Epoch-based annealing: alpha from 0->1 over delta_anneal_epochs
                current_epoch = self._current_epoch.item()
                alpha = min(current_epoch / self.delta_anneal_epochs, 1.0)
                blended = (1.0 - alpha) * dynamic_delta + alpha * self._delta
                scale_for_quant = torch.clamp(blended, min=1e-3, max=100.0)
            else:
                scale_for_quant = torch.clamp(dynamic_delta, min=1e-3, max=100.0)

        # Use StraightThroughQuantize with selected delta as scale
        result = StraightThroughQuantize.apply(
            y, scale_for_quant, self._zero_point, self.na, self.pa,
            True, False, 0, scale_for_quant, self._zero_point
        )

        return result, delta_loss

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
        
        # Quantization levels (MUST be defined BEFORE using in init_zp calculation)
        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1
        
        # Initialize scale parameter with correct shape
        # Use smaller initial scale for better precision (will be updated during training)
        init_scale = 0.01 if symmetric else 0.02  # Smaller for symmetric, slightly larger for asymmetric
        if per_channel:
            # We'll set the correct size during the first forward pass
            self.register_parameter('scale', nn.Parameter(torch.ones(1) * init_scale))
            self._scale_initialized = False
        else:
            self.register_parameter('scale', nn.Parameter(torch.ones(1) * init_scale))
            self._scale_initialized = True
        
        if not symmetric:
            # Learnable zero point for asymmetric quantization
            # Initialize to middle of range for better coverage of negative values
            init_zp = (self.qmax + self.qmin) / 2.0
            if per_channel:
                self.register_parameter('zero_point', nn.Parameter(torch.ones(1) * init_zp))
                self._zp_initialized = False
            else:
                self.register_parameter('zero_point', nn.Parameter(torch.ones(1) * init_zp))
                self._zp_initialized = True
        else:
            self.register_buffer('zero_point', torch.zeros(1))
            self._zp_initialized = True
        
        # Quantizer mode: controls when parameters are updated
        # 'calibration': collect statistics, update scales via EMA, no gradients
        # 'qat': learn scales via gradients only, NO EMA updates
        # 'fixed': freeze all parameters, no updates
        self._mode = 'calibration'
        
    def set_mode(self, mode: str):
        """
        Set quantizer mode:
        - 'calibration': Initialize scales using input statistics (EMA updates, no gradients)
        - 'qat': Learn scales via gradients (gradients enabled, NO EMA updates)
        - 'fixed': Freeze scales (no updates at all)
        """
        if mode not in ['calibration', 'qat', 'fixed']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'calibration', 'qat', or 'fixed'")
        
        old_mode = self._mode
        self._mode = mode
        
        if mode == 'calibration':
            # Calibration: use EMA updates, disable gradients
            self.scale.requires_grad = False
            if not self.symmetric:
                self.zero_point.requires_grad = False
        elif mode == 'qat':
            # QAT: enable gradients, will disable EMA updates in forward()
            self.scale.requires_grad = True
            if not self.symmetric:
                self.zero_point.requires_grad = True
        elif mode == 'fixed':
            # Fixed: no updates at all
            self.scale.requires_grad = False
            if not self.symmetric:
                self.zero_point.requires_grad = False
    
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
            
            # Skip update if input is all zeros (dead activations)
            if x.abs().max() < 1e-6:
                # print("Warning: Input is all zeros, skipping quantizer update")
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
        # Initialize parameters on first forward pass
        self._initialize_parameters(x)
        
        # Respect the mode: only update via EMA in 'calibration' mode
        # In 'qat' mode: scales are updated ONLY via gradients (optimizer.step())
        # In 'fixed' mode: no updates at all
        should_update_ema = (self._mode == 'calibration')
        
        # Allow override via update_stats parameter (for backward compatibility)
        if update_stats is not None:
            should_update_ema = update_stats and (self._mode == 'calibration')
        
        if should_update_ema:
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
        
        return result

class QATLinearADC(nn.Linear):
    """
    ADC-based Quantization-Aware Training Linear layer
    
    Implements ADC quantization from the paper with:
    - Equation 2: ADC quantization with floor operation
    - Equation 3: Fixed analytical delta calculation
    - Equation 4: A-shift (activation shifting) for better ADC utilization
    - Equation 6 & 7: W-reshape (kurtosis penalty) for weight distribution reshaping
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
                 signed_activations: bool = False,
                 use_dynamic_delta: bool = True,
                 use_delta_anneal: bool = True,
                 delta_loss_weight: float = 0.01,
                 delta_anneal_epochs: float = 1.0,
                 # W-reshape (kurtosis) parameters from paper Equation 6 & 7
                 use_kurtosis_loss: bool = True,
                 kurtosis_weight: float = 0.0006,  # λ_κ in paper
                 target_kurtosis: float = 1.8):  # Target kurtosis (uniform-like)
        super().__init__(in_features, out_features, bias)
        
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k
        self.ashift = ashift
        self.signed_activations = signed_activations
        
        # W-reshape (kurtosis) parameters
        self.use_kurtosis_loss = use_kurtosis_loss
        self.kurtosis_weight = kurtosis_weight
        self.target_kurtosis = target_kurtosis
        
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
        # HACK: For A-shift, force signed delta formula (codes are signed after shift)
        adc_signed_activations = True if ashift else signed_activations
        self.adc_quantizer = ADCQuantizer(
            M=in_features,
            bx=bx,
            bw=bw,
            ba=ba,
            k=k,
            signed_activations=adc_signed_activations,  #signed_activations
            use_dynamic_delta=use_dynamic_delta,
            use_delta_anneal=use_delta_anneal,
            delta_anneal_epochs=delta_anneal_epochs,
            delta_loss_weight=delta_loss_weight
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
    
    def set_quantizer_mode(self, mode: str):
        """
        Set mode for all quantizers in this layer.
        - 'calibration': Initialize scales using input statistics
        - 'qat': Learn scales via gradients during training
        - 'fixed': Freeze all quantization parameters
        """
        self.activation_quantizer.set_mode(mode)
        self.weight_quantizer.set_mode(mode)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.linear(x, self.weight, self.bias)

        # Store raw inputs for monitoring
        x_raw = x.clone().detach()
        w_raw = self.weight.clone().detach()

        # Integer-path computation:
        # 1) Build activation codes (per-tensor quantizer)
        act_q = self.activation_quantizer
        s_x = act_q.scale  # Use original scale (gradient clipping happens in safe_divide)
        
        if act_q.symmetric:
            # Signed path (no A-shift): symmetric quantization
            # Use safe_divide for gradient clipping, round_ste for STE
            code_x = round_ste(safe_divide(x, s_x))
            qmin_x, qmax_x = act_q.qmin, act_q.qmax
            code_x = torch.clamp(code_x, qmin_x, qmax_x)
        else:
            # Unsigned path: quantize to [0, 2^bx - 1] using zero_point offset
            zp_x = act_q.zero_point
            # Use safe_divide for gradient clipping, round_ste for STE
            code_x_temp = round_ste(safe_divide(x, s_x) + zp_x)
            code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
            
            if self.ashift:
                # A-shift: subtract fixed C instead of learned zp_x
                code_x = code_x_temp - self.C
            else:
                # Standard asymmetric: subtract learned zero_point to center
                code_x = code_x_temp - zp_x

        # Store quantized activations (dequantized for comparison)
        x_quantized = code_x * s_x

        # 2) Build weight codes (per-channel symmetric, channel_dim=0)
        w_q = self.weight_quantizer
        s_w_vec = w_q.scale  # Use original scale (gradient clipping happens in safe_divide)
        # Broadcast scales to weight shape for division
        s_w_b = s_w_vec.view(-1, 1)
        
        # Use safe_divide for gradient clipping, round_ste for STE
        code_w = round_ste(safe_divide(self.weight, s_w_b))
        qmin_w, qmax_w = w_q.qmin, w_q.qmax
        code_w = torch.clamp(code_w, qmin_w, qmax_w)

        # Store quantized weights (dequantized for comparison)
        w_quantized = code_w * s_w_b

        # 3) Integer MM in code domain
        y_int = F.linear(code_x, code_w, bias=None)

        # 4) ADC quantization with dynamic delta and annealing
        adc_output, delta_loss = self._adc_quantize_with_loss(y_int)
        # Store latest delta loss for external aggregation
        self._last_delta_loss = delta_loss
        
        # 4.5) Compute kurtosis loss for W-reshape (Paper Equation 6 & 7)
        # κ = E[((W - μ_W) / σ_W)^4], loss = (κ - target)^2
        if self.training and self.use_kurtosis_loss:
            kurtosis_loss = self.kurtosis_weight * compute_kurtosis_loss(
                self.weight, self.target_kurtosis
            )
            self._last_kurtosis_loss = kurtosis_loss
        else:
            self._last_kurtosis_loss = torch.tensor(0.0, device=y_int.device, dtype=y_int.dtype)

        # Store pipeline data for monitoring if we have a monitor attached
        if hasattr(self, '_pipeline_monitor') and self._pipeline_monitor is not None:
            self._pipeline_monitor(
                layer_name=getattr(self, '_layer_name', 'unknown'),
                x_raw=x_raw,
                x_quantized=x_quantized,
                w_raw=w_raw,
                w_quantized=w_quantized,
                before_adc=y_int,
                after_adc=adc_output
            )

        # 5) Dequantize back to real domain
        # y_real = adc_output * s_x * s_w (per out channel)
        y_real = adc_output * s_x
        y_real = y_real * s_w_vec  # broadcast over out_features

        # Corrections for asymmetric quantization
        if not act_q.symmetric:
            # For asymmetric, we quantized with +zp_x but subtracted different offsets
            # This leaves a residual offset: (zp_x - offset) that must be corrected
            wq_sum = code_w.sum(dim=1)  # shape: (out_features,)
            
            if self.ashift:
                # A-shift: we subtracted C, so residual is (zp_x - C)
                # Correction: subtract zp_x contribution, add back C contribution
                zp_x = act_q.zero_point
                y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)  # remove zp offset
                y_real = y_real + (self.C * s_x) * (s_w_vec * wq_sum)  # add back C offset
            else:
                # Standard asymmetric: we subtracted zp_x, so residual is (zp_x - zp_x) = 0
                # No correction needed - the offsets cancel perfectly
                pass

        # 6) Add bias if present
        if self.bias is not None:
            y_real = y_real + self.bias

        return y_real

    def _adc_quantize_with_loss(self, y_int: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply ADC quantization with dynamic delta, annealing, and loss calculation"""
        delta = self.adc_quantizer._delta
        na = self.adc_quantizer.na
        pa = self.adc_quantizer.pa

        # Start with analytical delta
        scale_for_quant = delta
        delta_loss = torch.tensor(0.0, device=y_int.device, dtype=y_int.dtype)
        
        # Debug logging removed per user request

        if self.adc_quantizer.use_dynamic_delta:
            with torch.no_grad():
                current_absmax = y_int.detach().abs().max()
                if torch.isfinite(current_absmax):
                    if self.adc_quantizer._running_absmax.item() == 0.0:
                        self.adc_quantizer._running_absmax.copy_(current_absmax)
                    else:
                        self.adc_quantizer._running_absmax.copy_(
                            (1 - self.adc_quantizer.delta_momentum) * self.adc_quantizer._running_absmax +
                            self.adc_quantizer.delta_momentum * current_absmax
                        )

            dynamic_delta = torch.clamp(self.adc_quantizer._running_absmax / max(pa, 1), min=1e-6)

            # Add MSE loss between dynamic and analytical delta
            if self.training and self.adc_quantizer.delta_loss_weight > 0:
                delta_loss = self.adc_quantizer.delta_loss_weight * F.mse_loss(dynamic_delta, delta)

            if self.adc_quantizer.use_delta_anneal:
                current_epoch = self.adc_quantizer._current_epoch.item()
                alpha = min(current_epoch / self.adc_quantizer.delta_anneal_epochs, 1.0)
                blended = (1.0 - alpha) * dynamic_delta + alpha * delta
                scale_for_quant = torch.clamp(blended, min=1e-3, max=100.0)
            else:
                scale_for_quant = torch.clamp(dynamic_delta, min=1e-3, max=100.0)

        # Apply ADC quantization (Paper Equation 2: uses floor, not round)
        # y_bar = floor(clip(y / delta, na, pa))
        y_adc_codes = floor_ste(y_int / scale_for_quant)
        y_adc_codes = torch.clamp(y_adc_codes, na, pa)

        # Dequantize back to the scale used for quantization
        adc_output = y_adc_codes * scale_for_quant

        return adc_output, delta_loss

    def set_epoch(self, epoch: float):
        """Set the current training epoch for delta annealing"""
        if hasattr(self, 'adc_quantizer'):
            self.adc_quantizer.set_epoch(epoch)
    
    def get_auxiliary_losses(self) -> dict:
        """
        Get all auxiliary losses for this layer.
        
        Returns:
            Dictionary with:
            - 'delta_loss': MSE loss between dynamic and analytical delta (if using dynamic delta)
            - 'kurtosis_loss': W-reshape kurtosis penalty (Equation 6 & 7)
            - 'total': Sum of all losses
        """
        losses = {}
        
        # Delta loss (extension, not in paper)
        delta_loss = getattr(self, '_last_delta_loss', torch.tensor(0.0))
        losses['delta_loss'] = delta_loss
        
        # Kurtosis loss (Paper Equation 6 & 7)
        kurtosis_loss = getattr(self, '_last_kurtosis_loss', torch.tensor(0.0))
        losses['kurtosis_loss'] = kurtosis_loss
        
        # Total loss for backprop
        losses['total'] = delta_loss + kurtosis_loss
        
        return losses


class TiledLinearADC(nn.Module):
    """
    Плиточный Linear с ADC по плиткам.
    Делит входные признаки вдоль in_features, чтобы каждая плитка укладывалась в mvm_limit.
    Каждая плитка — это QATLinearADC со своим M=in_features_tile (для корректного Δa).
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 bx: int = 8,
                 bw: int = 8,
                 ba: int = 8,
                 k:  int = 4,
                 ashift: bool = False,
                 signed_activations: bool = False,
                 mvm_limit: int = 512,
                 use_dynamic_delta: bool = True,
                 use_delta_anneal: bool = True,
                 delta_loss_weight: float = 0.01,
                 delta_anneal_epochs: float = 1.0,
                 # W-reshape (kurtosis) parameters from paper
                 use_kurtosis_loss: bool = True,
                 kurtosis_weight: float = 0.0006,
                 target_kurtosis: float = 1.8,
                 logger=None):
        super().__init__()
        self.logger = logger
        self.in_features_total = in_features
        self.out_features = out_features
        self.mvm_limit = mvm_limit

        # подбираем число плиток, как в твоём Conv-варианте (делим пополам, пока не влезет)
        n_tiles = 1
        tile_in = in_features
        while (tile_in > mvm_limit) and (tile_in % 2 == 0):
            n_tiles *= 2
            tile_in //= 2
        if tile_in > mvm_limit:
            raise ValueError("in_features is not divisible by a power of 2 to meet mvm_limit")

        self.n_tiles = n_tiles
        self.in_features_tile = tile_in

        # создаём плитки; bias кладём в первую (как в TiledConv2dADC)
        self.tiles = nn.ModuleList()
        for i in range(n_tiles):
            use_bias = bias if i == 0 else False
            self.tiles.append(
                QATLinearADC(
                    in_features=tile_in,
                    out_features=out_features,
                    bias=use_bias,
                    bx=bx, bw=bw, ba=ba, k=k,
                    ashift=ashift,
                    signed_activations=signed_activations,
                    use_dynamic_delta=use_dynamic_delta,
                    use_delta_anneal=use_delta_anneal,
                    delta_loss_weight=delta_loss_weight,
                    delta_anneal_epochs=delta_anneal_epochs,
                    use_kurtosis_loss=use_kurtosis_loss,
                    kurtosis_weight=kurtosis_weight,
                    target_kurtosis=target_kurtosis,
                )
            )

    def set_epoch(self, epoch: float):
        """Set the current training epoch for delta annealing"""
        # Set epoch for all tiles
        for tile in self.tiles:
            tile.set_epoch(epoch)
    
    def set_quantizer_mode(self, mode: str):
        """Set mode for all quantizers in all tiles"""
        for tile in self.tiles:
            tile.set_quantizer_mode(mode)
    
    def get_auxiliary_losses(self) -> dict:
        """
        Get all auxiliary losses from all tiles.
        
        Returns:
            Dictionary with aggregated losses from all tiles
        """
        total_delta = torch.tensor(0.0)
        total_kurtosis = torch.tensor(0.0)
        
        for tile in self.tiles:
            tile_losses = tile.get_auxiliary_losses()
            total_delta = total_delta + tile_losses['delta_loss']
            total_kurtosis = total_kurtosis + tile_losses['kurtosis_loss']
        
        return {
            'delta_loss': total_delta,
            'kurtosis_loss': total_kurtosis,
            'total': total_delta + total_kurtosis
        }

    # ===== служебные методы управления (по аналогии с TiledConv2dADC) =====

    def enable_adc(self, indices=None):
        """Включить ADC у всех плиток или у заданных индексов."""
        idxs = range(self.n_tiles) if indices is None else indices
        for i in idxs:
            self.tiles[i].use_adc = True

    def disable_adc(self, indices=None):
        """Выключить ADC у всех плиток или у заданных индексов."""
        idxs = range(self.n_tiles) if indices is None else indices
        for i in idxs:
            self.tiles[i].use_adc = False

    def _set_quantizer_state(self, enabled: bool):
        for t in self.tiles:
            if enabled:
                t.enable_quantization()
            else:
                t.disable_quantization()

    def train(self, mode: bool = True):
        super().train(mode)
        for t in self.tiles:
            t.train(mode)
        return self

    def eval(self):
        super().eval()
        for t in self.tiles:
            t.eval()
        return self

    def load_weights(self, linear: nn.Linear):
        """
        Разложить веса исходного nn.Linear по плиткам вдоль dim=1 (in_features).
        Bias копируем в первую плитку (если есть).
        """
        w = linear.weight  # [out_features, in_features]
        if w.shape[1] != self.in_features_total:
            raise ValueError("Input linear width mismatch.")

        splits = torch.split(w, self.in_features_tile, dim=1)
        if len(splits) != self.n_tiles:
            raise RuntimeError("Unexpected number of splits; check tiling.")

        with torch.no_grad():
            for i, t in enumerate(self.tiles):
                t.weight.copy_(splits[i])
                if (linear.bias is not None) and (t.bias is not None):
                    t.bias.copy_(linear.bias)

    # ===== основной forward =====
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (..., in_features)
        if x.shape[-1] != self.in_features_total:
            raise ValueError(f"Expected last dim={self.in_features_total}, got {x.shape[-1]}")

        orig_shape = x.shape                  # (..., F)
        x2d = x.reshape(-1, self.in_features_total)   # [B*, F]
        batch_size_2d = x2d.shape[0]

        # Pre-allocate output tensor for better memory efficiency
        y2d = torch.zeros(batch_size_2d, self.out_features, dtype=x2d.dtype, device=x2d.device)

        # Process tiles with optimized memory access and minimal slicing
        tile_in = self.in_features_tile
        for i, t in enumerate(self.tiles):
            # Use narrow() for zero-copy slicing when possible, fallback to slice
            if x2d.is_contiguous():
                xi = x2d.narrow(1, i * tile_in, tile_in)  # Zero-copy slice
            else:
                xi = x2d[:, i * tile_in:(i + 1) * tile_in]  # Regular slice

            yi = t(xi)                         # [B*, out_features]
            y2d.add_(yi)  # In-place addition for better performance

        y = y2d.reshape(*orig_shape[:-1], self.out_features)  # (..., out_features)
        return y
 