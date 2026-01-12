import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ADC.bert_clean.core.grad_functions import *


def compute_kurtosis_loss(weight: torch.Tensor, target_kurtosis: float = 1.8) -> torch.Tensor:
    # Equation 6: κ = E[((W - μ_W) / σ_W)^4]
    mean_w = weight.mean()
    std_w = weight.std()
    std_w = torch.clamp(std_w, min=1e-6)
    
    normalized = (weight - mean_w) / std_w
    kurtosis = (normalized ** 4).mean()
    
    loss = (kurtosis - target_kurtosis) ** 2
    return loss

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
        
        init_scale = 0.01 if symmetric else 0.02  # Smaller for symmetric, slightly larger for asymmetric
        if per_channel:
            # For per-channel: create placeholder (will be reshaped in _initialize_parameters)
            self.scale = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))
            self._scale_initialized = False
        else:
            # For per-tensor: create final parameter
            self.scale = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))
            self._scale_initialized = True
        
        # Initialize zero_point if asymmetric
        if not symmetric:
            init_zp = (self.qmax + self.qmin) / 2.0
            
            if per_channel:
                # For per-channel: create placeholder
                self.zero_point = nn.Parameter(torch.tensor([init_zp], dtype=torch.float32))
                self._zp_initialized = False
            else:
                # For per-tensor: create final parameter
                self.zero_point = nn.Parameter(torch.tensor([init_zp], dtype=torch.float32))
                self._zp_initialized = True
        else:
            # For symmetric: zero_point is always 0, no gradients needed
            self.register_buffer('zero_point', torch.zeros(1))
            self._zp_initialized = True  # Always initialized (it's just zeros)
        
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
        
        self._mode = mode
        
        if mode == 'calibration' or mode == 'fixed':
            self.scale.requires_grad = False
            if not self.symmetric:
                self.zero_point.requires_grad = False
        elif mode == 'qat':
            # QAT: enable gradients, will disable EMA updates in forward()
            self.scale.requires_grad = True
            if not self.symmetric:
                self.zero_point.requires_grad = True
    
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
                # Handle zero case
                new_scale = torch.where(new_scale == 0, torch.ones_like(new_scale) * 0.1, new_scale)
                
                # Exponential moving average update
                momentum = 0.01  # Reduced momentum for stability
                self.scale.data = (1 - momentum) * self.scale.data + momentum * new_scale
                
            else:
                new_scale = (x_max - x_min) / (2 ** self.num_bits - 1)
                new_zero_point = -x_min / new_scale
                new_zero_point = torch.clamp(new_zero_point, self.qmin, self.qmax)
                
                # Exponential moving average update
                momentum = 0.01  # Reduced momentum
                self.scale.data = (1 - momentum) * self.scale.data + momentum * new_scale
                self.zero_point.data = (1 - momentum) * self.zero_point.data + momentum * new_zero_point
                
    
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
        # а как с этим вообще ashift связан то епта
        adc_signed_activations = True if ashift else signed_activations
        # Precompute constants that don't depend on ba (used in delta calculation)
        if signed_activations:
            self._activation_level_magnitude = float(2 ** (bx - 1) - 1)
        else:
            self._activation_level_magnitude = float(2 ** bx - 1)
        self._weight_level_max = float(2 ** (bw - 1) - 1)
        
        # Calculate initial delta and clipping values based on ba
        denom = float((2 ** ba) * k)
        self.delta = (2.0 * float(in_features) * self._activation_level_magnitude * self._weight_level_max) / denom

        self.na = -(2 ** (ba - 1))  # Negative clipping value
        self.pa = 2 ** (ba - 1) - 1  # Positive clipping value
        
        # Ashift constant
        if ashift:
            self.C = 2 ** (bx - 1)
        else:
            self.C = 0
    
    def set_quantizer_mode(self, mode: str):
        """
        Set mode for all quantizers in this layer.
        - 'calibration': Initialize scales using input statistics
        - 'qat': Learn scales via gradients during training
        - 'fixed': Freeze all quantization parameters
        """
        self.activation_quantizer.set_mode(mode)
        self.weight_quantizer.set_mode(mode)
    
    def set_adc_bits(self, ba: int):
        """
        Dynamically set ADC bit precision (for BitAug).
        
        This recalculates delta (Eq. 3) and clipping values based on new ba.
        Used by BitAug to pass different bit precisions during training.
        
        Args:
            ba: New ADC bit precision
        """
        self.ba = ba
        
        # Recalculate delta using precomputed constants (Paper Equation 3)
        denom = float((2 ** ba) * self.k)
        self.delta = (2.0 * float(self.in_features) * self._activation_level_magnitude * self._weight_level_max) / denom
        
        # Recalculate clipping values
        self.na = -(2 ** (ba - 1))
        self.pa = 2 ** (ba - 1) - 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        y_int = F.linear(code_x, code_w, bias=None) # добавить где надо настоящее превращение в инт

        delta = self.delta
        na = self.na
        pa = self.pa

        # Apply ADC quantization (Paper Equation 2: uses floor, not round)
        y_adc_codes = floor_ste(y_int / delta)
        y_adc_codes = torch.clamp(y_adc_codes, na, pa)

        # Dequantize back to the scale used for quantization
        adc_output = y_adc_codes * delta

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

        # 6) Add bias if present ВОТ ЭТО ВАЖНО
        if self.bias is not None:
            # print('BIAS ', self.bias)
            y_real = y_real + self.bias

        return y_real
    
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
        
        # Kurtosis loss (Paper Equation 6 & 7)
        kurtosis_loss = getattr(self, '_last_kurtosis_loss', torch.tensor(0.0))
        losses['kurtosis_loss'] = kurtosis_loss
        
        # Total loss for backprop
        losses['total'] = kurtosis_loss
        
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
                    use_kurtosis_loss=use_kurtosis_loss,
                    kurtosis_weight=kurtosis_weight,
                    target_kurtosis=target_kurtosis,
                )
            )
    
    def set_quantizer_mode(self, mode: str):
        """Set mode for all quantizers in all tiles"""
        for tile in self.tiles:
            tile.set_quantizer_mode(mode)
    
    def set_adc_bits(self, ba: int):
        """
        Dynamically set ADC bit precision for all tiles (for BitAug).
        
        Args:
            ba: New ADC bit precision
        """
        for tile in self.tiles:
            tile.set_adc_bits(ba)
    
    def get_adc_bits(self) -> int:
        """Get current ADC bit precision from first tile"""
        return self.tiles[0].ba if self.tiles else 8
    
    def get_auxiliary_losses(self) -> dict:
        """
        Get all auxiliary losses from all tiles.
        
        Returns:
            Dictionary with aggregated losses from all tiles
        """
        total_kurtosis = torch.tensor(0.0)
        
        for tile in self.tiles:
            tile_losses = tile.get_auxiliary_losses()
            total_kurtosis = total_kurtosis + tile_losses['kurtosis_loss']
        
        return {
            'kurtosis_loss': total_kurtosis,
            'total': total_kurtosis
        }

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
 