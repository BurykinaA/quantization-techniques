import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ADC.llama.core.grad_functions import *


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
        # Ensure scale is on the same device as input
        s_x = act_q.scale.to(x.device)
        
        if act_q.symmetric:
            # Signed path (no A-shift): symmetric quantization
            # Use safe_divide for gradient clipping, round_ste for STE
            code_x = round_ste(safe_divide(x, s_x))
            qmin_x, qmax_x = act_q.qmin, act_q.qmax
            code_x = torch.clamp(code_x, qmin_x, qmax_x)
        else:
            # Unsigned path: quantize to [0, 2^bx - 1] using zero_point offset
            zp_x = act_q.zero_point.to(x.device)
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
        # Ensure scale is on the same device as weights
        s_w_vec = w_q.scale.to(self.weight.device)
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
                zp_x = act_q.zero_point.to(y_real.device)
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


class LoRAQATLinearADC(nn.Module):
    """
    ADC-LoRA: Low-Rank Adaptation for ADC quantization (Paper Section 3.4).
    
    Implements Equation 11: Y = QA(Qx(X)Qw(W + AB))
    
    Where:
    - W is the frozen pretrained weight
    - A ∈ R^{out_features × r}, B ∈ R^{r × in_features} are learnable low-rank matrices
    - r << min(out_features, in_features) is the rank
    - The effective weight (W + scaling * A @ B) is quantized through the ADC pipeline
    
    This reduces trainable parameters dramatically while maintaining ADC quantization.
    """
    
    def __init__(self,
                 base_layer: QATLinearADC,
                 r: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        """
        Args:
            base_layer: The QATLinearADC layer to wrap with LoRA
            r: LoRA rank (dimension of low-rank matrices)
            alpha: LoRA scaling factor (scaling = alpha / r)
            dropout: Dropout rate applied to LoRA path
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
        
        # LoRA parameters: A (out_features, r), B (r, in_features)
        # Following standard LoRA: A initialized with Kaiming, B initialized with zeros
        # This ensures initial output is identical to base layer (A @ B = 0)
        self.lora_A = nn.Parameter(torch.zeros(out_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, in_features))
        
        # Initialize A with Kaiming uniform (standard LoRA initialization)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B stays zero so initial A @ B = 0
        
        # Optional dropout for regularization
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Store dimensions for reference
        self.in_features = in_features
        self.out_features = out_features
    
    def set_quantizer_mode(self, mode: str):
        """Delegate to base layer"""
        self.base_layer.set_quantizer_mode(mode)
    
    def set_adc_bits(self, ba: int):
        """Delegate to base layer for BitAug"""
        self.base_layer.set_adc_bits(ba)
    
    def get_auxiliary_losses(self) -> dict:
        """Get auxiliary losses including kurtosis on effective weight"""
        # Compute kurtosis on effective weight (W + LoRA)
        effective_weight = self._get_effective_weight()
        
        if self.training and self.base_layer.use_kurtosis_loss:
            kurtosis_loss = self.base_layer.kurtosis_weight * compute_kurtosis_loss(
                effective_weight, self.base_layer.target_kurtosis
            )
        else:
            kurtosis_loss = torch.tensor(0.0, device=effective_weight.device)
        
        return {
            'kurtosis_loss': kurtosis_loss,
            'total': kurtosis_loss
        }
    
    def _get_effective_weight(self) -> torch.Tensor:
        """Compute effective weight: W + scaling * A @ B"""
        lora_weight = self.scaling * (self.lora_A @ self.lora_B)
        return self.base_layer.weight + lora_weight
    
    def compute_reference_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reference output: Qx(X) @ Qw(W) - quantized but NO ADC, NO LoRA.
        
        This is the target for MSE warmup optimization (Paper Equation 12).
        We want to find A, B such that:
            ||Qx(X)Qw(W) - QA(Qx(X)Qw(W + AB))||^2_F is minimized
        
        Returns:
            Reference output in real domain (dequantized)
        """
        # 1) Build activation codes (same as forward)
        act_q = self.base_layer.activation_quantizer
        s_x = act_q.scale.to(x.device)
        
        if act_q.symmetric:
            code_x = round_ste(safe_divide(x, s_x))
            code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
        else:
            zp_x = act_q.zero_point.to(x.device)
            code_x_temp = round_ste(safe_divide(x, s_x) + zp_x)
            code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
            
            if self.base_layer.ashift:
                code_x = code_x_temp - self.base_layer.C
            else:
                code_x = code_x_temp - zp_x
        
        # 2) Build weight codes using BASE weight W only (NO LoRA!)
        w_q = self.base_layer.weight_quantizer
        s_w_vec = w_q.scale.to(self.base_layer.weight.device)
        s_w_b = s_w_vec.view(-1, 1)
        
        code_w = round_ste(safe_divide(self.base_layer.weight, s_w_b))
        code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)
        
        # 3) Integer MM (NO ADC quantization!)
        y_int = F.linear(code_x, code_w, bias=None)
        
        # 4) Dequantize back to real domain (skip ADC step)
        y_real = y_int * s_x * s_w_vec
        
        # Corrections for asymmetric quantization
        if not act_q.symmetric:
            wq_sum = code_w.sum(dim=1)
            if self.base_layer.ashift:
                zp_x = act_q.zero_point.to(x.device)
                y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)
                y_real = y_real + (self.base_layer.C * s_x) * (s_w_vec * wq_sum)
        
        # Add bias
        if self.base_layer.bias is not None:
            y_real = y_real + self.base_layer.bias
        
        return y_real
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing ADC-LoRA: Y = QA(Qx(X)Qw(W + AB))
        
        The effective weight (W + scaling * A @ B) is quantized through the ADC pipeline.
        """
        # Compute effective weight with LoRA adaptation
        lora_weight = self.scaling * (self.lora_A @ self.lora_B)
        effective_weight = self.base_layer.weight + lora_weight
        
        # Apply dropout to the weight modification (during training)
        if self.training:
            # Apply dropout mask to lora contribution
            lora_weight_dropped = self.lora_dropout(lora_weight)
            if not isinstance(self.lora_dropout, nn.Identity):
                effective_weight = self.base_layer.weight + lora_weight_dropped
        
        # ===== ADC Quantization Pipeline (same as QATLinearADC.forward) =====
        # Store raw inputs for monitoring
        x_raw = x.clone().detach()
        
        # 1) Build activation codes (per-tensor quantizer)
        act_q = self.base_layer.activation_quantizer
        s_x = act_q.scale.to(x.device)
        
        if act_q.symmetric:
            code_x = round_ste(safe_divide(x, s_x))
            qmin_x, qmax_x = act_q.qmin, act_q.qmax
            code_x = torch.clamp(code_x, qmin_x, qmax_x)
        else:
            zp_x = act_q.zero_point.to(x.device)
            code_x_temp = round_ste(safe_divide(x, s_x) + zp_x)
            code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
            
            if self.base_layer.ashift:
                code_x = code_x_temp - self.base_layer.C
            else:
                code_x = code_x_temp - zp_x
        
        # 2) Build weight codes using EFFECTIVE weight (W + LoRA)
        w_q = self.base_layer.weight_quantizer
        s_w_vec = w_q.scale.to(effective_weight.device)
        s_w_b = s_w_vec.view(-1, 1)
        
        # Quantize effective weight (this is the key ADC-LoRA difference!)
        code_w = round_ste(safe_divide(effective_weight, s_w_b))
        qmin_w, qmax_w = w_q.qmin, w_q.qmax
        code_w = torch.clamp(code_w, qmin_w, qmax_w)
        
        # 3) Integer MM in code domain
        y_int = F.linear(code_x, code_w, bias=None)
        
        delta = self.base_layer.delta
        na = self.base_layer.na
        pa = self.base_layer.pa
        
        # Apply ADC quantization (Paper Equation 2: uses floor)
        y_adc_codes = floor_ste(y_int / delta)
        y_adc_codes = torch.clamp(y_adc_codes, na, pa)
        
        # Dequantize back to the scale
        adc_output = y_adc_codes * delta
        
        # Store kurtosis loss on effective weight
        if self.training and self.base_layer.use_kurtosis_loss:
            kurtosis_loss = self.base_layer.kurtosis_weight * compute_kurtosis_loss(
                effective_weight, self.base_layer.target_kurtosis
            )
            self.base_layer._last_kurtosis_loss = kurtosis_loss
        else:
            self.base_layer._last_kurtosis_loss = torch.tensor(0.0, device=y_int.device, dtype=y_int.dtype)
        
        # 5) Dequantize back to real domain
        y_real = adc_output * s_x
        y_real = y_real * s_w_vec
        
        # Corrections for asymmetric quantization
        if not act_q.symmetric:
            wq_sum = code_w.sum(dim=1)
            
            if self.base_layer.ashift:
                zp_x = act_q.zero_point.to(x.device)
                y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)
                y_real = y_real + (self.base_layer.C * s_x) * (s_w_vec * wq_sum)
        
        # 6) Add bias if present
        if self.base_layer.bias is not None:
            y_real = y_real + self.base_layer.bias
        
        return y_real
    
    def merge_lora_weights(self):
        """
        Merge LoRA weights into base weights permanently.
        Useful for inference after training.
        """
        with torch.no_grad():
            lora_weight = self.scaling * (self.lora_A @ self.lora_B)
            self.base_layer.weight.add_(lora_weight)
            # Reset LoRA to zero
            self.lora_A.zero_()
            self.lora_B.zero_()
    
    def get_num_trainable_params(self) -> int:
        """Return number of trainable LoRA parameters"""
        return self.lora_A.numel() + self.lora_B.numel()
    
    def get_compression_ratio(self) -> float:
        """Return compression ratio compared to full fine-tuning"""
        full_params = self.in_features * self.out_features
        lora_params = self.get_num_trainable_params()
        return full_params / lora_params


class LoRATiledLinearADC(nn.Module):
    """
    ADC-LoRA wrapper for TiledLinearADC.
    
    Applies LoRA adapters to each tile in the TiledLinearADC layer.
    Each tile gets its own LoRA A/B matrices, but they share the same rank and scaling.
    
    This maintains the tiling structure while reducing trainable parameters.
    """
    
    def __init__(self,
                 tiled_layer: TiledLinearADC,
                 r: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        """
        Args:
            tiled_layer: The TiledLinearADC layer to wrap with LoRA
            r: LoRA rank for each tile
            alpha: LoRA scaling factor
            dropout: Dropout rate for LoRA path
        """
        super().__init__()
        
        self.tiled_layer = tiled_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.n_tiles = tiled_layer.n_tiles
        self.in_features_total = tiled_layer.in_features_total
        self.out_features = tiled_layer.out_features
        self.in_features_tile = tiled_layer.in_features_tile
        
        # Freeze all base layer weights
        for tile in self.tiled_layer.tiles:
            tile.weight.requires_grad = False
            if tile.bias is not None:
                tile.bias.requires_grad = False
        
        # Create LoRA adapters for each tile
        # Each tile has shape (out_features, in_features_tile)
        # LoRA: A (out_features, r), B (r, in_features_tile)
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, r))
            for _ in range(self.n_tiles)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(r, self.in_features_tile))
            for _ in range(self.n_tiles)
        ])
        
        # Initialize A matrices with Kaiming uniform
        for lora_a in self.lora_A:
            nn.init.kaiming_uniform_(lora_a, a=5 ** 0.5)
        # B matrices stay zero for identity initialization
        
        # Dropout for regularization
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    
    def set_quantizer_mode(self, mode: str):
        """Set mode for all quantizers in all tiles"""
        self.tiled_layer.set_quantizer_mode(mode)
    
    def set_adc_bits(self, ba: int):
        """Set ADC bits for all tiles (for BitAug)"""
        self.tiled_layer.set_adc_bits(ba)
    
    def get_adc_bits(self) -> int:
        """Get current ADC bit precision"""
        return self.tiled_layer.get_adc_bits()
    
    def get_auxiliary_losses(self) -> dict:
        """
        Get all auxiliary losses from all tiles with LoRA-adjusted weights.
        """
        total_kurtosis = torch.tensor(0.0)
        
        for i, tile in enumerate(self.tiled_layer.tiles):
            # Compute effective weight for this tile
            lora_weight = self.scaling * (self.lora_A[i] @ self.lora_B[i])
            effective_weight = tile.weight + lora_weight
            
            if tile.use_kurtosis_loss:
                kurtosis_loss = tile.kurtosis_weight * compute_kurtosis_loss(
                    effective_weight, tile.target_kurtosis
                )
                total_kurtosis = total_kurtosis + kurtosis_loss
        
        return {
            'kurtosis_loss': total_kurtosis,
            'total': total_kurtosis
        }
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.tiled_layer.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.tiled_layer.eval()
        return self
    
    def compute_reference_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reference output: sum of Qx(X_i) @ Qw(W_i) for each tile.
        
        This is the target for MSE warmup (Paper Equation 12).
        No ADC quantization, no LoRA - just quantized activations times quantized weights.
        
        Returns:
            Reference output in real domain (dequantized)
        """
        if x.shape[-1] != self.in_features_total:
            raise ValueError(f"Expected last dim={self.in_features_total}, got {x.shape[-1]}")
        
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features_total)
        batch_size_2d = x2d.shape[0]
        
        y2d = torch.zeros(batch_size_2d, self.out_features, dtype=x2d.dtype, device=x2d.device)
        
        tile_in = self.in_features_tile
        for i, tile in enumerate(self.tiled_layer.tiles):
            # Get input slice for this tile
            if x2d.is_contiguous():
                xi = x2d.narrow(1, i * tile_in, tile_in)
            else:
                xi = x2d[:, i * tile_in:(i + 1) * tile_in]
            
            # Activation quantization
            act_q = tile.activation_quantizer
            s_x = act_q.scale.to(xi.device)
            
            if act_q.symmetric:
                code_x = round_ste(safe_divide(xi, s_x))
                code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
            else:
                zp_x = act_q.zero_point.to(xi.device)
                code_x_temp = round_ste(safe_divide(xi, s_x) + zp_x)
                code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
                
                if tile.ashift:
                    code_x = code_x_temp - tile.C
                else:
                    code_x = code_x_temp - zp_x
            
            # Weight quantization using BASE weight only (NO LoRA!)
            w_q = tile.weight_quantizer
            s_w_vec = w_q.scale.to(tile.weight.device)
            s_w_b = s_w_vec.view(-1, 1)
            
            code_w = round_ste(safe_divide(tile.weight, s_w_b))
            code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)
            
            # Integer MM (NO ADC quantization!)
            y_int = F.linear(code_x, code_w, bias=None)
            
            # Dequantize (skip ADC step)
            y_real = y_int * s_x * s_w_vec
            
            # Asymmetric corrections
            if not act_q.symmetric:
                wq_sum = code_w.sum(dim=1)
                if tile.ashift:
                    zp_x = act_q.zero_point.to(xi.device)
                    y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)
                    y_real = y_real + (tile.C * s_x) * (s_w_vec * wq_sum)
            
            # Add bias (only first tile has bias)
            if tile.bias is not None:
                y_real = y_real + tile.bias
            
            y2d.add_(y_real)
        
        y = y2d.reshape(*orig_shape[:-1], self.out_features)
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA-adapted tiles.
        
        Each tile computes: Y_i = QA(Qx(X_i)Qw(W_i + A_i @ B_i))
        """
        if x.shape[-1] != self.in_features_total:
            raise ValueError(f"Expected last dim={self.in_features_total}, got {x.shape[-1]}")
        
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features_total)
        batch_size_2d = x2d.shape[0]
        
        # Pre-allocate output tensor
        y2d = torch.zeros(batch_size_2d, self.out_features, dtype=x2d.dtype, device=x2d.device)
        
        tile_in = self.in_features_tile
        for i, tile in enumerate(self.tiled_layer.tiles):
            # Get input slice for this tile
            if x2d.is_contiguous():
                xi = x2d.narrow(1, i * tile_in, tile_in)
            else:
                xi = x2d[:, i * tile_in:(i + 1) * tile_in]
            
            # Compute effective weight with LoRA
            lora_weight = self.scaling * (self.lora_A[i] @ self.lora_B[i])
            if self.training:
                lora_weight = self.lora_dropout(lora_weight)
            effective_weight = tile.weight + lora_weight
            
            # ===== ADC Quantization Pipeline for this tile =====
            act_q = tile.activation_quantizer
            s_x = act_q.scale.to(xi.device)
            
            if act_q.symmetric:
                code_x = round_ste(safe_divide(xi, s_x))
                code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
            else:
                zp_x = act_q.zero_point.to(xi.device)
                code_x_temp = round_ste(safe_divide(xi, s_x) + zp_x)
                code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
                
                if tile.ashift:
                    code_x = code_x_temp - tile.C
                else:
                    code_x = code_x_temp - zp_x
            
            # Weight quantization with effective weight
            w_q = tile.weight_quantizer
            s_w_vec = w_q.scale.to(effective_weight.device)
            s_w_b = s_w_vec.view(-1, 1)
            
            code_w = round_ste(safe_divide(effective_weight, s_w_b))
            code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)
            
            # Integer MM
            y_int = F.linear(code_x, code_w, bias=None)
            
            # ADC quantization
            y_adc_codes = floor_ste(y_int / tile.delta)
            y_adc_codes = torch.clamp(y_adc_codes, tile.na, tile.pa)
            adc_output = y_adc_codes * tile.delta
            
            # Dequantize
            y_real = adc_output * s_x * s_w_vec
            
            # Asymmetric corrections
            if not act_q.symmetric:
                wq_sum = code_w.sum(dim=1)
                if tile.ashift:
                    zp_x = act_q.zero_point.to(xi.device)
                    y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)
                    y_real = y_real + (tile.C * s_x) * (s_w_vec * wq_sum)
            
            # Add bias (only first tile has bias)
            if tile.bias is not None:
                y_real = y_real + tile.bias
            
            y2d.add_(y_real)
        
        y = y2d.reshape(*orig_shape[:-1], self.out_features)
        return y
    
    def merge_lora_weights(self):
        """Merge all LoRA weights into base tile weights"""
        with torch.no_grad():
            for i, tile in enumerate(self.tiled_layer.tiles):
                lora_weight = self.scaling * (self.lora_A[i] @ self.lora_B[i])
                tile.weight.add_(lora_weight)
                self.lora_A[i].zero_()
                self.lora_B[i].zero_()
    
    def get_num_trainable_params(self) -> int:
        """Return total number of trainable LoRA parameters across all tiles"""
        total = 0
        for i in range(self.n_tiles):
            total += self.lora_A[i].numel() + self.lora_B[i].numel()
        return total
    
    def get_compression_ratio(self) -> float:
        """Return compression ratio compared to full fine-tuning"""
        full_params = self.in_features_total * self.out_features
        lora_params = self.get_num_trainable_params()
        return full_params / lora_params if lora_params > 0 else float('inf')
 