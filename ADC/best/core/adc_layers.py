import torch
import torch.nn as nn
import torch.nn.functional as F
from .grad_functions import *
from .utils import *


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
        
        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1
        
        init_scale = 0.01 if symmetric else 0.02
        self.scale = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))
        self._scale_initialized = not per_channel
        
        # Initialize zero_point if asymmetric
        if not symmetric:
            init_zp = (self.qmax + self.qmin) / 2.0
            self.zero_point = nn.Parameter(torch.tensor([init_zp], dtype=torch.float32))
            self._zp_initialized = not per_channel
        else:
            self.register_buffer('zero_point', torch.zeros(1))
            self._zp_initialized = True
        
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
                    x_absmax = x_reshaped.abs().max(dim=1)[0] # max by each channel -> [channel_size]
                else:
                    x_transposed = x.transpose(self.channel_dim, 0) # move channel_dim to the first dimension
                    x_reshaped = x_transposed.contiguous().view(channel_size, -1) # transpose create unsteady memory access pattern, so we need to contiguous() to make it continuous (view needed contiguous tensor as input)
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
        """Only for QAT mode: Update quantization parameters based on input statistics (for initialization)"""
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
                
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._initialize_parameters(x)
        
        if self._mode == 'calibration':
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
        
        # When True, model unipolar ADC (shift before floor, subtract after).
        # Physical optical device has range [0, 2^ba − 1] — only positive codes.
        # Mathematically equivalent to bipolar but z_shifted is guaranteed ≥ 0.
        self.unipolar_adc = False
        # offset_codes = |na| = 2^(ba-1), e.g. 128 for ba=8 — exact integer
        self._adc_offset_codes = -self.na

        # When True, skip ADC quantization (floor/clamp) in forward pass.
        self.bypass_adc = False
        # When True, skip ALL quantization (plain F.linear).
        self.bypass_all = False
        # Stats capture for per-layer k search (search_k_per_layer in pipeline.py).
        self._capturing = False
        self._y_int_samples: list = []
        # PACT-style learned activation clip threshold (set from FlatQuantLinear
        # after ADC conversion).  When not None, replaces per-token amax with a
        # fixed scalar clip value so activation codes are not outlier-dominated.
        self.alpha_adc: float | None = None
    
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
        self._adc_offset_codes = -self.na

    def set_k(self, new_k: int) -> None:
        """Update ADC parallelism k and recompute delta in-place."""
        self.k = new_k
        denom = float((2 ** self.ba) * self.k)
        self.delta = (2.0 * float(self.in_features)
                      * self._activation_level_magnitude
                      * self._weight_level_max) / denom
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bypass_all:
            w = self.weight.to(x.dtype)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, w, b)

        # Cast to float32 for quantization math (parameters are float32),
        # then cast output back at the end
        input_dtype = x.dtype
        x = x.float()

        # 1) Build activation codes — matching FlatQuantLinear._train_forward_adc.
        #    When alpha_adc is set (PACT-style), use the fixed learned clip threshold
        #    so codes are not dominated by per-token outliers (same logic as training).
        #    Otherwise fall back to per-token amax (original behaviour).
        #    activation_quantizer.scale (per-tensor, calibrated in STEP 2) is
        #    intentionally NOT used for the forward computation; it is still
        #    calibrated so that calibration diagnostics remain valid.
        act_q = self.activation_quantizer
        qmin_x, qmax_x = act_q.qmin, act_q.qmax
        act_levels = float(qmax_x)  # 127 (signed) or 255 (unsigned)

        if self.alpha_adc is not None and not self.bypass_adc:
            # PACT: fixed per-tile clip threshold — used for ADC path only.
            # bypass_adc path always uses per-token amax (identical to training).
            alpha_t = x.new_tensor(self.alpha_adc)
            x_c = x.clamp(-alpha_t, alpha_t)
            s_x = alpha_t / act_levels          # scalar — same for all tokens
            code_x = round_ste(x_c / s_x).clamp(qmin_x, qmax_x)
        else:
            # amax over the feature dimension → one scale per token [B, 1]
            s_x = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
            code_x = round_ste(x / s_x).clamp(qmin_x, qmax_x)

        # 2) Build weight codes (per-channel symmetric, channel_dim=0)
        w_q = self.weight_quantizer
        s_w_vec = w_q.scale  # Use original scale (gradient clipping happens in safe_divide)
        # Broadcast scales to weight shape for division
        s_w_b = s_w_vec.view(-1, 1)

        # Use safe_divide for gradient clipping, round_ste for STE
        code_w = round_ste(safe_divide(self.weight, s_w_b))
        qmin_w, qmax_w = w_q.qmin, w_q.qmax
        code_w = torch.clamp(code_w, qmin_w, qmax_w)

        # 3) Integer MVM in code domain + ADC quantization
        #
        # Single ADC on the full dot product (Eq. 2 from the paper).
        # K appears only in the delta formula (Eq. 3), making delta smaller
        # and giving the ADC finer resolution.

        y_int = F.linear(code_x, code_w, bias=None)

        if self._capturing:
            # Capture |y_int| for per-layer k search (both bipolar and unipolar).
            # Bypassing ADC during capture is handled externally (pipeline.py).
            with torch.no_grad():
                flat = y_int.detach().float().abs().flatten()
                if flat.numel() > 10_000:
                    idx = torch.randperm(flat.numel(), device=flat.device)[:10_000]
                    flat = flat[idx]
                self._y_int_samples.append(flat.cpu())

        if self.bypass_adc:
            adc_output = y_int
        elif getattr(self, 'unipolar_adc', False):
            # Unsigned shift-and-subtract for positive-only optical hardware.
            #
            # Shift signed codes to unsigned range, do one non-negative MVM,
            # then subtract correction terms digitally:
            #   code_x_u = code_x + 2^(bx-1)   →  [0, 2^bx - 1]
            #   code_w_u = code_w + 2^(bw-1)   →  [0, 2^bw - 1]
            #   y_uint   = code_x_u · code_w_u^T  ≥ 0   (one ADC read)
            #   y_int    = y_uint - zp_x·Σw_u - zp_w·Σx_u + tile_in·zp_x·zp_w
            #
            # δ = tile_in·(2^bx-1)·(2^bw-1) / ((2^ba-1)·k)
            # For INT4, k=16: δ ≈ 14.1  (single MVM, 4× faster than 4-quadrant)
            zp_x = float(1 << (self.bx - 1))         # = 8 for bx=4
            zp_w = float(1 << (self.bw - 1))         # = 8 for bw=4

            code_x_u = code_x + zp_x                 # [0, 2^bx - 1]
            code_w_u = code_w + zp_w                 # [0, 2^bw - 1]

            y_uint = F.linear(code_x_u, code_w_u, None)  # always ≥ 0

            qmax_u = float((1 << self.bx) - 1) * float((1 << self.bw) - 1)
            d_uni  = float(self.in_features) * qmax_u / (float(1 << self.ba) * float(self.k))
            # δ = (2^bx-1)*(2^bw-1)*M / (k * 2^ba)  e.g. 15*15*256/(16*256) = 14.0625
            # No clamp: y_uint is the digital sum of tile_in/k ADC readings.
            adc_out_u = floor_ste(y_uint / d_uni) * d_uni

            sum_w_u = code_w_u.sum(dim=1)                  # [out_features]
            sum_x_u = code_x_u.sum(dim=-1, keepdim=True)   # [B, 1]
            correction = (zp_x * sum_w_u
                          + zp_w * sum_x_u
                          - float(self.in_features) * zp_x * zp_w)
            adc_output = adc_out_u - correction
        else:
            y_adc_codes = floor_ste(y_int / self.delta)
            y_adc_codes = torch.clamp(y_adc_codes, self.na, self.pa)
            adc_output = y_adc_codes * self.delta

        # Kurtosis loss for W-reshape (Paper Equation 6 & 7)
        if self.training and self.use_kurtosis_loss:
            self._last_kurtosis_loss = self.kurtosis_weight * compute_kurtosis_loss(
                self.weight, self.target_kurtosis
            )
        else:
            self._last_kurtosis_loss = torch.tensor(0.0, device=adc_output.device, dtype=adc_output.dtype)

        # 4) Dequantize: s_x [B,1] * s_w_vec [out] → [B, out]
        y_real = adc_output * s_x * s_w_vec

        if self.bias is not None:
            y_real = y_real + self.bias

        return y_real.to(input_dtype)
    
    def get_auxiliary_losses(self) -> dict:
        """Get kurtosis (W-reshape) loss for this layer (Paper Equation 6 & 7)."""
        return {'total': getattr(self, '_last_kurtosis_loss', torch.tensor(0.0))}


class TiledLinearADC(nn.Module):
    """
    Tiled Linear layer with ADC quantization.
    Splits in_features so each tile fits within mvm_limit.
    Each tile is a QATLinearADC with its own M=in_features_tile for correct delta.
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
                 unipolar_adc: bool = False,
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

        n_tiles = 1
        tile_in = in_features
        while (tile_in > mvm_limit) and (tile_in % 2 == 0):
            n_tiles *= 2
            tile_in //= 2
        if tile_in > mvm_limit:
            raise ValueError("in_features is not divisible by a power of 2 to meet mvm_limit")

        self.n_tiles = n_tiles
        self.in_features_tile = tile_in
        self.unipolar_adc = unipolar_adc

        self.tiles = nn.ModuleList()
        for i in range(n_tiles):
            use_bias = bias if i == 0 else False
            tile = QATLinearADC(
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
            tile.unipolar_adc = unipolar_adc
            self.tiles.append(tile)
    
    def set_alpha_adc(self, alpha: "float | list[float] | None") -> None:
        """Set PACT-style activation clip threshold.

        Args:
            alpha: Scalar float (same for all tiles), list of per-tile floats
                   (length must match n_tiles), or None to revert to per-token
                   amax (original behaviour).
        """
        if alpha is None or isinstance(alpha, float):
            for tile in self.tiles:
                tile.alpha_adc = alpha
        else:
            # Per-tile list
            for i, tile in enumerate(self.tiles):
                tile.alpha_adc = float(alpha[i]) if i < len(alpha) else float(alpha[-1])

    def set_bypass_adc(self, bypass: bool):
        """Enable/disable ADC bypass for all tiles."""
        for tile in self.tiles:
            tile.bypass_adc = bypass
    
    def set_bypass_all(self, bypass: bool):
        """Enable/disable FULL bypass for all tiles (plain linear, no quantization)."""
        for tile in self.tiles:
            tile.bypass_all = bypass
    
    def set_quantizer_mode(self, mode: str):
        """Set mode for all quantizers in all tiles"""
        for tile in self.tiles:
            tile.set_quantizer_mode(mode)
    
    def set_adc_bits(self, ba: int):
        """Dynamically set ADC bit precision for all tiles (for BitAug)."""
        for tile in self.tiles:
            tile.set_adc_bits(ba)

    def set_k(self, new_k: int) -> None:
        """Update ADC parallelism k for all tiles in-place."""
        for tile in self.tiles:
            tile.set_k(new_k)
    
    def get_adc_bits(self) -> int:
        """Get current ADC bit precision from first tile"""
        return self.tiles[0].ba if self.tiles else 8
    
    def get_auxiliary_losses(self) -> dict:
        """Get aggregated kurtosis loss from all tiles."""
        total = torch.tensor(0.0)
        for tile in self.tiles:
            total = total + tile.get_auxiliary_losses()['total']
        return {'total': total}

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
        """Split weights from an nn.Linear across tiles along dim=1 (in_features)."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
