import torch
from torch import nn
from torch.ao.quantization.observer import MinMaxObserver
from ADC.ste import ste_round, ste_floor

class AffineQuantizerPerTensor(nn.Module):
    def __init__(self, bx=8):
        super().__init__()
        self.observer = MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine, 
                                       quant_min=0, quant_max=2**bx - 1)
        
        # Register scale and zero_point as buffers with initial placeholder values.
        # Their types will be preserved when updated.
        self.register_buffer('scale', torch.tensor(1.0)) 
        self.register_buffer('zero_point', torch.tensor(0, dtype=torch.int32)) # Observers often return int32 zero_point
        
        # Flag to indicate if parameters have been calculated
        self.params_calculated = False
        self.bx = bx
        self.enabled = True # Controls observer if model.training is False

    def enable(self):
        self.enabled = True
        # self.params_calculated = False # Optionally reset if re-enabling means re-calibrating

    def disable(self):
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.enabled: 
            self.observer(x.detach()) 
            _s, _zp = self.observer.calculate_qparams()
            # Update the buffers in-place using copy_()
            self.scale.copy_(_s)
            self.zero_point.copy_(_zp.to(self.zero_point.dtype)) # Ensure correct dtype for zero_point
            self.params_calculated = True

        if not self.params_calculated:
            if not self.training and self.enabled: # If in eval mode but observer is explicitly enabled
                print(f"Warning: {self.__class__.__name__} at eval, observer enabled, but params not calculated. Running observer once.")
                self.observer(x.detach())
                _s, _zp = self.observer.calculate_qparams()
                self.scale.copy_(_s)
                self.zero_point.copy_(_zp.to(self.zero_point.dtype))
                self.params_calculated = True
                if not self.params_calculated: # Should not happen if observer ran
                    raise RuntimeError(f"{self.__class__.__name__} failed to calculate params even with observer pass at eval.")
            else: # Not training, observer disabled, and params were never calculated (e.g. model loaded without training)
                raise RuntimeError(f"{self.__class__.__name__} must be calibrated (observer run during training or 'enabled' at eval) or have scale/zp loaded from a trained state_dict. Current scale/zp are at initial/default values.")

        # Use the buffer values for quantization
        current_scale = self.scale.to(x.device)
        current_zero_point = self.zero_point.to(x.device)

        xq = ste_round(x / current_scale + current_zero_point)
        xq = torch.clamp(xq, self.observer.quant_min, self.observer.quant_max)
        return xq


class SymmetricQuantizerPerTensor(nn.Module):
    def __init__(self, bw=8):
        super().__init__()
        # For symmetric, quant_min/max should span zero. qint8 is appropriate.
        # Max value for bw=1, qint8 range [-1, 0] can be tricky. If levels are {-1, 1}, this observer might not be ideal.
        # Let's assume standard symmetric range:
        q_min_val = -(2**(bw-1)) if bw > 1 else -1 
        q_max_val = (2**(bw-1))-1 if bw > 1 else (0 if bw == 1 else 0) # for bw=1, torch default is often [-1,0] or [-128,127] with specific mapping.
                                                                   # Using a narrow range for bw=1 like -1,0 can be problematic if inputs are positive.
                                                                   # It's safer to ensure the observer range covers expected input magnitudes.
        if bw == 1: # Special handling for binary weights, common to be {-alpha, alpha} mapped to {-1,1} or {0,1} levels.
                    # For simplicity here, we'll assume it can map to -1, 0 or -1, 1.
            q_min_val = -1
            q_max_val = 1 # Allow it to see positive values too, scale will adjust.
            # If you truly want {-1, 0} levels, then q_max_val = 0.

        self.observer = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, 
                                       quant_min=q_min_val, quant_max=q_max_val)
        
        self.register_buffer('scale', torch.tensor(1.0))
        # zero_point for symmetric should ideally be 0 after observer.
        self.register_buffer('zero_point', torch.tensor(0, dtype=torch.int32)) 
        
        self.params_calculated = False
        self.bw = bw
        self.enabled = True

    def enable(self):
        self.enabled = True
        # self.params_calculated = False

    def disable(self):
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.enabled:
            self.observer(x.detach())
            _s, _zp = self.observer.calculate_qparams()
            self.scale.copy_(_s)
            # For torch.per_tensor_symmetric, _zp should be 0.
            self.zero_point.copy_(_zp.to(self.zero_point.dtype)) 
            self.params_calculated = True

        if not self.params_calculated:
            if not self.training and self.enabled:
                print(f"Warning: {self.__class__.__name__} at eval, observer enabled, but params not calculated. Running observer once.")
                self.observer(x.detach())
                _s, _zp = self.observer.calculate_qparams()
                self.scale.copy_(_s)
                self.zero_point.copy_(_zp.to(self.zero_point.dtype))
                self.params_calculated = True
                if not self.params_calculated:
                    raise RuntimeError(f"{self.__class__.__name__} failed to calculate params even with observer pass at eval.")
            else:
                raise RuntimeError(f"{self.__class__.__name__} must be calibrated or have scale/zp loaded. Current scale is at initial/default value and observer is disabled.")

        current_scale = self.scale.to(x.device)
        # zero_point is typically not used in the symmetric formula, but we ensure it's available.
        # current_zero_point = self.zero_point.to(x.device) 

        xq = ste_round(x / current_scale) # zero_point is omitted for symmetric quantization formula
        xq = torch.clamp(xq, self.observer.quant_min, self.observer.quant_max)
        return xq

class ADCQuantizer(nn.Module):
    def __init__(self, M, bx, bw, ba = 8, k = 4):
        super().__init__()
        # delta calculation seems to assume symmetric quantization for weights (2**(bw-1)-1)
        # and affine for activations (2**bx - 1)
        self.delta = 2 * M * (2 ** bx - 1) * (2 ** (bw - 1) - 1) / ((2 ** ba - 1) * k)
        self.M = M
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = ste_floor(x / self.delta)
        mnval = -2 ** (self.ba - 1)
        mxval = 2 ** (self.ba - 1) - 1
        xq = torch.clamp(xq, mnval, mxval)
        return xq
    

class ADCQuantizerAshift(nn.Module):
    def __init__(self, M, bx, bw, ba=8, k=4, ashift_enabled=True): # ashift_enabled by default for this class
        super().__init__()
        self.M = M
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k
        self.ashift_enabled = ashift_enabled # Store the mode

        # Determine the maximum value/magnitude of weight levels
        max_w_level_val = (2**(self.bw - 1) - 1)
        if self.bw == 1: 
            max_w_level_val = 1
        if max_w_level_val <= 0 and self.bw > 1:
             print(f"Warning (Ashift): max_w_level_val for bw={self.bw} is {max_w_level_val}. Using 1.")
             max_w_level_val = 1

        # Determine the maximum magnitude of activation levels based on ashift_enabled
        if self.ashift_enabled:
            # With A-shift, activation levels (xs_levels) are effectively bx-bit signed symmetric.
            # Range is typically [-2**(bx-1), 2**(bx-1)-1].
            # The maximum *magnitude* of these levels is 2**(bx-1).
            max_x_level_magnitude = 2**(self.bx - 1)
        else:
            # Without A-shift (behaves like original ADCQuantizer for activation part of delta)
            # Activations are typically affine quantized. Levels are in [0, 2**bx-1].
            # Max *value* (and magnitude) is (2**bx-1).
            max_x_level_magnitude = (2**self.bx - 1)
        
        if max_x_level_magnitude <=0 and self.bx >=1:
            print(f"Warning (Ashift): max_x_level_magnitude for bx={self.bx} (ashift={self.ashift_enabled}) is {max_x_level_magnitude}. Using 1.")
            max_x_level_magnitude = 1

        denominator = (2**self.ba - 1) * self.k
        if denominator == 0:
            raise ValueError(f"ADCQuantizerAshift delta denominator is zero. ba={self.ba}, k={self.k}")

        self.delta = (2 * self.M * max_x_level_magnitude * max_w_level_val) / denominator
        
        if abs(self.delta) < 1e-12:
             print(f"Warning (Ashift): ADCQuantizerAshift delta is very small or zero: {self.delta}.")
             print(f"  M={self.M}, bx={self.bx}, bw={self.bw}, ba={self.ba}, k={self.k}, ashift_enabled={self.ashift_enabled}")
             print(f"  Calculated max_x_level_magnitude: {max_x_level_magnitude}")
             print(f"  Calculated max_w_level_val: {max_w_level_val}")
             self.delta = 1.0 # Fallback delta

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is the analog sum of products
        if abs(self.delta) < 1e-9:
            print(f"Critical Warning (Ashift): ADCQuantizerAshift forward pass with near-zero delta: {self.delta}")
            effective_delta = 1.0 
        else:
            effective_delta = self.delta
            
        xq = ste_floor(x / effective_delta)
            
        mnval = -2**(self.ba - 1)
        mxval = 2**(self.ba - 1) - 1
        xq = torch.clamp(xq, mnval, mxval)
        return xq # Outputting integer levels
    