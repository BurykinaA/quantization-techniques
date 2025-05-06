import torch
from torch import nn
from torch.ao.quantization.observer import MinMaxObserver
from ADC.ste import ste_round, ste_floor

class AffineQuantizerPerTensor(nn.Module):
    def __init__(self, bx=8):
        super().__init__()
        self.observer = MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min=0, quant_max=2**bx - 1)
        self.scale = None
        self.zero_point = None
        self.bx = bx
        self.enabled = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.enabled: # Observe only during training and when enabled
            self.observer(x.detach()) # Detach to prevent observer influencing gradients
            self.scale, self.zero_point = self.observer.calculate_qparams()

        if self.scale is None or self.zero_point is None:
            # If not training or not enabled, and params not set, try to calculate them once
            # This can happen if model is loaded directly to eval mode without prior training/calibration
            if not self.training and self.enabled:
                 self.observer(x.detach())
                 self.scale, self.zero_point = self.observer.calculate_qparams()
            else:
                raise RuntimeError("Quantizer must be calibrated (observer enabled during a forward pass) before use or have scale/zp loaded.")


        scale = self.scale.to(x.device)
        zero_point = self.zero_point.to(scale.dtype).to(x.device)

        xq = ste_round(x / scale + zero_point)
        xq = torch.clamp(xq, self.observer.quant_min, self.observer.quant_max)
        return xq


class SymmetricQuantizerPerTensor(nn.Module):
    def __init__(self, bw=8):
        super().__init__()
        maxval = 2 ** (bw - 1) - 1
        self.observer = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, quant_min=-maxval, quant_max=maxval) # Changed to qint8 for symmetric
        self.scale = None
        self.zero_point = None # Symmetric quantizers ideally have zero_point = 0
        self.bw = bw
        self.enabled = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.enabled: # Observe only during training and when enabled
            self.observer(x.detach())
            self.scale, self.zero_point = self.observer.calculate_qparams() # zero_point should be 0 for symmetric

        if self.scale is None:
            if not self.training and self.enabled:
                 self.observer(x.detach())
                 self.scale, self.zero_point = self.observer.calculate_qparams()
            else:
                raise RuntimeError("Quantizer must be calibrated (observer enabled during a forward pass) before use or have scale/zp loaded.")

        scale = self.scale.to(x.device)
        # For truly symmetric, zero_point from MinMaxObserver might not be exactly 0 if using quint8 scheme by mistake.
        # However, per_tensor_symmetric qscheme should handle this. We mostly care about scale.

        xq = ste_round(x / scale)
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
    