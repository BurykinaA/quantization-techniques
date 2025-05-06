import torch
from torch import nn
from ADC.quantizers import AffineQuantizerPerTensor, SymmetricQuantizerPerTensor, ADCQuantizer

class LinearADC(nn.Linear):
    def __init__(self, in_features, out_features, bx=8, bw=8, ba=8, k=4, bias=True):
        super(LinearADC, self).__init__(in_features, out_features, bias)
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)
        self.adc_quantizer = ADCQuantizer(M=in_features, bx=bx, bw=bw, ba=ba, k=k)

    def dequantize(self, yq_adc):
        y_sum_xq_wq = yq_adc * self.adc_quantizer.delta
        
        # Ensure quantizers have been calibrated and have scale/zero_point
        if self.x_quantizer.scale is None or self.x_quantizer.zero_point is None or \
           self.w_quantizer.scale is None:
            raise RuntimeError("Input/Weight quantizers in LinearADC must be calibrated before dequantization.")

        # Get quantized weights to calculate sum_wq_per_output
        # Temporarily disable observer updates for this internal w_quantizer call if it's not already
        w_quantizer_was_enabled = self.w_quantizer.enabled
        self.w_quantizer.enabled = False 
        wq = self.w_quantizer(self.weight.detach()) # Use detached weight
        self.w_quantizer.enabled = w_quantizer_was_enabled


        sum_wq_per_output = wq.sum(dim=1, keepdim=True).T 

        zp_x = self.x_quantizer.zero_point.to(y_sum_xq_wq.device, dtype=y_sum_xq_wq.dtype)
        scale_x = self.x_quantizer.scale.to(y_sum_xq_wq.device)
        scale_w = self.w_quantizer.scale.to(y_sum_xq_wq.device)
        
        out = scale_x * scale_w * (y_sum_xq_wq - zp_x * sum_wq_per_output)
        return out

    def _set_quantizer_state(self, enabled: bool):
        self.x_quantizer.enabled = enabled
        self.w_quantizer.enabled = enabled
        # ADC quantizer itself doesn't have an observer, its delta is fixed.

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_quantizer_state(mode) # Observers active in train mode
        return self

    def eval(self):
        super().eval()
        self._set_quantizer_state(False) # Observers inactive in eval mode
        return self

    def forward(self, x):
        xq = self.x_quantizer(x)
        
        # For weights, quantize them. If in eval mode, observer is off.
        # If in train mode, observer is on for the first few passes (calibration).
        wq = self.w_quantizer(self.weight) 

        y_for_adc = nn.functional.linear(xq, wq)
        yq_adc = self.adc_quantizer(y_for_adc)
        
        out = self.dequantize(yq_adc)
        
        if self.bias is not None:
            out = out + self.bias
        return out


class LinearQuant(nn.Linear):
    def __init__(self, in_features, out_features, bx=8, bw=8, bias=True):
        super(LinearQuant, self).__init__(in_features, out_features, bias)
        self.bx = bx
        self.bw = bw
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)

    def _set_quantizer_state(self, enabled: bool):
        self.x_quantizer.enabled = enabled
        self.w_quantizer.enabled = enabled

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_quantizer_state(mode)
        return self

    def eval(self):
        super().eval()
        self._set_quantizer_state(False)
        return self

    def forward(self, x):
        xq = self.x_quantizer(x)
        wq = self.w_quantizer(self.weight)

        if self.x_quantizer.scale is None or self.x_quantizer.zero_point is None or \
           self.w_quantizer.scale is None:
            raise RuntimeError("Input/Weight quantizers in LinearQuant must be calibrated.")

        scale_x = self.x_quantizer.scale.to(xq.device)
        zp_x = self.x_quantizer.zero_point.to(xq.device, dtype=xq.dtype)
        
        # Ensure xq is float before subtracting zero_point if zero_point is float
        if not torch.is_floating_point(xq):
             x_dequant = (xq.to(scale_x.dtype) - zp_x) * scale_x
        else:
             x_dequant = (xq - zp_x) * scale_x


        scale_w = self.w_quantizer.scale.to(wq.device)
        if not torch.is_floating_point(wq):
            w_dequant = wq.to(scale_w.dtype) * scale_w
        else:
            w_dequant = wq * scale_w
            
        out = nn.functional.linear(x_dequant, w_dequant, self.bias)
        return out
    