import torch
from torch import nn
from ADC.quantizers import AffineQuantizerPerTensor, SymmetricQuantizerPerTensor, ADCQuantizer, ADCQuantizerAshift

class LinearADC(nn.Linear):
    def __init__(self, in_features, out_features, bx=8, bw=8, ba=8, k=4, bias=True):
        super(LinearADC, self).__init__(in_features, out_features, bias)
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)
        self.adc_quantizer = ADCQuantizer(M=in_features, bx=bx, bw=bw, ba=ba, k=k)

    # def dequantize(self, yq_adc):
    #     y_sum_xq_wq = yq_adc * self.adc_quantizer.delta
        
    #     # Ensure quantizers have been calibrated and have scale/zero_point
    #     if self.x_quantizer.scale is None or self.x_quantizer.zero_point is None or \
    #        self.w_quantizer.scale is None:
    #         raise RuntimeError("Input/Weight quantizers in LinearADC must be calibrated before dequantization.")

    #     # Get quantized weights to calculate sum_wq_per_output
    #     # Temporarily disable observer updates for this internal w_quantizer call if it's not already
    #     w_quantizer_was_enabled = self.w_quantizer.enabled
    #     self.w_quantizer.enabled = False 
    #     wq = self.w_quantizer(self.weight.detach()) # Use detached weight
    #     self.w_quantizer.enabled = w_quantizer_was_enabled


    #     sum_wq_per_output = wq.sum(dim=1, keepdim=True).T 

    #     zp_x = self.x_quantizer.zero_point.to(y_sum_xq_wq.device, dtype=y_sum_xq_wq.dtype)
    #     scale_x = self.x_quantizer.scale.to(y_sum_xq_wq.device)
    #     scale_w = self.w_quantizer.scale.to(y_sum_xq_wq.device)
        
    #     out = scale_x * scale_w * (y_sum_xq_wq - zp_x * sum_wq_per_output)
    #     return out

    def dequantize(self, yq):

        # yq [B, O]

        # y = sum xq_i * wq_i
        # yq = y /

        y = yq * self.adc_quantizer.delta
        out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * self.weight.sum(axis=-1)
        out = out * self.x_quantizer.scale * self.w_quantizer.scale
        return out

    def _set_quantizer_state(self, enabled: bool):
        self.x_quantizer.enabled = enabled
        self.w_quantizer.enabled = enabled
        # ADC quantizer itself doesn't have an observer, its delta is fixed.

    # def train(self, mode: bool = True):
    #     super().train(mode)
    #     self._set_quantizer_state(mode) # Observers active in train mode
    #     return self

    # def eval(self):
    #     super().eval()
    #     self._set_quantizer_state(False) # Observers inactive in eval mode
    #     return self

    def train(self, mode=True):
        super().train(mode)
        if (mode == True):
            self.x_quantizer.enable()
            self.w_quantizer.enable()
        else:
            self.x_quantizer.disable()
            self.w_quantizer.disable()
        return self

    def eval(self, mode=True):
        super().eval(mode)
        self.train(not mode)
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


# class LinearADCAshift(nn.Linear):
    # def __init__(self, in_features, out_features, bx=8, bw=8, ba=8, k=4, ashift_enabled=True, bias=True):
    #     super(LinearADCAshift, self).__init__(in_features, out_features, bias)
    #     self.bx = bx
    #     self.bw = bw
    #     self.ba = ba
    #     self.ashift_enabled = ashift_enabled # Store ashift_enabled
    #     self.x_quantizer = AffineQuantizerPerTensor(bx)
    #     self.w_quantizer = SymmetricQuantizerPerTensor(bw)
    #     # Use ADCQuantizerAshift
    #     self.adc_quantizer = ADCQuantizerAshift(M=in_features, bx=bx, bw=bw, ba=ba, k=k, ashift_enabled=self.ashift_enabled)

    # def dequantize(self, yq_adc):
    #     y_sum_xq_wq = yq_adc * self.adc_quantizer.delta
        
    #     if self.x_quantizer.scale is None or self.x_quantizer.zero_point is None or \
    #        self.w_quantizer.scale is None:
    #         raise RuntimeError("Input/Weight quantizers in LinearADCAshift must be calibrated before dequantization.")

    #     w_quantizer_was_enabled = self.w_quantizer.enabled
    #     self.w_quantizer.enabled = False 
    #     wq = self.w_quantizer(self.weight.detach())
    #     self.w_quantizer.enabled = w_quantizer_was_enabled

    #     sum_wq_per_output = wq.sum(dim=1, keepdim=True).T 

    #     zp_x = self.x_quantizer.zero_point.to(y_sum_xq_wq.device, dtype=y_sum_xq_wq.dtype)
    #     scale_x = self.x_quantizer.scale.to(y_sum_xq_wq.device)
    #     scale_w = self.w_quantizer.scale.to(y_sum_xq_wq.device)
        
    #     # The dequantization formula needs to account for the ashift mode for activations if needed.
    #     # If ashift_enabled, activations x are treated as symmetric around zero for the purpose of delta calculation in ADCQuantizerAshift.
    #     # The original LinearADC dequantization formula subtracts zp_x * sum_wq_per_output.
    #     # This term corrects for the fact that x_q = (x/S_x) + Z_x, so x = S_x * (x_q - Z_x).
    #     # And the sum becomes SUM( (S_x * (x_q - Z_x)) * (S_w * w_q) )
    #     # = S_x * S_w * SUM( (x_q - Z_x) * w_q )
    #     # = S_x * S_w * ( SUM(x_q * w_q) - Z_x * SUM(w_q) )
    #     # yq_adc * delta_adc is an approximation of SUM(x_q * w_q).
    #     # So the formula out = scale_x * scale_w * (y_sum_xq_wq - zp_x * sum_wq_per_output) seems correct
    #     # as scale_x and scale_w are S_x and S_w, and zp_x is Z_x.
    #     # The ADCQuantizerAshift's delta already incorporates the specific way x_levels are interpreted (symmetric if ashift_enabled).
    #     # The AffineQuantizerPerTensor for x_quantizer will still produce xq with its own zero_point (zp_x).
    #     # The dequantization needs to correctly reverse this initial x_quantizer step.
        
    #     out = scale_x * scale_w * (y_sum_xq_wq - zp_x * sum_wq_per_output)
    #     return out

    # def _set_quantizer_state(self, enabled: bool):
    #     self.x_quantizer.enabled = enabled
    #     self.w_quantizer.enabled = enabled
    #     # ADCQuantizerAshift itself doesn't have an observer for its delta.

    # def train(self, mode: bool = True):
    #     super().train(mode)
    #     self._set_quantizer_state(mode)
    #     return self

    # def eval(self):
    #     super().eval()
    #     self._set_quantizer_state(False)
    #     return self

    # def forward(self, x):
    #     xq = self.x_quantizer(x)
    #     wq = self.w_quantizer(self.weight) 

    #     y_for_adc = nn.functional.linear(xq, wq) # This is SUM(xq * wq)
    #     yq_adc = self.adc_quantizer(y_for_adc)    # This is ADC_Quant(SUM(xq * wq))
        
    #     out = self.dequantize(yq_adc)
        
    #     if self.bias is not None:
    #         out = out + self.bias
    #     return out

class LinearADCAshift(LinearADC):
    def __init__(self, in_features, out_features, bx=8, bw=8, ba=8, k=4, ashift_enabled=True, bias=True):
        super(LinearADCAshift, self).__init__(in_features, out_features, bx, bw, ba, k, bias)
        self.C = 2 ** (bx - 1)
        self.ashift_enabled = ashift_enabled # Store ashift_enabled

    def dequantize(self, yq_adc, wq):
        y = yq_adc * self.adc_quantizer.delta + self.C * wq.sum(axis=-1)
        out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * self.weight.sum(axis=-1)
        out = out * self.x_quantizer.scale * self.w_quantizer.scale
        return out

    def forward(self, x):
        xq = self.x_quantizer(x)
        xq = xq - self.C
        wq = self.w_quantizer(self.weight) 

        y_for_adc = nn.functional.linear(xq, wq) # This is SUM(xq * wq)
        yq_adc = self.adc_quantizer(y_for_adc)    # This is ADC_Quant(SUM(xq * wq))
        
        out = self.dequantize(yq_adc, wq)
        
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


class Conv2dADC(nn.Conv2d):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=None, 
                 padding_mode='zeros', 
                 device=None, 
                 dtype=None,
                 bx=8,
                 bw=8,
                 ba=8,
                 k=4):
        super(Conv2dADC, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)
        if type(kernel_size) == int:
            Mv = in_channels*(kernel_size**2)
        else:
            Mv = in_channels*kernel_size[0]*kernel_size[1]
        self.adc_quantizer = ADCQuantizer(M=Mv, bx=bx, bw=bw, ba=ba, k=k)
    
    def dequantize(self, yq):
        # yq: out x H_out x W_out
        # self.weight: out x in x H_out x W_out
        y = yq * self.adc_quantizer.delta
        out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * (self.weight.sum(axis=(1, 2, 3)))[None, :, None, None]
        out = out * self.x_quantizer.scale * self.w_quantizer.scale
        return out
    
    def train(self, mode=True):
        super().train(mode)
        if (mode == True):
            self.x_quantizer.enable()
            self.w_quantizer.enable()
        else:
            self.x_quantizer.disable()
            self.w_quantizer.disable()
        return self
    def eval(self, mode=True):
        super().eval(mode)
        self.train(not mode)
        return self
    def forward(self, x):
        xq = self.x_quantizer(x)
        wq = self.w_quantizer(self.weight)
        y_for_adc = torch.nn.functional.conv2d(xq, 
                                               wq, 
                                               bias=self.bias, 
                                               stride=self.stride, 
                                               padding=self.padding, 
                                               dilation=self.dilation, 
                                               groups=self.groups)
        yq_adc = self.adc_quantizer(y_for_adc)
        out = self.dequantize(yq_adc)
        if self.bias is not None:
            out += self.bias
        return out
    