import torch
from torch import nn
from ADC.quantizers import AffineQuantizerPerTensor, SymmetricQuantizerPerTensor, ADCQuantizer, ADCQuantizerAshift
import wandb
import random


class LinearADC(nn.Linear):
    def __init__(self, in_features, out_features, bx=8, bw=8, ba=8, k=4, bias=True, ashift=False, logger=None):
        super(LinearADC, self).__init__(in_features, out_features, bias)
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)
        self.adc_quantizer = ADCQuantizer(M=in_features, bx=bx, bw=bw, ba=ba, k=k)
        self.ashift = ashift
        self.C = 2 ** (bx - 1)
        self.adc_enabled = True
        self.name = "Linear" + str(random.randint(10 ** 5, 10**6 - 1))
        self.logger = logger

    def enable_adc(self):
        self.adc_enabled = True
    def disable_adc(self):
        self.adc_enabled = False

    def dequantize(self, yq, wq):

        y = yq * self.adc_quantizer.delta
        if (self.ashift):
            y = y + self.C * wq.sum(axis=-1)
        out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * self.weight.sum(axis=-1)
        out = out * self.x_quantizer.scale * self.w_quantizer.scale
        return out

    def _set_quantizer_state(self, enabled: bool):
        self.x_quantizer.enabled = enabled
        self.w_quantizer.enabled = enabled
        # ADC quantizer itself doesn't have an observer, its delta is fixed.


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
        if (not self.adc_enabled):
            xq = self.x_quantizer.fake_quantize(x)
            wq = self.w_quantizer.fake_quantize(self.weight)
            out = nn.functional.linear(xq, wq, self.bias)
            return out

        xq = self.x_quantizer(x)
        if (self.ashift):
            xq = xq - self.C
        
        # For weights, quantize them. If in eval mode, observer is off.
        # If in train mode, observer is on for the first few passes (calibration).
        wq = self.w_quantizer(self.weight) 

        y_for_adc = nn.functional.linear(xq, wq)
        yq_adc = self.adc_quantizer(y_for_adc)
        
        out = self.dequantize(yq_adc, wq)
        
        if self.bias is not None:
            out = out + self.bias

        if self.logger:
            with torch.no_grad():
                out_gth = nn.functional.linear(x, self.weight, bias = self.bias)
                diff = torch.linalg.norm(out - out_gth).cpu().item()
                gth_norm = torch.linalg.norm(out_gth).cpu().item()
                self.logger.log(self.name, "out_norm_ratio", diff / gth_norm)
                #wandb.log({self.name + "_diff" : diff / gth_norm})
                #print(self.name + "_diff: ", diff / gth_norm)
        return out

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
                 k=4,
                 ashift=False,
                 logger=None):
        super(Conv2dADC, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.name = f"Conv2d" + str(random.randint(10 ** 5, 10**6 - 1))
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)
        self.ashift=ashift
        self.C = 2 ** (bx - 1)
        self.logger = logger
        if type(kernel_size) == int:
            Mv = in_channels*(kernel_size**2)
        else:
            Mv = in_channels*kernel_size[0]*kernel_size[1]
        self.adc_quantizer = ADCQuantizer(M=Mv, bx=bx, bw=bw, ba=ba, k=k, info=self.name, logger=self.logger)
        self.adc_enabled = True
    
    def enable_adc(self):
        self.adc_enabled = True
    def disable_adc(self):
        self.adc_enabled = False
    
    def _set_quantizer_state(self, enabled: bool):
        self.x_quantizer.enabled = enabled
        self.w_quantizer.enabled = enabled

    def dequantize(self, yq, wq):
        # yq: out x H_out x W_out
        # self.weight: out x in x H_out x W_out
        
        # Important!!!!!!!
        y = yq * self.adc_quantizer.delta
        #y = yq
        # -------------------------------------

        if (self.ashift):
            print("ashift dequantize")
            y = y + self.C * (wq.sum(axis=(1, 2, 3)))[None, :, None, None]
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
        if (not self.adc_enabled):
            xq = self.x_quantizer.fake_quantize(x)
            wq = self.w_quantizer.fake_quantize(self.weight)
            out = torch.nn.functional.conv2d(xq, 
                                               wq, 
                                               bias=self.bias, 
                                               stride=self.stride, 
                                               padding=self.padding, 
                                               dilation=self.dilation, 
                                               groups=self.groups)
            return out
        

        xq = self.x_quantizer(x)
        if (self.ashift):
            xq = xq - self.C
        wq = self.w_quantizer(self.weight)
        y_for_adc = torch.nn.functional.conv2d(xq, 
                                               wq, 
                                               bias=None, 
                                               stride=self.stride, 
                                               padding=self.padding, 
                                               dilation=self.dilation, 
                                               groups=self.groups)
        #yq_adc = y_for_adc
        yq_adc = self.adc_quantizer(y_for_adc)
        out = self.dequantize(yq_adc, wq)
        if self.bias is not None:
            out += self.bias
        if self.logger:
            with torch.no_grad():
                out_gth = torch.nn.functional.conv2d(x, 
                                               self.weight, 
                                               bias=self.bias, 
                                               stride=self.stride, 
                                               padding=self.padding, 
                                               dilation=self.dilation, 
                                               groups=self.groups)
                self.logger.log(self.name, "max_val", y_for_adc.max().item())
                self.logger.log(self.name, "min_val", y_for_adc.min().item())
                diff = torch.linalg.norm(out - out_gth).cpu().item()
                gth_norm = torch.linalg.norm(out_gth).cpu().item()
                self.logger.log(self.name, "out_norm_ratio", diff / gth_norm)
                #print(self.name + "_diff: ", diff / gth_norm)

        return out
    