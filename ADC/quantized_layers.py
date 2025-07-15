import torch
from torch import nn
from ADC.quantizers import AffineQuantizerPerTensor, SymmetricQuantizerPerTensor, ADCQuantizer, ADCQuantizerAshift
from torch.ao.quantization.observer import MinMaxObserver, ObserverBase
from ADC.ste import ste_round, ste_floor

# ДОБАВЛЕНИЕ: создаем кастомный MinMaxObserver с eps
class MinMaxObserverWithEps(MinMaxObserver):
    """
    Observer that tracks the min/max values of a tensor and computes quantization
    parameters, adding a small epsilon to the range to avoid division by zero.
    """

    def __init__(self, *args, **kwargs):
        super(MinMaxObserverWithEps, self).__init__(*args, **kwargs)
        self.register_buffer("eps", torch.tensor(torch.finfo(torch.float32).eps))

    def forward(self, x_orig):
        r"""Records the running min and max of ``x``."""
        # Добавляем eps в знаменатель
        scale = (self.max_val - self.min_val) / float(self.quant_max - self.quant_min + self.eps)
        # Убедимся, что scale не равен нулю
        scale = torch.max(scale, torch.tensor(self.eps).to(scale.device))

        zero_point = self.min_val - scale * self.quant_min
        zero_point = zero_point.round()

        return scale.to(torch.float32), zero_point.to(torch.int32)


class AffineQuantizerPerTensor(nn.Module):
    def __init__(self, bx=8):
        super().__init__()
        # ИСПОЛЬЗУЕМ НАШ НОВЫЙ OBSERVER
        self.observer = MinMaxObserverWithEps(dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min=0, quant_max=2**bx - 1)
        
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float32)) 
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int32))
        self.register_buffer('params_calculated', torch.tensor([False], dtype=torch.bool))
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return x
        if self.training:
            self.observer(x.detach())
        if not self.params_calculated:
            self.scale.copy_(self.observer.calculate_qparams()[0])
            self.zero_point.copy_(self.observer.calculate_qparams()[1])
            if self.training: self.params_calculated.fill_(True)
        return ste_round(torch.clamp(x / self.scale + self.zero_point, 0, 2**8 - 1))

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class SymmetricQuantizerPerTensor(nn.Module):
    def __init__(self, bw=8):
        super().__init__()
        q_min_val = -(2**(bw-1)) if bw > 1 else -1 
        q_max_val = 2**(bw-1) - 1 if bw > 1 else 0
        # ИСПОЛЬЗУЕМ НАШ НОВЫЙ OBSERVER
        self.observer = MinMaxObserverWithEps(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, 
                                       quant_min=q_min_val, quant_max=q_max_val)
        
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int32))
        self.register_buffer('params_calculated', torch.tensor([False], dtype=torch.bool))
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return x
        if self.training:
            self.observer(x.detach())
        if not self.params_calculated:
            self.scale.copy_(self.observer.calculate_qparams()[0])
            if self.training: self.params_calculated.fill_(True)
        return ste_round(torch.clamp(x / self.scale, self.observer.quant_min, self.observer.quant_max))

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class LinearADC(nn.Linear):
    def __init__(self, in_features, out_features, bx=8, bw=8, ba=8, k=4, bias=True, ashift=False):
        super(LinearADC, self).__init__(in_features, out_features, bias)
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)
        self.adc_quantizer = ADCQuantizer(M=in_features, bx=bx, bw=bw, ba=ba, k=k)
        self.ashift = ashift
        self.C = 2 ** (bx - 1)

    def dequantize(self, yq, wq):
        target_device = self.weight.device

        x_zp = self.x_quantizer.zero_point.to(target_device)
        w_s = self.w_quantizer.scale.to(target_device)
        x_s = self.x_quantizer.scale.to(target_device)

        y = yq * self.adc_quantizer.delta
        if (self.ashift):
            y = y + self.C * wq.sum(axis=-1)
        
        out = y - x_zp / w_s * self.weight.sum(axis=-1)
        out = out * x_s * w_s
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
        return out

class LinearADCAshift(LinearADC):
    def __init__(self, in_features, out_features, bx=8, bw=8, ba=8, k=4, ashift_enabled=True, bias=True):
        super(LinearADCAshift, self).__init__(in_features, out_features, bx, bw, ba, k, bias)
        self.C = 2 ** (bx - 1)
        self.ashift_enabled = ashift_enabled # Store ashift_enabled

    def dequantize(self, yq_adc, wq):
        target_device = self.weight.device
        
        x_zp = self.x_quantizer.zero_point.to(target_device)
        w_s = self.w_quantizer.scale.to(target_device)
        x_s = self.x_quantizer.scale.to(target_device)
        
        y = yq_adc * self.adc_quantizer.delta + self.C * wq.sum(axis=-1)
        out = y - x_zp / w_s * self.weight.sum(axis=-1)
        out = out * x_s * w_s
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
        # Активации (аффинное квантование)
        if self.training:
            self.x_quantizer.observer(x.detach())
        if not self.x_quantizer.params_calculated:
            # Вычисляем параметры, если еще не были вычислены (например, в режиме eval)
            self.x_quantizer.scale.copy_(self.x_quantizer.observer.calculate_qparams()[0])
            self.x_quantizer.zero_point.copy_(self.x_quantizer.observer.calculate_qparams()[1])
            if self.training: self.x_quantizer.params_calculated.fill_(True)

        scale_x = self.x_quantizer.scale.to(x.device)
        zp_x = self.x_quantizer.zero_point.to(x.device)
        x_q = ste_round(torch.clamp(x / scale_x + zp_x, 0, 2**self.bx - 1))
        x_dequant = (x_q - zp_x) * scale_x

        # Веса (симметричное квантование)
        if self.training:
            self.w_quantizer.observer(self.weight.detach())
        if not self.w_quantizer.params_calculated:
            self.w_quantizer.scale.copy_(self.w_quantizer.observer.calculate_qparams()[0])
            if self.training: self.w_quantizer.params_calculated.fill_(True)
        
        scale_w = self.w_quantizer.scale.to(self.weight.device)
        q_min = self.w_quantizer.observer.quant_min
        q_max = self.w_quantizer.observer.quant_max
        w_q = ste_round(torch.clamp(self.weight / scale_w, q_min, q_max))
        w_dequant = w_q * scale_w

        return nn.functional.linear(x_dequant, w_dequant, self.bias)


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
                 ashift=False):
        super(Conv2dADC, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k
        self.x_quantizer = AffineQuantizerPerTensor(bx)
        self.w_quantizer = SymmetricQuantizerPerTensor(bw)
        self.ashift=ashift
        self.C = 2 ** (bx - 1)
        if type(kernel_size) == int:
            Mv = in_channels*(kernel_size**2)
        else:
            Mv = in_channels*kernel_size[0]*kernel_size[1]
        self.adc_quantizer = ADCQuantizer(M=Mv, bx=bx, bw=bw, ba=ba, k=k)
    
    def dequantize(self, yq, wq):
        target_device = self.weight.device

        x_zp = self.x_quantizer.zero_point.to(target_device)
        w_s = self.w_quantizer.scale.to(target_device)
        x_s = self.x_quantizer.scale.to(target_device)

        # yq: out x H_out x W_out
        # self.weight: out x in x H_out x W_out
        y = yq * self.adc_quantizer.delta
        if (self.ashift):
            y = y + self.C * (wq.sum(axis=(1, 2, 3)))[None, :, None, None]
        out = y - x_zp / w_s * (self.weight.sum(axis=(1, 2, 3)))[None, :, None, None]
        out = out * x_s * w_s
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
        if (self.ashift):
            xq = xq - self.C
        wq = self.w_quantizer(self.weight)
        y_for_adc = torch.nn.functional.conv2d(xq, 
                                               wq, 
                                               bias=self.bias, 
                                               stride=self.stride, 
                                               padding=self.padding, 
                                               dilation=self.dilation, 
                                               groups=self.groups)
        yq_adc = self.adc_quantizer(y_for_adc)
        out = self.dequantize(yq_adc, wq)
        if self.bias is not None:
            out += self.bias
        return out
    