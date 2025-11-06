import torch
from torch import nn
from ADC.quantizers import AffineQuantizerPerTensor, SymmetricQuantizerPerTensor, ADCQuantizer, ADCQuantizerAshift, LearnableQuantizerPerTensor
import wandb
import random

MVM_LIMIT = 512

from ADC.ste import ste_floor, ste_round
ste_func_global = ste_floor

class BlockedConv2dADC(nn.Conv2d):
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
                 logger=None,
                 name=None,
                 max_block=512):
        super(BlockedConv2dADC, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        if name:
            self.name = name
        else:
            self.name = f"Conv2d" + str(random.randint(10 ** 5, 10**6 - 1))
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k
        #self.x_quantizer = AffineQuantizerPerTensor(bx, "histogram")
        #self.w_quantizer = SymmetricQuantizerPerTensor(bw, "histogram")
        self.x_quantizer = LearnableQuantizerPerTensor(self.bx, "histogram", symmetric=False)
        self.w_quantizer = LearnableQuantizerPerTensor(self.bw, "histogram", symmetric=True)
        self.ashift=ashift
        self.C = 2 ** (bx - 1)
        self.logger = logger
        # if type(kernel_size) == int:
        #     Mv = in_channels*(kernel_size**2)
        # else:
        #     Mv = in_channels*kernel_size[0]*kernel_size[1]
        self.adc_quantizer = ADCQuantizer(M=max_block, bx=bx, bw=bw, ba=ba, k=k, info=self.name, logger=self.logger, ste_func=ste_func_global)
        self.adc_enabled = True
        self.max_block = max_block
    
    def enable_adc(self):
        self.adc_enabled = True
    def disable_adc(self):
        self.adc_enabled = False
    
    def _set_quantizer_state(self, enabled: bool):
        self.x_quantizer.enabled = enabled
        self.w_quantizer.enabled = enabled

    def _batched_matmul(self, x, w, w_orig):
        # x (N, L, K*K*C)
        # w (K*K*C, O)
        # Split x, w into several matrices along dimension L
        Lmax = x.shape[-1]
        while (Lmax > self.max_block):
            assert Lmax % 2 == 0
            Lmax //= 2
        if (self.ashift):
            x = x - self.C
        x_splits = torch.split(x, Lmax, dim=-1)
        w_splits = torch.split(w, Lmax, dim=0)
        w_orig_splits = torch.split(w_orig, Lmax, dim=0)
        assert len(x_splits) == len(w_splits)
        y_for_adc = [x_splits[i] @ w_splits[i] for i in range(len(x_splits))]
        # ADC quantizers are different
        yq_adc = [self.adc_quantizer(y) for y in y_for_adc]
        if (self.logger and self.logger.enabled):
            self.logger.log_data(self, [y_for_adc, yq_adc], ["y_for_adc", "yq_adc"])
        out = [self.dequantize(yq_adc[i], w_splits[i], w_orig_splits[i]) for i in range(len(yq_adc))] # (N, O)_i
        return sum(out)

    def _unfold_conv(self, w, input):
        H_in = input.shape[-2]
        W_in = input.shape[-1]

        inp_unf = torch.nn.functional.unfold(input, kernel_size = self.kernel_size, stride = self.stride, dilation=self.dilation, padding=self.padding) # N, K * K, L
        inp_unf = inp_unf.transpose(1, 2) # N, L, K * K * IN
        w = w.view(w.size(0), -1).t() # K * K * IN, O
        w_orig = self.weight.view(self.weight.size(0), -1).t()
        
        #out_unf = inp_unf @ w # N, L, O
        #print("Internal dims:", inp_unf.shape, w.shape)
        out_unf = self._batched_matmul(inp_unf, w, w_orig)
        
        out_unf = out_unf.transpose(1, 2) # N, O, L
        H_out = ((H_in + 2*self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
        W_out = ((W_in + 2*self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1 
        out = out_unf.view(out_unf.shape[0], out_unf.shape[1], H_out, W_out)
    
        return out 

    def dequantize(self, yq, wq, w_orig):
        # yq: out x H_out x W_out
        # self.weight: out x in x H_out x W_out
        
        # Important!!!!!!!
        y = yq * self.adc_quantizer.delta
        if (self.ashift):
            y = y + self.C * wq.sum(0)
            #y = y + self.C * self.x_quantizer.scale * self.w_quantizer.scale * wq.sum(axis=0)
        #print(y.shape, w_orig.shape)
        #print(w_orig.sum(axis=-1).shape)
        #out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * w_orig.sum(axis=0)
        out = y - self.x_quantizer.zero_point * wq.sum(axis=0)
        out = out * self.x_quantizer.scale * self.w_quantizer.scale
        return out
    
    def train(self, mode=True):
        super().train(mode)
        #if (mode == True):
        #    self.x_quantizer.enable()
        #    self.w_quantizer.enable()
        #else:
        #    self.x_quantizer.disable()
        #    self.w_quantizer.disable()
        return self
    def eval(self, mode=True):
        super().eval(mode)
        self.train(not mode)
        return self
    def forward(self, x):
        #print("Layer input sizes:", x.shape, self.weight.shape)
        # print(self.adc_enabled)
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
            if (self.logger and self.logger.enabled):
                with torch.no_grad():
                    out_gth = torch.nn.functional.conv2d(x, 
                                                self.weight, 
                                                bias=self.bias, 
                                                stride=self.stride, 
                                                padding=self.padding, 
                                                dilation=self.dilation, 
                                                groups=self.groups)
                self.logger.log_data(self, [x, self.weight, xq, wq, out, out_gth], ["x", "w", "xq", "wq", "out", "out_gth"])
            return out
        
        xq = self.x_quantizer(x)
        #if (self.ashift):
        #    xq = xq - self.C
        wq = self.w_quantizer(self.weight)

        out = self._unfold_conv(wq, xq)

        if self.bias is not None:
            out += self.bias

        if self.logger and self.logger.enabled:
            with torch.no_grad():
                out_gth = torch.nn.functional.conv2d(x, 
                                               self.weight, 
                                               bias=self.bias, 
                                               stride=self.stride, 
                                               padding=self.padding, 
                                               dilation=self.dilation, 
                                               groups=self.groups)
            self.logger.log_data(self, [x, self.weight, xq, wq, out, out_gth], ["x", "w", "xq", "wq", "out", "out_gth"])
            self.logger.log_string(self.name, 'delta', self.adc_quantizer.delta)
           

        return out

class LinearADC(nn.Linear):
    def __init__(self, in_features, out_features, bx=8, bw=8, ba=8, k=4, bias=True, ashift=False, logger=None, name=None):
        super(LinearADC, self).__init__(in_features, out_features, bias)
        self.bx = bx
        self.bw = bw
        self.ba = ba
        #self.x_quantizer = AffineQuantizerPerTensor(bx)
        #self.w_quantizer = SymmetricQuantizerPerTensor(bw)
        self.x_quantizer = LearnableQuantizerPerTensor(bx, "histogram", symmetric=False)
        self.w_quantizer = LearnableQuantizerPerTensor(bw, "histogram", symmetric=True)
        self.adc_quantizer = ADCQuantizer(M=in_features, bx=bx, bw=bw, ba=ba, k=k)
        self.ashift = ashift
        self.C = 2 ** (bx - 1)
        self.adc_enabled = True
        if (name):
            self.name = name
        else:
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
        # if (mode == True):
        #     self.x_quantizer.enable()
        #     self.w_quantizer.enable()
        # else:
        #     self.x_quantizer.disable()
        #     self.w_quantizer.disable()
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
                 logger=None,
                 name=None):
        super(Conv2dADC, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        if name:
            self.name = name
        else:
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

    def dequantize(self, yq, wq, xshape=None):
        # yq: out x H_out x W_out
        # self.weight: out x in x H_out x W_out
        
        # Important!!!!!!!
        y = yq * self.adc_quantizer.delta
        #y = yq
        # -------------------------------------


        mask = torch.nn.functional.conv2d(torch.ones(xshape[1:]).unsqueeze(0).to(self.weight.device), 
                                               self.weight, 
                                               bias=None, 
                                               stride=self.stride, 
                                               padding=self.padding, 
                                               dilation=self.dilation, 
                                               groups=self.groups)

        if (self.ashift):
            #print("ashift dequantize")
            # Here we didn't take padding into account
            #y = y + self.C * (wq.sum(axis=(1, 2, 3)))[None, :, None, None]
            mask2 = torch.nn.functional.conv2d(torch.ones(xshape[1:]).unsqueeze(0).to(wq.device), 
                                               wq, 
                                               bias=None, 
                                               stride=self.stride, 
                                               padding=self.padding, 
                                               dilation=self.dilation, 
                                               groups=self.groups)
            y = y + self.C * mask2
        
        #out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * (self.weight.sum(axis=(1, 2, 3)))[None, :, None, None] # We might have an error here because of the padding
        out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * mask
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
        out = self.dequantize(yq_adc, wq, x.shape)
        if self.bias is not None:
            out += self.bias
        if self.logger and self.logger.enabled:
            with torch.no_grad():
                out_gth = torch.nn.functional.conv2d(x, 
                                               self.weight, 
                                               bias=self.bias, 
                                               stride=self.stride, 
                                               padding=self.padding, 
                                               dilation=self.dilation, 
                                               groups=self.groups)
                self.logger.log_data(self, [x, self.weight, xq, wq, y_for_adc, yq_adc, out, out_gth], ["x", "w", "xq", "wq", "y_for_adc", "yq_adc", "out", "out_gth"])
                
            #     self.logger.log(self.name, "max_val", y_for_adc.max().item())
            #     self.logger.log(self.name, "min_val", y_for_adc.min().item())
            #     diff = torch.linalg.norm(out - out_gth).cpu().item()
            #     gth_norm = torch.linalg.norm(out_gth).cpu().item()
            #     self.logger.log(self.name, "out_norm_ratio", diff / gth_norm)
            #     print(self.name + "_diff: ", diff / gth_norm)

        return out

# class BlockedConv2dADC(nn.Conv2d):
#     def __init__(self,
#                  in_channels, 
#                  out_channels, 
#                  kernel_size, 
#                  stride=1, 
#                  padding=0, 
#                  dilation=1, 
#                  groups=1, 
#                  bias=None, 
#                  padding_mode='zeros', 
#                  device=None, 
#                  dtype=None,
#                  bx=8,
#                  bw=8,
#                  ba=8,
#                  k=4,
#                  ashift=False,
#                  logger=None,
#                  name=None,
#                  max_block=512):
#         super(BlockedConv2dADC, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
#         if name:
#             self.name = name
#         else:
#             self.name = f"Conv2d" + str(random.randint(10 ** 5, 10**6 - 1))
#         self.bx = bx
#         self.bw = bw
#         self.ba = ba
#         self.k = k
#         self.x_quantizer = AffineQuantizerPerTensor(bx)
#         self.w_quantizer = SymmetricQuantizerPerTensor(bw)
#         self.ashift=ashift
#         self.C = 2 ** (bx - 1)
#         self.logger = logger
#         if type(kernel_size) == int:
#             Mv = in_channels*(kernel_size**2)
#         else:
#             Mv = in_channels*kernel_size[0]*kernel_size[1]
#         self.adc_quantizer = ADCQuantizer(M=Mv, bx=bx, bw=bw, ba=ba, k=k, info=self.name, logger=self.logger)
#         self.adc_enabled = True
#         self.max_block = max_block
    
#     def enable_adc(self):
#         self.adc_enabled = True
#     def disable_adc(self):
#         self.adc_enabled = False
    
#     def _set_quantizer_state(self, enabled: bool):
#         self.x_quantizer.enabled = enabled
#         self.w_quantizer.enabled = enabled

#     def _batched_matmul(self, x, w):
#         # x (N, L, K)
#         # w (K, O)
#         # Split x, w into several matrices along dimension L
#         x_splits = torch.split(x, self.max_block, dim=-1)
#         w_splits = torch.split(w, self.max_block, dim=0)
#         assert len(x_splits) == len(w_splits)
#         y_for_adc = [x_splits[i] @ w_splits[i] for i in range(len(x_splits))]
#         yq_adc = [self.adc_quantizer(y) for y in y_for_adc]
#         out = [self.dequantize(yq_adc[i], w_adc[i]) for i in range(len(yq_adc))] # (N, O)_i
#         if (self.logger and self.logger.enabled):
#             self.logger.log_data(self, [y_for_adc, yq_adc], ["y_for_adc", "yq_adc"])
#         return sum(out)

#     def _unfold_conv(self, w, input):
#         H_in = input.shape[-2]
#         W_in = input.shape[-1]

#         inp_unf = torch.nn.functional.unfold(input, kernel_size = self.kernel_size, stride = self.stride, dilation=self.dilation, padding=self.padding) # N, K * K, L
#         inp_unf = inp_unf.transpose(1, 2) # N, L, K * K * IN
#         w = w.view(w.size(0), -1).t() # K * K * IN, O
        
        
#         #out_unf = inp_unf @ w # N, L, O
#         out_unf = self._batched_matmul(inp_unf, w)
        
#         out_unf = out_unf.transpose(1, 2) # N, O, L
#         H_out = ((H_in + 2*self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0]) + 1
#         W_out = ((W_in + 2*self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1]) + 1 
#         out = out_unf.view(out_unf.shape[0], out_unf.shape[1], H_out, W_out)
    
#         return out 

#     def dequantize(self, yq, wq):
#         # yq: out x H_out x W_out
#         # self.weight: out x in x H_out x W_out
        
#         # Important!!!!!!!
#         y = yq * self.adc_quantizer.delta
#         if (self.ashift):
#             y = y + self.C * wq.sum(axis=-1)
#         out = y - self.x_quantizer.zero_point / self.w_quantizer.scale * self.weight.sum(axis=-1)
#         out = out * self.x_quantizer.scale * self.w_quantizer.scale
#         return out
    
#     def train(self, mode=True):
#         super().train(mode)
#         if (mode == True):
#             self.x_quantizer.enable()
#             self.w_quantizer.enable()
#         else:
#             self.x_quantizer.disable()
#             self.w_quantizer.disable()
#         return self
#     def eval(self, mode=True):
#         super().eval(mode)
#         self.train(not mode)
#         return self
#     def forward(self, x):
#         if (not self.adc_enabled):
#             xq = self.x_quantizer.fake_quantize(x)
#             wq = self.w_quantizer.fake_quantize(self.weight)
#             out = torch.nn.functional.conv2d(xq, 
#                                                wq, 
#                                                bias=self.bias, 
#                                                stride=self.stride, 
#                                                padding=self.padding, 
#                                                dilation=self.dilation, 
#                                                groups=self.groups)
#             return out
        
#         xq = self.x_quantizer(x)
#         if (self.ashift):
#             xq = xq - self.C
#         wq = self.w_quantizer(self.weight)

#         out = self._unfold_conv(wq, xq)

#         if self.bias is not None:
#             out += self.bias

#         if self.logger and self.logger.enabled:
#             with torch.no_grad():
#                 out_gth = torch.nn.functional.conv2d(x, 
#                                                self.weight, 
#                                                bias=self.bias, 
#                                                stride=self.stride, 
#                                                padding=self.padding, 
#                                                dilation=self.dilation, 
#                                                groups=self.groups)
#             self.logger.log_data(self, [x, self.weight, xq, wq, out, out_gth], ["x", "w", "xq", "wq", "out", "out_gth"])
           

#         return out


class TiledConv2dADC(nn.Module):
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
                 logger=None,
                 name=None):
        super(TiledConv2dADC, self).__init__()
        
        if (name):
            self.name = name
        else:
            self.name = "TiledConv2ADC" + str(random.randint(10 ** 5, 10**6 - 1))

        if type(kernel_size) == int:
            ksz = (kernel_size**2)
        else:
            ksz = kernel_size[0]*kernel_size[1]
        n_conv = 1
        while (ksz * in_channels > MVM_LIMIT and in_channels % 2 == 0):
            n_conv *= 2
            in_channels //= 2
        if (ksz * in_channels > MVM_LIMIT):
            raise ValueError("Number of input channels is not divided by power of 2")
        self.convs = nn.ModuleList()
        self.in_channels = in_channels
        self.logger = logger

        for i in range(n_conv):
            bias = bias if i == 0 else None
            self.convs.append(Conv2dADC(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype, bx, bw, ba, k, ashift, logger, self.name + f"_{i+1}/{n_conv}"))
    
    def enable_adc(self):
        for i in range(len(self.convs)):
            self.convs[i].enable_adc()
    def disable_adc(self):
        for i in range(len(self.convs)):
            self.convs[i].disable_adc()
    
    def _set_quantizer_state(self, enabled: bool):
        for i in range(len(self.convs)):
            self.convs[i]._set_quantizer_state(enabled)
    
    def train(self, mode=True):
        for i in range(len(self.convs)):
            self.convs[i].train(mode)
        return self

    def eval(self, mode=True):
        for i in range(len(self.convs)):
            self.convs[i].train(mode)
        return self

    def load_weights(self, conv):
        wshape = conv.weight.shape  # (out_channels, in_channels, kH, kW)
        ksz = wshape[2] * wshape[3]
        in_channels = wshape[1]
        n_conv = len(self.convs)
        split_in = in_channels // n_conv

        # Split weights and biases along the input channel axis
        w_splits = torch.split(conv.weight, split_in, dim=1)
        if conv.bias is not None:
            bias = conv.bias
        else:
            bias = None

        for i, subconv in enumerate(self.convs):
            subconv.weight.data.copy_(w_splits[i].clone())
            if bias is not None and subconv.bias is not None:
                subconv.bias.data.copy_(bias.clone())
        

    def forward(self, x):
        if (len(x.shape) == 4):
            # (N, C_in, H, W)
            results = [self.convs[i](x[:,self.in_channels * i : self.in_channels * (i + 1),:,:]) for i in range(len(self.convs))]
            return sum(results)
        elif (len(x.shape) == 3):
            # (C_in, H, W)
            results = [self.convs[i](x[self.in_channels * i : self.in_channels * (i + 1),:,:]) for i in range(len(self.convs))]
            return sum(results)
        else:
            raise ValueError("Incorrect dimension of input tensor")
