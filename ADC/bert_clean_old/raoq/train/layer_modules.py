import torch
import torch.nn as nn
from torch.autograd import Function
import numpy
import torch.nn.functional as F
from train.quantize_modules import *
from train.chip_modules import *
from train.layer_base import *
import warnings
import numpy as np
import sys
import pdb


class QuantLinearSensitive(LinearQuantInterfaceSensitive, nn.Linear):

    def __init__(self, *args, imc_mode:bool=False, mode:str='and', bx_int:int=0, 
                 bw_int:int=0, bx_shift:int=0, adc_quant:bool=False, 
                 b_adc:int=9, quant_flag:bool=False, by_scale:int=0, by_shift:int=0, 
                 set_bits:bool=False, noise:bool=False, writer=None, name='', **kwargs):
        super(QuantLinearSensitive, self).__init__(*args, **kwargs)
        self.imc_mode = False
        self.mode = mode
        self.bx_int = bx_int
        self.bw_int = bw_int
        self.bx_shift = bx_shift
        self.adc_quant = adc_quant
        self.b_adc = b_adc
        self.name = name
       
        self.quant_flag = quant_flag
        self.set_bits = set_bits
        self.noise = noise
        self.by_scale = by_scale
        self.by_shift = by_shift
        self.bank_size = 4
        self.strip_size = 4
        self.short_path = self.bw == self.strip_size and self.bx == self.bank_size
        self.bit_mask_x = torch.sum(2 ** torch.arange(self.bank_size))
        self.bit_mask_w = torch.sum(2 ** torch.arange(self.strip_size))
        self.writer = writer

        if self.mode != 'analog':
            self.bank_size, self.strip_size = 1, 1
 

    def forward(self, input, name:str=''):
        # quantize input
        input = self.x_quantizer(input, training=self.training) 

        # quantize weight
        weight = self.w_quantizer(self.weight, training=self.training)

        # convolution
        if not self.imc_mode:
            out = F.linear(input, weight, None)
        else:
            raise NotImplementedError

        if self.bias is not None:
            out += self.bias

        return out


class QuantConv2dSensitive(ConvQuantInterfaceSensitive, nn.Conv2d):

    def __init__(self, *args, imc_mode:bool=False, mode:str='and', bx_int:int=0, 
                 bw_int:int=0, bx_shift:int=0, x_signed:bool=False, adc_quant:bool=False, 
                 b_adc:int=9, quant_flag:bool=False, by_scale:int=0, by_shift:int=0, 
                 set_bits:bool=False, noise:bool=False, writer=None, name:str='', **kwargs):
        super(QuantConv2dSensitive, self).__init__(*args, **kwargs)
        self.imc_mode = False
        self.mode = mode
        self.bx_int = bx_int
        self.bw_int = bw_int
        self.bx_shift = bx_shift
        self.x_signed = x_signed
        self.adc_quant = adc_quant
        self.b_adc = b_adc
       
        self.quant_flag = quant_flag
        self.set_bits = set_bits
        self.noise = noise
        self.by_scale = by_scale
        self.by_shift = by_shift
        self.bank_size = 4
        self.strip_size = 4
        self.short_path = self.bw == self.strip_size and self.bx == self.bank_size
        self.bit_mask_x = torch.sum(2 ** torch.arange(self.bank_size))
        self.bit_mask_w = torch.sum(2 ** torch.arange(self.strip_size))
        self.writer = writer

        if self.mode != 'analog':
            self.bank_size, self.strip_size = 1, 1
 

    def forward(self, input, name:str=''):
        # quantize input
        input = self.x_quantizer(input, training=self.training) 

        # quantize weight
        weight = self.w_quantizer(self.weight, training=self.training)

        # convolution
        if not self.imc_mode:
            out = nn.functional.conv2d(input, weight, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            raise NotImplementedError

        return out


class QuantConv2dDW(ConvQuantInterface, nn.Conv2d):

    def __init__(self, *args, imc_mode:bool=False, mode:str='and', bx_int:int=0, 
                 bw_int:int=0, bx_shift:int=0, adc_quant:bool=False, 
                 b_adc:int=9, quant_flag:bool=False, by_scale:int=0, by_shift:int=0, 
                 set_bits:bool=False, noise:bool=False, writer=None, name='', **kwargs):
        super(QuantConv2dDW, self).__init__(*args, **kwargs)
        self.imc_mode = imc_mode
        self.mode = mode
        self.bx_int = bx_int
        self.bw_int = bw_int
        self.bx_shift = bx_shift
        self.adc_quant = adc_quant
        self.b_adc = b_adc
       
        self.quant_flag = quant_flag
        self.set_bits = set_bits
        self.noise = noise
        self.by_scale = by_scale
        self.by_shift = by_shift
        self.bank_size = 4
        self.strip_size = 4
        self.short_path = self.bw == self.strip_size and self.bx == self.bank_size
        self.bit_mask_x = torch.sum(2 ** torch.arange(self.bank_size))
        self.bit_mask_w = torch.sum(2 ** torch.arange(self.strip_size))
        self.writer = writer

        if self.mode != 'analog':
            self.bank_size, self.strip_size = 1, 1
 

    def forward(self, input, name:str=''):
        # determine quantization range of input
        if self._set_range > 0:
            self.x_quantizer.compute_scale(input, False, 'dist')

        # quantize input
        input_fp = input.clone()
        input, scale_x, bias_x = self.x_quantizer(input) 

        # determine quantization range of weight
        if self._set_range > 0:
            self.w_quantizer.compute_scale(self.weight, False, 'dist', act=False)

        # quantize weight
        weight, scale_w, _ = self.w_quantizer(self.weight)

        # convolution
        out = nn.functional.conv2d(input, weight, None, self.stride, self.padding, self.dilation, self.groups)
        if self.x_bias:
            conv_bias = nn.functional.conv2d(bias_x.repeat(input.shape[0], input.shape[1], input.shape[2], input.shape[3]), weight, None, self.stride, self.padding, self.dilation, self.groups)
            # dequantize
            out = scale_w * (scale_x * out + conv_bias)
        else:
            out = scale_x * scale_w * out

        # perform fp compute for the 1st (a few) batch(es)
        if self.run_1st_batch_fp and self._first_batch > 0:
            #out += (super(QuantConv2d, self).forward(input_fp) - out).detach()
            out += (F.conv2d(input_fp, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
                    - out).detach()

        if self.set_range_once:
            self._set_range = 0

        return out, 0


class QuantConv2d(ConvQuantInterface, nn.Conv2d):

    def __init__(self, *args, imc_mode:bool=False, mode:str='and', bx_int:int=0, 
                 bw_int:int=0, bx_shift:int=0, adc_quant:bool=False, 
                 b_adc:int=9, quant_flag:bool=False, by_scale:int=0, by_shift:int=0, 
                 set_bits:bool=False, noise:bool=False, writer=None, name='', **kwargs):
        super(QuantConv2d, self).__init__(*args, **kwargs)
        self.imc_mode = imc_mode
        self.mode = mode
        self.bx_int = bx_int
        self.bw_int = bw_int
        self.bx_shift = bx_shift
        self.adc_quant = adc_quant
        self.b_adc = b_adc
        self.name = name
       
        #self.quant_flag = False if not self.train_scale else quant_flag
        self.quant_flag = quant_flag
        self.set_bits = set_bits
        self.noise = noise
        self.by_scale = by_scale
        self.by_shift = by_shift
        self.bank_size = 4
        self.strip_size = 4
        self.short_path = self.bw == self.strip_size and self.bx == self.bank_size
        self.bit_mask_x = torch.sum(2 ** torch.arange(self.bank_size))
        self.bit_mask_w = torch.sum(2 ** torch.arange(self.strip_size))
        self.writer = writer

        if self.mode != 'analog':
            self.bank_size, self.strip_size = 1, 1
 

    def forward(self, input, adc_bit, name:str=''):
        # determine quantization range of input
        if self._set_range > 0:
            self.x_quantizer.compute_scale(input, False, 'dist')

        # quantize input
        input_fp = input.clone()
        input, scale_x, bias_x = self.x_quantizer(input) 

        # loss on weight
        kurt = ((self.weight - self.weight.mean()) / (self.weight.std())) ** 4
        kurt = kurt.mean()
        loss_w = (kurt - 1.8) ** 2

        # determine quantization range of weight
        if self._set_range > 0:
            self.w_quantizer.compute_scale(self.weight, False, 'dist', act=False)

        # quantize weight
        weight, scale_w, _ = self.w_quantizer(self.weight)

        # convolution
        if not self.imc_mode:
            out = nn.functional.conv2d(input, weight, None, self.stride, self.padding, self.dilation, self.groups)
            if self.x_bias:
                conv_bias = nn.functional.conv2d(bias_x.repeat(input.shape[0], input.shape[1], input.shape[2], input.shape[3]), weight, None, self.stride, self.padding, self.dilation, self.groups)
                # dequantize
                out = scale_w * (scale_x * out + conv_bias)
            else:
                out = scale_x * scale_w * out
        else:
            # zero padding
            zero_pad = (self.padding[0], )*4
            if not self.quant_flag:
                input_pad = nn.functional.pad(input, zero_pad, 'constant', 0.)
            else:
                input_pad = nn.functional.pad(input_fp, zero_pad, 'constant', 0.)

            # binarization
            if self.mode != 'analog':
                raise NotImplementedError
            else:
                if self.short_path:
                    x_bin = 2 * input_pad - (2**self.bx - 1)
                    w_bin = weight * 2 + 1
                else:
                    raise NotImplementedError
                bias = F.conv2d(x_bin, torch.ones(w_bin.shape, dtype=x_bin.dtype, device=x_bin.device), None, self.stride, 0, self.dilation, self.groups) \
                        - (2**self.bx - 1) * F.conv2d(torch.ones(x_bin.shape, dtype=x_bin.dtype, device=x_bin.device), w_bin, None, self.stride, 0, self.dilation, self.groups) \
                        + (2**self.bx - 1) * F.conv2d(torch.ones(x_bin.shape, device=x_bin.device), torch.ones(w_bin.shape, device=w_bin.device), None, self.stride, 0, self.dilation, self.groups)
                bias = bias.to(torch.float32)

            # ADC quantization
            if self.adc_quant:
                quant_level = int(2 ** adc_bit)
                wdim = 768
                if self.mode != 'analog':
                    raise NotImplementedError
                else:
                    qrange = int((2**self.bx-1) * (2**self.bw-1) * wdim) 
                    quant_interval = 2 * qrange / quant_level

                # IMC MVM
                w_row = np.prod(w_bin.shape[1:])
                num_part = math.ceil(w_row / wdim)
                out_bin = 0
                ch_start = 0
                for i in range(num_part):
                    if w_row >= wdim:
                        eff_row = wdim
                        w_row -= wdim
                        num_copy = 1
                        ch_step = wdim // np.prod(w_bin.shape[2:])
                    else:
                        eff_row = w_row
                        num_copy = wdim // eff_row
                        ch_step = w_row // np.prod(w_bin.shape[2:])
                    w_bin_temp = w_bin[:,ch_start:ch_start+ch_step,:,:]
                    x_bin_temp = x_bin[:,ch_start:ch_start+ch_step,:,:]
                    temp_out = num_copy * nn.functional.conv2d(x_bin_temp, w_bin_temp, None, self.stride, 0, self.dilation, self.groups)
                    temp_out = adc_quan(
                                   temp_out, 
                                   qrange=qrange, 
                                   mode=self.mode,
                                   b_adc=adc_bit, 
                                   quant_level=quant_level, 
                                   quant_interval=quant_interval,
                                   dtype=torch.float32)
                    out_bin += (temp_out / num_copy)
                    ch_start += ch_step
                out_bin = out_bin.to(torch.float32)

            # multi-bit reconstruction
            if self.mode == 'analog' and self.short_path:
                out = 0.25 * (out_bin - bias)
            else:
                raise NotImplementedError

            if self.x_bias:
                conv_bias = nn.functional.conv2d(bias_x.repeat(input.shape[0], input.shape[1], input.shape[2], input.shape[3]), weight, None, self.stride, self.padding, self.dilation, self.groups)
                # dequantize
                out = scale_w * (scale_x * out + conv_bias)
            else:
                out = scale_x * scale_w * out

        if self.set_range_once:
            self._set_range = 0

        return out, loss_w


