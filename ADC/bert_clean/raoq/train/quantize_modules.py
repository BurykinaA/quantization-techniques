import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from typing import Union
from torch import Tensor
from train.utils import round_ste, grad_scale
import pdb


class QuantizerBias(nn.Module):
    def __init__(self, num_bits:int, signed:bool, symmetric:bool, train_scale:bool):
        super(QuantizerBias,self).__init__()

        assert not signed
        if not signed:
            self.qrange = 2 ** (num_bits) - 1
        else:
            assert symmetric is True
            if symmetric:
                self.qrange = 2 ** (num_bits-1) - 1
            else:
                self.qrange = 2 ** (num_bits-1) - 0.5

        self.signed = signed
        self.symmetric = symmetric
        self.train_scale = train_scale
        self.register_parameter('scale', Parameter(torch.Tensor(1)))
        self.register_parameter('bias', Parameter(torch.zeros(1)))


    @torch.jit.export
    def compute_scale(self, x, channelwise:bool, range_type:str):
        assert self.train_scale
        if self.train_scale:
            assert not channelwise
            self.scale.data.copy_(x.detach().abs().mean() * 2 / math.sqrt(self.qrange))
            #self.bias.data.copy_(torch.min(x))
        else:
            raise NotImplementedError
            
            
    @torch.jit.export
    def get_params(self):
        return self.scale.detach(), self.bias.detach()


    def forward(self, x):
        grad_factor = 1.0 / (math.sqrt(self.qrange * x.numel()))
        scale = grad_scale(self.scale, grad_factor)
        bias = grad_scale(self.bias, grad_factor)

        x_norm = (x - bias) / scale
        if not self.signed:
            x_clip = torch.clip(x_norm, 0, self.qrange)
        else:
            if self.symmetric:
                x_clip = torch.clip(x_norm, -self.qrange, self.qrange)
            else:
                x_clip = torch.clip(x_norm, -self.qrange - 0.5, self.qrange - 0.5)
        x_round = round_ste(x_clip)
        return x_round, scale, bias


class QuantizerScale(nn.Module):
    def __init__(self, num_bits:int, signed:bool, symmetric:bool, train_scale:bool):
        super(QuantizerScale,self).__init__()

        if not signed:
            self.qrange = 2 ** (num_bits) - 1
        else:
            if symmetric:
                self.qrange = 2 ** (num_bits-1) - 1
            else:
                self.qrange = 2 ** (num_bits-1) - 0.5

        self.signed = signed
        self.symmetric = symmetric
        self.train_scale = train_scale
        self.register_parameter('scale', Parameter(torch.Tensor(1)))


    @torch.jit.export
    def compute_scale(self, x, channelwise:bool, range_type:str, act:bool=True):
        if self.train_scale:
            self.scale.data.copy_(x.detach().abs().mean() * 2 / math.sqrt(self.qrange))
        else:
            raise NotImplementedError
            

    @torch.jit.export
    def get_params(self):
        return self.scale.detach(), 0


    def forward(self, x):
        grad_factor = 1.0 / (math.sqrt(self.qrange * x.numel()))
        scale = grad_scale(self.scale, grad_factor)

        x_norm = x / scale
        if not self.signed:
            x_clip = torch.clip(x_norm, 0, self.qrange)
        else:
            if self.symmetric:
                x_clip = torch.clip(x_norm, -self.qrange, self.qrange)
            else:
                x_clip = torch.clip(x_norm, -self.qrange - 0.5, self.qrange - 0.5)
        x_round = round_ste(x_clip)
        return x_round, scale, 0 


class QuantizerSensitive(nn.Module):
    def __init__(self, num_bits:int, integer_bits:int, clip_negative:bool, symmetric:bool, train_scale:bool):
        super(QuantizerSensitive,self).__init__()
        if clip_negative:
            self.qrange = 2 ** (num_bits) - 1
        else:
            if symmetric:
                self.qrange = 2 ** (num_bits-1) - 1
            else:
                self.qrange = 2 ** (num_bits-1) - 0.5

        self.qrange_i = 2 ** integer_bits
        self.clip_negative = clip_negative
        self.symmetric = symmetric
        self.train_scale = train_scale
        self.num_bits = num_bits
        self.register_buffer('min_val', torch.zeros(1), persistent=True)
        self.register_buffer('max_val', torch.zeros(1), persistent=True)


    def ema(self, min_val, max_val, fac:float=0.1):
        self.min_val = fac * min_val + (1 - fac) * self.min_val
        self.max_val = fac * max_val + (1 - fac) * self.max_val


    def forward(self, x, training:bool=True):
        if training:
            with torch.no_grad():
                min_val, max_val = x.detach().data.min(), x.detach().data.max()
                self.ema(min_val, max_val)
        else:
            min_val, max_val = self.min_val, self.max_val

        x_norm = (x - min_val) / (max_val - min_val)
        qrange = 2 ** self.num_bits - 1
        x_round = round_ste(x_norm * qrange)
        x_quan = x_round / qrange * (max_val - min_val) + min_val

        return x_quan


