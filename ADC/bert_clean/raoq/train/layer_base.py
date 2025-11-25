import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from train.quantize_modules import QuantizerBias, QuantizerScale, QuantizerSensitive
from typing import Optional, Union


class LinearQuantInterfaceSensitive(nn.Linear):
    def __init__(self, *args, bx:int=8, bw:int=8, bbias:int=32, train_scale:bool=False, x_signed:bool=False, 
                 x_range_type:Optional[str]=None, w_range_type:Optional[str]=None, bias_range_type:Optional[str]=None,
                 set_range_once:bool=True, run_1st_batch_fp:bool=True, fp_mode:bool=False, x_bias:bool=False, **kwargs):
        super(LinearQuantInterfaceSensitive, self).__init__(*args, **kwargs)

        self.bx = 8
        self.bw = 8
        self.train_scale = train_scale
        self.x_range_type = x_range_type
        self.w_range_type = w_range_type
        self.x_signed = x_signed
        self.set_range_once = set_range_once
        self.run_1st_batch_fp = run_1st_batch_fp

        self.x_quantizer = QuantizerSensitive(num_bits=self.bx, integer_bits=0, 
                                     clip_negative=True, symmetric=False, train_scale=train_scale)
        self.w_quantizer = QuantizerSensitive(num_bits=self.bw, integer_bits=0, 
                                     clip_negative=False, symmetric=True, train_scale=train_scale)

        self._set_range = 0
        self._first_batch = 0
        self._running_tracked = 0


class ConvQuantInterfaceSensitive(nn.Conv2d):
    def __init__(self, *args, bx:int=8, bw:int=8, bbias:int=32, train_scale:bool=False, x_signed:bool=False,
                 x_range_type:Optional[str]=None, w_range_type:Optional[str]=None, bias_range_type:Optional[str]=None,
                 set_range_once:bool=True, run_1st_batch_fp:bool=True, fp_mode:bool=False, x_bias:bool=False, **kwargs):
        super(ConvQuantInterfaceSensitive, self).__init__(*args, **kwargs)

        self.bx = 8
        self.bw = 8
        self.bbias = bbias
        self.train_scale = train_scale
        self.x_range_type = x_range_type
        self.w_range_type = w_range_type
        self.bias_range_type = bias_range_type
        self.set_range_once = set_range_once
        self.run_1st_batch_fp = run_1st_batch_fp

        self.x_quantizer = QuantizerSensitive(num_bits=self.bx, integer_bits=0, 
                                     clip_negative=False, symmetric=True, train_scale=train_scale)
        self.w_quantizer = QuantizerSensitive(num_bits=self.bw, integer_bits=0, 
                                     clip_negative=False, symmetric=True, train_scale=train_scale)

        self._set_range = 0
        self._first_batch = 0
        self._running_tracked = False


class ConvQuantInterface(nn.Conv2d):
    def __init__(self, *args, bx:int=8, bw:int=8, bbias:int=32, train_scale:bool=False, x_signed:bool=False, 
                 x_range_type:Optional[str]=None, w_range_type:Optional[str]=None, bias_range_type:Optional[str]=None,
                 set_range_once:bool=True, run_1st_batch_fp:bool=True, fp_mode:bool=False, x_bias:bool=False, **kwargs):
        super(ConvQuantInterface, self).__init__(*args, **kwargs)

        self.bx = bx
        self.bw = bw
        self.bbias = bbias
        self.train_scale = train_scale
        self.x_range_type = x_range_type
        self.w_range_type = w_range_type
        self.x_signed = x_signed
        self.bias_range_type = bias_range_type
        self.set_range_once = set_range_once
        self.run_1st_batch_fp = run_1st_batch_fp

        self.x_bias = x_bias
        if x_bias:
            self.x_quantizer = torch.jit.script(
                                        QuantizerBias(
                                            num_bits=bx, 
                                            signed=self.x_signed, 
                                            symmetric=self.x_signed, 
                                            train_scale=train_scale,
                                                    )
                                               )
        else:
            self.x_quantizer = torch.jit.script(
                                        QuantizerScale(
                                            num_bits=bx, 
                                            signed=self.x_signed, 
                                            symmetric=self.x_signed, 
                                            train_scale=train_scale,
                                                    )
                                               )
        self.w_quantizer = torch.jit.script(
                                    QuantizerScale(
                                        num_bits=bw, 
                                        signed=True, 
                                        symmetric=True, 
                                        train_scale=train_scale
                                                )
                                           )
        self._set_range = 0
        self._first_batch = 0
        self._running_tracked = False

            
