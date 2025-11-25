import torch
from typing import List, Optional, Tuple, Union
from train.utils import *

@torch.jit.script
def adc_quan(x, qrange:int, mode:str, b_adc:int, quant_level:int, quant_interval:float, dtype:torch.dtype=torch.float32):
    # quantize input data
    assert mode == 'analog'
    if mode != 'analog':
        raise NotImplementedError
    else:
        interval_inv = 1.0 / quant_interval
        #xq = torch.floor((x + qrange) * interval_inv)
        xq = floor_ste((x + qrange) * interval_inv)

    xq = torch.clip(xq, 0, quant_level - 1)
    
    # convert to the original range
    xq = xq * quant_interval
    if mode == 'analog':
        xq = xq - qrange

    xq = xq.to(dtype=dtype)

    return xq

