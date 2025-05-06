from aihwkit.simulator.configs import NoiseModelType
from aihwkit.simulator.noise_models import BaseDriftNoiseModel
import torch
import numpy as np

from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightNoiseType, WeightClipType


class ADCNoiseModel(BaseDriftNoiseModel):
    def __init__(self, bx=8, bw=8, ba=8, k=4):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.ba = ba
        self.k = k
        
        # ADC quantization parameters
        self.M = None  # Will be set during forward pass
        self.delta = None  # Will be computed during forward pass
        
    def forward(self, hidden_states, weights, bias=None):
        # Store input dimensions for ADC scaling
        self.M = weights.shape[1]  # Input features dimension
        
        # Compute ADC step size (delta)
        max_value = self.M * (2**self.bx - 1) * (2**self.bw - 1)
        self.delta = max_value / ((2**self.ba - 1) * self.k)        
        # Simulate ADC quantization
        output = hidden_states.matmul(weights.t())
        if bias is not None:
            output += bias
            
        # Apply ADC quantization
        output_scaled = output / self.delta
        output_quantized = torch.round(output_scaled) * self.delta
        
        return output_quantized
    

def create_adc_tile_config(bx=8, bw=8, ba=8, k=4):
    rpu_config = InferenceRPUConfig()
    
    # Set up the custom ADC noise model
    rpu_config.noise_model = ADCNoiseModel(bx=bx, bw=bw, ba=ba, k=k)
    
    # Configure weight and input quantization
    rpu_config.forward.inp_res = 2**(-bx)  # Input quantization
    rpu_config.forward.out_res = 2**(-ba)  # Output quantization
    rpu_config.forward.w_res = 2**(-bw)    # Weight quantization
    
    # Set up weight clipping
    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    
    return rpu_config
