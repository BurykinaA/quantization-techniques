import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import from ADC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantizers import WeightQuantizer, ActivationQuantizer

class QuantizedBertSelfAttention(nn.Module):
    def __init__(self, original_attention, weight_bit_width=8, activation_bit_width=8):
        super(QuantizedBertSelfAttention, self).__init__()
        self.original_attention = original_attention
        
        # Create quantizers for weights
        self.query_weight_quantizer = WeightQuantizer(bit_width=weight_bit_width)
        self.key_weight_quantizer = WeightQuantizer(bit_width=weight_bit_width)
        self.value_weight_quantizer = WeightQuantizer(bit_width=weight_bit_width)
        self.output_weight_quantizer = WeightQuantizer(bit_width=weight_bit_width)
        
        # Create quantizers for activations
        self.input_act_quantizer = ActivationQuantizer(bit_width=activation_bit_width)
        self.attention_act_quantizer = ActivationQuantizer(bit_width=activation_bit_width)
        
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # Quantize input activations
        hidden_states_q = self.input_act_quantizer(hidden_states)
        
        # Get the original attention module components
        query = self.original_attention.query
        key = self.original_attention.key
        value = self.original_attention.value
        
        # Quantize weights
        query_weight_q = self.query_weight_quantizer(query.weight)
        key_weight_q = self.key_weight_quantizer(key.weight)
        value_weight_q = self.value_weight_quantizer(value.weight)
        
        # Apply quantized weights to linear layers
        query_layer = torch.nn.functional.linear(hidden_states_q, query_weight_q, query.bias)
        key_layer = torch.nn.functional.linear(hidden_states_q, key_weight_q, key.bias)
        value_layer = torch.nn.functional.linear(hidden_states_q, value_weight_q, value.bias)
        
        # Continue with the original attention mechanism
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.original_attention.attention_head_size, 
                                                                    dtype=attention_scores.dtype))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # Quantize attention probabilities
        attention_probs_q = self.attention_act_quantizer(attention_probs)
        
        if head_mask is not None:
            attention_probs_q = attention_probs_q * head_mask
            
        context_layer = torch.matmul(attention_probs_q, value_layer)
        
        # Apply output projection with quantized weights
        output_weight_q = self.output_weight_quantizer(self.original_attention.output.dense.weight)
        output = torch.nn.functional.linear(context_layer, output_weight_q, 
                                          self.original_attention.output.dense.bias)
        
        return output

class QuantizedBertLayer(nn.Module):
    def __init__(self, original_layer, weight_bit_width=8, activation_bit_width=8):
        super(QuantizedBertLayer, self).__init__()
        self.original_layer = original_layer
        
        # Quantize attention
        self.attention = QuantizedBertSelfAttention(
            original_layer.attention.self, 
            weight_bit_width=weight_bit_width,
            activation_bit_width=activation_bit_width
        )
        
        # Create quantizers for intermediate and output weights
        self.intermediate_weight_quantizer = WeightQuantizer(bit_width=weight_bit_width)
        self.output_weight_quantizer = WeightQuantizer(bit_width=weight_bit_width)
        
        # Create quantizers for activations
        self.intermediate_act_quantizer = ActivationQuantizer(bit_width=activation_bit_width)
        
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        
        # Apply layer norm (not quantized as it's parameter-free)
        attention_output = self.original_layer.attention.output.LayerNorm(
            attention_output + hidden_states
        )
        
        # Intermediate layer with quantized weights
        intermediate_weight_q = self.intermediate_weight_quantizer(
            self.original_layer.intermediate.dense.weight
        )
        intermediate_output = torch.nn.functional.linear(
            attention_output, 
            intermediate_weight_q,
            self.original_layer.intermediate.dense.bias
        )
        
        # Apply activation function
        intermediate_output = self.original_layer.intermediate.intermediate_act_fn(intermediate_output)
        
        # Quantize intermediate activations
        intermediate_output_q = self.intermediate_act_quantizer(intermediate_output)
        
        # Output layer with quantized weights
        output_weight_q = self.output_weight_quantizer(
            self.original_layer.output.dense.weight
        )
        layer_output = torch.nn.functional.linear(
            intermediate_output_q,
            output_weight_q,
            self.original_layer.output.dense.bias
        )
        
        # Apply layer norm (not quantized)
        layer_output = self.original_layer.output.LayerNorm(layer_output + attention_output)
        
        return layer_output 