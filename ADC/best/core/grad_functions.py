import torch


def floor_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.floor(x) - x).detach()


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()

####

class SafeDivideFunction(torch.autograd.Function):
    MAX_GRAD_NORM = 1000.0  # Maximum gradient norm for scale
    
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x, scale)
        return x / scale
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        
        # Gradient for x: grad_output / scale
        grad_x = grad_output / scale
        
        # Gradient for scale: -grad_output * x / scale²
        grad_scale = -grad_output * x / (scale ** 2)
        
        # Clip by norm to prevent explosion while preserving direction
        grad_norm = grad_scale.norm()
        if grad_norm > SafeDivideFunction.MAX_GRAD_NORM:
            grad_scale = grad_scale * (SafeDivideFunction.MAX_GRAD_NORM / grad_norm)
        
        return grad_x, grad_scale


def safe_divide(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return SafeDivideFunction.apply(x, scale)


class StraightThroughQuantize(torch.autograd.Function):
    def forward(ctx, input, scale, zero_point, qmin, qmax, symmetric, per_channel, channel_dim, original_scale, original_zp):
        ctx.save_for_backward(input, original_scale, original_zp)
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.symmetric = symmetric
        ctx.per_channel = per_channel
        ctx.channel_dim = channel_dim
        
        # Perform quantization using the broadcasted scale/zero_point
        if symmetric:
            x_scaled = input / scale
            x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
            output = x_quant * scale
        else:
            x_scaled = input / scale + zero_point
            x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
            output = (x_quant - zero_point) * scale
        
        return output
    
    def backward(ctx, grad_output):
        input, original_scale, original_zp = ctx.saved_tensors
        
        # For gradient computation, we need to use the broadcasted versions
        if ctx.per_channel:
            # Recreate the broadcasted scale and zero_point
            shape = [1] * input.ndim
            shape[ctx.channel_dim] = -1
            scale = original_scale.view(shape)
            
            if not ctx.symmetric:
                zero_point = original_zp.view(shape)
            else:
                zero_point = torch.zeros_like(scale)
        else:
            scale = original_scale
            if not ctx.symmetric:
                zero_point = original_zp
            else:
                zero_point = torch.zeros_like(scale)
        
        # Compute pre-quantized values to determine where clamping occurs
        if ctx.symmetric:
            pre_quant = input / scale
        else:
            pre_quant = input / scale + zero_point
        
        mask = (pre_quant > ctx.qmin) & (pre_quant < ctx.qmax) # Create mask for values that are NOT clamped
        grad_input = grad_output * mask.float() # Apply straight-through only where not clamped
        
        # Compute gradients for scale parameter
        if ctx.symmetric:
            quantized_levels = torch.clamp(torch.round(input / scale), ctx.qmin, ctx.qmax)
            scale_grad_per_element = grad_output * (quantized_levels - input / scale)
        else:
            x_scaled = input / scale + zero_point
            quantized_levels = torch.clamp(torch.round(x_scaled), ctx.qmin, ctx.qmax) - zero_point
            scale_grad_per_element = grad_output * (quantized_levels - input / scale)
        
        if ctx.per_channel:
            dims_to_sum = list(range(input.ndim))
            dims_to_sum.remove(ctx.channel_dim)
            grad_scale = torch.mean(scale_grad_per_element, dim=dims_to_sum, keepdim=False)
            
            if grad_scale.shape != original_scale.shape:
                grad_scale = grad_scale.view_as(original_scale)
        else:
            # For per-tensor, sum over all dimensions but keep as tensor with same shape as scale
            grad_scale = torch.mean(scale_grad_per_element).view_as(original_scale)
        
        # Compute gradients for zero_point
        if not ctx.symmetric:
            if ctx.per_channel:
                # For per-channel zero point - using gradient = -s approach
                zp_grad_per_element = -grad_output * scale
                zp_grad_per_element = torch.clamp(zp_grad_per_element, -1.0, 1.0)
                dims_to_sum = list(range(input.ndim))
                dims_to_sum.remove(ctx.channel_dim)
                grad_zero_point = torch.mean(zp_grad_per_element, dim=dims_to_sum, keepdim=False)
                
                if grad_zero_point.shape != original_zp.shape:
                    grad_zero_point = grad_zero_point.view_as(original_zp)
            else:
                # For per-tensor zero point
                zp_grad_per_element = -grad_output * scale
                zp_grad_per_element = torch.clamp(zp_grad_per_element, -1.0, 1.0)
                grad_zero_point = torch.mean(zp_grad_per_element).view_as(original_zp)
        else:
            grad_zero_point = None
        
        return (
            grad_input,    # input
            None,          # scale (broadcasted)
            None,          # zero_point (broadcasted)
            None,          # qmin
            None,          # qmax
            None,          # symmetric
            None,          # per_channel
            None,          # channel_dim
            grad_scale,    # original_scale
            grad_zero_point, # original_zp
        )
