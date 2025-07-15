import torch

class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient straight through
        return grad_output

class STEFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient straight through
        return grad_output

def ste_round(x):
    return STERound.apply(x)

def ste_floor(x):
    return STEFloor.apply(x)
