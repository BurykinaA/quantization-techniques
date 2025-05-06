import torch

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: just pass gradient through
        return grad_output

def ste_floor(x):
    return FloorSTE.apply(x)


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: just pass gradient through
        return grad_output

def ste_round(x):
    return RoundSTE.apply(x)
