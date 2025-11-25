import torch
import numpy as np


def round_ste(x):
    return x + (torch.round(x) - x).detach()


def floor_ste(x):
    return x + (torch.floor(x) - x).detach()


def grad_scale(x, scale:float):
    y_grad = x * scale
    return y_grad + (x - y_grad).detach()


class AugScheduler:
    def __init__(self, iter_max, start, end):
        self.iter_max = iter_max
        self.start, self.end = start, end

    def __call__(self, current):
        return self.end + 0.5 * (self.start - self.end) * (1 + np.cos(current  / self.iter_max * np.pi))
