"""
Tests for the quantizer primitives: STE gradients and the LearnableQuantizer.

Run:  python -m pytest ADC/best_clean/tests  (or: python tests/test_quant.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from core.grad_functions import round_ste, floor_ste
from core.adc_layers import LearnableQuantizer


def test_round_ste_value_and_grad():
    x = torch.tensor([0.2, 0.7, -1.4, 2.5], requires_grad=True)
    y = round_ste(x)
    assert torch.allclose(y, torch.round(x))   # forward = round
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))  # STE: gradient = 1


def test_floor_ste_value_and_grad():
    x = torch.tensor([0.2, 0.7, -1.4, 2.5], requires_grad=True)
    y = floor_ste(x)
    assert torch.allclose(y, torch.floor(x))
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_learnable_quantizer_symmetric_roundtrip():
    """Per-tensor symmetric quantizer: error stays within one step (scale)."""
    q = LearnableQuantizer(num_bits=8, symmetric=True, per_channel=False)
    q.set_mode("fixed")
    with torch.no_grad():
        q.scale.data.fill_(0.1)
    x = torch.linspace(-5.0, 5.0, 101)  # well inside +-127*0.1 = +-12.7
    y = q(x)
    assert torch.isfinite(y).all()
    assert (y - x).abs().max().item() <= 0.1 + 1e-6


def test_learnable_quantizer_per_channel_shape():
    """Per-channel quantizer initializes one scale per channel on first forward."""
    C, N = 6, 20
    q = LearnableQuantizer(num_bits=4, symmetric=True, per_channel=True, channel_dim=0)
    x = torch.randn(C, N)
    q(x)
    assert q.scale.shape == (C,)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
