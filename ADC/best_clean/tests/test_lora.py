"""
Tests for the post-ADC residual LoRA wrapper.

Run:  python -m pytest ADC/best_clean/tests  (or: python tests/test_lora.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from core.adc_layers import TiledLinearADC
from core.adc_lora import ResidualLoRATiledLinearADC


def _make_tiled(in_f=32, out_f=8):
    torch.manual_seed(0)
    linear = torch.nn.Linear(in_f, out_f, bias=False)
    tiled = TiledLinearADC(in_features=in_f, out_features=out_f, bias=False,
                           bx=4, bw=4, ba=8, k=16, signed_activations=True,
                           mvm_limit=16, use_kurtosis_loss=False)
    tiled.load_weights(linear)
    tiled.eval()
    return tiled


def test_zero_init_matches_base():
    """lora_B is zero-initialized → wrapped output equals the frozen base output."""
    tiled = _make_tiled()
    lora = ResidualLoRATiledLinearADC(tiled, r=4, alpha=8.0)
    lora.eval()
    x = torch.randn(4, 32)
    base = tiled(x)
    wrapped = lora(x)
    assert torch.allclose(base, wrapped, atol=1e-6), (base - wrapped).abs().max().item()


def test_scaling():
    lora = ResidualLoRATiledLinearADC(_make_tiled(), r=4, alpha=8.0)
    assert abs(lora.scaling - 2.0) < 1e-9


def test_trainable_param_count():
    in_f, out_f, r = 32, 8, 4
    lora = ResidualLoRATiledLinearADC(_make_tiled(in_f, out_f), r=r, alpha=8.0)
    assert lora.get_num_trainable_params() == r * in_f + out_f * r


def test_base_is_frozen():
    """The wrapped TiledLinearADC parameters must not require gradients."""
    lora = ResidualLoRATiledLinearADC(_make_tiled(), r=4, alpha=8.0)
    assert all(not p.requires_grad for p in lora.tiled_layer.parameters())
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad


def test_nonzero_lora_changes_output():
    """After perturbing lora_B, the output must differ from the base."""
    tiled = _make_tiled()
    lora = ResidualLoRATiledLinearADC(tiled, r=4, alpha=8.0)
    lora.eval()
    with torch.no_grad():
        lora.lora_B.weight.normal_(0.0, 0.1)
    x = torch.randn(4, 32)
    assert not torch.allclose(tiled(x), lora(x), atol=1e-4)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
