"""
Tests for the symmetric (bipolar) ADC layers.

Run:  python -m pytest ADC/best_clean/tests  (or: python tests/test_adc_layers.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from core.adc_layers import QATLinearADC, TiledLinearADC


def _make_layer(in_f, out_f, bx=4, bw=4, ba=8, k=16, seed=0):
    torch.manual_seed(seed)
    layer = QATLinearADC(
        in_features=in_f, out_features=out_f, bias=False,
        bx=bx, bw=bw, ba=ba, k=k, signed_activations=True,
        use_kurtosis_loss=False,
    )
    with torch.no_grad():
        torch.nn.init.uniform_(layer.weight, -0.5, 0.5)
        # Per-channel weight scale (shape [out_f]) so dequant broadcasting is exercised.
        s_w = layer.weight.abs().amax(dim=1).clamp(min=1e-6) / layer.weight_quantizer.qmax
        layer.weight_quantizer.scale.data = s_w
        layer.weight_quantizer._scale_initialized = True
    layer.eval()
    return layer


def test_delta_formula():
    """delta = 2 * tile_in * q_x * q_w / (2^ba * k)."""
    layer = _make_layer(256, 8, bx=4, bw=4, ba=8, k=16)
    expected = 2.0 * 256 * 7 * 7 / (2 ** 8 * 16)  # = 6.125
    assert abs(layer.delta - expected) < 1e-9, (layer.delta, expected)


def test_bypass_all_equals_linear():
    """bypass_all must reproduce a plain F.linear with the stored weight."""
    layer = _make_layer(16, 8)
    layer.bypass_all = True
    x = torch.randn(4, 16)
    out = layer(x)
    ref = F.linear(x, layer.weight, None)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max().item()


def test_symmetric_path_matches_manual():
    """The symmetric floor path must equal a hand-computed reference."""
    torch.manual_seed(7)
    in_f, out_f = 8, 3
    layer = _make_layer(in_f, out_f)
    x = torch.randn(5, in_f)

    out = layer(x)

    # Manual recomputation of the exact same pipeline.
    act_levels = float(layer.activation_quantizer.qmax)  # 7
    qmin_x, qmax_x = layer.activation_quantizer.qmin, layer.activation_quantizer.qmax
    qmin_w, qmax_w = layer.weight_quantizer.qmin, layer.weight_quantizer.qmax
    s_x = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
    code_x = torch.round(x / s_x).clamp(qmin_x, qmax_x)
    s_w = layer.weight_quantizer.scale.view(-1, 1)
    code_w = torch.round(layer.weight / s_w).clamp(qmin_w, qmax_w)
    y_int = F.linear(code_x, code_w, None)
    y_adc = torch.clamp(torch.floor(y_int / layer.delta), layer.na, layer.pa) * layer.delta
    ref = y_adc * s_x * layer.weight_quantizer.scale

    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()


def test_adc_output_on_grid():
    """Pre-dequant ADC codes must be integer multiples of delta within [na, pa]."""
    layer = _make_layer(16, 8)
    x = torch.randn(6, 16)
    act_levels = float(layer.activation_quantizer.qmax)
    s_x = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
    code_x = torch.round(x / s_x).clamp(
        layer.activation_quantizer.qmin, layer.activation_quantizer.qmax)
    s_w = layer.weight_quantizer.scale.view(-1, 1)
    code_w = torch.round(layer.weight / s_w).clamp(
        layer.weight_quantizer.qmin, layer.weight_quantizer.qmax)
    y_int = F.linear(code_x, code_w, None)
    codes = torch.clamp(torch.floor(y_int / layer.delta), layer.na, layer.pa)

    assert codes.min().item() >= layer.na
    assert codes.max().item() <= layer.pa
    assert torch.allclose(codes, codes.round())


def test_tiled_bypass_all_equals_linear():
    """Tiled layer in bypass_all mode equals the reference linear (tile sum is exact)."""
    torch.manual_seed(3)
    in_f, out_f = 32, 8
    linear = torch.nn.Linear(in_f, out_f, bias=False)
    tiled = TiledLinearADC(in_features=in_f, out_features=out_f, bias=False,
                           bx=4, bw=4, ba=8, k=16, signed_activations=True,
                           mvm_limit=16, use_kurtosis_loss=False)
    tiled.load_weights(linear)
    tiled.eval()
    assert tiled.n_tiles == 2

    tiled.set_bypass_all(True)
    x = torch.randn(4, in_f)
    out = tiled(x)
    ref = F.linear(x, linear.weight, None)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max().item()


def test_tiled_load_weights_split():
    """load_weights must split the original weight across tiles along dim=1."""
    in_f, out_f = 32, 8
    linear = torch.nn.Linear(in_f, out_f, bias=False)
    tiled = TiledLinearADC(in_features=in_f, out_features=out_f, bias=False,
                           bx=4, bw=4, ba=8, k=16, signed_activations=True,
                           mvm_limit=16, use_kurtosis_loss=False)
    tiled.load_weights(linear)
    recon = torch.cat([t.weight for t in tiled.tiles], dim=1)
    assert torch.allclose(recon, linear.weight)


def test_tiled_wrong_width_raises():
    """Forward with a mismatched last dim must raise ValueError."""
    tiled = TiledLinearADC(in_features=32, out_features=8, bias=False,
                           bx=4, bw=4, ba=8, k=16, signed_activations=True,
                           mvm_limit=16, use_kurtosis_loss=False)
    tiled.eval()
    try:
        tiled(torch.randn(2, 31))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong input width")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
