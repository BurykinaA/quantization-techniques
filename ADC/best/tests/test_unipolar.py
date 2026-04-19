"""
Tests for unipolar ADC mode in QATLinearADC / TiledLinearADC.

Unipolar math:
    offset_codes = -na = 2^(ba-1)          e.g. 128 for ba=8
    z_shifted = floor(y_int / δ) + offset   ∈ [0, 2^ba − 1]
    z         = z_shifted + na              = same value as bipolar clamp

The two modes must be numerically identical; additionally z_shifted must be ≥ 0.
"""

import sys
import os

# Allow importing core from ADC/best/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from core.adc_layers import QATLinearADC, TiledLinearADC


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_layer(in_f: int, out_f: int, ba: int = 8, bx: int = 4, bw: int = 4,
                unipolar: bool = False) -> QATLinearADC:
    """Small QATLinearADC with known weights, no bias, fixed quantizer scales."""
    layer = QATLinearADC(
        in_features=in_f, out_features=out_f, bias=False,
        bx=bx, bw=bw, ba=ba, k=16,
        signed_activations=True,
    )
    layer.unipolar_adc = unipolar

    # Initialise weight quantizer with known per-channel scales
    with torch.no_grad():
        torch.nn.init.uniform_(layer.weight, -0.5, 0.5)
        layer.weight_quantizer.scale.data = torch.full((out_f,), 0.1)
        layer.weight_quantizer._scale_initialized = True
        layer.activation_quantizer.scale.data = torch.tensor([0.1])
        layer.activation_quantizer._scale_initialized = True

    layer.eval()
    return layer


def _make_tiled(in_f: int, out_f: int, mvm_limit: int = 16,
                unipolar: bool = False) -> TiledLinearADC:
    """TiledLinearADC wrapping a small random nn.Linear."""
    linear = torch.nn.Linear(in_f, out_f, bias=False)
    tiled = TiledLinearADC(
        in_features=in_f, out_features=out_f, bias=False,
        bx=4, bw=4, ba=8, k=16,
        signed_activations=True,
        mvm_limit=mvm_limit,
        unipolar_adc=unipolar,
    )
    tiled.load_weights(linear)
    tiled.eval()
    return tiled


# ── Test 1: Numerical equivalence ─────────────────────────────────────────────

def test_bipolar_unipolar_equivalence():
    """Unipolar and bipolar modes must produce identical outputs."""
    torch.manual_seed(0)
    in_f, out_f = 16, 8
    x = torch.randn(4, in_f)

    layer_bi = _make_layer(in_f, out_f, unipolar=False)
    layer_uni = _make_layer(in_f, out_f, unipolar=True)

    # Copy identical weights so only the ADC path differs
    with torch.no_grad():
        layer_uni.weight.copy_(layer_bi.weight)
        layer_uni.weight_quantizer.scale.data.copy_(layer_bi.weight_quantizer.scale.data)
        layer_uni.activation_quantizer.scale.data.copy_(layer_bi.activation_quantizer.scale.data)

    with torch.no_grad():
        out_bi  = layer_bi(x)
        out_uni = layer_uni(x)

    assert torch.allclose(out_bi, out_uni, atol=1e-5), (
        f"Bipolar and unipolar outputs differ.\n"
        f"Max abs diff: {(out_bi - out_uni).abs().max().item():.6f}"
    )


# ── Test 2: z_shifted ≥ 0 ─────────────────────────────────────────────────────

def test_unipolar_z_shifted_nonnegative():
    """In unipolar mode, the intermediate ADC code z_shifted must be ≥ 0."""
    torch.manual_seed(1)
    in_f, out_f = 16, 8
    x = torch.randn(4, in_f)
    layer = _make_layer(in_f, out_f, unipolar=True)

    captured = {}

    original_forward = layer.forward.__func__

    def patched_forward(self, x_in):
        from core.grad_functions import floor_ste
        import torch.nn.functional as F

        x_in = x_in.float()
        act_q = self.activation_quantizer
        qmin_x, qmax_x = act_q.qmin, act_q.qmax
        act_levels = float(qmax_x)
        s_x = x_in.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
        code_x = torch.round(x_in / s_x).clamp(qmin_x, qmax_x)

        w_q = self.weight_quantizer
        s_w_b = w_q.scale.view(-1, 1)
        code_w = torch.round(self.weight / s_w_b).clamp(w_q.qmin, w_q.qmax)

        y_int = F.linear(code_x, code_w, bias=None)

        if self.unipolar_adc:
            z_shifted = floor_ste(y_int / self.delta) + self._adc_offset_codes
            captured["z_shifted"] = z_shifted.detach().clone()

        return original_forward(self, x_in)

    import types
    layer.forward = types.MethodType(patched_forward, layer)

    with torch.no_grad():
        layer(x)

    assert "z_shifted" in captured, "z_shifted was not captured"
    assert (captured["z_shifted"] >= 0).all(), (
        f"Some z_shifted values are negative: min={captured['z_shifted'].min().item()}"
    )


# ── Test 3: Saturation (large input) ──────────────────────────────────────────

def test_saturation_equivalence():
    """Large inputs that saturate the ADC must clamp identically in both modes."""
    torch.manual_seed(2)
    in_f, out_f = 16, 8

    layer_bi  = _make_layer(in_f, out_f, unipolar=False)
    layer_uni = _make_layer(in_f, out_f, unipolar=True)

    with torch.no_grad():
        layer_uni.weight.copy_(layer_bi.weight)
        layer_uni.weight_quantizer.scale.data.copy_(layer_bi.weight_quantizer.scale.data)
        layer_uni.activation_quantizer.scale.data.copy_(layer_bi.activation_quantizer.scale.data)

    # Very large input — guaranteed to saturate
    x = torch.full((4, in_f), 1e6)

    with torch.no_grad():
        out_bi  = layer_bi(x)
        out_uni = layer_uni(x)

    assert torch.allclose(out_bi, out_uni, atol=1e-5), (
        f"Saturation outputs differ.\nMax abs diff: {(out_bi - out_uni).abs().max().item():.6f}"
    )


# ── Test 4: TiledLinearADC propagates unipolar_adc to all tiles ───────────────

def test_tiled_propagates_unipolar():
    """TiledLinearADC must propagate unipolar_adc to every tile."""
    in_f, out_f = 32, 8
    tiled = _make_tiled(in_f, out_f, mvm_limit=16, unipolar=True)

    assert tiled.unipolar_adc, "TiledLinearADC.unipolar_adc should be True"
    for i, tile in enumerate(tiled.tiles):
        assert tile.unipolar_adc, f"Tile {i} unipolar_adc should be True"


# ── Test 5: TiledLinearADC numerical equivalence ─────────────────────────────

def test_tiled_bipolar_unipolar_equivalence():
    """TiledLinearADC unipolar and bipolar must produce identical outputs."""
    torch.manual_seed(3)
    in_f, out_f = 32, 8
    x = torch.randn(2, in_f)

    linear = torch.nn.Linear(in_f, out_f, bias=False)

    tiled_bi = _make_tiled(in_f, out_f, mvm_limit=16, unipolar=False)
    tiled_uni = _make_tiled(in_f, out_f, mvm_limit=16, unipolar=True)

    # Copy same weights
    tiled_bi.load_weights(linear)
    tiled_uni.load_weights(linear)

    with torch.no_grad():
        out_bi  = tiled_bi(x)
        out_uni = tiled_uni(x)

    assert torch.allclose(out_bi, out_uni, atol=1e-5), (
        f"TiledLinearADC bipolar/unipolar differ.\n"
        f"Max abs diff: {(out_bi - out_uni).abs().max().item():.6f}"
    )
