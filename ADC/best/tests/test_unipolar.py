"""
Tests for unipolar ADC mode in QATLinearADC / TiledLinearADC.
Run from ADC/best/:  python tests/test_unipolar.py

Unipolar math:
    offset_codes = -na = 2^(ba-1)          e.g. 128 for ba=8
    z_shifted = floor(y_int / delta) + offset   in [0, 2^ba - 1]
    z         = z_shifted + na              = same value as bipolar clamp
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from core.adc_layers import QATLinearADC, TiledLinearADC
from core.grad_functions import floor_ste


def _make_layer(in_f, out_f, ba=8, bx=4, bw=4, unipolar=False):
    layer = QATLinearADC(
        in_features=in_f, out_features=out_f, bias=False,
        bx=bx, bw=bw, ba=ba, k=16,
        signed_activations=True,
    )
    layer.unipolar_adc = unipolar
    with torch.no_grad():
        torch.nn.init.uniform_(layer.weight, -0.5, 0.5)
        layer.weight_quantizer.scale.data = torch.full((out_f,), 0.1)
        layer.weight_quantizer._scale_initialized = True
        layer.activation_quantizer.scale.data = torch.tensor([0.1])
        layer.activation_quantizer._scale_initialized = True
    layer.eval()
    return layer


def _copy_weights(src, dst):
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        dst.weight_quantizer.scale.data.copy_(src.weight_quantizer.scale.data)
        dst.activation_quantizer.scale.data.copy_(src.activation_quantizer.scale.data)


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        return False


# ── Test 1: Numerical equivalence ─────────────────────────────────────────────

def test_bipolar_unipolar_equivalence():
    torch.manual_seed(0)
    in_f, out_f = 16, 8
    x = torch.randn(4, in_f)
    layer_bi  = _make_layer(in_f, out_f, unipolar=False)
    layer_uni = _make_layer(in_f, out_f, unipolar=True)
    _copy_weights(layer_bi, layer_uni)
    with torch.no_grad():
        out_bi  = layer_bi(x)
        out_uni = layer_uni(x)
    diff = (out_bi - out_uni).abs().max().item()
    assert torch.allclose(out_bi, out_uni, atol=1e-5), \
        f"max abs diff = {diff:.2e}"


# ── Test 2: z_shifted >= 0 ────────────────────────────────────────────────────

def test_z_shifted_nonnegative():
    torch.manual_seed(1)
    in_f, out_f = 16, 8
    x = torch.randn(4, in_f)
    layer = _make_layer(in_f, out_f, unipolar=True)

    # Compute z_shifted directly using the layer's own delta and offset
    with torch.no_grad():
        x_f = x.float()
        act_levels = float(layer.activation_quantizer.qmax)
        s_x = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
        code_x = torch.round(x_f / s_x).clamp(
            layer.activation_quantizer.qmin, layer.activation_quantizer.qmax)
        s_w = layer.weight_quantizer.scale.view(-1, 1)
        code_w = torch.round(layer.weight / s_w).clamp(
            layer.weight_quantizer.qmin, layer.weight_quantizer.qmax)
        import torch.nn.functional as F
        y_int = F.linear(code_x, code_w, None)
        z_shifted = floor_ste(y_int / layer.delta) + layer._adc_offset_codes

    min_val = z_shifted.min().item()
    assert min_val >= 0, f"z_shifted has negative values: min = {min_val}"


# ── Test 3: Saturation equivalence ───────────────────────────────────────────

def test_saturation_equivalence():
    torch.manual_seed(2)
    in_f, out_f = 16, 8
    layer_bi  = _make_layer(in_f, out_f, unipolar=False)
    layer_uni = _make_layer(in_f, out_f, unipolar=True)
    _copy_weights(layer_bi, layer_uni)
    x = torch.full((4, in_f), 1e6)  # saturates the ADC
    with torch.no_grad():
        out_bi  = layer_bi(x)
        out_uni = layer_uni(x)
    diff = (out_bi - out_uni).abs().max().item()
    assert torch.allclose(out_bi, out_uni, atol=1e-5), \
        f"saturation max abs diff = {diff:.2e}"


# ── Test 4: TiledLinearADC propagates unipolar_adc to all tiles ───────────────

def test_tiled_propagates_unipolar():
    in_f, out_f = 32, 8
    linear = torch.nn.Linear(in_f, out_f, bias=False)
    tiled = TiledLinearADC(
        in_features=in_f, out_features=out_f, bias=False,
        bx=4, bw=4, ba=8, k=16,
        signed_activations=True, mvm_limit=16,
        unipolar_adc=True,
    )
    tiled.load_weights(linear)
    tiled.eval()
    assert tiled.unipolar_adc, "TiledLinearADC.unipolar_adc should be True"
    for i, tile in enumerate(tiled.tiles):
        assert tile.unipolar_adc, f"tile {i} unipolar_adc should be True"


# ── Test 5: TiledLinearADC bipolar/unipolar numerical equivalence ─────────────

def test_tiled_equivalence():
    torch.manual_seed(3)
    in_f, out_f = 32, 8
    linear = torch.nn.Linear(in_f, out_f, bias=False)
    x = torch.randn(2, in_f)

    def make_tiled(unipolar):
        t = TiledLinearADC(
            in_features=in_f, out_features=out_f, bias=False,
            bx=4, bw=4, ba=8, k=16,
            signed_activations=True, mvm_limit=16,
            unipolar_adc=unipolar,
        )
        t.load_weights(linear)
        t.eval()
        return t

    tiled_bi  = make_tiled(False)
    tiled_uni = make_tiled(True)
    with torch.no_grad():
        out_bi  = tiled_bi(x)
        out_uni = tiled_uni(x)
    diff = (out_bi - out_uni).abs().max().item()
    assert torch.allclose(out_bi, out_uni, atol=1e-5), \
        f"TiledLinearADC max abs diff = {diff:.2e}"


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("bipolar == unipolar (QATLinearADC)",     test_bipolar_unipolar_equivalence),
        ("z_shifted >= 0",                         test_z_shifted_nonnegative),
        ("saturation equivalence",                 test_saturation_equivalence),
        ("TiledLinearADC propagates unipolar_adc", test_tiled_propagates_unipolar),
        ("TiledLinearADC bipolar == unipolar",      test_tiled_equivalence),
    ]
    print(f"\nRunning {len(tests)} unipolar ADC tests ...\n")
    passed = sum(run_test(name, fn) for name, fn in tests)
    print(f"\n{passed}/{len(tests)} passed")
    if passed < len(tests):
        sys.exit(1)
