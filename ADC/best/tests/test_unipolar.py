"""
Tests for unipolar ADC mode — 4-quadrant decomposition.
Run from ADC/best/:  python tests/test_unipolar.py

x = x+ - x-,  w = w+ - w-  (all components >= 0)
y = x+·w+ + x-·w- - x+·w- - x-·w+   (4 positive MVMs, no DC offset)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from core.adc_layers import QATLinearADC, TiledLinearADC


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


# ── Test 1: All 4 branch inputs and MVMs are non-negative ────────────────────

def test_all_branches_nonnegative():
    """x+, x-, w+, w- and all 4 y_branch values must be >= 0."""
    torch.manual_seed(1)
    in_f, out_f = 16, 8
    x = torch.randn(4, in_f)
    layer = _make_layer(in_f, out_f, unipolar=True)

    with torch.no_grad():
        x_f = x.float()
        act_levels = float(layer.activation_quantizer.qmax)
        s_x = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
        code_x = torch.round(x_f / s_x).clamp(
            layer.activation_quantizer.qmin, layer.activation_quantizer.qmax)
        s_w = layer.weight_quantizer.scale.view(-1, 1)
        code_w = torch.round(layer.weight / s_w).clamp(
            layer.weight_quantizer.qmin, layer.weight_quantizer.qmax)

        x_pos = code_x.clamp(min=0)
        x_neg = (-code_x).clamp(min=0)
        w_pos = code_w.clamp(min=0)
        w_neg = (-code_w).clamp(min=0)

        y_pp = F.linear(x_pos, w_pos, None)
        y_pn = F.linear(x_pos, w_neg, None)
        y_np = F.linear(x_neg, w_pos, None)
        y_nn = F.linear(x_neg, w_neg, None)

    for name, t in [("x_pos", x_pos), ("x_neg", x_neg),
                    ("w_pos", w_pos), ("w_neg", w_neg),
                    ("y_pp",  y_pp),  ("y_pn",  y_pn),
                    ("y_np",  y_np),  ("y_nn",  y_nn)]:
        assert t.min().item() >= 0, f"{name} has negative values: min={t.min().item()}"


# ── Test 2: 4-quadrant algebra is exact without ADC ──────────────────────────

def test_4quadrant_algebra():
    """x+·w+ + x-·w- - x+·w- - x-·w+ must equal code_x · code_w^T exactly."""
    torch.manual_seed(2)
    in_f, out_f = 16, 8
    x = torch.randn(4, in_f)
    layer = _make_layer(in_f, out_f, unipolar=True)

    with torch.no_grad():
        x_f = x.float()
        act_levels = float(layer.activation_quantizer.qmax)
        s_x = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
        code_x = torch.round(x_f / s_x).clamp(
            layer.activation_quantizer.qmin, layer.activation_quantizer.qmax)
        s_w = layer.weight_quantizer.scale.view(-1, 1)
        code_w = torch.round(layer.weight / s_w).clamp(
            layer.weight_quantizer.qmin, layer.weight_quantizer.qmax)

        x_pos = code_x.clamp(min=0);  x_neg = (-code_x).clamp(min=0)
        w_pos = code_w.clamp(min=0);  w_neg = (-code_w).clamp(min=0)

        y_4q  = (F.linear(x_pos, w_pos, None) + F.linear(x_neg, w_neg, None)
                 - F.linear(x_pos, w_neg, None) - F.linear(x_neg, w_pos, None))
        y_ref = F.linear(code_x, code_w, None)

    diff = (y_4q - y_ref).abs().max().item()
    assert diff < 1e-4, f"4-quadrant algebra wrong: max diff = {diff:.2e}"


# ── Test 3: bypass_adc=True gives identical results for both modes ────────────

def test_bypass_equivalence():
    """With bypass_adc=True both modes give identical output."""
    torch.manual_seed(3)
    in_f, out_f = 16, 8
    x = torch.randn(4, in_f)

    layer_bi  = _make_layer(in_f, out_f, unipolar=False)
    layer_uni = _make_layer(in_f, out_f, unipolar=True)
    with torch.no_grad():
        layer_uni.weight.copy_(layer_bi.weight)
        layer_uni.weight_quantizer.scale.data.copy_(layer_bi.weight_quantizer.scale.data)
        layer_uni.activation_quantizer.scale.data.copy_(layer_bi.activation_quantizer.scale.data)

    layer_bi.bypass_adc  = True
    layer_uni.bypass_adc = True

    with torch.no_grad():
        out_bi  = layer_bi(x)
        out_uni = layer_uni(x)

    diff = (out_bi - out_uni).abs().max().item()
    assert diff < 1e-5, f"bypass mode outputs differ: max diff = {diff:.2e}"


# ── Test 4: TiledLinearADC propagates unipolar_adc to all tiles ───────────────

def test_tiled_propagates_unipolar():
    """TiledLinearADC must set unipolar_adc on every tile."""
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


# ── Test 5: Output is finite ──────────────────────────────────────────────────

def test_output_finite():
    """Unipolar forward pass must produce finite (non-NaN, non-Inf) outputs."""
    torch.manual_seed(4)
    in_f, out_f = 16, 8
    x = torch.randn(4, in_f)
    layer = _make_layer(in_f, out_f, unipolar=True)

    with torch.no_grad():
        out = layer(x)

    assert torch.isfinite(out).all(), f"Output has non-finite values"


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("all branch inputs/MVMs >= 0",            test_all_branches_nonnegative),
        ("4-quadrant algebra exact (no ADC noise)", test_4quadrant_algebra),
        ("bypass_adc=True: both modes identical",   test_bypass_equivalence),
        ("TiledLinearADC propagates unipolar_adc",  test_tiled_propagates_unipolar),
        ("output is finite",                        test_output_finite),
    ]
    print(f"\nRunning {len(tests)} unipolar ADC tests ...\n")
    passed = sum(run_test(name, fn) for name, fn in tests)
    print(f"\n{passed}/{len(tests)} passed")
    if passed < len(tests):
        sys.exit(1)
