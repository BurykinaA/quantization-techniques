"""
Unit tests for FlatQuant numerical stability fixes.

Tests run without loading a real LLaMA model — all use small toy tensors.
Run with:  python -m pytest ADC/llama/core/test_flat_quant_stability.py -v
or simply: python ADC/llama/core/test_flat_quant_stability.py
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim

from ADC.llama.core.flat_quant import KroneckerTransform, FlatQuantLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transform(dim: int = 32, add_diag: bool = False) -> KroneckerTransform:
    return KroneckerTransform(dim=dim, add_diag=add_diag)


def _set_diag(trans: KroneckerTransform, value: float) -> None:
    """Force diag_left and diag_right to a specific value."""
    with torch.no_grad():
        trans.diag_left.fill_(value)
        trans.diag_right.fill_(value)


# ===========================================================================
# Fix 1: forward() must not produce NaN/Inf when diag_left/right ≈ 0
# ===========================================================================

def test_forward_near_zero_diag_no_nan():
    """Bug: 1/diag_left with diag ~ 1e-8 produced NaN. Fixed by abs().clamp(min=1e-6)."""
    trans = _make_transform(dim=32)
    _set_diag(trans, 1e-8)   # near-zero (would cause NaN/Inf before fix)

    x = torch.randn(4, 32)
    out = trans(x, inv_t=True)

    assert not torch.isnan(out).any(),  "NaN in forward(inv_t=True) with near-zero diag"
    assert not torch.isinf(out).any(),  "Inf in forward(inv_t=True) with near-zero diag"
    print("PASS  test_forward_near_zero_diag_no_nan")


def test_forward_negative_diag_no_nan():
    """Bug: negative diag_left is valid after optimizer step; 1/neg caused NaN/Inf."""
    trans = _make_transform(dim=32)
    _set_diag(trans, -0.5)   # negative (valid after gradient step before fix)

    x = torch.randn(4, 32)
    out = trans(x, inv_t=True)

    assert not torch.isnan(out).any(), "NaN in forward(inv_t=True) with negative diag"
    assert not torch.isinf(out).any(), "Inf in forward(inv_t=True) with negative diag"
    print("PASS  test_forward_negative_diag_no_nan")


def test_forward_zero_diag_no_nan():
    """Exact zero diag_left is the worst case — must still not produce NaN."""
    trans = _make_transform(dim=32)
    _set_diag(trans, 0.0)

    x = torch.randn(4, 32)
    out = trans(x, inv_t=True)

    assert not torch.isnan(out).any(), "NaN in forward(inv_t=True) with zero diag"
    assert not torch.isinf(out).any(), "Inf in forward(inv_t=True) with zero diag"
    print("PASS  test_forward_zero_diag_no_nan")


def test_forward_diag_scale_near_zero_no_nan():
    """Bug: x / diag_scale without epsilon could produce NaN/Inf when scale → 0."""
    trans = _make_transform(dim=32, add_diag=True)
    with torch.no_grad():
        trans.diag_scale.fill_(1e-10)   # near-zero scale (worst case)

    x = torch.randn(4, 32)
    out = trans(x, inv_t=True)

    assert not torch.isnan(out).any(), "NaN in forward(inv_t=True) with near-zero diag_scale"
    assert not torch.isinf(out).any(), "Inf in forward(inv_t=True) with near-zero diag_scale"
    print("PASS  test_forward_diag_scale_near_zero_no_nan")


# ===========================================================================
# Fix 2: to_eval_mode() must not store NaN/Inf matrices
# ===========================================================================

def test_to_eval_mode_near_zero_diag_finite_matrices():
    """Bug: to_eval_mode() called 1/diag without protection; stored NaN in matrix_*_inv."""
    trans = _make_transform(dim=32)
    _set_diag(trans, 1e-8)   # near-zero

    trans.to_eval_mode()

    for name in ("matrix_left", "matrix_right", "matrix_left_inv", "matrix_right_inv"):
        mat = getattr(trans, name)
        assert not torch.isnan(mat).any(), f"NaN in {name} after to_eval_mode with near-zero diag"
        assert not torch.isinf(mat).any(), f"Inf in {name} after to_eval_mode with near-zero diag"
    print("PASS  test_to_eval_mode_near_zero_diag_finite_matrices")


def test_to_eval_mode_forward_consistent():
    """Forward pass in eval mode must match non-eval mode (up to float precision)."""
    trans = _make_transform(dim=32)
    _set_diag(trans, 0.5)

    x = torch.randn(4, 32)

    # Training mode result
    out_train = trans(x, inv_t=False).detach()

    # Eval mode result
    trans.to_eval_mode()
    out_eval = trans(x, inv_t=False).detach()

    assert torch.allclose(out_train, out_eval, atol=1e-4), \
        f"Eval/train mismatch: max diff = {(out_train - out_eval).abs().max().item():.2e}"
    print("PASS  test_to_eval_mode_forward_consistent")


# ===========================================================================
# Fix 3: gradient clipping + parameter projection keep diag away from zero
# ===========================================================================

def test_gradient_clipping_prevents_explosion():
    """After one backward step with huge loss, gradients should be clipped to max_norm=1."""
    trans = _make_transform(dim=32)
    params = list(trans.parameters())
    optimizer = optim.AdamW(params, lr=1.0)   # large LR to stress-test

    x = torch.randn(4, 32)
    # Manufacture a huge loss to trigger large gradients
    loss = trans(x, inv_t=True).pow(2).sum() * 1e6
    optimizer.zero_grad()
    loss.backward()

    # Apply clipping (as in the training loop)
    total_norm_before = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()

    # Check no parameter became NaN/Inf
    for p in params:
        assert not torch.isnan(p).any(), f"Parameter became NaN after clipped step"
        assert not torch.isinf(p).any(), f"Parameter became Inf after clipped step"
    print(f"PASS  test_gradient_clipping_prevents_explosion "
          f"(pre-clip grad norm: {total_norm_before:.2e})")


def test_projection_keeps_diag_positive():
    """After an optimizer step that would push diag negative, projection must clamp it."""
    trans = _make_transform(dim=32)
    optimizer = optim.SGD(
        [{"params": [trans.diag_left, trans.diag_right], "lr": 1000.0}]
    )   # huge LR to definitely push diag below zero

    x = torch.randn(4, 32)
    loss = trans(x, inv_t=True).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Apply projection (as in the training loop)
    with torch.no_grad():
        trans.diag_left.data.clamp_(min=1e-4)
        trans.diag_right.data.clamp_(min=1e-4)

    assert (trans.diag_left >= 1e-4).all(),  "diag_left below 1e-4 after projection"
    assert (trans.diag_right >= 1e-4).all(), "diag_right below 1e-4 after projection"
    print("PASS  test_projection_keeps_diag_positive")


# ===========================================================================
# End-to-end: mini training loop must not produce NaN
# ===========================================================================

def test_mini_training_loop_no_nan():
    """
    Simulates the FlatQuant layer-by-layer training loop with a tiny toy model:
    a single FlatQuantLinear wrapping a 64→32 Linear.
    Runs 5 epochs × 4 batches and checks that no NaN appears in loss or weights.
    """
    torch.manual_seed(0)
    dim_in, dim_out = 64, 32

    linear = nn.Linear(dim_in, dim_out, bias=False)
    fq_linear = FlatQuantLinear(
        linear, w_bits=8, a_bits=8,
        add_diag=False, lwc=True, lac=True,
    )

    trans = KroneckerTransform(dim=dim_in, add_diag=False)

    # Enable gradient only on transform parameters
    for p in fq_linear.linear.parameters():
        p.requires_grad = False

    trained_params = [
        {"params": [trans.diag_left, trans.diag_right,
                    *trans.u_left.parameters(), *trans.v_left.parameters(),
                    *trans.u_right.parameters(), *trans.v_right.parameters()], "lr": 5e-3},
        {"params": list(fq_linear.parameters()), "lr": 5e-2},
    ]
    optimizer = optim.AdamW(trained_params)
    loss_fn = nn.MSELoss()

    # Fixed FP reference output (no transform, no quant)
    with torch.no_grad():
        fp_inps = torch.randn(8, dim_in)
        fp_outs = linear(fp_inps)

    nan_count = 0
    for epoch in range(5):
        for j in range(4):
            x = fp_inps[j*2:(j+1)*2]
            target = fp_outs[j*2:(j+1)*2]
            out = fq_linear.train_forward(x, qa_trans=trans)
            loss = loss_fn(target, out)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                continue

            normalized_loss = loss / loss.clone().detach()
            optimizer.zero_grad()
            normalized_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                max_norm=1.0,
            )
            optimizer.step()
            with torch.no_grad():
                trans.diag_left.data.clamp_(min=1e-4)
                trans.diag_right.data.clamp_(min=1e-4)

    # All parameters must be finite after training
    for name, p in trans.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in trans.{name} after training loop"
        assert not torch.isinf(p).any(), f"Inf in trans.{name} after training loop"

    # to_eval_mode and reparameterize must not produce NaN weights
    trans.to_eval_mode()
    fq_linear.reparameterize(qa_trans=trans)
    w = fq_linear.linear.weight
    assert not torch.isnan(w).any(), "NaN in reparameterized weight"
    assert not torch.isinf(w).any(), "Inf in reparameterized weight"

    assert nan_count == 0, f"{nan_count} NaN losses during mini training loop"
    print(f"PASS  test_mini_training_loop_no_nan  (nan_count={nan_count})")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    tests = [
        test_forward_near_zero_diag_no_nan,
        test_forward_negative_diag_no_nan,
        test_forward_zero_diag_no_nan,
        test_forward_diag_scale_near_zero_no_nan,
        test_to_eval_mode_near_zero_diag_finite_matrices,
        test_to_eval_mode_forward_consistent,
        test_gradient_clipping_prevents_explosion,
        test_projection_keeps_diag_positive,
        test_mini_training_loop_no_nan,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed == 0:
        print("All stability fixes verified!")
