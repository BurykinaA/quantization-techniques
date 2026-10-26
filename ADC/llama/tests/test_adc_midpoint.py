"""Regression tests for midpoint ADC reconstruction."""

import sys

import pytest
import torch
import torch.nn as nn

from ADC.llama.core.adc_layers import TiledLinearADC
from ADC.llama.runs import llama_smooth_quant_adc_ptq as runner


def _tiled_copy(linear: nn.Linear, k: int) -> TiledLinearADC:
    tiled = TiledLinearADC(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=False,
        bx=4,
        bw=4,
        ba=8,
        k=k,
        signed_activations=True,
        mvm_limit=4,
    )
    tiled.load_weights(linear)
    for tile in tiled.tiles:
        scales = tile.weight.detach().abs().amax(dim=1).clamp(min=1e-8) / 7.0
        tile.weight_quantizer.scale.data = scales.clone()
        tile.weight_quantizer._scale_initialized = True
    return tiled


# k=1 leaves most codes inside the ADC range, k=16 saturates most of them.
@pytest.mark.parametrize("k", [1, 16])
def test_midpoint_shifts_only_unsaturated_codes_by_half_a_step(k: int) -> None:
    torch.manual_seed(5)
    tiled = _tiled_copy(nn.Linear(8, 6, bias=False), k)
    inputs = torch.randn(5, 8)

    with torch.no_grad():
        floor_output = tiled(inputs)
        tiled.set_adc_midpoint(True)
        midpoint_output = tiled(inputs)

    expected = torch.zeros_like(floor_output)
    for i, tile in enumerate(tiled.tiles):
        xi = inputs[:, i * tiled.in_features_tile:(i + 1) * tiled.in_features_tile]
        s_x = xi.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
        s_w = tile.weight_quantizer.scale.detach()
        code_x = torch.round(xi / s_x).clamp(-8, 7)
        code_w = torch.round(tile.weight.detach() / s_w.view(-1, 1)).clamp(-8, 7)
        codes = torch.floor((code_x @ code_w.T) / tile.delta).clamp(tile.na, tile.pa)
        inside = ((codes > tile.na) & (codes < tile.pa)).float()
        expected += 0.5 * tile.delta * inside * s_x * s_w

    torch.testing.assert_close(midpoint_output - floor_output, expected, rtol=1e-4, atol=1e-5)


def test_midpoint_leaves_the_bypassed_path_unchanged() -> None:
    torch.manual_seed(5)
    tiled = _tiled_copy(nn.Linear(8, 6, bias=False), k=16)
    tiled.set_bypass_adc(True)
    inputs = torch.randn(5, 8)

    with torch.no_grad():
        expected = tiled(inputs)
        tiled.set_adc_midpoint(True)
        actual = tiled(inputs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--adc_midpoint", "--adc_off_eval_only"], "needs the converter enabled"),
        (
            ["--adc_midpoint", "--lora_rank", "4", "--lora_adc_off_control"],
            "needs the converter enabled",
        ),
        (
            ["--adc_midpoint", "--lora_rank", "4", "--lora_mode", "pre_adc"],
            "requires --lora_mode residual",
        ),
    ],
)
def test_midpoint_rejects_invalid_flag_combinations(
    monkeypatch,
    capsys,
    argv,
    message,
) -> None:
    monkeypatch.setattr(sys, "argv", ["llama_smooth_quant_adc_ptq", *argv])

    with pytest.raises(SystemExit) as excinfo:
        runner.main()

    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err
