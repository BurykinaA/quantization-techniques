"""Regression tests for Stage 3 on backbones without FlatQuant transforms."""

import pytest
import torch
import torch.nn as nn

from ADC.llama.core.adc_layers import TiledLinearADC
from ADC.llama.core.adc_lora import ResidualLoRATiledLinearADC, apply_adc_lora


def _tiled_copy(linear: nn.Linear) -> TiledLinearADC:
    tiled = TiledLinearADC(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=False,
        bx=4,
        bw=4,
        ba=8,
        k=16,
        signed_activations=True,
        mvm_limit=4,
    )
    tiled.load_weights(linear)
    for tile in tiled.tiles:
        scales = tile.weight.detach().abs().amax(dim=1).clamp(min=1e-8) / 7.0
        tile.weight_quantizer.scale.data = scales.clone()
        tile.weight_quantizer._scale_initialized = True
    return tiled


def test_residual_lora_wraps_a_bare_tiled_projection() -> None:
    # --preprocess_method none converts the projection itself to the tiled
    # layer, with no FlatQuantLinear around it.
    torch.manual_seed(11)
    tiled = _tiled_copy(nn.Linear(12, 8, bias=False))
    model = nn.ModuleDict({"mlp": nn.ModuleDict({"down_proj": tiled})})
    inputs = torch.randn(3, 12)
    with torch.no_grad():
        expected = tiled(inputs)

    apply_adc_lora(model, target_modules=["down_proj"], rank=2, lora_alpha=4.0)

    wrapper = model["mlp"]["down_proj"]
    assert isinstance(wrapper, ResidualLoRATiledLinearADC)
    assert wrapper.tiled_layer is tiled
    with torch.no_grad():
        torch.testing.assert_close(wrapper(inputs), expected)

    wrapper(inputs).sum().backward()
    assert wrapper.lora_B.weight.grad.abs().sum() > 0


def test_apply_adc_lora_rejects_a_model_without_tiled_projections() -> None:
    model = nn.ModuleDict({"down_proj": nn.Linear(12, 8, bias=False)})

    with pytest.raises(ValueError, match="No TiledLinearADC"):
        apply_adc_lora(model, target_modules=["down_proj"], rank=2, lora_alpha=4.0)
