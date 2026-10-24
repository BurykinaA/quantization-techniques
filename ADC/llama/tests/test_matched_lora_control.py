"""Regression tests for the matched no-ADC LoRA control."""

import sys

import pytest
import torch
import torch.nn as nn

from ADC.llama.core.adc_layers import TiledLinearADC
from ADC.llama.core.adc_lora import (
    ResidualLoRATiledLinearADC,
    apply_adc_lora,
    load_adc_lora,
    save_adc_lora,
)
from ADC.llama.core.flat_quant import FlatQuantLinear
from ADC.llama.runs import llama_smooth_quant_adc_ptq as runner

ADC_CONFIG = {
    "bx": 4,
    "bw": 4,
    "ba": 8,
    "k": 16,
    "mvm_limit": 3,
    "signed_activations": True,
}


def _tiled_copy(linear: nn.Linear) -> TiledLinearADC:
    tiled = TiledLinearADC(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        bx=4,
        bw=4,
        ba=8,
        k=16,
        signed_activations=True,
        mvm_limit=ADC_CONFIG["mvm_limit"],
    )
    tiled.load_weights(linear)
    for tile in tiled.tiles:
        scales = tile.weight.detach().abs().amax(dim=1).clamp(min=1e-8) / 7.0
        tile.weight_quantizer.scale.data = scales.clone()
        tile.weight_quantizer._scale_initialized = True
    return tiled


def _lora_model(rank: int = 2, seed: int = 3) -> nn.Module:
    torch.manual_seed(seed)
    linear = nn.Linear(12, 8, bias=False)
    flat_linear = FlatQuantLinear(linear, w_bits=4, a_bits=4)
    flat_linear.linear = _tiled_copy(linear)
    model = nn.ModuleDict({"down_proj": flat_linear})
    apply_adc_lora(model, target_modules=["down_proj"], rank=rank, lora_alpha=4.0)
    return model


def test_bypass_floor_training_path_matches_adc_bypassed_inference() -> None:
    torch.manual_seed(7)
    linear = nn.Linear(12, 8, bias=True)
    flat_linear = FlatQuantLinear(
        linear,
        w_bits=4,
        a_bits=4,
        lwc=False,
        lac=False,
        adc_config={**ADC_CONFIG, "bypass_floor_ste": True},
    )
    tiled = _tiled_copy(linear)
    tiled.set_bypass_adc(True)

    inputs = torch.randn(2, 5, 12)

    torch.testing.assert_close(
        tiled(inputs),
        flat_linear.train_forward(inputs),
        rtol=1e-5,
        atol=1e-5,
    )


def test_bypass_floor_removes_converter_from_training_path() -> None:
    torch.manual_seed(7)
    linear = nn.Linear(12, 8, bias=False)
    with_floor = FlatQuantLinear(linear, w_bits=4, a_bits=4, adc_config=ADC_CONFIG)
    without_floor = FlatQuantLinear(
        linear,
        w_bits=4,
        a_bits=4,
        adc_config={**ADC_CONFIG, "bypass_floor_ste": True},
    )
    inputs = torch.randn(4, 12)

    assert not torch.allclose(
        with_floor.train_forward(inputs),
        without_floor.train_forward(inputs),
    )


def test_adc_bypass_reaches_tiled_layer_inside_residual_lora() -> None:
    model = _lora_model()
    wrapper = model["down_proj"].linear
    assert isinstance(wrapper, ResidualLoRATiledLinearADC)

    runner.set_adc_bypass(model, True)
    assert all(tile.bypass_adc for tile in wrapper.tiled_layer.tiles)
    assert not any(tile.bypass_all for tile in wrapper.tiled_layer.tiles)

    runner.set_adc_bypass(model, False)
    assert not any(tile.bypass_adc for tile in wrapper.tiled_layer.tiles)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--lora_adc_off_control"], "requires --lora_rank > 0"),
        (["--lora_rank", "4", "--run_adc_on_eval"], "requires --lora_adc_off_control"),
        (
            ["--lora_rank", "4", "--lora_adc_off_control", "--run_no_adc_eval"],
            "already evaluates without ADC",
        ),
        (
            ["--lora_rank", "4", "--lora_adc_off_control", "--lora_mode", "pre_adc"],
            "requires --lora_mode residual",
        ),
        (
            ["--lora_rank", "4", "--lora_save_path", "a.pt", "--lora_load_path", "b.pt"],
            "mutually exclusive",
        ),
        (["--fq_bypass_adc_floor"], "requires --preprocess_method flat_quant"),
    ],
)
def test_matched_control_rejects_invalid_flag_combinations(
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


def test_adc_lora_roundtrip_restores_adapter_and_metadata(tmp_path) -> None:
    trained = _lora_model()
    with torch.no_grad():
        trained["down_proj"].linear.lora_A.weight.normal_()
        trained["down_proj"].linear.lora_B.weight.normal_()
    path = tmp_path / "lora_adapter.pt"
    metadata = {"lora_rank": 2, "lora_adc_off_control": True}

    save_adc_lora(trained, str(path), metadata=metadata)
    restored = _lora_model()
    loaded_metadata = load_adc_lora(restored, str(path))

    assert loaded_metadata == metadata
    assert not list(tmp_path.glob("*.tmp.*"))
    for name in ("lora_A", "lora_B"):
        torch.testing.assert_close(
            getattr(restored["down_proj"].linear, name).weight,
            getattr(trained["down_proj"].linear, name).weight,
        )
    inputs = torch.randn(3, 12)
    with torch.no_grad():
        torch.testing.assert_close(
            restored["down_proj"].linear(inputs),
            trained["down_proj"].linear(inputs),
        )


def test_adc_lora_load_rejects_different_rank(tmp_path) -> None:
    path = tmp_path / "lora_adapter.pt"
    save_adc_lora(_lora_model(rank=2), str(path))

    with pytest.raises(ValueError, match="shape mismatch"):
        load_adc_lora(_lora_model(rank=3), str(path))


def test_adc_lora_load_rejects_missing_adapter_tensors(tmp_path) -> None:
    path = tmp_path / "lora_adapter.pt"
    save_adc_lora(_lora_model(), str(path))
    model = _lora_model()
    model["up_proj"] = _lora_model(seed=9)["down_proj"]

    with pytest.raises(ValueError, match="missing"):
        load_adc_lora(model, str(path))
