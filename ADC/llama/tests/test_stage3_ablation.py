"""Regression tests for Stage 3 on backbones without FlatQuant transforms."""

import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ADC.llama.core.adc_layers import TiledLinearADC
from ADC.llama.core.adc_lora import (
    ResidualLoRATiledLinearADC,
    apply_adc_lora,
    calibrate_adc_lora,
    load_adc_lora,
    save_adc_lora,
)
from ADC.llama.runs import llama_smooth_quant_adc_ptq as runner


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


def test_bfloat16_residual_starts_from_the_rounded_float32_initialization() -> None:
    tiled = _tiled_copy(nn.Linear(12, 8, bias=False))
    torch.manual_seed(13)
    float32 = ResidualLoRATiledLinearADC(tiled, r=2, alpha=4.0)
    torch.manual_seed(13)
    bfloat16 = ResidualLoRATiledLinearADC(tiled, r=2, alpha=4.0, dtype=torch.bfloat16)

    assert bfloat16.lora_A.weight.dtype == torch.bfloat16
    assert bfloat16.lora_B.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        bfloat16.lora_A.weight, float32.lora_A.weight.to(torch.bfloat16)
    )
    inputs = torch.randn(3, 12)
    with torch.no_grad():
        torch.testing.assert_close(bfloat16(inputs), tiled(inputs))


def test_apply_adc_lora_keeps_pre_adc_lora_in_float32() -> None:
    model = nn.ModuleDict({"down_proj": _tiled_copy(nn.Linear(12, 8, bias=False))})

    with pytest.raises(ValueError, match="requires mode='residual'"):
        apply_adc_lora(
            model,
            target_modules=["down_proj"],
            rank=2,
            lora_alpha=4.0,
            mode="pre_adc",
            lora_dtype=torch.bfloat16,
        )


def test_runner_rejects_a_bfloat16_pre_adc_lora(monkeypatch, capsys) -> None:
    argv = ["--lora_rank", "4", "--lora_mode", "pre_adc", "--lora_dtype", "bfloat16"]
    monkeypatch.setattr(sys, "argv", ["llama_smooth_quant_adc_ptq", *argv])

    with pytest.raises(SystemExit) as excinfo:
        runner.main()

    assert excinfo.value.code == 2
    assert "requires --lora_mode residual" in capsys.readouterr().err


def _wrapped_down_proj(linear: nn.Linear, rank: int = 2,
                       dtype: torch.dtype = torch.float32) -> nn.ModuleDict:
    model = nn.ModuleDict({"mlp": nn.ModuleDict({"down_proj": _tiled_copy(linear)})})
    apply_adc_lora(model, target_modules=["down_proj"], rank=rank, lora_alpha=4.0,
                   lora_dtype=dtype)
    return model


def test_saved_adapters_reload_into_a_fresh_wrapper(tmp_path) -> None:
    torch.manual_seed(19)
    linear = nn.Linear(12, 8, bias=False)
    trained = _wrapped_down_proj(linear)
    with torch.no_grad():
        for parameter in trained.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter))
    inputs = torch.randn(3, 12)
    with torch.no_grad():
        expected = trained["mlp"]["down_proj"](inputs)
    path = tmp_path / "lora_weights.pt"
    assert save_adc_lora(trained, str(path), {"lora_alpha": 4.0}) == 2

    # Same backbone, untrained adapters.
    fresh = _wrapped_down_proj(linear)
    with torch.no_grad():
        assert not torch.allclose(fresh["mlp"]["down_proj"](inputs), expected)
    assert load_adc_lora(fresh, str(path)) == {"lora_alpha": 4.0}
    with torch.no_grad():
        torch.testing.assert_close(fresh["mlp"]["down_proj"](inputs), expected)


@pytest.mark.parametrize("rank, dtype", [(3, torch.float32), (2, torch.bfloat16)])
def test_reload_rejects_adapters_of_another_rank_or_dtype(tmp_path, rank, dtype) -> None:
    linear = nn.Linear(12, 8, bias=False)
    path = tmp_path / "lora_weights.pt"
    save_adc_lora(_wrapped_down_proj(linear), str(path), {})

    with pytest.raises(ValueError):
        load_adc_lora(_wrapped_down_proj(linear, rank=rank, dtype=dtype), str(path))


class _StubCausalLM(nn.Module):
    """Embedding -> one tiled projection -> head, with the HF forward signature."""

    def __init__(self, vocab: int = 16, hidden: int = 12) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(use_cache=True)
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = _tiled_copy(nn.Linear(hidden, hidden, bias=False))
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None):
        logits = self.head(self.proj(self.embed(input_ids)))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return types.SimpleNamespace(loss=loss, logits=logits)


def test_ce_kl_training_with_a_preloaded_teacher_updates_only_the_adapters() -> None:
    torch.manual_seed(23)
    student = _StubCausalLM()
    teacher = _StubCausalLM()
    apply_adc_lora(student, target_modules=["proj"], rank=2, lora_alpha=4.0)
    backbone_before = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if "lora_" not in name
    }
    dataloader = [
        {"input_ids": torch.randint(0, 16, (2, 8)), "attention_mask": torch.ones(2, 8, dtype=torch.long)}
        for _ in range(3)
    ]

    calibrate_adc_lora(
        student,
        dataloader,
        device=torch.device("cpu"),
        nsamples=4,
        cali_bsz=2,
        epochs=1,
        lora_loss="ce_kl",
        teacher_model=teacher,
        teacher_device="cpu",
        microbatch_size=1,
        gradient_accumulation_steps=2,
    )

    assert student.proj.lora_B.weight.abs().sum() > 0
    for name, parameter in student.named_parameters():
        if "lora_" not in name:
            torch.testing.assert_close(parameter, backbone_before[name])
    assert not student.training
    assert student.config.use_cache is True


def test_runner_rejects_saving_adapters_without_lora(monkeypatch, capsys) -> None:
    argv = ["--lora_rank", "0", "--save_lora_weights"]
    monkeypatch.setattr(sys, "argv", ["llama_smooth_quant_adc_ptq", *argv])

    with pytest.raises(SystemExit) as excinfo:
        runner.main()

    assert excinfo.value.code == 2
    assert "require --lora_rank > 0" in capsys.readouterr().err
