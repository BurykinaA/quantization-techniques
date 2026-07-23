"""Regression tests for shared gated-decoder FlatQuant support."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ADC.llama.core.adc_layers import TiledLinearADC
from ADC.llama.core.flat_quant import (
    FlatQuantLinear,
    KroneckerTransform,
    _capture_calibration_batch,
    _project_flatquant_parameters,
    _reparameterize_ln,
    apply_flatquant_to_model,
    load_flat_transforms,
    reparameterize_model,
    save_flat_transforms,
    validate_model_layout,
)


class DummyAttention(nn.Module):
    def __init__(self, hidden_size: int = 8, bias: bool = False):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class DummyMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()


class DummyDecoderLayer(nn.Module):
    def __init__(self, affine_norm: bool = True, attention_bias: bool = False):
        super().__init__()
        self.self_attn = DummyAttention(bias=attention_bias)
        self.mlp = DummyMLP()
        self.input_layernorm = nn.LayerNorm(8, elementwise_affine=affine_norm)
        self.post_attention_layernorm = nn.LayerNorm(
            8,
            elementwise_affine=affine_norm,
        )


class DummyBackbone(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([layer])


class DummyCausalLM(nn.Module):
    def __init__(self, layer: nn.Module, model_type: str):
        super().__init__()
        self.model = DummyBackbone(layer)
        self.config = SimpleNamespace(model_type=model_type, hidden_size=8)


def test_affine_norm_absorbs_diagonal() -> None:
    norm = nn.LayerNorm(8)
    transform = KroneckerTransform(8, add_diag=True)
    diagonal = torch.linspace(0.5, 1.5, 8)
    transform.diag_scale.data.copy_(diagonal)

    _reparameterize_ln(norm, transform)

    torch.testing.assert_close(norm.weight, diagonal)
    assert transform.use_diag is False


def test_non_affine_norm_keeps_diagonal_active() -> None:
    norm = nn.LayerNorm(8, elementwise_affine=False)
    transform = KroneckerTransform(8, add_diag=True)
    diagonal_before = transform.diag_scale.detach().clone()

    _reparameterize_ln(norm, transform)

    assert norm.weight is None
    assert transform.use_diag is True
    torch.testing.assert_close(transform.diag_scale, diagonal_before)


def test_transform_projection_bounds_both_directions() -> None:
    transform = KroneckerTransform(8, add_diag=True)
    transform.diag_left.data.fill_(0.001)
    transform.diag_right.data.fill_(100.0)
    transform.diag_scale.data[0] = 100.0

    _project_flatquant_parameters(transform)

    assert transform.diag_left.min().item() == pytest.approx(0.1)
    assert transform.diag_right.max().item() == pytest.approx(10.0)
    assert transform.diag_scale.max().item() == pytest.approx(10.0)


def test_calibration_capture_keeps_every_batch_element() -> None:
    storage = torch.zeros(6, 2)
    first_batch = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    second_batch = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)

    next_index = _capture_calibration_batch(storage, first_batch, 0, 6)
    next_index = _capture_calibration_batch(storage, second_batch, next_index, 6)

    assert next_index == 6
    torch.testing.assert_close(
        storage,
        torch.cat((first_batch, second_batch[:2]), dim=0),
    )


@pytest.mark.parametrize(
    ("model_type", "affine_norm", "attention_bias"),
    [
        ("llama", True, False),
        ("qwen2", True, True),
        ("olmo", False, False),
    ],
)
def test_layout_validation_accepts_selected_architectures(
    model_type: str,
    affine_norm: bool,
    attention_bias: bool,
) -> None:
    model = DummyCausalLM(
        DummyDecoderLayer(
            affine_norm=affine_norm,
            attention_bias=attention_bias,
        ),
        model_type=model_type,
    )

    layout = validate_model_layout(model)

    assert layout["model_type"] == model_type
    assert layout["num_layers"] == 1
    assert layout["affine_input_norm"] is affine_norm
    assert layout["affine_post_attention_norm"] is affine_norm


def test_layout_validation_rejects_missing_projection() -> None:
    model = DummyCausalLM(DummyDecoderLayer(), model_type="unsupported")
    del model.model.layers[0].self_attn.q_proj

    with pytest.raises(ValueError, match="q_proj"):
        validate_model_layout(model)


def test_stage_a_transform_checkpoint_roundtrip(tmp_path) -> None:
    model = DummyCausalLM(DummyDecoderLayer(), model_type="llama")
    apply_flatquant_to_model(
        model,
        w_bits=4,
        a_bits=4,
        add_diag=True,
        lwc=True,
        lac=True,
    )
    expected = torch.linspace(0.5, 1.5, 8)
    transform = model.model.layers[0].self_attn.ln_trans
    transform.diag_scale.data.copy_(expected)
    checkpoint_path = tmp_path / "flat_quant_transforms_stage_a.pt"

    save_flat_transforms(model, str(checkpoint_path))
    transform.diag_scale.data.fill_(1.0)
    load_flat_transforms(model, str(checkpoint_path))

    torch.testing.assert_close(transform.diag_scale, expected)


def test_tiled_adc_preserves_qwen_style_linear_bias() -> None:
    linear = nn.Linear(8, 4, bias=True)
    tiled = TiledLinearADC(
        in_features=8,
        out_features=4,
        bias=True,
        bx=4,
        bw=4,
        ba=8,
        k=16,
        signed_activations=True,
        mvm_limit=4,
    )
    tiled.load_weights(linear)
    tiled.set_bypass_all(True)
    inputs = torch.randn(3, 8)

    assert tiled.tiles[0].bias is not None
    assert tiled.tiles[1].bias is None
    torch.testing.assert_close(tiled.tiles[0].bias, linear.bias)
    torch.testing.assert_close(tiled(inputs), linear(inputs))


def test_qwen_style_adc_training_path_matches_tiled_inference() -> None:
    torch.manual_seed(7)
    linear = nn.Linear(12, 8, bias=True)
    adc_config = {
        "bx": 4,
        "bw": 4,
        "ba": 8,
        "k": 16,
        "mvm_limit": 3,
        "signed_activations": True,
    }
    flat_linear = FlatQuantLinear(
        linear,
        w_bits=4,
        a_bits=4,
        lwc=False,
        lac=False,
        adc_config=adc_config,
    )
    tiled = TiledLinearADC(
        in_features=12,
        out_features=8,
        bias=True,
        bx=4,
        bw=4,
        ba=8,
        k=16,
        signed_activations=True,
        mvm_limit=3,
    )
    tiled.load_weights(linear)
    for tile in tiled.tiles:
        scales = tile.weight.detach().abs().amax(dim=1).clamp(min=1e-8) / 7.0
        tile.weight_quantizer.scale.data = scales.clone()
        tile.weight_quantizer._scale_initialized = True

    inputs = torch.randn(2, 5, 12)
    expected = flat_linear.train_forward(inputs)
    actual = tiled(inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_tiny_qwen_reparameterization_preserves_fp_logits() -> None:
    transformers = pytest.importorskip("transformers")
    qwen_config_class = getattr(transformers, "Qwen2Config", None)
    qwen_model_class = getattr(transformers, "Qwen2ForCausalLM", None)
    if qwen_config_class is None or qwen_model_class is None:
        pytest.skip("Installed transformers does not expose Qwen2")

    torch.manual_seed(11)
    config = qwen_config_class(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    model = qwen_model_class(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        expected = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    apply_flatquant_to_model(
        model,
        w_bits=16,
        a_bits=16,
        add_diag=True,
        lwc=False,
        lac=False,
    )
    reparameterize_model(model)

    with torch.no_grad():
        actual = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
