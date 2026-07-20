"""Regression tests for shared gated-decoder FlatQuant support."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ADC.llama.core.adc_layers import TiledLinearADC
from ADC.llama.core.flat_quant import (
    KroneckerTransform,
    _reparameterize_ln,
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
