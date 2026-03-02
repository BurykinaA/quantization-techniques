"""
FlatQuant-style preprocessing for LLaMA models.

This module provides a native in-repo approximation of FlatQuant behavior for
W4A4-first workflows by learning per-channel diagonal transforms from
calibration activations and absorbing them into (RMSNorm, Linear) groups.
"""

import logging

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def calibrate_flat_stats(
    model: nn.Module,
    dataloader,
    num_batches: int = 64,
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Collect activation statistics for each nn.Linear (excluding embeddings/head).

    Returns:
        dict[layer_name] with:
          - "amax": per-channel max abs activation, shape [in_features]
          - "std":  per-channel activation std, shape [in_features]
    """
    act_amax: dict[str, torch.Tensor] = {}
    act_sq_sum: dict[str, torch.Tensor] = {}
    act_sum: dict[str, torch.Tensor] = {}
    act_count: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name: str):
        def hook(module, inputs, output):
            x = inputs[0].detach().float()
            x_abs = x.abs()
            if x_abs.dim() == 3:
                channel_amax = x_abs.amax(dim=(0, 1))
                reduce_dims = (0, 1)
            else:
                channel_amax = x_abs.amax(dim=0)
                reduce_dims = 0

            channel_sum = x.sum(dim=reduce_dims)
            channel_sq_sum = (x * x).sum(dim=reduce_dims)
            sample_count = torch.tensor(float(x.numel() // x.shape[-1]), device=x.device)

            if name in act_amax:
                act_amax[name] = torch.max(act_amax[name], channel_amax)
                act_sum[name] = act_sum[name] + channel_sum
                act_sq_sum[name] = act_sq_sum[name] + channel_sq_sum
                act_count[name] = act_count[name] + sample_count
            else:
                act_amax[name] = channel_amax
                act_sum[name] = channel_sum
                act_sq_sum[name] = channel_sq_sum
                act_count[name] = sample_count

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "embed_tokens" in name or "lm_head" in name:
                continue
            hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc="FlatQuant calibration")):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device) if device else batch["input_ids"]
            attention_mask = batch["attention_mask"].to(device) if device else batch["attention_mask"]
            model(input_ids=input_ids, attention_mask=attention_mask)

    for hook in hooks:
        hook.remove()

    stats: dict[str, dict[str, torch.Tensor]] = {}
    eps = 1e-8
    for name in act_amax:
        count = act_count[name].clamp(min=1.0)
        mean = act_sum[name] / count
        second_moment = act_sq_sum[name] / count
        var = (second_moment - mean * mean).clamp(min=0.0)
        std = torch.sqrt(var + eps)
        stats[name] = {"amax": act_amax[name], "std": std}

    logger.info(f"Collected FlatQuant activation stats for {len(stats)} linear layers")
    return stats


def fit_flat_transforms(
    act_stats: dict[str, dict[str, torch.Tensor]],
    linears_by_name: dict[str, nn.Module],
    beta: float = 0.5,
    flatten_strength: float = 0.25,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """
    Fit per-channel diagonal transforms for each linear layer.

    The transform is designed to reduce dynamic-range outliers jointly across
    activations and weights:
        scale_j ~ (a_j / w_j)^beta * (a_j / std_j)^flatten_strength
    """
    transforms: dict[str, torch.Tensor] = {}
    for name, stats in act_stats.items():
        linear = linears_by_name.get(name)
        if linear is None:
            continue

        act_amax = stats["amax"].float().clamp(min=eps)
        act_std = stats["std"].float().clamp(min=eps)
        w_amax = linear.weight.detach().float().abs().amax(dim=0).clamp(min=eps)

        balance = act_amax.pow(beta) / w_amax.pow(1.0 - beta)
        flatness = (act_amax / act_std).pow(flatten_strength)
        scale = (balance / flatness).clamp(min=eps)

        # Guardrails for numerical stability in low-bit workflows.
        scale = scale.clamp(min=1e-2, max=1e2)
        transforms[name] = scale

    logger.info(f"Fitted FlatQuant transforms for {len(transforms)} linear layers")
    return transforms


def apply_flat_quant(
    model: nn.Module,
    transforms: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Apply FlatQuant transforms in-place to LLaMA (RMSNorm, Linear) groups.

    Returns:
        dict[group_key] -> applied shared scales for logging.
    """
    applied_scales: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not hasattr(module, "input_layernorm"):
            continue

        layer_prefix = name
        ln_attn = module.input_layernorm
        ln_mlp = module.post_attention_layernorm
        attn = module.self_attn
        mlp = module.mlp

        attn_linears = [attn.q_proj, attn.k_proj, attn.v_proj]
        attn_names = [f"{layer_prefix}.self_attn.{n}_proj" for n in ("q", "k", "v")]
        _apply_group_transforms(
            layernorm=ln_attn,
            linears=attn_linears,
            linear_names=attn_names,
            transforms=transforms,
            group_key=f"{layer_prefix}.attn",
            applied_scales=applied_scales,
        )

        mlp_linears = [mlp.gate_proj, mlp.up_proj]
        mlp_names = [f"{layer_prefix}.mlp.{n}_proj" for n in ("gate", "up")]
        _apply_group_transforms(
            layernorm=ln_mlp,
            linears=mlp_linears,
            linear_names=mlp_names,
            transforms=transforms,
            group_key=f"{layer_prefix}.mlp",
            applied_scales=applied_scales,
        )

    logger.info(f"Applied FlatQuant to {len(applied_scales)} groups")
    return applied_scales


def _apply_group_transforms(
    layernorm: nn.Module,
    linears: list[nn.Module],
    linear_names: list[str],
    transforms: dict[str, torch.Tensor],
    group_key: str,
    applied_scales: dict[str, torch.Tensor],
) -> None:
    available_names = [name for name in linear_names if name in transforms]
    if not available_names:
        logger.warning(f"No FlatQuant transforms for group {group_key}, skipping")
        return

    group_scale = torch.stack([transforms[name] for name in available_names]).mean(dim=0)
    group_scale = group_scale.clamp(min=1e-5)

    device = linears[0].weight.device
    dtype = linears[0].weight.dtype
    group_scale = group_scale.to(device)

    with torch.no_grad():
        layernorm.weight.div_(group_scale.to(dtype))
        for linear in linears:
            linear.weight.mul_(group_scale.unsqueeze(0).to(dtype))

    applied_scales[group_key] = group_scale.detach().cpu()
