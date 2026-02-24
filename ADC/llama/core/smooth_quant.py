"""
SmoothQuant: per-channel activation smoothing for LLaMA models.

Migrates quantization difficulty from activations to weights by computing
per-channel scaling factors and absorbing them into (RMSNorm, Linear) pairs.

Reference: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
Quantization for Large Language Models", 2023.
"""

import logging
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def calibrate_smooth_scales(
    model: nn.Module,
    dataloader,
    num_batches: int = 64,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Collect per-channel activation maximums for every nn.Linear layer
    (excluding embed_tokens and lm_head) by running calibration data.

    Returns:
        dict mapping layer name -> Tensor of shape [in_features] with
        max absolute activation value per input channel.
    """
    act_maxes: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name: str):
        def hook(module, input, output):
            x = input[0].detach().float()
            # x shape: [batch, seq_len, in_features] or [batch, in_features]
            x_abs = x.abs()
            if x_abs.dim() == 3:
                channel_max = x_abs.amax(dim=(0, 1))  # [in_features]
            else:
                channel_max = x_abs.amax(dim=0)
            if name in act_maxes:
                act_maxes[name] = torch.max(act_maxes[name], channel_max)
            else:
                act_maxes[name] = channel_max
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "embed_tokens" in name or "lm_head" in name:
                continue
            hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc="SmoothQuant calibration")):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device) if device else batch["input_ids"]
            attention_mask = batch["attention_mask"].to(device) if device else batch["attention_mask"]
            model(input_ids=input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    logger.info(f"Collected activation stats for {len(act_maxes)} linear layers")
    return act_maxes


def compute_smooth_scales(
    act_max: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Compute per-channel smooth scales: s_j = act_max_j^alpha / w_max_j^(1-alpha).

    Args:
        act_max: [in_features] max absolute activation per channel.
        weight:  [out_features, in_features] weight matrix.
        alpha:   migration strength (0 = all on weights, 1 = all on activations).

    Returns:
        Tensor of shape [in_features] with scaling factors.
    """
    w_max = weight.abs().amax(dim=0).float()  # [in_features]
    act_max = act_max.float()

    eps = 1e-5
    act_max = act_max.clamp(min=eps)
    w_max = w_max.clamp(min=eps)

    scales = act_max.pow(alpha) / w_max.pow(1.0 - alpha)
    scales = scales.clamp(min=eps)
    return scales


def apply_smooth_quant(
    model: nn.Module,
    act_maxes: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Apply SmoothQuant to a LLaMA model in-place.

    For each transformer layer, smooths the two (RMSNorm, Linear) groups:
      - input_layernorm  -> q_proj, k_proj, v_proj
      - post_attention_layernorm -> gate_proj, up_proj

    Returns:
        dict mapping group key -> computed smooth scales (for logging).
    """
    applied_scales = {}

    for name, module in model.named_modules():
        if not hasattr(module, "input_layernorm"):
            continue

        layer_prefix = name  # e.g. "model.layers.0"
        ln_attn = module.input_layernorm
        ln_mlp = module.post_attention_layernorm

        attn = module.self_attn
        mlp = module.mlp

        # --- Attention group: input_layernorm -> q/k/v_proj ---
        attn_linears = [attn.q_proj, attn.k_proj, attn.v_proj]
        attn_names = [f"{layer_prefix}.self_attn.{n}_proj" for n in ("q", "k", "v")]

        _apply_group(ln_attn, attn_linears, attn_names, act_maxes, alpha,
                     f"{layer_prefix}.attn", applied_scales)

        # --- MLP group: post_attention_layernorm -> gate/up_proj ---
        mlp_linears = [mlp.gate_proj, mlp.up_proj]
        mlp_names = [f"{layer_prefix}.mlp.{n}_proj" for n in ("gate", "up")]

        _apply_group(ln_mlp, mlp_linears, mlp_names, act_maxes, alpha,
                     f"{layer_prefix}.mlp", applied_scales)

    logger.info(f"Applied SmoothQuant to {len(applied_scales)} groups (alpha={alpha})")
    return applied_scales


def _apply_group(
    layernorm: nn.Module,
    linears: list[nn.Module],
    linear_names: list[str],
    act_maxes: dict[str, torch.Tensor],
    alpha: float,
    group_key: str,
    applied_scales: dict[str, torch.Tensor],
):
    """Apply smoothing to one (LayerNorm, [Linear, ...]) group."""
    available = [n for n in linear_names if n in act_maxes]
    if not available:
        logger.warning(f"No activation stats for group {group_key}, skipping")
        return

    # Use max across all linears in the group for shared scales
    combined_act_max = torch.stack([act_maxes[n] for n in available]).amax(dim=0)
    combined_w_max = torch.stack([l.weight.abs().amax(dim=0).float() for l in linears]).amax(dim=0)

    eps = 1e-5
    combined_act_max = combined_act_max.clamp(min=eps)
    combined_w_max = combined_w_max.clamp(min=eps)

    scales = combined_act_max.pow(alpha) / combined_w_max.pow(1.0 - alpha)
    scales = scales.clamp(min=eps)

    device = linears[0].weight.device
    dtype = linears[0].weight.dtype
    scales = scales.to(device)

    # Absorb 1/s into LayerNorm weight (gamma)
    with torch.no_grad():
        layernorm.weight.div_(scales.to(dtype))

        # Absorb s into each Linear weight on the input dimension
        for linear in linears:
            linear.weight.mul_(scales.unsqueeze(0).to(dtype))

    applied_scales[group_key] = scales.cpu()
