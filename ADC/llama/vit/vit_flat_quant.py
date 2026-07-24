"""
FlatQuant wrappers for timm Vision Transformers.

This is the ViT analogue of the LLaMA-specific wrappers in
``ADC/llama/core/flat_quant.py`` (FlatQuantLlamaMLP / FlatQuantLlamaAttention /
apply_flatquant_to_model / reparameterize_model / calibrate_flat_quant).
It reuses the architecture-agnostic building blocks unchanged:

    ADC.llama.core.flat_quant.FlatQuantLinear      — per-projection transform + fake/ADC quant
    ADC.llama.core.flat_quant.KroneckerTransform   — learnable Kronecker transform + diagonal
    ADC.llama.core.flat_quant._QuantProjectionWrapper — swaps a projection during attn training

Structural mapping (timm ViT block ← LLaMA decoder layer):

    blk.norm1 → attn(·)     ⟺   input_layernorm → self_attn
    blk.norm2 → mlp(·)      ⟺   post_attention_layernorm → mlp
    attn.qkv (fused q/k/v)  ⟺   q_proj / k_proj / v_proj
    attn.proj               ⟺   o_proj
    mlp.fc1 → GELU → fc2    ⟺   gate/up_proj → SiLU·× → down_proj

Transforms per block:
    ln_trans   : applied to attention input  (before qkv)
    in_trans   : applied to MLP input        (before fc1)
    mid_trans  : applied to fc2 input        (after GELU, before fc2)

The Kronecker transform is applied to activations at inference; its inverse is
folded into the projection weights by ``reparameterize()``.  The per-channel
``diag_scale`` is *not* folded into LayerNorm — in eval mode the activation-side
``·diag`` and the weight-side ``/diag`` cancel exactly over the shared feature
axis, so no LayerNorm/bias manipulation is needed (simpler than the LLaMA path,
and provably equivalent).

PACT (raw_alpha_adc) is intentionally left at its default so TiledLinearADC uses
per-token amax at inference — the documented baseline activation quantization.
"""

import gc
import functools
import logging
from contextlib import nullcontext

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ADC.llama.core.flat_quant import (
    FlatQuantLinear,
    KroneckerTransform,
    _QuantProjectionWrapper,
)

logger = logging.getLogger(__name__)


# =========================================================================
# ViT module wrappers
# =========================================================================

class FlatQuantViTAttention(nn.Module):
    """timm ViT attention wrapped with FlatQuant transforms.

    Keeps the original timm ``Attention`` module (so the softmax attention math
    stays intact) and only intercepts the ``qkv`` and ``proj`` linear layers.

    ln_trans : shared transform applied to the (LayerNorm'd) attention input.
    """

    def __init__(self, attn: nn.Module, w_bits, a_bits, add_diag, lwc, lac,
                 adc_config=None):
        super().__init__()
        self._orig_attn = attn
        in_dim = attn.qkv.weight.shape[1]

        self.qkv = FlatQuantLinear(attn.qkv, w_bits, a_bits, lwc, lac, adc_config)
        self.proj = FlatQuantLinear(attn.proj, w_bits, a_bits, lwc, lac, adc_config)

        self.ln_trans = KroneckerTransform(in_dim, add_diag=add_diag)

        self._ori_mode = False
        self._collect_smax = add_diag
        if self._collect_smax:
            self._ln_smax = torch.ones(in_dim, device="cpu") * 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ori_mode:
            return self._ori_forward(x)
        return self._train_forward(x)

    def _ori_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._collect_smax and hasattr(self, "_ln_smax"):
            self._ln_smax = torch.maximum(
                self._ln_smax.to(x.device),
                x.reshape(-1, x.shape[-1]).abs().amax(dim=0).detach(),
            )
        return self._orig_attn(x)

    def _train_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward-transform the attention input, then route qkv/proj through
        # quantised wrappers while the original attention runs the softmax math.
        x_t = self.ln_trans(x)
        saved = (self._orig_attn.qkv, self._orig_attn.proj)
        try:
            self._orig_attn.qkv = _QuantProjectionWrapper(self.qkv, self.ln_trans)
            self._orig_attn.proj = _QuantProjectionWrapper(self.proj, None)
            return self._orig_attn(x_t)
        finally:
            self._orig_attn.qkv, self._orig_attn.proj = saved

    def init_diag_scale(self, alpha: float = 0.5) -> None:
        if not hasattr(self, "_ln_smax"):
            return
        qkv_w = self.qkv.linear.weight.abs().amax(dim=0)
        eps = 1e-5
        self.ln_trans.diag_scale.data = (
            qkv_w.pow(1 - alpha) / self._ln_smax.to(qkv_w.device).pow(alpha)
        ).clamp(min=eps)
        del self._ln_smax
        self._collect_smax = False

    def reparameterize(self) -> None:
        self.ln_trans.to_eval_mode()
        self.qkv.reparameterize(qa_trans=self.ln_trans)
        self.proj.reparameterize()   # proj input is not ln_trans-transformed


class FlatQuantViTMlp(nn.Module):
    """timm ViT MLP (fc1 → GELU → fc2) wrapped with FlatQuant transforms.

    in_trans  : applied to the MLP input (before fc1).
    mid_trans : applied to fc2's input (after GELU).  Its diagonal cannot be
                folded into a preceding linear (GELU is nonlinear), so it stays
                active at inference — the ·diag / ÷diag pair cancels exactly.
    """

    def __init__(self, mlp: nn.Module, w_bits, a_bits, add_diag, lwc, lac,
                 adc_config=None):
        super().__init__()
        self.act = mlp.act

        self.fc1 = FlatQuantLinear(mlp.fc1, w_bits, a_bits, lwc, lac, adc_config)
        self.fc2 = FlatQuantLinear(mlp.fc2, w_bits, a_bits, lwc, lac, adc_config)

        in_dim = mlp.fc1.weight.shape[1]
        hidden_dim = mlp.fc2.weight.shape[1]

        self.in_trans = KroneckerTransform(in_dim, add_diag=add_diag)
        self.mid_trans = KroneckerTransform(hidden_dim, add_diag=add_diag)

        self._ori_mode = False
        self._collect_smax = add_diag
        if self._collect_smax:
            self._in_smax = torch.ones(in_dim, device="cpu") * 1e-5
            self._mid_smax = torch.ones(hidden_dim, device="cpu") * 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def _ori_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._collect_smax and hasattr(self, "_in_smax"):
            self._in_smax = torch.maximum(
                self._in_smax.to(x.device),
                x.reshape(-1, x.shape[-1]).abs().amax(dim=0).detach(),
            )
        h = self.act(self.fc1.ori_forward(x))
        if self._collect_smax and hasattr(self, "_mid_smax"):
            self._mid_smax = torch.maximum(
                self._mid_smax.to(h.device),
                h.reshape(-1, h.shape[-1]).abs().amax(dim=0).detach(),
            )
        return self.fc2.ori_forward(h)

    def _trans_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = self.in_trans(x)
        h = self.act(self.fc1.train_forward(x_t, qa_trans=self.in_trans))
        h_t = self.mid_trans(h)
        return self.fc2.train_forward(h_t, qa_trans=self.mid_trans)

    def init_diag_scale(self, alpha: float = 0.5) -> None:
        if not hasattr(self, "_in_smax"):
            return
        eps = 1e-5
        fc1_w = self.fc1.linear.weight.abs().amax(dim=0)
        fc2_w = self.fc2.linear.weight.abs().amax(dim=0)
        self.in_trans.diag_scale.data = (
            fc1_w.pow(1 - alpha) / self._in_smax.to(fc1_w.device).pow(alpha)
        ).clamp(min=eps)
        self.mid_trans.diag_scale.data = (
            fc2_w.pow(1 - alpha) / self._mid_smax.to(fc2_w.device).pow(alpha)
        ).clamp(min=eps)
        del self._in_smax, self._mid_smax
        self._collect_smax = False

    def reparameterize(self) -> None:
        self.in_trans.to_eval_mode()
        self.mid_trans.to_eval_mode()
        self.fc1.reparameterize(qa_trans=self.in_trans)
        self.fc2.reparameterize(qa_trans=self.mid_trans)


# =========================================================================
# Model-level operations
# =========================================================================

def _vit_blocks(model: nn.Module):
    """Return the ViT transformer blocks (timm exposes them as ``model.blocks``)."""
    return model.blocks


def apply_flatquant_to_vit(model, w_bits=4, a_bits=4, add_diag=True,
                           lwc=True, lac=True, adc_config=None):
    """Replace each timm ViT block's attn/mlp with FlatQuant-wrapped versions."""
    blocks = _vit_blocks(model)
    for i in tqdm(range(len(blocks)), desc="Applying FlatQuant wrappers (ViT)"):
        blk = blocks[i]
        blk.attn = FlatQuantViTAttention(blk.attn, w_bits, a_bits, add_diag, lwc, lac, adc_config)
        blk.mlp = FlatQuantViTMlp(blk.mlp, w_bits, a_bits, add_diag, lwc, lac, adc_config)
    return model


def reparameterize_vit(model):
    """Fold all FlatQuant transforms into projection weights.

    Wrappers are kept (their Kronecker transform is still applied to
    activations); ``diag_scale`` stays active and cancels analytically with the
    inverse-transformed weights, so no LayerNorm folding is performed.
    """
    for blk in _vit_blocks(model):
        if isinstance(blk.attn, FlatQuantViTAttention):
            blk.attn.reparameterize()
        if isinstance(blk.mlp, FlatQuantViTMlp):
            blk.mlp.reparameterize()
    return model


# =========================================================================
# Layer-by-layer MSE calibration (ViT)
# =========================================================================

def _get_params_by_pattern(module, patterns):
    params = []
    for name, param in module.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = True
            params.append(param)
    return params


@torch.no_grad()
def _capture_block0_inputs(model, dataloader, device, nsamples, embed_dim):
    """Run calibration images through patch-embed/pos stages to capture the
    input tensor of the first transformer block."""
    blocks = _vit_blocks(model)
    inps = None
    cache = {"i": 0}

    class _Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x):
            nonlocal inps
            n = x.shape[0]
            if cache["i"] < nsamples:
                if inps is None:
                    inps = torch.zeros((nsamples, x.shape[1], embed_dim),
                                       dtype=torch.float32, device=device)
                take = min(n, nsamples - cache["i"])
                inps[cache["i"]:cache["i"] + take] = x[:take].float()
                cache["i"] += take
            raise ValueError("catch")

    orig = blocks[0]
    blocks[0] = _Catcher(orig)
    model_dev = next(model.parameters()).device
    try:
        for imgs, _ in dataloader:
            if cache["i"] >= nsamples:
                break
            try:
                model(imgs.to(model_dev))
            except ValueError:
                pass
    finally:
        blocks[0] = orig

    n = cache["i"]
    logger.info(f"Captured {n} ViT block-0 input samples")
    return inps[:n], n


def calibrate_flat_quant_vit(
    model, dataloader, device,
    nsamples=1024, cali_bsz=16, epochs=30, flat_lr=5e-3,
    diag_alpha=0.5, add_diag=True, lwc=True, lac=True,
    propagate_quant_inputs=False, propagate_quant_alpha=0.5,
    diag_attn=True, diag_mlp=True,
):
    """Train FlatQuant transforms block-by-block via MSE against FP outputs.

    Mirrors core.flat_quant.calibrate_flat_quant, specialised to timm ViT:
    blocks take a single positional tensor (no attention masks / RoPE kwargs),
    and each block returns a single tensor.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    traincast = (functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)
                 if torch.cuda.is_available() else nullcontext)

    blocks = _vit_blocks(model)
    embed_dim = model.embed_dim

    # Move embedding stage to device for input capture
    orig_device = next(model.parameters()).device
    model.to(device)
    fp_inps, actual_nsamples = _capture_block0_inputs(
        model, dataloader, device, nsamples, embed_dim)
    # Free everything except blocks back to CPU to save memory
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fp_inps = fp_inps.float()
    fp_outs = torch.zeros_like(fp_inps)
    quant_inps = fp_inps.clone() if propagate_quant_inputs else fp_inps
    loss_func = nn.MSELoss()

    num_blocks = len(blocks)
    for i in tqdm(range(num_blocks), desc="FlatQuant blocks (ViT)", unit="blk"):
        logger.info(f"===== FlatQuant ViT calibration: block {i}/{num_blocks - 1} =====")
        blk = blocks[i].to(device)

        dtype_dict = {n: p.dtype for n, p in blk.named_parameters()}
        blk.float()

        # (a) FP reference outputs
        blk.attn._ori_mode = True
        blk.mlp._ori_mode = True
        with torch.no_grad():
            for j in range(actual_nsamples):
                fp_outs[j] = blk(fp_inps[j].unsqueeze(0)).squeeze(0)
        blk.attn._ori_mode = False
        blk.mlp._ori_mode = False

        if torch.isnan(fp_outs[:actual_nsamples]).any() or torch.isinf(fp_outs[:actual_nsamples]).any():
            logger.warning(f"block {i}: FP reference has NaN/Inf — training may be unstable")

        # (b) Diagonal init from activation/weight stats
        if add_diag:
            if diag_attn:
                blk.attn.init_diag_scale(alpha=diag_alpha)
            if diag_mlp:
                blk.mlp.init_diag_scale(alpha=diag_alpha)

        blk = blk.to(device)
        for p in blk.parameters():
            p.requires_grad = False

        # (c) Trainable parameters
        trained = [{
            "params": _get_params_by_pattern(
                blk, ["trans.u_", "trans.v_", "trans.diag_left", "trans.diag_right"]),
            "lr": flat_lr,
        }]
        if add_diag:
            diag_params = []
            for n, p in blk.named_parameters():
                if "diag_scale" not in n:
                    continue
                is_attn = "attn.ln_trans" in n
                is_mlp = "mlp.in_trans" in n or "mlp.mid_trans" in n
                if (is_attn and diag_attn) or (is_mlp and diag_mlp):
                    p.requires_grad_(True)
                    diag_params.append(p)
            if diag_params:
                trained.append({"params": diag_params, "lr": flat_lr})
        if lwc:
            trained.append({"params": _get_params_by_pattern(blk, ["clip_factor_w"]),
                            "lr": flat_lr * 10})
        if lac:
            trained.append({"params": _get_params_by_pattern(blk, ["clip_factor_a"]),
                            "lr": flat_lr * 10})

        optimizer = torch.optim.AdamW(trained)
        n_batches = max(1, actual_nsamples // cali_bsz)
        total_steps = epochs * n_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=flat_lr * 1e-3)

        # (d) MSE training
        for epoch in range(epochs):
            epoch_mse, nan_count = 0.0, 0
            for j in range(n_batches):
                idx = j * cali_bsz
                fp_ref = fp_outs[idx:idx + cali_bsz]
                with traincast():
                    if propagate_quant_inputs and 0.0 < propagate_quant_alpha < 1.0:
                        out_fp = blk(fp_inps[idx:idx + cali_bsz])
                        out_q = blk(quant_inps[idx:idx + cali_bsz])
                        loss = ((1.0 - propagate_quant_alpha) * loss_func(fp_ref, out_fp)
                                + propagate_quant_alpha * loss_func(fp_ref, out_q))
                    else:
                        train_inp = (quant_inps[idx:idx + cali_bsz]
                                     if propagate_quant_inputs else fp_inps[idx:idx + cali_bsz])
                        loss = loss_func(fp_ref, blk(train_inp))

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    scheduler.step()
                    continue
                epoch_mse += loss.detach().item()
                # float32 backward with normalised loss to avoid fp16 overflow
                loss_f32 = loss.float()
                norm_loss = loss_f32 / loss_f32.clone().detach()
                optimizer.zero_grad()
                norm_loss.backward()
                all_params = [p for g in optimizer.param_groups for p in g["params"]]
                if any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                       for p in all_params):
                    nan_count += 1
                    optimizer.zero_grad()
                    scheduler.step()
                    continue
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                # Keep diagonals in a safe range
                with torch.no_grad():
                    for n, p in blk.named_parameters():
                        if "diag_left" in n or "diag_right" in n:
                            p.data.clamp_(min=0.1)
                        elif "diag_scale" in n:
                            p.data.clamp_(min=1e-4, max=10.0)
                scheduler.step()
            logger.info(f"  block {i} epoch {epoch}: mse={epoch_mse:.4e} "
                        f"ok={n_batches - nan_count}/{n_batches} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Propagate ADC-quantized outputs to next block's input
        if propagate_quant_inputs:
            with torch.no_grad():
                for j in range(n_batches):
                    idx = j * cali_bsz
                    quant_inps[idx:idx + cali_bsz] = blk(
                        quant_inps[idx:idx + cali_bsz]).detach().float()

        # Feed this block's FP output as next block's input
        fp_inps, fp_outs = fp_outs, fp_inps

        # Restore dtypes, move block to CPU
        for n, p in blk.named_parameters():
            p.requires_grad = False
            if n in dtype_dict:
                p.data = p.to(dtype_dict[n])
        blocks[i] = blk.cpu()
        del blk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del fp_inps, fp_outs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to(orig_device)
    logger.info("FlatQuant ViT calibration complete")
    return model
