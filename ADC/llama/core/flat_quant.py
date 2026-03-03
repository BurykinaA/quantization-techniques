"""
FlatQuant: Flatness Matters for LLM Quantization.

Native implementation of the FlatQuant algorithm for LLaMA models,
following the official upstream: https://github.com/ruikangliu/FlatQuant

Core idea: learn per-layer affine transformations (Kronecker-decomposed
orthogonal matrices + diagonal scaling + learnable clipping) that make
weights and activations flatter and more quantization-friendly. Transforms
are trained via MSE loss between FP and quantized layer outputs, then
reparameterized into the model weights before downstream use.

Reference: Sun et al., "FlatQuant: Flatness Matters for LLM Quantization",
ICML 2025.
"""

import functools
import gc
import logging
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =========================================================================
# Quantizer utilities (used during FlatQuant calibration only)
# =========================================================================

class _WeightQuantizer(nn.Module):
    """Symmetric per-channel weight quantizer for FlatQuant training."""

    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.maxq = 2 ** (bits - 1) - 1
        self.scale: torch.Tensor | None = None

    def find_params(self, weight: torch.Tensor) -> None:
        w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        self.scale = w_max / self.maxq

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            self.find_params(weight)
        return (
            (weight / self.scale).round().clamp(-self.maxq - 1, self.maxq) * self.scale
        )


class _ActivationQuantizer(nn.Module):
    """Symmetric per-token activation quantizer for FlatQuant training.

    Optionally supports learnable activation clipping (lac).
    """

    def __init__(self, bits: int = 8, lac: bool = False):
        super().__init__()
        self.bits = bits
        self.maxq = 2 ** (bits - 1) - 1
        self.lac = lac
        if lac:
            self.clip_factor = nn.Parameter(torch.tensor(4.0), requires_grad=True)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.lac:
            x_max = x_max * self.sigmoid(self.clip_factor)
        scale = x_max / self.maxq
        return (x / scale).round().clamp(-self.maxq - 1, self.maxq) * scale


# =========================================================================
# Kronecker-decomposed orthogonal transform (SVD parametrization)
# =========================================================================

def _kronecker_matmul(
    x: torch.Tensor, mat_l: torch.Tensor, mat_r: torch.Tensor,
) -> torch.Tensor:
    """Efficient Kronecker product matmul: x @ kron(mat_l, mat_r)."""
    init_shape = x.shape
    x = x.reshape(-1, mat_l.shape[0], mat_r.shape[0])
    x = torch.matmul(x, mat_r)
    x = torch.matmul(mat_l.T, x)
    return x.reshape(init_shape)


def _get_decompose_dim(n: int) -> tuple[int, int]:
    """Find (a-b, a+b) such that (a-b)*(a+b) == n, for Kronecker split."""
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


def _random_orthogonal(size: int) -> torch.Tensor:
    """Random orthogonal matrix via QR decomposition."""
    h = torch.randn(size, size)
    q, r = torch.linalg.qr(h)
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    return q


class KroneckerTransform(nn.Module):
    """Kronecker-decomposed SVD orthogonal transform with optional diagonal scaling.

    Represents T = diag(d) @ kron(U_L @ diag(s_L) @ V_L^T, U_R @ diag(s_R) @ V_R^T)
    where U, V are constrained to be orthogonal via Cayley parametrization.
    """

    def __init__(self, dim: int, add_diag: bool = False):
        super().__init__()
        left_size, right_size = _get_decompose_dim(dim)
        self.left_size = left_size
        self.right_size = right_size
        self.dim = dim

        self.u_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.u_left.weight.data = _random_orthogonal(left_size)
        self.u_left = nn.utils.parametrizations.orthogonal(
            self.u_left, orthogonal_map="cayley", use_trivialization=False,
        )
        self.v_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.v_left.weight.data = _random_orthogonal(left_size)
        self.v_left = nn.utils.parametrizations.orthogonal(
            self.v_left, orthogonal_map="cayley", use_trivialization=False,
        )
        self.diag_left = nn.Parameter(torch.ones(left_size, dtype=torch.float32))

        self.u_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.u_right.weight.data = _random_orthogonal(right_size)
        self.u_right = nn.utils.parametrizations.orthogonal(
            self.u_right, orthogonal_map="cayley", use_trivialization=False,
        )
        self.v_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.v_right.weight.data = _random_orthogonal(right_size)
        self.v_right = nn.utils.parametrizations.orthogonal(
            self.v_right, orthogonal_map="cayley", use_trivialization=False,
        )
        self.diag_right = nn.Parameter(torch.ones(right_size, dtype=torch.float32))

        self.add_diag = add_diag
        self.use_diag = True
        if add_diag:
            self.diag_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        self._eval_mode = False

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        if self.add_diag and self.use_diag:
            if inv_t:
                x = x / self.diag_scale.to(x)
            else:
                x = x * self.diag_scale.to(x)

        if not self._eval_mode:
            dl = self.diag_left
            dr = self.diag_right
            if inv_t:
                dl = 1.0 / dl
                dr = 1.0 / dr
            mat_l = self.u_left.weight @ torch.diag(dl) @ self.v_left.weight.T
            mat_r = self.u_right.weight @ torch.diag(dr) @ self.v_right.weight.T
        else:
            if inv_t:
                mat_l, mat_r = self.matrix_left_inv, self.matrix_right_inv
            else:
                mat_l, mat_r = self.matrix_left, self.matrix_right

        return _kronecker_matmul(x, mat_l.to(x), mat_r.to(x))

    def to_eval_mode(self) -> None:
        if self._eval_mode:
            return
        with torch.no_grad():
            mat_l = self.u_left.weight @ torch.diag(self.diag_left) @ self.v_left.weight.T
            mat_r = self.u_right.weight @ torch.diag(self.diag_right) @ self.v_right.weight.T
            mat_l_inv = self.u_left.weight @ torch.diag(1.0 / self.diag_left) @ self.v_left.weight.T
            mat_r_inv = self.u_right.weight @ torch.diag(1.0 / self.diag_right) @ self.v_right.weight.T
        self.matrix_left = nn.Parameter(mat_l, requires_grad=False)
        self.matrix_right = nn.Parameter(mat_r, requires_grad=False)
        self.matrix_left_inv = nn.Parameter(mat_l_inv, requires_grad=False)
        self.matrix_right_inv = nn.Parameter(mat_r_inv, requires_grad=False)
        del self.u_left, self.v_left, self.diag_left
        del self.u_right, self.v_right, self.diag_right
        self._eval_mode = True


# =========================================================================
# FlatQuantized Linear layer
# =========================================================================

class FlatQuantLinear(nn.Module):
    """Linear layer with FlatQuant transform, quantizers and learnable clipping."""

    def __init__(
        self,
        linear: nn.Linear,
        w_bits: int = 8,
        a_bits: int = 8,
        lwc: bool = False,
        lac: bool = False,
    ):
        super().__init__()
        self.linear = linear
        self.w_quantizer = _WeightQuantizer(bits=w_bits)
        self.a_quantizer = _ActivationQuantizer(bits=a_bits, lac=lac)
        self.lwc = lwc
        if lwc:
            out_features = linear.weight.shape[0]
            self.clip_factor_w_max = nn.Parameter(
                torch.full((out_features, 1), 4.0), requires_grad=True,
            )
            self.clip_factor_w_min = nn.Parameter(
                torch.full((out_features, 1), 4.0), requires_grad=True,
            )
            self.sigmoid = nn.Sigmoid()

    def _apply_wclip(self, weight: torch.Tensor) -> torch.Tensor:
        wmin = weight.min(dim=1, keepdim=True).values
        wmax = weight.max(dim=1, keepdim=True).values
        wmax = wmax * self.sigmoid(self.clip_factor_w_max)
        wmin = wmin * self.sigmoid(self.clip_factor_w_min)
        return torch.clamp(weight, min=wmin, max=wmax)

    def _apply_trans(
        self, weight: torch.Tensor, qa_trans: KroneckerTransform,
    ) -> torch.Tensor:
        return qa_trans(weight, inv_t=True)

    def ori_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def train_forward(
        self,
        x: torch.Tensor,
        qa_trans: KroneckerTransform | None = None,
    ) -> torch.Tensor:
        weight = self.linear.weight.data
        if qa_trans is not None:
            weight = self._apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self._apply_wclip(weight)
        self.w_quantizer.find_params(weight)
        weight = self.w_quantizer(weight)
        x = self.a_quantizer(x)
        return F.linear(x, weight, self.linear.bias)

    def reparameterize(
        self, qa_trans: KroneckerTransform | None = None,
    ) -> None:
        """Fold transform + clipping into weights permanently."""
        weight = self.linear.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)
        if qa_trans is not None:
            weight = self._apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self._apply_wclip(weight)
        self.linear.weight.data = weight.to(ori_dtype)


# =========================================================================
# FlatQuant LLaMA modules
# =========================================================================

class FlatQuantLlamaMLP(nn.Module):
    """LLaMA MLP wrapped with FlatQuant transforms."""

    def __init__(
        self, mlp: nn.Module, w_bits: int, a_bits: int,
        add_diag: bool, lwc: bool, lac: bool,
    ):
        super().__init__()
        self.act_fn = mlp.act_fn

        self.gate_proj = FlatQuantLinear(mlp.gate_proj, w_bits, a_bits, lwc, lac)
        self.up_proj = FlatQuantLinear(mlp.up_proj, w_bits, a_bits, lwc, lac)
        self.down_proj = FlatQuantLinear(mlp.down_proj, w_bits, a_bits, lwc, lac)

        in_dim = mlp.up_proj.weight.shape[1]
        intermediate_dim = mlp.down_proj.weight.shape[1]

        self.up_gate_trans = KroneckerTransform(in_dim, add_diag=add_diag)
        self.down_trans = KroneckerTransform(intermediate_dim, add_diag=add_diag)

        self._ori_mode = False
        self._collect_smax = add_diag
        if self._collect_smax:
            self._up_smax = torch.ones(in_dim, device="cpu") * 1e-5
            self._down_smax = torch.ones(intermediate_dim, device="cpu") * 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def _ori_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._collect_smax and hasattr(self, "_up_smax"):
            self._up_smax = torch.maximum(
                self._up_smax.to(x.device),
                x.reshape(-1, x.shape[-1]).abs().amax(dim=0).detach(),
            )
        gate = self.act_fn(self.gate_proj.ori_forward(x))
        up = self.up_proj.ori_forward(x)
        intermediate = gate * up
        if self._collect_smax and hasattr(self, "_down_smax"):
            self._down_smax = torch.maximum(
                self._down_smax.to(intermediate.device),
                intermediate.reshape(-1, intermediate.shape[-1]).abs().amax(dim=0).detach(),
            )
        return self.down_proj.ori_forward(intermediate)

    def _trans_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ts = self.up_gate_trans(x)
        gate = self.act_fn(self.gate_proj.train_forward(x_ts, qa_trans=self.up_gate_trans))
        up = self.up_proj.train_forward(x_ts, qa_trans=self.up_gate_trans)
        intermediate = gate * up
        x_ts2 = self.down_trans(intermediate)
        return self.down_proj.train_forward(x_ts2, qa_trans=self.down_trans)

    def init_diag_scale(self, alpha: float = 0.5) -> None:
        if not hasattr(self, "_up_smax"):
            return
        up_w = torch.cat([
            self.up_proj.linear.weight, self.gate_proj.linear.weight,
        ], dim=0).abs().amax(dim=0)
        down_w = self.down_proj.linear.weight.abs().amax(dim=0)
        eps = 1e-5
        self.up_gate_trans.diag_scale.data = (
            up_w.pow(1 - alpha) / self._up_smax.to(up_w.device).pow(alpha)
        ).clamp(min=eps)
        self.down_trans.diag_scale.data = (
            down_w.pow(1 - alpha) / self._down_smax.to(down_w.device).pow(alpha)
        ).clamp(min=eps)
        del self._up_smax, self._down_smax
        self._collect_smax = False

    def reparameterize(self) -> None:
        self.up_gate_trans.to_eval_mode()
        self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        self.up_gate_trans.use_diag = False
        if self.down_trans.add_diag:
            w = self.up_proj.linear.weight
            ori_dtype = w.dtype
            w = w.to(torch.float64).T.mul(
                self.down_trans.diag_scale.to(torch.float64),
            ).T
            self.up_proj.linear.weight.data = w.to(ori_dtype)
            self.down_trans.use_diag = False


class FlatQuantLlamaAttention(nn.Module):
    """LLaMA self-attention wrapped with FlatQuant transforms.

    Only handles the linear projections and their transforms; the rest of the
    attention math is delegated to the original module during layer-level
    calibration.
    """

    def __init__(
        self, attn: nn.Module, w_bits: int, a_bits: int,
        add_diag: bool, lwc: bool, lac: bool,
    ):
        super().__init__()
        self._orig_attn = attn
        in_dim = attn.q_proj.weight.shape[1]

        self.q_proj = FlatQuantLinear(attn.q_proj, w_bits, a_bits, lwc, lac)
        self.k_proj = FlatQuantLinear(attn.k_proj, w_bits, a_bits, lwc, lac)
        self.v_proj = FlatQuantLinear(attn.v_proj, w_bits, a_bits, lwc, lac)
        self.o_proj = FlatQuantLinear(attn.o_proj, w_bits, a_bits, lwc, lac)

        self.ln_trans = KroneckerTransform(in_dim, add_diag=add_diag)

        self._ori_mode = False
        self._collect_smax = add_diag
        if self._collect_smax:
            self._ln_smax = torch.ones(in_dim, device="cpu") * 1e-5

    def forward_after_ln(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._ori_mode:
            return self._ori_forward_after_ln(hidden_states)
        return self._trans_forward_after_ln(hidden_states)

    def _ori_forward_after_ln(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._collect_smax and hasattr(self, "_ln_smax"):
            self._ln_smax = torch.maximum(
                self._ln_smax.to(hidden_states.device),
                hidden_states.reshape(-1, hidden_states.shape[-1]).abs().amax(dim=0).detach(),
            )
        q = self.q_proj.ori_forward(hidden_states)
        k = self.k_proj.ori_forward(hidden_states)
        v = self.v_proj.ori_forward(hidden_states)
        return q, k, v

    def _trans_forward_after_ln(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.ln_trans(hidden_states)
        q = self.q_proj.train_forward(h, qa_trans=self.ln_trans)
        k = self.k_proj.train_forward(h, qa_trans=self.ln_trans)
        v = self.v_proj.train_forward(h, qa_trans=self.ln_trans)
        return q, k, v

    def init_diag_scale(self, alpha: float = 0.5) -> None:
        if not hasattr(self, "_ln_smax"):
            return
        qkv_w = torch.cat([
            self.q_proj.linear.weight,
            self.k_proj.linear.weight,
            self.v_proj.linear.weight,
        ], dim=0).abs().amax(dim=0)
        eps = 1e-5
        self.ln_trans.diag_scale.data = (
            qkv_w.pow(1 - alpha) / self._ln_smax.to(qkv_w.device).pow(alpha)
        ).clamp(min=eps)
        del self._ln_smax
        self._collect_smax = False

    def reparameterize(self) -> None:
        self.ln_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        self.v_proj.reparameterize(qa_trans=self.ln_trans)
        self.o_proj.reparameterize()


# =========================================================================
# Apply / strip FlatQuant wrappers
# =========================================================================

def apply_flatquant_to_model(
    model: nn.Module,
    w_bits: int = 8,
    a_bits: int = 8,
    add_diag: bool = True,
    lwc: bool = True,
    lac: bool = True,
) -> nn.Module:
    """Replace LLaMA attention and MLP with FlatQuant-wrapped versions."""
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Applying FlatQuant wrappers"):
        layer = layers[i]
        layers[i].self_attn = FlatQuantLlamaAttention(
            layer.self_attn, w_bits, a_bits, add_diag, lwc, lac,
        )
        layers[i].mlp = FlatQuantLlamaMLP(
            layer.mlp, w_bits, a_bits, add_diag, lwc, lac,
        )
    return model


def reparameterize_model(model: nn.Module) -> nn.Module:
    """Fold trained FlatQuant transforms into model weights and LayerNorm."""
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        if isinstance(attn, FlatQuantLlamaAttention):
            attn.reparameterize()
            if attn.ln_trans.add_diag:
                _reparameterize_ln(layer.input_layernorm, attn.ln_trans)
        if isinstance(mlp, FlatQuantLlamaMLP):
            mlp.reparameterize()
            if mlp.up_gate_trans.add_diag:
                _reparameterize_ln(layer.post_attention_layernorm, mlp.up_gate_trans)
    return model


def _reparameterize_ln(ln: nn.Module, trans: KroneckerTransform) -> None:
    """Absorb diagonal scaling into LayerNorm weight."""
    w = ln.weight.data
    ori_dtype = w.dtype
    ln.weight.data = (w.to(torch.float64) * trans.diag_scale.to(torch.float64)).to(ori_dtype)
    trans.use_diag = False


def strip_flatquant_wrappers(model: nn.Module) -> nn.Module:
    """Replace FlatQuant wrapper modules back with plain nn.Linear layers.

    Call after reparameterize_model so the learned transforms are already
    folded into the weights.
    """
    for layer in model.model.layers:
        attn_wrapper = layer.self_attn
        if isinstance(attn_wrapper, FlatQuantLlamaAttention):
            orig = attn_wrapper._orig_attn
            orig.q_proj = attn_wrapper.q_proj.linear
            orig.k_proj = attn_wrapper.k_proj.linear
            orig.v_proj = attn_wrapper.v_proj.linear
            orig.o_proj = attn_wrapper.o_proj.linear
            layer.self_attn = orig

        mlp_wrapper = layer.mlp
        if isinstance(mlp_wrapper, FlatQuantLlamaMLP):
            from types import SimpleNamespace
            orig_mlp = SimpleNamespace()
            for attr in dir(layer.mlp):
                if not attr.startswith("_"):
                    try:
                        setattr(orig_mlp, attr, getattr(layer.mlp, attr))
                    except Exception:
                        pass

            parent_class = type(layer).mlp.fget.__class__ if hasattr(type(layer).mlp, 'fget') else None

            gate = mlp_wrapper.gate_proj.linear
            up = mlp_wrapper.up_proj.linear
            down = mlp_wrapper.down_proj.linear
            act_fn = mlp_wrapper.act_fn

            from transformers.models.llama.modeling_llama import LlamaMLP
            new_mlp = object.__new__(LlamaMLP)
            nn.Module.__init__(new_mlp)
            new_mlp.gate_proj = gate
            new_mlp.up_proj = up
            new_mlp.down_proj = down
            new_mlp.act_fn = act_fn
            layer.mlp = new_mlp

    return model


# =========================================================================
# Layer-by-layer MSE calibration (core FlatQuant training)
# =========================================================================

def calibrate_flat_quant(
    model: nn.Module,
    dataloader,
    device: torch.device,
    nsamples: int = 128,
    cali_bsz: int = 4,
    epochs: int = 15,
    flat_lr: float = 5e-3,
    diag_alpha: float = 0.5,
    add_diag: bool = True,
    lwc: bool = True,
    lac: bool = True,
) -> nn.Module:
    """Train FlatQuant transforms layer-by-layer using MSE loss.

    This follows the official FlatQuant `cali_flat_quant` procedure:
    1. Capture calibration inputs to each layer
    2. For each layer: compute FP outputs, then train transforms to minimize
       MSE between FP and quantized outputs
    3. Use trained layer output as input to the next layer
    """
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad = False

    dtype = torch.float16
    traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)
    if not torch.cuda.is_available():
        dtype = torch.float32
        traincast = nullcontext

    layers = model.model.layers
    hidden_size = model.config.hidden_size

    layers[0] = layers[0].to(device)
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    # Capture inputs to first layer
    inps = torch.zeros((nsamples, model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048, hidden_size), dtype=dtype, device=device)

    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class _Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if cache["i"] < nsamples:
                seq_len = inp.shape[1]
                inps[cache["i"], :seq_len, :] = inp[0]
                cache["i"] += 1
                if cache["attention_mask"] is None:
                    cache["attention_mask"] = kwargs.get("attention_mask")
                    cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError("catch")

    layers[0] = _Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= nsamples:
                break
            try:
                ids = batch[0] if isinstance(batch, (list, tuple)) else batch["input_ids"]
                model(ids.to(device))
            except ValueError:
                pass

    actual_nsamples = cache["i"]
    logger.info(f"Captured {actual_nsamples} calibration samples")
    inps = inps[:actual_nsamples]

    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    fp_inps = inps
    fp_outs = torch.zeros_like(inps)
    loss_func = nn.MSELoss()

    num_layers = len(layers)
    for i in range(num_layers):
        logger.info(f"========= FlatQuant calibration: Layer {i}/{num_layers - 1} =========")
        layer = layers[i].to(device)

        dtype_dict = {}
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        # Compute FP reference outputs
        layer.self_attn._ori_mode = True
        layer.mlp._ori_mode = True
        with torch.no_grad():
            for j in range(actual_nsamples):
                fp_outs[j] = layer(
                    fp_inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        layer.self_attn._ori_mode = False
        layer.mlp._ori_mode = False

        # Initialize diagonal scales from activation/weight statistics
        if add_diag:
            layer.self_attn.init_diag_scale(alpha=diag_alpha)
            layer.mlp.init_diag_scale(alpha=diag_alpha)

        layer = layer.to(device)

        # Set up trainable parameters
        for param in layer.parameters():
            param.requires_grad = False

        trained_params = []
        trained_params.append({
            "params": _get_params_by_pattern(layer, ["trans.u_", "trans.v_", "trans.diag_left", "trans.diag_right"]),
            "lr": flat_lr,
        })
        if add_diag:
            trained_params.append({
                "params": _get_params_by_pattern(layer, ["trans.diag_scale"]),
                "lr": flat_lr,
            })
        if lwc:
            trained_params.append({
                "params": _get_params_by_pattern(layer, ["clip_factor_w"]),
                "lr": flat_lr * 10,
            })
        if lac:
            trained_params.append({
                "params": _get_params_by_pattern(layer, ["clip_factor_a"]),
                "lr": flat_lr * 10,
            })

        optimizer = torch.optim.AdamW(trained_params)
        total_steps = epochs * (actual_nsamples // cali_bsz)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=flat_lr * 1e-3,
        )

        for epoch in range(epochs):
            epoch_mse = 0.0
            with traincast():
                for j in range(actual_nsamples // cali_bsz):
                    idx = j * cali_bsz
                    quant_out = layer(
                        fp_inps[idx:idx + cali_bsz],
                        attention_mask=attention_mask_batch,
                        position_ids=position_ids,
                    )[0]
                    loss = loss_func(fp_outs[idx:idx + cali_bsz], quant_out)
                    epoch_mse += loss.detach().item()
                    normalized_loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    normalized_loss.backward()
                    optimizer.step()
                    scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"  layer {i} epoch {epoch}, lr={lr:.8f}, mse={epoch_mse:.8f}")

        # Use quantized output as next layer's input
        fp_inps, fp_outs = fp_outs, fp_inps

        # Move layer back to CPU and clean up
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict:
                param.data = param.to(dtype_dict[name])
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model.config.use_cache = use_cache
    logger.info("FlatQuant calibration complete")
    return model


def _get_params_by_pattern(
    module: nn.Module, patterns: list[str],
) -> list[nn.Parameter]:
    """Collect parameters whose names match any of the given patterns."""
    params = []
    for name, param in module.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = True
            params.append(param)
    return params


# =========================================================================
# Save / load transforms
# =========================================================================

def save_flat_transforms(model: nn.Module, path: str) -> None:
    """Save FlatQuant transform state dicts for all layers."""
    transforms = {}
    for i, layer in enumerate(model.model.layers):
        state = {}
        if isinstance(layer.self_attn, FlatQuantLlamaAttention):
            state["attn_ln_trans"] = layer.self_attn.ln_trans.state_dict()
        if isinstance(layer.mlp, FlatQuantLlamaMLP):
            state["mlp_up_gate_trans"] = layer.mlp.up_gate_trans.state_dict()
            state["mlp_down_trans"] = layer.mlp.down_trans.state_dict()
        if state:
            transforms[i] = state
    torch.save(transforms, path)
    logger.info(f"Saved FlatQuant transforms to {path}")


def load_flat_transforms(model: nn.Module, path: str) -> nn.Module:
    """Load pre-trained FlatQuant transforms into an already-wrapped model."""
    transforms = torch.load(path, map_location="cpu")
    for i, state in transforms.items():
        layer = model.model.layers[i]
        if "attn_ln_trans" in state and isinstance(layer.self_attn, FlatQuantLlamaAttention):
            layer.self_attn.ln_trans.load_state_dict(state["attn_ln_trans"])
        if "mlp_up_gate_trans" in state and isinstance(layer.mlp, FlatQuantLlamaMLP):
            layer.mlp.up_gate_trans.load_state_dict(state["mlp_up_gate_trans"])
        if "mlp_down_trans" in state:
            layer.mlp.down_trans.load_state_dict(state["mlp_down_trans"])
    logger.info(f"Loaded FlatQuant transforms from {path}")
    return model
