"""
FlatQuant: Flatness-aware Post-Training Quantization for LLMs
==============================================================

Implementation based on the official FlatQuant repository:
  https://github.com/ruikangliu/FlatQuant

Algorithm overview
------------------
FlatQuant learns per-layer affine transformations that make weight and
activation distributions "flatter" (more uniform), reducing quantization
error.  Each transformation is a Kronecker-decomposed orthogonal matrix
with optional diagonal scaling and learnable clipping.

Pipeline (in order):
  1. apply_flatquant_to_model   — wrap attention / MLP with FlatQuant modules
  2. calibrate_flat_quant       — layer-by-layer MSE training of transforms
  3. reparameterize_model       — fold transforms into weights / LayerNorm
  4. strip_flatquant_wrappers   — restore plain nn.Linear for downstream use

During calibration each layer is trained independently:
  • FP reference output is computed with the original (unwrapped) weights
  • Quantized output is computed with transforms + fake quantisation
  • MSE loss between the two is minimised to learn optimal transforms

File layout
-----------
  Part 1 — Fake Quantizers           (weight & activation)
  Part 2 — Kronecker Transform       (learnable orthogonal via SVD)
  Part 3 — FlatQuantLinear           (linear + transform + quantisation)
  Part 4 — LLaMA Module Wrappers     (Attention, MLP)
  Part 5 — Model-Level Operations    (apply, calibrate, reparameterize, strip)
  Part 6 — Save / Load Transforms
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
# Part 1 — Fake Quantizers (used during FlatQuant calibration only)
# =========================================================================

class _WeightQuantizer(nn.Module):
    """Symmetric per-channel fake quantizer for weights.

    Quantizes and immediately dequantizes (fake-quant) to simulate
    quantization error during training while keeping gradients flowing.
    """

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
    """Symmetric per-token fake quantizer for activations.

    Optionally supports Learnable Activation Clipping (LAC), where a
    sigmoid-gated clip factor is trained to find optimal clipping bounds.
    """

    def __init__(self, bits: int = 8, lac: bool = False):
        super().__init__()
        self.bits = bits
        self.maxq = 2 ** (bits - 1) - 1
        self.lac = lac
        if lac:
            self.clip_factor = nn.Parameter(torch.tensor(4.0))
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.lac:
            x_max = x_max * self.sigmoid(self.clip_factor)
        scale = x_max / self.maxq
        return (x / scale).round().clamp(-self.maxq - 1, self.maxq) * scale


# =========================================================================
# Part 2 — Kronecker-decomposed SVD Transform
# =========================================================================
#
# Core FlatQuant transformation.
#
#   Forward:  T(x) = (x * diag_scale) @ kron(L, R)
#   Inverse:  T⁻¹(w) = (w / diag_scale) @ kron(L⁻¹, R⁻¹)
#
# L, R are factored via SVD:  L = U_L @ diag(s_L) @ V_L^T
# U, V are constrained orthogonal through the Cayley parametrization.
#
# Key property: without quantization the transforms cancel:
#   T(x) @ T⁻¹(W)^T  =  x @ W^T
# With quantization the error is reduced because both x and W have
# "flatter" distributions after transformation.
# =========================================================================


def _kronecker_matmul(
    x: torch.Tensor, mat_l: torch.Tensor, mat_r: torch.Tensor,
) -> torch.Tensor:
    """Efficient Kronecker product matmul:  x @ kron(mat_l, mat_r).

    Avoids forming the full Kronecker product by exploiting its structure:
    reshape x to 3-D, apply mat_r (right), then mat_l (left).
    """
    shape = x.shape
    x = x.reshape(-1, mat_l.shape[0], mat_r.shape[0])
    x = torch.matmul(x, mat_r)
    x = torch.matmul(mat_l.T, x)
    return x.reshape(shape)


def _get_decompose_dim(n: int) -> tuple[int, int]:
    """Factor *n* into (p, q) with p*q == n and p ≈ q.

    Uses the identity  n = (a − b)(a + b)  where  a² − b² = n.
    """
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
    """Random orthogonal matrix via QR decomposition (Haar measure).

    Computed in float64 for numerical precision (matching upstream which
    uses numpy/scipy), then converted to float32.
    """
    h = torch.randn(size, size, dtype=torch.float64)
    q, r = torch.linalg.qr(h)
    q = q @ torch.diag(torch.sign(torch.diag(r)))
    return q.float()


class KroneckerTransform(nn.Module):
    """Learnable Kronecker-decomposed transform with optional diagonal.

    Parameters
    ----------
    dim : int
        Input dimension (automatically factored into left × right).
    add_diag : bool
        Whether to include a learnable per-channel diagonal scale.
    """

    def __init__(self, dim: int, add_diag: bool = False):
        super().__init__()
        left_size, right_size = _get_decompose_dim(dim)
        self.left_size = left_size
        self.right_size = right_size
        self.dim = dim

        # Left Kronecker factor:  U_L @ diag(s_L) @ V_L^T
        # Using matrix_exp (not cayley) for the orthogonal map to avoid
        # singularity issues in torch.linalg.solve that cayley can hit.
        # use_trivialization=True ensures the initial weight IS the random
        # orthogonal matrix we set (Q₀ @ exp(0) = Q₀) and that small
        # parameter updates ≈ small rotations from Q₀.
        self.u_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.u_left.weight.data = _random_orthogonal(left_size)
        self.u_left = nn.utils.parametrizations.orthogonal(
            self.u_left, orthogonal_map="matrix_exp", use_trivialization=True,
        )
        self.v_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.v_left.weight.data = _random_orthogonal(left_size)
        self.v_left = nn.utils.parametrizations.orthogonal(
            self.v_left, orthogonal_map="matrix_exp", use_trivialization=True,
        )
        self.diag_left = nn.Parameter(torch.ones(left_size, dtype=torch.float32))

        # Right Kronecker factor:  U_R @ diag(s_R) @ V_R^T
        self.u_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.u_right.weight.data = _random_orthogonal(right_size)
        self.u_right = nn.utils.parametrizations.orthogonal(
            self.u_right, orthogonal_map="matrix_exp", use_trivialization=True,
        )
        self.v_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.v_right.weight.data = _random_orthogonal(right_size)
        self.v_right = nn.utils.parametrizations.orthogonal(
            self.v_right, orthogonal_map="matrix_exp", use_trivialization=True,
        )
        self.diag_right = nn.Parameter(torch.ones(right_size, dtype=torch.float32))

        # Optional per-channel diagonal scaling
        self.add_diag = add_diag
        self.use_diag = True
        if add_diag:
            self.diag_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        self._eval_mode = False

    def forward(self, x: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        """Apply forward (*inv_t=False*) or inverse (*inv_t=True*) transform.

        All computation is done in float32 to prevent overflow/NaN in float16
        AMP contexts (diag_scale can have large values after init_diag_scale,
        which would overflow float16 max ~65504).
        """
        orig_dtype = x.dtype
        x = x.float()

        if self.add_diag and self.use_diag:
            x = x / self.diag_scale if inv_t else x * self.diag_scale

        if not self._eval_mode:
            dl = 1.0 / self.diag_left if inv_t else self.diag_left
            dr = 1.0 / self.diag_right if inv_t else self.diag_right
            mat_l = self.u_left.weight @ torch.diag(dl) @ self.v_left.weight.T
            mat_r = self.u_right.weight @ torch.diag(dr) @ self.v_right.weight.T
        else:
            if inv_t:
                mat_l, mat_r = self.matrix_left_inv, self.matrix_right_inv
            else:
                mat_l, mat_r = self.matrix_left, self.matrix_right

        return _kronecker_matmul(x, mat_l, mat_r).to(orig_dtype)

    def to_eval_mode(self) -> None:
        """Pre-compute and cache forward / inverse matrices, free SVD params."""
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
# Part 3 — FlatQuantLinear
# =========================================================================

class FlatQuantLinear(nn.Module):
    """Linear layer augmented with FlatQuant transform and fake quantisation.

    Wraps an existing ``nn.Linear``.  During calibration::

        W' = T⁻¹(W)              # inverse transform on weights
        W' = clip(W')             # optional LWC
        W_q = fake_quant(W')      # weight quantisation
        x_q = fake_quant(x)       # activation quantisation
        y   = x_q @ W_q^T + bias

    After ``reparameterize()``, the transform is folded into weights
    permanently; only the activation quantiser stays active at eval time.
    """

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

    def ori_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Original FP forward — no transform, no quantisation."""
        return self.linear(x)

    def train_forward(
        self,
        x: torch.Tensor,
        qa_trans: KroneckerTransform | None = None,
    ) -> torch.Tensor:
        """Quantised forward with optional transform (calibration path)."""
        weight = self.linear.weight.data
        if qa_trans is not None:
            weight = qa_trans(weight, inv_t=True)
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
            weight = qa_trans(weight, inv_t=True)
        if self.lwc:
            weight = self._apply_wclip(weight)
        self.linear.weight.data = weight.to(ori_dtype)


# =========================================================================
# Part 4 — LLaMA Module Wrappers
# =========================================================================
#
# We delegate the actual computation (RoPE, attention math, etc.) to the
# original HuggingFace modules, making this code compatible across
# different transformers versions.  Only the linear projections are
# intercepted for transform + quantisation.
# =========================================================================


class FlatQuantLlamaMLP(nn.Module):
    """LLaMA MLP wrapped with FlatQuant transforms.

    Transforms
    ----------
    up_gate_trans : shared by gate_proj and up_proj
    down_trans    : applied before down_proj

    Data flow during calibration (_ori_mode=False)::

        x → up_gate_trans(x) → gate_proj(x', T) → act_fn ─┐
                               → up_proj(x', T)  ──────────× → intermediate
        intermediate → down_trans(intermediate) → down_proj(x'', T) → output
    """

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
        """FP forward — collects activation statistics for diagonal init."""
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
        """Quantised forward with transforms."""
        x_ts = self.up_gate_trans(x)
        gate = self.act_fn(self.gate_proj.train_forward(x_ts, qa_trans=self.up_gate_trans))
        up = self.up_proj.train_forward(x_ts, qa_trans=self.up_gate_trans)
        intermediate = gate * up
        x_ts2 = self.down_trans(intermediate)
        return self.down_proj.train_forward(x_ts2, qa_trans=self.down_trans)

    def init_diag_scale(self, alpha: float = 0.5) -> None:
        """SmoothQuant-style diagonal init from activation / weight stats."""
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
        """Fold transforms into projection weights permanently."""
        self.up_gate_trans.to_eval_mode()
        self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        self.up_gate_trans.use_diag = False
        # Absorb down_trans diagonal into up_proj weights so the
        # downstream intermediate tensor is already scaled.
        if self.down_trans.add_diag:
            w = self.up_proj.linear.weight
            ori_dtype = w.dtype
            w = w.to(torch.float64).T.mul(
                self.down_trans.diag_scale.to(torch.float64),
            ).T
            self.up_proj.linear.weight.data = w.to(ori_dtype)
            self.down_trans.use_diag = False


class _QuantProjectionWrapper(nn.Module):
    """Temporary drop-in for ``nn.Linear`` during attention calibration.

    Routes the forward call through ``FlatQuantLinear.train_forward``,
    applying both the inverse transform to weights and fake quantisation.
    Used to monkey-patch the original attention module's projections.
    """

    def __init__(
        self,
        fq_linear: FlatQuantLinear,
        qa_trans: KroneckerTransform | None = None,
    ):
        super().__init__()
        self._fq = fq_linear
        self._qa = qa_trans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fq.train_forward(x, qa_trans=self._qa)


class FlatQuantLlamaAttention(nn.Module):
    """LLaMA self-attention wrapped with FlatQuant transforms.

    Design: instead of re-implementing the full attention forward (which is
    HuggingFace-version-specific), we keep the original attention module
    and delegate to it.  Only the linear projections are intercepted.

    Transform
    ---------
    ln_trans : shared by q_proj, k_proj, v_proj (applied after LayerNorm)

    Modes
    -----
    _ori_mode = True  (FP reference)
        Collects activation statistics, delegates to ``_orig_attn`` as-is.

    _ori_mode = False (quantised training)
        1. Apply ``ln_trans`` to hidden_states (forward transform)
        2. Temporarily replace q/k/v/o projections with quantised wrappers
        3. Call ``_orig_attn.forward(...)``  (handles RoPE, masking, etc.)
        4. Restore original projections
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

    def forward(self, *args, **kwargs):
        """Dispatch to FP or quantised path based on ``_ori_mode``."""
        if self._ori_mode:
            return self._ori_forward(*args, **kwargs)
        return self._train_forward(*args, **kwargs)

    def _ori_forward(self, *args, **kwargs):
        """FP reference path: collect activation stats, delegate to original."""
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is not None and self._collect_smax and hasattr(self, "_ln_smax"):
            self._ln_smax = torch.maximum(
                self._ln_smax.to(hs.device),
                hs.reshape(-1, hs.shape[-1]).abs().amax(dim=0).detach(),
            )
        return self._orig_attn(*args, **kwargs)

    def _train_forward(self, *args, **kwargs):
        """Quantised training path via projection swapping."""
        # 1. Apply forward transform to hidden_states
        if args:
            args = (self.ln_trans(args[0]),) + args[1:]
        elif "hidden_states" in kwargs:
            kwargs["hidden_states"] = self.ln_trans(kwargs["hidden_states"])

        # 2. Swap projections → quantised wrappers, call, restore
        saved = (
            self._orig_attn.q_proj,
            self._orig_attn.k_proj,
            self._orig_attn.v_proj,
            self._orig_attn.o_proj,
        )
        try:
            self._orig_attn.q_proj = _QuantProjectionWrapper(self.q_proj, self.ln_trans)
            self._orig_attn.k_proj = _QuantProjectionWrapper(self.k_proj, self.ln_trans)
            self._orig_attn.v_proj = _QuantProjectionWrapper(self.v_proj, self.ln_trans)
            self._orig_attn.o_proj = _QuantProjectionWrapper(self.o_proj, None)
            return self._orig_attn(*args, **kwargs)
        finally:
            (
                self._orig_attn.q_proj,
                self._orig_attn.k_proj,
                self._orig_attn.v_proj,
                self._orig_attn.o_proj,
            ) = saved

    def init_diag_scale(self, alpha: float = 0.5) -> None:
        """SmoothQuant-style diagonal init from activation / weight stats."""
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
        """Fold transforms into projection weights permanently."""
        self.ln_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        self.v_proj.reparameterize(qa_trans=self.ln_trans)
        self.o_proj.reparameterize()


# =========================================================================
# Part 5 — Model-Level Operations
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
    """Fold all FlatQuant transforms into model weights and LayerNorm.

    After this the model still carries FlatQuant wrappers, but their
    transforms are effectively identity.  Call ``strip_flatquant_wrappers``
    next to remove the wrappers entirely.
    """
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
    """Absorb diagonal scaling from *trans* into LayerNorm weight."""
    w = ln.weight.data
    ori_dtype = w.dtype
    ln.weight.data = (w.to(torch.float64) * trans.diag_scale.to(torch.float64)).to(ori_dtype)
    trans.use_diag = False


def strip_flatquant_wrappers(model: nn.Module) -> nn.Module:
    """Remove FlatQuant wrappers, restoring plain ``nn.Linear`` layers.

    **Must** be called after ``reparameterize_model()`` so the learned
    transforms are already folded into weights.
    """
    for layer in model.model.layers:
        # --- Attention: put original attn back with reparameterised projs ---
        attn_wrapper = layer.self_attn
        if isinstance(attn_wrapper, FlatQuantLlamaAttention):
            orig = attn_wrapper._orig_attn
            orig.q_proj = attn_wrapper.q_proj.linear
            orig.k_proj = attn_wrapper.k_proj.linear
            orig.v_proj = attn_wrapper.v_proj.linear
            orig.o_proj = attn_wrapper.o_proj.linear
            layer.self_attn = orig

        # --- MLP: reconstruct a standard LlamaMLP with reparameterised projs ---
        mlp_wrapper = layer.mlp
        if isinstance(mlp_wrapper, FlatQuantLlamaMLP):
            from transformers.models.llama.modeling_llama import LlamaMLP
            new_mlp = object.__new__(LlamaMLP)
            nn.Module.__init__(new_mlp)
            new_mlp.gate_proj = mlp_wrapper.gate_proj.linear
            new_mlp.up_proj = mlp_wrapper.up_proj.linear
            new_mlp.down_proj = mlp_wrapper.down_proj.linear
            new_mlp.act_fn = mlp_wrapper.act_fn
            layer.mlp = new_mlp

    return model


# =========================================================================
# Part 5b — Layer-by-layer MSE Calibration  (core training loop)
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

    Follows the official ``cali_flat_quant`` procedure:

    1. Run calibration data through the embedding layer to capture
       first-layer inputs (hidden states + all kwargs like masks and
       position embeddings).
    2. For each decoder layer (sequentially):

       a. Compute FP reference outputs  (``_ori_mode = True``)
       b. Initialise diagonal scales from activation / weight statistics
       c. Minimise  ``MSE(FP_output, quantised_output)``  over *epochs*
       d. Feed this layer's trained output as the next layer's input

    Parameters
    ----------
    model       : LLaMA model with FlatQuant wrappers already applied.
    dataloader  : calibration data — iterable of (input_ids, …) tuples.
    device      : GPU device.
    nsamples    : number of calibration samples.
    cali_bsz    : mini-batch size for transform training.
    epochs      : training epochs per layer.
    flat_lr     : base learning rate for transform parameters.
    diag_alpha  : SmoothQuant-style α for diagonal initialisation.
    add_diag, lwc, lac : whether the respective features are enabled.
    """
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad = False

    # ── AMP setup ──
    dtype = torch.float16
    traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)
    if not torch.cuda.is_available():
        dtype = torch.float32
        traincast = nullcontext

    layers = model.model.layers
    hidden_size = model.config.hidden_size

    # Move embedding + first layer to device for input capture
    layers[0] = layers[0].to(device)
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    # ── Step 1: capture first-layer inputs ─────────────────────────
    #
    # We intercept the first decoder layer's forward to collect:
    #   • hidden-state inputs  (stored in `inps`)
    #   • all kwargs the decoder layer receives (attention_mask,
    #     position_embeddings, position_ids, etc.) so we can replay
    #     them during calibration.
    #
    # The inps tensor is allocated lazily on the first capture so its
    # sequence-length dimension matches the actual calibration data
    # (avoids the max_position_embeddings=131072 mismatch that would
    # break RoPE).

    inps = None  # shape will be (nsamples, actual_seqlen, hidden_size)
    cache = {"i": 0, "layer_kwargs": None}

    class _Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            nonlocal inps
            if cache["i"] < nsamples:
                if inps is None:
                    inps = torch.zeros(
                        (nsamples, inp.shape[1], hidden_size),
                        dtype=dtype, device=device,
                    )
                inps[cache["i"]] = inp[0]
                cache["i"] += 1
                if cache["layer_kwargs"] is None:
                    cache["layer_kwargs"] = kwargs
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

    # Build kwargs for single-sample and batched layer calls.
    # The Catcher may have captured kwargs from a multi-sample batch
    # (depending on dataloader batch_size), so we normalise everything
    # to batch=1 first.  position_embeddings / position_ids broadcast
    # naturally from batch=1; only attention_mask needs explicit
    # expansion for the batched training calls.
    layer_kwargs = cache["layer_kwargs"] or {}

    for key in list(layer_kwargs.keys()):
        val = layer_kwargs[key]
        if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] > 1:
            layer_kwargs[key] = val[:1]
        elif isinstance(val, tuple):
            layer_kwargs[key] = tuple(
                v[:1] if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] > 1 else v
                for v in val
            )

    batch_kwargs = dict(layer_kwargs)
    attention_mask = layer_kwargs.get("attention_mask")
    if attention_mask is not None:
        batch_kwargs["attention_mask"] = attention_mask.expand(cali_bsz, -1, -1, -1)

    # Free GPU memory from embedding / rotary
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 2: layer-by-layer calibration ─────────────────────────
    fp_inps = inps.float()
    fp_outs = torch.zeros_like(fp_inps)
    loss_func = nn.MSELoss()

    num_layers = len(layers)
    for i in range(num_layers):
        logger.info(f"========= FlatQuant calibration: Layer {i}/{num_layers - 1} =========")
        layer = layers[i].to(device)

        # Remember original dtypes so we can restore after float32 training
        dtype_dict = {}
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        # (a) Compute FP reference outputs ──────────────────────────
        layer.self_attn._ori_mode = True
        layer.mlp._ori_mode = True
        with torch.no_grad():
            for j in range(actual_nsamples):
                out = layer(fp_inps[j].unsqueeze(0), **layer_kwargs)
                if isinstance(out, tuple):
                    out = out[0]
                fp_outs[j] = out.squeeze(0)
        layer.self_attn._ori_mode = False
        layer.mlp._ori_mode = False

        # (b) Initialise diagonal scales from activation / weight stats
        if add_diag:
            layer.self_attn.init_diag_scale(alpha=diag_alpha)
            layer.mlp.init_diag_scale(alpha=diag_alpha)

        layer = layer.to(device)

        # (c) Set up trainable parameters & optimiser ───────────────
        for param in layer.parameters():
            param.requires_grad = False

        trained_params = [
            {
                "params": _get_params_by_pattern(
                    layer, ["trans.u_", "trans.v_", "trans.diag_left", "trans.diag_right"],
                ),
                "lr": flat_lr,
            },
        ]
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

        # (d) Train transforms via MSE loss ─────────────────────────
        nan_detected = False
        for epoch in range(epochs):
            epoch_mse = 0.0
            with traincast():
                for j in range(actual_nsamples // cali_bsz):
                    idx = j * cali_bsz
                    out = layer(fp_inps[idx:idx + cali_bsz], **batch_kwargs)
                    quant_out = out[0] if isinstance(out, tuple) else out
                    loss = loss_func(fp_outs[idx:idx + cali_bsz], quant_out)
                    if torch.isnan(loss) or torch.isinf(loss):
                        if not nan_detected:
                            nan_detected = True
                            logger.warning(
                                f"  layer {i} NaN/Inf loss at epoch {epoch}, batch {j}. "
                                f"quant_out has NaN: {torch.isnan(quant_out).any().item()}, "
                                f"fp_outs slice has NaN: {torch.isnan(fp_outs[idx:idx+cali_bsz]).any().item()}"
                            )
                        scheduler.step()
                        continue
                    epoch_mse += loss.detach().item()
                    normalized_loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    normalized_loss.backward()
                    optimizer.step()
                    scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"  layer {i} epoch {epoch}, lr={lr:.8f}, mse={epoch_mse:.8f}")

        # Feed this layer's output as the next layer's input
        fp_inps, fp_outs = fp_outs, fp_inps

        # Restore dtypes and move to CPU
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict:
                param.data = param.to(dtype_dict[name])
        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    logger.info("FlatQuant calibration complete")
    return model


def _get_params_by_pattern(
    module: nn.Module, patterns: list[str],
) -> list[nn.Parameter]:
    """Collect parameters whose names contain any of *patterns*."""
    params = []
    for name, param in module.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = True
            params.append(param)
    return params


# =========================================================================
# Part 6 — Save / Load Transforms
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
