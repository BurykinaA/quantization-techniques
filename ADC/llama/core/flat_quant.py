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

from ADC.llama.core.grad_functions import floor_ste, round_ste

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
        # Cast to float32: in float16 the clamp(min=1e-8) is a no-op
        # (1e-8 < float16 min subnormal ~6e-8 → rounds to 0), making
        # scale=0 possible → 0/0=NaN in backward even though forward
        # looks fine (round(inf).clamp()*0 = 0).
        orig_dtype = x.dtype
        x = x.float()
        x_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        if self.lac:
            x_max = x_max * self.sigmoid(self.clip_factor)
        scale = x_max / self.maxq
        return ((x / scale).round().clamp(-self.maxq - 1, self.maxq) * scale).to(orig_dtype)


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


def _hadamard_matrix(n: int) -> torch.Tensor:
    """Return the normalized n×n Hadamard matrix (float32).

    Uses recursive Sylvester construction:
        H_1 = [[1]],  H_{2k} = (1/√2) [[H_k, H_k], [H_k, -H_k]]

    The returned matrix satisfies H @ H.T = I (orthogonal).
    n must be a power of two.
    """
    assert n >= 1 and (n & (n - 1)) == 0, f"Hadamard requires power-of-2 size, got {n}"
    h = torch.ones(1, 1, dtype=torch.float64)
    cur = 1
    while cur < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
        cur *= 2
    return (h / math.sqrt(n)).float()


def _init_kronecker_hadamard(trans: "KroneckerTransform") -> None:
    """Re-initialize Kronecker factors so P ≈ H_left ⊗ H_right.

    Sets  u_left = H_left,  v_left = I  →  mat_l = H_left @ I @ I.T = H_left
    and equivalently for the right factor, so the full transform starts as a
    pure (normalized) Hadamard rotation.  diag_left / diag_right are reset
    to ones (no anisotropic scaling at initialization).

    Only applied when both left_size and right_size are powers of two;
    silently falls back to the current weights otherwise.
    """
    def _is_pow2(n: int) -> bool:
        return n >= 1 and (n & (n - 1)) == 0

    if not (_is_pow2(trans.left_size) and _is_pow2(trans.right_size)):
        logger.debug(
            "_init_kronecker_hadamard: skipping (left=%d right=%d not both pow2)",
            trans.left_size, trans.right_size,
        )
        return

    dev = trans.diag_left.device

    def _reinit(m: nn.Linear, w: torch.Tensor) -> None:
        """Replace the parametrized weight's base point with *w*."""
        torch.nn.utils.parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        m.weight.data = w.to(dev)
        nn.utils.parametrizations.orthogonal(
            m, orthogonal_map="matrix_exp", use_trivialization=True,
        )

    h_l   = _hadamard_matrix(trans.left_size)
    h_r   = _hadamard_matrix(trans.right_size)
    eye_l = torch.eye(trans.left_size)
    eye_r = torch.eye(trans.right_size)

    _reinit(trans.u_left,  h_l)
    _reinit(trans.v_left,  eye_l)
    _reinit(trans.u_right, h_r)
    _reinit(trans.v_right, eye_r)

    trans.diag_left.data.fill_(1.0)
    trans.diag_right.data.fill_(1.0)


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
            _ds = self.diag_scale.abs().clamp(min=1e-8)
            x = x / _ds if inv_t else x * _ds

        if not self._eval_mode:
            _dl = self.diag_left.abs().clamp(min=1e-6)
            _dr = self.diag_right.abs().clamp(min=1e-6)
            dl = 1.0 / _dl if inv_t else _dl
            dr = 1.0 / _dr if inv_t else _dr
            mat_l = self.u_left.weight @ torch.diag(dl) @ self.v_left.weight.T
            mat_r = self.u_right.weight @ torch.diag(dr) @ self.v_right.weight.T
        else:
            if inv_t:
                mat_l, mat_r = self.matrix_left_inv, self.matrix_right_inv
            else:
                mat_l, mat_r = self.matrix_left, self.matrix_right

        return _kronecker_matmul(x, mat_l, mat_r).to(orig_dtype)

    def to_eval_mode(self) -> None:
        """Pre-compute and cache forward / inverse / transpose matrices, free SVD params."""
        if self._eval_mode:
            return
        with torch.no_grad():
            _dl = self.diag_left.abs().clamp(min=1e-6)
            _dr = self.diag_right.abs().clamp(min=1e-6)
            mat_l = self.u_left.weight @ torch.diag(_dl) @ self.v_left.weight.T
            mat_r = self.u_right.weight @ torch.diag(_dr) @ self.v_right.weight.T
            mat_l_inv = self.u_left.weight @ torch.diag(1.0 / _dl) @ self.v_left.weight.T
            mat_r_inv = self.u_right.weight @ torch.diag(1.0 / _dr) @ self.v_right.weight.T
            # L^T = V @ D @ U^T  (used in reparameterize to cancel Kron^{-T})
            mat_l_T = self.v_left.weight @ torch.diag(_dl) @ self.u_left.weight.T
            mat_r_T = self.v_right.weight @ torch.diag(_dr) @ self.u_right.weight.T
        self.matrix_left = nn.Parameter(mat_l, requires_grad=False)
        self.matrix_right = nn.Parameter(mat_r, requires_grad=False)
        self.matrix_left_inv = nn.Parameter(mat_l_inv, requires_grad=False)
        self.matrix_right_inv = nn.Parameter(mat_r_inv, requires_grad=False)
        self.matrix_left_T = nn.Parameter(mat_l_T, requires_grad=False)
        self.matrix_right_T = nn.Parameter(mat_r_T, requires_grad=False)
        del self.u_left, self.v_left, self.diag_left
        del self.u_right, self.v_right, self.diag_right
        self._eval_mode = True

    def apply_kron_transpose(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Kron^T = kron(V_L D_L U_L^T, V_R D_R U_R^T) to x (no diag_scale).

        Used in reparameterize() to cancel the Kron^{-T} already folded into the
        weight, so that the stored weight = w / diag (correct for strip-mode inference
        where Kron is no longer applied to activations).
        """
        orig_dtype = x.dtype
        x = x.float()
        if not self._eval_mode:
            _dl = self.diag_left.abs().clamp(min=1e-6)
            _dr = self.diag_right.abs().clamp(min=1e-6)
            mat_l = self.v_left.weight @ torch.diag(_dl) @ self.u_left.weight.T
            mat_r = self.v_right.weight @ torch.diag(_dr) @ self.u_right.weight.T
        else:
            mat_l, mat_r = self.matrix_left_T, self.matrix_right_T
        return _kronecker_matmul(x, mat_l, mat_r).to(orig_dtype)


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
        adc_config: dict | None = None,
    ):
        super().__init__()
        self.linear = linear
        self.w_quantizer = _WeightQuantizer(bits=w_bits)
        self.a_quantizer = _ActivationQuantizer(bits=a_bits, lac=lac)
        self.lwc = lwc
        self._adc_config = dict(adc_config) if adc_config is not None else None
        self._reparameterized = False  # set True after reparameterize(); skips weight transform
        # Penalty attrs — training-only, not part of adc_config
        self._penalty_lambda_clip    = 0.0
        self._penalty_lambda_dead    = 0.0
        self._penalty_clip_margin    = 1.0
        self._penalty_dead_threshold = 1.0
        self._penalty_enabled        = False
        # Last computed penalties (mean over tiles, set during forward)
        self._last_clip_penalty: torch.Tensor | None = None
        self._last_dead_penalty: torch.Tensor | None = None
        # Band-occupancy penalty: encourages z-mass into (tau_lo, tau_hi)
        # L_band = 1 - E[σ(β(|z|−τ_lo)) · σ(β(τ_hi−|z|))]
        self._band_enabled      = False
        self._band_lambda       = 0.0
        self._band_tau_lo       = 1.0
        self._band_tau_hi       = 64.0
        self._band_beta         = 5.0
        self._band_topk_frac    = 0.2   # top-k% worst tiles by band loss
        self._last_band_penalty: torch.Tensor | None = None
        # PACT-style learnable activation range for ADC path.
        # alpha_adc = softplus(raw_alpha_adc) + eps — a fixed clip threshold
        # replacing per-token amax in _train_forward_adc.
        # None when adc_config is None (non-ADC path).
        self.raw_alpha_adc: nn.Parameter | None = None
        self._alpha_adc_initialized = False
        if adc_config is not None:
            # Initialize raw_alpha_adc so softplus(raw) ≈ 1.0 (softplus(0.541)≈1)
            # Will be overwritten by init_alpha_adc() from calibration stats.
            self.raw_alpha_adc = nn.Parameter(torch.tensor(0.541))
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

    def init_alpha_adc(self, act_stats: torch.Tensor) -> None:
        """Initialise raw_alpha_adc from percentile of activation magnitudes.

        act_stats: 1-D tensor of per-channel max activations (abs) collected
                   during the FP forward pass.  We use p99 of the distribution
                   as the initial clip threshold.
        """
        if self.raw_alpha_adc is None:
            return
        p99 = torch.quantile(act_stats.float(), 0.99).clamp(min=1e-3)
        # softplus^{-1}(x) = log(exp(x) - 1)
        raw = torch.log(torch.expm1(p99))
        self.raw_alpha_adc.data.fill_(raw.item())
        self._alpha_adc_initialized = True

    def ori_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Original FP forward — no transform, no quantisation."""
        return self.linear(x)

    def train_forward(
        self,
        x: torch.Tensor,
        qa_trans: KroneckerTransform | None = None,
    ) -> torch.Tensor:
        """Quantised forward with optional transform (calibration path).

        When ``self._reparameterized`` is True (after reparameterize()), the
        weight transform is already baked in and the Kronecker transform is
        applied to activations at a higher level — just delegate to self.linear
        directly (which may be nn.Linear or TiledLinearADC after ADC conversion).

        When ``self._adc_config`` is set, simulates the full
        TiledLinearADC pipeline (integer MVM + ADC floor/clamp) so the
        Kronecker transforms learn to minimise ADC quantisation error.
        Otherwise falls back to simple INT8 fake-quant.
        """
        if self._reparameterized:
            return self.linear(x)

        if self._adc_config is not None:
            return self._train_forward_adc(x, qa_trans)

        weight = self.linear.weight.data
        if qa_trans is not None:
            weight = qa_trans(weight, inv_t=True)
        if self.lwc:
            weight = self._apply_wclip(weight)
        self.w_quantizer.find_params(weight)
        weight = self.w_quantizer(weight)
        x = self.a_quantizer(x)
        return F.linear(x, weight.to(x.dtype), self.linear.bias)

    def _train_forward_adc(
        self,
        x: torch.Tensor,
        qa_trans: KroneckerTransform | None = None,
    ) -> torch.Tensor:
        """ADC-aware forward: mirrors TiledLinearADC tile-by-tile.

        Steps per tile (matching QATLinearADC.forward):
          1. Quantize activations → integer codes
          2. Quantize weights → integer codes (per output channel)
          3. Integer MVM in code domain
          4. ADC quantization: floor(y / Δ).clamp(na, pa) * Δ  (Eq 2-3)
          5. Dequantize: multiply by activation and weight scales
        """
        cfg = self._adc_config
        bx: int = cfg["bx"]
        bw: int = cfg["bw"]
        ba: int = cfg["ba"]
        k: int  = cfg["k"]
        mvm_limit: int = cfg["mvm_limit"]
        signed: bool   = cfg.get("signed_activations", True)

        # Quantization bounds (matching QATLinearADC)
        if signed:
            qmax_x = 2 ** (bx - 1) - 1
            qmin_x = -(2 ** (bx - 1))
        else:
            qmax_x = 2 ** bx - 1
            qmin_x = 0
        act_levels = float(qmax_x)  # 127 (signed) or 255 (unsigned)

        qmax_w = 2 ** (bw - 1) - 1
        qmin_w = -(2 ** (bw - 1))
        w_levels = float(qmax_w)    # 127

        na = -(2 ** (ba - 1))
        pa = 2 ** (ba - 1) - 1

        # Tiling — same logic as TiledLinearADC.__init__
        in_features = self.linear.in_features
        tile_in = in_features
        while tile_in > mvm_limit and tile_in % 2 == 0:
            tile_in //= 2
        n_tiles = in_features // tile_in

        # ADC step size (Eq. 3 from paper)
        delta = 2.0 * tile_in * act_levels * w_levels / (float(2 ** ba) * k)

        # Get (optionally transformed) weight in float32
        weight = self.linear.weight.data
        if qa_trans is not None:
            weight = qa_trans(weight, inv_t=True)
        if self.lwc:
            weight = self._apply_wclip(weight)
        w_f32 = weight.float()

        x_f32 = x.float()
        orig_shape = x_f32.shape
        x2d = x_f32.reshape(-1, in_features)                # [B, in_features]

        y2d = torch.zeros(
            x2d.shape[0], self.linear.out_features,
            device=x2d.device, dtype=x2d.dtype,
        )

        # Penalty accumulators — initialised on the same device as x
        clip_acc = x2d.new_zeros(()).float()
        dead_acc = x2d.new_zeros(()).float()
        band_tile_losses: list[torch.Tensor] = []   # one scalar per tile, for top-k selection
        n_tiles_eff = 0

        for i in range(n_tiles):
            xi = x2d[:, i * tile_in:(i + 1) * tile_in]     # [B, tile_in]
            wi = w_f32[:, i * tile_in:(i + 1) * tile_in]   # [out, tile_in]

            # Per-channel weight quantization for this tile
            s_wi = wi.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / w_levels
            code_wi = round_ste(wi / s_wi).clamp(qmin_w, qmax_w)

            # Per-tile activation quantization
            # PACT-style: use fixed learned clip threshold alpha_adc
            # so that code_xi is not dominated by per-token outliers.
            if self.raw_alpha_adc is not None:
                alpha = F.softplus(self.raw_alpha_adc).clamp(min=1e-6)
                xi_c = xi.clamp(-alpha, alpha)
                s_xi = alpha / act_levels           # scalar — same for all tokens
                code_xi = round_ste(xi_c / s_xi).clamp(qmin_x, qmax_x)
            else:
                s_xi = xi.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
                code_xi = round_ste(xi / s_xi).clamp(qmin_x, qmax_x)

            # Integer MVM → ADC quantization (Eq. 2-3)
            # y_int can reach tile_in * 127^2 ≈ 4M, which overflows float16
            # (~65504) when running under float16 autocast.  Force float32.
            _dev_type = "cuda" if code_xi.is_cuda else "cpu"
            with torch.amp.autocast(device_type=_dev_type, enabled=False):
                y_int = F.linear(code_xi.float(), code_wi.float())       # [B, out], float32
            y_adc  = floor_ste(y_int / delta).clamp(na, pa) * delta  # [B, out]

            # Accumulate penalties (only when enabled)
            if self._penalty_enabled or self._band_enabled:
                z = y_int / delta   # normalised ADC input, same shape as y_int
                if self._penalty_enabled:
                    clip_acc = clip_acc + F.relu(z.abs() - (float(pa) - self._penalty_clip_margin)).pow(2).mean()
                    dead_acc = dead_acc + F.relu(self._penalty_dead_threshold - z.abs()).mean()
                if self._band_enabled:
                    # per-tile loss stored for top-k selection after loop
                    z_abs = z.abs()
                    in_band = (torch.sigmoid(self._band_beta * (z_abs - self._band_tau_lo))
                               * torch.sigmoid(self._band_beta * (self._band_tau_hi - z_abs)))
                    band_tile_losses.append(1.0 - in_band.mean())
                n_tiles_eff += 1

            # Dequantize: s_xi [B,1] × s_wi.T [out] → [B, out]
            y2d = y2d + y_adc * s_xi * s_wi.squeeze(1)

        # Store per-tile mean penalties for the training loop
        self._last_clip_penalty = clip_acc / max(n_tiles_eff, 1)
        self._last_dead_penalty = dead_acc / max(n_tiles_eff, 1)
        if band_tile_losses:
            stacked = torch.stack(band_tile_losses)          # [n_tiles]
            k = max(1, int(len(stacked) * self._band_topk_frac))
            # topk returns the k largest values (= worst tiles)
            self._last_band_penalty = stacked.topk(k).values.mean()
        else:
            self._last_band_penalty = x2d.new_zeros(()).float()

        y = y2d.reshape(*orig_shape[:-1], self.linear.out_features)
        if self.linear.bias is not None:
            y = y + self.linear.bias
        return y.to(x.dtype)

    def reparameterize(
        self, qa_trans: KroneckerTransform | None = None,
    ) -> None:
        """Fold transform + clipping into weights permanently.

        Stores W_stored = clip(w / diag @ Kron^{-T}) — matching the official
        FlatQuant approach.  The Kronecker transform is still applied to
        activations at inference time (via up_gate_trans / ln_trans in the
        MLP/Attention wrappers), so the full product is correct:

            (x * D @ Kron) @ (W/D @ Kron^{-T})^T = x @ W^T  ✓

        After this call, _reparameterized=True makes train_forward() bypass
        all weight transforms and delegate directly to self.linear (which may
        later be replaced by TiledLinearADC by the ADC converter).
        """
        weight = self.linear.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)
        if qa_trans is not None:
            weight = qa_trans(weight, inv_t=True)  # w / diag @ Kron^{-T}
        if self.lwc:
            weight = self._apply_wclip(weight)
        self.linear.weight.data = weight.to(ori_dtype)
        self._reparameterized = True


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
        adc_config: dict | None = None,
    ):
        super().__init__()
        self.act_fn = mlp.act_fn

        self.gate_proj = FlatQuantLinear(mlp.gate_proj, w_bits, a_bits, lwc, lac, adc_config)
        self.up_proj   = FlatQuantLinear(mlp.up_proj,   w_bits, a_bits, lwc, lac, adc_config)
        self.down_proj = FlatQuantLinear(mlp.down_proj, w_bits, a_bits, lwc, lac, adc_config)

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
        adc_config: dict | None = None,
    ):
        super().__init__()
        self._orig_attn = attn
        in_dim = attn.q_proj.weight.shape[1]

        self.q_proj = FlatQuantLinear(attn.q_proj, w_bits, a_bits, lwc, lac, adc_config)
        self.k_proj = FlatQuantLinear(attn.k_proj, w_bits, a_bits, lwc, lac, adc_config)
        self.v_proj = FlatQuantLinear(attn.v_proj, w_bits, a_bits, lwc, lac, adc_config)
        self.o_proj = FlatQuantLinear(attn.o_proj, w_bits, a_bits, lwc, lac, adc_config)

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
    adc_config: dict | None = None,
) -> nn.Module:
    """Replace LLaMA attention and MLP with FlatQuant-wrapped versions.

    Args:
        adc_config: When provided, ``train_forward`` simulates the full
            ADC pipeline (tiled integer MVM + ``floor_ste`` + delta clamp)
            instead of simple INT8 fake-quant.  Expected keys::

                {bx, bw, ba, k, mvm_limit, signed_activations}
    """
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Applying FlatQuant wrappers"):
        layer = layers[i]
        layers[i].self_attn = FlatQuantLlamaAttention(
            layer.self_attn, w_bits, a_bits, add_diag, lwc, lac, adc_config,
        )
        layers[i].mlp = FlatQuantLlamaMLP(
            layer.mlp, w_bits, a_bits, add_diag, lwc, lac, adc_config,
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
    lambda_clip: float = 0.0,
    lambda_dead: float = 0.0,
    clip_margin: float = 1.0,
    dead_threshold: float = 1.0,
    penalty_projections: list[str] | None = None,
    lambda_band: float = 0.0,
    band_tau_lo: float = 1.0,
    band_tau_hi: float = 64.0,
    band_beta: float = 5.0,
    band_topk_frac: float = 0.2,
    freeze_clip: bool = False,
    kronecker_init: str = "random",
    loss_type: str = "mse",
    huber_delta: float = 1.0,
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
    if loss_type == "l1":
        loss_func = nn.L1Loss()
    elif loss_type == "huber":
        loss_func = nn.HuberLoss(delta=huber_delta)
    else:
        loss_func = nn.MSELoss()

    num_layers = len(layers)
    layer_bar = tqdm(range(num_layers), desc="FlatQuant layers", unit="layer")
    for i in layer_bar:
        layer_bar.set_postfix(layer=i)
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

        # Sanity-check: warn if FP reference data contains NaN/Inf
        _inp_bad = torch.isnan(fp_inps[:actual_nsamples]).any() or torch.isinf(fp_inps[:actual_nsamples]).any()
        _out_bad = torch.isnan(fp_outs[:actual_nsamples]).any() or torch.isinf(fp_outs[:actual_nsamples]).any()
        if _inp_bad:
            logger.warning(f"Layer {i}: fp_inps contains NaN/Inf — skipping calibration")
            fp_inps, fp_outs = fp_outs, fp_inps
            continue
        if _out_bad:
            logger.warning(f"Layer {i}: fp_outs (FP reference) contains NaN/Inf — training may be unstable")

        # (b) Initialise diagonal scales from activation / weight stats
        if add_diag:
            layer.self_attn.init_diag_scale(alpha=diag_alpha)
            layer.mlp.init_diag_scale(alpha=diag_alpha)

        # (b2) Optionally re-initialize Kronecker factors to Hadamard rotation
        if kronecker_init == "hadamard":
            for _name, _m in layer.named_modules():
                if isinstance(_m, KroneckerTransform):
                    _init_kronecker_hadamard(_m)

        # (b3) Initialize alpha_adc from activation statistics
        # Use p99 of fp_inps as a rough proxy for each projection's input range.
        # This is the layer input; the actual projection inputs differ (especially
        # down_proj which sees intermediate activations), but it gives a better
        # starting point than a fixed value and lets the optimizer fine-tune.
        for _name, _m in layer.named_modules():
            if isinstance(_m, FlatQuantLinear) and _m.raw_alpha_adc is not None:
                p99 = torch.quantile(fp_inps[:actual_nsamples].abs().float(), 0.99)
                raw = torch.log(torch.expm1(p99.clamp(min=0.01)))
                _m.raw_alpha_adc.data.fill_(raw.clamp(min=-5.0, max=5.0).item())

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
        if lwc and not freeze_clip:
            trained_params.append({
                "params": _get_params_by_pattern(layer, ["clip_factor_w"]),
                "lr": flat_lr * 10,
            })
        if lac and not freeze_clip:
            trained_params.append({
                "params": _get_params_by_pattern(layer, ["clip_factor_a"]),
                "lr": flat_lr * 10,
            })
        # PACT alpha_adc: use higher LR like clip factors (10x base LR)
        # but only when adc_config is set (raw_alpha_adc is not None).
        # _get_params_by_pattern also sets requires_grad=True, so use it here too.
        alpha_params = _get_params_by_pattern(layer, ["raw_alpha_adc"])
        if alpha_params:
            trained_params.append({
                "params": alpha_params,
                "lr": flat_lr * 10,
            })

        # Inject penalty attrs — separate from adc_config, not mutating shared dict
        _penalty_projs = penalty_projections or ["o_proj", "down_proj"]
        for _name, _m in layer.named_modules():
            if isinstance(_m, FlatQuantLinear) and _m._adc_config is not None:
                _is_target = any(p in _name for p in _penalty_projs)
                _m._penalty_enabled        = _is_target and (lambda_clip > 0.0 or lambda_dead > 0.0)
                _m._penalty_lambda_clip    = lambda_clip
                _m._penalty_lambda_dead    = lambda_dead
                _m._penalty_clip_margin    = clip_margin
                _m._penalty_dead_threshold = dead_threshold
                _m._band_enabled           = _is_target and (lambda_band > 0.0)
                _m._band_lambda            = lambda_band
                _m._band_tau_lo            = band_tau_lo
                _m._band_tau_hi            = band_tau_hi
                _m._band_beta              = band_beta
                _m._band_topk_frac         = band_topk_frac

        optimizer = torch.optim.AdamW(trained_params)
        total_steps = epochs * (actual_nsamples // cali_bsz)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=flat_lr * 1e-3,
        )

        # (d) Train transforms via MSE loss ─────────────────────────
        n_batches = actual_nsamples // cali_bsz

        # Attach forward hooks to catch the first NaN-producing tensor (only
        # on the first NaN batch so we don't spam the log).
        _nan_found: list[str] = []

        def _make_nan_hook(tag: str) -> callable:
            def _hook(module, inp, out):
                if _nan_found:          # already reported once, stop checking
                    return
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
                    _nan_found.append(tag)
                    logger.warning(f"Layer {i}: first NaN/Inf tensor → {tag}  shape={list(t.shape)}")
            return _hook

        _hooks = []
        for name, mod in layer.named_modules():
            _hooks.append(mod.register_forward_hook(_make_nan_hook(name)))

        epoch_bar = tqdm(range(epochs), desc=f"  L{i} epochs", unit="ep", leave=False)
        for epoch in epoch_bar:
            epoch_mse = 0.0
            epoch_dead = 0.0
            epoch_clip = 0.0
            epoch_band = 0.0
            epoch_pen_count = 0
            nan_count = 0
            batch_bar = tqdm(range(n_batches), desc=f"    L{i} E{epoch} batches", unit="batch", leave=False)
            for j in batch_bar:
                idx = j * cali_bsz
                _nan_found.clear()
                # Forward only under autocast — backward must run in float32
                # to avoid float16 overflow (1/loss can exceed float16 max).
                with traincast():
                    out = layer(fp_inps[idx:idx + cali_bsz], **batch_kwargs)
                    quant_out = out[0] if isinstance(out, tuple) else out
                    loss = loss_func(fp_outs[idx:idx + cali_bsz], quant_out)
                    if lambda_clip > 0.0 or lambda_dead > 0.0 or lambda_band > 0.0:
                        _clip_acc = loss.new_zeros(())
                        _dead_acc = loss.new_zeros(())
                        _band_acc = loss.new_zeros(())
                        _n_pen = 0
                        for _, _m in layer.named_modules():
                            if isinstance(_m, FlatQuantLinear):
                                if (_m._penalty_enabled
                                        and _m._last_clip_penalty is not None):
                                    _clip_acc = _clip_acc + _m._last_clip_penalty.to(loss.device)
                                    _dead_acc = _dead_acc + _m._last_dead_penalty.to(loss.device)
                                    _n_pen += 1
                                if (_m._band_enabled
                                        and _m._last_band_penalty is not None):
                                    _band_acc = _band_acc + _m._last_band_penalty.to(loss.device)
                        if _n_pen > 0:
                            loss = loss + lambda_clip * _clip_acc / _n_pen \
                                       + lambda_dead * _dead_acc / _n_pen
                            epoch_clip += (_clip_acc / _n_pen).item()
                            epoch_dead += (_dead_acc / _n_pen).item()
                            epoch_pen_count += 1
                        _n_band = sum(
                            1 for _, _m in layer.named_modules()
                            if isinstance(_m, FlatQuantLinear) and _m._band_enabled
                        )
                        if _n_band > 0:
                            loss = loss + lambda_band * _band_acc / _n_band
                            epoch_band += (_band_acc / _n_band).item()
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    batch_bar.set_postfix(loss="NaN", nan=nan_count)
                    scheduler.step()
                    continue
                epoch_mse += loss.detach().item()
                # Cast to float32 before backward so 1/loss doesn't overflow
                loss_f32 = loss.float()
                normalized_loss = loss_f32 / loss_f32.clone().detach()
                optimizer.zero_grad()
                normalized_loss.backward()
                # Guard: skip step if any gradient is NaN/Inf.
                # clip_grad_norm_ with a NaN grad returns NaN norm,
                # which then poisons ALL gradients and all parameters.
                all_params = [p for g in optimizer.param_groups for p in g["params"]]
                has_nan_grad = any(
                    p.grad is not None
                    and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in all_params
                )
                if has_nan_grad:
                    # Log which parameter has NaN/Inf gradient
                    param_name_map = {id(lp): n for n, lp in layer.named_parameters()}
                    for g in optimizer.param_groups:
                        for p in g["params"]:
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                pname = param_name_map.get(id(p), "<unknown>")
                                finite_mask = ~torch.isnan(p.grad) & ~torch.isinf(p.grad)
                                if finite_mask.any():
                                    gmin = p.grad[finite_mask].min().item()
                                    gmax = p.grad[finite_mask].max().item()
                                else:
                                    gmin = gmax = float("nan")
                                n_nan = torch.isnan(p.grad).sum().item()
                                n_inf = torch.isinf(p.grad).sum().item()
                                logger.warning(
                                    f"Layer {i} batch {j}: NaN/Inf grad in '{pname}'  "
                                    f"nan={n_nan} inf={n_inf}  finite_range=[{gmin:.3e}, {gmax:.3e}]"
                                )
                    nan_count += 1
                    batch_bar.set_postfix(loss="NaN/grad", nan=nan_count)
                    optimizer.zero_grad()
                    scheduler.step()
                    continue
                # Clip gradients to prevent explosion through 1/diag paths
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                # Project diag parameters to stay in safe range.
                # diag_left/diag_right: ≥ 0.1 → T⁻¹ amplifies weights ≤ 10×.
                # diag_scale: absorbed into LayerNorm at reparameterize time,
                #   so large values create large inference activations → huge
                #   per-tile act_scale → typical activations round to 0.
                #   Clamp to [1e-4, 10] to keep inference activation scales sane.
                # Project diag parameters to stay in safe range.
                # diag_left/diag_right: ≥ 0.1 → T⁻¹ amplifies weights ≤ 10×.
                # diag_scale: absorbed into LayerNorm at reparameterize time,
                #   so large values create large inference activations → huge
                #   per-tile act_scale → typical activations round to 0.
                #   Clamp to [1e-4, 10] to keep inference activation scales sane.
                with torch.no_grad():
                    for name, param in layer.named_parameters():
                        if "diag_left" in name or "diag_right" in name:
                            param.data.clamp_(min=0.1)
                        elif "diag_scale" in name:
                            param.data.clamp_(min=1e-4, max=10.0)
                        elif "raw_alpha_adc" in name:
                            # Keep alpha = softplus(raw) in [0.1, 100]
                            # softplus^{-1}(0.1) ≈ -2.25, softplus^{-1}(100) ≈ 100
                            param.data.clamp_(min=-2.25, max=100.0)
                scheduler.step()
                _pf: dict = dict(
                    loss=f"{loss.item():.3e}",
                    mse=f"{epoch_mse / (j + 1 - nan_count):.3e}" if (j + 1 - nan_count) > 0 else "N/A",
                    nan=nan_count,
                )
                if lambda_clip > 0.0 or lambda_dead > 0.0:
                    if _n_pen > 0:
                        _pf["clip"] = f"{(_clip_acc / _n_pen).item():.3e}"
                        _pf["dead"] = f"{(_dead_acc / _n_pen).item():.3e}"
                if lambda_band > 0.0 and _n_band > 0:
                    _pf["band"] = f"{(_band_acc / _n_band).item():.3e}"
                batch_bar.set_postfix(**_pf)
            lr = optimizer.param_groups[0]["lr"]
            ok = n_batches - nan_count
            _epf = dict(lr=f"{lr:.2e}", mse=f"{epoch_mse:.3e}", ok=f"{ok}/{n_batches}")
            _log_extra = ""
            if epoch_pen_count > 0:
                _mean_dead = epoch_dead / epoch_pen_count
                _mean_clip = epoch_clip / epoch_pen_count
                _epf["dead"] = f"{_mean_dead:.3e}"
                if lambda_clip > 0.0:
                    _epf["clip"] = f"{_mean_clip:.3e}"
                _log_extra = f", dead={_mean_dead:.4e}, clip={_mean_clip:.4e}"
            if lambda_band > 0.0 and n_batches > 0:
                _mean_band = epoch_band / n_batches
                _epf["band"] = f"{_mean_band:.3e}"
                _log_extra += f", band={_mean_band:.4e}"
            epoch_bar.set_postfix(**_epf)
            logger.info(
                f"  layer {i} epoch {epoch}, lr={lr:.8f}, "
                f"mse={epoch_mse:.4e}, ok_batches={ok}/{n_batches}{_log_extra}"
            )
        # Log diag parameter ranges to catch blow-up early
        for name, param in layer.named_parameters():
            if "diag_scale" in name or "diag_left" in name or "diag_right" in name:
                with torch.no_grad():
                    p = param.data
                    logger.info(
                        f"  layer {i} [{name}] "
                        f"min={p.min():.4f} max={p.max():.4f} mean={p.mean():.4f}"
                    )
        for h in _hooks:
            h.remove()

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
# Part 6 — E3 Sanity-check helpers
# =========================================================================

def capture_layer_outputs(
    model: nn.Module,
    sample_input: dict,
    device: torch.device,
) -> dict:
    """Capture per-layer FlatQuant wrapper outputs for a fixed sample.

    Registers forward hooks on every ``FlatQuantLlamaMLP`` and
    ``FlatQuantLlamaAttention`` instance and runs a single no-grad forward
    pass.  Returns::

        {layer_idx: {'mlp': tensor_cpu, 'attn': tensor_cpu}}

    Works regardless of whether the model is in calibration mode or
    inference mode, so you can call it before *and* after
    ``reparameterize_model()`` + ADC conversion to compare the outputs.
    """
    outputs: dict = {}
    hooks: list = []

    for i, layer in enumerate(model.model.layers):
        outputs[i] = {}

        if isinstance(layer.mlp, FlatQuantLlamaMLP):
            def _mlp_hook(__m, __inp, out, idx=i):  # noqa: ARG001
                outputs[idx]["mlp"] = out.detach().cpu()
            hooks.append(layer.mlp.register_forward_hook(_mlp_hook))

        if isinstance(layer.self_attn, FlatQuantLlamaAttention):
            def _attn_hook(__m, __inp, out, idx=i):  # noqa: ARG001
                # attention forward returns (hidden_state, past_kv, ...)
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                outputs[idx]["attn"] = tensor.detach().cpu()
            hooks.append(layer.self_attn.register_forward_hook(_attn_hook))

    # Use the device the model's embedding table actually lives on —
    # after FlatQuant layer-by-layer calibration the model may still be on CPU
    # even if `device` is cuda.
    try:
        actual_device = next(model.parameters()).device
    except StopIteration:
        actual_device = device

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            input_ids = sample_input["input_ids"].to(actual_device)
            attn_mask = sample_input.get("attention_mask")
            kwargs = {}
            if attn_mask is not None:
                kwargs["attention_mask"] = attn_mask.to(actual_device)
            model(input_ids=input_ids, **kwargs)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return outputs


def compare_layer_outputs(
    before: dict,
    after: dict,
    log: "logging.Logger | None" = None,
) -> dict:
    """Compare per-layer outputs from two :func:`capture_layer_outputs` calls.

    Args:
        before: Outputs captured in calibration/FlatQuant mode.
        after:  Outputs captured in inference/ADC mode.
        log:    Optional logger; if *None* the module-level logger is used.

    Returns:
        Nested dict ``{layer_idx: {part: {'mse': float, 'rel_error': float}}}``
        where *part* is ``'mlp'`` or ``'attn'``.
    """
    if log is None:
        log = logger

    results: dict = {}
    all_mse: list = []
    all_rel: list = []

    for i in sorted(before.keys()):
        results[i] = {}
        for part in ("mlp", "attn"):
            b_dict = before.get(i, {})
            a_dict = after.get(i, {})
            if part not in b_dict or part not in a_dict:
                continue
            b = b_dict[part].float()
            a = a_dict[part].float()
            mse = float((b - a).pow(2).mean())
            ref = float(b.pow(2).mean())
            rel = mse / (ref + 1e-10)
            results[i][part] = {"mse": mse, "rel_error": rel}
            all_mse.append(mse)
            all_rel.append(rel)
            log.debug(f"  Layer {i:3d} {part}: MSE={mse:.4e}  rel={rel:.3%}")

    if all_mse:
        log.info(
            "E3 Sanity Check — calibration vs inference mismatch:\n"
            f"  Mean MSE:       {sum(all_mse) / len(all_mse):.4e}\n"
            f"  Max  MSE:       {max(all_mse):.4e}\n"
            f"  Mean rel error: {sum(all_rel) / len(all_rel):.3%}\n"
            f"  Max  rel error: {max(all_rel):.3%}"
        )
    else:
        log.warning("E3: no layer outputs were captured — check model wrapping")

    return results


# =========================================================================
# Part 7 — Save / Load Transforms
# =========================================================================

def save_flat_transforms(model: nn.Module, path: str) -> None:
    """Save FlatQuant transform state dicts for all layers.

    Saves Kronecker transforms AND LWC/LAC clip factors — both are required
    to reproduce the reparameterized weights on reload.
    """
    transforms = {}
    for i, layer in enumerate(model.model.layers):
        state = {}
        if isinstance(layer.self_attn, FlatQuantLlamaAttention):
            state["attn_ln_trans"] = layer.self_attn.ln_trans.state_dict()
            # LWC/LAC clip factors and alpha_adc for attention projections
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                fql = getattr(layer.self_attn, proj_name, None)
                if isinstance(fql, FlatQuantLinear):
                    proj_state = {}
                    if fql.lwc and hasattr(fql, "clip_factor_w_max"):
                        proj_state["clip_factor_w_max"] = fql.clip_factor_w_max.data
                        proj_state["clip_factor_w_min"] = fql.clip_factor_w_min.data
                    if hasattr(fql, "a_quantizer") and fql.a_quantizer.lac:
                        proj_state["clip_factor_a"] = fql.a_quantizer.clip_factor.data
                    if fql.raw_alpha_adc is not None:
                        proj_state["raw_alpha_adc"] = fql.raw_alpha_adc.data.clone()
                    if proj_state:
                        state[f"attn_{proj_name}_clip"] = proj_state
        if isinstance(layer.mlp, FlatQuantLlamaMLP):
            state["mlp_up_gate_trans"] = layer.mlp.up_gate_trans.state_dict()
            state["mlp_down_trans"] = layer.mlp.down_trans.state_dict()
            # LWC/LAC clip factors and alpha_adc for MLP projections
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                fql = getattr(layer.mlp, proj_name, None)
                if isinstance(fql, FlatQuantLinear):
                    proj_state = {}
                    if fql.lwc and hasattr(fql, "clip_factor_w_max"):
                        proj_state["clip_factor_w_max"] = fql.clip_factor_w_max.data
                        proj_state["clip_factor_w_min"] = fql.clip_factor_w_min.data
                    if hasattr(fql, "a_quantizer") and fql.a_quantizer.lac:
                        proj_state["clip_factor_a"] = fql.a_quantizer.clip_factor.data
                    if fql.raw_alpha_adc is not None:
                        proj_state["raw_alpha_adc"] = fql.raw_alpha_adc.data.clone()
                    if proj_state:
                        state[f"mlp_{proj_name}_clip"] = proj_state
        if state:
            transforms[i] = state
    torch.save(transforms, path)
    logger.info(f"Saved FlatQuant transforms to {path}")


def load_flat_transforms(model: nn.Module, path: str) -> nn.Module:
    """Load pre-trained FlatQuant transforms into an already-wrapped model.

    Restores both Kronecker transforms and LWC/LAC clip factors so that
    fq_reparameterize_model() produces the same weights as during training.
    """
    try:
        _device = next(model.parameters()).device
    except StopIteration:
        _device = torch.device("cpu")

    transforms = torch.load(path, map_location=_device)
    for i, state in transforms.items():
        layer = model.model.layers[i]

        # --- Attention ---
        if "attn_ln_trans" in state and isinstance(layer.self_attn, FlatQuantLlamaAttention):
            layer.self_attn.ln_trans.load_state_dict(state["attn_ln_trans"])
            layer.self_attn.ln_trans.to(_device)
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            clip_key = f"attn_{proj_name}_clip"
            if clip_key in state and isinstance(layer.self_attn, FlatQuantLlamaAttention):
                fql = getattr(layer.self_attn, proj_name, None)
                if isinstance(fql, FlatQuantLinear):
                    ps = state[clip_key]
                    if "clip_factor_w_max" in ps and hasattr(fql, "clip_factor_w_max"):
                        fql.clip_factor_w_max.data.copy_(ps["clip_factor_w_max"].to(_device))
                        fql.clip_factor_w_min.data.copy_(ps["clip_factor_w_min"].to(_device))
                    if "clip_factor_a" in ps and fql.a_quantizer.lac:
                        fql.a_quantizer.clip_factor.data.copy_(ps["clip_factor_a"].to(_device))
                    if "raw_alpha_adc" in ps and fql.raw_alpha_adc is not None:
                        fql.raw_alpha_adc.data.copy_(ps["raw_alpha_adc"].to(_device))

        # --- MLP ---
        if "mlp_up_gate_trans" in state and isinstance(layer.mlp, FlatQuantLlamaMLP):
            layer.mlp.up_gate_trans.load_state_dict(state["mlp_up_gate_trans"])
            layer.mlp.up_gate_trans.to(_device)
        if "mlp_down_trans" in state and isinstance(layer.mlp, FlatQuantLlamaMLP):
            layer.mlp.down_trans.load_state_dict(state["mlp_down_trans"])
            layer.mlp.down_trans.to(_device)
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            clip_key = f"mlp_{proj_name}_clip"
            if clip_key in state and isinstance(layer.mlp, FlatQuantLlamaMLP):
                fql = getattr(layer.mlp, proj_name, None)
                if isinstance(fql, FlatQuantLinear):
                    ps = state[clip_key]
                    if "clip_factor_w_max" in ps and hasattr(fql, "clip_factor_w_max"):
                        fql.clip_factor_w_max.data.copy_(ps["clip_factor_w_max"].to(_device))
                        fql.clip_factor_w_min.data.copy_(ps["clip_factor_w_min"].to(_device))
                    if "clip_factor_a" in ps and fql.a_quantizer.lac:
                        fql.a_quantizer.clip_factor.data.copy_(ps["clip_factor_a"].to(_device))
                    if "raw_alpha_adc" in ps and fql.raw_alpha_adc is not None:
                        fql.raw_alpha_adc.data.copy_(ps["raw_alpha_adc"].to(_device))

    logger.info(f"Loaded FlatQuant transforms from {path}")
    return model


def propagate_alpha_adc_to_tiled(model: nn.Module) -> None:
    """Propagate learned alpha_adc from FlatQuantLinear to TiledLinearADC.

    After ``replace_linear_with_adc``, each ``FlatQuantLinear.linear`` is a
    ``TiledLinearADC``.  This function reads ``raw_alpha_adc`` from every
    ``FlatQuantLinear`` and calls ``TiledLinearADC.set_alpha_adc`` so that
    inference uses the same PACT clip threshold that was learned during
    FlatQuant calibration.

    Must be called **after** ``replace_linear_with_adc`` and **before**
    any quantized inference.
    """
    from ADC.llama.core.adc_layers import TiledLinearADC
    n_propagated = 0
    for _, module in model.named_modules():
        if not isinstance(module, FlatQuantLinear):
            continue
        if module.raw_alpha_adc is None:
            continue
        if not isinstance(module.linear, TiledLinearADC):
            continue
        with torch.no_grad():
            alpha_val = float(
                torch.nn.functional.softplus(module.raw_alpha_adc).clamp(min=1e-6).item()
            )
        module.linear.set_alpha_adc(alpha_val)
        n_propagated += 1
    logger.info(f"Propagated alpha_adc to {n_propagated} TiledLinearADC layers")
