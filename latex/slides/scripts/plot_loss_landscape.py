"""
Loss landscape visualisation for FlatQuant INT4 with varying propagation alpha.

Method: Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets"
  — filter-normalised random directions in FlatQuant parameter space.

For each of the 3 checkpoints (α=0, 0.5, 1.0), we:
  1. Extract float32 parameters (the learnable FlatQuant transforms).
  2. Generate two shared filter-normalised random directions d1, d2.
  3. Sweep a 21×21 grid: θ_perturbed = θ_base + a·d1 + b·d2.
  4. Evaluate ADC cross-entropy loss on a small calibration batch.
  5. Plot three contourf panels side-by-side.

Expected story:
  α=0 (FP loss only)  → narrow basin in FP direction, wide in ADC direction
  α=0.5 (mixed)       → flat wide basin in both directions
  α=1.0 (ADC loss)    → deep steep basin in ADC direction

Usage (run from repo root or latex/slides/):
  python scripts/plot_loss_landscape.py \\
      --checkpoint_alpha0   path/to/landscape_alpha0/model_full.pt \\
      --checkpoint_alpha05  path/to/landscape_alpha05/model_full.pt \\
      --checkpoint_alpha1   path/to/landscape_alpha1/model_full.pt \\
      --output              figs/plot_loss_landscape_alpha.pdf
"""

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ── Make TiledLinearADC importable (needed for torch.load) ───────────────────
_here = Path(__file__).resolve().parent
_adc_llama = _here.parent.parent.parent / "ADC" / "llama"
if str(_adc_llama) not in sys.path:
    sys.path.insert(0, str(_adc_llama))

# ── Shared style (matches plot_all.py) ───────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ── Filter-normalised directions (Li et al. 2018) ────────────────────────────

def filter_normalize(direction: dict, reference: dict) -> dict:
    """
    Scale each tensor in `direction` so that each filter (row for 2D tensors,
    full tensor otherwise) has the same norm as the corresponding filter in
    `reference`. This makes the perturbation scale independent of layer size.
    """
    out = {}
    for k, d in direction.items():
        ref = reference[k]
        if d.dim() >= 2:
            # per-output-channel (row) normalisation
            d_norms   = d.view(d.shape[0], -1).norm(dim=1, keepdim=True).clamp(min=1e-8)
            ref_norms = ref.view(ref.shape[0], -1).norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = (ref_norms / d_norms).view(d.shape[0], *([1] * (d.dim() - 1)))
            out[k] = d * scale
        else:
            d_norm   = d.norm().clamp(min=1e-8)
            ref_norm = ref.norm().clamp(min=1e-8)
            out[k] = d * (ref_norm / d_norm)
    return out


def random_direction(reference: dict) -> dict:
    """Random filter-normalised direction matching the scale of `reference`."""
    raw = {k: torch.randn_like(v) for k, v in reference.items()}
    return filter_normalize(raw, reference)


# ── Calibration batch ─────────────────────────────────────────────────────────

_CAL_TEXTS = [
    "The history of computing dates back to ancient times, but modern computers "
    "emerged in the mid-twentieth century with the development of transistors.",
    "Neural networks are inspired by the structure of the human brain, consisting "
    "of layers of interconnected nodes that process information.",
    "Quantization reduces the precision of model weights and activations from "
    "floating point to lower-bit integers, enabling efficient hardware deployment.",
    "The Eiffel Tower is located in Paris and was built in 1889 as the entrance "
    "arch for the 1889 World Fair.",
    "Language models are trained on large corpora of text and learn to predict "
    "the next token given a sequence of previous tokens.",
    "Analog in-memory computing performs matrix-vector multiplication directly "
    "in memory, avoiding data movement between memory and compute units.",
    "The gradient descent algorithm updates model parameters by moving in the "
    "direction that reduces the training loss.",
    "Attention mechanisms allow neural networks to focus on relevant parts of "
    "the input sequence when generating each output token.",
]


def get_cal_batch(tokenizer_name: str, batch_size: int, seq_len: int,
                  device: torch.device) -> torch.Tensor:
    """
    Tokenise a small set of fixed texts and return input_ids.
    Falls back to random tokens if the tokenizer cannot be loaded.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        texts = (_CAL_TEXTS * ((batch_size // len(_CAL_TEXTS)) + 1))[:batch_size]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=seq_len)
        return enc["input_ids"].to(device)
    except Exception as e:
        print(f"  [warn] tokenizer load failed ({e}), using random tokens")
        return torch.randint(0, 32000, (batch_size, seq_len), device=device)


# ── Landscape evaluation ──────────────────────────────────────────────────────

def get_fp32_params(model) -> dict:
    """
    Extract learnable float32 parameters — these are the FlatQuant transform
    matrices. The base model weights are fp16; fp32 = FlatQuant params.
    """
    return {k: p.data.clone().cpu()
            for k, p in model.named_parameters()
            if p.dtype == torch.float32}


def eval_loss(model, input_ids: torch.Tensor) -> float:
    """CE loss of the (ADC-active) model on input_ids. Clamped at 20 for plotting."""
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    with torch.no_grad():
        try:
            out = model(input_ids=ids, labels=ids)
            loss = out.loss.item()
        except Exception:
            loss = 20.0
    return min(float(loss), 20.0)


def compute_landscape(model, base_params: dict,
                      d1: dict, d2: dict,
                      input_ids: torch.Tensor,
                      grid_range: float, grid_n: int) -> np.ndarray:
    """
    Sweep a grid_n × grid_n grid around base_params and evaluate CE loss.
    Returns Z array of shape (grid_n, grid_n).
    """
    device = next(model.parameters()).device
    xs = np.linspace(-grid_range, grid_range, grid_n)
    Z = np.zeros((grid_n, grid_n))

    # Pre-move directions to device
    d1_dev = {k: v.to(device) for k, v in d1.items()}
    d2_dev = {k: v.to(device) for k, v in d2.items()}
    base_dev = {k: v.to(device) for k, v in base_params.items()}

    # Map param name → actual parameter tensor in model (for in-place update)
    param_map = {k: p for k, p in model.named_parameters() if k in base_params}

    total = grid_n * grid_n
    for i, a in enumerate(xs):
        for j, b in enumerate(xs):
            # Perturb in-place
            for k, p in param_map.items():
                p.data.copy_(base_dev[k] + a * d1_dev[k] + b * d2_dev[k])

            Z[i, j] = eval_loss(model, input_ids)

        if (i + 1) % 5 == 0:
            print(f"    row {i+1}/{grid_n} done")

    # Restore original params
    for k, p in param_map.items():
        p.data.copy_(base_dev[k])

    return Z


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_landscapes(landscapes: list[tuple], alphas: list[str], out_path: str):
    """
    landscapes: list of (xs, Z) where xs is the 1D grid and Z is (n,n).
    alphas: list of alpha label strings.
    """
    n = len(landscapes)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))

    # Shared color range: clip at 95th percentile to avoid outlier domination
    all_vals = np.concatenate([Z.ravel() for _, Z in landscapes])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 95))

    for ax, (xs, Z), alpha in zip(axes, landscapes, alphas):
        cf = ax.contourf(xs, xs, Z.T, levels=30, cmap="RdYlBu_r",
                         vmin=vmin, vmax=vmax)
        ax.contour(xs, xs, Z.T, levels=12, colors="k", linewidths=0.25, alpha=0.35)
        ax.set_title(f"α = {alpha}", fontsize=13)
        ax.set_xlabel("direction $d_1$")
        if ax is axes[0]:
            ax.set_ylabel("direction $d_2$")
        ax.plot(0, 0, "w+", markersize=10, markeredgewidth=1.5)  # trained checkpoint
        plt.colorbar(cf, ax=ax, label="CE loss", shrink=0.85)

    fig.suptitle("Loss landscape: FlatQuant INT4, varying propagation α",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_alpha0",  required=True,
                        help="Path to model_full.pt for α=0")
    parser.add_argument("--checkpoint_alpha05", required=True,
                        help="Path to model_full.pt for α=0.5")
    parser.add_argument("--checkpoint_alpha1",  required=True,
                        help="Path to model_full.pt for α=1.0")
    parser.add_argument("--output", required=True,
                        help="Output PDF path (e.g. figs/plot_loss_landscape_alpha.pdf)")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B",
                        help="Model name for tokenizer loading")
    parser.add_argument("--layer_idx", type=int, default=None,
                        help="(unused) reserved for future per-layer proxy mode")
    parser.add_argument("--grid_n", type=int, default=21,
                        help="Grid resolution per axis (default 21 → 441 points)")
    parser.add_argument("--grid_range", type=float, default=0.5,
                        help="Perturbation range [-r, r] in filter-norm units (default 0.5)")
    parser.add_argument("--cal_batch_size", type=int, default=8,
                        help="Number of sentences in calibration batch (default 8)")
    parser.add_argument("--seq_len", type=int, default=64,
                        help="Sequence length for calibration batch (default 64)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Grid: {args.grid_n}×{args.grid_n}, range ±{args.grid_range}")

    checkpoints = [
        ("0",   args.checkpoint_alpha0),
        ("0.5", args.checkpoint_alpha05),
        ("1.0", args.checkpoint_alpha1),
    ]

    # Calibration batch (same for all models)
    print("Loading calibration batch …")
    input_ids = get_cal_batch(args.model_name, args.cal_batch_size,
                               args.seq_len, device)
    print(f"  input_ids shape: {input_ids.shape}")

    # Shared random directions — generated from first model's params
    print("Loading α=0 checkpoint to generate shared directions …")
    model0 = torch.load(args.checkpoint_alpha0, weights_only=False,
                        map_location=device)
    model0.eval()
    base_params_ref = get_fp32_params(model0)
    print(f"  float32 params: {len(base_params_ref)} tensors, "
          f"{sum(v.numel() for v in base_params_ref.values()):,} values")

    torch.manual_seed(0)
    d1 = random_direction(base_params_ref)
    d2 = random_direction(base_params_ref)
    del model0   # free memory before loading next

    landscapes = []
    for alpha_str, ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"  [skip] checkpoint not found: {ckpt_path}")
            continue

        print(f"\nProcessing α={alpha_str}  ({ckpt_path})")
        model = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.eval()

        base_params = get_fp32_params(model)
        xs = np.linspace(-args.grid_range, args.grid_range, args.grid_n)

        print(f"  Computing {args.grid_n}×{args.grid_n} landscape …")
        Z = compute_landscape(model, base_params, d1, d2,
                              input_ids, args.grid_range, args.grid_n)

        print(f"  loss range: {Z.min():.3f} – {Z.max():.3f}  "
              f"(center = {Z[args.grid_n//2, args.grid_n//2]:.3f})")
        landscapes.append((xs, Z))
        del model  # free GPU memory before next model

    if len(landscapes) < 3:
        print(f"Warning: only {len(landscapes)}/3 checkpoints found. "
              f"Plotting with available data.")

    alphas = ["0", "0.5", "1.0"][:len(landscapes)]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot_landscapes(landscapes, alphas, args.output)


if __name__ == "__main__":
    main()
